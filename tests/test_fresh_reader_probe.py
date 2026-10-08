"""Disposable probe selection/attachment boundaries; no CUDA execution here."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knee.fresh_reader import CONFIG, PREPROCESS_VERSION, SOURCE_HASHES, fingerprint
from knee.fresh_reader_probe import select_probe_records
try:
    import numpy as np
    import torch
except ImportError:
    np = None


class ProbeCoreTests(unittest.TestCase):
    def test_selects_only_cached_training_without_label_or_order_dependence(self):
        rows = [{'uid': str(i), 'split': 'train' if i < 45 else 'weak_validation', 'labels': [.1]*12}
                for i in range(64)]
        rows += [{'uid':'uncached', 'split':'train', 'labels':[.9]*12},
                 {'uid':'gold', 'split':'gold', 'labels':[1.]*12}]
        index = {str(i): {} for i in range(64)} | {'gold': {}}
        expected = select_probe_records({'records': rows}, index)
        actual = select_probe_records({'records': [dict(r, labels=[.9]*12) for r in reversed(rows)]}, index)
        self.assertEqual([r['uid'] for r in actual], [r['uid'] for r in expected])
        self.assertEqual(len(actual), 32)
        self.assertTrue(all(r['split'] == 'train' and int(r['uid']) < 45 for r in actual))
        with self.assertRaises(ValueError):
            select_probe_records({'records': rows}, index, 33)
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select_probe_records({'records': rows}, {}, 32)

    def test_probe_builder_attaches_only_private_pilot_cache(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('probe_builder_test', root/'scripts/build_fresh_reader.py')
        builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d)/'sources', Path(d)/'notebook'
            src.mkdir()
            for name in ('preprocess.py','knee.py'): (src/name).write_text('pass\n')
            with patch.object(builder, 'check_sources'):
                contract = builder.build(src, out, 'probe', 'probe', cache_kernel='ozgung/pilot', plan_fingerprint='abc')
            metadata = json.loads((out/'kernel-metadata.json').read_text())
            self.assertEqual(metadata['dataset_sources'], [])
            self.assertEqual(metadata['competition_sources'], [])
            self.assertEqual(metadata['model_sources'], [])
            self.assertEqual(metadata['kernel_sources'], ['ozgung/pilot'])
            self.assertTrue(metadata['is_private'] and metadata['enable_gpu'] and metadata['enable_internet'])
            self.assertEqual(contract['effective_stage_seconds'], 600)
            nb = json.loads((out/'notebook.ipynb').read_text())
            generated = ''.join(nb['cells'][1]['source'])
            compile(generated, 'probe', 'exec')
            self.assertIn('STAGE_BUDGET = 600', generated)
            self.assertIn("if STAGE != 'probe':", generated)


@unittest.skipIf(np is None, 'Optional ML dependencies unavailable')
class ProbeMLTests(unittest.TestCase):
    def test_encoder_hash_is_order_independent_and_binds_dtype_shape_values(self):
        from knee.fresh_reader import tensor_state_sha256
        state = {'b': torch.tensor(1.), 'a': torch.tensor([[1.,2.]])}
        digest = tensor_state_sha256(state)
        self.assertEqual(digest, tensor_state_sha256(dict(reversed(list(state.items())))))
        for modified in ({'b':state['b'], 'a':state['a'].reshape(2)},
                         {'b':state['b'], 'a':state['a'].double()},
                         {'b':state['b'], 'a':state['a'] + 1}):
            self.assertNotEqual(digest, tensor_state_sha256(modified))
        self.assertEqual(len(tensor_state_sha256({'bfloat': torch.ones(2, dtype=torch.bfloat16)})), 64)

    def test_probe_checks_selected_hashes_without_reading_validation_arrays(self):
        from knee.fresh_reader_probe import verify_probe_cache
        from knee.fresh_reader import load_cached_batch
        from knee.validation import file_sha256
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'arrays').mkdir()
            rows = [{'uid': 'train'+str(i), 'split':'train', 'labels':[.1+.8*i]*12} for i in range(2)]
            rows.append({'uid':'validation','split':'weak_validation','labels':[.5]*12})
            plan = {'records':rows, 'config':copy.deepcopy(CONFIG), 'targets':['t'+str(i) for i in range(12)],
                    'source_hashes':SOURCE_HASHES, 'preprocessing':PREPROCESS_VERSION}
            plan['fingerprint'] = fingerprint(plan)
            (root/'plan-private.json').write_text(json.dumps(plan))
            (root/'cache-pilot.json').write_text(json.dumps({'status':'PASS','failures':0,
                'plan_fingerprint':plan['fingerprint'],'cached_studies':3,'gate':{'passed':False}}))
            index = {'validation': {'file':'must-not-be-read.npz'}}
            for row in rows[:2]:
                name = hashlib.sha256(row['uid'].encode()).hexdigest()+'.npz'
                x = np.zeros((6,4,3,224,224),np.uint8); x[0] = 1
                mask = np.array([True,False,False,False,False,False])
                pos = np.zeros((6,4),np.float32); pos[0] = [.2,.4,.6,.8]
                np.savez_compressed(root/'arrays'/name,x=x,mask=mask,pos=pos)
                index[row['uid']] = {'file':name,'bytes':(root/'arrays'/name).stat().st_size,
                                     'sha256':file_sha256(root/'arrays'/name)}
            (root/'cache-index-private.json').write_text(json.dumps(index))
            actual, receipt, loaded_index, selected = verify_probe_cache(root, plan['fingerprint'], count=2)
            self.assertFalse(receipt['gate']['passed'])  # CPU time gate is preserved, not repurposed.
            self.assertTrue(all(row['split']=='train' for row in selected))
            x, mask, pos, y = load_cached_batch(selected, loaded_index, root, [0,1], device='cpu')
            self.assertEqual(tuple(x.shape), (2,6,4,3,224,224))
            self.assertTrue(mask[:,0].all()); self.assertFalse(mask[:,1:].any())
            self.assertEqual(tuple(y.shape), (2,12))
            path = root/'arrays'/index[rows[0]['uid']]['file']; path.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'array changed'):
                verify_probe_cache(root, plan['fingerprint'], count=2)

    def test_no_cuda_fails_and_writes_only_diagnostic_report(self):
        from knee.fresh_reader_probe import probe
        with tempfile.TemporaryDirectory() as d, patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'no CPU fallback'):
                probe('unused','unused',d,'expected')
            self.assertEqual([p.name for p in Path(d).iterdir()], ['probe-report.json'])
            self.assertEqual(json.loads((Path(d)/'probe-report.json').read_text())['status'],'FAILED')


if __name__ == '__main__':
    unittest.main()
