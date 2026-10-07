"""Synthetic contracts for the bounded fresh reader; default tests need no ML."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from knee.fresh_reader import (CONFIG, budget_gate, check_sources, fingerprint, grouped_subset,
                               pilot_order, validate_plan)
try:
    import numpy as np
    import torch
    import pandas as pd
except ImportError:
    np = None


class FreshReaderCoreTests(unittest.TestCase):
    def test_subset_preserves_roles_and_indivisible_groups(self):
        rows = [{'uid': str(i), 'group': str(i // 2), 'split': 'train' if i < 6 else 'weak_validation'}
                for i in range(10)]
        train = grouped_subset(rows, 'train', 3)
        val = grouped_subset(rows, 'weak_validation', 3)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(val), 2)
        self.assertFalse({r['group'] for r in train} & {r['group'] for r in val})
        self.assertTrue(all(r['split'] == 'train' for r in train))

    def test_pilot_order_does_not_depend_on_labels_or_input_order(self):
        rows = [{'uid': str(i), 'labels': [0]} for i in range(100)]
        expected = [r['uid'] for r in pilot_order(rows, 64)]
        changed = [dict(r, labels=[1]) for r in reversed(rows)]
        self.assertEqual(expected, [r['uid'] for r in pilot_order(changed, 64)])

    def test_budget_gate_refuses_time_or_disk_expansion(self):
        kw = dict(count=64, total=712, seconds=100, byte_count=1000,
                  seconds_limit=1800, bytes_limit=15000)
        self.assertTrue(budget_gate(**kw)['passed'])
        self.assertFalse(budget_gate(**dict(kw, seconds_limit=1000))['passed'])
        self.assertFalse(budget_gate(**dict(kw, bytes_limit=12000))['passed'])
        with self.assertRaises(ValueError):
            budget_gate(**dict(kw, count=0))

    def test_unreviewed_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'preprocess.py').write_text('pass\n')
            with self.assertRaisesRegex(ValueError, 'Unreviewed'):
                check_sources(d)

    def test_plan_fingerprint_detects_label_or_config_mutation(self):
        plan = {'labels': [0., 1.], 'config': {'windows': 4}}
        plan['fingerprint'] = fingerprint(plan)
        validate_plan(plan)
        plan['labels'][0] = .5
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            validate_plan(plan)

    def test_builder_enforces_cache_gate_and_private_stage_settings(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('fresh_builder_test', root/'scripts/build_fresh_reader.py')
        builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as d:
            sources = Path(d)/'source'; sources.mkdir()
            for name in ('preprocess.py','knee.py'): (sources/name).write_text('pass\n')
            with patch.object(builder, 'check_sources'):
                with self.assertRaisesRegex(ValueError, 'reviewed plan'):
                    builder.build(sources, Path(d)/'bad', 'synthetic', 'train')
                for stage in ('cache-pilot', 'cache', 'train'):
                    out = Path(d)/stage
                    builder.build(sources, out, 'synthetic', stage,
                                  cache_kernel=None if stage == 'cache-pilot' else 'synthetic/cache',
                                  plan_fingerprint=None if stage == 'cache-pilot' else 'abc')
                    meta = json.loads((out/'kernel-metadata.json').read_text())
                    self.assertTrue(meta['is_private'])
                    self.assertEqual(meta['enable_gpu'], stage == 'train')
                    self.assertEqual(meta['enable_internet'], stage == 'train')
                    nb = json.loads((out/'notebook.ipynb').read_text())
                    compile(''.join(nb['cells'][1]['source']), 'generated', 'exec')

    def test_training_bounds_are_predeclared(self):
        self.assertEqual(CONFIG['image_size'], 224)
        self.assertEqual(CONFIG['windows'], 4)
        self.assertLessEqual(CONFIG['training_seconds'], 5400)
        self.assertLessEqual(CONFIG['max_epochs'], 4)


@unittest.skipIf(np is None, 'Optional ML dependencies unavailable')
class FreshReaderMLTests(unittest.TestCase):
    def test_masked_soft_loss_has_zero_unknown_gradient(self):
        from knee.fresh_reader import loss_terms, macro_loss, training_prior
        logits = torch.tensor([[1000., 2., -1000.]], requires_grad=True)
        labels = torch.tensor([[0., .5, 1.]])
        n, d = loss_terms(logits, labels)
        self.assertTrue(torch.isfinite(n).all())
        macro_loss(n, d).backward()
        self.assertEqual(float(logits.grad[0, 1]), 0.)
        prior, support = training_prior([[0., .9], [1., .5]])
        np.testing.assert_allclose(prior, [.5, .9])
        np.testing.assert_allclose(support, [2., .8])
        with self.assertRaisesRegex(ValueError, 'no known'):
            training_prior([[.5, .9], [.5, .1]])

    def test_timing_subset_can_have_unknown_target_without_nan(self):
        from knee.fresh_reader import loss_terms, macro_loss, per_target_bce
        from knee.features import masked_loss
        y = torch.tensor([[.1, .5]] * 16 + [[.9, 1.]])
        logits = torch.zeros_like(y)
        n, d = loss_terms(logits[:16], y[:16])
        self.assertTrue(torch.isfinite(macro_loss(n, d)))
        self.assertIsNone(per_target_bce(n, d)[1])
        json.dumps(per_target_bce(n, d), allow_nan=False)
        full_n, full_d = loss_terms(logits, y)
        self.assertGreater(float(full_d[1]), 0.)
        torch.testing.assert_close(macro_loss(full_n, full_d),
                                   masked_loss(logits, y, (y - .5).abs() * 2))

    def test_rng_checkpoint_resumes_same_optimizer_trajectory(self):
        from knee.fresh_reader import rng_state, restore_rng, save_checkpoint
        random.seed(4); np.random.seed(4); torch.manual_seed(4)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        def step():
            optimizer.zero_grad()
            x = torch.randn(2, 2) + random.random() + float(np.random.rand())
            loss = model(x).square().mean()
            loss.backward(); optimizer.step()
        step()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'last.pt'
            save_checkpoint(path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng_state()})
            step()
            expected = copy.deepcopy(model.state_dict())
            state = torch.load(path, weights_only=False)
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            restore_rng(state['rng']); step()
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key], atol=0, rtol=0)

    def test_smaller_holdout_does_not_reassign_baseline_validation(self):
        from knee.fresh_reader import make_plan
        from knee.experiment import choose_studies
        from knee.validation import make_gold_holdout
        targets = [f't{i}' for i in range(12)]
        rows = [{'StudyInstanceUID': str(i), 'Report': f'report {i}', **{t: '' for t in targets}}
                for i in range(20)]
        rows.append({'StudyInstanceUID': 'gold', 'Report': 'gold report', **{t: '1' for t in targets}})
        rows.append({'StudyInstanceUID': 'duplicate', 'Report': 'gold report', **{t: '' for t in targets}})
        def write(path, columns, values):
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(values)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            write(p/'train.csv', ['StudyInstanceUID', 'Report'] + targets, rows)
            write(p/'sample_submission.csv', ['StudyInstanceUID'] + targets,
                  [{'StudyInstanceUID': 'test', **{t: '.5' for t in targets}}])
            write(p/'train_series.csv', ['StudyInstanceUID', 'SeriesInstanceUID'], [])
            write(p/'weak.csv', ['StudyInstanceUID'] + targets,
                  [{'StudyInstanceUID': r['StudyInstanceUID'], **{t: '.2' for t in targets}} for r in rows])
            config = CONFIG | {'baseline_train_limit': 8, 'baseline_validation_limit': 6,
                               'train_limit': 4, 'validation_limit': 2}
            from knee.validation import file_sha256
            with patch('knee.fresh_reader.WEAK_LABELS_SHA256', file_sha256(p/'weak.csv')):
                plan = make_plan(p, p/'weak.csv', config)
            assignments, _ = make_gold_holdout(rows, targets, 'StudyInstanceUID')
            selected, roles = choose_studies(rows, assignments, 8, 6)
            original_val = {r['StudyInstanceUID'] for r, role in zip(selected, roles) if role == 'weak_validation'}
            train_ids = {r['uid'] for r in plan['records'] if r['split'] == 'train'}
            self.assertFalse(train_ids & original_val)
            self.assertFalse({'gold', 'duplicate'} & {r['uid'] for r in plan['records']})
            self.assertEqual(plan['gold_ids'], ['gold'])

    def test_cache_masks_windows_resume_and_strict_decode_failure(self):
        from knee.fresh_reader import prepare_cache
        labels = [f't{i}' for i in range(12)]
        plan = {'records': [{'uid': 'a', 'split': 'train', 'group': 'a', 'labels': [.1]*12},
                            {'uid': 'b', 'split': 'weak_validation', 'group': 'b', 'labels': [.9]*12}],
                'gold_ids': [], 'targets': labels, 'config': CONFIG, 'source_hashes': {}, 'files': {}}
        volume = np.stack([np.full((224, 224), 30*i, np.uint8) for i in range(6)])
        prep = SimpleNamespace(MAX_SLICES=64, load_series=lambda *a: (volume, {'n_bad': 0}))
        code = SimpleNamespace(LABELS=labels, PLANES=['Sagittal','Coronal','Axial'],
                               window_centres=lambda n,k,t: np.array([1,2,3,4]),
                               build_study_table=lambda *a: {'a': [['series-a'],[],[],[],[],[]],
                                                           'b': [[],['series-b'],[],[],[],[]]})
        resize = SimpleNamespace(INTER_AREA=1, resize=lambda a,*args,**kwargs: a.copy())
        def source(*args):
            return prep if args[-1] == 'preprocess' else code
        with tempfile.TemporaryDirectory() as d:
            data, out = Path(d)/'data', Path(d)/'cache'
            data.mkdir()
            for uid in ('a','b'):
                folder = data/'train_series'/uid/('series-'+uid); folder.mkdir(parents=True)
                for z in range(3): (folder/str(z)).write_bytes(b'fixture')
            pd.DataFrame([{'StudyInstanceUID': uid, 'SeriesInstanceUID': 'series-'+uid} for uid in ('a','b')]).to_csv(data/'train_series.csv',index=False)
            with patch.dict('sys.modules', {'cv2': resize}), patch('knee.fresh_reader.load_source', side_effect=source), \
                 patch('knee.fresh_reader.make_plan', side_effect=lambda *a: copy.deepcopy(plan)), \
                 patch('knee.fresh_reader.patient_exclusions', side_effect=lambda p,*a,**kw: p):
                result = prepare_cache(data, 'unused', 'unused', out)
                self.assertEqual(result['status'], 'PASS')
                index = json.loads((out/'cache-index-private.json').read_text())
                with np.load(out/'arrays'/index['a']['file']) as z:
                    self.assertEqual(z['mask'].tolist(), [True,False,False,False,False,False])
                    np.testing.assert_array_equal(z['x'][0,0,:,0,0], [0,30,60])
                    self.assertFalse(z['x'][1:].any())
                    self.assertEqual(z['x'].shape, (6,4,3,224,224))
                again = prepare_cache(data, 'unused', 'unused', out, pilot=False)
                self.assertEqual(again['new_studies'], 0)
                bad_out = Path(d)/'bad-cache'
                prep.load_series = lambda *a: (volume, {'n_bad': 1})
                with self.assertRaisesRegex(ValueError, 'strict decoding'):
                    prepare_cache(data, 'unused', 'unused', bad_out)
                self.assertEqual(json.loads((bad_out/'cache-pilot.json').read_text())['status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
