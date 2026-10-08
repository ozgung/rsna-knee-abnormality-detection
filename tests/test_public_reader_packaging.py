import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import copy

from knee.public_reader_contract import INPUT_POLICY_VERSION, STRESS_CASES

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('public_reader_builder', ROOT/'scripts/build_public_reader.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class ReaderPackagingTests(unittest.TestCase):
    def fixtures(self, directory):
        src = Path(directory)/'source'; src.mkdir()
        hashes={}
        for name in ['knee.py','preprocess.py']:
            data='"""Synthetic source fixture."""\nVALUE = 1\n'
            (src/name).write_text(data)
            hashes[name]=hashlib.sha256(data.encode()).hexdigest()
        return src,hashes

    def test_unreviewed_source_fails_before_package_write(self):
        with tempfile.TemporaryDirectory() as d:
            src,hashes=self.fixtures(d); (src/'knee.py').write_text('raise RuntimeError("changed")')
            dest=Path(d)/'dest'
            with patch.object(builder,'SOURCE_HASHES',hashes), self.assertRaisesRegex(ValueError,'Unreviewed'):
                builder.build(src,dest,'test')
            self.assertFalse(dest.exists())

    def test_submission_requires_all_rehearsed_checkpoint_identities(self):
        with tempfile.TemporaryDirectory() as d:
            src,hashes=self.fixtures(d)
            with patch.object(builder,'SOURCE_HASHES',hashes):
                for weights in [None,{'cnxt_v0_fold0.pt':'a'*64}]:
                    with self.assertRaisesRegex(ValueError,'all three'):
                        builder.build(src,Path(d)/'dest','test',rehearsal=False,weight_hashes=weights)

    def test_rehearsal_is_private_offline_and_compiles(self):
        with tempfile.TemporaryDirectory() as d:
            src,hashes=self.fixtures(d); dest=Path(d)/'dest'
            with patch.object(builder,'SOURCE_HASHES',hashes):
                builder.build(src,dest,'test')
            meta=json.loads((dest/'kernel-metadata.json').read_text())
            self.assertTrue(meta['is_private']); self.assertFalse(meta['enable_internet'])
            self.assertEqual(meta['dataset_sources'],['goodpjw2008/rsna-knee-2-5d-convnext-reader'])
            notebook=json.loads((dest/'notebook.ipynb').read_text())
            self.assertEqual(notebook['metadata']['kernelspec']['name'],'python3')
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    source = ''.join(cell['source'])
                    compile(source,'generated','exec')
                    self.assertIn('ENTRY_SHA256 = ', source)
            contract=json.loads((dest/'build-contract.json').read_text())
            self.assertEqual(contract['input_policy_version'], INPUT_POLICY_VERSION)
            self.assertEqual(contract['train_studies_requested'], 64)

    def test_full_entry_requires_fresh_expanded_rehearsal_not_just_weight_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            src, hashes = self.fixtures(directory)
            root = Path(directory)
            weights = {f'cnxt_v0_fold{i}.pt': str(i)*64 for i in range(3)}
            report = dict(status='PASS', rehearsal=True, source_hashes=hashes, weight_hashes=weights,
                entry_sha256=hashlib.sha256((ROOT/'scripts/public_reader_entry.py').read_bytes()).hexdigest(),
                input_policy_version=INPUT_POLICY_VERSION, train_studies_requested=64,
                studies=67, test_studies=3, sample_fingerprint='a'*64, zero_studies=0, resource_errors=0,
                repeat_max_abs_error=0, cache_parity_max_abs_error=0, elapsed_seconds=500,
                decode_failures=3, partial_series=1, failed_series=1, short_series=1,
                zero_decoded_series=0, missing_slots=10, selected_series=392,
                stress_checks=dict(status='PASS', uses_raw_data=False, uses_labels=False,
                                   cases=dict.fromkeys(STRESS_CASES, 'PASS')))
            path = root/'receipt.json'
            with patch.object(builder, 'SOURCE_HASHES', hashes):
                with self.assertRaisesRegex(ValueError, 'expanded rehearsal'):
                    builder.build(src,root/'none','test',rehearsal=False,weight_hashes=weights)
                for key, value in [('entry_sha256', 'b'*64), ('train_studies_requested',16),
                                   ('stress_checks',{}), ('resource_errors',1), ('elapsed_seconds',3501),
                                   ('weight_hashes',{**weights,'cnxt_v0_fold0.pt':'f'*64})]:
                    wrong=copy.deepcopy(report); wrong[key]=value; path.write_text(json.dumps(wrong))
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        builder.build(src,root/'wrong','test',rehearsal=False,weight_hashes=weights,rehearsal_report=path)
                    self.assertFalse((root/'wrong').exists())
                path.write_text(json.dumps(report))
                builder.build(src,root/'valid','test',rehearsal=False,weight_hashes=weights,rehearsal_report=path)
            contract=json.loads((root/'valid/build-contract.json').read_text())
            self.assertEqual(contract['rehearsal_report_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
