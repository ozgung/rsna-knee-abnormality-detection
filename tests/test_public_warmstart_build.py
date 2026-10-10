import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class PublicWarmstartBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        import sys
        sys.path.insert(0, str(root / 'scripts'))
        spec = importlib.util.spec_from_file_location('warmstart_builder', root / 'scripts/build_public_warmstart_smoke.py')
        cls.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.builder)

    def test_modified_public_source_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.py'
            source.write_text('print("unreviewed")')
            destination = Path(directory) / 'build'
            with self.assertRaisesRegex(ValueError, 'audited snapshot'):
                self.builder.build_smoke(source, destination, 'test-warmstart')
            self.assertFalse(destination.exists())

    def test_existing_build_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'receipt.json'
            marker.write_text('preserved')
            with self.assertRaisesRegex(ValueError, 'immutable'):
                self.builder.build_smoke('missing-source', directory, 'test-warmstart')
            self.assertEqual(marker.read_text(), 'preserved')

    def test_package_is_offline_image_only_and_has_no_model_download(self):
        # This tests packaging permissions, not checkpoint or GPU execution.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.py'
            source.write_text('mocked audited fixture')
            destination = Path(directory) / 'build'
            manifest = {'attribution': 'fixture', 'source_url': 'https://example.test/audited',
                        'weight_dataset': self.builder.WEIGHT_DATASET, 'checkpoint_file': 'fixture.pt'}
            with patch.object(self.builder, 'audit_source', return_value=('', '', manifest)):
                self.builder.build_smoke(source, destination, 'test-warmstart')
            metadata = json.loads((destination / 'kernel-metadata.json').read_text())
            self.assertFalse(metadata['enable_internet'])
            self.assertTrue(metadata['is_private'])
            self.assertEqual(metadata['model_sources'], [])
            self.assertEqual(metadata['kernel_sources'], [])
            self.assertEqual(metadata['dataset_sources'], [self.builder.WEIGHT_DATASET, self.builder.CONVNEXT_DATASET])
            nb = json.loads((destination / 'notebook.ipynb').read_text())
            code = ''.join(nb['cells'][1]['source'])
            compile(code, '<test-build>', 'exec')
            self.assertIn('EXPECTED_CODECS=', code)
            entry = code.split('PUBLIC_SOURCE=', 1)[1]
            self.assertNotIn('pretrained=True', entry)
            self.assertNotIn('stevenleehans', json.dumps(metadata))


if __name__ == '__main__':
    unittest.main()
