import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import hashlib

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
                if cell['cell_type']=='code': compile(''.join(cell['source']),'generated','exec')


if __name__ == '__main__':
    unittest.main()
