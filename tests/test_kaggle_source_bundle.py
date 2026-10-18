"""Offline source transport preserves bytes and rejects altered identity."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('source_bundle_builder', ROOT / 'scripts/build_kaggle.py')
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class SourceBundleTests(unittest.TestCase):
    def test_compressed_roundtrip_and_fail_closed_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            digest = BUILDER.build(out / 'build', 'synthetic-source', '', compress_sources=True)
            notebook = out / 'build/notebook.ipynb'
            self.assertLess(notebook.stat().st_size, 900_000)
            code = ''.join(json.loads(notebook.read_text())['cells'][1]['source'])
            bootstrap = code.split('sys.path.insert')[0].replace(
                "'/kaggle/working/project'", repr(str(out / 'project')))
            compile(code, '<compressed-source>', 'exec')
            exec(bootstrap, {})
            for path in (ROOT / 'knee').glob('*.py'):
                self.assertEqual((out / 'project/knee' / path.name).read_bytes(), path.read_bytes())
            self.assertEqual(digest, hashlib.sha256(json.dumps({
                str(p.relative_to(ROOT)): p.read_text()
                for p in sorted((ROOT / 'knee').glob('*.py'))
            }, sort_keys=True).encode()).hexdigest())
            rejected = bootstrap.replace(digest, '0' * 64).replace(
                str(out / 'project'), str(out / 'rejected'))
            with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
                exec(rejected, {})
            self.assertFalse((out / 'rejected').exists())

