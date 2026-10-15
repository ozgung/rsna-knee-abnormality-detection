"""Optional local64 window parity against immutable historical training source."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import random
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT', '/nonexistent'))
TRAINER = EVIDENCE / 'outputs/kaggle/trackA-coatnet224-fold01-20261009/train_knee_patched.py'
READER = EVIDENCE / 'outputs/research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py'
HAVE_ML = all(importlib.util.find_spec(m) is not None for m in ('numpy', 'torch'))


@unittest.skipUnless(HAVE_ML and TRAINER.is_file() and READER.is_file(),
                     'Pinned historical source fixtures and NumPy/PyTorch required')
class Local64WindowParityTests(unittest.TestCase):
    def test_npz_352_matches_pinned_training_and_public_384_windows(self):
        import numpy as np
        import torch
        from torch.nn import functional as F
        from knee import local64_cache as cache, local64_loader as loader
        from knee import reference, raptor64_cache
        text = TRAINER.read_text()
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),
                         'eb3d34ffc6aa6f7c500f8ef2039dba0f313b42e371e075e58d4b7a47dde0bd3c')
        tree = ast.parse(text)
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StudyWindows')
        ns = dict(np=np, torch=torch, F=F, random=random, Dataset=torch.utils.data.Dataset,
                  _roi_square_box=None, _roi_compartments=None)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<pinned-StudyWindows>', 'exec'), ns)
        module, decoder, manifest = reference.audit_source(
            READER.read_text(), profile='raptor',
            checkpoint_sha256=raptor64_cache.V4_CHECKPOINT_SHA256, raptor_windows=42)
        self.assertEqual(decoder, '')
        public = raptor64_cache.load_public(module, manifest)
        rng = np.random.default_rng(42)
        volume = rng.integers(1, 256, (64, 352, 352), dtype=np.uint8)
        mask = np.ones(64, dtype=np.uint8)
        volume[44:52] = 0; mask[44:52] = 0
        # A dark local crop retains source availability, unlike the old pixel-derived mask.
        volume[5] = 0
        joint = np.zeros((352, 352), dtype=np.uint8)
        previous_threads = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'study.npz'
                hashes = cache.save_npz(path, volume, mask, joint)
                loaded, available, _ = loader.load_study(path, expected_hashes=hashes)
                labels = {'fixture-local64': np.zeros(12, np.float32)}
                ds = ns['StudyWindows']('', ['fixture-local64'], {'fixture-local64': 0}, labels,
                                        384, 42, train=False, aug=False, norm='imagenet')
                ds.vols, ds.masks = loaded[None], available[None]
                historical, target = ds[0]
                current = loader.eval_windows(public, loaded, available, k=42)
                self.assertEqual(tuple(current.shape), (42, 3, 384, 384))
                self.assertTrue(torch.equal(current, historical))
                self.assertTrue(torch.isfinite(current).all())
                self.assertEqual(tuple(target.shape), (12,))
                self.assertEqual(int(available[5]), 1)
                # Training's K8 interface also consumes the serialized shape unchanged.
                ds.train = True; ds.k = 8
                random.seed(42); from_cache, _ = ds[0]
                ds.vols, ds.masks = volume[None], mask[None]
                random.seed(42); from_dense, _ = ds[0]
                self.assertEqual(tuple(from_cache.shape), (8, 3, 384, 384))
                self.assertTrue(torch.equal(from_cache, from_dense))
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
