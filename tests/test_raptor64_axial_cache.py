"""it4 derived axial-only cache: the axial arms must see byte-identical inputs (synthetic data only)."""
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT', '/nonexistent'))
HISTORICAL = EVIDENCE / 'outputs/kaggle/trackA-coatnet224-fold01-20261009'
HAVE = all(importlib.util.find_spec(p) for p in ('numpy', 'torch')) and (HISTORICAL / 'train_knee_patched.py').is_file()


def load():
    spec = importlib.util.spec_from_file_location('raptor64_axial_cache', ROOT / 'scripts/raptor64_axial_cache.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy required')
class DerivedArraysTests(unittest.TestCase):
    def test_axial_bytes_kept_rest_zeroed_and_loader_schema(self):
        import numpy as np
        from knee import raptor64_cache as cache
        ax = load(); rng = np.random.default_rng(0)
        v = rng.integers(1, 255, (64, 336, 336), dtype=np.uint8); m = np.ones(64, np.uint8); v[44:48] = 0; m[44:48] = 0
        dv, dm = ax.derive_arrays(v, m)
        self.assertEqual(dv[52:64].tobytes(), v[52:64].tobytes()); self.assertEqual(dm[52:64].tobytes(), m[52:64].tobytes())
        self.assertFalse(dv[:52].any()); self.assertFalse(dm[:52].any())
        cache.validate_arrays(dv, dm)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '000000.npz'; ax.save_npz(path, dv, dm)
            rv, rm = cache.read_bounded(path)
            self.assertTrue(np.array_equal(rv, dv) and np.array_equal(rm, dm))
            self.assertLess(path.stat().st_size, 64 * 336 * 336 // 3)
        empty = v.copy(); empty[52:64] = 0; me = m.copy(); me[52:64] = 0
        with self.assertRaises(ValueError):
            ax.derive_arrays(empty, me)


@unittest.skipUnless(HAVE, 'NumPy/PyTorch and the pinned historical trainer required')
class DerivedInputsIdenticalTests(unittest.TestCase):
    def test_axial_arms_see_identical_windows_targets_and_masks(self):
        import numpy as np
        import torch
        from knee import raptor64_training as guard, public_training_smoke as smoke
        import tests.test_raptor64_training as base_tests
        ax = load(); rng = np.random.default_rng(1)
        ids = ['a', 'b', 'c']; id2row = {u: i for i, u in enumerate(ids)}
        vols = rng.integers(1, 255, (3, 64, 336, 336), dtype=np.uint8); masks = np.ones((3, 64), np.uint8)
        vols[1, 60:64] = 0; masks[1, 60:64] = 0                      # partial axial block
        derived = [ax.derive_arrays(vols[i], masks[i]) for i in range(3)]
        dvols = np.stack([d[0] for d in derived]); dmasks = np.stack([d[1] for d in derived])
        labels = {u: np.linspace(0, 1, 12).astype(np.float32) * (i + 1) / 3 for i, u in enumerate(ids)}
        probs = rng.uniform(0, 1, (3, 12)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 't.npz'; np.savez(path, ids=np.asarray(ids), probs=probs)
            SW, _, _ = base_tests.historical_classes()
            for arm in ('ax-crop', 'ax-full'):
                policy = {'variant': 'axial', 'axial': guard.AXIAL_SPECS[arm], 'teacher': {'sha256': smoke.file_sha(path), 'weight': 0.5}}
                with patch.dict(os.environ, {'RAPTOR64_TEACHER': str(path)}), patch.object(guard, '_CORPUS', ('v', 'm', np.asarray(ids))), \
                        patch.object(guard, '_CACHE_IDENTITY', {'fixture': False}), patch.object(guard, 'recipe', return_value=policy), \
                        patch.object(guard, '_AXIAL', {u: True for u in ids}):
                    AX = guard.recipe_variant(SW, 'x')
                    for train, k in ((True, 10), (False, 10)):
                        full = AX('', ids, id2row, labels, 32, k, train=train, norm='imagenet'); full.vols, full.masks = vols, masks
                        der = AX('', ids, id2row, labels, 32, k, train=train, norm='imagenet'); der.vols, der.masks = dvols, dmasks
                        for i in range(3):
                            random.seed(100 + i); xf, yf = full[i]
                            random.seed(100 + i); xd, yd = der[i]
                            self.assertTrue(torch.equal(xf, xd), f'{arm} train={train} windows differ')
                            self.assertTrue(torch.equal(yf, yd), f'{arm} train={train} targets differ')
                            self.assertTrue(np.array_equal(full.masks[i], der.masks[i]))



@unittest.skipUnless(importlib.util.find_spec('numpy'), 'NumPy required')
class DeriveAndStageTests(unittest.TestCase):
    """Synthetic four-shard cache -> derive (kernel step) -> stage (VM step) -> unchanged loader opens it."""
    def test_derive_then_stage_round_trip(self):
        import numpy as np
        from knee import raptor64_cache as cache, raptor64_loader as loader, public_training_smoke as smoke
        ax = load(); rng = np.random.default_rng(2)
        ids = [f's{i}' for i in range(8)]; folds = {u: (0 if i < 3 else 1 + i % 4) for i, u in enumerate(ids)}
        train = [u for u in ids if folds[u] != 0]; dev = [u for u in ids if folds[u] == 0]
        membership = {'eligible': 8, 'training': len(train), 'development': len(dev), 'ordered_eligible_sha256': loader.digest_document(ids),
                      'ordered_training_sha256': loader.digest_document(train), 'ordered_development_sha256': loader.digest_document(dev)}
        validation = {'fixture': True, 'membership': membership, 'folds': folds}
        slots = [{'plane': p, 'selected': True, 'fluid_sensitive': int(f == 1), 'frames': 20, 'count': c} for p, f, c in cache.SLOTS]
        parts = [{'studies': 2, 'ordinals_sha256': loader.digest_document(list(range(s, 8, 4))),
                  'ordered_ids_sha256': loader.digest_document([ids[n] for n in range(s, 8, 4)])} for s in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); src = tmp / 'input'; pins = {}; originals = {}
            for s in range(4):
                root = src / f'shard{s}/raptor64_cache'; root.mkdir(parents=True); entries = []
                for n in range(s, 8, 4):
                    v = rng.integers(1, 255, (64, 336, 336), dtype=np.uint8); m = np.ones(64, np.uint8); originals[n] = (v, m)
                    saved = cache.atomic_npz(root / cache.filename(n), v, m)
                    entries.append({'ordinal': n, 'StudyInstanceUID': ids[n], 'file': cache.filename(n), **saved, 'windows': {}, 'slots': slots})
                contract = {'version': cache.VERSION, 'config': cache.CONFIG, 'membership': membership, 'shard_index': s, 'shard_count': 4,
                            'partitions': parts, 'reader': {}, 'input_pins': {}, 'qualified_environment': {}, 'sources': {}}
                cache.atomic_json(root / 'run-contract.json', contract)
                cache.atomic_json(root / 'index-private.json', {'version': cache.VERSION, 'contract_sha256': loader.digest_document(contract), 'entries': entries})
                cache.atomic_json(root / 'receipt.json', {'status': 'PASS', 'version': cache.VERSION, 'contract_sha256': loader.digest_document(contract),
                                  'private_index_sha256': smoke.file_sha(root / 'index-private.json'), 'completed_studies': 2, 'shard_index': s, 'shard_count': 4})
                pins[str(s)] = {'contract_file_sha256': smoke.file_sha(root / 'run-contract.json'), 'index_sha256': smoke.file_sha(root / 'index-private.json'),
                                'receipt_sha256': smoke.file_sha(root / 'receipt.json'), 'ref': f'fixture/shard{s}'}
            out = tmp / 'working'; out.mkdir()
            result = ax.derive(src, out, pins, workers=1)
            self.assertEqual(result['studies'], 8); self.assertEqual(sorted(p.name for p in out.iterdir()),
                                                                     ['axial-derived-pins.json'] + [f'axial_shard{s}.tar' for s in range(4)])
            config = tmp / 'stage.json'; config.write_text(json.dumps({'validation': validation}))
            pins_path = out / 'axial-derived-pins.json'
            with self.assertRaises(ValueError):
                ax.stage(out, tmp / 'bad', pins_path, '0' * 64, config)
            receipt = ax.stage(out, tmp / 'cache', pins_path, smoke.file_sha(pins_path), config)
            self.assertEqual((receipt['status'], receipt['studies']), ('PASS', 8))
            manifest = json.loads((tmp / 'cache/cache64-manifest-private.json').read_bytes())
            vols, masks, got = loader.open_corpus(manifest, expected_manifest_sha256=receipt['manifest_sha256'], allow_fixture=True)
            self.assertEqual(list(got), ids)
            for n in range(8):
                self.assertEqual(vols[n][52:64].tobytes(), originals[n][0][52:64].tobytes())
                self.assertEqual(masks[n][52:64].tobytes(), originals[n][1][52:64].tobytes())
                self.assertFalse(vols[n][:52].any())
            again = tmp / 'cache2'
            self.assertEqual(ax.stage(out, again, pins_path, smoke.file_sha(pins_path), config)['content_sha256'], receipt['content_sha256'])


if __name__ == '__main__':
    unittest.main()
