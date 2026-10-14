"""Iteration-2 single-factor variants (M1 mirror, M2 distil): synthetic tests only."""
import importlib.util
import json
import os
from pathlib import Path
import random
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT', '/nonexistent'))
HISTORICAL = EVIDENCE / 'outputs/kaggle/trackA-coatnet224-fold01-20261009'
BUNDLE_V1 = EVIDENCE / 'outputs/analysis/raptor64-20261013/colab-bundle-v1/raptor64-colab-bundle.tar.gz'
HAVE = all(importlib.util.find_spec(p) for p in ('numpy', 'torch')) and (HISTORICAL / 'train_knee_patched.py').is_file()

if HAVE:
    from knee import raptor64_training as guard, public_training_smoke as smoke
    import tests.test_raptor64_training as base_tests


@unittest.skipUnless(HAVE, 'NumPy/PyTorch and the pinned historical trainer required')
class MirrorTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        rng = np.random.default_rng(0)
        self.vol = rng.integers(0, 255, (64, 336, 336), dtype=np.uint8)
        self.mask = np.ones(64, np.uint8); self.mask[44:52] = 0; self.vol[44:52] = 0

    def test_mirror_twice_is_identity_and_geometry(self):
        import numpy as np
        m = guard.mirror_volume(self.vol)
        self.assertTrue(np.array_equal(guard.mirror_volume(m), self.vol))
        self.assertTrue(np.array_equal(m[0], self.vol[17])); self.assertTrue(np.array_equal(m[18], self.vol[31]))
        self.assertTrue(np.array_equal(m[40], self.vol[40][:, ::-1])); self.assertTrue(np.array_equal(m[63], self.vol[63][:, ::-1]))
        mm = guard.mirror_volume(self.mask)
        self.assertTrue(np.array_equal(guard.mirror_volume(mm), self.mask)); self.assertTrue(np.array_equal(mm[32:], self.mask[32:]))
        y = np.arange(12)
        self.assertEqual(guard.mirror_targets(y).tolist(), [0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11])
        self.assertTrue(np.array_equal(guard.mirror_targets(guard.mirror_targets(y)), y))
        self.assertEqual(guard.LAB[guard.MCL_INDEX], 'MCL')
        self.assertEqual([guard.LAB[i] for i in (2, 3, 4, 5)], ['Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA'])

    def datasets(self, train, k):
        import numpy as np
        SW, _, _ = base_tests.historical_classes()
        policy = {'variant': 'mirror'}
        with patch.object(guard, 'recipe', return_value=policy):
            MSW = guard.recipe_variant(SW, 'x')
        ids = ['a', 'b']; id2row = {'a': 0, 'b': 1}; labels = {u: np.linspace(0, 1, 12).astype(np.float32) for u in ids}
        vols = np.stack([self.vol, self.vol[::-1].copy()]); masks = np.stack([self.mask, self.mask])
        out = []
        for cls in (SW, MSW):
            ds = cls('', ids, id2row, labels, 32, k, train=train, aug=train, norm='imagenet'); ds.vols, ds.masks = vols, masks; out.append(ds)
        return out

    def test_sampler_and_window_count_unchanged(self):
        import torch
        base, mirror = self.datasets(True, 8)
        for i in range(2):
            random.seed(11 + i); draw = random.random(); x0, y0 = base[i]          # base after consuming the mirror draw
            random.seed(11 + i); x1, y1 = mirror[i]
            self.assertEqual(tuple(x1.shape), (8, 3, 32, 32)); self.assertEqual(tuple(y1.shape), (24,))
            if draw >= guard.MIRROR_P:
                self.assertTrue(torch.equal(x0, x1)); self.assertTrue(torch.equal(y1[:12], y0)); self.assertTrue(bool((y1[12:] == 1).all()))
            else:
                self.assertTrue(torch.equal(y1[:12], y0[guard.SIDE_PERM])); self.assertEqual(float(y1[12 + guard.MCL_INDEX]), 0.0)
        with patch('random.random', return_value=0.0):   # always mirrored
            x, y = mirror[0]
        self.assertEqual(tuple(x.shape), (8, 3, 32, 32))
        eb, em = self.datasets(False, 42)
        for i in range(2):
            self.assertTrue(torch.equal(eb[i][0], em[i][0])); self.assertTrue(torch.equal(eb[i][1], em[i][1]))
        em.force_mirror = True
        ref = eb; ref.vols = __import__('numpy').stack([guard.mirror_volume(v) for v in eb.vols]); ref.masks = __import__('numpy').stack([guard.mirror_volume(m) for m in eb.masks])
        self.assertTrue(torch.equal(em[0][0], ref[0][0]))

    def test_masked_loss_equals_historical_when_unmasked(self):
        import torch
        torch.manual_seed(0)
        logits, target = torch.randn(4, 12), torch.rand(4, 12); pw = torch.rand(12) * 3 + 1
        hist = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        loss = guard.MaskedPosWeightBCE(pw)
        self.assertAlmostEqual(float(loss(logits, torch.cat([target, torch.ones(4, 12)], 1))), float(hist(logits, target)), places=6)
        self.assertAlmostEqual(float(loss(logits, target)), float(hist(logits, target)), places=6)
        mask = torch.ones(4, 12); mask[:, guard.MCL_INDEX] = 0
        manual = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction='none')
        self.assertAlmostEqual(float(loss(logits, torch.cat([target, mask], 1))), float((manual * mask).sum() / mask.sum()), places=6)


@unittest.skipUnless(HAVE, 'NumPy/PyTorch and the pinned historical trainer required')
class DistilTests(unittest.TestCase):
    def test_distil_loss_and_teacher_pinning(self):
        import numpy as np
        import torch
        torch.manual_seed(0)
        logits, target, teacher = torch.randn(4, 12), torch.rand(4, 12), torch.rand(4, 12); pw = torch.rand(12) + 1
        loss = guard.DistilBCE(pw, 0.5)
        expected = torch.nn.BCEWithLogitsLoss(pos_weight=pw)(logits, target) + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(logits, teacher)
        self.assertAlmostEqual(float(loss(logits, torch.cat([target, teacher], 1))), float(expected), places=6)
        with tempfile.TemporaryDirectory() as tmp:
            ids = np.asarray(['a', 'b', 'c']); probs = np.random.default_rng(0).uniform(0, 1, (3, 12)).astype(np.float32)
            path = Path(tmp) / 't.npz'; np.savez(path, ids=ids, probs=probs)
            policy = {'teacher': {'sha256': smoke.file_sha(path)}}
            with patch.dict(os.environ, {'RAPTOR64_TEACHER': str(path)}), patch.object(guard, '_CORPUS', ('v', 'm', ids)), patch.object(guard, '_CACHE_IDENTITY', {'fixture': False}):
                teacher_map = guard.load_teacher(policy)
                self.assertTrue(np.array_equal(teacher_map['b'], probs[1]))
                with self.assertRaises(ValueError):
                    guard.load_teacher({'teacher': {'sha256': '0' * 64}})
            with patch.dict(os.environ, {'RAPTOR64_TEACHER': str(path)}), patch.object(guard, '_CORPUS', ('v', 'm', ids[::-1])), patch.object(guard, '_CACHE_IDENTITY', {'fixture': False}):
                with self.assertRaises(ValueError):
                    guard.load_teacher(policy)
            SW, _, _ = base_tests.historical_classes()
            with patch.dict(os.environ, {'RAPTOR64_TEACHER': str(path)}), patch.object(guard, '_CORPUS', ('v', 'm', ids)), patch.object(guard, '_CACHE_IDENTITY', {'fixture': False}), \
                    patch.object(guard, 'recipe', return_value={'variant': 'distil', **policy}):
                DSW = guard.recipe_variant(SW, 'x')
            vols = np.ones((3, 64, 336, 336), np.uint8); masks = np.ones((3, 64), np.uint8)
            labels = {u: np.full(12, .25, np.float32) for u in ids}
            for train, size in ((True, 24), (False, 12)):
                ds = DSW('', list(ids), {u: i for i, u in enumerate(ids)}, labels, 32, 8 if train else 42, train=train, norm='imagenet')
                ds.vols, ds.masks = vols, masks
                x, y = ds[1]
                self.assertEqual(tuple(y.shape), (size,))
                if train:
                    self.assertTrue(np.allclose(y[12:].numpy(), probs[1])); self.assertTrue(np.allclose(y[:12].numpy(), .25))


@unittest.skipUnless(HAVE, 'NumPy/PyTorch and the pinned historical trainer required')
class FlipDistilTests(unittest.TestCase):
    """it3 in-distil-flip: image-only mirror; labels, masks and teacher targets must be unchanged."""
    def test_flip_preserves_labels_teacher_and_window_count(self):
        import numpy as np
        import torch
        rng = np.random.default_rng(0)
        ids = ['a', 'b', 'c']; id2row = {u: i for i, u in enumerate(ids)}
        vols = rng.integers(1, 255, (3, 64, 336, 336), dtype=np.uint8); masks = np.ones((3, 64), np.uint8); masks[:, 44:52] = 0; vols[:, 44:52] = 0
        labels = {u: np.linspace(0, 1, 12).astype(np.float32) * (i + 1) / 3 for i, u in enumerate(ids)}
        probs = rng.uniform(0, 1, (3, 12)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 't.npz'; np.savez(path, ids=np.asarray(ids), probs=probs)
            policy = {'variant': 'distil_flip', 'flip': guard.FLIP_SPEC, 'teacher': {'sha256': smoke.file_sha(path), 'weight': 0.5}}
            SW, _, _ = base_tests.historical_classes()
            with patch.dict(os.environ, {'RAPTOR64_TEACHER': str(path)}), patch.object(guard, '_CORPUS', ('v', 'm', np.asarray(ids))), \
                    patch.object(guard, '_CACHE_IDENTITY', {'fixture': False}), patch.object(guard, 'recipe', return_value=policy):
                FD = guard.recipe_variant(SW, 'x')
            self.assertEqual(guard.FLIP_SPEC['perm'], list(range(12))); self.assertEqual(guard.FLIP_SPEC['masked'], [])
            ds = FD('', ids, id2row, labels, 32, 8, train=True, aug=False, norm='imagenet'); ds.vols, ds.masks = vols, masks
            ref = SW('', ids, id2row, labels, 32, 8, train=True, aug=False, norm='imagenet')
            for forced in (0.0, 0.99):   # always flipped, never flipped
                for i, u in enumerate(ids):
                    random.seed(100 + i)
                    with patch('random.random', return_value=forced):
                        x, y = ds[i]
                    self.assertEqual(tuple(x.shape), (8, 3, 32, 32)); self.assertEqual(tuple(y.shape), (24,))
                    self.assertTrue(np.array_equal(y[:12].numpy(), labels[u]), 'labels changed under flip')
                    self.assertTrue(np.array_equal(y[12:].numpy(), probs[i]), 'teacher targets changed under flip')
                    ref.vols = np.stack([guard.mirror_volume(v) for v in vols]) if forced < 0.5 else vols
                    ref.masks = np.stack([guard.mirror_volume(m) for m in masks]) if forced < 0.5 else masks
                    random.seed(100 + i)
                    with patch('random.random', return_value=forced):
                        xr, _ = ref[i]
                    self.assertTrue(torch.equal(x, xr))
            self.assertTrue(np.array_equal(guard.mirror_volume(guard.mirror_volume(vols[0])), vols[0]))
            ev = FD('', ids, id2row, labels, 32, 42, train=False, norm='imagenet'); ev.vols, ev.masks = vols, masks
            er = SW('', ids, id2row, labels, 32, 42, train=False, norm='imagenet'); er.vols, er.masks = vols, masks
            self.assertTrue(torch.equal(ev[0][0], er[0][0])); self.assertEqual(tuple(ev[0][1].shape), (12,))

    def test_it3_policy_weights(self):
        self.assertEqual(guard.DISTIL_WEIGHTS, {'in-distil': 0.5, 'in-distil-w1': 1.0, 'in-distil-flip': 0.5, 'ax-crop': 0.5, 'ax-full': 0.5})
        self.assertEqual((guard.VARIANTS['in-distil-w1'], guard.VARIANTS['in-distil-flip']), ('distil', 'distil_flip'))
        for arm in ('in-distil-w1', 'in-distil-flip'):
            self.assertLessEqual(len(f'knee-raptor64-{arm}-fold0-final-20261014'), 50)


class ProbeTests(unittest.TestCase):
    def test_runtime_probe_slices_variant_targets(self):
        """Regression (it2 attempt 1): variant datasets emit 24-wide targets; the controller probe must use the first 12."""
        source = (ROOT / 'scripts/colab_raptor64_run.py').read_text()
        spec = importlib.util.spec_from_file_location('r64run_probe', ROOT / 'scripts/colab_raptor64_run.py')
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        self.assertIn('binary_cross_entropy_with_logits(out.float(),y[:,:12].cuda())', runner.PROBE)
        self.assertNotIn('(out.float(),y.cuda())', runner.PROBE)
        compile(runner.PROBE, 'probe', 'exec')
        if importlib.util.find_spec('torch'):
            import torch
            out, y = torch.randn(4, 12), torch.rand(4, 24)
            torch.nn.functional.binary_cross_entropy_with_logits(out, y[:, :12])


@unittest.skipUnless(HAVE and BUNDLE_V1.is_file(), 'pinned evidence and bundle v1 required')
class BuilderTests(unittest.TestCase):
    def test_variant_packages_and_unchanged_baseline_trainer(self):
        spec = importlib.util.spec_from_file_location('b64', ROOT / 'scripts/build_raptor64_training.py')
        builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            builder.build(tmp / 'in-orig', evidence_root=EVIDENCE, arm='in-orig')
            with tarfile.open(BUNDLE_V1) as t:
                frozen = t.extractfile('raptor64-in-orig-pkg/train_knee_patched.py').read()
                frozen_entry = t.extractfile('raptor64-in-orig-pkg/tracka_entry.py').read()
                frozen_recipe = t.extractfile('raptor64-in-orig-pkg/raptor64-recipe.json').read()
            self.assertEqual((tmp / 'in-orig/train_knee_patched.py').read_bytes(), frozen)
            self.assertEqual((tmp / 'in-orig/tracka_entry.py').read_bytes(), frozen_entry)
            self.assertEqual((tmp / 'in-orig/raptor64-recipe.json').read_bytes(), frozen_recipe)
            r = builder.build(tmp / 'in-mirror', evidence_root=EVIDENCE, arm='in-mirror')
            self.assertTrue(r['scientific_ast_equal'])
            trainer = (tmp / 'in-mirror/train_knee_patched.py').read_text(); entry = (tmp / 'in-mirror/tracka_entry.py').read_text()
            self.assertIn(builder.VARIANT_HOOK.strip(), trainer); self.assertIn('lossf = raptor64.loss_variant(lossf, __file__)', trainer)
            self.assertIn('mirror_tta_csv', entry)
            policy = guard.recipe(tmp / 'in-mirror/train_knee_patched.py')
            self.assertEqual((policy['variant'], policy['mirror']['p'], policy['frozen']), ('mirror', 0.5, guard.FROZEN))
            with self.assertRaises(ValueError):
                builder.build(tmp / 'd0', evidence_root=EVIDENCE, arm='in-distil')
            builder.build(tmp / 'in-distil', evidence_root=EVIDENCE, arm='in-distil', teacher_sha256='a' * 64)
            policy = guard.recipe(tmp / 'in-distil/train_knee_patched.py')
            self.assertEqual((policy['variant'], policy['teacher']['weight'], policy['teacher']['sha256']), ('distil', 0.5, 'a' * 64))
            self.assertNotIn('mirror_tta_csv', (tmp / 'in-distil/tracka_entry.py').read_text())
            with self.assertRaises(ValueError):
                builder.build(tmp / 'o2', evidence_root=EVIDENCE, arm='in-orig', teacher_sha256='a' * 64)
            v5 = EVIDENCE / 'outputs/analysis/raptor64-20261013/colab-bundle-v5/raptor64-colab-bundle.tar.gz'
            if v5.is_file():
                builder.build(tmp / 'd5', evidence_root=EVIDENCE, arm='in-distil', teacher_sha256='dab0621a2302dc46a867d007d8d90d014d62ddb2f2ad515b8c571d9e17e692f9')
                with tarfile.open(v5) as tf:
                    for name in ('train_knee_patched.py', 'tracka_entry.py', 'raptor64-recipe.json'):
                        self.assertEqual((tmp / 'd5' / name).read_bytes(), tf.extractfile('raptor64-in-distil-pkg/' + name).read(), name)
            builder.build(tmp / 'w1', evidence_root=EVIDENCE, arm='in-distil-w1', teacher_sha256='a' * 64)
            builder.build(tmp / 'fl', evidence_root=EVIDENCE, arm='in-distil-flip', teacher_sha256='a' * 64)
            w1 = guard.recipe(tmp / 'w1/train_knee_patched.py'); fl = guard.recipe(tmp / 'fl/train_knee_patched.py')
            self.assertEqual((w1['variant'], w1['teacher']['weight'], 'flip' in w1), ('distil', 1.0, False))
            self.assertEqual((fl['variant'], fl['teacher']['weight'], fl['flip']), ('distil_flip', 0.5, guard.FLIP_SPEC))
            self.assertNotIn('mirror_tta_csv', (tmp / 'fl/tracka_entry.py').read_text())
            strip = lambda p: {k: v for k, v in p.items() if k not in ('recipe', 'arm', 'teacher', 'flip', 'variant', 'head_definition', 'initializer')}
            self.assertEqual(strip(w1), strip(fl))


def axial_phantom(anterior_top=True, seed=0):
    """Synthetic 64-slot volume: non-axial slots constant 7; axial slots = air, a wide posterior soft-tissue mass and a
    narrow anterior (patella-like) blob plus per-slice texture; vertically mirrored when anterior_top is False."""
    import numpy as np
    rng = np.random.default_rng(seed)
    v = np.full((64, 336, 336), 7, np.uint8)
    for s in range(52, 64):
        img = np.zeros((336, 336), np.uint8)
        img[150:300, 40:296] = 120; img[60:110, 140:196] = 200
        img = np.where(img > 0, np.clip(img.astype(int) + rng.integers(-20, 20, img.shape) + (s - 52), 1, 255), 0).astype(np.uint8)
        v[s] = img if anterior_top else img[::-1]
    return v


def axial_slots(selected, frames):
    from knee import raptor64_cache as cache
    return ([{'plane': p, 'selected': True, 'fluid_sensitive': 0, 'frames': 20, 'count': c} for p, _, c in cache.SLOTS[:4]] +
            [{'plane': 'Axial', 'selected': selected, 'fluid_sensitive': 0 if selected else None, 'frames': frames, 'count': 12}])


@unittest.skipUnless(HAVE, 'NumPy/PyTorch and the pinned historical trainer required')
class AxialArmTests(unittest.TestCase):
    """it4 ax-crop / ax-full: axial slots only, anterior crop, all 10 windows, no-axial exclusion, label-preserving AP flip."""
    CROP = {'anterior': 0.55, 'central': 0.65}

    def test_orientation_rule_and_crop_geometry(self):
        import numpy as np
        top, bottom = axial_phantom(True), axial_phantom(False)
        self.assertEqual(guard.axial_anterior_side(top[52:64]), 'top')
        self.assertEqual(guard.axial_anterior_side(bottom[52:64]), 'bottom')
        self.assertEqual(guard.axial_anterior_side(np.zeros((12, 336, 336), np.uint8)), 'top')   # undecidable -> storage order
        self.assertEqual(guard.axial_crop_box(336, 336, self.CROP), (0, 185, 59, 277))
        crop, full = guard.AXIAL_SPECS['ax-crop'], guard.AXIAL_SPECS['ax-full']
        a = guard.axial_block(top, crop)
        self.assertEqual(a.shape, (12, 185, 218)); self.assertTrue(np.array_equal(a, top[52:64, 0:185, 59:277]))
        self.assertTrue(np.array_equal(guard.axial_block(bottom, crop), bottom[52:64, 0:185, 59:277]), 'storage order; the heuristic is audit-only')
        self.assertTrue(np.array_equal(guard.axial_block(top, crop, flip=True), a[:, ::-1]))
        f = guard.axial_block(top, full)
        self.assertEqual(f.shape, (12, 336, 336)); self.assertTrue(np.array_equal(f, top[52:64]))
        self.assertTrue(np.array_equal(guard.axial_block(bottom, full), bottom[52:64]))
        self.assertTrue(np.array_equal(guard.axial_block(top, full, flip=True), top[52:64, ::-1]))
        self.assertFalse((guard.axial_block(axial_phantom(True), full) == 7).any(), 'only axial slots are read')

    def datasets(self, arm):
        import numpy as np
        ids = ['a', 'b', 'c']; id2row = {u: i for i, u in enumerate(ids)}
        vols = np.stack([axial_phantom(True, 0), axial_phantom(False, 1), axial_phantom(True, 2)]); masks = np.ones((3, 64), np.uint8)
        vols[2, 52:64] = 0; masks[2, 52:64] = 0           # study c: no axial series -> all-masked axial block
        labels = {u: np.linspace(0, 1, 12).astype(np.float32) * (i + 1) / 3 for i, u in enumerate(ids)}
        probs = np.random.default_rng(3).uniform(0, 1, (3, 12)).astype(np.float32)
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / 't.npz'; np.savez(path, ids=np.asarray(ids), probs=probs)
        policy = {'variant': 'axial', 'axial': guard.AXIAL_SPECS[arm], 'teacher': {'sha256': smoke.file_sha(path), 'weight': 0.5}}
        SW, _, _ = base_tests.historical_classes()
        stack = [patch.dict(os.environ, {'RAPTOR64_TEACHER': str(path)}), patch.object(guard, '_CORPUS', ('v', 'm', np.asarray(ids))),
                 patch.object(guard, '_CACHE_IDENTITY', {'fixture': False}), patch.object(guard, 'recipe', return_value=policy),
                 patch.object(guard, '_AXIAL', {'a': True, 'b': True, 'c': False})]
        for s in stack:
            s.start(); self.addCleanup(s.stop)
        AX = guard.recipe_variant(SW, 'x')
        tr = AX('', ids, id2row, labels, 32, 10, train=True, aug=False, norm='imagenet'); tr.vols, tr.masks = vols, masks
        ev = AX('', ids, id2row, labels, 32, 10, train=False, norm='imagenet'); ev.vols, ev.masks = vols, masks
        return tr, ev, vols, labels, probs

    def reference(self, block):
        import torch
        from torch.nn import functional as F
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tiles = []
        for c in range(1, 11):
            tri = torch.from_numpy(block[c - 1:c + 2].astype('float32') / 255.0)
            tiles.append((F.interpolate(tri[None], size=(32, 32), mode='bilinear', align_corners=False)[0] - mean) / std)
        return torch.stack(tiles)

    def test_axial_slots_ten_windows_exclusion_and_label_preserving_flip(self):
        import numpy as np
        import torch
        self.assertEqual(guard.eval_centers(np.ones(12), 12, 10), list(range(1, 11)))
        for arm in ('ax-crop', 'ax-full'):
            with self.subTest(arm=arm):
                tr, ev, vols, labels, probs = self.datasets(arm)
                spec = guard.AXIAL_SPECS[arm]
                self.assertEqual(tr.ids, ['a', 'b'], 'training excludes the study without axial content')
                self.assertEqual(ev.ids, ['a', 'b', 'c'], 'evaluation keeps every development study')
                for i, u in enumerate(['a', 'b']):
                    x, y = ev[i]
                    self.assertEqual(tuple(x.shape), (10, 3, 32, 32)); self.assertEqual(tuple(y.shape), (12,))
                    self.assertTrue(torch.allclose(x, self.reference(guard.axial_block(vols[i], spec)), atol=1e-6), 'windows = axial slots 52-63 only')
                    if arm == 'ax-crop':
                        self.assertTrue(torch.allclose(x, self.reference(vols[i, 52:64, 0:185, 59:277]), atol=1e-6), 'anterior 55% x central 65%')
                    with patch('random.random', return_value=0.99), patch('random.shuffle', lambda seq: None):
                        xn, yn = tr[i]                       # no flip; identity shuffle -> every centre once, in order
                    with patch('random.random', return_value=0.0), patch('random.shuffle', lambda seq: None):
                        xf, yf = tr[i]                       # AP flip
                    self.assertTrue(torch.equal(xn, x), 'training uses all 10 windows')
                    self.assertTrue(torch.allclose(xf, self.reference(guard.axial_block(vols[i], spec, flip=True)), atol=1e-6))
                    self.assertTrue(torch.allclose(xf, x.flip(-2), atol=1e-5))
                    for yy in (yn, yf):
                        self.assertEqual(tuple(yy.shape), (24,))
                        self.assertTrue(np.array_equal(yy[:12].numpy(), labels[u]), 'labels changed under AP flip')
                        self.assertTrue(np.array_equal(yy[12:].numpy(), probs[i]), 'teacher targets changed under AP flip')
                xc, yc = ev[2]                               # no-axial study still predicted (blend falls back to in-distil)
                self.assertEqual(tuple(xc.shape), (10, 3, 32, 32)); self.assertEqual(tuple(yc.shape), (12,))

    def test_shuffled_training_windows_are_a_permutation_of_all_centres(self):
        import torch
        tr, ev, _, _, _ = self.datasets('ax-crop')
        x_eval = ev[0][0]
        random.seed(7)
        with patch('random.random', return_value=0.99):
            x, _ = tr[0]
        matches = sorted(int(torch.argmin(((x_eval - w) ** 2).flatten(1).sum(1))) for w in x)
        self.assertEqual(matches, list(range(10)))

    def test_axial_presence_from_slot_metadata(self):
        manifest = {'entries': [{'StudyInstanceUID': 'a', 'slots': axial_slots(True, 30)}, {'StudyInstanceUID': 'b', 'slots': axial_slots(False, 0)},
                                {'StudyInstanceUID': 'c', 'slots': axial_slots(True, 0)}]}
        self.assertEqual(guard.axial_presence(manifest), {'a': True, 'b': False, 'c': False})
        bad = {'entries': [{'StudyInstanceUID': 'a', 'slots': axial_slots(True, 30)[:4] + [{'plane': 'Coronal', 'selected': True, 'frames': 3, 'count': 12}]}]}
        with self.assertRaises(ValueError):
            guard.axial_presence(bad)

    def test_it4_policy_pins(self):
        from knee import raptor64_cache as cache
        self.assertEqual(sum(c for _, _, c in cache.SLOTS[:4]), 52); self.assertEqual(cache.SLOTS[4], ('Axial', -1, 12))
        self.assertEqual((guard.AXIAL_SLOTS, guard.AXIAL_WINDOWS, guard.AXIAL_INDEX), ((52, 64), 10, 4))
        common = {'slots': [52, 64], 'windows': 10, 'ap_flip_p': 0.5, 'orientation': 'storage-order-anterior-top-v1 (centroid heuristic audited and rejected)',
                  'train_exclusion': 'studies with no axial content (axial slot not selected or zero frames)',
                  'eval': 'all 870 predicted; studies without axial content fall back to in-distil alone in the blend'}
        self.assertEqual(guard.AXIAL_SPECS, {'ax-crop': {**common, 'crop': {'anterior': 0.55, 'central': 0.65}}, 'ax-full': {**common, 'crop': None}})
        self.assertEqual((guard.VARIANTS['ax-crop'], guard.VARIANTS['ax-full']), ('axial', 'axial'))
        self.assertEqual((guard.DISTIL_WEIGHTS['ax-crop'], guard.DISTIL_WEIGHTS['ax-full']), (0.5, 0.5))
        self.assertEqual(guard.ARMS['ax-crop'], ('imagenet', 'original')); self.assertEqual(guard.ARMS['ax-full'], ('imagenet', 'original'))
        self.assertEqual(guard.FROZEN_AXIAL, {**guard.FROZEN, 'k': 10, 'k_eval': 10})
        self.assertEqual((guard.frozen_for('ax-crop'), guard.export_k_for('ax-full')), (guard.FROZEN_AXIAL, 10))
        self.assertEqual((guard.frozen_for('in-distil'), guard.export_k_for('in-distil'), guard.frozen_for('dr-full')), (guard.FROZEN, 42, guard.FROZEN_FULL))
        spec = importlib.util.spec_from_file_location('r64run_ax', ROOT / 'scripts/colab_raptor64_run.py')
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        self.assertEqual((runner.arm_windows('ax-crop'), runner.arm_windows('in-distil')), ((10, 10, 10), (8, 16, 42)))
        self.assertEqual(runner.comparison_endpoint('ax-full'), 'K10 primary (selection K10; axial windows)')
        for arm in ('ax-crop', 'ax-full'):
            self.assertLessEqual(len(f'knee-raptor64-{arm}-fold0-final-20261014'), 50)
            policy = {'version': 1, 'arm': arm, 'init': 'imagenet', 'head': 'original', 'frozen': guard.FROZEN_AXIAL, 'export_k': 10,
                      'experiment_split': guard.SPLITS[0], 'experiment_split_sha256': smoke.digest(smoke.canonical(guard.SPLITS[0])),
                      'variant': 'axial', 'axial': guard.AXIAL_SPECS[arm], 'teacher': {'weight': 0.5, 'sha256': 'a' * 64, 'source': 'dread_v4_full_k42'}}
            guard.validate_policy(policy)
            for bad in ({'frozen': guard.FROZEN}, {'export_k': 42}, {'axial': {**guard.AXIAL_SPECS[arm], 'ap_flip_p': 0.0}},
                        {'axial': guard.AXIAL_SPECS['ax-full' if arm == 'ax-crop' else 'ax-crop']}, {'teacher': {**policy['teacher'], 'weight': 1.0}}):
                with self.subTest(arm=arm, bad=bad), self.assertRaises(ValueError):
                    guard.validate_policy({**policy, **bad})


@unittest.skipUnless(HAVE and BUNDLE_V1.is_file(), 'pinned evidence and bundle v1 required')
class AxialBuilderTests(unittest.TestCase):
    def test_axial_packages(self):
        import ast
        spec = importlib.util.spec_from_file_location('b64ax', ROOT / 'scripts/build_raptor64_training.py')
        builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaises(ValueError):
                builder.build(tmp / 'none', evidence_root=EVIDENCE, arm='ax-crop')
            policies = {}
            for arm in ('ax-crop', 'ax-full'):
                r = builder.build(tmp / arm, evidence_root=EVIDENCE, arm=arm, teacher_sha256='a' * 64)
                self.assertTrue(r['scientific_ast_equal'])
                trainer = (tmp / arm / 'train_knee_patched.py').read_text(); entry = (tmp / arm / 'tracka_entry.py').read_text()
                self.assertIn(builder.VARIANT_HOOK.strip(), trainer); self.assertIn('lossf = raptor64.loss_variant(lossf, __file__)', trainer)
                self.assertIn('k_eval = raptor64.AXIAL_WINDOWS', entry); self.assertNotIn('k_eval = raptor64.EXPORT_K', entry)
                self.assertIn('"comparison_endpoint": "K10 primary (selection K10; axial windows)"', entry); self.assertNotIn('mirror_tta_csv', entry)
                tree = ast.parse(entry)
                defaults = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == 'PARAM_DEFAULTS' for t in n.targets)))
                self.assertEqual((defaults['K'], defaults['K_EVAL'], defaults['EPOCHS']), ('10', '10', '12'))
                policies[arm] = guard.recipe(tmp / arm / 'train_knee_patched.py')
                self.assertEqual((policies[arm]['variant'], policies[arm]['axial'], policies[arm]['frozen'], policies[arm]['export_k']),
                                 ('axial', guard.AXIAL_SPECS[arm], guard.FROZEN_AXIAL, 10))
                self.assertEqual(policies[arm]['teacher']['weight'], 0.5)
            strip = lambda p: {k: v for k, v in p.items() if k not in ('recipe', 'arm', 'axial')}
            self.assertEqual(strip(policies['ax-crop']), strip(policies['ax-full']), 'the crop is the only factor between the arms')
            self.assertEqual({k: v for k, v in policies['ax-crop']['axial'].items() if k != 'crop'},
                             {k: v for k, v in policies['ax-full']['axial'].items() if k != 'crop'})


if __name__ == '__main__':
    unittest.main()
