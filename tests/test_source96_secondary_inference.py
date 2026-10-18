"""Synthetic-only tests for the K16 / K94 / header-gated and legacy ungated mirror secondary inference module.

No competition data, no timm download, no GPU: a tiny stand-in backbone and a stand-in trainer source
(same eval-path semantics as the pinned StudyWindows/RaptorClassifier) drive the real module code. When the
byte-pinned public trainer source is available locally the same end-to-end tests also run against it.
"""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import tempfile
import types
import unittest
from unittest.mock import patch

PREREQUISITES = ('public_training_smoke', 'source96_training', 'source96_loader', 'source96_cache', 'source_span_pilot', 'corpus44_parity')
_MISSING = [name for name in PREREQUISITES if importlib.util.find_spec('knee.' + name) is None]
if _MISSING:   # e.g. a worktree cut from a commit that predates the source96 modules
    raise unittest.SkipTest('Prerequisite modules absent from this checkout: ' + ', '.join(_MISSING))

from knee import source96_loader as loader, source96_secondary_inference as sec, source96_training as guard, public_training_smoke as smoke  # noqa: E402

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
HAS_TORCH = HAS_NUMPY and importlib.util.find_spec('torch') is not None
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DOC = ROOT / 'docs/experiments/source96-secondary-inference-protocol-20261012.md'
PINNED_TRAINER = Path(os.environ.get('SOURCE96_TRAIN_SOURCE', ROOT / 'outputs/analysis/public-train-recipe-20261009/assets/train_knee.py'))
H = W = 12     # tiny spatial size: the mirror and planning logic do not depend on 336
TEST_RES = 32
EPOCHS = [1, 4, 8]
SAG0, SAG1, COR0, COR1, AX = (0, 26), (26, 48), (48, 66), (66, 78), (78, 96)

# Same eval-path code as the pinned trainer (StudyWindows with train=False/roi=False, RaptorClassifier,
# build_backbone); loaded through the module's AST extractor exactly like a real trainer source.
STAND_IN_TRAINER = '''
class StudyWindows(Dataset):
    def __init__(self, root, ids, id2row, labels, res, k, train, aug=True, norm="none", roi=False):
        self.root, self.ids, self.id2row, self.labels = root, ids, id2row, labels
        self.res, self.k, self.train, self.aug, self.norm, self.roi = res, k, train, aug, norm, roi
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.vols = None; self.masks = None
    def __len__(self): return len(self.ids)
    def _centers(self, valid, count):
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        if not cs: cs = [max(1, min((lo + hi) // 2, self._D - 2))]
        idx = np.linspace(0, len(cs) - 1, count).round().astype(int)
        return [cs[i] for i in idx]
    def __getitem__(self, i):
        uid = self.ids[i]; row = self.id2row[uid]
        self._D = self.vols.shape[1]
        valid = np.where(self.masks[row] > 0)[0]
        if len(valid) < 3: valid = np.arange(min(3, self._D))
        cs = self._centers(valid, self.k)
        vol = self.vols[row]
        tiles = []
        for c in cs:
            c = max(1, min(c, self._D - 2))
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
            t = torch.from_numpy(np.ascontiguousarray(tri))
            if t.shape[-1] != self.res or t.shape[-2] != self.res:
                t = F.interpolate(t[None], size=(self.res, self.res), mode="bilinear", align_corners=False)[0]
            tiles.append(t.numpy())
        x = torch.from_numpy(np.stack(tiles, 0))
        if self.norm == "imagenet": x = (x - self._mean) / self._std
        return x, torch.from_numpy(self.labels[uid])

def build_backbone(arch="vit_small_patch16_224", pretrained=False):
    return timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=3, global_pool="avg")

class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=384, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n
    win_chunk = 0
    def encode(self, x):
        B, K = x.shape[:2]
        flat = x.flatten(0, 1)
        if self.win_chunk > 0:
            f = torch.cat([self.backbone(c) for c in flat.split(self.win_chunk, 0)], 0)
        else:
            f = self.backbone(flat)
        return f.view(B, K, -1)
    def head(self, feats):
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb
    def forward(self, x):
        return self.head(self.encode(x))
'''
STAND_IN_SHA = hashlib.sha256(STAND_IN_TRAINER.encode()).hexdigest()


def volume_with_marker(slice_index, row, col, value=200):
    import numpy as np
    v = np.zeros((96, H, W), np.uint8)
    v[slice_index, row, col] = value
    return v


def study(rng, *, missing_axial=False, partial_sag0=False, missing_coronal=False):
    """Random fully-populated stack (every slice nonzero) with optional missing/partial planes."""
    import numpy as np
    v = rng.integers(1, 256, size=(96, H, W), dtype=np.uint8)
    if partial_sag0:
        v[20:26] = 0
    if missing_axial:
        v[AX[0]:AX[1]] = 0
    if missing_coronal:
        v[COR0[0]:COR1[1]] = 0
    return v, v.reshape(96, -1).any(axis=1).astype(np.uint8)


class LayoutTests(unittest.TestCase):
    """Dependency-free: runs on the plain python3.12 check as well."""

    def test_layout_is_26_22_18_12_18_and_matches_cache_policy(self):
        from knee import source96_cache
        self.assertEqual(sec.SLOT_SIZES, (26, 22, 18, 12, 18))
        self.assertEqual(sec.SLOT_PLANES, ('Sagittal', 'Sagittal', 'Coronal', 'Coronal', 'Axial'))
        self.assertEqual(sec.SLOT_BOUNDS, (SAG0, SAG1, COR0, COR1, AX))
        self.assertEqual(sec.DEPTH, 96)
        self.assertEqual([s['mirror'] for s in sec.slot_layout()], ['slice_order', 'slice_order', 'width', 'width', 'width'])
        self.assertEqual([[p, f, n] for p, f, n in zip(sec.SLOT_PLANES, sec.SLOT_FLUID, sec.SLOT_SIZES)], source96_cache.CONFIG['slots'])

    def test_cli_requires_explicit_device(self):
        with self.assertRaises(SystemExit):
            sec.main(['--work', 'w', '--trainer-source', 't', '--labels', 'l', '--truth', 'x', '--out', 'o'])

    def test_mirror_orientation_is_declared_unverified_with_a_specific_audit_requirement(self):
        self.assertIs(sec.MIRROR_ORIENTATION_VERIFIED, False)
        self.assertIn('unverified', sec.MIRROR_ORIENTATION_ASSUMPTION)
        for needle in ('label-blind', 'per plane', 'dominant axis', 'row', 'column', 'oblique', 'nonstandard'):
            self.assertIn(needle, sec.ORIENTATION_AUDIT_REQUIRED)

    def test_no_orientation_overclaims_or_retired_runtime_estimates(self):
        overclaim = re.compile(r'\btrue(?:\s+study-level)?\s+(?:anatomical|mirror)', re.I)
        self.assertIsNone(overclaim.search(Path(sec.__file__).read_text()))
        if PROTOCOL_DOC.is_file():
            doc = PROTOCOL_DOC.read_text()
            self.assertIsNone(overclaim.search(doc))
            self.assertIn('unverified', doc)
            self.assertIn('unmeasured', doc)
            for retired in ('25 min', '2.4 h', '5.3 h'):
                self.assertNotIn(retired, doc)

    def test_only_k16_and_k94_are_supported(self):
        self.assertEqual(sec.WINDOW_COUNTS, (16, 94))
        self.assertEqual(sec.ARMS, ('k16_plain', 'k94_plain', 'k94_mirror_tta'))


@unittest.skipUnless(HAS_NUMPY, 'NumPy unavailable')
class MirrorTests(unittest.TestCase):

    def test_sagittal_marker_moves_to_the_other_end_of_its_own_slot_without_inplane_flip(self):
        import numpy as np
        for lo, hi in (SAG0, SAG1):
            v = volume_with_marker(lo, 3, 2)
            out = sec.mirror_volume(v)
            expected = volume_with_marker(hi - 1, 3, 2)   # opposite L/R end of the same series, same pixel (A/P and S/I untouched)
            self.assertTrue(np.array_equal(out, expected))
        # a marker one slice in from the lateral end lands one slice in from the other end
        self.assertTrue(np.array_equal(sec.mirror_volume(volume_with_marker(SAG1[0] + 1, 5, 7)), volume_with_marker(SAG1[1] - 2, 5, 7)))

    def test_sagittal_series_never_swap_slots(self):
        import numpy as np
        v = np.zeros((96, H, W), np.uint8)
        v[SAG0[0]:SAG0[1]] = 10
        v[SAG1[0]:SAG1[1]] = 20
        out = sec.mirror_volume(v)
        self.assertTrue((out[SAG0[0]:SAG0[1]] == 10).all() and (out[SAG1[0]:SAG1[1]] == 20).all())

    def test_coronal_and_axial_flip_width_only(self):
        import numpy as np
        for lo, hi in (COR0, COR1, AX):
            mid = (lo + hi) // 2
            out = sec.mirror_volume(volume_with_marker(mid, 4, 1))
            self.assertTrue(np.array_equal(out, volume_with_marker(mid, 4, W - 2)))   # same slice and row, mirrored column
            out = sec.mirror_volume(volume_with_marker(lo, 9, 0))
            self.assertTrue(np.array_equal(out, volume_with_marker(lo, 9, W - 1)))

    def test_every_plane_actually_moves_something_on_asymmetric_data(self):
        import numpy as np
        v = np.random.default_rng(0).integers(1, 256, size=(96, H, W), dtype=np.uint8)
        out = sec.mirror_volume(v)
        for lo, hi in sec.SLOT_BOUNDS:
            self.assertFalse(np.array_equal(v[lo:hi], out[lo:hi]))

    def test_mirror_twice_is_identity_and_input_is_untouched_even_if_read_only(self):
        import numpy as np
        rng = np.random.default_rng(1)
        v, m = study(rng, partial_sag0=True, missing_axial=True)
        v.flags.writeable = False
        m.flags.writeable = False
        v2, m2 = sec.mirror_study(v, m)
        self.assertFalse(np.array_equal(v2, v))
        v3, m3 = sec.mirror_study(v2, m2)
        self.assertTrue(np.array_equal(v3, v) and np.array_equal(m3, m) and v3.dtype == np.uint8)
        self.assertTrue(v2.flags.writeable)

    def test_mask_follows_the_data_and_stays_pixel_consistent(self):
        import numpy as np
        v, m = study(np.random.default_rng(2), partial_sag0=True)
        v2, m2 = sec.mirror_study(v, m)
        sec.validate_study(v2, m2)
        self.assertTrue(np.array_equal(m2.astype(bool), v2.reshape(96, -1).any(axis=1)))
        self.assertEqual(int(m2.sum()), int(m.sum()))
        # first sagittal slot: valid slices 0..19 move to 6..25 together with their pixels; masked ones stay masked
        self.assertTrue(m[0:20].all() and not m[20:26].any())
        self.assertTrue(m2[6:26].all() and not m2[0:6].any())
        self.assertTrue(np.array_equal(v2[SAG0[1] - 1], v[SAG0[0]]))
        # width planes keep their mask unchanged
        self.assertTrue(np.array_equal(m2[COR0[0]:], m[COR0[0]:]))

    def test_plane_with_zero_valid_slices_stays_empty_and_masked(self):
        import numpy as np
        v, m = study(np.random.default_rng(3), missing_axial=True, missing_coronal=True)
        self.assertFalse(m[COR0[0]:].any())
        v2, m2 = sec.mirror_study(v, m)
        self.assertFalse(v2[COR0[0]:].any() or m2[COR0[0]:].any())      # nothing mirrored into empty positions
        self.assertTrue(m2[:COR0[0]].all())                              # sagittal planes fully valid and mirrored
        sec.validate_study(v2, m2)
        # a study whose only populated plane is sagittal series 2: mirror keeps it there
        v = np.zeros((96, H, W), np.uint8)
        v[SAG1[0]:SAG1[1]] = np.random.default_rng(4).integers(1, 256, size=(22, H, W), dtype=np.uint8)
        m = v.reshape(96, -1).any(axis=1).astype(np.uint8)
        v2, m2 = sec.mirror_study(v, m)
        self.assertEqual(np.flatnonzero(m2).tolist(), list(range(*SAG1)))
        self.assertTrue(np.array_equal(v2[SAG1[0]:SAG1[1]], v[SAG1[0]:SAG1[1]][::-1]))

    def test_full_valid_synthetic_reflected_centre_comparison_with_public_raptor_rule(self):
        """Full-valid synthetic reflected-centre comparison (every slice populated): the public rule (centre slot
        decides; sagittal reverses the 3 channels, others flip width) equals the triplet at the reflected centre of
        the study-level mirrored volume for each window lying inside ONE series. Of the 8 K94 windows that straddle
        two series, the 4 pure coronal/axial ones (65, 66, 77, 78) also coincide; the 4 that involve a sagittal
        slice (25, 26, 47, 48) differ by construction (and are already cross-series mixtures in the plain pass).
        This is NOT evidence about partial masks: there the mirrored mask re-plans the centres, so sampled bags
        are not claimed to be identical."""
        import numpy as np
        v = np.random.default_rng(6).integers(1, 256, size=(96, H, W), dtype=np.uint8)
        t = sec.mirror_volume(v)
        slot = np.repeat(np.arange(5), sec.SLOT_SIZES)
        straddling, differing = [], []
        for c in range(1, 95):
            straddles = len({slot[c - 1], slot[c], slot[c + 1]}) > 1
            lo, hi = sec.SLOT_BOUNDS[slot[c]]
            tri = v[[c - 1, c, c + 1]]
            if sec.SLOT_PLANES[slot[c]] == 'Sagittal':
                public, mirrored_centre = tri[::-1], lo + hi - 1 - c       # the study-level mirror maps centre c to the reflected slice
            else:
                public, mirrored_centre = tri[:, :, ::-1], c
            equal = 1 <= mirrored_centre <= 94 and np.array_equal(public, t[[mirrored_centre - 1, mirrored_centre, mirrored_centre + 1]])
            if straddles:
                straddling.append(c)
            if not equal:
                differing.append(c)
        self.assertEqual(straddling, [25, 26, 47, 48, 65, 66, 77, 78])
        self.assertEqual(differing, [25, 26, 47, 48])

    def test_mirror_invariant_studies_are_fixed_points(self):
        import numpy as np
        v = np.random.default_rng(5).integers(1, 256, size=(96, H, W), dtype=np.uint8)
        v = np.maximum(v, sec.mirror_volume(v))     # symmetrised under the assumed L/R mirror
        self.assertTrue(np.array_equal(sec.mirror_volume(v), v))

    def test_bad_geometry_rejected(self):
        import numpy as np
        with self.assertRaises(ValueError):
            sec.mirror_volume(np.zeros((44, H, W), np.uint8))
        with self.assertRaises(ValueError):
            sec.mirror_volume(np.zeros((96, H, W), np.float32))
        with self.assertRaises(ValueError):
            sec.mirror_mask(np.full(96, 2, np.uint8))
        with self.assertRaises(ValueError):
            sec.validate_study(np.zeros((96, H, W), np.uint8), np.zeros(96, np.uint8))     # whole-empty
        with self.assertRaises(ValueError):
            sec.validate_study(volume_with_marker(0, 0, 0), np.ones(96, np.uint8))         # mask/pixel disagreement


@unittest.skipUnless(HAS_NUMPY, 'NumPy unavailable')
class EvalCentreTests(unittest.TestCase):
    K16_FULL = [1, 7, 13, 20, 26, 32, 38, 44, 51, 57, 63, 69, 75, 82, 88, 94]    # hand-derived: linspace(0,93,16).round()+1

    def test_full_mask_plans(self):
        import numpy as np
        m = np.ones(96, np.uint8)
        self.assertEqual(sec.eval_centres(m, 16), self.K16_FULL)
        self.assertEqual(sec.eval_centres(m, 94), list(range(1, 95)))       # K94 = every interior triplet exactly once

    def test_bounds_determinism_and_masked_ends(self):
        import numpy as np
        m = np.ones(96, np.uint8)
        m[:10] = 0
        m[90:] = 0
        for k in (16, 94):
            plan = sec.eval_centres(m, k)
            self.assertEqual(len(plan), k)
            self.assertEqual(plan, sec.eval_centres(m.copy(), k))
            self.assertTrue(all(11 <= c <= 88 for c in plan))
        self.assertEqual(len(set(sec.eval_centres(m, 94))), 78)             # 78 interior centres, duplicates preserved

    def test_fewer_than_three_valid_slices_falls_back(self):
        import numpy as np
        m = np.zeros(96, np.uint8)
        m[[40, 41]] = 1
        self.assertEqual(sec.eval_centres(m, 16), [1] * 16)


@unittest.skipUnless(HAS_TORCH, 'PyTorch unavailable; CPU model tests skipped')
class _Fixture:
    """Shared synthetic world: tiny backbone, stand-in trainer, work directory, corpus."""
    sha = STAND_IN_SHA
    source = STAND_IN_TRAINER

    def definitions(self):
        import torch
        from torch import nn

        class TinyBackbone(nn.Module):
            num_features = 8

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((2, 2))
                self.fc = nn.Linear(16, 8)

            def forward(self, x):
                return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))

        torch.manual_seed(0)
        timm = types.SimpleNamespace(create_model=lambda arch, **kwargs: TinyBackbone())
        return sec.load_definitions(self.source, self.sha, timm_module=timm)

    def setUp(self):
        import numpy as np
        import torch
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.defs = self.definitions()
        torch.manual_seed(7)
        backbone = self.defs.build_backbone(guard.ARCH, pretrained=False)
        self.model = self.defs.RaptorClassifier(backbone, F_dim=backbone.num_features)
        rng = np.random.default_rng(11)
        studies = [study(rng), study(rng, missing_axial=True), study(rng, partial_sag0=True), study(rng, missing_coronal=True), study(rng)]
        self.vols = np.stack([s[0] for s in studies])
        self.masks = np.stack([s[1] for s in studies])
        self.ids = [f'synthetic-study-{i}' for i in range(len(studies))]
        self.id2row = {u: i for i, u in enumerate(self.ids)}
        self.work = self.root / 'work'
        self.work.mkdir()
        self.checkpoint = self.work / 'raptor_ft_fold0_swa.pt'
        self.write_primary()

    def write_primary(self, res=TEST_RES, **overrides):
        import torch
        state = {'model': self.model.state_dict(), 'gold_auc': None, 'aucs': {}, 'src': 'synthetic', 'res': res, 'arch': guard.ARCH,
                 'lab': sec.TARGETS, 'epoch': EPOCHS, 'swa_over': EPOCHS}
        torch.save(state, self.checkpoint)
        (self.work / 'raptor_ft_fold0.json').write_text(json.dumps({'partial': False, 'hist': [{}] * 10}))
        record = {'export': {'k_eval': 16}, 'checkpoint_sha256': smoke.file_sha(self.checkpoint), 'selected_epochs': EPOCHS}
        record.update(overrides)
        (self.work / 'run_fold0.json').write_text(json.dumps(record))

    def run_it(self, out=None, vols=None, masks=None, **kwargs):
        settings = dict(work=self.work, fold=0, checkpoint=self.checkpoint, definitions=self.defs, vols=self.vols if vols is None else vols,
                        masks=self.masks if masks is None else masks, ids=self.ids, id2row=self.id2row, out=out or self.root / 'out',
                        device='cpu', res=TEST_RES, batch_studies=2, k94_win_chunk=16,
                        mode=sec.MODE_UNGATED)   # historical tests request the legacy ungated arm explicitly
        settings.update(kwargs)
        return sec.run_secondary(**settings)

    def reference(self, k, vols, masks, group):
        """Independent reference: pinned sampler windows through a fresh model with the same weights."""
        import numpy as np
        import torch
        backbone = self.defs.build_backbone(guard.ARCH, pretrained=False)
        model = self.defs.RaptorClassifier(backbone, F_dim=backbone.num_features)
        model.load_state_dict(self.model.state_dict())
        model.eval()
        model.win_chunk = 0
        labels = {u: np.zeros(12, np.float32) for u in self.ids}
        ds = self.defs.StudyWindows('', self.ids, self.id2row, labels, TEST_RES, k, False, aug=False, norm='imagenet', roi=False)
        ds.vols, ds.masks = vols, masks
        rows = []
        with torch.no_grad():
            for s in range(0, len(self.ids), group):
                x = torch.stack([ds[j][0] for j in range(s, min(s + group, len(self.ids)))])
                rows.extend(torch.sigmoid(model(x).float()).numpy())
        return np.asarray(rows, dtype=np.float32)

    @staticmethod
    def manual_window(vol, c):
        """Triplet centred at c, resized bilinearly and ImageNet-normalised, written out independently."""
        import numpy as np
        import torch
        from torch.nn import functional as F
        tri = torch.from_numpy(np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0)
        t = F.interpolate(tri[None], size=(TEST_RES, TEST_RES), mode='bilinear', align_corners=False)[0]
        return (t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def npz(self, out, arm):
        import numpy as np
        return np.load(Path(out) / f'{arm}-private.npz')


class EndToEndTests:
    """Mixed into fixtures for each definitions provider (stand-in and, when available, the pinned trainer)."""

    def test_three_separate_outputs_bound_to_one_checkpoint(self):
        import numpy as np
        out = self.root / 'out'
        receipt = self.run_it(out)
        self.assertEqual(receipt['status'], 'PASS')
        self.assertEqual(receipt['optimizer_updates'], 0)
        sha = smoke.file_sha(self.checkpoint)
        self.assertEqual(set(receipt['outputs']), set(sec.ARMS))
        self.assertEqual({o['checkpoint_sha256'] for o in receipt['outputs'].values()}, {sha})
        self.assertEqual(receipt['binding']['checkpoint_sha256'], sha)
        self.assertEqual(receipt['binding']['checkpoint_sha256_after_run'], sha)
        self.assertEqual(receipt['binding']['primary_record_checkpoint_sha256'], sha)
        self.assertTrue(receipt['binding']['all_outputs_same_checkpoint'])
        self.assertEqual(receipt['binding']['model_state_sha256'], receipt['binding']['model_state_sha256_after_run'])
        saved = json.loads((out / 'receipt.json').read_text())
        self.assertEqual(saved['outputs'], receipt['outputs'])
        for arm in sec.ARMS:
            data = self.npz(out, arm)
            self.assertEqual(str(data['checkpoint_sha256']), sha)         # artifacts are self-binding
            p = data['probabilities']
            self.assertEqual(p.shape, (5, 12))
            self.assertTrue(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all())
            self.assertEqual(data['ids'].tolist(), self.ids)
            self.assertEqual(smoke.file_sha(out / f'{arm}-private.npz'), receipt['outputs'][arm]['npz_sha256'])
            header = (out / f'{arm}.csv').read_text().splitlines()[0]
            self.assertEqual(header, ','.join(['StudyInstanceUID', *sec.TARGETS]))
        self.assertEqual(receipt['outputs']['k16_plain']['k'], 16)
        self.assertEqual((receipt['outputs']['k94_plain']['k'], receipt['outputs']['k94_mirror_tta']['k']), (94, 94))
        self.assertEqual([receipt['outputs'][a]['mirror_tta'] for a in sec.ARMS], [False, False, True])
        text = (out / 'receipt.json').read_text()
        self.assertFalse(any(u in text for u in self.ids))                # aggregate receipt carries no identifiers
        self.assertIn('Secondary diagnostics only', text)

    def test_k16_reproduces_primary_arithmetic_and_k94_views_match_references(self):
        import numpy as np
        out = self.root / 'out'
        self.run_it(out)
        k16 = self.npz(out, 'k16_plain')['probabilities']
        np.testing.assert_allclose(k16, self.reference(16, self.vols, self.masks, group=2), rtol=0, atol=1e-6)   # primary grouping: 2 studies per forward
        plain = self.npz(out, 'k94_plain')['probabilities']
        np.testing.assert_allclose(plain, self.reference(94, self.vols, self.masks, group=1), rtol=0, atol=1e-6)
        mirrored_v = np.stack([sec.mirror_volume(v) for v in self.vols])
        mirrored_m = np.stack([sec.mirror_mask(m) for m in self.masks])
        tta = self.npz(out, 'k94_mirror_tta')
        np.testing.assert_allclose(tta['mirror_view'], self.reference(94, mirrored_v, mirrored_m, group=1), rtol=0, atol=1e-6)
        np.testing.assert_allclose(tta['plain_view'], plain, rtol=0, atol=0)
        np.testing.assert_allclose(tta['probabilities'], 0.5 * (tta['plain_view'] + tta['mirror_view']), rtol=0, atol=1e-7)
        self.assertGreater(float(np.abs(tta['mirror_view'] - tta['plain_view']).max()), 1e-5)   # mirror is not a no-op on asymmetric knees
        self.assertGreater(float(np.abs(tta['probabilities'] - plain).max()), 1e-6)

    def test_k94_win_chunk_does_not_change_probabilities(self):
        import numpy as np
        a = self.run_it(self.root / 'a', k94_win_chunk=0)
        b = self.run_it(self.root / 'b', k94_win_chunk=7)
        for arm in sec.ARMS:
            np.testing.assert_allclose(self.npz(self.root / 'a', arm)['probabilities'], self.npz(self.root / 'b', arm)['probabilities'], rtol=0, atol=1e-6)
        self.assertEqual(a['settings']['k94_win_chunk'], 0)
        self.assertEqual(b['settings']['k94_win_chunk'], 7)

    def test_mirror_invariant_corpus_gives_tta_equal_to_plain_k94(self):
        import numpy as np
        vols = np.stack([np.maximum(v, sec.mirror_volume(v)) for v in self.vols])
        masks = np.stack([v.reshape(96, -1).any(axis=1).astype(np.uint8) for v in vols])
        out = self.root / 'sym'
        self.run_it(out, vols=vols, masks=masks)
        tta = self.npz(out, 'k94_mirror_tta')
        np.testing.assert_allclose(tta['mirror_view'], tta['plain_view'], rtol=0, atol=1e-6)
        np.testing.assert_allclose(tta['probabilities'], self.npz(out, 'k94_plain')['probabilities'], rtol=0, atol=1e-6)

    def test_window_plans_are_recorded_and_mirror_replans_on_the_mirrored_mask(self):
        import numpy as np
        import torch
        receipt = self.run_it(self.root / 'out')
        plans = receipt['window_plans']
        self.assertEqual(plans['k16_plain']['windows'], 16 * 5)
        self.assertEqual(plans['k94_plain']['windows'], 94 * 5)
        self.assertEqual(plans['k94_mirror_view']['windows'], 94 * 5)
        self.assertEqual(plans['k94_plain']['fallback_studies'], 0)
        self.assertGreater(plans['k94_plain']['masked_triplets'], 0)       # missing planes produce masked triplets, as in the primary path
        self.assertGreater(plans['k94_plain']['boundary_triplets'], 0)
        # study 2 has sagittal slices 20..25 masked: plain valid range starts at 0, mirrored valid range starts at 6
        ds = self.defs.StudyWindows('', self.ids, self.id2row, {}, TEST_RES, 94, False, aug=False, norm='imagenet', roi=False)
        plain, _ = sec.center_plan(self.masks[2], ds, 94)
        mirrored, _ = sec.center_plan(sec.mirror_mask(self.masks[2]), ds, 94)
        self.assertEqual((min(plain), min(mirrored)), (1, 7))
        self.assertEqual(max(plain), max(mirrored))
        # recorded aggregates equal the sum of per-study plans computed from the plain and the MIRRORED masks
        for name, transform, k in (('k16_plain', lambda m: m, 16), ('k94_plain', lambda m: m, 94), ('k94_mirror_view', sec.mirror_mask, 94)):
            total = {}
            for m in self.masks:
                for key, value in sec.center_plan(transform(m), ds, k)[1].items():
                    total[key] = total.get(key, 0) + value
            self.assertEqual({key: plans[name][key] for key in total}, total, name)
        self.assertNotEqual(plans['k94_plain']['unique_centers'], plans['k94_mirror_view']['unique_centers'])
        # the windows actually fed to the model for the mirrored view use exactly those mirrored-mask centres and mirrored pixels
        sampler = sec.make_sampler(self.defs, self.ids, self.id2row, sec.MappedCorpus(self.vols, sec.mirror_volume),
                                   sec.MappedCorpus(self.masks, sec.mirror_mask), 94, TEST_RES)
        windows = sampler[2][0]
        v2 = sec.mirror_volume(self.vols[2])
        for i in (0, 1, 40, 93):
            self.assertTrue(torch.allclose(windows[i], self.manual_window(v2, mirrored[i]), atol=1e-6))
        self.assertFalse(torch.allclose(windows[0], self.manual_window(self.vols[2], plain[0]), atol=1e-3))

    def test_center_plan_counts_bounds_determinism_and_unsupported_k(self):
        import numpy as np
        ds = self.defs.StudyWindows('', self.ids, self.id2row, {}, TEST_RES, 94, False, aug=False, norm='imagenet', roi=False)
        full = np.ones(96, np.uint8)
        c16, f16 = sec.center_plan(full, ds, 16)
        c94, f94 = sec.center_plan(full, ds, 94)
        self.assertEqual(c16, EvalCentreTests.K16_FULL)
        self.assertEqual(c94, list(range(1, 95)))
        self.assertEqual((f16['windows'], f94['windows'], f94['unique_centers'], f94['duplicate_centers']), (16, 94, 94, 0))
        self.assertEqual((f94['masked_triplets'], f94['boundary_triplets']), (0, 8))     # 8 centres straddle the four slot boundaries
        self.assertEqual(sec.center_plan(full, ds, 94)[0], c94)
        partial = full.copy()
        partial[:10] = 0
        partial[90:] = 0
        for k in (16, 94):
            centers, flags = sec.center_plan(partial, ds, k)
            self.assertEqual(len(centers), k)
            self.assertTrue(all(11 <= c <= 88 for c in centers))
        few = np.zeros(96, np.uint8)
        few[[3, 4]] = 1
        self.assertEqual(sec.center_plan(few, ds, 16)[1]['fallback_studies'], 1)
        for bad in (8, 42, 96):
            with self.assertRaises(ValueError):
                sec.center_plan(full, ds, bad)
        with self.assertRaises(ValueError):
            sec.center_plan(np.full(96, 2, np.uint8), ds, 16)

    def test_k16_parity_pass_and_fail(self):
        import numpy as np
        first = self.run_it(self.root / 'first')
        k16 = self.npz(self.root / 'first', 'k16_plain')['probabilities']
        oof = self.root / 'oof_fold0.csv'
        header = ['StudyInstanceUID', *sec.TARGETS, 'set']
        def write(delta):
            lines = [','.join(header)]
            for sid, row in zip(self.ids, k16):
                lines.append(','.join([sid, *[repr(float(v) + delta) for v in row], 'val']))
            oof.write_text('\n'.join(lines) + '\n')
        write(0.0)
        receipt = self.run_it(self.root / 'pass', saved_oof=oof)
        self.assertEqual(receipt['status'], 'PASS')
        self.assertTrue(receipt['parity']['k16_vs_primary']['within_tolerance'])
        write(0.01)
        with self.assertRaisesRegex(ValueError, 'primary export'):
            self.run_it(self.root / 'fail', saved_oof=oof)
        saved = json.loads((self.root / 'fail/receipt.json').read_text())
        self.assertEqual(saved['status'], 'PARITY_FAIL')
        self.assertEqual(sorted(os.listdir(self.root / 'fail')), ['receipt.json'])         # failure receipt only: no arm output, K16 included
        self.assertEqual(saved['outputs'], {})
        self.assertIs(saved['k94_and_mirror_views_run'], False)
        self.assertFalse(saved['parity']['k16_vs_primary']['within_tolerance'])
        recorded = self.run_it(self.root / 'recorded', saved_oof=oof, enforce_parity=False)
        self.assertEqual(recorded['status'], 'PASS')
        self.assertFalse(recorded['parity']['k16_vs_primary']['within_tolerance'])
        write(0.0)
        oof.write_text(oof.read_text().replace('synthetic-study-4', 'unknown'))
        with self.assertRaises(ValueError):
            self.run_it(self.root / 'coverage', saved_oof=oof)
        self.assertEqual(first['status'], 'PASS')

    def test_k94_refused_without_unchanged_primary_k16_record(self):
        class Exploding:
            shape = (5, 96, H, W)
            def __getitem__(self, row):
                raise AssertionError('corpus must not be read when the checkpoint is refused')
        (self.work / 'run_fold0.json').unlink()
        out = self.root / 'refused'
        with self.assertRaisesRegex(ValueError, 'primaryK16'):
            self.run_it(out, vols=Exploding(), masks=Exploding())
        self.assertFalse(out.exists())
        base = {'export': {'k_eval': 16}, 'checkpoint_sha256': smoke.file_sha(self.checkpoint), 'selected_epochs': EPOCHS}
        for name, change in {'k42 record': {'export': {'k_eval': 42}}, 'other checkpoint': {'checkpoint_sha256': '0' * 64},
                             'other epochs': {'selected_epochs': [2, 4, 8]}, 'no sha': {'checkpoint_sha256': None},
                             'no export': {'export': None}}.items():
            record = {**base, **change}
            if change.get('checkpoint_sha256', 1) is None:
                record.pop('checkpoint_sha256')
            if change.get('export', 1) is None:
                record.pop('export')
            (self.work / 'run_fold0.json').write_text(json.dumps(record))
            with self.subTest(name), self.assertRaises(ValueError):
                self.run_it(self.root / ('refused-' + name.replace(' ', '-')), vols=Exploding(), masks=Exploding())
        (self.work / 'run_fold0.json').write_text(json.dumps(base))
        (self.work / 'raptor_ft_fold0.json').write_text(json.dumps({'partial': False, 'hist': [{}] * 9}))      # unfinished training
        with self.assertRaises(ValueError):
            self.run_it(self.root / 'unfinished', vols=Exploding(), masks=Exploding())

    def test_checkpoint_changed_after_primary_record_is_refused(self):
        import torch
        state = torch.load(self.checkpoint, weights_only=True)
        state['model'] = {k: v + 1 if v.is_floating_point() else v for k, v in state['model'].items()}
        torch.save(state, self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'primaryK16'):
            self.run_it(self.root / 'tampered')

    def test_wrong_identity_checkpoint_is_refused(self):
        import torch
        for change in ({'arch': 'other'}, {'res': 31}, {'swa_over': [1, 4], 'epoch': [1, 4]}, {'lab': list(reversed(sec.TARGETS))}):
            self.write_primary()                                          # fresh consistent record + checkpoint per case
            state = torch.load(self.checkpoint, weights_only=True)
            state.update(change)
            torch.save(state, self.checkpoint)
            (self.work / 'run_fold0.json').write_text(json.dumps({'export': {'k_eval': 16}, 'checkpoint_sha256': smoke.file_sha(self.checkpoint),
                                                                   'selected_epochs': EPOCHS}))        # primary record agrees: only identity is wrong
            with self.subTest(change), self.assertRaisesRegex(ValueError, 'identity|Top-3'):
                self.run_it(self.root / 'identity')

    def test_explicit_device_required_and_no_cpu_fallback(self):
        import torch
        for device in ('auto', None, 'tpu'):
            with self.subTest(device), self.assertRaises(ValueError):
                self.run_it(self.root / f'device-{device}', device=device)
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                self.run_it(self.root / 'cuda', device='cuda')
        self.assertFalse((self.root / 'cuda').exists())

    def test_corrupt_study_fails_closed_with_failure_receipt(self):
        import numpy as np
        vols, masks = self.vols.copy(), self.masks.copy()
        masks[3, 10] = 0                                                  # mask disagrees with pixels
        out = self.root / 'bad'
        with self.assertRaisesRegex(ValueError, 'consistency'):
            self.run_it(out, vols=vols, masks=masks)
        self.assertEqual(json.loads((out / 'receipt.json').read_text())['status'], 'FAIL')
        self.assertFalse((out / 'k16_plain.csv').exists())

    def test_model_state_digest_matches_the_checkpoint_weights(self):
        receipt = self.run_it(self.root / 'out')
        from knee import source96_training
        reference = source96_training.tensor_hash(self.model.state_dict())
        self.assertEqual(receipt['binding']['model_state_sha256'], reference)

    def test_receipt_and_mirror_artifact_mark_orientation_unverified(self):
        out = self.root / 'out'
        receipt = self.run_it(out)
        saved = json.loads((out / 'receipt.json').read_text())
        for r in (receipt, saved):
            self.assertIs(r['mirror_orientation_verified'], False)
            self.assertIs(r['outputs']['k94_mirror_tta']['mirror_orientation_verified'], False)
            self.assertIn('unverified', r['mirror_orientation_assumption'])
            self.assertIn('dominant axis', r['orientation_audit_required_before_tta_interpretation'])
            self.assertNotIn('mirror_orientation_verified', r['outputs']['k94_plain'])
        self.assertFalse(bool(self.npz(out, 'k94_mirror_tta')['mirror_orientation_verified']))

    def test_corrupt_primary_reference_fails_closed_before_any_k94_work_and_writes_no_arm_output(self):
        k16 = None
        self.run_it(self.root / 'first')
        k16 = self.npz(self.root / 'first', 'k16_plain')['probabilities']
        oof = self.root / 'oof_fold0.csv'
        rows = [[sid, *[repr(float(v)) for v in row], 'val'] for sid, row in zip(self.ids, k16)]
        def write(table):
            with oof.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['StudyInstanceUID', *sec.TARGETS, 'set'])
                writer.writerows(table)
        nan_row = [rows[0][0], 'nan', *rows[0][2:]]
        high_row = [rows[0][0], '1.5', *rows[0][2:]]
        cases = {'duplicate row': rows + [rows[1]], 'nan': [nan_row, *rows[1:]], 'out of range': [high_row, *rows[1:]],
                 'short row': [rows[0][:-2] + rows[0][-1:], *rows[1:]]}
        for name, table in cases.items():
            write(table)
            for enforce in (True, False):         # a corrupt reference is never a tolerance question
                out = self.root / f'reference-{name.replace(" ", "-")}-{enforce}'
                with self.subTest(name, enforce=enforce), self.assertRaises(ValueError):
                    self.run_it(out, saved_oof=oof, enforce_parity=enforce)
                saved = json.loads((out / 'receipt.json').read_text())
                self.assertEqual(saved['status'], 'FAIL')
                self.assertEqual(sorted(os.listdir(out)), ['receipt.json'])      # the corrupt reference stops the run before any arm output

    # ----- CLI orchestration: the frozen primary is bound and qualified before any corpus open
    def cli_argv(self, out=None, sha=None, mode=('--ungated-mirror-tta',)):
        trainer = self.root / 'trainer_source.py'
        trainer.write_text(self.source)
        return ['--work', str(self.work), '--trainer-source', str(trainer), '--trainer-sha256', sha or self.sha, '--labels', 'labels.parquet',
                '--truth', 'truth.parquet', '--out', str(out or self.root / 'cli-out'), '--device', 'cpu', '--skip-parity', *mode]

    def test_cli_binds_and_qualifies_primary_before_any_corpus_open(self):
        self.write_primary(res=sec.RES)       # the CLI is pinned to the frozen 224 resolution
        (self.work / 'folds_v0.json').write_text(json.dumps({'folds': {u: 0 for u in self.ids}}))
        (self.work / 'excluded_ids.json').write_text(json.dumps({'quarantined_study_ids': []}))
        events, verified = [], []
        real_bind = sec.bind_checkpoint

        def bind(*args, **kwargs):
            events.append('bind')
            return real_bind(*args, **kwargs)

        def definitions(source, sha, **kwargs):
            events.append('definitions')
            self.assertEqual(sha, self.sha)
            return self.defs

        def corpus(**kwargs):
            events.append('corpus')
            return self.vols, self.masks, self.ids

        def verify(labels, truth, folds, excluded, dev, **kwargs):
            events.append('verify_inputs')
            verified.append(list(dev))

        with patch.object(sec, 'bind_checkpoint', bind), patch.object(sec, 'load_definitions', definitions), \
                patch.object(guard, 'staged_corpus', corpus), patch.object(guard, 'verify_export_inputs', verify), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            sec.main(self.cli_argv())
        self.assertEqual(events[:4], ['bind', 'definitions', 'corpus', 'verify_inputs'])    # guard first, corpus only afterwards
        self.assertEqual(events[-1], 'bind')           # run_secondary re-binds immediately before inference
        self.assertEqual(verified, [self.ids])
        saved = json.loads((self.root / 'cli-out' / 'receipt.json').read_text())
        self.assertEqual(saved['status'], 'PASS')
        self.assertIs(saved['mirror_orientation_verified'], False)
        self.assertIs(json.loads(printed.getvalue())['mirror_orientation_verified'], False)

    def test_cli_invalid_primary_cannot_open_the_corpus(self):
        import torch
        opened = []

        def spy(*args, **kwargs):
            opened.append('corpus opener called')
            raise AssertionError('corpus opened before the frozen primary was bound and qualified')

        def record(**changes):
            record_path = self.work / 'run_fold0.json'
            data = json.loads(record_path.read_text())
            data.update(changes)
            record_path.write_text(json.dumps(data))

        def tamper():
            state = torch.load(self.checkpoint, weights_only=True)
            state['model'] = {k: v + 1 if v.is_floating_point() else v for k, v in state['model'].items()}
            torch.save(state, self.checkpoint)

        def wrong_identity():
            state = torch.load(self.checkpoint, weights_only=True)
            state['res'] = 31
            torch.save(state, self.checkpoint)
            record(checkpoint_sha256=smoke.file_sha(self.checkpoint))     # the record agrees: only identity is wrong

        cases = {'no primary record': lambda: (self.work / 'run_fold0.json').unlink(),
                 'k42 record': lambda: record(export={'k_eval': 42}),
                 'other checkpoint sha': lambda: record(checkpoint_sha256='0' * 64),
                 'other epochs': lambda: record(selected_epochs=[2, 4, 8]),
                 'unfinished training': lambda: (self.work / 'raptor_ft_fold0.json').write_text(json.dumps({'partial': False, 'hist': [{}] * 9})),
                 'tampered checkpoint': tamper, 'wrong identity': wrong_identity,
                 'missing checkpoint': lambda: self.checkpoint.unlink()}
        for name, mutate in cases.items():
            out = self.root / ('cli-' + name.replace(' ', '-'))
            with self.subTest(name):
                self.write_primary(res=sec.RES)
                mutate()
                with patch.object(guard, 'staged_corpus', spy), patch.object(guard, 'verify_export_inputs', spy), patch.object(loader, 'open_corpus', spy):
                    with self.assertRaises(ValueError):
                        sec.main(self.cli_argv(out))
                self.assertEqual(opened, [])
                self.assertFalse(out.exists())

    def test_cli_unpinned_trainer_source_cannot_open_the_corpus(self):
        self.write_primary(res=sec.RES)       # qualified primary: only the trainer digest is wrong
        opened = []

        def spy(*args, **kwargs):
            opened.append('corpus opener called')
            raise AssertionError('corpus opened before the trainer definitions were verified')

        with patch.object(guard, 'staged_corpus', spy), patch.object(guard, 'verify_export_inputs', spy), patch.object(loader, 'open_corpus', spy):
            with self.assertRaisesRegex(ValueError, 'pinned digest'):
                sec.main(self.cli_argv(sha='0' * 64))
        self.assertEqual(opened, [])
        self.assertFalse((self.root / 'cli-out').exists())


# ---------------------------------------------------------------------------------------------- pinned orientation artifacts
ORIENT_VERSION = 'orientation-audit-kernel-v2'
REAL_AUDIT = ROOT / 'outputs/kaggle/orientation-audit-v2-20261012-result-v1/orientation_audit'
REAL_CONTRACT = ROOT / 'outputs/kaggle/orientation-audit-v2-20261012/build-contract.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


class SyntheticAudit:
    """Writes a small synthetic header-audit artifact set (same structure and hash rules as knee.orientation_audit) and its pins.

    edit(docs) runs after the consistent documents and their integrity hashes are built, so it creates exactly one
    inconsistency; file pins are then computed from the final bytes (a tamper must trip the guard under test, not the file hash).
    """
    NAMES = {'aggregate': 'train-orientation.json', 'routing': 'routing-private-train.json', 'contract': 'build-contract.json'}

    def __init__(self, root, ids, routes, edit=None, repin=True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        ids, routes = list(ids), list(routes)
        contract = {'version': ORIENT_VERSION, 'train_series_csv_sha256': 'c' * 64, 'synthetic': True}
        canonical = sha(json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        counts = {r: routes.count(r) for r in sec.ROUTES}
        n = len(ids)
        ids_hash, routing_hash = sha('\n'.join(ids).encode()), sha(json.dumps(routes).encode())
        shares = {'Sagittal': 954, 'Coronal': 971, 'Axial': 981}
        aggregate = {'version': ORIENT_VERSION, 'status': 'COMPLETE', 'studies_done': n, 'studies_planned': n, 'labels_read': False,
                     'pixels_decoded': False, 'identifiers_in_aggregate': False, 'private_routing_file': self.NAMES['routing'],
                     'config': {'split': 'train', 'contract_sha256': canonical, 'series_csv_sha256': 'c' * 64},
                     'aggregate': {'studies_total': n, 'study_routing': counts, 'ordered_study_ids_sha256': ids_hash,
                                   'routing_array_sha256': routing_hash},
                     'descriptive_summary': {'threshold_fraction': 0.99, 'min_studies': 100, 'role': 'descriptive only; per-study routing governs',
                                             'per_plane': {p: {'all_ok_standard': k, 'studies': 1000, 'fraction': k / 1000, 'at_or_above_threshold': False}
                                                           for p, k in shares.items()}}}
        routing = {'version': ORIENT_VERSION, 'private': True, 'split': 'train', 'routes': list(sec.ROUTES), 'study_ids': ids, 'routing': routes,
                   'slot_names': ['sag0', 'sag1', 'cor0', 'cor1', 'ax'], 'slot_route': [[r] * 5 for r in routes],
                   'slot_status': [['ok_standard'] * 5 for _ in routes]}
        self.docs = {'aggregate': aggregate, 'routing': routing, 'contract': contract}
        if edit:
            edit(self.docs)
        self.paths = {k: self.root / name for k, name in self.NAMES.items()}
        for key, path in self.paths.items():
            path.write_text(json.dumps(self.docs[key], indent=1))
        self.pins = {'orientation_version': ORIENT_VERSION, 'split': 'train', 'studies': n,
                     'contract_canonical_sha256': canonical, 'ordered_study_ids_sha256': ids_hash, 'routing_array_sha256': routing_hash}
        for key, name in self.NAMES.items():
            self.pins[f'{key}_file'] = name
            self.pins[f'{key}_file_sha256'] = sha(self.paths[key].read_bytes())

    def load(self, **overrides):
        return sec.load_routing_artifacts(self.paths['aggregate'], self.paths['routing'], self.paths['contract'], pins={**self.pins, **overrides})

    def append(self, key):
        with self.paths[key].open('ab') as stream:
            stream.write(b' ')


IDS5 = [f'synthetic-study-{i}' for i in range(5)]
ROUTES5 = ['mirror_ok', 'bypass_plain', 'mirror_ok', 'needs_transform', 'mirror_ok']


class RoutingArtifactTests(unittest.TestCase):
    """Dependency-free: strict consumption of the pinned audit artifacts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def build(self, name='a', ids=IDS5, routes=ROUTES5, **kwargs):
        return SyntheticAudit(self.root / name, ids, routes, **kwargs)

    def test_valid_artifacts_load_and_record_the_failed_descriptor(self):
        audit = self.build()
        verified = audit.load()
        self.assertEqual(verified.population_route_counts, {'mirror_ok': 3, 'bypass_plain': 1, 'needs_transform': 1})
        self.assertEqual(len(verified), 5)
        descriptor = verified.descriptor
        self.assertFalse(descriptor['all_planes_pass'])
        self.assertEqual(descriptor['threshold_fraction'], 0.99)
        self.assertEqual({p: v['passes_99_percent'] for p, v in descriptor['per_plane'].items()}, {'Sagittal': False, 'Coronal': False, 'Axial': False})
        self.assertAlmostEqual(descriptor['per_plane']['Sagittal']['all_standard_share'], 0.954)
        self.assertFalse(verified.receipt['pins_are_frozen_default'])     # synthetic pins are never mistaken for the frozen ones
        self.assertEqual(set(verified.receipt['file_sha256']), set(SyntheticAudit.NAMES.values()))
        self.assertNotIn(IDS5[0], json.dumps(verified.receipt) + json.dumps(descriptor))

    def test_join_is_by_study_id_not_position(self):
        verified = self.build().load()
        shuffled = [IDS5[3], IDS5[0], IDS5[4], IDS5[1]]                    # different order and a subset of the audited studies
        self.assertEqual(sec.join_routes(shuffled, verified), ['needs_transform', 'mirror_ok', 'mirror_ok', 'bypass_plain'])
        self.assertEqual(sec.join_routes(IDS5, verified), ROUTES5)

    def test_missing_route_for_an_evaluation_study_is_refused(self):
        verified = self.build().load()
        with self.assertRaisesRegex(ValueError, 'Missing routes for 1 '):
            sec.join_routes([*IDS5[:2], 'not-audited'], verified)
        with self.assertRaisesRegex(ValueError, 'Missing routes for 2 '):
            sec.join_routes(['x', 'y'], verified)

    def test_duplicate_evaluation_ids_and_forged_routes_are_refused(self):
        verified = self.build().load()
        with self.assertRaisesRegex(ValueError, 'Unique'):
            sec.join_routes([IDS5[0], IDS5[0]], verified)
        with self.assertRaises(ValueError):
            sec.join_routes([], verified)
        with self.assertRaisesRegex(ValueError, 'load_routing_artifacts'):
            sec.join_routes(IDS5, dict(zip(IDS5, ROUTES5)))                 # an unchecked mapping is not accepted
        with self.assertRaisesRegex(ValueError, 'only by load_routing_artifacts'):
            sec.VerifiedRouting(object(), {}, '', '', {}, {}, {}, {})

    def test_duplicate_routing_ids_are_refused(self):
        audit = self.build('dup', ids=[*IDS5[:4], IDS5[0]])                 # every hash is consistent with the duplicated list
        with self.assertRaisesRegex(ValueError, 'Duplicate study IDs'):
            audit.load()

    def test_unknown_route_strings_are_refused(self):
        audit = self.build('unknown', routes=['mirror_ok', 'maybe_ok', 'mirror_ok', 'bypass_plain', 'mirror_ok'])
        with self.assertRaisesRegex(ValueError, 'Unknown route'):
            audit.load()

    def test_altered_file_bytes_fail_each_file_pin(self):
        for key in ('aggregate', 'routing', 'contract'):
            with self.subTest(key):
                audit = self.build('hash-' + key)
                audit.load()
                audit.append(key)
                with self.assertRaisesRegex(ValueError, 'file hash differs'):
                    audit.load()

    def test_altered_contract_is_refused(self):
        def edit(docs):
            docs['contract']['train_series_csv_sha256'] = 'd' * 64
        # (a) contract edited and its FILE pin refreshed: the canonical-contract pin still refuses it
        audit = self.build('contract-a', edit=edit)
        with self.assertRaisesRegex(ValueError, 'Contract hash differs'):
            audit.load()
        # (b) both contract pins refreshed too: the aggregate still refers to the original contract
        canonical = sha(json.dumps(audit.docs['contract'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        with self.assertRaisesRegex(ValueError, 'different contract'):
            audit.load(contract_canonical_sha256=canonical)

    def test_changed_id_order_and_route_array_are_refused(self):
        def reorder(docs):
            ids = docs['routing']['study_ids']
            ids[0], ids[1] = ids[1], ids[0]
        with self.assertRaisesRegex(ValueError, 'order differs from its hash'):
            self.build('order', edit=reorder).load()

        def reroute(docs):
            docs['routing']['routing'][1] = 'mirror_ok'
            docs['routing']['slot_route'][1] = ['mirror_ok'] * 5
        with self.assertRaisesRegex(ValueError, 'Route array differs from its hash'):
            self.build('route', edit=reroute).load()

    def test_other_integrity_failures_are_refused(self):
        def set_(path, value):
            def edit(docs):
                node = docs[path[0]]
                for k in path[1:-1]:
                    node = node[k]
                node[path[-1]] = value
            return edit
        cases = {'partial audit': set_(('aggregate', 'status'), 'RUNNING'), 'labels read': set_(('aggregate', 'labels_read'), True),
                 'identifiers in aggregate': set_(('aggregate', 'identifiers_in_aggregate'), True),
                 'relaxed descriptor threshold': set_(('aggregate', 'descriptive_summary', 'threshold_fraction'), 0.9),
                 'count mismatch': set_(('aggregate', 'aggregate', 'study_routing'), {'mirror_ok': 5, 'bypass_plain': 0, 'needs_transform': 0}),
                 'wrong split': set_(('routing', 'split'), 'test'), 'not private': set_(('routing', 'private'), False),
                 'other version': set_(('routing', 'version'), 'v1'), 'short per-slot array': set_(('routing', 'slot_route'), []),
                 'route vocabulary': set_(('routing', 'routes'), ['mirror_ok']),
                 'descriptor flag': set_(('aggregate', 'descriptive_summary', 'per_plane', 'Sagittal', 'at_or_above_threshold'), True)}
        for name, edit in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.build('case-' + name.replace(' ', '-'), edit=edit).load()

    def test_file_name_and_symlink_are_checked(self):
        audit = self.build('names')
        renamed = audit.root / 'copy-of-routing.json'
        renamed.write_bytes(audit.paths['routing'].read_bytes())
        with self.assertRaisesRegex(ValueError, 'file name differs'):
            sec.load_routing_artifacts(audit.paths['aggregate'], renamed, audit.paths['contract'], pins=audit.pins)
        link_dir = self.root / 'links'
        link_dir.mkdir()
        link = link_dir / 'routing-private-train.json'
        link.symlink_to(audit.paths['routing'])
        with self.assertRaisesRegex(ValueError, 'file required'):
            sec.load_routing_artifacts(audit.paths['aggregate'], link, audit.paths['contract'], pins=audit.pins)

    def test_verified_routing_payload_is_immutable(self):
        verified = self.build().load()
        before = sec.join_routes(IDS5, verified)
        with self.assertRaises(TypeError):
            verified.route_by_id[IDS5[1]] = 'mirror_ok'                      # in-place route edit: bypass_plain -> mirror_ok
        for call in (lambda: verified.route_by_id.update({IDS5[1]: 'mirror_ok'}), lambda: verified.route_by_id.pop(IDS5[1]),
                     lambda: verified.route_by_id.__delitem__(IDS5[1])):
            with self.assertRaises((AttributeError, TypeError)):
                call()
        for name in ('route_by_id', 'ordered_study_ids_sha256', 'routing_array_sha256', '_counts', '_pins', '_receipt', 'population_route_counts', 'new_attribute'):
            with self.subTest(name), self.assertRaises(AttributeError):
                setattr(verified, name, {} if name == 'route_by_id' else 'x')   # replacing the route map or any attribute
            with self.subTest('delete ' + name), self.assertRaises(AttributeError):
                delattr(verified, name)
        verified.descriptor['all_planes_pass'] = True                        # nested data is handed out as copies
        verified.population_route_counts['mirror_ok'] = 99
        verified.receipt['pins_are_frozen_default'] = True
        verified.pins['routing_array_sha256'] = 'x'
        self.assertFalse(verified.descriptor['all_planes_pass'])
        self.assertEqual(verified.population_route_counts['mirror_ok'], 3)
        self.assertFalse(verified.receipt['pins_are_frozen_default'])
        self.assertEqual(verified.pins['routing_array_sha256'], self.build('again').pins['routing_array_sha256'])
        self.assertEqual(sec.join_routes(IDS5, verified), before)

    def test_join_revalidates_the_content_binding_at_use(self):
        import gc
        verified = self.build().load()
        underlying = gc.get_referents(verified.route_by_id)[0]
        underlying[IDS5[1]] = 'mirror_ok'                                    # swapped behind the read-only view
        with self.assertRaisesRegex(ValueError, 'no longer match'):
            sec.join_routes(IDS5, verified)
        underlying[IDS5[1]] = 'bypass_plain'
        self.assertEqual(sec.join_routes(IDS5, verified), ROUTES5)
        object.__setattr__(verified, 'route_by_id', {**verified.route_by_id, IDS5[1]: 'mirror_ok'})     # attribute replaced behind __setattr__
        with self.assertRaisesRegex(ValueError, 'no longer match'):
            sec.join_routes(IDS5, verified)
        reordered = self.build('reordered').load()
        object.__setattr__(reordered, 'route_by_id', dict(reversed(list(reordered.route_by_id.items()))))   # same routes per ID, other order
        with self.assertRaisesRegex(ValueError, 'ID order'):
            sec.join_routes(IDS5, reordered)

    def test_forged_verified_routing_with_original_hashes_is_refused(self):
        verified = self.build().load()
        altered = {**verified.route_by_id, IDS5[1]: 'mirror_ok'}
        with self.assertRaisesRegex(ValueError, 'no longer match'):
            sec.VerifiedRouting(sec._VERIFIED, altered, verified.ordered_study_ids_sha256, verified.routing_array_sha256,
                                verified.population_route_counts, verified.descriptor, verified.pins, verified.receipt)
        ids_hash, routing_hash, counts = sec.route_hashes(altered)        # hashes recomputed for the altered routes, pins still original
        with self.assertRaisesRegex(ValueError, 'no longer match'):
            sec.VerifiedRouting(sec._VERIFIED, altered, ids_hash, routing_hash, counts, verified.descriptor, verified.pins, verified.receipt)
        with self.assertRaisesRegex(ValueError, 'Unknown route'):
            sec.VerifiedRouting(sec._VERIFIED, {**verified.route_by_id, IDS5[1]: 'maybe'}, verified.ordered_study_ids_sha256,
                                verified.routing_array_sha256, verified.population_route_counts, verified.descriptor, verified.pins, verified.receipt)
        genuine = sec.VerifiedRouting(sec._VERIFIED, verified.route_by_id, verified.ordered_study_ids_sha256, verified.routing_array_sha256,
                                      verified.population_route_counts, verified.descriptor, verified.pins, verified.receipt)
        self.assertEqual(sec.join_routes(IDS5, genuine), ROUTES5)

    def test_default_pins_are_the_frozen_audit_and_route_vocabulary_matches_the_audit_module(self):
        from knee import orientation_audit
        self.assertEqual(sec.ROUTES, orientation_audit.ROUTES)
        self.assertEqual(sec.DESCRIPTOR_MIN_FRACTION, orientation_audit.DESCRIPTIVE_MIN_FRACTION)
        self.assertEqual(sec.ORIENTATION_PINS['studies'], 4407)
        self.assertEqual(sec.ORIENTATION_PINS['aggregate_file_sha256'], 'bb44eedc9c487aee653b0cea7e4e93256e05e65d68fe8fccdcbc039965ded92a')
        self.assertIs(sec.UNIVERSAL_GEOMETRY_VERIFIED, False)
        self.assertIs(sec.MIRROR_ORIENTATION_VERIFIED, False)

    @unittest.skipUnless(all((REAL_AUDIT / name).is_file() for name in ('train-orientation.json', 'routing-private-train.json'))
                         and REAL_CONTRACT.is_file(), 'Real audit artifacts not present locally')
    def test_real_audit_artifacts_verify_under_the_frozen_pins(self):
        verified = sec.load_routing_artifacts(REAL_AUDIT / 'train-orientation.json', REAL_AUDIT / 'routing-private-train.json', REAL_CONTRACT)
        self.assertEqual(verified.population_route_counts, {'mirror_ok': 4128, 'bypass_plain': 279, 'needs_transform': 0})
        self.assertTrue(verified.receipt['pins_are_frozen_default'])
        self.assertFalse(verified.descriptor['all_planes_pass'])
        shares = {p: v['all_standard_share'] for p, v in verified.descriptor['per_plane'].items()}
        self.assertAlmostEqual(shares['Sagittal'], 0.95462, places=4)
        self.assertAlmostEqual(shares['Coronal'], 0.97164, places=4)
        self.assertAlmostEqual(shares['Axial'], 0.98162, places=4)
        self.assertEqual(len(verified), 4407)


@unittest.skipUnless(HAS_NUMPY, 'NumPy unavailable')
class ComparisonHelperTests(unittest.TestCase):
    """compare_plain_vs_gated only computes; synthetic labels only."""

    @staticmethod
    def brute_auc(y, p):
        pos, neg = [a for a, b in zip(p, y) if b == 1], [a for a, b in zip(p, y) if b == 0]
        if not pos or not neg:
            return None
        return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))

    def data(self, n=40, seed=0):
        import numpy as np
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, size=(n, 12))
        m = rng.integers(0, 2, size=(n, 12))
        m[:, 11] = 1
        y[:, 11] = 1                                        # target 11: one class only, so its AUC is undefined
        plain = np.round(rng.random((n, 12)), 1).astype(np.float32)   # rounding creates ties
        eligible = rng.random(n) < 0.6
        gated = plain.copy()
        gated[eligible] = np.clip(plain[eligible] + rng.normal(0, .1, size=(int(eligible.sum()), 12)), 0, 1).astype(np.float32)
        return y, m, plain, gated, eligible

    def test_auc_matches_brute_force_including_ties_and_support(self):
        import numpy as np
        y, m, plain, gated, eligible = self.data()
        result = sec.compare_plain_vs_gated(y, m, plain, gated, eligible)
        for population, rows in (('full_population', np.ones(len(y), bool)), ('eligible_subset', eligible)):
            for arm, pred in (('plain', plain), ('gated', gated)):
                for t, target in enumerate(sec.TARGETS):
                    use = rows & (m[:, t] == 1)
                    entry = result[population][arm]['per_target'][target]
                    expected = self.brute_auc(y[use, t].tolist(), pred[use, t].tolist())
                    if expected is None:
                        self.assertIsNone(entry['auc'])
                    else:
                        self.assertAlmostEqual(entry['auc'], expected, places=12)
                    self.assertEqual((entry['support'], entry['positives'], entry['negatives']),
                                     (int(use.sum()), int((y[use, t] == 1).sum()), int((y[use, t] == 0).sum())))
        self.assertEqual(result['full_population']['studies'], len(y))
        self.assertEqual(result['eligible_subset']['studies'], int(eligible.sum()))
        self.assertEqual(result['full_population']['plain']['undefined_targets'], [sec.TARGETS[11]])
        defined = [v['auc'] for v in result['full_population']['gated']['per_target'].values() if v['auc'] is not None]
        self.assertAlmostEqual(result['full_population']['gated']['macro_auc'], sum(defined) / len(defined))
        self.assertEqual(result['full_population']['gated']['defined_targets'], 11)
        json.dumps(result, allow_nan=False)                 # JSON-safe: no NaN

    def test_equal_arms_give_zero_delta_and_non_eligible_rows_must_match_plain(self):
        import numpy as np
        y, m, plain, _gated, eligible = self.data(seed=1)
        same = sec.compare_plain_vs_gated(y, m, plain, plain.copy(), eligible)
        self.assertTrue(all(v in (0.0, None) for v in same['eligible_subset']['gated_minus_plain_auc'].values()))
        bad = plain.copy()
        index = int((~eligible).nonzero()[0][0])
        bad[index, 0] = np.float32((float(bad[index, 0]) + 0.37) % 1.0)
        with self.assertRaisesRegex(ValueError, 'non-eligible'):
            sec.compare_plain_vs_gated(y, m, plain, bad, eligible)

    def test_inputs_are_not_mutated_and_bad_inputs_are_refused(self):
        import numpy as np
        y, m, plain, gated, eligible = self.data(seed=2)
        arrays = [a.copy() for a in (y, m, plain, gated, eligible)]
        for a in (y, m, plain, gated, eligible):
            a.flags.writeable = False
        sec.compare_plain_vs_gated(y, m, plain, gated, eligible)
        for before, after in zip(arrays, (y, m, plain, gated, eligible)):
            self.assertTrue(np.array_equal(before, after))
        for name, args in {'labels not binary': (y + 1, m, plain, gated, eligible), 'mask not binary': (y, m + 2, plain, gated, eligible),
                           'prob > 1': (y, m, plain, gated + 1, eligible), 'nan': (y, m, plain, gated * np.nan, eligible),
                           'eligible not bool': (y, m, plain, gated, eligible.astype(int)), 'shape': (y[:-1], m, plain, gated, eligible)}.items():
            with self.subTest(name), self.assertRaises(ValueError):
                sec.compare_plain_vs_gated(*args)

    def test_empty_eligible_subset_is_undefined_not_an_error(self):
        import numpy as np
        y, m, plain, _gated, _eligible = self.data(seed=3)
        result = sec.compare_plain_vs_gated(y, m, plain, plain.copy(), np.zeros(len(y), bool))
        self.assertEqual(result['eligible_subset']['studies'], 0)
        self.assertIsNone(result['eligible_subset']['plain']['macro_auc'])
        self.assertEqual(result['eligible_subset']['plain']['defined_targets'], 0)


class RoutedEndToEndTests:
    """Plain-only and header-gated modes through the real module path (mixed into each definitions provider)."""
    ROUTES = ROUTES5

    def audit(self, routes=None, name='audit', ids=None, **kwargs):
        return SyntheticAudit(self.root / name, ids or self.ids, routes or self.ROUTES, **kwargs)

    def run_gated(self, out=None, routes=None, audit_name='audit', **kwargs):
        audit = self.audit(routes, audit_name)
        return audit, self.run_it(out or self.root / 'gated', mode=sec.MODE_GATED, routing=audit.load(), **kwargs)

    def mirrored(self):
        import numpy as np
        return (np.stack([sec.mirror_volume(v) for v in self.vols]), np.stack([sec.mirror_mask(m) for m in self.masks]))

    def test_gated_arm_set_flags_and_membership_receipt(self):
        out = self.root / 'gated'
        audit, receipt = self.run_gated(out)
        self.assertEqual(receipt['status'], 'PASS')
        self.assertEqual(receipt['mode'], sec.MODE_GATED)
        self.assertEqual(receipt['version'], sec.VERSION_ROUTED)
        self.assertEqual(tuple(receipt['outputs']), ('k16_plain', 'k94_plain', 'k94_header_gated_approx_lr_tta'))
        self.assertEqual(sec.GATED_ARM, 'k94_header_gated_approx_lr_tta')
        saved = json.loads((out / 'receipt.json').read_text())
        for r in (receipt, saved):
            self.assertIs(r['mirror_orientation_verified'], False)
            self.assertIs(r['universal_geometry_verified'], False)
            arm = r['outputs'][sec.GATED_ARM]
            self.assertIs(arm['mirror_orientation_verified'], False)
            self.assertIs(arm['universal_geometry_verified'], False)
            self.assertTrue(arm['bypass_rows_byte_identical_to_k94_plain'] and arm['full_membership_preserved'])
            self.assertEqual((arm['mirrored_studies'], arm['plain_passthrough_studies'], arm['rows']), (3, 2, 5))
            self.assertNotIn('mirror_orientation_verified', r['outputs']['k94_plain'])
            routing = r['routing']
            self.assertEqual(routing['full_population']['studies'], 5)
            self.assertEqual(routing['eligible_subset']['studies'], 3)
            self.assertEqual(routing['route_counts'], {'mirror_ok': 3, 'bypass_plain': 1, 'needs_transform': 1})
            self.assertEqual(routing['full_population']['ids_sha256'], smoke.digest(smoke.canonical(self.ids)))
            self.assertEqual(r['ids_sha256'], routing['full_population']['ids_sha256'])
            eligible = [i for i, route in zip(self.ids, self.ROUTES) if route == 'mirror_ok']
            self.assertEqual(routing['eligible_subset']['ids_sha256'], smoke.digest(smoke.canonical(eligible)))
            self.assertNotEqual(routing['eligible_subset']['ids_sha256'], routing['full_population']['ids_sha256'])
            self.assertEqual(routing['artifacts']['ordered_study_ids_sha256'], audit.pins['ordered_study_ids_sha256'])
            self.assertEqual(routing['artifacts']['routing_array_sha256'], audit.pins['routing_array_sha256'])
            self.assertFalse(routing['cohort_descriptor_99_percent']['all_planes_pass'])
            self.assertEqual(set(routing['cohort_descriptor_99_percent']['per_plane']), {'Sagittal', 'Coronal', 'Axial'})
            self.assertIn('label invariance', routing['note'])
            self.assertEqual(r['window_plans']['k94_mirror_view']['windows'], 94 * 3)   # only the eligible studies are mirrored
        text = (out / 'receipt.json').read_text()
        self.assertFalse(any(u in text for u in self.ids))

    def test_bypass_rows_are_byte_identical_to_plain_and_eligible_rows_are_the_equal_mean(self):
        import numpy as np
        out = self.root / 'gated'
        self.run_gated(out)
        plain = self.npz(out, 'k94_plain')['probabilities']
        gated = self.npz(out, sec.GATED_ARM)
        tta = gated['probabilities']
        bypass = [i for i, r in enumerate(self.ROUTES) if r != 'mirror_ok']
        eligible = [i for i, r in enumerate(self.ROUTES) if r == 'mirror_ok']
        self.assertEqual(bypass, [1, 3])
        for i in bypass:
            self.assertEqual(tta[i].tobytes(), plain[i].tobytes())
        self.assertEqual(tta[bypass].tobytes(), plain[bypass].tobytes())
        plain_csv = list(csv.reader((out / 'k94_plain.csv').read_text().splitlines()))
        gated_csv = list(csv.reader((out / f'{sec.GATED_ARM}.csv').read_text().splitlines()))
        for i in bypass:
            self.assertEqual(plain_csv[1 + i], gated_csv[1 + i])
        mv, mm = self.mirrored()
        mirror_reference = self.reference(94, mv, mm, group=1)
        np.testing.assert_allclose(gated['mirror_view_eligible'], mirror_reference[eligible], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(gated['plain_view'], plain)
        expected = (0.5 * (plain[eligible] + gated['mirror_view_eligible'])).astype(np.float32)
        np.testing.assert_array_equal(tta[eligible], expected)
        self.assertGreater(float(np.abs(tta[eligible] - plain[eligible]).max()), 1e-6)    # the mirror is applied to mirror_ok studies
        self.assertEqual(gated['eligible'].tolist(), [True, False, True, False, True])
        self.assertEqual(gated['routes'].tolist(), self.ROUTES)

    def test_full_membership_is_preserved_in_every_artifact_and_order_follows_the_evaluation_ids(self):
        out = self.root / 'gated'
        self.run_gated(out)
        for arm in ('k16_plain', 'k94_plain', sec.GATED_ARM):
            self.assertEqual(self.npz(out, arm)['ids'].tolist(), self.ids)
            rows = list(csv.reader((out / f'{arm}.csv').read_text().splitlines()))
            self.assertEqual([r[0] for r in rows[1:]], self.ids)
        # the route join is by study ID: evaluating the studies in a different order moves the routes with the IDs
        order = [4, 2, 0, 3, 1]
        ids = [self.ids[i] for i in order]
        audit = self.audit()
        receipt = self.run_it(self.root / 'reordered', ids=ids, mode=sec.MODE_GATED, routing=audit.load())
        self.assertEqual(receipt['routing']['route_counts'], {'mirror_ok': 3, 'bypass_plain': 1, 'needs_transform': 1})
        data = self.npz(self.root / 'reordered', sec.GATED_ARM)
        self.assertEqual(data['routes'].tolist(), [self.ROUTES[i] for i in order])
        plain = self.npz(self.root / 'reordered', 'k94_plain')['probabilities']
        for position, i in enumerate(order):
            if self.ROUTES[i] != 'mirror_ok':
                self.assertEqual(data['probabilities'][position].tobytes(), plain[position].tobytes())

    def test_all_bypass_equals_plain_and_all_eligible_equals_the_legacy_ungated_arm(self):
        import numpy as np
        out = self.root / 'none'
        _audit, receipt = self.run_gated(out, routes=['bypass_plain', 'needs_transform'] * 2 + ['bypass_plain'], audit_name='none-audit')
        self.assertEqual(receipt['routing']['eligible_subset']['studies'], 0)
        self.assertEqual(receipt['window_plans']['k94_mirror_view'].get('windows', 0), 0)
        self.assertEqual(self.npz(out, sec.GATED_ARM)['probabilities'].tobytes(), self.npz(out, 'k94_plain')['probabilities'].tobytes())
        all_ok = self.root / 'all'
        self.run_gated(all_ok, routes=['mirror_ok'] * 5, audit_name='all-audit')
        legacy = self.root / 'legacy'
        self.run_it(legacy)
        np.testing.assert_array_equal(self.npz(all_ok, sec.GATED_ARM)['probabilities'], self.npz(legacy, 'k94_mirror_tta')['probabilities'])
        for arm in ('k16_plain', 'k94_plain'):          # the plain arms do not depend on the mode
            np.testing.assert_array_equal(self.npz(all_ok, arm)['probabilities'], self.npz(legacy, arm)['probabilities'])

    def test_inputs_and_masks_are_not_mutated_by_gated_inference(self):
        import numpy as np
        vols, masks = self.vols.copy(), self.masks.copy()
        before = (vols.tobytes(), masks.tobytes())
        vols.flags.writeable = False
        masks.flags.writeable = False
        audit = self.audit()
        verified = audit.load()
        route_before = dict(verified.route_by_id)
        ids = list(self.ids)
        self.run_it(self.root / 'ro', vols=vols, masks=masks, ids=ids, mode=sec.MODE_GATED, routing=verified)
        self.assertEqual((vols.tobytes(), masks.tobytes()), before)
        self.assertEqual(ids, self.ids)
        self.assertEqual(verified.route_by_id, route_before)

    def test_gated_refuses_missing_and_unverified_routing_before_creating_outputs(self):
        audit = self.audit(name='partial', ids=self.ids[:4], routes=self.ROUTES[:4])        # no route for the fifth evaluation study
        out = self.root / 'missing'
        with self.assertRaisesRegex(ValueError, 'Missing routes for 1 '):
            self.run_it(out, mode=sec.MODE_GATED, routing=audit.load())
        self.assertFalse(out.exists())
        for routing in (None, dict(zip(self.ids, self.ROUTES))):
            with self.assertRaisesRegex(ValueError, 'verified routing'):
                self.run_it(self.root / 'forged', mode=sec.MODE_GATED, routing=routing)
        self.assertFalse((self.root / 'forged').exists())

    def test_gated_k16_parity_is_still_enforced(self):
        first = self.run_it(self.root / 'first')
        k16 = self.npz(self.root / 'first', 'k16_plain')['probabilities']
        oof = self.root / 'oof_fold0.csv'
        def write(delta):
            lines = [','.join(['StudyInstanceUID', *sec.TARGETS, 'set'])]
            for sid, row in zip(self.ids, k16):
                lines.append(','.join([sid, *[repr(float(v) + delta) for v in row], 'val']))
            oof.write_text('\n'.join(lines) + '\n')
        write(0.0)
        passed = self.run_it(self.root / 'pass', saved_oof=oof, mode=sec.MODE_GATED, routing=self.audit(name='a1').load())
        self.assertTrue(passed['parity']['k16_vs_primary']['within_tolerance'])
        self.assertEqual(passed['parity']['k16_vs_primary']['absolute_tolerance'], 1e-4)
        write(0.01)
        with self.assertRaisesRegex(ValueError, 'primary export'):
            self.run_it(self.root / 'fail', saved_oof=oof, mode=sec.MODE_GATED, routing=self.audit(name='a2').load())
        self.assertEqual(json.loads((self.root / 'fail/receipt.json').read_text())['status'], 'PARITY_FAIL')
        self.assertEqual(first['status'], 'PASS')

    # ----- K16 parity first
    def oof_file(self, delta=0.0):
        self.run_it(self.root / 'oof-reference', mode=sec.MODE_PLAIN)
        k16 = self.npz(self.root / 'oof-reference', 'k16_plain')['probabilities']
        oof = self.root / f'oof-{delta}.csv'
        lines = [','.join(['StudyInstanceUID', *sec.TARGETS, 'set'])]
        for sid, row in zip(self.ids, k16):
            lines.append(','.join([sid, *[repr(float(v) + delta) for v in row], 'val']))
        oof.write_text('\n'.join(lines) + '\n')
        return oof

    def spied_run(self, out, **kwargs):
        """Run with spies on every model forward (batch, windows) and every window-centre plan (k); returns (calls, plans, result)."""
        calls, plans = [], []
        real_predict, real_plan = sec.predict, sec.center_plan

        def predict(model, x, device, fp16):
            calls.append((int(x.shape[0]), int(x.shape[1])))
            return real_predict(model, x, device, fp16)

        def center_plan(mask, sampler, k):
            plans.append(k)
            return real_plan(mask, sampler, k)

        with patch.object(sec, 'predict', predict), patch.object(sec, 'center_plan', center_plan):
            result = self.run_it(out, **kwargs)
        return calls, plans, result

    def mode_kwargs(self, mode, name):
        return {'mode': mode, 'routing': self.audit(name=name).load() if mode == sec.MODE_GATED else None}

    def test_parity_mismatch_stops_after_k16_with_no_k94_or_mirror_forward_and_no_arm_output(self):
        oof = self.oof_file(0.01)
        for mode in (sec.MODE_PLAIN, sec.MODE_GATED, sec.MODE_UNGATED):
            out = self.root / f'mismatch-{mode}'
            calls, plans, forwarded = [], [], []
            real_predict, real_plan = sec.predict, sec.center_plan

            def predict(model, x, device, fp16):
                calls.append((int(x.shape[0]), int(x.shape[1])))
                return real_predict(model, x, device, fp16)

            def center_plan(mask, sampler, k):
                plans.append(k)
                return real_plan(mask, sampler, k)

            def boom(*args, **kwargs):
                raise AssertionError('no mirror work may happen after a K16 parity mismatch')

            def no_arm_write(*args, **kwargs):
                raise AssertionError('no arm output may be written after a K16 parity mismatch')

            with self.subTest(mode), patch.object(sec, 'predict', predict), patch.object(sec, 'center_plan', center_plan), \
                    patch.object(sec, 'mirror_volume', boom), patch.object(sec, 'mirror_mask', boom), \
                    patch.object(sec, 'write_npz', no_arm_write), patch.object(sec, 'write_csv', no_arm_write):
                with self.assertRaisesRegex(ValueError, 'primary export'):
                    self.run_it(out, saved_oof=oof, **self.mode_kwargs(mode, f'a-{mode}'))
            self.assertEqual(calls, [(2, 16), (2, 16), (1, 16)], mode)           # only the three K16 batches of 2+2+1 studies
            self.assertEqual(set(plans), {16}, mode)
            self.assertEqual(plans.count(16), 5, mode)
            self.assertEqual(sorted(os.listdir(out)), ['receipt.json'], mode)
            saved = json.loads((out / 'receipt.json').read_text())
            self.assertEqual(saved['status'], 'PARITY_FAIL')
            self.assertEqual(saved['outputs'], {})
            self.assertIs(saved['k94_and_mirror_views_run'], False)
            self.assertEqual(set(saved['window_plans']), {'k16_plain'})
            self.assertFalse(saved['parity']['k16_vs_primary']['within_tolerance'])
            self.assertGreater(saved['parity']['k16_vs_primary']['max_absolute_error'], 1e-4)
            self.assertEqual(saved['k16_completed_studies'], 5)
            self.assertEqual(saved['completed_studies'], 0)

    def test_parity_pass_keeps_k16_first_ordering_full_membership_and_identical_results(self):
        import numpy as np
        oof = self.oof_file(0.0)
        for mode in (sec.MODE_GATED, sec.MODE_UNGATED, sec.MODE_PLAIN):
            with self.subTest(mode):
                calls, plans, checked = self.spied_run(self.root / f'checked-{mode}', saved_oof=oof, **self.mode_kwargs(mode, f'b-{mode}'))
                n_mirrored = {sec.MODE_GATED: 3, sec.MODE_UNGATED: 5, sec.MODE_PLAIN: 0}[mode]
                self.assertEqual(calls, [(2, 16), (2, 16), (1, 16)] + [(1, 94)] * (5 + n_mirrored))     # every K16 forward precedes every K94 forward
                self.assertEqual(plans[:5], [16] * 5)
                self.assertNotIn(16, plans[5:])
                self.assertTrue(checked['parity']['k16_vs_primary']['within_tolerance'])
                self.assertEqual(checked['completed_studies'], 5)
                _calls, _plans, unchecked = self.spied_run(self.root / f'unchecked-{mode}', **self.mode_kwargs(mode, f'c-{mode}'))
                self.assertEqual(tuple(checked['outputs']), sec.MODE_ARMS[mode])
                for arm in sec.MODE_ARMS[mode]:
                    a, b = self.npz(self.root / f'checked-{mode}', arm), self.npz(self.root / f'unchecked-{mode}', arm)
                    self.assertEqual(a['probabilities'].tobytes(), b['probabilities'].tobytes())
                    self.assertEqual(a['ids'].tolist(), self.ids)
                    self.assertEqual(checked['outputs'][arm]['csv_sha256'], unchecked['outputs'][arm]['csv_sha256'])
                self.assertEqual(checked['window_plans'], unchecked['window_plans'])
                if mode == sec.MODE_GATED:
                    plain = self.npz(self.root / f'checked-{mode}', 'k94_plain')['probabilities']
                    gated = self.npz(self.root / f'checked-{mode}', sec.GATED_ARM)['probabilities']
                    for i in (1, 3):
                        self.assertEqual(gated[i].tobytes(), plain[i].tobytes())

    def test_unenforced_parity_mismatch_is_recorded_after_k16_and_the_run_continues(self):
        oof = self.oof_file(0.01)
        calls, plans, receipt = self.spied_run(self.root / 'unenforced', saved_oof=oof, enforce_parity=False, **self.mode_kwargs(sec.MODE_GATED, 'd'))
        self.assertEqual(calls[:3], [(2, 16), (2, 16), (1, 16)])
        self.assertEqual(calls[3:], [(1, 94)] * 8)
        self.assertEqual(receipt['status'], 'PASS')
        self.assertFalse(receipt['parity']['k16_vs_primary']['within_tolerance'])
        self.assertFalse(receipt['parity']['k16_vs_primary']['enforced'])
        self.assertEqual(tuple(receipt['outputs']), sec.MODE_ARMS[sec.MODE_GATED])
        self.assertTrue((self.root / 'unenforced' / f'{sec.GATED_ARM}.csv').exists())

    # ----- plain-only mode
    def test_plain_only_runs_without_routing_or_any_mirror(self):
        import numpy as np
        out = self.root / 'plain'
        def boom(*args, **kwargs):
            raise AssertionError('plain_only must not mirror anything')
        with patch.object(sec, 'mirror_volume', boom), patch.object(sec, 'mirror_mask', boom), \
                patch.object(sec, 'load_routing_artifacts', boom), patch.object(sec, 'join_routes', boom):
            receipt = self.run_it(out, mode=sec.MODE_PLAIN)
        self.assertEqual(receipt['status'], 'PASS')
        self.assertEqual(tuple(receipt['outputs']), ('k16_plain', 'k94_plain'))
        self.assertEqual(receipt['mode'], sec.MODE_PLAIN)
        self.assertFalse(receipt['routing']['used'])
        self.assertNotIn('k94_mirror_view', receipt['window_plans'])
        self.assertIs(receipt['mirror_orientation_verified'], False)
        self.assertIs(receipt['universal_geometry_verified'], False)
        self.assertFalse(any(name.startswith('k94_mirror') or 'gated' in name for name in os.listdir(out)))
        legacy = self.run_it(self.root / 'legacy')
        for arm in ('k16_plain', 'k94_plain'):
            np.testing.assert_array_equal(self.npz(out, arm)['probabilities'], self.npz(self.root / 'legacy', arm)['probabilities'])
            self.assertEqual(receipt['outputs'][arm]['k'], legacy['outputs'][arm]['k'])

    def test_python_api_default_mode_is_plain_only_and_never_mirrors_or_reads_routing(self):
        import inspect
        self.assertEqual(inspect.signature(sec.run_secondary).parameters['mode'].default, sec.MODE_PLAIN)
        def boom(*args, **kwargs):
            raise AssertionError('an omitted mode must not mirror or open routing')
        settings = dict(work=self.work, fold=0, checkpoint=self.checkpoint, definitions=self.defs, vols=self.vols,
                        masks=self.masks, ids=self.ids, id2row=self.id2row, out=self.root / 'default-mode',
                        device='cpu', res=TEST_RES, batch_studies=2, k94_win_chunk=16)   # no mode argument
        with patch.object(sec, 'mirror_volume', boom), patch.object(sec, 'mirror_mask', boom), \
                patch.object(sec, 'load_routing_artifacts', boom), patch.object(sec, 'join_routes', boom):
            receipt = sec.run_secondary(**settings)
        self.assertEqual(receipt['mode'], sec.MODE_PLAIN)
        self.assertEqual(tuple(receipt['outputs']), ('k16_plain', 'k94_plain'))
        self.assertFalse(receipt['routing']['used'])

    def test_plain_only_rejects_routing_and_k16_parity_still_enforced(self):
        verified = self.audit().load()
        with self.assertRaisesRegex(ValueError, 'plain_only'):
            self.run_it(self.root / 'with-routing', mode=sec.MODE_PLAIN, routing=verified)
        self.assertFalse((self.root / 'with-routing').exists())
        with self.assertRaisesRegex(ValueError, 'ungated'):
            self.run_it(self.root / 'legacy-routing', mode=sec.MODE_UNGATED, routing=verified)
        with self.assertRaisesRegex(ValueError, 'Unknown mode'):
            self.run_it(self.root / 'bad-mode', mode='gated')
        self.run_it(self.root / 'first')
        k16 = self.npz(self.root / 'first', 'k16_plain')['probabilities']
        oof = self.root / 'oof_fold0.csv'
        lines = [','.join(['StudyInstanceUID', *sec.TARGETS, 'set'])]
        for sid, row in zip(self.ids, k16):
            lines.append(','.join([sid, *[repr(float(v) + 0.01) for v in row], 'val']))
        oof.write_text('\n'.join(lines) + '\n')
        with self.assertRaisesRegex(ValueError, 'primary export'):
            self.run_it(self.root / 'plain-fail', saved_oof=oof, mode=sec.MODE_PLAIN)
        self.assertEqual(json.loads((self.root / 'plain-fail/receipt.json').read_text())['status'], 'PARITY_FAIL')

    # ----- legacy ungated arm unchanged
    def test_legacy_ungated_arm_keeps_its_semantics_and_is_marked_ungated(self):
        import numpy as np
        out = self.root / 'legacy'
        receipt = self.run_it(out)
        self.assertEqual(receipt['mode'], sec.MODE_UNGATED)
        self.assertEqual(receipt['version'], sec.VERSION)
        self.assertEqual(tuple(receipt['outputs']), sec.ARMS)
        self.assertIs(receipt['ungated'], True)
        self.assertIs(receipt['outputs']['k94_mirror_tta']['ungated'], True)
        self.assertIn('dominant axis', receipt['orientation_audit_required_before_tta_interpretation'])
        self.assertNotIn('routing', receipt)
        self.assertEqual(receipt['window_plans']['k94_mirror_view']['windows'], 94 * 5)    # every study is mirrored
        tta = self.npz(out, 'k94_mirror_tta')
        self.assertEqual(tta['mirror_view'].shape, (5, 12))
        np.testing.assert_array_equal(tta['probabilities'], (0.5 * (tta['plain_view'] + tta['mirror_view'])).astype(np.float32))

    # ----- CLI
    def prepare_cli(self):
        self.write_primary(res=sec.RES)
        (self.work / 'folds_v0.json').write_text(json.dumps({'folds': {u: 0 for u in self.ids}}))
        (self.work / 'excluded_ids.json').write_text(json.dumps({'quarantined_study_ids': []}))

    def cli_patches(self, events, audit=None):
        real_load = sec.load_routing_artifacts

        def definitions(source, sha, **kwargs):
            events.append('definitions')
            return self.defs

        def corpus(**kwargs):
            events.append('corpus')
            return self.vols, self.masks, self.ids

        def verify(*args, **kwargs):
            events.append('verify_inputs')

        def routing(aggregate, routes, contract):
            events.append('routing')
            return real_load(aggregate, routes, contract, pins=audit.pins)

        patches = [patch.object(sec, 'load_definitions', definitions), patch.object(guard, 'staged_corpus', corpus),
                   patch.object(guard, 'verify_export_inputs', verify)]
        if audit is not None:
            patches.append(patch.object(sec, 'load_routing_artifacts', routing))
        return patches

    def gated_argv(self, audit, out=None):
        return self.cli_argv(out, mode=('--header-gated-tta', '--orientation-aggregate', str(audit.paths['aggregate']),
                                        '--orientation-routing', str(audit.paths['routing']), '--orientation-contract', str(audit.paths['contract'])))

    def test_cli_plain_only_skips_routing_artifacts_entirely(self):
        self.prepare_cli()
        events = []
        def boom(*args, **kwargs):
            raise AssertionError('plain-only must not touch routing artifacts')
        with contextlib.ExitStack() as stack:
            for patcher in self.cli_patches(events):
                stack.enter_context(patcher)
            stack.enter_context(patch.object(sec, 'load_routing_artifacts', boom))
            stack.enter_context(patch.object(sec, 'join_routes', boom))
            stack.enter_context(patch.object(sec, 'read_pinned_file', boom))
            printed = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            sec.main(self.cli_argv(mode=('--plain-only',)))
        self.assertEqual(events, ['definitions', 'corpus', 'verify_inputs'])
        saved = json.loads((self.root / 'cli-out' / 'receipt.json').read_text())
        self.assertEqual((saved['status'], saved['mode']), ('PASS', sec.MODE_PLAIN))
        self.assertEqual(tuple(saved['outputs']), ('k16_plain', 'k94_plain'))
        self.assertEqual(json.loads(printed.getvalue())['mode'], sec.MODE_PLAIN)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            sec.main(self.cli_argv(self.root / 'cli-bad', mode=('--plain-only', '--orientation-aggregate', 'x')))

    def test_cli_requires_exactly_one_explicit_mode_and_all_three_artifacts_for_gating(self):
        self.prepare_cli()
        with contextlib.redirect_stderr(io.StringIO()):
            for mode in ((), ('--plain-only', '--ungated-mirror-tta'), ('--header-gated-tta',),
                         ('--header-gated-tta', '--orientation-aggregate', 'a', '--orientation-routing', 'r'),
                         ('--ungated-mirror-tta', '--orientation-contract', 'c')):
                with self.subTest(mode), self.assertRaises(SystemExit):
                    sec.main(self.cli_argv(mode=mode))

    def test_cli_gated_verifies_routing_after_the_checkpoint_and_before_the_corpus(self):
        self.prepare_cli()
        audit = self.audit()
        events = []
        real_bind = sec.bind_checkpoint
        def bind(*args, **kwargs):
            events.append('bind')
            return real_bind(*args, **kwargs)
        with contextlib.ExitStack() as stack:
            for patcher in self.cli_patches(events, audit):
                stack.enter_context(patcher)
            stack.enter_context(patch.object(sec, 'bind_checkpoint', bind))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            sec.main(self.gated_argv(audit))
        self.assertEqual(events[:5], ['bind', 'routing', 'definitions', 'corpus', 'verify_inputs'])
        saved = json.loads((self.root / 'cli-out' / 'receipt.json').read_text())
        self.assertEqual((saved['status'], saved['mode']), ('PASS', sec.MODE_GATED))
        self.assertEqual(saved['routing']['eligible_subset']['studies'], 3)

    def test_cli_gated_with_altered_artifacts_cannot_open_the_corpus(self):
        self.prepare_cli()
        opened = []
        def spy(*args, **kwargs):
            opened.append('corpus opener called')
            raise AssertionError('corpus opened before the routing artifacts were verified')
        for name in ('aggregate', 'routing', 'contract'):
            audit = self.audit(name='altered-' + name)
            audit.append(name)
            events = []
            with self.subTest(name), contextlib.ExitStack() as stack:
                for patcher in self.cli_patches(events, audit):
                    stack.enter_context(patcher)
                stack.enter_context(patch.object(guard, 'staged_corpus', spy))
                stack.enter_context(patch.object(guard, 'verify_export_inputs', spy))
                stack.enter_context(patch.object(loader, 'open_corpus', spy))
                with self.assertRaisesRegex(ValueError, 'file hash differs'):
                    sec.main(self.gated_argv(audit, self.root / ('cli-' + name)))
                self.assertEqual(opened, [])
                self.assertEqual(events, ['routing'])
                self.assertFalse((self.root / ('cli-' + name)).exists())

    def test_cli_gated_missing_route_for_a_development_study_fails_before_outputs(self):
        self.prepare_cli()
        audit = self.audit(name='short', ids=self.ids[:4], routes=self.ROUTES[:4])
        events = []
        with contextlib.ExitStack() as stack:
            for patcher in self.cli_patches(events, audit):
                stack.enter_context(patcher)
            with self.assertRaisesRegex(ValueError, 'Missing routes'):
                sec.main(self.gated_argv(audit))
        self.assertFalse((self.root / 'cli-out').exists())


class StandInDefinitionsTests(EndToEndTests, RoutedEndToEndTests, _Fixture, unittest.TestCase):
    pass


@unittest.skipUnless(PINNED_TRAINER.is_file() and HAS_TORCH, 'Byte-pinned public trainer source unavailable')
class PinnedTrainerTests(EndToEndTests, RoutedEndToEndTests, _Fixture, unittest.TestCase):
    """Same tests, but StudyWindows/RaptorClassifier/build_backbone are the real pinned definitions."""
    sha = smoke.SOURCE_SHA

    @property
    def source(self):
        return PINNED_TRAINER.read_text()

    def test_stand_in_windows_equal_pinned_windows(self):
        import numpy as np
        import torch
        stand_in = _Fixture.definitions(types.SimpleNamespace(source=STAND_IN_TRAINER, sha=STAND_IN_SHA))
        labels = {u: np.zeros(12, np.float32) for u in self.ids}
        mirrored_v = np.stack([sec.mirror_volume(v) for v in self.vols])
        mirrored_m = np.stack([sec.mirror_mask(m) for m in self.masks])
        for k in (16, 94):
            for vols, masks in ((self.vols, self.masks), (mirrored_v, mirrored_m)):
                a = self.defs.StudyWindows('', self.ids, self.id2row, labels, TEST_RES, k, False, aug=False, norm='imagenet', roi=False)
                b = stand_in.StudyWindows('', self.ids, self.id2row, labels, TEST_RES, k, False, aug=False, norm='imagenet', roi=False)
                a.vols = b.vols = vols
                a.masks = b.masks = masks
                for j in range(len(self.ids)):
                    self.assertTrue(torch.equal(a[j][0], b[j][0]))


@unittest.skipUnless(HAS_TORCH, 'PyTorch unavailable')
class DefinitionLoaderTests(unittest.TestCase):
    def test_digest_and_inventory_are_enforced(self):
        with self.assertRaisesRegex(ValueError, 'pinned digest'):
            sec.load_definitions(STAND_IN_TRAINER + '\n# drift', STAND_IN_SHA)
        partial = STAND_IN_TRAINER.split('class RaptorClassifier')[0]
        with self.assertRaisesRegex(ValueError, 'inventory'):
            sec.load_definitions(partial, hashlib.sha256(partial.encode()).hexdigest())

    def test_timm_is_not_imported_until_a_backbone_is_built(self):
        with patch.dict('sys.modules', {'timm': None}):
            defs = sec.load_definitions(STAND_IN_TRAINER, STAND_IN_SHA)
            self.assertTrue(callable(defs.StudyWindows) and callable(defs.RaptorClassifier))
            with self.assertRaises(ImportError):
                defs.build_backbone('any-arch')


@unittest.skipUnless(HAS_NUMPY, 'NumPy unavailable')
class SavedPrimaryTests(unittest.TestCase):
    """The primary-export reference must be intact before it can serve as a parity reference."""
    IDS = ['study-a', 'study-b', 'study-c']

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'oof_fold0.csv'

    @staticmethod
    def row(sid, base=0.2, split='val'):
        return [sid, *[repr(base + 0.01 * i) for i in range(len(sec.TARGETS))], split]

    def write(self, rows, header=None):
        with self.path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(header or ['StudyInstanceUID', *sec.TARGETS, 'set'])
            writer.writerows(rows)

    def good(self):
        return [self.row(sid, 0.1 * (i + 1)) for i, sid in enumerate(self.IDS)]

    def with_value(self, sid, text, column=0):
        row = self.row(sid)
        row[1 + column] = text
        return row

    def test_valid_rows_are_aligned_to_the_requested_order_and_gold_rows_are_skipped(self):
        import numpy as np
        rows = self.good()
        self.write(rows + [self.row('gold-only', 0.9, 'gold')])
        out = sec.saved_primary(self.path, list(reversed(self.IDS)))
        self.assertEqual(out.shape, (3, 12))
        np.testing.assert_array_equal(out[0], [float(v) for v in rows[2][1:-1]])
        np.testing.assert_array_equal(out[2], [float(v) for v in rows[0][1:-1]])

    def test_unknown_set_labels_are_rejected_not_skipped(self):
        rows = self.good()
        for label in ('train', 'VAL', 'val ', 'test', ''):
            with self.subTest(label=label):
                self.write(rows + [self.row('extra', 0.5, label)])
                with self.assertRaisesRegex(ValueError, 'unknown set label'):
                    sec.saved_primary(self.path, self.IDS)

    def test_only_val_and_gold_are_permitted_set_labels(self):
        self.assertEqual(sec.PRIMARY_OOF_SETS, ('val', 'gold'))

    def test_duplicate_validation_rows_are_rejected_even_when_the_id_set_is_complete(self):
        rows = self.good()
        for name, duplicate in (('identical', rows[1]), ('conflicting', self.row(self.IDS[1], 0.7))):
            with self.subTest(name):
                self.write(rows + [duplicate])      # set(rows) == set(ids): a dict-based reader would silently accept this
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    sec.saved_primary(self.path, self.IDS)
        self.write([rows[0], rows[0], rows[1]])      # duplicate that also leaves one study uncovered
        with self.assertRaises(ValueError):
            sec.saved_primary(self.path, self.IDS)

    def test_malformed_rows_are_rejected(self):
        good = self.good()
        short_split = self.row(self.IDS[0])[:-1]                                      # 'set' field missing
        short_target = self.row(self.IDS[0])[:-2] + ['val']                           # one target missing
        long_row = self.row(self.IDS[0]) + ['0.5']
        for name, bad, pattern in (('missing set field', short_split, 'malformed row'), ('missing target', short_target, 'malformed row'),
                                   ('extra field', long_row, 'malformed row'), ('non-numeric', self.with_value(self.IDS[0], 'abc'), 'malformed value'),
                                   ('empty value', self.with_value(self.IDS[0], ''), 'malformed value')):
            with self.subTest(name):
                self.write([bad, *good[1:]])
                with self.assertRaisesRegex(ValueError, pattern):
                    sec.saved_primary(self.path, self.IDS)
        with self.subTest('malformed row in another set'):
            self.write(good + [self.row('gold-only', 0.9, 'gold')[:-2] + ['gold']])
            with self.assertRaisesRegex(ValueError, 'malformed row'):
                sec.saved_primary(self.path, self.IDS)
        with self.subTest('wrong header'):
            self.write(good, header=['StudyInstanceUID', *reversed(sec.TARGETS), 'set'])
            with self.assertRaisesRegex(ValueError, 'schema'):
                sec.saved_primary(self.path, self.IDS)

    def test_unexpected_missing_and_extra_study_ids_are_rejected(self):
        good = self.good()
        for name, rows in (('missing study', good[:2]), ('unexpected replaces one', [*good[:2], self.row('unknown')]),
                           ('extra study', good + [self.row('study-d')]), ('no validation rows', [self.row('x', split='gold')])):
            with self.subTest(name):
                self.write(rows)
                with self.assertRaisesRegex(ValueError, 'cover exactly'):
                    sec.saved_primary(self.path, self.IDS)
        self.write(good)
        with self.assertRaises(ValueError):
            sec.saved_primary(self.path, [*self.IDS, 'study-a'])       # the requested IDs must be unique too
        with self.assertRaises(ValueError):
            sec.saved_primary(self.path, [])

    def test_non_finite_reference_probabilities_are_rejected(self):
        good = self.good()
        for text in ('nan', 'NaN', 'inf', '-inf', '1e999'):
            for column in (0, len(sec.TARGETS) - 1):
                with self.subTest(text, column=column):
                    self.write([self.with_value(self.IDS[0], text, column), *good[1:]])
                    with self.assertRaisesRegex(ValueError, 'not finite within'):
                        sec.saved_primary(self.path, self.IDS)

    def test_out_of_range_reference_probabilities_are_rejected_and_bounds_are_accepted(self):
        import numpy as np
        good = self.good()
        for text in ('1.0000001', '-1e-9', '2', '-0.5'):
            with self.subTest(text):
                self.write([self.with_value(self.IDS[0], text, 5), *good[1:]])
                with self.assertRaisesRegex(ValueError, 'not finite within'):
                    sec.saved_primary(self.path, self.IDS)
        edge = self.row(self.IDS[0])
        edge[1], edge[2] = '0', '1.0'
        self.write([edge, *good[1:]])
        out = sec.saved_primary(self.path, self.IDS)
        self.assertEqual((float(out[0, 0]), float(out[0, 1])), (0.0, 1.0))
        self.assertTrue(np.isfinite(out).all())


@unittest.skipUnless(HAS_NUMPY, 'NumPy unavailable')
class HelperTests(unittest.TestCase):
    def test_development_ids_follow_cache_order_and_drop_excluded(self):
        ids = ['a', 'b', 'c', 'd', 'e']
        folds = {'a': 0, 'b': 1, 'c': 0, 'd': 0, 'e': 2}
        self.assertEqual(sec.development_ids(ids, folds, {'d'}, 0), ['a', 'c'])
        self.assertEqual(sec.development_ids(ids, {}, set(), 0), [])

    def test_probability_schema(self):
        import numpy as np
        sec.probabilities(np.full((2, 12), .5), 2)
        for bad in (np.full((2, 12), 1.5), np.full((2, 12), np.nan), np.full((1, 12), .5), np.full((2, 11), .5)):
            with self.assertRaises(ValueError):
                sec.probabilities(bad, 2)


if __name__ == '__main__':
    unittest.main()
