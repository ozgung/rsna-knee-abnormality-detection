"""R6-flip frame, observer, recovery and packet tests: CPU and synthetic only.

Nothing here touches a GPU, a DICOM or the network. Tests that need the ignored local evidence
(archived A-prime packet, pinned R6 sources/metadata, CPU pins) skip when it is absent.
"""
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from knee import public_r6flip as R6F
from knee import public_r6flip_entry as ENTRY
from knee import public_r6flip_worker as WORKER

try:
    MAIN = Path(subprocess.check_output(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], cwd=PROJECT,
                                        text=True, stderr=subprocess.DEVNULL).strip()).parent
except Exception:
    MAIN = PROJECT
EVIDENCE = MAIN / 'outputs/research/public-0958-20261017'
SOURCES = EVIDENCE / 'r6-v1-sources'
SMALL = EVIDENCE / 'r6-v1-small'

try:
    import numpy as np
    import pandas as pd
    import torch
except ImportError:          # the dependency-free default environment
    np = pd = torch = None


def pinned_compact_infer():
    from knee.public_r6_profile import SOURCE_SHA256
    path = SOURCES / 'compact_infer.py'
    if torch is None or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256['compact_infer.py']:
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location('r6flip_test_compact_infer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrameTests(unittest.TestCase):
    def test_frozen_weights_and_order(self):
        w = R6F.raptor_weight_vector()
        self.assertEqual(len(w), 12)
        expected = {'Lateral OA': .2, 'PF OA': .6, 'Synovitis': .8, "Baker's": .35}
        for name, value in zip(R6F.LABELS, w):
            self.assertEqual(value, expected.get(name, .5))
        self.assertEqual((R6F.FRAME['raptor_half']['old320'], R6F.FRAME['raptor_half']['crop224']), (.65, .35))
        self.assertTrue(R6F.FLIP_TTA)
        self.assertEqual(R6F.FRAME['excluded'], ['na-full', 'V2', 'extra readers', 'R224 mirror'])
        with self.assertRaises(ValueError):
            R6F.raptor_weight_vector(list(reversed(R6F.LABELS)))

    def test_import_is_dependency_free(self):
        code = ('import sys; sys.path.insert(0, %r); import knee.public_r6flip, knee.public_r6flip_worker, knee.public_r6flip_entry; '
                'bad=[m for m in ("numpy","torch","pandas","pydicom") if m in sys.modules]; print(bad); raise SystemExit(bool(bad))') % str(PROJECT)
        r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


@unittest.skipIf(np is None, 'numpy/pandas/torch not installed')
class RecoveryTests(unittest.TestCase):
    def arms(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        return [rng.random((n, 12)) for _ in range(3)], [np.zeros(n, bool) for _ in range(3)]

    def public_formula(self, old, r6, crop):
        """The captured notebook's stage-3 arithmetic with no failures (each model ranked over all rows)."""
        from knee.public_ws2_recovery import quantize9, source224_csv_values
        pct = lambda v: pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values
        raw224, ours = source224_csv_values(crop), quantize9(r6)
        out = np.zeros_like(old)
        w = R6F.raptor_weight_vector()
        for j in range(12):
            rr = (1.0 - .35) * pct(old[:, j]) + .35 * pct(raw224[:, j])
            out[:, j] = w[j] * rr + (1.0 - w[j]) * pct(ours[:, j])
        return out

    def test_blend_equals_public_arithmetic_and_aprime_half_weights(self):
        from knee import public_ws2_recovery as REC
        (old, r6, crop), masks = self.arms()
        rec = R6F.recover(old, r6, crop, *masks)
        self.assertTrue(np.array_equal(rec['blend'], self.public_formula(old, r6, crop)))
        aprime = REC.recover(old, r6, crop, *masks)['blend']
        half = [j for j, w in enumerate(R6F.raptor_weight_vector()) if w == .5]
        self.assertTrue(np.array_equal(rec['blend'][:, half], aprime[:, half]))
        self.assertFalse(np.array_equal(rec['blend'], aprime))
        self.assertEqual(rec['raptor_weights']['Synovitis'], .8)

    def test_r6_missing_rows_take_quantized_genuine_mean_under_union_cap(self):
        from knee.public_ws2_recovery import quantize9
        n = 100
        (old, r6, crop), masks = self.arms(n)
        r6[[3, 7]] = np.nan
        masks[1][[3, 7]] = True
        masks[0][[3]] = True          # union counts a study once
        old[3] = np.nan
        rec = R6F.recover(old, r6, crop, *masks)
        self.assertEqual((rec['cap'], rec['union_recovered']), (3, 2))
        mean = quantize9(quantize9(np.delete(r6, [3, 7], 0)).mean(0)[None])[0]
        self.assertEqual(rec['means']['r6'], mean.tolist())
        self.assertEqual(rec['statuses']['r6'][7], 'recovered')
        summary = R6F.recovery_receipt(rec)
        self.assertEqual(summary['recovered_by_arm'], {'old320': 1, 'r6': 2, 'crop224': 0})
        masks[2][[10, 11]] = True
        with self.assertRaisesRegex(ValueError, 'UNION_RECOVERY_CAP'):
            R6F.recover(old, r6, crop, *masks)

    def test_guarded_complete_population_equals_aprime_path(self):
        (old, r6, crop), masks = self.arms()
        st = {a: ['genuine'] * 40 for a in R6F.ARMS}
        rec = R6F.recover_guarded({'old320': old, 'r6': r6, 'crop224': crop}, st)
        self.assertEqual((rec['mode'], rec['fallback_studies']), ('aprime_complete_population', 0))
        self.assertTrue(np.array_equal(rec['blend'], R6F.recover(old, r6, crop, *masks)['blend']))

    def test_deadline_fallback_never_aborts_and_counts(self):
        from knee.public_ws2_recovery import quantize9, source224_csv_values
        n = 40
        (old, r6, crop), _ = self.arms(n)
        st = {'old320': ['genuine'] * n, 'crop224': ['genuine'] * n, 'r6': ['genuine'] * 20 + ['unavailable'] * 20}
        st['crop224'][0] = 'unavailable'
        st['old320'][25] = st['crop224'][25] = 'unavailable'
        r6[20:] = np.nan
        rec = R6F.recover_guarded({'old320': old, 'r6': r6, 'crop224': crop}, st)
        self.assertEqual(rec['mode'], 'deadline_fallback_public_semantics')
        self.assertEqual(rec['fallback_studies'], 21)
        self.assertEqual(rec['fallback'], {'raptor_only': 19, 'r6_only': 0, 'constant': 1, 'partial_raptor': 1})
        self.assertTrue(np.all(rec['blend'][25] == .5) and np.isfinite(rec['blend']).all())
        pct = lambda v: pd.Series(v).rank(method='average', pct=True).values
        ok_old = np.array([s == 'genuine' for s in st['old320']])
        ok_crop = np.array([s == 'genuine' for s in st['crop224']])
        r384 = np.full(n, np.nan); r384[ok_old] = pct(old[ok_old, 6])
        r224 = np.full(n, np.nan); r224[ok_crop] = pct(source224_csv_values(crop[ok_crop])[:, 6])
        ro = np.full(n, np.nan); ro[:20] = pct(quantize9(r6[:20])[:, 6])
        self.assertAlmostEqual(rec['blend'][0, 6], .6 * r384[0] + .4 * ro[0], places=12)        # PF OA, R224 missing
        self.assertAlmostEqual(rec['blend'][30, 6], .65 * r384[30] + .35 * r224[30], places=12)   # R6 missing
        st['r6'] = ['unavailable'] * n
        st['old320'] = st['crop224'] = ['unavailable'] * n
        self.assertTrue(np.all(R6F.recover_guarded({'old320': old, 'r6': r6, 'crop224': crop}, st)['blend'] == .5))

    def test_masks_must_be_explicit_booleans(self):
        (old, r6, crop), masks = self.arms()
        with self.assertRaisesRegex(ValueError, 'EXPLICIT_ROW_MASK_REQUIRED'):
            R6F.recover(old, r6, crop, masks[0].astype(int), masks[1], masks[2])
        with self.assertRaisesRegex(ValueError, 'FATAL_ERRORS_REQUIRE_ABORT'):
            R6F.recover(old, r6, crop, *masks, fatal_errors=['resource'])


@unittest.skipIf(torch is None, 'torch not installed')
class FlipTests(unittest.TestCase):
    spec = {'categories': {l: ['none', 'some'] for l in R6F.LABELS}, 'prior': {l: [.5, .5] for l in R6F.LABELS},
            'binaries': {'tear': ['Medial Meniscus', 'Lateral Meniscus'], 'severe': ['Medial OA', 'Lateral OA', 'PF OA']}}
    bins = [('tear', 'Medial Meniscus'), ('tear', 'Lateral Meniscus'), ('severe', 'Medial OA'), ('severe', 'Lateral OA'), ('severe', 'PF OA')]

    def test_permutation_is_the_published_involution(self):
        perm, bin_perm = R6F.mirror_perm(R6F.LABELS, self.spec, self.bins)
        self.assertEqual(perm, [0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11])
        self.assertEqual(bin_perm, [1, 0, 3, 2, 4])
        bad = json.loads(json.dumps(self.spec))
        bad['categories']['Lateral OA'] = ['none']
        with self.assertRaisesRegex(ValueError, 'FLIP_STATE_CATEGORIES'):
            R6F.mirror_perm(R6F.LABELS, bad, self.bins)
        with self.assertRaisesRegex(ValueError, 'FLIP_BINARY_PARTNER'):
            R6F.mirror_perm(R6F.LABELS, self.spec, self.bins[:1])

    def test_deployed_logits_average_mirror_with_mcl_plain(self):
        perm, _ = R6F.mirror_perm()
        z = [torch.arange(12, dtype=torch.float32) + 10 * k for k in range(4)]
        zm = [torch.arange(12, dtype=torch.float32) * -1 + 100 * k for k in range(4)]
        out = R6F.deployed_logits(z, zm, perm)
        for k in range(4):
            for j in range(12):
                expected = z[k][1] if j == 1 else .5 * (z[k][j] + zm[k][perm[j]])
                self.assertEqual(float(out[k, j]), float(expected))

    def test_mirror_inputs_and_readout_match_pinned_source(self):
        CI = pinned_compact_infer()
        g = torch.Generator().manual_seed(7)
        img = torch.randint(0, 256, (1, 12, 3, 8, 8), generator=g, dtype=torch.uint8)
        meta = torch.zeros((1, 12, 9))
        meta[0, :, 7] = torch.tensor([0, 1, 2] * 4, dtype=torch.float32)
        meta[0, :, 1] = torch.tensor([0, 1, 2, 3, 4, 5] * 2, dtype=torch.float32)
        meta[0, :, 2] = torch.linspace(-1, 1, 12)
        ours, ours_meta = R6F.mirror_inputs(img, meta)
        sag = meta[0, :, 7] == 0
        self.assertTrue(torch.equal(ours[0, sag], img[0, sag].flip(1)))
        self.assertTrue(torch.equal(ours[0, ~sag], img[0, ~sag].flip(-1)))
        self.assertTrue(torch.equal(ours_meta[0, sag, 2], -meta[0, sag, 2]))
        self.assertEqual(ours_meta[0, :, 1].tolist(), [0, 1, 3, 2, 4, 5] * 2)
        if CI is None:
            self.skipTest('pinned compact_infer.py not staged')
        a, b = CI.mirror_windows(img, meta)
        self.assertTrue(torch.equal(a, ours) and torch.equal(b, ours_meta))
        self.assertEqual(CI.mirror_perm(R6F.LABELS, self.spec, self.bins), tuple(R6F.mirror_perm(R6F.LABELS, self.spec, self.bins)))
        sl, bl = torch.randn(1, 12, 2), torch.randn(1, 5)
        holder = type('M', (), {'state_k': [2] * 12})()
        self.assertTrue(torch.equal(CI.state_readout(holder, sl, bl), R6F.state_readout(holder.state_k, sl, bl)))

    @unittest.skipUnless((SMALL / 'state_spec.json').is_file(), 'R6 metadata not staged')
    def test_reconstruction_with_actual_r6_stackers(self):
        from knee import public_r7_stackers as stackers
        spec = json.loads((SMALL / 'state_spec.json').read_bytes())
        layout = stackers.state_layout(spec, R6F.LABELS)
        state_k = [len(spec['categories'][l]) for l in R6F.LABELS]
        bins = [(b, l) for b, labs in spec['binaries'].items() for l in labs]
        fluid = stackers.load_fluid((SMALL / 'fluid_stacker.json').read_bytes(), R6F.LABELS)
        state = stackers.load_state((SMALL / 'state_stacker.json').read_bytes(), R6F.LABELS, layout,
                                    hashlib.md5((SMALL / 'state_spec.json').read_bytes()).hexdigest())
        g = torch.Generator().manual_seed(3)
        kmax = max(state_k)
        aux = lambda: (torch.randn(1, 12, kmax, generator=g), torch.randn(1, len(bins), generator=g))
        obs = {'plain': [torch.randn(12, generator=g) for _ in range(4)], 'mirror': [torch.randn(12, generator=g) for _ in range(4)],
               'plain_aux': [aux() for _ in range(4)], 'mirror_aux': [aux() for _ in range(4)], 'state_k': state_k, 'bin_names': bins}
        record = {'status': 'ok', 'eff_sag_p99_open': 12.5, 'bak_sag_p99_open': 3.0}
        p, detail = R6F.reconstruct(obs, record, fluid, state, spec)
        perm, bin_perm = R6F.mirror_perm(R6F.LABELS, spec, bins)
        zs = R6F.deployed_logits(obs['plain'], obs['mirror'], perm)
        p_model = torch.sigmoid(zs.mean(0)).double().numpy()
        untouched = [j for j, l in enumerate(R6F.LABELS) if l not in ('Effusion', "Baker's")]
        self.assertTrue(np.array_equal(p[untouched], p_model[untouched]))
        co = fluid['classes']['Effusion']
        z_eff = co['b0'] + co['b_z'] * float(zs.mean(0).double()[7]) + co['b_lv'] * math.log1p(12.5)
        self.assertAlmostEqual(p[7], 1 / (1 + math.exp(-z_eff)), places=12)
        self.assertNotEqual(p[9], p_model[9])
        fallback, _ = R6F.reconstruct(obs, record, fluid, state, spec, stack_fallback=["Baker's"])
        self.assertEqual(fallback[9], p_model[9])
        CI = pinned_compact_infer()
        if CI is not None:
            mcl = [i for c in CI.state_layout(spec, R6F.LABELS)['columns'] if c['class'] == 'MCL'
                   for i in (range(c['start'], c['stop']) if 'categories' in c else [c['column']])]
            self.assertEqual(mcl, R6F.mcl_state_columns(layout))


@unittest.skipIf(torch is None, 'torch not installed')
class DigestTests(unittest.TestCase):
    def test_compile_wrapper_and_layout_do_not_change_digest_mutation_does(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
        before, buffers, ctor = R6F.module_digest(net), R6F.buffers_digest(net), R6F.constructor_digest(net)
        outer = torch.nn.Module()
        holder = torch.nn.Module()
        holder._orig_mod = net
        outer.enc = holder
        plain = torch.nn.Module()
        plain.enc = net
        self.assertEqual(R6F.module_digest(outer), R6F.module_digest(plain))
        self.assertEqual(R6F.module_digest(net.to(memory_format=torch.channels_last)), before)
        self.assertEqual(R6F.buffers_digest(net), buffers)
        self.assertEqual(R6F.constructor_digest(net), ctor)
        sums = R6F.checksum(net)
        with torch.no_grad():
            net[0].weight[0, 0, 0, 0] += 1e-3
        self.assertNotEqual(R6F.module_digest(net), before)
        self.assertNotEqual(R6F.checksum(net), sums)

    def test_array_sha_matches_aprime_definition(self):
        from knee.public_s6_evaluation import array_sha
        for a in (np.arange(24, dtype=np.uint8).reshape(2, 3, 4), np.linspace(0, 1, 18, dtype=np.float32).reshape(2, 9)):
            self.assertEqual(WORKER.array_sha(a), array_sha(a))


class WorkerArgsTests(unittest.TestCase):
    manifest = {'arm': 'E_lrprior', 'backbone': 'convnext_small', 'size': 384, 'crop': 352, 'eval_stride': 1, 'views': 'v4',
                'state_spec': 'state_spec.json', 'side_model': 'side_cls.pt', 'side_threshold': 2.0, 'side_position_mm': 20.0,
                'side_max_x': 0.0, 'slices': 3, 'phys_mm': 3.5, 'models2': None, 'trainer_md5': 'abc', 'compile': True,
                'fluid_stacker': 'fluid_stacker.json', 'state_stacker': 'state_stacker.json'}

    def test_notebook_worker_args_with_flip_and_no_deadline(self):
        cfg = {'r6': {'manifest': self.manifest}, 'r6_mount': '/m', 'data': '/d', 'code': '/c', 'anatomy': '/a/anatomy.pt'}
        args = WORKER.worker_args(cfg, '/o/ids.json', '/o/compact')
        self.assertEqual(args, ['--data', '/d', '--split', 'train', '--code', '/c', '--arm', 'E_lrprior', '--assignment', '/o/ids.json',
                                '--output', '/o/compact', '--amp', 'auto', '--workers', '2', '--deadline', '0', '--models', '/m',
                                '--backbone', 'convnext_small', '--size', '384', '--crop', '352', '--eval-stride', '1', '--views', 'v4',
                                '--state-spec', '/m/state_spec.json', '--side-model', '/m/side_cls.pt', '--side-threshold', '2.0',
                                '--side-position-mm', '20.0', '--side-max-x', '0.0', '--slices', '3', '--phys-mm', '3.5',
                                '--trainer-md5', 'abc', '--compile', '--fluid-stacker', '/m/fluid_stacker.json', '--flip-tta',
                                '--state-stacker', '/m/state_stacker.json', '--anchor-weights', '/a/anatomy.pt'])


@unittest.skipIf(torch is None, 'torch not installed')
class ObserverTests(unittest.TestCase):
    """compact_infer-shaped synthetic flow through the real observer: binding, reload, flip input check, boundaries."""

    def fake_ct(self):
        class Net(torch.nn.Module):
            def __init__(self, backbone, init, hidden=4, freeze_blocks=16, crop=8, state_spec=None, in_chans=3, spatial=False,
                         extent=False, syn_decomp=False):
                super().__init__()
                self.enc = torch.nn.Linear(3, 2)
                self.head = torch.nn.Linear(2, 12)
                self.register_buffer('mean', torch.zeros(1))
                self.state_k = [2] * 12
                self.bin_names = [('tear', 'Medial Meniscus'), ('tear', 'Lateral Meniscus')]

            def forward(self, img, meta, mask, chunk=0, return_aux=False):
                x = img.float().mean(dim=(-1, -2))[0]
                z = self.head(self.enc(x)).mean(0, keepdim=True)
                if return_aux:
                    return z, torch.zeros(1, 12, 2), torch.zeros(1, 2)
                return z
        return type('CT', (), {'Net': Net, 'LABELS': R6F.LABELS})

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.CT = self.fake_ct()
        self.original_load = torch.load
        torch.manual_seed(0)
        model = self.CT.Net('convnext_small', None, crop=8, state_spec={'a': 1}, spatial=True, extent=True, syn_decomp=True)
        self.path = self.root / 'E_lrprior_s0_f0.pt'
        torch.save(model.state_dict(), self.path)
        import inspect
        bound = inspect.signature(self.CT.Net.__init__).bind(model, 'convnext_small', None, crop=8, state_spec={'a': 1},
                                                             spatial=True, extent=True, syn_decomp=True)
        bound.apply_defaults()
        args = {k: v for k, v in bound.arguments.items() if k != 'self'}
        args['state_spec'] = R6F.digest({'a': 1})
        self.pin = {'file': self.path.name, 'file_sha256': WORKER.file_sha(self.path), 'state': R6F.module_digest(model),
                    'buffers': R6F.buffers_digest(model), 'constructor': R6F.constructor_digest(model), 'constructor_args': args}

    def tearDown(self):
        self.tmp.cleanup()

    def observer(self, pin=None):
        cfg = {'r6': {'cpu_pins': {'folds': [pin or self.pin]}, 'manifest': {'crop': 8, 'slices': 3}}, 'inject_uid': None}
        obs = WORKER.Observer(cfg, self.root / 'obs', torch, self.CT)
        obs.install()
        return obs

    def load(self):
        sd = torch.load(self.path, map_location='cpu', weights_only=True)
        m = self.CT.Net('convnext_small', None, crop=8, state_spec={'a': 1}, spatial=True, extent=True, syn_decomp=True)
        m.load_state_dict(sd, strict=True)
        return m.eval().requires_grad_(False)

    def item(self, uid, w=4):
        img = torch.randint(0, 256, (w, 3, 8, 8), dtype=torch.uint8)
        meta = torch.zeros((w, 9))
        meta[:, 7] = torch.tensor([0, 1, 2, 1][:w], dtype=torch.float32)
        return {'uid': uid, 'error': None, 'images': img, 'meta': meta, 'seconds': [1., 2., 3.], 'fluid': {'status': 'ok'}}

    def score(self, m, item, mirror_fn=R6F.mirror_inputs):
        img, meta = item['images'][None], item['meta'][None]
        mask = torch.ones((1, img.shape[1]), dtype=torch.bool)
        m(img, meta, mask, chunk=192)                      # compile-check style call, not observed as scoring
        z = m(img, meta, mask, chunk=192, return_aux=True)
        img_m, meta_m = mirror_fn(img, meta)
        zm = m(img_m, meta_m, mask, chunk=192, return_aux=True)
        return z, zm

    def test_binding_reload_flip_check_and_boundaries(self):
        obs = self.observer()
        try:
            m = self.load()
            self.assertEqual(len(obs.models), 1)
            entry = obs.models[0]['entry']
            self.assertTrue(entry['strict'] and entry['reload']['state'] == entry['state'] == self.pin['state'])
            items = {}
            for uid in ('a', 'b'):
                items[uid] = item = self.item(uid)
                obs.begin(item, uid == 'a')
                self.score(m, item)
                obs.end()
            self.assertEqual([s['id'] for s in obs.studies], ['a', 'b'])
            s = obs.studies[0]
            self.assertTrue(s['plain'] and s['mirror'] and s['mirror_input_equal'] and s['checksum_equal'])
            self.assertEqual(s['compile_check_calls'], 1)
            self.assertEqual(s['input'], {'render_windows': WORKER.array_sha(items['a']['images'].numpy()),
                                          'metadata': WORKER.array_sha(items['a']['meta'].numpy())})
            o = WORKER.Observer.observation(s, obs.models)
            self.assertEqual((o['calls'], o['fp32_rescues']), (2, 2))     # CPU: autocast off, recorded as fp32 mode
            obs.begin(self.item('c'), False)
            with self.assertRaisesRegex(ValueError, 'R6_FLIP_INPUT_TRANSFORM'):
                self.score(m, self.item('c'), mirror_fn=lambda i, mm: (i.flip(-1), mm))
            obs.abort_current('test')
            self.assertEqual(obs.studies[-1]['aborted'], 'test')
            with torch.no_grad():
                m.head.bias += 1
            obs.begin(self.item('d'), False)
            with self.assertRaisesRegex(ValueError, 'R6_PARAMETER_BUFFER_MUTATION'):
                obs.end()
        finally:
            obs.uninstall()
        self.assertIs(torch.load, self.original_load)
        self.assertFalse(hasattr(self.CT.Net.forward, '__wrapped__'))

    def test_wrong_pin_rejects_at_load(self):
        obs = self.observer({**self.pin, 'state': '0' * 64})
        try:
            with self.assertRaisesRegex(ValueError, 'QUALIFIED_R6_FOLD_STATE'):
                self.load()
        finally:
            obs.uninstall()

    def test_signature_preserved_for_syn_decomp_detection(self):
        import inspect
        obs = self.observer()
        try:
            self.assertIn('syn_decomp', inspect.signature(self.CT.Net.__init__).parameters)
        finally:
            obs.uninstall()


@unittest.skipIf(torch is None, 'torch not installed')
class LoaderChildTests(unittest.TestCase):
    def test_child_binding_and_single_source_seam_injection(self):
        class DS:
            def __init__(self):
                self.uids, self.native = ['s0', 's1'], None

            def _setup(self):
                self.native = True
                self.bank = type('B', (), {'model': torch.nn.Linear(2, 2)})()
                self.side_net = torch.nn.Linear(3, 1)
                self.window_rows = lambda *a, **k: [1]

            def __getitem__(self, i):              # compact_infer's shape: every exception becomes an item error
                try:
                    if self.native is None:
                        self._setup()
                    if not self.window_rows():
                        raise ValueError('no usable window')
                    return {'uid': self.uids[i], 'error': None}
                except Exception as exc:
                    return {'uid': self.uids[i], 'error': repr(exc)}
        ds = DS()
        with tempfile.TemporaryDirectory() as d, \
                mock.patch('torch.utils.data.get_worker_info', return_value=type('I', (), {'dataset': ds, 'num_workers': 2})()):
            original = DS.__getitem__
            WORKER.child_init(d, 's0', 0)
            first, second = ds[0], ds[1]
            self.assertIn('no usable window', first['error'])
            self.assertIsNone(second['error'])
            self.assertEqual(ds.window_rows(), [1])
            receipt = json.loads(next(Path(d).glob('child-*.json')).read_bytes())
            self.assertEqual((receipt['items'], len(receipt['injected']), receipt['changed']), (2, 1, False))
            self.assertEqual(receipt['before'], receipt['after'])
            with torch.no_grad():
                ds.side_net.weight += 1
            ds[1]
            self.assertTrue(json.loads(next(Path(d).glob('child-*.json')).read_bytes())['changed'])
            DS.__getitem__ = original


@unittest.skipIf(np is None, 'numpy not installed')
class MergeProjectionTests(unittest.TestCase):
    def shards(self, root, ids, fail=None):
        from knee.public_s6_recovery_entry import assignments
        from knee.public_s6_recovery_runtime import save_ledger
        reports = []
        for board, part in enumerate(assignments(ids)):
            folder = root / ('board%d-r6-production' % board) / 'r6'
            folder.mkdir(parents=True)
            probs, statuses, logits, records = [], [], [], []
            for sid in part:
                bad = sid == fail
                p = np.full(12, np.nan) if bad else np.full(12, .25 + .01 * int(sid[1:]))
                probs.append(p)
                statuses.append('data_local_failure' if bad else 'genuine')
                logits.append(np.full((4, 12), np.nan if bad else .1, dtype=np.float32))
                records.append({'id': sid, 'status': statuses[-1], 'probabilities': None if bad else p.tolist(),
                                'production_seconds': 10., 'windows': 100})
            save_ledger(folder / 'predictions.npz', part, probs, statuses, logits=logits)
            reports.append({'version': R6F.VERSION, 'status': 'PASS', 'board': board, 'arm': 'r6', 'pass': 'production', 'assigned': part,
                            'completed': len(part), 'state_before': 's', 'state_after': 's', 'zero_imputed': True,
                            'ledger_sha256': WORKER.file_sha(folder / 'predictions.npz'), 'records': records,
                            'genuine': statuses.count('genuine'), 'failed': statuses.count('data_local_failure')})
        return reports

    def test_complete_population_merge_and_tampering(self):
        ids = ['s%d' % i for i in range(9)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reports = self.shards(root, ids, fail='s4')
            p, missing = ENTRY.merge_r6(ids, reports, root)
            self.assertEqual(missing.tolist(), [i == 4 for i in range(9)])
            self.assertTrue(np.isnan(p[4]).all() and np.isfinite(np.delete(p, 4, 0)).all())
            bad = json.loads(json.dumps(reports))
            bad[0]['assigned'] = bad[0]['assigned'][::-1]
            with self.assertRaisesRegex(ValueError, 'R6_WORKER_TERMINAL_GATE'):
                ENTRY.merge_r6(ids, bad, root)
            bad = json.loads(json.dumps(reports))
            bad[1]['ledger_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'R6_ATOMIC_LEDGER_BINDING'):
                ENTRY.merge_r6(ids, bad, root)
            bad = json.loads(json.dumps(reports))
            bad[0]['state_after'] = 'mutated'
            with self.assertRaisesRegex(ValueError, 'R6_WORKER_STATE_GATE'):
                ENTRY.merge_r6(ids, bad, root)
            with self.assertRaisesRegex(ValueError, 'R6_WORKER_TERMINAL_GATE'):
                ENTRY.merge_r6(ids + ['s9'], reports, root)
            bad = json.loads(json.dumps(reports))
            bad[0]['records'][0]['probabilities'][0] = .9
            with self.assertRaisesRegex(ValueError, 'R6_LEDGER_EQUALS_RECORD'):
                ENTRY.merge_r6(ids, bad, root)

    def test_projection_rule_and_gate(self):
        rows = lambda n, s, w=None: [dict({'production_seconds': s, 'status': 'genuine'}, **({'windows': w} if w else {})) for _ in range(n)]
        workers = {'old': [{'records': rows(20, 6.)}, {'records': rows(19, 6.)}],
                   'crop224': [{'records': rows(20, 3.)}, {'records': rows(19, 3.)}],
                   'r6': [{'records': rows(20, 25., 578)}, {'records': rows(19, 25., 578)}]}
        phases = {'old': {'wall_seconds': 600.}, 'crop224': {'wall_seconds': 150.}, 'r6': {'wall_seconds': 900.},
                  'r6_sentinel1': {'wall_seconds': 400.}, 'r6_sentinel2': {'wall_seconds': 400.}}
        d = [578] * 870
        p = ENTRY.projection(14., phases, workers, 1., d)
        overhead = (600 - 120) + (150 - 60) + (900 - 500 + 800)
        expected = 15 + overhead + 120 * 1300 / 39 * 1.15 + 60 * 1300 / 39 * 1.15 + 500 * (1300 * 578 * 1.25 / (39 * 578)) * 1.15
        self.assertAlmostEqual(p['conservative_seconds'], expected, places=6)
        self.assertEqual(p['gate_pass'], expected <= 8.3 * 3600)
        self.assertAlmostEqual(p['point_seconds'], 15 + overhead + 120 * 1300 / 39 + 60 * 1300 / 39 + 500 * 1300 / 39, places=6)
        with self.assertRaisesRegex(ValueError, 'FROZEN870'):
            ENTRY.projection(14., phases, workers, 1., d[:-1])


def r6_worker_receipt(cfg, board, ids, pins):
    folds = [dict(p, strict=True, reload={'state': p['state'], 'buffers': p['buffers'], 'constructor': p['constructor'], 'strict': True},
                  sentinel={'finite': True, 'max_abs_dprob': 1e-4, 'compiled_deployed': True}) for p in pins['folds']]
    child = {'before': {'side': dict(pins['side']), 'anatomy': {'state': 'anat'}}, 'changed': False, 'items': len(ids), 'injected': []}
    child['after'] = json.loads(json.dumps(child['before']))
    return {'version': R6F.VERSION, 'status': 'PASS', 'board': board, 'pass': 'production', 'assigned': ids, 'completed': len(ids),
            'environment': {'torch': cfg['binding']['environment']['torch'], 'packages': cfg['binding']['environment']['packages'],
                            'physical_uuid': 'GPU-%d' % board, 'cuda_device_count': 1},
            'folds': folds, 'state_before': R6F.digest([p['state'] for p in pins['folds']]),
            'state_after': R6F.digest([p['state'] for p in pins['folds']]), 'flip': dict(pins['flip']),
            'compile': {'requested': True, 'active': True, 'fallback': None}, 'fp32_recoveries': 0,
            'loaders': [{'num_workers': 2, 'prefetch_factor': 1, 'timeout': 900., 'context': 'spawn'}], 'children': [child],
            'peak_cuda_bytes': 1, 'peak_rss_bytes': 1, 'failed': 0, 'genuine': len(ids),
            'records': [{'id': s, 'status': 'genuine', 'reconstruction_max_abs': 0., 'mirror_input_equal': True, 'checksum_equal': True,
                         'fluid_status_ok': True, 'fp32_rescues': 0, 'stack_fallback': [], 'calls': 8} for s in ids]}


class R6WorkerGateTests(unittest.TestCase):
    pins = {'folds': [{'file': 'f%d' % i, 'file_sha256': 'h%d' % i, 'state': 's%d' % i, 'buffers': 'b%d' % i, 'constructor': 'c',
                       'constructor_args': {'crop': 352}} for i in range(4)],
            'side': {'state': 'side', 'buffers': 'sb', 'constructor': 'sc'}, 'flip': {'perm': [0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11],
                                                                                  'bin_perm': [1, 0]}}
    cfg = {'binding': {'environment': {'torch': '2.10.0+cu128', 'packages': {'timm': '1.0.26'}}},
           'r6flip_binding': {'r6': {'cpu_pins': pins}}}
    r = {'boards': [{'uuid': 'GPU-0'}, {'uuid': 'GPU-1'}]}

    def test_valid_receipt_passes_and_each_mutation_rejects(self):
        ids = ['a', 'b']
        good = r6_worker_receipt(self.cfg, 0, ids, self.pins)
        self.assertEqual(ENTRY.require_r6_worker(good, self.r, self.cfg, 0, 'production', ids), {'anat'})
        mutations = {
            'R6_FOLD_BINDING': lambda x: x['folds'][2].update(state='other'),
            'R6_FRESH_RELOAD_SENTINEL': lambda x: x['folds'][0]['sentinel'].update(max_abs_dprob=2e-3),
            'R6_STATE_BEFORE_AFTER': lambda x: x.update(state_after='mutated'),
            'R6_FLIP_PERMUTATION_PIN': lambda x: x['flip'].update(perm=list(range(12))),
            'R6_COMPILED_NO_FALLBACK': lambda x: x['compile'].update(fallback='eager'),
            'R6_NO_FP32_RESCUE': lambda x: x.update(fp32_recoveries=1),
            'R6_TWO_SPAWN_LOADER_WORKERS': lambda x: x['loaders'][0].update(num_workers=0),
            'R6_LOADER_CHILD_ANATOMY_SIDE_BINDING': lambda x: x['children'][0].update(changed=True),
            'R6_STUDY_GATES': lambda x: x['records'][1].update(reconstruction_max_abs=1e-3),
            'R6_ISOLATED_PINNED_BOARD': lambda x: x['environment'].update(physical_uuid='GPU-1'),
            'R6_WORKER_PASS': lambda x: x.update(assigned=['b', 'a']),
        }
        for code, mutate in mutations.items():
            x = json.loads(json.dumps(good))
            mutate(x)
            with self.assertRaisesRegex(ValueError, code):
                ENTRY.require_r6_worker(x, self.r, self.cfg, 0, 'production', ids)
        x = json.loads(json.dumps(good))
        x['children'][0]['before']['side']['state'] = 'other'
        x['children'][0]['after']['side']['state'] = 'other'
        with self.assertRaisesRegex(ValueError, 'R6_LOADER_CHILD_ANATOMY_SIDE_BINDING'):
            ENTRY.require_r6_worker(x, self.r, self.cfg, 0, 'production', ids)


ARCHIVE_READY = (MAIN / 'outputs/kaggle/ws2-r224-v2-qual-20261015/notebook.ipynb').is_file() and \
    (MAIN / 'outputs/analysis/r6flip-20261017/cpu-pins.json').is_file() and (EVIDENCE / 'r6-v1-assets/E_lrprior_s0_f3.pt').is_file()


@unittest.skipUnless(ARCHIVE_READY, 'archived A-prime packet, R6 assets or CPU pins not staged')
class PacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import build_r6flip_entry as B
        cls.B = B
        cls.rehearsal = B.packet('rehearsal')

    def test_source_preservation_and_new_modules(self):
        archived = self.B.archived()
        for name, text in archived['sources'].items():
            self.assertEqual(self.rehearsal['sources'][name], text)
        for name in self.B.NEW_SOURCES:
            self.assertEqual(self.rehearsal['sources'][name], (PROJECT / name).read_text())
        cfg = self.rehearsal['cfg']
        self.assertFalse(set(self.B.R7_ONLY_CFG) & set(cfg))
        b = cfg['r6flip_binding']
        self.assertEqual(b['r6']['bundle_id'], 'fe055406fad6a12f')
        self.assertEqual(b['raptor_weight_vector'], R6F.raptor_weight_vector())
        self.assertEqual(len(b['r6']['cpu_pins']['folds']), 4)
        self.assertEqual(self.B.packet('rehearsal'), self.rehearsal)

    def test_entry_strips_private_fixtures_and_keeps_binding(self):
        q = {'r224_state': 'x', 'anatomy_state': 'y', 'status': 'PASS'}
        entry = self.B.packet('entry', q)
        cfg = entry['cfg']
        self.assertFalse(set(self.B.PRIVATE_CFG) & set(cfg))
        self.assertFalse(set(self.B.PRIVATE_BINDING) & set(cfg['binding']))
        self.assertEqual(cfg['r6flip_binding'], self.rehearsal['cfg']['r6flip_binding'])
        self.assertEqual((cfg['r224_state_pin'], cfg['mode']), ('x', 'entry'))
        self.assertEqual(ENTRY.r6_worker_config({**cfg, 'boards': [{}, {}], 'staged': 's', 'staged_ids': [], 'stage_binding': {},
                                                 'code': 'c', 'r6_mount': 'm', 'anatomy': 'a'}, 0, 'production', [], 'o', None)['r6']['anatomy_state'], 'y')

    def test_metadata_private_offline_t4_title_is_id(self):
        meta = self.B.metadata(self.B.SLUGS['rehearsal'])
        self.assertEqual(meta['id'], 'ozgung/' + meta['title'])
        self.assertTrue(meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'])
        self.assertEqual(meta['machine_shape'], 'NvidiaTeslaT4')
        self.assertNotIn('timeout', meta)
        self.assertIn('heliosli/rsna-knee-r6-frozen-fe055406-w1/1', meta['dataset_sources'])
        self.assertNotIn('heliosli/rsna-knee-compact-r7cnxfsss/1', meta['dataset_sources'])

    def test_fresh_isolated_unpacked_import(self):
        """Materialize the payload exactly like the bootstrap and import the shipped copies with python -I."""
        with tempfile.TemporaryDirectory() as d:
            project = Path(d) / 'r6flip-project'
            for name, content in self.rehearsal['sources'].items():
                p = project / name if name.startswith('knee/') else project / 'source' / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content.encode())
            code = ('import sys; sys.path.insert(0, %r)\n'
                    'import knee.public_r6flip_entry as E, knee.public_r6flip_worker as W, knee.public_r6flip as F\n'
                    'assert all(m not in sys.modules for m in ("numpy", "torch", "pandas")), "ML import at module import"\n'
                    'assert E.__file__.startswith(%r) and W.__file__.startswith(%r)\n'
                    'from knee.public_r6flip_entry import projection, merge_r6, require_rehearsal\n'
                    'try:\n    import numpy as np, pandas\nexcept ImportError:\n    np = None\n'
                    'if np is not None:\n'
                    '    rng = np.random.default_rng(1); a = [rng.random((30, 12)) for _ in range(3)]; m = [np.zeros(30, bool)] * 3\n'
                    '    r = F.recover(*a, *m); assert r["union_recovered"] == 0 and r["blend"].shape == (30, 12)\n'
                    '    from knee.public_r7_recovery_entry import launch, merge; from knee.public_ws2_entry import launch_crop224\n'
                    'print("FRESH_IMPORT_OK")\n') % (str(project), str(project), str(project))
            env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
            r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, cwd=d, env=env)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('FRESH_IMPORT_OK', r.stdout)
            script = self.B.bootstrap(self.rehearsal)
            compile(script, '<bootstrap>', 'exec')
            self.assertIn("signal.alarm(int(8.9*3600))", script)
            self.assertIn("emergency_csv", script)


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(torch is not None and (SMALL / 'state_spec.json').is_file(), 'torch or R6 metadata not staged')
class WorkerExecuteSpawnTests(unittest.TestCase):
    """execute() end to end on CPU: compact_infer-shaped fake main(), two real spawn loader children, injection, ledger."""

    def test_execute_with_spawn_children_injection_and_ledger(self):
        import inspect
        fake = PROJECT / 'tests/fixtures/r6flip_fake'
        saved_path, saved = list(sys.path), {k: sys.modules.pop(k) for k in ('compact_train', 'compact_infer') if k in sys.modules}
        sys.path.insert(0, str(fake))
        try:
            import compact_train as CT
            spec = json.loads((SMALL / 'state_spec.json').read_bytes())
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                mount = d / 'mount'
                mount.mkdir()
                for name in ('state_spec.json', 'fluid_stacker.json', 'state_stacker.json'):
                    (mount / name).write_bytes((SMALL / name).read_bytes())
                pins = []
                for i in range(4):
                    torch.manual_seed(i)
                    m = CT.Net('convnext_small', None, freeze_blocks=16, crop=8, state_spec=spec, in_chans=3, spatial=True, extent=True,
                               syn_decomp=True)
                    path = mount / ('E_lrprior_s0_f%d.pt' % i)
                    torch.save(m.state_dict(), path)
                    b = inspect.signature(CT.Net.__init__).bind(m, 'convnext_small', None, freeze_blocks=16, crop=8, state_spec=spec,
                                                                in_chans=3, spatial=True, extent=True, syn_decomp=True)
                    b.apply_defaults()
                    args = {k: v for k, v in b.arguments.items() if k != 'self'}
                    args['state_spec'] = R6F.digest(spec)
                    pins.append({'file': path.name, 'file_sha256': WORKER.file_sha(path), 'state': R6F.module_digest(m),
                                 'buffers': R6F.buffers_digest(m), 'constructor': R6F.constructor_digest(m), 'constructor_args': args})
                torch.manual_seed(11)
                side = torch.nn.Linear(3, 1)
                manifest = {'arm': 'E_lrprior', 'backbone': 'convnext_small', 'size': 12, 'crop': 8, 'eval_stride': 1, 'views': 'v4',
                            'state_spec': 'state_spec.json', 'side_model': 'side_cls.pt', 'slices': 3, 'phys_mm': 3.5,
                            'fluid_stacker': 'fluid_stacker.json', 'state_stacker': 'state_stacker.json', 'compile': True,
                            'models': [p['file'] for p in pins]}
                cfg = {'ids': ['u0', 'u1', 'u2', 'u3'], 'inject_uid': 'u1', 'r6_mount': str(mount), 'data': str(d), 'code': str(fake),
                       'anatomy': str(d / 'anatomy.pt'), 'pass': 'sentinel1',
                       'r6': {'manifest': manifest, 'cpu_pins': {'folds': pins, 'side': {'state': R6F.module_digest(side),
                                                                                        'constructor': R6F.constructor_digest(side)}}}}
                out = d / 'board0' / 'r6'
                out.mkdir(parents=True)
                r = {'records': []}
                from knee.public_r6flip_worker import execute, load_stackers
                spec2, fluid, state, pair = load_stackers(mount, manifest, R6F.LABELS)
                import compact_infer
                execute(cfg, out, r, torch, CT, lambda: compact_infer, manifest, spec2, fluid, state, pair, check_ci=lambda ci, m: None)
            self.assertEqual((r['status'], r['genuine'], r['failed']), ('PASS', 3, 1))
            rows = {x['id']: x for x in r['records']}
            self.assertEqual(rows['u1']['status'], 'data_local_failure')
            self.assertIn('no usable window', rows['u1']['item_error'])
            for sid in ('u0', 'u2', 'u3'):
                x = rows[sid]
                self.assertTrue(x['mirror_input_equal'] and x['checksum_equal'] and x['calls'] == 8 and x['reconstruction_max_abs'] <= 1e-6)
            self.assertEqual(len(r['children']), 2)
            self.assertEqual(sum(len(c['injected']) for c in r['children']), 1)
            self.assertTrue(all(c['before'] == c['after'] and not c['changed'] for c in r['children']))
            self.assertEqual(r['loaders'][0]['num_workers'], 2)
            self.assertEqual(r['state_before'], r['state_after'])
        finally:
            sys.path[:] = saved_path
            for k in ('compact_train', 'compact_infer'):
                sys.modules.pop(k, None)
            sys.modules.update(saved)


@unittest.skipIf(np is None, 'numpy not installed')
class NeverAbortTests(unittest.TestCase):
    def test_salvage_partial_ledgers_and_emergency_constant_csv(self):
        import csv
        from knee.public_s6_recovery_runtime import save_ledger
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ids = ['a', 'b', 'c', 'd']
            (d / 'board0-old' / 'old').mkdir(parents=True)
            save_ledger(d / 'board0-old' / 'old' / 'predictions.npz', ['a', 'c'], [np.full(12, .3), np.full(12, np.nan)],
                        ['genuine', 'data_local_failure'])
            p, st = ENTRY.salvage(ids, d, 'old', ['board0-old', 'board1-old'])
            self.assertEqual(st, ['genuine', 'unavailable', 'unavailable', 'unavailable'])
            compact = d / 'board1-r6-production' / 'r6' / 'compact'
            compact.mkdir(parents=True)
            (compact / 'predictions.csv').write_text('StudyInstanceUID,' + ','.join(R6F.LABELS) + '\nd,' + ','.join(['0.5'] * 12)
                                                     + '\nb,0.1,0.2\n')
            p, st = ENTRY.salvage(ids, d, 'r6', ['board0-r6-production', 'board1-r6-production'])
            self.assertEqual(st, ['unavailable', 'unavailable', 'unavailable', 'genuine'])
            root = d / 'comp'
            root.mkdir()
            (root / 'sample_submission.csv').write_text('StudyInstanceUID,' + ','.join(R6F.LABELS) + '\n')
            (root / 'test.csv').write_text('StudyInstanceUID\nx1\nx2\n')
            (root / 'test_series.csv').write_text('StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n'
                                                  'x1,s1,1,0,Sagittal\nx2,s2,0,0,Coronal\n')
            ENTRY.emergency_csv(root, d / 'submission.csv')
            rows = list(csv.reader((d / 'submission.csv').open()))
            self.assertEqual([r[0] for r in rows], ['StudyInstanceUID', 'x1', 'x2'])
            self.assertTrue(all(float(v) == .5 for r in rows[1:] for v in r[1:]))
