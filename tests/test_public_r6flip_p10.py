"""Parent request 2026-10-20 22:55 UTC: the p9s2 grid (compact weight w, member eval stride, R224), P10 (Raptor half + own-only
compact, mean logit or mean rank, up to eight members) and P11 (own-only compact alone, no Raptor, no R6).

Synthetic data only; fake modules for the loader-child hook; tiny CPU modules for the member forwards. Heavy parts skip when
numpy/pandas/torch are absent; packet tests skip when the archive is not staged.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from knee import public_r6flip as R6F
from knee import public_r6flip_entry as ENTRY
from knee import public_s6c as S6C

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = pd = None
try:
    import torch
except ImportError:
    torch = None


def members(n=3, labels=('v3', 'v3', 'v4b'), no_reference=()):
    out = []
    for i in range(n):
        seed = 43 + i
        m = {'seed': seed, 'labels': labels[i % len(labels)], 'file': 'E_prior_s%d_x.pt' % seed, 'sha256': ('%02x' % seed) * 32,
             'bytes': 200000000 + seed, 'dataset': 'ozgung/knee-s6-x-s%d-20261021' % seed, 'dataset_version': 1}
        if seed in no_reference:
            m['no_reference'] = True
        out.append(m)
    return out


class GridSpecTests(unittest.TestCase):
    def test_p9s2_grid_names_and_blends(self):
        self.assertEqual({k: v for k, v in R6F.VARIANTS['p9s2'].items()}, R6F.VARIANTS['p9s'])
        self.assertEqual(R6F.VARIANTS['p9s2w60']['compact_arms'], [['r6', .4], ['s6c', .6]])
        self.assertEqual(R6F.VARIANTS['p9s2w70']['compact_arms'], [['r6', .3], ['s6c', .7]])
        self.assertEqual((R6F.own_stride(R6F.VARIANTS['p9s2s2']), R6F.own_stride(R6F.VARIANTS['p9s2']), R6F.own_stride(None)), (2, 1, 1))
        self.assertEqual(R6F.resolve_blend(R6F.VARIANTS['p9s2rw60s2'])[0], R6F.RAPTOR_ARMS)
        self.assertEqual(R6F.resolve_compact(R6F.VARIANTS['p9s2rw60s2']), ((('r6', .4), ('s6c', .6)), False))
        grid = sorted(k for k in R6F.VARIANTS if k.startswith('p9s2') and k != 'p9s2s2b')      # p9s2s2b: the Oct 22 four-best list
        self.assertEqual(len(grid), 12)
        self.assertTrue(all(R6F.own_shared(R6F.VARIANTS[k]) and R6F.own_compact_arm(R6F.VARIANTS[k]) for k in grid))
        with self.assertRaises(ValueError):
            R6F.resolve_blend({**copy.deepcopy(R6F.VARIANTS['p9']), 'own_stride': 2})           # stride needs the shared decode
        with self.assertRaises(ValueError):
            R6F.resolve_blend({**copy.deepcopy(R6F.VARIANTS['p9s2']), 'own_stride': 4})

    def test_p10_and_p11_blends(self):
        p10, p11 = R6F.VARIANTS['p10'], R6F.VARIANTS['p11']
        self.assertEqual(R6F.resolve_blend(p10)[0], R6F.RAPTOR_ARMS)
        self.assertEqual(R6F.resolve_blend(R6F.VARIANTS['p10p'])[0], (('old320', 1.0),))
        self.assertEqual((R6F.own_aggregation(p10), R6F.own_aggregation(R6F.VARIANTS['p10mr']), R6F.own_aggregation(R6F.VARIANTS['p7'])),
                         ('mean_logit', 'mean_rank', 'mean_rank'))
        self.assertEqual((R6F.blend_arm_names(p11), R6F.compact_arm_names(p11), R6F.own_only(p11), R6F.own_only(p10)), ([], ['r6'], True, False))
        self.assertEqual(R6F.resolve_compact(p11), (R6F.DEFAULT_COMPACT, True))                 # the s6c block takes the compact slot
        self.assertTrue(all(R6F.COMPACT_VARIANTS[k] == 's6c' for k in ('p10', 'p10mr', 'p10p', 'p10pmr', 'p11', 'p11mr')))
        self.assertEqual(R6F.raptor_weight_vector(blend=p11), [0.0] * 12)
        self.assertEqual(R6F.raptor_weight_vector(blend=p10), R6F.raptor_weight_vector())
        with self.assertRaises(ValueError):
            R6F.resolve_blend({**copy.deepcopy(p11), 'raptor_arms': [['old320', 1.0]]})          # own_only means no Raptor arm
        with self.assertRaises(ValueError):
            R6F.resolve_blend({**copy.deepcopy(p10), 'own_aggregation': 'median'})

    def test_slugs_dates_and_rehearsal_reuse(self):
        from scripts import build_r6flip_entry as B
        self.assertEqual(B.slug_for('p9s2w60s2', 'rehearsal'), 'r6flip-p9s2w60s2-rehearsal-20261021')
        self.assertEqual(B.slug_for('p11mr', 'entry'), 'r6flip-p11mr-entry-20261021')
        self.assertEqual(B.slug_for('p9s', 'entry'), 'r6flip-p9s-entry-20261020')                 # the submitted name is unchanged
        self.assertTrue(B.same_stages('p9s2', 'p9s2w60') and B.same_stages('p9s2w70', 'p9s2'))    # only the compact weight differs
        self.assertFalse(B.same_stages('p9s2', 'p9s2s2') or B.same_stages('p9s2', 'p9s2r') or B.same_stages('p9s2', 'p9s'))
        self.assertFalse(B.same_stages('p10', 'p10mr') or B.same_stages('p10', 'p10p') or B.same_stages('p11', 'p11mr') or B.same_stages('p10', 'p11'))
        self.assertTrue(all(len(B.slug_for(k, 'rehearsal')) <= 50 for k in R6F.VARIANTS))
        for k in R6F.VARIANTS:
            self.assertIn(B.S6C_MODELS_BY_VARIANT.get(k, 'S6C_MODELS'), ('S6C_MODELS', 'S6C_MODELS_P9', 'S6C_MODELS_P9S2', 'S6C_MODELS_P10',
                                                                         'S6C_MODELS_P12N', 'S6C_MODELS_V5', 'S6C_MODELS_P9B', 'S6C_MODELS_E1',
                                                                         'S6C_MODELS_C1K3', 'S6C_MODELS_C1K2', 'S6C_MODELS_S4'))
        self.assertTrue(all(B.S6C_MODELS_BY_VARIANT[k] == 'S6C_MODELS_P10' for k in R6F.VARIANTS if k.startswith(('p10', 'p11')) and k not in ('p11v5', 'p10v5', 'p10pv5', 'p10s4', 'p10s4s')))
        self.assertTrue(all(B.S6C_MODELS_BY_VARIANT[k] == 'S6C_MODELS_P9S2' for k in R6F.VARIANTS if k.startswith('p9s2') and k != 'p9s2s2b'))


class MemberRuleTests(unittest.TestCase):
    def test_up_to_eight_members_labels_and_references(self):
        self.assertEqual(len(S6C.validate_models(members(8, ('v3',)))), 8)
        with self.assertRaises(ValueError):
            S6C.validate_models(members(9, ('v3',)))
        with self.assertRaises(ValueError) as cm:
            S6C.validate_models(members(3))                                     # v3 + v4b mixed without the sealed allowance
        self.assertEqual(str(cm.exception), 'S6C_ONE_LABEL_TABLE_NO_MIXING')
        self.assertEqual(len(S6C.validate_models(members(3), allow_mixed_labels=True)), 3)
        for labels in ('v4a', 'v4b', 'round1', 'v3'):
            S6C.validate_models(members(2, (labels,)))
        with self.assertRaises(ValueError):
            S6C.validate_models(members(2, ('labels-x',)))
        ms = members(2, ('v3',), no_reference=(44,))
        self.assertEqual(S6C.validate_models(ms), ['E_prior_s43_f0.pt', 'E_prior_s44_f0.pt'])
        bad = copy.deepcopy(ms)
        bad[0]['no_reference'] = 'yes'
        with self.assertRaises(ValueError):
            S6C.validate_models(bad)
        ref = {'source': 'A100', 'probabilities': {'E_prior_s43_f0.pt': {'u': [0.5] * 12}}}
        block = S6C.block(ms, ref, aggregation='mean_logit')                   # s44 has no reference, and says so
        self.assertEqual((block['aggregation_key'], block['labels'], block['aggregation']), ('mean_logit', 'v3', S6C.AGGREGATIONS['mean_logit']))
        self.assertEqual(S6C.block(members(2, ('v3',)), ref)['mixed_labels_allowed'], False)   # block() itself: P7's partial references stay legal
        with self.assertRaises(ValueError):
            S6C.block(ms, ref, aggregation='median')
        self.assertEqual(S6C.block(members(3), None, allow_mixed_labels=True)['labels'], 'v3+v4b')
        self.assertEqual(S6C.aggregation_key(S6C.block(members(2, ('v3',)))), 'mean_rank')
        self.assertEqual(S6C.aggregation_key(None), 'mean_rank')

    @unittest.skipIf(np is None or pd is None, 'numpy/pandas not installed')
    def test_mean_logit_merge_and_salvage(self):
        from tests.test_public_s6c import ArrayTests
        from knee.public_s6_recovery_entry import assignments
        ids = ['s%02d' % i for i in range(10)]
        shards = assignments(ids)
        helper = ArrayTests()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            reports = [helper.write_board(d, b, shards[b], 7 + b)[0] for b in range(2)]
            values, mask, seed_p = S6C.merge(ids, reports, d, 3, aggregation='mean_logit')
            ledger_p = {}
            for b in range(2):
                with np.load(S6C.ledger_path(d, b), allow_pickle=False) as a:
                    ledger_p.update(dict(zip(a['study_ids'].tolist(), a['probabilities'])))
            self.assertTrue(np.allclose(values, np.array([ledger_p[s] for s in ids]), atol=1e-12))      # the worker's own CSV arithmetic
            z = np.log(np.clip(seed_p, 1e-12, 1 - 1e-12) / (1 - np.clip(seed_p, 1e-12, 1 - 1e-12)))
            self.assertTrue(np.allclose(values, S6C.sigmoid(z.mean(1)), atol=1e-5))
            rank_values = S6C.merge(ids, reports, d, 3)[0]
            self.assertFalse(np.allclose(values, rank_values))
            sal, st = S6C.salvage(ids, d, 3, aggregation='mean_logit')
            self.assertTrue(np.allclose(sal, values, atol=1e-5) and st.count('genuine') == 10)
            self.assertTrue(np.allclose(S6C.aggregate(seed_p, np.ones(10, bool), 'mean_rank'), rank_values))
            with self.assertRaises(ValueError):
                S6C.aggregate(seed_p, np.ones(10, bool), 'median')


@unittest.skipIf(np is None or pd is None, 'numpy/pandas not installed')
class ArithmeticTests(unittest.TestCase):
    @staticmethod
    def pct(v):
        return pd.DataFrame(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).to_numpy()

    def test_compact_weight_reaches_the_blend(self):
        from knee.public_ws2_recovery import quantize9
        rng = np.random.default_rng(3)
        n = 40
        a = {'old320': rng.random((n, 12)), 'r6': rng.random((n, 12)), 's6c': S6C.mean_rank(rng.random((n, 3, 12)), np.ones(n, bool))}
        st = {k: ['genuine'] * n for k in a}
        w = np.asarray(R6F.raptor_weight_vector())
        for name, wo in (('p9s2', .5), ('p9s2w60', .6), ('p9s2w70', .7), ('p9s2w60s2', .6)):
            rec = R6F.recover_guarded(a, st, blend=R6F.VARIANTS[name])
            compact = (1 - wo) * self.pct(quantize9(a['r6'])) + wo * self.pct(quantize9(a['s6c']))
            self.assertTrue(np.allclose(rec['blend'], w * self.pct(a['old320']) + (1 - w) * compact, rtol=0, atol=1e-12), name)
            self.assertEqual(rec['compact_weights'], {'r6': round(1 - wo, 2), 's6c': wo})
        st['s6c'] = ['genuine'] * 30 + ['unavailable'] * 10          # guard cut: rank(R6 plain) alone, whatever w
        rec = R6F.recover_guarded(a, st, blend=R6F.VARIANTS['p9s2w70'])
        r6 = self.pct(quantize9(a['r6']))
        self.assertTrue(np.allclose(rec['blend'][30:], (w * self.pct(a['old320']) + (1 - w) * r6)[30:], rtol=0, atol=1e-12))

    def test_own_only_is_the_ranked_own_compact(self):
        from knee.public_ws2_recovery import quantize9
        rng = np.random.default_rng(4)
        n = 40
        own = S6C.mean_logit(rng.random((n, 4, 12)), np.ones(n, bool))
        rec = R6F.recover_guarded({'r6': own}, {'r6': ['genuine'] * n}, blend=R6F.VARIANTS['p11'])
        self.assertEqual((rec['mode'], rec['fallback_studies'], sorted(rec['ranks'])), ('variant_complete_population', 0, ['r6']))
        self.assertTrue(np.allclose(rec['blend'], self.pct(quantize9(own)), rtol=0, atol=1e-12))
        self.assertTrue(np.all(rec['raptor_slot'] == 0))
        arrays = {'r6': own}
        arrays['r6'] = own.copy()
        arrays['r6'][30:] = np.nan
        fb = R6F.recover_guarded(arrays, {'r6': ['genuine'] * 30 + ['unavailable'] * 10}, blend=R6F.VARIANTS['p11'])
        self.assertEqual((fb['mode'], fb['fallback_studies'], fb['fallback']['constant']), ('deadline_fallback_public_semantics', 10, 10))
        self.assertTrue(np.allclose(fb['blend'][:30], self.pct(quantize9(own[:30])), rtol=0, atol=1e-12))
        self.assertTrue(np.all(fb['blend'][30:] == .5))
        # P10: P7's arithmetic with the Raptor half
        a = {'old320': rng.random((n, 12)), 'crop224': rng.random((n, 12)), 'r6': own}
        st = {k: ['genuine'] * n for k in a}
        p10 = R6F.recover_guarded(a, st, blend=R6F.VARIANTS['p10'])
        p7 = R6F.recover_guarded(a, st, blend=R6F.VARIANTS['p7'])
        self.assertTrue(np.array_equal(p10['blend'], p7['blend']))
        self.assertEqual(p10['mode'], 'aprime_complete_population')


def stub_cfg(variant, block=None, mode='entry'):
    cols = ['StudyInstanceUID'] + R6F.LABELS
    cfg = {'mode': mode, 'r6flip_binding': {'official_columns': cols, 'entry_variant': {'name': variant}, 'r6': {}}, 'environment': {},
           'blend': copy.deepcopy(R6F.VARIANTS[variant])}
    if block is not None:
        cfg['s6c'] = block
    return cfg, cols


def run_entry(cfg, cols, shared_reports=None):
    ids = ['s1', 's2', 's3']
    calls, written = [], {}

    def launch(c, o, arm, deadline, pass_name='production', **kw):
        calls.append((arm, pass_name))
        return [{'status': 'NO_RECEIPT'}] * 2, {'wall_seconds': 1., 'deadline_hit': False, 'peaks': {'host_rss_bytes': 1, 'output_disk_bytes': 1},
                                                'minimum_disk_free_bytes': 1, 'exit_codes': [1, 1]}
    with tempfile.TemporaryDirectory() as d:
        with mock.patch.object(ENTRY, 'launch_guarded', launch), \
                mock.patch.object(ENTRY, 'shared_reports', shared_reports or ENTRY.shared_reports), \
                mock.patch.object(ENTRY, 'inventory', lambda: [{'index': 0, 'uuid': 'a', 'name': 'Tesla T4'}, {'index': 1, 'uuid': 'b', 'name': 'Tesla T4'}]), \
                mock.patch('knee.public_s6_recovery_entry.stage', lambda *a, **k: {}), \
                mock.patch('knee.public_fullfield_entry.test_contract', lambda root, labels, visible=False: (cols, list(ids), {})), \
                mock.patch('knee.public_fullfield_entry.write_csv', lambda path, columns, sids, rows: written.update(rows=np.asarray(rows)) or 'sha'):
            report = ENTRY.run(Path(d), cfg, Path(d) / 'o', started=time.monotonic())
    return report, calls, written


@unittest.skipIf(np is None or pd is None, 'numpy/pandas not installed')
class StagePlanTests(unittest.TestCase):
    def test_p10_and_p11_stage_plans(self):
        block = S6C.block(members(3), None, aggregation='mean_logit', allow_mixed_labels=True)
        for variant, plan, arms in (('p11', ['s6c'], ['r6']), ('p11mr', ['s6c'], ['r6']), ('p10', ['old', 'crop224', 's6c'], ['crop224', 'old320', 'r6']),
                                    ('p10p', ['old', 's6c'], ['old320', 'r6'])):
            cfg, cols = stub_cfg(variant, block)
            report, calls, written = run_entry(cfg, cols)
            self.assertEqual([a for a, _ in calls], plan, variant)
            self.assertEqual(sorted(report['arm_merge']), arms, variant)
            self.assertEqual(report['status'], 'PASS_WITH_FALLBACK', variant)
            self.assertTrue(np.all(written['rows'] == .5), variant)
            self.assertNotIn('old', report['phases']) if variant.startswith('p11') else self.assertIn('old', report['phases'])
        cfg, cols = stub_cfg('p11')                                           # the member block is required
        report, calls, _ = run_entry(cfg, cols)
        self.assertEqual((calls, report['status']), ([], 'EMERGENCY_CONSTANT'))

    def test_projection_counts_the_stride1_check_once_as_overhead(self):
        def shard(seconds, windows=10, extra=None):
            rec = {'production_seconds': seconds, 'status': 'genuine', 'windows': windows}
            if extra is not None:
                rec = {**rec, **extra}
            return {'records': [dict(rec) for _ in range(5)], 'peak_cuda_bytes': 7}
        phase = lambda: {'wall_seconds': 100., 'peaks': {'host_rss_bytes': 1, 'output_disk_bytes': 1}, 'minimum_disk_free_bytes': 1}
        phases = {a: phase() for a in ('old', 'r6', 'r6_sentinel1', 'r6_sentinel2')}
        windows = [10] * 870
        base = {'old': [shard(2.), shard(3.)], 'r6': [shard(4.), shard(5.)], 's6c': [shard(1., 20), shard(1., 20)]}
        p0 = ENTRY.projection(10., phases, base, 5., windows)
        checked = {**base, 's6c': [shard(1., 20, {'stride1_seconds': [.3, .2]}), shard(1., 20, {'stride1_seconds': [.3, .2]})]}
        p1 = ENTRY.projection(10., phases, checked, 5., windows)
        self.assertEqual(p1['s6c_stride1_check_seconds'], [2.5, 2.5])
        self.assertNotIn('s6c_stride1_check_seconds', p0)
        # R6 production 25 s -> 22.5 s (scaled x 1300 x 10 / 100 = 130); the 2.5 s check stays once in the fixed overhead (100 - 22.5)
        self.assertAlmostEqual(p0['point_seconds'] - p1['point_seconds'], 2.5 * 130 - 2.5, places=6)
        self.assertEqual(p1['measured_production_wall_seconds']['r6'], 22.5)

    def test_worker_config_seals_the_member_stride_and_the_stride1_check(self):
        cfg, cols = stub_cfg('p9s2s2', S6C.block(members(2, ('v3',))), mode='rehearsal')
        cfg.update(boards=[{'uuid': 'a'}, {'uuid': 'b'}], staged='st', staged_ids=['s1'], stage_binding={}, code='c', source_pins={},
                   r6_mount='m', anatomy='an', deadline_epoch=1.)
        cfg['r6flip_binding']['r6'] = {'manifest': {}, 'cpu_pins': {}}
        c = ENTRY.r6_worker_config(cfg, 0, 'production', ['s1'], Path('/o/board0-r6-production'), None)
        self.assertEqual((c['s6c_shared']['geometry']['eval_stride'], c['s6c_shared']['stride1_check']), (2, True))
        self.assertEqual({k: v for k, v in c['s6c_shared']['geometry'].items() if k != 'eval_stride'},
                         {k: v for k, v in S6C.GEOMETRY_CHILD.items() if k != 'eval_stride'})
        cfg['mode'] = 'entry'
        c = ENTRY.r6_worker_config(cfg, 0, 'production', ['s1'], Path('/o/board0-r6-production'), None)
        self.assertEqual((c['s6c_shared']['geometry']['eval_stride'], c['s6c_shared']['stride1_check']), (2, False))
        cfg['blend'] = copy.deepcopy(R6F.VARIANTS['p9s2w60'])
        cfg['mode'] = 'rehearsal'
        c = ENTRY.r6_worker_config(cfg, 1, 'sentinel1', ['s1'], Path('/o/board1-r6-sentinel1'), None)
        self.assertEqual((c['s6c_shared']['geometry']['eval_stride'], c['s6c_shared']['stride1_check']), (1, False))


@unittest.skipIf(torch is None or np is None, 'torch/numpy not installed')
class Stride2ChildTests(unittest.TestCase):
    def test_stride_two_windows_plus_the_stride_one_check_set(self):
        from knee import public_s6c_shared as SH
        from tests.test_public_r6flip_p9s import fake_modules
        mods = fake_modules(n_slices=8)
        with mock.patch.dict(sys.modules, mods):
            ds = types.SimpleNamespace(builder=mods['px_cache'].StudySlices('/c', [], 384, 'v4', anchors='/c/anchors'), native=object())
            SH.install_child(ds, {**SH.CHILD_GEOMETRY, 'eval_stride': 2, 'stride1_check': True})
            ds.builder.build('u1')
            item = SH.attach(ds, {'uid': 'u1'})
            self.assertEqual((tuple(item['s6c_images'].shape), tuple(item['s6c_images_ref'].shape)), ((3, 3, 320, 320), (6, 3, 320, 320)))
            self.assertEqual(item['s6c_note']['eval_stride'], 2)
            ds2 = types.SimpleNamespace(builder=mods['px_cache'].StudySlices('/c', [], 384, 'v4'), native=object())
            SH.install_child(ds2, {**SH.CHILD_GEOMETRY, 'eval_stride': 2, 'stride1_check': False})     # entry: no check set
            ds2.builder.build('u2')
            item = SH.attach(ds2, {'uid': 'u2'})
            self.assertEqual(tuple(item['s6c_images'].shape), (3, 3, 320, 320))
            self.assertIsNone(item['s6c_images_ref'])


@unittest.skipIf(torch is None or np is None or pd is None, 'torch/numpy/pandas not installed')
class SharedStride2Tests(unittest.TestCase):
    ENV = {'torch': '2.10.0+cu128', 'packages': {'timm': '1.0.26'}, 'cuda_device_count': 1}

    def run_pass(self, out, board, pass_name, ids, skip=()):
        from tests.test_public_r6flip_p9s import make_shared, study_input
        sh = make_shared(out, board, pass_name, ids, self.ENV, measure=False)
        sh.geometry = {**sh.geometry, 'eval_stride': 2}
        sh.stride1_check = True
        obs = types.SimpleNamespace(fresh_depth=0, studies=[])
        for sid in ids:
            s6 = study_input(sid, windows=3)
            full = study_input(sid, windows=6)
            s6['images_ref'], s6['meta_ref'] = full['images'], full['meta']
            cur = {'id': sid, 's6c_input': s6, 'handle_seconds': 10., 'wait_seconds': 0., 'first_of_loader': not obs.studies}
            if sid not in skip:
                sh.on_study(obs, cur, None, None, None, 192)
                self.assertFalse(any(k in cur['s6c_input'] for k in ('images', 'meta', 'images_ref', 'meta_ref')))
            obs.studies.append(cur)
        rec = json.loads(Path(sh.finalize(obs, {'environment': dict(self.ENV, physical_uuid='u%d' % board)})['receipt']).read_text())
        rec['peak_cuda_bytes'] = 1
        return rec

    def test_stride_two_rows_carry_the_stride_one_check_and_pass_the_gates(self):
        from knee.public_s6_recovery_entry import assignments
        ids = ['p%02d' % i for i in range(39)]
        sentinels = ids[:4]
        shards = assignments(ids)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            reports = {'production': [self.run_pass(out, b, 'production', shards[b]) for b in range(2)],
                       'sentinel1': [self.run_pass(out, b, 'sentinel1', sentinels, skip=(sentinels[0],)) for b in range(2)],
                       'sentinel2': [self.run_pass(out, b, 'sentinel2', sentinels) for b in range(2)]}
            x = reports['production'][0]
            row = x['records'][0]
            self.assertEqual((x['geometry']['eval_stride'], x['stride1_check'], row['windows'], row['windows_stride1']), (2, True, 3, 6))
            self.assertEqual(len(row['seed_probabilities_stride1']), 2)
            self.assertNotEqual(row['seed_probabilities_stride1'], row['seed_probabilities'])
            block = S6C.block([{'seed': 42, 'labels': 'v3', 'file': 'a.pt', 'sha256': x['models'][0]['sha256'], 'bytes': 7, 'dataset': 'ozgung/d1', 'dataset_version': 1},
                               {'seed': 43, 'labels': 'v3', 'file': 'b.pt', 'sha256': x['models'][1]['sha256'], 'bytes': 8, 'dataset': 'ozgung/d2', 'dataset_version': 1}])
            s1 = {r_['id']: r_['seed_probabilities_stride1'] for w in reports['production'] for r_ in w['records']}
            r = {'boards': [{'uuid': 'u0'}, {'uuid': 'u1'}],
                 'workers': {'s6c': reports['production'], 's6c_sentinels': {k: reports[k] for k in ('sentinel1', 'sentinel2')},
                             'r6': [{'s6c_shared': {'status': 'PASS'}}] * 2}}
            cfg = {'s6c': dict(block, reference={'source': 'A100', 'probabilities': {block['aliases'][1]: {ids[5]: [min(1., v + 2e-3) for v in s1[ids[5]][1]]}}}),
                   'binding': {'environment': {'torch': self.ENV['torch'], 'packages': self.ENV['packages']}}, 's6_sentinels': sentinels,
                   'staged_ids': ids,
                   'p9_reference': {'aliases': [block['aliases'][1]], 'receipt_sha256': 'ref',          # a reference for one member only
                                    'probabilities': {s: [[min(1., v + 4e-4) for v in s1[s][1]]] for s in ids}}}
            q = S6C.require_compact(r, cfg, key='s6c', injected=('sentinel1', sentinels[0]), reference_key='seed_probabilities_stride1')
            self.assertEqual((q['reference_studies'], q['reference_key']), (1, 'seed_probabilities_stride1'))
            self.assertAlmostEqual(q['reference_max_abs_dprob'], 2e-3, places=9)
            q2 = S6C.require_compact(r, cfg, key='s6c', injected=('sentinel1', sentinels[0]))     # the default key reads the stride-2 rows
            self.assertNotEqual(q2['reference_max_abs_dprob'], q['reference_max_abs_dprob'])
            g = S6C.require_shared(r, cfg, stride=2)
            self.assertEqual((g['parity_key'], g['studies'], sorted(g['separate_pass_max_abs_by_member'])), ('seed_probabilities_stride1', 39, [block['aliases'][1]]))
            self.assertLessEqual(g['separate_pass_max_abs'], 4.0001e-4)
            self.assertEqual((g['member_cost']['stride'], g['member_cost']['windows_per_study'], g['member_cost']['stride1_windows_per_study']), (2, 3., 6.))
            self.assertEqual(len(g['member_cost']['per_member_study_seconds']), 2)
            self.assertEqual(len(g['member_cost']['stride1_check_per_member_study_seconds']), 2)
            c3 = copy.deepcopy(cfg)                                                                  # no member with a separate-pass read-out
            c3['p9_reference']['aliases'] = ['E_prior_s99_f0.pt']
            g3 = S6C.require_shared(r, c3, stride=2)
            self.assertEqual((g3['parity_applicable'], g3['separate_pass_max_abs']), (False, None))
            self.assertTrue(g['parity_applicable'])
            with self.assertRaises(ValueError) as cm:
                S6C.require_shared(r, cfg, stride=1)                                                # the sealed stride must match the receipts
            self.assertEqual(str(cm.exception), 'S6C_SHARED_WORKER_PASS')


class ResourceSampleTests(unittest.TestCase):
    def test_a_vanishing_temp_file_skips_the_sample_instead_of_aborting_the_stage(self):
        calls = []

        def snapshot(out):
            calls.append(out)
            if len(calls) == 1:
                raise FileNotFoundError('/o/board0-s6c-production/s6c/tmp/s_x/series/y.npy')
            return {'host_rss_bytes': 5, 'output_disk_bytes': 7, 'disk_free_bytes': 100}
        high = {'host_rss_bytes': 1, 'output_disk_bytes': 9}
        minimum = ENTRY.sample_resources('/o', high, 500, snapshot)
        self.assertEqual((minimum, high), (500, {'host_rss_bytes': 1, 'output_disk_bytes': 9}))      # skipped, nothing changed
        minimum = ENTRY.sample_resources('/o', high, minimum, snapshot)
        self.assertEqual((minimum, high), (100, {'host_rss_bytes': 5, 'output_disk_bytes': 9}))
        with self.assertRaises(RuntimeError):                                                           # only OS errors are skipped
            ENTRY.sample_resources('/o', high, minimum, lambda out: (_ for _ in ()).throw(RuntimeError('x')))


def staged():
    try:
        from scripts import build_r6flip_entry as B
        return B.ARCH.is_dir() and B.CPU_PINS.is_file()
    except Exception:
        return False


@unittest.skipUnless(staged(), 'A-prime archive not staged')
class PacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import build_r6flip_entry as B
        cls.B = B
        cls.ms = members(3, ('v3', 'v3', 'v4b'), no_reference=(45,))
        cls.p10 = mock.patch.object(B, 'S6C_MODELS_P10', cls.ms)
        cls.p9s2 = mock.patch.object(B, 'S6C_MODELS_P9S2', copy.deepcopy(B.S6C_MODELS_P9))      # the gen-2 grid with the two reference members
        cls.ref = mock.patch.object(B, 'S6C_REFERENCE', None)
        for p in (cls.p10, cls.p9s2):
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in (cls.p10, cls.p9s2):
            p.stop()

    def test_unsealed_lists_refuse_and_sealed_lists_build(self):
        with mock.patch.object(self.B, 'S6C_MODELS_P10', None):
            with self.assertRaises(ValueError) as cm:
                self.B.packet('rehearsal', variant='p11')
            self.assertEqual(str(cm.exception), 'S6C_MODELS_NOT_SEALED')
        with mock.patch.object(self.B, 'S6C_MODELS_P9S2', None):
            with self.assertRaises(ValueError):
                self.B.packet('rehearsal', variant='p9s2w60')

    def test_p11_packet(self):
        with self.ref:
            pack = self.B.packet('rehearsal', variant='p11')
        cfg, b = pack['cfg'], pack['cfg']['r6flip_binding']
        self.assertEqual(cfg['blend'], R6F.VARIANTS['p11'])
        self.assertEqual((cfg['s6c']['aggregation_key'], cfg['s6c']['labels'], len(cfg['s6c']['models'])), ('mean_logit', 'v3+v4b', 3))
        self.assertNotIn('r6', b)
        self.assertEqual(b['entry_variant']['stages'], ['r6'])
        self.assertIn('alone; no Raptor half', b['variant'])
        self.assertEqual(b['frame']['raptor_half']['arms'], {})
        self.assertEqual(b['s6c']['aggregation_key'], 'mean_logit')
        meta = self.B.metadata(self.B.slug_for('p11', 'rehearsal'), 'p11')
        self.assertFalse(any(d.startswith('heliosli/rsna-knee-r6') for d in meta['dataset_sources']))
        self.assertIn('ozgung/knee-s6-x-s45-20261021/1', meta['dataset_sources'])
        script = self.B.bootstrap(pack)
        self.assertIn('r6_mount=None', script)
        self.assertNotIn('EXACT_R6_V1_MANIFEST', script)
        with self.ref:
            entry = self.B.packet('entry', {'status': 'PASS', 'r224_state': None, 'anatomy_state': 'y'}, 'p11')
        self.assertNotIn('r224_state_pin', entry['cfg'])
        self.assertEqual(entry['cfg']['s6c']['models'], cfg['s6c']['models'])

    def test_p10_variants_share_the_member_list_and_differ_by_aggregation(self):
        with self.ref:
            p10 = self.B.packet('rehearsal', variant='p10')
            p10mr = self.B.packet('rehearsal', variant='p10mr')
        self.assertEqual(p10['cfg']['s6c']['models'], p10mr['cfg']['s6c']['models'])
        self.assertEqual((p10['cfg']['s6c']['aggregation_key'], p10mr['cfg']['s6c']['aggregation_key']), ('mean_logit', 'mean_rank'))
        self.assertEqual(p10['cfg']['r6flip_binding']['entry_variant']['stages'], ['old320', 'crop224', 'r6'])
        self.assertIn('mean member logit', ' '.join(p10['cfg']['r6flip_binding']['deviations']))

    def test_p9s2_grid_packets_seal_weight_and_stride(self):
        if self.B.P9_REHEARSAL_RECEIPT_SHA is None or not Path(self.B.P9_REHEARSAL_OUT / 'receipt.json').is_file():
            self.skipTest('P9 rehearsal reference not staged')
        pack = self.B.packet('rehearsal', variant='p9s2w60s2')
        cfg = pack['cfg']
        self.assertEqual(cfg['blend'], R6F.VARIANTS['p9s2w60s2'])
        self.assertEqual(cfg['p9_reference']['aliases'], ['E_prior_s42_f1.pt', 'E_prior_s43_f0.pt'])
        self.assertEqual(cfg['r6flip_binding']['frame']['compact_half']['s6c']['eval_stride'], 2)
        self.assertEqual(cfg['r6flip_binding']['entry_variant']['compact_arms'], [['r6', .4], ['s6c', .6]])
        self.assertIn('eval stride 2', ' '.join(cfg['r6flip_binding']['deviations']))
        entry = self.B.packet('entry', {'status': 'PASS', 'r224_state': None, 'anatomy_state': 'y'}, 'p9s2w60s2')
        self.assertNotIn('p9_reference', entry['cfg'])


if __name__ == '__main__':
    unittest.main()


# ---- P12 (parent 2026-10-21 12:55): no Raptor; compact = 1/2 rank(R6 flip|plain) + 1/2 rank(own mean logit), shared decode -----
class P12Tests(unittest.TestCase):
    def test_spec(self):
        for name, flip in (('p12', False), ('p12f', True), ('p12fn', True)):
            b = R6F.VARIANTS[name]
            self.assertEqual((R6F.blend_arm_names(b), R6F.resolve_compact(b)), ([], ((('r6', .5), ('s6c', .5)), flip)))
            self.assertTrue(R6F.own_compact_arm(b) and R6F.own_shared(b))
            self.assertEqual((R6F.own_stride(b), R6F.own_aggregation(b), R6F.raptor_weight_vector(blend=b)), (2, 'mean_logit', [0.0] * 12))
            self.assertNotIn(name, R6F.COMPACT_VARIANTS)                         # the R6 bundle runs
        with self.assertRaises(ValueError):                                      # flip with an own arm needs the shared decode
            R6F.own_compact_arm({**copy.deepcopy(R6F.VARIANTS['p9']), 'compact_flip': True})
        with self.assertRaises(ValueError):
            R6F.resolve_blend({**copy.deepcopy(R6F.VARIANTS['p12f']), 'raptor_arms': [['old320', 1.0]]})
        from scripts import build_r6flip_entry as B
        self.assertEqual((B.S6C_MODELS_BY_VARIANT['p12f'], B.S6C_MODELS_BY_VARIANT['p12fn'], B.slug_for('p12f', 'entry')),
                         ('S6C_MODELS_P10', 'S6C_MODELS_P12N', 'r6flip-p12f-entry-20261021'))
        self.assertFalse(B.same_stages('p12', 'p12f') or B.same_stages('p12f', 'p12fn'))

    @unittest.skipIf(np is None or pd is None, 'numpy/pandas not installed')
    def test_arithmetic_and_guard_fallback(self):
        from knee.public_ws2_recovery import quantize9
        pct = lambda v: pd.DataFrame(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).to_numpy()
        rng = np.random.default_rng(12)
        n = 40
        a = {'r6': rng.random((n, 12)), 's6c': S6C.mean_logit(rng.random((n, 8, 12)), np.ones(n, bool))}
        st = {k: ['genuine'] * n for k in a}
        rec = R6F.recover_guarded(a, st, blend=R6F.VARIANTS['p12f'])
        self.assertEqual((rec['mode'], rec['fallback_studies']), ('variant_complete_population', 0))
        self.assertTrue(np.allclose(rec['blend'], .5 * pct(quantize9(a['r6'])) + .5 * pct(quantize9(a['s6c'])), rtol=0, atol=1e-12))
        st['s6c'] = ['genuine'] * 30 + ['unavailable'] * 10                     # members cut at the guard: rank(R6) alone
        a['s6c'][30:] = np.nan
        fb = R6F.recover_guarded(a, st, blend=R6F.VARIANTS['p12f'])
        self.assertEqual(fb['compact_fallback'], {'partial_compact': 10, 'r6_only': 10, 's6c_only': 0})
        self.assertTrue(np.allclose(fb['blend'][30:], pct(quantize9(a['r6']))[30:], rtol=0, atol=1e-12))

    @unittest.skipIf(np is None or pd is None, 'numpy/pandas not installed')
    def test_stage_plan_and_worker_config(self):
        block = S6C.block(members(3), None, aggregation='mean_logit', allow_mixed_labels=True)
        seen = []
        cfg, cols = stub_cfg('p12f', block)
        report, calls, written = run_entry(cfg, cols, lambda out, arm, pass_name='production': seen.append((arm, pass_name)) or [{'status': 'NO_RECEIPT'}] * 2)
        self.assertEqual(calls, [('r6', 'production')])
        self.assertEqual(seen, [('s6c', 'production')])
        self.assertEqual(sorted(report['arm_merge']), ['r6', 's6c'])
        self.assertTrue(np.all(written['rows'] == .5))
        cfg.update(mode='rehearsal', boards=[{'uuid': 'a'}, {'uuid': 'b'}], staged='st', staged_ids=['s1'], stage_binding={}, code='c',
                   source_pins={}, r6_mount='m', anatomy='an', deadline_epoch=1.)
        cfg['r6flip_binding']['r6'] = {'manifest': {}, 'cpu_pins': {}}
        c = ENTRY.r6_worker_config(cfg, 0, 'production', ['s1'], Path('/o/board0-r6-production'), None)
        self.assertNotIn('flip', c)                                             # R6 flip TTA (P2's worker path)
        self.assertEqual((c['s6c_shared']['geometry']['eval_stride'], c['s6c_shared']['stride1_check']), (2, True))


# ---- P13 (parent pre-registration 2026-10-21 22:00): P2 exactly + own members in the compact via the shared decode -------------
class P13Tests(unittest.TestCase):
    def test_spec_lists_and_rehearsal_reuse(self):
        from scripts import build_r6flip_entry as B
        for name, w, stride, k in (('p13a', .5, 2, 3), ('p13b', .5, 3, 3), ('p13c', .5, 3, 2), ('p13aw65', .65, 2, 3), ('p13bw65', .65, 3, 3)):
            b = R6F.VARIANTS[name]
            self.assertEqual(R6F.resolve_blend(b)[0], R6F.RAPTOR_ARMS)                     # P2's Raptor half
            self.assertEqual(R6F.raptor_weight_vector(blend=b), R6F.raptor_weight_vector())  # v1 per-finding weights
            self.assertEqual(R6F.resolve_compact(b), ((('r6', round(1 - w, 2)), ('s6c', w)), True))   # R6 FLIP
            self.assertEqual((R6F.own_stride(b), R6F.own_aggregation(b), R6F.own_shared(b)), (stride, 'mean_logit', True))
            self.assertEqual(len(B.s6c_models(name)), k)
        self.assertEqual([m['file'] for m in B.s6c_models('p13c')], ['E_prior_s43_v4ball.pt', 'E_prior_s44_v5all.pt'])
        self.assertTrue(B.same_stages('p13a', 'p13aw65') and B.same_stages('p13b', 'p13bw65'))   # slot 4 reuses C1's rehearsal
        self.assertFalse(B.same_stages('p13a', 'p13b') or B.same_stages('p13b', 'p13c'))
        with self.assertRaises(ValueError):
            R6F.resolve_blend({**copy.deepcopy(R6F.VARIANTS['p13a']), 'own_stride': 4})

    @unittest.skipIf(torch is None or np is None, 'torch/numpy not installed')
    def test_stride_three_child_render(self):
        from knee import public_s6c_shared as SH
        from tests.test_public_r6flip_p9s import fake_modules
        mods = fake_modules(n_slices=10)
        with mock.patch.dict(sys.modules, mods):
            ds = types.SimpleNamespace(builder=mods['px_cache'].StudySlices('/c', [], 384, 'v4'), native=object())
            SH.install_child(ds, {**SH.CHILD_GEOMETRY, 'eval_stride': 3, 'stride1_check': True})
            ds.builder.build('u1')
            item = SH.attach(ds, {'uid': 'u1'})
            self.assertEqual((item['s6c_images'].shape[0], item['s6c_images_ref'].shape[0], item['s6c_note']['eval_stride']), (3, 8, 3))


class S4StrideTests(unittest.TestCase):
    def test_block_variants_and_command(self):
        from knee import public_s6c_worker as W
        ms = members(2, ('v5',))
        self.assertNotIn('eval_stride', S6C.block(ms))                                   # every earlier block unchanged
        self.assertEqual(S6C.block(ms, eval_stride=2)['eval_stride'], 2)
        with self.assertRaises(ValueError):
            S6C.block(ms, eval_stride=4)
        self.assertEqual((R6F.own_stride(R6F.VARIANTS['p10s4']), R6F.own_stride(R6F.VARIANTS['p10s4s'])), (2, 3))
        self.assertTrue(R6F.COMPACT_VARIANTS['p10s4'] == R6F.COMPACT_VARIANTS['p10s4s'] == 's6c')
        self.assertEqual(R6F.resolve_blend(R6F.VARIANTS['p10s4'])[0], R6F.RAPTOR_ARMS)
        cmd = ['py', '-u', 'L', '--eval-stride', '1', '--assignment', 'a.json', '--output', 'o', '--tmp', 't', '--dump-logits', 'o/l.json']
        self.assertEqual(W.with_args(cmd, eval_stride=2)[3:5], ['--eval-stride', '2'])
        c1 = W.with_args(cmd, eval_stride=1, assignment='r.json', output='o2', tmp='t2', dump_logits='o2/l.json')
        self.assertEqual(c1[4:], ['1', '--assignment', 'r.json', '--output', 'o2', '--tmp', 't2', '--dump-logits', 'o2/l.json'])
        with self.assertRaises(ValueError):
            W.with_args(cmd, chunk=96)                                                   # unknown flags refuse
        from scripts import build_r6flip_entry as B
        with mock.patch.object(B, 'S6C_MODELS_S4', members(2, ('v5',))), mock.patch.object(B, 'S6C_REFERENCE', None):
            self.assertEqual(B.s6c_block('p10s4')[0]['eval_stride'], 2)
            self.assertEqual(B.s6c_block('p10s4s')[0]['eval_stride'], 3)
            self.assertNotIn('eval_stride', B.s6c_block('p10v5')[0] if B.S6C_MODELS_E1 else {})
