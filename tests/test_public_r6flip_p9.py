"""P9 (Lane P9, 2026-10-20): compact slot = 1/2 rank(R6 plain) + 1/2 rank(own S6 members' mean rank); own stage last.

Synthetic data only. Heavy parts skip when numpy/pandas or the staged build evidence are absent.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
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

P9 = R6F.VARIANTS['p9']
P9R = R6F.VARIANTS['p9r']


def members():
    return [{'seed': 42, 'slot': 1, 'labels': 'v3', 'file': 'E_prior_s42_fmall.pt', 'sha256': 'aa' * 32, 'bytes': 200000001,
             'dataset': 'ozgung/knee-laned-fmall-s42-20261020', 'dataset_version': 1},
            {'seed': 43, 'labels': 'v3', 'file': 'E_prior_s43_v3all.pt', 'sha256': 'bb' * 32, 'bytes': 200000002,
             'dataset': 'ozgung/knee-s6-v3all-s43-20261020', 'dataset_version': 1}]


class P9SpecTests(unittest.TestCase):
    def test_p9_is_p2_without_r224_and_a_half_half_own_compact(self):
        arms, weights, default = R6F.resolve_blend(P9)
        self.assertEqual(arms, (('old320', 1.0),))
        self.assertEqual((weights, default), (R6F.RAPTOR_WEIGHTS, R6F.DEFAULT_RAPTOR_WEIGHT))
        self.assertEqual(R6F.resolve_compact(P9), ((('r6', .5), ('s6c', .5)), False))
        self.assertEqual(P9['compact_arms'], [['r6', .5], ['s6c', .5]])          # sealed equal weights
        self.assertEqual(R6F.resolve_blend(P9R)[0], R6F.RAPTOR_ARMS)
        self.assertEqual(R6F.resolve_compact(P9R), R6F.resolve_compact(P9))
        self.assertTrue(R6F.own_compact_arm(P9) and R6F.own_compact_arm(P9R))
        self.assertFalse(R6F.compact_shared(P9) or R6F.own_compact_arm(R6F.VARIANTS['p7']) or R6F.own_compact_arm(R6F.VARIANTS['p6s'])
                         or R6F.own_compact_arm(None))
        self.assertEqual(R6F.raptor_weight_vector(blend=P9), R6F.raptor_weight_vector())
        self.assertEqual(R6F.compact_arm_names(P9), ['r6', 's6c'])
        self.assertEqual(R6F.PRECISION['s6c'], '%.9g')
        self.assertNotIn('p9', R6F.COMPACT_VARIANTS)                              # the R6 bundle still runs in P9

    def test_malformed_own_compact_is_rejected(self):
        for arms, flip, code in (([['s6c', 1.0]], False, 'COMPACT_ARM_ORDER'), ([['s6c', .5], ['r6', .5]], False, 'COMPACT_ARM_ORDER'),
                                 ([['r6', .5], ['s6c', .5]], True, 'OWN_COMPACT_NEEDS_R6_PLAIN_PAIR'),
                                 ([['r6', .4], ['r7', .3], ['s6c', .3]], False, 'OWN_COMPACT_NEEDS_R6_PLAIN_PAIR')):
            bad = {**copy.deepcopy(P9), 'compact_arms': arms, 'compact_flip': flip}
            with self.assertRaises(ValueError) as cm:
                R6F.own_compact_arm(bad)
            self.assertEqual(str(cm.exception), code, arms)

    def test_slugs(self):
        from scripts import build_r6flip_entry as B
        self.assertEqual((B.slug_for('p9', 'rehearsal'), B.slug_for('p9', 'entry'), B.slug_for('p9r', 'rehearsal')),
                         ('r6flip-p9-rehearsal-20261020', 'r6flip-p9-entry-20261020', 'r6flip-p9r-rehearsal-20261020'))
        self.assertTrue(B.own_dual('p9') and B.has_s6c('p9') and B.has_s6c('p7') and not B.own_dual('p7') and not B.has_s6c('p6s'))
        self.assertFalse(B.same_stages('p9', 'p9r') or B.same_stages('p9', 'p6p') or B.same_stages('p9', 'p7'))


@unittest.skipIf(np is None or pd is None, 'numpy/pandas not installed')
class P9ArithmeticTests(unittest.TestCase):
    @staticmethod
    def pct(v):
        return pd.DataFrame(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).to_numpy()

    def data(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        a = {k: rng.random((n, 12)) for k in ('old320', 'crop224', 'r6')}
        a['s6c'] = S6C.mean_rank(rng.random((n, 2, 12)), np.ones(n, bool))
        return a

    def test_half_weights_reach_the_blend(self):
        from knee.public_ws2_recovery import quantize9
        n = 40
        a = self.data(n)
        st = {k: ['genuine'] * n for k in a}
        w = np.asarray(R6F.raptor_weight_vector())
        for blend, raptor in ((P9, self.pct(a['old320'])), (P9R, None)):
            rec = R6F.recover_guarded(a, st, blend=blend)
            self.assertEqual((rec['mode'], rec['fallback_studies'], rec['compact_weights']), ('variant_complete_population', 0, {'r6': .5, 's6c': .5}))
            compact = .5 * self.pct(quantize9(a['r6'])) + .5 * self.pct(quantize9(a['s6c']))
            if raptor is None:
                from knee.public_ws2_recovery import source224_csv_values
                raptor = .65 * self.pct(a['old320']) + .35 * self.pct(source224_csv_values(a['crop224']))
            self.assertTrue(np.allclose(rec['blend'], w * raptor + (1 - w) * compact, rtol=0, atol=1e-12))
            self.assertTrue(np.allclose(rec['compact_slot'], compact, rtol=0, atol=1e-12))
        with self.assertRaises(ValueError):          # the frozen P2 path refuses the dual compact
            R6F.recover(a['old320'], a['r6'], a['crop224'], *[np.zeros(n, bool)] * 3, blend=P9R)

    def test_guard_cut_own_studies_fall_back_to_r6_only_ranks(self):
        from knee.public_ws2_recovery import quantize9
        n, cut = 40, 9
        a = self.data(n, 1)
        st = {k: ['genuine'] * n for k in a}
        st['s6c'] = ['genuine'] * (n - cut) + ['unavailable'] * cut        # the guard stopped the own stage before the last 9 studies
        a['s6c'][n - cut:] = np.nan
        rec = R6F.recover_guarded(a, st, blend=P9)
        self.assertEqual((rec['mode'], rec['fallback_studies']), ('deadline_fallback_public_semantics', cut))
        self.assertEqual(rec['compact_fallback'], {'partial_compact': cut, 'r6_only': cut, 's6c_only': 0})
        self.assertEqual(rec['fallback'], {'raptor_only': 0, 'r6_only': 0, 'constant': 0, 'partial_raptor': 0})
        w = np.asarray(R6F.raptor_weight_vector())
        r6 = self.pct(quantize9(a['r6']))
        own = np.full((n, 12), np.nan)
        own[:n - cut] = self.pct(quantize9(a['s6c'][:n - cut]))             # own members ranked over their own available rows
        compact = np.where(np.arange(n)[:, None] < n - cut, .5 * r6 + .5 * np.nan_to_num(own), r6)
        raptor = self.pct(a['old320'])
        self.assertTrue(np.allclose(rec['blend'], w * raptor + (1 - w) * compact, rtol=0, atol=1e-12))
        self.assertTrue(np.allclose(rec['blend'][n - cut:], (w * raptor + (1 - w) * r6)[n - cut:], rtol=0, atol=1e-12))
        # own stage lost entirely (stage error): every study is the R6-plain-only arithmetic
        st['s6c'] = ['unavailable'] * n
        whole = R6F.recover_guarded(a, st, blend=P9)
        self.assertTrue(np.allclose(whole['blend'], w * raptor + (1 - w) * r6, rtol=0, atol=1e-12))
        self.assertEqual(whole['compact_fallback']['r6_only'], n)
        # R6 lost, own stage complete: the own ranks alone are the compact slot; nothing at all: 0.5
        st['s6c'], st['r6'] = ['genuine'] * n, ['unavailable'] * n
        a = self.data(n, 1)
        only = R6F.recover_guarded(a, st, blend=P9)
        self.assertTrue(np.allclose(only['blend'], w * raptor + (1 - w) * self.pct(quantize9(a['s6c'])), rtol=0, atol=1e-12))
        for k in st:
            st[k] = ['unavailable'] * n
        self.assertTrue(np.all(R6F.recover_guarded(a, st, blend=P9)['blend'] == .5))


def stub_cfg(variant, block=True):
    cols = ['StudyInstanceUID'] + R6F.LABELS
    cfg = {'mode': 'entry', 'r6flip_binding': {'official_columns': cols, 'entry_variant': {'name': variant}, 'r6': {}}, 'environment': {},
           'blend': copy.deepcopy(R6F.VARIANTS[variant])}
    if block:
        cfg['s6c'] = S6C.block(members())
    return cfg, cols


def run_entry(cfg, cols, launch, deadline_seconds=None):
    ids = ['s1', 's2', 's3']
    written = {}
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / 'o'
        patches = [mock.patch.object(ENTRY, 'launch_guarded', launch),
                   mock.patch.object(ENTRY, 'inventory', lambda: [{'index': 0, 'uuid': 'a', 'name': 'Tesla T4'},
                                                                 {'index': 1, 'uuid': 'b', 'name': 'Tesla T4'}]),
                   mock.patch('knee.public_s6_recovery_entry.stage', lambda *a, **k: {}),
                   mock.patch('knee.public_fullfield_entry.test_contract', lambda root, labels, visible=False: (cols, list(ids), {})),
                   mock.patch('knee.public_fullfield_entry.write_csv',
                              lambda path, columns, sids, rows: written.update(rows=np.asarray(rows)) or 'sha')]
        if deadline_seconds is not None:
            patches.append(mock.patch.object(R6F, 'DEADLINE_SECONDS', deadline_seconds))
        for p in patches:
            p.start()
        try:
            report = ENTRY.run(Path(d), cfg, out, started=time.monotonic())
        finally:
            for p in reversed(patches):
                p.stop()
    return report, written


def recorder(calls, fail=None, slow=None):
    def launch(c, o, arm, deadline, pass_name='production', **kw):
        calls.append((arm, pass_name, kw.get('inject_uid')))
        if arm == fail:
            raise RuntimeError('own stage bug')
        if arm == slow:
            time.sleep(.4)
        return [{'status': 'NO_RECEIPT'}] * 2, {'wall_seconds': 1., 'deadline_hit': False, 'peaks': {'host_rss_bytes': 1, 'output_disk_bytes': 1},
                                                'minimum_disk_free_bytes': 1, 'exit_codes': [1, 1]}
    return launch


@unittest.skipIf(np is None or pd is None, 'numpy/pandas not installed')
class P9StageOrderTests(unittest.TestCase):
    def test_own_stage_is_last_and_r6_still_runs(self):
        for variant, plan in (('p9', ['old', 'r6', 's6c']), ('p9r', ['old', 'crop224', 'r6', 's6c'])):
            calls = []
            cfg, cols = stub_cfg(variant)
            report, written = run_entry(cfg, cols, recorder(calls))
            self.assertEqual([a for a, _, _ in calls], plan, variant)
            self.assertEqual(sorted(report['workers']), sorted(plan), variant)
            self.assertEqual(report['status'], 'PASS_WITH_FALLBACK', variant)
            self.assertTrue(np.all(written['rows'] == .5))
        self.assertEqual(sorted(report['arm_merge']), ['crop224', 'old320', 'r6', 's6c'])

    def test_own_stage_error_never_aborts_the_entry(self):
        calls = []
        cfg, cols = stub_cfg('p9')
        report, _ = run_entry(cfg, cols, recorder(calls, fail='s6c'))
        self.assertEqual([a for a, _, _ in calls], ['old', 'r6', 's6c'])
        self.assertIn('s6c', report['skipped_stages'])
        self.assertEqual(report['phases']['s6c']['stage_error'], 'RuntimeError')
        self.assertEqual(report['status'], 'PASS_WITH_FALLBACK')

    def test_deadline_before_the_own_stage_skips_only_it(self):
        calls = []
        cfg, cols = stub_cfg('p9')
        report, _ = run_entry(cfg, cols, recorder(calls, slow='r6'), deadline_seconds=ENTRY.STAGE_RESERVE_SECONDS + .2)
        self.assertEqual([a for a, _, _ in calls], ['old', 'r6'])
        self.assertEqual(report['skipped_stages'], ['s6c'])
        self.assertEqual(report['status'], 'PASS_WITH_FALLBACK')

    def test_stage_config_must_match_the_sealed_blend(self):
        cfg, cols = stub_cfg('p9', block=False)               # P9 blend without the member block: refused before any stage
        calls = []
        report, _ = run_entry(cfg, cols, recorder(calls))
        self.assertEqual((calls, report['status']), ([], 'EMERGENCY_CONSTANT'))
        cfg, cols = stub_cfg('p9')                              # a member block on a variant that has no own arm
        cfg['blend'] = copy.deepcopy(R6F.VARIANTS['p6p'])
        cfg['r6flip_binding']['r7'] = {}
        report, _ = run_entry(cfg, cols, recorder(calls))
        self.assertEqual(report['status'], 'EMERGENCY_CONSTANT')

    def test_rehearsal_runs_r6_sentinels_with_injection_and_own_sentinels_without(self):
        cfg, cols = stub_cfg('p9')
        probes = ['q%02d' % i for i in range(36)]
        cfg.update(mode='rehearsal', probes={q: {} for q in probes}, probe_order=probes, s6_sentinels=['z1', 'z2', 'z3', 'z4'])
        calls = []
        with mock.patch('knee.public_fullfield_entry.read_series', lambda p: {q: {} for q in probes}):
            try:
                run_entry(cfg, cols, recorder(calls))
            except Exception:
                pass                                            # the stub rehearsal fails at the strict merge; the launch order is what matters
        self.assertEqual(calls[:4], [('old', 'production', None), ('r6', 'sentinel1', 'z1'), ('r6', 'sentinel2', None), ('r6', 'production', None)])
        self.assertEqual([c for c in calls if c[0] == 's6c'], [('s6c', 'sentinel1', None), ('s6c', 'sentinel2', None), ('s6c', 'production', None)])

    def test_salvage_csv_reads_r6_and_own_ledgers_in_their_own_slots(self):
        import csv
        from knee.public_s6_recovery_runtime import save_ledger
        from tests.test_public_s6c import ArrayTests, models
        ids = ['x1', 'x2', 'x3', 'x4']
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'board0-old' / 'old').mkdir(parents=True)
            save_ledger(d / 'board0-old' / 'old' / 'predictions.npz', ids, list(rng.random((4, 12))), ['genuine'] * 4)
            compact = d / 'board0-r6-production' / 'r6' / 'compact'
            compact.mkdir(parents=True)
            (compact / 'predictions.csv').write_text('StudyInstanceUID,' + ','.join(R6F.LABELS) + '\n' + ''.join(
                '%s,%s\n' % (s, ','.join(str(x) for x in rng.random(12))) for s in ids))
            ArrayTests().write_board(d, 0, ids[:3], 7)           # own ledger: three of four studies
            root = d / 'comp'
            root.mkdir()
            (root / 'sample_submission.csv').write_text('StudyInstanceUID,' + ','.join(R6F.LABELS) + '\n')
            (root / 'test.csv').write_text('StudyInstanceUID\n' + '\n'.join(ids) + '\n')
            (root / 'test_series.csv').write_text('StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n'
                                                  + ''.join('%s,t%d,1,0,Sagittal\n' % (s, i) for i, s in enumerate(ids)))
            got = ENTRY.salvage_csv(root, d, ids, d / 'out.csv', 'RuntimeError', P9, S6C.block(models()))   # three-member ledger
            self.assertEqual(got['genuine_rows'], {'old320': 4, 'r6': 4, 's6c': 3})
            self.assertEqual(got['fallback_studies'], 1)
            rows = list(csv.reader((d / 'out.csv').open()))
            self.assertEqual([r[0] for r in rows], ['StudyInstanceUID'] + ids)


@unittest.skipIf(np is None or pd is None, 'numpy/pandas not installed')
class P9GateTests(unittest.TestCase):
    def test_own_gates_read_the_s6c_worker_key(self):
        from tests.test_public_s6c import valid_compact_rehearsal
        r, cfg = valid_compact_rehearsal()
        r9 = {'boards': r['boards'], 'workers': {'s6c': r['workers']['r6'], 's6c_sentinels': r['workers']['r6_sentinels'],
                                                 'r6': [{'status': 'PASS'}] * 2}}
        q = S6C.require_compact(r9, cfg, key='s6c')
        self.assertEqual((q['seeds'], q['sentinel_max_abs_dprob'], q['reference_studies']), (3, 0.0, 0))
        with self.assertRaises(KeyError):
            S6C.require_compact(r9, cfg)                                # the P7 key does not see the own stage
        r9['workers']['s6c'][1] = dict(r9['workers']['s6c'][1], failed=1)
        with self.assertRaises(ValueError):
            S6C.require_compact(r9, cfg, key='s6c')

    def test_own_production_and_sentinels_are_projected_like_r6(self):
        def shard(seconds, windows=10):
            return {'records': [{'production_seconds': seconds, 'status': 'genuine', 'windows': windows}] * 5, 'peak_cuda_bytes': 7}
        phase = lambda: {'wall_seconds': 100., 'peaks': {'host_rss_bytes': 1, 'output_disk_bytes': 1}, 'minimum_disk_free_bytes': 1}
        workers = {'old': [shard(2.), shard(3.)], 'r6': [shard(4.), shard(5.)]}
        phases = {a: phase() for a in ('old', 'r6', 'r6_sentinel1', 'r6_sentinel2')}
        windows = [10] * 870
        base = ENTRY.projection(10., phases, workers, 5., windows)
        workers9 = {**workers, 's6c': [shard(3., 20), shard(6., 20)]}
        phases9 = {**phases, 's6c': phase(), 's6c_sentinel1': phase(), 's6c_sentinel2': phase()}
        p9 = ENTRY.projection(10., phases9, workers9, 5., windows)
        self.assertEqual((p9['measured_production_wall_seconds']['s6c'], p9['measured_s6c_windows'], p9['s6c_sentinel_pass_wall_seconds']),
                         (30., 200, 200.))
        self.assertNotIn('measured_s6c_windows', base)
        # own stage: 30 s over 200 windows -> 1300 x 10 windows; fixed overhead (100 - 30) plus the two sentinel walls
        self.assertAlmostEqual(p9['point_seconds'] - base['point_seconds'], 30. * 1300 * 10 / 200 + 70. + 200., places=6)
        self.assertEqual(sorted(ENTRY.resources(phases9, workers9)['peak_cuda_bytes_by_arm']), ['old', 'r6', 's6c'])


def staged():
    try:
        from scripts import build_r6flip_entry as B
        return B.ARCH.is_dir() and Path(B.S6C_REFERENCE['path']).is_file() and B.CPU_PINS.is_file()
    except Exception:
        return False


@unittest.skipUnless(staged(), 'A-prime archive / reference not staged')
class P9PacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import build_r6flip_entry as B
        cls.B = B
        cls.default = B.packet('rehearsal')
        cls.pack = B.packet('rehearsal', variant='p9')

    def test_rehearsal_packet_seals_members_and_keeps_the_r6_bundle(self):
        cfg, b = self.pack['cfg'], self.pack['cfg']['r6flip_binding']
        self.assertEqual(cfg['blend'], P9)
        block = cfg['s6c']
        self.assertEqual(block['aliases'], ['E_prior_s42_f1.pt', 'E_prior_s43_f0.pt'])
        self.assertEqual(block['labels'], 'v3')
        self.assertEqual([(m['dataset'], m['dataset_version'], m['sha256'][:8]) for m in block['models']],
                         [('ozgung/knee-laned-fmall-s42-20261020', 1, '6a13f52e'), ('ozgung/knee-s6-v3all-s43-20261020', 1, 'a99ad897')])
        self.assertEqual(sorted(block['reference']['probabilities']), block['aliases'])
        self.assertTrue(all(len(v) == 36 for v in block['reference']['probabilities'].values()))
        self.assertEqual(b['r6'], self.default['cfg']['r6flip_binding']['r6'])          # the R6 bundle binding is P2's
        self.assertEqual((b['s6c']['block_digest'], b['s6c']['aliases']), (R6F.digest(block), block['aliases']))
        self.assertEqual(b['entry_variant']['stages'], ['old320', 'r6', 's6c'])
        self.assertEqual((b['entry_variant']['compact_arms'], b['entry_variant']['compact_flip']), ([['r6', .5], ['s6c', .5]], False))
        self.assertEqual(b['frame']['compact_half']['arms'], {'r6': .5, 's6c': .5})
        self.assertEqual(b['frame']['compact_half']['s6c']['members'], block['aliases'])
        self.assertNotIn('heliosli R6 bundle', b['frame']['excluded'])
        self.assertEqual(b['raptor_weight_vector'], self.default['cfg']['r6flip_binding']['raptor_weight_vector'])
        self.assertEqual(set(self.pack['sources']) - set(self.default['sources']), {*self.B.S6C_SOURCES, S6C.CAPTURED})
        for key, value in self.default['cfg'].items():
            if key not in ('r6flip_binding', 'blend', 's6c'):
                self.assertEqual(cfg[key], value, key)

    def test_member_list_is_sealed_per_variant(self):
        self.assertEqual([m['sha256'][:8] for m in self.B.s6c_models('p7')], ['03da2cac', '6a13f52e', 'a99ad897'])    # P7 unchanged
        self.assertEqual(self.B.s6c_models('p9r'), self.B.s6c_models('p9'))
        with mock.patch.object(self.B, 'S6C_MODELS_P9', None):
            with self.assertRaises(ValueError):
                self.B.packet('rehearsal', variant='p9')
        with mock.patch.object(self.B, 'S6C_MODELS_P9', self.B.S6C_MODELS_P9 + [dict(members()[1], seed=99, sha256='cc' * 32)]):
            with self.assertRaises(ValueError) as cm:                         # a member without an A100 reference cannot be sealed
                self.B.packet('rehearsal', variant='p9')
            self.assertEqual(str(cm.exception), 'S6C_REFERENCE_COVERS_SEALED_MEMBERS')

    def test_metadata_and_bootstrap_mount_both_r6_and_members(self):
        slug = self.B.slug_for('p9', 'rehearsal')
        meta = self.B.metadata(slug, 'p9')
        self.assertEqual((meta['id'], meta['title']), ('ozgung/' + slug, slug))
        self.assertNotIn('timeout', meta)
        self.assertEqual(meta['dataset_sources'], self.B.DATASETS + ['nartaa/rsna-knee-s6fall-weights/3', 'ozgung/knee-laned-fmall-s42-20261020/1',
                                                                     'ozgung/knee-s6-v3all-s43-20261020/1'])
        self.assertEqual(meta['model_sources'], [])
        script = self.B.bootstrap(self.pack)
        compile(script, '<bootstrap>', 'exec')
        for text in ('EXACT_R6_V1_MANIFEST', 'EXACT_S6C_CODE_MOUNT', 'EXACT_S6C_SEED_BYTES', 'r6_mount=str(r6)', 'helios_mount=str(anatomy.parent)'):
            self.assertIn(text, script)
        self.assertNotIn('r6_mount=None', script)
        self.assertNotIn('EXACT_S6C_SEED_BYTES', self.B.bootstrap(self.default))         # the default build is untouched

    def test_entry_strips_private_fixtures_and_keeps_the_block(self):
        q = {'r224_state': None, 'anatomy_state': 'y', 'status': 'PASS'}
        entry = self.B.packet('entry', q, 'p9')
        cfg = entry['cfg']
        self.assertTrue(all(k not in cfg for k in self.B.PRIVATE_CFG))
        self.assertEqual(cfg['s6c'], self.pack['cfg']['s6c'])
        self.assertNotIn('r224_state_pin', cfg)
        text = json.dumps(cfg['r6flip_binding'])
        self.assertNotIn(self.pack['cfg']['probe_order'][0], text)

    def test_unpacked_sources_import_without_ml(self):
        with tempfile.TemporaryDirectory() as d:
            for name, content in self.pack['sources'].items():
                if name.startswith('knee/'):
                    p = Path(d) / name
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(content.encode())
            code = ('import sys; sys.path.insert(0, %r); import knee.public_s6c as S, knee.public_r6flip_entry as E, '
                    'knee.public_r6flip as F; assert F.own_compact_arm(F.VARIANTS["p9"]) and E._s6c() is S; print("ok")' % d)
            r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, cwd=d)
            self.assertEqual(r.stdout.strip(), 'ok', r.stderr[-2000:])


if __name__ == '__main__':
    unittest.main()
