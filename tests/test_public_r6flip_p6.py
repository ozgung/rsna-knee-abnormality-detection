"""P6 dual compact (final-push plan 2026-10-19, Lane A): 1/2 rank(R6 plain) + 1/2 rank(R7 plain), R7 stage last.

Synthetic data only. Heavy parts are skipped when numpy/pandas/torch or the staged build evidence are absent.
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
from knee import public_r6flip_worker as WORKER

try:
    MAIN = Path(subprocess.check_output(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], cwd=PROJECT,
                                        text=True, stderr=subprocess.DEVNULL).strip()).parent
except Exception:
    MAIN = PROJECT
R7_SMALL = MAIN / 'outputs/analysis/r7-recovery-20261013/r7-v1-assets'
try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = pd = None
try:
    import torch
except ImportError:
    torch = None

P6 = R6F.VARIANTS['p6']
P2_BLEND = {'raptor_arms': [list(a) for a in R6F.RAPTOR_ARMS], 'weights': dict(R6F.RAPTOR_WEIGHTS),
            'default_weight': R6F.DEFAULT_RAPTOR_WEIGHT}


class P6SpecTests(unittest.TestCase):
    def test_p6_is_p2_with_half_half_plain_compact(self):
        self.assertEqual(R6F.resolve_blend(P6), R6F.resolve_blend(None))            # Raptor arms, per-finding weights, default
        self.assertEqual(R6F.raptor_weight_vector(blend=P6), R6F.raptor_weight_vector())
        self.assertEqual(R6F.resolve_compact(P6), ((('r6', .5), ('r7', .5)), False))
        self.assertEqual(R6F.compact_arm_names(P6), ['r6', 'r7'])
        self.assertEqual(R6F.resolve_compact(None), ((('r6', 1.0),), True))
        for v in ('p3', 'p4', 'p5'):
            self.assertEqual(R6F.resolve_compact(R6F.VARIANTS[v]), (R6F.DEFAULT_COMPACT, True), v)
        p6p = R6F.VARIANTS['p6p']
        self.assertEqual(R6F.resolve_blend(p6p)[0], (('old320', 1.0),))
        self.assertEqual(R6F.resolve_compact(p6p), R6F.resolve_compact(P6))
        self.assertEqual(R6F.resolve_blend(p6p)[1:], R6F.resolve_blend(P6)[1:])

    def test_malformed_compact_arms_are_rejected(self):
        for name, mutate, message in (
                ('order', lambda b: b['compact_arms'].reverse(), 'COMPACT_ARM_ORDER'),
                ('r7_only', lambda b: b.update(compact_arms=[['r7', 1.0]]), 'COMPACT_ARM_ORDER'),
                ('unknown', lambda b: b['compact_arms'].__setitem__(1, ['s6', .5]), 'COMPACT_ARM_ORDER'),
                ('sum', lambda b: b['compact_arms'].__setitem__(1, ['r7', .6]), 'COMPACT_ARM_WEIGHTS_SUM_TO_ONE'),
                ('flip_type', lambda b: b.update(compact_flip=0), 'COMPACT_FLIP_FLAG'),
                ('half_schema', lambda b: b.pop('compact_flip'), 'BLEND_SCHEMA')):
            bad = copy.deepcopy(P6)
            mutate(bad)
            with self.assertRaisesRegex(ValueError, message, msg=name):
                R6F.resolve_compact(bad)


@unittest.skipIf(np is None or pd is None, 'numpy/pandas not installed')
class P6ArithmeticTests(unittest.TestCase):
    def data(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        return {a: rng.random((n, 12)) for a in ('old320', 'crop224', 'r6', 'r7')}

    @staticmethod
    def pct(v):
        return pd.DataFrame(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).to_numpy()

    def raptor(self, a):
        from knee.public_ws2_recovery import source224_csv_values
        return .65 * self.pct(a['old320']) + .35 * self.pct(source224_csv_values(a['crop224']))

    def test_half_weights_reach_the_blend(self):
        from knee.public_ws2_recovery import quantize9
        a = self.data()
        rec = R6F.recover_guarded(a, {k: ['genuine'] * 40 for k in a}, blend=P6)
        self.assertEqual((rec['mode'], rec['fallback_studies']), ('variant_complete_population', 0))
        compact = .5 * self.pct(quantize9(a['r6'])) + .5 * self.pct(quantize9(a['r7']))
        w = np.asarray(R6F.raptor_weight_vector())
        self.assertTrue(np.allclose(rec['blend'], w * self.raptor(a) + (1 - w) * compact, rtol=0, atol=1e-12))
        self.assertTrue(np.array_equal(rec['compact_slot'], .5 * rec['ranks']['r6'] + .5 * rec['ranks']['r7']))
        self.assertEqual(rec['compact_weights'], {'r6': .5, 'r7': .5})
        self.assertEqual(rec['precision']['r7'], '%.9g')
        self.assertEqual(R6F.guarded_receipt(rec)['compact_weights'], {'r6': .5, 'r7': .5})

    def test_identical_r7_reproduces_p2_arithmetic(self):
        """With R7 == R6 the compact slot is rank(R6): P6 must then equal P2's frozen A-prime path bit for bit."""
        a = self.data(60, 2)
        a['r7'] = a['r6'].copy()
        st = {k: ['genuine'] * 60 for k in a}
        p2 = R6F.recover_guarded({k: a[k] for k in ('old320', 'crop224', 'r6')}, {k: st[k] for k in ('old320', 'crop224', 'r6')})
        p6 = R6F.recover_guarded(a, st, blend=P6)
        self.assertEqual(p2['mode'], 'aprime_complete_population')
        self.assertTrue(np.allclose(p6['blend'], p2['blend'], rtol=0, atol=1e-15))
        # data-local rows inside the cap take the arm's global genuine mean, R7 included
        st['r7'][5] = 'data_local_failure'
        a['r7'][5] = np.nan
        rec = R6F.recover_guarded(a, st, blend=P6)
        self.assertEqual((rec['mode'], rec['union_recovered']), ('variant_complete_population', 1))
        self.assertEqual(rec['statuses']['r7'][5], 'recovered')

    def test_frozen_recover_refuses_the_dual_compact(self):
        a = self.data()
        m = np.zeros(40, dtype=bool)
        with self.assertRaisesRegex(ValueError, 'RECOVER_FROZEN_COMPACT_ONLY'):
            R6F.recover(a['old320'], a['r6'], a['crop224'], m, m, m, blend=P6)

    def test_guard_cut_r7_studies_fall_back_to_r6_only_ranks(self):
        from knee.public_ws2_recovery import quantize9
        n, cut = 40, 12
        a = self.data(n, 4)
        st = {k: ['genuine'] * n for k in a}
        st['r7'] = ['genuine'] * (n - cut) + ['unavailable'] * cut          # the guard stopped R7 before the last 12 studies
        a['r7'][n - cut:] = np.nan
        rec = R6F.recover_guarded(a, st, blend=P6)
        self.assertEqual((rec['mode'], rec['fallback_studies']), ('deadline_fallback_public_semantics', cut))
        self.assertEqual(rec['compact_fallback'], {'partial_compact': cut, 'r6_only': cut, 'r7_only': 0})
        self.assertEqual(rec['fallback'], {'raptor_only': 0, 'r6_only': 0, 'constant': 0, 'partial_raptor': 0})
        r6 = self.pct(quantize9(a['r6']))
        r7 = np.full((n, 12), np.nan)
        r7[:n - cut] = self.pct(quantize9(a['r7'][:n - cut]))                # R7 ranked over its own available rows
        compact = np.where(np.arange(n)[:, None] < n - cut, .5 * r6 + .5 * np.nan_to_num(r7), r6)
        w = np.asarray(R6F.raptor_weight_vector())
        self.assertTrue(np.allclose(rec['blend'], w * self.raptor(a) + (1 - w) * compact, rtol=0, atol=1e-12))
        self.assertTrue(np.allclose(rec['blend'][n - cut:], (w * self.raptor(a) + (1 - w) * r6)[n - cut:], rtol=0, atol=1e-12))
        # R7 lost entirely (stage error): every study is P2's arithmetic on rank(R6 plain)
        st['r7'] = ['unavailable'] * n
        whole = R6F.recover_guarded(a, st, blend=P6)
        p2 = R6F.recover_guarded({k: a[k] for k in ('old320', 'crop224', 'r6')}, {k: st[k] for k in ('old320', 'crop224', 'r6')})
        self.assertTrue(np.allclose(whole['blend'], p2['blend'], rtol=0, atol=1e-12))
        self.assertEqual(whole['compact_fallback']['r6_only'], n)
        # R6 missing too: Raptor half alone, nothing at all: 0.5 (never an abort)
        st['r6'] = ['unavailable'] * n
        self.assertTrue(np.allclose(R6F.recover_guarded(a, st, blend=P6)['blend'], self.raptor(a), rtol=0, atol=1e-12))
        for k in st:
            st[k] = ['unavailable'] * n
        self.assertTrue(np.all(R6F.recover_guarded(a, st, blend=P6)['blend'] == .5))


def stub_cfg(variant):
    cols = ['StudyInstanceUID'] + R6F.LABELS
    cfg = {'mode': 'entry', 'r6flip_binding': {'official_columns': cols}, 'environment': {}}
    if variant is not None:
        cfg['blend'] = copy.deepcopy(R6F.VARIANTS[variant])
        if 'r7' in R6F.compact_arm_names(cfg['blend']):
            cfg['r6flip_binding']['r7'] = {}
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
        calls.append(arm)
        if arm == fail:
            raise RuntimeError('r7 stage bug')
        if arm == slow:
            time.sleep(.4)
        return [{'status': 'NO_RECEIPT'}] * 2, {'wall_seconds': 1., 'deadline_hit': False, 'peaks': {'host_rss_bytes': 1, 'output_disk_bytes': 1},
                                                'minimum_disk_free_bytes': 1, 'exit_codes': [1, 1]}
    return launch


@unittest.skipIf(np is None or pd is None or torch is None, 'numpy/pandas/torch not installed')
class P6StageOrderTests(unittest.TestCase):
    def test_r7_is_the_last_stage_and_p2_is_unchanged(self):
        for variant, plan in ((None, ['old', 'crop224', 'r6']), ('p6', ['old', 'crop224', 'r6', 'r7']), ('p6p', ['old', 'r6', 'r7'])):
            calls = []
            cfg, cols = stub_cfg(variant)
            report, written = run_entry(cfg, cols, recorder(calls))
            self.assertEqual(calls, plan, variant)
            self.assertEqual(sorted(report['workers']), sorted(plan), variant)
            self.assertEqual(report['status'], 'PASS_WITH_FALLBACK', variant)
            self.assertTrue(np.all(written['rows'] == .5))
        self.assertEqual(sorted(report['arm_merge']), ['old320', 'r6', 'r7'])

    def test_r7_stage_error_never_aborts_the_entry(self):
        calls = []
        cfg, cols = stub_cfg('p6')
        report, _ = run_entry(cfg, cols, recorder(calls, fail='r7'))
        self.assertEqual(calls, ['old', 'crop224', 'r6', 'r7'])
        self.assertIn('r7', report['skipped_stages'])
        self.assertEqual(report['phases']['r7']['stage_error'], 'RuntimeError')
        self.assertEqual(report['status'], 'PASS_WITH_FALLBACK')

    def test_deadline_before_r7_skips_only_r7(self):
        calls = []
        cfg, cols = stub_cfg('p6')
        # deadline = start + 0.2 s: the R6 stage (sleeps 0.4 s) ends after it, so R7 is never launched
        report, _ = run_entry(cfg, cols, recorder(calls, slow='r6'), deadline_seconds=ENTRY.STAGE_RESERVE_SECONDS + .2)
        self.assertEqual(calls, ['old', 'crop224', 'r6'])
        self.assertIn('r7', report['skipped_stages'])
        self.assertEqual(report['status'], 'PASS_WITH_FALLBACK')

    def test_stage_config_must_match_the_sealed_blend(self):
        cfg, cols = stub_cfg('p6')
        del cfg['r6flip_binding']['r7']
        report, _ = run_entry(cfg, cols, recorder([]))
        self.assertEqual(report['status'], 'EMERGENCY_CONSTANT')
        cfg, cols = stub_cfg(None)
        cfg['r6flip_binding']['r7'] = {}
        report, _ = run_entry(cfg, cols, recorder([]))
        self.assertEqual(report['status'], 'EMERGENCY_CONSTANT')
        cfg, cols = stub_cfg('p6')
        cfg['r7_reference'] = {}                       # rehearsal-only fixture in an entry
        report, _ = run_entry(cfg, cols, recorder([]))
        self.assertEqual(report['status'], 'EMERGENCY_CONSTANT')

    def test_salvage_csv_uses_r6_alone_when_r7_left_nothing(self):
        import csv
        from knee.public_s6_recovery_runtime import save_ledger
        ids = ['x1', 'x2', 'x3']
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for folder, arm in (('board0-old', 'old'), ('board0-crop224', 'crop224')):
                (d / folder / arm).mkdir(parents=True)
                save_ledger(d / folder / arm / 'predictions.npz', ids, list(rng.random((3, 12))), ['genuine'] * 3)
            compact = d / 'board0-r6-production' / 'r6' / 'compact'
            compact.mkdir(parents=True)
            (compact / 'predictions.csv').write_text('StudyInstanceUID,' + ','.join(R6F.LABELS) + '\n' + ''.join(
                '%s,%s\n' % (s, ','.join(str(x) for x in rng.random(12))) for s in ids))
            root = d / 'comp'
            root.mkdir()
            (root / 'sample_submission.csv').write_text('StudyInstanceUID,' + ','.join(R6F.LABELS) + '\n')
            (root / 'test.csv').write_text('StudyInstanceUID\n' + '\n'.join(ids) + '\n')
            (root / 'test_series.csv').write_text('StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n'
                                                  + ''.join('%s,t%d,1,0,Sagittal\n' % (s, i) for i, s in enumerate(ids)))
            got = ENTRY.salvage_csv(root, d, ids, d / 'out.csv', 'RuntimeError', P6)
            self.assertEqual(got['genuine_rows'], {'old320': 3, 'crop224': 3, 'r6': 3, 'r7': 0})
            self.assertEqual(got['fallback_studies'], 3)
            rows = list(csv.reader((d / 'out.csv').open()))
            self.assertEqual([r[0] for r in rows], ['StudyInstanceUID'] + ids)
            # now an R7 compact CSV flushed by compact_infer before the crash is salvaged too
            c7 = d / 'board1-r7-production' / 'r7' / 'compact'
            c7.mkdir(parents=True)
            (c7 / 'predictions.csv').write_text('StudyInstanceUID,' + ','.join(R6F.LABELS) + '\n' + ''.join(
                '%s,%s\n' % (s, ','.join(str(x) for x in rng.random(12))) for s in ids[:2]))
            got = ENTRY.salvage_csv(root, d, ids, d / 'out2.csv', 'RuntimeError', P6)
            self.assertEqual(got['genuine_rows']['r7'], 2)
            self.assertEqual(got['fallback_studies'], 1)


class P6WorkerConfigTests(unittest.TestCase):
    def cfg(self, variant):
        cfg = {'mode': 'entry', 'boards': [{'uuid': 'a'}, {'uuid': 'b'}], 'environment': {}, 'staged': '/s', 'staged_ids': ['i0', 'i1'],
               'stage_binding': {}, 'code': '/c', 'source_pins': {}, 'r6_mount': '/m6', 'r7_mount': '/m7', 'anatomy': '/a.pt',
               'qualified_rehearsal': {'anatomy_state': 'anat'},
               'r6flip_binding': {'r6': {'manifest': {'bundle_id': 'r6'}, 'cpu_pins': {'p': 6}},
                                  'r7': {'manifest': {'bundle_id': 'r7'}, 'cpu_pins': {'p': 7}}}}
        if variant:
            cfg['blend'] = copy.deepcopy(R6F.VARIANTS[variant])
        return cfg

    def test_p2_r6_config_is_unchanged_and_p6_arms_are_plain(self):
        p2 = ENTRY.r6_worker_config(self.cfg(None), 1, 'production', ['i1'], Path('/o/b1'), None)
        self.assertFalse({'flip', 'profile', 'compact_arm'} & set(p2))
        self.assertEqual((p2['r6_mount'], p2['r6']['manifest'], p2['r6']['anatomy_state']), ('/m6', {'bundle_id': 'r6'}, 'anat'))
        self.assertTrue(WORKER.flip_enabled(p2))
        with self.assertRaisesRegex(ValueError, 'SEALED_COMPACT_ARM'):
            ENTRY.r6_worker_config(self.cfg(None), 1, 'production', ['i1'], Path('/o/b1'), None, 'r7')
        r6 = ENTRY.r6_worker_config(self.cfg('p6'), 0, 'production', ['i0'], Path('/o/b0'), None, 'r6')
        self.assertEqual((r6['flip'], r6['r6_mount'], r6['r6']['cpu_pins']), (False, '/m6', {'p': 6}))
        self.assertNotIn('profile', r6)
        r7 = ENTRY.r6_worker_config(self.cfg('p6'), 0, 'production', ['i0'], Path('/o/b0'), None, 'r7')
        self.assertEqual((r7['flip'], r7['profile'], r7['compact_arm'], r7['r6_mount'], r7['r6']['manifest'], r7['r6']['cpu_pins']),
                         (False, 'r7', 'r7', '/m7', {'bundle_id': 'r7'}, {'p': 7}))
        self.assertEqual((WORKER.profile_name(r7), WORKER.compact_arm(r7), WORKER.flip_enabled(r7)), ('r7', 'r7', False))
        with self.assertRaisesRegex(ValueError, 'COMPACT_PROFILE'):
            WORKER.profile_name({**r7, 'flip': True})
        with self.assertRaisesRegex(ValueError, 'COMPACT_ARM_IS_PROFILE'):
            WORKER.compact_arm({**r7, 'compact_arm': 'r6'})

    def test_worker_args_drop_flip_tta_only_for_plain_arms(self):
        manifest = {'arm': 'E_lrprior', 'backbone': 'convnext_small', 'size': 384, 'crop': 352, 'eval_stride': 1, 'views': 'v4',
                    'state_spec': 'state_spec.json', 'fluid_stacker': 'fluid_stacker.json', 'state_stacker': 'state_stacker.json',
                    'compile': True, 'flip_tta': False}
        base = {'r6': {'manifest': manifest}, 'r6_mount': '/m', 'data': '/d', 'code': '/c', 'anatomy': '/a/anatomy.pt'}
        self.assertIn('--flip-tta', WORKER.worker_args(base, '/o/ids.json', '/o/compact'))       # P2: flag forced on, as sealed
        plain = WORKER.worker_args({**base, 'flip': False}, '/o/ids.json', '/o/compact')
        self.assertNotIn('--flip-tta', plain)
        self.assertEqual([x for x in WORKER.worker_args(base, '/o/ids.json', '/o/compact') if x != '--flip-tta'], plain)


def r7_rehearsal_fixture():
    """A synthetic P6 rehearsal receipt (R6 production + R7 production workers) and its cfg, valid for require_r7."""
    from tests.test_public_r6flip import R6WorkerGateTests, r6_worker_receipt
    from knee.public_s6_recovery_entry import assignments
    ids = ['id%02d' % i for i in range(39)]
    pins = copy.deepcopy(R6WorkerGateTests.pins)
    pins7 = copy.deepcopy(pins)
    for f in pins7['folds']:
        f['state'] = 'r7' + f['state']
    pins7['flip'] = {}
    cfg = {'staged_ids': ids, 'binding': copy.deepcopy(R6WorkerGateTests.cfg['binding']), 'blend': copy.deepcopy(P6),
           'r6flip_binding': {'r6': {'cpu_pins': pins}, 'r7': {'cpu_pins': pins7}}}
    shards = assignments(ids)
    probs = {s: [round(.01 * (i % 90) + .001 * j, 6) for j in range(12)] for i, s in enumerate(ids)}
    workers = {'r6': [], 'r7': []}
    for arm, p in (('r6', pins), ('r7', pins7)):
        for board in range(2):
            x = r6_worker_receipt(cfg, board, shards[board], p)
            x.update(arm=arm, slot=arm + '-plain', flip={'enabled': False})
            for row in x['records']:
                row.update(calls=4, mirror_input_equal=None, windows=50, input={'render_windows': 'w' + row['id'], 'metadata': 'm'},
                           probabilities=probs[row['id']])
            workers[arm].append(x)
    cfg['r7_reference'] = {'probabilities': {s: [v + 4e-3 for v in probs[s]] for s in ids}, 'receipts_sha256': ['a', 'b']}
    return {'boards': [{'uuid': 'GPU-0'}, {'uuid': 'GPU-1'}], 'workers': workers}, cfg


class P6RehearsalGateTests(unittest.TestCase):
    def test_valid_r7_stage_passes_and_each_mutation_rejects(self):
        r, cfg = r7_rehearsal_fixture()
        got = ENTRY.require_r7(r, cfg, 'anat')
        self.assertEqual((got['studies'], got['input_equals_r6']), (39, True))
        self.assertAlmostEqual(got['aprime_reference_max_abs'], 4e-3, places=9)
        mutations = {
            'R7_APRIME_REFERENCE_PARITY': lambda r, c: c['r7_reference']['probabilities']['id03'].__setitem__(5, .9),
            'R7_INPUT_EQUALS_R6_INPUT': lambda r, c: r['workers']['r7'][1]['records'][2]['input'].update(render_windows='other'),
            'R7_INPUT_EQUALS_R6_INPUT ': lambda r, c: r['workers']['r7'][0]['records'][0].update(windows=49),
            'R7_ANATOMY_STATE_EQUALS_R6': lambda r, c: [ch[k]['anatomy'].update(state='x') for w in r['workers']['r7']
                                                         for ch in w['children'] for k in ('before', 'after')],
            'R6_FOLD_BINDING': lambda r, c: r['workers']['r7'][0]['folds'][1].update(state=c['r6flip_binding']['r6']['cpu_pins']['folds'][1]['state']),
            'R6_WORKER_PASS': lambda r, c: r['workers']['r7'][0].update(slot='r6flip'),
            'PLAIN_PASS_NO_FLIP': lambda r, c: r['workers']['r7'][1].update(flip={'perm': [], 'bin_perm': []}),
            'R6_STUDY_GATES': lambda r, c: r['workers']['r7'][1]['records'][0].update(calls=8),
            'R7_NO_INJECTION': lambda r, c: r['workers']['r7'][0].update(failed=1),
            'R7_REFERENCE_COVERAGE': lambda r, c: c['r7_reference']['probabilities'].pop('id00'),
        }
        for code, mutate in mutations.items():
            with self.subTest(code=code):
                r, cfg = r7_rehearsal_fixture()
                mutate(r, cfg)
                with self.assertRaisesRegex(ValueError, code.strip()):
                    ENTRY.require_r7(r, cfg, 'anat')

    def test_r6_plain_worker_gate_uses_the_sealed_flip(self):
        r, cfg = r7_rehearsal_fixture()
        from knee.public_s6_recovery_entry import assignments
        x = r['workers']['r6'][0]
        ids = assignments(cfg['staged_ids'])[0]
        self.assertEqual(ENTRY.require_r6_worker(x, r, cfg, 0, 'production', ids), {'anat'})
        p2 = {k: v for k, v in cfg.items() if k != 'blend'}           # the same plain receipt fails P2's flip gates
        with self.assertRaisesRegex(ValueError, 'R6_WORKER_PASS'):
            ENTRY.require_r6_worker(x, r, p2, 0, 'production', ids)


@unittest.skipIf(np is None or torch is None, 'numpy/torch not installed')
class P6ProjectionTests(unittest.TestCase):
    def test_r7_production_is_projected_like_r6(self):
        def shard(seconds, windows=10):
            return {'records': [{'production_seconds': seconds, 'status': 'genuine', 'windows': windows}] * 5, 'peak_cuda_bytes': 7}
        phase = lambda: {'wall_seconds': 100., 'peaks': {'host_rss_bytes': 1, 'output_disk_bytes': 1}, 'minimum_disk_free_bytes': 1}
        workers = {'old': [shard(2.), shard(3.)], 'crop224': [shard(2.), shard(3.)], 'r6': [shard(4.), shard(5.)]}
        phases = {a: phase() for a in ('old', 'crop224', 'r6', 'r6_sentinel1', 'r6_sentinel2')}
        windows = [10] * 870
        p2 = ENTRY.projection(10., phases, workers, 5., windows)
        workers7 = {**workers, 'r7': [shard(3., 20), shard(6., 20)]}
        p6 = ENTRY.projection(10., {**phases, 'r7': phase()}, workers7, 5., windows)
        self.assertEqual(p6['measured_production_wall_seconds']['r7'], 30.)
        self.assertEqual(p6['measured_r7_windows'], 200)
        self.assertNotIn('measured_r7_windows', p2)
        # R7: 30 s over 200 measured windows -> 1300 studies x 10 windows each; plus its fixed overhead (100 - 30)
        self.assertAlmostEqual(p6['point_seconds'] - p2['point_seconds'], 30. * 1300 * 10 / 200 + 70., places=6)
        self.assertEqual(sorted(ENTRY.resources({**phases, 'r7': phase()}, workers7)['peak_cuda_bytes_by_arm']),
                         ['crop224', 'old', 'r6', 'r7'])


@unittest.skipUnless(torch is not None and np is not None and (R7_SMALL / 'state_spec.json').is_file(), 'torch or R7 metadata not staged')
class P6PlainReconstructionTests(unittest.TestCase):
    def test_plain_reconstruction_with_actual_r7_stackers_ignores_mirror_entries(self):
        import hashlib
        import math
        from knee import public_r7_stackers as stackers
        spec = json.loads((R7_SMALL / 'state_spec.json').read_bytes())
        layout = stackers.state_layout(spec, R6F.LABELS)
        state_k = [len(spec['categories'][l]) for l in R6F.LABELS]
        bins = [(b, l) for b, labs in spec['binaries'].items() for l in labs]
        fluid = stackers.load_fluid((R7_SMALL / 'fluid_stacker.json').read_bytes(), R6F.LABELS)
        state = stackers.load_state((R7_SMALL / 'state_stacker.json').read_bytes(), R6F.LABELS, layout,
                                    hashlib.md5((R7_SMALL / 'state_spec.json').read_bytes()).hexdigest())
        g = torch.Generator().manual_seed(5)
        kmax = max(state_k)
        aux = lambda: (torch.randn(1, 12, kmax, generator=g), torch.randn(1, len(bins), generator=g))
        obs = {'plain': [torch.randn(12, generator=g) for _ in range(4)], 'mirror': None,
               'plain_aux': [aux() for _ in range(4)], 'mirror_aux': None, 'state_k': state_k, 'bin_names': bins}
        record = {'status': 'ok', 'eff_sag_p99_open': 7.5, 'bak_sag_p99_open': 2.0}
        p, detail = R6F.reconstruct(obs, record, fluid, state, spec, flip=False)
        zs = torch.stack(obs['plain'])
        p_model = torch.sigmoid(zs.mean(0)).double().numpy()
        untouched = [j for j, l in enumerate(R6F.LABELS) if l not in ('Effusion', "Baker's")]
        self.assertTrue(np.array_equal(p[untouched], p_model[untouched]))
        co = fluid['classes']['Effusion']
        z_eff = co['b0'] + co['b_z'] * float(zs.mean(0).double()[7]) + co['b_lv'] * math.log1p(7.5)
        self.assertAlmostEqual(p[7], 1 / (1 + math.exp(-z_eff)), places=12)
        stv = torch.stack([R6F.state_readout(state_k, sl, bl)[0] for sl, bl in obs['plain_aux']]).double().mean(0).numpy()
        expect, _ = stackers.apply(p_model, zs.mean(0).double().numpy(), stv, record, fluid, state, R6F.LABELS)
        self.assertTrue(np.array_equal(p, np.asarray(expect)))
        self.assertIsNone(detail['perm'])
        with self.assertRaises(Exception):                 # the flip path needs the mirror pass
            R6F.reconstruct(obs, record, fluid, state, spec)


@unittest.skipUnless(torch is not None and (MAIN / 'outputs/research/public-0958-20261017/r6-v1-small/state_spec.json').is_file(),
                     'torch or R6 metadata not staged')
class P6WorkerExecuteTests(unittest.TestCase):
    """execute() on CPU with the fake compact_infer and two spawn loader children, plain pass (no --flip-tta)."""

    def test_execute_plain_pass_ledger_and_receipt(self):
        import inspect
        small = MAIN / 'outputs/research/public-0958-20261017/r6-v1-small'
        fake = PROJECT / 'tests/fixtures/r6flip_fake'
        saved_path, saved = list(sys.path), {k: sys.modules.pop(k) for k in ('compact_train', 'compact_infer') if k in sys.modules}
        sys.path.insert(0, str(fake))
        try:
            import compact_train as CT
            spec = json.loads((small / 'state_spec.json').read_bytes())
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                mount = d / 'mount'
                mount.mkdir()
                for name in ('state_spec.json', 'fluid_stacker.json', 'state_stacker.json'):
                    (mount / name).write_bytes((small / name).read_bytes())
                pins = []
                for i in range(4):
                    torch.manual_seed(20 + i)
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
                            'models': [p['file'] for p in pins], 'flip_tta': False}
                cfg = {'ids': ['u0', 'u1', 'u2'], 'inject_uid': None, 'r6_mount': str(mount), 'data': str(d), 'code': str(fake),
                       'anatomy': str(d / 'anatomy.pt'), 'pass': 'production', 'flip': False, 'profile': 'r7', 'compact_arm': 'r7',
                       'r6': {'manifest': manifest, 'cpu_pins': {'folds': pins, 'side': {'state': R6F.module_digest(side),
                                                                                        'constructor': R6F.constructor_digest(side)}}}}
                out = d / 'board0' / 'r7'
                out.mkdir(parents=True)
                r = {'records': []}
                from knee.public_r6flip_worker import execute, load_stackers
                spec2, fluid, state, pair = load_stackers(mount, manifest, R6F.LABELS)
                import compact_infer
                execute(cfg, out, r, torch, CT, lambda: compact_infer, manifest, spec2, fluid, state, pair, check_ci=lambda ci, m: None)
                self.assertNotIn('--flip-tta', r['worker_args'])
                from knee.public_s6_qualification import file_sha
                self.assertEqual(r['ledger_sha256'], file_sha(out / 'predictions.npz'))
            self.assertEqual((r['status'], r['genuine'], r['failed']), ('PASS', 3, 0))
            self.assertEqual(r['flip'], {'enabled': False})
            for x in r['records']:
                self.assertEqual((x['calls'], x['mirror_input_equal'], x['checksum_equal']), (4, None, True))
                self.assertLessEqual(x['reconstruction_max_abs'], 1e-6)       # independent fake arithmetic vs reconstruct(flip=False)
            self.assertEqual(len(r['children']), 2)
        finally:
            sys.path[:] = saved_path
            for k in ('compact_train', 'compact_infer'):
                sys.modules.pop(k, None)
            sys.modules.update(saved)


P6_EVIDENCE = all(p.is_file() for p in (MAIN / 'outputs/kaggle/ws2-r224-v2-qual-20261015/notebook.ipynb',
                                        MAIN / 'outputs/analysis/lanea-20261019/r7-cpu-pins.json',
                                        MAIN / 'outputs/analysis/lanea-20261019/r7-v1-assets/E_lrprior_s0_f3.pt',
                                        MAIN / 'outputs/research/public-0958-20261017/r6-v1-assets/E_lrprior_s0_f3.pt'))


@unittest.skipUnless(P6_EVIDENCE, 'sealed A-prime build, R6/R7 assets or R7 CPU pins not staged')
class P6PacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import build_r6flip_entry as B
        cls.B = B
        cls.default = B.packet('rehearsal')
        cls.pack = B.packet('rehearsal', variant='p6')

    def test_rehearsal_packet_seals_the_dual_compact(self):
        cfg, b = self.pack['cfg'], self.pack['cfg']['r6flip_binding']
        from knee import public_r7_profile as R7P
        self.assertEqual(cfg['blend'], P6)
        self.assertEqual((b['r7']['bundle_id'], b['r7']['dataset'], b['r7']['dataset_version'], b['r7']['manifest_sha256']),
                         ('8c7b666e27dc0e8f', 'heliosli/rsna-knee-compact-r7cnxfsss', 1, R7P.MANIFEST_SHA256))
        self.assertEqual(b['r7']['manifest']['flip_tta'], False)
        self.assertEqual([f['file_sha256'] for f in b['r7']['cpu_pins']['folds']],
                         [R7P.ASSET_SHA256['E_lrprior_s0_f%d.pt' % i] for i in range(4)])
        self.assertEqual(b['entry_variant']['stages'], ['old320', 'crop224', 'r6', 'r7'])
        self.assertEqual((b['entry_variant']['compact_arms'], b['entry_variant']['compact_flip']), ([['r6', .5], ['r7', .5]], False))
        self.assertEqual(b['raptor_weight_vector'], self.default['cfg']['r6flip_binding']['raptor_weight_vector'])
        self.assertEqual(set(self.pack['sources']) - set(self.default['sources']), {'knee/public_r7_profile.py'})
        ref = cfg['r7_reference']['probabilities']
        self.assertEqual(list(ref), cfg['probe_order'] + cfg['binding']['visible_ids'])
        self.assertTrue(all(len(v) == 12 and all(0 <= x <= 1 for x in v) for v in ref.values()))
        for key, value in self.default['cfg'].items():
            if key != 'r6flip_binding':
                self.assertEqual(cfg[key], value, key)

    def test_slug_metadata_and_bootstrap_mount(self):
        slug = self.B.slug_for('p6', 'rehearsal')
        self.assertEqual((slug, self.B.slug_for('p6', 'entry')), ('r6flip-p6-rehearsal-20261019', 'r6flip-p6-entry-20261019'))
        meta = self.B.metadata(slug, 'p6')
        self.assertEqual((meta['id'], meta['title']), ('ozgung/' + slug, slug))
        self.assertNotIn('timeout', meta)
        self.assertEqual(meta['dataset_sources'], self.B.DATASETS + ['heliosli/rsna-knee-compact-r7cnxfsss/1'])
        script = self.B.bootstrap(self.pack)
        self.assertIn("cfg['r7_mount']=str(r7)", script)
        self.assertIn('EXACT_R7_V1_MANIFEST', script)
        self.assertNotIn('r7_mount', self.B.bootstrap(self.default))
        self.assertFalse(self.B.same_stages('p6', 'p6p') or self.B.same_stages('p6', 'p3'))

    def test_entry_strips_the_private_r7_reference(self):
        q = {'r224_state': 'x', 'anatomy_state': 'y', 'status': 'PASS'}
        entry = self.B.packet('entry', q, 'p6')
        self.assertNotIn('r7_reference', entry['cfg'])
        self.assertIn('r7', entry['cfg']['r6flip_binding'])
        text = json.dumps(entry['cfg']['r6flip_binding'])
        self.assertNotIn(self.pack['cfg']['probe_order'][0], text)

    def test_unpacked_sources_import_without_ml(self):
        with tempfile.TemporaryDirectory() as d:
            for name, content in self.pack['sources'].items():
                if name.startswith('knee/'):
                    p = Path(d) / name
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(content.encode())
            code = ('import sys; sys.path.insert(0, %r); import knee.public_r7_profile as P, knee.public_r6flip_entry as E, '
                    'knee.public_r6flip as F; assert P.BUNDLE_ID == "8c7b666e27dc0e8f" and F.compact_arm_names(F.VARIANTS["p6"]) '
                    '== ["r6", "r7"]; print("ok")' % d)
            r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, cwd=d)
            self.assertEqual(r.stdout.strip(), 'ok', r.stderr[-2000:])


if __name__ == '__main__':
    unittest.main()
