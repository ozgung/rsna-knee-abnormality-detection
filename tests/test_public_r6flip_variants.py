"""Sealed entry variants P3/P4/P5 of the R6-flip controller (cfg[blend], na-full stage) and the unchanged default.

Synthetic data only. Heavy parts are skipped when numpy/pandas or the staged build evidence are absent.
"""
import copy
import hashlib
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

try:
    MAIN = Path(subprocess.check_output(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], cwd=PROJECT,
                                        text=True, stderr=subprocess.DEVNULL).strip()).parent
except Exception:
    MAIN = PROJECT
try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = pd = None
try:
    import torch  # noqa: F401
except ImportError:
    torch = None

DEFAULT_BLEND = {'raptor_arms': [list(a) for a in R6F.RAPTOR_ARMS], 'weights': dict(R6F.RAPTOR_WEIGHTS),
                 'default_weight': R6F.DEFAULT_RAPTOR_WEIGHT}
# Output of the code before cfg[blend] existed (commit 896407d) on seeded inputs; the default must reproduce it bit for bit.
GOLDEN = {'complete': '4839d09796c342739e29facec5ae40ac88f6f19a068c341cf2c09801655a53de',
          'fallback': '4408e955c4323472c3a149265354ab4874b702d7fe22883b36e619d9bdc0c844',
          'recovered': 'f04cd91c341cdd1c287e4fa8e44012789a63c79e80f92b87bfb4d11412d656b5',
          'weights': '7d21e0b91aaf645936eafc7c9274edcfdbf70f3c04887092de51cdb124117b56'}
V2_COLUMNS = ['Lateral OA', 'PF OA', 'Synovitis', "Baker's"]


class VariantTableTests(unittest.TestCase):
    def test_variants_match_the_spec(self):
        self.assertEqual(sorted(R6F.VARIANTS), ['p3', 'p4', 'p5'])
        self.assertEqual(R6F.resolve_blend(R6F.VARIANTS['p3'])[0], (('old320', .52), ('crop224', .28), ('nafull', .2)))
        self.assertEqual(R6F.resolve_blend(R6F.VARIANTS['p4'])[0], R6F.resolve_blend(R6F.VARIANTS['p3'])[0])
        self.assertEqual(R6F.resolve_blend(R6F.VARIANTS['p5'])[0], (('old320', .65), ('nafull', .35)))
        self.assertEqual(R6F.raptor_weight_vector(blend=R6F.VARIANTS['p3']), R6F.raptor_weight_vector())
        self.assertEqual(R6F.raptor_weight_vector(blend=R6F.VARIANTS['p5']), R6F.raptor_weight_vector())
        v2 = R6F.raptor_weight_vector(blend=R6F.VARIANTS['p4'])
        self.assertEqual(dict(zip(R6F.LABELS, v2)), {**dict.fromkeys(R6F.LABELS, .5), 'Lateral OA': .15, 'PF OA': .65,
                                                     'Synovitis': .85, "Baker's": .30})
        self.assertEqual(R6F.blend_arm_names(R6F.VARIANTS['p5']), ['old320', 'nafull'])
        self.assertEqual(R6F.blend_arm_names(), ['old320', 'crop224'])

    def test_default_blend_is_the_frozen_one(self):
        self.assertEqual(R6F.resolve_blend(None), (R6F.RAPTOR_ARMS, R6F.RAPTOR_WEIGHTS, R6F.DEFAULT_RAPTOR_WEIGHT))
        self.assertEqual(R6F.resolve_blend(DEFAULT_BLEND)[0], R6F.RAPTOR_ARMS)
        self.assertEqual(R6F.raptor_weight_vector(blend=DEFAULT_BLEND), R6F.raptor_weight_vector())

    def test_malformed_blends_are_rejected(self):
        good = copy.deepcopy(R6F.VARIANTS['p3'])
        for name, mutate, message in (
                ('sum', lambda b: b['raptor_arms'].__setitem__(2, ['nafull', .25]), 'BLEND_ARM_WEIGHTS_SUM_TO_ONE'),
                ('order', lambda b: b['raptor_arms'].reverse(), 'BLEND_ARM_ORDER'),
                ('unknown', lambda b: b['raptor_arms'].__setitem__(1, ['crop320', .28]), 'BLEND_ARM_ORDER'),
                ('no_old320', lambda b: b.update(raptor_arms=[['crop224', .5], ['nafull', .5]]), 'BLEND_ARM_ORDER'),
                ('weight_range', lambda b: b['weights'].update({'PF OA': 1.2}), 'BLEND_WEIGHTS'),
                ('weight_target', lambda b: b['weights'].update({'Meniscus': .5}), 'BLEND_WEIGHTS'),
                ('extra_key', lambda b: b.update(extra=1), 'BLEND_SCHEMA')):
            bad = copy.deepcopy(good)
            mutate(bad)
            with self.assertRaisesRegex(ValueError, message, msg=name):
                R6F.resolve_blend(bad)


@unittest.skipIf(np is None or pd is None, 'numpy/pandas not installed')
class BlendArithmeticTests(unittest.TestCase):
    def data(self, n=40, seed=0, arms=('old320', 'r6', 'crop224', 'nafull')):
        rng = np.random.default_rng(seed)
        return {a: rng.random((n, 12)) for a in arms}

    def expected(self, arrays, blend):
        """Independent restatement of the published arithmetic for a sealed blend with every arm complete."""
        from knee.public_ws2_recovery import quantize9, source224_csv_values
        pct = lambda v: pd.DataFrame(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).to_numpy()
        arms, weights, default = R6F.resolve_blend(blend)
        slot = sum(w * pct(source224_csv_values(arrays[a]) if a == 'crop224' else arrays[a]) for a, w in arms)
        r6 = pct(quantize9(arrays['r6']))
        out = np.empty_like(slot)
        for j, name in enumerate(R6F.LABELS):
            w = weights.get(name, default)
            out[:, j] = w * slot[:, j] + (1.0 - w) * r6[:, j]
        return out

    def test_default_reproduces_the_pre_config_arithmetic_bit_for_bit(self):
        got = {}
        a = self.data(40, 0, ('old320', 'r6', 'crop224'))
        st = {k: ['genuine'] * 40 for k in a}
        got['complete'] = R6F.recover_guarded(a, st)['blend']
        st2 = {k: list(v) for k, v in st.items()}
        st2['r6'] = ['genuine'] * 20 + ['unavailable'] * 20
        st2['crop224'][0] = 'unavailable'
        st2['old320'][25] = st2['crop224'][25] = 'unavailable'
        a2 = {k: v.copy() for k, v in a.items()}
        a2['r6'][20:] = np.nan
        got['fallback'] = R6F.recover_guarded(a2, st2)['blend']
        a3 = self.data(100, 1, ('old320', 'r6', 'crop224'))
        st3 = {k: ['genuine'] * 100 for k in a3}
        st3['r6'][3] = st3['r6'][7] = st3['old320'][3] = 'data_local_failure'
        a3['r6'][[3, 7]] = np.nan
        a3['old320'][3] = np.nan
        rec = R6F.recover_guarded(a3, st3)
        self.assertEqual(rec['mode'], 'aprime_complete_population')
        got['recovered'] = rec['blend']
        got['weights'] = np.asarray(R6F.raptor_weight_vector())
        self.assertEqual({k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in got.items()}, GOLDEN)
        # the default sealed as an explicit blend takes the same frozen path
        for arrays, statuses, key in ((a, st, 'complete'), (a2, st2, 'fallback'), (a3, st3, 'recovered')):
            self.assertTrue(np.array_equal(R6F.recover_guarded(arrays, statuses, blend=DEFAULT_BLEND)['blend'], got[key]))

    def test_each_variants_arms_and_weights_reach_the_blend(self):
        for variant, blend in R6F.VARIANTS.items():
            arrays = self.data()
            names = R6F.blend_arm_names(blend) + ['r6']
            statuses = {a: ['genuine'] * 40 for a in names}
            rec = R6F.recover_guarded({a: arrays[a] for a in names}, statuses, blend=blend)
            self.assertEqual((rec['mode'], rec['fallback_studies']), ('variant_complete_population', 0), variant)
            self.assertTrue(np.allclose(rec['blend'], self.expected(arrays, blend), rtol=0, atol=1e-12), variant)
            self.assertEqual(rec['raptor_weights'], dict(zip(R6F.LABELS, R6F.raptor_weight_vector(blend=blend))))
            self.assertEqual(sorted(rec['ranks']), sorted(names))
            self.assertEqual(rec['recipe'], R6F.VERSION)
            receipt = R6F.guarded_receipt(rec)
            self.assertEqual(receipt['fallback_studies'], 0)

    def test_p4_differs_from_p3_only_in_the_four_heliosli_columns(self):
        arrays = self.data()
        names = ['old320', 'crop224', 'nafull', 'r6']
        statuses = {a: ['genuine'] * 40 for a in names}
        p3 = R6F.recover_guarded(arrays, statuses, blend=R6F.VARIANTS['p3'])['blend']
        p4 = R6F.recover_guarded(arrays, statuses, blend=R6F.VARIANTS['p4'])['blend']
        changed = {R6F.LABELS[j] for j in range(12) if not np.array_equal(p3[:, j], p4[:, j])}
        self.assertEqual(changed, set(V2_COLUMNS))

    def test_p5_ignores_crop224_and_needs_none(self):
        arrays = self.data()
        statuses = {a: ['genuine'] * 40 for a in ('old320', 'nafull', 'r6')}
        rec = R6F.recover_guarded({a: arrays[a] for a in statuses}, statuses, blend=R6F.VARIANTS['p5'])
        self.assertNotIn('crop224', rec['ranks'])
        from knee.public_ws2_recovery import quantize9
        pct = lambda v: pd.DataFrame(v).rank(method='average', pct=True).to_numpy()
        slot = .65 * pct(arrays['old320']) + .35 * pct(arrays['nafull'])      # ACL carries the 0.5 default weight
        self.assertTrue(np.allclose(rec['blend'][:, 0], .5 * slot[:, 0] + .5 * pct(quantize9(arrays['r6']))[:, 0], rtol=0, atol=1e-12))
        self.assertTrue(np.allclose(rec['blend'], self.expected(arrays, R6F.VARIANTS['p5']), rtol=0, atol=1e-12))

    def test_variant_data_local_rows_take_the_global_genuine_mean_under_the_cap(self):
        n = 100
        arrays = self.data(n, 3)
        statuses = {a: ['genuine'] * n for a in arrays}
        statuses['nafull'][4] = statuses['nafull'][9] = 'data_local_failure'
        arrays['nafull'][[4, 9]] = np.nan
        rec = R6F.recover_guarded(arrays, statuses, blend=R6F.VARIANTS['p3'])
        self.assertEqual((rec['mode'], rec['union_recovered'], rec['cap']), ('variant_complete_population', 2, 3))
        self.assertEqual(rec['statuses']['nafull'][4], 'recovered')
        self.assertTrue(np.allclose(rec['means']['nafull'], np.delete(arrays['nafull'], [4, 9], 0).mean(0)))
        statuses['crop224'][1] = statuses['old320'][2] = 'data_local_failure'     # union 4 > cap 3: fallback, never an abort
        over = R6F.recover_guarded(arrays, statuses, blend=R6F.VARIANTS['p3'])
        self.assertEqual(over['mode'], 'deadline_fallback_public_semantics')
        self.assertTrue(np.isfinite(over['blend']).all())

    def test_missing_nafull_falls_back_to_the_other_raptor_ranks(self):
        from knee.public_ws2_recovery import quantize9, source224_csv_values
        n = 40
        arrays = self.data(n, 5)
        names = ['old320', 'crop224', 'nafull', 'r6']
        statuses = {a: ['genuine'] * n for a in names}
        statuses['nafull'] = ['unavailable'] * n
        rec = R6F.recover_guarded(arrays, statuses, blend=R6F.VARIANTS['p3'])      # whole arm missing: must not raise
        self.assertEqual((rec['mode'], rec['fallback_studies']), ('deadline_fallback_public_semantics', n))
        self.assertEqual(rec['fallback']['partial_raptor'], n)
        pct = lambda v: pd.DataFrame(v).rank(method='average', pct=True).to_numpy()
        slot = (.52 * pct(arrays['old320']) + .28 * pct(source224_csv_values(arrays['crop224']))) / (.52 + .28)
        w = np.asarray(R6F.raptor_weight_vector(blend=R6F.VARIANTS['p3']))
        self.assertTrue(np.allclose(rec['blend'], w * slot + (1 - w) * pct(quantize9(arrays['r6'])), rtol=0, atol=1e-12))
        for a in names:
            statuses[a] = ['unavailable'] * n
        self.assertTrue(np.all(R6F.recover_guarded(arrays, statuses, blend=R6F.VARIANTS['p3'])['blend'] == .5))


def run_entry(testcase, variant, *, nafull_error=False, out=None):
    """Drive ENTRY.run in entry mode with every worker launch replaced by a recording stub (no ledgers, no GPU)."""
    cols = ['StudyInstanceUID'] + R6F.LABELS
    ids = ['s1', 's2', 's3']
    cfg = {'mode': 'entry', 'r6flip_binding': {'official_columns': cols}, 'environment': {}}
    if variant is not None:
        cfg['blend'] = copy.deepcopy(R6F.VARIANTS[variant])
        if 'nafull' in R6F.blend_arm_names(cfg['blend']):
            cfg['nafull'] = {'worker': {}, 'qualified_state': {}, 'checkpoint': {}}
    calls = []

    def launch(c, o, arm, deadline, pass_name='production', **kw):
        calls.append(arm if arm != 'r6' else 'r6')
        if arm == 'nafull' and nafull_error:
            raise (RuntimeError('config bug') if nafull_error is True else nafull_error)
        return [{'status': 'NO_RECEIPT'}] * 2, {'wall_seconds': 1., 'deadline_hit': False, 'peaks': {'host_rss_bytes': 1, 'output_disk_bytes': 1},
                                                'minimum_disk_free_bytes': 1, 'exit_codes': [1, 1]}
    written = {}
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / 'o'
        with mock.patch.object(ENTRY, 'launch_guarded', launch), \
                mock.patch.object(ENTRY, 'inventory', lambda: [{'index': 0, 'uuid': 'a', 'name': 'Tesla T4'}, {'index': 1, 'uuid': 'b', 'name': 'Tesla T4'}]), \
                mock.patch('knee.public_s6_recovery_entry.stage', lambda *a, **k: {}), \
                mock.patch('knee.public_fullfield_entry.test_contract', lambda root, labels, visible=False: (cols, list(ids), {})), \
                mock.patch('knee.public_fullfield_entry.write_csv', lambda path, columns, sids, rows: written.update(rows=np.asarray(rows)) or 'sha'):
            report = ENTRY.run(Path(d), cfg, out, started=time.monotonic())
    return report, calls, written


@unittest.skipIf(np is None or pd is None or torch is None, 'numpy/pandas/torch not installed')
class EntryStageTests(unittest.TestCase):
    def test_stage_plan_per_variant_and_default(self):
        for variant, plan in ((None, ['old', 'crop224', 'r6']), ('p3', ['old', 'crop224', 'nafull', 'r6']),
                              ('p4', ['old', 'crop224', 'nafull', 'r6']), ('p5', ['old', 'nafull', 'r6'])):
            report, calls, written = run_entry(self, variant)
            self.assertEqual(calls, plan, variant)
            self.assertIn(report['status'], ('PASS_WITH_FALLBACK',), variant)
            self.assertTrue(np.all(written['rows'] == .5))     # no ledgers at all: the constant is the last resort, never an abort
            self.assertEqual(sorted(report['workers']), sorted(plan), variant)

    def test_p5_runs_no_crop224_anywhere(self):
        report, calls, _ = run_entry(self, 'p5')
        self.assertNotIn('crop224', calls)
        self.assertNotIn('crop224', report['workers'])
        self.assertNotIn('crop224', report['phases'])
        self.assertNotIn('crop224', report['arm_merge'])
        self.assertEqual(sorted(report['arm_merge']), ['nafull', 'old320', 'r6'])

    def test_nafull_stage_failure_never_aborts_the_entry(self):
        report, calls, written = run_entry(self, 'p3', nafull_error=True)
        self.assertEqual(calls, ['old', 'crop224', 'nafull', 'r6'])
        self.assertIn('nafull', report['skipped_stages'])
        self.assertEqual(report['phases']['nafull']['stage_error'], 'RuntimeError')
        self.assertEqual(report['status'], 'PASS_WITH_FALLBACK')

    def test_platform_interruption_in_the_nafull_stage_goes_to_the_outer_salvage_not_the_stage_fallback(self):
        report, calls, written = run_entry(self, 'p3', nafull_error=InterruptedError('platform interruption'))
        self.assertEqual(calls, ['old', 'crop224', 'nafull'])          # the r6 stage is not reached
        self.assertEqual(report['failure_type'], 'InterruptedError')
        self.assertEqual(report['status'], 'EMERGENCY_CONSTANT')       # no ledgers on disk in this stub run
        self.assertNotIn('nafull', report['skipped_stages'])

    def test_arm_table_entry_salvages_a_missing_nafull_and_only_tabulates_sealed_arms(self):
        ids = ['s1', 's2']
        with tempfile.TemporaryDirectory() as d:
            workers = {'old': [{}, {}], 'crop224': [{}, {}], 'r6': [{}, {}]}      # no 'nafull' entry at all
            arrays, statuses, notes = ENTRY.arm_table(ids, workers, Path(d), entry=True, blend=R6F.VARIANTS['p3'])
            self.assertEqual(sorted(arrays), ['crop224', 'nafull', 'old320', 'r6'])
            self.assertEqual(statuses['nafull'], ['unavailable'] * 2)
            self.assertTrue(notes['nafull'].startswith('salvaged after'))
            arrays, statuses, notes = ENTRY.arm_table(ids, {'old': [{}, {}], 'r6': [{}, {}]}, Path(d), entry=True, blend=R6F.VARIANTS['p5'])
            self.assertEqual(sorted(arrays), ['nafull', 'old320', 'r6'])
            arrays, statuses, notes = ENTRY.arm_table(ids, workers, Path(d), entry=True)
            self.assertEqual(sorted(arrays), ['crop224', 'old320', 'r6'])
            with self.assertRaises(Exception):
                ENTRY.arm_table(ids, workers, Path(d), entry=False, blend=R6F.VARIANTS['p3'])      # the rehearsal stays strict

    def test_salvage_csv_with_a_missing_nafull_ledger(self):
        import csv
        from knee.public_s6_recovery_runtime import save_ledger
        self.assertIn('nafull', ENTRY.SALVAGE_FOLDERS)
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
            got = ENTRY.salvage_csv(root, d, ids, d / 'out.csv', 'RuntimeError', R6F.VARIANTS['p4'])
            self.assertEqual(got['genuine_rows'], {'old320': 3, 'crop224': 3, 'nafull': 0, 'r6': 3})
            self.assertEqual(got['fallback_studies'], 3)
            rows = list(csv.reader((d / 'out.csv').open()))
            self.assertEqual([r[0] for r in rows], ['StudyInstanceUID'] + ids)
            self.assertTrue(all(0 <= float(v) <= 1 for r in rows[1:] for v in r[1:]))
            default = ENTRY.salvage_csv(root, d, ids, d / 'default.csv', 'RuntimeError')
            self.assertEqual(sorted(default['genuine_rows']), ['crop224', 'old320', 'r6'])
            self.assertEqual(default['fallback_studies'], 0)


@unittest.skipIf(np is None or torch is None, 'numpy/torch not installed')
class ProjectionTests(unittest.TestCase):
    def workers(self, arms):
        def shard(seconds):
            return {'records': [{'production_seconds': seconds, 'status': 'genuine', 'windows': 10}] * 5, 'peak_cuda_bytes': 7}
        out = {a: [shard(2.), shard(3.)] for a in arms}
        out['r6'] = [shard(4.), shard(5.)]
        return out

    def test_projection_covers_exactly_the_stages_that_ran(self):
        windows = [10] * 870
        for arms in (('old', 'crop224'), ('old', 'crop224', 'nafull'), ('old', 'nafull')):
            phases = {a: {'wall_seconds': 100., 'peaks': {'host_rss_bytes': 1, 'output_disk_bytes': 1}, 'minimum_disk_free_bytes': 1}
                      for a in (*arms, 'r6', 'r6_sentinel1', 'r6_sentinel2')}
            p = ENTRY.projection(10., phases, self.workers(arms), 5., windows)
            self.assertEqual(sorted(p['measured_production_wall_seconds']), sorted((*arms, 'r6')))
            self.assertEqual(sorted(ENTRY.resources(phases, self.workers(arms))['peak_cuda_bytes_by_arm']), sorted((*arms, 'r6')))
        two = ENTRY.projection(10., {a: {'wall_seconds': 100., 'peaks': {'host_rss_bytes': 1, 'output_disk_bytes': 1}, 'minimum_disk_free_bytes': 1}
                                     for a in ('old', 'crop224', 'r6', 'r6_sentinel1', 'r6_sentinel2')},
                               self.workers(('old', 'crop224')), 5., windows)
        three = ENTRY.projection(10., {a: {'wall_seconds': 100., 'peaks': {'host_rss_bytes': 1, 'output_disk_bytes': 1}, 'minimum_disk_free_bytes': 1}
                                       for a in ('old', 'crop224', 'nafull', 'r6', 'r6_sentinel1', 'r6_sentinel2')},
                                 self.workers(('old', 'crop224', 'nafull')), 5., windows)
        self.assertGreater(three['conservative_seconds'], two['conservative_seconds'])


@unittest.skipIf(np is None or torch is None, 'numpy/torch not installed')
class NafullGateTests(unittest.TestCase):
    """require_nafull on a synthetic receipt; the A-triple-prime K62 checker itself is replaced by a recorder."""

    def setUp(self):
        from knee import public_nafull_aprime as NA
        self.NA = NA
        self.ids = ['id%02d' % i for i in range(39)]
        from knee.public_s6_recovery_entry import assignments
        shards = assignments(self.ids)
        self.sentinels = self.ids[:4]
        env = {'torch': '2.0', 'packages': {'numpy': '1'}, 'cuda_device_count': 1}
        self.cfg = {'staged_ids': self.ids, 'binding': {'environment': {'torch': '2.0', 'packages': {'numpy': '1'}},
                                                        's6_probe_ids': self.ids[:18], 'old_historical_ids': self.ids[18:36]},
                    'nafull': {'checkpoint': dict(NA.CHECKPOINT), 'reader_reference': {'marker': 1},
                               'qualified_state': {'old_state': NA.CHECKPOINT['state_sha256'], 'old_grid': NA.CONSTRUCTOR_GRID_SHA256},
                               'worker': {'public_manifest': {'checkpoint_sha256': NA.CHECKPOINT['sha256'], 'checkpoint_file': NA.CHECKPOINT['file'],
                                                              'weight_dataset': NA.CHECKPOINT['dataset']},
                                          'control_checkpoint': '/kaggle/input/x/' + NA.CHECKPOINT['file'],
                                          'control_state_sha256': NA.CHECKPOINT['state_sha256'],
                                          'control_grid_sha256': NA.CONSTRUCTOR_GRID_SHA256, 'old_sentinels': self.sentinels}}}
        sent = [{'id': s, 'input': {'volume_sha256': s}, 'probabilities': [.1] * 12, 'reloaded_probabilities': [.1] * 12} for s in self.sentinels]
        self.r = {'boards': [{'uuid': 'u0'}, {'uuid': 'u1'}], 'visible_ids': self.ids[36:], 'workers': {'nafull': [
            {'version': NA.VERSION, 'status': 'PASS', 'board': b, 'arm': 'old', 'assigned': shards[b], 'completed': len(shards[b]),
             'environment': {**env, 'physical_uuid': 'u%d' % b}, 'state_before': {'state': NA.CHECKPOINT['state_sha256']},
             'state_after': {'state': NA.CHECKPOINT['state_sha256']}, 'constructor_grid_sha256': NA.CONSTRUCTOR_GRID_SHA256,
             'zero_failed': True, 'zero_imputed': True, 'fp32_recoveries': 0, 'peak_cuda_bytes': 1, 'peak_rss_bytes': 1, 'ledger_sha256': 'x',
             'records': [{'id': s, 'status': 'genuine'} for s in shards[b]], 'sentinels': copy.deepcopy(sent)} for b in range(2)]}}

    def check(self):
        calls = []
        from knee import public_nafull_aprime_runtime as RT
        with mock.patch.object(RT, '_nafull_rehearsal_checks', lambda r, b: calls.append((r, b))):
            ENTRY.require_nafull(self.r, self.cfg, {})
        return calls

    def test_valid_receipt_passes_and_runs_the_aprime_k62_checks(self):
        (shim, binding), = self.check()
        self.assertEqual(set(shim['workers']), {'old'})
        self.assertEqual(len(shim['workers']['old']), 2)
        self.assertEqual(binding['nafull_reader_reference'], {'marker': 1})
        self.assertEqual(binding['s6_probe_ids'] + binding['old_historical_ids'], self.ids[:36])

    def test_each_mutation_rejects(self):
        def board(f):
            return lambda s: f(s.r['workers']['nafull'][1])
        mutations = {
            'NAFULL_EXACT_CHECKPOINT_PINS': lambda s: s.cfg['nafull']['worker']['public_manifest'].update(checkpoint_sha256='0' * 64),
            'NAFULL_TWO_PHYSICAL_T4': lambda s: s.r['workers']['nafull'][1]['environment'].update(physical_uuid='u0'),
            'NAFULL_WORKER_STATE_RESOURCE': board(lambda w: w['state_before'].update(state='0' * 64)),
            'NAFULL_WORKER_STATE_RESOURCE ': board(lambda w: w.update(fp32_recoveries=1)),
            'NAFULL_WORKER_STATE_RESOURCE  ': board(lambda w: w.update(version='r6flip')),
            'NAFULL_FROZEN39_GENUINE_COMPLETE': board(lambda w: w['records'][0].update(status='data_local_failure')),
            'NAFULL_CROSS_BOARD_SENTINEL': board(lambda w: w['sentinels'][2].update(probabilities=[.1000011] * 12)),
            'NAFULL_CROSS_BOARD_SENTINEL ': board(lambda w: w['sentinels'][0].update(input={'volume_sha256': 'other'})),
            'NAFULL_SENTINEL_IDENTITY': board(lambda w: w['sentinels'].pop()),
        }
        for message, mutate in mutations.items():
            with self.subTest(message=message):
                self.setUp()
                mutate(self)
                with self.assertRaisesRegex(ValueError, message.strip()):
                    self.check()
        self.setUp()
        self.r['workers']['nafull'][0]['records'].pop()
        self.r['workers']['nafull'][0]['assigned'].pop()
        with self.assertRaises(ValueError):
            self.check()


@unittest.skipUnless((MAIN / 'outputs/kaggle/nafull-aprime-rehearsal-r3-20261016/notebook.ipynb').is_file()
                     and (MAIN / 'outputs/kaggle/ws2-r224-v2-qual-20261015/notebook.ipynb').is_file(), 'sealed build evidence not staged')
class VariantPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import build_r6flip_entry as B
        cls.B = B
        cls.default = B.packet('rehearsal')
        cls.packs = {v: B.packet('rehearsal', variant=v) for v in R6F.VARIANTS}

    def test_default_build_has_no_variant_artifacts(self):
        cfg = self.default['cfg']
        self.assertNotIn('blend', cfg)
        self.assertNotIn('nafull', cfg)
        self.assertFalse(set(self.B.NAFULL_SOURCES) & set(self.default['sources']))
        self.assertEqual(self.B.slug_for(None, 'rehearsal'), 'r6flip-rehearsal-20261017')
        self.assertEqual(self.B.slug_for(None, 'entry'), 'r6flip-entry-20261017')
        meta = self.B.metadata(self.B.SLUGS['rehearsal'])
        self.assertEqual(meta, self.B.metadata(self.B.SLUGS['rehearsal'], None))
        self.assertEqual(meta['dataset_sources'], self.B.DATASETS)
        script = self.B.bootstrap(self.default)
        self.assertNotIn('nafull', script)
        self.assertEqual(self.B.packet('rehearsal'), self.default)
        self.assertEqual(cfg['r6flip_binding']['raptor_weight_vector'], R6F.raptor_weight_vector())

    def test_default_payload_differs_from_a_variant_only_by_variant_content(self):
        """Everything shipped for the default except the variant additions is shared by the variants."""
        for variant, pack in self.packs.items():
            self.assertEqual(pack['codecs'], self.default['codecs'])
            for name, text in self.default['sources'].items():
                self.assertEqual(pack['sources'][name], text, name)
            extra = set(pack['sources']) - set(self.default['sources'])
            self.assertEqual(extra, set(self.B.NAFULL_SOURCES))
            for key, value in self.default['cfg'].items():
                if key != 'r6flip_binding':
                    self.assertEqual(pack['cfg'][key], value, key)
            self.assertEqual(set(pack['cfg']) - set(self.default['cfg']), {'blend', 'nafull'})

    def test_variant_slugs_metadata_and_blend_are_sealed(self):
        for variant, pack in self.packs.items():
            self.assertEqual(pack['cfg']['blend'], R6F.VARIANTS[variant])
            self.assertEqual(pack['cfg']['r6flip_binding']['raptor_weight_vector'], R6F.raptor_weight_vector(blend=R6F.VARIANTS[variant]))
            self.assertEqual(pack['cfg']['r6flip_binding']['entry_variant']['name'], variant)
            slug = self.B.slug_for(variant, 'rehearsal')
            self.assertEqual(slug, 'r6flip-%s-rehearsal-20261018' % variant)
            self.assertEqual(self.B.slug_for(variant, 'entry'), 'r6flip-%s-entry-20261018' % variant)
            meta = self.B.metadata(slug, variant)
            self.assertEqual((meta['id'], meta['title']), ('ozgung/' + slug, slug))
            self.assertNotIn('timeout', meta)
            self.assertIn('ozgung/knee-raptor64-na-full-fold0-final-20261016/1', meta['dataset_sources'])
            self.assertEqual(meta['dataset_sources'][:len(self.B.DATASETS)], self.B.DATASETS)
            script = self.B.bootstrap(pack)
            self.assertIn("cfg['nafull']['worker']['control_checkpoint']=str(nafile)", script)
            self.assertIn('NAFULL_VERSION1_CHECKPOINT_BYTES', script)
            self.assertEqual(pack['cfg']['mode'], 'rehearsal')
            self.assertIn('probes', pack['cfg']['nafull']['worker'])
        self.assertEqual(self.packs['p3']['cfg']['nafull'], self.packs['p4']['cfg']['nafull'])
        self.assertNotEqual(self.packs['p3']['cfg']['blend'], self.packs['p4']['cfg']['blend'])
        self.assertTrue(self.B.same_stages('p3', 'p4') and not self.B.same_stages('p3', 'p5') and not self.B.same_stages(None, 'p3'))

    def test_na_full_sources_are_the_sealed_aprime_bytes_and_import_without_ml(self):
        for name in self.B.NAFULL_SOURCES:
            self.assertEqual(self.packs['p3']['sources'][name], (PROJECT / name).read_text())
        import ast
        for name in self.B.NAFULL_SOURCES:
            ast.parse(self.packs['p3']['sources'][name])

    @unittest.skipIf(torch is None, 'torch not installed')
    def test_unpacked_p3_na_full_worker_wiring_up_to_the_gpu_check(self):
        """Unpack the p3 payload like the bootstrap, assemble a board-1 na-full worker config with the controller, run the A-triple-prime
        entry module in a child process and require that it reaches the isolated-T4 check (a CPU host has no T4) with the config intact."""
        pack = self.packs['p3']
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            project = d / 'r6flip-project'
            for name, content in pack['sources'].items():
                p = project / name if name.startswith('knee/') else project / 'source' / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content.encode())
            cfg = copy.deepcopy(pack['cfg'])
            cfg['nafull']['worker']['control_checkpoint'] = str(d / 'raptor_ft_fold0_swa.pt')
            ids = cfg['probe_order'][:2]
            cfg.update(boards=[{'index': 0, 'uuid': 'GPU-0'}, {'index': 1, 'uuid': 'GPU-1'}], assignment=[ids[:1], ids[1:]],
                       staged=str(d / 'staged'), staged_ids=ids, stage_binding={})
            worker_cfg = ENTRY.nafull_worker_config(cfg, 1, d / 'board1-nafull')
            (d / 'board1-nafull').mkdir()
            (d / 'worker.json').write_text(json.dumps(worker_cfg))
            env = {'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(project), 'KNEE_ENTRY_PROJECT': str(project), 'CUDA_VISIBLE_DEVICES': '1',
                   'PYTHONDONTWRITEBYTECODE': '1'}
            for module in (ENTRY.NAFULL_REHEARSAL_MODULE,):
                r = subprocess.run([sys.executable, '-u', '-m', module, '--worker', str(d / 'worker.json')], cwd=d, env=env,
                                   capture_output=True, text=True)
                receipt = json.loads((d / 'board1-nafull' / 'old' / 'receipt.json').read_bytes())
                failure = json.loads((d / 'board1-nafull' / 'old' / 'failure-private.json').read_bytes())
                self.assertEqual((receipt['status'], receipt['board'], receipt['arm'], receipt['assigned']), ('FAIL', 1, 'old', ids[1:]), r.stderr[-2000:])
                self.assertEqual(failure['type'], 'ValueError', failure)
                self.assertIn('ISOLATED_T4_REQUIRED', failure['detail'])
                self.assertEqual(receipt['version'], self.NA_VERSION)

    @property
    def NA_VERSION(self):
        from knee import public_nafull_aprime as NA
        return NA.VERSION

    def test_entry_variant_strips_private_na_full_fixtures(self):
        q = {'r224_state': 'x', 'anatomy_state': 'y', 'status': 'PASS'}
        entry = self.B.packet('entry', q, 'p3')
        na = entry['cfg']['nafull']
        self.assertNotIn('reader_reference', na)
        self.assertFalse(set(ENTRY.NAFULL_WORKER_PRIVATE) & set(na['worker']))
        self.assertEqual(entry['cfg']['blend'], R6F.VARIANTS['p3'])
        self.assertEqual(entry['cfg']['r224_state_pin'], 'x')
        text = json.dumps(na)
        self.assertNotIn(self.packs['p3']['cfg']['probe_order'][0], text)
        p5 = self.B.packet('entry', {'r224_state': None, 'anatomy_state': 'y', 'status': 'PASS'}, 'p5')
        self.assertNotIn('r224_state_pin', p5['cfg'])
        self.assertNotIn('probes', p5['cfg'])

    def test_worker_config_assembles_the_na_full_profile(self):
        cfg = copy.deepcopy(self.packs['p5']['cfg'])
        cfg.update(boards=[{'uuid': 'a'}, {'uuid': 'b'}], assignment=[['i0'], ['i1']], staged='s', staged_ids=['i0', 'i1'])
        cfg['nafull']['worker']['control_checkpoint'] = '/kaggle/input/na/raptor_ft_fold0_swa.pt'
        c = ENTRY.nafull_worker_config(cfg, 1, Path('/o/board1-nafull'))
        self.assertEqual((c['arm'], c['board'], c['dev'], c['out']), ('old', 1, ['i1'], '/o/board1-nafull'))
        self.assertEqual(c['control_checkpoint'], '/kaggle/input/na/raptor_ft_fold0_swa.pt')
        self.assertEqual(c['control_state_sha256'], cfg['nafull']['qualified_state']['old_state'])
        self.assertEqual(c['binding'], {'qualified_state': cfg['nafull']['qualified_state']})
        self.assertNotIn('nafull', c)
        self.assertNotIn('blend', c)
        self.assertEqual(set(c['probes']), set(cfg['nafull']['worker']['probes']))
        self.assertTrue(all(set(p) == {'old'} for p in c['probes'].values()))
        self.assertEqual(c['old_sentinels'], cfg['nafull']['worker']['old_sentinels'])
        self.assertNotEqual(c['public_manifest'], cfg['public_manifest'])      # K62 manifest, not the R384 one


if __name__ == '__main__':
    unittest.main()
