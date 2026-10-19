"""P7-prime compact stage (knee.public_s6c, knee.public_s6c_worker) and its R6F controller/builder wiring.

Synthetic data only; no GPU, image or network use. Heavy parts skip when numpy/pandas are absent.
"""
import copy
import csv
import hashlib
import json
import os
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


def models(labels='v2', seeds=(42, 43, 44)):
    return [{'seed': s, 'labels': labels, 'file': 'E_prior_s%d_%sall.pt' % (s, labels), 'sha256': ('%02x' % s) * 32,
             'bytes': 200000000 + s, 'dataset': 'ozgung/knee-s6-%sall-s%d-20261019' % (labels, s), 'dataset_version': 1} for s in seeds]


class FrameTests(unittest.TestCase):
    def test_variant_is_p2s_raptor_half_with_a_compact_stage(self):
        self.assertEqual(R6F.COMPACT_VARIANTS, {'p7': S6C.FOLDER})
        self.assertEqual(R6F.resolve_blend(R6F.VARIANTS['p7'])[0], R6F.RAPTOR_ARMS)
        self.assertEqual(R6F.raptor_weight_vector(blend=R6F.VARIANTS['p7']), R6F.raptor_weight_vector())
        self.assertEqual(ENTRY.S6C_ARM, S6C.FOLDER)

    def test_import_is_dependency_free(self):
        code = ('import sys; sys.path.insert(0, %r); import knee.public_s6c, knee.public_s6c_worker, knee.public_r6flip_entry\n'
                'assert all(m not in sys.modules for m in ("numpy", "torch", "pandas"))\nprint("OK")') % str(PROJECT)
        r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_seed_set_rules(self):
        self.assertEqual(S6C.validate_models(models()), ['E_prior_s42_f0.pt', 'E_prior_s43_f0.pt', 'E_prior_s44_f0.pt'])
        mixed = models() + models('v3', (45,))
        for bad, message in ((mixed, 'S6C_ONE_LABEL_TABLE_NO_MIXING'), (models(seeds=(42, 42)), 'S6C_DISTINCT_SEEDS'),
                             (models(seeds=(43, 42)), 'S6C_SEEDS_IN_LOADER_ORDER'), (models(seeds=(42,)), 'S6C_TWO_TO_SIX_SEEDS'),
                             ([dict(models()[0], extra=1)] + models()[1:], 'S6C_MODEL_SCHEMA'),
                             ([dict(models()[0], dataset='heliosli/x')] + models()[1:], 'S6C_MODEL_DATASET')):
            with self.assertRaises(ValueError) as e:
                S6C.validate_models(bad)
            self.assertEqual(str(e.exception), message)
        b = S6C.block(models('v3', (42, 43)))
        self.assertEqual((b['labels'], b['aliases'], b['flip_tta']), ('v3', ['E_prior_s42_f0.pt', 'E_prior_s43_f0.pt'], False))
        self.assertEqual(S6C.dataset_sources(b), ['nartaa/rsna-knee-s6fall-weights/3', 'ozgung/knee-s6-v3all-s42-20261019/1',
                                                  'ozgung/knee-s6-v3all-s43-20261019/1'])


def ci_receipt(ids, scored, fallback, n_seeds, seconds=100.0):
    return {'status': 'PASS', 'amp': 'fp16', 'models': [S6C.alias(s) for s in (42, 43, 44)[:n_seeds]], 'spatial_models': n_seeds,
            'extent_models': 0, 'timed_out': False, 'studies': len(ids), 'scored': len(scored), 'seconds': seconds,
            'anchor_weights_sha256': S6C.ANATOMY_SHA256, 'sha256': dict(S6C.SOURCE_PINS), 'fallback': fallback,
            'per_study': [[u, 40 + i, 1., 1., 1.] for i, u in enumerate(scored)]}


@unittest.skipIf(np is None or pd is None, 'numpy/pandas not installed')
class ArrayTests(unittest.TestCase):
    def synthetic(self, ids, n_seeds=3, seed=0, fail=()):
        from knee import public_s6c_worker as W
        rng = np.random.default_rng(seed)
        z = {u: rng.normal(size=(n_seeds, 12)) for u in ids if u not in fail}
        pred = {u: [float('%.6f' % x) for x in S6C.sigmoid(v.mean(0))] for u, v in z.items()}
        fallback = {u: 'not reached before the deadline' for u in fail}
        ci = ci_receipt(ids, [u for u in ids if u not in fail], fallback, n_seeds)
        return W.summarize(ids, ci, pred, {u: v.tolist() for u, v in z.items()}, n_seeds, 20.0), z

    def test_mean_rank_is_the_mean_of_per_seed_average_pct_ranks(self):
        rng = np.random.default_rng(3)
        p = rng.random((25, 3, 12))
        p[4, 1, 2] = p[5, 1, 2]                                 # ties take the average rank
        good = np.ones(25, bool)
        good[[7, 9]] = False
        got = S6C.mean_rank(p, good)
        want = np.mean([pd.DataFrame(p[good][:, s]).rank(method='average', pct=True).to_numpy() for s in range(3)], axis=0)
        self.assertTrue(np.array_equal(got[good], want) and np.isnan(got[~good]).all())
        with self.assertRaises(ValueError):
            S6C.mean_rank(np.full((3, 2, 12), np.nan), np.ones(3, bool))

    def test_worker_summary_reconstructs_csv_from_seed_logits_and_marks_deadline_rows(self):
        ids = ['a', 'b', 'c', 'd']
        (probs, statuses, logits, records), z = self.synthetic(ids, fail=('d',))
        self.assertEqual(statuses, ['genuine'] * 3 + ['data_local_failure'])
        self.assertTrue(records[3]['deadline'])
        self.assertEqual([r['windows'] for r in records[:3]], [40, 41, 42])
        self.assertAlmostEqual(records[0]['production_seconds'], (100. - 20.) / 3)
        self.assertTrue(np.allclose(records[1]['seed_probabilities'], S6C.sigmoid(z['b'])))
        from knee import public_s6c_worker as W
        ci = ci_receipt(ids, ids, {}, 3)
        bad_pred = {u: [0.5] * 12 for u in ids}
        with self.assertRaises(ValueError) as e:
            W.summarize(ids, ci, bad_pred, {u: np.zeros((3, 12)).tolist() for u in ids[:3]}, 3, 0.)
        self.assertEqual(str(e.exception), 'S6C_DUMPED_LOGITS_COVER_SCORED')
        with self.assertRaises(ValueError) as e:
            W.summarize(ids, ci, bad_pred, {u: np.ones((3, 12)).tolist() for u in ids}, 3, 0.)
        self.assertEqual(str(e.exception), 'S6C_MEAN_LOGIT_RECONSTRUCTION')

    def write_board(self, out, board, ids, seed, fail=(), pass_name='production'):
        from knee import public_s6c_worker as W
        (probs, statuses, logits, records), z = self.synthetic(ids, seed=seed, fail=fail)
        folder = out / ('board%d-s6c-%s' % (board, pass_name)) / 's6c'
        folder.mkdir(parents=True)
        W.save_ledger(folder / 'predictions.npz', ids, probs, statuses, logits)
        state = S6C.digest([{'alias': 'x', 'sha256': 'y', 'bytes': 1}])
        return {'version': S6C.VERSION, 'status': 'PASS', 'board': board, 'arm': 's6c', 'pass': pass_name, 'assigned': ids,
                'completed': len(ids), 'state_before': state, 'state_after': state, 'ledger_sha256': W.file_sha(folder / 'predictions.npz'),
                'records': records, 'genuine': statuses.count('genuine'), 'failed': statuses.count('data_local_failure')}, z

    def test_merge_complete_population_mean_rank_and_tampering(self):
        from knee.public_s6_recovery_entry import assignments
        ids = ['s%02d' % i for i in range(10)]
        shards = assignments(ids)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            r0, z0 = self.write_board(out, 0, shards[0], 1)
            r1, z1 = self.write_board(out, 1, shards[1], 2, fail=(shards[1][-1],))
            values, mask, seed_p = S6C.merge(ids, [r0, r1], out, 3)
            z = {**z0, **z1}
            self.assertEqual(mask.tolist(), [s == shards[1][-1] for s in ids])
            good = ~mask
            want = S6C.mean_rank(np.array([S6C.sigmoid(z[s]) if s in z else np.full((3, 12), np.nan) for s in ids]), good)
            self.assertTrue(np.allclose(values[good], want[good], rtol=0, atol=1e-6) and np.isnan(values[mask]).all())
            for key, value in (('status', 'FAIL'), ('ledger_sha256', '0' * 64), ('assigned', shards[1]), ('state_after', 'x')):
                with self.assertRaises(ValueError):
                    S6C.merge(ids, [dict(r0, **{key: value}), r1], out, 3)
            with self.assertRaises(ValueError):
                S6C.merge(ids, [r0, r1], out, 2)                       # seed count is sealed

    def test_salvage_reads_ledgers_then_the_flushed_csv(self):
        from knee.public_s6_recovery_entry import assignments
        ids = ['s%02d' % i for i in range(8)]
        shards = assignments(ids)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self.write_board(out, 0, shards[0], 1)
            compact = out / 'board1-s6c-production' / 's6c' / 'compact'      # board 1 interrupted: CSV rows only
            compact.mkdir(parents=True)
            rows = [[u] + ['%.6f' % x for x in np.linspace(.1, .9, 12) * (i + 1) / 5] for i, u in enumerate(shards[1][:3])]
            with (compact / 'predictions.csv').open('w', newline='') as h:
                w = csv.writer(h)
                w.writerow(['StudyInstanceUID'] + S6C.LABELS)
                w.writerows(rows)
            values, statuses = S6C.salvage(ids, out, 3)
            self.assertEqual(statuses, ['genuine' if (s in shards[0] or s in shards[1][:3]) else 'unavailable' for s in ids])
            got = np.asarray(values)
            self.assertTrue(np.isfinite(got[[i for i, s in enumerate(statuses) if s == 'genuine']]).all())

    def test_frozen_recovery_with_the_compact_slot_and_deadline_fallback(self):
        rng = np.random.default_rng(5)
        n = 40
        arrays = {'old320': rng.random((n, 12)), 'crop224': rng.random((n, 12)), 'r6': S6C.mean_rank(rng.random((n, 3, 12)), np.ones(n, bool))}
        statuses = {a: ['genuine'] * n for a in arrays}
        p7 = R6F.recover_guarded(arrays, statuses, blend=R6F.VARIANTS['p7'])
        p2 = R6F.recover_guarded(arrays, statuses)
        self.assertEqual((p7['mode'], p7['fallback_studies']), ('aprime_complete_population', 0))
        self.assertTrue(np.array_equal(p7['blend'], p2['blend']))     # same arithmetic as P2; only the compact column differs
        statuses['r6'] = ['genuine'] * 30 + ['unavailable'] * 10
        fb = R6F.recover_guarded(arrays, statuses, blend=R6F.VARIANTS['p7'])
        self.assertEqual((fb['mode'], fb['fallback']['raptor_only']), ('deadline_fallback_public_semantics', 10))


def run_entry(testcase, s6c, *, ledgers=None):
    """ENTRY.run in entry mode with recording stub launches (the variant-test harness, with the P7-prime compact stage)."""
    cols = ['StudyInstanceUID'] + R6F.LABELS
    ids = ['s1', 's2', 's3']
    cfg = {'mode': 'entry', 'r6flip_binding': {'official_columns': cols, 'entry_variant': {'name': 'p7'}}, 'environment': {},
           'blend': copy.deepcopy(R6F.VARIANTS['p7'])}
    if s6c is not None:
        cfg['s6c'] = s6c
    calls = []

    def launch(c, o, arm, deadline, pass_name='production', **kw):
        calls.append(arm)
        return [{'status': 'NO_RECEIPT'}] * 2, {'wall_seconds': 1., 'deadline_hit': False, 'peaks': {'host_rss_bytes': 1, 'output_disk_bytes': 1},
                                                'minimum_disk_free_bytes': 1, 'exit_codes': [1, 1]}
    written = {}
    with tempfile.TemporaryDirectory() as d:
        with mock.patch.object(ENTRY, 'launch_guarded', launch), \
                mock.patch.object(ENTRY, 'inventory', lambda: [{'index': 0, 'uuid': 'a', 'name': 'Tesla T4'}, {'index': 1, 'uuid': 'b', 'name': 'Tesla T4'}]), \
                mock.patch('knee.public_s6_recovery_entry.stage', lambda *a, **k: {}), \
                mock.patch('knee.public_fullfield_entry.test_contract', lambda root, labels, visible=False: (cols, list(ids), {})), \
                mock.patch('knee.public_fullfield_entry.write_csv', lambda path, columns, sids, rows: written.update(rows=np.asarray(rows)) or 'sha'):
            report = ENTRY.run(Path(d), cfg, Path(d) / 'o', started=time.monotonic())
    return report, calls, written


@unittest.skipIf(np is None or pd is None, 'numpy/pandas not installed')
class EntryStageTests(unittest.TestCase):
    def test_p7_stage_plan_never_runs_r6_and_never_aborts(self):
        report, calls, written = run_entry(self, S6C.block(models()))
        self.assertEqual(calls, ['old', 'crop224', 's6c'])
        self.assertEqual(report['status'], 'PASS_WITH_FALLBACK')
        self.assertTrue(np.all(written['rows'] == .5))
        self.assertEqual(sorted(report['arm_merge']), ['crop224', 'old320', 'r6'])

    def test_compact_config_must_match_the_variant(self):
        report, calls, _ = run_entry(self, None)            # p7 binding without cfg['s6c']: refused before any stage
        self.assertEqual(calls, [])
        self.assertEqual(report['status'], 'EMERGENCY_CONSTANT')

    def test_salvage_csv_uses_the_compact_ledgers(self):
        from knee.public_s6_recovery_runtime import save_ledger
        ids = ['x1', 'x2', 'x3', 'x4']
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for folder, arm in (('board0-old', 'old'), ('board0-crop224', 'crop224')):
                (d / folder / arm).mkdir(parents=True)
                save_ledger(d / folder / arm / 'predictions.npz', ids, list(rng.random((4, 12))), ['genuine'] * 4)
            helper = ArrayTests()
            helper.write_board(d, 0, ids, 7)
            root = d / 'comp'
            root.mkdir()
            (root / 'sample_submission.csv').write_text('StudyInstanceUID,' + ','.join(R6F.LABELS) + '\n')
            (root / 'test.csv').write_text('StudyInstanceUID\n' + '\n'.join(ids) + '\n')
            (root / 'test_series.csv').write_text('StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n'
                                                  + ''.join('%s,t%d,1,0,Sagittal\n' % (s, i) for i, s in enumerate(ids)))
            got = ENTRY.salvage_csv(root, d, ids, d / 'out.csv', 'RuntimeError', R6F.VARIANTS['p7'], S6C.block(models()))
            self.assertEqual(got['genuine_rows'], {'old320': 4, 'crop224': 4, 'r6': 4})
            self.assertEqual(got['fallback_studies'], 0)


def valid_compact_rehearsal():
    """A synthetic rehearsal receipt that satisfies require_compact (39 = 35 + 4 sentinel ids in this fixture)."""
    from knee.public_s6_recovery_entry import assignments
    ids = ['p%02d' % i for i in range(39)]
    sentinels = ids[:4]
    ms = models()
    block = S6C.block(ms)
    staged = [{'alias': S6C.alias(m['seed']), 'sha256': m['sha256'], 'bytes': m['bytes']} for m in ms]
    env = {'torch': '2.10.0+cu128', 'packages': {'timm': '1.0.26'}, 'cuda_device_count': 1}
    seed_p = {u: [[(i * 7 + j + k) % 97 / 100. for j in range(12)] for k in range(3)] for i, u in enumerate(ids)}

    def worker(board, pass_name, assigned):
        recs = [{'id': u, 'status': 'genuine', 'windows': 50, 'production_seconds': 1., 'reconstruction_max_abs': 1e-7,
                 'seed_probabilities': seed_p[u]} for u in assigned]
        return {'version': S6C.VERSION, 'status': 'PASS', 'board': board, 'pass': pass_name, 'arm': 's6c', 'assigned': assigned,
                'completed': len(assigned), 'environment': dict(env, physical_uuid='u%d' % board), 'models': staged,
                'state_before': S6C.digest(staged), 'state_after': S6C.digest(staged), 'failed': 0, 'records': recs,
                'compact': {'status': 'PASS', 'amp': 'fp16', 'models': block['aliases'], 'spatial_models': 3, 'extent_models': 0,
                            'timed_out': False, 'anchor_weights_sha256': S6C.ANATOMY_SHA256, 'sha256': dict(S6C.SOURCE_PINS)},
                'code': {'captured_sha256': S6C.CAPTURED_SHA256, 'extras_ok': True, 'source_sha_ok': True},
                'anatomy_sha256': S6C.ANATOMY_SHA256, 'peak_cuda_bytes': 1, 'peak_rss_bytes': 1}
    shards = assignments(ids)
    r = {'boards': [{'uuid': 'u0'}, {'uuid': 'u1'}],
         'workers': {'r6': [worker(b, 'production', shards[b]) for b in range(2)],
                     'r6_sentinels': {p: [worker(b, p, sentinels) for b in range(2)] for p in ('sentinel1', 'sentinel2')}}}
    cfg = {'s6c': block, 'binding': {'environment': {'torch': env['torch'], 'packages': env['packages']}}, 's6_sentinels': sentinels,
           'staged_ids': ids}
    return r, cfg


class CompactGateTests(unittest.TestCase):
    def test_valid_rehearsal_passes_and_mutations_reject(self):
        r, cfg = valid_compact_rehearsal()
        q = S6C.require_compact(r, cfg)
        self.assertEqual((q['seeds'], q['labels'], q['anatomy_state'], q['sentinel_max_abs_dprob']), (3, 'v2', S6C.ANATOMY_SHA256, 0.0))
        def mutate(path, value):
            rr = copy.deepcopy(r)
            obj = rr
            for k in path[:-1]:
                obj = obj[k]
            obj[path[-1]] = value
            return rr
        cases = [(('workers', 'r6', 0, 'status'), 'FAIL'), (('workers', 'r6', 1, 'state_after'), 'x'),
                 (('workers', 'r6', 0, 'compact', 'amp'), 'bf16'), (('workers', 'r6', 0, 'compact', 'extent_models'), 1),
                 (('workers', 'r6', 0, 'environment', 'torch'), '2.8.0'), (('workers', 'r6', 0, 'code', 'extras_ok'), False),
                 (('workers', 'r6', 0, 'records', 0, 'status'), 'data_local_failure'),
                 (('workers', 'r6', 0, 'records', 0, 'reconstruction_max_abs'), 1e-3),
                 (('workers', 'r6_sentinels', 'sentinel2', 1, 'records', 0, 'seed_probabilities'), [[0.9] * 12] * 3)]
        for path, value in cases:
            with self.assertRaises((ValueError, KeyError, IndexError), msg=str(path)):
                S6C.require_compact(mutate(path, value), cfg)

    def test_reference_gate(self):
        r, cfg = valid_compact_rehearsal()
        sid = cfg['staged_ids'][10]
        p = [row['seed_probabilities'][1] for row in r['workers']['r6'][0]['records'] + r['workers']['r6'][1]['records'] if row['id'] == sid][0]
        cfg['s6c'] = S6C.block(models(), {'source': 'A100 fp16', 'probabilities': {'E_prior_s43_f0.pt': {sid: [x + 0.01 for x in p]}}})
        self.assertEqual(S6C.require_compact(r, cfg)['reference_studies'], 1)
        cfg['s6c'] = S6C.block(models(), {'source': 'A100 fp16', 'probabilities': {'E_prior_s43_f0.pt': {sid: [min(x + 0.05, 1.0) for x in p]}}})
        with self.assertRaises(ValueError) as e:
            S6C.require_compact(r, cfg)
        self.assertEqual(str(e.exception), 'S6C_A100_REFERENCE')


@unittest.skipIf(np is None or pd is None, 'numpy/pandas not installed')
class PacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import build_r6flip_entry as B
        cls.B = B
        cls.patch = mock.patch.object(B, 'S6C_MODELS', models())
        cls.patch.start()
        cls.pack = B.packet('rehearsal', variant='p7')

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()

    def test_cfg_binding_sources_and_no_r6_bundle(self):
        cfg, b = self.pack['cfg'], self.pack['cfg']['r6flip_binding']
        self.assertEqual(cfg['blend'], R6F.VARIANTS['p7'])
        self.assertEqual(cfg['s6c'], S6C.block(models()))
        self.assertNotIn('r6', b)
        self.assertEqual(b['s6c']['block_digest'], R6F.digest(cfg['s6c']))
        self.assertEqual(b['entry_variant']['stages'], ['old320', 'crop224', 'r6'])
        self.assertIn('heliosli R6 bundle', b['frame']['excluded'])
        for name in ('knee/public_s6c.py', 'knee/public_s6c_worker.py', S6C.CAPTURED):
            self.assertIn(name, self.pack['sources'])
        self.assertEqual(hashlib.sha256(self.pack['sources'][S6C.CAPTURED].encode()).hexdigest(), S6C.CAPTURED_SHA256)
        meta = self.B.metadata(self.B.slug_for('p7', 'rehearsal'), 'p7')
        self.assertFalse(any(d.startswith('heliosli/rsna-knee-r6') for d in meta['dataset_sources']))
        self.assertIn('nartaa/rsna-knee-s6fall-weights/3', meta['dataset_sources'])
        self.assertIn('ozgung/knee-s6-v2all-s44-20261019/1', meta['dataset_sources'])
        self.assertNotIn('timeout', meta)

    def test_bootstrap_mounts_our_seeds_not_r6(self):
        script = self.B.bootstrap(self.pack)
        compile(script, '<bootstrap>', 'exec')
        self.assertNotIn('EXACT_R6_V1_MANIFEST', script)
        self.assertIn('EXACT_S6C_SEED_BYTES', script)
        self.assertIn('r6_mount=None', script)
        self.assertIn('salvage', open(PROJECT / 'knee/public_r6flip_entry.py').read())
        default = self.B.bootstrap(self.B.packet('rehearsal'))
        self.assertIn('EXACT_R6_V1_MANIFEST', default)          # the default build is untouched

    def test_entry_packet_strips_private_fixtures(self):
        entry = self.B.packet('entry', qualified={'status': 'PASS', 'r224_state': 'x'}, variant='p7')
        cfg = entry['cfg']
        self.assertTrue(all(k not in cfg for k in self.B.PRIVATE_CFG))
        self.assertEqual(cfg['s6c'], self.pack['cfg']['s6c'])

    def test_fresh_isolated_unpacked_import_and_captured_contract(self):
        with tempfile.TemporaryDirectory() as d:
            project = Path(d) / 'r6flip-project'
            for name, content in self.pack['sources'].items():
                p = project / name if name.startswith('knee/') else project / 'source' / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content.encode())
            code = ('import sys; sys.path.insert(0, %r)\n'
                    'import knee.public_r6flip_entry as E, knee.public_s6c_worker as W, knee.public_s6c as S\n'
                    'assert W.__file__.startswith(%r) and E._s6c().__file__.startswith(%r)\n'
                    'from pathlib import Path\n'
                    'k1 = W.load_controller(Path(%r) / S.CAPTURED)\n'
                    'import types; I = types.SimpleNamespace(launcher="L", comp="D", code="C", models="M", anatomy="A")\n'
                    'cmd = k1.b_cmd(I, "train", "ids.json", "out", "tmp", 0.)\n'
                    'assert cmd[1:3] == ["-u", "L"] and cmd[cmd.index("--amp") + 1] == "fp16" and cmd[cmd.index("--eval-stride") + 1] == "1"\n'
                    'assert cmd[cmd.index("--dump-logits") + 1].endswith("logits.json") and cmd[cmd.index("--arm") + 1] == "E_prior"\n'
                    'assert k1.b_env(1)["CUDA_VISIBLE_DEVICES"] == "1"\n'
                    'print("FRESH_IMPORT_OK")\n') % (str(project), str(project), str(project), str(project))
            env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
            r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, cwd=d, env=env)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('FRESH_IMPORT_OK', r.stdout)

    def test_unsealed_models_refuse_to_build(self):
        with mock.patch.object(self.B, 'S6C_MODELS', None):
            with self.assertRaises(ValueError):
                self.B.packet('rehearsal', variant='p7')


if __name__ == '__main__':
    unittest.main()
