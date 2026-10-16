"""Synthetic tests of the S6 solo-pair runtime, rehearsal layer and builder.

The captured publisher controller is executed for real (its archived bytes are
required, so these tests skip when the ignored evidence tree is absent). No
GPU, real image, real checkpoint or network is used.
"""
import copy
import ast
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from knee import s6_solo_runtime as RT

try:
    MAIN = Path(subprocess.check_output(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                                        cwd=PROJECT, text=True).strip()).parent
except Exception:
    MAIN = PROJECT
CAPTURED = MAIN / 'outputs/research/public-0957-20261015/kernels/nartaa_rsna-knee-s6fall-solo/rsna-knee-s6fall-solo.py'
import importlib.util
HAVE_ML = all(importlib.util.find_spec(m) is not None for m in ('numpy', 'pandas'))
HAVE_CAPTURED = CAPTURED.is_file() and HAVE_ML   # the captured controller imports numpy/pandas
HAVE_EVIDENCE = HAVE_CAPTURED and (MAIN / 'outputs/analysis/s6v2-final-acceptance-20261016/acceptance.json').is_file() \
    and (MAIN / 'outputs/analysis/s6solo-pair-20261016/control-v3/code_manifest.json').is_file()
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
UID = 'StudyInstanceUID'
_N = [0]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return path


def load(kroot=None, tmp=None):
    env = {}
    if kroot is not None:
        env['K1_KAGGLE_ROOT'] = str(kroot)
    if tmp is not None:
        env['K1_TMP'] = str(tmp)
    _N[0] += 1
    with mock.patch.dict(os.environ, env):
        k1 = RT.load_controller(CAPTURED, name='k1_test_%d' % _N[0])
    k1.log = lambda *a: None        # quiet test output only; the controller's logic is untouched
    return k1


def synthetic(tmp, variant='v2'):
    """A tiny Kaggle-like input tree plus a matching cfg with synthetic pins."""
    tmp = Path(tmp)
    kroot, ainput = tmp / 'kaggle', tmp / 'ainput'
    inp = kroot / 'input'
    (kroot / 'working').mkdir(parents=True)
    comp = inp / 'rsna-knee-abnormality-detection'
    ids = ['S1', 'S2', 'S3']
    write(comp / 'test.csv', UID + '\n' + ''.join(i + '\n' for i in ids))
    write(comp / 'sample_submission.csv', ','.join([UID] + LAB) + '\n')
    write(comp / 'test_series.csv', UID + ',SeriesInstanceUID\nS1,a\nS1,b\nS2,c\nS3,d\n')
    for i in ids:
        write(comp / 'test_series' / i / 'x' / '1.dcm', b'dicom')
    write(comp / 'train_series.csv', UID + ',SeriesInstanceUID\n' + ''.join('T%d,t%d\n' % (k, k) for k in range(1, 5)))
    for k in range(1, 5):
        write(comp / 'train_series' / ('T%d' % k) / 'x' / '1.dcm', b'dicom')
    ds = inp / 'rsna-knee-s6fall-weights'
    helios = inp / 'rsna-orthofoundation-fold0-epoch7-20261004'
    heliosli = {n: ('# synthetic %s\n' % n).encode() for n in
                ('px_cache.py', 'compact_train.py', 'compact_infer.py', 'probe_infer.py', 'knee_anchor_bank_v2.py')}
    common = b'# synthetic common\n'
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:') as t:
        for n, b in heliosli.items():
            info = tarfile.TarInfo('heliosli/' + n); info.size = len(b)
            t.addfile(info, io.BytesIO(b))
    extras = b'K1PAYLOAD1\n' + buf.getvalue()
    write(ds / 's6_extras.bin', extras)
    members = {'heliosli/' + n: {'bytes': len(b), 'sha256': sha(b)} for n, b in heliosli.items()}
    members['cloud_code/training/common.py'] = {'bytes': len(common), 'sha256': sha(common)}
    manifest = json.dumps({'members': members, 'from_this_dataset': sorted('heliosli/' + n for n in heliosli)}).encode()
    write(ds / 'code_manifest.json', manifest)
    write(ds / 'checkpoint-manifest.json', b'{"sha256": "x"}')
    anatomy = write(helios / 'anatomy.pt', b'anatomy')
    write(helios / 'cloud_code/training/common.py', common)
    aw = write(ainput / 'nartaa-pub' / 'A.pt', b'A weights')
    ck = b'B checkpoint bytes'
    if variant == 'v2':
        bm = inp / 'knee-s6v2-v2all-20261016'
        source, alias = 'E_prior_s42_v2all.pt', 'E_prior_s42_fall_v2.pt'
        recs = {'run_config.json': json.dumps({'crop': 288, 'arms': 'E_prior'}).encode(),
                'SHA256SUMS': ('%s  /content/x/%s\n' % (sha(ck), source)).encode()}
        json_expect = {'run_config.json': {'crop': 288, 'arms': 'E_prior'}}
        text_expect = {'SHA256SUMS': '%s  /content/x/%s' % (sha(ck), source)}
        desc_extra = {'runtime_alias': alias}
    else:
        bm, source = ds, 'E_prior_s42_fall.pt'
        alias = source
        recs = {'E_prior_s42_fall.DONE.json': json.dumps({'model': source, 'epochs': 16}).encode()}
        json_expect = {'E_prior_s42_fall.DONE.json': {'model': source, 'epochs': 16}}
        text_expect, desc_extra = {}, {}
    write(bm / source, ck)
    for n, b in recs.items():
        write(bm / n, b)
    desc = {'variant': variant, 'checkpoint_filename': source, 'checkpoint_sha256': sha(ck), 'checkpoint_size': len(ck), **desc_extra}
    cfg = {'mode': 'entry',
           'common': {'targets': LAB, 'embedded_a_sha256': RT.EMBEDDED_A_SHA256,
                      'code_mount': {'mount': 'rsna-knee-s6fall-weights', 'files': {
                          'code_manifest.json': {'sha256': sha(manifest), 'bytes': len(manifest)},
                          's6_extras.bin': {'sha256': sha(extras), 'bytes': len(extras)}}},
                      'a_weight': {'file': 'A.pt', 'sha256': sha(b'A weights'), 'bytes': len(b'A weights')}},
           'arm': {'variant': variant, 'label': 'solo-' + variant, 'descriptor': desc,
                   'b_model': {'file': alias, 'source_file': source, 'sha256': sha(ck), 'bytes': len(ck), 'arm': 'E_prior',
                               'dataset': 'x/' + bm.name, 'dataset_version': 1, 'mount': bm.name},
                   'receipts': {n: sha(b) for n, b in recs.items()}, 'json_expect': json_expect, 'text_expect': text_expect}}
    pins = {'ANATOMY_SHA256': sha(b'anatomy'), 'SOURCE_SHA': {n: sha(b) for n, b in heliosli.items()},
            'CODE_SHA': {'common.py': sha(common)}}
    return {'kroot': kroot, 'tmp': tmp / 'k1tmp', 'ainput': ainput, 'cfg': cfg, 'pins': pins, 'bmount': bm, 'source': source,
            'alias': alias, 'ck': ck, 'ids': ids, 'anatomy': anatomy, 'aw': aw}


def prepared(s):
    k1 = load(s['kroot'], s['tmp'])
    RT.bind(k1, s['cfg'], a_root=s['ainput'])
    for k, v in s['pins'].items():
        setattr(k1, k, v)
    k1.TMP.mkdir(parents=True, exist_ok=True)
    return k1


@unittest.skipUnless(HAVE_CAPTURED, 'archived publisher solo source or numpy/pandas not present')
class ControllerBindingTests(unittest.TestCase):
    def test_exact_three_sites_and_frozen_contract(self):
        text = CAPTURED.read_text()
        func = RT.derived_prepare_inputs_source(text)
        self.assertEqual(func.count('solo_verify_inputs(I)'), 1)
        self.assertEqual(func.count('solo_stage_b_model(I)'), 1)
        self.assertNotIn('DONE marker', func)
        original = text[text.index('\ndef prepare_inputs():\n') + 1:text.index('\n# ---- component A ', text.index('\ndef prepare_inputs'))]
        kept = [l for l in original.splitlines() if l not in ''.join(o for o, _ in RT.REPLACEMENTS).splitlines()]
        self.assertEqual([l for l in func.rstrip().splitlines() if l in kept], [l for l in original.rstrip().splitlines() if l in kept])
        with self.assertRaisesRegex(ValueError, 'CAPTURED_PUBLISHER_SOURCE_SHA256'):
            RT.derived_prepare_inputs_source(text.replace('B_TIMEOUT_S = 14400', 'B_TIMEOUT_S = 14401'))

    def test_load_refuses_changed_source_and_bind_installs_arm(self):
        with tempfile.TemporaryDirectory() as d:
            bad = write(Path(d) / 'x.py', CAPTURED.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'CAPTURED_PUBLISHER_SOURCE_SHA256'):
                RT.load_controller(bad, name='k1_bad')
            s = synthetic(d)
            k1 = prepared(s)
            self.assertEqual(k1.B_MODEL['file'], 'E_prior_s42_fall_v2.pt')
            self.assertEqual(k1.prepare_inputs.__code__.co_filename, '<s6solo-prepare_inputs>')
            self.assertEqual((k1.MODE, k1.W_A, k1.W_B, k1.B_AMP), ('submit', 0.0, 1.0, 'fp16'))
            cfg = copy.deepcopy(s['cfg']); cfg['common']['embedded_a_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'EMBEDDED_A_SOURCE'):
                RT.bind(load(s['kroot'], s['tmp']), cfg)


@unittest.skipUnless(HAVE_CAPTURED, 'archived publisher solo source or numpy/pandas not present')
class PrepareInputsTests(unittest.TestCase):
    def run_prepare(self, variant='v2', mutate=None):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        s = synthetic(d.name, variant)
        if mutate:
            mutate(s)
        k1 = prepared(s)
        k1._RECEIPT['env'] = {'n_gpu': 2}
        return s, k1, k1.prepare_inputs

    def test_v2_alias_staged_alone_with_identical_bytes(self):
        s, k1, prep = self.run_prepare('v2')
        I = prep()
        self.assertEqual(sorted(p.name for p in I.models.iterdir()), ['E_prior_s42_fall_v2.pt'])
        self.assertEqual((I.models / 'E_prior_s42_fall_v2.pt').read_bytes(), s['ck'])
        self.assertFalse((I.models / 'E_prior_s42_fall_v2.pt').is_symlink())
        self.assertEqual(I.test_ids, s['ids'])
        self.assertEqual(k1._RECEIPT['solo_stage']['alias_sha256'], sha(s['ck']))
        self.assertEqual(k1._RECEIPT['inputs']['model']['variant'], 'v2')
        self.assertTrue(I.launcher.is_file())

    def test_control_reads_checkpoint_from_code_mount(self):
        s, k1, prep = self.run_prepare('control')
        I = prep()
        self.assertEqual(sorted(p.name for p in I.models.iterdir()), ['E_prior_s42_fall.pt'])
        self.assertEqual(I.bmount, I.ds)

    def test_fail_closed_on_every_bound_input(self):
        def ck(s): write(s['bmount'] / s['source'], s['ck'][:-1] + b'X')
        def receipt(s): write(s['bmount'] / 'run_config.json', json.dumps({'crop': 288, 'arms': 'E_prior', 'x': 1}))
        def expect(s):
            write(s['bmount'] / 'run_config.json', b'{"crop": 256, "arms": "E_prior"}')
            s['cfg']['arm']['receipts']['run_config.json'] = sha(b'{"crop": 256, "arms": "E_prior"}')
        def text(s): s['cfg']['arm']['text_expect']['SHA256SUMS'] = 'f' * 64 + '  /content/x/E_prior_s42_v2all.pt'
        def extras(s): s['cfg']['common']['code_mount']['files']['s6_extras.bin']['sha256'] = '0' * 64
        def aweight(s): write(s['aw'], b'other A')
        def amissing(s): s['aw'].unlink()
        def alias(s): s['cfg']['arm']['b_model']['file'] = 'E_prior_s42_fall_other.pt'
        def missing(s): (s['bmount'] / s['source']).unlink()
        def anatomy(s): write(s['anatomy'], b'other anatomy')
        cases = {'checkpoint': (ck, 'B checkpoint bytes'), 'receipt': (receipt, 'B receipt run_config.json missing'),
                 'expectation': (expect, 'differs .found, expected'), 'text': (text, 'lacks its bound line'),
                 'extras': (extras, 'publisher code mount'), 'a_weight': (aweight, 'component A weight'),
                 'a_missing': (amissing, 'component A weight'), 'alias': (alias, 'solo staging'),
                 'missing_checkpoint': (missing, 'cannot resolve'), 'anatomy': (anatomy, 'anatomy.pt differs')}
        for name, (mutate, message) in cases.items():
            with self.subTest(name=name):
                s, k1, prep = self.run_prepare('v2', mutate)
                with self.assertRaisesRegex(k1.K1Error, message):
                    prep()


class FakeWorkers:
    """Stands in for compact_infer processes: writes receipts/logits/predictions per spec, per a scripted plan."""

    def __init__(self, k1, plan):
        self.k1, self.plan, self.calls = k1, plan, 0

    def __call__(self, specs, timeout_s, what):
        k1, step = self.k1, self.plan[min(self.calls, len(self.plan) - 1)]
        self.calls += 1
        out = []
        for s in specs:
            cmd = s['cmd']
            ids = json.loads(Path(cmd[cmd.index('--assignment') + 1]).read_text())
            pdir = Path(cmd[cmd.index('--output') + 1])
            fail = {u: step.get('reason', 'loader: synthetic') for u in ids if u in step.get('fail', ())}
            if step.get('deadline'):
                fail.update({u: 'not reached before the deadline' for u in ids[-1:]})
            ok = [u for u in ids if u not in fail]
            rec = {'status': 'PASS', 'studies': len(ids), 'scored': len(ok), 'fallback': fail, 'timed_out': bool(step.get('deadline')),
                   'amp': step.get('amp', 'fp16'), 'models': [step.get('model', k1.B_MODEL['file'])], 'spatial_models': 1,
                   'anchor_weights_sha256': k1.ANATOMY_SHA256, 'anchor_methods': {'skm': len(ok)},
                   'sha256': {n: k1.SOURCE_SHA[n] for n in k1.B_RECEIPT_SOURCES}, 'seconds': 1.0, 'stats': {'windows': len(ok)},
                   'per_study': [[u, 1, 0.1, 0.1, 0.1] for u in ok]}
            logits = {u: [[(hash(u) % 97) / 50.0 - 1 + j / 100.0 for j in range(12)]] for u in ok}
            if step.get('extra'):
                logits['OUTSIDE'] = [[0.0] * 12]
            (pdir / 'receipt.json').write_text(json.dumps(rec))
            (pdir / 'logits.json').write_text(json.dumps(logits))
            with open(pdir / 'predictions.csv', 'w', newline='') as f:
                w = csv.writer(f); w.writerow([UID] + LAB)
                for u, z in logits.items():
                    w.writerow([u] + ['%.6f' % (1 / (1 + math.exp(-x))) for x in z[0]])
            prog = [json.dumps({'done': k, 'of': len(ids), 'elapsed_s': 0.05 + 0.1 * (k - 1)}) for k in range(1, len(ids) + 1)]
            (pdir / 'log.txt').write_text('\n'.join(prog) + '\nWORKER_DONE\n')
            if s['env'].get('S6SOLO_MODE') == 'rehearsal':
                idx = int(s['env'].get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
                q = {'source': 'runtime_observer',
                     'inference_source_sha256': k1._s6_expected_models['b']['inference_source_sha256'],
                     'boards': [{'gpu_uuid': 'GPU-%d' % idx,
                                 'b': synthetic_model_record(k1._s6_expected_models['b'])}],
                     'overhead_seconds': 0.01, 'overhead_scope': 'synthetic test fixture only'}
                write(pdir / 'model-qualification.json', json.dumps(q))
            px = s['env'].get('K1_PX_DUMP')
            if px:
                with open(Path(px) / 'px_1.jsonl', 'a') as f:
                    for u in ok:
                        f.write(json.dumps({'uid': u, 'error': None, 'n': 1, 'img_sha256': sha(u.encode()),
                                            'meta_sha256': sha(b'm' + u.encode()), 'anchor': {'method': 'skm'}}) + '\n')
            out.append((s['name'], 0, 1.0, ''))
        time.sleep(step.get('sleep', 0))
        return out


@unittest.skipUnless(HAVE_CAPTURED, 'archived publisher solo source or numpy/pandas not present')
class CapturedRecoveryTests(unittest.TestCase):
    """Fault injection through the captured run_b/collect_part/blend, not re-implementations."""

    def controller(self, plan):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        s = synthetic(d.name)
        k1 = RT.bind(load(s['kroot'], s['tmp']), s['cfg'])     # captured pins unchanged
        k1.OUT.mkdir(parents=True, exist_ok=True)
        k1._RECEIPT['env'] = {'n_gpu': 2}
        k1.run_group = FakeWorkers(k1, plan)
        I = types.SimpleNamespace(launcher=Path(d.name, 'l.py'), comp=Path(d.name), code=Path(d.name), models=Path(d.name),
                                  anatomy=Path(d.name, 'a.pt'), cost={})
        return k1, I

    def ids(self, n):
        return ['U%04d' % i for i in range(n)]

    def test_healthy_and_retry_recovery(self):
        k1, I = self.controller([{}])
        P, Z, info, failed = k1.run_b(I, 'B', self.ids(39), 'test', 10 ** 6)
        self.assertEqual(failed, {}); self.assertIsNone(info['retry']); self.assertEqual(info['fail_cap'], 0)
        k1, I = self.controller([{'fail': {'U0003'}}, {}])
        P, Z, info, failed = k1.run_b(I, 'B', self.ids(39), 'test', 10 ** 6)
        self.assertEqual(failed, {}); self.assertEqual(info['retry']['ids'], ['U0003'])
        self.assertTrue(math.isfinite(P.sum()))

    def test_caps_zero_below_100_and_rounded_quarter_percent_at_1300(self):
        k1, I = self.controller([{'fail': {'U0003'}}])
        with self.assertRaisesRegex(k1.K1Error, 'allowed 0'):
            k1.run_b(I, 'B', self.ids(39), 'test', 10 ** 6)
        ids = self.ids(1300)
        k1, I = self.controller([{'fail': set(ids[:4])}])
        P, Z, info, failed = k1.run_b(I, 'B', ids, 'test', 10 ** 6)
        self.assertEqual(sorted(failed), ids[:4]); self.assertEqual(info['fail_cap'], 4)
        k1, I = self.controller([{'fail': set(ids[:5])}])
        with self.assertRaisesRegex(k1.K1Error, 'allowed 4'):
            k1.run_b(I, 'B', ids, 'test', 10 ** 6)

    def test_first_and_retry_deadlines_are_hard_errors(self):
        k1, I = self.controller([{'deadline': True}])
        with self.assertRaisesRegex(k1.K1Error, 'reached its deadline'):
            k1.run_b(I, 'B', self.ids(1300), 'test', 10 ** 6)
        k1, I = self.controller([{'fail': {'U0001'}}, {'deadline': True}])
        with self.assertRaisesRegex(k1.K1Error, 'deadline in the retry'):
            k1.run_b(I, 'B', self.ids(1300), 'test', 10 ** 6)
        k1, I = self.controller([{'fail': {'U0001'}, 'reason': 'not reached before the deadline'}])
        with self.assertRaisesRegex(k1.K1Error, 'reached its deadline'):
            k1.run_b(I, 'B', self.ids(1300), 'test', 10 ** 6)

    def test_wrong_checkpoint_precision_or_assignment_receipts_fail(self):
        for step, message in (({'model': 'E_prior_s42_fall.pt'}, 'receipt amp'), ({'amp': 'bf16'}, 'receipt amp'),
                              ({'extra': True}, 'outside the assignment')):
            with self.subTest(step=step):
                k1, I = self.controller([step])
                with self.assertRaisesRegex(k1.K1Error, message):
                    k1.run_b(I, 'B', self.ids(39), 'test', 10 ** 6)

    def test_full_population_recovery_and_failure_deletion(self):
        import numpy as np
        k1, I = self.controller([{}])
        work = Path(k1.OUT).parent
        k1.WORK = work
        rng = np.random.default_rng(0)
        n = 200
        ids = self.ids(n)
        A, B = rng.random((n, 12)), rng.random((n, 12))
        B = 0.7 * A + 0.3 * B                      # correlated enough for the inherited rho floor
        B[5] = np.nan
        I.test_ids, I.sub_cols = ids, [UID] + LAB
        info = k1.blend_and_write(I, A, B, {'U0005': 'synthetic'})
        rows = list(csv.reader(open(work / 'submission.csv')))
        rA = k1.rank01(A)
        Bf = B.copy(); Bf[5] = np.nanmean(np.delete(B, 5, 0), axis=0)
        expect = k1.rank01(Bf); expect[5] = rA[5]
        got = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
        self.assertTrue(np.array_equal(got.astype(np.float32), expect.astype(np.float32)))
        self.assertEqual(info['b_failed_rank_A_only'], {'U0005': 'synthetic'})
        with self.assertRaisesRegex(k1.K1Error, 'sanity floor'):
            k1.blend_and_write(I, A, rng.random((n, 12)), None)
        k1.remove_submission()
        self.assertFalse((work / 'submission.csv').exists())


def real_input(tmp, ids, splits):
    inp = Path(tmp) / 'input'
    comp = inp / 'competitions' / 'rsna-knee-abnormality-detection'
    write(comp / 'test.csv', UID + '\n')
    write(comp / 'sample_submission.csv', ','.join([UID] + LAB) + '\n')
    for split in ('train', 'test'):
        rows = [(i, 'ser' + i) for i, s in zip(ids, splits) if s == split] + [('OTHER' + split, 'x')]
        write(comp / (split + '_series.csv'), UID + ',SeriesInstanceUID\n' + ''.join('%s,%s\n' % r for r in rows))
        for i, s in zip(ids, splits):
            if s == split:
                write(comp / (split + '_series') / i / ('ser' + i) / '1.dcm', b'dicom')
    write(inp / 'datasets' / 'nartaa' / 'rsna-knee-s6fall-weights' / 'checkpoint-manifest.json', b'{}')
    write(inp / 'datasets' / 'heliosli' / 'rsna-orthofoundation-fold0-epoch7-20261004' / 'anatomy.pt', b'x')
    write(inp / 'datasets' / 'ozgung' / 'knee-s6v2-v2all-20261016' / 'E_prior_s42_v2all.pt', b'x')
    return inp


def rehearsal_cfg(ids, splits):
    return {'mode': 'rehearsal', 'common': {'targets': LAB, 'embedded_a_sha256': RT.EMBEDDED_A_SHA256,
                       'a_weight': {'sha256': 'a' * 64},
                       'model_sources': {'compact_train.py': {'sha256': 'c' * 64}, 'compact_infer.py': {'sha256': 'd' * 64}}},
            'arm': {'variant': 'v2', 'cpu': {'state_sha256': 'e' * 64, 'constructor_buffers_sha256': 'f' * 64,
                                             'constructor': {}, 'buffer_inventory': {}},
                   'b_model': {'mount': 'knee-s6v2-v2all-20261016', 'source_file': 'E_prior_s42_v2all.pt',
                               'sha256': 'b' * 64}},
            'rehearsal': {'fixture': {'ids': ids, 'splits': splits},
                          'mounts': [['rsna-knee-s6fall-weights', 'checkpoint-manifest.json'],
                                     ['rsna-orthofoundation-fold0-epoch7-20261004', 'anatomy.pt']],
                          'sentinel_per_board': 2, 'sentinel_probability_atol': 1e-3, 'projection_studies': 1300,
                          'gate_seconds': 29880, 'post_controller_allowance_seconds': 60}}


class FixtureTests(unittest.TestCase):
    def test_fixture_lists_frozen_ids_as_test_with_per_study_links(self):
        from knee import s6_solo_rehearsal as R
        ids, splits = ['T1', 'T2', 'V1'], ['train', 'train', 'test']
        with tempfile.TemporaryDirectory() as d:
            inp = real_input(d, ids, splits)
            cfg = rehearsal_cfg(ids, splits)
            fx = R.build_fixture(cfg, inp, Path(d) / 'fx', Path(d) / 'work')
            comp = Path(fx['comp'])
            self.assertEqual(list(csv.reader(open(comp / 'test.csv'))), [[UID], ['T1'], ['T2'], ['V1']])
            series = list(csv.reader(open(comp / 'test_series.csv')))
            self.assertEqual([r[0] for r in series[1:]], ids)
            self.assertEqual(len(list(csv.reader(open(comp / 'sample_submission.csv')))), 4)
            for i, s in zip(ids, splits):
                link = comp / 'test_series' / i
                self.assertTrue(link.is_symlink() and (link / ('ser' + i) / '1.dcm').is_file())
                self.assertIn(s + '_series', os.readlink(link))
            self.assertEqual(sorted(fx['mounts']), ['knee-s6v2-v2all-20261016', 'rsna-knee-s6fall-weights',
                                                    'rsna-orthofoundation-fold0-epoch7-20261004'])
            self.assertEqual(os.readlink(Path(d) / 'fx' / 'working'), str(Path(d) / 'work'))
            with self.assertRaisesRegex(ValueError, 'FRESH_FIXTURE_ROOT'):
                R.build_fixture(cfg, inp, Path(d) / 'fx', Path(d) / 'work')
            bad = rehearsal_cfg(['T1', 'MISSING'], ['train', 'train'])
            with self.assertRaisesRegex(ValueError, 'FIXTURE_STUDY_SERIES_PRESENT'):
                R.build_fixture(bad, inp, Path(d) / 'fx2', Path(d) / 'work')

    @unittest.skipUnless(HAVE_CAPTURED, 'archived publisher solo source or numpy/pandas not present')
    def test_controller_resolves_fixture_through_its_own_hook(self):
        from knee import s6_solo_rehearsal as R
        ids, splits = ['T1', 'V1'], ['train', 'test']
        with tempfile.TemporaryDirectory() as d:
            cfg = rehearsal_cfg(ids, splits)
            fx = R.build_fixture(cfg, real_input(d, ids, splits), Path(d) / 'fx', Path(d) / 'work')
            k1 = load(fx['root'])
            self.assertEqual(str(k1.find_mount(k1.COMP, 'test.csv')), fx['comp'])
            self.assertTrue(str(k1.find_mount(k1.DS_SLUG, 'checkpoint-manifest.json')).startswith(fx['root']))
            self.assertEqual(k1.WORK.resolve(), (Path(d) / 'work').resolve())


class ARunnerTests(unittest.TestCase):
    def test_exact_bytes_fixture_root_and_fork_pool(self):
        from knee import s6_solo_a_runner as A
        fake = ('import multiprocessing as mp, sys\n'
                'def find_test_root():\n    return "/kaggle/input/rsna-knee-abnormality-detection"\n'
                'def _one(x):\n    return x * 2\n'
                'def main():\n    root = find_test_root()\n'
                '    with mp.get_context("fork").Pool(2) as p:\n        r = sorted(p.imap_unordered(_one, [1, 2, 3]))\n'
                '    print("ROOT", root, r, flush=True)\n'
                'if __name__ == "__main__":\n    raise SystemExit("guard must not fire")\n')
        with tempfile.TemporaryDirectory() as d:
            run = write(Path(d) / 'run.py', fake)
            root = Path(d) / 'fx'
            write(root / 'test.csv', 'x'); write(root / 'test_series.csv', 'x')
            code = 'import sys; sys.path.insert(0, %r); from knee import s6_solo_a_runner as A; A.run(%r, %r, %r)' % (
                str(PROJECT), str(run), str(root), sha(fake.encode()))
            r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('ROOT %s [2, 4, 6]' % root, r.stdout)
            with self.assertRaises(SystemExit):
                A.run(run, root)                    # the real A hash is required by default

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None
                         and (MAIN / 'outputs/analysis/s6solo-pair-20261016/embedded-a.py').is_file(),
                         'torch or embedded A reference source is absent')
    def test_a_observer_preserves_forward_output_on_synthetic_model(self):
        import torch
        from knee import s6_solo_a_runner as A
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([2.0]))
                self.register_buffer('mean', torch.tensor([0.0]))
        source = MAIN / 'outputs/analysis/s6solo-pair-20261016/embedded-a.py'
        m = types.SimpleNamespace(__file__=str(source), ARMS=[{'arch': 'fixture'}])
        def load_model(*args, **kwargs): return Model(), 320
        def infer_pair_probs(model, plain, mirrored, device):
            return (plain.sum() + mirrored.sum()) * model.weight
        m.load_model, m.infer_pair_probs = load_model, infer_pair_probs
        with tempfile.TemporaryDirectory() as d:
            ck = write(Path(d) / 'weights.pt', b'synthetic checkpoint')
            result_path = Path(d) / 'model-qualification.json'
            finalize = A._install_model_observer(m, result_path)
            model, _ = m.load_model(ck, 'fixture', 320, 'cuda:0')
            plain, mirrored = torch.ones(2), torch.full((2,), 3.0)
            with mock.patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '0, GPU-A\n', '')):
                observed = m.infer_pair_probs(model, plain, mirrored, 'cuda:0')
                self.assertFalse(result_path.exists())
                finalize()
            expected = infer_pair_probs(model, plain, mirrored, 'cuda:0')
            receipt = json.loads(result_path.read_text())
            self.assertTrue(torch.equal(observed, expected))
            self.assertEqual(receipt['source'], 'runtime_observer')
            self.assertEqual(receipt['boards'][0]['gpu_uuid'], 'GPU-A')
            self.assertTrue(receipt['boards'][0]['a']['strict_reload'])
            self.assertTrue(receipt['boards'][0]['a']['strict_loaded_before'])


def synthetic_model_record(expected):
    state, buffers = expected['state_sha256'], expected['buffers_sha256']
    model = {'checkpoint_sha256': expected['checkpoint_sha256'], 'constructor_sha256': '1' * 64,
             'constructor_source_sha256': expected['constructor_source_sha256'],
             'state_before_sha256': state, 'state_after_sha256': state, 'reload_state_sha256': state,
             'buffers_before_sha256': buffers, 'buffers_after_sha256': buffers, 'reload_buffers_sha256': buffers,
             'reload_constructor_sha256': '1' * 64, 'input_sha256': '2' * 64, 'reload_input_sha256': '2' * 64,
             'strict_loaded_before': True, 'strict_reload': True, 'sentinel_same_input': True,
             'sentinel_max_abs': 0.0, 'state_unchanged_from_load': True, 'buffers_unchanged_from_load': True,
             'constructor_spec': expected.get('constructor_spec'), 'buffer_inventory': expected.get('buffer_inventory')}
    if expected.get('grid_sha256'):
        model.update(grid_before_sha256=expected['grid_sha256'], grid_after_sha256=expected['grid_sha256'],
                     grid_unchanged_from_load=True)
    return model


def synthetic_model_receipts(cfg):
    from knee import s6_solo_rehearsal as R
    expected = R._expected_model_qualification(cfg)
    ab, bb = [], []
    for uuid in ('GPU-0', 'GPU-1'):
        ab.append({'gpu_uuid': uuid, 'a': synthetic_model_record(expected['a'])})
        bb.append({'gpu_uuid': uuid, 'b': synthetic_model_record(expected['b'])})
    return ({'source': 'runtime_observer', 'component_a_source_sha256': expected['a']['component_source_sha256'],
             'boards': ab, 'overhead_seconds': 0.01, 'overhead_scope': 'synthetic test fixture only'},
            {'source': 'runtime_observer', 'inference_source_sha256': expected['b']['inference_source_sha256'],
             'boards': bb, 'overhead_seconds': 0.01, 'overhead_scope': 'synthetic test fixture only'})


def synthetic_result(tmp, cfg, k1, slow=False, payload='p' * 64, board_names=('Tesla T4', 'Tesla T4'), elapsed=712.0):
    """Write a complete /kaggle/working-like result tree the way the runtime would, then assemble it."""
    import numpy as np
    from knee import s6_solo_rehearsal as R
    root = Path(tmp) / 'working'
    k1d, s = root / 'k1', root / 's6solo'
    ids = cfg['rehearsal']['fixture']['ids']
    n = len(ids)
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(n, 12))
    B = 1 / (1 + np.exp(-Z))
    A = k1.rank01(B + rng.normal(scale=0.3, size=B.shape)).astype(np.float32).astype(np.float64)
    write(k1d / 'A' / 'submission_A.csv', ','.join([UID] + LAB) + '\n' + ''.join(
        ','.join([u] + [repr(float(np.float32(x))) for x in A[i]]) + '\n' for i, u in enumerate(ids)))
    scale = 400 if slow else 1
    write(k1d / 'A' / 'log.txt', '[gpu cuda:0] loaded 1 arm(s) | 40s\n[gpu cuda:1] loaded 1 arm(s) | 42s\n'
          '[gpu] done: %d/%d studies on 2 device(s) | %ds\nDONE %ds\n' % (n, n, 42 + 60 * scale, 50 + 60 * scale))
    with open(write(k1d / 'B' / 'B_probs.csv', b''), 'w', newline='') as f:
        w = csv.writer(f); w.writerow([UID] + LAB + ['z|' + l for l in LAB])
        for i, u in enumerate(ids):
            w.writerow([u] + ['%.17g' % x for x in B[i]] + ['%.17g' % x for x in Z[i]])
    parts = [ids[0::2], ids[1::2]]
    px = {}
    for gi, part in enumerate(parts):
        p = k1d / 'B' / ('part%d' % gi)
        write(p / 'ids.json', json.dumps(part))
        write(p / 'receipt.json', json.dumps({'seconds': 100.0 * scale, 'stats': {'windows': 10 * len(part)},
                                              'per_study': [[u, 10, 0.5, 0.5, 0.2] for u in part]}))
        lines = [json.dumps({'done': k, 'of': len(part), 'elapsed_s': 30.0 + (k - 1) * 3.0 * scale}) for k in (1, 2, 3, len(part))]
        write(p / 'log.txt', '\n'.join(lines) + '\nWORKER_DONE\n')
        for u in part:
            px[u] = {'uid': u, 'error': None, 'img_sha256': sha(u.encode()), 'meta_sha256': sha(b'm' + u.encode()), 'n': 10, 'anchor': None}
    write(k1d / 'B' / 'px_hashes.json', json.dumps(px))
    aq, bq = synthetic_model_receipts(cfg)
    write(k1d / 'A' / 'model-qualification.json', json.dumps(aq))
    write(k1d / 'B' / 'model-qualification.json', json.dumps(bq))
    for rerun, orig in ((1, 0), (0, 1)):
        sd = k1d / 'sentinel' / ('board%d' % rerun)
        sids = parts[orig][:2]
        write(sd / 'ids.json', json.dumps(sids))
        write(sd / 'logits.json', json.dumps({u: [Z[ids.index(u)].tolist()] for u in sids}))
        write(sd / 'px' / 'px_1.jsonl', ''.join(json.dumps(px[u]) + '\n' for u in sids))
    final = k1.blend_with_failures(A, B, np.zeros(n, bool)).astype(np.float32)
    write(s / 'rehearsal-final.csv', ','.join([UID] + LAB) + '\n' + ''.join(
        ','.join([u] + [repr(float(x)) for x in final[i]]) + '\n' for i, u in enumerate(ids)))
    stages = {'env': {'start_s': 1.0, 'seconds': 5.0}, 'inputs': {'start_s': 6.0, 'seconds': 20.0},
              'B_probe': {'start_s': 26.0, 'seconds': 30.0}, 'A': {'start_s': 56.0, 'seconds': 60.0 + 60.0 * scale},
              'B': {'start_s': 200.0, 'seconds': 150.0 * scale}, 'blend': {'start_s': 400.0, 'seconds': 1.0}}
    kr = {'env': {'n_gpu': 2, 'gpus': ['Tesla T4', 'Tesla T4']}, 'stages': stages, 'total_seconds': sum(v['seconds'] for v in stages.values()) + 3,
          'inputs': {'sub_cols': [UID] + LAB, 'model': {'sha256': cfg['arm']['b_model']['sha256']}},
          'solo': {'binding_sha256': RT.binding_digest(cfg), 'payload_sha256': payload},
          'solo_stage': {'alias_sha256': cfg['arm']['b_model']['sha256']},
          'A': {'internal_fallbacks': [], 'attempts': [{'attempt': 1}]},
          'B': {'failed_after_retry': {}, 'retry': None, 'fail_cap': 0}, 'blend': {'b_failed_rank_A_only': {}, 'rows': n}}
    write(k1d / 'k1_receipt.json', json.dumps(kr))
    t0 = 1000.0
    resources = {'interval_seconds': 10, 'boards': [{'index': i, 'uuid': 'GPU-%d' % i, 'name': board_names[i], 'total_mib': 15360}
                                                    for i in range(2)],
                 'samples': [{'t': t0 + k * 10, 'host_used_bytes': 5e9 + k, 'host_total_bytes': 31e9, 'gpu_used_mib': [1000, 900],
                              'disk_free_bytes': {'/kaggle/working': 10e9, '/tmp': 50e9}} for k in range(60)]}
    timing = {'notebook_started': t0 - 12, 'fixture_started': t0 - 8, 'fixture_seconds': 2.0, 'controller_t0': t0,
              'controller_end': t0 + 500, 'finished': t0 - 12 + elapsed}
    sentinel_run = {'seconds': 90.0, 'runs': [['sentinel.board1', 0, 80.0, ''], ['sentinel.board0', 0, 85.0, '']]}
    receipt, ledger = R.assemble(k1, cfg, k1d, s / 'rehearsal-final.csv', timing, sentinel_run, resources)
    RT.save(s / 'resources.json', resources)
    RT.save(s / 'ledger.json', ledger)
    receipt.update(ledger_sha256=RT.file_sha(s / 'ledger.json'), resources_sha256=RT.file_sha(s / 'resources.json'),
                   final_csv_sha256=RT.file_sha(s / 'rehearsal-final.csv'))
    RT.save(s / 'receipt.json', receipt)
    return root, receipt


@unittest.skipUnless(HAVE_CAPTURED, 'archived publisher solo source or numpy/pandas not present')
class AssembleVerifyTests(unittest.TestCase):
    def cfg(self):
        ids = ['R%02d' % i for i in range(39)]
        cfg = rehearsal_cfg(ids, ['train'] * 36 + ['test'] * 3)
        cfg['rehearsal'].update(sentinel_timeout_s=1800)
        return cfg

    def test_pass_replays_and_tamper_fails(self):
        from knee import s6_solo_rehearsal as R
        k1 = load()
        with tempfile.TemporaryDirectory() as d:
            cfg = self.cfg()
            root, receipt = synthetic_result(d, cfg, k1)
            self.assertEqual(receipt['status'], 'PASS', receipt['failed_checks'])
            p = receipt['projection']
            self.assertTrue(p['passes'] and 0 < p['conservative_seconds'] < 29880)
            # A: 60 s fixed, 60 s production over 39 -> 1300 at 1.25 x 1.15
            self.assertAlmostEqual(p['conservative']['A_seconds'], 60 + 60 / 39 * 1300 * 1.25 * 1.15, places=6)
            self.assertEqual(R.verify_result(root, cfg, k1)['ledger_sha256'], receipt['ledger_sha256'])
            led = json.loads((root / 's6solo' / 'ledger.json').read_text())
            led['rows'][0]['b_probability'][0] += 1e-9
            RT.save(root / 's6solo' / 'ledger.json', led)
            with self.assertRaisesRegex(ValueError, 'SAVED_FILE_BINDINGS'):
                R.verify_result(root, cfg, k1)
            other = copy.deepcopy(cfg); other['arm']['b_model']['sha256'] = 'c' * 64
            with self.assertRaisesRegex(ValueError, 'PASSING_BOUND_REHEARSAL'):
                R.verify_result(root, other, k1)

    def test_slow_projection_one_board_and_changed_final_fail(self):
        k1 = load()
        with tempfile.TemporaryDirectory() as d:
            _, slow = synthetic_result(Path(d) / 'a', self.cfg(), k1, slow=True)
            self.assertEqual(slow['status'], 'FAIL')
            self.assertIn('projection_8_3h_and_inherited_guards', slow['failed_checks'])
            _, odd = synthetic_result(Path(d) / 'b', self.cfg(), k1, board_names=('Tesla T4', 'Tesla P100'))
            self.assertIn('two_physical_t4', odd['failed_checks'])

    def test_actual_elapsed_requires_finite_strictly_under_nine_hours(self):
        k1 = load()
        with tempfile.TemporaryDirectory() as d:
            _, under = synthetic_result(Path(d) / 'under', self.cfg(), k1, elapsed=32399.0)
            _, exact = synthetic_result(Path(d) / 'exact', self.cfg(), k1, elapsed=32400.0)
            _, over = synthetic_result(Path(d) / 'over', self.cfg(), k1, elapsed=32401.0)
            self.assertNotIn('actual_notebook_elapsed_under_9h', under['failed_checks'])
            self.assertIn('actual_notebook_elapsed_under_9h', exact['failed_checks'])
            self.assertIn('actual_notebook_elapsed_under_9h', over['failed_checks'])
            self.assertEqual(exact['elapsed_seconds'], 32400.0)
            self.assertEqual(over['elapsed_seconds'], 32401.0)
            root, _ = synthetic_result(Path(d) / 'nan', self.cfg(), k1)
            k1d = root / 'k1'
            resources = json.loads((root / 's6solo' / 'resources.json').read_text())
            timing = json.loads((root / 's6solo' / 'receipt.json').read_text())['timing']
            timing['finished'] = float('nan')
            receipt, _ = __import__('knee.s6_solo_rehearsal', fromlist=['assemble']).assemble(
                k1, self.cfg(), k1d, root / 's6solo' / 'rehearsal-final.csv', timing,
                json.loads((root / 's6solo' / 'receipt.json').read_text())['sentinel']['run'], resources)
            self.assertIn('actual_notebook_elapsed_under_9h', receipt['failed_checks'])

    def test_collector_rejects_duplicate_or_unmatched_physical_gpu_receipts(self):
        from knee import s6_solo_rehearsal as R
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = {'source': 'runtime_observer', 'overhead_seconds': 0.0, 'boards': [
                {'gpu_uuid': 'GPU-A', 'a': {}}, {'gpu_uuid': 'GPU-A', 'a': {}}]}
            b = {'source': 'runtime_observer', 'overhead_seconds': 0.0, 'inference_source_sha256': 'f' * 64, 'boards': [
                {'gpu_uuid': 'GPU-A', 'b': {}}, {'gpu_uuid': 'GPU-B', 'b': {}}]}
            write(root / 'A' / 'model-qualification.json', json.dumps(a))
            write(root / 'B' / 'model-qualification.json', json.dumps(b))
            self.assertIsNone(R._collect_model_qualification(root))

    def test_model_qualification_rejects_missing_or_wrong_board_and_reload(self):
        from knee.s6_solo_rehearsal import _valid_model_receipt
        boards = [{'uuid': 'GPU-A'}, {'uuid': 'GPU-B'}]
        expected = {'a': {'checkpoint_sha256': 'a' * 64}, 'b': {'checkpoint_sha256': 'b' * 64}}
        self.assertFalse(_valid_model_receipt(None, boards, expected))
        synthetic = {'source': 'synthetic_fixture', 'boards': [], 'overhead_seconds': 0}
        self.assertFalse(_valid_model_receipt(synthetic, boards, expected))


class FakeGroups:
    """The controller's subprocess layer for a whole rehearsal: A writes its CSV/log, B and sentinels via FakeWorkers."""

    def __init__(self, k1, runner, comp):
        self.k1, self.runner, self.comp, self.calls = k1, runner, comp, []
        self.b = FakeWorkers(k1, [{'sleep': 0.6}])

    def __call__(self, specs, timeout_s, what):
        self.calls.append((what, [s['cmd'] for s in specs]))
        if what != 'A':
            return self.b(specs, timeout_s, what)
        cmd = specs[0]['cmd']
        if self.runner is None:       # entry: the captured `python -u run.py`
            assert len(cmd) == 3 and cmd[2].endswith('/A/run.py'), cmd
        else:
            assert cmd[2] == self.runner and cmd[4] == self.comp and cmd[3].endswith('/A/run.py'), cmd
        with open(Path(self.comp) / 'test.csv') as f:
            ids = [r[UID] for r in csv.DictReader(f)]
        with open(self.k1.WORK / 'submission.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow([UID] + LAB)
            for k, u in enumerate(ids):
                w.writerow([u] + ['%.6f' % ((k + 1 + j) % (len(ids) + 1) / (len(ids) + 1)) for j in range(12)])
        Path(specs[0]['log']).write_text('device cuda | gpus 2\n[gpu cuda:0] loaded 1 arm(s) | 0s\n[gpu cuda:1] loaded 1 arm(s) | 0s\n'
                                         '[gpu] done: %d/%d studies on 2 device(s) | 1s\nwrote /kaggle/working/submission.csv | x\nDONE 1s\n'
                                         % (len(ids), len(ids)))
        if self.runner is not None and hasattr(self.k1, '_s6_expected_models'):
            expected = self.k1._s6_expected_models
            aq = {'source': 'runtime_observer', 'component_a_source_sha256': expected['a']['component_source_sha256'],
                  'boards': [{'gpu_uuid': 'GPU-%d' % i, 'a': synthetic_model_record(expected['a'])} for i in range(2)],
                  'overhead_seconds': 0.01, 'overhead_scope': 'synthetic test fixture only'}
            write(Path(specs[0]['log']).parent / 'model-qualification.json', json.dumps(aq))
        time.sleep(1.2)
        return [('A', 0, 1.2, '')]


@unittest.skipUnless(HAVE_CAPTURED, 'archived publisher solo source or numpy/pandas not present')
class RehearsalIntegrationTests(unittest.TestCase):
    """RT.run in rehearsal mode end to end, with only the GPU subprocess layer and host probes faked."""

    def test_rehearsal_run_passes_replays_and_leaves_no_submission(self):
        from knee import s6_solo_rehearsal as R
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('K1_KAGGLE_ROOT', None)
            s = synthetic(d, 'v2')
            project = Path(d) / 'project'
            for n in ('knee/__init__.py', 'knee/s6_solo_pair.py', 'knee/s6_solo_runtime.py', 'knee/s6_solo_rehearsal.py',
                      'knee/s6_solo_a_runner.py', 'knee/s6_solo_b_observer.py'):
                write(project / n, (PROJECT / n).read_bytes())
            write(project / RT.CAPTURED, CAPTURED.read_bytes())
            ids, splits = ['T1', 'T2', 'T3', 'T4', 'S1'], ['train'] * 4 + ['test']
            cfg = copy.deepcopy(s['cfg'])
            cfg['mode'] = 'rehearsal'
            cfg['rehearsal'] = {'fixture': {'ids': ids, 'splits': splits},
                                'mounts': [['rsna-knee-s6fall-weights', 'checkpoint-manifest.json'],
                                           ['rsna-orthofoundation-fold0-epoch7-20261004', 'anatomy.pt']],
                                'sentinel_per_board': 2, 'sentinel_timeout_s': 600, 'sentinel_probability_atol': 1e-3,
                                'projection_studies': 1300, 'gate_seconds': 29880, 'post_controller_allowance_seconds': 60}
            work = s['kroot'] / 'working'
            real_bind, seen = RT.bind, {}

            def bind(k1, cfg_):
                real_bind(k1, cfg_)
                for k, v in s['pins'].items():
                    setattr(k1, k, v)
                k1.probe_env = lambda: {'cuda_available': True, 'n_gpu': 2, 'gpus': ['Tesla T4', 'Tesla T4']}
                k1.probe_b_stack = lambda I: {'synthetic': True}
                k1.log = lambda *a: None
                k1.run_group = FakeGroups(k1, str(project / 'knee/s6_solo_a_runner.py'), str(Path(d) / 'fx/input/rsna-knee-abnormality-detection'))
                seen['k1'] = k1
                return k1
            boards = [{'index': i, 'uuid': 'GPU-%d' % i, 'name': 'Tesla T4', 'used_mib': 100.0, 'total_mib': 15360.0} for i in range(2)]
            with mock.patch.object(RT, 'WORK', work), mock.patch.object(RT, 'A_INPUT_ROOT', s['ainput']), \
                    mock.patch.object(R, 'INPUT_ROOT', s['kroot'] / 'input'), mock.patch.object(R, 'FIXTURE_ROOT', Path(d) / 'fx'), \
                    mock.patch.object(RT, 'bind', bind), mock.patch.object(RT, 'gpu_query', lambda: boards), \
                    mock.patch.object(RT, 'meminfo', lambda: {'MemTotal': 31 << 30, 'MemAvailable': 25 << 30}):
                with self.assertRaisesRegex(RuntimeError, 'S6 solo rehearsal did not pass'):
                    RT.run(cfg, project)
            k1 = seen['k1']
            failure = json.loads((work / 's6solo' / 'failure.json').read_text())
            self.assertIn('did not pass', failure['detail'])
            self.assertFalse((work / 'submission.csv').exists())
            self.assertTrue((work / 's6solo' / 'rehearsal-final.csv').is_file())
            whats = [w for w, _ in k1.run_group.calls]
            self.assertEqual(whats, ['A', 'B', 'sentinel'])
            b_cmds = k1.run_group.calls[1][1]
            for c in b_cmds:
                self.assertEqual(c[c.index('--amp') + 1], 'fp16')
                self.assertEqual(c[c.index('--models') + 1], str(k1.TMP / 'k1models'))
            self.assertEqual(sorted(p.name for p in (k1.TMP / 'k1models').iterdir()), ['E_prior_s42_fall_v2.pt'])
            self.assertEqual(sorted(json.loads((work / 'k1/B/px_hashes.json').read_text())), sorted(ids))
            led = json.loads((work / 's6solo' / 'ledger.json').read_text())
            self.assertEqual([r['id'] for r in led['rows']], ids)
            with self.assertRaisesRegex(ValueError, 'PASSING_BOUND_REHEARSAL'):
                R.verify_result(work, cfg, k1)
            self.assertEqual(os.environ.get('K1_KAGGLE_ROOT'), str(Path(d) / 'fx'))
        os.environ.pop('K1_KAGGLE_ROOT', None)

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'torch is not installed')
    def test_generated_b_launcher_observes_imported_compact_infer_source(self):
        import torch
        tree = ast.parse(CAPTURED.read_text())
        launcher = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'LAUNCHER' for t in n.targets))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); src, code, hooks, proj = (root / n for n in ('src', 'code', 'hooks', 'project'))
            for p in (src, code, hooks, proj / 'knee'):
                p.mkdir(parents=True, exist_ok=True)
            (proj / 'knee' / 's6_solo_b_observer.py').write_bytes((PROJECT / 'knee/s6_solo_b_observer.py').read_bytes())
            (hooks / 'sitecustomize.py').write_text('from knee.s6_solo_b_observer import install\ninstall()\n')
            ct_text = ('import torch\nfrom torch import nn\n'
                       'class Net(nn.Module):\n'
                       ' def __init__(self):\n  super().__init__(); self.weight=nn.Parameter(torch.tensor([1.0])); self.register_buffer("bias", torch.tensor([0.25]))\n'
                       ' def forward(self, x): return x * self.weight + self.bias\n')
            ci_text = ('import os, torch, compact_train as CT\n'
                       'class CStudies: pass\n'
                       'def main():\n'
                       ' net=CT.Net(); net.load_state_dict(torch.load(os.environ["S6SOLO_TEST_CHECKPOINT"], weights_only=True), strict=True)\n'
                       ' net.eval().requires_grad_(False); x=torch.tensor([1.0]); net(x); net(x)\n')
            (src / 'compact_train.py').write_text(ct_text)
            (src / 'compact_infer.py').write_text(ci_text)
            state = {'weight': torch.tensor([2.0]), 'bias': torch.tensor([0.5])}
            ck = root / 'tiny.pt'; torch.save(state, ck)
            code_text = launcher.replace('__SRC__', repr(str(src))).replace('__CODE__', repr(str(code)))
            launch = root / 'k1_b_infer.py'; launch.write_text(code_text)
            receipt = root / 'qualification.json'
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (hooks, proj, src, code))),
                       S6SOLO_MODE='rehearsal', S6SOLO_TEST_CHECKPOINT=str(ck),
                       S6SOLO_MODEL_RECEIPT=str(receipt), S6SOLO_B_CHECKPOINT_SHA256=sha(ck.read_bytes()),
                       PYTHONDONTWRITEBYTECODE='1')
            run = subprocess.run([sys.executable, str(launch)], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            record = json.loads(receipt.read_text())
            self.assertEqual(record['inference_source_sha256'], sha(ci_text.encode()))
            self.assertEqual(len(record['boards']), 1)
            model = record['boards'][0]['b']
            self.assertEqual(model['checkpoint_sha256'], sha(ck.read_bytes()))
            self.assertTrue(model['strict_loaded_before'] and model['strict_reload'])
            self.assertEqual(model['state_before_sha256'], model['state_after_sha256'])


@unittest.skipUnless(HAVE_CAPTURED, 'archived publisher solo source or numpy/pandas not present')
class EntryIntegrationTests(unittest.TestCase):
    """RT.run in entry mode: captured A command, official-style test.csv, submission written or deleted."""

    def run_entry(self, d, plan):
        s = synthetic(d, 'control')
        project = Path(d) / 'project'
        for n in ('knee/__init__.py', 'knee/s6_solo_pair.py', 'knee/s6_solo_runtime.py'):
            write(project / n, (PROJECT / n).read_bytes())
        write(project / RT.CAPTURED, CAPTURED.read_bytes())
        work = s['kroot'] / 'working'
        real_load, real_bind = RT.load_controller, RT.bind

        def load_with_root(path, name=RT.CONTROLLER_MODULE):
            with mock.patch.dict(os.environ, {'K1_KAGGLE_ROOT': str(s['kroot']), 'K1_TMP': str(s['tmp'])}):
                return real_load(path, name)

        def bind(k1, cfg_):
            real_bind(k1, cfg_)
            for k, v in s['pins'].items():
                setattr(k1, k, v)
            k1.probe_env = lambda: {'cuda_available': True, 'n_gpu': 2, 'gpus': ['Tesla T4', 'Tesla T4']}
            k1.probe_b_stack = lambda I: {}
            k1.log = lambda *a: None
            k1.run_group = FakeGroups(k1, None, str(s['kroot'] / 'input/rsna-knee-abnormality-detection'))
            k1.run_group.b = FakeWorkers(k1, plan)
            return k1
        with mock.patch.object(RT, 'WORK', work), mock.patch.object(RT, 'A_INPUT_ROOT', s['ainput']), \
                mock.patch.object(RT, 'load_controller', load_with_root), mock.patch.object(RT, 'bind', bind):
            try:
                return RT.run(s['cfg'], project), work, None
            except Exception as e:
                return None, work, e

    def test_entry_writes_exact_submission(self):
        with tempfile.TemporaryDirectory() as d:
            receipt, work, err = self.run_entry(d, [{}])
            self.assertIsNone(err)
            self.assertEqual(receipt['status'], 'PASS')
            with open(work / 'submission.csv') as f:
                rows = list(csv.reader(f))
            self.assertEqual([r[0] for r in rows], [UID, 'S1', 'S2', 'S3'])
            self.assertEqual(receipt['submission']['sha256'], RT.file_sha(work / 'submission.csv'))
            self.assertTrue((work / 's6solo' / 'entry-receipt.json').is_file())

    def test_entry_failure_deletes_submission(self):
        with tempfile.TemporaryDirectory() as d:
            receipt, work, err = self.run_entry(d, [{'fail': {'S2'}}])     # N < 100: any B failure after retry is fatal
            self.assertIsNotNone(err)
            self.assertFalse((work / 'submission.csv').exists())
            self.assertIn('allowed 0', json.loads((work / 's6solo' / 'failure.json').read_text())['detail'])


@unittest.skipUnless(HAVE_EVIDENCE, 'accepted S6 solo evidence tree not present')
class BuilderPairTests(unittest.TestCase):
    def setUp(self):
        from scripts import build_s6_solo_pair as B
        self.B = B
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / 'outputs'
        self.out.mkdir()
        patcher = mock.patch.object(B, 'OUT', self.out); patcher.start(); self.addCleanup(patcher.stop)

    def test_pair_differs_only_in_weight_binding_and_replays(self):
        B = self.B
        B.build_rehearsal(self.out / 'kaggle')
        roots = {v: self.out / 'kaggle' / B.SLUGS[v] for v in ('v2', 'control')}
        from scripts.check_s6_solo_pair import pair_difference, probe
        for root in roots.values():
            pack, _ = B.unpack(root / 'notebook.ipynb')
            probe(pack)
        self.assertEqual(pair_difference(roots)['identical_sources'], len(B.COMMON_SOURCES) + len(B.REHEARSAL_SOURCES) + 1)
        for v, root in roots.items():
            B.verify_build(root, v)
            meta = json.loads((root / 'kernel-metadata.json').read_text())
            self.assertEqual((meta['id'], meta['title'], meta['is_private'], meta['enable_internet'], meta['machine_shape']),
                             ('ozgung/' + B.SLUGS[v], B.SLUGS[v], True, False, 'NvidiaTeslaT4'))
        from scripts import check_s6_solo_pair as C
        diff = C.pair_difference(roots)
        self.assertEqual(diff['bootstrap_differing_lines'], ['encoded', 'expected_payload'])
        self.assertNotIn('rehearsal', diff['arm_differing_fields'])
        a, _ = B.unpack(roots['v2'] / 'notebook.ipynb')
        self.assertEqual(a['cfg']['arm']['b_model']['file'], 'E_prior_s42_fall_v2.pt')
        self.assertEqual(a['cfg']['rehearsal']['fixture']['splits'].count('train'), 36)
        with self.assertRaisesRegex(ValueError, 'FRESH_IGNORED_OUTPUT'):
            B.emit(roots['v2'], B.SLUGS['v2'], 'v2', a, 'rehearsal')
        nb = roots['control'] / 'notebook.ipynb'
        nb.write_text(nb.read_text().replace('S6 solo payload changed', 'S6 solo payload edited'))
        with self.assertRaisesRegex(ValueError, 'IMMUTABLE_BUILD_notebook.ipynb'):
            B.verify_build(roots['control'], 'control')

    def test_bootstrap_unpacks_verifies_and_launches_the_runtime_module(self):
        B = self.B
        B.build_rehearsal(self.out / 'kaggle')
        _, code = B.unpack(self.out / 'kaggle' / B.SLUGS['v2'] / 'notebook.ipynb')
        work = Path(self.tmp.name) / 'kw'
        work.mkdir()
        site = "work=Path('/kaggle/working')"
        self.assertEqual(code.count(site), 1)
        r = subprocess.run([sys.executable, '-c', code.replace(site, 'work=Path(%r)' % str(work))], capture_output=True, text=True,
                           timeout=300, cwd=self.tmp.name, env={k: v for k, v in os.environ.items() if k != 'K1_KAGGLE_ROOT'})
        self.assertNotEqual(r.returncode, 0)                      # no /kaggle here: the child must stop, not the bootstrap
        self.assertIn('S6SOLO rehearsal solo-V2 payload', r.stdout)
        self.assertIn('s6_solo_runtime.py', r.stdout)            # the child imported the unpacked runtime module
        self.assertIn('S6 solo runtime exited 1; submission.csv removed', r.stderr)
        pack, _ = B.unpack(self.out / 'kaggle' / B.SLUGS['v2'] / 'notebook.ipynb')
        for name, digest in pack['source_sha256'].items():
            self.assertEqual(RT.file_sha(work / 's6solo-project' / name), digest)

    def test_entry_requires_this_packets_passing_result(self):
        B = self.B
        B.build_rehearsal(self.out / 'kaggle')
        build = self.out / 'kaggle' / B.SLUGS['v2']
        prepared = json.loads((build / 'build-receipt.json').read_text())
        cfg = B.packet('v2')['cfg']
        k1 = load()
        root, receipt = synthetic_result(Path(self.tmp.name) / 'res', cfg, k1, payload=prepared['payload_sha256'])
        self.assertEqual(receipt['status'], 'PASS', receipt['failed_checks'])
        from knee import s6_solo_rehearsal as R
        self.assertEqual(R.verify_result(root, cfg, k1)['receipt_sha256'], RT.file_sha(root / 's6solo' / 'receipt.json'))
        entry = B.build_entry('v2', build, root, destination=self.out / 'kaggle' / 'entry-v2', slug='s6solo-v2-entry-20261016')
        self.assertEqual(entry['mode'], 'entry')
        root2, _ = synthetic_result(Path(self.tmp.name) / 'res2', cfg, k1, payload='0' * 64)
        with self.assertRaisesRegex(ValueError, 'PASSING_BOUND_REHEARSAL|RESULT_FROM_THIS_REHEARSAL_PACKET'):
            B.build_entry('v2', build, root2, destination=self.out / 'kaggle' / 'entry-v2b')
        root3, _ = synthetic_result(Path(self.tmp.name) / 'res3', cfg, k1, slow=True, payload=prepared['payload_sha256'])
        with self.assertRaisesRegex(ValueError, 'PASSING_BOUND_REHEARSAL'):
            B.build_entry('v2', build, root3, destination=self.out / 'kaggle' / 'entry-v2c')
        control_build = self.out / 'kaggle' / B.SLUGS['control']
        with self.assertRaisesRegex(ValueError, 'PASSING_BOUND_REHEARSAL|RESULT_FROM'):
            B.build_entry('control', control_build, root, destination=self.out / 'kaggle' / 'entry-c')

    def test_builder_cfg_binds_real_b_constructor_buffers_sources_and_aggregate_receipt(self):
        from knee import s6_solo_rehearsal as R
        B = self.B
        B.build_rehearsal(self.out / 'kaggle')
        pack, _ = B.unpack(self.out / 'kaggle' / B.SLUGS['v2'] / 'notebook.ipynb')
        cfg = pack['cfg']
        expected = R._expected_model_qualification(cfg)
        cpu = cfg['arm']['cpu']
        self.assertEqual(expected['b']['buffers_sha256'], cpu['constructor_buffers_sha256'])
        self.assertEqual(expected['b']['constructor_source_sha256'],
                         '14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80')
        self.assertEqual(expected['b']['inference_source_sha256'],
                         '34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a')
        boards = [{'uuid': 'GPU-A'}, {'uuid': 'GPU-B'}]
        records = []
        for uuid in ('GPU-A', 'GPU-B'):
            row = {'gpu_uuid': uuid}
            for component, want in expected.items():
                model = {'checkpoint_sha256': want['checkpoint_sha256'],
                         'state_before_sha256': want['state_sha256'], 'state_after_sha256': want['state_sha256'],
                         'reload_state_sha256': want['state_sha256'], 'buffers_before_sha256': want['buffers_sha256'],
                         'buffers_after_sha256': want['buffers_sha256'], 'reload_buffers_sha256': want['buffers_sha256'],
                         'constructor_sha256': 'a' * 64, 'reload_constructor_sha256': 'a' * 64,
                         'constructor_source_sha256': want['constructor_source_sha256'],
                         'input_sha256': 'b' * 64, 'reload_input_sha256': 'b' * 64,
                         'strict_reload': True, 'strict_loaded_before': True, 'sentinel_same_input': True,
                         'sentinel_max_abs': 0.0, 'state_unchanged_from_load': True,
                         'buffers_unchanged_from_load': True,
                         'constructor_spec': want.get('constructor_spec'),
                         'buffer_inventory': want.get('buffer_inventory')}
                if component == 'a':
                    model.update(grid_before_sha256=want['grid_sha256'], grid_after_sha256=want['grid_sha256'],
                                 grid_unchanged_from_load=True)
                row[component] = model
            records.append(row)
        receipt = {'source': 'runtime_observer', 'boards': records,
                   'component_a_source_sha256': expected['a']['component_source_sha256'],
                   'inference_source_sha256': expected['b']['inference_source_sha256'],
                   'overhead_scope': 'reload_and_same_input_sentinel_only; other observer work remains in wall stages'}
        self.assertTrue(R._valid_model_receipt(receipt, boards, expected))
        receipt.pop('inference_source_sha256')
        self.assertFalse(R._valid_model_receipt(receipt, boards, expected))
        control_pack, _ = B.unpack(self.out / 'kaggle' / B.SLUGS['control'] / 'notebook.ipynb')
        control_cfg = control_pack['cfg']
        control_expected = R._expected_model_qualification(control_cfg)
        self.assertEqual(control_expected['b']['buffers_sha256'], control_cfg['arm']['cpu']['constructor_buffers_sha256'])
        a_receipt, b_receipt = synthetic_model_receipts(control_cfg)
        merged = {'source': 'runtime_observer', 'boards': [{'gpu_uuid': a['gpu_uuid'], 'a': a['a'], 'b': b['b']}
                                                            for a, b in zip(a_receipt['boards'], b_receipt['boards'])],
                  'component_a_source_sha256': a_receipt['component_a_source_sha256'],
                  'inference_source_sha256': b_receipt['inference_source_sha256']}
        self.assertTrue(R._valid_model_receipt(merged, [{'uuid': 'GPU-0'}, {'uuid': 'GPU-1'}], control_expected))


if __name__ == '__main__':
    unittest.main()
