"""P9s (Lane P9, 2026-10-20): shared decode -- our S6 members rendered at 320 px from R6's native cache and scored inside the
R6 plain worker (knee.public_s6c_shared); no separate own stage.

Synthetic data only: fake px_cache / compact_infer / compact_train modules for the loader-child hook, tiny CPU modules for the
member forwards. Heavy parts skip when numpy/pandas/torch are absent.
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

P9S = R6F.VARIANTS['p9s']


class P9sSpecTests(unittest.TestCase):
    def test_p9s_is_p9_arithmetic_with_a_shared_decode(self):
        p9 = R6F.VARIANTS['p9']
        self.assertEqual({k: v for k, v in P9S.items() if k != 'own_shared'}, p9)
        self.assertTrue(R6F.own_shared(P9S) and R6F.own_shared(R6F.VARIANTS['p9sr']) and R6F.own_compact_arm(P9S))
        self.assertFalse(R6F.own_shared(p9) or R6F.own_shared(None) or R6F.compact_shared(P9S))
        self.assertEqual(R6F.resolve_compact(P9S), ((('r6', .5), ('s6c', .5)), False))
        self.assertEqual(R6F.resolve_blend(R6F.VARIANTS['p9sr'])[0], R6F.RAPTOR_ARMS)
        with self.assertRaises(ValueError):
            R6F.own_shared({**copy.deepcopy(R6F.VARIANTS['p6p']), 'own_shared': True})
        with self.assertRaises(ValueError):
            R6F.resolve_blend({**copy.deepcopy(P9S), 'compact_shared': True})          # one shared scheme at a time
        from scripts import build_r6flip_entry as B
        self.assertEqual(B.slug_for('p9s', 'rehearsal'), 'r6flip-p9s-rehearsal-20261020')
        self.assertTrue(B.has_s6c('p9s') and B.own_dual('p9s'))
        self.assertFalse(B.same_stages('p9s', 'p9'))                                     # a P9 rehearsal does not qualify P9s


def stub_cfg(variant, block=True, mode='entry'):
    cols = ['StudyInstanceUID'] + R6F.LABELS
    cfg = {'mode': mode, 'r6flip_binding': {'official_columns': cols, 'entry_variant': {'name': variant}, 'r6': {}}, 'environment': {},
           'blend': copy.deepcopy(R6F.VARIANTS[variant])}
    if block:
        from tests.test_public_s6c import models
        cfg['s6c'] = S6C.block(models())
    return cfg, cols


@unittest.skipIf(np is None or pd is None, 'numpy/pandas not installed')
class P9sStageTests(unittest.TestCase):
    def run_entry(self, cfg, cols, shared_reports=None):
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

    def test_no_own_stage_and_member_reports_come_from_the_r6_stage(self):
        seen = []

        def reports(out, arm, pass_name='production'):
            seen.append((arm, pass_name))
            return [{'status': 'NO_RECEIPT'}] * 2
        for variant, plan in (('p9s', [('old', 'production'), ('r6', 'production')]),
                              ('p9sr', [('old', 'production'), ('crop224', 'production'), ('r6', 'production')])):
            cfg, cols = stub_cfg(variant)
            report, calls, written = self.run_entry(cfg, cols, reports)
            self.assertEqual(calls, plan, variant)
            self.assertIn('s6c', report['workers'])
            self.assertNotIn('s6c', report['phases'])
            self.assertEqual(report['status'], 'PASS_WITH_FALLBACK')
            self.assertTrue(np.all(written['rows'] == .5))
        self.assertEqual(seen, [('s6c', 'production')] * 2)
        self.assertEqual(sorted(report['arm_merge']), ['crop224', 'old320', 'r6', 's6c'])

    def test_rehearsal_collects_member_reports_after_each_sentinel_pass(self):
        seen = []

        def reports(out, arm, pass_name='production'):
            seen.append((arm, pass_name))
            return [{'status': 'NO_RECEIPT'}] * 2
        cfg, cols = stub_cfg('p9s', mode='rehearsal')
        probes = ['q%02d' % i for i in range(36)]
        cfg.update(probes={q: {} for q in probes}, probe_order=probes, s6_sentinels=['z1', 'z2', 'z3', 'z4'], p9_reference={})
        with mock.patch('knee.public_fullfield_entry.read_series', lambda p: {q: {} for q in probes}):
            try:
                self.run_entry(cfg, cols, reports)
            except Exception:
                pass                                            # the stub rehearsal fails at the strict merge
        self.assertEqual(seen, [('s6c', 'sentinel1'), ('s6c', 'sentinel2'), ('s6c', 'production')])
        cfg.pop('p9_reference')                                 # a P9s rehearsal without the separate-pass reference is refused
        calls = []
        with mock.patch('knee.public_fullfield_entry.read_series', lambda p: {q: {} for q in probes}):
            with self.assertRaises(ValueError) as cm:
                self.run_entry(cfg, cols, lambda *a, **k: calls.append(a) or [{'status': 'NO_RECEIPT'}] * 2)
        self.assertEqual(str(cm.exception), 'P9S_REFERENCE_MATCHES_BLEND')
        cfg, cols = stub_cfg('p9s')
        cfg['p9_reference'] = {}                                # the private fixture in an entry
        report, calls, _ = self.run_entry(cfg, cols)
        self.assertEqual((calls, report['status']), ([], 'EMERGENCY_CONSTANT'))

    def test_worker_config_carries_the_shared_block_on_every_r6_pass(self):
        cfg, cols = stub_cfg('p9s', mode='rehearsal')
        cfg.update(boards=[{'uuid': 'a'}, {'uuid': 'b'}], staged='st', staged_ids=['s1'], stage_binding={}, code='c', source_pins={},
                   r6_mount='m', anatomy='an', deadline_epoch=123.)
        cfg['r6flip_binding']['r6'] = {'manifest': {}, 'cpu_pins': {}}
        for pass_name in ('sentinel1', 'sentinel2', 'production'):
            c = ENTRY.r6_worker_config(cfg, 1, pass_name, ['s1'], Path('/o/board1-r6-' + pass_name), None)
            self.assertEqual(c['flip'], False)
            self.assertEqual(c['s6c_shared']['block'], cfg['s6c'])
            self.assertEqual((c['s6c_shared']['deadline_epoch'], c['s6c_shared']['measure_stride2']), (123., False))   # proxy measured once
            self.assertEqual(c['s6c_shared']['out'], '/o/board1-s6c-' + pass_name)
            self.assertNotIn('r7_shared', c)
        cfg['mode'] = 'entry'
        self.assertFalse(ENTRY.r6_worker_config(cfg, 0, 'production', ['s1'], Path('/o/board0-r6-production'), None)['s6c_shared']['measure_stride2'])
        cfg['blend'] = copy.deepcopy(R6F.VARIANTS['p9'])
        self.assertNotIn('s6c_shared', ENTRY.r6_worker_config(cfg, 0, 'production', ['s1'], Path('/o/x'), None))

    def test_projection_counts_members_inside_the_r6_phase_only(self):
        def shard(seconds, windows=10):
            return {'records': [{'production_seconds': seconds, 'status': 'genuine', 'windows': windows}] * 5, 'peak_cuda_bytes': 7}
        phase = lambda: {'wall_seconds': 100., 'peaks': {'host_rss_bytes': 1, 'output_disk_bytes': 1}, 'minimum_disk_free_bytes': 1}
        workers = {'old': [shard(2.), shard(3.)], 'r6': [shard(4.), shard(5.)], 's6c': [shard(1., 20), shard(1., 20)]}
        phases = {a: phase() for a in ('old', 'r6', 'r6_sentinel1', 'r6_sentinel2')}
        p = ENTRY.projection(10., phases, workers, 5., [10] * 870)
        self.assertNotIn('s6c', p['measured_production_wall_seconds'])
        self.assertNotIn('measured_s6c_windows', p)
        self.assertEqual(sorted(ENTRY.resources(phases, workers)['peak_cuda_bytes_by_arm']), ['old', 'r6', 's6c'])


# ---- loader-child hook -------------------------------------------------------------------------------------------------------
def fake_modules(n_slices=6, size_px=None):
    """px_cache / compact_infer / compact_train stand-ins: one sagittal series of n_slices, view 0, windows of 3 slices."""
    import torch

    class StudySlices:
        built = []

        def __init__(self, cache, uids, size, views='v1', anchors=None):
            self.cache, self.uids, self.size, self.views, self.anchors = cache, uids, size, views, anchors

        def build(self, uid):
            StudySlices.built.append((self.cache, self.size, self.views, self.anchors))
            S = int(self.size)
            px = torch.arange(n_slices, dtype=torch.uint8)[:, None, None].expand(n_slices, S, S).contiguous() + (S % 7)
            meta = torch.zeros((n_slices, 12), dtype=torch.float32)
            meta[:, 1] = 2                      # slot
            meta[:, 3] = 0                      # view id
            meta[:, 4] = torch.linspace(-1, 1, n_slices)
            meta[:, 7] = 0                      # plane sagittal
            return {'uid': uid, 'error': None, 'px': px, 'meta': meta, 'blocks': [{'n_slices': n_slices, 'views': [[0, 0]]}]}

    def window_rows(blocks, meta_of, mode, slices, eval_stride=3, cover_stride=3, phys_mm=0.0, **kw):
        rows = []
        for b in blocks:
            for f, v, st in b['views']:
                for c in range(1, b['n_slices'] - 1, eval_stride):
                    rows.append((f, st + c, st + c - 1, st + c, st + c + 1))
        return rows
    px_cache = types.ModuleType('px_cache')
    px_cache.StudySlices = StudySlices
    compact_infer = types.ModuleType('compact_infer')
    compact_infer.FORCED_SIDE = {}
    compact_train = types.ModuleType('compact_train')
    compact_train.window_rows = window_rows
    return {'px_cache': px_cache, 'compact_infer': compact_infer, 'compact_train': compact_train}


@unittest.skipIf(torch is None or np is None, 'torch/numpy not installed')
class ChildHookTests(unittest.TestCase):
    def test_every_r6_render_also_yields_the_members_windows(self):
        from knee import public_s6c_shared as SH
        mods = fake_modules()
        with mock.patch.dict(sys.modules, mods):
            builder = mods['px_cache'].StudySlices('/cache/work', [], 384, 'v4', anchors='/cache/work/anchors')
            ds = types.SimpleNamespace(builder=builder, native=object())
            SH.install_child(ds, SH.CHILD_GEOMETRY)
            SH.install_child(ds, SH.CHILD_GEOMETRY)                     # idempotent
            item384 = ds.builder.build('u1')
            self.assertEqual(tuple(item384['px'].shape), (6, 384, 384))
            item = SH.attach(ds, {'uid': 'u1', 'images': item384['px'], 'error': None})
            self.assertIsNone(item['s6c_error'])
            self.assertEqual(tuple(item['s6c_images'].shape), (4, 3, 320, 320))          # 4 windows of 3 slices at stride 1
            self.assertEqual(item['s6c_images'].dtype, torch.uint8)
            self.assertEqual(tuple(item['s6c_meta'].shape), (4, 6))
            self.assertEqual(item['s6c_note']['separate_render'], False)
            built = mods['px_cache'].StudySlices.built
            self.assertEqual(built, [('/cache/work', 384, 'v4', '/cache/work/anchors'), ('/cache/work', 320, 'v4', '/cache/work/anchors')])
            self.assertEqual(SH.attach(ds, {'uid': 'u1'})['s6c_error'], 'no shared render for this item')   # consumed once

    def test_a_side_flipped_study_is_rendered_again_without_the_forced_side(self):
        from knee import public_s6c_shared as SH
        mods = fake_modules()
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(sys.modules, mods):
            work = Path(d) / 's_abc'
            work.mkdir()
            builder = mods['px_cache'].StudySlices(str(work), [], 384, 'v4', anchors=str(work / 'anchors'))
            prepared = []

            def prepare(uid, w):
                prepared.append((uid, str(w), dict(mods['compact_infer'].FORCED_SIDE)))
                (w / 'studies').mkdir(parents=True)
                (w / 'geometry').mkdir()
                (w / 'studies' / 'key.json').write_text(json.dumps({'series': [{'side': 'L'}]}))
            bank = types.SimpleNamespace(record=lambda uid, study, geo_dir: {'method': 'fake'})
            ds = types.SimpleNamespace(builder=builder, native=object(), _prepare=prepare, bank=bank,
                                       KB=types.SimpleNamespace(study_key=lambda uid: 'key'),
                                       PI=types.SimpleNamespace(OneStudySource=types.SimpleNamespace(current=None)))
            mods['compact_infer'].FORCED_SIDE['u2'] = 'R'
            SH.install_child(ds, SH.CHILD_GEOMETRY)
            ds.builder.build('u2')
            item = SH.attach(ds, {'uid': 'u2'})
            self.assertIsNone(item['s6c_error'])
            self.assertEqual(item['s6c_note']['separate_render'], True)
            self.assertEqual(item['s6c_note']['r6_forced_side'], 'R')
            self.assertEqual(prepared[0][0], 'u2')
            self.assertEqual(prepared[0][2], {})                                           # prepared without the forced side
            self.assertEqual(mods['compact_infer'].FORCED_SIDE, {'u2': 'R'})                # restored for R6
            self.assertFalse(Path(prepared[0][1]).exists())                                 # second cache removed
            self.assertEqual(mods['px_cache'].StudySlices.built[-1][0], prepared[0][1])

    def test_render_failure_is_carried_not_raised(self):
        from knee import public_s6c_shared as SH
        mods = fake_modules()
        mods['compact_train'].window_rows = lambda *a, **k: []
        with mock.patch.dict(sys.modules, mods):
            ds = types.SimpleNamespace(builder=mods['px_cache'].StudySlices('/c', [], 384, 'v4'), native=object())
            SH.install_child(ds, SH.CHILD_GEOMETRY)
            ds.builder.build('u3')
            item = SH.attach(ds, {'uid': 'u3'})
            self.assertIn('no usable window', item['s6c_error'])
            self.assertIsNone(item['s6c_images'])


# ---- main process: SharedS6 on CPU with tiny member modules --------------------------------------------------------------------
class TinyNet(torch.nn.Module if torch is not None else object):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(12, generator=g))
        self.register_buffer('b', torch.randn(12, generator=g))

    def forward(self, img, meta, mask, chunk=0, return_aux=False):
        x = img[mask].float().mean(dim=(1, 2, 3)) / 255.            # [W]
        return (x.mean() * self.w + self.b)[None]                    # [1, 12]


def make_shared(out, board, pass_name, ids, env, deadline=0., measure=True):
    from knee import public_s6c_shared as SH
    sh = SH.SharedS6.__new__(SH.SharedS6)
    staged = [{'alias': 'E_prior_s42_f0.pt', 'sha256': 'aa' * 32, 'bytes': 7}, {'alias': 'E_prior_s43_f0.pt', 'sha256': 'bb' * 32, 'bytes': 8}]
    models = [TinyNet(1).eval(), TinyNet(2).eval()]
    folder = Path(out) / ('board%d-s6c-%s' % (board, pass_name))
    sh.__dict__.update(torch=torch, block={'aliases': [s['alias'] for s in staged]}, ids=list(ids), board=board, pass_name=pass_name,
                       device='cpu', deadline=deadline, measure_stride2=measure, out=folder / 's6c', geometry=dict(SH.CHILD_GEOMETRY),
                       models=models, staged=staged, active=True, cut=None, rows={}, csv=None, stride1_check=False,
                       code_receipt={'captured_sha256': S6C.CAPTURED_SHA256, 'extras_ok': True, 'source_sha_ok': True, 'members': 1, 'extras': 1},
                       entries=[{'alias': s['alias'], 'sha256': s['sha256'], 'bytes': s['bytes'], 'spatial': True, 'state': R6F.module_digest(m),
                                 'buffers': R6F.buffers_digest(m), 'constructor': R6F.constructor_digest(m)} for s, m in zip(staged, models)])
    sh.checksums = [R6F.checksum(m) for m in models]
    (sh.out / 'compact').mkdir(parents=True)
    (sh.out / 'models').mkdir()
    for s in staged:                                                  # restaged() re-hashes the alias files
        (sh.out / 'models' / s['alias']).write_bytes(b'x' * s['bytes'])
    import hashlib
    for s, e in zip(staged, sh.entries):
        s['sha256'] = e['sha256'] = hashlib.sha256(b'x' * s['bytes']).hexdigest()
    sh.csv = open(sh.out / 'compact' / 'predictions.csv', 'w', newline='', encoding='utf-8')
    sh.csv.write(','.join(['StudyInstanceUID'] + S6C.LABELS) + '\n')
    return sh


def study_input(uid, windows=5, seed=0):
    g = torch.Generator().manual_seed(hash(uid) % 1000 + seed)
    img = torch.randint(0, 256, (windows, 3, 320, 320), generator=g, dtype=torch.uint8)
    meta = torch.zeros((windows, 6))
    return {'images': img, 'meta': meta, 'error': None, 'note': {'separate_render': False, 'r6_forced_side': None},
            'input': {'render_windows': 'h' + uid, 'metadata': 'm' + uid, 'windows': windows}}


@unittest.skipIf(torch is None or np is None or pd is None, 'torch/numpy/pandas not installed')
class SharedS6Tests(unittest.TestCase):
    ENV = {'torch': '2.10.0+cu128', 'packages': {'timm': '1.0.26'}, 'cuda_device_count': 1}

    def run_pass(self, out, board, pass_name, ids, skip=(), deadline=0.):
        sh = make_shared(out, board, pass_name, ids, self.ENV, deadline=deadline)
        obs = types.SimpleNamespace(fresh_depth=0, studies=[])
        for sid in ids:
            cur = {'id': sid, 's6c_input': study_input(sid), 'handle_seconds': 10., 'wait_seconds': 0., 'first_of_loader': not obs.studies}
            if sid not in skip:
                sh.on_study(obs, cur, None, None, None, 192)
                self.assertNotIn('images', cur['s6c_input'])                  # no tensors kept in the study record
            obs.studies.append(cur)
        rec = json.loads(Path(sh.finalize(obs, {'environment': dict(self.ENV, physical_uuid='u%d' % board)})['receipt']).read_text())
        rec['peak_cuda_bytes'] = 1                                            # CPU test: no CUDA peak
        return rec

    def test_shared_pass_ledger_receipt_merge_and_gates(self):
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
            self.assertEqual((x['slot'], x['arm'], x['version'], x['cut'], x['flip']), ('compact-shared', 's6c', S6C.VERSION, None, {'enabled': False}))
            self.assertEqual(x['state_before'], x['state_after'])
            self.assertEqual(len(x['records'][0]['seed_probabilities']), 2)
            self.assertEqual(x['records'][0]['stride2_proxy']['windows'], 3)
            self.assertLessEqual(x['records'][0]['reconstruction_max_abs'], 5e-7)
            self.assertEqual(reports['sentinel1'][1]['failed'], 1)
            self.assertEqual(reports['sentinel1'][1]['records'][0]['failure_code'], 'r6 study not scored')
            values, mask, seed_p = S6C.merge(ids, reports['production'], out, 2)       # the P7/P9 merge applies unchanged
            self.assertEqual((values.shape, int(mask.sum()), seed_p.shape), ((39, 12), 0, (39, 2, 12)))
            sal, st = S6C.salvage(ids, out, 2)
            self.assertTrue(np.allclose(sal, values, equal_nan=True) and st.count('genuine') == 39)
            # the rehearsal gates: the own worker key, the injected sentinel skipped in pass 1, the separate-pass parity
            block = S6C.block([{'seed': 42, 'labels': 'v3', 'file': 'a.pt', 'sha256': x['models'][0]['sha256'], 'bytes': 7,
                                'dataset': 'ozgung/d1', 'dataset_version': 1},
                               {'seed': 43, 'labels': 'v3', 'file': 'b.pt', 'sha256': x['models'][1]['sha256'], 'bytes': 8,
                                'dataset': 'ozgung/d2', 'dataset_version': 1}])
            rows = {row['id']: row['seed_probabilities'] for w in reports['production'] for row in w['records']}
            r = {'boards': [{'uuid': 'u0'}, {'uuid': 'u1'}],
                 'workers': {'s6c': reports['production'], 's6c_sentinels': {k: reports[k] for k in ('sentinel1', 'sentinel2')},
                             'r6': [{'s6c_shared': {'status': 'PASS'}}] * 2}}
            cfg = {'s6c': block, 'binding': {'environment': {'torch': self.ENV['torch'], 'packages': self.ENV['packages']}},
                   's6_sentinels': sentinels, 'staged_ids': ids,
                   'p9_reference': {'aliases': block['aliases'], 'receipt_sha256': 'ref',
                                    'probabilities': {s: [[min(1., v + 4e-4) for v in m] for m in rows[s]] for s in ids}}}
            q = S6C.require_compact(r, cfg, key='s6c', injected=('sentinel1', sentinels[0]))
            self.assertEqual((q['seeds'], q['sentinel_repeats'][sentinels[0]], q['sentinel_repeats'][sentinels[1]]), (2, 3, 5))
            self.assertEqual(q['sentinel_max_abs_dprob'], 0.0)                          # deterministic tiny members
            with self.assertRaises(ValueError):
                S6C.require_compact(r, cfg, key='s6c')                                   # the injection must be declared
            g = S6C.require_shared(r, cfg)
            self.assertLessEqual(g['separate_pass_max_abs'], 4.0001e-4)
            self.assertEqual((g['studies'], g['side_flipped_separate_renders']), (39, 0))
            for code, mutate in {
                'P9S_SEPARATE_PASS_PARITY': lambda r, c: c['p9_reference']['probabilities'][ids[5]][1].__setitem__(0, .99),
                'P9_REFERENCE_COVERAGE': lambda r, c: c['p9_reference'].update(aliases=['x', 'y']),
                'S6C_SHARED_WORKER_PASS': lambda r, c: r['workers']['s6c'][0].update(cut={'after_studies': 1}),
                'S6C_SHARED_WORKER_PASS ': lambda r, c: r['workers']['r6'][1].__setitem__('s6c_shared', {'status': 'FAIL'}),
                'S6C_SHARED_STUDY_GATES': lambda r, c: r['workers']['s6c'][1]['records'][0].update(r6_aborted=True),
            }.items():
                r2, c2 = copy.deepcopy(r), copy.deepcopy(cfg)
                if code.endswith(' '):
                    r2['workers']['r6'] = [dict(w) for w in r2['workers']['r6']]
                mutate(r2, c2)
                with self.assertRaises(ValueError, msg=code) as cm:
                    S6C.require_shared(r2, c2)
                self.assertEqual(str(cm.exception), code.strip())

    def test_members_stop_first_near_the_deadline_and_cut_rows_are_deadline_failures(self):
        ids = ['s%d' % i for i in range(10)]
        with tempfile.TemporaryDirectory() as d:
            sh = make_shared(Path(d), 0, 'production', ids, self.ENV, deadline=time.time() + 1.)
            obs = types.SimpleNamespace(fresh_depth=0, studies=[])
            for i, sid in enumerate(ids):
                cur = {'id': sid, 's6c_input': study_input(sid), 'handle_seconds': 5., 'wait_seconds': 0., 'first_of_loader': i == 0}
                sh.on_study(obs, cur, None, None, None, 192)
                obs.studies.append(cur)
            self.assertFalse(sh.active)
            self.assertEqual(sh.cut['after_studies'], 3)                                    # 7 x 5 s x 1.3 > 1 s: cut after the 3rd study
            rec = json.loads(Path(sh.finalize(obs, {'environment': self.ENV})['receipt']).read_text())
            self.assertEqual((rec['genuine'], rec['failed']), (3, 7))
            self.assertTrue(all(row['deadline'] for row in rec['records'] if row['status'] != 'genuine'))
            self.assertEqual(rec['records'][-1]['failure_code'], 'members cut before the deadline')
            arrays = {'old320': np.random.default_rng(0).random((10, 12)), 'r6': np.random.default_rng(1).random((10, 12))}
            with np.load(sh.out / 'predictions.npz') as a:
                own = S6C.mean_rank(S6C.sigmoid(a['seed_logits']), np.asarray(a['statuses']) == 'genuine')
            arrays['s6c'] = own
            statuses = {'old320': ['genuine'] * 10, 'r6': ['genuine'] * 10, 's6c': [r['status'] if r['status'] == 'genuine' else 'unavailable'
                                                                                   for r in rec['records']]}
            out = R6F.recover_guarded(arrays, statuses, blend=P9S)
            self.assertEqual(out['compact_fallback'], {'partial_compact': 7, 'r6_only': 7, 's6c_only': 0})


if __name__ == '__main__':
    unittest.main()
