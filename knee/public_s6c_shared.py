"""P9s shared decode: our S6 members scored inside the R6 plain worker from ONE DICOM decode per study.

Both pipelines are the same lineage (heliosli CStudies): native render of the DICOM series + geometry bank + knee anchors
into a per-study `work` cache, then px_cache.StudySlices renders the views at size S. R6 renders at 384 px (crop 352); our
S6 members at 320 px (crop 288). px_cache.py, probe_infer.py, knee_anchor_bank_v2.py, the training/ package and the text of
compact_train.window_rows are byte-identical in the two code trees, so the loader child of R6's compact_infer can render the
320 px windows from the same `work` cache right after R6's 384 px windows (StudySlices.build is wrapped per dataset instance;
the cache is still on disk at that point). The S6 path has no image side model: when R6's side check flipped a study, the
members' windows are rendered from a second native render without the forced side, as the separate s6c worker would see it.
The windows travel in the loader item ('s6c_images', 's6c_meta'); the main process scores every member right after R6's
last plain fold forward (SharedS6.on_study), fp16, chunk 192, exactly compact_infer 34f433fe's score(): centre crop, one
forward per member, fp32 rescue of a non-finite output, probabilities = sigmoid(mean logits), per-member logits kept.
No flip TTA: the members were trained without mirror augmentation (--aug crop) and their compact_infer has no --flip-tta.
Ledger, CSV and receipt have the s6c worker's shape (knee.public_s6c.merge / salvage / require_compact apply unchanged).
Near the deadline the members stop first (R6-only pace x 1.3, the P6s rule); studies without members keep rank(R6 plain).
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from knee import public_r6flip as R6F
from knee import public_s6c as S6C

require = S6C.require
VERSION = 's6c-shared-decode-v1'
CHILD_GEOMETRY = dict(S6C.GEOMETRY_CHILD)
CHUNK = S6C.GEOMETRY['chunk']
CUT_SAFETY = 1.3                 # members stop when the remaining studies at the R6-only pace x 1.3 would pass the deadline
CUT_MIN_STUDIES = 3
PARITY_TOLERANCE = 1e-3          # per member vs the separate s6c worker on the frozen 39 (the fresh-process tolerance)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(json.dumps(value, indent=1, allow_nan=False, default=str).encode() + b'\n')
    os.replace(tmp, path)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


# ---- loader child: the 320 px render from R6's work cache ----------------------------------------------------------------------
def windows_from_item(item, window_rows, geometry):
    """CStudies.__getitem__'s window assembly (eval mode, no extent meta) on a StudySlices.build output."""
    import numpy as np
    import torch
    px, meta = item['px'].numpy(), item['meta'].numpy()
    blocks = [{'n_slices': b['n_slices'], 'views': [(0, v, st) for v, st in b['views']]} for b in item['blocks']]
    cand = window_rows(blocks, lambda f, r: meta[r], 'eval', int(geometry['slices']), int(geometry['eval_stride']),
                       phys_mm=float(geometry['phys_mm']))
    if not cand:
        raise ValueError('no usable window')
    rows = np.array(cand, np.int64)
    img = np.stack([px[rows[:, 2 + d]] for d in range(int(geometry['slices']))], 1)
    m = meta[rows[:, 1]]
    meta6 = np.stack([m[:, 1], m[:, 3], m[:, 4], m[:, 5], m[:, 6], m[:, 7]], 1).astype(np.float32)
    return torch.from_numpy(img), torch.from_numpy(meta6)


def render_members_input(ds, uid, geometry, px_cache, window_rows, forced_side):
    """The members' windows for one study from the dataset's current work cache (R6's native render, geometry and anchors);
    with a forced side (R6's image side check flipped the study) from a second native render without it."""
    builder = ds.builder
    t0 = time.perf_counter()
    note = {'r6_forced_side': forced_side, 'separate_render': forced_side is not None}
    if forced_side is None:
        b = px_cache.StudySlices(builder.cache, [], int(geometry['size']), geometry['views'], anchors=builder.anchors)
        item = b.build(uid)
    else:
        work2 = Path(builder.cache).with_name(Path(builder.cache).name + '-s6c')
        shutil.rmtree(work2, ignore_errors=True)
        try:
            ds.PI.OneStudySource.current = uid
            ds._prepare(uid, work2)
            key = ds.KB.study_key(uid)
            with open(work2 / 'studies' / (key + '.json'), encoding='utf-8') as fh:
                study = json.load(fh)
            rec = ds.bank.record(uid, study, geo_dir=str(work2 / 'geometry'))
            (work2 / 'anchors').mkdir(exist_ok=True)
            with open(work2 / 'anchors' / (key + '.json'), 'w', encoding='utf-8') as fh:
                json.dump(rec, fh)
            note['sides'] = sorted({str(s.get('side')) for s in study['series']})
            b = px_cache.StudySlices(str(work2), [], int(geometry['size']), geometry['views'], anchors=str(work2 / 'anchors'))
            item = b.build(uid)
        finally:
            shutil.rmtree(work2, ignore_errors=True)
    img, meta = windows_from_item(item, window_rows, geometry)
    ref = None
    if geometry.get('stride1_check') and int(geometry['eval_stride']) != 1:     # rehearsal at stride 2: the stride-1 windows too
        ref = windows_from_item(item, window_rows, {**geometry, 'eval_stride': 1})
    note['render_seconds'] = time.perf_counter() - t0
    note['eval_stride'] = int(geometry['eval_stride'])
    return img, meta, note, ref


def install_child(ds, geometry):
    """Wrap this dataset instance's StudySlices.build so every R6 render also produces the members' windows (stashed on the
    dataset; the __getitem__ wrapper in knee.public_r6flip_worker.child_init attaches them to the item)."""
    if getattr(ds, '_s6c_shared_installed', False):
        return
    import px_cache
    import compact_infer as CI
    import compact_train as CT
    builder = ds.builder
    original = builder.build
    window_rows = CT.window_rows          # the module function: identical text in both trees; the R6 injection seam is not used

    def build(uid, _original=original):
        item = _original(uid)
        try:
            forced = CI.FORCED_SIDE.get(str(uid))
            if forced is not None:
                CI.FORCED_SIDE.pop(str(uid), None)
            try:
                img, meta, note, ref = render_members_input(ds, uid, geometry, px_cache, window_rows, forced)
            finally:
                if forced is not None:
                    CI.FORCED_SIDE[str(uid)] = forced
            ds._s6c_shared_item = {'images': img, 'meta': meta, 'error': None, 'note': note,
                                   'images_ref': None if ref is None else ref[0], 'meta_ref': None if ref is None else ref[1]}
        except Exception as exc:
            ds._s6c_shared_item = {'images': None, 'meta': None, 'error': repr(exc)[:300], 'note': None, 'images_ref': None, 'meta_ref': None}
        return item

    builder.build = build
    ds._s6c_shared_installed = True


def attach(ds, item):
    """Move the stashed members' windows onto the loader item (keys the R6 source ignores)."""
    stash = getattr(ds, '_s6c_shared_item', None)
    ds._s6c_shared_item = None
    if stash is None:
        item['s6c_error'] = 'no shared render for this item'
        return item
    item['s6c_images'], item['s6c_meta'], item['s6c_error'], item['s6c_note'] = stash['images'], stash['meta'], stash['error'], stash['note']
    item['s6c_images_ref'], item['s6c_meta_ref'] = stash.get('images_ref'), stash.get('meta_ref')
    return item


# ---- main process ----------------------------------------------------------------------------------------------------------
class SharedS6:
    KEY = 's6c_shared'

    def __init__(self, cfg, torch, device=None):
        self.cfg, self.torch = cfg['s6c_shared'], torch
        self.block = self.cfg['block']
        self.ids, self.board, self.pass_name = list(cfg['ids']), cfg.get('board'), cfg.get('pass')
        self.project = Path(cfg['code']).resolve().parent
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.deadline = float(self.cfg.get('deadline_epoch') or 0)
        self.measure_stride2 = bool(self.cfg.get('measure_stride2'))
        self.out = Path(self.cfg['out']) / S6C.FOLDER
        self.geometry = dict(self.cfg.get('geometry') or CHILD_GEOMETRY)
        require(set(self.geometry) == set(CHILD_GEOMETRY) and all(self.geometry[k] == CHILD_GEOMETRY[k] for k in CHILD_GEOMETRY if k != 'eval_stride')
                and int(self.geometry['eval_stride']) in (1, 2, 3), 'S6C_SHARED_GEOMETRY')
        self.stride1_check = bool(self.cfg.get('stride1_check')) and int(self.geometry['eval_stride']) != 1
        self.models, self.staged, self.entries, self.checksums = [], [], [], []
        self.active, self.cut = True, None
        self.rows = {}
        self.csv = None
        self.code_receipt = None

    @property
    def child_cfg(self):
        return {**self.geometry, 'stride1_check': self.stride1_check}

    def load(self):
        """The exact S6 code tree (captured controller, publisher extras, heliosli files), the sealed member files under their
        aliases, S6's own compact_train.Net per member (strict load), digests and checksums; CSV header."""
        import importlib.util
        torch = self.torch
        from knee import public_s6c_worker as S6CW
        t0 = time.perf_counter()
        self.out.mkdir(parents=True, exist_ok=True)
        k1 = S6CW.load_controller(self.project / S6C.CAPTURED)
        src, code, self.code_receipt = S6CW.code_tree(k1, self.block, self.out / 'k1src')
        anatomy = Path(self.block['helios_mount']) / 'anatomy.pt'
        require(file_sha(anatomy) == S6C.ANATOMY_SHA256, 'S6C_SHARED_ANATOMY_FILE')
        self.staged = S6CW.stage_models(self.block, self.out / 'models')
        spec = importlib.util.spec_from_file_location('s6c_shared_compact_train', str(src / 'compact_train.py'))
        CT = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = CT
        spec.loader.exec_module(CT)
        require(file_sha(src / 'compact_train.py') == S6C.SOURCE_PINS['compact_train.py'] and CT.LABELS == S6C.LABELS, 'S6C_SHARED_TRAINER')
        self.CT = CT
        g = S6C.GEOMETRY
        for st in self.staged:
            path = self.out / 'models' / st['alias']
            sd = torch.load(path, map_location='cpu', weights_only=True)
            spatial = any(k == 'sp_q' or k.endswith('.sp_q') for k in sd)
            extent = any(k.split('.')[-1] in ('ext_u', 'ext_view', 'ext_gamma') for k in sd)
            require(not extent, 'S6C_SHARED_NO_EXTENT_MEMBERS')
            kw = {'spatial': True} if spatial else {}
            m = CT.Net(g['backbone'], None, freeze_blocks=16, crop=g['crop'], state_spec=None, in_chans=g['slices'], **kw)   # load_group's call
            res = m.load_state_dict(sd, strict=True)
            del sd
            require(not res.missing_keys and not res.unexpected_keys, 'S6C_SHARED_STRICT_LOAD')
            self.entries.append({'alias': st['alias'], 'sha256': st['sha256'], 'bytes': st['bytes'], 'spatial': spatial,
                                 'state': R6F.module_digest(m), 'buffers': R6F.buffers_digest(m), 'constructor': R6F.constructor_digest(m)})
            self.models.append(m.eval().requires_grad_(False).to(self.device).to(memory_format=torch.channels_last))
        self.checksums = [R6F.checksum(m) for m in self.models]
        (self.out / 'compact').mkdir(exist_ok=True)
        self.csv = open(self.out / 'compact' / 'predictions.csv', 'w', newline='', encoding='utf-8')
        self.csv.write(','.join(['StudyInstanceUID'] + S6C.LABELS) + '\n')
        self.csv.flush()
        return {'members': [e['alias'] for e in self.entries], 'spatial_models': sum(e['spatial'] for e in self.entries),
                'code': self.code_receipt, 'seconds': time.perf_counter() - t0, 'deadline_epoch': self.deadline, 'device': self.device}

    def _forward(self, m, img, meta, mask, mode):
        torch = self.torch
        if mode == 'fp32' or img.device.type != 'cuda':
            with torch.autocast(img.device.type, enabled=False):
                return m(img, meta, mask, chunk=CHUNK).float()
        with torch.autocast('cuda', dtype=torch.float16):
            return m(img, meta, mask, chunk=CHUNK).float()

    def r6_only_pace(self, obs):
        rows = [(s, self.rows.get(s['id'], {})) for s in obs.studies if not s.get('aborted')]
        paces = [s['handle_seconds'] + (0. if s['first_of_loader'] else s['wait_seconds']) - x.get('s6c_seconds', 0.) for s, x in rows]
        return sum(paces) / len(paces) if len(paces) >= CUT_MIN_STUDIES else None

    def should_cut(self, obs, now=None):
        if not self.active or not self.deadline:
            return False
        pace = self.r6_only_pace(obs)
        if pace is None:
            return False
        remaining = len(self.ids) - len(obs.studies)
        now = time.time() if now is None else now
        if now + remaining * max(pace, 0.) * CUT_SAFETY > self.deadline:
            self.active = False
            self.cut = {'after_studies': len(obs.studies), 'remaining': remaining, 'r6_only_pace_seconds': pace, 'epoch': now,
                        'deadline_epoch': self.deadline}
            return True
        return False

    def on_study(self, obs, cur, img, meta, mask, chunk):
        """Once per study after R6's last plain fold forward: the members on the study's own 320 px windows."""
        torch = self.torch
        if 's6c' in cur:
            return
        sid = cur['id']
        self.should_cut(obs)
        s6 = cur.get('s6c_input') or {}
        cur['s6c_input'] = {k: v for k, v in s6.items() if k not in ('images', 'meta', 'images_ref', 'meta_ref')}   # no tensors kept
        if not self.active:
            cur['s6c'] = None
            self.rows[sid] = {'status': 'deadline'}
            return
        if s6.get('error') or s6.get('images') is None or not len(s6['images']):
            cur['s6c'] = False
            self.rows[sid] = {'status': 'render_failure', 'error': s6.get('error') or 'no members input'}
            return
        t0 = time.perf_counter()
        g = S6C.GEOMETRY
        off = (g['size'] - g['crop']) // 2
        obs.fresh_depth += 1                  # not R6 scoring calls for the observer
        try:
            with torch.inference_mode():
                full = s6['images'].to(self.device, non_blocking=True)
                x = full[None, :, :, off:off + g['crop'], off:off + g['crop']]
                mt = s6['meta'].to(self.device)[None]
                mk = torch.ones((1, x.shape[1]), dtype=torch.bool, device=self.device)
                require(x.dtype == torch.uint8 and tuple(x.shape[2:]) == (g['slices'], g['crop'], g['crop']) and mt.shape[-1] == 6,
                        'S6C_SHARED_INPUT_SCHEMA')
                zs, rescues, member_seconds = [], 0, []
                for m in self.models:
                    tm = time.perf_counter()
                    z = self._forward(m, x, mt, mk, 'fp16')
                    if not bool(torch.isfinite(z).all()):
                        z = self._forward(m, x, mt, mk, 'fp32')
                        rescues += 1
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    member_seconds.append(time.perf_counter() - tm)
                    zs.append(z[0].detach().cpu())
                check = None
                if self.stride1_check and s6.get('images_ref') is not None and len(s6['images_ref']):   # rehearsal at stride 2
                    xr = s6['images_ref'].to(self.device, non_blocking=True)[None, :, :, off:off + g['crop'], off:off + g['crop']]
                    mr = s6['meta_ref'].to(self.device)[None]
                    kr = torch.ones((1, xr.shape[1]), dtype=torch.bool, device=self.device)
                    zr, sr = [], []
                    for m in self.models:
                        tm = time.perf_counter()
                        z1 = self._forward(m, xr, mr, kr, 'fp16')
                        if not bool(torch.isfinite(z1).all()):
                            z1 = self._forward(m, xr, mr, kr, 'fp32')
                        if self.device == 'cuda':
                            torch.cuda.synchronize()
                        sr.append(time.perf_counter() - tm)
                        zr.append(z1[0].detach().cpu())
                    check = {'seed_probabilities': torch.sigmoid(torch.stack(zr).double()).numpy().tolist(), 'seconds': sr,
                             'windows': int(s6['images_ref'].shape[0])}
                    del xr, mr, kr
                stride2 = None
                if self.measure_stride2 and x.shape[1] >= 2:
                    ts = time.perf_counter()
                    self._forward(self.models[0], x[:, ::2].contiguous(), mt[:, ::2].contiguous(), mk[:, ::2].contiguous(), 'fp16')
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    stride2 = {'windows': int(x[:, ::2].shape[1]), 'seconds': time.perf_counter() - ts}
                del full, x, mt, mk
            checksum_equal = all(R6F.checksum(m) == c for m, c in zip(self.models, self.checksums))
        finally:
            obs.fresh_depth -= 1
        require(checksum_equal, 'S6C_SHARED_PARAMETER_BUFFER_MUTATION')
        z = torch.stack(zs).double()                                  # [S, 12]
        p = torch.sigmoid(z.mean(0)).numpy()
        seed_p = torch.sigmoid(z).numpy()
        finite = bool(all(math.isfinite(float(v)) and 0. <= float(v) <= 1. for v in p)) and bool(torch.isfinite(z).all())
        seconds = time.perf_counter() - t0
        row = {'status': 'genuine' if finite else 'nonfinite', 'probabilities': [float(v) for v in p], 'logits': z.float().numpy().tolist(),
               'seed_probabilities': seed_p.tolist(), 'fp32_rescues': rescues, 'checksum_equal': checksum_equal,
               's6c_seconds': seconds, 'member_seconds': member_seconds, 'stride2_proxy': stride2,
               'windows': int(s6['images'].shape[0]), 'note': s6.get('note'), 'input': s6.get('input'), 'stride1_check': check}
        self.rows[sid] = row
        cur['s6c'] = True
        if finite:
            text = ['%.6f' % v for v in row['probabilities']]
            row['reconstruction_max_abs'] = max(abs(float(t) - v) for t, v in zip(text, row['probabilities']))
            self.csv.write(','.join([sid] + text) + '\n')
            self.csv.flush()

    def finalize(self, obs, r6_receipt):
        """Members' ledger (assigned order, the s6c worker's arrays) and receipt (the s6c worker's shape plus shared-decode facts)."""
        import numpy as np
        import resource
        from knee import public_s6c_worker as S6CW
        if self.csv is not None:
            self.csv.close()
        studies = {s['id']: s for s in obs.studies}
        n = len(self.models)
        probs, statuses, seed_logits, records = [], [], [], []
        for ordinal, sid in enumerate(self.ids):
            x, s = self.rows.get(sid), studies.get(sid)
            if x is not None and x['status'] == 'genuine' and s is not None:
                probs.append(np.asarray(x['probabilities'], dtype=np.float64))
                statuses.append('genuine')
                seed_logits.append(np.asarray(x['logits'], dtype=np.float32))
                chk = x.get('stride1_check') or {}
                records.append({'id': sid, 'ordinal': ordinal, 'status': 'genuine', 'windows': x['windows'],
                                'production_seconds': x['s6c_seconds'], 's6c_seconds': x['s6c_seconds'], 'member_seconds': x['member_seconds'],
                                'stride2_proxy': x['stride2_proxy'], 'reconstruction_max_abs': x['reconstruction_max_abs'],
                                'probabilities': x['probabilities'], 'seed_probabilities': x['seed_probabilities'],
                                'fp32_rescues': x['fp32_rescues'], 'checksum_equal': x['checksum_equal'], 'shared_decode': True,
                                'render': x['note'], 'input': x.get('input'), 'r6_aborted': bool(s.get('aborted')),
                                **({'seed_probabilities_stride1': chk['seed_probabilities'], 'stride1_seconds': chk['seconds'],
                                    'windows_stride1': chk['windows']} if chk else {})})
            else:
                probs.append(np.full(12, np.nan))
                statuses.append('data_local_failure')
                seed_logits.append(np.full((n, 12), np.nan, dtype=np.float32))
                code = ('members cut before the deadline' if (x or {}).get('status') == 'deadline' else
                        'nonfinite member output' if (x or {}).get('status') == 'nonfinite' else
                        'members render failure' if (x or {}).get('status') == 'render_failure' else 'r6 study not scored')
                records.append({'id': sid, 'ordinal': ordinal, 'status': 'data_local_failure', 'phase': 's6c_shared', 'failure_code': code,
                                'deadline': x is None or x.get('status') == 'deadline', 'production_seconds': 0.,
                                'error': (x or {}).get('error')})
        ledger = self.out / 'predictions.npz'
        S6CW.save_ledger(ledger, self.ids, probs, statuses, seed_logits)
        after = S6CW.restaged(self.out / 'models', self.staged)
        digests_after = [{'state': R6F.module_digest(m), 'buffers': R6F.buffers_digest(m)} for m in self.models]
        require(all(a['state'] == e['state'] and a['buffers'] == e['buffers'] for a, e in zip(digests_after, self.entries)),
                'S6C_SHARED_STATE_AFTER')
        cuda = self.torch.cuda.is_available()
        rec = {'status': 'PASS', 'version': S6C.VERSION, 'shared_version': VERSION, 'arm': S6C.FOLDER, 'slot': 'compact-shared',
               'pass': self.pass_name, 'board': self.board, 'assigned': self.ids, 'completed': len(records), 'records': records,
               'genuine': statuses.count('genuine'), 'failed': statuses.count('data_local_failure'),
               'zero_failed': all(s == 'genuine' for s in statuses), 'zero_imputed': True,
               'models': self.staged, 'state_before': S6C.digest(self.staged), 'state_after': S6C.digest(after),
               'module_digests': self.entries, 'environment': r6_receipt.get('environment'),
               'compact': {'status': 'PASS', 'amp': S6C.AMP, 'models': self.block['aliases'], 'spatial_models': sum(e['spatial'] for e in self.entries),
                           'extent_models': 0, 'timed_out': False, 'anchor_weights_sha256': S6C.ANATOMY_SHA256,
                           'sha256': dict(S6C.SOURCE_PINS), 'chunk': CHUNK, 'scored': statuses.count('genuine'), 'studies': len(self.ids)},
               'code': self.code_receipt, 'anatomy_sha256': S6C.ANATOMY_SHA256, 'geometry': {**S6C.GEOMETRY, **self.geometry},
               'stride1_check': self.stride1_check, 'flip': {'enabled': False},
               'cut': self.cut, 'shared_with': 'r6 plain production worker (same process; members rendered at 320 px from R6\'s native cache)',
               'ledger_sha256': file_sha(ledger), 'fp32_recoveries': sum(x.get('fp32_rescues', 0) for x in records),
               'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               'peak_cuda_bytes': self.torch.cuda.max_memory_allocated() if cuda else 0}
        require(rec['state_after'] == rec['state_before'], 'S6C_SHARED_FILES_UNCHANGED')
        save(self.out / 'receipt.json', rec)
        return {'status': 'PASS', 'genuine': rec['genuine'], 'failed': rec['failed'], 'cut': self.cut, 'receipt': str(self.out / 'receipt.json'),
                'members': len(self.models)}
