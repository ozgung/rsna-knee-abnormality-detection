"""R6 flip-TTA board worker: heliosli's compact_infer v4c main() run unchanged, observed from outside.

One process per board and pass, launched like probe_dual.py launches compact_infer.py (CUDA_VISIBLE_DEVICES,
one-thread BLAS env, --amp auto, two spawn loader workers, the notebook's WORKER_ARGS including --flip-tta).
Declared differences from probe_dual: --data/--split point at the staged images-only tree, --deadline is the
controller's T0 + 8.3 h stage deadline (unreached studies fall back in the controller), and main() is called
in-process so the observers below can bind what it constructs. No source byte is changed.

Observers (rehearsal and entry alike, so rehearsal timing includes them):
* torch.load / compact_train.Net.__init__ / load_state_dict: checkpoint file SHA256, constructor arguments,
  full state, buffers and architecture digests of every fold model, compared with the actual-checkpoint CPU
  pins; a fresh constructor strict-reloads the same file and must reproduce every digest.
* Net.forward (return_aux passes only): the plain and mirrored windows of every study; the mirrored input
  must equal our independent mirror transform; raw logits/state outputs are kept so the deployed
  probabilities can be recomputed independently (flip permutation, MCL plain pass, stackers).
  The first scoring forward of each fold is repeated on the fresh strict-reloaded eager model.
* DataLoader: per-study item hashes (background thread), timings, fluid/side/anchor notes, per-study exact
  parameter checksums; worker_init_fn binds the anatomy V-Net and side net inside each loader child
  (state before/after every item) and, in rehearsal sentinel pass 1 only, injects one no-usable-window
  failure through the source's own window_rows seam.

P6 dual compact (sealed cfg['blend'] compact_arms): the same worker runs R6 and then heliosli's R7 v1 bundle without
--flip-tta. Its board config then carries 'flip': False and, for R7, 'profile': 'r7' / 'compact_arm': 'r7' with the R7
manifest, CPU pins and mount in the 'r6' / 'r6_mount' slots; only plain passes are observed and the deployed probabilities
are recomputed from them. A config without these keys is the unchanged R6 flip-TTA worker.

P6s shared decode (cfg 'r7_shared', production pass only): the R6 plain worker also loads the four R7 v1 fold models
(pinned by file SHA256 and actual-checkpoint CPU state digests, compiled like compact_infer --compile) and scores them on
the very windows compact_infer just rendered for R6 (one decode per study), right after R6's last fold. R7 deployed
probabilities are recomputed with R7's own state spec and stackers (knee.public_r7_stackers) and written as an R7 ledger,
receipt and a flushed CSV under <stage>/board<b>-r7-production/r7. Near the deadline R7 stops first (R6 keeps going), so
cut studies keep R6 plain as their compact slot.
"""
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from knee import public_r6flip as R6F

VERSION = R6F.VERSION
WORKER_MODULE = 'knee.public_r6flip_worker'
RECONSTRUCTION_ATOL = 1e-6
RELOAD_DPROB_TOL = 5e-3          # fresh eager vs deployed (compiled) forward; 1e-3 failed on T4 (fold 4 sentinel 1.85e-3, identical on both boards = compile precision, not drift)
FLUID_OK_PREFIX = 'no region view for '
FLUID_MISSING = ('no fluid-sensitive sagittal series',)
PROBE_DUAL_ENV = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MALLOC_ARENA_MAX': '2',
                  'PYTHONDONTWRITEBYTECODE': '1'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(json.dumps(value, indent=1, allow_nan=False, default=str).encode() + b'\n')
    os.replace(tmp, path)


def array_sha(a):
    """Identical to knee.public_s6_evaluation.array_sha (dtype/shape header, then the raw bytes)."""
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(R6F.canonical([str(a.dtype), list(a.shape)]))
    h.update(memoryview(a).cast('B') if a.size else b'')
    return h.hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def flip_enabled(cfg):
    """--flip-tta unless the board config seals 'flip': False (P6 plain compact arms)."""
    flip = cfg.get('flip', R6F.FLIP_TTA)
    require(isinstance(flip, bool), 'R6_FLIP_FLAG')
    return flip


def profile_name(cfg):
    name = cfg.get('profile', 'r6')
    require(name in ('r6', 'r7') and (name == 'r6' or flip_enabled(cfg) is False), 'COMPACT_PROFILE')   # R7 only as a plain arm
    return name


def compact_arm(cfg):
    arm = cfg.get('compact_arm', 'r6')
    require(arm == profile_name(cfg), 'COMPACT_ARM_IS_PROFILE')
    return arm


def worker_args(cfg, assignment, output):
    """probe_dual.py's worker command (minus interpreter/script) plus the notebook's WORKER_ARGS, in their order."""
    m = cfg['r6']['manifest']
    mount = Path(cfg['r6_mount'])
    args = ['--data', str(cfg['data']), '--split', 'train', '--code', str(cfg['code']), '--arm', m['arm'],
            '--assignment', str(assignment), '--output', str(output), '--amp', 'auto', '--workers', '2',
            '--deadline', repr(float(cfg.get('deadline_epoch') or 0)) if cfg.get('deadline_epoch') else '0',
            '--models', str(mount)]
    args += ['--backbone', m['backbone'], '--size', str(m['size']), '--crop', str(m['crop']),
             '--eval-stride', str(m.get('eval_stride', 3)), '--views', m.get('views', 'v1')]
    if m.get('state_spec'):
        args += ['--state-spec', str(mount / m['state_spec'])]
    if m.get('side_model'):
        args += ['--side-model', str(mount / m['side_model']), '--side-threshold', str(m.get('side_threshold', 2.0)),
                 '--side-position-mm', str(m.get('side_position_mm', 20.0)), '--side-max-x', str(m.get('side_max_x', 0.0))]
    args += ['--slices', str(m.get('slices', 3)), '--phys-mm', str(m.get('phys_mm', 0.0))]
    require(not m.get('models2'), 'R6_ONE_MODEL_GROUP')
    if m.get('trainer_md5'):
        args += ['--trainer-md5', m['trainer_md5']]
    if m.get('compile'):
        args += ['--compile']
    if m.get('fluid_stacker'):
        args += ['--fluid-stacker', str(mount / m['fluid_stacker'])]
    if flip_enabled(cfg):
        args += ['--flip-tta']
    if m.get('state_stacker'):
        args += ['--state-stacker', str(mount / m['state_stacker'])]
    if m.get('views') in ('v3k', 'v4'):
        args += ['--anchor-weights', str(cfg['anatomy'])]
    return args


# ---- loader children ---------------------------------------------------------------------------------------------------
def _child_digests(ds):
    out = {}
    for name, module in (('anatomy', getattr(getattr(ds, 'bank', None), 'model', None)), ('side', getattr(ds, 'side_net', None))):
        if module is None:
            out[name] = None
            continue
        out[name] = {'state': R6F.module_digest(module), 'buffers': R6F.buffers_digest(module),
                     'constructor': R6F.constructor_digest(module), 'training': bool(module.training)}
    return out


def child_init(child_dir, inject_uid, worker_id, shared=None):
    """DataLoader worker_init_fn: bind the loader child's anatomy and side models; optional rehearsal injection.
    shared: P9s geometry dict -> every R6 render also yields our members' 320 px windows (knee.public_s6c_shared)."""
    import torch.utils.data as tud
    info = tud.get_worker_info()
    ds = info.dataset
    cls = type(ds)
    original = cls.__getitem__
    path = Path(child_dir) / ('child-%d-%d.json' % (os.getpid(), worker_id))
    state = {'pid': os.getpid(), 'worker_id': worker_id, 'num_workers': info.num_workers, 'items': 0,
             'before': None, 'after': None, 'changed': False, 'injected': []}

    def getitem(self, i):
        if self.native is None:
            try:
                self._setup()
            except Exception:
                pass              # the source's own try block retries setup and records the item error
        if state['before'] is None and self.native is not None:
            state['before'] = _child_digests(self)
            save(path, state)
        if shared is not None and self.native is not None:
            from knee import public_s6c_shared as SH
            SH.install_child(self, shared)
        uid = self.uids[i]
        if inject_uid is not None and uid == inject_uid and not state['injected'] and self.native is not None:
            seam = self.window_rows
            self.window_rows = lambda *a, **k: []
            try:
                item = original(self, i)
            finally:
                self.window_rows = seam
            state['injected'].append({'ordinal': int(i), 'error': item.get('error')})
        else:
            item = original(self, i)
        if shared is not None:
            from knee import public_s6c_shared as SH
            item = SH.attach(self, item)
        state['items'] += 1
        if self.native is not None:
            state['after'] = _child_digests(self)
            state['changed'] = state['changed'] or state['after'] != state['before']
        save(path, state)
        return item

    cls.__getitem__ = getitem


# ---- main-process observer -------------------------------------------------------------------------------------------------
class Observer:
    def __init__(self, cfg, out, torch, CT):
        from concurrent.futures import ThreadPoolExecutor
        self.cfg, self.out, self.torch, self.CT = cfg, Path(out), torch, CT
        self.pins = cfg['r6']['cpu_pins']['folds']
        self.crop = int(cfg['r6']['manifest']['crop'])
        self.slices = int(cfg['r6']['manifest']['slices'])
        self.models, self.loaded, self.fresh_depth = [], {}, 0
        self.studies, self.current, self.calls_other, self.loaders = [], None, 0, []
        self.pool = ThreadPoolExecutor(1)
        self.mark = None
        self.saved = {}
        self.child_dir = self.out / 'children'
        self.child_dir.mkdir(parents=True, exist_ok=True)
        self.inject = cfg.get('inject_uid')
        self.flip = flip_enabled(cfg)
        self.shared = None             # SharedR7 (P6s production pass only)

    # -- installation
    def install(self):
        torch, CT, obs = self.torch, self.CT, self
        Net = CT.Net
        self.saved = {'load': torch.load, 'init': Net.__init__, 'lsd': Net.load_state_dict, 'forward': Net.forward,
                      'loader': torch.utils.data.DataLoader}
        original_load, original_init, original_lsd, original_forward = (self.saved[k] for k in ('load', 'init', 'lsd', 'forward'))

        @functools.wraps(original_load)
        def load(f, *a, **k):
            result = original_load(f, *a, **k)
            if isinstance(f, (str, os.PathLike)) and Path(f).is_file() and obs.fresh_depth == 0:
                obs.loaded[id(result)] = (str(f), file_sha(f), result)
            return result

        @functools.wraps(original_init)
        def init(self, *a, **k):
            original_init(self, *a, **k)
            if obs.fresh_depth == 0:
                object.__setattr__(self, '_r6flip_ctor', (a, k))

        @functools.wraps(original_lsd)
        def load_state_dict(self, state_dict, *a, **k):
            result = original_lsd(self, state_dict, *a, **k)
            if obs.fresh_depth == 0:
                strict = k.get('strict', a[0] if a else True)
                obs.register(self, state_dict, strict, result)
            return result

        @functools.wraps(original_forward)
        def forward(self, img, meta, mask, chunk=0, return_aux=False):
            out = original_forward(self, img, meta, mask, chunk=chunk, return_aux=return_aux)
            if obs.fresh_depth == 0:
                if return_aux:
                    obs.on_scoring(self, img, meta, mask, chunk, out)
                else:
                    obs.calls_other += 1
            return out

        base_loader = self.saved['loader']

        class ObservedLoader(base_loader):
            def __init__(self, dataset, *a, **k):
                if k.get('num_workers', 0):
                    require(k.get('worker_init_fn') is None, 'SOURCE_WORKER_INIT_UNCHANGED')
                    import importlib
                    named = importlib.import_module(WORKER_MODULE)      # picklable by its importable name under spawn
                    k['worker_init_fn'] = functools.partial(named.child_init, str(obs.child_dir), obs.inject,
                                                            **({'shared': obs.shared.child_cfg} if getattr(obs.shared, 'KEY', None) == 's6c_shared' else {}))
                obs.loaders.append({'num_workers': k.get('num_workers'), 'prefetch_factor': k.get('prefetch_factor'),
                                    'timeout': k.get('timeout'), 'batch_size': k.get('batch_size', 'default'),
                                    'context': str(k.get('multiprocessing_context')), 'studies': len(dataset)})
                super().__init__(dataset, *a, **k)

            def __iter__(self):
                first = True
                for item in super().__iter__():
                    obs.begin(item, first)
                    first = False
                    yield item
                    obs.end()

        torch.load = load
        Net.__init__, Net.load_state_dict, Net.forward = init, load_state_dict, forward
        torch.utils.data.DataLoader = ObservedLoader
        self.mark = time.perf_counter()

    def uninstall(self):
        if not self.saved:
            return
        torch, Net = self.torch, self.CT.Net
        torch.load = self.saved['load']
        Net.__init__, Net.load_state_dict, Net.forward = self.saved['init'], self.saved['lsd'], self.saved['forward']
        torch.utils.data.DataLoader = self.saved['loader']
        self.saved = {}
        self.pool.shutdown(wait=True)

    # -- model binding
    def register(self, model, state_dict, strict, result):
        import inspect
        source = self.loaded.get(id(state_dict))
        require(source is not None, 'R6_STATE_FROM_OBSERVED_FILE')
        path, sha, _ = source
        name = Path(path).name
        index = len(self.models)
        pin = self.pins[index]
        a, k = getattr(model, '_r6flip_ctor', ((), {}))
        bound = inspect.signature(self.saved['init']).bind(model, *a, **k)
        bound.apply_defaults()
        args = {key: value for key, value in bound.arguments.items() if key != 'self'}
        args['state_spec'] = None if args.get('state_spec') is None else R6F.digest(args['state_spec'])
        entry = {'index': index, 'file': name, 'file_sha256': sha, 'strict': strict is True,
                 'missing_keys': list(getattr(result, 'missing_keys', [])), 'unexpected_keys': list(getattr(result, 'unexpected_keys', [])),
                 'constructor_args': args, 'state': R6F.module_digest(model), 'buffers': R6F.buffers_digest(model),
                 'constructor': R6F.constructor_digest(model), 'bin_names': [list(x) for x in model.bin_names],
                 'state_k': [int(x) for x in model.state_k]}
        require(entry['strict'] and not entry['missing_keys'] and not entry['unexpected_keys'], 'R6_STRICT_LOAD')
        require(name == pin['file'] and sha == pin['file_sha256'], 'R6_CHECKPOINT_FILE_PIN')
        require(entry['state'] == pin['state'] and entry['buffers'] == pin['buffers'] and entry['constructor'] == pin['constructor']
                and args == pin['constructor_args'], 'QUALIFIED_R6_FOLD_STATE')
        # fresh constructor + strict reload of the same file must reproduce every digest
        t0 = time.perf_counter()
        self.fresh_depth += 1
        try:
            fresh = self.CT.Net(*a, **k)
            reloaded = fresh.load_state_dict(self.saved['load'](path, map_location='cpu', weights_only=True), strict=True)
        finally:
            self.fresh_depth -= 1
        entry['reload'] = {'state': R6F.module_digest(fresh), 'buffers': R6F.buffers_digest(fresh),
                           'constructor': R6F.constructor_digest(fresh),
                           'strict': not reloaded.missing_keys and not reloaded.unexpected_keys}
        require(entry['reload']['strict'] and all(entry['reload'][x] == entry[x] for x in ('state', 'buffers', 'constructor')),
                'R6_FRESH_STRICT_RELOAD')
        entry['binding_seconds'] = time.perf_counter() - t0
        self.models.append({'module': model, 'fresh': fresh, 'entry': entry, 'checksum': None, 'sentinel': None})
        self.loaded.pop(id(state_dict), None)

    def model_index(self, model):
        for i, m in enumerate(self.models):
            if m['module'] is model:
                return i
        raise ValueError('R6_UNREGISTERED_MODEL_FORWARD')

    # -- per-study observation
    def abort_current(self, reason):
        """A study whose handling raised out of compact_infer's loop (the source then records a fallback)."""
        cur = self.current
        if cur is None:
            return
        cur.pop('hash', None)
        cur.pop('plain_meta', None)
        if cur.get('s6c_input'):
            cur['s6c_input'] = {k: v for k, v in cur['s6c_input'].items() if k not in ('images', 'meta')}
        cur.update(aborted=reason, handle_seconds=time.perf_counter() - cur.pop('t_begin'),
                   compile_check_calls=self.calls_other - cur.pop('calls_other_before'),
                   plain=cur['plain'] is not None, mirror=cur['mirror'] is not None, checksum_equal=None)
        self.studies.append(cur)
        self.current = None

    def begin(self, item, first):
        now = time.perf_counter()
        self.abort_current('superseded_by_next_item')
        if self.models and self.models[0]['checksum'] is None:
            for m in self.models:
                m['checksum'] = R6F.checksum(m['module'])
        study = {'id': item['uid'], 'error': item['error'], 'windows': int(len(item['images'])),
                 'seconds': list(item.get('seconds') or []), 'fluid': item.get('fluid'), 'fluid_s': item.get('fluid_s'),
                 'anchor': item.get('anchor'), 'side': item.get('side'), 'first_of_loader': bool(first),
                 'wait_seconds': now - self.mark, 'calls': [], 'plain': None, 'mirror': None, 'mirror_input_equal': None,
                 'calls_other_before': self.calls_other, 't_begin': now}
        if item['error'] is None and len(item['images']):
            images, meta = item['images'], item['meta']
            study['hash'] = self.pool.submit(lambda: {'render_windows': array_sha(images.numpy()), 'metadata': array_sha(meta.numpy())})
        if getattr(self.shared, 'KEY', None) == 's6c_shared':      # P9s: the members' windows travel in the same item
            s6 = {'images': item.get('s6c_images'), 'meta': item.get('s6c_meta'), 'error': item.get('s6c_error'), 'note': item.get('s6c_note')}
            if s6['error'] is None and s6['images'] is not None and len(s6['images']):
                im6, me6 = s6['images'], s6['meta']
                s6['input'] = {'render_windows': array_sha(im6.numpy()), 'metadata': array_sha(me6.numpy()), 'windows': int(len(im6))}
            study['s6c_input'] = s6
        self.current = study

    def on_scoring(self, model, img, meta, mask, chunk, out):
        torch = self.torch
        cur = self.current
        require(cur is not None, 'R6_FORWARD_OUTSIDE_STUDY')
        k = self.model_index(model)
        try:
            autocast = torch.is_autocast_enabled('cuda')
        except TypeError:
            autocast = torch.is_autocast_enabled()
        z, sl, bl = out
        if cur['plain'] is None:
            require(img.dim() == 5 and img.shape[0] == 1 and img.dtype == torch.uint8 and tuple(img.shape[2:]) == (self.slices, self.crop, self.crop)
                    and meta.shape[:2] == img.shape[:2] and bool(mask.all()) and chunk == 192, 'R6_SCORING_INPUT_SCHEMA')
            cur['plain'], cur['plain_meta'] = img, meta
        if img is cur['plain']:
            kind = 'plain'
        else:
            require(self.flip, 'R6_PLAIN_ONLY_UNEXPECTED_SCORING_INPUT')
            if cur['mirror'] is None:
                expected, expected_meta = R6F.mirror_inputs(cur['plain'], cur['plain_meta'])
                cur['mirror_input_equal'] = bool(torch.equal(img, expected) and torch.equal(meta, expected_meta))
                del expected, expected_meta
                require(cur['mirror_input_equal'], 'R6_FLIP_INPUT_TRANSFORM')
                cur['mirror'] = img
            require(img is cur['mirror'], 'R6_UNEXPECTED_SCORING_INPUT')
            kind = 'mirror'
        cur['calls'].append({'model': k, 'pass': kind, 'mode': 'fp16' if autocast else 'fp32',
                             'z': z.detach().float().cpu()[0], 'sl': sl.detach().float().cpu(), 'bl': bl.detach().float().cpu()})
        slot = self.models[k]
        if slot['sentinel'] is None and kind == 'plain' and autocast:
            t0 = time.perf_counter()
            fresh = slot.pop('fresh').eval().requires_grad_(False).to(img.device).to(memory_format=torch.channels_last)
            self.fresh_depth += 1
            try:
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    zf, slf, blf = fresh(img, meta, mask, chunk=chunk, return_aux=True)
            finally:
                self.fresh_depth -= 1
            zf = zf.float()
            dz = float((zf - z.float()).abs().max())
            dp = float((torch.sigmoid(zf.double()) - torch.sigmoid(z.double())).abs().max())
            slot['sentinel'] = {'study_ordinal': len(self.studies), 'finite': bool(torch.isfinite(zf).all()),
                                'max_abs_dlogit': dz, 'max_abs_dprob': dp, 'tolerance_dprob': RELOAD_DPROB_TOL,
                                'compiled_deployed': type(model.enc).__name__ == 'OptimizedModule',
                                'seconds': time.perf_counter() - t0}
            del fresh, zf, slf, blf
            torch.cuda.empty_cache()
            require(slot['sentinel']['finite'] and dp <= RELOAD_DPROB_TOL, 'R6_FRESH_RELOAD_SAME_INPUT')
        if self.shared is not None and kind == 'plain' and k == len(self.pins) - 1:
            self.shared.on_study(self, cur, img, meta, mask, chunk)

    def finish_or_abort(self):
        """At main() return: a study compact_infer fully scored before leaving its loop (deadline break) is closed like a
        normal boundary; anything else is aborted as before."""
        cur = self.current
        if cur is None:
            return
        passes = ('plain', 'mirror') if self.flip else ('plain',)
        done = {(c['model'], c['pass']) for c in cur['calls']}
        if cur['error'] is None and self.models and all((k, p) in done for k in range(len(self.models)) for p in passes):
            self.end()
        else:
            self.abort_current('main_returned_without_boundary')

    def end(self):
        cur = self.current
        require(cur is not None, 'R6_STUDY_BOUNDARY')
        now = time.perf_counter()
        cur['handle_seconds'] = now - cur.pop('t_begin')
        cur['compile_check_calls'] = self.calls_other - cur.pop('calls_other_before')
        if 'hash' in cur:
            cur['input'] = cur.pop('hash').result()
        cur.pop('plain_meta', None)
        if cur.get('s6c_input'):
            cur['s6c_input'] = {k: v for k, v in cur['s6c_input'].items() if k not in ('images', 'meta')}
        cur['plain'] = cur['plain'] is not None
        cur['mirror'] = cur['mirror'] is not None
        cur['checksum_equal'] = all(R6F.checksum(m['module']) == m['checksum'] for m in self.models) if self.models else None
        require(cur['checksum_equal'] is not False, 'R6_PARAMETER_BUFFER_MUTATION')
        self.studies.append(cur)
        self.current = None
        self.mark = now

    @staticmethod
    def observation(study, models, flip=True):
        """The deployed (last) plain and mirrored call of every fold model (plain only without flip TTA)."""
        last = {}
        for c in study['calls']:
            last[(c['model'], c['pass'])] = c
        n = len(models)
        passes = ('plain', 'mirror') if flip else ('plain',)
        require(all((k, p) in last for k in range(n) for p in passes)
                and all(c['pass'] in passes for c in study['calls']), 'R6_FOLD_PASS_COVERAGE')
        if not flip:
            entry = models[0]['entry']
            require(all(m['entry']['state_k'] == entry['state_k'] and m['entry']['bin_names'] == entry['bin_names'] for m in models),
                    'R6_FOLD_HEADS')
            return {'plain': [last[(k, 'plain')]['z'] for k in range(n)], 'mirror': None,
                    'plain_aux': [(last[(k, 'plain')]['sl'], last[(k, 'plain')]['bl']) for k in range(n)], 'mirror_aux': None,
                    'state_k': entry['state_k'], 'bin_names': entry['bin_names'],
                    'fp32_rescues': sum(1 for c in study['calls'] if c['mode'] == 'fp32'), 'calls': len(study['calls'])}
        entry = models[0]['entry']
        require(all(m['entry']['state_k'] == entry['state_k'] and m['entry']['bin_names'] == entry['bin_names'] for m in models),
                'R6_FOLD_HEADS')
        return {'plain': [last[(k, 'plain')]['z'] for k in range(n)], 'mirror': [last[(k, 'mirror')]['z'] for k in range(n)],
                'plain_aux': [(last[(k, 'plain')]['sl'], last[(k, 'plain')]['bl']) for k in range(n)],
                'mirror_aux': [(last[(k, 'mirror')]['sl'], last[(k, 'mirror')]['bl']) for k in range(n)],
                'state_k': entry['state_k'], 'bin_names': entry['bin_names'],
                'fp32_rescues': sum(1 for c in study['calls'] if c['mode'] == 'fp32'),
                'calls': len(study['calls'])}


def read_predictions(path, labels):
    import csv
    rows = {}
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        require(next(reader) == ['StudyInstanceUID'] + labels, 'R6_PREDICTIONS_SCHEMA')
        for row in reader:
            require(len(row) == 13 and row[0] not in rows, 'R6_PREDICTIONS_ROW')
            rows[row[0]] = [float(x) for x in row[1:]]
    return rows


def fluid_status_ok(record):
    status = (record or {}).get('status')
    return isinstance(status, str) and (status == 'ok' or status in FLUID_MISSING or status.startswith(FLUID_OK_PREFIX))


def summarize(cfg, obs, ci_receipt, predictions, labels, spec, fluid_doc, state_doc):
    """Per-study records and the complete assigned-population ledger arrays (worker-local checks only)."""
    import numpy as np
    ids = cfg['ids']
    flip = flip_enabled(cfg)
    fallback = ci_receipt.get('fallback') or {}
    require(set(predictions) <= set(ids) and set(fallback) <= set(ids) and not set(predictions) & set(fallback)
            and set(predictions) | set(fallback) == set(ids), 'R6_SCORED_OR_FALLBACK_PARTITION')
    by_id = {}
    for s in obs.studies:
        require(s['id'] not in by_id, 'R6_STUDY_SEEN_ONCE')
        by_id[s['id']] = s
    state_fallback = set((ci_receipt.get('state_stacker') or {}).get('stack_fallback') or {})
    fluid_fallback = set((ci_receipt.get('fluid_stacker') or {}).get('stack_fallback') or {})
    probs, statuses, logits, records = [], [], [], []
    for ordinal, sid in enumerate(ids):
        s = by_id.get(sid)
        if sid in predictions:
            require(s is not None and not s.get('aborted') and s['error'] is None and s['windows'] > 0 and s['plain'] and s['mirror'] == flip,
                    'R6_SCORED_STUDY_OBSERVED')
            o = Observer.observation(s, obs.models, flip)
            fall = [n for n, hit in (("Baker's", sid in state_fallback), ('Effusion', sid in fluid_fallback)) if hit]
            p_recon, detail = R6F.reconstruct(o, s['fluid'], fluid_doc, state_doc, spec, labels, stack_fallback=fall, flip=flip)
            p = np.asarray(predictions[sid], dtype=np.float64)
            err = float(np.max(np.abs(p - p_recon)))
            require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() and err <= RECONSTRUCTION_ATOL, 'R6_DEPLOYED_RECONSTRUCTION')
            probs.append(p)
            statuses.append('genuine')
            logits.append(np.asarray(detail['logits'], dtype=np.float32))
            production = s['handle_seconds'] + (0.0 if s['first_of_loader'] else s['wait_seconds'])
            records.append({'id': sid, 'ordinal': ordinal, 'status': 'genuine', 'windows': s['windows'], 'input': s['input'],
                            'source_seconds': s['seconds'], 'wait_seconds': s['wait_seconds'], 'handle_seconds': s['handle_seconds'],
                            'first_of_loader': s['first_of_loader'], 'compile_check_calls': s['compile_check_calls'],
                            'production_seconds': production, 'fluid': {k: (s['fluid'] or {}).get(k) for k in
                                                                         ('status', 'eff_sag_p99_open', 'bak_sag_p99_open', 'n_sag_fluid_series')},
                            'fluid_status_ok': fluid_status_ok(s['fluid']), 'anchor': s['anchor'], 'side': s['side'],
                            'mirror_input_equal': s['mirror_input_equal'], 'checksum_equal': s['checksum_equal'],
                            'calls': o['calls'], 'fp32_rescues': o['fp32_rescues'], 'reconstruction_max_abs': err,
                            'stack_fallback': fall, 'model_probabilities': detail['model_probabilities'],
                            'stackers': detail['stackers'], 'probabilities': p.tolist(), 'logits': detail['logits']})
        else:
            probs.append(np.full(12, np.nan))
            statuses.append('data_local_failure')
            logits.append(np.full((4, 12), np.nan, dtype=np.float32))
            records.append({'id': sid, 'ordinal': ordinal, 'status': 'data_local_failure', 'phase': 'r6_compact_infer',
                            'failure_code': str(fallback[sid]).split('(')[0][:80], 'observed_item': s is not None,
                            'deadline': str(fallback[sid]) == 'not reached before the deadline',
                            'item_error': None if s is None else s['error'], 'windows': None if s is None else s['windows'],
                            'production_seconds': 0.0 if s is None else s['handle_seconds']})
    return probs, statuses, logits, records


def load_stackers(mount, manifest, labels):
    from knee import public_r7_stackers as stackers
    spec = json.loads((mount / manifest['state_spec']).read_bytes())
    fluid_doc = stackers.load_fluid((mount / manifest['fluid_stacker']).read_bytes(), labels)
    layout = stackers.state_layout(spec, labels)
    state_doc = stackers.load_state((mount / manifest['state_stacker']).read_bytes(), labels, layout,
                                    hashlib.md5((mount / manifest['state_spec']).read_bytes()).hexdigest())
    pair = stackers.check_pair(fluid_doc, state_doc)
    require(tuple(pair['fluid']) == stackers.FLUID_ADOPTED and tuple(pair['state']) == stackers.STATE_ADOPTED
            and not state_doc['needs_volume'], 'R6_ADOPTED_STACKERS')
    return spec, fluid_doc, state_doc, pair


def profile_module(name):
    if name == 'r7':
        from knee import public_r7_profile as P
    else:
        from knee import public_r6_profile as P
    return P


def check_compact_receipt(ci, manifest):
    """The captured worker's own receipt must name exactly the pinned bundle, sources and switches."""
    from knee.public_r6_profile import SOURCE_SHA256
    try:                                   # the R7 profile ships only with the R7 variants (P6*); every other build runs the R6 bundle
        from knee import public_r7_profile
        r7_bundle = public_r7_profile.BUNDLE_ID
    except ImportError:
        r7_bundle = None
    ASSET_SHA256 = profile_module('r7' if r7_bundle is not None and manifest.get('bundle_id') == r7_bundle else 'r6').ASSET_SHA256
    require(ci.get('status') == 'PASS' and ci.get('timed_out') is False and ci.get('amp') == 'fp16'
            and ci.get('models') == manifest['models'] and ci.get('extent_models') == 4 and ci.get('syn_decomp_models') == 4
            and ci.get('spatial_models') == 4 and ci.get('trainer_md5') == manifest['trainer_md5']
            and ci.get('anchor_weights_sha256') == manifest['anchor_weights_sha256']
            and (ci.get('side_check') or {}).get('model_sha256') == ASSET_SHA256['side_cls.pt']
            and (ci.get('state_stacker') or {}).get('sha256') == ASSET_SHA256['state_stacker.json']
            and (ci.get('fluid_stacker') or {}).get('sha256') == ASSET_SHA256['fluid_stacker.json']
            and len(ci.get('sha256', {})) == 6 and all(ci['sha256'][n] == SOURCE_SHA256[n] for n in ci['sha256']),
            'R6_SOURCE_RECEIPT')


def execute(cfg, out, r, torch, CT, import_ci, manifest, spec, fluid_doc, state_doc, pair, check_ci=check_compact_receipt,
            shared=None):
    """Observe one compact_infer main() over cfg['ids'] and turn it into a bound, complete ledger and receipt."""
    import resource
    from knee.public_s6_recovery_runtime import save_ledger
    labels = R6F.LABELS
    obs = Observer(cfg, out, torch, CT)
    obs.shared = shared
    obs.install()
    try:
        CI = import_ci()
        assignment = out / 'ids.json'
        assignment.write_text(json.dumps(cfg['ids']))
        args = worker_args(cfg, assignment, out / 'compact')
        r.update(worker_args=args, phase='compact_infer')
        save(out / 'receipt.json', r)
        argv = sys.argv
        sys.argv = [str(Path(CI.__file__)), *args]
        main_start = time.perf_counter()
        try:
            CI.main()
        finally:
            sys.argv = argv
    finally:
        obs.uninstall()
    r['main_seconds'] = time.perf_counter() - main_start
    obs.finish_or_abort()
    if shared is not None:                 # R7 / members ledger and receipt first: an R6 summary failure must not lose them
        r[getattr(shared, 'KEY', 'r7_shared')] = shared.finalize(obs, r)
    r['aborted_observations'] = sum(1 for s in obs.studies if s.get('aborted'))
    ci = json.loads((out / 'compact' / 'receipt.json').read_bytes())
    r['compact_receipt_sha256'] = file_sha(out / 'compact' / 'receipt.json')
    require(len(obs.models) == len(manifest['models']) == 4 and [m['entry']['file'] for m in obs.models] == manifest['models'],
            'R6_FOUR_FOLDS_BOUND')
    if flip_enabled(cfg):
        perm, bin_perm = R6F.mirror_perm(labels, spec, obs.models[0]['entry']['bin_names'])
        flip = ci.get('flip_tta') or {}
        require(flip.get('perm') == perm and flip.get('bin_perm') == bin_perm and flip.get('mcl') == 'plain pass only', 'R6_FLIP_PERMUTATION')
        flip_receipt = {'perm': perm, 'bin_perm': bin_perm, 'receipt': flip}
    else:
        require('flip_tta' not in ci, 'PLAIN_PASS_NO_FLIP_RECEIPT')
        flip_receipt = {'enabled': False}
    check_ci(ci, manifest)
    predictions = read_predictions(out / 'compact' / 'predictions.csv', labels)
    probs, statuses, logits, records = summarize(cfg, obs, ci, predictions, labels, spec, fluid_doc, state_doc)
    ledger = out / 'predictions.npz'
    save_ledger(ledger, cfg['ids'], probs, statuses, logits=logits)
    after = [{'state': R6F.module_digest(m['module']), 'buffers': R6F.buffers_digest(m['module'])} for m in obs.models]
    require(all(a['state'] == m['entry']['state'] and a['buffers'] == m['entry']['buffers'] for a, m in zip(after, obs.models)),
            'R6_STATE_AFTER')
    children = [json.loads(p.read_bytes()) for p in sorted((out / 'children').glob('child-*.json'))]
    side_pin, anatomy_pin = cfg['r6']['cpu_pins']['side'], cfg['r6'].get('anatomy_state')
    require(children and all(ch['before'] is not None and not ch['changed'] for ch in children), 'R6_LOADER_CHILD_STATE')
    require(all(ch['before']['side']['state'] == side_pin['state'] and ch['before']['side']['constructor'] == side_pin['constructor']
                for ch in children), 'QUALIFIED_R6_SIDE_NET')
    require(len({ch['before']['anatomy']['state'] for ch in children}) == 1
            and (anatomy_pin is None or children[0]['before']['anatomy']['state'] == anatomy_pin), 'QUALIFIED_R6_ANATOMY')
    compile_receipt = {k: (ci.get('compile') or {}).get(k) for k in ('requested', 'active', 'fallback', 'decision', 'study_dprob',
                                                                     'check_seconds', 'max_abs_dprob', 'graphs')}
    genuine = [x for x in records if x['status'] == 'genuine']
    cuda = torch.cuda.is_available()
    r.update(status='PASS', phase='complete', completed=len(records), records=records,
             genuine=statuses.count('genuine'), failed=statuses.count('data_local_failure'),
             zero_failed=all(s == 'genuine' for s in statuses), zero_imputed=True,
             fp32_recoveries=sum(x['fp32_rescues'] for x in genuine),
             folds=[dict(m['entry'], sentinel=m['sentinel'], checksum=m['checksum']) for m in obs.models],
             state_before=R6F.digest([m['entry']['state'] for m in obs.models]),
             state_after=R6F.digest([a['state'] for a in after]),
             flip=flip_receipt, compile=compile_receipt,
             loaders=obs.loaders, children=children, compile_check_forward_calls=obs.calls_other,
             compact_stats=ci.get('stats'), compact_seconds=ci.get('seconds'),
             fallback_reasons={k: str(v)[:200] for k, v in (ci.get('fallback') or {}).items()},
             stackers={'fluid_sha256': fluid_doc['sha256'], 'state_sha256': state_doc['sha256'], 'adopted': pair,
                       'state_fallback': sorted((ci.get('state_stacker') or {}).get('stack_fallback') or {}),
                       'fluid_fallback': sorted((ci.get('fluid_stacker') or {}).get('stack_fallback') or {})},
             ledger_sha256=file_sha(ledger), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
             peak_cuda_bytes=torch.cuda.max_memory_allocated() if cuda else 0,
             peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else 0)
    save(out / 'receipt.json', r)
    return r


R7_SHARED_COMPILE_TOL = 1e-3     # compact_infer --compile-tol default / R7 recipe tol_prob
R7_CUT_SAFETY = 1.3              # R7 stops when the remaining studies at the R6-only pace x 1.3 would pass the deadline
R7_CUT_MIN_STUDIES = 3


class SharedR7:
    """P6s: R7 v1 fold models on R6's rendered windows inside the R6 plain worker (see the module docstring)."""
    KEY = 'r7_shared'

    def __init__(self, cfg, torch, CT, device=None):
        self.cfg, self.torch, self.CT = cfg['r7_shared'], torch, CT
        self.r6_manifest = (cfg.get('r6') or {}).get('manifest')
        self.ids, self.board, self.pass_name = list(cfg['ids']), cfg.get('board'), cfg.get('pass')
        self.code = cfg.get('code')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.deadline = float(self.cfg.get('deadline_epoch') or 0)
        self.out = Path(self.cfg['out']) / 'r7'
        self.models, self.entries, self.eager, self.checksums = [], [], {}, []
        self.compile = {'requested': bool(self.cfg['manifest'].get('compile')), 'active': False, 'fallback': None, 'decision': None,
                        'study_dprob': None, 'warmup_shapes': [], 'warmup_max_dprob': None, 'check_seconds': None}
        self.active, self.cut = True, None
        self.rows = {}
        self.csv = None

    def load(self):
        """Pins first (file SHA256, manifest, strict constructor/state/buffers vs the CPU pins), then device, compile, warm-up."""
        import inspect
        torch, CT = self.torch, self.CT
        from knee import public_r7_profile as R7P
        from knee import public_r7_stackers as stackers
        t0 = time.perf_counter()
        mount = Path(self.cfg['mount'])
        profile = R7P.validate_r7_assets(mount, source_dir=self.code) if self.code else None
        manifest = json.loads((mount / 'manifest.json').read_bytes())
        require(manifest == self.cfg['manifest'] and file_sha(mount / 'manifest.json') == R7P.MANIFEST_SHA256, 'R7_SHARED_EXACT_MANIFEST')
        self.manifest = manifest
        if self.r6_manifest is not None:     # R7 must see exactly R6's rendered windows: same render/crop recipe
            require(all(manifest.get(k) == self.r6_manifest.get(k) for k in ('size', 'crop', 'slices', 'views', 'phys_mm', 'eval_stride',
                                                                             'backbone', 'anchor_weights_sha256', 'side_model')),
                    'R7_SHARED_SAME_RENDER_RECIPE')
        self.spec = json.loads((mount / manifest['state_spec']).read_bytes())
        self.fluid_doc = stackers.load_fluid((mount / manifest['fluid_stacker']).read_bytes(), R6F.LABELS)
        layout = stackers.state_layout(self.spec, R6F.LABELS)
        self.state_doc = stackers.load_state((mount / manifest['state_stacker']).read_bytes(), R6F.LABELS, layout,
                                             hashlib.md5((mount / manifest['state_spec']).read_bytes()).hexdigest())
        pair = stackers.check_pair(self.fluid_doc, self.state_doc)
        require(tuple(pair['fluid']) == stackers.FLUID_ADOPTED and tuple(pair['state']) == stackers.STATE_ADOPTED
                and not self.state_doc['needs_volume'], 'R7_SHARED_ADOPTED_STACKERS')
        pins = self.cfg['cpu_pins']['folds']
        require([p['file'] for p in pins] == manifest['models'] and len(pins) == 4, 'R7_SHARED_FOUR_FOLDS')
        for name, pin in zip(manifest['models'], pins):
            path = mount / name
            sha = file_sha(path)
            sd = torch.load(path, map_location='cpu', weights_only=True)
            kw = {}
            if any(k == 'sp_q' or k.endswith('.sp_q') for k in sd):
                kw['spatial'] = True
            if any(k.split('.')[-1] in ('ext_u', 'ext_view', 'ext_gamma') for k in sd):
                kw['extent'] = True
            if any(k == 'syn_beta' or k.endswith('.syn_beta') for k in sd):
                kw['syn_decomp'] = True
            ctor = dict(freeze_blocks=manifest['freeze_blocks'], crop=manifest['crop'], state_spec=self.spec, in_chans=manifest['slices'], **kw)
            m = CT.Net(manifest['backbone'], None, **ctor)          # compact_infer load_group's exact call
            res = m.load_state_dict(sd, strict=True)
            del sd
            bound = inspect.signature(CT.Net.__init__).bind(m, manifest['backbone'], None, **ctor)
            bound.apply_defaults()
            args = {k: v for k, v in bound.arguments.items() if k != 'self'}
            args['state_spec'] = R6F.digest(args['state_spec'])
            entry = {'file': name, 'file_sha256': sha, 'strict': not res.missing_keys and not res.unexpected_keys,
                     'constructor_args': args, 'state': R6F.module_digest(m), 'buffers': R6F.buffers_digest(m),
                     'constructor': R6F.constructor_digest(m), 'bin_names': [list(x) for x in m.bin_names],
                     'state_k': [int(x) for x in m.state_k]}
            require(entry['strict'] and sha == pin['file_sha256'] and all(entry[k] == pin[k] for k in ('state', 'buffers', 'constructor',
                                                                                                       'constructor_args')),
                    'QUALIFIED_R7_SHARED_FOLD_STATE')
            self.entries.append(entry)
            self.models.append(m.eval().requires_grad_(False).to(self.device).to(memory_format=torch.channels_last))
        require(all(e['state_k'] == self.entries[0]['state_k'] and e['bin_names'] == self.entries[0]['bin_names'] for e in self.entries),
                'R7_SHARED_FOLD_HEADS')
        if self.compile['requested']:
            import torch._dynamo as _dynamo
            for name_, lim_ in (('cache_size_limit', 64), ('recompile_limit', 64), ('accumulated_cache_size_limit', 2048),
                                ('accumulated_recompile_limit', 2048)):        # compact_infer's own settings (R6 + R7 = 8 encoders)
                if hasattr(_dynamo.config, name_):
                    setattr(_dynamo.config, name_, max(lim_, int(getattr(_dynamo.config, name_))))
            for m in self.models:
                self.eager[id(m)] = m.enc
                m.enc = torch.compile(self.eager[id(m)], dynamic=False, mode='default')
            self.compile['active'] = True
            self.warmup()
        self.checksums = [R6F.checksum(m) for m in self.models]
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'compact').mkdir(exist_ok=True)
        self.csv = open(self.out / 'compact' / 'predictions.csv', 'w', newline='', encoding='utf-8')
        self.csv.write(','.join(['StudyInstanceUID'] + R6F.LABELS) + '\n')
        self.csv.flush()
        return {'profile': profile, 'folds': len(self.models), 'compile': dict(self.compile), 'seconds': time.perf_counter() - t0,
                'deadline_epoch': self.deadline}

    def _forward(self, m, img, meta, mask, chunk, mode, eager=False):
        torch = self.torch
        enc = m.enc
        if eager and id(m) in self.eager:
            m.enc = self.eager[id(m)]
        try:
            if mode == 'fp32' or img.device.type != 'cuda':      # also leaves compact_infer's enclosing fp16 autocast
                with torch.autocast(img.device.type, enabled=False):
                    z, sl, bl = m(img, meta, mask, chunk=chunk, return_aux=True)
            else:
                with torch.autocast('cuda', dtype=torch.float16):
                    z, sl, bl = m(img, meta, mask, chunk=chunk, return_aux=True)
        finally:
            m.enc = enc
        return z.float(), sl.float(), bl.float()

    def warmup(self):
        """Every encoder batch shape (32..192 windows) compiled before the first study; compiled vs eager only as a sanity bound."""
        torch = self.torch
        gen = torch.Generator().manual_seed(20261020)
        crop, slices = int(self.manifest['crop']), int(self.manifest['slices'])
        worst = 0.
        with torch.inference_mode():
            for s in range(32, 193, 32):
                img = torch.randint(0, 256, (1, s, slices, crop, crop), generator=gen, dtype=torch.uint8).to(self.device)
                meta = torch.zeros((1, s, 9), dtype=torch.float32, device=self.device)
                mask = torch.ones((1, s), dtype=torch.bool, device=self.device)
                for m in self.models:
                    zc = self._forward(m, img, meta, mask, 192, 'fp16')[0]
                    ze = self._forward(m, img, meta, mask, 192, 'fp16', eager=True)[0]
                    require(bool(torch.isfinite(zc).all()) == bool(torch.isfinite(ze).all()), 'R7_SHARED_WARMUP_FINITENESS')
                    worst = max(worst, float((torch.sigmoid(zc.double()) - torch.sigmoid(ze.double())).abs().max()))
                self.compile['warmup_shapes'].append(s)
        self.compile['warmup_max_dprob'] = worst
        if worst > .05:                                   # compact_infer COMPILE_SANITY_DP: a broken compiled kernel
            self.set_eager('warm-up compiled vs eager |dp| %.3g > 0.05' % worst)

    def set_eager(self, reason):
        for m in self.models:
            if id(m) in self.eager:
                m.enc = self.eager[id(m)]
        self.eager.clear()
        self.compile.update(active=False, fallback=str(reason)[:400])

    def check(self, img, meta, mask, chunk):
        """compact_infer's study-level gate on the first real study: compiled vs eager fold-mean probability within 1e-3."""
        torch = self.torch
        t0 = time.perf_counter()
        zc = torch.stack([self._forward(m, img, meta, mask, chunk, 'fp16')[0][0] for m in self.models])
        ze = torch.stack([self._forward(m, img, meta, mask, chunk, 'fp16', eager=True)[0][0] for m in self.models])
        d = float((torch.sigmoid(zc.mean(0).double()) - torch.sigmoid(ze.mean(0).double())).abs().max())
        self.compile['study_dprob'] = d
        self.compile['decision'] = 'compiled' if d <= R7_SHARED_COMPILE_TOL else 'eager'
        if d > R7_SHARED_COMPILE_TOL:
            self.set_eager('study check |dp| %.3g > %.0e' % (d, R7_SHARED_COMPILE_TOL))
        self.compile['check_seconds'] = time.perf_counter() - t0

    def r6_only_pace(self, obs):
        rows = [(s, self.rows.get(s['id'], {})) for s in obs.studies if not s.get('aborted')]
        paces = [s['handle_seconds'] + (0. if s['first_of_loader'] else s['wait_seconds']) - x.get('r7_seconds', 0.) for s, x in rows]
        return sum(paces) / len(paces) if len(paces) >= R7_CUT_MIN_STUDIES else None

    def should_cut(self, obs, now=None):
        """R7 last: stop R7 when R6 alone could no longer finish the remaining studies before the deadline (x 1.3)."""
        if not self.active or not self.deadline:
            return False
        pace = self.r6_only_pace(obs)
        if pace is None:
            return False
        remaining = len(self.ids) - len(obs.studies)
        now = time.time() if now is None else now
        if now + remaining * max(pace, 0.) * R7_CUT_SAFETY > self.deadline:
            self.active = False
            self.cut = {'after_studies': len(obs.studies), 'remaining': remaining, 'r6_only_pace_seconds': pace,
                        'epoch': now, 'deadline_epoch': self.deadline}
            return True
        return False

    def on_study(self, obs, cur, img, meta, mask, chunk):
        """Called once per study right after R6's last plain fold forward, on the same rendered windows."""
        torch = self.torch
        if 'r7' in cur:
            return
        sid = cur['id']
        self.should_cut(obs)
        if not self.active:
            cur['r7'] = None
            self.rows[sid] = {'status': 'deadline'}
            return
        t0 = time.perf_counter()
        obs.fresh_depth += 1                  # R7 forwards are not R6 scoring calls for the observer
        try:
            with torch.inference_mode():
                if self.compile['active'] and self.compile['decision'] is None:
                    self.check(img, meta, mask, chunk)
                zs, aux, rescues = [], [], 0
                for m in self.models:
                    z, sl, bl = self._forward(m, img, meta, mask, chunk, 'fp16')
                    if not bool(torch.isfinite(z).all()):          # compact_infer's fp32 rescue, eager encoder
                        z, sl, bl = self._forward(m, img, meta, mask, chunk, 'fp32', eager=True)
                        rescues += 1
                    zs.append(z.detach().cpu()[0])
                    aux.append((sl.detach().cpu(), bl.detach().cpu()))
            checksum_equal = all(R6F.checksum(m) == c for m, c in zip(self.models, self.checksums))
        finally:
            obs.fresh_depth -= 1
        require(checksum_equal, 'R7_SHARED_PARAMETER_BUFFER_MUTATION')
        e = self.entries[0]
        p, detail = R6F.reconstruct({'plain': zs, 'mirror': None, 'plain_aux': aux, 'mirror_aux': None, 'state_k': e['state_k'],
                                     'bin_names': e['bin_names']}, cur['fluid'], self.fluid_doc, self.state_doc, self.spec, R6F.LABELS,
                                    flip=False)
        finite = bool(all(math.isfinite(float(x)) and 0. <= float(x) <= 1. for x in p))
        seconds = time.perf_counter() - t0
        row = {'status': 'genuine' if finite else 'nonfinite', 'probabilities': [float(x) for x in p], 'logits': detail['logits'],
               'model_probabilities': detail['model_probabilities'], 'stackers': detail['stackers'], 'fp32_rescues': rescues,
               'checksum_equal': checksum_equal, 'r7_seconds': seconds, 'windows': int(img.shape[1])}
        self.rows[sid] = row
        cur['r7'] = True
        if finite:
            self.csv.write(','.join([sid] + ['%.9g' % x for x in row['probabilities']]) + '\n')
            self.csv.flush()

    def finalize(self, obs, r6_receipt):
        """R7 ledger (assigned order) and receipt; studies without R7 output are deadline/data-local rows."""
        import numpy as np
        import resource
        from knee.public_s6_recovery_runtime import save_ledger
        if self.csv is not None:
            self.csv.close()
        studies = {s['id']: s for s in obs.studies}
        probs, statuses, logits, records = [], [], [], []
        for ordinal, sid in enumerate(self.ids):
            x, s = self.rows.get(sid), studies.get(sid)
            if x is not None and x['status'] == 'genuine' and s is not None:
                probs.append(np.asarray(x['probabilities'], dtype=np.float64))
                statuses.append('genuine')
                logits.append(np.asarray(x['logits'], dtype=np.float32))
                records.append({'id': sid, 'ordinal': ordinal, 'status': 'genuine', 'windows': x['windows'], 'input': s.get('input'),
                                'r7_seconds': x['r7_seconds'], 'production_seconds': x['r7_seconds'], 'fp32_rescues': x['fp32_rescues'],
                                'checksum_equal': x['checksum_equal'], 'fluid_status_ok': fluid_status_ok(s['fluid']),
                                'stackers': x['stackers'], 'model_probabilities': x['model_probabilities'],
                                'probabilities': x['probabilities'], 'logits': x['logits'], 'calls': 4, 'shared_decode': True,
                                'r6_aborted': bool(s.get('aborted'))})
            else:
                probs.append(np.full(12, np.nan))
                statuses.append('data_local_failure')
                logits.append(np.full((4, 12), np.nan, dtype=np.float32))
                records.append({'id': sid, 'ordinal': ordinal, 'status': 'data_local_failure', 'phase': 'r7_shared',
                                'failure_code': 'r7 cut before the deadline' if (x or {}).get('status') == 'deadline' else
                                'nonfinite r7 output' if x is not None else 'r6 study not scored',
                                'deadline': x is None or x.get('status') == 'deadline', 'production_seconds': 0.})
        ledger = self.out / 'predictions.npz'
        save_ledger(ledger, self.ids, probs, statuses, logits=logits)
        after = [{'state': R6F.module_digest(m), 'buffers': R6F.buffers_digest(m)} for m in self.models]
        require(all(a['state'] == e['state'] and a['buffers'] == e['buffers'] for a, e in zip(after, self.entries)), 'R7_SHARED_STATE_AFTER')
        cuda = self.torch.cuda.is_available()
        rec = {'status': 'PASS', 'version': VERSION, 'arm': 'r7', 'slot': 'r7-shared', 'pass': self.pass_name, 'board': self.board,
               'assigned': self.ids, 'completed': len(records), 'records': records, 'genuine': statuses.count('genuine'),
               'failed': statuses.count('data_local_failure'), 'zero_failed': all(s == 'genuine' for s in statuses), 'zero_imputed': True,
               'fp32_recoveries': sum(x.get('fp32_rescues', 0) for x in records), 'folds': self.entries,
               'state_before': R6F.digest([e['state'] for e in self.entries]), 'state_after': R6F.digest([a['state'] for a in after]),
               'flip': {'enabled': False}, 'compile': dict(self.compile), 'cut': self.cut, 'environment': r6_receipt.get('environment'),
               'shared_with': 'r6 production worker (same process, same rendered windows)', 'ledger_sha256': file_sha(ledger),
               'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               'peak_cuda_bytes': self.torch.cuda.max_memory_allocated() if cuda else 0}
        save(self.out / 'receipt.json', rec)
        return {'status': 'PASS', 'genuine': rec['genuine'], 'failed': rec['failed'], 'cut': self.cut, 'compile': rec['compile'],
                'receipt': str(self.out / 'receipt.json')}


def r6_worker(config_path):
    """Kaggle board worker: environment, staged inputs, source closure, assets and pins first; then execute()."""
    cfg = json.loads(Path(config_path).read_bytes())
    arm, name, flip = compact_arm(cfg), profile_name(cfg), flip_enabled(cfg)
    out = Path(cfg['out']) / arm
    out.mkdir()
    start = time.perf_counter()
    r = {'status': 'RUNNING', 'version': VERSION, 'arm': arm, 'slot': 'r6flip' if flip else name + '-plain', 'pass': cfg['pass'],
         'board': cfg['board'], 'assigned': cfg['ids'], 'completed': 0, 'records': []}
    save(out / 'receipt.json', r)
    try:
        import torch
        from knee.public_r6_profile import SOURCE_SHA256
        P = profile_module(name)
        MANIFEST_SHA256 = P.MANIFEST_SHA256
        from knee.public_s6_qualification import verify_sources
        from knee.public_s6_recovery_entry import board_environment, verify_staged
        r['environment'] = board_environment(torch, cfg)
        require(sys.version_info[:2] == (3, 12) and torch.cuda.get_device_name(0) == 'Tesla T4', 'R6_CUDA_ENVIRONMENT')
        require(all(os.environ.get(k) == v for k, v in PROBE_DUAL_ENV.items()), 'R6_PROBE_DUAL_ENV')
        verify_staged(cfg['data'], cfg['data_ids'], cfg['data_binding'])
        verify_sources(cfg['code'], cfg['source_pins'])
        r['profile'] = (P.validate_r7_assets if name == 'r7' else P.validate_r6_assets)(cfg['r6_mount'], source_dir=cfg['code'])
        mount = Path(cfg['r6_mount'])
        manifest = json.loads((mount / 'manifest.json').read_bytes())
        require(manifest == cfg['r6']['manifest'] and hashlib.sha256((mount / 'manifest.json').read_bytes()).hexdigest() == MANIFEST_SHA256,
                'R6_EXACT_MANIFEST')
        require(sorted(p.name for p in mount.glob(manifest['arm'] + '_s*_f*.pt')) == manifest['models'], 'R6_FOLD_GLOB_CLOSURE')
        require(file_sha(cfg['anatomy']) == manifest['anchor_weights_sha256'], 'R6_ANATOMY_FILE')
        code = Path(cfg['code'])
        require(hashlib.md5((code / 'compact_train.py').read_bytes()).hexdigest() == manifest['trainer_md5']
                and manifest['src_sha256']['compact_train.py'] == SOURCE_SHA256['compact_train.py'], 'R6_TRAINER_PIN')
        require('compact_infer' not in sys.modules and 'compact_train' not in sys.modules, 'FRESH_COMPACT_NAMESPACE')
        sys.path.insert(0, str(code))
        import compact_train as CT
        require(Path(CT.__file__).resolve() == (code / 'compact_train.py').resolve() and CT.LABELS == R6F.LABELS, 'R6_EXACT_TRAINER_IMPORT')
        spec, fluid_doc, state_doc, pair = load_stackers(mount, manifest, R6F.LABELS)
        shared = None
        if cfg.get('r7_shared'):
            require(name == 'r6' and arm == 'r6' and not flip and cfg['pass'] == 'production', 'R7_SHARED_ON_R6_PLAIN_PRODUCTION_ONLY')
            shared = SharedR7(cfg, torch, CT)
            r['r7_shared_setup'] = shared.load()
            save(out / 'receipt.json', r)
        elif cfg.get('s6c_shared'):          # P9s: our members from the same decode, every pass (sentinels give the repeat gates)
            from knee import public_s6c_shared as SH
            require(name == 'r6' and arm == 'r6' and not flip, 'S6C_SHARED_ON_R6_PLAIN_ONLY')
            shared = SH.SharedS6(cfg, torch)
            r['s6c_shared_setup'] = shared.load()
            save(out / 'receipt.json', r)

        def import_ci():
            import compact_infer as CI
            require(Path(CI.__file__).resolve() == (code / 'compact_infer.py').resolve()
                    and file_sha(CI.__file__) == SOURCE_SHA256['compact_infer.py'], 'R6_EXACT_WORKER_IMPORT')
            return CI
        execute(cfg, out, r, torch, CT, import_ci, manifest, spec, fluid_doc, state_doc, pair, shared=shared)
    except BaseException as e:
        save(out / 'failure-private.json', {'type': type(e).__name__, 'detail': str(e)[:4000]})
        r.update(status='INTERRUPTED' if isinstance(e, (InterruptedError, KeyboardInterrupt, TimeoutError)) else 'FAIL',
                 failure_type=type(e).__name__)
        raise
    finally:
        r['elapsed_seconds'] = time.perf_counter() - start
        save(out / 'receipt.json', r)


if __name__ == '__main__':
    import argparse
    require(Path(__file__).resolve() == (Path(os.environ['KNEE_ENTRY_PROJECT']) / 'knee/public_r6flip_worker.py').resolve(),
            'EXACT_FRESH_R6_WORKER_IMPORT')
    p = argparse.ArgumentParser()
    p.add_argument('--worker', required=True)
    r6_worker(p.parse_args().worker)
