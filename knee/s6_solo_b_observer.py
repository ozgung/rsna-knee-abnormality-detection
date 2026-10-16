"""Rehearsal-only observer injected into compact_infer workers via sitecustomize."""
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import subprocess
import time
from pathlib import Path


def _hash_state(state):
    import torch
    def canonical_bytes(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    h = hashlib.sha256()
    def token(value):
        data = canonical_bytes(value); h.update(len(data).to_bytes(8, 'big')); h.update(data)
    def add(value):
        if torch.is_tensor(value):
            v = value.detach().cpu().contiguous()
            token(['tensor', str(v.dtype), list(v.shape)])
            raw = v.reshape(-1).view(torch.uint8).numpy().tobytes()
            h.update(len(raw).to_bytes(8, 'big')); h.update(raw)
        elif isinstance(value, dict):
            token(['dict', len(value)])
            for key in sorted(value, key=lambda k: (type(k).__name__, canonical_bytes(k))):
                token([type(key).__name__, key]); add(value[key])
        elif isinstance(value, (list, tuple)):
            token([type(value).__name__, len(value)])
            for item in value: add(item)
        else:
            token([type(value).__name__, value])
    add(state)
    return h.hexdigest()


def _constructor(model):
    cls = next((c for c in model.__class__.mro() if c.__name__ == 'Net'), model.__class__)
    spec = {'class': cls.__module__ + '.' + cls.__qualname__,
            'parameters': [(n, str(p.dtype), list(p.shape), bool(p.requires_grad)) for n, p in model.named_parameters()],
            'buffers': [(n, str(b.dtype), list(b.shape)) for n, b in model.named_buffers()]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _gpu_uuid():
    try:
        rows = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=20).stdout.splitlines()
        index = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
        for row in rows:
            i, uuid = [x.strip() for x in row.split(',', 1)]
            if int(i) == index:
                return uuid
    except Exception:
        pass
    return None


class _CompactTrainFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'compact_train':
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _CompactTrainLoader(spec.loader)
        return spec


class _CompactTrainLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        method = getattr(self.wrapped, 'create_module', None)
        return method(spec) if method else None

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        _patch(module)


def _patch(ct):
    """Record actual CT.Net construction/load/first forward and run a strict same-input reload sentinel."""
    import torch
    original_net = ct.Net
    original_torch_load = torch.load
    latest = {'checkpoint_sha256': None, 'state': None}
    records = []

    def observed_torch_load(path, *args, **kwargs):
        result = original_torch_load(path, *args, **kwargs)
        if isinstance(path, (str, os.PathLike)):
            p = Path(path)
            if p.is_file():
                h = hashlib.sha256()
                with p.open('rb') as stream:
                    for block in iter(lambda: stream.read(1 << 20), b''):
                        h.update(block)
                latest['checkpoint_sha256'], latest['state'] = h.hexdigest(), result
        return result

    class ObservedNet(original_net):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._s6_ctor_args, self._s6_ctor_kwargs = args, kwargs
            bound = inspect.signature(original_net.__init__).bind_partial(self, *args, **kwargs)
            bound.apply_defaults()
            self._s6_constructor_spec = {k: v for k, v in bound.arguments.items() if k != 'self'}
            self._s6_checkpoint_sha = latest['checkpoint_sha256']
            self._s6_constructor_sha = _constructor(self)
            self._s6_constructor_source_sha = hashlib.sha256(Path(ct.__file__).read_bytes()).hexdigest()
            self._s6_loaded = False
            self._s6_state = latest['state']

        def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
            out = super().load_state_dict(state_dict, strict=strict, *args, **kwargs)
            self._s6_loaded = bool(strict is True)
            self._s6_state_sha = _hash_state(self.state_dict())
            self._s6_buffers_sha = _hash_state(dict(self.named_buffers()))
            return out

        def load_deploy_state_dict(self, state_dict, *args, **kwargs):
            out = super().load_deploy_state_dict(state_dict, *args, **kwargs)
            self._s6_loaded = True
            self._s6_state_sha = _hash_state(self.state_dict())
            self._s6_buffers_sha = _hash_state(dict(self.named_buffers()))
            return out

        def forward(self, *args, **kwargs):
            before = _hash_state(self.state_dict()) if not getattr(self, '_s6_observed', False) else None
            buffers_before = _hash_state(dict(self.named_buffers())) if before else None
            t0 = time.time()
            out = super().forward(*args, **kwargs)
            if before is not None:
                self._s6_observed = True
                # Build a fresh instance from the actual constructor and strict-load the exact loaded state.
                reload_t0 = time.time()
                device = next(self.parameters()).device
                with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
                    fresh = (original_net(*self._s6_ctor_args, **self._s6_ctor_kwargs).eval().requires_grad_(False)
                             .to(device).to(memory_format=torch.channels_last))
                    reload_result = fresh.load_state_dict(self._s6_state, strict=True)
                    fresh_state = _hash_state(fresh.state_dict())
                    fresh_buffers = _hash_state(dict(fresh.named_buffers()))
                    fresh_constructor = _constructor(fresh)
                    with torch.no_grad():
                        sentinel_input = _hash_state({'args': {'arg%d' % i: v for i, v in enumerate(args)}, 'kwargs': kwargs})
                        sentinel = fresh(*args, **kwargs)
                max_abs = float((out.detach().float() - sentinel.detach().float()).abs().max().cpu())
                input_h = _hash_state({'args': {'arg%d' % i: v for i, v in enumerate(args)}, 'kwargs': kwargs})
                state_after = _hash_state(self.state_dict())
                buffers_after = _hash_state(dict(self.named_buffers()))
                records.append({'checkpoint_sha256': self._s6_checkpoint_sha,
                                'constructor_sha256': self._s6_constructor_sha,
                                'constructor_source_sha256': self._s6_constructor_source_sha,
                                'constructor_spec': self._s6_constructor_spec,
                                'buffer_inventory': {n: {'dtype': str(v.dtype), 'shape': list(v.shape)}
                                                     for n, v in self.named_buffers()},
                                'buffers_before_sha256': buffers_before, 'buffers_after_sha256': buffers_after,
                                'state_before_sha256': before, 'state_after_sha256': state_after,
                                'reload_state_sha256': fresh_state, 'reload_constructor_sha256': fresh_constructor,
                                'reload_buffers_sha256': fresh_buffers, 'input_sha256': input_h,
                                'reload_input_sha256': sentinel_input,
                                'strict_reload': bool(self._s6_loaded and getattr(reload_result, 'missing_keys', []) == []
                                                      and getattr(reload_result, 'unexpected_keys', []) == []),
                                'sentinel_same_input': input_h == sentinel_input,
                                'sentinel_max_abs': max_abs,
                                'state_unchanged_from_load': before == self._s6_state_sha,
                                'buffers_unchanged_from_load': buffers_before == self._s6_buffers_sha,
                                'qualification_overhead_seconds': round(time.time() - reload_t0, 4),
                                'forward_seconds': round(time.time() - t0, 4)})
                del fresh, sentinel
            return out

    torch.load = observed_torch_load
    ct.Net = ObservedNet

    receipt_path = os.environ.get('S6SOLO_MODEL_RECEIPT')
    if not receipt_path:
        raise RuntimeError('S6SOLO_MODEL_RECEIPT missing in rehearsal worker')

    def save_receipt():
        # One B model is expected in this pair; multiple model loads are rejected by admission.
        record = records[0] if len(records) == 1 else None
        boards = [{'gpu_uuid': _gpu_uuid(), 'b': record}] if record is not None else []
        main_path = Path(__import__('sys').modules['__main__'].__file__)
        inference_sha = hashlib.sha256(main_path.read_bytes()).hexdigest()
        result = {'source': 'runtime_observer', 'inference_source_sha256': inference_sha, 'boards': boards,
                  'overhead_seconds': sum(r['qualification_overhead_seconds'] for r in records),
                  'observed_model_count': len(records)}
        p = Path(receipt_path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, sort_keys=True, indent=1) + '\n')

    import atexit
    atexit.register(save_receipt)


def install():
    if os.environ.get('S6SOLO_MODE') != 'rehearsal':
        return
    if not any(isinstance(x, _CompactTrainFinder) for x in __import__('sys').meta_path):
        __import__('sys').meta_path.insert(0, _CompactTrainFinder())
