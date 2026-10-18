"""Rehearsal only: run component A's exact embedded bytes with the 39-study fixture as its test root.

Component A hard-codes /kaggle/input/{competitions/,}rsna-knee-abnormality-detection
as its test root, so the frozen fixture cannot be presented to it by paths alone.
This runner executes the unchanged, hash-checked A source as a module (its
``__main__`` guard does not fire), replaces only ``find_test_root`` and then
calls A's own ``main``. Weights discovery, preprocessing (fork pool), TTA,
fallbacks and the /kaggle/working/submission.csv write stay A's code. The entry
never uses this file: the captured controller runs A as ``python -u run.py``.
"""
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

A_SOURCE_SHA256 = 'bcd864cbf349882e876959b49cb348cbb89b4c62fc204d4e4261049c5e3a79c5'
MODULE = 'k1_component_a'


def _fingerprint(state):
    """Match knee.public_adaptation_pilot.state_fingerprint exactly."""
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


def _constructor_fingerprint(model):
    spec = {'class': model.__class__.__module__ + '.' + model.__class__.__qualname__,
            'parameters': [(n, str(p.dtype), list(p.shape), bool(p.requires_grad)) for n, p in model.named_parameters()],
            'buffers': [(n, str(b.dtype), list(b.shape)) for n, b in model.named_buffers()]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _tensor_fingerprint(value):
    return _fingerprint({'input': value})


def _grid_names():
    return frozenset('backbone.stages.%d.blocks.%d.attn.rel_pos.%s' % (stage, block, kind)
                     for stage, count in ((2, 14), (3, 2)) for block in range(count)
                     for kind in ('relative_position_index', 'rel_coords_log'))


def _max_abs(a, b):
    import numpy as np
    if hasattr(a, 'detach'): a = a.detach().cpu().numpy()
    if hasattr(b, 'detach'): b = b.detach().cpu().numpy()
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _install_model_observer(module, output_path):
    """Wrap only A's loader/forward boundary; production inference still calls the original functions."""
    original_load, original_infer = module.load_model, module.infer_pair_probs
    records, loaded = {}, {}
    final_models = {}
    source = Path(module.__file__).read_bytes()
    source_ok = hashlib.sha256(source).hexdigest() == A_SOURCE_SHA256 and b'model.load_state_dict(ck["model"], strict=True)' in source
    if not source_ok:
        raise RuntimeError('S6SOLO_A_RUNNER: exact A source lacks its pinned strict loader')

    def observed_load(path, arch, res, device, ngpu=1):
        model, actual_res = original_load(path, arch, res, device, ngpu)
        key = str(device)
        loaded[key] = {'path': str(path), 'model': model, 'res': actual_res,
                       'checkpoint_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                       'constructor_sha256': _constructor_fingerprint(model),
                       'state_sha256': _fingerprint(model.state_dict()),
                       'buffers_sha256': _fingerprint(dict(model.named_buffers())),
                       'grid_sha256': _fingerprint({n: b for n, b in model.named_buffers() if n in _grid_names()}),
                       'strict_loaded_before': True,
                       'device': key, 'used': False}
        return model, actual_res

    def observed_infer(model, plain, mirrored, device):
        key = str(device)
        entry = loaded.get(key)
        before = _fingerprint(model.state_dict()) if entry is not None and not entry['used'] and entry['model'] is model else None
        buffers_before = _fingerprint(dict(model.named_buffers())) if before is not None else None
        grid_before = _fingerprint({n: b for n, b in model.named_buffers() if n in _grid_names()}) if before is not None else None
        input_sha = hashlib.sha256((_tensor_fingerprint(plain) + _tensor_fingerprint(mirrored)).encode()).hexdigest() if before is not None else None
        # Keep normal predictions byte-for-byte on the original path. The one-time qualification
        # reload/sentinel is deliberately outside the returned production arithmetic.
        out = original_infer(model, plain, mirrored, device)
        overhead = 0.0
        if before is not None:
            entry['used'] = True
            reload_t0 = time.time()
            import torch
            dev = torch.device(device)
            with torch.random.fork_rng(devices=[dev.index or 0] if dev.type == 'cuda' else []):
                fresh, _ = original_load(entry['path'], module.ARMS[0]['arch'], entry['res'], device)
                reloaded_state = _fingerprint(fresh.state_dict())
                reloaded_buffers = _fingerprint(dict(fresh.named_buffers()))
                reloaded_constructor = _constructor_fingerprint(fresh)
                sentinel_input_sha = hashlib.sha256((_tensor_fingerprint(plain) + _tensor_fingerprint(mirrored)).encode()).hexdigest()
                sentinel = original_infer(fresh, plain, mirrored, device)
            max_abs = _max_abs(out, sentinel)
            overhead = time.time() - reload_t0
            record = {'checkpoint_sha256': entry['checkpoint_sha256'],
                      'constructor_sha256': entry['constructor_sha256'],
                      'constructor_source_sha256': A_SOURCE_SHA256,
                      'buffers_before_sha256': buffers_before, 'buffers_after_sha256': _fingerprint(dict(model.named_buffers())),
                      'state_before_sha256': before, 'state_after_sha256': _fingerprint(model.state_dict()),
                      'grid_before_sha256': grid_before, 'grid_after_sha256': _fingerprint({n: b for n, b in model.named_buffers() if n in _grid_names()}),
                      'reload_state_sha256': reloaded_state, 'reload_constructor_sha256': reloaded_constructor,
                      'reload_buffers_sha256': reloaded_buffers, 'input_sha256': input_sha,
                      'reload_input_sha256': sentinel_input_sha,
                      'strict_loaded_before': entry['strict_loaded_before'],
                      'strict_reload': True, 'sentinel_same_input': True, 'sentinel_max_abs': max_abs,
                      'state_unchanged_from_load': before == entry['state_sha256'],
                      'buffers_unchanged_from_load': buffers_before == entry['buffers_sha256'],
                      'qualification_overhead_seconds': round(overhead, 4)}
            records[key] = record
            final_models[key] = model
            del fresh
        return out

    module.load_model = observed_load
    module.infer_pair_probs = observed_infer

    def write_receipt():
        if not records or set(records) != set(final_models):
            return
        for key, model in final_models.items():
            record = records[key]
            record['state_after_sha256'] = _fingerprint(model.state_dict())
            record['buffers_after_sha256'] = _fingerprint(dict(model.named_buffers()))
            grid = {n: b for n, b in model.named_buffers() if n in _grid_names()}
            record['grid_after_sha256'] = _fingerprint(grid)
            record['state_unchanged_from_load'] = record['state_before_sha256'] == record['state_after_sha256']
            record['buffers_unchanged_from_load'] = record['buffers_before_sha256'] == record['buffers_after_sha256']
            record['grid_unchanged_from_load'] = record['grid_before_sha256'] == record['grid_after_sha256']
        # Index mapping is explicit and physical UUIDs come from the host inventory.
        import subprocess
        proc = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=20)
        uuid_by_index = {}
        for line in proc.stdout.splitlines():
            bits = [x.strip() for x in line.split(',')]
            if len(bits) == 2:
                uuid_by_index[int(bits[0])] = bits[1]
        visible = [x.strip() for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x.strip()]
        boards = []
        for device, record in sorted(records.items()):
            idx = int(device.split(':')[-1])
            physical = visible[idx] if idx < len(visible) else str(idx)
            uuid = physical if physical.startswith('GPU-') else uuid_by_index.get(int(physical)) if physical.isdigit() else physical
            boards.append({'gpu_uuid': uuid, 'a': {k: v for k, v in record.items()
                            if k not in ('qualification_overhead_seconds',)},
                           'qualification_overhead_seconds': record['qualification_overhead_seconds']})
        result = {'source': 'runtime_observer', 'component_a_source_sha256': A_SOURCE_SHA256,
                  'boards': boards, 'overhead_seconds': sum(x['qualification_overhead_seconds'] for x in boards),
                  'overhead_scope': 'reload_and_same_input_sentinel_only; other observer work remains in wall stages'}
        path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(result, sort_keys=True, indent=1) + '\n')
        os.replace(tmp, path)
    return write_receipt


def run(run_py, test_root, expected_sha256=A_SOURCE_SHA256, module=MODULE, output_path='/kaggle/working/k1/A/model-qualification.json'):
    run_py, test_root = Path(run_py), str(Path(test_root))
    raw = run_py.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise SystemExit('S6SOLO_A_RUNNER: component A source differs from %s' % expected_sha256)
    if not (Path(test_root) / 'test.csv').is_file() or not (Path(test_root) / 'test_series.csv').is_file():
        raise SystemExit('S6SOLO_A_RUNNER: fixture test root lacks test.csv/test_series.csv')
    m = types.ModuleType(module)
    m.__file__ = str(run_py)
    sys.modules[module] = m          # fork-pool workers resolve pickled A functions through this entry
    exec(compile(raw, str(run_py), 'exec'), m.__dict__)
    if not callable(getattr(m, 'find_test_root', None)) or not callable(getattr(m, 'main', None)):
        raise SystemExit('S6SOLO_A_RUNNER: component A lacks find_test_root/main')
    m.find_test_root = lambda: test_root
    write_observation = (_install_model_observer(m, output_path)
                        if expected_sha256 == A_SOURCE_SHA256 else (lambda: None))
    print('S6SOLO_A_RUNNER exact A %s... with the rehearsal fixture as its test root' % expected_sha256[:12], flush=True)
    result = m.main()
    write_observation()
    return result


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        raise SystemExit('usage: s6_solo_a_runner.py RUN_PY FIXTURE_TEST_ROOT [MODEL_RECEIPT]')
    run(sys.argv[1], sys.argv[2], output_path=sys.argv[3] if len(sys.argv) == 4 else '/kaggle/working/k1/A/model-qualification.json')
