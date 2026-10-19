"""P7-prime compact board worker: our S6 seed checkpoints through the exact nartaa S6 code tree and B command.

One process per board and pass (launched by knee.public_r6flip_entry.launch_guarded with CUDA_VISIBLE_DEVICES = board).
Steps: pinned board environment, staged images-only tree, the captured solo controller (6e87350d..., loaded unchanged
as a module; only its pure helpers extract_payload / write_launcher / b_cmd / b_env are called), the code tree
(publisher extras + heliosli files, every member hash-checked, SOURCE_SHA / CODE_SHA re-checked), the seed checkpoints
copied byte for byte into a fresh folder under loader-discoverable aliases, then ONE compact_infer subprocess per board
over the assigned studies (all seeds on the same windows, --dump-logits, --deadline = the controller's stage deadline).
The ledger keeps the CSV mean-logit probabilities and the per-seed logits; every genuine row must satisfy
CSV == sigmoid(mean seed logits) to RECONSTRUCTION_ATOL. On SIGTERM the child is forwarded the signal and no ledger is
written (the controller salvages compact_infer's flushed CSV).
"""
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
import types

from knee import public_s6c as S6C

require = S6C.require


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(json.dumps(value, indent=1, allow_nan=False, default=str).encode() + b'\n')
    os.replace(tmp, path)


def file_sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def worker_config(cfg, board, pass_name, ids, folder):
    return {'mode': cfg['mode'], 'board': board, 'physical': cfg['boards'][board], 'environment': cfg['environment'],
            'pass': pass_name, 'ids': list(ids), 'out': str(folder), 'data': cfg['staged'], 'data_ids': cfg['staged_ids'],
            'data_binding': cfg['stage_binding'], 'deadline_epoch': cfg.get('deadline_epoch') or 0, 's6c': cfg['s6c'],
            'project': str(Path(__file__).resolve().parents[1])}


def load_controller(path):
    """The archived publisher solo source, executed unchanged as a module (its __main__ guard does not fire)."""
    import hashlib
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == S6C.CAPTURED_SHA256, 'S6C_CAPTURED_PUBLISHER_SOURCE')
    module = types.ModuleType('k1_s6c_controller')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    require(module.B_GEOMETRY == S6C.GEOMETRY and module.B_AMP == S6C.AMP and module.B_WORKERS == S6C.LOADER_WORKERS
            and module.LAB == S6C.LABELS and module.B_MODEL['arm'] == 'E_prior' and module.ANATOMY_SHA256 == S6C.ANATOMY_SHA256
            and all(module.SOURCE_SHA[n] == S6C.SOURCE_PINS[n] for n in S6C.SOURCE_PINS), 'S6C_FROZEN_B_CONTRACT')
    return module


def code_tree(k1, s6c, root):
    """The scored S6 file set: extras from the publisher dataset, everything else from heliosli (prepare_inputs rules)."""
    ds, helios = Path(s6c['code_mount']), Path(s6c['helios_mount'])
    for name, pin in s6c['code']['files'].items():
        require((ds / name).is_file() and (ds / name).stat().st_size == pin['bytes'] and file_sha(ds / name) == pin['sha256'],
                'S6C_CODE_MOUNT_FILE_' + name)
    man = json.loads((ds / 'code_manifest.json').read_text())
    extras = {n: man['members'][n] for n in man['from_this_dataset']}
    k1.extract_payload(ds / k1.EXTRAS_FILE, root, extras)
    for n, rec in man['members'].items():
        if n in extras:
            continue
        src, dst = helios / k1.safe_rel(n), root / k1.safe_rel(n)
        require(src.is_file() and src.stat().st_size == rec['bytes'], 'S6C_HELIOS_FILE_SIZE')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        require(file_sha(dst) == rec['sha256'], 'S6C_HELIOS_FILE_SHA')
    src, code = root / 'heliosli', root / 'cloud_code'
    ok = all(file_sha(src / n) == v for n, v in k1.SOURCE_SHA.items()) and all(
        file_sha(code / 'training' / n) == v for n, v in k1.CODE_SHA.items())
    require(ok, 'S6C_SOURCE_CODE_SHA')
    return src, code, {'captured_sha256': S6C.CAPTURED_SHA256, 'manifest_sha256': file_sha(ds / 'code_manifest.json'),
                       'members': len(man['members']), 'extras': len(extras), 'extras_ok': True, 'source_sha_ok': True}


def stage_models(s6c, folder):
    """Exact bytes of every sealed seed file under its loader alias; nothing else in the folder."""
    folder = Path(folder)
    require(not folder.exists(), 'S6C_FRESH_MODEL_FOLDER')
    folder.mkdir(parents=True)
    staged = []
    for m in s6c['models']:
        src = Path(m['path'])
        require(src.is_file() and not src.is_symlink() and src.name == m['file'] and src.stat().st_size == m['bytes']
                and file_sha(src) == m['sha256'], 'S6C_SEED_SOURCE_BYTES')
        dst = folder / S6C.alias(m['seed'])
        shutil.copyfile(src, dst)
        require(file_sha(dst) == m['sha256'], 'S6C_SEED_STAGED_BYTES')
        staged.append({'alias': dst.name, 'sha256': m['sha256'], 'bytes': m['bytes']})
    require(sorted(p.name for p in folder.iterdir()) == s6c['aliases'] == [x['alias'] for x in staged], 'S6C_MODEL_FOLDER_CLOSURE')
    return staged


def restaged(folder, staged):
    return [{'alias': x['alias'], 'sha256': file_sha(Path(folder) / x['alias']), 'bytes': (Path(folder) / x['alias']).stat().st_size}
            for x in staged]


class GpuSampler:
    """Peak memory.used of this board (nvidia-smi, physical index = board) every 5 s."""
    def __init__(self, board):
        self.board, self.peak, self._stop = board, 0, threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(['nvidia-smi', '-i', str(self.board), '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=20).stdout.strip()
                self.peak = max(self.peak, int(float(out.splitlines()[0])) * 1024 * 1024)
            except Exception:
                pass
            self._stop.wait(5.0)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.thread.join(timeout=30)


def model_ready_seconds(log_path):
    for line in Path(log_path).read_text(errors='replace').splitlines():
        if line.startswith('MODEL_READY '):
            return float(json.loads(line[len('MODEL_READY '):])['seconds'])
    return None


def read_predictions(path):
    import csv
    rows = {}
    with open(path, newline='', encoding='utf-8') as h:
        reader = csv.reader(h)
        require(next(reader) == ['StudyInstanceUID'] + S6C.LABELS, 'S6C_PREDICTIONS_SCHEMA')
        for row in reader:
            require(len(row) == 13 and row[0] not in rows, 'S6C_PREDICTIONS_ROW')
            rows[row[0]] = [float(x) for x in row[1:]]
    return rows


def summarize(ids, ci, predictions, logits, n_seeds, ready_seconds):
    """Per-study records and the assigned-population ledger arrays (worker-local checks only)."""
    import numpy as np
    fallback = ci.get('fallback') or {}
    require(set(predictions) <= set(ids) and set(fallback) <= set(ids) and not set(predictions) & set(fallback)
            and set(predictions) | set(fallback) == set(ids), 'S6C_SCORED_OR_FALLBACK_PARTITION')
    require(set(logits) == set(predictions), 'S6C_DUMPED_LOGITS_COVER_SCORED')
    windows = {row[0]: int(row[1]) for row in ci.get('per_study') or []}
    scored = max(len(predictions), 1)
    production = max(float(ci.get('seconds') or 0.0) - float(ready_seconds or 0.0), 0.0) / scored
    probs, statuses, seed_logits, records = [], [], [], []
    for ordinal, sid in enumerate(ids):
        if sid in predictions:
            z = np.asarray(logits[sid], dtype=np.float64)
            require(z.shape == (n_seeds, 12) and np.isfinite(z).all(), 'S6C_SEED_LOGITS')
            p = np.asarray(predictions[sid], dtype=np.float64)
            err = float(np.max(np.abs(p - S6C.sigmoid(z.mean(0)))))
            require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() and err <= S6C.RECONSTRUCTION_ATOL, 'S6C_MEAN_LOGIT_RECONSTRUCTION')
            probs.append(p)
            statuses.append('genuine')
            seed_logits.append(z.astype(np.float32))
            records.append({'id': sid, 'ordinal': ordinal, 'status': 'genuine', 'windows': windows.get(sid),
                            'production_seconds': production, 'reconstruction_max_abs': err, 'probabilities': p.tolist(),
                            'seed_probabilities': S6C.sigmoid(z).tolist()})
        else:
            probs.append(np.full(12, np.nan))
            statuses.append('data_local_failure')
            seed_logits.append(np.full((n_seeds, 12), np.nan, dtype=np.float32))
            records.append({'id': sid, 'ordinal': ordinal, 'status': 'data_local_failure', 'phase': 's6c_compact_infer',
                            'failure_code': str(fallback[sid]).split('(')[0][:80],
                            'deadline': str(fallback[sid]) == 'not reached before the deadline', 'production_seconds': 0.0})
    return probs, statuses, seed_logits, records


def save_ledger(path, ids, probabilities, statuses, seed_logits):
    import numpy as np
    from knee.public_adaptation_fit import atomic_save
    p = np.asarray(probabilities, dtype=np.float64).reshape((len(ids), 12))
    z = np.asarray(seed_logits, dtype=np.float32)
    good = np.asarray(statuses) == 'genuine'
    require(len(ids) == len(set(ids)) and len(statuses) == len(ids) and z.shape[0] == len(ids) and z.ndim == 3, 'S6C_LEDGER_SCHEMA')
    require(np.isfinite(p[good]).all() and np.isnan(p[~good]).all() and np.isfinite(z[good]).all() and np.isnan(z[~good]).all(),
            'S6C_GENUINE_ONLY_LEDGER')
    arrays = dict(study_ids=np.asarray(ids), probabilities=p, statuses=np.asarray(statuses), seed_logits=z)
    atomic_save(path, None, lambda _, h: np.savez(h, **arrays))


def run_worker(config_path):
    cfg = json.loads(Path(config_path).read_bytes())
    out = Path(cfg['out']) / S6C.FOLDER
    out.mkdir()
    start = time.perf_counter()
    s6c = cfg['s6c']
    r = {'status': 'RUNNING', 'version': S6C.VERSION, 'arm': S6C.FOLDER, 'slot': 'compact', 'pass': cfg['pass'], 'board': cfg['board'],
         'assigned': cfg['ids'], 'completed': 0, 'records': []}
    save(out / 'receipt.json', r)
    child = None

    def forward(signum, frame):
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
        raise InterruptedError('stage deadline (SIGTERM)')
    signal.signal(signal.SIGTERM, forward)
    try:
        import torch
        from knee.public_s6_recovery_entry import board_environment, verify_staged
        r['environment'] = board_environment(torch, cfg)
        verify_staged(cfg['data'], cfg['data_ids'], cfg['data_binding'])
        k1 = load_controller(Path(cfg['project']) / S6C.CAPTURED)
        src, code, r['code'] = code_tree(k1, s6c, out / 'k1src')
        anatomy = Path(s6c['helios_mount']) / 'anatomy.pt'
        require(file_sha(anatomy) == S6C.ANATOMY_SHA256, 'S6C_ANATOMY_FILE')
        r['anatomy_sha256'] = S6C.ANATOMY_SHA256
        models = out / 'models'
        staged = stage_models(s6c, models)
        r['models'], r['state_before'] = staged, S6C.digest(staged)
        save(out / 'receipt.json', r)
        I = types.SimpleNamespace(launcher=k1.write_launcher(src, code, out / 'k1launch' / 'k1_b_infer.py'), comp=Path(cfg['data']),
                                  code=code, models=models, anatomy=anatomy)
        assignment = out / 'ids.json'
        assignment.write_text(json.dumps(cfg['ids']))
        cmd = k1.b_cmd(I, 'train', assignment, out / 'compact', out / 'tmp', float(cfg['deadline_epoch'] or 0))
        r.update(command=[str(x) for x in cmd[1:]], phase='compact_infer')
        save(out / 'receipt.json', r)
        log_path = out / 'compact.log'
        main_start = time.perf_counter()
        with GpuSampler(cfg['board']) as gpu, log_path.open('xb') as log:
            child = subprocess.Popen(cmd, env=k1.b_env(cfg['board']), stdout=log, stderr=subprocess.STDOUT, cwd=str(out))
            rc = child.wait()
        r['main_seconds'] = time.perf_counter() - main_start
        r['peak_cuda_bytes'] = gpu.peak
        require(rc == 0, 'S6C_COMPACT_INFER_EXIT_%s' % rc)
        ci = json.loads((out / 'compact' / 'receipt.json').read_bytes())
        r['compact_receipt_sha256'] = file_sha(out / 'compact' / 'receipt.json')
        r['compact'] = {k: ci.get(k) for k in ('status', 'amp', 'models', 'spatial_models', 'extent_models', 'timed_out', 'studies',
                                               'scored', 'seconds', 'stats', 'anchor_weights_sha256', 'sha256', 'gpu', 'torch', 'timm',
                                               'views', 'phys_mm', 'anchor_methods')}
        require(ci.get('status') == 'PASS' and ci.get('amp') == S6C.AMP and ci.get('models') == s6c['aliases']
                and ci.get('spatial_models') == len(staged) and ci.get('extent_models') == 0
                and ci.get('anchor_weights_sha256') == S6C.ANATOMY_SHA256
                and all((ci.get('sha256') or {}).get(n) == v for n, v in S6C.SOURCE_PINS.items()), 'S6C_SOURCE_RECEIPT')
        predictions = read_predictions(out / 'compact' / 'predictions.csv')
        logits = json.loads((out / 'compact' / 'logits.json').read_bytes())
        probs, statuses, seed_logits, records = summarize(cfg['ids'], ci, predictions, logits, len(staged), model_ready_seconds(log_path))
        ledger = out / 'predictions.npz'
        save_ledger(ledger, cfg['ids'], probs, statuses, seed_logits)
        after = restaged(models, staged)
        r.update(status='PASS', phase='complete', completed=len(records), records=records, genuine=statuses.count('genuine'),
                 failed=statuses.count('data_local_failure'), zero_failed=all(s == 'genuine' for s in statuses), zero_imputed=True,
                 state_after=S6C.digest(after), ledger_sha256=file_sha(ledger),
                 fallback_reasons={k: str(v)[:200] for k, v in (ci.get('fallback') or {}).items()},
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024)
        require(r['state_after'] == r['state_before'], 'S6C_STATE_AFTER')
        save(out / 'receipt.json', r)
        return r
    except BaseException as e:
        save(out / 'failure-private.json', {'type': type(e).__name__, 'detail': str(e)[:4000]})
        r.update(status='INTERRUPTED' if isinstance(e, (InterruptedError, KeyboardInterrupt, TimeoutError)) else 'FAIL',
                 failure_type=type(e).__name__)
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                child.kill()
        raise
    finally:
        r['elapsed_seconds'] = time.perf_counter() - start
        save(out / 'receipt.json', r)


if __name__ == '__main__':
    import argparse
    require(Path(__file__).resolve() == (Path(os.environ['KNEE_ENTRY_PROJECT']) / 'knee/public_s6c_worker.py').resolve(),
            'EXACT_FRESH_S6C_WORKER_IMPORT')
    p = argparse.ArgumentParser()
    p.add_argument('--worker', required=True)
    run_worker(p.parse_args().worker)
