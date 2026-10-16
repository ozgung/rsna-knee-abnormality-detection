"""Common runtime of the solo-V2 / solo-S6fall pair: the captured publisher controller.

Both notebooks ship this file, the archived publisher solo source (6e87350d...)
and knee/s6_solo_pair.py byte for byte; only ``cfg['arm']`` differs. The
captured source is executed unchanged as a module. Exactly three sites of its
``prepare_inputs`` are replaced, matched on exact source text:

1. the B checkpoint sha check becomes ``solo_verify_inputs(I)``: the B
   checkpoint, its own genuine export receipts, the separately pinned publisher
   code extras and component A's weight file are all hash-bound from cfg;
2. the publisher run_config/DONE assertions are removed from the common code;
   the control arm re-imposes the same values through its receipt expectations,
   V2 binds its own genuine receipts instead (no manufactured DONE);
3. the model-folder symlink becomes ``solo_stage_b_model(I)``: the exact bytes
   are copied by ``stage_checkpoint`` into a fresh folder holding one
   loader-discoverable file (V2 under its alias), so nothing else co-loads.

Everything else (A, B, retry, caps, deadlines, rank recovery, rho floor,
guards, submission writing and deletion) is the captured code. Rehearsal-only
hooks live in knee.s6_solo_rehearsal and are refused in entry mode.
"""
import csv
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

CAPTURED = 'publisher/rsna-knee-s6fall-solo.py'
CAPTURED_SHA256 = '6e87350d5ae092b7a889546ca13517303523bc469f7e611ffe945913434d22fe'
EMBEDDED_A_SHA256 = 'bcd864cbf349882e876959b49cb348cbb89b4c62fc204d4e4261049c5e3a79c5'
CONTROLLER_MODULE = 'k1_solo_controller'
WORK = Path('/kaggle/working')
A_INPUT_ROOT = Path('/kaggle/input')     # component A's own hard-coded search root
B_GEOMETRY = {'backbone': 'convnext_small', 'size': 320, 'crop': 288, 'views': 'v4', 'slices': 3, 'phys_mm': 3.5,
              'eval_stride': 1, 'chunk': 192, 'item_timeout': 900}

SITE_MODEL_SHA = ('    if sha256_file(I.ds / B_MODEL["file"]) != B_MODEL["sha256"]:\n'
                  '        raise K1Error("model sha differs from the kernel constant")\n')
SITE_RECEIPTS = (
    '    cfg = json.loads((I.ds / B_MODEL["run_config"]).read_text())\n'
    '    want = {"size": B_GEOMETRY["size"], "crop": B_GEOMETRY["crop"], "views": B_GEOMETRY["views"], "slices": B_GEOMETRY["slices"],\n'
    '            "phys_mm": B_GEOMETRY["phys_mm"], "backbone": B_GEOMETRY["backbone"], "spatial_attn": True, "arms": B_MODEL["arm"]}\n'
    '    bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}\n'
    '    if bad:\n'
    '        raise K1Error("run_config differs from the inference arguments (trained, inferred): %s" % bad)\n'
    '    done = json.loads((I.ds / B_MODEL["done"]).read_text())\n'
    '    if done.get("model_sha256") != B_MODEL["sha256"] or os.path.basename(done.get("model", "")) != B_MODEL.get("source_file", B_MODEL["file"]):\n'
    '        raise K1Error("DONE marker does not record this model")\n'
    '    for k in ("epochs", "train_studies"):\n'
    '        if B_MODEL.get("expect_" + k) is not None and done.get(k) != B_MODEL["expect_" + k]:\n'
    '            raise K1Error("DONE marker %s=%s, expected %s" % (k, done.get(k), B_MODEL["expect_" + k]))\n')
SITE_SYMLINK = ('    I.models.mkdir(parents=True)\n'
                '    os.symlink(I.ds / B_MODEL["file"], I.models / B_MODEL["file"])        # the models folder holds exactly this one file\n')
REPLACEMENTS = ((SITE_MODEL_SHA, '    solo_verify_inputs(I)\n'),
                (SITE_RECEIPTS, ''),
                (SITE_SYMLINK, '    I.models.parent.mkdir(parents=True, exist_ok=True)\n    solo_stage_b_model(I)\n'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False, default=str) + '\n')
    os.replace(tmp, path)


def binding_digest(cfg):
    """The identity a rehearsal result is bound to: common contract plus this arm."""
    return digest({'common': cfg['common'], 'arm': cfg['arm']})


# ---- the captured controller ------------------------------------------------------------------------------------------------
def load_controller(path, name=CONTROLLER_MODULE):
    """Execute the archived publisher source unchanged as module ``name`` (its __main__ guard does not fire)."""
    path = Path(path)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CAPTURED_SHA256, 'CAPTURED_PUBLISHER_SOURCE_SHA256')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def derived_prepare_inputs_source(text):
    """prepare_inputs with exactly the three documented sites replaced (exact text, each exactly once)."""
    require(hashlib.sha256(text.encode()).hexdigest() == CAPTURED_SHA256, 'CAPTURED_PUBLISHER_SOURCE_SHA256')
    start = text.index('\ndef prepare_inputs():\n') + 1
    end = text.index('\n# ---- component A ', start) + 1
    func = text[start:end]
    for old, new in REPLACEMENTS:
        require(func.count(old) == 1, 'EXACT_PREPARE_INPUTS_SITE')
        func = func.replace(old, new)
    require('B_MODEL["run_config"]' not in func and 'B_MODEL["done"]' not in func and 'os.symlink' not in func,
            'NO_PUBLISHER_RECEIPT_ASSERTION_LEFT')
    return func


def a_weight_path(fname, root=None):
    """Component A's find_weight_file rule, applied to the same root, so the hashed file is the file A loads."""
    root = str(root or A_INPUT_ROOT).rstrip('/')
    for p in (f'{root}/raptor-knee-arms/{fname}', f'{root}/raptor-knee-arms/1/{fname}', f'{root}/raptor-cnn336/{fname}'):
        if os.path.exists(p):
            return Path(p)
    for d in sorted(glob.glob(root + '/*/')):
        if 'competition' in d.lower() or 'abnormality-detection' in d or any(
                os.path.exists(os.path.join(d, x)) for x in ('test_series', 'train_images', 'train.csv', 'test.csv')):
            continue
        hits = glob.glob(os.path.join(d, '**', fname), recursive=True)
        if hits:
            return Path(hits[0])
    return None


def verify_inputs(k1, cfg, I, a_root=None):
    """Replaces the captured B sha check: every input file the run reads is bound by full hash before use."""
    def need(ok, message):
        if not ok:
            raise k1.K1Error('solo inputs: ' + message)
    common, arm = cfg['common'], cfg['arm']
    b = arm['b_model']
    for name, pin in sorted(common['code_mount']['files'].items()):
        p = I.ds / name
        need(p.is_file() and p.stat().st_size == pin['bytes'] and k1.sha256_file(p) == pin['sha256'],
             'publisher code mount file %s differs from its pin' % name)
    aw = common['a_weight']
    ap = a_weight_path(aw['file'], a_root)
    need(ap is not None and ap.is_file() and ap.stat().st_size == aw['bytes'] and k1.sha256_file(ap) == aw['sha256'],
         'component A weight file %s missing or differs' % aw['file'])
    I.bmount = k1.find_mount(b['mount'], b['source_file'])
    ck = I.bmount / b['source_file']
    need(ck.is_file() and not ck.is_symlink() and ck.stat().st_size == b['bytes'] and k1.sha256_file(ck) == b['sha256'],
         'B checkpoint bytes differ from the bound identity')
    for name, sha in sorted(arm['receipts'].items()):
        p = I.bmount / name
        need(p.is_file() and k1.sha256_file(p) == sha, 'B receipt %s missing or differs' % name)
    for name, want in sorted(arm['json_expect'].items()):
        data = json.loads((I.bmount / name).read_text())
        bad = {k: (data.get(k), v) for k, v in want.items() if data.get(k) != v}
        need(not bad, 'B receipt %s differs (found, expected): %s' % (name, bad))
    for name, line in sorted(arm['text_expect'].items()):
        need(line in (I.bmount / name).read_text().splitlines(), 'B receipt %s lacks its bound line' % name)
    k1._RECEIPT['solo_inputs'] = {'variant': arm['variant'], 'b_mount': str(I.bmount), 'b_checkpoint': b,
                                  'a_weight': str(ap), 'receipts': arm['receipts']}
    k1.log('[solo] inputs pinned: %s B %s %s..., A weight %s..., %d B receipts, %d code-mount files' % (
        arm['variant'], b['source_file'], b['sha256'][:12], aw['sha256'][:12], len(arm['receipts']),
        len(common['code_mount']['files'])))


def stage_b_model(k1, cfg, I):
    """Replaces the captured symlink: identical bytes, fresh folder, exactly one discoverable checkpoint."""
    from knee.s6_solo_pair import ALIAS_RE, stage_checkpoint
    b = cfg['arm']['b_model']
    try:
        r = stage_checkpoint(cfg['arm']['descriptor'], I.bmount / b['source_file'], I.models)
    except ValueError as e:
        raise k1.K1Error('solo staging: %s' % e)
    names = sorted(p.name for p in I.models.iterdir())
    if (names != [b['file']] or r['alias_file'] != b['file'] or not ALIAS_RE.fullmatch(b['file'])
            or len(list(I.models.glob(b['arm'] + '_s*_f*.pt'))) != 1 or r['alias_sha256'] != b['sha256']):
        raise k1.K1Error('solo staging: model folder must hold exactly %s, found %s' % (b['file'], names))
    k1._RECEIPT['solo_stage'] = r
    k1.log('[solo] staged %s as %s (%d bytes, identical)' % (r['source_file'], r['alias_file'], r['alias_size']))


def bind(k1, cfg, a_root=None):
    """Install the arm's B binding into the loaded controller; refuse a controller that is not the frozen contract."""
    arm, common = cfg['arm'], cfg['common']
    b = arm['b_model']
    import base64
    require(k1.MODE == 'submit' and k1.W_A == 0.0 and k1.W_B == 1.0, 'FROZEN_SOLO_WEIGHTS')
    require(k1.A_SOURCE_SHA256 == EMBEDDED_A_SHA256 == common['embedded_a_sha256']
            and hashlib.sha256(base64.b64decode(k1.A_B64)).hexdigest() == EMBEDDED_A_SHA256, 'EMBEDDED_A_SOURCE')
    require(k1.B_GEOMETRY == B_GEOMETRY and k1.B_AMP == 'fp16' and k1.B_WORKERS == 2 and k1.LAB == common['targets'],
            'FROZEN_B_GEOMETRY_PRECISION_TARGETS')
    require(k1.DS_SLUG == common['code_mount']['mount'], 'CODE_MOUNT_SLUG')
    k1.B_MODEL = {'file': b['file'], 'source_file': b['source_file'], 'sha256': b['sha256'], 'bytes': b['bytes'],
                  'arm': b['arm'], 'dataset': b['dataset'], 'dataset_version': b['dataset_version'], 'variant': arm['variant']}
    k1.solo_verify_inputs = lambda I: verify_inputs(k1, cfg, I, a_root)
    k1.solo_stage_b_model = lambda I: stage_b_model(k1, cfg, I)
    text = Path(k1.__file__).read_text()
    func = derived_prepare_inputs_source(text)
    exec(compile(func, '<s6solo-prepare_inputs>', 'exec'), k1.__dict__)
    k1._RECEIPT['solo'] = {'variant': arm['variant'], 'binding_sha256': binding_digest(cfg), 'mode': cfg['mode'],
                           'captured_source_sha256': CAPTURED_SHA256,
                           'derived_prepare_inputs_sha256': hashlib.sha256(func.encode()).hexdigest(),
                           'payload_sha256': os.environ.get('S6SOLO_PAYLOAD')}
    return k1


# ---- resources ----------------------------------------------------------------------------------------------------------------
def meminfo():
    out = {}
    try:
        for line in open('/proc/meminfo'):
            k, v = line.split(':', 1)
            if k in ('MemTotal', 'MemAvailable'):
                out[k] = int(v.split()[0]) * 1024
    except Exception:
        pass
    return out


def gpu_query():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=20)
        rows = []
        for line in r.stdout.splitlines():
            f = [x.strip() for x in line.split(',')]
            if len(f) == 5:
                rows.append({'index': int(f[0]), 'uuid': f[1], 'name': f[2], 'used_mib': float(f[3]), 'total_mib': float(f[4])})
        return rows
    except Exception:
        return []


class Sampler:
    """Observer thread: host memory, per-GPU memory and free disk. Sampled, so true peaks may be missed between samples."""

    def __init__(self, interval=10.0, paths=('/kaggle/working', '/tmp')):
        self.interval, self.paths, self.samples = interval, paths, []
        self._stop = threading.Event()
        self.boards = []
        self._thread = threading.Thread(target=self._run, name='s6solo-sampler', daemon=True)

    def sample(self):
        m, g = meminfo(), gpu_query()
        if g and not self.boards:
            self.boards = [{k: x[k] for k in ('index', 'uuid', 'name', 'total_mib')} for x in g]
        disk = {}
        for p in self.paths:
            try:
                disk[p] = shutil.disk_usage(p).free
            except OSError:
                pass
        used = m['MemTotal'] - m['MemAvailable'] if len(m) == 2 else None
        self.samples.append({'t': round(time.time(), 1), 'host_used_bytes': used, 'host_total_bytes': m.get('MemTotal'),
                             'gpu_used_mib': [x['used_mib'] for x in g], 'disk_free_bytes': disk})

    def _run(self):
        while not self._stop.is_set():
            try:
                self.sample()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=30)
        try:
            self.sample()
        except Exception:
            pass

    def report(self):
        return {'interval_seconds': self.interval, 'boards': self.boards, 'samples': self.samples}


def peaks(report, windows):
    """Peaks over the whole run and per named (start, end) wall-time window."""
    def over(rows):
        host = [r['host_used_bytes'] for r in rows if r['host_used_bytes'] is not None]
        gpus = [r['gpu_used_mib'] for r in rows if r['gpu_used_mib']]
        disk = {}
        for r in rows:
            for p, v in r['disk_free_bytes'].items():
                disk[p] = min(disk.get(p, v), v)
        return {'samples': len(rows), 'host_used_peak_bytes': max(host) if host else None,
                'gpu_used_peak_mib': [max(col) for col in zip(*gpus)] if gpus else [], 'disk_free_min_bytes': disk}
    rows = report['samples']
    out = {'all': over(rows)}
    for name, (a, b) in windows.items():
        out[name] = over([r for r in rows if a <= r['t'] <= b])
    total = [r['host_total_bytes'] for r in rows if r.get('host_total_bytes')]
    out['host_total_bytes'] = total[0] if total else None
    return out


# ---- entry ----------------------------------------------------------------------------------------------------------------------
def remove_submission(work=None):
    work = Path(work or WORK)
    for p in (work / 'submission.csv', work / '.submission.csv.tmp'):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def check_submission(path, test_csv, sub_cols, targets):
    with open(test_csv, newline='') as f:
        ids = [r['StudyInstanceUID'] for r in csv.DictReader(f)]
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    require(rows and rows[0] == list(sub_cols) and sorted(sub_cols[1:]) == sorted(targets), 'SUBMISSION_SCHEMA')
    require([r[0] for r in rows[1:]] == ids and len(ids) == len(set(ids)), 'SUBMISSION_EXACT_ID_COVERAGE')
    for r in rows[1:]:
        require(len(r) == len(sub_cols), 'SUBMISSION_ROW_WIDTH')
        for v in r[1:]:
            x = float(v)
            require(math.isfinite(x) and 0.0 <= x <= 1.0, 'SUBMISSION_FINITE_UNIT_INTERVAL')
    return {'rows': len(ids), 'sha256': file_sha(path)}


def finish_entry(k1, cfg, sampler, started, out):
    inputs = k1._RECEIPT['inputs']
    sub = WORK / 'submission.csv'
    checked = check_submission(sub, Path(inputs['comp']) / 'test.csv', inputs['sub_cols'], cfg['common']['targets'])
    require(checked['sha256'] == k1._RECEIPT['blend']['submission_sha256'], 'SUBMISSION_IS_THE_CONTROLLER_FILE')
    receipt = {'status': 'PASS', 'mode': 'entry', 'variant': cfg['arm']['variant'], 'binding_sha256': binding_digest(cfg),
               'payload_sha256': os.environ.get('S6SOLO_PAYLOAD'), 'submission': checked,
               'b_failed_rank_A_only': k1._RECEIPT['blend']['b_failed_rank_A_only'],
               'a_internal_fallbacks': len(k1._RECEIPT['A']['internal_fallbacks']),
               'stages': {k: v.get('seconds') for k, v in k1._RECEIPT['stages'].items()},
               'controller_seconds': k1._RECEIPT.get('total_seconds'), 'elapsed_seconds': round(time.time() - started, 1),
               'resources': peaks(sampler.report(), {}), 'qualified_rehearsal': cfg.get('qualified_rehearsal')}
    save(out / 'entry-receipt.json', receipt)
    return receipt


def run(cfg, project):
    """One notebook execution: entry or rehearsal, for the arm bound in cfg."""
    require(cfg.get('mode') in ('rehearsal', 'entry'), 'MODE')
    started = float(os.environ.get('S6SOLO_STARTED', time.time()))
    out = WORK / 's6solo'
    out.mkdir(parents=True, exist_ok=True)
    sampler = Sampler().start()
    state = None
    try:
        if cfg['mode'] == 'rehearsal':
            from knee import s6_solo_rehearsal as R
            state = R.prepare(cfg, project)                  # sets K1_KAGGLE_ROOT before the controller is imported
        else:
            require('K1_KAGGLE_ROOT' not in os.environ and 'rehearsal' not in cfg, 'ENTRY_HAS_NO_REHEARSAL_HOOKS')
        k1 = bind(load_controller(project / CAPTURED), cfg)
        if state is not None:
            R.install(k1, cfg, state)
        k1.main()
        if state is not None:
            receipt = R.finish(k1, cfg, state, sampler, started, out)
        else:
            receipt = finish_entry(k1, cfg, sampler, started, out)
        print('S6SOLO_DONE ' + json.dumps({'status': receipt['status'], 'mode': cfg['mode'], 'variant': cfg['arm']['variant'],
                                            'elapsed_seconds': receipt.get('elapsed_seconds')}), flush=True)
        if receipt['status'] != 'PASS':
            raise RuntimeError('S6 solo %s did not pass: %s' % (cfg['mode'], receipt.get('failed_checks')))
    except BaseException as e:
        remove_submission()
        try:
            save(out / 'failure.json', {'mode': cfg['mode'], 'variant': cfg['arm']['variant'], 'type': type(e).__name__,
                                        'detail': repr(e)[:4000], 'elapsed_seconds': round(time.time() - started, 1),
                                        'resources': peaks(sampler.report(), {})})
        except Exception:
            pass
        raise
    finally:
        sampler.stop()
    return receipt


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config', required=True)
    a = p.parse_args(argv)
    run(json.loads(Path(a.config).read_text()), Path(__file__).resolve().parents[1])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
