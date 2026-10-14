"""Single-arm Raptor64 fold-0 VM controller. CLI: config.json --sha256 REVIEWED_CONFIG_SHA.

Parallel to the qualified source96 controller (unchanged). Config v1 keys: name, arm,
vm_session_id, package_root/package_manifest_sha256, workdir, input_root, stage_root,
manifest_sha256/content_sha256/stage_receipt_sha256, initializer_descriptor(+_sha256),
persister/persister_sha256, persistence {dataset, private:true, staging, status_file},
controller_lock, workers (default 2), worker_choice_name, resume, resume_sha256,
actual_platform_deadline (optional UTC seconds). Every path/dataset name must carry the arm
slug so concurrent arms on one VM stay isolated. No installation/staging/initializer download
happens here; the private backup dataset must already exist.
"""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time

PERSISTER_SHA = '8923bd872500b87f85be78aefd9fc799a6f88942e94503045b5e01ea66c118df'
VERSION = 'colab-raptor64-run-v1'
ARMS = {'in-orig': ('imagenet', 'original'), 'in-plane': ('imagenet', 'plane'),
        'dr-orig': ('dread_v4', 'original'), 'dr-plane': ('dread_v4', 'plane'), 'na-orig': ('nartaa_oai', 'original'),
        'dr-full': ('dread_v4', 'original'), 'in-mirror': ('imagenet', 'original'), 'in-distil': ('imagenet', 'original'),
        'in-distil-w1': ('imagenet', 'original'), 'in-distil-flip': ('imagenet', 'original')}
FULL_ARMS = frozenset({'dr-full'})
EPOCHS = 12
FINAL_EPOCH = EPOCHS - 1
DEVELOPMENT = 870
EXPORT_K = 42
PREFIX = 'raptor64'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def checked(path, digest):
    p = Path(path)
    if p.is_symlink() or not re.fullmatch('[0-9a-f]{64}', str(digest)) or sha(p) != digest:
        raise ValueError('Artifact pin differs')
    return json.loads(p.read_bytes())


def atomic_json(path, value):
    p = Path(path); tmp = p.with_name(p.name + '.tmp')
    if p.is_symlink() or tmp.is_symlink():
        raise ValueError('Unsafe receipt path')
    with open(tmp, 'w') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def private_diagnostic(path, text):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, 'a') as f:
        f.write(text); f.flush()


def validate(c):
    allowed = {'version', 'name', 'arm', 'vm_session_id', 'package_root', 'package_manifest_sha256', 'workdir', 'input_root', 'stage_root',
               'manifest_sha256', 'content_sha256', 'stage_receipt_sha256', 'initializer_descriptor', 'initializer_descriptor_sha256',
               'persister', 'persister_sha256', 'persistence', 'controller_lock', 'workers', 'worker_choice_name', 'resume', 'resume_sha256',
               'actual_platform_deadline', 'fold'}
    if set(c) - allowed:
        raise ValueError('Unknown runner config parameter')
    arm = c.get('arm')
    if c.get('version') != VERSION or arm not in ARMS or c.get('fold', 0) != 0:
        raise ValueError('Raptor64 fold-0 arm config required')
    if not re.fullmatch('[a-z0-9][a-z0-9-]{2,63}', c.get('name', '')) or not re.fullmatch('[a-zA-Z0-9_-]{3,80}', c.get('vm_session_id', '')):
        raise ValueError('Named immutable VM config required')
    for name in ('package_root', 'workdir', 'input_root', 'stage_root', 'initializer_descriptor', 'persister', 'controller_lock'):
        p = Path(c[name])
        if not p.is_absolute() or p.is_symlink():
            raise ValueError('Explicit safe absolute VM paths required')
    work = Path(c['workdir']); package = Path(c['package_root']); persist = c['persistence']
    dataset = persist.get('dataset', '')
    if persist.get('private') is not True or not re.fullmatch('[a-z0-9_-]+/[a-z0-9-]{3,50}', dataset):
        raise ValueError('Private persistence dataset required')
    for name in ('staging', 'status_file'):
        p = Path(persist[name])
        if not p.is_absolute() or p.is_symlink():
            raise ValueError('Explicit private persistence paths required')
    if work == package or work in package.parents or package in work.parents or Path(persist['staging']) == work or work in Path(persist['staging']).parents:
        raise ValueError('Package/work/persistence staging must be separate')
    tag = f'{PREFIX}-{arm}'
    isolated = (c['name'], package.name, work.name, Path(persist['staging']).name, Path(persist['status_file']).name,
                Path(c['controller_lock']).name, dataset.split('/')[-1])
    # Boundary-aware arm parsing (longest arm first): 'raptor64-in-distil' must not match 'raptor64-in-distil-w1-...'.
    alternatives = '|'.join(re.escape(a) for a in sorted(ARMS, key=len, reverse=True))
    parsed = [re.search(rf'{PREFIX}-({alternatives})(?=-(?:pkg|fold0|backup|persist|controller|run)\b)', value) for value in isolated]
    if any(m is None or m.group(1) != arm for m in parsed):
        raise ValueError('Each arm requires isolated arm-named package/work/backup/status/lock paths')
    m = checked(package / 'manifest.json', c['package_manifest_sha256'])
    for file, key in [('train_knee_original.py', 'original_sha256'), ('train_knee_patched.py', 'patched_sha256'), ('tracka_entry.py', 'entry_sha256'),
                      ('train_knee.patch', 'patch_sha256'), ('folds_v0.json', 'folds_sha256'), ('excluded_ids.json', 'excluded_ids_sha256'),
                      ('raptor64-recipe.json', 'recipe_sha256')]:
        if (package / file).is_symlink() or sha(package / file) != m[key]:
            raise ValueError('Package artifact differs')
    for file, digest in m['sources'].items():
        p = Path(file)
        if p.is_absolute() or '..' in p.parts or (package / p).is_symlink() or sha(package / p) != digest:
            raise ValueError('Runtime module pin differs')
    policy = json.loads((package / 'raptor64-recipe.json').read_bytes())
    if policy.get('arm') != arm or (policy.get('init'), policy.get('head')) != ARMS[arm]:
        raise ValueError('Package arm differs from config arm')
    if (c.get('workers', 2) != policy['workers'] or c.get('worker_choice_name') != policy.get('worker_choice_name') or
            (policy['workers'] != 2 and not policy.get('worker_choice_name'))):
        raise ValueError('Frozen worker choice differs')
    f = policy['frozen']; full = arm in FULL_ARMS
    if (f['epochs'], f['k'], f['k_eval'], f['res'], f['fold'], f['folds'], f['time_budget_h'], policy.get('export_k')) != (
            EPOCHS, 8, 16, 384, -1 if full else 0, 0 if full else 5, 0, EXPORT_K) or (policy.get('mode') == 'full') != full:
        raise ValueError('Frozen training recipe differs')
    if c['persister_sha256'] != PERSISTER_SHA or sha(c['persister']) != PERSISTER_SHA:
        raise ValueError('Authoritative persister source differs')
    stage = Path(c['stage_root']); r = checked(stage / 'staging-receipt.json', c['stage_receipt_sha256'])
    if r.get('status') != 'PASS' or r.get('manifest_sha256') != c['manifest_sha256'] or r.get('content_sha256') != c['content_sha256']:
        raise ValueError('Verified staged cache required')
    d = checked(c['initializer_descriptor'], c['initializer_descriptor_sha256'])
    if d.get('init') != ARMS[arm][0] or Path(d['weights_file']).is_symlink() or sha(d['weights_file']) != d['weights_sha256']:
        raise ValueError('Initializer descriptor/weights differ from the arm')
    if work.exists() and any(re.search(r'fold([0-9])', p.name) and re.search(r'fold([0-9])', p.name).group(1) != '0' for p in work.glob('raptor_*')):
        raise ValueError('Cross-fold work artifacts forbidden')
    resume = work / 'raptor_resume_fold0.pt'
    if c.get('resume', False):
        if type(c['resume']) is not bool or not resume.is_file() or resume.is_symlink() or sha(resume) != c.get('resume_sha256'):
            raise ValueError('Explicit pinned resume required')
    elif work.exists() and any(work.glob('raptor_*')):
        raise ValueError('Existing training artifacts require explicit resume')
    return m, policy, r, d


def environment(c):
    tree = ast.parse((Path(c['package_root']) / 'tracka_entry.py').read_text())
    defaults = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PARAM_DEFAULTS' for t in n.targets)))
    env = dict(os.environ); env.update({k: str(v) for k, v in defaults.items()})
    env.update(FOLD_A='0', FOLD_B='', WORKDIR=c['workdir'], ASSET_DIR=c['package_root'], RAPTOR_INPUT=c['input_root'], RESUME='1' if c.get('resume') else '0',
               RAPTOR_PRELOAD='0', PYTHONPATH=c['package_root'], PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               RAPTOR64_STAGE_ROOT=c['stage_root'], RAPTOR64_MANIFEST_SHA256=c['manifest_sha256'], RAPTOR64_CONTENT_SHA256=c['content_sha256'],
               RAPTOR64_STAGE_RECEIPT_SHA256=c['stage_receipt_sha256'], RAPTOR64_INITIALIZER_DESCRIPTOR=c['initializer_descriptor'],
               RAPTOR64_INITIALIZER_DESCRIPTOR_SHA256=c['initializer_descriptor_sha256'])
    for key in ('RAPTOR64_RESUME_SHA256', 'RAPTOR64_ACTUAL_PLATFORM_DEADLINE_UTC_SECONDS', 'ORIGINAL_SCRIPT'):
        env.pop(key, None)
    if c.get('resume'):
        env['RAPTOR64_RESUME_SHA256'] = c['resume_sha256']
    if c.get('actual_platform_deadline') is not None:
        env['RAPTOR64_ACTUAL_PLATFORM_DEADLINE_UTC_SECONDS'] = str(float(c['actual_platform_deadline']))
    return env


PROBE = '''import json,platform,random,sys,torch,timm,numpy
from knee import raptor64_training as s
s.strict_cuda(torch)
v,m,ids=s.staged_corpus()
sys.argv=["probe"]
import train_knee_patched as tk
b=tk.build_backbone(s.ARCH,pretrained=False);i=s.initialize_backbone(b,"train_knee_patched.py")
b.set_grad_checkpointing(True)
model=tk.RaptorClassifier(b,F_dim=b.num_features).cuda();model.win_chunk=8
state={k:t.detach().clone() for k,t in model.state_dict().items()}
rows=[0,1,2,3];id2row={str(u):j for j,u in enumerate(ids)};lab={str(ids[j]):numpy.zeros(12,numpy.float32) for j in rows}
py=random.getstate();npst=numpy.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
ds=tk.StudyWindows("",[str(ids[j]) for j in rows],id2row,lab,384,8,train=True,norm="imagenet")
x,y=tk.collate([ds[j] for j in range(4)])
torch.cuda.reset_peak_memory_stats();model.train()
with torch.autocast("cuda",dtype=torch.float16):out=model(x.to("cuda"))
loss=torch.nn.functional.binary_cross_entropy_with_logits(out.float(),y[:,:12].cuda());loss.backward()
finite=bool(torch.isfinite(out).all()) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
train_peak=torch.cuda.max_memory_allocated()
model.eval();ev=tk.StudyWindows("",[str(ids[0])],id2row,lab,384,42,train=False,norm="imagenet")
xe,_=tk.collate([ev[0]])
with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):oe=model(xe.to("cuda"))
model.load_state_dict(state);random.setstate(py);numpy.random.set_state(npst);torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"timm":timm.__version__,"numpy":numpy.__version__,"cuda":torch.version.cuda,
 "gpu":torch.cuda.get_device_name(0),"threads":torch.get_num_threads(),"studies":len(ids),"initializer":i,"device":"cuda",
 "probe":{"train_batch_shape":list(x.shape),"eval_windows":int(xe.shape[1]),"finite":finite and bool(torch.isfinite(oe).all()),
 "train_peak_cuda_bytes_bs4_k8":train_peak,"head":type(model).__name__,"synthetic":False,"rows":"first four corpus rows; no labels"}}))
'''


def runtime_probe(c, env):
    p = subprocess.run([sys.executable, '-c', PROBE], cwd=c['package_root'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    private_diagnostic(Path(c['workdir']) / f'{PREFIX}-runtime-stdout.log', p.stdout)
    private_diagnostic(Path(c['workdir']) / f'{PREFIX}-runtime-stderr.log', p.stderr)
    if p.returncode:
        raise RuntimeError('CUDA/cache/initializer/forward runtime preflight failed; private diagnostics retained in workdir')
    result = json.loads(p.stdout.strip().splitlines()[-1])
    init = result.get('initializer', {})
    if (result.get('device') != 'cuda' or result.get('threads') != 1 or result.get('studies') != 4345 or result.get('timm') != '1.0.26' or
            init.get('missing') != 0 or init.get('unexpected') != 0 or init.get('init') != ARMS[c['arm']][0] or
            result.get('probe', {}).get('finite') is not True or result['probe'].get('eval_windows') != EXPORT_K):
        raise ValueError('Runtime environment/initializer/forward probe differs')
    return result


def private_dataset(dataset):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    with tempfile.TemporaryDirectory() as t:
        m = json.loads(Path(api.dataset_metadata(dataset, t)).read_bytes())
    info = m.get('info', m)
    identity = (str(info.get('ownerUser')) + '/' + str(info.get('datasetSlug'))) if 'info' in m else info.get('id')
    if identity != dataset or info.get('isPrivate') is not True:
        raise ValueError('Existing backup dataset identity/privacy is unverified/public')
    return {'dataset': dataset, 'private': True}


def sanitized(line):
    line = re.sub(r'^\[(?:train:fold0|persist [0-9:]{8})\]\s*', '', line.strip())
    for pattern in (r'^ep\d+ loss [0-9.eE+-]+ \| OOF masked macro-AUC [0-9.eE+-]+', r'^\[resume\] (?:recoverable completed epoch|completed epoch checkpoint saved)\s*\d*',
                    r'^FINAL PERSIST (?:VERIFIED epoch \d+ version \d+|UNVERIFIED)', r'^\[raptor64\] init \S+ loaded \d+ tensors, missing 0, unexpected 0',
                    r'^ep\d+ loss [0-9.eE+-]+ \| FULL-DATA fixed-epoch score [01]\.0'):
        match = re.search(pattern, line.strip())
        if match:
            return match.group(0)
    return None


def stream(process, label, diagnostic_path):
    for line in process.stdout:
        private_diagnostic(diagnostic_path, line)
        safe = sanitized(line)
        if safe:
            print('[' + label + '] ' + safe, flush=True)


def primary_complete(work, fold=0, development=DEVELOPMENT, experiment_split=None, cache_content=None, arm=None):
    """Complete twelve-epoch, top-3 SWA, K42 export of the fixed 870 development studies."""
    if fold != 0:
        raise ValueError('Raptor64 is fold-0 only')
    record = json.loads((Path(work) / 'run_fold0.json').read_bytes())
    meta = json.loads((Path(work) / 'raptor_ft_fold0.json').read_bytes())
    if experiment_split is not None and record.get('raptor64_experiment_split') != experiment_split:
        raise ValueError('Export experiment split differs')
    if cache_content is not None and record.get('raptor64_cache_content_sha256') != cache_content:
        raise ValueError('Export cache identity differs')
    if arm is not None and (record.get('raptor64_policy') or {}).get('arm') != arm:
        raise ValueError('Export arm differs')
    if record.get('mode') == 'full' or (arm is not None and arm in FULL_ARMS):
        return full_complete(work, record, meta, experiment_split, cache_content, arm)
    if (record.get('fold') != 0 or record.get('epochs_requested') != EPOCHS or record.get('epochs_completed') != EPOCHS or
            record.get('stopped_by_time_budget') or record.get('checkpoint_kind') != 'swa_topk' or
            record.get('comparison_endpoint') != 'K42 primary (selection K16)' or record.get('export', {}).get('k_eval') != EXPORT_K or
            record['export'].get('n_val') != development or record['export'].get('n_gold') != 0 or
            len(record.get('selected_epochs', [])) != 3 or len(set(record['selected_epochs'])) != 3 or
            record.get('history') != meta.get('hist') or meta.get('partial') or [h['ep'] for h in meta.get('hist', [])] != list(range(EPOCHS))):
        raise ValueError('Complete twelve-epoch K42 primary export required')
    file = record['checkpoint_file']
    if file != 'raptor_ft_fold0_swa.pt' or (Path(work) / file).is_symlink() or sha(Path(work) / file) != record['checkpoint_sha256']:
        raise ValueError('Primary checkpoint identity differs')
    if (Path(work) / 'oof_fold0.csv').is_symlink() or not (Path(work) / 'oof_fold0.csv').is_file():
        raise ValueError('Primary prediction artifact missing')
    return record


def full_complete(work, record, meta, experiment_split, cache_content, arm):
    """Full-data fit: twelve epochs, pre-registered fixed-epoch SWA, no predictions."""
    policy = record.get('raptor64_policy') or {}
    if (arm not in FULL_ARMS or policy.get('arm') != arm or policy.get('mode') != 'full' or record.get('mode') != 'full' or
            record.get('epochs_requested') != EPOCHS or record.get('epochs_completed') != EPOCHS or record.get('stopped_by_time_budget') or
            record.get('checkpoint_kind') != 'swa_fixed_epochs' or sorted(record.get('selected_epochs', [])) != sorted(policy.get('swa_epochs', [None])) or
            record.get('export') is not None or record.get('history') != meta.get('hist') or meta.get('partial') or
            [h['ep'] for h in meta.get('hist', [])] != list(range(EPOCHS)) or
            (experiment_split is not None and record.get('raptor64_experiment_split') != experiment_split) or
            (cache_content is not None and record.get('raptor64_cache_content_sha256') != cache_content)):
        raise ValueError('Complete twelve-epoch full-data fixed-epoch SWA record required')
    file = record['checkpoint_file']
    if file != 'raptor_ft_fold0_swa.pt' or (Path(work) / file).is_symlink() or sha(Path(work) / file) != record['checkpoint_sha256']:
        raise ValueError('Primary checkpoint identity differs')
    return record


def published_manifest(dataset, version):
    """Read the exact published version; the persister deletes local staging."""
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.run(['kaggle', 'datasets', 'download', f'{dataset}/{version}', '-f', 'payload_manifest.json', '-p', directory,
                                 '--unzip', '-o', '-q'], capture_output=True, text=True, timeout=300)
        path = Path(directory) / 'payload_manifest.json'
        if result.returncode or not path.is_file() or path.is_symlink() or path.stat().st_size > 1024 * 1024:
            raise ValueError('Exact published backup manifest unavailable')
        return json.loads(path.read_bytes())


def final_verified(c, status):
    p = c['persistence']; resume = Path(c['workdir']) / 'raptor_resume_fold0.pt'; st = resume.stat()
    final = status.get('final') or {}; version = status.get('verified_version')
    if (status.get('dataset') != p['dataset'] or status.get('fold') != 0 or final.get('state') != 'verified' or
            status.get('verified_epoch') != FINAL_EPOCH or final.get('epoch') != FINAL_EPOCH or type(version) is not int or version <= 0 or
            final.get('version') != version or status.get('verified_signature') != [st.st_size, st.st_mtime_ns]):
        raise ValueError('Latest final epoch11 backup is not verified')
    payload = published_manifest(p['dataset'], version)
    if payload.get('dataset') != p['dataset'] or payload.get('fold') != 0 or payload.get('epoch') != FINAL_EPOCH or payload.get('resume_sha256') != sha(resume):
        raise ValueError('Final backup payload differs from latest resume')
    after = resume.stat()
    if (after.st_size, after.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
        raise ValueError('Resume changed during final backup verification')
    return {'epoch': FINAL_EPOCH, 'version': version, 'resume_sha256': payload['resume_sha256']}


def run(c, *, popen=subprocess.Popen, probe=runtime_probe, privacy=private_dataset, sleep=time.sleep, read_status=None, install_signals=True,
        monotonic=time.monotonic):
    m, policy, stage, init = validate(c); env = environment(c); arm = c['arm']
    lock = Path(c['controller_lock']); lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.is_symlink():
        raise ValueError('Unsafe controller lock')
    with open(lock, 'a+') as locked:
        try:
            fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another controller owns this arm lock') from None
        work = Path(c['workdir']); work.mkdir(parents=True, exist_ok=True)
        runtime = probe(c, env); privacy(c['persistence']['dataset'])
        fingerprint = {'controller_sha256': sha(__file__), 'config_sha256_canonical': hashlib.sha256(json.dumps(c, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                       'version': VERSION, 'name': c['name'], 'arm': arm, 'init': ARMS[arm][0], 'head': ARMS[arm][1], 'vm_session_id': c['vm_session_id'],
                       'package_manifest_sha256': c['package_manifest_sha256'], 'cache_manifest_sha256': c['manifest_sha256'],
                       'cache_content_sha256': c['content_sha256'], 'initializer_descriptor_sha256': c['initializer_descriptor_sha256'],
                       'initializer_weights_sha256': init['weights_sha256'], 'persister_sha256': PERSISTER_SHA, 'policy': policy, 'runtime': runtime,
                       'workers': c.get('workers', 2), 'RAPTOR_PRELOAD': '0', 'fold': 0, 'experiment_split': policy.get('experiment_split'),
                       'resume': bool(c.get('resume')), 'worker_rng_resume': 'Main RNG only; persistent worker RNG/prefetch not restored',
                       'benchmark': 'Forward/backward probe on four corpus rows only; no throughput benchmark'}
        atomic_json(work / f'{PREFIX}-run-fingerprint.json', fingerprint)
        p = c['persistence']
        pcmd = [sys.executable, '-u', c['persister'], '--workdir', c['workdir'], '--fold', '0', '--dataset', p['dataset'], '--staging', p['staging'],
                '--status-file', p['status_file'], '--keep-old-versions']
        persist = train = None; interrupted = {'signal': None, 'deadline': None}; old = {}; threads = []; receipt = {**fingerprint, 'status': 'FAIL'}

        def stop(signum, _frame):
            if interrupted['signal'] is not None:
                return
            interrupted['signal'] = signum; interrupted['deadline'] = monotonic() + 60
            if train is not None and train.poll() is None:
                try:
                    os.killpg(train.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if install_signals:
            for signum in (signal.SIGTERM, signal.SIGINT):
                old[signum] = signal.signal(signum, stop)
        try:
            persist = popen(pcmd, env=env, cwd=c['package_root'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            threads.append(threading.Thread(target=stream, args=(persist, 'backup', work / f'{PREFIX}-backup.log'), daemon=True)); threads[-1].start()
            if interrupted['signal']:
                raise RuntimeError('Interrupted before training launch')
            train = popen([sys.executable, '-u', str(Path(c['package_root']) / 'tracka_entry.py')], env=env, cwd=c['package_root'],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            threads.append(threading.Thread(target=stream, args=(train, 'training', work / f'{PREFIX}-training.log'), daemon=True)); threads[-1].start()
            while train.poll() is None:
                if persist.poll() is not None:
                    stop(signal.SIGTERM, None)
                if interrupted['deadline'] is not None and monotonic() >= interrupted['deadline']:
                    break
                sleep(.5)
            receipt['training_returncode'] = train.returncode
        except Exception as error:
            receipt['failure'] = type(error).__name__
        finally:
            if train is not None and train.poll() is None:
                stop(signal.SIGTERM, None)
                try:
                    train.wait(timeout=max(0, interrupted['deadline'] - monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(train.pid, signal.SIGKILL); train.wait(timeout=60)
            if train is not None:
                receipt['training_returncode'] = train.returncode
            if persist is not None:
                if persist.poll() is None:
                    persist.send_signal(signal.SIGUSR1)
                try:
                    persist.wait(timeout=900)  # Persister final-timeout 840 plus shutdown margin; no training cap.
                except subprocess.TimeoutExpired:
                    persist.send_signal(signal.SIGTERM)
                    try:
                        persist.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(persist.pid, signal.SIGKILL); persist.wait(timeout=60)
                receipt['persister_returncode'] = persist.returncode
            for t in threads:
                t.join(timeout=1)
            for signum, handler in old.items():
                signal.signal(signum, handler)
        try:
            if train is None or train.returncode != 0 or interrupted['signal'] or persist is None or persist.returncode != 0:
                raise ValueError('Training/controller/backup subprocess failed')
            primary = primary_complete(work, 0, DEVELOPMENT, policy.get('experiment_split'), c['content_sha256'], arm)
            status = read_status(c) if read_status else json.loads(Path(p['status_file']).read_bytes())
            backup = final_verified(c, status); privacy(p['dataset'])
            receipt.update(status='PASS', checkpoint_sha256=primary['checkpoint_sha256'], backup=backup)
        except Exception as error:
            receipt.setdefault('failure', type(error).__name__)
        receipt['interrupted_signal'] = interrupted['signal']; receipt['resume_preserved'] = (work / 'raptor_resume_fold0.pt').is_file()
        atomic_json(work / f'{PREFIX}-run-receipt.json', receipt)
        return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('config', type=Path); parser.add_argument('--sha256', required=True)
    a = parser.parse_args(); config = checked(a.config, a.sha256); result = run(config)
    print(json.dumps({'status': result['status'], 'arm': config['arm'], 'failure': result.get('failure'), 'resume_preserved': result['resume_preserved']}))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
