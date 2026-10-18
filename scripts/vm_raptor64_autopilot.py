"""VM foreground autopilot: launch the reviewed arms, supervise, apply the dr-orig health stop, export privately.

Reads /content/raptor64-autopilot-plan.json: {"session", "runner_sha256", "source_revision",
"arms": {ARM: {"config_sha256", "final_dataset"}}}. For each arm: start the pinned controller detached
(own session), refresh /content/raptor64-health.json every 60 s, and when the controller ends with a PASS
receipt run the reviewed terminal exporter once to the named PRIVATE final dataset (create-fresh). A FAIL
receipt is recorded and not exported as complete. dr-orig: if its epoch-1 K16 selection AUC is below 0.85,
SIGTERM its controller (which finalizes its backup) and record the stop. Runs until every arm is handled,
so the kernel never sits idle mid-run; works with the laptop offline. State: /content/raptor64-autopilot.json.
"""
import datetime, fcntl, hashlib, json, os, re, signal, subprocess, sys, time
from pathlib import Path

root = Path('/content'); tools = root / 'raptor64-tools/scripts'
lock = open(root / 'raptor64-autopilot.lock', 'a+')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print('Another autopilot is active'); sys.exit(0)
plan = json.loads((root / 'raptor64-autopilot-plan.json').read_bytes())
env = {**os.environ, 'KAGGLE_CONFIG_DIR': '/content'}
state_path = root / 'raptor64-autopilot.json'
STOP_K16 = 0.85


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


state = json.loads(state_path.read_bytes()) if state_path.exists() else {'started': now(), 'arms': {}}


def save():
    state['updated'] = now(); tmp = state_path.with_name(state_path.name + '.tmp'); tmp.write_text(json.dumps(state, indent=2) + '\n'); tmp.replace(state_path)


controller = tools / 'colab_raptor64_run.py'
if sha(controller) != plan['runner_sha256']:
    raise SystemExit('Controller source differs from the reviewed pin')
procs = {}
for arm, spec in plan['arms'].items():
    s = state['arms'].setdefault(arm, {})
    config = root / f'raptor64-{arm}-run-config.json'
    if sha(config) != spec['config_sha256']:
        raise SystemExit(f'{arm}: reviewed config SHA differs')
    launch = root / f'raptor64-{arm}-launch.json'
    if launch.exists():
        s.setdefault('note', 'launch receipt existed; supervising existing PID'); s['pid'] = json.loads(launch.read_bytes())['pid']
        continue
    log = root / f'raptor64-{arm}-controller-private.log'
    with log.open('xb') as stream:
        p = subprocess.Popen([sys.executable, '-u', str(controller), str(config), '--sha256', spec['config_sha256']],
                             env={**env, **spec.get('extra_env', {}), 'PYTHONPATH': f'/content/raptor64-{arm}-pkg'},
                             cwd=str(root), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    procs[arm] = p
    receipt = {'utc': now(), 'arm': arm, 'pid': p.pid, 'config_sha256': spec['config_sha256'], 'controller_sha256': plan['runner_sha256'],
               'log': str(log), 'launched_by': 'autopilot'}
    with launch.open('x') as f:
        json.dump(receipt, f, indent=2)
    s.update(pid=p.pid, launched=receipt['utc'], phase='running')
    save()


def alive(arm):
    p = procs.get(arm)
    if p is not None:
        return p.poll() is None
    return Path(f"/proc/{state['arms'][arm]['pid']}").exists()


def epoch_auc(arm, epoch):
    log = root / f'raptor64-{arm}-controller-private.log'
    if not log.exists():
        return None
    m = re.findall(rf'\[training\] ep{epoch} loss [0-9.eE+-]+ \| OOF masked macro-AUC ([0-9.eE+-]+)', log.read_text(errors='replace'))
    return float(m[-1]) if m else None


def export(arm, spec):
    c = json.loads((root / f'raptor64-{arm}-run-config.json').read_bytes())
    cmd = [sys.executable, '-u', str(tools / 'colab_raptor64_export_terminal.py'), '--arm', arm, '--workdir', c['workdir'],
           '--config', str(root / f'raptor64-{arm}-run-config.json'), '--config-sha256', spec['config_sha256'],
           '--package-manifest-sha256', c['package_manifest_sha256'], '--cache-content-sha256', c['content_sha256'],
           '--runner', str(controller), '--runner-sha256', plan['runner_sha256'], '--controller-sha256', plan['runner_sha256'],
           '--dataset', spec['final_dataset'], '--create-fresh', spec['final_dataset'],
           '--staging', f'/content/raptor64-{arm}-export-staging', '--verify-root', f'/content/raptor64-{arm}-export-verify',
           '--receipt', f'/content/raptor64-{arm}-export-receipt.json', '--source-revision', plan['source_revision'],
           '--ready-timeout', str(plan.get('ready_timeout', 900))]
    with open(root / f'raptor64-{arm}-export-private.log', 'ab') as log:
        r = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt = root / f'raptor64-{arm}-export-receipt.json'
    out = {'returncode': r.returncode, 'finished': now()}
    if receipt.exists():
        d = json.loads(receipt.read_bytes()); out.update({k: d.get(k) for k in ('state', 'dataset', 'version', 'manifest_sha256')})
    return out


while True:
    subprocess.run([sys.executable, str(tools / 'colab_raptor64_vm.py'), 'health'], capture_output=True, env=env)
    pending = False
    for arm, spec in plan['arms'].items():
        s = state['arms'][arm]
        if s.get('phase') in ('exported', 'export_failed', 'failed', 'stopped_health'):
            continue
        pending = True
        stop_k16 = spec.get('health_stop_k16', STOP_K16 if arm == 'dr-orig' else None)
        if stop_k16 is not None and not s.get('health_checked'):
            auc = epoch_auc(arm, 1)
            if auc is not None:
                s['epoch1_k16'] = auc; s['health_checked'] = True
                if auc < stop_k16 and alive(arm):
                    os.kill(s['pid'], signal.SIGTERM); s['health_stop'] = {'utc': now(), 'reason': f'epoch-1 K16 {auc} < {stop_k16}'}
                save()
        if alive(arm):
            continue
        receipt = root / f'raptor64-{arm}-fold0/raptor64-run-receipt.json'
        r = json.loads(receipt.read_bytes()) if receipt.exists() else {}
        s.update(ended=now(), controller_status=r.get('status'), failure=r.get('failure'), checkpoint_sha256=r.get('checkpoint_sha256'), backup=r.get('backup'))
        if s.get('health_stop'):
            s['phase'] = 'stopped_health'
        elif r.get('status') == 'PASS':
            s['phase'] = 'exporting'; save()
            s['export'] = export(arm, spec)
            s['phase'] = 'exported' if s['export'].get('returncode') == 0 and s['export'].get('state') == 'verified' else 'export_failed'
        else:
            s['phase'] = 'failed'
        save()
    save()
    if not pending:
        print(json.dumps(state, indent=1)); break
    time.sleep(60)
