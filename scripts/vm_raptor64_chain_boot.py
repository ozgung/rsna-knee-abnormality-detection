"""Chained-VM bootstrap for Raptor64 arms (runs ON a fresh Colab VM, detached; generalised from the it4 vm-chain).

Reads /content/chain-params.json:
  {session, arms: [...], new_run_started_utc_epoch, excluded_resume_sha256: [...], pins: {uploaded file: sha256},
   bundle, stage_config_sha256, runner_sha256, source_revision, final_date, health_stop_epoch1 (or null), diag_name}
Steps: verify uploaded pins; start the per-minute diagnostics logger (uptime, host RAM, GPU memory/utilisation,
largest processes); extract the bundle; setup (pinned packages + initializers); stage the full v4 336 cache with the
unchanged loader (resumable retries); per arm: inputs, private backup dataset, restore the latest backup of THIS run
(publish time after new_run_started and not an excluded resume), config; then launch the reviewed autopilot.
Progress: /content/chain-boot-state.json ({steps: {...}, status: running|launched|failed}). Never deletes artifacts.
"""
import datetime, hashlib, importlib.util, json, os, pathlib, shutil, subprocess, sys, tarfile, tempfile, time, traceback

root = pathlib.Path('/content')
STATE = root / 'chain-boot-state.json'
P = json.loads((root / 'chain-params.json').read_bytes())
env = {**os.environ, 'KAGGLE_CONFIG_DIR': '/content'}
t0 = time.time()
state = {'session': P['session'], 'started': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'running', 'steps': {}}

DIAG = r'''
import json, subprocess, sys, time
path = sys.argv[1]
while True:
    up = float(open("/proc/uptime").read().split()[0]); mem = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":"); mem[k] = int(v.split()[0])
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    procs = subprocess.run(["bash", "-c", "ps -eo rss,comm --sort=-rss | head -4"], capture_output=True, text=True).stdout.split()
    disk = subprocess.run(["df", "-k", "/content"], capture_output=True, text=True).stdout.split()
    with open(path, "a") as f:
        f.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "uptime_s": up, "mem_total_kb": mem["MemTotal"],
                            "mem_available_kb": mem["MemAvailable"], "swap_free_kb": mem.get("SwapFree"), "gpu": smi, "top_rss": procs[2:8],
                            "disk_avail_kb": int(disk[-3]) if len(disk) >= 3 and disk[-3].isdigit() else None}) + "\n")
    time.sleep(60)
'''


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def step(name, **kw):
    state['steps'][name] = {'t': round(time.time() - t0, 1), **kw}
    tmp = STATE.with_name(STATE.name + '.tmp'); tmp.write_text(json.dumps(state, indent=1) + '\n'); tmp.replace(STATE)
    print(name, json.dumps(kw)[:400], flush=True)


def retry(cmd, attempts=3, wait=30, **kw):
    for attempt in range(attempts):
        r = subprocess.run(cmd, capture_output=True, text=True, env=kw.pop('env', env), **kw)
        if r.returncode == 0:
            return r
        time.sleep(wait)
    raise RuntimeError(f'{cmd[2:4]} failed after {attempts} attempts: ' + r.stderr[-1500:])


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def restore_latest(arm, vm):
    """Restore the arm's latest verified backup of THIS run into a fresh workdir; None when there is none."""
    dataset = vm.paths(arm)['dataset']; work = root / f'raptor64-{arm}-fold0'
    with tempfile.TemporaryDirectory(dir='/content') as d:
        subprocess.run(['kaggle', 'datasets', 'download', dataset, '-f', 'payload_manifest.json', '-p', d, '-o', '-q'], capture_output=True, text=True, env=env)
        mp = pathlib.Path(d) / 'payload_manifest.json'
        m = json.loads(mp.read_bytes()) if mp.exists() else {}
        if m.get('resume_sha256') in P['excluded_resume_sha256'] or m.get('publish_started_at_epoch', 0) <= P['new_run_started_utc_epoch']:
            return None
        subprocess.run(['kaggle', 'datasets', 'download', dataset, '-f', 'raptor_resume_fold0.pt', '-p', d, '-o', '-q'], capture_output=True, text=True, env=env)
        for z in pathlib.Path(d).glob('*.zip'):
            shutil.unpack_archive(str(z), d); z.unlink()
        f = pathlib.Path(d) / 'raptor_resume_fold0.pt'
        if not f.exists() or sha(f) != m['resume_sha256']:
            raise RuntimeError(f'{arm}: restored resume sha differs')
        if work.exists():
            raise RuntimeError('Fresh workdir required for restore')
        work.mkdir(mode=0o700); shutil.move(str(f), work / 'raptor_resume_fold0.pt')
        return {'epoch': m['epoch'], 'resume_sha256': m['resume_sha256'], 'publish_started_at': m.get('publish_started_at')}


def main():
    for name, digest in P['pins'].items():
        if sha(root / name) != digest:
            raise RuntimeError(f'uploaded pin differs: {name}')
    os.chmod(root / 'kaggle.json', 0o600)
    (root / 'chain-diag.py').write_text(DIAG)
    subprocess.Popen([sys.executable, str(root / 'chain-diag.py'), str(root / P['diag_name'])], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    step('diagnostics_started')
    if not (root / 'raptor64-tools').exists():
        with tarfile.open(root / P['bundle']) as t:
            t.extractall(root, filter='data')
    vm_path = root / 'raptor64-tools/scripts/colab_raptor64_vm.py'
    if not (root / 'raptor64-init-imagenet/initializer.json').is_file():
        r = subprocess.run([sys.executable, str(vm_path), 'setup'], capture_output=True, text=True, env=env)
        step('setup_returncode', returncode=r.returncode, note='the dread_v4 initializer is not needed by imagenet arms')
    if not (root / 'raptor64-init-imagenet/initializer.json').is_file():
        raise RuntimeError('ImageNet initializer missing after setup')
    setup = json.loads((root / 'raptor64-runtime-setup.json').read_bytes()) if (root / 'raptor64-runtime-setup.json').exists() else {}
    step('setup_done', device=setup.get('device'), vram_bytes=setup.get('vram_bytes'), cpu_count=setup.get('cpu_count'))
    if not (root / 'raptor64_cache/staging-receipt.json').exists():
        for attempt in range(4):   # transient download errors resume from completed files
            r = subprocess.run([sys.executable, str(vm_path), 'stage', '--config-sha', P['stage_config_sha256'], '--workers', '8'],
                               capture_output=True, text=True, env=env)
            step(f'stage_attempt_{attempt}', returncode=r.returncode, tail=r.stdout.strip()[-300:])
            if r.returncode == 0:
                break
            time.sleep(30)
    receipt = json.loads((root / 'raptor64_cache/staging-receipt.json').read_bytes())
    if receipt.get('status') != 'PASS' or receipt.get('studies') != 4345:
        raise RuntimeError('staging did not PASS')
    step('staged', content_sha256=receipt['content_sha256'], manifest_sha256=receipt['manifest_sha256'], stage_seconds=receipt.get('elapsed_seconds'))
    vm = load('vmtool', vm_path); runner_path = root / 'raptor64-tools/scripts/colab_raptor64_run.py'
    if sha(runner_path) != P['runner_sha256']:
        raise RuntimeError('runner differs from the reviewed pin')
    runner = load('runner', runner_path)
    for arm in P['arms']:   # e.g. na-orig needs the nartaa_oai initializer prepared by setup
        if not (root / f'raptor64-init-{vm.INIT[arm]}/initializer.json').is_file():
            raise RuntimeError(f'{vm.INIT[arm]} initializer missing after setup')
    configs = {}
    for arm in P['arms']:
        for s in (['inputs', arm], ['bootstrap', arm]):
            retry([sys.executable, str(vm_path), *s])
        restore = restore_latest(arm, vm)
        c = vm.make_config(arm, vm_session_id=P['session'])
        if restore:
            c['resume'] = True; c['resume_sha256'] = restore['resume_sha256']
        runner.validate(c)
        cp = root / f'raptor64-{arm}-run-config.json'
        if cp.exists():
            raise RuntimeError('existing config preserved; reconcile instead of overwriting')
        cp.write_text(json.dumps(c, indent=2) + '\n'); configs[arm] = sha(cp)
        step(f'{arm}_ready', restore=restore, config_sha256=configs[arm])
    plan = {'session': P['session'], 'runner_sha256': P['runner_sha256'], 'source_revision': P['source_revision'], 'ready_timeout': 2700,
            'arms': {arm: {'config_sha256': digest, 'final_dataset': f'ozgung/knee-raptor64-{arm}-fold0-final-{P["final_date"]}',
                           'extra_env': {'RAPTOR64_TEACHER': '/content/teacher-dread-v4-k42-private.npz'},
                           **({'health_stop_k16': P['health_stop_epoch1']} if P.get('health_stop_epoch1') is not None else {})}
                     for arm, digest in configs.items()}}
    (root / 'raptor64-autopilot-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    p = subprocess.Popen([sys.executable, '-u', str(root / 'raptor64-tools/scripts/vm_raptor64_autopilot.py')], cwd=str(root),
                         stdout=open(root / 'raptor64-autopilot-private.log', 'ab'), stderr=subprocess.STDOUT, start_new_session=True, env=env)
    state['status'] = 'launched'
    step('launched', autopilot_pid=p.pid, plan_sha256=sha(root / 'raptor64-autopilot-plan.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        state['status'] = 'failed'; step('failed', error=f'{type(e).__name__}: {e}'[:1500], traceback=traceback.format_exc()[-1500:])
        sys.exit(1)
