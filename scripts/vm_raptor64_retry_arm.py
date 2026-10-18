"""VM (detached, nohup): retry one resume-only recovery arm after a non-scientific failure (e.g. export OOM).

Usage: python vm_raptor64_retry_arm.py ARM VERSION RESUME_SHA256 CONFIG_SHA256 FINAL_DATASET WAIT_FOR_ARM
1. Waits until the original controller is dead and (if given) WAIT_FOR_ARM's autopilot phase is terminal (frees GPU).
2. Preserves the failed attempt: renames the workdir, launch receipt, controller log and export logs with an
   -attemptN suffix (never deletes).
3. Restores the exact pinned resume into a fresh workdir, re-checks the unchanged reviewed config SHA, launches
   the pinned controller detached, waits for it, and on a PASS receipt runs the reviewed exporter once.
State: /content/raptor64-ARM-retry.json.
"""
import datetime, hashlib, json, os, subprocess, sys, time
from pathlib import Path

arm, version, resume_sha, config_sha, final_dataset, wait_arm = sys.argv[1:7]
root = Path('/content'); tools = root / 'raptor64-tools/scripts'; env = {**os.environ, 'KAGGLE_CONFIG_DIR': '/content'}
state_path = root / f'raptor64-{arm}-retry.json'; state = {'arm': arm, 'started': datetime.datetime.now(datetime.timezone.utc).isoformat()}
controller = tools / 'colab_raptor64_run.py'; RUNNER = '7691c13111262ea6106e421e8694903cb5cdd3e5a21630eb3858118ea24aa972'


def save(**kw):
    state.update(kw, updated=datetime.datetime.now(datetime.timezone.utc).isoformat()); state_path.write_text(json.dumps(state, indent=2) + '\n')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


launch = root / f'raptor64-{arm}-launch.json'
old_pid = json.loads(launch.read_bytes())['pid']
save(phase='waiting', waiting_for=[f'pid {old_pid} exit', f'{wait_arm} terminal'])
while Path(f'/proc/{old_pid}').exists():
    time.sleep(30)
if wait_arm != '-':
    while True:
        phase = json.loads((root / 'raptor64-autopilot.json').read_bytes())['arms'].get(wait_arm, {}).get('phase')
        if phase in ('exported', 'export_failed', 'failed', 'stopped_health'):
            break
        time.sleep(30)
work = root / f'raptor64-{arm}-fold0'
old_receipt = json.loads((work / 'raptor64-run-receipt.json').read_bytes())
if old_receipt.get('status') == 'PASS':
    save(phase='not_needed', note='original controller PASS'); sys.exit(0)
n = 1
while (root / f'raptor64-{arm}-fold0-attempt{n}').exists():
    n += 1
preserved = {}
for src in [work, launch, root / f'raptor64-{arm}-controller-private.log', root / f'raptor64-{arm}-export-private.log',
            root / f'raptor64-{arm}-persist-status.json', root / f'raptor64-{arm}-backup-staging']:
    if src.exists():
        dst = src.with_name(src.name + (f'.attempt{n}' if src.is_file() else f'-attempt{n}'))
        src.rename(dst); preserved[src.name] = dst.name
save(phase='restoring', failed_attempt=n, failed_receipt={k: old_receipt.get(k) for k in ('status', 'failure', 'training_returncode', 'persister_returncode')},
     preserved=preserved)
r = subprocess.run([sys.executable, '/content/vm_raptor64_restore_resume.py', arm, version, resume_sha], capture_output=True, text=True, env=env)
if r.returncode:
    save(phase='restore_failed', detail=r.stdout[-500:] + r.stderr[-500:]); sys.exit(1)
config = root / f'raptor64-{arm}-run-config.json'
if sha(config) != config_sha or sha(controller) != RUNNER:
    save(phase='pin_failed'); sys.exit(1)
log = root / f'raptor64-{arm}-controller-private.log'
with log.open('xb') as stream:
    p = subprocess.Popen([sys.executable, '-u', str(controller), str(config), '--sha256', config_sha], env={**env, 'PYTHONPATH': f'/content/raptor64-{arm}-pkg'},
                         cwd=str(root), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
with launch.open('x') as f:
    json.dump({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'arm': arm, 'pid': p.pid, 'config_sha256': config_sha,
               'controller_sha256': RUNNER, 'log': str(log), 'launched_by': f'retry attempt {n + 1}'}, f, indent=2)
save(phase='running', pid=p.pid)
p.wait()
receipt = json.loads((work / 'raptor64-run-receipt.json').read_bytes())
save(phase='controller_ended', controller_status=receipt.get('status'), failure=receipt.get('failure'), checkpoint_sha256=receipt.get('checkpoint_sha256'),
     backup=receipt.get('backup'))
if receipt.get('status') != 'PASS':
    save(phase='failed'); sys.exit(1)
c = json.loads(config.read_bytes())
cmd = [sys.executable, '-u', str(tools / 'colab_raptor64_export_terminal.py'), '--arm', arm, '--workdir', c['workdir'], '--config', str(config),
       '--config-sha256', config_sha, '--package-manifest-sha256', c['package_manifest_sha256'], '--cache-content-sha256', c['content_sha256'],
       '--runner', str(controller), '--runner-sha256', RUNNER, '--controller-sha256', RUNNER, '--dataset', final_dataset, '--create-fresh', final_dataset,
       '--staging', f'/content/raptor64-{arm}-export-staging', '--verify-root', f'/content/raptor64-{arm}-export-verify',
       '--receipt', f'/content/raptor64-{arm}-export-receipt.json', '--source-revision', 'unversioned']
with open(root / f'raptor64-{arm}-export-private.log', 'ab') as out:
    e = subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT)
er = root / f'raptor64-{arm}-export-receipt.json'
d = json.loads(er.read_bytes()) if er.exists() else {}
save(phase='exported' if e.returncode == 0 and d.get('state') == 'verified' else 'export_failed', export={'returncode': e.returncode, 'state': d.get('state')})
