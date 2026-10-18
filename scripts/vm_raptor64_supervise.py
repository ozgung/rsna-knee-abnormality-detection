"""Foreground VM supervision cell: keeps the kernel busy while any raptor64 controller lives.

Starts/stops nothing. Every 60 s refreshes /content/raptor64-health.json through the reviewed
`colab_raptor64_vm.py health` subcommand; exits when no launched controller is alive for 10 minutes
(allowing a later arm to be launched by a separate exec). Single instance via a lock file.
"""
import fcntl, json, os, subprocess, sys, time
from pathlib import Path
root = Path('/content'); lock = open(root / 'raptor64-supervision.lock', 'a+')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print('Another supervision cell is active'); sys.exit(0)
state = root / 'raptor64-supervision.json'; idle_since = None; started = time.time()
while True:
    env = {**os.environ, 'KAGGLE_CONFIG_DIR': '/content'}
    # Launch requests arrive by `colab upload` (works while this cell keeps the kernel busy).
    for request in sorted(root.glob('raptor64-launch-request-*.json')):
        try:
            r = json.loads(request.read_bytes()); arm = r['arm']; sha = r['sha']
            if request.name != f'raptor64-launch-request-{arm}.json':
                raise ValueError('request name/arm mismatch')
            p = subprocess.run([sys.executable, '/content/raptor64-tools/scripts/colab_raptor64_vm.py', 'launch', arm, '--sha', sha],
                               capture_output=True, text=True, env=env)
            result = {'arm': arm, 'sha': sha, 'returncode': p.returncode, 'stdout': p.stdout[-2000:], 'stderr': p.stderr[-2000:]}
        except Exception as error:
            result = {'error': type(error).__name__, 'detail': str(error)[:300]}
        (root / (request.stem + '.result.json')).write_text(json.dumps(result, indent=2) + '\n')
        request.rename(root / (request.stem + '.consumed.json'))
    subprocess.run([sys.executable, '/content/raptor64-tools/scripts/colab_raptor64_vm.py', 'health'], capture_output=True, env=env)
    alive = []
    for launch in root.glob('raptor64-*-launch.json'):
        pid = json.loads(launch.read_bytes())['pid']
        if Path(f'/proc/{pid}').exists():
            alive.append(launch.name.split('-launch')[0])
    now = time.time(); idle_since = None if alive else (idle_since or now)
    state.write_text(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'alive': sorted(alive),
                                 'supervising_seconds': round(now - started), 'pid': os.getpid()}) + '\n')
    if idle_since and now - idle_since > 600:
        print('No live raptor64 controller for 10 minutes; supervision ends'); break
    time.sleep(60)
