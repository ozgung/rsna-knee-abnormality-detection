"""Local chained-VM driver for Raptor64 arms on Colab (generalised from the it4 chain driver).

One Colab session at a time under a private `colab --config` state file; per-minute VM diagnostics; auto-chain to a
fresh VM on loss, resuming each arm from its latest verified backup of this run; GPU preference order with fallback.

Usage: python scripts/colab_raptor64_chain.py LANE.json FIRST_INDEX
LANE.json: {prefix, date, colab_config, gpus: ["H100", "A100"], first_gpu_wait_s, max_vms, arms, state_dir,
            min_life_s, short_death_limit, params: {...VM chain-params without session...}, uploads: [[local, remote], ...],
            fetch: [remote paths under /content], post_launch: [local script, remote output] or null}
Sessions: PREFIX-DATE for index 1, PREFIX-rN-DATE for index N >= 2.
State/log under state_dir: chain-state.json, chain-driver.log, diag/SESSION.jsonl, final/.
Exit 0: every arm exported; 2: setup/boot failure; 3: short-lived VM limit; 4: arm failure; 5: VM cap reached.
"""
import json, pathlib, re, subprocess, sys, time

COLAB = '~/.local/bin/colab'
CPY = '~/.local/share/uv/tools/google-colab-cli/bin/python'
TERMINAL = ('exported', 'export_failed', 'failed', 'stopped_health')
LAUNCHER = '''import json, pathlib, subprocess, sys
s = pathlib.Path("/content/chain-boot-state.json")
if s.exists():
    print("BOOT_ALREADY " + json.loads(s.read_bytes()).get("status", "?"))
else:
    subprocess.Popen([sys.executable, "-u", "/content/vm_raptor64_chain_boot.py"], cwd="/content", stdout=open("/content/chain-boot.log", "ab"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    print("BOOT_STARTED")
'''
RECONNECT = '''"""Refresh the proxy token of the exact existing runtime; never allocate or stop a VM."""
import json, os, requests
from colab_cli.common import state
from colab_cli.state import SessionState
from colab_cli.commands.session import spawn_keep_alive
state.config_path = CONFIG
hits = [a for a in state.client.list_assignments() if a.endpoint == ENDPOINT]
assert len(hits) == 1, 'Exact existing runtime not present; no allocation/restart allowed'
proxy = hits[0].runtime_proxy_info
headers = {'X-Colab-Client-Agent': 'colab-cli', 'X-Colab-Runtime-Proxy-Token': proxy.token, 'Authorization': 'token ' + proxy.token}
r = requests.get(proxy.url.rstrip('/') + '/api/kernels', headers=headers, timeout=30)
assert r.status_code == 200, f'kernel listing HTTP {r.status_code}'
kid = sorted(r.json(), key=lambda k: k.get('last_activity', ''))[-1]['id']
previous = state.store.get(NAME)
assert previous is None or previous.endpoint == ENDPOINT, 'Session alias points to a different runtime'
s = previous or SessionState(name=NAME, token=proxy.token, url=proxy.url, endpoint=ENDPOINT, variant='GPU', accelerator=ACCEL, kernel_id=kid)
s.token = proxy.token; s.url = proxy.url; s.kernel_id = kid
state.client.keep_alive_assignment(ENDPOINT); state.store.add(s)
alive = False
if s.keep_alive_pid:
    try:
        os.kill(s.keep_alive_pid, 0); alive = True
    except ProcessLookupError:
        pass
if not alive:
    s.keep_alive_pid = spawn_keep_alive(ENDPOINT, NAME, auth_provider=state.auth_provider, config_path=state.config_path); state.store.add(s)
print(json.dumps({'status': 'RECONNECTED_EXISTING', 'session': NAME, 'endpoint': ENDPOINT, 'kernel_id': kid, 'keep_alive_pid': s.keep_alive_pid}))
'''


def session_name(prefix, date, n):
    return f'{prefix}-{date}' if n == 1 else f'{prefix}-r{n}-{date}'


def lost(text):
    return 'appears to be lost' in text or 'Session not found' in text or 'not found in local state' in text


class Driver:
    def __init__(self, lane):
        self.L = lane; self.dir = pathlib.Path(lane['state_dir']); (self.dir / 'diag').mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / 'chain-state.json'; self.log_path = self.dir / 'chain-driver.log'
        self.state = json.loads(self.state_path.read_bytes()) if self.state_path.exists() else {'vms': [], 'new_run_started_utc_epoch': time.time()}
        self.save()

    def log(self, msg):
        line = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + msg
        with open(self.log_path, 'a') as f:
            f.write(line + '\n')
        print(line, flush=True)

    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=1) + '\n')

    def colab(self, *args, timeout=600):
        try:
            r = subprocess.run([COLAB, '--config', self.L['colab_config'], *args], capture_output=True, text=True, timeout=timeout)
            text = r.stdout + r.stderr
        except subprocess.TimeoutExpired as e:
            return 124, str(e)
        return r.returncode, '\n'.join(l for l in text.splitlines() if 'new version' not in l and 'colab update' not in l and 'silence this' not in l)

    def endpoint_of(self, name):
        return json.loads(pathlib.Path(self.L['colab_config']).read_bytes())[name]['endpoint']

    def reconnect(self, name, endpoint, accel):
        src = (f'CONFIG = {self.L["colab_config"]!r}\nNAME = {name!r}\nENDPOINT = {endpoint!r}\nACCEL = {accel!r}\n') + RECONNECT
        path = self.dir / f'reconnect-{name}.py'; path.write_text(src)
        r = subprocess.run([CPY, str(path)], capture_output=True, text=True, timeout=300)
        return (r.stdout + r.stderr).strip().splitlines()[-1:]

    def allocate(self, name):
        for i, gpu in enumerate(self.L['gpus']):
            deadline = time.time() + (self.L['first_gpu_wait_s'] if i < len(self.L['gpus']) - 1 else 40 * 180)
            attempt = 0
            while time.time() < deadline:
                rc, text = self.colab('new', '-s', name, '--gpu', gpu, timeout=900)
                if rc == 0 and 'READY' in text:
                    return gpu, text
                last = text.strip().splitlines()[-1][:200] if text.strip() else str(rc)
                self.log(f'{name}: {gpu} allocation attempt {attempt} failed: {last}')
                self.state.setdefault('allocation_failures', []).append({'utc': time.strftime('%H:%M:%S', time.gmtime()), 'gpu': gpu, 'error': last})
                self.save(); attempt += 1
                if 'rejected accelerator' in text:
                    break   # no entitlement: next GPU type at once
                self.colab('sessions'); time.sleep(180)
        return None, text

    def status(self, name):
        rc, text = self.colab('exec', '-s', name, '-f', str(self.L['status_script']), '--timeout', '120', timeout=300)
        m = re.search(r'STATUS (\{.*\})', text)
        return (json.loads(m.group(1)) if m else None), text

    def fetch(self, name):
        out = self.dir / 'final'; out.mkdir(exist_ok=True)
        for f in self.L['fetch']:
            self.colab('download', '-s', name, '/content/' + f, str(out / f.replace('/', '__')), timeout=900)

    def run(self, first):
        short = sum(1 for v in self.state['vms'] if v.get('short_lived'))
        for n in range(first, first + self.L['max_vms']):
            if len(self.state['vms']) >= self.L['max_vms']:
                break
            name = session_name(self.L['prefix'], self.L['date'], n)
            rc, text = self.colab('sessions')
            if self.L['prefix'] in text:
                self.log('one of my sessions is still listed; refusing to allocate: ' + text[-300:]); return 2
            gpu, text = self.allocate(name)
            alloc = time.time(); vm = {'session': name, 'gpu': gpu, 'allocated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(alloc)), 'events': []}
            if gpu is None:
                self.state['vms'].append({**vm, 'allocation_failed': True}); self.save()
                self.log(f'{name}: allocation failed: {text[-400:]}'); return 2
            self.state['vms'].append(vm); self.save()
            endpoint = self.endpoint_of(name); vm['endpoint'] = endpoint; self.log(f'{name}: allocated {gpu} {endpoint}')
            params = {**self.L['params'], 'session': name, 'arms': self.L['arms'], 'new_run_started_utc_epoch': self.state['new_run_started_utc_epoch']}
            pp = self.dir / 'chain-params.json'; pp.write_text(json.dumps(params, indent=1))
            lp = self.dir / 'chain-launch.py'; lp.write_text(LAUNCHER)
            for src, dst in [*self.L['uploads'], [str(pp), 'chain-params.json']]:
                rc, text = self.colab('upload', '-s', name, src, '/content/' + dst, timeout=900)
                if rc != 0:
                    self.log(f'{name}: upload {dst} failed: {text[-300:]}'); self.colab('stop', '-s', name); return 2
            rc, text = self.colab('exec', '-s', name, '-f', str(lp), '--timeout', '60', timeout=300)
            self.log(f'{name}: boot launcher: {text.strip()[-120:]}')
            last_reconnect = time.time(); last_diag = 0; misses = 0; launched = False; post_done = False
            while True:
                time.sleep(60)
                if time.time() - last_reconnect > 1800:
                    vm['events'].append({'utc': time.strftime('%H:%M:%S', time.gmtime()), 'reconnect': self.reconnect(name, endpoint, gpu)})
                    last_reconnect = time.time()
                status, text = self.status(name)
                if status is None:
                    misses += 1
                    if lost(text) or misses >= 3:
                        break
                    continue
                misses = 0; vm['last_status'] = status; vm['last_status_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                if time.time() - last_diag > 300:
                    self.colab('download', '-s', name, '/content/' + self.L['params']['diag_name'], str(self.dir / 'diag' / f'{name}.jsonl'), timeout=300)
                    last_diag = time.time()
                boot = status['boot']
                if boot['status'] == 'failed':
                    self.colab('download', '-s', name, '/content/chain-boot-state.json', str(self.dir / f'chain-boot-{name}.json'))
                    self.log(f'{name}: boot failed (not a VM loss): {boot["error"]}'); self.save(); self.colab('stop', '-s', name); return 2
                if boot['status'] == 'launched' and not launched:
                    launched = True; vm['boot_seconds'] = round(time.time() - alloc)
                    self.colab('download', '-s', name, '/content/chain-boot-state.json', str(self.dir / f'chain-boot-{name}.json'))
                    self.log(f'{name}: launched after {vm["boot_seconds"]} s')
                if launched and not post_done and self.L.get('post_launch'):
                    script, remote = self.L['post_launch']
                    rc, text = self.colab('exec', '-s', name, '-f', script, '--timeout', '1500', timeout=1800)
                    self.colab('download', '-s', name, '/content/' + remote, str(self.dir / remote))
                    self.log(f'{name}: post-launch: {text.strip()[-300:]}'); post_done = True
                if not launched and time.time() - alloc > 5400:
                    self.log(f'{name}: boot not launched after 90 min'); self.save(); self.colab('stop', '-s', name); return 2
                self.save()
                phases = status['phases']
                if phases and all(phases.get(a) in TERMINAL for a in self.L['arms']):
                    self.log(f'{name}: terminal phases {phases}')
                    self.fetch(name); vm['terminal_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); self.save()
                    self.colab('stop', '-s', name); self.log(f'{name}: stopped')
                    return 0 if all(phases[a] == 'exported' for a in self.L['arms']) else 4
            lost_t = time.time(); vm['lost_detected_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(lost_t))
            vm['alloc_to_loss_detected_s'] = round(lost_t - alloc)
            rc, hist = self.colab('log', '-s', name, '-n', '400')
            vm['keepalive_events'] = [l[:200] for l in hist.splitlines() if l.startswith('[20') and ('KEEP: error' in l or 'KEEP: stopped' in l or 'terminated' in l)][:6]
            vm['short_lived'] = lost_t - alloc < self.L['min_life_s']; self.save()
            self.log(f'{name}: VM lost after {vm["alloc_to_loss_detected_s"]} s (detected); keepalive: {vm["keepalive_events"][:2]}')
            self.colab('sessions')   # prunes the stale local entry
            short += vm['short_lived']
            if short >= self.L['short_death_limit']:
                self.log(f'{short} VMs died in under {self.L["min_life_s"] // 60} min; stopping the chain'); return 3
        self.log('VM cap reached'); return 5


if __name__ == '__main__':
    lane = json.loads(pathlib.Path(sys.argv[1]).read_bytes())
    sys.exit(Driver(lane).run(int(sys.argv[2])))
