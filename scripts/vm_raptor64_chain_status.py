"""One-line sanitized status of a chained Raptor64 VM (runs ON the VM): boot state, autopilot phases, per-arm epoch
progress and verified backups, last diagnostics sample. Prints `STATUS {json}`; no study IDs or labels."""
import json, pathlib, re

root = pathlib.Path('/content'); P = json.loads((root / 'chain-params.json').read_bytes()); out = {}
b = root / 'chain-boot-state.json'
boot = json.loads(b.read_bytes()) if b.exists() else {}
out['boot'] = {'status': boot.get('status'), 'last_step': max(boot.get('steps', {}).items(), key=lambda kv: kv[1]['t'])[0] if boot.get('steps') else None,
               'error': (boot.get('steps', {}).get('failed') or {}).get('error')}
a = root / 'raptor64-autopilot.json'
out['phases'] = {k: v.get('phase') for k, v in json.loads(a.read_bytes()).get('arms', {}).items()} if a.exists() else {}
out['arms'] = {}
for arm in P['arms']:
    log = root / f'raptor64-{arm}-controller-private.log'
    lines = re.findall(r'\[training\] (ep\d+ loss [0-9.]+ \| OOF masked macro-AUC [0-9.]+)', log.read_text(errors='replace')) if log.exists() else []
    st = root / f'raptor64-{arm}-persist-status.json'
    s = json.loads(st.read_bytes()) if st.exists() else {}
    tl = root / f'raptor64-{arm}-fold0/raptor64-training.log'
    tail = [l[:200] for l in tl.read_text(errors='replace').splitlines() if re.search(r'Traceback|Error|resum', l)][-2:] if tl.exists() else []
    out['arms'][arm] = {'epochs': lines[-3:], 'n_epochs': len(lines), 'verified_epoch': s.get('verified_epoch'), 'verified_version': s.get('verified_version'), 'alerts': tail}
d = root / P['diag_name']
text = d.read_text() if d.exists() else ''
out['diag'] = json.loads(text.splitlines()[-1]) if text.strip() else None
print('STATUS ' + json.dumps(out))
