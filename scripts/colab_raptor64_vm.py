"""Raptor64 Colab VM orchestration (runs ON the VM). Subcommands:

  setup                  pip-install pinned packages, check credential, prepare both initializers
  inputs ARM             stage pinned small supervision inputs for ARM's package (per-arm input root)
  stage                  stage + verify the four private cache shards (/content/raptor64_cache)
  bootstrap ARM          create (if absent) and verify the arm's PRIVATE resume-backup dataset
  config ARM             write /content/raptor64-ARM-run-config.json and validate it with the runner
  launch ARM --sha SHA   start the arm controller detached (one per arm; separate lock/work/backup)
  health                 aggregate sanitized progress for all arms to /content/raptor64-health.json

Layout (all under /content): packages raptor64-ARM-pkg, work raptor64-ARM-fold0, inputs
input-raptor64-ARM, initializers raptor64-init-{imagenet,dread_v4}, cache raptor64_cache,
backup dataset ozgung/knee-raptor64-ARM-fold0-20261013. Credential: /content/kaggle.json (mode 600).
Nothing here starts training except `launch`, which requires a reviewed config SHA.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/content')
ARMS = ('in-orig', 'in-plane', 'dr-orig', 'dr-plane', 'dr-full', 'in-mirror', 'in-distil', 'in-distil-w1', 'in-distil-flip', 'ax-crop', 'ax-full')
INIT = {'in-orig': 'imagenet', 'in-plane': 'imagenet', 'dr-orig': 'dread_v4', 'dr-plane': 'dread_v4', 'dr-full': 'dread_v4', 'in-mirror': 'imagenet', 'in-distil': 'imagenet', 'in-distil-w1': 'imagenet', 'in-distil-flip': 'imagenet', 'ax-crop': 'imagenet', 'ax-full': 'imagenet'}
DATE = '20261013'
PACKAGES = ['timm==1.0.26', 'kaggle==2.2.4', 'safetensors', 'pydicom', 'opencv-python-headless', 'pyarrow']
PERSISTER_SHA = '8923bd872500b87f85be78aefd9fc799a6f88942e94503045b5e01ea66c118df'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def arm_check(arm):
    if arm not in ARMS:
        raise SystemExit('Unknown arm; use one of ' + ', '.join(ARMS))
    return arm


def paths(arm, root=ROOT):
    t = f'raptor64-{arm}'
    return {'package': root / f'{t}-pkg', 'work': root / f'{t}-fold0', 'input': root / f'input-{t}',
            'init': root / f'raptor64-init-{INIT[arm]}', 'config': root / f'{t}-run-config.json',
            'lock': root / f'{t}-controller.lock', 'staging': root / f'{t}-backup-staging', 'status': root / f'{t}-persist-status.json',
            'log': root / f'{t}-controller-private.log', 'launch': root / f'{t}-launch.json',
            'dataset': f'ozgung/knee-{t}-fold0-{DATE}'}


def env():
    e = dict(os.environ); e['KAGGLE_CONFIG_DIR'] = str(ROOT)
    return e


def run(cmd, log, **kw):
    with open(log, 'ab') as stream:
        subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True, env=env(), **kw)


def setup(args):
    os.umask(0o077)
    cred = ROOT / 'kaggle.json'
    if not cred.is_file() or cred.is_symlink():
        raise SystemExit('Upload the Kaggle credential to /content/kaggle.json first')
    os.chmod(cred, 0o600)
    run([sys.executable, '-m', 'pip', 'install', *PACKAGES], ROOT / 'raptor64-setup-private.log')
    import torch, importlib.metadata as md
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    info = {'device': torch.cuda.get_device_name(0), 'vram_bytes': torch.cuda.get_device_properties(0).total_memory,
            'cpu_count': os.cpu_count(), 'disk_free': __import__('shutil').disk_usage(ROOT).free,
            'packages': {k: md.version(k) for k in ('torch', 'timm', 'numpy', 'safetensors', 'kaggle', 'pydicom', 'pyarrow')}}
    if info['packages']['timm'] != '1.0.26':
        raise SystemExit('timm 1.0.26 required')
    results = {}
    for init in ('imagenet', 'dread_v4'):
        target = ROOT / f'raptor64-init-{init}'
        if (target / 'initializer.json').exists():
            results[init] = {'descriptor_sha256': sha(target / 'initializer.json'), 'reused': True}
            continue
        run([sys.executable, str(ROOT / 'raptor64-tools/scripts/prepare_raptor64_initializer.py'), str(target), '--init', init],
            ROOT / f'raptor64-init-{init}-private.log', cwd=str(ROOT / 'raptor64-tools'))
        results[init] = {'descriptor_sha256': sha(target / 'initializer.json'), 'reused': False}
    info['initializers'] = results
    (ROOT / 'raptor64-runtime-setup.json').write_text(json.dumps(info, indent=2) + '\n'); print(json.dumps(info))


def inputs(args):
    p = paths(arm_check(args.arm))
    manifest = sha(p['package'] / 'manifest.json')
    run([sys.executable, str(ROOT / 'raptor64-tools/scripts/colab_source96_inputs.py'), '--input-root', str(p['input']),
         '--package', str(p['package']), '--manifest-sha256', manifest], ROOT / f'raptor64-{args.arm}-inputs-private.log')
    print(json.dumps({'arm': args.arm, 'inputs': 'PASS', 'receipt_sha256': sha(p['input'] / 'source96-inputs-receipt.json')}))


def stage(args):
    config = ROOT / 'raptor64-stage-config-private.json'
    if sha(config) != args.config_sha:
        raise SystemExit('Reviewed stage config SHA differs')
    e = env(); e['RAPTOR64_CODE_ROOT'] = str(ROOT / 'raptor64-tools'); e['PYTHONPATH'] = str(ROOT / 'raptor64-tools')
    with open(ROOT / 'raptor64-stage-private.log', 'ab') as log:
        subprocess.run([sys.executable, '-u', str(ROOT / 'raptor64-tools/scripts/colab_raptor64_prepare.py'), '--config', str(config),
                        '--download-workers', str(args.workers)], stdout=log, stderr=subprocess.STDOUT, check=True, env=e)
    r = json.loads((ROOT / 'raptor64_cache/staging-receipt.json').read_bytes())
    print(json.dumps({k: r.get(k) for k in ('status', 'studies', 'manifest_sha256', 'content_sha256', 'elapsed_seconds')}))


def bootstrap(args):
    p = paths(arm_check(args.arm)); ds = p['dataset']; d = ROOT / f'raptor64-{args.arm}-bootstrap'
    os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    status = subprocess.run(['kaggle', 'datasets', 'status', ds], capture_output=True, text=True, env=env())
    if status.returncode != 0:
        d.mkdir(exist_ok=False)
        (d / 'bootstrap.json').write_text(json.dumps({'kind': 'bootstrap-only', 'training_started': False, 'dataset': ds}) + '\n')
        (d / 'dataset-metadata.json').write_text(json.dumps({'title': ds.split('/')[1].replace('-', ' ')[:50], 'id': ds, 'licenses': [{'name': 'CC0-1.0'}]}) + '\n')
        created = subprocess.run(['kaggle', 'datasets', 'create', '-p', str(d), '-q'], capture_output=True, text=True, env=env())
        if created.returncode != 0:
            raise SystemExit('Private dataset create failed; see Kaggle CLI output on the VM')
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        m = json.loads(Path(api.dataset_metadata(ds, t)).read_bytes())
    info = m.get('info', m)
    identity = (str(info.get('ownerUser')) + '/' + str(info.get('datasetSlug'))) if 'info' in m else info.get('id')
    if identity != ds or info.get('isPrivate') is not True:
        raise SystemExit('Backup dataset identity/privacy unverified')
    print(json.dumps({'arm': args.arm, 'dataset': ds, 'private': True}))


def make_config(arm, root=ROOT, *, vm_session_id):
    p = paths(arm, root); stage_receipt = root / 'raptor64_cache/staging-receipt.json'; r = json.loads(stage_receipt.read_bytes())
    if r.get('status') != 'PASS' or r.get('studies') != 4345:
        raise ValueError('Verified 4345-study staging required')
    return {'version': 'colab-raptor64-run-v1', 'name': f'raptor64-{arm}-fold0-v1', 'arm': arm, 'vm_session_id': vm_session_id,
            'package_root': str(p['package']), 'package_manifest_sha256': sha(p['package'] / 'manifest.json'), 'workdir': str(p['work']),
            'input_root': str(p['input']), 'stage_root': str(root / 'raptor64_cache'), 'manifest_sha256': r['manifest_sha256'],
            'content_sha256': r['content_sha256'], 'stage_receipt_sha256': sha(stage_receipt),
            'initializer_descriptor': str(p['init'] / 'initializer.json'), 'initializer_descriptor_sha256': sha(p['init'] / 'initializer.json'),
            'persister': str(root / 'colab_tracka_persist.py'), 'persister_sha256': PERSISTER_SHA,
            'persistence': {'dataset': p['dataset'], 'private': True, 'staging': str(p['staging']), 'status_file': str(p['status'])},
            'controller_lock': str(p['lock']), 'workers': 2, 'worker_choice_name': None, 'resume': False, 'fold': 0}


def load_runner():
    spec = importlib.util.spec_from_file_location('raptor64_runner', ROOT / 'raptor64-tools/scripts/colab_raptor64_run.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def config(args):
    p = paths(arm_check(args.arm))
    if p['config'].exists():
        raise SystemExit('Existing config preserved; reconcile instead of overwriting')
    c = make_config(args.arm, vm_session_id=args.session)
    runner = load_runner(); os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT)
    runner.validate(c); runner.private_dataset(c['persistence']['dataset'])
    p['config'].write_text(json.dumps(c, indent=2) + '\n')
    print(json.dumps({'arm': args.arm, 'config': str(p['config']), 'config_sha256': sha(p['config']), 'training_started': False}))


def launch(args):
    p = paths(arm_check(args.arm)); controller = ROOT / 'raptor64-tools/scripts/colab_raptor64_run.py'
    if sha(p['config']) != args.sha:
        raise SystemExit('Reviewed config SHA differs')
    for key in ('launch', 'work', 'lock'):
        if p[key].exists():
            raise SystemExit(f'Existing {key} artifact; reconcile rather than relaunch')
    cmd = [sys.executable, '-u', str(controller), str(p['config']), '--sha256', args.sha]
    e = env(); e['PYTHONPATH'] = str(p['package'])
    with p['log'].open('xb') as stream:
        proc = subprocess.Popen(cmd, env=e, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'arm': args.arm, 'pid': proc.pid, 'command': cmd,
               'config_sha256': args.sha, 'controller_sha256': sha(controller), 'log': str(p['log']), 'phase': 'controller_started_not_yet_training_verified'}
    with p['launch'].open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


SAFE = re.compile(r'\[(training|backup)\] (ep\d+ loss [0-9.eE+-]+ \| (?:OOF masked macro-AUC [0-9.eE+-]+|FULL-DATA fixed-epoch score [01]\.0)|\[resume\][^\n]*|FINAL PERSIST[^\n]*|\[raptor64\] init[^\n]*)')


def health(args):
    out = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'arms': {}}
    try:
        import torch
        out['gpu'] = {'allocated_note': 'per-process memory in nvidia-smi', 'device': torch.cuda.get_device_name(0)}
    except Exception:
        pass
    smi = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True)
    out['nvidia_smi'] = smi.stdout.strip()
    for arm in ARMS:
        p = paths(arm)
        if not p['launch'].exists():
            continue
        launch = json.loads(p['launch'].read_bytes()); alive = Path(f'/proc/{launch["pid"]}').exists()
        lines = SAFE.findall(p['log'].read_text(errors='replace')) if p['log'].exists() else []
        status = json.loads(p['status'].read_bytes()) if p['status'].exists() else {}
        receipt = p['work'] / 'raptor64-run-receipt.json'
        out['arms'][arm] = {'controller_alive': alive, 'progress': [m[1] for m in lines][-30:],
                            'backup': {k: status.get(k) for k in ('verified_epoch', 'verified_version')},
                            'receipt_status': json.loads(receipt.read_bytes()).get('status') if receipt.exists() else None}
    (ROOT / 'raptor64-health.json').write_text(json.dumps(out, indent=2) + '\n'); print(json.dumps(out, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('setup')
    for name in ('inputs', 'bootstrap'):
        s = sub.add_parser(name); s.add_argument('arm')
    s = sub.add_parser('stage'); s.add_argument('--config-sha', required=True); s.add_argument('--workers', type=int, default=8)
    s = sub.add_parser('config'); s.add_argument('arm'); s.add_argument('--session', required=True)
    s = sub.add_parser('launch'); s.add_argument('arm'); s.add_argument('--sha', required=True)
    sub.add_parser('health')
    a = p.parse_args(argv)
    {'setup': setup, 'inputs': inputs, 'stage': stage, 'bootstrap': bootstrap, 'config': config, 'launch': launch, 'health': health}[a.cmd](a)


if __name__ == '__main__':
    main()
