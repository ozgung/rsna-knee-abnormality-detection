"""VM: restore one arm's exact resume file from its private backup dataset version into a FRESH workdir.

Usage: python vm_raptor64_restore_resume.py ARM VERSION RESUME_SHA256
Downloads payload_manifest.json and raptor_resume_fold0.pt of exactly OWNER/SLUG/VERSION, requires the
manifest's dataset/fold/epoch 11/resume SHA to match, verifies the file SHA and size, then places it at
/content/raptor64-ARM-fold0/raptor_resume_fold0.pt (the workdir must not exist yet). Nothing else is written there.
"""
import hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

arm, version, expected = sys.argv[1], int(sys.argv[2]), sys.argv[3]
assert arm in ('in-orig', 'in-plane', 'dr-orig') and len(expected) == 64
dataset = f'ozgung/knee-raptor64-{arm}-fold0-20261013'
work = Path(f'/content/raptor64-{arm}-fold0')
assert not work.exists(), 'Fresh workdir required for restore'
env = {**os.environ, 'KAGGLE_CONFIG_DIR': '/content'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


with tempfile.TemporaryDirectory(dir='/content', prefix=f'raptor64-{arm}-restore-') as d:
    for name in ('payload_manifest.json', 'raptor_resume_fold0.pt'):
        r = subprocess.run(['kaggle', 'datasets', 'download', f'{dataset}/{version}', '-f', name, '-p', d, '--unzip', '-o', '-q'],
                           capture_output=True, text=True, env=env, timeout=3600)
        if r.returncode:
            raise SystemExit(f'download of {name} failed rc={r.returncode}')
        for z in Path(d).glob('*.zip'):
            shutil.unpack_archive(str(z), d); z.unlink()
    m = json.loads((Path(d) / 'payload_manifest.json').read_bytes())
    f = Path(d) / 'raptor_resume_fold0.pt'
    if (m.get('dataset') != dataset or m.get('fold') != 0 or m.get('epoch') != 11 or m.get('resume_sha256') != expected or
            m['files']['raptor_resume_fold0.pt']['size'] != f.stat().st_size):
        raise SystemExit('Backup manifest differs from the pinned final epoch-11 resume')
    actual = sha(f)
    if actual != expected:
        raise SystemExit('Restored resume SHA differs')
    work.mkdir(mode=0o700)
    shutil.move(str(f), work / 'raptor_resume_fold0.pt')
print(json.dumps({'arm': arm, 'dataset': dataset, 'version': version, 'resume_sha256': actual,
                  'bytes': (work / 'raptor_resume_fold0.pt').stat().st_size, 'epoch': 11, 'placed': str(work / 'raptor_resume_fold0.pt')}))
