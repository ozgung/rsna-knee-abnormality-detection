"""Local independent per-file verification of a private raptor64 final export (when the bulk archive is not built yet).

Usage: python verify_raptor64_export_per_file.py DATASET DEST RECEIPT_JSON
Downloads MANIFEST.json and each listed file individually (`kaggle datasets download -f`), checks every
sha256/size against MANIFEST.json, checks dataset privacy, and writes a receipt. Never modifies the dataset.
"""
import hashlib, json, subprocess, sys, tempfile, zipfile
from pathlib import Path

KAGGLE = '~/miniconda3/bin/kaggle'
dataset, dest, receipt_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
assert not dest.exists() and not receipt_path.exists(), 'fresh paths required'
dest.mkdir(parents=True)


def fetch(name):
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([KAGGLE, 'datasets', 'download', dataset, '-f', name, '-p', d, '-q'], capture_output=True, text=True, timeout=3600)
        if r.returncode:
            raise SystemExit(f'download {name} failed')
        files = list(Path(d).iterdir())
        assert len(files) == 1, files
        f = files[0]
        if f.suffix == '.zip' and f.name == name + '.zip':
            with zipfile.ZipFile(f) as z:
                assert [i.filename for i in z.infolist()] == [name]; z.extract(name, d)
            f = Path(d) / name
        assert f.name == name
        target = dest / name; f.replace(target)
        return target


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


manifest = json.loads(fetch('MANIFEST.json').read_bytes())
entries = manifest['files']
rows = entries.items() if isinstance(entries, dict) else [(e['name'], e) for e in entries]
results = {}
for name, e in rows:
    p = fetch(name)
    results[name] = {'sha256_match': sha(p) == e['sha256'], 'size_match': p.stat().st_size == e.get('size', e.get('bytes')), 'sha256': e['sha256']}
meta = subprocess.run([KAGGLE, 'datasets', 'metadata', dataset, '-p', str(dest)], capture_output=True, text=True)
md = json.loads((dest / 'dataset-metadata.json').read_bytes()) if (dest / 'dataset-metadata.json').exists() else {}
info = md.get('info', md)
private = info.get('isPrivate')
ok = all(r['sha256_match'] and r['size_match'] for r in results.values())
receipt = {'dataset': dataset, 'method': 'independent per-file download (bulk archive not yet built at exporter verify time)',
           'manifest_sha256': sha(dest / 'MANIFEST.json'), 'files': results, 'all_match': ok, 'is_private': private,
           'state': 'verified' if ok and private is True else 'failed'}
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'state': receipt['state'], 'files': len(results), 'all_match': ok, 'is_private': private}))
