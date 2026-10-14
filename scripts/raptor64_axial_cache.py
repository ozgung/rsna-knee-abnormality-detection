"""Iteration-4 derived axial-only Raptor64 cache (hash-pinned) for short-lived Colab VMs.

The ax-crop / ax-full arms read only slots 52-63 (images and masks) of each 64-slot study. The derived cache keeps
every study file in the unchanged loader format (64 x 336 x 336 uint8 volume + 64 mask) but zeroes slots 0-51 and
their mask entries, so the files compress to a fraction of the original and stage in minutes. Axial bytes are
copied unchanged, so the arms see byte-identical inputs (tests/test_raptor64_axial_cache.py; every study is also
re-read and compared in `derive`). The derived cache has its own content identity (new file/volume/mask hashes);
it is never presented as the full cache.

  derive  (Kaggle CPU kernel, no internet): read the four accepted shard outputs from /kaggle/input, verify their
          pinned metadata and per-file hashes, write axial_shardN.tar (run-contract.json verbatim, a re-hashed
          index-private.json, a derived receipt.json, NNNNNN.npz) plus axial-derived-pins.json to /kaggle/working.
  stage   (Colab VM): verify the downloaded tars against pinned SHA-256s, extract to /content/raptor64_cache/sourceN,
          assemble the standard private manifest with the unchanged loader, and write staging-receipt.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time

AXIAL = (52, 64)
DEPTH, STORAGE = 64, 336
VERSION = 'raptor64-v4-336-sharded-v1'
DERIVATION = 'raptor64-axial-only-v1: slots 0-51 and their mask entries zeroed; slots 52-63 copied byte-for-byte'


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def derive_arrays(volume, mask):
    """Axial-only copy in the unchanged (64, 336, 336) / (64,) uint8 format."""
    import numpy as np
    if volume.shape != (DEPTH, STORAGE, STORAGE) or volume.dtype != np.uint8 or mask.shape != (DEPTH,) or mask.dtype != np.uint8:
        raise ValueError('Source geometry differs')
    a, b = AXIAL
    v = np.zeros_like(volume); m = np.zeros_like(mask)
    v[a:b] = volume[a:b]; m[a:b] = mask[a:b]
    if not m.any() or not np.array_equal(m.astype(bool), v.reshape(DEPTH, -1).any(axis=1)):
        raise ValueError('Derived axial block is empty or inconsistent')
    return v, m


def save_npz(path, volume, mask):
    import numpy as np
    with open(path, 'wb') as stream:
        np.savez_compressed(stream, volume=volume, mask=mask)


def _derive_one(args):
    import numpy as np
    src, dst, entry = args
    if sha_file(src) != entry['file_sha256']:
        raise ValueError('Source file hash differs: ' + entry['file'])
    with np.load(src, allow_pickle=False) as z:
        volume, mask = z['volume'], z['mask']
    v, m = derive_arrays(volume, mask)
    save_npz(dst, v, m)
    with np.load(dst, allow_pickle=False) as z:
        rv, rm = z['volume'], z['mask']
    a, b = AXIAL
    if not (np.array_equal(rv, v) and np.array_equal(rm, m) and rv[a:b].tobytes() == volume[a:b].tobytes() and rm[a:b].tobytes() == mask[a:b].tobytes()):
        raise ValueError('Derived round trip differs')
    out = dict(entry)
    out.update(file_sha256=sha_file(dst), volume_sha256=sha_bytes(v.tobytes()), mask_sha256=sha_bytes(m.tobytes()), bytes=os.path.getsize(dst),
               derived_from={'file_sha256': entry['file_sha256'], 'volume_sha256': entry['volume_sha256'], 'mask_sha256': entry['mask_sha256']})
    return out


def derive(input_root, out_root, pins, workers=4):
    from multiprocessing import Pool
    out_root = Path(out_root); started = time.time()
    shard_roots = {}
    for contract_path in Path(input_root).rglob('run-contract.json'):
        root = contract_path.parent
        shard = json.loads(contract_path.read_bytes())['shard_index']
        if shard in shard_roots:
            raise ValueError('Duplicate shard input')
        shard_roots[shard] = root
    if sorted(shard_roots) != [0, 1, 2, 3]:
        raise ValueError(f'Four accepted shards required, found {sorted(shard_roots)}')
    result = {'derivation': DERIVATION, 'shards': {}, 'studies': 0}
    for shard in range(4):
        root = shard_roots[shard]; pin = pins[str(shard)]
        for name, key in (('run-contract.json', 'contract_file_sha256'), ('index-private.json', 'index_sha256'), ('receipt.json', 'receipt_sha256')):
            if sha_file(root / name) != pin[key]:
                raise ValueError(f'shard {shard} {name} differs from the accepted pin')
        index = json.loads((root / 'index-private.json').read_bytes()); receipt = json.loads((root / 'receipt.json').read_bytes())
        if receipt.get('status') != 'PASS':
            raise ValueError('Accepted shard receipt is not PASS')
        dst = out_root / f'source{shard}'; dst.mkdir(parents=True)
        jobs = [(root / e['file'], dst / e['file'], e) for e in index['entries']]
        if workers > 1:
            with Pool(workers) as pool:
                entries = pool.map(_derive_one, jobs, chunksize=8)
        else:
            entries = [_derive_one(j) for j in jobs]
        (dst / 'run-contract.json').write_bytes((root / 'run-contract.json').read_bytes())
        new_index = {**{k: v for k, v in index.items() if k != 'entries'}, 'entries': entries, 'derivation': DERIVATION,
                     'parent_index_sha256': pin['index_sha256']}
        (dst / 'index-private.json').write_text(json.dumps(new_index, indent=1) + '\n')
        new_receipt = {'status': 'PASS', 'version': VERSION, 'contract_sha256': receipt['contract_sha256'],
                       'private_index_sha256': sha_file(dst / 'index-private.json'), 'completed_studies': len(entries),
                       'shard_index': shard, 'shard_count': 4, 'derivation': DERIVATION,
                       'parent': {'receipt_sha256': pin['receipt_sha256'], 'index_sha256': pin['index_sha256'], 'kernel': pin.get('ref')}}
        (dst / 'receipt.json').write_text(json.dumps(new_receipt, indent=1) + '\n')
        tar_path = out_root / f'axial_shard{shard}.tar'
        with tarfile.open(tar_path, 'w') as t:
            for p in sorted(dst.iterdir()):
                t.add(p, arcname=f'source{shard}/{p.name}')
        result['shards'][str(shard)] = {'tar': tar_path.name, 'tar_sha256': sha_file(tar_path), 'tar_bytes': tar_path.stat().st_size,
                                        'contract_file_sha256': sha_file(dst / 'run-contract.json'),
                                        'index_sha256': sha_file(dst / 'index-private.json'), 'receipt_sha256': sha_file(dst / 'receipt.json'),
                                        'studies': len(entries), 'npz_bytes': sum(e['bytes'] for e in entries)}
        result['studies'] += len(entries)
        for p in dst.iterdir():
            p.unlink()
        dst.rmdir()
    result['seconds'] = round(time.time() - started, 1)
    (out_root / 'axial-derived-pins.json').write_text(json.dumps(result, indent=1) + '\n')
    return result


def stage(download_dir, cache_root, pins_path, pins_sha256, stage_config):
    """Colab: verify pinned tars, extract, assemble with the unchanged loader, write the staging receipt."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from knee import raptor64_loader as loader, raptor64_cache as cache
    started = time.time(); download_dir, cache_root = Path(download_dir), Path(cache_root)
    if sha_file(pins_path) != pins_sha256:
        raise ValueError('Derived-cache pins differ')
    pins = json.loads(Path(pins_path).read_bytes())
    if cache_root.exists() and any(cache_root.iterdir()):
        raise ValueError('Fresh cache root required')
    cache_root.mkdir(parents=True, exist_ok=True)
    descriptors = []
    for shard in range(4):
        pin = pins['shards'][str(shard)]; tar_path = download_dir / pin['tar']
        if sha_file(tar_path) != pin['tar_sha256']:
            raise ValueError(f'axial shard {shard} tar differs from the pin')
        with tarfile.open(tar_path) as t:
            names = t.getnames()
            if any(not n.startswith(f'source{shard}/') or '..' in n for n in names):
                raise ValueError('Unsafe tar member')
            t.extractall(cache_root, filter='data')
        root = cache_root / f'source{shard}'
        descriptors.append({'root': str(root), 'contract_file_sha256': pin['contract_file_sha256'], 'index_sha256': pin['index_sha256'],
                            'receipt_sha256': pin['receipt_sha256']})
    validation = json.loads(Path(stage_config).read_bytes())['validation']
    manifest_path = cache_root / 'cache64-manifest-private.json'
    manifest = loader.assemble(descriptors, validation, manifest_path=manifest_path)
    receipt = {'version': 'colab-raptor64-axial-derived-stage-v1', 'status': 'PASS', 'studies': len(manifest['entries']),
               'manifest_sha256': manifest['manifest_sha256'], 'content_sha256': manifest['content_sha256'],
               'manifest_file_sha256': sha_file(manifest_path), 'derivation': DERIVATION, 'pins_sha256': pins_sha256,
               'elapsed_seconds': round(time.time() - started, 1), 'aggregate_contains_ids': False}
    cache.atomic_json(cache_root / 'staging-receipt.json', receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('derive'); d.add_argument('--input-root', default='/kaggle/input'); d.add_argument('--out', default='/kaggle/working')
    d.add_argument('--pins-json', required=True, help='JSON {shard: {contract_file_sha256, index_sha256, receipt_sha256, ref}}'); d.add_argument('--workers', type=int, default=4)
    s = sub.add_parser('stage'); s.add_argument('--download-dir', required=True); s.add_argument('--cache-root', default='/content/raptor64_cache')
    s.add_argument('--pins', required=True); s.add_argument('--pins-sha256', required=True); s.add_argument('--stage-config', required=True)
    a = p.parse_args(argv)
    if a.cmd == 'derive':
        print(json.dumps(derive(a.input_root, a.out, json.loads(a.pins_json), a.workers), indent=1))
    else:
        print(json.dumps(stage(a.download_dir, a.cache_root, a.pins, a.pins_sha256, a.stage_config)))


if __name__ == '__main__':
    main()
