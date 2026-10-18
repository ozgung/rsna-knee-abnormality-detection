"""VM-only private Raptor64 staging; no model, training or local image relay.

Config: version, cache_root, validation (loader's private membership/folds), and four sources:
ref, version=1, notebook_path, notebook_sha256, contract_file_sha256, index_sha256, receipt_sha256.
Same transport/binding rules as the qualified source96 stager (whose generic helpers are reused
unchanged); only the cache module, artifact prefix and manifest name differ.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

CODE_ROOT = Path(os.environ.get('RAPTOR64_CODE_ROOT', '/content/raptor64_stage_code'))
if CODE_ROOT.is_dir():
    sys.path.insert(0, str(CODE_ROOT))
from knee import raptor64_cache as cache, raptor64_loader as loader, public_training_smoke as smoke

_spec = importlib.util.spec_from_file_location('colab_source96_prepare_helpers', Path(__file__).resolve().parent / 'colab_source96_prepare.py')
s96 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s96)
KaggleTransport, checked_url, _binding, _snapshot, _download = s96.KaggleTransport, s96.checked_url, s96._binding, s96._snapshot, s96._download

VERSION = 'colab-raptor64-stage-v1'
PREFIX = 'raptor64_cache/'
METADATA_MAX_BYTES = 16 * 1024 * 1024
DISK_RESERVE = 5 * 1024**3
METADATA = {'run-contract.json': 'contract_file_sha256', 'index-private.json': 'index_sha256', 'receipt.json': 'receipt_sha256'}
MANIFEST = 'cache64-manifest-private.json'


def checked_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'raptor64_cache/(?:run-contract\.json|index-private\.json|receipt\.json|[0-9]{6}\.npz)', name):
        raise ValueError('Unsafe or unrequested artifact path')
    return name.split('/')[-1]


def _inventory(transport, binding):
    _snapshot(transport, binding); items = transport.list_files(binding['ref']); _snapshot(transport, binding)
    selected = {}
    for item in items:
        name = item['name']
        if not name.startswith(PREFIX):
            continue
        if name.endswith('.npz') or name.split('/')[-1] in METADATA:
            checked_name(name)
            if name in selected:
                raise ValueError('Duplicate artifact listing')
            selected[name] = checked_url(item['url'])
    return selected


def _metadata_plan(plans, validation):
    _, counts = loader._validate_profile(validation); common = None; seen = set(); rows = {}; files = []
    for plan in plans:
        root = plan['root']; contract = json.loads((root / 'run-contract.json').read_bytes())
        index = json.loads((root / 'index-private.json').read_bytes()); receipt = json.loads((root / 'receipt.json').read_bytes())
        n = contract['shard_index']; contract_sha = loader.digest_document(contract)
        if type(n) is not int or not 0 <= n < 4 or n in seen or contract['shard_count'] != 4:
            raise ValueError('Shard identity differs')
        seen.add(n)
        if contract['version'] != cache.VERSION or contract['config'] != cache.CONFIG or contract['membership'] != validation['membership']:
            raise ValueError('Cache policy/population differs')
        current = {k: contract[k] for k in loader.COMMON}
        if common is None:
            common = current
        elif current != common:
            raise ValueError('Cache source/pins differ across shards')
        members = list(range(n, counts[0], 4))
        if (len(contract['partitions']) != 4 or contract['partitions'][n]['ordinals_sha256'] != loader.digest_document(members) or
                contract['partitions'][n]['studies'] != len(members)):
            raise ValueError('Partition differs')
        if (receipt.get('status') != 'PASS' or receipt.get('version') != cache.VERSION or receipt.get('contract_sha256') != contract_sha or
                receipt.get('private_index_sha256') != plan['binding']['index_sha256'] or receipt.get('completed_studies') != len(members) or
                receipt.get('shard_index') != n or receipt.get('shard_count') != 4 or index.get('version') != cache.VERSION or
                index.get('contract_sha256') != contract_sha):
            raise ValueError('Cache metadata not fully qualified')
        entries = index['entries']
        if [e['ordinal'] for e in entries] != members:
            raise ValueError('Index partition incomplete/reordered')
        ids = []
        for row in entries:
            ordinal = row['ordinal']; uid = cache.safe_id(row['StudyInstanceUID']); name = cache.filename(ordinal)
            if (row['file'] != name or type(row['bytes']) is not int or not 0 < row['bytes'] <= cache.MAX_FILE_BYTES or
                    not all(loader._hash(row[k]) for k in ('file_sha256', 'volume_sha256', 'mask_sha256'))):
                raise ValueError('Private file index invalid')
            loader.validate_slots(row.get('slots'))
            if ordinal in rows:
                raise ValueError('Overlapping ordinals')
            rows[ordinal] = uid; ids.append(uid); remote = PREFIX + name
            if remote not in plan['inventory']:
                raise ValueError('Expected cache artifact absent')
            files.append({'path': cache.safe_file(root, name), 'url': plan['inventory'][remote], 'row': row, 'binding': plan['binding']})
        if contract['partitions'][n]['ordered_ids_sha256'] != loader.digest_document(ids):
            raise ValueError('Shard ID order differs')
        if {name for name in plan['inventory'] if name.endswith('.npz')} != {PREFIX + cache.filename(i) for i in members}:
            raise ValueError('Extra or missing output NPZ')
    ids = [rows[i] for i in range(counts[0])]
    if len(set(ids)) != len(ids) or loader.digest_document(ids) != validation['membership']['ordered_eligible_sha256']:
        raise ValueError('Global eligible order differs')
    folds = validation['folds']
    if any(uid not in folds or type(folds[uid]) is not int or not 0 <= folds[uid] <= 4 for uid in ids):
        raise ValueError('Eligible fold metadata invalid')
    train = [u for u in ids if folds[u] != 0]; dev = [u for u in ids if folds[u] == 0]
    if ((len(train), len(dev)) != counts[1:] or loader.digest_document(train) != validation['membership']['ordered_training_sha256'] or
            loader.digest_document(dev) != validation['membership']['ordered_development_sha256']):
        raise ValueError('Historical train/development order differs')
    return files


def _transfer(items, transport, workers, complete):
    largest = max((item['row']['bytes'] for item in items), default=0)

    def download(item):
        if shutil.disk_usage(item['path'].parent).free < 2 * workers * largest + DISK_RESERVE:
            raise OSError('Disk changed during staging')
        row = item['row']
        return _download(transport, item['url'], item['path'], row['file_sha256'], expected_bytes=row['bytes'], max_bytes=cache.MAX_FILE_BYTES)
    if workers == 1:
        for item in items:
            complete(download(item))
        return
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    remaining = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()

        def fill():
            while len(pending) < workers:
                item = next(remaining, None)
                if item is None:
                    break
                pending.add(pool.submit(download, item))
        fill()
        try:
            while pending:
                done, pending_rest = wait(pending, return_when=FIRST_COMPLETED)
                pending.clear(); pending.update(pending_rest)
                for result in [future.result() for future in done]:
                    complete(result)
                fill()
        except BaseException:
            for future in pending:
                future.cancel()
            raise


def prepare(config, transport, *, allow_fixture=False, download_workers=1):
    if type(download_workers) is not int or not 1 <= download_workers <= 8:
        raise ValueError('Download workers must be1..8')
    root = Path(config['cache_root'])
    if config.get('version') != VERSION or root.is_symlink() or len(config['sources']) != 4:
        raise ValueError('Staging configuration differs')
    if config['validation'].get('fixture') and not allow_fixture:
        raise ValueError('Fixture staging not allowed in production')
    bindings = [_binding(d) for d in config['sources']]
    if len({b['ref'] for b in bindings}) != 4 or any('raptor64' not in b['ref'] for b in bindings):
        raise ValueError('Four unique Raptor64 source slugs required')
    root.mkdir(parents=True, exist_ok=True)
    authority = {'version': VERSION, 'bindings': bindings, 'validation': config['validation']}
    authority_sha = loader.digest_document(authority); control = root / 'staging-contract-private.json'
    if control.exists():
        if control.is_symlink() or json.loads(control.read_bytes()) != authority:
            raise ValueError('Prior staged authority differs')
    elif any(root.iterdir()):
        raise ValueError('Unbound existing staging root')
    else:
        cache.atomic_json(control, authority)
    for name in ('staging-receipt.json', MANIFEST):
        previous = root / name
        if previous.exists():
            if previous.is_symlink():
                raise ValueError('Prior metadata symlink forbidden')
            history = root / 'staging-history-private'; history.mkdir(exist_ok=True)
            previous.replace(history / (name + '.' + smoke.file_sha(previous)))
    report = {'version': VERSION, 'status': 'RUNNING', 'phase': 'metadata', 'authority_sha256': authority_sha,
              'sources': [{'ref': b['ref'], 'version': 1, 'cell_sources_sha256': b['cell_sources_sha256']} for b in bindings],
              'new_downloaded_bytes': 0, 'completed_files': 0, 'aggregate_contains_ids': False, 'download_workers': download_workers}
    started = time.monotonic()

    def write():
        report['elapsed_seconds'] = time.monotonic() - started; cache.atomic_json(root / 'staging-receipt.json', report)
    write()
    try:
        plans = []
        if shutil.disk_usage(root).free < 4 * 3 * METADATA_MAX_BYTES + DISK_RESERVE:
            raise OSError('Insufficient metadata staging disk')
        for i, binding in enumerate(bindings):
            directory = root / f'source{i}'
            if directory.is_symlink():
                raise ValueError('Shard destination symlink forbidden')
            directory.mkdir(exist_ok=True); inventory = _inventory(transport, binding)
            for name, field in METADATA.items():
                remote = PREFIX + name
                if remote not in inventory:
                    raise ValueError('Required metadata artifact absent')
                report['new_downloaded_bytes'] += _download(transport, inventory[remote], directory / name, binding[field], max_bytes=METADATA_MAX_BYTES)
            _snapshot(transport, binding); plans.append({'root': directory, 'binding': binding, 'inventory': inventory})
        files = _metadata_plan(plans, config['validation'])
        for plan in plans:
            allowed = {item['path'].name for item in files if item['path'].parent == plan['root']}
            if not {p.name for p in plan['root'].glob('*.npz')} <= allowed:
                raise ValueError('Unexpected existing cache file; preserve and stop')
        missing = []
        for item in files:
            path = item['path']; row = item['row']
            if path.exists() or path.is_symlink():
                if path.is_symlink() or path.stat().st_size != row['bytes'] or smoke.file_sha(path) != row['file_sha256']:
                    raise ValueError('Completed private artifact differs; preserve and stop')
            else:
                missing.append(item)
        required = sum(i['row']['bytes'] for i in missing) + 2 * download_workers * max((i['row']['bytes'] for i in files), default=0) + DISK_RESERVE
        free = shutil.disk_usage(root).free; report['disk_preflight'] = {'free_bytes': free, 'required_bytes': required, 'remaining_files': len(missing)}
        if free < required:
            raise OSError('Insufficient complete cache staging disk')
        report['phase'] = 'images'; report['completed_files'] = len(files) - len(missing); write()

        def completed(count):
            report['new_downloaded_bytes'] += count; report['completed_files'] += 1; write()
        _transfer(missing, transport, download_workers, completed)
        for binding in bindings:
            _snapshot(transport, binding)
        report['phase'] = 'assemble'; write()
        descriptors = []
        for plan in plans:
            for path in plan['root'].glob('*.npz'):
                if path.stat().st_mode & 0o222:
                    path.chmod(0o444)
            descriptors.append({'root': str(plan['root']), **{k: plan['binding'][k] for k in ('contract_file_sha256', 'index_sha256', 'receipt_sha256')}})
        manifest = loader.assemble(descriptors, config['validation'])
        for binding in bindings:
            _snapshot(transport, binding)
        cache.atomic_json(root / MANIFEST, manifest)
        report.update(status='PASS', phase='complete', studies=len(manifest['entries']), manifest_sha256=manifest['manifest_sha256'],
                      content_sha256=manifest['content_sha256'], manifest_file_sha256=smoke.file_sha(root / MANIFEST))
        write(); return report
    except BaseException as error:
        report.update(status='ERROR', failure={'phase': report['phase'], 'type': type(error).__name__}); write()
        raise RuntimeError('Raptor64 staging failed; completed files and sanitized receipt preserved') from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True); parser.add_argument('--download-workers', type=int, default=1)
    args = parser.parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_bytes()); report = prepare(config, KaggleTransport(), download_workers=args.download_workers)
    except BaseException:
        raise RuntimeError('Raptor64 staging failed; inspect sanitized staging receipt') from None
    print(json.dumps({'event': 'RAPTOR64_PREPARE_OK', 'studies': report['studies'], 'manifest_sha256': report['manifest_sha256'], 'content_sha256': report['content_sha256']}))


if __name__ == '__main__':
    main()
