"""Local metadata-only qualification of the four Raptor64 cache kernels; writes the private VM stage config.

For each shard: exact private v1 COMPLETE kernel whose cell sources equal the pushed build notebook,
download only raptor64_cache/{run-contract.json,index-private.json,receipt.json} (no images), and
require the contract to equal the local build contract, receipt PASS with full partition coverage,
index bound to the receipt and the exact NPZ inventory. Writes an aggregate receipt (no IDs) and a
PRIVATE stage config (membership hashes, private fold map, source bindings) for colab_raptor64_prepare.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee import raptor64_cache as cache, raptor64_loader as loader, public_training_smoke as smoke

_spec = importlib.util.spec_from_file_location('colab_raptor64_prepare', Path(__file__).resolve().parent / 'colab_raptor64_prepare.py')
stage = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(stage)
SLUG = 'knee-raptor64-cache-shard{}-20261013'
VM_NOTEBOOK = '/content/raptor64-accepted/shard{}.ipynb'


def qualify(evidence, transport, *, slug=SLUG, build_dir=None, out_root=None):
    build_dir = build_dir or slug.removeprefix('knee-')
    evidence = Path(evidence); shards = []; contracts = []
    for i in range(4):
        build = evidence / f'outputs/kaggle/{build_dir.format(i)}'
        notebook = build / 'notebook.ipynb'; contract = json.loads((build / 'build-contract.json').read_bytes())
        ref = 'ozgung/' + slug.format(i)
        binding = {'ref': ref, 'version': 1, 'notebook_sha256': smoke.file_sha(notebook), 'contract_file_sha256': '0' * 64,
                   'index_sha256': '0' * 64, 'receipt_sha256': '0' * 64, 'cell_sources_sha256': stage.s96.source_cells_sha256(notebook.read_text())}
        inventory = stage._inventory(transport, binding)
        target = Path(out_root or evidence / 'outputs/kaggle') / f'{build_dir.format(i)}-result-v1/raptor64_cache'
        target.mkdir(parents=True, exist_ok=True)
        pins = {}
        for name in stage.METADATA:
            remote = stage.PREFIX + name
            if remote not in inventory:
                raise ValueError('Required metadata artifact absent')
            path = target / name
            if not path.exists():
                tmp = path.with_name(path.name + '.tmp')
                with tmp.open('wb') as f:
                    for chunk in transport.iter_chunks(inventory[remote]):
                        f.write(chunk)
                tmp.replace(path)
            pins[name] = smoke.file_sha(path)
        observed = json.loads((target / 'run-contract.json').read_bytes())
        if observed != contract:
            raise ValueError(f'Shard {i} run contract differs from the pushed build contract')
        receipt = json.loads((target / 'receipt.json').read_bytes()); index = json.loads((target / 'index-private.json').read_bytes())
        members = cache.make_shards([str(n) for n in range(contract['membership']['eligible'])], 4)[i]
        if (receipt.get('status') != 'PASS' or receipt.get('completed_studies') != len(members) or receipt.get('contract_sha256') != loader.digest_document(contract) or
                receipt.get('private_index_sha256') != pins['index-private.json'] or [e['ordinal'] for e in index['entries']] != members):
            raise ValueError(f'Shard {i} is not a complete PASS')
        for entry in index['entries']:
            loader.validate_slots(entry['slots'])
        if {n for n in inventory if n.endswith('.npz')} != {stage.PREFIX + cache.filename(n) for n in members}:
            raise ValueError(f'Shard {i} NPZ inventory differs')
        contracts.append(contract)
        shards.append({'ref': ref, 'version': 1, 'notebook_path': VM_NOTEBOOK.format(i), 'notebook_sha256': binding['notebook_sha256'],
                       'contract_file_sha256': pins['run-contract.json'], 'index_sha256': pins['index-private.json'], 'receipt_sha256': pins['receipt.json'],
                       '_aggregate': {k: receipt.get(k) for k in ('completed_studies', 'bytes', 'elapsed_seconds', 'decode_seconds', 'window_seconds',
                                                                   'write_seconds', 'peak_host_rss_bytes', 'counters')}})
    if any({k: c[k] for k in loader.COMMON} != {k: contracts[0][k] for k in loader.COMMON} for c in contracts):
        raise ValueError('Shards differ in scientific source/pins')
    return shards, contracts[0]['membership']


def stage_config(evidence, shards, membership):
    import numpy as np
    evidence = Path(evidence)
    folds = json.loads((evidence / 'outputs/kaggle/trackA-coatnet224-fold01-20261009/folds_v0.json').read_bytes())['folds']
    ids = []
    for sub, prefix in [('corpus', 'all'), ('corpus-ext', 'extra')]:
        ids.extend(np.load(evidence / f'outputs/analysis/public-train-recipe-20261009/{sub}/{prefix}_ids.npy', allow_pickle=False).astype(str).tolist())
    eligible_fold = {u: folds[u] for u in ids if u in folds and type(folds[u]) is int and 0 <= folds[u] <= 4}
    validation = {'fixture': False, 'membership': membership, 'folds': eligible_fold}
    return {'version': stage.VERSION, 'cache_root': '/content/raptor64_cache', 'validation': validation,
            'sources': [{k: v for k, v in s.items() if k != '_aggregate'} for s in shards]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--evidence-root', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True); p.add_argument('--stage-config', type=Path, required=True)
    a = p.parse_args(argv)
    if a.receipt.exists() or a.stage_config.exists():
        raise SystemExit('Fresh receipt/stage-config paths required')
    shards, membership = qualify(a.evidence_root, stage.KaggleTransport())
    config = stage_config(a.evidence_root, shards, membership)
    a.stage_config.write_text(json.dumps(config, indent=2) + '\n'); a.stage_config.chmod(0o600)
    receipt = {'status': 'PASS', 'shards': [{k: v for k, v in s.items() if k not in ('notebook_path',)} for s in shards],
               'membership_counts': {k: membership[k] for k in ('eligible', 'training', 'development')},
               'stage_config_sha256': smoke.file_sha(a.stage_config), 'images_downloaded': False, 'aggregate_contains_ids': False}
    a.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'stage_config_sha256': receipt['stage_config_sha256'],
                      'studies': sum(s['_aggregate']['completed_studies'] for s in shards)}))


if __name__ == '__main__':
    main()
