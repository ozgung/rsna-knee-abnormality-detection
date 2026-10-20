"""Lane C solo probe (C1) for one new all-data S6 seed, on the accepted s6solo pair machinery. Never pushes or submits.

  accept     download the seed's private export file by file, verify SHA256SUMS / run_config / membership against the
             V2-all contract (only seed and labels may differ), run the strict CPU state/constructor/repeat check
             (scripts/qualify_s6solo_cpu.py with the seed's identity) and write <seed dir>/seed-acceptance.json
  rehearsal  the private/offline 2xT4 frozen-39 qualification packet of the 'v2' arm rebound to this seed
  verify     re-derive the emitted rehearsal packet byte for byte
  entry      the image-only entry, only from that exact rehearsal build plus its collected PASS result

Everything except cfg['arm'] (B checkpoint binding, receipts, CPU pins), the kernel id/title and the seed dataset
attachment is the accepted pair's shared source and config (scripts/build_s6_solo_pair.py), unchanged.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import build_s6_solo_pair as B
from knee import s6_solo_pair as PAIR
from knee.s6_solo_runtime import file_sha, require

LANE = B.OUT / 'analysis/lanec-20261019'
SEED_SOURCES = ('run_config.json', 'SHA256SUMS', 'membership_E_prior_s%d_f0.json', 'partial_preds_E_prior_s%d.json')
_ORIGINAL_ARMS, _ORIGINAL_METADATA = B.arms, B.metadata


SOURCE = {'kind': 'model'}       # 'dataset': the solo fixture links input/<mount> one level deep, so a C1 packet uses a dataset copy
MODEL = {'slug': 'knee-s6-lanec-seeds-20261019', 'lane': LANE}   # lane F (Oct 20) passes --model-slug knee-s6-lanef-20261020 --lane-dir


def names(tag, seed):
    ck = 'E_prior_s%d_%sall.pt' % (seed, tag)
    ref = ('ozgung/knee-s6-%sall-s%d-20261020' % (tag, seed) if SOURCE['kind'] == 'dataset'
           else 'ozgung/%s/pyTorch/%s-s%d' % (MODEL['slug'], tag, seed))
    return {'dir': MODEL['lane'] / ('%s-s%d' % (tag, seed)), 'dataset': ref,
            'checkpoint': ck,
            'alias': 'E_prior_s%d_f0.pt' % seed, 'membership': 'membership_E_prior_s%d_f0.json' % seed,
            'rehearsal_slug': 's6solo-%ss%d-qual-20261020' % (tag, seed), 'entry_slug': 's6solo-%ss%d-entry-20261020' % (tag, seed)}


def accept(tag, seed, version, python=sys.executable):
    """Local acceptance of the exported seed (counts and hashes only)."""
    n = names(tag, seed)
    d = n['dir'] / 'dl'
    d.mkdir(parents=True, exist_ok=True)
    files = [n['checkpoint']] + [x % seed if '%d' in x else x for x in SEED_SOURCES]
    if not all((d / f).is_file() for f in files):      # private Kaggle Model variation version (dataset quota exhausted)
        subprocess.run(['kaggle', 'models', 'instances', 'versions', 'download', '%s/%d' % (n['dataset'], version), '-p', str(d),
                        '--untar', '-q', '--force'], check=True)
        for z in list(d.glob('*.tar.gz')) + list(d.glob('*.zip')):
            import shutil
            shutil.unpack_archive(str(z), str(d))
            z.unlink()
    require(all((d / f).is_file() for f in files), 'SEED_EXPORT_FILES')
    sums = {}
    for line in (d / 'SHA256SUMS').read_text().splitlines():
        sha, name = line.split(None, 1)
        sums[name.strip()] = sha
    require(set(files) - {'SHA256SUMS'} <= set(sums), 'SEED_SHA256SUMS_COVERAGE')     # lane F exports also list targets.csv
    for name, sha in sums.items():
        require((d / name).is_file() and file_sha(d / name) == sha, 'SEED_SHA256SUMS_' + name)
    rc = json.loads((d / 'run_config.json').read_text())
    want = {'arms': 'E_prior', 'backbone': 'convnext_small', 'crop': 288, 'slices': 3, 'phys_mm': 3.5, 'spatial_attn': True,
            'extent': False, 'state_spec': None, 'freeze_blocks': 16, 'epochs': 16, 'seed_base': seed, 'seeds': 1,
            'include_review_holdout': False, 'pool_studies': 4345, 'gold_studies': 58, 'labels': PAIR.TARGETS,
            'batch': 2, 'train_windows': 64, 'lr': 1e-4, 'only_folds': '0'}
    bad = {k: (rc.get(k), v) for k, v in want.items() if rc.get(k) != v}
    require(not bad, 'SEED_RUN_CONFIG %s' % sorted(bad))
    v2acc = json.loads((B.ACCEPT / 'acceptance.json').read_text())['membership']
    mem = json.loads((d / n['membership']).read_text())
    mwant = {**{k: v2acc[k] for k in ('arm', 'all_data', 'train', 'train_sha256', 'holdout', 'gold', 'gold_sha256', 'train_gold_overlap')},
             'seed': seed}
    bad = {k: (mem.get(k), v) for k, v in mwant.items() if mem.get(k) != v}
    require(not bad, 'SEED_MEMBERSHIP %s' % sorted(bad))
    ck = d / n['checkpoint']
    cpu = n['dir'] / 'cpu.json'
    if not cpu.exists():
        extras = B.OUT / 's6v2/dl/extras/heliosli'
        subprocess.run([python, str(PROJECT / 'scripts/qualify_s6solo_cpu.py'), '--train', str(extras / 'compact_train.py'),
                        '--infer', str(extras / 'compact_infer.py'), '--control', str(B.ACCEPT / 'control/E_prior_s42_fall.pt'),
                        '--v2', str(ck), '--v2-sha', sums[n['checkpoint']], '--v2-bytes', str(ck.stat().st_size), '--out', str(cpu)],
                       check=True, cwd=str(PROJECT))
    c = json.loads(cpu.read_text())
    require(c['status'] == 'PASS_SYNTHETIC_CPU_ONLY' and c['checkpoints']['v2']['checkpoint_sha256'] == sums[n['checkpoint']]
            and c['checkpoints']['v2']['strict_load'] is True, 'SEED_CPU_STRICT')
    line = [x for x in (d / 'SHA256SUMS').read_text().splitlines() if x.split()[1:] == [n['checkpoint']]][0]
    acc = {'tag': tag, 'seed': seed, 'dataset': n['dataset'], 'dataset_version': version, 'checkpoint_filename': n['checkpoint'],
           'checkpoint_sha256': sums[n['checkpoint']], 'checkpoint_size': ck.stat().st_size, 'runtime_alias': n['alias'],
           'receipts': {name: file_sha(d / name) for name in ('run_config.json', 'SHA256SUMS', n['membership'])},
           'run_config_expect': want, 'membership_expect': mwant, 'sha256sums_line': line, 'cpu_receipt_sha256': file_sha(cpu),
           'state_sha256': c['checkpoints']['v2']['state_sha256'], 'partial_preds_sha256': file_sha(d / ('partial_preds_E_prior_s%d.json' % seed))}
    out = n['dir'] / 'seed-acceptance.json'
    out.write_text(json.dumps(acc, indent=1, sort_keys=True) + '\n')
    return acc


def seed_arm(tag, seed):
    n = names(tag, seed)
    acc = json.loads((n['dir'] / 'seed-acceptance.json').read_text())
    d = n['dir'] / 'dl'
    for name, sha in acc['receipts'].items():
        require(file_sha(d / name) == sha, 'SEED_RECEIPT_' + name)
    B._json_checked(d / 'run_config.json', acc['receipts']['run_config.json'], acc['run_config_expect'])
    B._json_checked(d / n['membership'], acc['receipts'][n['membership']], acc['membership_expect'])
    line = acc['sha256sums_line']
    require(line in (d / 'SHA256SUMS').read_text().splitlines() and line.split()[0] == acc['checkpoint_sha256'], 'SEED_SHA256SUMS_LINE')
    cpu = json.loads((n['dir'] / 'cpu.json').read_text())
    require(file_sha(n['dir'] / 'cpu.json') == acc['cpu_receipt_sha256'] and cpu['status'] == 'PASS_SYNTHETIC_CPU_ONLY'
            and cpu['checkpoints']['v2']['checkpoint_sha256'] == acc['checkpoint_sha256'] and cpu['checkpoints']['v2']['strict_load'] is True,
            'SEED_CPU_STRICT')
    ck = cpu['checkpoints']['v2']
    desc = {'variant': 'v2', 'targets': PAIR.TARGETS, 'dataset_slug': acc['dataset'], 'dataset_version': acc['dataset_version'],
            'checkpoint_sha256': acc['checkpoint_sha256'], 'checkpoint_size': acc['checkpoint_size'],
            'checkpoint_filename': acc['checkpoint_filename'], 'runtime_alias': acc['runtime_alias'], 'common': PAIR.COMMON}
    require(PAIR.ALIAS_RE.fullmatch(desc['runtime_alias']), 'LOADER_DISCOVERABLE_ALIAS')
    return {'variant': 'v2', 'label': 'solo-%s-s%d' % (tag.upper(), seed), 'descriptor': desc,
            'b_model': {'file': desc['runtime_alias'], 'source_file': desc['checkpoint_filename'], 'sha256': desc['checkpoint_sha256'],
                        'bytes': desc['checkpoint_size'], 'arm': 'E_prior', 'dataset': desc['dataset_slug'],
                        'dataset_version': desc['dataset_version'], 'descriptor_dataset_version': desc['dataset_version'],
                        'mount': (desc['dataset_slug'].split('/')[1] if len(desc['dataset_slug'].split('/')) == 2 else
                                  'models/ozgung/%s/pytorch/%s-s%d/%d' % (MODEL['slug'], tag, seed, desc['dataset_version']))},
            'receipts': dict(acc['receipts']), 'json_expect': {'run_config.json': acc['run_config_expect'], n['membership']: acc['membership_expect']},
            'text_expect': {'SHA256SUMS': line},
            'cpu': {'receipt_sha256': acc['cpu_receipt_sha256'], 'state_sha256': ck['state_sha256'],
                    'constructor_buffers_sha256': ck['constructor_buffers_sha256'], 'buffer_inventory': ck['buffer_inventory'],
                    'constructor': cpu['constructor'], 'strict_load': True},
            'provenance': 'Own Lane C all-data ConvNeXt-small S6 (V2-all recipe), labels %s, seed %d, 16 epochs, all 4345 pool studies; '
                          'gold-58 read once (selection-exposed for the probe choice only).' % (tag, seed)}


def install(tag, seed):
    """Rebind the pair builder's 'v2' arm, slugs and metadata to this seed (module globals, so verify replays the same)."""
    n = names(tag, seed)
    B.arms = lambda raw: {**_ORIGINAL_ARMS(raw), 'v2': seed_arm(tag, seed)}

    version = json.loads((n['dir'] / 'seed-acceptance.json').read_text())['dataset_version']

    def metadata(slug, variant):
        meta = _ORIGINAL_METADATA(slug, 'control')
        if variant == 'v2' and SOURCE['kind'] == 'dataset':
            meta['dataset_sources'] = meta['dataset_sources'] + ['%s/%d' % (n['dataset'], version)]
        elif variant == 'v2':
            meta['model_sources'] = ['%s/%d' % (n['dataset'], version)]
        return meta
    B.metadata = metadata
    B.SLUGS = {**B.SLUGS, 'v2': n['rehearsal_slug']}
    B.ENTRY_SLUGS = {**B.ENTRY_SLUGS, 'v2': n['entry_slug']}
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=['accept', 'rehearsal', 'verify', 'entry'])
    p.add_argument('--tag', required=True, choices=['v2', 'v3', 'v4b', 'v5'])
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--python', default=sys.executable, help='interpreter with torch/timm for the CPU check')
    p.add_argument('--rehearsal-result')
    p.add_argument('--version', type=int, default=1, help='Kaggle Model variation / dataset version of the export')
    p.add_argument('--source', choices=['model', 'dataset'], default='model')
    p.add_argument('--model-slug', default=MODEL['slug'], help='Kaggle Model holding the export (lane C default; lane F: knee-s6-lanef-20261020)')
    p.add_argument('--lane-dir', default=None, help='local analysis directory for the seed (default: lane C)')
    a = p.parse_args(argv)
    SOURCE['kind'] = a.source
    MODEL['slug'] = a.model_slug
    if a.lane_dir:
        MODEL['lane'] = Path(a.lane_dir)
    if a.command == 'accept':
        print(json.dumps({k: v for k, v in accept(a.tag, a.seed, a.version, python=a.python).items() if k not in ('run_config_expect',)}, indent=1))
        return 0
    n = install(a.tag, a.seed)
    root = B.OUT / 'kaggle'
    if a.command == 'rehearsal':
        r = B.emit(root / n['rehearsal_slug'], n['rehearsal_slug'], 'v2', B.packet('v2'), 'rehearsal')
    elif a.command == 'verify':
        r = B.verify_build(root / n['rehearsal_slug'], 'v2')
    else:
        require(a.rehearsal_result, 'REHEARSAL_RESULT_REQUIRED')
        r = B.build_entry('v2', root / n['rehearsal_slug'], a.rehearsal_result, root / n['entry_slug'], n['entry_slug'])
    print(json.dumps({k: r.get(k) for k in ('status', 'slug', 'payload_sha256', 'files', 'notebook_bytes', 'b_model')}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
