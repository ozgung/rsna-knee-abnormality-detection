"""Build-only private CPU shard of the Raptor64 (Dread widedense v4) image cache.

The reader is the audited `knee.reference` raptor profile of the pinned Dread inference
source (unmodified v4 geometry). No labels, report text, models or training are used.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee import raptor64_cache as cache, public_training_smoke as smoke, reference
from knee.raptor_input import policy_contract
from knee.reference_blend import DOCKER_IMAGE, CONVNEXT_DATASET
from knee.coverage_inference import CODECS

# Same pinned Kaggle docker image as the four qualified source96 CPU shards; their observed
# CPU environment is the qualified one (receipt pinned by SHA).
ENVIRONMENT_RECEIPT = 'outputs/kaggle/source96-cache-shard0-20261012-result-v1/source96_cache/receipt.json'
ENVIRONMENT_RECEIPT_SHA = '898c6a24271a48490ecfcef0adc5aa43a9a11343d9dde0bc8a116d2ad5a1cc45'
TRAIN_CSV_SHA = '8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33'
FILES = ('knee/__init__.py', 'knee/raptor64_cache.py', 'knee/source96_cache.py', 'knee/source_span_pilot.py',
         'knee/corpus44_parity.py', 'knee/public_training_smoke.py', 'knee/public_rich_cache.py', 'knee/reference.py',
         'knee/fresh_reader_setup.py', 'knee/raptor_input.py')
OWNER = 'ozgung'


def reader(public_source):
    """Audited raptor-profile extraction (source SHA pinned inside knee.reference)."""
    module, decoder, manifest = reference.audit_source(public_source, profile='raptor',
                                                      checkpoint_sha256=cache.V4_CHECKPOINT_SHA256, raptor_windows=42)
    if decoder:
        raise ValueError('Raptor profile must not embed a decoder')
    return module, manifest


def build(destination, slug, *, evidence_root, public_source, shard_index, shard_count=4, resume_kernel=None, resume_dataset=None):
    import numpy as np
    out = Path(destination); evidence = Path(evidence_root)
    if out.exists():
        raise ValueError('Fresh immutable build directory required')
    cache.check_slug(slug)
    if len(OWNER + '/' + slug) > 64:
        raise ValueError('Kernel reference too long')
    if type(shard_index) is not int or type(shard_count) is not int or not 0 <= shard_index < shard_count <= 64:
        raise ValueError('Invalid shard index/count')
    if resume_kernel and resume_dataset:
        raise ValueError('Select one resume transport')
    for ref in (resume_kernel, resume_dataset):
        if ref:
            owner, _, name = ref.partition('/')
            if owner != OWNER:
                raise ValueError('Resume must be an explicit owned private attachment')
            cache.check_slug(name)
    public_path = Path(public_source) if Path(public_source).is_absolute() else evidence / public_source
    module, manifest = reader(public_path.read_text())
    public = cache.load_public(module, manifest)  # executes only audited definitions; verifies v4 geometry
    env_path = evidence / ENVIRONMENT_RECEIPT
    smoke.require_file(env_path, ENVIRONMENT_RECEIPT_SHA)
    env_receipt = json.loads(env_path.read_bytes())
    if env_receipt.get('status') != 'PASS':
        raise ValueError('Qualified environment receipt is not PASS')
    identity = evidence / 'outputs/kaggle/sealed-identity-full-result/sealed_identity/sealed-manifest-private.json'
    smoke.require_file(identity, smoke.MANIFEST_SHA); identity_document = json.loads(identity.read_bytes())
    historical = []
    for sub, prefix in [('corpus', 'all'), ('corpus-ext', 'extra')]:
        path = evidence / f'outputs/analysis/public-train-recipe-20261009/{sub}/{prefix}_ids.npy'
        smoke.require_file(path, smoke.PINS[prefix + '_ids.npy']); historical.extend(np.load(path, allow_pickle=False).astype(str).tolist())
    folds_path = evidence / 'outputs/kaggle/trackA-coatnet224-fold01-20261009/folds_v0.json'
    folds_sha = smoke.file_sha(folds_path); folds = json.loads(folds_path.read_bytes())['folds']
    raw = evidence / 'data/raw'; smoke.require_file(raw / 'train.csv', TRAIN_CSV_SHA)
    ids, summary = cache.membership(historical, raw / 'train.csv', folds, identity_document)
    shards = cache.make_shards(ids, shard_count)
    partitions = [{'studies': len(members), 'ordinals_sha256': smoke.digest(smoke.canonical(members)),
                   'ordered_ids_sha256': smoke.digest(smoke.canonical([ids[n] for n in members]))} for members in shards]
    sources = {name: (ROOT / name).read_text() for name in FILES}
    for name, source in sources.items():
        compile(source, name, 'exec')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    contract = {'version': cache.VERSION, 'config': cache.CONFIG, 'membership': summary, 'shard_index': shard_index,
                'shard_count': shard_count, 'partitions': partitions,
                'reader': {'module_sha256': smoke.digest(module.encode()), 'manifest_sha256': smoke.digest(smoke.canonical(manifest)),
                           'source_sha256': reference.RAPTOR_SOURCE_SHA256, 'input_policy': policy_contract(),
                           'slots': [list(s) for s in public.SLOTS], 'depth': public.MAXS, 'storage': public.IMG,
                           'qualification': 'raptor production rehearsal v2 (67 studies, strict comparisons PASS) and B0 entry; same audited reader'},
                'code_revision': revision,
                'working_source_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).strip(),
                'sources': {k: smoke.digest(v.encode()) for k, v in sources.items()},
                'input_pins': {'competition': {n: smoke.file_sha(raw / n) for n in ('train.csv', 'train_series.csv')},
                               'historical_ids': {n: v for n, v in smoke.PINS.items() if n.endswith('_ids.npy')},
                               'folds_sha256': folds_sha, 'identity_sha256': smoke.MANIFEST_SHA, 'codecs': CODECS,
                               'environment_receipt_sha256': ENVIRONMENT_RECEIPT_SHA},
                'qualified_environment': env_receipt['environment'],
                'policy': 'Private cache only; unmodified Raptor v4 reader 64 slots/336px; no labels/report text/outcomes/models/training; fail on any study failure',
                'resume_policy': 'Exact same contract, aligned membership, file/raw hashes; external transport does not change scientific contract',
                'storage_upper_bound_bytes': cache.disk_required(len(shards[shard_index])),
                'runtime_limit': 'Actual CPU platform 12h only'}
    code = ("import time,sys,json,hashlib,subprocess\nfrom pathlib import Path\nSTARTED=time.monotonic()\n"
            f'SOURCES={sources!r}\nCONTRACT={contract!r}\nMODULE={module!r}\nMANIFEST={manifest!r}\n'
            "base=Path('/kaggle/working/raptor64_code')\n"
            "for name,source in SOURCES.items():\n"
            "    if hashlib.sha256(source.encode()).hexdigest()!=CONTRACT['sources'][name]:raise ValueError('Packaged source differs')\n"
            "    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
            "sys.path.insert(0,str(base))\nfrom knee.fresh_reader_setup import dataset_root\nfrom knee.public_training_smoke import require_file\n"
            f"codecs=dataset_root('/kaggle/input',{CONVNEXT_DATASET!r},list(CONTRACT['input_pins']['codecs']))\n"
            "wheels=[]\nfor name,sha in CONTRACT['input_pins']['codecs'].items():\n"
            "    require_file(codecs/name,sha);wheels.append(str(codecs/name))\n"
            "if sys.version_info[:2]!=(3,12):raise ValueError('Pinned Python environment changed')\n"
            "subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*wheels],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            "from knee.raptor64_cache import run\n"
            "r=run('/kaggle/input','/kaggle/working/raptor64_cache',CONTRACT,MODULE,MANIFEST,CONTRACT['input_pins']['folds_sha256'],"
            f"started=STARTED,resume_kernel={resume_kernel!r},resume_dataset={resume_dataset!r})\n"
            "print(json.dumps({'event':'raptor64_cache_done','status':r['status'],'studies':r['completed_studies'],'bytes':r['bytes']}))\n")
    compile(code, 'raptor64-cache-notebook', 'exec')
    if any(r['uid'] in code for r in identity_document['records']):
        raise ValueError('Private study identity embedded in source')
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python3', 'language': 'python'}}, 'cells': [
        {'cell_type': 'markdown', 'id': 'protocol', 'metadata': {}, 'source': ['# Private Raptor64 v4 CPU cache shard\n',
            f'Shard {shard_index}/{shard_count}: {len(shards[shard_index])} of 4345 non-gold/non-quarantine studies in historical corpus order.\n',
            'Unmodified audited Raptor v4 reader: 336 px/140 mm, slots 18/14/12/8/12, 6-94% span. One-study streaming, atomic compressed files, private aligned index.\n',
            'Aggregate receipt has no IDs/report text; raw cache artifacts and index MUST remain private. No public dataset, model or training.\n']},
        {'cell_type': 'code', 'id': 'run', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': code.splitlines(True)}]}
    blob = (json.dumps(nb, indent=2) + '\n').encode()
    if len(blob) > 900000:
        raise ValueError('Notebook source limit')
    metadata = {'id': OWNER + '/' + slug, 'title': slug.replace('-', ' ').title(), 'code_file': 'notebook.ipynb', 'language': 'python',
                'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False, 'enable_internet': False,
                'competition_sources': ['rsna-knee-abnormality-detection'],
                'dataset_sources': [*smoke.CORPORA, smoke.ASSETS, CONVNEXT_DATASET] + ([resume_dataset] if resume_dataset else []),
                'kernel_sources': [smoke.IDENTITY_KERNEL] + ([resume_kernel] if resume_kernel else []), 'model_sources': [],
                'docker_image': DOCKER_IMAGE}
    receipt = {'notebook_sha256': smoke.digest(blob), 'notebook_bytes': len(blob), 'contract_sha256': smoke.digest(smoke.canonical(contract)),
               'bundle_sha256': smoke.digest(smoke.canonical(sources)), 'code_revision': revision, 'build_only': True,
               'kernel': metadata['id'], 'shard_index': shard_index, 'shard_count': shard_count, 'studies': len(shards[shard_index]),
               'required_free_disk_upper_bound_bytes': contract['storage_upper_bound_bytes']}
    out.mkdir(parents=True); (out / 'notebook.ipynb').write_bytes(blob)
    for name, value in [('kernel-metadata.json', metadata), ('build-contract.json', contract), ('build-receipt.json', receipt)]:
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('destination', type=Path); p.add_argument('slug')
    for name in ('evidence-root', 'public-source'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--shard-index', type=int, required=True); p.add_argument('--shard-count', type=int, default=4)
    group = p.add_mutually_exclusive_group(); group.add_argument('--resume-kernel'); group.add_argument('--resume-dataset')
    print(json.dumps(build(**vars(p.parse_args())), indent=2))
