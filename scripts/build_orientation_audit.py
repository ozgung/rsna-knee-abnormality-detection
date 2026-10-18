"""Build only: private CPU/offline, header-only DICOM orientation audit with per-study mirror routing.

Never pushes, launches or reads labels. Reuses the qualified wide96 reader (SLOTS96, pinned public reader
SHA 859aae12...) for the series/plane assignment, exactly as the 96-slot cache did.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee import orientation_audit as audit, source_span_pilot as pilot, public_training_smoke as smoke
from knee.reference_blend import DOCKER_IMAGE

PILOT_RECEIPT_SHA = '48655a7ef2863e2038beb9d7247f91afb7f9aba1c8153813b0cdaf03fb400e45'
PILOT_RECEIPT = 'outputs/kaggle/source-span-pilot-20261012-result-v1/source_span_pilot/receipt.json'
DEFAULT_PUBLIC = 'outputs/research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py'
CACHE_CONTRACTS = tuple(f'outputs/kaggle/source96-cache-shard{i}-20261012-reviewed-v1/build-contract.json' for i in range(4))
DEFAULT_DESTINATION = 'outputs/kaggle/orientation-audit-v2-20261012'
DEFAULT_SLUG = 'knee-orientation-audit-v2-20261012'
FILES = ('knee/__init__.py', 'knee/orientation_audit.py', 'knee/source_span_pilot.py', 'knee/corpus44_parity.py',
         'knee/public_training_smoke.py', 'knee/public_rich_cache.py', 'knee/reference.py',
         'knee/fresh_reader_setup.py', 'knee/raptor_input.py')
DENOMINATORS = {
    'study': 'every study in the series CSV (train: all studies; test: the visible studies); each gets exactly one routing category',
    'plane': 'studies with at least one selected series in that plane (Sagittal: sag0/sag1, Coronal: cor0/cor1, Axial: ax)',
    'slot': 'studies whose slot has a selected series; slot names are plane abbreviation plus index within the plane '
            '(sag0 = Sagittal fluid-pref 1, sag1 = Sagittal fluid-pref 0, cor0 = Coronal fluid-pref 1, cor1 = Coronal fluid-pref 0, ax = Axial)',
    'series': 'selected series that exist for a slot; series without readable .dcm files stay in the denominator as unverifiable',
    'slice': 'header files read (all instances when instance_cap is null); used slices are the 18/26/22/12 reader picks per slot',
}
ROUTING_RULE = {
    'mirror_ok': 'every present selected series is ok_standard: mirror vector L/R, standard class, complete (IOP, IPP, PixelSpacing), '
                 'consistent, fully observed (never sampled) geometry, strictly ascending unambiguous slice order, no reader fallback',
    'needs_transform': 'no series needs bypass, and at least one has complete consistent geometry with a standard-class L/R vector on a different image axis (exact flip exists but is unqualified)',
    'bypass_plain': 'any present series is oblique, nonstandard, unverifiable, inconsistent, incomplete, sampled, degenerate/ambiguous order '
                    'or has no standard L/R vector; also a study with no present series',
    'governs': 'per-study routing governs; cohort fractions are descriptive only',
}


def require_cache_contracts(evidence, hashes, train_series_sha):
    """All four cache-shard build contracts must exist and agree on reader hashes, layout, span and train CSV."""
    for name in CACHE_CONTRACTS:
        path = Path(evidence) / name
        if not path.is_file():
            raise ValueError(f'Required cache build contract missing: {name}')
        contract = json.loads(path.read_bytes())
        if (contract['reader_hashes'] != hashes or contract['config']['slots'] != [list(s) for s in pilot.SLOTS96]
                or contract['config']['span'] != list(audit.SPAN96)):
            raise ValueError('Reader hashes or layout differ from a cache build contract')
        if contract['input_pins']['competition']['train_series.csv'] != train_series_sha:
            raise ValueError('Local train_series.csv differs from the file the cache used')
    return len(CACHE_CONTRACTS)


def build(destination, slug, *, evidence_root, public_source, workers=8, instance_cap=None):
    out, evidence = Path(destination), Path(evidence_root)
    if out.exists():
        raise ValueError('Fresh immutable build directory required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*', slug):
        raise ValueError('Invalid notebook slug')
    if type(workers) is not int or not 1 <= workers <= 16 or instance_cap is not None and (type(instance_cap) is not int or instance_cap < 2):
        raise ValueError('Invalid workers or instance cap')
    receipt_path = evidence / PILOT_RECEIPT
    smoke.require_file(receipt_path, PILOT_RECEIPT_SHA)
    qualification = json.loads(receipt_path.read_bytes())
    if qualification['status'] != 'PASS' or not qualification['exact_control_equal'] or qualification['config'] != pilot.CONFIG:
        raise ValueError('Qualified real32 pilot differs')
    source_path = Path(public_source)
    codes = pilot.recipes((source_path if source_path.is_absolute() else evidence / source_path).read_text())
    hashes = {k: smoke.digest(v.encode()) for k, v in codes.items()}
    if hashes != qualification['input_pins']['readers']:
        raise ValueError('Reader differs from the qualified real pilot')
    train_csv = evidence / 'data/raw/train_series.csv'
    verified_cache_contracts = require_cache_contracts(evidence, hashes, smoke.file_sha(train_csv))
    pick, slots = audit.pinned_selector(codes)
    records = audit.load_series_records(train_csv)   # metadata columns only; train.csv (targets) is never opened
    selected = sum(uid is not None for rows in records.values() for _s, _p, uid in audit.select_series(pick, slots, rows))
    sources = {name: (ROOT / name).read_text() for name in FILES}
    for name, source in sources.items():
        compile(source, name, 'exec')
        if name in qualification['source_hashes'] and smoke.digest(source.encode()) != qualification['source_hashes'][name]:
            raise ValueError('Reused helper differs from the qualified real pilot')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    contract = {'version': audit.VERSION, 'thresholds': audit.thresholds(), 'slots': [list(s) for s in slots],
                'span': list(audit.SPAN96), 'slot_names': audit.slot_names(slots), 'reader_hashes': hashes,
                'reader_original_sha256': pilot.parity.SOURCE_SHA, 'qualification_receipt_sha256': PILOT_RECEIPT_SHA,
                'verified_cache_build_contracts': verified_cache_contracts,
                'train_series_csv_sha256': smoke.file_sha(train_csv), 'train_studies': len(records),
                'train_series_listed': sum(len(r) for r in records.values()),
                'expected_selected_series': {'train': selected}, 'workers': workers, 'instance_cap': instance_cap,
                'denominators': DENOMINATORS, 'routing_rule': ROUTING_RULE,
                'descriptive_threshold': {'fraction': audit.DESCRIPTIVE_MIN_FRACTION, 'min_studies': audit.DESCRIPTIVE_MIN_STUDIES},
                'sources': {k: smoke.digest(v.encode()) for k, v in sources.items()}, 'code_revision': revision,
                'working_source_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).strip(),
                'policy': 'Header metadata only (stop_before_pixels); no labels, reports or targets opened; no models; '
                          'aggregate output has no identifiers; routing arrays are separate private files',
                'runtime_limit': 'Actual CPU platform12h only'}
    code = ("import time,sys,json,hashlib\nfrom pathlib import Path\nSTARTED=time.monotonic()\n"
            f'SOURCES={sources!r}\nCONTRACT={contract!r}\nREADERS={codes!r}\n'
            "base=Path('/kaggle/working/orientation_code')\n"
            "for name,source in SOURCES.items():\n"
            "    if hashlib.sha256(source.encode()).hexdigest()!=CONTRACT['sources'][name]:raise ValueError('Packaged source differs')\n"
            "    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
            "sys.path.insert(0,str(base))\n"
            "if sys.version_info[:2]!=(3,12):raise ValueError('Pinned Python environment changed')\n"
            "from knee.orientation_audit import run\n"
            "r=run('/kaggle/input','/kaggle/working/orientation_audit',CONTRACT,READERS,started=STARTED)\n"
            "print(json.dumps({'event':'orientation_audit_done','status':r['status'],'splits':r['splits'],'elapsed_seconds':r['elapsed_seconds']}))\n")
    compile(code, 'orientation-audit-notebook', 'exec')
    if re.search(r'1\.2\.826\.0\.1\.3680043', code):
        raise ValueError('A study or series identity is embedded in the notebook source')
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python3', 'language': 'python'}}, 'cells': [
        {'cell_type': 'markdown', 'id': 'protocol', 'metadata': {}, 'source': [
            '# Header-only DICOM orientation audit and per-study mirror routing\n',
            'CPU, offline, private. Series chosen by the qualified wide96 reader policy (slots 26/22/18/12/18). Reads DICOM headers only;\n',
            'no pixels, labels, reports or models. Aggregate JSON has counts only; routing-private-*.json hold per-study routes and must stay private.\n']},
        {'cell_type': 'code', 'id': 'run', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': code.splitlines(True)}]}
    blob = (json.dumps(nb, indent=2) + '\n').encode()
    if len(blob) > 900000:
        raise ValueError('Notebook source limit')
    metadata = {'id': 'ozgung/' + slug, 'title': slug.replace('-', ' ').title(), 'code_file': 'notebook.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False,
                'enable_tpu': False, 'enable_internet': False, 'competition_sources': ['rsna-knee-abnormality-detection'],
                'dataset_sources': [], 'kernel_sources': [], 'model_sources': [], 'docker_image': DOCKER_IMAGE}
    receipt = {'notebook_sha256': smoke.digest(blob), 'notebook_bytes': len(blob),
               'contract_sha256': smoke.digest(smoke.canonical(contract)), 'bundle_sha256': smoke.digest(smoke.canonical(sources)),
               'code_revision': revision, 'build_only': True, 'expected_selected_series_train': selected,
               'train_studies': len(records), 'verified_cache_build_contracts': verified_cache_contracts}
    out.mkdir(parents=True)
    (out / 'notebook.ipynb').write_bytes(blob)
    for name, value in [('kernel-metadata.json', metadata), ('build-contract.json', contract), ('build-receipt.json', receipt)]:
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination', type=Path, nargs='?', default=Path(DEFAULT_DESTINATION))
    p.add_argument('slug', nargs='?', default=DEFAULT_SLUG)
    p.add_argument('--evidence-root', type=Path, default=ROOT)
    p.add_argument('--public-source', type=Path, default=Path(DEFAULT_PUBLIC))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--instance-cap', type=int, default=None)
    args = p.parse_args()
    result = build(**vars(args))
    print(json.dumps(result, indent=2))
    print('BUILD ONLY. Push NOT executed. When authorized, run from the project root:')
    print(f'  /Users/ozgun/miniconda3/bin/kaggle kernels push -p {args.destination}')
