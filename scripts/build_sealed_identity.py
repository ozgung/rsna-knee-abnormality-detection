"""Build, never launch, a private offline CPU identity audit notebook."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee.sealed_identity import (POLICY, PILOT_SECONDS, FULL_SECONDS, OUTPUT_BYTES, METADATA,
    fingerprint, stamped, validate_stamp, load_metadata, runtime_contract, validate_pilot, require)
from knee.validation import file_sha256

DOCKER = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
KAGGLE_SOURCE_BYTES = 1_000_000
LEGACY_BASELINE = '19683d3463f0a299abaaafc92ef4fd88d72b25017e6a22ad085cecc96f16d09f'
BASELINE = '47697ebc8045d178f44488f4c4c6004aed765d8bdb7b4b4d3d409a7ac2da6a82'
FEATURES = 'a8190ab6129dccae859171eec9f1d024d9882652b84edc2735980360084ea76b'
PATIENTS = '6e5a26633f7a5a1222947084f5b280d386ac8c404f7840a81a8845701b15cda3'
PLAN = '1bd346c05c0aac20f54b82b790e356d3d6c1b95606accaa4a5d1da7e3e421942'


def original_contract(data_dir, baseline_dir, plan_path):
    # Local builder only. Never copy feature tensors, reports or labels into the notebook.
    import numpy as np
    baseline_dir, data_dir = Path(baseline_dir), Path(data_dir)
    extraction = json.loads((baseline_dir / 'extraction.json').read_text())
    plan = json.loads(Path(plan_path).read_text())
    validate_stamp(plan)
    require(plan['fingerprint'] == PLAN and plan['baseline_fingerprint'] == LEGACY_BASELINE,
            'reviewed_fresh_plan_changed')
    require(file_sha256(baseline_dir / 'features.npz') == extraction['features_sha256'] == FEATURES and
            file_sha256(baseline_dir / 'patients-private.json') == PATIENTS, 'reviewed_original_artifacts_changed')
    metadata = {name: file_sha256(data_dir / name) for name in METADATA}
    require(metadata == extraction['metadata_fingerprints'] == plan['files'], 'original_metadata_changed')
    patients = json.loads((baseline_dir / 'patients-private.json').read_text())
    with np.load(baseline_dir / 'features.npz', allow_pickle=False) as source:
        ids, splits, groups, targets = (source[k].astype(str).tolist() for k in ('ids', 'splits', 'groups', 'targets'))
    require(len(ids) == len(splits) == len(groups) == len(patients) == 2358, 'original_array_alignment_changed')
    records = [dict(uid=uid, split=role, group=group, patient_hashes=sorted(set(hashes)))
               for uid, role, group, hashes in zip(ids, splits, groups, patients)]
    baseline = [{k: r[k] for k in ('uid', 'split', 'group')} for r in records if r['split'] != 'gold']
    require(fingerprint(baseline) == BASELINE, 'original_ordered_assignments_changed')
    # Historical fresh make_plan adds labels to the shared baseline dictionaries
    # for its selected712 rows before hashing. Reproduce that legacy digest only
    # locally; the new private contract retains no label values.
    plan_labels = {r['uid']: r['labels'] for r in plan['records']}
    legacy = [r | ({'labels': plan_labels[r['uid']]} if r['uid'] in plan_labels else {}) for r in baseline]
    require(fingerprint(legacy) == LEGACY_BASELINE, 'legacy_baseline_fingerprint_changed')
    original = stamped({'policy': POLICY, 'records': records, 'targets': targets,
        'baseline_fingerprint': BASELINE, 'legacy_baseline_fingerprint': LEGACY_BASELINE,
        'metadata_sha256': metadata, 'study_count': 4407, 'series_count': 24371,
        'original_counts': {'gold': 58, 'train': 2000, 'weak_validation': 300},
        'artifact_sha256': {'features.npz': FEATURES, 'patients-private.json': PATIENTS,
                           'extraction.json': file_sha256(baseline_dir / 'extraction.json'),
                           'fresh-plan-private.json': file_sha256(plan_path)}, 'fresh_plan_fingerprint': PLAN})
    load_metadata(data_dir, original)
    lookup = {r['uid']: r for r in records}
    require(all(r['uid'] in lookup and all(r[k] == lookup[r['uid']][k] for k in ('split', 'group'))
                for r in plan['records']), 'fresh_plan_changed_original_assignments')
    return original


def build(data_dir, baseline_dir, plan_path, destination, slug, *, stage='pilot', pilot_dir=None):
    require(stage in ('pilot', 'full') and ((stage == 'full') == (pilot_dir is not None)), 'stage_contract_mismatch')
    require(slug.startswith('knee-') and all(c.isalnum() or c == '-' for c in slug), 'invalid_notebook_slug')
    original = original_contract(data_dir, baseline_dir, plan_path)
    contract = runtime_contract(original)
    pilot = None
    if pilot_dir:
        pilot_dir = Path(pilot_dir)
        require(sum((pilot_dir / name).stat().st_size for name in ('identity-private.json', 'receipt.json'))
                <= OUTPUT_BYTES, 'pilot_artifact_size_exceeded')
        pilot = {'identity': json.loads((pilot_dir / 'identity-private.json').read_text()),
                 'receipt': json.loads((pilot_dir / 'receipt.json').read_text())}
        groups, gold, keys = load_metadata(data_dir, original)
        validate_pilot(pilot, contract, groups, original, keys)
    private = json.dumps({'original': original, 'pilot': pilot}, sort_keys=True, allow_nan=False)
    require(len(private.encode()) <= OUTPUT_BYTES, 'embedded_private_contract_size_exceeded')
    own = {name: (ROOT / name).read_text() for name in
           ('knee/__init__.py', 'knee/validation.py', 'knee/fresh_reader_setup.py', 'knee/sealed_identity.py')}
    for name, content in own.items():
        compile(content, name, 'exec')
    stage_seconds = PILOT_SECONDS if stage == 'pilot' else FULL_SECONDS
    code = f'''import time
_audit_started = time.monotonic()
import json, sys
from pathlib import Path
CODE = {own!r}
base = Path('/kaggle/working/sealed_identity_code')
for name, content in CODE.items():
    path = base / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
sys.path.insert(0, str(base))
from knee.fresh_reader_setup import exact_root, time_limit
from knee.sealed_identity import run_audit, runtime_contract, require
try:
    with time_limit({stage_seconds} - (time.monotonic() - _audit_started), 'whole identity stage'):
        private = json.loads({private!r})
        require(runtime_contract(private['original']) == {contract!r}, 'embedded_runtime_source_mismatch')
        inputs = Path('/kaggle/input')
        data = exact_root([inputs/'competitions/rsna-knee-abnormality-detection',
                           inputs/'rsna-knee-abnormality-detection'], ['train.csv', 'train_series.csv', 'sample_submission.csv'])
        result = run_audit(data, private['original'], '/kaggle/working/sealed_identity',
                           stage={stage!r}, pilot=private['pilot'], started=_audit_started)
except Exception as exc:
    failure = Path('/kaggle/working/sealed_identity/failure-receipt.json')
    failure.parent.mkdir(parents=True, exist_ok=True)
    if not failure.exists():
        failure.write_text(json.dumps({{'status': 'FAIL', 'stage': {stage!r}, 'error_code': type(exc).__name__,
                                      'elapsed_seconds': time.monotonic() - _audit_started}}))
    print(json.dumps({{'event': 'identity_stage_failed', 'error_type': type(exc).__name__}}), flush=True)
    raise RuntimeError('Identity audit failed; see aggregate receipt') from None
print(json.dumps(result, sort_keys=True), flush=True)
'''
    compile(code, 'sealed-identity-notebook', 'exec')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'cells': [
        {'cell_type': 'markdown', 'id': 'scope', 'metadata': {}, 'source': [
            '# Private global identity audit\n',
            'Header-only, no pixels or target values in selection; immutable original assignments. '
            'Private outputs contain pseudonymous study/series identities and must remain private.\n']},
        {'cell_type': 'code', 'id': 'audit', 'metadata': {}, 'execution_count': None,
         'outputs': [], 'source': code.splitlines(True)}]}
    metadata = {'id': 'ozgung/' + slug, 'title': slug.replace('-', ' ').title(), 'code_file': 'notebook.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False,
        'enable_tpu': False, 'enable_internet': False, 'competition_sources': ['rsna-knee-abnormality-detection'],
        'dataset_sources': [], 'kernel_sources': [], 'model_sources': [], 'docker_image': DOCKER}
    notebook_bytes = json.dumps(notebook, indent=2).encode()
    require(max(len(code.encode()), len(notebook_bytes)) <= KAGGLE_SOURCE_BYTES,
            'kaggle_source_size_exceeded')
    build_contract = {'stage': stage, 'runtime_contract': contract,
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'stage_seconds': stage_seconds,
        'embedded_private_sha256': hashlib.sha256(private.encode()).hexdigest(),
        'notebook_sha256': hashlib.sha256(notebook_bytes).hexdigest(),
        'notebook_bytes': len(notebook_bytes), 'joined_code_bytes': len(code.encode()),
        'pilot_receipt_fingerprint': pilot['receipt']['fingerprint'] if pilot else None}
    destination = Path(destination)
    require(not destination.exists() or not any(destination.iterdir()), 'build_destination_not_empty')
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in [('notebook.ipynb', notebook), ('kernel-metadata.json', metadata),
                        ('build-contract.json', build_contract)]:
        (destination / name).write_text(json.dumps(value, indent=2))
    return build_contract


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data_dir', 'baseline_dir', 'plan_path', 'destination', 'slug'):
        parser.add_argument(name)
    parser.add_argument('--stage', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--pilot-dir')
    args = parser.parse_args()
    result = build(args.data_dir, args.baseline_dir, args.plan_path, args.destination, args.slug,
                   stage=args.stage, pilot_dir=args.pilot_dir)
    print(json.dumps(result, indent=2))
