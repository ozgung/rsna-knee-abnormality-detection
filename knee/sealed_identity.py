"""Header-only global identity audit and label-blind sealed cohort.

Only stdlib is imported at module load. The runtime additionally needs pydicom;
it never requests pixels, saves reports, or modifies the original assignments.
"""
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import platform
import re
import time
import warnings

from knee.fresh_reader_setup import progress, time_limit
from knee.validation import _report_key, file_sha256, read_csv, read_schema

POLICY = 'global-series-patient-sealed-v1'
SALT = 'fresh-sealed-v1:20261008'
PILOT_STUDIES = 64
SEALED_STUDIES = 300
PILOT_SECONDS = 600
FULL_SECONDS = 3600
OUTPUT_BYTES = 32 * 1024**2
METADATA = ('train.csv', 'train_series.csv', 'sample_submission.csv')
LIMITATIONS = ('One lexicographically first DICOM header per metadata series; within-series identity '
               'consistency is not established. Anonymized PatientID may not identify repeat patients. '
               'Exact normalized reports do not detect paraphrases. No independent-label claim is made.')


class AuditFailure(ValueError):
    """Messages are fixed aggregate codes, never data-derived values."""


def require(condition, code):
    if not condition:
        raise AuditFailure(code)


def fingerprint(value):
    # Identical canonical JSON convention to the existing fresh plan.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def stamped(value):
    return value | {'fingerprint': fingerprint(value)}


def validate_stamp(value):
    require(value.get('fingerprint') == fingerprint({k: v for k, v in value.items() if k != 'fingerprint'}),
            'private_artifact_fingerprint_mismatch')


def patient_hash(value):
    # Exact strip/prefix/hash convention used by knee.dicom and fresh_reader.
    patient = str(value).strip()
    require(bool(patient), 'missing_patient_identity')
    return hashlib.sha256(('knee-patient-v1:' + patient).encode()).hexdigest()


def report_group(row):
    # Reuse the dependency-free normalization shared with validation; identical
    # to experiment.report_group, without importing its ML dependencies.
    return hashlib.sha256((_report_key(row['Report']) or row['StudyInstanceUID']).encode()).hexdigest()


def safe_id(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) is not None


def digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def load_metadata(data_dir, original):
    """Gold membership is exclusion-only; no target values enter cohort ordering."""
    data_dir = Path(data_dir)
    validate_stamp(original)
    require(original.get('policy') == POLICY, 'original_contract_policy_mismatch')
    require({name: file_sha256(data_dir / name) for name in METADATA} == original['metadata_sha256'],
            'competition_metadata_hash_mismatch')
    id_column, targets = read_schema(data_dir / 'sample_submission.csv')
    columns, rows = read_csv(data_dir / 'train.csv')
    require(id_column == 'StudyInstanceUID' and columns == [id_column, 'Report', *targets] and
            targets == original['targets'], 'study_target_schema_mismatch')
    ids = [row[id_column] for row in rows]
    require(len(ids) == original['study_count'] and len(set(ids)) == len(ids) and all(map(safe_id, ids)),
            'study_coverage_mismatch')
    groups = {row[id_column]: report_group(row) for row in rows}
    # No gold label values are retained or used. Only already-held-out row presence.
    gold = {row[id_column] for row in rows if any(row[t].strip() for t in targets)}
    series_columns, series = read_csv(data_dir / 'train_series.csv')
    require(series_columns == [id_column, 'SeriesInstanceUID', 'Fluid_Sensitive', 'Fat_Suppression',
                               'Anatomical_Plane'], 'series_schema_mismatch')
    keys = [(row[id_column], row['SeriesInstanceUID']) for row in series]
    require(len(keys) == original['series_count'] and len(set(keys)) == len(keys) and
            len({s for _, s in keys}) == len(keys) and all(safe_id(u) and safe_id(s) for u, s in keys) and
            {u for u, _ in keys} == set(ids), 'series_coverage_mismatch')
    records = original['records']
    require(len({r['uid'] for r in records}) == len(records) and all(r['uid'] in groups for r in records),
            'original_id_coverage_mismatch')
    require(all(set(r) == {'uid', 'split', 'group', 'patient_hashes'} and r['group'] == groups[r['uid']] and
                r['split'] in ('train', 'weak_validation', 'gold') and isinstance(r['patient_hashes'], list) and
                r['patient_hashes'] and r['patient_hashes'] == sorted(set(r['patient_hashes'])) and
                all(map(digest, r['patient_hashes'])) for r in records), 'original_identity_contract_mismatch')
    require(gold == {r['uid'] for r in records if r['split'] == 'gold'}, 'gold_membership_mismatch')
    baseline = [{k: r[k] for k in ('uid', 'split', 'group')} for r in records if r['split'] != 'gold']
    require(fingerprint(baseline) == original['baseline_fingerprint'] and
            dict(Counter(r['split'] for r in records)) == original['original_counts'],
            'original_assignment_fingerprint_mismatch')
    return groups, gold, sorted(keys)


def pilot_ids(groups, original):
    selected = {r['uid'] for r in original['records'] if r['split'] != 'gold'}
    pools = {'original': selected, 'outside_original': set(groups) - selected}
    require(all(len(pool) >= PILOT_STUDIES // 2 for pool in pools.values()), 'pilot_stratum_too_small')
    return {name: sorted(pool, key=lambda uid: hashlib.sha256(
        (SALT + ':pilot:' + name + ':' + uid).encode()).hexdigest())[:PILOT_STUDIES // 2]
        for name, pool in pools.items()}


def runtime_contract(original):
    from knee import fresh_reader_setup, validation
    return stamped({'policy': POLICY, 'salt': SALT, 'sealed_target': SEALED_STUDIES,
        'pilot_studies': PILOT_STUDIES, 'pilot_seconds': PILOT_SECONDS, 'full_seconds': FULL_SECONDS,
        'output_bytes': OUTPUT_BYTES, 'original_fingerprint': original['fingerprint'],
        'sources': {name: file_sha256(path) for name, path in
            [('sealed_identity.py', __file__), ('validation.py', validation.__file__),
             ('fresh_reader_setup.py', fresh_reader_setup.__file__)]}})


def projection(measured_series, total_series, header_seconds, overhead_seconds):
    require(0 < measured_series <= total_series and all(math.isfinite(x) and x >= 0 for x in
            (header_seconds, overhead_seconds)) and header_seconds > 0, 'invalid_pilot_timing')
    estimate = overhead_seconds + 1.5 * header_seconds * total_series / measured_series
    return {'passed': estimate <= FULL_SECONDS, 'estimated_seconds_with_margin': estimate,
            'seconds_limit': FULL_SECONDS, 'safety_factor': 1.5,
            'measured_series': measured_series, 'target_series': total_series,
            'header_listing_seconds': header_seconds, 'nonheader_overhead_seconds': overhead_seconds}


def validate_identity(identity, contract, expected_keys):
    validate_stamp(identity)
    require(identity.get('contract_fingerprint') == contract['fingerprint'], 'header_contract_mismatch')
    records = identity['series']
    require([(r['uid'], r['series_uid']) for r in records] == sorted(expected_keys),
            'header_series_coverage_mismatch')
    require(all(set(r) == {'uid', 'series_uid', 'representative', 'patient_hash'} and
                safe_id(r['representative']) and r['representative'].lower().endswith('.dcm') and
                digest(r['patient_hash']) for r in records), 'header_identity_schema_mismatch')


def validate_pilot(bundle, contract, groups, original, keys):
    require(set(bundle) == {'identity', 'receipt'}, 'pilot_bundle_schema_mismatch')
    receipt, identity = bundle['receipt'], bundle['identity']
    validate_stamp(receipt)
    strata = pilot_ids(groups, original)
    chosen = {uid for values in strata.values() for uid in values}
    expected = [key for key in keys if key[0] in chosen]
    expected_strata = {name: {'studies': len(members), 'series': sum(u in set(members) for u, _ in expected)}
                       for name, members in strata.items()}
    validate_identity(identity, contract, expected)
    require(receipt['status'] == 'PASS' and receipt['stage'] == 'pilot' and
            receipt['contract_fingerprint'] == contract['fingerprint'] and
            receipt['identity_fingerprint'] == identity['fingerprint'] and
            receipt['identity_sha256'] == hashlib.sha256(json_bytes(identity)).hexdigest() and
            receipt['studies'] == PILOT_STUDIES and receipt['series'] == len(expected) and
            receipt['strata'] == expected_strata and receipt['read_series'] == len(expected) and
            receipt['reused_series'] == 0 and 0 < receipt['elapsed_seconds'] <= PILOT_SECONDS and
            receipt['gate'] == projection(
                len(expected), len(keys), receipt['header_listing_seconds'], receipt['nonheader_overhead_seconds']) and
            receipt['gate']['passed'], 'pilot_qualification_failed')
    return identity['series']


def read_headers(data_dir, keys, *, counts, cached=(), deadline_at=None):
    import pydicom
    cached_by_key = {(r['uid'], r['series_uid']): r for r in cached}
    result = []
    for uid, series_uid in keys:
        if deadline_at is not None and time.monotonic() >= deadline_at:
            raise TimeoutError('Header audit deadline exceeded')
        if (uid, series_uid) in cached_by_key:
            result.append(cached_by_key[(uid, series_uid)])
            counts['reused_series'] += 1
            continue
        tick = time.monotonic()
        folder = Path(data_dir) / 'train_series' / uid / series_uid
        # Listing is bounded to this exact metadata series; never traverse pixels.
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.dcm')
        require(bool(files), 'missing_series_header')
        # Pydicom can warn/log raw field values before raising. Keep only an
        # aggregate count and restore its logger even on resource/deadline errors.
        logger = logging.getLogger('pydicom')
        disabled = logger.disabled
        try:
            logger.disabled = True
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always')
                header = pydicom.dcmread(files[0], stop_before_pixels=True, specific_tags=['PatientID'])
                patient = header.get('PatientID', '')
                require(isinstance(patient, str), 'invalid_patient_identity_type')
                hashed = patient_hash(patient)
                counts['parser_warnings'] = counts.get('parser_warnings', 0) + len(notices)
        finally:
            logger.disabled = disabled
        if deadline_at is not None and time.monotonic() >= deadline_at:
            raise TimeoutError('Header audit deadline exceeded')
        result.append({'uid': uid, 'series_uid': series_uid, 'representative': files[0].name,
                       'patient_hash': hashed})
        counts['header_listing_seconds'] += time.monotonic() - tick
        counts['read_series'] += 1
        if counts['read_series'] % 128 == 0:
            progress('identity_headers_progress', read_series=counts['read_series'],
                     reused_series=counts['reused_series'], total_series=len(keys),
                     header_listing_seconds=counts['header_listing_seconds'])
    return result


def make_manifest(groups, gold, original, series, contract, *, target=SEALED_STUDIES):
    """Whole transitive components, ordered by salted canonical membership hash."""
    identities = defaultdict(set)
    for row in series:
        identities[row['uid']].add(row['patient_hash'])
    for row in original['records']:
        identities[row['uid']].update(row['patient_hashes'])
    require(set(identities) == set(groups) and all(identities.values()), 'global_identity_coverage_mismatch')
    parent = {uid: uid for uid in groups}
    def find(uid):
        while parent[uid] != uid:
            parent[uid] = parent[parent[uid]]
            uid = parent[uid]
        return uid
    seen = {}
    for uid in sorted(groups):
        for key in [('report', groups[uid]), *(('patient', p) for p in sorted(identities[uid]))]:
            if key in seen:
                parent[find(uid)] = find(seen[key])
            else:
                seen[key] = uid
    components = defaultdict(list)
    for uid in sorted(groups):
        components[find(uid)].append(uid)
    original_roles = {r['uid']: r['split'] for r in original['records']}
    original_weak = {u for u, role in original_roles.items() if role != 'gold'}
    dev = {u for u, role in original_roles.items() if role == 'weak_validation'}
    eligible = [members for members in components.values() if not (set(members) & (gold | original_weak))]
    eligible.sort(key=lambda members: (hashlib.sha256((SALT + ':' + fingerprint(members)).encode()).hexdigest(), members))
    require(sum(map(len, eligible)) >= target and target > 0, 'insufficient_sealable_studies')
    selected = []
    for members in eligible:
        selected.extend(members)
        if len(selected) >= target:
            break
    sealed = set(selected)
    records = []
    for members in sorted(components.values()):
        flags = {'gold': bool(set(members) & gold), 'development': bool(set(members) & dev),
                 'sealed': bool(set(members) & sealed)}
        component = fingerprint(members)
        for uid in members:
            blocked = [name for name, touches in flags.items() if touches]
            role = ('gold' if uid in gold else 'development' if uid in dev else
                    'sealed' if uid in sealed else 'quarantined' if blocked else 'training_candidate')
            records.append({'uid': uid, 'report_group': groups[uid], 'patient_hashes': sorted(identities[uid]),
                            'component': component, 'original_split': original_roles.get(uid),
                            'future_role': role, 'quarantine_reasons': blocked if role == 'quarantined' else []})
    return stamped({'policy': POLICY, 'salt': SALT, 'contract_fingerprint': contract['fingerprint'],
                    'original_fingerprint': original['fingerprint'], 'baseline_fingerprint': original['baseline_fingerprint'],
                    'target_studies': target, 'sealed_order': selected, 'sealed_studies': len(selected),
                    'overshoot': len(selected) - target, 'eligible_studies': sum(map(len, eligible)),
                    'eligible_components': len(eligible), 'components': len(components),
                    'records': sorted(records, key=lambda r: r['uid']), 'limitations': LIMITATIONS})


def validate_manifest(manifest, groups, gold, original, series, contract):
    validate_stamp(manifest)
    require(manifest == make_manifest(groups, gold, original, series, contract), 'sealed_manifest_mismatch')


def json_bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def write_outputs(output_dir, values):
    encoded = {name: json_bytes(value) for name, value in values.items()}
    require(sum(map(len, encoded.values())) <= OUTPUT_BYTES, 'private_output_budget_exceeded')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Never replace a previously sealed cohort with a different one.
    path = output_dir / 'sealed-manifest-private.json'
    if path.exists() and path.name in encoded:
        require(path.read_bytes() == encoded[path.name], 'existing_sealed_manifest_differs')
    require(not any(p.name not in encoded for p in output_dir.iterdir()), 'output_directory_not_fresh')
    for name, payload in encoded.items():
        path = output_dir / name
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(payload)
        temporary.replace(path)


def run_audit(data_dir, original, output_dir, *, stage='pilot', pilot=None, started=None):
    started = time.monotonic() if started is None else started
    counts = {'read_series': 0, 'reused_series': 0, 'header_listing_seconds': 0., 'parser_warnings': 0}
    receipt = {'status': 'FAIL', 'stage': stage, 'policy': POLICY, 'limitations': LIMITATIONS}
    owns_output = False
    try:
        require(stage in ('pilot', 'full') and ((stage == 'full') == (pilot is not None)), 'stage_contract_mismatch')
        output_dir = Path(output_dir)
        require(not output_dir.exists() or not any(output_dir.iterdir()), 'output_directory_not_fresh')
        owns_output = True
        seconds = PILOT_SECONDS if stage == 'pilot' else FULL_SECONDS
        with time_limit(seconds - (time.monotonic() - started), 'global identity audit'):
            groups, gold, keys = load_metadata(data_dir, original)
            contract = runtime_contract(original)
            strata = pilot_ids(groups, original)
            chosen = {u for members in strata.values() for u in members}
            selected_keys = [key for key in keys if key[0] in chosen] if stage == 'pilot' else keys
            cached = validate_pilot(pilot, contract, groups, original, keys) if pilot else ()
            receipt.update(contract_fingerprint=contract['fingerprint'], startup_seconds=time.monotonic() - started,
                           environment={'python': platform.python_version(), 'pydicom': importlib.metadata.version('pydicom')},
                           original_counts=original['original_counts'], original_fingerprint=original['fingerprint'])
            progress('identity_headers_begin', stage=stage, studies=len({u for u, _ in selected_keys}),
                     series=len(selected_keys), startup_seconds=receipt['startup_seconds'])
            series = read_headers(data_dir, selected_keys, counts=counts, cached=cached, deadline_at=started + seconds)
            identity = stamped({'contract_fingerprint': contract['fingerprint'], 'series': series})
            validate_identity(identity, contract, selected_keys)
            values = {'identity-private.json': identity}
            if stage == 'full':
                manifest = make_manifest(groups, gold, original, series, contract)
                validate_manifest(manifest, groups, gold, original, series, contract)
                values['sealed-manifest-private.json'] = manifest
                receipt.update(sealed_studies=manifest['sealed_studies'], overshoot=manifest['overshoot'],
                               component_count=manifest['components'], eligible_studies=manifest['eligible_studies'],
                               future_roles=dict(Counter(r['future_role'] for r in manifest['records'])),
                               quarantined_original_train=sum(r['original_split'] == 'train' and r['future_role'] ==
                                                              'quarantined' for r in manifest['records']),
                               studies_with_multiple_patient_hashes=sum(len(r['patient_hashes']) > 1 for r in manifest['records']),
                               sealed_manifest_fingerprint=manifest['fingerprint'],
                               reused_parser_warnings=pilot['receipt']['parser_warnings'],
                               pilot_receipt_fingerprint=pilot['receipt']['fingerprint'])
            elapsed = time.monotonic() - started
            receipt.update(status='PASS', studies=len({u for u, _ in selected_keys}), series=len(selected_keys),
                           **counts, elapsed_seconds=elapsed,
                           nonheader_overhead_seconds=elapsed - counts['header_listing_seconds'],
                           identity_fingerprint=identity['fingerprint'],
                           identity_sha256=hashlib.sha256(json_bytes(identity)).hexdigest(),
                           private_output_bytes=sum(len(json_bytes(v)) for v in values.values()))
            if stage == 'pilot':
                receipt['strata'] = {name: {'studies': len(members),
                    'series': sum(u in set(members) for u, _ in selected_keys)} for name, members in strata.items()}
                receipt['gate'] = projection(len(series), len(keys), counts['header_listing_seconds'],
                                              receipt['nonheader_overhead_seconds'])
            receipt = stamped(receipt)
            values['receipt.json'] = receipt
            write_outputs(output_dir, values)
            progress('identity_audit_complete', stage=stage, studies=receipt['studies'], series=receipt['series'],
                     elapsed_seconds=receipt['elapsed_seconds'])
            return receipt
    except Exception as exc:
        # No raw exception text/tracebacks: third-party errors can contain paths/UIDs.
        failure = {'status': 'FAIL', 'stage': stage, 'policy': POLICY,
                   'error_code': str(exc) if isinstance(exc, AuditFailure) else type(exc).__name__,
                   **counts, 'elapsed_seconds': time.monotonic() - started}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if owns_output:
            for name in ('identity-private.json', 'sealed-manifest-private.json', 'receipt.json'):
                (output_dir / name).unlink(missing_ok=True)
                (output_dir / name).with_suffix('.tmp').unlink(missing_ok=True)
        (output_dir / 'failure-receipt.json').write_bytes(json_bytes(failure))
        progress('identity_audit_failed', **failure)
        raise AuditFailure('identity_audit_failed_see_aggregate_receipt') from None
