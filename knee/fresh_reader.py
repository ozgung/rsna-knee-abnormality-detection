"""Bounded fresh-ImageNet 2.5D pilot. Heavy dependencies are imported on demand.

External model/preprocessing code is supplied separately, hash checked and used
under goodpjw2008's Apache-2.0 dataset license. No public knee weights are loaded.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from knee.validation import file_sha256, make_gold_holdout, read_csv, read_schema

SOURCE_HASHES = {
    'preprocess.py': '4491963e4d2156625c9cf79051f4a25c32d2d761f1f07df3117a439ed8029ae2',
    'knee.py': '8e9fd05dd94e9be70d4b4a6cbdf5201c219b5e6eb3d57c07bb8992c8dbe72ae7',
}
PREPROCESS_VERSION = 'fresh-reader-canonical384-resize224-k4-v1'
BACKBONE = 'convnext_tiny.fb_in1k'
WEAK_LABELS_SHA256 = '96444cd3c7d396bae4a71454317d9cbe0689759b5278b30939781e04d0774537'
CONFIG = {'seed': 42, 'image_size': 224, 'windows': 4, 'train_limit': 512,
          'validation_limit': 200, 'baseline_train_limit': 2000, 'baseline_validation_limit': 300,
          'pilot_studies': 64, 'pilot_seconds': 1800, 'cache_seconds': 3600,
          'cache_bytes': 2 * 1024**3, 'disk_reserve_bytes': 1024**3,
          'training_seconds': 5400, 'max_epochs': 4, 'batch_size': 2,
          'lr_encoder': 1e-4, 'lr_head': 5e-4, 'weight_decay': .02, 'ema': .998}
DURATION_PROFILE = 'same512-duration12-v1'
DURATION_PLAN = '1bd346c05c0aac20f54b82b790e356d3d6c1b95606accaa4a5d1da7e3e421942'
INITIAL_ENCODER_SHA256 = 'b1d4d0986c66bb1d02eddf7bf4bceea7fde12aced5def666c28d3b89951b569e'
TRAINING_SOURCE_FILES = ('knee/__init__.py', 'knee/validation.py', 'knee/features.py', 'knee/experiment.py',
                        'knee/fresh_reader.py', 'knee/fresh_reader_setup.py', 'knee/fresh_reader_probe.py',
                        'scripts/build_fresh_reader.py')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_plan(plan):
    actual = fingerprint({k: v for k, v in plan.items() if k != 'fingerprint'})
    if plan.get('fingerprint') != actual:
        raise ValueError('Plan contents do not match their fingerprint')
    return actual


def duration_experiment(plan_fingerprint, config, *, runner_sha256=None, own_sources=None):
    """One named duration comparison; never rewrite the immutable cache plan."""
    if plan_fingerprint != DURATION_PLAN or config != CONFIG:
        raise ValueError('Duration experiment requires the exact original512/200 cache plan/config')
    runner_sha256 = file_sha256(__file__) if runner_sha256 is None else runner_sha256
    if not isinstance(runner_sha256, str) or len(runner_sha256) != 64 or any(c not in '0123456789abcdef' for c in runner_sha256):
        raise ValueError('Invalid experiment runner hash')
    own_sources = ({name: file_sha256(Path(__file__).resolve().parents[1] / name) for name in TRAINING_SOURCE_FILES}
                   if own_sources is None else own_sources)
    if (set(own_sources) != set(TRAINING_SOURCE_FILES) or own_sources['knee/fresh_reader.py'] != runner_sha256 or
            any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                for v in own_sources.values())):
        raise ValueError('Invalid experiment build/runtime source hashes')
    contract = {'profile': DURATION_PROFILE, 'plan_fingerprint': plan_fingerprint,
        'config': config | {'max_epochs': 12}, 'runner_sha256': runner_sha256,
        'source_hashes': SOURCE_HASHES, 'own_sources': own_sources, 'backbone': BACKBONE,
        'initial_encoder_sha256': INITIAL_ENCODER_SHA256,
        'initialization': 'fresh generic ImageNet; new optimizer, EMA and12epoch cosine; no warm start',
        'selection': 'minimum confidence-weighted macro weak-validation BCE; EMA; no gold',
        'gate': {'warmup_batches': 16, 'timed_batches': 48, 'batch_size': 2,
                 'warmup_backward_batches': 16, 'timed_backward_batches': 48,
                 'optimizer_updates': 0, 'safety_factor': 1.5, 'checkpoint_reserve_seconds': 180,
                 'require_all_epochs': 12, 'seconds_limit': 5400}}
    return contract | {'fingerprint': fingerprint(contract)}


def validate_experiment(plan, experiment, *, current_runner=True):
    if experiment is None:
        return
    expected = duration_experiment(plan['fingerprint'], plan['config'],
        runner_sha256=None if current_runner else experiment.get('runner_sha256'),
        own_sources=None if current_runner else experiment.get('own_sources'))
    if experiment != expected:
        raise ValueError('Named training experiment differs from its immutable contract')


def duration_gate_budget(train_seconds, validation_seconds, diagnostic_seconds, elapsed_seconds):
    values = (train_seconds, validation_seconds, diagnostic_seconds, elapsed_seconds)
    if any(not math.isfinite(x) or x < 0 for x in values) or min(values[:3]) <= 0:
        raise ValueError('Invalid duration timing measurements')
    epoch_seconds = (256 * train_seconds + 100 * validation_seconds + 32 * diagnostic_seconds) * 1.5
    remaining = 5400 - elapsed_seconds - 180
    return {'passed': 12 * epoch_seconds <= remaining, 'planned_epochs': 12,
            'estimated_epoch_seconds_with_margin': epoch_seconds,
            'estimated_training_seconds_with_margin': 12 * epoch_seconds,
            'elapsed_before_fit_seconds': elapsed_seconds, 'remaining_after_checkpoint_reserve_seconds': remaining,
            'checkpoint_reserve_seconds': 180, 'safety_factor': 1.5}


def duration_timing_gate(model, ema, optimizer, scaler, batch, validate, tr, va, diagnostic_indices, config, started):
    """Same first64 no-update batches:16 warmup,48 steady; all time is charged."""
    import torch
    if (len(tr), len(va), len(diagnostic_indices), config['batch_size']) != (512, 200, 64, 2):
        raise ValueError('Duration gate requires the exact512/200/train64 cohort and batch2')
    before = rng_state()
    warmup_started = time.monotonic()
    measured = 0
    warmup_backward = timed_backward = 0
    try:
        model.train()
        for batch_number in range(64):
            if time.monotonic() - started > config['training_seconds'] - 180:
                raise TimeoutError('Duration timing gate exhausted the training budget')
            if batch_number == 16:
                torch.cuda.synchronize()
                timed_started = time.monotonic()
                warmup_seconds = timed_started - warmup_started
            offset = batch_number * 2
            x, mask, pos, y = batch(tr[offset:offset + 2])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                numerator, denominator = loss_terms(model(x, mask, pos, res=config['image_size'], train=True), y)
            if not torch.isfinite(numerator).all():
                raise ValueError('Nonfinite duration pilot supervision')
            if (denominator > 0).any():
                scaler.scale(macro_loss(numerator, denominator)).backward()
                measured += 1
                if batch_number < 16:
                    warmup_backward += 1
                else:
                    timed_backward += 1
        torch.cuda.synchronize()
        timed_seconds = time.monotonic() - timed_started
        if (warmup_backward, timed_backward) != (16, 48):
            raise ValueError('Duration gate requires16warmup and48timed supervised backward batches')
        tick = time.monotonic()
        validate(ema, va[:16]); torch.cuda.synchronize()
        validation_seconds = (time.monotonic() - tick) / 8
        tick = time.monotonic()
        validate(ema, diagnostic_indices[:16]); torch.cuda.synchronize()
        diagnostic_seconds = (time.monotonic() - tick) / 8
    finally:
        optimizer.zero_grad(set_to_none=True)
        restore_rng(before)
    return duration_gate_budget(timed_seconds / 48, validation_seconds, diagnostic_seconds,
                                time.monotonic() - started) | {
        'warmup_batches': 16, 'timed_batches': 48, 'warmup_seconds': warmup_seconds,
        'warmup_backward_batches': warmup_backward, 'timed_backward_batches': timed_backward,
        'timed_seconds': timed_seconds, 'measured_training_steps': measured, 'optimizer_updates': 0,
        'train_batch_seconds': timed_seconds / 48, 'validation_batch_seconds': validation_seconds,
        'training_diagnostic_batch_seconds': diagnostic_seconds, 'training_diagnostic_studies': 64}


def validate_duration_resume(state, experiment):
    if (state.get('experiment') != experiment or state.get('experiment_fingerprint') != experiment['fingerprint'] or
            state.get('config') != experiment['config'] or state.get('planned_epochs') != 12 or
            state.get('initial_encoder_sha256') != INITIAL_ENCODER_SHA256 or
            not math.isfinite(state.get('best_loss', math.inf)) or
            not isinstance(state.get('best_evidence_fingerprint'), str) or
            len(state.get('best_evidence_fingerprint', '')) != 64 or
            not 0 <= state.get('epoch', -1) < 12 or not state.get('gate', {}).get('passed') or
            state.get('gate', {}).get('planned_epochs') != 12 or
            state.get('gate', {}).get('experiment_fingerprint') != experiment['fingerprint'] or
            state.get('scheduler', {}).get('T_max') != 12 * 256):
        raise ValueError('Duration resume requires its own incomplete12epoch experiment/checkpoint pair')


def remaining_training_budget(elapsed_before, started, *, reserve=180):
    if (isinstance(elapsed_before, bool) or not isinstance(elapsed_before, (int, float)) or
            not math.isfinite(elapsed_before) or elapsed_before < 0):
        raise ValueError('Invalid cumulative training elapsed time')
    remaining = 5400 - elapsed_before - (time.monotonic() - started) - reserve
    if remaining <= 0:
        raise TimeoutError('Cumulative training budget exhausted before preparation')
    return remaining


def duration_resume_report(resume, experiment, started):
    """Small receipt first: reject exhausted/mixed runs before loading tensors."""
    report = json.loads((Path(resume).parent / 'training-report.json').read_text())
    if (report.get('experiment') != experiment or report.get('experiment_fingerprint') != experiment['fingerprint'] or
            report.get('runner_sha256') != experiment['runner_sha256'] or
            report.get('plan_fingerprint') != experiment['plan_fingerprint']):
        raise ValueError('Duration resume training report differs')
    elapsed = report.get('elapsed_seconds')
    remaining_training_budget(elapsed, started)
    return elapsed


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def check_sources(source_dir):
    for name, digest in SOURCE_HASHES.items():
        if file_sha256(Path(source_dir) / name) != digest:
            raise ValueError(f'Unreviewed external source: {name}')


def load_source(source_dir, name):
    check_sources(source_dir)
    spec = importlib.util.spec_from_file_location('fresh_external_' + name, Path(source_dir) / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grouped_subset(records, role, limit):
    """Keep whole groups without reassigning any baseline validation study."""
    groups = {}
    for row in records:
        if row['split'] == role:
            groups.setdefault(row['group'], []).append(row)
    selected = []
    for members in groups.values():
        if len(selected) + len(members) <= limit:
            selected.extend(members)
    return selected


def pilot_order(records, count):
    # No target value enters the order.
    return sorted(records, key=lambda row: hashlib.sha256(('fresh-pilot-v1:' + row['uid']).encode()).hexdigest())[:count]


def budget_gate(count, total, seconds, byte_count, *, seconds_limit, bytes_limit, safety=1.5):
    if not 0 < count <= total or seconds <= 0 or byte_count < 0:
        raise ValueError('Invalid throughput measurement')
    estimate_seconds = seconds / count * total * safety
    estimate_bytes = byte_count / count * total * 1.25
    return {'passed': estimate_seconds <= seconds_limit and estimate_bytes <= bytes_limit,
            'measured_studies': count, 'target_studies': total,
            'estimated_seconds_with_margin': estimate_seconds,
            'estimated_bytes_with_margin': estimate_bytes,
            'seconds_limit': seconds_limit, 'bytes_limit': bytes_limit}


def cache_time_budget(stage, override=None, *, default=CONFIG['cache_seconds']):
    """An operational full-cache limit; never insert it into the data plan."""
    if override is not None:
        if stage != 'cache' or type(override) is not int or not 0 < override <= 7200:
            raise ValueError('cache-seconds requires stage cache and an integer in 1..7200')
        return override
    return default


def assess_cache_expansion(pilot_receipt, sample, total, *, seconds_limit, bytes_limit):
    """Reassess measured decode + nondecode work without changing the old gate."""
    if pilot_receipt.get('status') != 'PASS' or pilot_receipt.get('failures', 0) != 0:
        raise ValueError('Full cache requires a successful decoding pilot')
    overhead = pilot_receipt['elapsed_seconds'] - pilot_receipt['new_decode_seconds']
    decode = sum(row['seconds'] for row in sample)
    byte_count = sum(row['bytes'] for row in sample)
    if not all(math.isfinite(v) for v in (overhead, decode, byte_count)) or overhead < 0:
        raise ValueError('Invalid measured pilot timing or size')
    if bytes_limit > CONFIG['cache_bytes']:
        raise ValueError('Full cache retains the original 2 GiB ceiling')
    gate = budget_gate(len(sample), total, decode, byte_count,
                       seconds_limit=seconds_limit, bytes_limit=bytes_limit)
    gate['estimated_decode_seconds_with_margin'] = gate['estimated_seconds_with_margin']
    gate['pilot_nondecode_seconds'] = overhead
    gate['estimated_seconds_with_margin'] += overhead
    gate['passed'] = (gate['estimated_seconds_with_margin'] <= seconds_limit and
                      gate['estimated_bytes_with_margin'] <= bytes_limit)
    gate['original_gate_passed'] = pilot_receipt['gate']['passed']
    gate['basis'] = '1.5x full-split decode projection + measured pilot nondecode overhead; 1.25x bytes'
    return gate


def make_plan(data_dir, weak_path, config):
    """Use the existing baseline splitter, then subset its fixed assignments."""
    from knee.experiment import choose_studies, report_group
    data_dir = Path(data_dir)
    if file_sha256(weak_path) != WEAK_LABELS_SHA256:
        raise ValueError('Expected the original pinned Steven v1 weak-label table')
    id_column, targets = read_schema(data_dir / 'sample_submission.csv')
    columns, rows = read_csv(data_dir / 'train.csv')
    patient_columns = [c for c in columns if c.lower().replace('_', '').replace(' ', '')
                       in ('patientid', 'patientidentifier', 'subjectid', 'patientinstanceuid')]
    if len(patient_columns) > 1:
        raise ValueError('Ambiguous patient identity fields')
    # Official values are consulted only to identify/exclude trusted-label rows.
    assignments, audit = make_gold_holdout(rows, targets, id_column,
                                           patient_columns[0] if patient_columns else None)
    selected, splits = choose_studies(rows, assignments, config['baseline_train_limit'],
                                     config['baseline_validation_limit'], config['seed'])
    baseline = [{'uid': row[id_column], 'split': str(role), 'group': report_group(row)}
                for row, role in zip(selected, splits) if role != 'gold']
    records = (grouped_subset(baseline, 'train', config['train_limit']) +
               grouped_subset(baseline, 'weak_validation', config['validation_limit']))
    if not all(any(r['split'] == split for r in records) for split in ('train', 'weak_validation')):
        raise ValueError('Both training and validation partitions are required')
    _, weak_rows = read_csv(weak_path)
    weak = {row[id_column]: row for row in weak_rows}
    if len(weak) != len(weak_rows):
        raise ValueError('Duplicate weak-label IDs')
    for record in records:
        record['labels'] = [float(weak[record['uid']][target]) for target in targets]
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in record['labels']):
            raise ValueError('Weak labels must be finite probabilities')
    return {'targets': targets, 'records': records,
            'gold_ids': [row[id_column] for row in rows if assignments[row[id_column]] == 'validation'],
            'baseline_fingerprint': fingerprint(baseline), 'gold_exclusion_audit': audit,
            'config': config, 'source_hashes': SOURCE_HASHES, 'preprocessing': PREPROCESS_VERSION,
            'files': {name: file_sha256(data_dir / name) for name in
                      ('train.csv', 'train_series.csv', 'sample_submission.csv')},
            'weak_labels_sha256': file_sha256(weak_path)}


def environment():
    import platform
    import torch
    versions = {}
    for name in ('numpy', 'pandas', 'torch', 'timm', 'pydicom', 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'versions': versions,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'cuda_devices': [{'name': torch.cuda.get_device_name(i),
                              'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                             for i in range(torch.cuda.device_count())],
            'code_revision': os.environ.get('KNEE_CODE_REVISION', 'unversioned'),
            'source_sha256': os.environ.get('KNEE_SOURCE_SHA256', 'unversioned')}


def patient_exclusions(plan, data_dir, table, *, deadline_at=None):
    """Header-only identity check includes gold, without reading its pixels/labels.

    One header per selected candidate series is checked; anonymization and
    within-series identity consistency remain limitations, recorded in the audit.
    """
    import pydicom
    from knee.experiment import exclude_related
    records = plan['records']
    ids = [r['uid'] for r in records] + plan['gold_ids']
    patient_hashes = []
    for uid in ids:
        if deadline_at is not None and time.monotonic() > deadline_at:
            raise TimeoutError('CPU header-audit budget exhausted')
        identities = set()
        for candidates in table.get(uid, [[] for _ in range(6)]):
            if not candidates:
                continue
            folder = Path(data_dir) / 'train_series' / uid / candidates[0]
            files = sorted(p for p in folder.iterdir() if p.is_file())
            if not files:
                continue
            header = pydicom.dcmread(files[0], stop_before_pixels=True, specific_tags=['PatientID'])
            patient = str(header.get('PatientID', '')).strip()
            if patient:
                identities.add(hashlib.sha256(('knee-patient-v1:' + patient).encode()).hexdigest())
        patient_hashes.append(sorted(identities))
    roles = [r['split'] for r in records] + ['gold'] * len(plan['gold_ids'])
    groups = [r['group'] for r in records] + ['gold:' + uid for uid in plan['gold_ids']]
    filtered = exclude_related(roles, groups, patient_hashes)
    retained = [r for r, role in zip(records, filtered) if role != 'excluded']
    plan['identity_audit'] = {'checked_studies': len(ids), 'with_patient_id': sum(bool(p) for p in patient_hashes),
                             'excluded_weak': len(records) - len(retained),
                             'limitation': 'First header of each selected series; anonymized PatientID may not identify repeat patients.'}
    plan['records'] = retained
    return plan


def prepare_cache(data_dir, weak_path, source_dir, output_dir, *, pilot=True, config=None, cache_seconds=None):
    import cv2
    import numpy as np
    import pandas as pd
    config = dict(CONFIG if config is None else config)
    effective_cache_seconds = cache_time_budget('cache-pilot' if pilot else 'cache', cache_seconds, default=config['cache_seconds'])
    started = time.monotonic()
    deadline = config['pilot_seconds'] if pilot else effective_cache_seconds
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / 'arrays'
    cache_dir.mkdir(exist_ok=True)
    prep, model = load_source(source_dir, 'preprocess'), load_source(source_dir, 'knee')
    plan = make_plan(data_dir, weak_path, config)
    if plan['targets'] != model.LABELS:
        raise ValueError('External target order differs from official schema')
    se = pd.read_csv(data_dir / 'train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    wanted = {r['uid'] for r in plan['records']} | set(plan['gold_ids'])
    se = se[se.StudyInstanceUID.isin(wanted)]
    counts = {}
    for row in se.itertuples():
        if time.monotonic() - started > deadline:
            raise TimeoutError('CPU metadata-scan budget exhausted')
        folder = data_dir / 'train_series' / row.StudyInstanceUID / row.SeriesInstanceUID
        counts[row.SeriesInstanceUID] = min(sum(p.is_file() for p in folder.iterdir()), prep.MAX_SLICES) if folder.is_dir() else 0
    table = model.build_study_table(se, counts)
    plan = patient_exclusions(plan, data_dir, table, deadline_at=started + deadline)
    plan['fingerprint'] = fingerprint(plan)
    plan_path = output_dir / 'plan-private.json'
    if plan_path.exists() and json.loads(plan_path.read_text())['fingerprint'] != plan['fingerprint']:
        raise ValueError('Existing cache plan differs; use a separate output directory')
    write_json(plan_path, plan)
    records = plan['records']
    selected = pilot_order(records, config['pilot_studies']) if pilot else records
    index_path = output_dir / 'cache-index-private.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    revised_gate = None
    if not pilot:
        pilot_path = output_dir / 'cache-pilot.json'
        pilot_receipt = json.loads(pilot_path.read_text())
        if pilot_receipt['plan_fingerprint'] != plan['fingerprint']:
            raise ValueError('Full cache requires a pilot for this exact plan')
        sample = []
        for record in pilot_order(records, config['pilot_studies']):
            if record['uid'] not in index:
                raise ValueError('Pilot cache is incomplete')
            item = index[record['uid']]
            path = cache_dir / item['file']
            if path.stat().st_size != item['bytes'] or file_sha256(path) != item['sha256']:
                raise ValueError('Cached pilot array changed')
            sample.append(item)
        revised_gate = assess_cache_expansion(pilot_receipt, sample, len(records),
            seconds_limit=effective_cache_seconds, bytes_limit=config['cache_bytes'])
        write_json(output_dir / 'cache-revised-gate.json', {
            'plan_fingerprint': plan['fingerprint'], 'original_pilot_sha256': file_sha256(pilot_path),
            'original_gate': pilot_receipt['gate'], 'cache_seconds_override': cache_seconds,
            'effective_cache_seconds': effective_cache_seconds, 'gate': revised_gate})
        if not pilot_receipt['gate']['passed'] and cache_seconds is None:
            raise ValueError('Failed original time gate requires an explicit full-cache budget override')
        if not revised_gate['passed']:
            raise ValueError('Full cache fails the revised measured time/disk gate')
    new_seconds, new_count, new_bytes = 0., 0, 0
    audit = {'status': 'RUNNING', 'plan_fingerprint': plan['fingerprint'], 'environment': environment(),
             'selected_studies': len(selected), 'selected_series': 0, 'missing_slots': 0, 'failures': 0,
             'effective_cache_seconds': effective_cache_seconds, 'cache_seconds_override': cache_seconds}
    report_path = output_dir / ('cache-pilot.json' if pilot else 'cache-complete.json')
    try:
        for record in selected:
            if time.monotonic() - started > deadline:
                raise TimeoutError('CPU preprocessing wall-clock budget exhausted')
            uid = record['uid']
            if uid in index:
                if file_sha256(cache_dir / index[uid]['file']) != index[uid]['sha256']:
                    raise ValueError('Cached array changed')
                continue
            if shutil.disk_usage(output_dir).free < config['disk_reserve_bytes'] + 10 * 1024**2:
                raise RuntimeError('Disk reserve reached')
            tick = time.monotonic()
            k, res = config['windows'], config['image_size']
            x = np.zeros((6, k, 3, res, res), np.uint8)
            mask, pos = np.zeros(6, bool), np.zeros((6, k), np.float32)
            for slot, candidates in enumerate(table.get(uid, [[] for _ in range(6)])):
                if not candidates:
                    audit['missing_slots'] += 1
                    continue
                volume, info = prep.load_series(str(data_dir / 'train_series' / uid / candidates[0]),
                                               model.PLANES[slot % 3])
                if info['n_bad'] or len(volume) < 3 or not np.any(volume):
                    raise ValueError('Selected series failed strict decoding or is empty')
                centers = model.window_centres(len(volume), k, False)
                for j, center in enumerate(centers):
                    for channel, z in enumerate((center - 1, center, center + 1)):
                        x[slot, j, channel] = cv2.resize(volume[z], (res, res), interpolation=cv2.INTER_AREA)
                mask[slot], pos[slot] = True, centers / (len(volume) - 1)
                audit['selected_series'] += 1
            if not mask.any():
                raise ValueError('No valid series in study')
            filename = hashlib.sha256(uid.encode()).hexdigest() + '.npz'
            path = cache_dir / filename
            tmp = path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, x=x, mask=mask, pos=pos)
            byte_count = tmp.stat().st_size
            if sum(v['bytes'] for v in index.values()) + byte_count > config['cache_bytes']:
                tmp.unlink()
                raise RuntimeError('Cache byte budget exceeded')
            os.replace(tmp, path)
            duration = time.monotonic() - tick
            index[uid] = {'file': filename, 'bytes': byte_count, 'sha256': file_sha256(path), 'seconds': duration}
            write_json(index_path, index)
            new_seconds += duration; new_count += 1; new_bytes += byte_count
            if len(index) % 8 == 0:
                print(json.dumps({'event': 'cache_progress', 'completed': len(index), 'seconds': time.monotonic() - started}), flush=True)
        sample = [index[r['uid']] for r in pilot_order(records, config['pilot_studies'])]
        audit['gate'] = revised_gate or budget_gate(len(sample), len(records), sum(s['seconds'] for s in sample),
                                   sum(s['bytes'] for s in sample), seconds_limit=config['cache_seconds'],
                                   bytes_limit=config['cache_bytes'])
        audit.update(status='PASS', cached_studies=len(index), cache_bytes=sum(v['bytes'] for v in index.values()),
                     new_studies=new_count, new_decode_seconds=new_seconds, new_bytes=new_bytes)
        if not pilot and any(r['uid'] not in index for r in records):
            raise ValueError('Incomplete training cache')
    except Exception as exc:
        audit.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        audit['elapsed_seconds'] = time.monotonic() - started
        write_json(report_path, audit)
    return audit


def loss_terms(logits, labels):
    import torch
    weights = (labels - .5).abs() * 2
    numerator = (torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction='none') * weights).sum(0)
    return numerator, weights.sum(0)


def macro_loss(numerator, denominator):
    valid = denominator > 0
    if not valid.any():
        raise ValueError('No supervised target')
    return (numerator[valid] / denominator[valid]).mean()


def per_target_bce(numerator, denominator):
    return [float(n / d) if d > 0 else None for n, d in zip(numerator, denominator)]


def training_prior(labels):
    import numpy as np
    y = np.asarray(labels, dtype=np.float32)
    weight = np.abs(y - .5) * 2
    support = weight.sum(0)
    if (support <= 0).any():
        raise ValueError('A training target has no known supervision')
    return np.clip((y * weight).sum(0) / support, 1e-4, 1 - 1e-4), support


def rng_state():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_checkpoint(path, payload):
    import torch
    path = Path(path)
    torch.save(payload, path.with_suffix('.tmp'))
    os.replace(path.with_suffix('.tmp'), path)


def weak_validation_evidence(plan, indices, predictions, *, logits, epoch, weak_bce, experiment=None):
    """Bind already-computed best-EMA predictions to the full ordered weak holdout.

    This private payload includes original soft labels (including unknown 0.5),
    never gold labels. It does not fit, recompute, or change the selection metric.
    """
    validate_plan(plan)
    validate_experiment(plan, experiment, current_runner=False)
    expected = [i for i, row in enumerate(plan['records']) if row['split'] == 'weak_validation']
    if not expected or list(indices) != expected:
        raise ValueError('Evidence must cover the exact ordered weak-validation plan records')
    targets = list(plan['targets'])
    records = [plan['records'][i] for i in expected]
    ids = [row['uid'] for row in records]
    if not targets or len(set(targets)) != len(targets) or len(set(ids)) != len(ids) or any(not uid for uid in ids):
        raise ValueError('Invalid weak-validation target or ID schema')
    labels = [[float(value) for value in row['labels']] for row in records]
    probabilities = [[float(value) for value in row] for row in predictions]
    logit_values = [[float(value) for value in row] for row in logits]
    if (len(probabilities) != len(expected) or
            any(len(row) != len(targets) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in row)
                for row in labels + probabilities)):
        raise ValueError('Weak-validation evidence has invalid shape or probabilities')
    if (len(logit_values) != len(expected) or
            any(len(row) != len(targets) or any(not math.isfinite(v) for v in row) for row in logit_values)):
        raise ValueError('Weak-validation evidence has invalid logits')
    for probability_row, logit_row in zip(probabilities, logit_values):
        for probability, logit in zip(probability_row, logit_row):
            sigmoid = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
            if abs(sigmoid - probability) > 1e-6:
                raise ValueError('Weak-validation probabilities and logits are misaligned')
    if (not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1 or
            not math.isfinite(weak_bce) or weak_bce < 0):
        raise ValueError('Invalid selected epoch or weak BCE')
    evidence = {'version': 1, 'split': 'weak_validation', 'model': 'selected_best_ema',
                'plan_fingerprint': plan['fingerprint'], 'plan_record_indices': expected,
                'targets': targets, 'ids': ids, 'weak_labels': labels, 'probabilities': probabilities,
                'logits': logit_values, 'loss_dtype': 'float32 logits and float32-cast original weak labels',
                'selected_epoch': epoch, 'selected_weak_bce': float(weak_bce),
                'selection_metric': 'confidence-weighted per-target macro weak BCE; exact 0.5 unknown'}
    if experiment is not None:
        if epoch > experiment['config']['max_epochs']:
            raise ValueError('Selected epoch exceeds its named experiment')
        evidence.update(version=2, experiment=experiment, experiment_fingerprint=experiment['fingerprint'])
    return evidence | {'fingerprint': fingerprint(evidence)}


def _check_best_evidence(plan, payload, expected_fingerprint=None):
    evidence = payload.get('weak_validation_evidence')
    if not isinstance(evidence, dict):
        raise ValueError('Best checkpoint has no matching weak-validation evidence')
    if payload.get('plan_fingerprint') != plan['fingerprint'] or payload.get('targets') != plan['targets']:
        raise ValueError('Best evidence checkpoint plan or target order differs')
    experiment = payload.get('experiment')
    if experiment is not None and (payload.get('experiment_fingerprint') != experiment['fingerprint'] or
            payload.get('runner_sha256') != experiment['runner_sha256'] or
            payload.get('config') != experiment['config'] or
            payload.get('initial_encoder_sha256') != experiment['initial_encoder_sha256']):
        raise ValueError('Best checkpoint experiment binding differs')
    expected = weak_validation_evidence(plan, evidence.get('plan_record_indices', []),
        evidence.get('probabilities', []), logits=evidence.get('logits', []),
        epoch=payload['epoch'], weak_bce=payload['weak_bce'], experiment=experiment)
    if evidence != expected or (expected_fingerprint is not None and evidence['fingerprint'] != expected_fingerprint):
        raise ValueError('Best weak-validation evidence differs from its plan or selected checkpoint')
    return evidence


def write_best_evidence(output_dir, plan, payload, *, expected_fingerprint=None):
    evidence = _check_best_evidence(plan, payload, expected_fingerprint)
    output_dir = Path(output_dir)
    checkpoint_sha256 = file_sha256(output_dir / 'best.pt')
    name = 'best-weak-validation-private.json'
    write_json(output_dir / name, {'best_checkpoint': 'best.pt', 'best_checkpoint_sha256': checkpoint_sha256,
                                  'evidence': evidence})
    return {'file': name, 'sha256': file_sha256(output_dir / name),
            'evidence_fingerprint': evidence['fingerprint'], 'best_checkpoint_sha256': checkpoint_sha256,
            'selected_epoch': evidence['selected_epoch']}


def restore_best_evidence(previous_dir, output_dir, plan, *, best_loss, expected_fingerprint, experiment=None):
    """Recover the sidecar from the selected checkpoint, even if no epoch improves."""
    import torch
    if not isinstance(expected_fingerprint, str) or len(expected_fingerprint) != 64:
        raise ValueError('Resume has no selected weak-validation evidence fingerprint')
    previous = Path(previous_dir) / 'best.pt'
    payload = torch.load(previous, map_location='cpu', weights_only=True)
    if payload.get('experiment') != experiment:
        raise ValueError('Resume best checkpoint belongs to a different experiment')
    if payload.get('weak_bce') != best_loss:
        raise ValueError('Resume best checkpoint loss does not match')
    _check_best_evidence(plan, payload, expected_fingerprint)
    destination = Path(output_dir) / 'best.pt'
    if previous.resolve() != destination.resolve():
        shutil.copy2(previous, destination)
    return write_best_evidence(output_dir, plan, payload, expected_fingerprint=expected_fingerprint)


def training_diagnostic_indices(records, limit=64):
    """Fixed ID-hash cohort of training rows, independent of labels and validation."""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 64:
        raise ValueError('Training diagnostic is bounded to at most 64 rows')
    eligible = [i for i, row in enumerate(records) if row['split'] == 'train']
    if not eligible or len({records[i]['uid'] for i in eligible}) != len(eligible):
        raise ValueError('Training diagnostic requires unique training IDs')
    return sorted(eligible, key=lambda i: hashlib.sha256(
        ('fresh-ema-training-diagnostic:' + records[i]['uid']).encode()).hexdigest())[:limit]


def record_training_step(trace, loss, *, overflow):
    """Track attempted known-label batches without changing optimizer decisions."""
    if not math.isfinite(loss) or loss < 0:
        raise ValueError('Invalid diagnostic online batch loss')
    trace['online_loss_sum'] += float(loss)
    trace['online_training_batches'] += 1
    trace['amp_overflow_skips'] += int(overflow)
    trace['successful_optimizer_steps'] += int(not overflow)


def training_trace_summary(trace, lr_end):
    return {key: trace[key] for key in ('successful_optimizer_steps', 'amp_overflow_skips',
            'online_training_batches', 'lr_start')} | {
        'lr_end': list(lr_end),
        'online_training_batch_loss_mean': trace['online_loss_sum'] / trace['online_training_batches']
            if trace['online_training_batches'] else None,
        'online_loss_note': 'Mean of confidence-weighted macro batch BCE over known-label attempted batches, including AMP-overflow skips; online weights and training augmentation.'}


def tensor_state_sha256(state):
    """Hash actual tensors deterministically without saving another weight file.

    v1: sorted state names; length-prefixed JSON name/dtype/shape, then raw
    contiguous CPU bytes viewed as uint8 (including dtypes NumPy cannot decode).
    One tensor is copied at a time, keeping peak memory bounded.
    """
    import torch
    digest = hashlib.sha256(b'fresh-encoder-state-v1\0')
    for name, tensor in sorted(state.items()):
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode()
        digest.update(len(metadata).to_bytes(8, 'big')); digest.update(metadata)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, 'big')); digest.update(raw)
    return digest.hexdigest()


def load_cached_batch(records, index, cache_dir, indices, *, device='cuda'):
    import numpy as np
    import torch
    arrays = []
    for i in indices:
        with np.load(Path(cache_dir) / 'arrays' / index[records[i]['uid']]['file'], allow_pickle=False) as z:
            if not z['mask'].any():
                raise ValueError('Empty study in training cache')
            arrays.append(tuple(z[key].copy() for key in ('x', 'mask', 'pos')))
    x, mask, pos = [torch.from_numpy(np.stack([a[j] for a in arrays])).to(device) for j in range(3)]
    y = torch.tensor([records[i]['labels'] for i in indices], dtype=torch.float32, device=device)
    return x, mask, pos, y


def train(cache_dir, source_dir, output_dir, *, resume=None, experiment=None, started_at=None):
    import copy
    import numpy as np
    import torch
    from knee.fresh_reader_setup import progress, time_limit
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for fresh training; no CPU fallback')
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'last.pt').exists() and resume is None:
        raise ValueError('Existing training output requires explicit resume or a new directory')
    started = time.monotonic() if started_at is None else started_at
    if not math.isfinite(started) or started > time.monotonic():
        raise ValueError('Invalid whole-stage start time')
    elapsed_before = 0.
    plan = json.loads((cache_dir / 'plan-private.json').read_text())
    validate_plan(plan)
    complete = json.loads((cache_dir / 'cache-complete.json').read_text())
    if complete['status'] != 'PASS' or complete['plan_fingerprint'] != plan['fingerprint']:
        raise ValueError('A completed cache with matching fingerprint is required')
    validate_experiment(plan, experiment)
    config = dict(plan['config']) if experiment is None else dict(experiment['config'])
    experiment_fields = {} if experiment is None else {'experiment': experiment, 'experiment_fingerprint': experiment['fingerprint']}
    if experiment is None and (config['max_epochs'] > 4 or config['training_seconds'] > 5400):
        raise ValueError('Pilot exceeds its authorized four-epoch/90-minute budget')
    state = None
    if resume:
        if experiment is not None:
            elapsed_before = duration_resume_report(resume, experiment, started)
        with time_limit(remaining_training_budget(elapsed_before, started), 'resume checkpoint loading'):
            state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['plan_fingerprint'] != plan['fingerprint'] or state['source_hashes'] != SOURCE_HASHES or state['backbone'] != BACKBONE or state.get('runner_sha256') != file_sha256(__file__):
            raise ValueError('Resume training contract differs')
        if state.get('experiment') != experiment:
            raise ValueError('Resume belongs to a different training experiment')
        if experiment is not None:
            validate_duration_resume(state, experiment)
            if state['elapsed_seconds'] > elapsed_before:
                raise ValueError('Duration resume report predates its last checkpoint')
        remaining_training_budget(state['elapsed_seconds'], started)
        elapsed_before = max(elapsed_before, state['elapsed_seconds'])
    remaining_training_budget(elapsed_before, started)
    records = plan['records']
    index = json.loads((cache_dir / 'cache-index-private.json').read_text())
    cache_verification_started = time.monotonic()
    verified_bytes = 0
    progress('train_cache_verification_begin', verified_studies=0, total_studies=len(records),
             verified_bytes=0, elapsed_seconds=0.)
    with time_limit(remaining_training_budget(elapsed_before, started), 'training cache verification'):
        for verified_studies, row in enumerate(records, 1):
            if file_sha256(cache_dir / 'arrays' / index[row['uid']]['file']) != index[row['uid']]['sha256']:
                raise ValueError('Training cache hash mismatch')
            verified_bytes += index[row['uid']]['bytes']
            if verified_studies % 64 == 0:
                progress('train_cache_verification_progress', verified_studies=verified_studies,
                         total_studies=len(records), verified_bytes=verified_bytes,
                         elapsed_seconds=time.monotonic() - cache_verification_started)
    progress('train_cache_verification_complete', verified_studies=len(records), total_studies=len(records),
             verified_bytes=verified_bytes, elapsed_seconds=time.monotonic() - cache_verification_started)
    tr = [i for i, row in enumerate(records) if row['split'] == 'train']
    va = [i for i, row in enumerate(records) if row['split'] == 'weak_validation']
    if not tr or not va or {records[i]['group'] for i in tr} & {records[i]['group'] for i in va}:
        raise ValueError('Empty partitions or report-group leakage')
    diagnostic_indices = training_diagnostic_indices(records)
    diagnostic_contract = {'count': len(diagnostic_indices),
        'ordered_ids_fingerprint': fingerprint([records[i]['uid'] for i in diagnostic_indices]),
        'selection': 'SHA256(fresh-ema-training-diagnostic:ID), training rows only; no label selection',
        'model': 'EMA; unaugmented; diagnostic only, never checkpoint selection'}
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    prior, support = training_prior([records[i]['labels'] for i in tr])
    progress('model_initialization_begin', backbone=BACKBONE, resume=resume is not None)
    # Fixed network timeouts plus the POSIX deadline bound fresh ImageNet loading.
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
    os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '30')
    with time_limit(min(300, remaining_training_budget(elapsed_before, started)), 'ImageNet model initialization'):
        code = load_source(source_dir, 'knee')
        model = code.KneeNet(BACKBONE, pretrained=resume is None, img_size=config['image_size']).cuda()
        initial_encoder_sha256 = tensor_state_sha256(model.enc.state_dict()) if resume is None else None
        if experiment is not None and resume is None and initial_encoder_sha256 != INITIAL_ENCODER_SHA256:
            raise ValueError('Duration experiment generic ImageNet encoder hash differs')
        progress('model_initialization_complete', elapsed_seconds=time.monotonic() - started)
        model.cls_b.data.copy_(torch.as_tensor(np.log(prior / (1 - prior)), device='cuda'))
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW([
            {'params': model.enc.parameters(), 'lr': config['lr_encoder']},
            {'params': [p for n, p in model.named_parameters() if not n.startswith('enc.')], 'lr': config['lr_head']}],
            weight_decay=config['weight_decay'])
        scaler = torch.amp.GradScaler('cuda')
    batch_size = config['batch_size']

    def batch(indices):
        return load_cached_batch(records, index, cache_dir, indices)

    @torch.inference_mode()
    def validate(current, indices):
        current.eval()
        numerator, denominator = torch.zeros(len(plan['targets'])), torch.zeros(len(plan['targets']))
        predictions, saved_logits = [], []
        for offset in range(0, len(indices), batch_size):
            if elapsed_before + time.monotonic() - started > config['training_seconds'] - 120:
                raise TimeoutError('Validation wall-clock budget exhausted')
            x, mask, pos, y = batch(indices[offset:offset + batch_size])
            with torch.autocast('cuda', dtype=torch.float16):
                logit = current(x, mask, pos, res=config['image_size'], train=False)
            n, d = loss_terms(logit, y)
            numerator += n.cpu(); denominator += d.cpu()
            predictions.extend(logit.float().sigmoid().cpu().tolist())
            saved_logits.extend(logit.float().cpu().tolist())
        if not (denominator > 0).any() or not torch.isfinite(numerator).all():
            raise ValueError('Invalid validation loss/support')
        from knee.validation import evaluate_predictions
        truth = {str(i): [None if y == .5 else float(y > .5) for y in records[i]['labels']] for i in indices}
        predicted = {str(i): p for i, p in zip(indices, predictions)}
        auc = evaluate_predictions(truth, predicted, plan['targets'])
        return {'weak_bce': float(macro_loss(numerator, denominator)),
                'confidence_support': denominator.tolist(),
                'per_target_bce': per_target_bce(numerator, denominator),
                'undefined_bce_targets': [t for t, d in zip(plan['targets'], denominator) if d <= 0],
                'thresholded_weak_auc': auc}, predictions, saved_logits

    best_evidence_receipt = None
    if state:
        with time_limit(remaining_training_budget(elapsed_before, started), 'matching checkpoint restoration'):
            if math.isfinite(state['best_loss']):
                best_evidence_receipt = restore_best_evidence(Path(resume).parent, output_dir, plan,
                    best_loss=state['best_loss'], expected_fingerprint=state.get('best_evidence_fingerprint'), experiment=experiment)
            model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
            optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        epochs, gate = state['planned_epochs'], state['gate']
        initial_encoder_sha256 = state['initial_encoder_sha256']
    elif experiment is not None:
        gate_started = time.monotonic()
        progress('gpu_gate_begin', train_studies=len(tr), validation_studies=len(va),
                 diagnostic_studies=len(diagnostic_indices), elapsed_seconds=0.)
        gate = duration_timing_gate(model, ema, optimizer, scaler, batch, validate, tr, va,
                                   diagnostic_indices, config, started)
        gate.update(experiment_fingerprint=experiment['fingerprint'])
        write_json(output_dir / 'gpu-gate.json', gate)
        progress('gpu_gate_complete', elapsed_seconds=time.monotonic() - gate_started,
                 passed=gate['passed'], planned_epochs=gate['planned_epochs'],
                 warmup_batches=16, timed_batches=48,
                 estimated_epoch_seconds_with_margin=gate['estimated_epoch_seconds_with_margin'])
        if not gate['passed']:
            raise RuntimeError('All12 duration epochs must fit the90minute budget; no shortened horizon')
        epochs = 12
    else:
        # Forward/backward timing only; no optimizer updates. Restore RNG afterward.
        gate_started = time.monotonic()
        progress('gpu_gate_begin', train_studies=len(tr), validation_studies=len(va),
                 diagnostic_studies=len(diagnostic_indices), elapsed_seconds=0.)
        before_gate = rng_state()
        torch.cuda.synchronize(); tick = time.monotonic()
        model.train()
        measured = 0
        for offset in range(0, min(len(tr), 64 * batch_size), batch_size):
            if time.monotonic() - started > config['training_seconds'] - 180:
                raise TimeoutError('GPU timing gate exhausted the training budget')
            x, mask, pos, y = batch(tr[offset:offset + batch_size])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                numerator, denominator = loss_terms(model(x, mask, pos, res=config['image_size'], train=True), y)
            if not torch.isfinite(numerator).all():
                raise ValueError('Nonfinite pilot supervision')
            if not (denominator > 0).any():
                continue
            scaler.scale(macro_loss(numerator, denominator)).backward()
            measured += 1
        torch.cuda.synchronize()
        if measured == 0:
            raise ValueError('GPU pilot has no known supervision')
        train_batch_seconds = (time.monotonic() - tick) / measured
        optimizer.zero_grad(set_to_none=True)
        tick = time.monotonic()
        validate(ema, va[:min(len(va), 8 * batch_size)])
        torch.cuda.synchronize()
        val_batch_seconds = (time.monotonic() - tick) / math.ceil(min(len(va), 8 * batch_size) / batch_size)
        tick = time.monotonic()
        validate(ema, diagnostic_indices[:min(len(diagnostic_indices), 8 * batch_size)])
        torch.cuda.synchronize()
        diagnostic_batch_seconds = (time.monotonic() - tick) / math.ceil(min(len(diagnostic_indices), 8 * batch_size) / batch_size)
        restore_rng(before_gate)
        epoch_seconds = (math.ceil(len(tr) / batch_size) * train_batch_seconds +
                         math.ceil(len(va) / batch_size) * val_batch_seconds +
                         math.ceil(len(diagnostic_indices) / batch_size) * diagnostic_batch_seconds) * 1.5
        remaining = config['training_seconds'] - (time.monotonic() - started) - 180
        epochs = min(config['max_epochs'], int(remaining // epoch_seconds))
        gate = {'passed': epochs >= 1, 'measured_training_steps': measured, 'train_batch_seconds': train_batch_seconds,
                'validation_batch_seconds': val_batch_seconds, 'estimated_epoch_seconds_with_margin': epoch_seconds,
                'training_diagnostic_batch_seconds': diagnostic_batch_seconds,
                'training_diagnostic_studies': len(diagnostic_indices),
                'planned_epochs': epochs}
        write_json(output_dir / 'gpu-gate.json', gate)
        progress('gpu_gate_complete', elapsed_seconds=time.monotonic() - gate_started,
                 passed=gate['passed'], measured_training_steps=measured, planned_epochs=epochs,
                 train_batch_seconds=train_batch_seconds, validation_batch_seconds=val_batch_seconds,
                 diagnostic_batch_seconds=diagnostic_batch_seconds,
                 estimated_epoch_seconds_with_margin=epoch_seconds)
        if epochs < 1:
            raise RuntimeError('GPU throughput gate predicts no complete epoch within budget')
    steps_per_epoch = math.ceil(len(tr) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs * steps_per_epoch))
    if state:
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'])
    epoch = state['epoch'] if state else 0
    cursor = state['cursor'] if state else 0
    order = state['order'] if state else None
    epoch_training_trace = state['epoch_training_trace'] if state else None
    history = state['history'] if state else []
    best = state['best_loss'] if state else float('inf')
    best_evidence_fingerprint = best_evidence_receipt['evidence_fingerprint'] if best_evidence_receipt else None
    step = state['step'] if state else 0
    run_contract = {**experiment_fields, 'runner_sha256': file_sha256(__file__),
                    'plan_fingerprint': plan['fingerprint'], 'source_hashes': SOURCE_HASHES,
                    'backbone': BACKBONE, 'config': config, 'environment': environment(),
                    'pretrained_cfg': model.enc.pretrained_cfg, 'initial_encoder_sha256': initial_encoder_sha256,
                    'encoder_hash_scheme': 'fresh-encoder-state-v1', 'targets': plan['targets'],
                    'train_studies': len(tr), 'validation_studies': len(va), 'training_prior': prior.tolist(),
                    'training_diagnostic': diagnostic_contract,
                    'training_confidence_support': support.tolist(), 'gate': gate,
                    'normalization': 'Fixed ImageNet mean/std; no fitted image statistics.',
                    'metric': 'Per-target macro confidence-weighted weak BCE, excluding and listing unsupported targets; exact 0.5 masked. No gold evaluation.'}
    write_json(output_dir / 'run-contract.json', run_contract)
    # The validation baseline uses training-fitted prevalence only.
    vy = torch.tensor([records[i]['labels'] for i in va], dtype=torch.float32)
    prior_logits = torch.tensor(np.log(prior / (1 - prior))).expand_as(vy)
    n, d = loss_terms(prior_logits, vy)
    prior_bce = float(macro_loss(n, d))
    run_contract['validation_confidence_support'] = d.tolist()
    run_contract['prior_per_target_bce'] = per_target_bce(n, d)
    write_json(output_dir / 'run-contract.json', run_contract)

    def checkpoint():
        checkpoint_started = time.monotonic()
        progress('checkpoint_save_begin', kind='last', epoch=epoch,
                 elapsed_seconds=elapsed_before + time.monotonic() - started)
        save_checkpoint(output_dir / 'last.pt', {
            **experiment_fields, 'config': config,
            'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'rng': rng_state(),
            'epoch': epoch, 'cursor': cursor, 'order': order, 'history': history, 'best_loss': best,
            'best_evidence_fingerprint': best_evidence_fingerprint,
            'epoch_training_trace': epoch_training_trace,
            'step': step, 'planned_epochs': epochs, 'gate': gate, 'backbone': BACKBONE,
            'plan_fingerprint': plan['fingerprint'], 'source_hashes': SOURCE_HASHES,
            'runner_sha256': file_sha256(__file__), 'initial_encoder_sha256': initial_encoder_sha256,
            'elapsed_seconds': elapsed_before + time.monotonic() - started})
        progress('checkpoint_save_complete', kind='last', epoch=epoch,
                 save_seconds=time.monotonic() - checkpoint_started,
                 elapsed_seconds=elapsed_before + time.monotonic() - started)

    status = 'RUNNING'
    try:
        while epoch < epochs:
            if order is None:
                order = np.random.permutation(tr).tolist()
            if epoch_training_trace is None:
                epoch_training_trace = {'successful_optimizer_steps': 0, 'amp_overflow_skips': 0,
                    'online_training_batches': 0, 'online_loss_sum': 0.,
                    'lr_start': [group['lr'] for group in optimizer.param_groups]}
            progress('train_epoch_begin', epoch=epoch + 1, planned_epochs=epochs,
                     attempted_supervised_batches=epoch_training_trace['online_training_batches'],
                     successful_optimizer_steps=epoch_training_trace['successful_optimizer_steps'],
                     elapsed_seconds=elapsed_before + time.monotonic() - started)
            model.train()
            while cursor < len(order):
                if elapsed_before + time.monotonic() - started > config['training_seconds'] - 120:
                    status = 'BUDGET_STOP'; checkpoint(); return status
                x, mask, pos, y = batch(order[cursor:cursor + batch_size])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    numerator, denominator = loss_terms(model(x, mask, pos, res=config['image_size'], train=True), y)
                if not torch.isfinite(numerator).all():
                    raise ValueError('Invalid training loss')
                if not (denominator > 0).any():
                    cursor += batch_size
                    continue
                batch_loss = macro_loss(numerator, denominator)
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                cursor += batch_size
                overflow = scaler.get_scale() < old_scale
                record_training_step(epoch_training_trace, float(batch_loss.detach()), overflow=overflow)
                if epoch_training_trace['online_training_batches'] % 32 == 0:
                    progress('train_batch_progress', epoch=epoch + 1,
                             attempted_supervised_batches=epoch_training_trace['online_training_batches'],
                             successful_optimizer_steps=epoch_training_trace['successful_optimizer_steps'],
                             amp_overflow_skips=epoch_training_trace['amp_overflow_skips'],
                             online_training_batch_loss_mean=epoch_training_trace['online_loss_sum'] / epoch_training_trace['online_training_batches'],
                             elapsed_seconds=elapsed_before + time.monotonic() - started)
                if overflow:
                    continue  # FP16 overflow: no optimizer/scheduler/EMA update occurred.
                scheduler.step()
                step += 1
                decay = min(config['ema'], (1 + step) / (10 + step))
                with torch.no_grad():
                    for a, b in zip(ema.parameters(), model.parameters()):
                        a.lerp_(b, 1 - decay)
                    for a, b in zip(ema.buffers(), model.buffers()):
                        a.copy_(b)
            diagnostic, _, _ = validate(ema, diagnostic_indices)
            validation, validation_predictions, validation_logits = validate(ema, va)
            score = validation['weak_bce']
            history.append({'epoch': epoch + 1, **validation, 'prevalence_weak_bce': prior_bce,
                'training_trace': training_trace_summary(epoch_training_trace,
                    [group['lr'] for group in optimizer.param_groups]),
                'training_diagnostic': {key: diagnostic[key] for key in
                    ('weak_bce', 'confidence_support', 'undefined_bce_targets')}})
            if score < best:
                evidence = weak_validation_evidence(plan, va, validation_predictions,
                    logits=validation_logits, epoch=epoch + 1, weak_bce=score, experiment=experiment)
                best_payload = {**experiment_fields, 'runner_sha256': file_sha256(__file__),
                    'model': ema.state_dict(), 'backbone': BACKBONE,
                    'targets': plan['targets'], 'source_hashes': SOURCE_HASHES, 'config': config,
                    'plan_fingerprint': plan['fingerprint'], 'epoch': epoch + 1, 'weak_bce': score,
                    'training_prior': prior.tolist(), 'initial_encoder_sha256': initial_encoder_sha256,
                    'weak_validation_evidence': evidence}
                checkpoint_started = time.monotonic()
                progress('checkpoint_save_begin', kind='best', epoch=epoch + 1,
                         elapsed_seconds=elapsed_before + time.monotonic() - started)
                save_checkpoint(output_dir / 'best.pt', best_payload)
                progress('checkpoint_save_complete', kind='best', epoch=epoch + 1,
                         save_seconds=time.monotonic() - checkpoint_started,
                         elapsed_seconds=elapsed_before + time.monotonic() - started)
                best, best_evidence_fingerprint = score, evidence['fingerprint']
                best_evidence_receipt = None
                best_evidence_receipt = write_best_evidence(output_dir, plan, best_payload)
                del best_payload
            epoch += 1; cursor = 0; order = None; epoch_training_trace = None
            checkpoint()
            print(json.dumps({'event': 'epoch', 'epoch': epoch, 'weak_bce': score, 'prevalence_weak_bce': prior_bce}), flush=True)
        status = 'PASS'
    except TimeoutError:
        status = 'BUDGET_STOP'; checkpoint()
    except Exception:
        status = 'FAILED'; checkpoint(); raise
    finally:
        write_json(output_dir / 'training-report.json', {**experiment_fields, 'runner_sha256': file_sha256(__file__),
            'status': status, 'history': history,
            'prevalence_weak_bce': prior_bce, 'planned_epochs': epochs, 'completed_epochs': epoch,
            'best_weak_bce': best if math.isfinite(best) else None,
            'best_weak_validation_evidence': best_evidence_receipt,
            'training_diagnostic': diagnostic_contract,
            'partial_epoch_training_trace': training_trace_summary(epoch_training_trace,
                [group['lr'] for group in optimizer.param_groups]) if epoch_training_trace else None,
            'elapsed_seconds': elapsed_before + time.monotonic() - started,
            'cuda_peak_bytes': torch.cuda.max_memory_allocated(), 'plan_fingerprint': plan['fingerprint']})
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('cache-pilot', 'cache', 'train'))
    parser.add_argument('--sources', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data'); parser.add_argument('--weak-labels'); parser.add_argument('--cache')
    parser.add_argument('--resume')
    parser.add_argument('--cache-seconds', type=int)
    args = parser.parse_args()
    cache_time_budget(args.stage, args.cache_seconds)
    if args.stage == 'train':
        train(args.cache, args.sources, args.output, resume=args.resume)
    else:
        prepare_cache(args.data, args.weak_labels, args.sources, args.output, pilot=args.stage == 'cache-pilot', cache_seconds=args.cache_seconds)


if __name__ == '__main__':
    main()
