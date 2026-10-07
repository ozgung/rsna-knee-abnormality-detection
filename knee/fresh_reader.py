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


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_plan(plan):
    actual = fingerprint({k: v for k, v in plan.items() if k != 'fingerprint'})
    if plan.get('fingerprint') != actual:
        raise ValueError('Plan contents do not match their fingerprint')
    return actual


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
    versions = {}
    for name in ('numpy', 'pandas', 'torch', 'timm', 'pydicom', 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'versions': versions,
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


def prepare_cache(data_dir, weak_path, source_dir, output_dir, *, pilot=True, config=None):
    import cv2
    import numpy as np
    import pandas as pd
    config = dict(CONFIG if config is None else config)
    started = time.monotonic()
    deadline = config['pilot_seconds'] if pilot else config['cache_seconds']
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
    if not pilot:
        gate = json.loads((output_dir / 'cache-pilot.json').read_text())
        if gate['plan_fingerprint'] != plan['fingerprint'] or not gate['gate']['passed']:
            raise ValueError('Full cache requires a passing pilot for this exact plan')
    selected = pilot_order(records, config['pilot_studies']) if pilot else records
    index_path = output_dir / 'cache-index-private.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    new_seconds, new_count, new_bytes = 0., 0, 0
    audit = {'status': 'RUNNING', 'plan_fingerprint': plan['fingerprint'], 'environment': environment(),
             'selected_studies': len(selected), 'selected_series': 0, 'missing_slots': 0, 'failures': 0}
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
        audit['gate'] = budget_gate(len(sample), len(records), sum(s['seconds'] for s in sample),
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


def train(cache_dir, source_dir, output_dir, *, resume=None):
    import copy
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for fresh training; no CPU fallback')
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'last.pt').exists() and resume is None:
        raise ValueError('Existing training output requires explicit resume or a new directory')
    started = time.monotonic()
    elapsed_before = 0.
    plan = json.loads((cache_dir / 'plan-private.json').read_text())
    validate_plan(plan)
    complete = json.loads((cache_dir / 'cache-complete.json').read_text())
    if complete['status'] != 'PASS' or complete['plan_fingerprint'] != plan['fingerprint']:
        raise ValueError('A completed cache with matching fingerprint is required')
    config = plan['config']
    if config['max_epochs'] > 4 or config['training_seconds'] > 5400:
        raise ValueError('Pilot exceeds its authorized four-epoch/90-minute budget')
    records = plan['records']
    index = json.loads((cache_dir / 'cache-index-private.json').read_text())
    for row in records:
        if file_sha256(cache_dir / 'arrays' / index[row['uid']]['file']) != index[row['uid']]['sha256']:
            raise ValueError('Training cache hash mismatch')
    tr = [i for i, row in enumerate(records) if row['split'] == 'train']
    va = [i for i, row in enumerate(records) if row['split'] == 'weak_validation']
    if not tr or not va or {records[i]['group'] for i in tr} & {records[i]['group'] for i in va}:
        raise ValueError('Empty partitions or report-group leakage')
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    code = load_source(source_dir, 'knee')
    prior, support = training_prior([records[i]['labels'] for i in tr])
    model = code.KneeNet(BACKBONE, pretrained=resume is None, img_size=config['image_size']).cuda()
    model.cls_b.data.copy_(torch.as_tensor(np.log(prior / (1 - prior)), device='cuda'))
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {'params': model.enc.parameters(), 'lr': config['lr_encoder']},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('enc.')], 'lr': config['lr_head']}],
        weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')
    batch_size = config['batch_size']

    def batch(indices):
        arrays = []
        for i in indices:
            with np.load(cache_dir / 'arrays' / index[records[i]['uid']]['file'], allow_pickle=False) as z:
                if not z['mask'].any():
                    raise ValueError('Empty study in training cache')
                arrays.append(tuple(z[key].copy() for key in ('x', 'mask', 'pos')))
        x, mask, pos = [torch.from_numpy(np.stack([a[j] for a in arrays])).cuda() for j in range(3)]
        y = torch.tensor([records[i]['labels'] for i in indices], dtype=torch.float32, device='cuda')
        return x, mask, pos, y

    @torch.inference_mode()
    def validate(current, indices):
        current.eval()
        numerator, denominator = torch.zeros(len(plan['targets'])), torch.zeros(len(plan['targets']))
        predictions = []
        for offset in range(0, len(indices), batch_size):
            if elapsed_before + time.monotonic() - started > config['training_seconds'] - 120:
                raise TimeoutError('Validation wall-clock budget exhausted')
            x, mask, pos, y = batch(indices[offset:offset + batch_size])
            with torch.autocast('cuda', dtype=torch.float16):
                logit = current(x, mask, pos, res=config['image_size'], train=False)
            n, d = loss_terms(logit, y)
            numerator += n.cpu(); denominator += d.cpu()
            predictions.extend(logit.float().sigmoid().cpu().tolist())
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
                'thresholded_weak_auc': auc}

    state = torch.load(resume, map_location='cpu', weights_only=False) if resume else None
    if state:
        if state['plan_fingerprint'] != plan['fingerprint'] or state['source_hashes'] != SOURCE_HASHES or state['backbone'] != BACKBONE or state.get('runner_sha256') != file_sha256(__file__):
            raise ValueError('Resume training contract differs')
        if math.isfinite(state['best_loss']):
            previous_best = Path(resume).parent / 'best.pt'
            best_payload = torch.load(previous_best, map_location='cpu', weights_only=True)
            if (best_payload['plan_fingerprint'] != plan['fingerprint'] or
                    best_payload['weak_bce'] != state['best_loss']):
                raise ValueError('Resume best checkpoint is missing or does not match')
            if previous_best.resolve() != (output_dir / 'best.pt').resolve():
                shutil.copy2(previous_best, output_dir / 'best.pt')
            del best_payload
        model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        epochs, gate = state['planned_epochs'], state['gate']
    else:
        # Forward/backward timing only; no optimizer updates. Restore RNG afterward.
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
        restore_rng(before_gate)
        epoch_seconds = (math.ceil(len(tr) / batch_size) * train_batch_seconds +
                         math.ceil(len(va) / batch_size) * val_batch_seconds) * 1.5
        remaining = config['training_seconds'] - (time.monotonic() - started) - 180
        epochs = min(config['max_epochs'], int(remaining // epoch_seconds))
        gate = {'passed': epochs >= 1, 'measured_training_steps': measured, 'train_batch_seconds': train_batch_seconds,
                'validation_batch_seconds': val_batch_seconds, 'estimated_epoch_seconds_with_margin': epoch_seconds,
                'planned_epochs': epochs}
        write_json(output_dir / 'gpu-gate.json', gate)
        if epochs < 1:
            raise RuntimeError('GPU throughput gate predicts no complete epoch within budget')
    steps_per_epoch = math.ceil(len(tr) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs * steps_per_epoch))
    if state:
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'])
    elapsed_before = state['elapsed_seconds'] if state else 0.
    epoch = state['epoch'] if state else 0
    cursor = state['cursor'] if state else 0
    order = state['order'] if state else None
    history = state['history'] if state else []
    best = state['best_loss'] if state else float('inf')
    step = state['step'] if state else 0
    run_contract = {'plan_fingerprint': plan['fingerprint'], 'source_hashes': SOURCE_HASHES,
                    'backbone': BACKBONE, 'config': config, 'environment': environment(),
                    'pretrained_cfg': model.enc.pretrained_cfg, 'targets': plan['targets'],
                    'train_studies': len(tr), 'validation_studies': len(va), 'training_prior': prior.tolist(),
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
        save_checkpoint(output_dir / 'last.pt', {
            'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'rng': rng_state(),
            'epoch': epoch, 'cursor': cursor, 'order': order, 'history': history, 'best_loss': best,
            'step': step, 'planned_epochs': epochs, 'gate': gate, 'backbone': BACKBONE,
            'plan_fingerprint': plan['fingerprint'], 'source_hashes': SOURCE_HASHES,
            'runner_sha256': file_sha256(__file__),
            'elapsed_seconds': elapsed_before + time.monotonic() - started})

    status = 'RUNNING'
    try:
        while epoch < epochs:
            if order is None:
                order = np.random.permutation(tr).tolist()
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
                scaler.scale(macro_loss(numerator, denominator)).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                cursor += batch_size
                if scaler.get_scale() < old_scale:
                    continue  # FP16 overflow: no optimizer/scheduler/EMA update occurred.
                scheduler.step()
                step += 1
                decay = min(config['ema'], (1 + step) / (10 + step))
                with torch.no_grad():
                    for a, b in zip(ema.parameters(), model.parameters()):
                        a.lerp_(b, 1 - decay)
                    for a, b in zip(ema.buffers(), model.buffers()):
                        a.copy_(b)
            validation = validate(ema, va)
            score = validation['weak_bce']
            history.append({'epoch': epoch + 1, **validation, 'prevalence_weak_bce': prior_bce})
            if score < best:
                best = score
                save_checkpoint(output_dir / 'best.pt', {'model': ema.state_dict(), 'backbone': BACKBONE,
                    'targets': plan['targets'], 'source_hashes': SOURCE_HASHES, 'config': config,
                    'plan_fingerprint': plan['fingerprint'], 'epoch': epoch + 1, 'weak_bce': score,
                    'training_prior': prior.tolist()})
            epoch += 1; cursor = 0; order = None
            checkpoint()
            print(json.dumps({'event': 'epoch', 'epoch': epoch, 'weak_bce': score, 'prevalence_weak_bce': prior_bce}), flush=True)
        status = 'PASS'
    except TimeoutError:
        status = 'BUDGET_STOP'; checkpoint()
    except Exception:
        status = 'FAILED'; checkpoint(); raise
    finally:
        write_json(output_dir / 'training-report.json', {'status': status, 'history': history,
            'prevalence_weak_bce': prior_bce, 'planned_epochs': epochs, 'completed_epochs': epoch,
            'best_weak_bce': best if math.isfinite(best) else None,
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
    args = parser.parse_args()
    if args.stage == 'train':
        train(args.cache, args.sources, args.output, resume=args.resume)
    else:
        prepare_cache(args.data, args.weak_labels, args.sources, args.output, pilot=args.stage == 'cache-pilot')


if __name__ == '__main__':
    main()
