"""One fixed, CPU-only512/200 frozen-DINO control; never open gold labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import subprocess
import time

from knee.fresh_reader import fingerprint, validate_plan
from knee.fresh_reader_setup import time_limit
from knee.paired_validation import metrics, require
from knee.validation import file_sha256

NAME = 'matched512-dino-attention-v1'
PINS = {
    'plan': '7f3afd222e7286e68e49feff14e02eb59fc06dc066ff606f6d3a03cc36db912e',
    'features': 'a8190ab6129dccae859171eec9f1d024d9882652b84edc2735980360084ea76b',
    'extraction': '28e74f00f745c5fa6d0a6a0d96fd4fa907714bc68b0776447a8c46f8b9d53daa',
    'sealed': '21c71745703d742585684298494a6269ae4217fadc5fc4c26f3bf0d10a289467',
}
PLAN = '1bd346c05c0aac20f54b82b790e356d3d6c1b95606accaa4a5d1da7e3e421942'
FEATURE_SOURCE = '81a44f1dcf6e88f488c2312797518d096328be1e245fbd0915853b7313801890'
COUNTS = {'train': 512, 'weak_validation': 200}
FEATURE_SHAPE = (3, 8, 768)
RECIPE = {'kind': 'attention', 'seed': 42, 'hidden': 128, 'dropout': .2,
          'lr': .001, 'weight_decay': .02, 'batch_size': 64, 'clip_grad_norm': 5,
          'max_epochs': 80, 'patience': 12, 'improvement_tolerance': 1e-5,
          'device': 'cpu', 'dtype': 'float32', 'threads': 2, 'max_seconds': 300}
SOURCES = ('knee/matched_dino.py', 'knee/features.py', 'knee/fresh_reader.py',
           'knee/fresh_reader_setup.py', 'knee/paired_validation.py', 'knee/validation.py')


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Matched control whole-fit deadline exceeded')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_inputs(plan_path, artifacts, sealed_path):
    """Hash-pinned metadata and selected features only; npz['gold'] is never read.

    NumPy decompresses the full feature member before indexing; unselected rows
    are immediately discarded and never enter statistics, fitting or inference.
    """
    import numpy as np
    artifacts = Path(artifacts)
    paths = {'plan': Path(plan_path), 'features': artifacts / 'features.npz',
             'extraction': artifacts / 'extraction.json', 'sealed': Path(sealed_path)}
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    require(hashes == PINS, 'Fixed input file hashes differ')
    plan = json.loads(paths['plan'].read_text())
    require(validate_plan(plan) == PLAN and 'identity_audit' in plan, 'Need the exact post-header512/200 plan')
    records, targets = plan['records'], plan['targets']
    require(Counter(r['split'] for r in records) == COUNTS, 'Fixed cohort counts or roles differ')
    ids = [r['uid'] for r in records]
    require(len(set(ids)) == len(ids) and all(ids), 'Duplicate or empty plan IDs')
    require(len(targets) == 12 and len(set(targets)) == 12, 'Expected the12 official targets')
    extraction = json.loads(paths['extraction'].read_text())
    require(extraction['status'] == 'completed' and extraction['features_sha256'] == hashes['features'] and
            extraction['weak_labels_sha256'] == plan['weak_labels_sha256'] and
            extraction['metadata_fingerprints'] == plan['files'], 'Extraction or Steven v1 source differs')
    sealed = json.loads(paths['sealed'].read_text())
    require(fingerprint({k: v for k, v in sealed.items() if k != 'fingerprint'}) == sealed['fingerprint'],
            'Sealed manifest self-fingerprint differs')
    identity = {r['uid']: r for r in sealed['records']}
    require(len(identity) == len(sealed['records']), 'Duplicate identity rows')
    for row in records:
        require(row['uid'] in identity, 'Plan row absent from identity audit')
        found = identity[row['uid']]
        expected = 'training_candidate' if row['split'] == 'train' else 'development'
        require(found['original_split'] == row['split'] and found['future_role'] == expected and
                found['report_group'] == row['group'] and not found['quarantine_reasons'],
                'Plan row changed original role or touches a protected component')
    train_components = {identity[r['uid']]['component'] for r in records if r['split'] == 'train'}
    protected = {r['component'] for r in sealed['records'] if r['future_role'] in ('development', 'gold', 'sealed')}
    require(not train_components & protected and not set(ids) & set(plan['gold_ids']), 'Protected component leakage')
    with np.load(paths['features'], allow_pickle=False) as cache:
        cached_ids = cache['ids'].astype(str).tolist()
        require(len(set(cached_ids)) == len(cached_ids) and all(cached_ids), 'Duplicate or empty cache IDs')
        require(cache['targets'].tolist() == targets, 'Feature target order differs')
        lookup = {uid: i for i, uid in enumerate(cached_ids)}
        require(all(uid in lookup for uid in ids), 'Plan row absent from feature cache')
        take = np.asarray([lookup[uid] for uid in ids])
        splits, groups = cache['splits'], cache['groups']
        require(splits.shape == groups.shape == (len(cached_ids),), 'Cache role/group shape differs')
        baseline = [{'uid': uid, 'split': str(splits[i]), 'group': str(groups[i])}
                    for i, uid in enumerate(cached_ids) if splits[i] != 'gold']
        require(fingerprint(baseline) == sealed['baseline_fingerprint'], 'Clean original assignment hash differs')
        # The historical fresh builder attached labels to its selected shared
        # dictionaries before hashing. Preserve that legacy digest separately.
        labels_by_id = {r['uid']: r['labels'] for r in records}
        legacy = [r | ({'labels': labels_by_id[r['uid']]} if r['uid'] in labels_by_id else {}) for r in baseline]
        require(fingerprint(legacy) == plan['baseline_fingerprint'], 'Legacy original assignment hash differs')
        require(splits[take].tolist() == [r['split'] for r in records] and
                groups[take].tolist() == [r['group'] for r in records], 'Original role/group alignment differs')
        # Do not access the gold member, including for schema or support checks.
        y = cache['weak'][take].astype(np.float32)
        mask = cache['masks'][take]
        x = cache['features'][take].astype(np.float32)
    require(x.shape == (len(records), *FEATURE_SHAPE) and mask.shape == x.shape[:3] and
            mask.dtype == np.bool_, 'Feature/mask shape or dtype differs')
    require(y.shape == (len(records), len(targets)) and np.isfinite(y).all() and
            ((y >= 0) & (y <= 1)).all() and
            np.array_equal(y, np.asarray([r['labels'] for r in records], np.float32)), 'Exact weak labels differ')
    require(np.isfinite(x).all() and mask.any((1, 2)).all() and (x[~mask] == 0).all(),
            'Empty/nonfinite study or nonzero masked padding')
    train = np.asarray([i for i, r in enumerate(records) if r['split'] == 'train'])
    validation = np.asarray([i for i, r in enumerate(records) if r['split'] == 'weak_validation'])
    require(not {records[i]['group'] for i in train} & {records[i]['group'] for i in validation},
            'Training/validation group overlap')
    return {'x': x, 'mask': mask, 'y': y, 'train': train, 'validation': validation,
            'records': records, 'targets': targets, 'plan': plan, 'input_sha256': hashes}


def make_model(data):
    import numpy as np
    import torch
    from knee.features import StudyHead, fit_feature_stats
    train = data['train']
    confidence = 2 * np.abs(data['y'] - .5)
    support = confidence[train].sum(0)
    require((support > 0).all(), 'A training target has no supervision')
    prior = (data['y'][train] * confidence[train]).sum(0) / support
    mean, std = fit_feature_stats(data['x'], data['mask'], train)
    model = StudyHead(mean, std, prior, kind=RECIPE['kind'], hidden=RECIPE['hidden'], dropout=RECIPE['dropout'])
    return model.to(device='cpu', dtype=torch.float32), prior, confidence


def fit(data, output, contract, deadline, report):
    import numpy as np
    import torch
    from knee.features import masked_loss
    check_time(deadline)
    torch.manual_seed(RECIPE['seed']); np.random.seed(RECIPE['seed'])
    model, prior, confidence = make_model(data)
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()),
            'CPU float32 recipe violated')
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])
    x, mask, y, weight = map(torch.from_numpy, (data['x'], data['mask'], data['y'], confidence))
    train, validation = data['train'], data['validation']
    truth = data['y'][validation]
    report['prevalence_baseline'] = metrics(truth, np.broadcast_to(prior, truth.shape), data['targets'])
    report['training_prior'] = prior.tolist()
    report['training_confidence_support'] = confidence[train].sum(0).tolist()
    report['training_count'], report['validation_count'] = len(train), len(validation)
    rng = np.random.default_rng(RECIPE['seed'])
    best, best_epoch = float('inf'), -1
    for epoch in range(RECIPE['max_epochs']):
        model.train(); losses = []
        for batch in np.array_split(rng.permutation(train), max(1, int(np.ceil(len(train) / RECIPE['batch_size'])))):
            check_time(deadline)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(x[batch], mask[batch]), y[batch], weight[batch])
            require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE['clip_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            report['successful_optimizer_steps'] += 1
            losses.append(float(loss.detach()))
            check_time(deadline)
        model.eval(); check_time(deadline)
        with torch.inference_mode():
            logits = model(x[validation], mask[validation])
            value = float(masked_loss(logits, y[validation], weight[validation]))
            probabilities = logits.sigmoid().numpy()
        require(np.isfinite(value), 'Nonfinite validation loss')
        check_time(deadline)
        report['history'].append({'epoch': epoch + 1, 'train_batch_bce': float(np.mean(losses)),
                                  'weak_validation_bce': value, 'successful_optimizer_steps': len(losses)})
        if value < best - RECIPE['improvement_tolerance']:
            best, best_epoch = value, epoch
            selected = metrics(truth, probabilities, data['targets'], logits=logits.numpy())
            require(abs(selected['macro_confidence_weighted_bce'] - best) < 2e-6, 'Selection metric mismatch')
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            checkpoint = {'state_dict': state, 'targets': data['targets'], 'contract': contract,
                          'selected_epoch': epoch + 1, 'selected_weak_bce': best}
            temporary = output / 'best.pt.tmp'
            torch.save(checkpoint, temporary); temporary.replace(output / 'best.pt')
            evidence = {'contract_fingerprint': contract['fingerprint'], 'plan_fingerprint': PLAN,
                'selected_epoch': epoch + 1, 'selected_weak_bce': best, 'targets': data['targets'],
                'plan_record_indices': validation.tolist(),
                'ids': [data['records'][i]['uid'] for i in validation],
                'groups': [data['records'][i]['group'] for i in validation],
                'weak_labels': truth.tolist(), 'probabilities': probabilities.tolist(), 'logits': logits.tolist(),
                'best_checkpoint_sha256': file_sha256(output / 'best.pt')}
            evidence['fingerprint'] = fingerprint(evidence)
            write_json(output / 'weak-predictions-private.json', evidence)
            report.update(selected_epoch=epoch + 1, selected_weak_bce=best, selected_metrics=selected,
                best_checkpoint_sha256=evidence['best_checkpoint_sha256'], evidence_fingerprint=evidence['fingerprint'])
        if (epoch + 1) % 5 == 0:
            print(json.dumps({'event': 'matched_control_epoch', 'epoch': epoch + 1,
                'weak_bce': value, 'successful_optimizer_steps': report['successful_optimizer_steps']}), flush=True)
        if epoch - best_epoch >= RECIPE['patience']:
            return 'early_stopping'
    return 'max_epochs'


def run(plan_path, artifacts, sealed_path, output):
    started = time.monotonic(); deadline = started + RECIPE['max_seconds']
    output = Path(output)
    claimed = False
    report = {'name': NAME, 'status': 'FAILED', 'config': dict(RECIPE), 'history': [],
              'successful_optimizer_steps': 0}
    previous_threads = None
    try:
        with time_limit(deadline - time.monotonic(), NAME):
            # The directory claim also uses the setup-inclusive deadline.
            output.mkdir(parents=True, exist_ok=False)
            claimed = True
            import numpy as np
            import torch
            root = Path(__file__).resolve().parents[1]
            sources = {name: file_sha256(root / name) for name in SOURCES}
            require(sources['knee/features.py'] == FEATURE_SOURCE, 'Audited StudyHead/statistics/loss source changed')
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
            report['environment'] = {'python': platform.python_version(), 'numpy': np.__version__,
                'torch': torch.__version__, 'platform': platform.platform(), 'device': 'cpu',
                'threads': 2, 'dtype': 'float32', 'revision': revision}
            data = load_inputs(plan_path, artifacts, sealed_path)
            check_time(deadline)
            contract = {'name': NAME, 'recipe': dict(RECIPE), 'plan_fingerprint': PLAN,
                'input_sha256': data['input_sha256'], 'source_sha256': sources, 'revision': revision,
                'targets': data['targets'],
                'ordered_assignment_fingerprint': fingerprint([
                    {k: r[k] for k in ('uid', 'split', 'group')} for r in data['records']]),
                'selection': 'confidence-weighted macro weak BCE; tolerance1e-5; no gold/sealed evaluation'}
            contract['fingerprint'] = fingerprint(contract)
            report['contract'] = contract
            write_json(output / 'run-contract.json', contract)
            previous_threads = torch.get_num_threads(); torch.set_num_threads(RECIPE['threads'])
            reason = fit(data, output, contract, deadline, report)
            check_time(deadline)
            report.update(status='PASS', stop_reason=reason)
    except FileExistsError:
        raise
    except TimeoutError:
        report.update(status='BUDGET_STOP', stop_reason='whole_fit_deadline', comparison_complete=False)
    except Exception as exc:
        # Exception text can contain paths/IDs. Keep the aggregate failure private-data free.
        report.update(status='FAILED', stop_reason=type(exc).__name__, comparison_complete=False)
    finally:
        report['runtime_seconds'] = time.monotonic() - started
        report.setdefault('comparison_complete', report['status'] == 'PASS')
        report['limitations'] = ['Reused Steven v1 weak development labels; no independent clinical validation.',
            'Old2000-head also used300 validation rows and CUDA, so its comparison is descriptive.',
            'Matching training counts does not match inputs: frozen3planes x8slice tokens versus fresh6slots x4three-frame windows; preprocessing and representation differ. No causal architecture claim.',
            'Full feature member decompressed before selecting712; no gold-label member access or unselected-row model/statistics use.',
            'Budget-stopped output is incomplete; do not extend the allowance or infer convergence.']
        if claimed:
            write_json(output / 'training-report.json', report)
        if previous_threads is not None:
            torch.set_num_threads(previous_threads)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--sealed-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run(args.plan, args.artifacts, args.sealed_manifest, args.output)
    print(json.dumps({k: report[k] for k in ('status', 'stop_reason', 'runtime_seconds', 'successful_optimizer_steps')}))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
