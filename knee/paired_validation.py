"""Paired, aggregate-only analysis of saved fresh and frozen weak predictions.

Requires NumPy only when analysis runs. No fitting, checkpoint deserialization,
image reads, gold-array access, model selection, or blend search occurs here.
"""
import argparse
import json
import math
from pathlib import Path

from knee.fresh_reader import fingerprint, validate_plan, validate_experiment, weak_validation_evidence
from knee.validation import file_sha256

MEMBERS = ('baseline', 'attention_seed43', 'attention_seed44', 'meanmax_seed42')
SEED = 20261008
DRAWS = 2000
FRESH = 'fresh_selected_ema'
BASELINES = ('frozen_dino_single', 'frozen_dino_four_head')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _matrix(values, shape, name, *, probability=False):
    import numpy as np
    result = np.asarray(values, dtype=np.float64)
    require(result.shape == shape and np.isfinite(result).all(), name + ' shape or finite values differ')
    if probability:
        require(((result >= 0) & (result <= 1)).all(), name + ' must be in [0,1]')
    return result


def loss_matrix(labels, values, *, logits=False):
    """Float64 stable BCE; labels match the float32 values used in training."""
    import numpy as np
    y = np.asarray(labels, dtype=np.float32).astype(np.float64)
    require(y.ndim == 2 and len(y) > 0 and y.shape[1] > 0, 'Empty or invalid label matrix')
    y = _matrix(y, y.shape, 'Labels', probability=True)
    value = _matrix(values, y.shape, 'Predictions', probability=not logits)
    if logits:
        # Avoid cancellation and retain finite losses for sigmoid-saturated values.
        loss = y * np.logaddexp(0, -value) + (1 - y) * np.logaddexp(0, value)
    else:
        p = np.clip(value, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
        loss = -y * np.log(p) - (1 - y) * np.log1p(-p)
    require(np.isfinite(loss).all(), 'Nonfinite BCE')
    return loss, 2 * np.abs(y - .5)


def _auc_draws(labels, predictions, counts):
    """Exact tie-aware AUC for repeated rows, vectorized over bootstrap draws."""
    import numpy as np
    result = np.full((len(counts), labels.shape[1]), np.nan)
    for column in range(labels.shape[1]):
        known = np.flatnonzero(labels[:, column] != .5)
        if not len(known):
            continue
        order = known[np.argsort(predictions[known, column], kind='stable')]
        starts = np.r_[0, np.flatnonzero(np.diff(predictions[order, column]) != 0) + 1]
        positive = labels[order, column] > .5
        pos = np.add.reduceat(counts[:, order] * positive, starts, axis=1)
        neg = np.add.reduceat(counts[:, order] * ~positive, starts, axis=1)
        pos_total, neg_total = pos.sum(1), neg.sum(1)
        defined = (pos_total > 0) & (neg_total > 0)
        wins = (pos * (np.cumsum(neg, axis=1) - .5 * neg)).sum(1)
        result[defined, column] = wins[defined] / (pos_total[defined] * neg_total[defined])
    return result


def metrics(labels, probabilities, targets, *, logits=None):
    import numpy as np
    y = np.asarray(labels, dtype=np.float32).astype(np.float64)
    require(y.ndim == 2 and y.shape[1] == len(targets) and len(set(targets)) == len(targets),
            'Target schema differs')
    p = _matrix(probabilities, y.shape, 'Probabilities', probability=True)
    losses, weights = loss_matrix(y, p if logits is None else logits, logits=logits is not None)
    support = weights.sum(0)
    defined = support > 0
    require(defined.any(), 'No observed validation supervision')
    bce = [float((losses[:, j] * weights[:, j]).sum() / support[j]) if defined[j] else None
           for j in range(len(targets))]
    auc = _auc_draws(y, p, np.ones((1, len(y)), dtype=np.int64))[0]
    valid_auc = np.isfinite(auc)
    rows = {}
    for j, target in enumerate(targets):
        known = y[:, j] != .5
        positive = int(((y[:, j] > .5) & known).sum())
        rows[target] = {'known': int(known.sum()), 'positive': positive,
                        'negative': int(known.sum()) - positive, 'missing': int((~known).sum()),
                        'auc': float(auc[j]) if valid_auc[j] else None,
                        'bce': bce[j], 'confidence_support': float(support[j])}
    return {'macro_confidence_weighted_bce': float(np.mean([v for v in bce if v is not None])),
            'macro_all_targets_auc': float(auc.mean()) if valid_auc.all() else None,
            'macro_defined_targets_auc': float(auc[valid_auc].mean()) if valid_auc.any() else None,
            'undefined_bce_targets': [t for t, ok in zip(targets, defined) if not ok],
            'undefined_auc_targets': [t for t, ok in zip(targets, valid_auc) if not ok],
            'per_target': rows}


def resample_counts(groups, *, draws=DRAWS, seed=SEED):
    """Sample G groups G times with replacement; every member has that multiplicity."""
    import numpy as np
    require(bool(groups) and all(isinstance(g, str) and g for g in groups), 'Need nonempty group keys')
    require(isinstance(draws, int) and not isinstance(draws, bool) and draws > 0, 'Invalid draw count')
    names = sorted(set(groups))
    lookup = {name: i for i, name in enumerate(names)}
    membership = np.asarray([lookup[g] for g in groups])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(names), size=(draws, len(names)))
    group_counts = np.stack([np.bincount(row, minlength=len(names)) for row in samples])
    return group_counts[:, membership]


def _interval(values, estimate):
    import numpy as np
    valid = np.isfinite(values)
    return {'estimate': estimate,
            'percentile_95_interval': np.quantile(values[valid], [.025, .975], method='linear').tolist()
                if valid.any() else None,
            'valid_draws': int(valid.sum()), 'invalid_draws': int((~valid).sum())}


def paired_bootstrap(labels, predictions, targets, groups, *, fresh_logits, draws=DRAWS, seed=SEED):
    """Paired fresh-minus-frozen intervals; undefined draws never change target sets."""
    import numpy as np
    require(set(predictions) == {FRESH, *BASELINES}, 'Need the fresh model and both fixed baselines')
    y = np.asarray(labels, dtype=np.float32).astype(np.float64)
    require(len(groups) == len(y), 'Group coverage differs')
    point = {name: metrics(y, p, targets, logits=fresh_logits if name == FRESH else None)
             for name, p in predictions.items()}
    counts = resample_counts(groups, draws=draws, seed=seed)
    weights = 2 * np.abs(y - .5)
    supported = weights.sum(0) > 0
    denominator = counts @ weights
    bce_defined = (denominator[:, supported] > 0).all(1)
    values = {}
    for name, p in predictions.items():
        loss, _ = loss_matrix(y, fresh_logits if name == FRESH else p, logits=name == FRESH)
        numerator = counts @ (loss * weights)
        bce = np.full(draws, np.nan)
        bce[bce_defined] = (numerator[bce_defined][:, supported] /
                            denominator[bce_defined][:, supported]).mean(1)
        per_auc = _auc_draws(y, np.asarray(p), counts)
        auc = np.full(draws, np.nan)
        auc_defined = np.isfinite(per_auc).all(1)
        auc[auc_defined] = per_auc[auc_defined].mean(1)
        values[name] = {'macro_confidence_weighted_bce': bce, 'macro_all_targets_auc': auc}
    comparisons = {}
    for baseline in BASELINES:
        comparisons['fresh_minus_' + baseline] = {}
        for metric in ('macro_confidence_weighted_bce', 'macro_all_targets_auc'):
            fresh_point, base_point = point[FRESH][metric], point[baseline][metric]
            estimate = fresh_point - base_point if fresh_point is not None and base_point is not None else None
            comparisons['fresh_minus_' + baseline][metric] = _interval(
                values[FRESH][metric] - values[baseline][metric], estimate)
    return {'draws': draws, 'seed': seed, 'resampling_units': len(set(groups)),
            'sampled_studies_min': int(counts.sum(1).min()), 'sampled_studies_max': int(counts.sum(1).max()),
            'interval_method': 'Paired differences; 2.5/97.5 percentiles, linear interpolation; no model reselection.',
            'bce_draw_policy': 'Fixed targets supported in the original cohort; a draw losing any such target is invalid.',
            'auc_draw_policy': 'All plan targets must have both known classes in each draw; otherwise the draw is invalid.',
            'direction': 'Negative BCE and positive AUC differences favor fresh.', 'comparisons': comparisons}


def _load_predictions(path, targets):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == {'ids', 'targets', 'predictions'}, 'Unexpected prediction archive schema')
        ids = data['ids'].astype(str).tolist()
        require(data['targets'].tolist() == targets, 'Prediction target order differs')
        values = _matrix(data['predictions'], (len(ids), len(targets)), 'Saved probabilities', probability=True)
    require(ids and len(ids) == len(set(ids)) and all(ids), 'Duplicate or empty prediction IDs')
    return ids, values


def evidence_experiment(plan, private, directory):
    """Explicit v1 compatibility; named v2 evidence must match its run artifacts."""
    evidence = private['evidence']
    if evidence.get('version') == 1:
        require('experiment' not in evidence and 'experiment_fingerprint' not in evidence,
                'Legacy v1 evidence cannot carry a named experiment')
        require(1 <= evidence.get('selected_epoch', 0) <= min(4, plan.get('config', {}).get('max_epochs', 4)),
                'Legacy v1 selected epoch exceeds its original pilot horizon')
        for name in ('run-contract.json', 'training-report.json'):
            path = Path(directory) / name
            if path.exists():
                artifact = json.loads(path.read_text())
                require(artifact.get('experiment') is None and artifact.get('experiment_fingerprint') is None,
                        'Legacy v1 evidence cannot accompany a named experiment run')
        return None
    require(evidence.get('version') == 2 and isinstance(evidence.get('experiment'), dict),
            'Unknown fresh evidence version or missing experiment')
    experiment = evidence['experiment']
    validate_experiment(plan, experiment, current_runner=False)
    require(evidence.get('experiment_fingerprint') == experiment['fingerprint'], 'Evidence experiment hash differs')
    artifacts = [json.loads((Path(directory) / name).read_text()) for name in
                 ('run-contract.json', 'training-report.json')]
    for artifact in artifacts:
        require(artifact.get('experiment') == experiment and
                artifact.get('experiment_fingerprint') == experiment['fingerprint'] and
                artifact.get('plan_fingerprint') == plan['fingerprint'] and
                artifact.get('runner_sha256') == experiment['runner_sha256'], 'Evidence/run experiment binding differs')
    run, report = artifacts
    require(run.get('config') == experiment['config'] and
            run.get('initial_encoder_sha256') == experiment['initial_encoder_sha256'], 'Named run config or initialization differs')
    selected = report.get('best_weak_validation_evidence') or {}
    require(selected.get('evidence_fingerprint') == evidence['fingerprint'] and
            selected.get('best_checkpoint_sha256') == private['best_checkpoint_sha256'], 'Selected report/evidence pair differs')
    return experiment


def load_inputs(plan_path, fresh_path, baseline_dir, comparison_dir, *, fresh_checkpoint=None):
    """Return private aligned arrays in memory, never in the aggregate report."""
    import numpy as np
    plan_path, fresh_path, base, comp = map(Path, (plan_path, fresh_path, baseline_dir, comparison_dir))
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    require('identity_audit' in plan, 'Need the final post-header plan')
    records, targets = plan['records'], plan['targets']
    require(len({r['uid'] for r in records}) == len(records) and all(r['uid'] for r in records),
            'Duplicate or empty plan IDs')
    require({r['split'] for r in records} == {'train', 'weak_validation'}, 'Unexpected plan split roles')
    cohort = [r for r in records if r['split'] == 'weak_validation']
    require(not ({r['group'] for r in records if r['split'] == 'train'} & {r['group'] for r in cohort}),
            'Plan has train/validation report-group overlap')
    indices = [i for i, row in enumerate(records) if row['split'] == 'weak_validation']
    private = json.loads(fresh_path.read_text())
    require(set(private) == {'best_checkpoint', 'best_checkpoint_sha256', 'evidence'} and
            private['best_checkpoint'] == 'best.pt', 'Unexpected fresh evidence envelope')
    evidence = private['evidence']
    experiment = evidence_experiment(plan, private, fresh_path.parent)
    expected = weak_validation_evidence(plan, indices, evidence['probabilities'], logits=evidence['logits'],
                                       epoch=evidence['selected_epoch'], weak_bce=evidence['selected_weak_bce'],
                                       experiment=experiment)
    require(evidence == expected, 'Fresh evidence differs from exact ordered plan, labels, or fingerprint')
    checkpoint = Path(fresh_checkpoint) if fresh_checkpoint else fresh_path.parent / 'best.pt'
    require(file_sha256(checkpoint) == private['best_checkpoint_sha256'], 'Fresh checkpoint hash differs')
    y = np.asarray([r['labels'] for r in cohort], np.float32).astype(np.float64)
    fresh_p = np.asarray(evidence['probabilities'], dtype=np.float64)
    fresh_z = np.asarray(evidence['logits'], dtype=np.float64)
    fresh_metric = metrics(y, fresh_p, targets, logits=fresh_z)['macro_confidence_weighted_bce']
    require(math.isclose(fresh_metric, evidence['selected_weak_bce'], rel_tol=2e-6, abs_tol=2e-6),
            'Saved fresh logits do not reproduce selected BCE within float32 accumulation tolerance')

    extraction = json.loads((base / 'extraction.json').read_text())
    comparison = json.loads((comp / 'comparison.json').read_text())
    require(comparison['status'] == 'completed', 'Original comparison did not complete')
    require(comparison['selected']['members'] == list(MEMBERS) and comparison['selected']['weights'] == [.25] * 4,
            'Recorded selected ensemble changed')
    require(plan['weak_labels_sha256'] == extraction['weak_labels_sha256'], 'Weak-label source changed')
    cache = base / 'features.npz'
    cache_sha = file_sha256(cache)
    require(cache_sha == extraction['features_sha256'] == comparison['feature_sha256'],
            'Feature cache differs from extraction/comparison receipts')
    require(targets == comparison['targets'], 'Plan/comparison target order differs')
    with np.load(cache, allow_pickle=False) as data:
        # Do not request features, gold labels or private reports.
        cached_ids = data['ids'].astype(str).tolist()
        require(len(set(cached_ids)) == len(cached_ids) and all(cached_ids), 'Duplicate or empty cache IDs')
        require(data['targets'].tolist() == targets, 'Feature target order differs')
        lookup = {uid: i for i, uid in enumerate(cached_ids)}
        splits, groups = data['splits'], data['groups']
        weak, masks = data['weak'], data['masks']
        require(splits.shape == groups.shape == (len(cached_ids),) and
                weak.shape == (len(cached_ids), len(targets)) and masks.shape[0] == len(cached_ids),
                'Feature metadata shapes differ')
        for row in records:
            require(row['uid'] in lookup, 'Plan study absent from original feature cache')
            i = lookup[row['uid']]
            require(splits[i] == row['split'], 'Plan study changed its original split role')
            require(groups[i] == row['group'], 'Plan grouping differs from original baseline')
            require(np.array_equal(np.asarray(row['labels'], np.float32), weak[i]), 'Plan labels differ from original weak labels')
        require(all(masks[lookup[r['uid']]].any() for r in cohort), 'Validation study had no original image features')
        full_ids = [uid for uid, i in lookup.items() if splits[i] == 'weak_validation' and masks[i].any()]
        full_truth = np.stack([weak[lookup[uid]] for uid in full_ids])
        baseline_training_count = int((splits == 'train').sum())
    arrays, hashes = {}, {}
    for name in (*MEMBERS, 'selected'):
        path = comp / f'{name}_weak_predictions.npz'
        ids, values = _load_predictions(path, targets)
        require(set(ids) == set(full_ids), 'Saved predictions must cover the original weak holdout exactly')
        row_of = {uid: i for i, uid in enumerate(ids)}
        arrays[name] = np.stack([values[row_of[uid]] for uid in full_ids])
        hashes[name] = file_sha256(path)
    require(np.array_equal(arrays['selected'], np.stack([arrays[n] for n in MEMBERS]).mean(0)),
            'Selected saved probabilities differ from the fixed four-member mean')
    head_hashes = {}
    for name in MEMBERS:
        head = base / 'attention/head.pt' if name == 'baseline' else comp / name / 'head.pt'
        head_hashes[name] = file_sha256(head)
        require(head_hashes[name] == comparison['candidates'][name]['checkpoint_sha256'], 'Frozen checkpoint hash differs')
    single_bce = metrics(full_truth, arrays['baseline'], targets)['macro_confidence_weighted_bce']
    ensemble_bce = metrics(full_truth, arrays['selected'], targets)['macro_confidence_weighted_bce']
    expected_single = [r['weak_bce'] for r in comparison['comparisons'] if r['members'] == ['baseline']]
    require(len(expected_single) == 1 and abs(single_bce - expected_single[0]) <= 1e-12 and
            abs(ensemble_bce - comparison['selected']['weak_bce']) <= 1e-12,
            'Saved probabilities no longer reproduce original comparison scores')
    selected_ids = [r['uid'] for r in cohort]
    positions = {uid: i for i, uid in enumerate(full_ids)}
    take = [positions[uid] for uid in selected_ids]
    predictions = {FRESH: fresh_p, BASELINES[0]: arrays['baseline'][take], BASELINES[1]: arrays['selected'][take]}
    receipt = {'status': 'PASS', 'targets': targets, 'cohort_studies': len(cohort),
               'fresh_evidence_version': evidence['version'],
               'fresh_experiment_fingerprint': experiment['fingerprint'] if experiment else None,
               'fresh_training_studies': sum(r['split'] == 'train' for r in records),
               'original_baseline_training_studies': baseline_training_count,
               'plan_fingerprint': plan['fingerprint'], 'plan_file_sha256': file_sha256(plan_path),
               'ordered_cohort_sha256': fingerprint(selected_ids),
               'ordered_report_groups_sha256': fingerprint([r['group'] for r in cohort]),
               'fresh_evidence_sha256': file_sha256(fresh_path), 'fresh_evidence_fingerprint': evidence['fingerprint'],
               'fresh_checkpoint_sha256': private['best_checkpoint_sha256'],
               'fresh_selected_epoch': evidence['selected_epoch'], 'fresh_recorded_float32_bce': evidence['selected_weak_bce'],
               'feature_sha256': cache_sha, 'weak_labels_sha256': plan['weak_labels_sha256'],
               'prediction_archive_sha256': hashes, 'frozen_checkpoint_sha256': head_hashes,
               'comparison_receipt_sha256': file_sha256(comp / 'comparison.json'),
               'extraction_receipt_sha256': file_sha256(base / 'extraction.json'),
               'selected_recipe': {'members': list(MEMBERS), 'weights': [.25] * 4, 'selection_repeated': False}}
    return y, predictions, fresh_z, [r['group'] for r in cohort], receipt


def analyze(plan_path, fresh_path, baseline_dir, comparison_dir, *, fresh_checkpoint=None):
    y, predictions, logits, groups, report = load_inputs(plan_path, fresh_path, baseline_dir, comparison_dir,
                                                       fresh_checkpoint=fresh_checkpoint)
    import numpy as np
    targets = report['targets']
    report['analysis_version'] = 1
    report['analysis_source_sha256'] = file_sha256(__file__)
    report['numpy_version'] = np.__version__
    report['random_generator'] = 'NumPy default_rng (PCG64); sorted group keys'
    report['metrics'] = {name: metrics(y, values, targets, logits=logits if name == FRESH else None)
                         for name, values in predictions.items()}
    report['paired_bootstrap'] = {
        'study': paired_bootstrap(y, predictions, targets, [str(i) for i in range(len(y))], fresh_logits=logits),
        'report_group': paired_bootstrap(y, predictions, targets, groups, fresh_logits=logits)}
    report['metric_policy'] = ('Labels are float32-cast original weak values; exact 0.5 is unknown. BCE uses '
        '2*abs(y-0.5) confidence, normalizes within each supported target, then averages targets. '
        'Fresh BCE is recomputed stably from stored logits in float64; frozen BCE uses saved probabilities '
        'with float64 machine-epsilon endpoint clipping. AUC thresholds known truth at >0.5 and uses average ties. '
        'Primary AUC requires every target (12 in this competition); secondary point AUC lists excluded targets.')
    report['limitations'] = [
        'Matched validation cohort, not matched training budget: frozen models used 2000 training studies versus at most 512 fresh.',
        'This reused weak development cohort selected heads, ensemble and fresh epoch; intervals condition on those selections and do not correct selection optimism.',
        'Report-group resampling preserves duplicate reports, not an independently verified patient grouping; anonymized patient identity remains limited.',
        'Public weak-label prompt independence is unverified. No gold labels, gold metrics or images are used. These are not independent clinical labels.',
        'Percentile intervals are descriptive finite-cohort uncertainty, not a leaderboard prediction or permission to tune blends.']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--fresh-evidence', type=Path, required=True)
    parser.add_argument('--fresh-checkpoint', type=Path)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--comparison-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a new aggregate output path')
    report = analyze(args.plan, args.fresh_evidence, args.baseline_dir, args.comparison_dir,
                     fresh_checkpoint=args.fresh_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'cohort_studies': report['cohort_studies'],
                      'bootstrap_draws_per_scheme': DRAWS}, indent=2))


if __name__ == '__main__':
    main()
