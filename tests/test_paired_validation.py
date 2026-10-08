"""Synthetic alignment, stable loss and paired cluster-bootstrap contracts."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from knee.paired_validation import (BASELINES, DRAWS, FRESH, MEMBERS, SEED, analyze,
    fingerprint, load_inputs, loss_matrix, metrics, paired_bootstrap, resample_counts)
from knee.validation import evaluate_predictions, file_sha256

try:
    import numpy as np
except ImportError:
    np = None


class ImportTests(unittest.TestCase):
    def test_import_does_not_load_ml_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable, '-S', '-c',
                        'import sys; import knee.paired_validation; '
                        'assert "numpy" not in sys.modules; assert "torch" not in sys.modules'],
                       cwd=root, check=True, capture_output=True)


@unittest.skipIf(np is None, 'optional NumPy unavailable')
class BootstrapTests(unittest.TestCase):
    def test_stable_logits_masked_soft_labels_and_tied_auc(self):
        y = [[.9, .5], [.1, .8], [.5, .2]]
        z = np.array([[1000., -1000.], [-1000., 0.], [1000., 0.]])
        p = np.array([[1., 0.], [0., .5], [1., .5]])
        result = metrics(y, p, ['a', 'b'], logits=z)
        self.assertAlmostEqual(result['per_target']['a']['bce'], 100., places=4)
        self.assertAlmostEqual(result['per_target']['b']['bce'], np.log(2))
        self.assertEqual(result['per_target']['b']['auc'], .5)
        self.assertEqual(result['per_target']['b']['missing'], 1)
        self.assertEqual(result['macro_all_targets_auc'], .75)
        loss, weight = loss_matrix(y, z, logits=True)
        self.assertEqual(float((loss * weight)[0, 1]), 0.)
        undefined = metrics([[.9, .5], [.8, .5]], [[.5, .5]] * 2, ['a', 'b'])
        self.assertIsNone(undefined['macro_all_targets_auc'])
        self.assertEqual(undefined['undefined_bce_targets'], ['b'])
        self.assertEqual(undefined['undefined_auc_targets'], ['a', 'b'])
        json.dumps(undefined, allow_nan=False)

    def test_group_resampling_preserves_all_members_and_variable_study_count(self):
        groups = ['paired', 'solo-a', 'paired', 'solo-b']
        counts = resample_counts(groups, draws=40, seed=7)
        np.testing.assert_array_equal(counts, resample_counts(groups, draws=40, seed=7))
        np.testing.assert_array_equal(counts[:, 0], counts[:, 2])
        self.assertGreater(len(set(counts.sum(1).tolist())), 1)
        self.assertTrue((counts.sum(1) >= 3).all())
        study_counts = resample_counts(['0', '1', '2', '3'], draws=40, seed=7)
        np.testing.assert_array_equal(study_counts.sum(1), np.full(40, 4))

    def test_paired_intervals_match_explicit_resampling_with_ties_and_invalid_draws(self):
        y = np.asarray([[.9, .5], [.1, .9], [.5, .1], [.2, .5]], dtype=np.float32)
        fresh = np.asarray([[.8, .5], [.4, .8], [.4, .2], [.4, .5]])
        baseline = np.full_like(fresh, .5)
        predictions = {FRESH: fresh, BASELINES[0]: baseline, BASELINES[1]: baseline}
        logits = np.log(fresh / (1 - fresh))
        groups = ['g0', 'g0', 'g1', 'g2']
        result = paired_bootstrap(y, predictions, ['a', 'b'], groups, fresh_logits=logits, draws=80, seed=6)
        counts = resample_counts(groups, draws=80, seed=6)
        bce_deltas, auc_deltas = [], []
        for count in counts:
            take = np.repeat(np.arange(len(y)), count)
            yy, pp = y[take], fresh[take]
            weight = 2 * np.abs(yy.astype(np.float64) - .5)
            if (weight.sum(0) > 0).all():
                fm = metrics(yy, pp, ['a', 'b'], logits=logits[take])
                bm = metrics(yy, baseline[take], ['a', 'b'])
                bce_deltas.append(fm['macro_confidence_weighted_bce'] - bm['macro_confidence_weighted_bce'])
            truth = {str(i): [None if v == .5 else float(v > .5) for v in row] for i, row in enumerate(yy)}
            score = {str(i): row.tolist() for i, row in enumerate(pp)}
            ref = evaluate_predictions(truth, score, ['a', 'b'])['macro_all_targets_auc']
            if ref is not None:
                auc_deltas.append(ref - .5)
        comparison = result['comparisons']['fresh_minus_' + BASELINES[0]]
        for name, values in [('macro_confidence_weighted_bce', bce_deltas), ('macro_all_targets_auc', auc_deltas)]:
            interval = comparison[name]
            self.assertEqual(interval['valid_draws'], len(values))
            self.assertEqual(interval['invalid_draws'], 80 - len(values))
            np.testing.assert_allclose(interval['percentile_95_interval'], np.quantile(values, [.025, .975]))
        self.assertGreater(comparison['macro_all_targets_auc']['invalid_draws'], 0)
        self.assertGreater(comparison['macro_confidence_weighted_bce']['invalid_draws'], 0)
        self.assertEqual(comparison, result['comparisons']['fresh_minus_' + BASELINES[1]])

    def test_undefined_all_target_auc_never_becomes_partial_macro_interval(self):
        y = [[.9, .9], [.1, .8], [.1, .5]]
        p = np.full((3, 2), .5)
        result = paired_bootstrap(y, {FRESH: p, BASELINES[0]: p, BASELINES[1]: p}, ['a', 'b'],
                                  ['a', 'b', 'c'], fresh_logits=np.zeros((3, 2)), draws=20)
        interval = result['comparisons']['fresh_minus_' + BASELINES[0]]['macro_all_targets_auc']
        self.assertIsNone(interval['estimate'])
        self.assertIsNone(interval['percentile_95_interval'])
        self.assertEqual(interval['invalid_draws'], 20)
        json.dumps(result, allow_nan=False)


def make_fixture(root):
    from knee.fresh_reader import weak_validation_evidence
    base, comp, fresh = root / 'base', root / 'comparison', root / 'fresh'
    for directory in (base / 'attention', comp, fresh):
        directory.mkdir(parents=True)
    targets = ['target-b', 'target-a']
    ids = ['private-train'] + [f'private-weak-{i}' for i in range(6)] + ['private-gold']
    weak = np.asarray([[.1, .9], [.9, .5], [.1, .9], [.9, .1], [.1, .5], [.2, .8], [.8, .2], [.5, .5]], dtype=np.float32)
    groups = ['train-group', 'paired', 'paired', 'g2', 'g3', 'g4', 'g5', 'gold-group']
    splits = ['train'] + ['weak_validation'] * 6 + ['validation']
    np.savez(base / 'features.npz', ids=ids, targets=targets, weak=weak, masks=np.ones((8, 2), bool),
             groups=groups, splits=splits, gold=np.array([{'never': 'read'}], dtype=object),
             features=np.array([{'never': 'read'}], dtype=object))
    take = [0, 5, 1, 4, 2]
    records = [{'uid': ids[i], 'split': splits[i], 'group': groups[i], 'labels': weak[i].tolist()} for i in take]
    plan = {'targets': targets, 'records': records, 'identity_audit': {}, 'weak_labels_sha256': 'weak-label-fixture'}
    plan['fingerprint'] = fingerprint(plan)
    plan_path = root / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    extraction = {'weak_labels_sha256': plan['weak_labels_sha256'], 'features_sha256': file_sha256(base / 'features.npz')}
    (base / 'extraction.json').write_text(json.dumps(extraction))
    p = np.asarray([[.8, .5], [.3, .7], [.75, .2], [.2, .6], [.4, .6], [.7, .25]])
    predictions = {name: p * (1 - i * .1) + .5 * i * .1 for i, name in enumerate(MEMBERS)}
    predictions['selected'] = np.stack([predictions[n] for n in MEMBERS]).mean(0)
    for name, values in predictions.items():
        # A different archive order must be explicitly aligned to the plan by ID.
        np.savez(comp / f'{name}_weak_predictions.npz', ids=list(reversed(ids[1:7])), targets=targets,
                 predictions=values[::-1])
    candidates = {}
    for name in MEMBERS:
        head = base / 'attention/head.pt' if name == 'baseline' else comp / name / 'head.pt'
        head.parent.mkdir(exist_ok=True)
        head.write_bytes(name.encode())
        candidates[name] = {'checkpoint_sha256': file_sha256(head)}
    comparison = {'status': 'completed', 'targets': targets, 'feature_sha256': extraction['features_sha256'],
        'selected': {'members': list(MEMBERS), 'weights': [.25] * 4,
                     'weak_bce': metrics(weak[1:7], predictions['selected'], targets)['macro_confidence_weighted_bce']},
        'comparisons': [{'members': ['baseline'],
                         'weak_bce': metrics(weak[1:7], p, targets)['macro_confidence_weighted_bce']}],
        'candidates': candidates}
    (comp / 'comparison.json').write_text(json.dumps(comparison))
    fp = p[np.asarray(take[1:]) - 1]
    fz = np.log(fp / (1 - fp))
    selected_bce = metrics(weak[take[1:]], fp, targets, logits=fz)['macro_confidence_weighted_bce']
    evidence = weak_validation_evidence(plan, [1, 2, 3, 4], fp, logits=fz, epoch=2, weak_bce=selected_bce)
    (fresh / 'best.pt').write_bytes(b'synthetic fresh checkpoint, never deserialized')
    fresh_path = fresh / 'best-weak-validation-private.json'
    fresh_path.write_text(json.dumps({'best_checkpoint': 'best.pt',
        'best_checkpoint_sha256': file_sha256(fresh / 'best.pt'), 'evidence': evidence}))
    return plan_path, fresh_path, base, comp


@unittest.skipIf(np is None, 'optional NumPy unavailable')
class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = make_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_full_analysis_aligns_permuted_archives_and_outputs_only_aggregates(self):
        y, predictions, _, groups, receipt = load_inputs(*self.paths)
        np.testing.assert_allclose(predictions[FRESH], predictions[BASELINES[0]])
        self.assertEqual(groups, ['g4', 'paired', 'g3', 'paired'])
        report = analyze(*self.paths)
        self.assertEqual(report['cohort_studies'], 4)
        for scheme in ('study', 'report_group'):
            result = report['paired_bootstrap'][scheme]
            self.assertEqual(result['draws'], DRAWS)
            self.assertEqual(result['seed'], SEED)
            self.assertEqual(result['resampling_units'], 4 if scheme == 'study' else 3)
        serialized = json.dumps(report, allow_nan=False)
        self.assertNotIn('private-', serialized)
        self.assertNotIn('probabilities', report)
        self.assertNotIn('logits', report)
        self.assertEqual(receipt['ordered_cohort_sha256'], report['ordered_cohort_sha256'])

    def test_fresh_alignment_tampering_is_rejected_even_with_recomputed_digest(self):
        path = self.paths[1]
        original = json.loads(path.read_text())
        for key, value in [('ids', list(reversed(original['evidence']['ids']))),
                           ('targets', list(reversed(original['evidence']['targets']))),
                           ('weak_labels', [[.1, .9]] * 4), ('selected_weak_bce', .123)]:
            changed = copy.deepcopy(original)
            changed['evidence'][key] = value
            changed['evidence']['fingerprint'] = fingerprint({k: v for k, v in changed['evidence'].items() if k != 'fingerprint'})
            path.write_text(json.dumps(changed))
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_inputs(*self.paths)
        path.write_text(json.dumps(original))

    def test_plan_and_checkpoint_hash_drift_is_rejected(self):
        plan_path, fresh_path, base, _ = self.paths
        original = plan_path.read_text()
        plan = json.loads(original)
        plan['records'][0]['labels'][0] = .2
        plan_path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            load_inputs(*self.paths)
        plan_path.write_text(original)
        head = fresh_path.parent / 'best.pt'
        old = head.read_bytes()
        head.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Fresh checkpoint hash'):
            load_inputs(*self.paths)
        head.write_bytes(old)
        (base / 'attention/head.pt').write_bytes(b'changed frozen head')
        with self.assertRaisesRegex(ValueError, 'Frozen checkpoint hash'):
            load_inputs(*self.paths)

    def test_prediction_id_coverage_target_order_and_fixed_mean_are_checked(self):
        path = self.paths[3] / 'selected_weak_predictions.npz'
        with np.load(path) as data:
            original = {name: data[name] for name in data.files}
        for key, values in [('ids', np.array(['duplicate'] * 6)),
                            ('targets', original['targets'][::-1]),
                            ('predictions', np.full((6, 2), .5))]:
            np.savez(path, **(original | {key: values}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_inputs(*self.paths)
        np.savez(path, **original)

    def test_original_labels_and_original_scores_are_checked(self):
        plan_path, fresh_path, base, comp = self.paths
        path = base / 'features.npz'
        with np.load(path, allow_pickle=True) as data:
            original = {name: data[name] for name in data.files}
        changed = original['weak'].copy()
        changed[1, 0] = .3
        np.savez(path, **(original | {'weak': changed}))
        for receipt_path, key in [(base / 'extraction.json', 'features_sha256'), (comp / 'comparison.json', 'feature_sha256')]:
            receipt = json.loads(receipt_path.read_text())
            receipt[key] = file_sha256(path)
            receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'labels differ'):
            load_inputs(*self.paths)
        np.savez(path, **original)
        for receipt_path, key in [(base / 'extraction.json', 'features_sha256'), (comp / 'comparison.json', 'feature_sha256')]:
            receipt = json.loads(receipt_path.read_text())
            receipt[key] = file_sha256(path)
            if receipt_path.name == 'comparison.json':
                receipt['selected']['weak_bce'] += .01
            receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'original comparison scores'):
            load_inputs(*self.paths)


if __name__ == '__main__':
    unittest.main()
