"""Synthetic fixed-cohort, leakage, weak-only and CPU-budget regressions."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from knee import matched_dino as control

try:
    import numpy as np
    import torch
except ImportError:
    np = None


class ImportTests(unittest.TestCase):
    def test_default_import_has_no_ml_dependency(self):
        subprocess.run([sys.executable, '-S', '-c',
            'import sys; import knee.matched_dino; assert "torch" not in sys.modules; assert "numpy" not in sys.modules'],
            cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)


@unittest.skipIf(np is None, 'Optional ML environment unavailable')
class MatchedControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.base = self.root / 'base'; self.base.mkdir()
        self.targets = [f'target{i}' for i in range(12)]
        self.ids = [f'private{i}' for i in range(10)]
        self.roles = ['train'] * 4 + ['weak_validation'] * 5 + ['gold']
        self.groups = [f'group{i}' for i in range(10)]
        self.y = np.asarray([[.1 if (i + j) % 2 else .9 for j in range(12)] for i in range(10)], np.float32)
        self.x = np.arange(10 * 3 * 2 * 4, dtype=np.float32).reshape(10, 3, 2, 4) / 100
        self.mask = np.ones((10, 3, 2), bool)
        self.mask[:, 2, 1] = False; self.x[~self.mask] = 0
        self.take = [3, 0, 2, 1, 7, 4, 6, 5]
        records = [{'uid': self.ids[i], 'split': self.roles[i], 'group': self.groups[i],
                    'labels': self.y[i].tolist()} for i in self.take]
        baseline = [{'uid': self.ids[i], 'split': self.roles[i], 'group': self.groups[i]} for i in range(9)]
        labels_by_id = {r['uid']: r['labels'] for r in records}
        legacy = [r | ({'labels': labels_by_id[r['uid']]} if r['uid'] in labels_by_id else {}) for r in baseline]
        self.plan = {'targets': self.targets, 'records': records, 'gold_ids': [self.ids[9]],
            'identity_audit': {}, 'weak_labels_sha256': 'steven-v1', 'files': {'train.csv': 'metadata'},
            'baseline_fingerprint': control.fingerprint(legacy)}
        self.plan['fingerprint'] = control.fingerprint(self.plan)
        self.plan_path = self.root / 'plan.json'; self.plan_path.write_text(json.dumps(self.plan))
        identities = [{'uid': self.ids[i], 'original_split': self.roles[i], 'report_group': self.groups[i],
            'future_role': 'training_candidate' if i < 4 else 'development' if i < 9 else 'gold',
            'component': f'component{i}', 'quarantine_reasons': []} for i in range(10)]
        identities.append({'uid': 'private-sealed', 'original_split': None, 'report_group': 'sealed',
            'future_role': 'sealed', 'component': 'sealed', 'quarantine_reasons': []})
        self.sealed = {'records': identities, 'baseline_fingerprint': control.fingerprint(baseline)}
        self.sealed['fingerprint'] = control.fingerprint(self.sealed)
        self.sealed_path = self.root / 'sealed.json'; self.sealed_path.write_text(json.dumps(self.sealed))
        self.write_cache()
        self.patches = [patch.object(control, 'COUNTS', {'train': 4, 'weak_validation': 4}),
            patch.object(control, 'FEATURE_SHAPE', (3, 2, 4)), patch.object(control, 'PLAN', self.plan['fingerprint'])]
        for p in self.patches: p.start()
        self.pinpatch = patch.object(control, 'PINS', self.pins()); self.pinpatch.start()
        self.threads = torch.get_num_threads(); torch.set_num_threads(2)

    def tearDown(self):
        self.pinpatch.stop()
        for p in reversed(self.patches): p.stop()
        torch.set_num_threads(self.threads); self.tmp.cleanup()

    def write_cache(self):
        # Object-valued gold cannot be loaded with allow_pickle=False; success proves it is unopened.
        np.savez(self.base / 'features.npz', ids=self.ids, targets=self.targets, weak=self.y,
                 features=self.x, masks=self.mask, splits=self.roles, groups=self.groups,
                 gold=np.asarray([{'forbidden': 'gold label'}], dtype=object))
        extraction = {'status': 'completed', 'features_sha256': control.file_sha256(self.base / 'features.npz'),
            'weak_labels_sha256': 'steven-v1', 'metadata_fingerprints': {'train.csv': 'metadata'}}
        (self.base / 'extraction.json').write_text(json.dumps(extraction))

    def pins(self):
        return {key: control.file_sha256(path) for key, path in {'plan': self.plan_path,
            'features': self.base / 'features.npz', 'extraction': self.base / 'extraction.json',
            'sealed': self.sealed_path}.items()}

    def load(self):
        return control.load_inputs(self.plan_path, self.base, self.sealed_path)

    def repin(self):
        self.pinpatch.stop(); self.pinpatch = patch.object(control, 'PINS', self.pins()); self.pinpatch.start()

    def test_exact_order_no_gold_member_and_train_only_statistics_prior(self):
        self.assertNotEqual(self.plan['baseline_fingerprint'], self.sealed['baseline_fingerprint'])
        data = self.load()
        np.testing.assert_array_equal(data['x'], self.x[self.take])
        np.testing.assert_array_equal(data['y'], self.y[self.take])
        self.assertEqual(data['train'].tolist(), [0, 1, 2, 3])
        self.assertEqual(data['validation'].tolist(), [4, 5, 6, 7])
        model, prior, confidence = control.make_model(data)
        selected = self.x[:4][self.mask[:4]]
        np.testing.assert_allclose(model.mean.numpy(), selected.mean(0), atol=1e-6)
        np.testing.assert_allclose(model.std.numpy(), selected.std(0), atol=1e-6)
        np.testing.assert_allclose(prior, (data['y'][:4] * confidence[:4]).sum(0) / confidence[:4].sum(0))
        changed = copy.deepcopy(data); changed['x'][4:] = 1e6; changed['y'][4:] = .99
        other, other_prior, _ = control.make_model(changed)
        torch.testing.assert_close(other.mean, model.mean); torch.testing.assert_close(other.std, model.std)
        np.testing.assert_array_equal(other_prior, prior)
        self.assertEqual(model.kind, 'attention'); self.assertEqual(model.proj[0].out_features, 128)
        self.assertEqual(model.proj[2].p, .2)
        self.assertTrue(all(v.device.type == 'cpu' and v.dtype == torch.float32 for v in model.parameters()))

    def test_tampered_source_hash_rejected_before_feature_load(self):
        self.plan_path.write_text(self.plan_path.read_text() + ' ')
        with patch('numpy.load', side_effect=AssertionError('no arrays after changed source')):
            with self.assertRaisesRegex(ValueError, 'hashes'): self.load()

    def test_wrong_cache_roles_groups_labels_masks_rejected_after_repin(self):
        for kind in ('roles', 'groups', 'labels', 'empty', 'padding'):
            with self.subTest(kind=kind):
                old = copy.deepcopy((self.roles, self.groups, self.y, self.mask, self.x))
                if kind == 'roles': self.roles[0] = 'weak_validation'
                if kind == 'groups': self.groups[0] = 'changed'
                if kind == 'labels': self.y[0, 0] = .3
                if kind == 'empty': self.mask[0] = False; self.x[0] = 0
                if kind == 'padding': self.x[0, 2, 1] = 1
                self.write_cache(); self.repin()
                with self.assertRaises(ValueError): self.load()
                self.roles, self.groups, self.y, self.mask, self.x = old
        self.write_cache(); self.repin()

    def test_crosslinks_to_unused_development_gold_or_sealed_rejected(self):
        for protected in ('development', 'gold', 'sealed'):
            with self.subTest(protected=protected):
                changed = copy.deepcopy(self.sealed)
                target = next(r for r in changed['records'] if r['future_role'] == protected and r['uid'] not in self.ids[4:8])
                changed['records'][0]['component'] = target['component']
                changed['fingerprint'] = control.fingerprint({k: v for k, v in changed.items() if k != 'fingerprint'})
                self.sealed_path.write_text(json.dumps(changed)); self.repin()
                with self.assertRaisesRegex(ValueError, 'leakage'): self.load()

    def test_exact_recipe_synthetic_fit_saves_ordered_best_and_only_selected_rows(self):
        data = self.load(); out = self.root / 'synthetic-fit'; out.mkdir()
        contract = {'fingerprint': 'synthetic-contract'}
        report = {'history': [], 'successful_optimizer_steps': 0}
        calls = []
        from knee.features import StudyHead
        forward = StudyHead.forward
        def record(model, features, masks):
            calls.append((model.training, features.detach().clone()))
            return forward(model, features, masks)
        with patch.object(control, 'RECIPE', {**control.RECIPE, 'max_epochs': 2}), patch.object(StudyHead, 'forward', record):
            reason = control.fit(data, out, contract, time.monotonic() + 10, report)
        self.assertEqual(reason, 'max_epochs'); self.assertEqual(report['successful_optimizer_steps'], 2)
        for training, values in calls:
            source = data['x'][data['train'] if training else data['validation']]
            self.assertTrue(all(any(np.array_equal(row.numpy(), allowed) for allowed in source) for row in values))
        evidence = json.loads((out / 'weak-predictions-private.json').read_text())
        self.assertEqual(evidence['ids'], [self.ids[i] for i in self.take[4:]])
        self.assertEqual(evidence['targets'], self.targets)
        self.assertEqual(evidence['best_checkpoint_sha256'], control.file_sha256(out / 'best.pt'))
        checkpoint = torch.load(out / 'best.pt', weights_only=True)
        self.assertEqual(checkpoint['contract'], contract)
        self.assertEqual(checkpoint['selected_epoch'], evidence['selected_epoch'])
        self.assertFalse(report['selected_metrics']['undefined_auc_targets'])
        self.assertNotIn('private', json.dumps(report))
        self.assertEqual({k: control.RECIPE[k] for k in ('seed', 'lr', 'weight_decay', 'batch_size', 'clip_grad_norm', 'max_epochs', 'patience', 'improvement_tolerance')},
            {'seed': 42, 'lr': .001, 'weight_decay': .02, 'batch_size': 64, 'clip_grad_norm': 5,
             'max_epochs': 80, 'patience': 12, 'improvement_tolerance': 1e-5})

    def test_deadline_interrupts_setup_no_fit_and_cannot_overwrite(self):
        out = self.root / 'budget'
        with patch.object(control, 'RECIPE', {**control.RECIPE, 'max_seconds': .05}), \
             patch.object(control, 'load_inputs', side_effect=lambda *a: time.sleep(.5)), \
             patch.object(control, 'fit', side_effect=AssertionError('must not fit')):
            started = time.monotonic(); report = control.run(self.plan_path, self.base, self.sealed_path, out)
        self.assertEqual(report['status'], 'BUDGET_STOP'); self.assertFalse(report['comparison_complete'])
        self.assertLess(time.monotonic() - started, .4)
        self.assertEqual(report['successful_optimizer_steps'], 0)
        before = (out / 'training-report.json').read_bytes()
        with self.assertRaises(FileExistsError): control.run(self.plan_path, self.base, self.sealed_path, out)
        self.assertEqual(before, (out / 'training-report.json').read_bytes())

    def test_per_batch_deadline_stops_before_next_update(self):
        data = self.load(); out = self.root / 'partial'; out.mkdir()
        report = {'history': [], 'successful_optimizer_steps': 0}
        # Entry, first batch begin/end; then the second batch must stop.
        with patch.object(control, 'RECIPE', {**control.RECIPE, 'batch_size': 2}), \
             patch.object(control, 'check_time', side_effect=[None, None, None, TimeoutError('budget')]):
            with self.assertRaises(TimeoutError):
                control.fit(data, out, {'fingerprint': 'synthetic'}, time.monotonic() + 10, report)
        self.assertEqual(report['successful_optimizer_steps'], 1)
        self.assertFalse((out / 'best.pt').exists())


if __name__ == '__main__':
    unittest.main()
