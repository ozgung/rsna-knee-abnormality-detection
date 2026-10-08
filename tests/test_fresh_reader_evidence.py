"""Private selected-prediction alignment and diagnostic-only training evidence."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from knee.fresh_reader import (fingerprint, record_training_step, restore_best_evidence,
    training_diagnostic_indices, training_trace_summary, weak_validation_evidence, write_best_evidence)
from knee.validation import file_sha256


def fixture():
    plan = {'targets': ['z-target', 'a-target'], 'records': [
        {'uid': 'train', 'split': 'train', 'labels': [.1, .9]},
        {'uid': 'z-weak', 'split': 'weak_validation', 'labels': [.9, .5]},
        {'uid': 'excluded-gold', 'split': 'gold', 'labels': None},
        {'uid': 'a-weak', 'split': 'weak_validation', 'labels': [.2, .8]}]}
    plan['fingerprint'] = fingerprint(plan)
    probabilities = [[.8, .45], [.15, .7]]
    logits = [[math.log(p / (1 - p)) for p in row] for row in probabilities]
    evidence = weak_validation_evidence(plan, [1, 3], probabilities, logits=logits, epoch=2, weak_bce=.42)
    payload = {'plan_fingerprint': plan['fingerprint'], 'targets': plan['targets'],
               'epoch': 2, 'weak_bce': .42, 'weak_validation_evidence': evidence}
    return plan, probabilities, evidence, payload


class EvidenceCoreTests(unittest.TestCase):
    def test_predictions_keep_exact_plan_order_soft_labels_and_selected_metadata(self):
        plan, probabilities, evidence, _ = fixture()
        self.assertEqual(evidence['ids'], ['z-weak', 'a-weak'])
        self.assertEqual(evidence['targets'], ['z-target', 'a-target'])
        self.assertEqual(evidence['plan_record_indices'], [1, 3])
        self.assertEqual(evidence['weak_labels'], [[.9, .5], [.2, .8]])
        self.assertEqual(evidence['probabilities'], probabilities)
        self.assertEqual(evidence['plan_fingerprint'], plan['fingerprint'])
        self.assertEqual(evidence['selected_epoch'], 2)
        self.assertEqual(evidence['selected_weak_bce'], .42)
        self.assertEqual(evidence['fingerprint'], fingerprint({k: v for k, v in evidence.items() if k != 'fingerprint'}))
        self.assertNotIn('excluded-gold', json.dumps(evidence))

    def test_wrong_subset_order_shape_range_or_selection_is_rejected(self):
        plan, probabilities, evidence, _ = fixture()
        for indices, values, epoch, score in [([3, 1], probabilities, 2, .42),
                ([1], probabilities[:1], 2, .42), ([0, 1, 3], probabilities, 2, .42),
                ([1, 3], [[.8], [.1]], 2, .42), ([1, 3], [[float('nan'), .4], [.1, .8]], 2, .42),
                ([1, 3], [[1.1, .4], [.1, .8]], 2, .42), ([1, 3], probabilities, 0, .42),
                ([1, 3], probabilities, 2, float('nan'))]:
            with self.subTest(indices=indices, epoch=epoch), self.assertRaises(ValueError):
                weak_validation_evidence(plan, indices, values, logits=evidence['logits'], epoch=epoch, weak_bce=score)
        for invalid in ([[float('inf'), 0], [0, 0]], [[1], [2]], [[100, 100], [100, 100]]):
            with self.subTest(logits=invalid), self.assertRaises(ValueError):
                weak_validation_evidence(plan, [1, 3], probabilities, logits=invalid, epoch=2, weak_bce=.42)

    def test_sidecar_binds_checkpoint_bytes_and_rejects_relabeling_reordering_or_epoch_drift(self):
        plan, _, evidence, payload = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'best.pt').write_bytes(b'synthetic checkpoint bytes')
            receipt = write_best_evidence(root, plan, payload)
            sidecar = json.loads((root / receipt['file']).read_text())
            self.assertEqual(sidecar['evidence'], evidence)
            self.assertEqual(sidecar['best_checkpoint_sha256'], file_sha256(root / 'best.pt'))
            self.assertEqual(receipt['sha256'], file_sha256(root / receipt['file']))
            for key, value in [('ids', ['a-weak', 'z-weak']), ('weak_labels', [[0., 1.], [0., 1.]]),
                               ('selected_epoch', 3), ('selected_weak_bce', .4), ('targets', ['a-target', 'z-target'])]:
                changed = copy.deepcopy(payload)
                changed['weak_validation_evidence'][key] = value
                changed['weak_validation_evidence']['fingerprint'] = fingerprint({
                    k: v for k, v in changed['weak_validation_evidence'].items() if k != 'fingerprint'})
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'differs'):
                    write_best_evidence(root, plan, changed)

    def test_diagnostic_training_cohort_is_bounded_label_blind_and_partition_safe(self):
        rows = [{'uid': f'train-{i}', 'split': 'train', 'labels': [.1]} for i in range(80)]
        rows += [{'uid': 'weak', 'split': 'weak_validation'}, {'uid': 'gold', 'split': 'gold'}]
        first = [rows[i]['uid'] for i in training_diagnostic_indices(rows)]
        changed = [row | {'labels': [.9]} for row in reversed(rows)]
        second = [changed[i]['uid'] for i in training_diagnostic_indices(changed)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertTrue(all(uid.startswith('train-') for uid in first))
        self.assertEqual(len(training_diagnostic_indices(rows[:7])), 7)
        with self.assertRaises(ValueError):
            training_diagnostic_indices(rows, 65)

    def test_partial_trace_roundtrip_preserves_success_overflow_lr_and_batch_mean(self):
        trace = {'successful_optimizer_steps': 0, 'amp_overflow_skips': 0,
                 'online_training_batches': 0, 'online_loss_sum': 0., 'lr_start': [1e-4, 5e-4]}
        record_training_step(trace, .8, overflow=True)
        record_training_step(trace, .6, overflow=False)
        resumed = json.loads(json.dumps(trace))
        record_training_step(resumed, .4, overflow=False)
        summary = training_trace_summary(resumed, [5e-5, 2.5e-4])
        self.assertEqual(summary['successful_optimizer_steps'], 2)
        self.assertEqual(summary['amp_overflow_skips'], 1)
        self.assertEqual(summary['online_training_batches'], 3)
        self.assertAlmostEqual(summary['online_training_batch_loss_mean'], .6)
        self.assertEqual(summary['lr_start'], [1e-4, 5e-4])
        self.assertEqual(summary['lr_end'], [5e-5, 2.5e-4])
        with self.assertRaises(ValueError):
            record_training_step(resumed, float('nan'), overflow=False)

    def test_training_notebook_has_only_cache_resume_and_skips_competition_resolution(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('evidence_builder_test', root / 'scripts/build_fresh_reader.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as directory:
            sources, out = Path(directory) / 'source', Path(directory) / 'notebook'
            sources.mkdir()
            for name in ('preprocess.py', 'knee.py'):
                (sources / name).write_text('pass\n')
            with patch.object(builder, 'check_sources'):
                builder.build(sources, out, 'synthetic-training', 'train', cache_kernel='ozgung/cache',
                              resume_kernel='ozgung/prior-training', plan_fingerprint='reviewed')
            metadata = json.loads((out / 'kernel-metadata.json').read_text())
            self.assertEqual(metadata['competition_sources'], [])
            self.assertEqual(metadata['dataset_sources'], [])
            self.assertEqual(metadata['model_sources'], [])
            self.assertEqual(metadata['kernel_sources'], ['ozgung/cache', 'ozgung/prior-training'])
            notebook = json.loads((out / 'notebook.ipynb').read_text())
            source = ''.join(notebook['cells'][1]['source'])
            compile(source, 'generated-training', 'exec')
            guarded = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.If)
                       and any(isinstance(child, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'data'
                           for t in child.targets) for child in node.body)]
            self.assertEqual(len(guarded), 1)
            resolve = Mock(side_effect=AssertionError('Training must not resolve competition data'))
            exec(compile(ast.Module(body=guarded, type_ignores=[]), 'generated-cache-only-setup', 'exec'),
                 {'STAGE': 'train', 'exact_root': resolve})
            resolve.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch unavailable')
class EvidenceResumeTests(unittest.TestCase):
    def test_saved_logits_preserve_finite_bce_when_sigmoid_saturates(self):
        import torch
        from knee.fresh_reader import loss_terms, macro_loss
        plan, _, _, _ = fixture()
        logits = torch.tensor([[1000., -1000.], [-20., 20.]])
        labels = torch.tensor([plan['records'][i]['labels'] for i in [1, 3]])
        numerator, support = loss_terms(logits, labels)
        loss = float(macro_loss(numerator, support))
        evidence = weak_validation_evidence(plan, [1, 3], logits.sigmoid().tolist(),
                                            logits=logits.tolist(), epoch=1, weak_bce=loss)
        self.assertEqual(evidence['probabilities'][0], [1., 0.])
        restored_n, restored_d = loss_terms(torch.tensor(evidence['logits']), torch.tensor(evidence['weak_labels']))
        self.assertEqual(float(macro_loss(restored_n, restored_d)), loss)
        self.assertTrue(math.isfinite(loss))

    def test_resume_reconstructs_missing_sidecar_and_preserves_selected_best_pair(self):
        import torch
        from knee.fresh_reader import save_checkpoint
        plan, _, evidence, payload = fixture()
        payload['model'] = {'weight': torch.ones(1)}
        with tempfile.TemporaryDirectory() as directory:
            prior, resumed = Path(directory) / 'prior', Path(directory) / 'resumed'
            prior.mkdir(); resumed.mkdir()
            save_checkpoint(prior / 'best.pt', payload)
            original = write_best_evidence(prior, plan, payload)
            (prior / original['file']).unlink()  # Partial output transfer can omit the private sidecar.
            receipt = restore_best_evidence(prior, resumed, plan, best_loss=.42,
                                            expected_fingerprint=evidence['fingerprint'])
            self.assertEqual(receipt, original)
            self.assertEqual(file_sha256(prior / 'best.pt'), file_sha256(resumed / 'best.pt'))
            self.assertEqual(json.loads((resumed / receipt['file']).read_text())['evidence'], evidence)
            for score, digest in [(.43, evidence['fingerprint']), (.42, None), (.42, 'f' * 64)]:
                with self.subTest(score=score), self.assertRaises(ValueError):
                    restore_best_evidence(prior, resumed, plan, best_loss=score, expected_fingerprint=digest)


if __name__ == '__main__':
    unittest.main()
