"""Synthetic saved-result portability and fail-closed admission checks."""
import json
import tempfile
import unittest
from pathlib import Path

from tests import test_s6_solo_runtime as fixtures
from knee import s6_solo_rehearsal as producer
from knee import s6_solo_replay as replay


@unittest.skipUnless(fixtures.HAVE_CAPTURED, 'captured source and optional numpy/pandas are absent')
class HostReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg, self.k1 = fixtures.AssembleVerifyTests().cfg(), fixtures.load()
        self.root, self.receipt = fixtures.synthetic_result(self.temp.name, self.cfg, self.k1)
        self.path = self.root / 's6solo/receipt.json'
        self.receipt = json.loads(self.path.read_text())

    def save(self):
        self.path.write_text(json.dumps(self.receipt, sort_keys=True) + '\n')

    def test_exact_result_keeps_original_receipt_binding(self):
        before = self.path.read_bytes()
        checked = replay.verify_result(self.root, self.cfg, self.k1)
        self.assertEqual(checked['ledger_sha256'], self.receipt['ledger_sha256'])
        self.assertEqual(checked['host_replay']['differences'], [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_only_probability_diagnostic_roundoff_is_portable(self):
        self.receipt['sentinel']['comparisons'][0]['max_abs_probability'] += 1e-14
        self.save()
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'INDEPENDENT_RECEIPT_LEDGER_REPLAY'):
            producer.verify_result(self.root, self.cfg, self.k1)
        checked = replay.verify_result(self.root, self.cfg, self.k1)
        self.assertEqual(len(checked['host_replay']['differences']), 1)
        self.assertEqual(self.path.read_bytes(), before)

    def test_difference_below_scientific_gate_still_exceeds_replay_roundoff(self):
        self.receipt['sentinel']['comparisons'][0]['max_abs_probability'] += 1e-6
        self.save()
        with self.assertRaisesRegex(ValueError, 'SENTINEL_DIAGNOSTIC_NUMERIC_REPLAY'):
            replay.verify_result(self.root, self.cfg, self.k1)

    def test_other_diagnostics_remain_exact(self):
        self.receipt['sentinel']['comparisons'][0]['max_abs_logit'] += 1e-14
        self.save()
        with self.assertRaisesRegex(ValueError, 'INDEPENDENT_RECEIPT_LEDGER_REPLAY'):
            replay.verify_result(self.root, self.cfg, self.k1)

    def test_missing_input_hash_evidence_fails_recomputed_gate(self):
        for path in (self.root / 'k1/sentinel').glob('board*/px/px_*.jsonl'):
            path.unlink()
        with self.assertRaisesRegex(ValueError, 'PASSING_RECOMPUTED_REHEARSAL'):
            replay.verify_result(self.root, self.cfg, self.k1)

    def test_saved_artifact_binding_remains_required(self):
        resources = self.root / 's6solo/resources.json'
        resources.write_text(resources.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'SAVED_FILE_BINDINGS'):
            replay.verify_result(self.root, self.cfg, self.k1)


if __name__ == '__main__':
    unittest.main()
