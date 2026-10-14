import csv
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT', '/nonexistent'))
DIAG = EVIDENCE / 'outputs/kaggle/own224-window-samerun-20261012-result-v1/own224_window_diagnostic'
spec = importlib.util.spec_from_file_location('eval_r64', ROOT / 'scripts/evaluate_raptor64_endpoint.py')
evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)


@unittest.skipUnless((DIAG / 'endpoint-private.npz').is_file() and importlib.util.find_spec('numpy'), 'Pinned endpoint evidence required')
class EndpointTests(unittest.TestCase):
    def write(self, path, ids, p, rows=None):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['StudyInstanceUID', *evaluator.smoke.TARGETS, 'set'])
            for uid, row in zip(ids, p):
                w.writerow([uid, *row.tolist(), 'val'])
            for extra in rows or []:
                w.writerow(extra)

    def test_control_copy_has_zero_delta_and_contrasts_and_gates(self):
        import numpy as np
        c = np.load(DIAG / 'k42-private.npz'); ids = c['ids'].astype(str); p = c['probabilities'].astype(np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            same = Path(tmp) / 'same.csv'; self.write(same, ids, p)
            noisy = Path(tmp) / 'noisy.csv'; self.write(noisy, ids, np.clip(p + np.random.default_rng(0).normal(0, .05, p.shape), 0, 1))
            r = evaluator.evaluate(DIAG / 'endpoint-private.npz', DIAG / 'k42-private.npz', {'in-orig': same, 'in-plane': noisy, 'dr-orig': same})
            self.assertAlmostEqual(r['control']['macro_auc'], evaluator.CONTROL_MACRO, places=12)
            self.assertAlmostEqual(r['arms']['in-orig']['delta_vs_own44_k42'], 0.0, places=12)
            self.assertNotIn('sanity_gate', r['arms']['dr-orig']); self.assertNotIn('hypothesis_gate', r['arms']['in-orig'])
            self.assertIn('in-plane minus in-orig', r['contrasts']); self.assertLess(r['contrasts']['in-plane minus in-orig']['delta'], 0)
            self.assertEqual(r['groups']['clusters'], 843); self.assertAlmostEqual(r['prevalence_baseline']['macro_auc'], 0.5)
            bad = Path(tmp) / 'bad.csv'; self.write(bad, ids[::-1], p)
            with self.assertRaises(ValueError):
                evaluator.evaluate(DIAG / 'endpoint-private.npz', DIAG / 'k42-private.npz', {'in-orig': bad})
            gold = Path(tmp) / 'gold.csv'; self.write(gold, ids, p, rows=[['gold-row', *[0.5] * 12, 'gold']])
            with self.assertRaises(ValueError):
                evaluator.evaluate(DIAG / 'endpoint-private.npz', DIAG / 'k42-private.npz', {'in-orig': gold})


if __name__ == '__main__':
    unittest.main()
