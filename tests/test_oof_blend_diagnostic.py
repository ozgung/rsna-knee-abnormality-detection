import unittest, importlib.util, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("oofbd", Path(__file__).resolve().parents[1] / "scripts" / "oof_blend_diagnostic.py")
try:
    import numpy, pandas  # noqa: F401
except ImportError:
    numpy = None

@unittest.skipIf(numpy is None, "numpy/pandas not installed")
class AucTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.m)

    def test_rank_auc_matches_hand_computed_values_with_ties(self):
        auc = self.m.auc
        self.assertEqual(auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertEqual(auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)     # all tied -> 0.5
        self.assertEqual(auc([0, 0, 1, 1], [0.2, 0.3, 0.3, 0.9]), 0.875)   # one cross-class tie counts half
        self.assertIsNone(auc([1, 1], [0.2, 0.3]))                          # undefined without both classes

    def test_report_masks_and_undefined_targets(self):
        import numpy as np
        m = self.m
        y = np.zeros((4, 12), int); y[2:, 0] = 1; y[0, 1] = 1; y[3, 1] = 1
        p = np.linspace(0, 1, 48).reshape(4, 12)
        mask = np.ones((4, 12)); mask[:, 1] = 0; mask[3, 1] = 1
        r = m.report(y, p, mask)
        self.assertEqual(r["per_target"]["ACL"], 1.0)
        self.assertIn("MCL", r["undefined_targets"])   # masked to one positive only
        self.assertEqual(r["n_defined"], 1)

if __name__ == "__main__":
    unittest.main()
