"""Synthetic-fixture tests for scripts/build_bootstrap_labels.py (no real data)."""
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:  # plain python3.12 without the ML environment
    np = pd = None

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@unittest.skipIf(np is None, "numpy/pandas not installed")
class BootstrapLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SCRIPTS))
        import build_bootstrap_labels as bbl
        cls.b = bbl
        cls.T = bbl.TARGETS

    def frames(self, n=6):
        T = self.T
        ids = [f"s{i}" for i in range(n)]
        base = pd.DataFrame(0.2, index=pd.Index(ids, name="StudyInstanceUID"), columns=T)
        mask = pd.DataFrame(1.0, index=base.index, columns=T)
        oof = pd.DataFrame(0.9, index=base.index, columns=T)
        return base, mask, oof

    def test_mixing_addressed_vs_silent_and_clip(self):
        T = self.T
        y = np.full((2, 12), 0.2)
        m = np.ones((2, 12)); m[1, :] = 0
        p = np.full((2, 12), 0.9)
        out = self.b.mix(y, m, p, 0.5, 0.8, (0.02, 0.98))
        np.testing.assert_allclose(out[0], 0.5 * 0.2 + 0.5 * 0.9)
        np.testing.assert_allclose(out[1], 0.2 * 0.2 + 0.8 * 0.9)
        self.assertNotEqual(out[0, 0], round(out[0, 0]))  # fractions kept
        hi = self.b.mix(np.ones((1, 12)), np.ones((1, 12)), np.ones((1, 12)), 0.5, 0.8, (0.02, 0.98))
        lo = self.b.mix(np.zeros((1, 12)), np.ones((1, 12)), np.zeros((1, 12)), 0.5, 0.8, (0.02, 0.98))
        np.testing.assert_allclose(hi, 0.98)
        np.testing.assert_allclose(lo, 0.02)

    def test_synovitis_prior_branches(self):
        k = self.b.SYN
        e = self.b.EFF
        y = np.full((4, 12), 0.5)
        y[:, e] = [0.1, 0.5, 0.9, 0.7]
        m = np.zeros((4, 12))  # all silent
        m[3, k] = 1  # addressed synovitis keeps its base value
        p = np.full((4, 12), 0.0)
        out = self.b.mix(y, m, p, 0.5, 0.8, (0.0, 1.0), synovitis_effusion_prior=True)
        np.testing.assert_allclose(out[:3, k], 0.2 * np.array([0.15, 0.45, 0.60]))
        np.testing.assert_allclose(out[3, k], 0.5 * 0.5)
        plain = self.b.mix(y, m, p, 0.5, 0.8, (0.0, 1.0))
        np.testing.assert_allclose(plain[:3, k], 0.2 * 0.5)
        np.testing.assert_allclose(out[:, e], plain[:, e])  # other targets untouched

    def test_duplicate_study_abort_and_fold_requirement(self):
        base, mask, oof = self.frames(4)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for k in (0, 1):
                df = oof.reset_index()
                df = df[df["StudyInstanceUID"].isin(["s0", "s1"] if k == 0 else ["s1", "s2"])].copy()  # s1 twice
                df["set"] = "val"
                df.to_csv(d / f"oof_fold{k}.csv", index=False)
            with self.assertRaises(ValueError):
                self.b.load_oof(self.b.expand_oof_paths([d]))
            (d / "oof_fold1.csv").unlink()
            with self.assertRaises(ValueError):
                self.b.load_oof(self.b.expand_oof_paths([d]), require_folds=5)
            val, gold, folds = self.b.load_oof(self.b.expand_oof_paths([d]))
            self.assertEqual(folds, [0])

    def test_gold_rows_averaged_and_excluded_from_output(self):
        base, mask, oof = self.frames(4)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for k, g in ((0, 0.2), (1, 0.6)):
                v = oof.reset_index().iloc[2 * k:2 * k + 2].copy(); v["set"] = "val"
                gr = pd.DataFrame({"StudyInstanceUID": ["g1"], "set": "gold"})
                for t in self.T:
                    gr[t] = g
                pd.concat([v, gr]).to_csv(d / f"oof_fold{k}.csv", index=False)
            val, gold, _ = self.b.load_oof(self.b.expand_oof_paths([d]), require_folds=2)
        np.testing.assert_allclose(gold.loc["g1"].to_numpy(), 0.4)
        self.assertEqual(len(val), 4)
        # base table that carries a gold row: it is dropped from the output
        base.loc["g1"] = 0.5
        mask.loc["g1"] = 1.0
        y1, y1s, info = self.b.build_tables(base, mask, val, ["g1"], 0.5, 0.8, (0.02, 0.98))
        self.assertNotIn("g1", y1.index)
        self.assertNotIn("g1", y1s.index)
        self.assertEqual(info["base_rows_dropped_as_gold"], 1)
        self.assertEqual(len(y1), 4)
        # a gold study among OOF val rows aborts
        bad = val.copy(); bad.loc["g1"] = 0.5
        with self.assertRaises(ValueError):
            self.b.build_tables(base, mask, bad, ["g1"], 0.5, 0.8, (0.02, 0.98))

    def test_partial_requires_flag(self):
        base, mask, oof = self.frames(5)
        part = oof.iloc[:3]
        with self.assertRaises(ValueError):
            self.b.build_tables(base, mask, part, [], 0.5, 0.8, (0.02, 0.98))
        y1, _, info = self.b.build_tables(base, mask, part, [], 0.5, 0.8, (0.02, 0.98), allow_partial=True)
        self.assertEqual(len(y1), 3)

    def test_silent_fraction_reported(self):
        base, mask, oof = self.frames(4)
        mask.iloc[:1, 0] = 0
        _, _, info = self.b.build_tables(base, mask, oof, [], 0.5, 0.8, (0.02, 0.98))
        self.assertAlmostEqual(info["silent_fraction_per_target"][self.T[0]], 0.25)
        self.assertEqual(info["silent_fraction_per_target"][self.T[1]], 0.0)

    def test_table_independent_of_gold_labels(self):
        """build_tables has no gold-label input, and the gold check returns metrics without touching tables."""
        import inspect
        params = set(inspect.signature(self.b.build_tables).parameters)
        self.assertEqual(params, {"base", "mask", "oof_val", "gold_ids", "alpha_addressed", "alpha_silent", "clip",
                                  "allow_partial"})
        rng = np.random.default_rng(0)
        gold_y = (rng.random((58, 12)) > 0.5).astype(float)
        gold_y[0, :] = 1; gold_y[1, :] = 0
        p = np.clip(gold_y * 0.6 + rng.random((58, 12)) * 0.4, 0, 1)
        base_g = np.full((58, 12), 0.3)
        m_g = np.ones((58, 12))
        res = self.b.teacher_side_check(p, base_g, m_g, gold_y, 0.5, 0.8, (0.02, 0.98))
        self.assertGreater(res["oof_on_gold"]["macro_auc"], 0.8)
        self.assertEqual(len(res["alpha_grid_descriptive_not_for_selection"]), 16)
        self.assertIn("NOT FOR SELECTION", res["note"])
        self.assertEqual(len(res["calibration_mixed"][self.T[0]]), 5)
        # gold values are used only as arguments of the check; no frame is returned
        self.assertFalse(any(isinstance(v, pd.DataFrame) for v in res.values()))

    def test_rank_auc_matches_benchmark_helper(self):
        import benchmark_label_tables as blt
        rng = np.random.default_rng(1)
        y = (rng.random(60) > 0.6).astype(float)
        p = np.round(rng.random(60), 1)  # ties
        self.assertAlmostEqual(self.b.rank_auc(p, y), blt.auc_pairwise(p, y))
        self.assertTrue(np.isnan(self.b.rank_auc(p, np.zeros(60))))


if __name__ == "__main__":
    unittest.main()
