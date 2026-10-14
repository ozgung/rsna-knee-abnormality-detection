import importlib.util, sys, unittest
from pathlib import Path

try:
    import numpy as np, pandas, sklearn, scipy  # noqa: F401
except ImportError:
    np = None

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stacking_e4.py"


@unittest.skipIf(np is None, "numpy/pandas/sklearn/scipy not installed")
class StackingE4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("stacking_e4", SCRIPT)
        cls.m = importlib.util.module_from_spec(spec)
        sys.modules["stacking_e4"] = cls.m
        spec.loader.exec_module(cls.m)

    def synth(self, n=500, seed=0):
        rng = np.random.default_rng(seed)
        nl = len(self.m.LAB)
        fold = np.arange(n) % 5
        z = rng.normal(size=(n, nl))
        Y = (z + rng.normal(scale=0.7, size=(n, nl)) > 0.3).astype(int)
        M = (rng.random((n, nl)) > 0.15).astype(float)
        P = 1 / (1 + np.exp(-(z + rng.normal(scale=0.5, size=(n, nl)))))
        return fold, Y, M, P

    def test_training_rows_never_include_heldout_fold_and_respect_mask(self):
        fold, Y, M, P = self.synth()
        seen = []
        X = self.m.feats_logit([P])
        self.m.nested_stack(X, Y, M, fold, "lr", 0.1, record=lambda k, t, tr: seen.append((k, t, tr.copy())))
        self.assertEqual(len(seen), 5 * len(self.m.LAB))
        for k, t, tr in seen:
            self.assertFalse(np.any(fold[tr] == k), "stacker trained on its own fold")
            self.assertTrue(np.all(M[tr, t] > 0), "unmasked-label rows used for training")
            self.assertEqual(set(np.unique(fold[tr])), set(range(5)) - {k})

    def test_heldout_predictions_are_invariant_to_heldout_labels_and_masks(self):
        fold, Y, M, P = self.synth(seed=1)
        X = self.m.feats_logit([P])
        base = self.m.nested_stack(X, Y, M, fold, "lr", None, grid=(0.01, 0.3))
        k = 2
        Y2, M2 = Y.copy(), M.copy()
        rng = np.random.default_rng(5)
        Y2[fold == k] = rng.integers(0, 2, Y2[fold == k].shape)
        M2[fold == k] = 1.0
        alt = self.m.nested_stack(X, Y2, M2, fold, "lr", None, grid=(0.01, 0.3))
        np.testing.assert_allclose(base[fold == k], alt[fold == k])          # fold k scores do not see fold k labels
        self.assertFalse(np.allclose(base[fold != k], alt[fold != k]))         # other folds legitimately do see them

    def test_inner_selection_never_reads_heldout_fold(self):
        fold, Y, M, P = self.synth(seed=2)
        X = self.m.feats_logit([P])
        trall = np.where(fold != 0)[0]
        c1 = self.m.inner_select_C(X, Y, M, fold, trall, 3, (0.003, 0.03, 0.3))
        Y2 = Y.copy(); Y2[fold == 0] = 1 - Y2[fold == 0]
        c2 = self.m.inner_select_C(X, Y2, M, fold, trall, 3, (0.003, 0.03, 0.3))
        self.assertEqual(c1, c2)

    def test_stacker_removes_planted_crosstalk(self):
        # head B = signal B + 0.9 * signal A, head A clean: stacking on both outputs should beat identity B
        rng = np.random.default_rng(3)
        n, nl = 3000, len(self.m.LAB)
        fold = np.arange(n) % 5
        a, b = rng.normal(size=n), rng.normal(size=n)
        Y = np.zeros((n, nl), int); Y[:, 4] = a + rng.normal(size=n) * .5 > 0; Y[:, 5] = b + rng.normal(size=n) * .5 > 0
        L = rng.normal(size=(n, nl)) * 0.1
        L[:, 4] = a + .05 * rng.normal(size=n); L[:, 5] = b + 0.9 * a + .05 * rng.normal(size=n)
        P = 1 / (1 + np.exp(-L)); M = np.ones((n, nl))
        S = self.m.nested_stack(self.m.feats_logit([P]), Y, M, fold, "lr", 1.0)
        ident = self.m.auc(Y[:, 5], P[:, 5]); stacked = self.m.auc(Y[:, 5], S[:, 5])
        self.assertGreater(stacked, ident + 0.02)

    def test_fast_auc_matches_reference_and_bootstrap_is_seeded(self):
        fold, Y, M, P = self.synth(seed=4)
        res = self.m.fast_per_target(Y, M, {"p": P}, np.arange(len(Y)))["p"]
        ref = self.m.report(Y, P, M)["per_target"]
        for j, t in enumerate(self.m.LAB):
            self.assertAlmostEqual(res[j], ref[t], places=10)
        Q = P + np.random.default_rng(0).normal(scale=0.05, size=P.shape)
        r1 = self.m.paired_bootstrap(Y, M, {"p": P, "q": Q}, {"d": ("q", "p")}, draws=20, seed=42)
        r2 = self.m.paired_bootstrap(Y, M, {"p": P, "q": Q}, {"d": ("q", "p")}, draws=20, seed=42)
        self.assertEqual(r1, r2)

    def test_load_val_drops_gold_and_flags_duplicates(self):
        import tempfile, pandas as pd
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for k in range(5):
                df = pd.DataFrame({"StudyInstanceUID": [f"s{k}a", f"s{k}b", "gold1"], "set": ["val", "val", "gold"]})
                for t in self.m.LAB: df[t] = 0.5
                p = Path(d) / f"oof_fold{k}.csv"; df.to_csv(p, index=False); paths.append(p)
            v = self.m.load_val(paths)
            self.assertEqual(len(v), 10); self.assertNotIn("gold1", v.index)
            self.assertEqual(sorted(v["fold"].unique()), [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
