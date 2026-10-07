import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
    from knee.compare import compare, select_candidates, weak_view, weighted_bce
    from knee.experiment import train
    from knee.features import masked_loss
    from knee.validation import file_sha256
except ImportError:
    np = None


@unittest.skipIf(np is None, "Optional ML dependencies unavailable")
class ComparisonTests(unittest.TestCase):
    def test_bce_matches_training_and_ignores_unknown_labels(self):
        y = np.array([[1., .5, 0.1], [0., .5, .9]], np.float32)
        logits = torch.tensor([[2., 50., -1.], [-2., -50., 1.]])
        expected = float(masked_loss(logits, torch.from_numpy(y), torch.from_numpy(abs(y - .5) * 2)))
        actual = weighted_bce(y, logits.double().sigmoid().numpy())
        self.assertAlmostEqual(actual, expected, places=6)
        with self.assertRaises(ValueError):
            weighted_bce(np.full((2, 3), .5), np.full((2, 3), .6))
        with self.assertRaises(ValueError):
            weighted_bce(y, np.full((2, 3), np.nan))

    def test_equal_averaging_selected_only_by_weak_bce(self):
        truth = np.array([[1.], [0.]])
        predictions = {"baseline": np.array([[.9], [.6]]), "candidate": np.array([[.4], [.1]])}
        comparisons, selected = select_candidates(truth, predictions, ensembles=True)
        self.assertEqual(selected["members"], ["baseline", "candidate"])
        self.assertEqual(selected["weights"], [.5, .5])
        self.assertLess(selected["weak_bce"], comparisons[0]["weak_bce"])
        # Exact ties preserve the original candidate and avoid unnecessary heads.
        _, selected = select_candidates(truth, {"baseline": predictions["baseline"], "same": predictions["baseline"]}, ensembles=True)
        self.assertEqual(selected["members"], ["baseline"])

    def make_cache(self, root, *, include_gold=True):
        rng = np.random.default_rng(13)
        n = 28
        masks = np.ones((n, 3, 2), bool)
        masks[20] = False
        targets = np.array([f"target_{i}" for i in range(12)])
        arrays = dict(features=rng.normal(size=(n, 3, 2, 8)).astype(np.float16), masks=masks,
                      weak=rng.choice([0., .1, .5, .9, 1.], (n, 12)).astype(np.float32),
                      ids=np.array([f"synthetic_{i}" for i in range(n)]), targets=targets,
                      splits=np.array(["train"]*16 + ["weak_validation"]*8 + ["gold"]*4),
                      groups=np.array([f"group_{i}" for i in range(n)]))
        if include_gold:
            gold = np.full((n, 12), np.nan, np.float32)
            gold[24:] = np.array([[0]*12, [1]*12, [0]*12, [1]*12])
            arrays["gold"] = gold
        cache = root / "features.npz"
        np.savez_compressed(cache, **arrays)
        extraction = root / "extraction.json"
        extraction.write_text(json.dumps({"features_sha256": file_sha256(cache), "config": {"synthetic": True}}))
        return cache, extraction

    def test_selection_inputs_do_not_read_gold_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            cache, _ = self.make_cache(Path(directory), include_gold=False)
            view = weak_view(cache)
            self.assertEqual(len(view["ids"]), 7)  # Empty weak-validation study omitted.
            self.assertNotIn("gold", view)
            with np.load(cache) as data:
                arrays = {key: data[key] for key in data.files}
            arrays["groups"][24] = arrays["groups"][0]
            np.savez_compressed(cache, **arrays)
            with self.assertRaisesRegex(ValueError, "leakage"):
                weak_view(cache)

    def test_failed_candidates_cannot_replace_baseline(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            cache, extraction = self.make_cache(root)
            baseline = root / "baseline"
            train(cache, extraction, baseline, device="cpu", max_epochs=1, max_seconds=30)
            with patch("knee.compare.train", side_effect=RuntimeError("synthetic failure")):
                result = compare(cache, extraction, baseline, root / "failed_comparison", device="cpu",
                                 max_epochs=1, max_seconds=30, ensembles=True)
            self.assertEqual(result["status"], "completed_with_failures")
            self.assertEqual(result["selected"]["members"], ["baseline"])
            self.assertEqual(len(result["comparisons"]), 1)
            self.assertTrue(all(info["status"] == "failed" for name, info in result["candidates"].items() if name != "baseline"))

    def test_complete_cpu_run_preserves_baseline_and_uses_same_cache(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            cache, extraction = self.make_cache(root)
            baseline = root / "baseline"
            train(cache, extraction, baseline, device="cpu", max_epochs=2, max_seconds=30)
            before = {p.name: file_sha256(p) for p in baseline.iterdir()}
            result = compare(cache, extraction, baseline, root / "comparison", device="cpu", max_epochs=2,
                             patience=2, max_seconds=30, ensembles=True)
            self.assertEqual(result["status"], "completed")
            self.assertFalse(result["selection_uses_gold"])
            self.assertEqual(len(result["comparisons"]), 11)  # 4 singles + 6 pairs + all 4.
            self.assertEqual(before, {p.name: file_sha256(p) for p in baseline.iterdir()})
            self.assertEqual(set(result["candidates"]), {"baseline", "attention_seed43", "attention_seed44", "meanmax_seed42"})
            persisted = json.loads((root / "comparison/comparison.json").read_text())
            self.assertEqual(persisted["selected"], result["selected"])
            with np.load(root / "comparison/selected_weak_predictions.npz") as data:
                self.assertAlmostEqual(weighted_bce(weak_view(cache)["truth"], data["predictions"]), result["selected"]["weak_bce"])
            with self.assertRaises(FileExistsError):
                compare(cache, extraction, baseline, root / "comparison", device="cpu", max_epochs=2, max_seconds=30)
            broken = json.loads(extraction.read_text())
            broken["features_sha256"] = "wrong"
            extraction.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "does not match"):
                compare(cache, extraction, baseline, root / "new", device="cpu", max_epochs=2, max_seconds=30)


if __name__ == "__main__":
    unittest.main()
