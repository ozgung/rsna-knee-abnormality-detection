import contextlib
import csv
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from knee.ensemble import validate_weights

try:
    import numpy as np
    import torch
    from knee.dicom import PREPROCESSING_VERSION, StudyLoadError, StudyVolume
    from knee.ensemble import PredictionHeads
    from knee.features import StudyHead
    from knee.runtime import infer
except ImportError:
    np = None


class WeightTests(unittest.TestCase):
    def test_finite_nonnegative_sum_to_one_required(self):
        self.assertEqual(validate_weights(2), (.5, .5))
        self.assertAlmostEqual(math.fsum(validate_weights(3)), 1)
        for count, weights in ((0, []), (2, [1]), (2, [-.1, 1.1]), (2, [.2, .2]),
                               (2, [float("nan"), 1]), (2, [float("inf"), 0])):
            with self.subTest(count=count, weights=weights), self.assertRaises(ValueError):
                validate_weights(count, weights)

    def test_runtime_import_does_not_load_ml_dependencies(self):
        code = "import sys; import knee.runtime, knee.ensemble; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        subprocess.run([sys.executable, "-S", "-c", code], check=True,
                       cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)


@unittest.skipIf(np is None, "Optional ML dependencies unavailable")
class EnsembleTests(unittest.TestCase):
    def save_head(self, path, *, kind="attention", bias=.2, prior=.3, changes=None):
        model = StudyHead(np.zeros(8), np.ones(8), [prior] * 12, hidden=4, kind=kind).eval()
        with torch.no_grad():
            model.out.weight.zero_()
            model.out.bias.fill_(torch.logit(torch.tensor(bias)))
        checkpoint = {"state_dict": model.state_dict(), "targets": [f"t{i}" for i in range(12)],
                      "config": {"kind": kind, "hidden": 4, "dropout": .2}, "feature_sha256": "a" * 64,
                      "preprocessing": {"preprocessing_version": PREPROCESSING_VERSION,
                                        "image_size": 32, "slices_per_plane": 2, "crop_mm": 160}}
        if changes:
            changes(checkpoint)
        torch.save(checkpoint, path)
        return model, checkpoint

    def test_probability_average_and_training_prior_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, checkpoint = self.save_head(root / "a.pt", bias=.2, prior=.1)
            self.save_head(root / "b.pt", kind="meanmax", bias=.6, prior=.9)
            heads = PredictionHeads([root / "a.pt", root / "b.pt"], checkpoint["targets"],
                                    PREPROCESSING_VERSION, device="cpu")
            features = torch.randn(2, 3, 2, 8)
            masks = torch.ones(2, 3, 2, dtype=torch.bool)
            masks[1] = False
            values = heads(features, masks)
            torch.testing.assert_close(values[0], torch.full((12,), .4, dtype=torch.float64))
            torch.testing.assert_close(values[1], torch.full((12,), .5, dtype=torch.float64))
            torch.testing.assert_close(values[1], heads.prior_probabilities)
            weighted = PredictionHeads([root / "a.pt", root / "b.pt"], checkpoint["targets"],
                                       PREPROCESSING_VERSION, device="cpu", weights=[.25, .75])
            torch.testing.assert_close(weighted(features, masks)[0], torch.full((12,), .5, dtype=torch.float64))
            torch.testing.assert_close(weighted.prior_probabilities, torch.full((12,), .7, dtype=torch.float64))

    def test_single_head_is_exactly_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.pt"
            original, checkpoint = self.save_head(path)
            heads = PredictionHeads([path], checkpoint["targets"], PREPROCESSING_VERSION, device="cpu")
            features, masks = torch.randn(2, 3, 2, 8), torch.ones(2, 3, 2, dtype=torch.bool)
            masks[1] = False
            with torch.no_grad():
                expected = original(features, masks).sigmoid()
            self.assertTrue(torch.equal(heads(features, masks), expected))
            self.assertEqual(heads(features, masks).dtype, expected.dtype)

    def test_mismatched_preprocessing_targets_and_features_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, checkpoint = self.save_head(root / "a.pt")
            mutations = [
                ("preprocessing", lambda c: c["preprocessing"].update(crop_mm=140)),
                ("target order", lambda c: c.update(targets=c["targets"][::-1])),
                ("feature caches", lambda c: c.update(feature_sha256="b" * 64)),
                ("feature caches", lambda c: c.pop("feature_sha256")),
                ("version", lambda c: c["preprocessing"].update(preprocessing_version="other")),
            ]
            for message, mutate in mutations:
                with self.subTest(message=message):
                    self.save_head(root / "b.pt", changes=mutate)
                    with self.assertRaisesRegex(ValueError, message):
                        PredictionHeads([root / "a.pt", root / "b.pt"], checkpoint["targets"],
                                        PREPROCESSING_VERSION, device="cpu")

    def test_nonfinite_head_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.pt"
            _, checkpoint = self.save_head(path, changes=lambda c: c["state_dict"]["out.bias"].fill_(float("nan")))
            heads = PredictionHeads([path], checkpoint["targets"], PREPROCESSING_VERSION, device="cpu")
            with self.assertRaisesRegex(ValueError, "invalid head probability"):
                heads(torch.zeros(1, 3, 2, 8), torch.ones(1, 3, 2, dtype=torch.bool))

    def test_inference_encodes_once_and_averages_fallback(self):
        # Real head math, mocked image/CUDA boundary: this is a CPU integration
        # check, not evidence of real decoding or a CUDA backend passing.
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            _, checkpoint = self.save_head(root / "a.pt", bias=.2, prior=.1)
            self.save_head(root / "b.pt", kind="meanmax", bias=.6, prior=.9)
            ids = ["synthetic_a", "synthetic_b", "synthetic_empty"]
            tables = {
                "sample_submission.csv": (["StudyInstanceUID", *checkpoint["targets"]], [[uid, *([.5] * 12)] for uid in ids]),
                "test.csv": (["StudyInstanceUID"], [[uid] for uid in ids]),
                "test_series.csv": (["StudyInstanceUID", "SeriesInstanceUID"], [[uid, f"series_{i}"] for i, uid in enumerate(ids)]),
            }
            for name, (header, rows) in tables.items():
                with (root / name).open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(header)
                    writer.writerows(rows)
            volume = StudyVolume(np.zeros((3, 2, 32, 32), np.float32), np.ones((3, 2), bool),
                                 np.ones(3, bool), {"counts": {}, "errors": {}, "missing_planes": 0})
            loader_results = [volume, volume, StudyLoadError({"errors": {"synthetic_empty": 1}})]
            real_constructor = PredictionHeads
            def cpu_heads(*args, **kwargs):
                return real_constructor(*args, **{**kwargs, "device": "cpu"})
            with patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.max_memory_allocated", return_value=0), \
                 patch.object(torch.Tensor, "cuda", lambda self: self), \
                 patch("knee.ensemble.PredictionHeads", side_effect=cpu_heads), \
                 patch("knee.dicom.load_study", side_effect=loader_results) as load, \
                 patch("knee.features.load_encoder") as encoder, \
                 patch("knee.features.encode_volume", return_value=np.zeros((3, 2, 8), np.float16)) as encode, \
                 patch("knee.experiment.environment", return_value={"device": "mocked CPU boundary"}):
                report = infer(root, "unused", root / "a.pt", root / "submission.csv",
                               additional_head_paths=[root / "b.pt"])
            self.assertEqual(load.call_count, 3)
            self.assertEqual(encode.call_count, 2)
            encoder.assert_called_once()
            self.assertEqual(report["head_count"], 2)
            self.assertEqual(report["head_weights"], [.5, .5])
            self.assertEqual(report["fallback_studies"], 1)
            with (root / "submission.csv").open() as handle:
                rows = list(csv.reader(handle))[1:]
            np.testing.assert_allclose([float(row[1]) for row in rows], [.4, .4, .5], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
