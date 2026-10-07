"""Local inference integration checks; model execution is CPU-only and synthetic."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None


@unittest.skipIf(np is None or torch is None, "Optional ML dependencies unavailable")
class InferenceTests(unittest.TestCase):
    def setUp(self):
        from knee.dicom import PREPROCESSING_VERSION
        from knee.features import StudyHead

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.targets = [f"target_{i}" for i in range(12)]
        self.ids = ["hidden_4", "hidden_0", "hidden_3", "hidden_1", "hidden_2"]
        self._csv("sample_submission.csv", ["StudyInstanceUID", *self.targets],
                  [[f"example_{i}", *([.5] * 12)] for i in range(3)])
        self._csv("test.csv", ["StudyInstanceUID"], [[uid] for uid in self.ids])
        self._csv("test_series.csv", ["StudyInstanceUID", "SeriesInstanceUID"],
                  [[uid, f"series_{i}"] for i, uid in enumerate(reversed(self.ids))])
        torch.manual_seed(42)
        head = StudyHead(np.zeros(768), np.ones(768), np.full(12, .4), hidden=8).eval()
        self.checkpoint = {
            "state_dict": head.state_dict(), "targets": self.targets,
            "config": {"kind": "attention", "hidden": 8},
            "preprocessing": {"preprocessing_version": PREPROCESSING_VERSION,
                              "image_size": 224, "slices_per_plane": 8, "crop_mm": 160},
        }
        self.head_path = self.root / "head.pt"
        torch.save(self.checkpoint, self.head_path)

    def _csv(self, name, columns, rows):
        with (self.root / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)

    def test_hidden_ids_replace_example_ids_with_masked_repeat(self):
        from knee.dicom import StudyVolume
        from knee.runtime import infer

        mask = np.zeros((3, 8), bool)
        mask[0, :2] = True
        mask[1, 0] = True
        volume = StudyVolume(np.zeros((3, 8, 1, 1), np.float32), mask, mask.any(1),
                             {"counts": {"slices_decoded": 3}, "errors": {}, "missing_planes": 1})
        first = np.ones((3, 8, 768), np.float16)
        second = first.copy()
        second[~mask] = 10000  # Padding must not change repeated predictions.
        output = self.root / "submission.csv"
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("torch.cuda.is_available", return_value=True))
            stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=0))
            stack.enter_context(patch.object(torch.Tensor, "cuda", lambda tensor, *a, **kw: tensor))
            stack.enter_context(patch.object(torch.nn.Module, "cuda", lambda model, *a, **kw: model))
            stack.enter_context(patch("knee.features.load_encoder", return_value=object()))
            stack.enter_context(patch("knee.features.encode_volume",
                                      side_effect=[first] * len(self.ids) + [second] * len(self.ids)))
            loader = stack.enter_context(patch("knee.dicom.load_study", return_value=volume))
            stack.enter_context(patch("knee.experiment.environment", return_value={"device": "mocked CPU"}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            report = infer(self.root, self.root, self.head_path, output, repeat=2)
        with output.open(newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["StudyInstanceUID", *self.targets])
        self.assertEqual([row[0] for row in rows[1:]], self.ids)
        probabilities = np.asarray([row[1:] for row in rows[1:]], dtype=float)
        self.assertEqual(probabilities.shape, (len(self.ids), 12))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertEqual([call.args[1] for call in loader.call_args_list], self.ids * 2)
        self.assertEqual(report["studies"], len(self.ids))
        self.assertEqual(report["fallback_studies"], 0)
        self.assertFalse((self.root / "runtime-template.csv").exists())

    def test_checkpoint_target_order_mismatch_fails_before_encoding(self):
        from knee.runtime import infer

        self.checkpoint["targets"] = self.targets[::-1]
        torch.save(self.checkpoint, self.head_path)
        output = self.root / "submission.csv"
        with patch("torch.cuda.is_available", return_value=True), \
                patch("knee.features.load_encoder") as encoder:
            with self.assertRaisesRegex(ValueError, "Checkpoint target order mismatch"):
                infer(self.root, self.root, self.head_path, output)
        encoder.assert_not_called()
        self.assertFalse(output.exists())
