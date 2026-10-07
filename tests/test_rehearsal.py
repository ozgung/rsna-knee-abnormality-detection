import contextlib
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from knee.rehearsal import rehearse, SERIES_FIELDS
from knee.validation import read_csv

try:
    import numpy as np
except ImportError:
    np = None


class RehearsalImportTests(unittest.TestCase):
    def test_import_needs_no_ml_dependencies(self):
        subprocess.run([sys.executable, "-S", "-c", "import sys; import knee.rehearsal; assert 'numpy' not in sys.modules; assert 'torch' not in sys.modules"],
                       cwd=Path(__file__).resolve().parents[1], capture_output=True, check=True)


@unittest.skipIf(np is None, "Optional NumPy unavailable")
class RehearsalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root, self.out = self.base / "data", self.base / "outputs"
        self.root.mkdir()
        (self.root / "train_series").mkdir()
        self.targets = [f"t{i}" for i in range(12)]
        self.ids = ["synthetic_gold_a", "synthetic_gold_b"]
        self.expected = np.array([[.2] * 12, [.8] * 12])
        self._csv(self.root / "sample_submission.csv", ["StudyInstanceUID", *self.targets], [["visible", *([.5] * 12)]])
        self._csv(self.root / "train.csv", ["StudyInstanceUID", "Report", *self.targets], [
            [self.ids[0], "private narrative alpha", *([0] * 12)],
            [self.ids[1], "private narrative beta", *([1] * 12)],
        ])
        self._csv(self.root / "train_series.csv", ["StudyInstanceUID", *SERIES_FIELDS, "Report", "PatientID", self.targets[0]], [
            [uid, f"series_{i}", "True", "True", "Sagittal", "private series narrative", "private patient", "1"]
            for i, uid in enumerate(self.ids)
        ])
        self.expected_path = self.base / "expected.npz"
        np.savez_compressed(self.expected_path, ids=np.array(self.ids), predictions=self.expected, targets=np.array(self.targets))
        self.heads = [self.base / "a.pt", self.base / "b.pt"]

    def _csv(self, path, columns, rows):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)

    def assert_gold_contract(self, path):
        self.assertTrue((path / "test_series").is_symlink())
        self.assertEqual((path / "test_series").resolve(), self.root / "train_series")
        self.assertEqual(set(p.name for p in path.iterdir()), {"test.csv", "sample_submission.csv", "test_series.csv", "test_series"})
        fields, rows = read_csv(path / "test.csv")
        self.assertEqual(fields, ["StudyInstanceUID"])
        self.assertEqual([row[fields[0]] for row in rows], self.ids)
        fields, rows = read_csv(path / "sample_submission.csv")
        self.assertEqual(fields, ["StudyInstanceUID", *self.targets])
        self.assertTrue(all(float(row[target]) == .5 for row in rows for target in self.targets))
        fields, _ = read_csv(path / "test_series.csv")
        self.assertEqual(fields, ["StudyInstanceUID", *SERIES_FIELDS])
        for metadata in path.glob("*.csv"):
            text = metadata.read_text()
            self.assertNotIn("Report", text)
            self.assertNotIn("PatientID", text)
            self.assertNotIn("private", text)

    def test_gold_metadata_is_image_only_and_selected_heads_reach_both_runs(self):
        calls = []
        def fake_infer(data, encoder, head, output, **kwargs):
            calls.append((data, head, kwargs))
            self.assertEqual(head, self.heads[0])
            self.assertEqual(kwargs["additional_head_paths"], self.heads[1:])
            self.assertEqual(kwargs["head_weights"], (.25, .75))
            if len(calls) == 1:
                self.assert_gold_contract(data)
                self.assertNotIn("repeat", kwargs)
                self._csv(output, ["StudyInstanceUID", *self.targets],
                          [[uid, *values] for uid, values in zip(self.ids, self.expected)])
            else:
                self.assertEqual(data, self.root)
                self.assertEqual(kwargs["repeat"], 2)
                self.assertFalse((self.out / "gold-rehearsal/test_series").exists())
            return {"status": "completed", "synthetic": True}
        with patch("knee.rehearsal.infer", side_effect=fake_infer), contextlib.redirect_stdout(io.StringIO()):
            result = rehearse(self.root, "encoder", self.heads, self.expected_path, self.out, head_weights=[.25, .75])
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["gold_prediction_max_error"], 0)
        self.assertEqual(result["gold_metrics"]["macro_all_targets_auc"], 1)
        self.assertTrue((self.out / "rehearsal.json").exists())
        self.assertFalse((self.out / "gold-rehearsal/test_series").is_symlink())
        with patch("knee.rehearsal.infer") as inference:
            with self.assertRaises(FileExistsError):
                rehearse(self.root, "encoder", self.heads, self.expected_path, self.out)
            inference.assert_not_called()

    def test_link_removed_when_inference_raises_and_labels_never_loaded(self):
        def failing_infer(data, *args, **kwargs):
            self.assert_gold_contract(data)
            raise RuntimeError("synthetic inference failure")
        with patch("knee.rehearsal.infer", side_effect=failing_infer) as inference, \
             patch("knee.rehearsal.read_csv", wraps=read_csv) as reader:
            with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
                rehearse(self.root, "encoder", self.heads, self.expected_path, self.out)
        inference.assert_called_once()
        self.assertFalse((self.out / "gold-rehearsal/test_series").exists())
        self.assertFalse((self.out / "gold-rehearsal/test_series").is_symlink())
        self.assertFalse((self.out / "visible-rehearsal").exists())
        self.assertFalse((self.out / "rehearsal.json").exists())
        self.assertNotIn("train.csv", [Path(call.args[0]).name for call in reader.call_args_list])

    def test_parity_failure_prevents_gold_evaluation_and_visible_run(self):
        def bad_predictions(data, encoder, head, output, **kwargs):
            self._csv(output, ["StudyInstanceUID", *self.targets], [[uid, *([.5] * 12)] for uid in self.ids])
            return {"status": "completed"}
        with patch("knee.rehearsal.infer", side_effect=bad_predictions) as inference, \
             patch("knee.rehearsal.read_csv", wraps=read_csv) as reader:
            with self.assertRaisesRegex(RuntimeError, "disagree"):
                rehearse(self.root, "encoder", [self.heads[0]], self.expected_path, self.out)
        inference.assert_called_once()
        self.assertFalse((self.out / "gold-rehearsal/test_series").is_symlink())
        self.assertNotIn("train.csv", [Path(call.args[0]).name for call in reader.call_args_list])


if __name__ == "__main__":
    unittest.main()
