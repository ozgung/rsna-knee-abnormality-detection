import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from knee.audit import audit
from knee.device import select_device
from knee.submission import write_submission


class DeviceTests(unittest.TestCase):
    def test_auto_priorities_and_explicit_failure(self):
        for cuda, mps, expected in [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]:
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                                    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))
            with patch("knee.device.importlib.import_module", return_value=torch):
                self.assertEqual(select_device(), expected)
                if not cuda:
                    with self.assertRaises(RuntimeError):
                        select_device("cuda")

    def test_missing_torch_and_invalid_device(self):
        with patch("knee.device.importlib.import_module", side_effect=ModuleNotFoundError(name="torch")):
            self.assertEqual(select_device(), "cpu")
            with self.assertRaises(RuntimeError):
                select_device("mps")
        with self.assertRaises(ValueError):
            select_device("typo")


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.sample = Path(self.temp.name) / "sample.csv"
        self.output = Path(self.temp.name) / "output.csv"
        self.header = ["id", *[f"t{i}" for i in range(12)]]
        with self.sample.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.header)
            writer.writerows([["b", *([0] * 12)], ["a", *([0] * 12)]])

    def test_order_headers_and_no_overwrite(self):
        write_submission(self.sample, {"a": [0.1] * 12, "b": [0.9] * 12}, self.output)
        with self.output.open(newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], self.header)
        self.assertEqual([row[0] for row in rows[1:]], ["b", "a"])
        self.assertEqual(float(rows[1][1]), 0.9)
        with self.assertRaises(FileExistsError):
            write_submission(self.sample, {"a": [0.1] * 12, "b": [0.9] * 12}, self.output)

    def test_rejects_invalid_predictions_before_writing(self):
        for values in [[float("nan")] * 12, [float("inf")] * 12, [-0.1] * 12, [1.1] * 12, [0.5] * 11]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                write_submission(self.sample, {"a": values, "b": [0.5] * 12}, self.output)
            self.assertFalse(self.output.exists())
        with self.assertRaises(ValueError):
            write_submission(self.sample, {"a": [0.5] * 12}, self.output)

    def test_csv_audit(self):
        result = audit(self.sample, "id")
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["unique_nonempty_ids"], 2)
        with self.assertRaises(ValueError):
            audit(self.sample, "missing_column")


if __name__ == "__main__":
    unittest.main()
