"""Synthetic checks for the s6v2 lane helpers: targets schema/ID-coverage assertions and the npy header finaliser."""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "s6v2" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CT = load("check_targets")
L = CT.LABELS


def write_case(d, n_pool=8, gold_target=False, quarantine_target=False):
    ids = ["s%02d" % i for i in range(n_pool + 2 + 4)]
    gold = set(ids[n_pool:n_pool + 2])
    quar = set(ids[n_pool + 2:])
    with open(d / "train.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["StudyInstanceUID", "Report"] + L)
        for u in ids:
            w.writerow([u, "x"] + (["0"] * 12 if u in gold else [""] * 12))
    folds = {u: (-1 if u in gold else i % 5) for i, u in enumerate(ids)}
    json.dump({"folds": folds}, open(d / "folds.json", "w"))
    with open(d / "t.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CT.COLS)
        for u in ids:
            g, q = u in gold, u in quar
            pri = ["0.5"] * 12 if (not g and not q) or (g and gold_target) or (q and quarantine_target) else [""] * 12
            w.writerow([u, "1" if g else "0", "1" if q else "0", str(folds[u])] + pri + [""] * 12)
    return gold


class TargetsCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_check(self):
        return CT.check(self.d / "train.csv", self.d / "folds.json", self.d / "t.csv", n_rows=14, n_gold=2, n_quarantined=4)

    def test_valid_case_passes(self):
        write_case(self.d)
        r = self.run_check()
        self.assertEqual((r["pool"], r["gold_blank"], r["quarantined_blank"], r["gold_in_pool"]), (8, 2, 4, 0))

    def test_gold_target_rejected(self):
        write_case(self.d, gold_target=True)
        with self.assertRaises(AssertionError):
            self.run_check()

    def test_quarantine_target_rejected(self):
        write_case(self.d, quarantine_target=True)
        with self.assertRaises(AssertionError):
            self.run_check()


class NpyFinalize(unittest.TestCase):
    def test_truncates_and_rewrites_header(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        mod = load("vm_build_cache")
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "m.npy"
            mm = np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(1000, 12))
            mm[:7] = np.arange(84, dtype=np.float32).reshape(7, 12)
            mm.flush(); del mm
            mod.finalize_npy(p, 7)
            out = np.load(p)
            self.assertEqual(out.shape, (7, 12))
            self.assertTrue((out == np.arange(84, dtype=np.float32).reshape(7, 12)).all())


if __name__ == "__main__":
    unittest.main()
