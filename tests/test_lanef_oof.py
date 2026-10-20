"""Lane F: hash-4 fold targets and the assembly of own out-of-fold teacher columns (synthetic fixtures, no real data)."""
import csv, hashlib, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the dependency-free discover run skips the numpy parts
    np = None

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]


def hash_fold(uid):
    return int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 4


def write_targets(path, uids, gold, folds5=True):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(["StudyInstanceUID", "is_gold", "review_holdout", "fold"] + [l + "|prior" for l in LABELS] + [l + "|strict" for l in LABELS])
        for i, u in enumerate(uids):
            g = u in gold
            w.writerow([u, int(g), 0, -1 if g else (i % 5 if folds5 else hash_fold(u))] + ([""] * 12 if g else ["%.6f" % ((i * 7 % 13) / 13) for _ in LABELS]) + [""] * 12)


class LaneFTests(unittest.TestCase):
    def _need_numpy(self):
        if np is None:
            self.skipTest('numpy not installed')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.d = Path(self.tmp.name)
        self.uids = ["1.2.3.%d" % i for i in range(60)]
        self.gold = set(self.uids[:6]); self.pool = [u for u in self.uids if u not in self.gold]

    def tearDown(self):
        self.tmp.cleanup()

    def test_make_hash4_targets_rewrites_only_the_fold_column(self):
        src, dst = self.d / "in.csv", self.d / "out.csv"
        write_targets(src, self.uids, self.gold)
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/lanef/make_hash4_targets.py"), str(src), str(dst)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        a = list(csv.reader(open(src, newline=""))); b = list(csv.reader(open(dst, newline="")))
        self.assertEqual(len(a), len(b))
        for x, y in zip(a[1:], b[1:]):
            self.assertEqual(x[:3] + x[4:], y[:3] + y[4:])
            self.assertEqual(int(y[3]), -1 if x[1] == "1" else hash_fold(x[0]))
        self.assertIn(b"\r\n", dst.read_bytes())

    def _fold_dir(self, k, rng, z_all, g_all):
        fd = self.d / "raw" / ("f%d" % k); fd.mkdir(parents=True)
        fo = np.asarray([hash_fold(u) for u in self.pool])
        z = np.zeros((len(self.pool), 12), np.float32); z[fo == k] = z_all[fo == k]
        np.save(fd / "partial_oof_E_prior_s42.npy", z)
        json.dump({"arm": "E_prior", "seed": 42, "epochs": 12, "only_folds": [k], "holdout_fold": None, "labels": LABELS,
                   "gold_uids": sorted(self.gold), "gold_logits": (g_all[k] / 4).round(5).tolist(), "loss_first_last": [[0.51, 0.37]],
                   "train_seconds": 1.0, "optimizer_steps": 10}, open(fd / "partial_preds_E_prior_s42.json", "w"))
        json.dump(self.pool, open(fd / "pool_uids.json", "w"))
        json.dump({"folds": 4}, open(fd / "run_config.json", "w"))
        write_targets(fd / "targets.csv", self.uids, self.gold, folds5=False)

    def test_analyze_oof_assembles_fold_rows_and_undoes_the_gold_division(self):
        self._need_numpy()
        rng = np.random.default_rng(0)
        z_all = rng.normal(size=(len(self.pool), 12)).astype(np.float32)
        g_all = {k: rng.normal(size=(6, 12)) for k in range(4)}
        for k in range(4):
            self._fold_dir(k, rng, z_all, g_all)
        blend, train, laneb = self.d / "blend.csv", self.d / "train.csv", self.d / "laneb"; laneb.mkdir()
        with open(blend, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["StudyInstanceUID"] + LABELS)
            for u in self.pool:
                w.writerow([u] + ["%.3f" % v for v in rng.uniform(size=12)])
        with open(train, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["StudyInstanceUID", "Report"] + LABELS)
            for u in self.uids:
                w.writerow([u, "x"] + ([str(int(v)) for v in rng.integers(0, 2, 12)] if u in self.gold else [""] * 12))
        with open(laneb / "oof_columns.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["StudyInstanceUID", "fold"] + ["R6 flip|" + l for l in LABELS] + ["R7 plain|" + l for l in LABELS])
            for u in self.pool:
                w.writerow([u, hash_fold(u)] + ["%.4f" % v for v in rng.uniform(size=24)])
        out = self.d / "out"
        r = subprocess.run([sys.executable, str(ROOT / "scripts/lanef/analyze_oof.py"), "--raw", str(self.d / "raw"), "--out", str(out),
                            "--laneb", str(laneb), "--blend", str(blend), "--train", str(train)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = list(csv.DictReader(open(out / "oof_columns.csv")))
        self.assertEqual(len(rows), len(self.pool))
        for row in rows:
            i = self.pool.index(row["StudyInstanceUID"])
            self.assertEqual(int(row["fold"]), hash_fold(row["StudyInstanceUID"]))
            self.assertAlmostEqual(float(row["own logit|ACL"]), float(z_all[i, 0]), places=5)
        g = list(csv.DictReader(open(out / "gold58_columns.csv")))
        self.assertEqual(len(g), 6)
        want = np.mean([g_all[k] for k in range(4)], 0)
        for i, u in enumerate(sorted(self.gold)):
            self.assertAlmostEqual(float(g[i]["mean logit|ACL"]), float(want[i, 0]), places=3)
        s = json.load(open(out / "summary.json"))
        self.assertEqual(s["decision_table"]["n"], len(self.pool))
        self.assertIn("own - R6 flip", s["decision_table"]["paired_bootstrap"])
        self.assertTrue(s["gold58"]["exposed"])

    def test_analyze_oof_rejects_nonzero_rows_outside_the_fold(self):
        self._need_numpy()
        rng = np.random.default_rng(1)
        z_all = rng.normal(size=(len(self.pool), 12)).astype(np.float32)
        g_all = {k: rng.normal(size=(6, 12)) for k in range(4)}
        for k in range(4):
            self._fold_dir(k, rng, z_all, g_all)
        bad = np.load(self.d / "raw/f1/partial_oof_E_prior_s42.npy"); bad[:] = 1.0
        np.save(self.d / "raw/f1/partial_oof_E_prior_s42.npy", bad)
        r = subprocess.run([sys.executable, str(ROOT / "scripts/lanef/analyze_oof.py"), "--raw", str(self.d / "raw"), "--out", str(self.d / "out")],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not zero", r.stderr)


if __name__ == "__main__":
    unittest.main()
