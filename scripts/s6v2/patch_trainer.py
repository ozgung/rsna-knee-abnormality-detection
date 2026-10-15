"""Derive compact_train_s6v2.py from heliosli's pinned compact_train.py (sha256 14767c7a...), the trainer nartaa's s6fall
code_manifest pins.  nartaa's run wrapper (trainer sha256 ae18d2e1..., options pool / folds_json / all_data) is not public;
these minimal edits reproduce its documented options and change no model, loss, sampling or optimiser code:
  1. pool folds come from the targets CSV 'fold' column (= folds_v0.json) instead of sha1(uid) % K;
  2. the output-bias initialisation (logit of the mean prior target) is fitted on the fold's TRAINING rows only
     (the pinned file uses the whole pool, including the held-out fold);
  3. S6V2_ALL_DATA=1 trains one model on every pool study (nartaa's all_data + no_final_predict: no OOF prediction);
  4. the exact train / held-out membership of every fold (count + sha256 of the sorted IDs) is written to
     <out>/membership_<arm>_s<seed>_f<k>.json before training starts;
  5. Studies reads the lossless per-row PNG store when px.u8.npy is absent (storage only: bit-identical uint8 rows).
Usage: python -I patch_trainer.py <pinned compact_train.py> <output .py>"""
import hashlib, sys
from pathlib import Path

PIN = "14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80"
src = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(src).hexdigest() == PIN, "not the pinned compact_train.py"
s = src.decode("utf-8")
EDITS = [
    ("""    pool_fold = np.array([int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % K for u in pool])""",
     """    pool_fold = np.array([int(by_uid[u]["fold"]) for u in pool])          # s6v2: folds_v0 from the targets CSV
    assert ((pool_fold >= 0) & (pool_fold < K)).all(), "pool fold outside 0..K-1"
    ALL_DATA = os.environ.get("S6V2_ALL_DATA") == "1\""""),
    ("""    bias0 = torch.logit(col(pool, "prior", float("nan")).mean(0).clamp(1e-3, 1 - 1e-3))""",
     """    prior_all = col(pool, "prior", float("nan"))
    bias0 = torch.logit(prior_all.mean(0).clamp(1e-3, 1 - 1e-3))     # s6v2: check-targets only; training uses the fold's rows"""),
    ("""            tr = np.flatnonzero((pool_fold != k) & (pool_fold != H)) if H is not None else np.flatnonzero(pool_fold != k)
            ho = np.flatnonzero(pool_fold == k)
            torch.manual_seed(1000 + 17 * seed)          # same init""",
     """            tr = np.flatnonzero((pool_fold != k) & (pool_fold != H)) if H is not None else np.flatnonzero(pool_fold != k)
            ho = np.flatnonzero(pool_fold == k)
            if ALL_DATA:                                   # s6v2: nartaa's all_data (one model on every pool study)
                tr, ho = np.arange(len(pool)), np.zeros(0, dtype=np.int64)
            def _h(ix):
                return hashlib.sha256("\\n".join(sorted(pool[i] for i in ix)).encode()).hexdigest()
            json.dump({"arm": arm, "seed": seed, "fold": k, "all_data": ALL_DATA, "train": int(len(tr)), "train_sha256": _h(tr),
                       "holdout": int(len(ho)), "holdout_sha256": _h(ho), "gold": len(gold),
                       "gold_sha256": hashlib.sha256("\\n".join(sorted(gold)).encode()).hexdigest(),
                       "train_gold_overlap": len({pool[i] for i in tr} & set(gold))},
                      open(os.path.join(a.out, "membership_%s_s%d_f%d.json" % (arm, seed, k)), "w"), indent=1)
            print("  membership fold %d: train %d (%s) holdout %d (%s)" % (k, len(tr), _h(tr)[:16], len(ho), _h(ho)[:16]), flush=True)
            torch.manual_seed(1000 + 17 * seed)          # same init"""),
    ("""            with torch.no_grad():
                model.bias.copy_(bias0)""",
     """            with torch.no_grad():                         # s6v2: bias from the training rows of this fold only
                model.bias.copy_(torch.logit(prior_all[torch.from_numpy(tr).to(dev)].mean(0).clamp(1e-3, 1 - 1e-3)))"""),
    ("""            z_pool[torch.from_numpy(ho).to(dev)] = predict(model, [pool[i] for i in ho], seed)
            if H is not None:
                hh = np.flatnonzero(pool_fold == H)
                z_pool[torch.from_numpy(hh).to(dev)] += predict(model, [pool[i] for i in hh], seed) / (K - 1)
            logs.append""",
     """            if len(ho):
                z_pool[torch.from_numpy(ho).to(dev)] = predict(model, [pool[i] for i in ho], seed)
            if H is not None:
                hh = np.flatnonzero(pool_fold == H)
                z_pool[torch.from_numpy(hh).to(dev)] += predict(model, [pool[i] for i in hh], seed) / (K - 1)
            logs.append"""),
    ("""        if self.px is None:
            self.px = [np.load(os.path.join(d, "px.u8.npy"), mmap_mode="r") for d in self.dirs]""",
     """        if self.px is None:                           # s6v2: px.u8.npy, or the lossless per-row PNG store (same uint8 rows)
            self.px = [np.load(os.path.join(d, "px.u8.npy"), mmap_mode="r") if os.path.exists(os.path.join(d, "px.u8.npy"))
                       else PngRows(d) for d in self.dirs]"""),
    ("""def collate(batch):""",
     """class PngRows:
    \"\"\"s6v2: read-only row store px.png.bin + px.png.off.npy written by vm_build_cache.py; rows[idx] -> uint8 [k, S, S],
    bit-identical to the px.u8.npy rows px_cache would have written (PNG is lossless).\"\"\"

    def __init__(self, d):
        import cv2
        cv2.setNumThreads(1)
        self.cv2 = cv2
        self.off = np.load(os.path.join(d, "px.png.off.npy"))
        self.buf = np.memmap(os.path.join(d, "px.png.bin"), dtype=np.uint8, mode="r")
        first = self._row(0)
        self.shape = (len(self.off) - 1,) + first.shape

    def _row(self, i):
        return self.cv2.imdecode(np.asarray(self.buf[int(self.off[i]):int(self.off[i + 1])]), self.cv2.IMREAD_UNCHANGED)

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        if idx.ndim == 0:
            return self._row(int(idx))
        return np.stack([self._row(int(i)) for i in idx.ravel()]).reshape(idx.shape + self.shape[1:])


def collate(batch):"""),
]
for a, b in EDITS:
    assert s.count(a) == 1, "edit anchor not found exactly once: " + a[:70]
    s = s.replace(a, b)
Path(sys.argv[2]).write_text(s, encoding="utf-8")
print("patched", hashlib.sha256(s.encode()).hexdigest())
