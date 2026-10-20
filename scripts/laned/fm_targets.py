"""Lane D (local): joint-fit targets = the competition all-data targets (labels v2 or v3, unchanged cells, weight 1) plus one row
per fastMRI exam whose three teacher arms (P2's R384 two-view 'old', R224 'crop224', R6 flip 'r6') are all genuine:
  target_f = w_f * (0.65 p_R384 + 0.35 p_R224) + (1 - w_f) * p_R6flip      (P2's v1 per-finding Raptor weights; probabilities, no ranks)
written to 'L|prior' with 'L|pw' = 0.5 (competition pool rows 1).  fastMRI rows: is_gold 0, review_holdout 0, fold sha1 % 5.
Usage: python fm_targets.py <base targets csv> <out csv> <teacher out dir> [<teacher out dir> ...]   (prints counts only)"""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]
RAPTOR_W = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker's": 0.35}
W = np.array([RAPTOR_W.get(L, 0.5) for L in LABELS])
ARMS = {"old": ("board%d-old", "old"), "crop224": ("board%d-crop224", "crop224"), "r6": ("board%d-r6-production", "r6")}
EFFECTIVE_FM_WEIGHT = 0.5      # plan: teacher targets at weight 0.5 relative to a competition study


def cell_weight(f, target=EFFECTIVE_FM_WEIGHT, batch=2):
    """compact_train's bce_soft normalises by the batch's weight sum per class, so with batch 2 a nominal cell weight w is not
    the effective one.  Solve for w such that E[per-sample weight | fastMRI] / E[per-sample weight | competition] = target
    under uniform sampling with fastMRI fraction f (batch 2: partner is fastMRI with probability f)."""
    assert batch == 2
    def ratio(w):
        ef = (1 - f) * w / (1 + w) + f * 0.5
        ec = (1 - f) * 0.5 + f * 1 / (1 + w)
        return ef / ec
    lo, hi = 1e-4, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ratio(mid) < target else (lo, mid)
    return round(0.5 * (lo + hi), 4)


def arm_rows(dirs, arm):
    out = {}
    folder, sub = ARMS[arm]
    for d in dirs:
        for b in (0, 1):
            p = Path(d) / "r6flip" / (folder % b) / sub / "predictions.npz"
            c = Path(d) / "r6flip" / (folder % b) / "r6" / "compact" / "predictions.csv"
            if p.exists():
                z = np.load(p, allow_pickle=False)
                rows = zip(z["study_ids"], z["probabilities"], z["statuses"])
            elif arm == "r6" and c.exists():      # P2's own entry salvage: compact_infer's flushed CSV rows are genuine
                rr = [r for r in csv.reader(open(c, newline=""))]
                assert rr[0] == ["StudyInstanceUID"] + LABELS
                rows = [(r[0], np.array([float(x) for x in r[1:]]), "genuine") for r in rr[1:] if len(r) == 13]
            else:
                continue
            for u, pr, st in rows:
                pr = np.asarray(pr, dtype=np.float64)
                if not ((pr >= 0) & (pr <= 1)).all():
                    continue
                if str(st) == "genuine" and np.isfinite(pr).all():
                    assert str(u) not in out, "duplicate teacher row"
                    out[str(u)] = pr.astype(np.float64)
    return out


def main(base, out, *dirs):
    arms = {a: arm_rows(dirs, a) for a in ARMS}
    ids = sorted(set.intersection(*[set(v) for v in arms.values()]))
    rec = {"teacher_rows_genuine": {a: len(v) for a, v in arms.items()}, "fm_rows_all_three": len(ids)}
    from scipy.stats import rankdata

    def pct(a):                                   # average percentile rank in (0, 1], per finding
        return np.stack([rankdata(a[:, j]) / len(a) for j in range(a.shape[1])], 1)
    A = {a: np.stack([arms[a][u] for u in ids]) for a in ARMS}
    rec["arm_prob_mean"] = {a: dict(zip(LABELS, np.round(A[a].mean(0), 4).tolist())) for a in ARMS}
    S = W * (0.65 * pct(A["old"]) + 0.35 * pct(A["crop224"])) + (1 - W) * pct(A["r6"])
    R = pct(S)
    rows = list(csv.DictReader(open(base, encoding="utf-8")))
    cols = list(rows[0].keys())
    assert cols[:4] == ["StudyInstanceUID", "is_gold", "review_holdout", "fold"]
    base_ids = {r["StudyInstanceUID"] for r in rows}
    assert not (base_ids & set(ids)), "fastMRI IDs collide with competition IDs"
    pool = [r for r in rows if r["is_gold"] == "0" and r["review_holdout"] == "0"]
    pm = np.array([[float(r[L + "|prior"]) for L in LABELS] for r in pool])
    T = np.stack([np.quantile(pm[:, j], np.clip(R[:, j] - 0.5 / len(R), 0, 1)) for j in range(12)], 1)
    assert np.isfinite(T).all() and (T >= 0).all() and (T <= 1).all()
    rec["calibration"] = "P2 rank blend over fastMRI exams, quantile-mapped onto the competition pool target marginal"
    rec["fm_target_mean"] = dict(zip(LABELS, np.round(T.mean(0), 4).tolist()))
    rec["competition_pool_rows"] = len(pool)
    rec["competition_target_mean"] = dict(zip(LABELS, np.round(pm.mean(0), 4).tolist()))
    f = len(ids) / (len(ids) + len(pool))
    FM_WEIGHT = cell_weight(f)
    rec["fm_fraction"], rec["fm_cell_weight"], rec["fm_effective_weight"] = round(f, 4), FM_WEIGHT, EFFECTIVE_FM_WEIGHT
    newcols = cols + [L + "|pw" for L in LABELS]
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=newcols); w.writeheader()
        for r in rows:
            inpool = r["is_gold"] == "0" and r["review_holdout"] == "0"
            w.writerow({**r, **{L + "|pw": ("1" if inpool else "") for L in LABELS}})
        for u, t in zip(ids, T):
            row = {"StudyInstanceUID": u, "is_gold": "0", "review_holdout": "0", "fold": str(int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 5)}
            row.update({L + "|prior": "%.6f" % x for L, x in zip(LABELS, t)})
            row.update({c: "" for c in cols if c.endswith("|strict")})
            row.update({L + "|pw": str(FM_WEIGHT) for L in LABELS})
            w.writerow(row)
    rec["out_sha256"] = hashlib.sha256(Path(out).read_bytes()).hexdigest()
    rec["base_sha256"] = hashlib.sha256(Path(base).read_bytes()).hexdigest()
    json.dump(rec, open(str(out) + ".receipt.json", "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
