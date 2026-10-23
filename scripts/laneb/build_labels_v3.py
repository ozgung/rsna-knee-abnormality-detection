"""Labels v3 (Lane B, Oct 19): the labels-v2 headline construction (base_own224_w30) with the image teacher swapped for
heliosli's R6 bundle under flip TTA, out of fold.

  base    = labels_v2_mean6_plus_dread (parameter-free mean of the report tables + Dread LLM; gold rows = mean5)
  teacher = R6 flip (compact_infer v4c, stackers applied): non-gold rows = the study's hash-fold model (out of fold);
            gold rows = mean of the four fold models (gold is out of fold for every model)
  cal     = per-finding quantile map of the teacher onto the base's NON-GOLD marginal (fit on non-gold rows with a teacher value),
            exactly build_labels_v2.calibrate
  v3      = (1 - w) base + w cal, w = 0.30 (MCL 0.15), the a-priori labels-v2 weights; rows without a teacher keep the base.
Nothing is fitted on gold.  Gold-58 is read once at the end (score + paired bootstrap vs labels v2) and recorded as exposed.
Kill (plan Lane B): gold-58 macro < labels v2's 0.920 -> FAILED.json, else the table + READY.json.
Writes IDs only under --out (ignored outputs/); prints aggregates only.
  python3 build_labels_v3.py --oof DIR(oof_columns.csv, gold58_columns.csv) --out DIR"""
import argparse, csv, hashlib, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path("~/Dev/rsna-knee-abnormality-detection")
sys.path.insert(0, str(ROOT))
from knee.paired_validation import _auc_draws, resample_counts  # noqa: E402

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]
K = len(LABELS)
V2 = ROOT / "outputs/labels/labels-v2-20261015"
BASE = V2 / "labels_v2_mean6_plus_dread.csv"
V2_TABLE = V2 / "labels_v2_base_own224_w30.csv"
TRAIN = ROOT / "data/raw/train.csv"
W_DEFAULT, W_MCL = 0.30, 0.15
V2_GOLD58 = 0.920          # labels v2 headline gold-58 macro (docs/experiments/labels-v2-20261015.md)
TEACHER = "R6 flip"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(path, prefix=""):
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["StudyInstanceUID"]] = [float(r[prefix + l]) for l in LABELS]
    return out


def quantile_map(mat, fit_rows):
    """build_labels_v2.quantile_map: per column, mid-rank empirical CDF of the fit rows' values."""
    out = np.full(mat.shape, np.nan)
    for k in range(K):
        ref = mat[fit_rows, k]
        ref = np.sort(ref[~np.isnan(ref)])
        v = mat[:, k]; ok = ~np.isnan(v)
        lo = np.searchsorted(ref, v[ok], side="left"); hi = np.searchsorted(ref, v[ok], side="right")
        out[ok, k] = (lo + hi) / (2.0 * len(ref))
    return out


def macro(y, p, counts=None):
    c = np.ones((1, len(y)), dtype=np.int64) if counts is None else counts
    return _auc_draws(y, p, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    ids, gold = [], {}
    with open(TRAIN, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            ids.append(r["StudyInstanceUID"])
            if all(r[l] != "" for l in LABELS):
                gold[r["StudyInstanceUID"]] = [float(r[l]) for l in LABELS]
    assert len(ids) == 4407 and len(gold) == 58
    base_t = read(BASE)
    assert set(base_t) == set(ids)
    n = len(ids)
    is_gold = np.array([u in gold for u in ids])
    fit_rows = ~is_gold
    base = np.array([base_t[u] for u in ids])
    oof = read(a.oof / "oof_columns.csv", TEACHER + "|")
    g58 = read(a.oof / "gold58_columns.csv", TEACHER + "|")
    assert not set(oof) & set(gold) and set(g58) == set(gold)
    T = np.full((n, K), np.nan)
    for i, u in enumerate(ids):
        if u in oof:
            T[i] = oof[u]
        elif u in g58:
            T[i] = g58[u]
    ht = ~np.isnan(T)
    q = quantile_map(T, fit_rows)
    cal = np.full(T.shape, np.nan)
    for k in range(K):
        ok = ht[:, k] & fit_rows
        ref = np.sort(base[ok, k])
        cal[ht[:, k], k] = np.quantile(ref, np.clip(q[ht[:, k], k], 0, 1))
    w = np.array([W_MCL if l == "MCL" else W_DEFAULT for l in LABELS])
    v3 = base.copy()
    v3[ht] = ((1 - w)[None, :] * base + w[None, :] * np.nan_to_num(cal))[ht]
    assert np.isfinite(v3).all() and (v3 >= 0).all() and (v3 <= 1).all()
    table = out / "labels_v3_base_r6flipoof_w30.csv"
    with open(table, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh); wr.writerow(["StudyInstanceUID"] + LABELS)
        for i, u in enumerate(ids):
            wr.writerow([u] + ["%.5f" % v for v in v3[i]])
    # ---- gold-58: read once (exposed)
    gi = np.flatnonzero(is_gold)
    gu = [ids[i] for i in gi]
    yg = np.array([gold[u] for u in gu])
    v2t = read(V2_TABLE)
    v2g = np.array([v2t[u] for u in gu])
    cnt = resample_counts(gu, draws=2000, seed=42)
    d3, d2, db = macro(yg, v3[gi], cnt), macro(yg, v2g, cnt), macro(yg, base[gi], cnt)
    m3, m2, mb = (float(np.mean(macro(yg, x)[0])) for x in (v3[gi], v2g, base[gi]))
    ok3 = np.isfinite(d3).all(1) & np.isfinite(d2).all(1) & np.isfinite(db).all(1)
    diff, diffb = d3[ok3].mean(1) - d2[ok3].mean(1), d3[ok3].mean(1) - db[ok3].mean(1)
    tq = macro(yg, cal[gi])[0]
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "table": table.name, "rows": n,
           "non_gold_rows": int(fit_rows.sum()), "non_gold_rows_with_teacher": int((ht.all(1) & fit_rows).sum()),
           "gold_rows_with_teacher": int((ht.all(1) & is_gold).sum()),
           "construction": "labels_v2_mean6_plus_dread base; teacher R6 flip (hash-fold OOF on non-gold, 4-model mean on gold), "
                           "quantile-calibrated onto the base's non-gold marginal (non-gold fit), w 0.30 (MCL 0.15); rows without "
                           "teacher keep the base",
           "inputs_sha256": {"base": sha256(BASE), "oof_columns": sha256(a.oof / "oof_columns.csv"),
                             "gold58_columns": sha256(a.oof / "gold58_columns.csv"), "builder": sha256(__file__)},
           "table_sha256": sha256(table),
           "gold58": {"labels_v3_macro": m3, "labels_v2_macro_recomputed": m2, "base_macro": mb, "teacher_cal_macro": float(np.nanmean(tq)),
                      "per_target_v3": dict(zip(LABELS, macro(yg, v3[gi])[0].tolist())),
                      "paired_v3_minus_v2": {"delta": m3 - m2, "ci95": np.quantile(diff, [.025, .975]).tolist(), "p_gt0": float((diff > 0).mean())},
                      "paired_v3_minus_base": {"delta": m3 - mb, "ci95": np.quantile(diffb, [.025, .975]).tolist()},
                      "bootstrap": "2,000 study draws, seed 42", "exposure": "EXPOSED: gold-58 read once for the Lane B kill check; no fitting or selection on it"},
           "kill_rule": "gold-58 macro < labels v2 (recomputed with the same AUC code; headline 0.920) -> FAILED"}
    passed = m3 >= m2 and m3 >= V2_GOLD58 - 5e-4      # same AUC code for v2 and v3; 0.920 is the rounded headline
    res["status"] = "READY" if passed else "FAILED"
    json.dump(res, open(out / ("READY.json" if passed else "FAILED.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("status", "rows", "non_gold_rows_with_teacher", "gold_rows_with_teacher", "table_sha256")}))
    print(json.dumps(res["gold58"], indent=1))


if __name__ == "__main__":
    main()
