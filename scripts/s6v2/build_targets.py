"""Build the two compact_train --targets CSVs of the s6v2 main lane (J0 = nartaa JEV-blend reconstruction, V2 = labels v2).

Schema read by heliosli's pinned compact_train.py (sha256 14767c7a...): StudyInstanceUID, is_gold, review_holdout, fold and, per
finding L, "L|prior" (arm E_prior: soft target, weight 1 per cell, weight-normalised BCE) and "L|strict" (unused by E_prior; gold
rows must have both blank).  Gold-58 rows are kept as is_gold=1 rows with blank targets so the trainer predicts them and can never
train on them (its pool excludes is_gold=1 and it asserts blank targets).  Pool rows = the 4,349 non-gold train studies (the
trainer intersects with the pixel cache).  fold = folds_v0.json fold (gold -1).

J0 cell = mean of the available values of nartaa labels_v1_blend (non-gold) and riadmohamed42 JEV HYBRID; a cell with neither
value (unaddressed) gets the class prior = mean of that class's J0 values over the TRAINING-fold rows that have one
(folds != --holdout-fold; all pool rows for an all-data fit).  The 4 gold-relative studies quarantined by the sealed identity
audit (sealed-manifest-private.json future_role 'quarantined') are written with review_holdout=1 and blank targets, so the
trainer leaves them out of the pool (review_holdout and blanked-label exclusions both apply).
V2 cell = labels_v2_base_own224_w30 with the same fill rule (the table is dense, so the rule fills nothing).
Prints counts only (no IDs, no report text).  Usage: python -I build_targets.py <repo root> <out dir> <holdout fold, or -1 for all-data> <tag>"""
import csv, hashlib, json, sys
from pathlib import Path

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
UID = "StudyInstanceUID"


def read_table(path):
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            vals = []
            for L in LABELS:
                v = r.get(L, "")
                vals.append(None if v in ("", "nan", "NaN", None) else float(v))
            out[r[UID]] = vals
    return out


def gold_ids(train_csv):
    gold, allids = set(), []
    with open(train_csv, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            allids.append(r[UID])
            if all(r[L] != "" for L in LABELS):
                gold.add(r[UID])
    return allids, gold


def fill(table_by_uid, pool, fit_rows):
    prior, filled = [], [0] * len(LABELS)
    for j in range(len(LABELS)):
        v = [table_by_uid[u][j] for u in fit_rows if table_by_uid[u][j] is not None]
        prior.append(sum(v) / len(v))
    out = {}
    for u in pool:
        row = []
        for j, x in enumerate(table_by_uid[u]):
            if x is None:
                x = prior[j]; filled[j] += 1
            assert 0.0 <= x <= 1.0, "target outside [0, 1]"
            row.append(x)
        out[u] = row
    return out, prior, filled


def write(path, allids, gold, folds, vals, quarantined):
    cols = [UID, "is_gold", "review_holdout", "fold"] + [L + "|prior" for L in LABELS] + [L + "|strict" for L in LABELS]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for u in allids:
            g, q = u in gold, u in quarantined
            pri = [""] * len(LABELS) if (g or q) else ["%.6f" % x for x in vals[u]]
            w.writerow([u, "1" if g else "0", "1" if q else "0", str(folds[u])] + pri + [""] * len(LABELS))
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root, out, holdout, tag = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    out.mkdir(parents=True, exist_ok=True)
    allids, gold = gold_ids(root / "data/raw/train.csv")
    folds = json.load(open(root / "outputs/s6v2/dl/hpo/folds_v0.json"))["folds"]
    assert set(folds) == set(allids) and {u for u, f in folds.items() if f == -1} == gold and len(gold) == 58
    import json as _j
    man = _j.load(open(root / "outputs/kaggle/sealed-identity-full-result/sealed_identity/sealed-manifest-private.json"))
    quarantined = {x["uid"] for x in man["records"] if x["future_role"] == "quarantined"}
    assert len(quarantined) == 4 and not (quarantined & gold)
    pool = sorted(u for u in allids if u not in gold and u not in quarantined)
    fit_rows = [u for u in pool if folds[u] != holdout]
    blend = read_table(root / "data/weak-labels/nartaa_rsna-knee-hpo-assets/labels_v1_blend.parquet.csv")
    jev = read_table(root / "data/weak-labels/riadmohamed42_jev-knee-labels/HYBRID_labels_scores.csv")
    v2 = read_table(root / "outputs/labels/labels-v2-20261015/labels_v2_base_own224_w30.csv")
    j0 = {}
    n_src = {"both": 0, "blend_only": 0, "jev_only": 0, "none": 0}
    for u in pool:
        a, b = blend.get(u, [None] * 12), jev.get(u, [None] * 12)
        row = []
        for x, y in zip(a, b):
            if x is not None and y is not None:
                row.append(0.5 * (x + y)); n_src["both"] += 1
            elif x is not None:
                row.append(x); n_src["blend_only"] += 1
            elif y is not None:
                row.append(y); n_src["jev_only"] += 1
            else:
                row.append(None); n_src["none"] += 1
        j0[u] = row
    v2p = {u: v2.get(u, [None] * 12) for u in pool}
    j0f, j0prior, j0filled = fill(j0, pool, fit_rows)
    v2f, v2prior, v2filled = fill(v2p, pool, fit_rows)
    rec = {"holdout_fold": holdout, "pool_rows": len(pool), "prior_fit_rows": len(fit_rows), "gold_rows": len(gold),
           "quarantined_rows": len(quarantined), "j0_cell_sources": n_src,
           "j0_prior": dict(zip(LABELS, [round(p, 4) for p in j0prior])), "j0_filled": dict(zip(LABELS, j0filled)),
           "v2_prior": dict(zip(LABELS, [round(p, 4) for p in v2prior])), "v2_filled": dict(zip(LABELS, v2filled)),
           "inputs_sha256": {k: hashlib.sha256((root / p).read_bytes()).hexdigest() for k, p in {
               "labels_v1_blend": "data/weak-labels/nartaa_rsna-knee-hpo-assets/labels_v1_blend.parquet.csv",
               "jev_hybrid": "data/weak-labels/riadmohamed42_jev-knee-labels/HYBRID_labels_scores.csv",
               "labels_v2_base_own224_w30": "outputs/labels/labels-v2-20261015/labels_v2_base_own224_w30.csv",
               "folds_v0": "outputs/s6v2/dl/hpo/folds_v0.json"}.items()}}
    rec["targets_sha256"] = {"targets_J0_%s.csv" % tag: write(out / ("targets_J0_%s.csv" % tag), allids, gold, folds, j0f, quarantined),
                             "targets_V2_%s.csv" % tag: write(out / ("targets_V2_%s.csv" % tag), allids, gold, folds, v2f, quarantined)}
    json.dump(rec, open(out / ("targets_receipt_%s.json" % tag), "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
