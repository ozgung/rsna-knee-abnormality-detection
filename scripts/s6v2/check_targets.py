"""Schema and ID-coverage assertions for the s6v2 targets CSVs (prints counts only).
Usage: python -I check_targets.py <train.csv> <folds_v0.json> <targets.csv> [...]"""
import csv, json, math, sys

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
UID = "StudyInstanceUID"
COLS = [UID, "is_gold", "review_holdout", "fold"] + [L + "|prior" for L in LABELS] + [L + "|strict" for L in LABELS]


def check(train_csv, folds_json, path, n_rows=4407, n_gold=58, n_quarantined=4):
    gold, ids = set(), []
    with open(train_csv, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            ids.append(r[UID])
            if all(r[L] != "" for L in LABELS):
                gold.add(r[UID])
    folds = json.load(open(folds_json))["folds"]
    with open(path, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        assert rd.fieldnames == COLS, "column schema differs"
        rows = list(rd)
    got = [r[UID] for r in rows]
    assert len(got) == len(set(got)), "duplicate study IDs"
    assert set(got) == set(ids) and len(got) == n_rows, "ID coverage differs from train.csv"
    assert len(gold) == n_gold
    n_pool = n_q = 0
    for r in rows:
        u = r[UID]
        g = r["is_gold"] == "1"
        assert r["is_gold"] in ("0", "1") and g == (u in gold), "is_gold flag disagrees with the gold-58 IDs"
        assert r["review_holdout"] in ("0", "1") and int(r["fold"]) == folds[u]
        if r["review_holdout"] == "1":
            assert not g and all(r[L + "|prior"] == "" for L in LABELS), "quarantined row carries a target"
            n_q += 1
            continue
        if g:
            assert all(r[L + "|prior"] == "" and r[L + "|strict"] == "" for L in LABELS), "gold row carries a target"
            assert folds[u] == -1
        else:
            n_pool += 1
            assert 0 <= folds[u] <= 4
            for L in LABELS:
                v = float(r[L + "|prior"])
                assert math.isfinite(v) and 0.0 <= v <= 1.0, "pool target missing or outside [0, 1]"
                assert r[L + "|strict"] == ""
    trainable_gold = {r[UID] for r in rows if r["is_gold"] == "0"} & gold
    assert not trainable_gold
    assert n_q == n_quarantined
    return {"rows": len(rows), "pool": n_pool, "quarantined_blank": n_q, "gold_blank": len(gold), "gold_in_pool": len(trainable_gold),
            "fold_counts": {k: sum(1 for r in rows if r["is_gold"] == "0" and r["review_holdout"] == "0" and int(r["fold"]) == k) for k in range(5)}}


if __name__ == "__main__":
    for p in sys.argv[3:]:
        print(p.split("/")[-1], json.dumps(check(sys.argv[1], sys.argv[2], p)))
    print("CHECK_OK")
