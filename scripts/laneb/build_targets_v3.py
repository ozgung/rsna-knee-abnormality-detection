"""compact_train --targets CSV for labels v3 (all-data fit), the s6v2 build_targets.py schema and rules with the V2 table swapped:
StudyInstanceUID, is_gold, review_holdout, fold, L|prior (labels_v3 cell; dense, so the class-prior fill fills nothing), L|strict (blank).
Gold-58 and the 4 quarantined studies keep blank targets; folds = folds_v0.json.  Checks: identical rows / is_gold / review_holdout /
fold columns to targets_V2_all.csv.  Prints counts only.
  python3 build_targets_v3.py <labels_v3 table> <out dir>"""
import csv, hashlib, importlib.util, json, sys
from pathlib import Path

ROOT = Path("~/Dev/rsna-knee-abnormality-detection")
spec = importlib.util.spec_from_file_location("bt", ROOT / "scripts/s6v2/build_targets.py")
BT = importlib.util.module_from_spec(spec); spec.loader.exec_module(BT)


def main():
    table, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    allids, gold = BT.gold_ids(ROOT / "data/raw/train.csv")
    folds = json.load(open(ROOT / "outputs/s6v2/dl/hpo/folds_v0.json"))["folds"]
    assert set(folds) == set(allids) and {u for u, f in folds.items() if f == -1} == gold and len(gold) == 58
    man = json.load(open(ROOT / "outputs/kaggle/sealed-identity-full-result/sealed_identity/sealed-manifest-private.json"))
    quarantined = {x["uid"] for x in man["records"] if x["future_role"] == "quarantined"}
    assert len(quarantined) == 4 and not (quarantined & gold)
    pool = sorted(u for u in allids if u not in gold and u not in quarantined)
    v3 = BT.read_table(table)
    v3p = {u: v3.get(u, [None] * 12) for u in pool}
    v3f, prior, filled = BT.fill(v3p, pool, pool)
    path = out / "targets_V3_all.csv"
    h = BT.write(path, allids, gold, folds, v3f, quarantined)
    ref = list(csv.DictReader(open(ROOT / "outputs/s6v2/targets/targets_V2_all.csv", encoding="utf-8")))
    new = list(csv.DictReader(open(path, encoding="utf-8")))
    assert [tuple(r[k] for k in ("StudyInstanceUID", "is_gold", "review_holdout", "fold")) for r in ref] == \
           [tuple(r[k] for k in ("StudyInstanceUID", "is_gold", "review_holdout", "fold")) for r in new], "row schema differs from V2_all"
    changed = sum(any(a[L + "|prior"] != b[L + "|prior"] for L in BT.LABELS) for a, b in zip(ref, new))
    rec = {"pool_rows": len(pool), "gold_rows": len(gold), "quarantined_rows": len(quarantined), "filled": dict(zip(BT.LABELS, filled)),
           "v3_prior": dict(zip(BT.LABELS, [round(p, 4) for p in prior])), "rows_differing_from_V2_all": changed,
           "inputs_sha256": {"labels_v3": hashlib.sha256(table.read_bytes()).hexdigest(),
                             "folds_v0": hashlib.sha256((ROOT / "outputs/s6v2/dl/hpo/folds_v0.json").read_bytes()).hexdigest(),
                             "build_targets_py": hashlib.sha256((ROOT / "scripts/s6v2/build_targets.py").read_bytes()).hexdigest()},
           "targets_sha256": {"targets_V3_all.csv": h}}
    json.dump(rec, open(out / "targets_receipt_V3_all.json", "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
