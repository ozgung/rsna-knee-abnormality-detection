"""All-data compact_train --targets CSV from any dense label table (Lane C: labels v3), with build_targets.py's exact rules:
gold-58 rows is_gold=1 with blank targets, the 4 quarantined rows review_holdout=1 with blank targets, pool = the other train
studies, fold = folds_v0, a missing cell gets the class prior over the pool (the dense v2/v3 tables fill nothing).
Prints counts only.  Usage: python -I build_targets_table.py <repo root> <out dir> <label table csv> <name, e.g. V3_all>"""
import hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_targets as BT  # noqa: E402


def main():
    root, out, table, name = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    out.mkdir(parents=True, exist_ok=True)
    allids, gold = BT.gold_ids(root / "data/raw/train.csv")
    folds = json.load(open(root / "outputs/s6v2/dl/hpo/folds_v0.json"))["folds"]
    assert set(folds) == set(allids) and {u for u, f in folds.items() if f == -1} == gold and len(gold) == 58
    man = json.load(open(root / "outputs/kaggle/sealed-identity-full-result/sealed_identity/sealed-manifest-private.json"))
    quarantined = {x["uid"] for x in man["records"] if x["future_role"] == "quarantined"}
    assert len(quarantined) == 4 and not (quarantined & gold)
    pool = sorted(u for u in allids if u not in gold and u not in quarantined)
    t = BT.read_table(table)
    missing_rows = sum(1 for u in pool if u not in t)
    vals, prior, filled = BT.fill({u: t.get(u, [None] * 12) for u in pool}, pool, pool)
    path = out / ("targets_%s.csv" % name)
    rec = {"pool_rows": len(pool), "gold_rows": len(gold), "quarantined_rows": len(quarantined), "table_rows_missing": missing_rows,
           "prior": dict(zip(BT.LABELS, [round(p, 4) for p in prior])), "filled": dict(zip(BT.LABELS, filled)),
           "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
           "gold_in_table_nonblank": sum(1 for u in gold if u in t and any(x is not None for x in t[u])),
           "targets_sha256": BT.write(path, allids, gold, folds, vals, quarantined)}
    json.dump(rec, open(out / ("targets_receipt_%s.json" % name), "w"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
