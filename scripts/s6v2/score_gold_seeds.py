"""Lane C gold-58 read-out (descriptive, selection-exposed): per-seed macro AUC from compact_infer predictions.csv and the
seed-ensemble mean rank (mean over seeds of per-seed average percentile ranks over the 58), per-target AUC and supports.
Macro = unweighted mean over the 12 targets; a target with a single class would be undefined and makes the macro NaN.
Usage: python score_gold_seeds.py <repo root> <out json> <name>=<compact_infer out dir> [...] [--ensemble a,b,...]
Prints aggregates only (no study IDs)."""
import csv, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_pair import LABELS, UID, endpoint_support, gold_table, macro, per  # noqa: E402


def read_pred(d):
    with open(Path(d) / "predictions.csv", newline="") as h:
        r = csv.reader(h)
        assert next(r) == [UID] + LABELS, "prediction schema"
        rows = {x[0]: [float(v) for v in x[1:]] for x in r}
    rec = json.load(open(Path(d) / "receipt.json"))
    return rows, rec


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    args = sys.argv[3:]
    ens = []
    if "--ensemble" in args:
        i = args.index("--ensemble"); ens = [args[i + 1].split(",")]; args = args[:i] + args[i + 2:]
    gold = gold_table(root / "data/raw/train.csv")
    ids = sorted(gold)
    assert len(ids) == 58
    y = np.array([gold[u] for u in ids])
    res = {"gold58_selection_exposed": True, "support": endpoint_support(y), "macro_policy": "unweighted mean of 12 per-target AUCs",
           "runs": {}, "ensembles": {}}
    preds = {}
    for a in args:
        name, d = a.split("=", 1)
        rows, rec = read_pred(d)
        assert set(rows) == set(ids) and not rec.get("fallback"), "gold coverage / fallback in %s" % name
        p = np.array([rows[u] for u in ids]); preds[name] = p
        m, _ = macro(y, p)
        res["runs"][name] = {"macro": m, "per_target": dict(zip(LABELS, per(y, p))), "amp": rec.get("amp"), "models": rec.get("models"),
                             "scored": rec.get("scored"), "seconds": rec.get("seconds")}
    for group in ens:
        r = np.mean([pd.DataFrame(preds[g]).rank(method="average", pct=True).to_numpy() for g in group], axis=0)
        m, _ = macro(y, r)
        res["ensembles"]["+".join(group)] = {"macro": m, "per_target": dict(zip(LABELS, per(y, r))), "aggregation": "mean per-seed pct rank"}
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({"runs": {k: round(v["macro"], 4) for k, v in res["runs"].items()},
                      "ensembles": {k: round(v["macro"], 4) for k, v in res["ensembles"].items()}}))


if __name__ == "__main__":
    main()
