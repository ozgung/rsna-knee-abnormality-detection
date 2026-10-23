"""Lane C gold-58 read-out (descriptive, selection-exposed): per-seed macro AUC from compact_infer predictions.csv and the
seed-ensemble mean rank (mean over seeds of per-seed average percentile ranks over the 58), per-target AUC and supports.
Macro = unweighted mean over the 12 targets; a target with a single class would be undefined and makes the macro NaN.
Usage: python score_gold_seeds.py <repo root> <out json> <name>=<compact_infer out dir> [...] [--ensemble a,b,...]
Prints aggregates only (no study IDs)."""
import csv, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
UID = "StudyInstanceUID"


def auc(y, p):
    """Binary ROC AUC = Mann-Whitney U with average ranks for ties (equals sklearn's roc_auc_score); None if one class."""
    y = np.asarray(y, float); r = pd.Series(np.asarray(p, float)).rank(method="average").to_numpy()
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    return None if n1 == 0 or n0 == 0 else float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def per(y, p):
    return [auc(y[:, j], p[:, j]) for j in range(y.shape[1])]


def macro(y, p):
    a = per(y, p)
    return (float(np.mean(a)) if all(x is not None for x in a) else float("nan")), a


def gold_table(train_csv):
    out = {}
    with open(train_csv, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if all(r[L] != "" for L in LABELS):
                out[r[UID]] = [float(r[L]) for L in LABELS]
    return out


def endpoint_support(y):
    return {L: {"positive": int((y[:, j] == 1).sum()), "negative": int((y[:, j] == 0).sum())} for j, L in enumerate(LABELS)}


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
        items = [(name, np.array([rows[u] for u in ids]))]
        if len(rec.get("models") or []) > 1:          # one decode, several seeds: split the dumped per-model logits
            z = json.load(open(Path(d) / "logits.json"))
            zz = np.array([z[u] for u in ids], dtype=np.float64)          # [58, M, 12]
            assert zz.shape == (58, len(rec["models"]), 12)
            assert np.abs(1 / (1 + np.exp(-zz.mean(1))) - items[0][1]).max() < 2e-6, "CSV != sigmoid(mean logits)"
            items += [(m.replace(".pt", ""), 1 / (1 + np.exp(-zz[:, k]))) for k, m in enumerate(rec["models"])]
        for nm, p in items:
            preds[nm] = p
            mm, _ = macro(y, p)
            res["runs"][nm] = {"macro": mm, "per_target": dict(zip(LABELS, per(y, p))), "amp": rec.get("amp"), "source": name,
                               "models": rec.get("models") if nm == name else [nm + ".pt"], "scored": rec.get("scored"),
                               "seconds": rec.get("seconds")}
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
