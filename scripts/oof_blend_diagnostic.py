"""Bounded OOF blend diagnostic for two Track A families (no training, no uploads).

Reads the exported oof_fold{k}.csv files of two 5-fold families, rebuilds the held-out
truth (train_labels_v1 soft labels >= 0.5 with m_ masks) and the 58-study gold set
(train.csv rows with all twelve targets), reproduces each export's single-fold scores,
then scores pooled 5-fold OOF and probability / rank blends.  AUC is a dependency-free
rank (Mann-Whitney) implementation with tie handling identical to sklearn's.

Example:
  python scripts/oof_blend_diagnostic.py --out outputs/analysis/oof-blend-384-r2-20261011 \
    --family c384r 'outputs/kaggle/trackA-export-c384r-fold{0,1,2}-20261010-result/trackA/oof_fold{k}.csv' ...
"""
import argparse, json, csv
from pathlib import Path
import numpy as np, pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
def _nk(s): return "".join(ch for ch in s.lower() if ch.isalnum())

def auc(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if not npos or not nneg: return None
    r = pd.Series(p).rank(method="average").values
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def report(y, p, m=None):
    per, undefined = {}, []
    for j, n in enumerate(LAB):
        keep = slice(None) if m is None else (m[:, j] > 0)
        a = auc(y[keep, j], p[keep, j]); per[n] = a
        if a is None: undefined.append(n)
    vals = [v for v in per.values() if v is not None]
    return {"macro_auc_defined": float(np.mean(vals)) if vals else None, "n_defined": len(vals),
            "undefined_targets": undefined, "per_target": per}

def rank01(a):
    return (pd.DataFrame(a).rank(method="average").values - 1) / max(len(a) - 1, 1)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth-csv", default="data/weak-labels/nartaa_rsna-knee-hpo-assets/train_labels_v1.parquet.csv")
    ap.add_argument("--train-csv", default="data/raw/train.csv")
    ap.add_argument("--family", nargs="+", action="append", metavar=("NAME", "OOF_CSV"), required=True,
                    help="family name followed by its five oof_fold{k}.csv paths (fold order 0..4)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    truth = pd.read_csv(a.truth_csv, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    yc = {_nk(c[2:]): c for c in truth.columns if c.startswith("y_")}
    mc = {_nk(c[2:]): c for c in truth.columns if c.startswith("m_")}
    tr = pd.read_csv(a.train_csv, dtype={"StudyInstanceUID": str})
    gold = tr[tr[LAB].notna().all(axis=1)].set_index("StudyInstanceUID")[LAB].astype(float)
    fams = {}
    for spec in a.family:
        name, paths = spec[0], spec[1:]
        assert len(paths) == 5, name
        fams[name] = [pd.read_csv(p, dtype={"StudyInstanceUID": str}) for p in paths]
    names = list(fams)
    res = {"families": names, "gold_n": int(len(gold)), "single_fold": {}, "pooled_oof": {}, "gold_mean5": {}, "blends": {}}
    # single-fold reproduction + pooled val per family
    pooled = {n: [] for n in names}; goldp = {n: [] for n in names}
    for n in names:
        res["single_fold"][n] = {}
        for k, df in enumerate(fams[n]):
            v = df[df.set == "val"]; g = df[df.set == "gold"].set_index("StudyInstanceUID").loc[gold.index]
            rows = truth.loc[v.StudyInstanceUID.values]
            vy = (np.stack([rows[yc[_nk(t)]].values for t in LAB], 1).astype(float) >= 0.5).astype(int)
            vm = np.stack([rows[mc[_nk(t)]].values for t in LAB], 1).astype(float)
            gy = (gold.values >= 0.5).astype(int)
            rv, rg = report(vy, v[LAB].values, vm), report(gy, g[LAB].values)
            res["single_fold"][n][f"fold{k}"] = {"n_val": int(len(v)), "heldout_masked_macro_auc": rv["macro_auc_defined"],
                                                 "gold_macro_auc": rg["macro_auc_defined"], "gold_undefined": rg["undefined_targets"]}
            pooled[n].append((v.StudyInstanceUID.values, v[LAB].values, vy, vm)); goldp[n].append(g[LAB].values)
    # align pooled OOF on the intersection of study ids across families (fold-wise)
    for n in names:
        ids = np.concatenate([x[0] for x in pooled[n]]); P = np.concatenate([x[1] for x in pooled[n]])
        Y = np.concatenate([x[2] for x in pooled[n]]); M = np.concatenate([x[3] for x in pooled[n]])
        pooled[n] = pd.DataFrame(P, index=ids, columns=LAB), pd.DataFrame(Y, index=ids, columns=LAB), pd.DataFrame(M, index=ids, columns=LAB)
        r = report(Y, P, M); res["pooled_oof"][n] = {"n": int(len(ids)), "macro_auc": r["macro_auc_defined"], "per_target": r["per_target"]}
        gm = np.mean(goldp[n], 0); rg = report((gold.values >= 0.5).astype(int), gm)
        res["gold_mean5"][n] = {"macro_auc": rg["macro_auc_defined"], "per_target": rg["per_target"]}
    common = pooled[names[0]][0].index
    for n in names[1:]: common = common.intersection(pooled[n][0].index)
    res["pooled_common_n"] = int(len(common))
    Y = pooled[names[0]][1].loc[common].values; M = pooled[names[0]][2].loc[common].values
    gy = (gold.values >= 0.5).astype(int)
    Ps = {n: pooled[n][0].loc[common].values for n in names}; Gs = {n: np.mean(goldp[n], 0) for n in names}
    Rs = {n: rank01(Ps[n]) for n in names}; RG = {n: rank01(Gs[n]) for n in names}
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        ws = [w, 1 - w] if len(names) == 2 else [1 / len(names)] * len(names)
        pb = sum(wi * Ps[n] for wi, n in zip(ws, names)); gb = sum(wi * Gs[n] for wi, n in zip(ws, names))
        rb = sum(wi * Rs[n] for wi, n in zip(ws, names)); rgb = sum(wi * RG[n] for wi, n in zip(ws, names))
        key = "+".join(f"{wi:.1f}*{n}" for wi, n in zip(ws, names))
        res["blends"][key] = {"prob_oof_macro_auc": report(Y, pb, M)["macro_auc_defined"], "prob_gold_macro_auc": report(gy, gb)["macro_auc_defined"],
                              "rank_oof_macro_auc": report(Y, rb, M)["macro_auc_defined"], "rank_gold_macro_auc": report(gy, rgb)["macro_auc_defined"]}
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "oof_blend_diagnostic.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("families", "gold_n", "pooled_common_n")}))
    for n in names:
        print(n, "single-fold heldout/gold:", [(round(res["single_fold"][n][f"fold{k}"]["heldout_masked_macro_auc"], 4), round(res["single_fold"][n][f"fold{k}"]["gold_macro_auc"], 4)) for k in range(5)])
        print(n, "pooled OOF", round(res["pooled_oof"][n]["macro_auc"], 4), "| gold mean-of-5", round(res["gold_mean5"][n]["macro_auc"], 4))
    for k, v in res["blends"].items(): print(k, {kk: round(vv, 4) for kk, vv in v.items()})

if __name__ == "__main__":
    main()
