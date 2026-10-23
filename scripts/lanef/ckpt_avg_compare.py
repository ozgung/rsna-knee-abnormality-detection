"""Lane F checkpoint-averaging test, local comparison: averaged (epochs 9 + 10 backups + final) vs final fold models on their
held-out hash folds.  --raw holds the final fits' f<k>/ directories (analyze_oof.py layout); --avg holds f<k>/ directories with
the reeval outputs of r1_avg.sh (partial_oof_E_prior_s<seed>.npy, partial_preds_E_prior_s<seed>.json, avg_receipt.json).
Truth labels_v1_blend >= 0.5; paired study bootstrap 2,000 draws, seed 42.  Rule (parent, Oct 20 22:55): apply the averaging to
the all-data members if averaged - final >= +0.001 with the CI above 0.  Prints aggregates only.

  python3 scripts/lanef/ckpt_avg_compare.py --raw oof_r1/raw --avg oof_r1/avg_raw --out oof_r1/ckpt_avg.json"""
import argparse, csv, glob, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/lanef"))
from analyze_oof import LABELS, BLEND, TRAIN, sigmoid, macro_auc, boot_table, load_fold  # noqa: E402


def load_avg(d, k, pool_ref):
    d = Path(d)
    pj = sorted(glob.glob(str(d / "partial_preds_E_prior_s*.json")))
    assert len(pj) == 1, d
    meta = json.load(open(pj[0])); seed = int(meta["seed"])
    assert meta["only_folds"] == [k] and meta.get("reeval_of"), "not a reeval of fold %d" % k
    z = np.load(d / ("partial_oof_E_prior_s%d.npy" % seed)).astype(np.float64)
    assert z.shape == (len(pool_ref), 12)
    g = np.asarray(meta["gold_logits"], np.float64) * 4
    rec = json.load(open(d / "avg_receipt.json")) if (d / "avg_receipt.json").exists() else {}
    return z, g, rec, meta.get("eval_stride")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", required=True); ap.add_argument("--avg", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--blend", default=str(BLEND)); ap.add_argument("--train", default=str(TRAIN))
    a = ap.parse_args()
    folds = sorted(int(p.name[1:]) for p in Path(a.avg).glob("f[0-3]") if (p / "avg_receipt.json").exists() or list(p.glob("partial_oof_*.npy")))
    assert folds, "no averaged fold outputs under %s" % a.avg
    fin, avg, fold_of, gold_fin, gold_avg, recs = {}, {}, {}, {}, {}, {}
    pool_ref = gold_uids = None
    for k in folds:
        pool, uids, z, gu, g, info = load_fold(Path(a.raw) / ("f%d" % k), k)
        pool_ref = pool_ref or pool; gold_uids = gold_uids or gu
        assert pool == pool_ref and gu == gold_uids
        za, ga, rec, stride = load_avg(Path(a.avg) / ("f%d" % k), k, pool)
        idx = {u: i for i, u in enumerate(pool)}
        for u, row in zip(uids, z):
            fin[u] = row; fold_of[u] = k; avg[u] = za[idx[u]]
            assert np.abs(za[idx[u]]).sum() > 0, "averaged reeval left a held-out row empty (fold %d)" % k
        gold_fin[k] = g; gold_avg[k] = ga; recs[k] = {"receipt": rec, "eval_stride": stride, "final_fit": {kk: info[kk] for kk in ("fold", "n_heldout", "loss_first_last")}}
    studies = sorted(fin)
    blend = {}
    with open(a.blend, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            blend[r["StudyInstanceUID"]] = [float(r[l]) for l in LABELS]
    studies = [u for u in studies if u in blend]
    y = np.asarray([[1.0 if v >= 0.5 else 0.0 for v in blend[u]] for u in studies])
    fo = np.asarray([fold_of[u] for u in studies])
    cols = {"final": sigmoid(np.stack([fin[u] for u in studies])), "averaged e9+e10+e11": sigmoid(np.stack([avg[u] for u in studies]))}
    res = {"folds": folds, "n": len(studies), "truth": "labels_v1_blend >= 0.5", "eval": "pixel cache, stride 1, held-out hash fold per model",
           "macro": {n: macro_auc(y, p)[0] for n, p in cols.items()},
           "per_fold_macro": {n: {str(k): macro_auc(y[fo == k], p[fo == k])[0] for k in folds} for n, p in cols.items()},
           "paired": boot_table(y, cols, [("averaged e9+e10+e11", "final")], studies),
           "per_fold_paired": {str(k): boot_table(y[fo == k], {n: p[fo == k] for n, p in cols.items()}, [("averaged e9+e10+e11", "final")],
                                                   [u for u, f_ in zip(studies, fo) if f_ == k])["averaged e9+e10+e11 - final"] for k in folds},
           "per_target": {n: dict(zip(LABELS, macro_auc(y, p)[1].tolist())) for n, p in cols.items()}, "receipts": recs}
    gold = {}
    with open(a.train, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if all(r[l] not in ("", None) for l in LABELS):
                gold[r["StudyInstanceUID"]] = [float(r[l]) for l in LABELS]
    yg = np.asarray([gold[u] for u in gold_uids])
    gf = np.mean([gold_fin[k] for k in folds], 0); ga = np.mean([gold_avg[k] for k in folds], 0)
    res["gold58"] = {"n": 58, "folds": folds, "exposed": True, "note": "averaged models' gold logits come from the reeval (one extra trainer read per averaged fold model)",
                     "final_mean_logit_macro": macro_auc(yg, gf)[0], "averaged_mean_logit_macro": macro_auc(yg, ga)[0],
                     "per_fold_model": {str(k): {"final": macro_auc(yg, gold_fin[k])[0], "averaged": macro_auc(yg, gold_avg[k])[0]} for k in folds},
                     "paired_averaged_minus_final": boot_table(yg, {"final": sigmoid(gf), "averaged": sigmoid(ga)}, [("averaged", "final")], list(gold_uids))["averaged - final"]}
    d = res["paired"]["averaged e9+e10+e11 - final"]
    res["rule"] = {"text": "apply last-three-epoch averaging to the all-data members if averaged - final >= +0.001 with CI > 0 (pooled OOF)",
                   "delta": d["delta"], "ci95": d["ci95"], "passed": bool(d["delta"] >= 0.001 and d["ci95"][0] > 0), "folds_covered": folds}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"n": res["n"], "folds": folds, "macro": res["macro"], "per_fold_macro": res["per_fold_macro"], "paired": res["paired"],
                      "per_fold_paired": {k: {"delta": v["delta"], "ci95": v["ci95"]} for k, v in res["per_fold_paired"].items()},
                      "gold58": {k: v for k, v in res["gold58"].items() if k in ("final_mean_logit_macro", "averaged_mean_logit_macro", "per_fold_model")},
                      "rule": res["rule"]}, indent=1))


if __name__ == "__main__":
    main()
