"""Read-out of the s6v2 fold-0 pair (J0 vs V2): gold-58 macro AUC with a paired study-level bootstrap and per-finding deltas,
fold-0 OOF against report-derived references, Spearman V2 vs J0 on fold 0, and the pre-registered rules 1-3.
Aggregates only (no IDs).  Usage:
  python -I evaluate_pair.py <repo root> <J0 run dir> <V2 run dir> <out json> [--gold-csv-J0 f --gold-csv-V2 f]
A run dir holds the trainer's partial_preds_E_prior_s42.json, partial_oof_E_prior_s42.npy and pool_uids.json.  Optional
--gold-npz-<arm> files (uids, logits) from compact_infer replace the trainer's gold logits for the primary endpoint."""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
UID = "StudyInstanceUID"
NARTAA_GOLD, NARTAA_OOF = 0.931, 0.907


def macro(y, p):
    a = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(a)) if len(a) == y.shape[1] else float("nan"), a


def per(y, p):
    return [float(roc_auc_score(y[:, j], p[:, j])) if 0 < y[:, j].sum() < len(y) else None for j in range(y.shape[1])]


def load_run(d):
    d = Path(d)
    pr = json.load(open(d / "partial_preds_E_prior_s42.json"))
    oof = np.load(d / "partial_oof_E_prior_s42.npy")
    pool = json.load(open(d / "pool_uids.json"))
    return pr, oof, pool


def gold_table(train_csv):
    out = {}
    with open(train_csv, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if all(r[L] != "" for L in LABELS):
                out[r[UID]] = [float(r[L]) for L in LABELS]
    return out


def targets(path):
    out, fold = {}, {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            fold[r[UID]] = int(r["fold"])
            if r["is_gold"] == "0" and r["review_holdout"] == "0":
                out[r[UID]] = [float(r[L + "|prior"]) for L in LABELS]
    return out, fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("j0"); ap.add_argument("v2"); ap.add_argument("out")
    ap.add_argument("--gold-npz-J0"); ap.add_argument("--gold-npz-V2")
    a = ap.parse_args()
    root = Path(a.root)
    gold = gold_table(root / "data/raw/train.csv")
    runs = {"J0": load_run(a.j0), "V2": load_run(a.v2)}
    res = {"labels": LABELS, "arms": {}}
    # ---- gold-58
    G = {}
    for arm, (pr, oof, pool) in runs.items():
        npz = getattr(a, "gold_npz_" + arm)
        if npz:
            z = np.load(npz, allow_pickle=False); uids = [str(u) for u in z["uids"]]; lg = np.asarray(z["logits"], float)
            src = "compact_infer"
        else:
            uids, lg, src = pr["gold_uids"], np.asarray(pr["gold_logits"], float), "trainer predict (eval stride 3)"
        m = dict(zip(uids, lg))
        assert set(m) == set(gold) and len(m) == 58, "gold coverage differs"
        G[arm] = (m, src)
    order = sorted(gold)
    y = np.array([gold[u] for u in order])
    P = {arm: np.array([G[arm][0][u] for u in order]) for arm in G}
    for arm in P:
        mac, _ = macro(y, P[arm])
        res["arms"][arm] = {"gold58_macro": mac, "gold58_per_finding": dict(zip(LABELS, per(y, P[arm]))), "gold_source": G[arm][1]}
    rng = np.random.default_rng(42)
    deltas, skipped = [], 0
    for _ in range(2000):
        ix = rng.integers(0, len(order), len(order))
        yb = y[ix]
        if any(not (0 < yb[:, j].sum() < len(ix)) for j in range(12)):
            skipped += 1; continue
        deltas.append(macro(yb, P["V2"][ix])[0] - macro(yb, P["J0"][ix])[0])
    deltas = np.array(deltas)
    d_point = res["arms"]["V2"]["gold58_macro"] - res["arms"]["J0"]["gold58_macro"]
    res["gold58_delta"] = {"point": d_point, "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
                           "p_gt_0": float((deltas > 0).mean()), "draws_used": int(len(deltas)), "draws_skipped_single_class": skipped,
                           "per_finding": {L: (res["arms"]["V2"]["gold58_per_finding"][L] - res["arms"]["J0"]["gold58_per_finding"][L])
                                           for L in LABELS}}
    # ---- fold-0 OOF (report-derived references; descriptive)
    tJ0, fold = targets(root / "outputs/s6v2/targets/targets_J0_f0.csv")
    blend = {}
    with open(root / "data/weak-labels/nartaa_rsna-knee-hpo-assets/labels_v1_blend.parquet.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            blend[r[UID]] = [float(r[L]) for L in LABELS]
    oofs = {}
    for arm, (pr, oof, pool) in runs.items():
        ix = [i for i, u in enumerate(pool) if fold[u] == 0]
        oofs[arm] = ({pool[i]: oof[i] for i in ix})
    common = sorted(set(oofs["J0"]) & set(oofs["V2"]))
    res["fold0_studies"] = {"J0": len(oofs["J0"]), "V2": len(oofs["V2"]), "common": len(common)}
    for ref_name, ref in (("J0_targets_ge_0.5", tJ0), ("labels_v1_blend_ge_0.5", blend)):
        yr = np.array([[1.0 if v >= 0.5 else 0.0 for v in ref[u]] for u in common])
        for arm in oofs:
            po = np.array([oofs[arm][u] for u in common])
            mac, _ = macro(yr, po)
            res["arms"][arm]["fold0_oof_macro_" + ref_name] = mac
            res["arms"][arm]["fold0_oof_per_finding_" + ref_name] = dict(zip(LABELS, per(yr, po)))
        res["fold0_oof_delta_" + ref_name] = res["arms"]["V2"]["fold0_oof_macro_" + ref_name] - res["arms"]["J0"]["fold0_oof_macro_" + ref_name]
    pj = np.array([oofs["J0"][u] for u in common]); pv = np.array([oofs["V2"][u] for u in common])
    res["fold0_spearman_V2_vs_J0"] = {L: float(spearmanr(pj[:, j], pv[:, j]).statistic) for j, L in enumerate(LABELS)}
    # ---- pre-registered rules (primary OOF reference: J0 targets >= 0.5)
    j0g, j0o = res["arms"]["J0"]["gold58_macro"], res["arms"]["J0"]["fold0_oof_macro_J0_targets_ge_0.5"]
    od = res["fold0_oof_delta_J0_targets_ge_0.5"]
    # rule 1 gates every actionable verdict: NaN / undefined sanity compares False and blocks (DEBUG); raw threshold flags are kept apart
    res["rules"] = {
        "rule1_j0_sanity": {"gold_in_band": bool(abs(j0g - NARTAA_GOLD) <= 0.02), "oof_in_band": bool(abs(j0o - NARTAA_OOF) <= 0.01),
                            "pass": bool(abs(j0g - NARTAA_GOLD) <= 0.02 and abs(j0o - NARTAA_OOF) <= 0.01),
                            "j0_gold": j0g, "j0_oof": j0o,
                            "bands": "gold 0.931 +- 0.02, oof 0.907 +- 0.01; nartaa's 0.907 is on their labels / eval path and may not be comparable to ours"},
        "raw_delta_flags": {"promote_thresholds_met": bool(d_point >= 0.005 and od >= -0.010), "kill_threshold_met": bool(d_point <= -0.010),
                            "gold_delta": d_point, "oof_delta": od},
        "rule2_promote": bool(abs(j0g - NARTAA_GOLD) <= 0.02 and abs(j0o - NARTAA_OOF) <= 0.01 and d_point >= 0.005 and od >= -0.010),
        "rule3_kill": bool(abs(j0g - NARTAA_GOLD) <= 0.02 and abs(j0o - NARTAA_OOF) <= 0.01 and d_point <= -0.010),
        "verdict": ("BLOCKED_DEBUG_J0_SANITY" if not (abs(j0g - NARTAA_GOLD) <= 0.02 and abs(j0o - NARTAA_OOF) <= 0.01)
                    else "PROMOTE" if (d_point >= 0.005 and od >= -0.010) else "KILL" if d_point <= -0.010 else "NEITHER")}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("gold58_delta", "rules", "fold0_studies")}, indent=1))
    for arm in res["arms"]:
        print(arm, "gold58 %.4f" % res["arms"][arm]["gold58_macro"], "oof(J0ref) %.4f" % res["arms"][arm]["fold0_oof_macro_J0_targets_ge_0.5"])


if __name__ == "__main__":
    main()
