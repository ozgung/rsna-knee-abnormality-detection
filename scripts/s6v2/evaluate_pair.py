"""Read-out of the s6v2 fold-0 pair (J0 vs V2): gold-58 macro AUC with a paired study-level bootstrap and per-finding deltas,
fold-0 OOF against report-derived references, Spearman V2 vs J0 on fold 0, and the pre-registered rules 1-3.
Aggregates only (no IDs).  Usage:
  python -I evaluate_pair.py <repo root> <J0 run dir> <V2 run dir> <out json> [--gold-csv-J0 f --gold-csv-V2 f]
A run dir holds the trainer's partial_preds_E_prior_s42.json, partial_oof_E_prior_s42.npy and pool_uids.json.  Optional
--gold-npz-<arm> files (uids, logits) from compact_infer replace the trainer's gold logits for the primary endpoint."""
import argparse, csv, hashlib, json, sys
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


def endpoint_support(y):
    """Aggregate observed binary-label support; missing labels never become negatives."""
    y = np.asarray(y, float)
    if y.ndim != 2 or y.shape[1] != len(LABELS):
        raise ValueError("support array must have the canonical 12 target columns")
    known = np.isfinite(y)
    if not np.isin(y[known], [0., 1.]).all():
        raise ValueError("support expects binary observed reference labels")
    out = {}
    for j, label in enumerate(LABELS):
        positive = int(((y[:, j] == 1) & known[:, j]).sum())
        negative = int(((y[:, j] == 0) & known[:, j]).sum())
        out[label] = {"observed": positive + negative, "positive": positive, "negative": negative,
                      "missing": int((~known[:, j]).sum()), "undefined": positive == 0 or negative == 0}
    return out


def idsha(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def load_run(d):
    """Fails closed on label order, duplicate IDs or shape mismatches (Codex read-out audit r1)."""
    d = Path(d)
    pr = json.load(open(d / "partial_preds_E_prior_s42.json"))
    oof = np.load(d / "partial_oof_E_prior_s42.npy")
    pool = json.load(open(d / "pool_uids.json"))
    if list(pr.get("labels", [])) != LABELS:
        raise ValueError("prediction label order differs from the canonical 12-label order (%s)" % d.name)
    if len(pool) != len(set(pool)) or oof.shape != (len(pool), len(LABELS)):
        raise ValueError("pool IDs duplicated or OOF shape mismatch (%s)" % d.name)
    g = pr["gold_uids"]
    if len(g) != len(set(g)) or np.asarray(pr["gold_logits"]).shape != (len(g), len(LABELS)):
        raise ValueError("gold IDs duplicated or gold logits shape mismatch (%s)" % d.name)
    if not np.isfinite(oof).all() or not np.isfinite(np.asarray(pr["gold_logits"], float)).all():
        raise ValueError("non-finite predictions (%s)" % d.name)
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
    ap.add_argument("--gold-csv-J0"); ap.add_argument("--gold-csv-V2")   # compact_infer predictions.csv (columns bound by name)
    ap.add_argument("--preflight", help="vm_preflight receipt: fold-0 holdout and gold ID hashes must match it")
    a = ap.parse_args()
    root = Path(a.root)
    gold = gold_table(root / "data/raw/train.csv")
    runs = {"J0": load_run(a.j0), "V2": load_run(a.v2)}
    res = {"labels": LABELS, "arms": {}, "reporting": {
        "macro_policy": "Unweighted arithmetic mean of all 12 target AUCs; macro undefined unless every target has both observed classes.",
        "prediction_export": "Partial fold-0 predictions; not full OOF.",
        "gold58_selection_exposed": True,
        "gold58_interpretation": "Selection-exposed development: V2 was chosen among about ten label variants using gold-58. Gradient-training exclusion does not restore independence.",
        "fold0_interpretation": "Report-derived, descriptive development comparison; teacher fold lineage can expose fold-0 reference labels.",
        "support": {}}}
    # ---- gold-58
    G = {}
    for arm, (pr, oof, pool) in runs.items():
        npz = getattr(a, "gold_npz_" + arm)
        gcsv = getattr(a, "gold_csv_" + arm)
        if gcsv:
            with open(gcsv, encoding="utf-8", newline="") as fh:
                rd = csv.DictReader(fh)
                if set(LABELS) - set(rd.fieldnames or []):
                    raise ValueError("compact_infer CSV lacks label columns (%s)" % arm)
                rr = list(rd)
            uids = [r[UID] for r in rr]; lg = np.array([[float(r[L]) for L in LABELS] for r in rr])
            src = "compact_infer predictions.csv (eval stride 1, raw DICOM)"
        elif npz:
            z = np.load(npz, allow_pickle=False); uids = [str(u) for u in z["uids"]]; lg = np.asarray(z["logits"], float)
            src = "compact_infer"
        else:
            uids, lg, src = pr["gold_uids"], np.asarray(pr["gold_logits"], float), "trainer predict (eval stride 3)"
        if len(uids) != len(set(uids)) or lg.shape != (len(uids), len(LABELS)):
            raise ValueError("gold IDs duplicated or logits shape mismatch (%s)" % arm)
        m = dict(zip(uids, lg))
        if set(m) != set(gold) or len(m) != 58:
            raise ValueError("gold coverage differs from the 58 gold studies (%s)" % arm)
        G[arm] = (m, src)
    order = sorted(gold)
    y = np.array([gold[u] for u in order])
    res["reporting"]["support"]["gold58"] = endpoint_support(y)
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
    expect0 = {u for u in tJ0 if fold[u] == 0}
    for arm in oofs:
        if set(oofs[arm]) != expect0:
            raise ValueError("fold-0 prediction IDs of %s differ from the targets' %d fold-0 studies" % (arm, len(expect0)))
    if a.preflight:
        pf = json.load(open(a.preflight))
        for name, rec in pf["arms"].items():
            if rec["fold0_holdout_sha256"] != idsha(expect0) or rec["fold0_holdout"] != len(expect0):
                raise ValueError("fold-0 holdout differs from the preflight membership (%s)" % name)
            if rec["gold_sha256"] != idsha(gold) or rec["gold"] != 58:
                raise ValueError("gold set differs from the preflight membership (%s)" % name)
        res["preflight_bound"] = {"fold0_holdout_sha256": idsha(expect0), "gold_sha256": idsha(gold)}
    common = sorted(set(oofs["J0"]) & set(oofs["V2"]))
    res["fold0_studies"] = {"J0": len(oofs["J0"]), "V2": len(oofs["V2"]), "common": len(common)}
    for ref_name, ref in (("J0_targets_ge_0.5", tJ0), ("labels_v1_blend_ge_0.5", blend)):
        yr = np.array([[1.0 if v >= 0.5 else 0.0 for v in ref[u]] for u in common])
        res["reporting"]["support"]["fold0_" + ref_name] = endpoint_support(yr)
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
