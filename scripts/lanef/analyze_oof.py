"""Lane F local analysis (CPU): assemble own out-of-fold (OOF) teacher columns from four hash-fold S6 fits and compare them with
Lane B's R6-flip / R7-plain OOF columns on the same studies.

  python3.12 scripts/lanef/analyze_oof.py --raw DIR --out DIR [--laneb outputs/laneb-20261019/analysis]
                                          [--baseline-own PREV/oof_columns.csv] [--folds 0,1,2,3]

--raw holds one sub-directory per fold, f<k>/, with the trainer's --only-folds k outputs: partial_oof_E_prior_s<seed>.npy
(pool x 12 logits; only fold-k rows are filled), partial_preds_E_prior_s<seed>.json (gold logits divided by K by the trainer),
pool_uids.json and targets.csv (the hash-fold column).  Only the fold-k rows of each array are used; the rows of every other
fold must be exactly zero.  Truth on the pool = labels_v1_blend >= 0.5; macro = mean of 12 per-target AUCs; paired study
bootstrap 2,000 draws, seed 42 (knee.paired_validation), identical to Lane B.  Gold-58 = mean logit of the fold models (one
read).  Writes IDs only under --out (ignored outputs/); prints aggregates only."""
import argparse, csv, glob, hashlib, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from knee.paired_validation import _auc_draws, resample_counts  # noqa: E402

MAIN = Path("~/Dev/rsna-knee-abnormality-detection")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]
BLEND = MAIN / "data/weak-labels/nartaa_rsna-knee-hpo-assets/labels_v1_blend.parquet.csv"
TRAIN = MAIN / "data/raw/train.csv"
DRAWS, SEED = 2000, 42


def sigmoid(z):
    z = np.asarray(z, np.float64)
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def ranks(p):
    """per-column average rank / n in (0, 1] (ties averaged)."""
    out = np.empty_like(p, dtype=np.float64)
    for j in range(p.shape[1]):
        v = p[:, j]; order = np.argsort(v, kind="stable"); sv = v[order]
        r = np.empty(len(v)); i = 0
        while i < len(v):
            k = i
            while k + 1 < len(v) and sv[k + 1] == sv[i]:
                k += 1
            r[order[i:k + 1]] = 0.5 * (i + k) + 1; i = k + 1
        out[:, j] = r / len(v)
    return out


def macro_auc(y, p):
    a = _auc_draws(y, p, np.ones((1, len(y)), dtype=np.int64))[0]
    return float(np.nanmean(a)), a


def boot_table(y, cols, pairs, groups, draws=DRAWS, seed=SEED):
    counts = resample_counts(groups, draws=draws, seed=seed)
    per = {n: _auc_draws(y, p, counts) for n, p in cols.items()}
    out = {}
    for a, b in pairs:
        ok = np.isfinite(per[a]).all(1) & np.isfinite(per[b]).all(1)
        d = per[a][ok].mean(1) - per[b][ok].mean(1)
        point = macro_auc(y, cols[a])[0] - macro_auc(y, cols[b])[0]
        out["%s - %s" % (a, b)] = {"delta": point, "ci95": np.quantile(d, [.025, .975]).tolist(), "sd": float(d.std()),
                                   "p_gt0": float((d > 0).mean()), "valid_draws": int(ok.sum())}
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_fold(d, k):
    """Return (pool_uids, fold-k uids, fold-k logits [n,12], gold uids, gold logits of this fold model [58,12], meta)."""
    d = Path(d)
    pj = sorted(glob.glob(str(d / "partial_preds_E_prior_s*.json")))
    assert len(pj) == 1, "expected one partial_preds json in %s" % d
    meta = json.load(open(pj[0]))
    seed = int(meta["seed"])
    assert meta["only_folds"] == [k], "fold dir %s holds only_folds %s" % (d, meta["only_folds"])
    assert meta["labels"] == LABELS and meta["holdout_fold"] is None
    pool = json.load(open(d / "pool_uids.json"))
    z = np.load(d / ("partial_oof_E_prior_s%d.npy" % seed)).astype(np.float64)
    assert z.shape == (len(pool), 12)
    fold = {}
    with open(d / "targets.csv", newline="") as f:
        for r in csv.DictReader(f):
            fold[r["StudyInstanceUID"]] = int(r["fold"])
    fo = np.asarray([fold[u] for u in pool])
    K = int(json.load(open(d / "run_config.json"))["folds"])
    assert K == 4 and set(np.unique(fo)) <= {0, 1, 2, 3}
    ho = np.flatnonzero(fo == k); other = np.flatnonzero(fo != k)
    assert (z[other] == 0).all(), "rows outside fold %d are not zero in %s" % (k, d)
    assert np.isfinite(z[ho]).all() and (np.abs(z[ho]).sum(1) > 0).all(), "unfilled held-out rows in %s" % d
    g = np.asarray(meta["gold_logits"], np.float64) * K                   # the trainer divides the gold logits by K
    info = {"fold": k, "seed": seed, "n_train": int(len(other)), "n_heldout": int(len(ho)), "epochs": meta["epochs"],
            "loss_first_last": meta["loss_first_last"], "train_seconds": meta["train_seconds"],
            "optimizer_steps": meta["optimizer_steps"], "pool_sha256": hashlib.sha256("\n".join(pool).encode()).hexdigest(),
            "targets_sha256": sha(d / "targets.csv"), "oof_npy_sha256": sha(d / ("partial_oof_E_prior_s%d.npy" % seed))}
    return pool, [pool[i] for i in ho], z[ho], meta["gold_uids"], g, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--laneb", default=str(MAIN / "outputs/laneb-20261019/analysis"))
    ap.add_argument("--baseline-own", default=None, help="a previous round's oof_columns.csv for the stop rule")
    ap.add_argument("--folds", default="0,1,2,3")
    ap.add_argument("--name", default="own")
    ap.add_argument("--blend", default=str(BLEND)); ap.add_argument("--train", default=str(TRAIN))
    a = ap.parse_args()
    BLEND_P, TRAIN_P = Path(a.blend), Path(a.train)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    folds = [int(x) for x in a.folds.split(",")]

    own, fold_of, infos, gold_uids, gold_z = {}, {}, [], None, {}
    pool_ref = None
    for k in folds:
        pool, uids, z, gu, g, info = load_fold(Path(a.raw) / ("f%d" % k), k)
        if pool_ref is None:
            pool_ref, gold_uids = pool, gu
        assert pool == pool_ref and gu == gold_uids, "pool or gold differ between fold runs"
        for u, row in zip(uids, z):
            own[u] = row; fold_of[u] = k
        gold_z[k] = g; infos.append(info)
    studies = sorted(own)
    Z = np.stack([own[u] for u in studies]); P = sigmoid(Z); fo = np.asarray([fold_of[u] for u in studies])

    with open(out / "oof_columns.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["StudyInstanceUID", "fold"] + ["%s logit|%s" % (a.name, l) for l in LABELS] + ["%s|%s" % (a.name, l) for l in LABELS])
        for i, u in enumerate(studies):
            w.writerow([u, int(fo[i])] + ["%.6f" % v for v in Z[i]] + ["%.6f" % v for v in P[i]])

    # ---- truth and Lane B columns on the same studies
    blend = {}
    with open(BLEND_P, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            blend[r["StudyInstanceUID"]] = [float(r[l]) for l in LABELS]
    lb = {}
    with open(Path(a.laneb) / "oof_columns.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lb[r["StudyInstanceUID"]] = r
    common = [u for u in studies if u in blend and u in lb]
    idx = np.asarray([studies.index(u) for u in common])
    y = np.asarray([[1.0 if v >= 0.5 else 0.0 for v in blend[u]] for u in common])
    lbfold = np.asarray([int(lb[u]["fold"]) for u in common])
    assert (lbfold == fo[idx]).all(), "own hash folds differ from Lane B's fold ids"
    cols = {a.name: P[idx],
            "R6 flip": np.asarray([[float(lb[u]["R6 flip|%s" % l]) for l in LABELS] for u in common]),
            "R7 plain": np.asarray([[float(lb[u]["R7 plain|%s" % l]) for l in LABELS] for u in common])}
    rk = {n: ranks(p) for n, p in cols.items()}
    cols["1/2 rank(R6 flip) + 1/2 rank(%s)" % a.name] = 0.5 * rk["R6 flip"] + 0.5 * rk[a.name]
    cols["1/2 rank(R6 flip) + 1/2 rank(R7 plain)"] = 0.5 * rk["R6 flip"] + 0.5 * rk["R7 plain"]
    cols["1/3 rank(R6 flip, R7 plain, %s)" % a.name] = (rk["R6 flip"] + rk["R7 plain"] + rk[a.name]) / 3
    pairs = [(a.name, "R6 flip"), (a.name, "R7 plain"),
             ("1/2 rank(R6 flip) + 1/2 rank(%s)" % a.name, "R6 flip"),
             ("1/2 rank(R6 flip) + 1/2 rank(%s)" % a.name, "1/2 rank(R6 flip) + 1/2 rank(R7 plain)"),
             ("1/3 rank(R6 flip, R7 plain, %s)" % a.name, "1/2 rank(R6 flip) + 1/2 rank(R7 plain)")]
    if a.baseline_own:
        base = {}
        with open(a.baseline_own, newline="", encoding="utf-8") as f:
            hdr = None
            for r in csv.DictReader(f):
                hdr = hdr or [c for c in r if c.endswith("|" + LABELS[0]) and "logit" not in c][0].split("|")[0]
                base[r["StudyInstanceUID"]] = [float(r["%s|%s" % (hdr, l)]) for l in LABELS]
        keep = [i for i, u in enumerate(common) if u in base]
        if len(keep) < len(common):
            print("baseline covers %d of %d studies; the stop-rule comparison uses the overlap" % (len(keep), len(common)))
        cols_b = {n: p[keep] for n, p in cols.items()}; cols_b["baseline own"] = np.asarray([base[common[i]] for i in keep])
        yb = y[keep]
        bt = boot_table(yb, {a.name: cols_b[a.name], "baseline own": cols_b["baseline own"]}, [(a.name, "baseline own")],
                        [common[i] for i in keep])
        d = bt["%s - baseline own" % a.name]
        stop_rule = {"delta": d["delta"], "ci95": d["ci95"], "n": len(keep),
                     "passed": bool(d["delta"] >= 0.002 and d["ci95"][0] > 0),
                     "text": "round passes only if own OOF pooled macro beats the previous round by >= +0.002 with the paired 95% CI excluding 0"}
    else:
        stop_rule = None

    tab = {"n": len(common), "truth": "labels_v1_blend >= 0.5", "folds_present": folds,
           "fold_counts": np.bincount(fo[idx], minlength=4).tolist(), "support_pos": y.sum(0).astype(int).tolist(),
           "macro": {}, "per_target": {}, "per_fold_macro": {}}
    for n, p in cols.items():
        m, per = macro_auc(y, p)
        tab["macro"][n] = m; tab["per_target"][n] = dict(zip(LABELS, per.tolist()))
        tab["per_fold_macro"][n] = {str(k): macro_auc(y[fo[idx] == k], p[fo[idx] == k])[0] for k in folds}
    tab["paired_bootstrap"] = boot_table(y, cols, pairs, common)
    tab["bootstrap"] = {"draws": DRAWS, "seed": SEED, "unit": "study", "macro": "mean of 12 per-target AUCs; draws with an undefined target dropped"}

    # ---- gold-58: mean logit of the fold models (one read; exposed)
    gold = {}
    with open(TRAIN_P, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if all(r[l] not in ("", None) for l in LABELS):
                gold[r["StudyInstanceUID"]] = [float(r[l]) for l in LABELS]
    assert set(gold_uids) <= set(gold), "gold uids missing from train.csv labels"
    yg = np.asarray([gold[u] for u in gold_uids])
    gz_mean = np.mean([gold_z[k] for k in folds], 0)
    g58 = {"n": len(gold_uids), "folds": folds, "macro_mean_logit": macro_auc(yg, gz_mean)[0],
           "per_fold_model": {str(k): macro_auc(yg, gold_z[k])[0] for k in folds},
           "per_target_mean_logit": dict(zip(LABELS, macro_auc(yg, gz_mean)[1].tolist())), "exposed": True,
           "read": "trainer end-of-fit prediction on the pixel cache, eval stride 1 (one read per fold model)"}
    lbg = Path(a.laneb) / "gold58_columns.csv"
    if lbg.exists():
        gl = {}
        with open(lbg, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                gl[r["StudyInstanceUID"]] = r
        if all(u in gl for u in gold_uids):
            r6 = np.asarray([[float(gl[u]["R6 flip|%s" % l]) for l in LABELS] for u in gold_uids])
            g58["laneb_R6_flip_4model_macro"] = macro_auc(yg, r6)[0]
            g58["paired_own_minus_R6_flip"] = boot_table(yg, {"own": sigmoid(gz_mean), "R6 flip": r6}, [("own", "R6 flip")], list(gold_uids))["own - R6 flip"]
    with open(out / "gold58_columns.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["StudyInstanceUID"] + ["f%d logit|%s" % (k, l) for k in folds for l in LABELS] + ["mean logit|%s" % l for l in LABELS])
        for i, u in enumerate(gold_uids):
            w.writerow([u] + ["%.5f" % gold_z[k][i, j] for k in folds for j in range(12)] + ["%.5f" % v for v in gz_mean[i]])

    summary = {"name": a.name, "folds": infos, "decision_table": tab, "gold58": g58, "stop_rule": stop_rule,
               "files": {"oof_columns.csv": sha(out / "oof_columns.csv"), "gold58_columns.csv": sha(out / "gold58_columns.csv")},
               "laneb_reference": {"R6 flip single-fold OOF macro (Lane B)": 0.9066, "R7 plain": 0.9051}}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps({"n": tab["n"], "fold_counts": tab["fold_counts"], "macro": tab["macro"], "per_fold_macro": tab["per_fold_macro"],
                      "paired_bootstrap": tab["paired_bootstrap"], "gold58": {k: v for k, v in g58.items() if k != "per_target_mean_logit"},
                      "stop_rule": stop_rule, "fits": [{k: v for k, v in i.items() if k in ("fold", "n_train", "n_heldout", "loss_first_last", "train_seconds")} for i in infos]}, indent=1))


if __name__ == "__main__":
    main()
