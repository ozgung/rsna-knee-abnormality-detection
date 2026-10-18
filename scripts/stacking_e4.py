"""E4: second-stage per-finding stacking on clean OOF predictions (zero GPU, no training of image models).

Question: can a per-finding logistic regression on the 12 OOF predictions of our own models
(own224 r0, CoAtNet-384 raw labels, CoAtNet-224 R2) decorrelate compartment-confused findings
and raise pooled masked macro AUC?

Leakage design (nested CV over the existing grouped folds): for held-out fold k the stacker for
target t is fitted ONLY on val-OOF rows of the other four folds with m_t = 1; the stacker's
regularisation strength C is chosen by an inner leave-one-fold-out loop over those four folds.
Fold k labels are never read while producing fold k predictions.  Rows are the exported `val`
rows of the five oof_fold{k}.csv files (gold-58 rows are `set == gold` and are dropped; the 4345
common studies are the non-gold, non-quarantined training studies).

Truth, AUC and masking conventions are reused from scripts/oof_blend_diagnostic.py.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oof_blend_diagnostic import LAB, _nk, auc, report  # noqa: E402

PAIRS = [("Medial OA", "Lateral OA"), ("Lateral OA", "PF OA"), ("Contusion", "Fracture"), ("Effusion", "Synovitis")]
DIFFS = [("Lateral OA", "Medial OA"), ("Lateral OA", "PF OA"), ("Contusion", "Fracture"), ("Synovitis", "Effusion")]
C_GRID = (0.003, 0.03, 0.3)
EPS = 1e-6
SPOTLIGHT = ["Lateral OA", "PF OA", "Medial OA", "Fracture", "Contusion", "Synovitis", "Lateral Meniscus"]

R = Path("outputs")
FAMILY_PATHS = {
    "own224": [R / f"analysis/oof-coatnet224-r0/oof_fold{k}.csv" for k in range(5)],
    "c384r": [R / "kaggle" / d / "trackA" / f"oof_fold{k}.csv" for k, d in enumerate([
        "trackA-export-c384r-fold0-20261010-result", "trackA-export-c384r-fold1-20261010-result",
        "trackA-export-c384r-fold2-20261010-result", "trackA-export-c384r-fold3-provenance-20261011-result",
        "trackA-export-c384r-fold4-provenance-20261011-result"])],
    "r2c224": [R / "kaggle" / d / "trackA" / f"oof_fold{k}.csv" for k, d in enumerate([
        "trackA-coatnet224-r2-fold01-20261010-result", "trackA-coatnet224-r2-fold01-20261010-result",
        "trackA-coatnet224-r2-fold23-20261010-result", "trackA-coatnet224-r2-fold23-20261010-result",
        "trackA-coatnet224-r2-fold4-20261010-result"])],
}


# ----------------------------------------------------------------------------- data
def load_val(paths):
    """Concatenate the `val` rows of five oof_fold{k}.csv files; column `fold` = k."""
    parts = []
    for k, p in enumerate(paths):
        df = pd.read_csv(p, dtype={"StudyInstanceUID": str})
        v = df[df["set"] == "val"].copy()
        v["fold"] = k
        parts.append(v)
    out = pd.concat(parts).set_index("StudyInstanceUID")
    if out.index.has_duplicates:
        raise ValueError("duplicate val study across folds")
    return out


def load_truth(truth_csv, ids):
    truth = pd.read_csv(truth_csv, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    yc = {_nk(c[2:]): c for c in truth.columns if c.startswith("y_")}
    mc = {_nk(c[2:]): c for c in truth.columns if c.startswith("m_")}
    rows = truth.loc[ids]
    Y = (np.stack([rows[yc[_nk(t)]].values for t in LAB], 1).astype(float) >= 0.5).astype(int)
    M = np.stack([rows[mc[_nk(t)]].values for t in LAB], 1).astype(float)
    return Y, M


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ----------------------------------------------------------------------------- stacker
def make_model(kind, C=0.03):
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_depth=2, max_iter=120, learning_rate=0.04, min_samples_leaf=40,
                                              l2_regularization=10.0, early_stopping=False, random_state=42)
    raise ValueError(kind)


def _score(model, X):
    return model.decision_function(X) if hasattr(model, "decision_function") else logit(model.predict_proba(X)[:, 1])


def _fit_ok(y):
    return len(np.unique(y)) == 2 and min(int(y.sum()), int((1 - y).sum())) >= 5


def inner_select_C(X, Y, M, fold, rows, t, grid):
    """Pick C for target t by leave-one-fold-out AUC inside `rows` (indices of the outer training rows)."""
    best, best_a = grid[len(grid) // 2], -1.0
    folds = np.unique(fold[rows])
    for C in grid:
        pooled_y, pooled_p = [], []
        for f in folds:
            tr = rows[(fold[rows] != f) & (M[rows, t] > 0)]
            te = rows[(fold[rows] == f) & (M[rows, t] > 0)]
            if not _fit_ok(Y[tr, t]) or len(te) == 0:
                continue
            m = make_model("lr", C).fit(X[tr], Y[tr, t])
            pooled_y.append(Y[te, t]); pooled_p.append(_score(m, X[te]))
        if pooled_y:
            a = auc(np.concatenate(pooled_y), np.concatenate(pooled_p))
            if a is not None and a > best_a:
                best, best_a = C, a
    return best


def nested_stack(X, Y, M, fold, kind="lr", C=None, grid=C_GRID, record=None, coefs=None):
    """Out-of-fold stacked scores (n x 12).  C=None -> inner-selected per (outer fold, target).

    `record(k, t, train_rows)` is called before each outer fit so tests can assert that the
    training rows never include fold k.  `coefs` (dict) collects LR coefficients per (k, t).
    """
    n = len(X)
    out = np.full((n, len(LAB)), np.nan)
    for k in np.unique(fold):
        te = np.where(fold == k)[0]
        trall = np.where(fold != k)[0]
        for t in range(len(LAB)):
            tr = trall[M[trall, t] > 0]
            if record is not None:
                record(k, t, tr)
            if not _fit_ok(Y[tr, t]):
                out[te, t] = 0.0
                continue
            if kind == "lr":
                c = C if C is not None else inner_select_C(X, Y, M, fold, trall, t, grid)
            else:
                c = None
            m = make_model(kind, c if c is not None else 0.03).fit(X[tr], Y[tr, t])
            out[te, t] = _score(m, X[te])
            if coefs is not None and kind == "lr":
                coefs[(int(k), t)] = (m[-1].coef_[0].copy(), float(c))
    return out


# ----------------------------------------------------------------------------- features
def feats_logit(probs):
    return np.concatenate([logit(p) for p in probs], axis=1)


def feats_diff(probs):
    """Logits plus compartment differences (for the tree model; linear models make these redundant)."""
    L = [logit(p) for p in probs]
    ix = {n: i for i, n in enumerate(LAB)}
    extra = [l[:, ix[a]] - l[:, ix[b]] for l in L for a, b in DIFFS]
    return np.concatenate(L + [np.stack(extra, 1)], axis=1)


# ----------------------------------------------------------------------------- bootstrap
def fast_per_target(Y, M, mats, idx):
    """Per-target AUC for each score matrix on resampled rows idx; returns {name: (12,) with nan}."""
    from scipy.stats import rankdata
    res = {k: np.full(len(LAB), np.nan) for k in mats}
    for t in range(len(LAB)):
        keep = idx[M[idx, t] > 0]
        y = Y[keep, t]
        npos, nneg = int(y.sum()), int((1 - y).sum())
        if not npos or not nneg:
            continue
        for k, P in mats.items():
            r = rankdata(P[keep, t])
            res[k][t] = (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return res


def paired_bootstrap(Y, M, mats, pairs, draws=2000, seed=42):
    """pairs: {label: (stacked_name, identity_name)}.  Resamples studies; returns macro and per-target
    mean difference with 95% percentile CI plus P(diff <= 0)."""
    rng = np.random.default_rng(seed)
    n = len(Y)
    diffs_macro = {k: np.empty(draws) for k in pairs}
    diffs_t = {k: np.empty((draws, len(LAB))) for k in pairs}
    for d in range(draws):
        idx = rng.integers(0, n, n)
        res = fast_per_target(Y, M, mats, idx)
        for k, (a, b) in pairs.items():
            dt = res[a] - res[b]
            diffs_t[k][d] = dt
            diffs_macro[k][d] = np.nanmean(res[a]) - np.nanmean(res[b])
    out = {}
    for k in pairs:
        def summ(v):
            v = v[~np.isnan(v)]
            return {"mean": float(v.mean()), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                    "p_le_0": float((v <= 0).mean())}
        out[k] = {"macro": summ(diffs_macro[k]), "per_target": {LAB[t]: summ(diffs_t[k][:, t]) for t in range(len(LAB))}}
    return out


# ----------------------------------------------------------------------------- confusion diagnostics
def phi(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def pair_diagnostics(Y, M, P_id, P_st, L_id):
    """Per compartment pair: label co-occurrence/phi, prediction correlation (identity logit vs stacked score),
    cross-AUC (rank target a by head b) for identity, and own AUC for identity and stacked."""
    ix = {n: i for i, n in enumerate(LAB)}
    out = {}
    for a, b in PAIRS:
        i, j = ix[a], ix[b]
        both = (M[:, i] > 0) & (M[:, j] > 0)
        co = float(((Y[both, i] == 1) & (Y[both, j] == 1)).sum() / max(1, ((Y[both, i] == 1) | (Y[both, j] == 1)).sum()))
        d = {"n_both_labelled": int(both.sum()), "label_phi": phi(Y[both, i], Y[both, j]), "label_jaccard": co,
             "pred_corr_identity_logit": phi(L_id[:, i], L_id[:, j]), "pred_corr_stacked": phi(P_st[:, i], P_st[:, j])}
        for tgt, oth in ((a, b), (b, a)):
            ti, oi = ix[tgt], ix[oth]
            k = M[:, ti] > 0
            d[f"{tgt}|own_head_identity"] = auc(Y[k, ti], P_id[k, ti])
            d[f"{tgt}|ranked_by_{oth}_head"] = auc(Y[k, ti], P_id[k, oi])
            d[f"{tgt}|stacked"] = auc(Y[k, ti], P_st[k, ti])
        out[f"{a}/{b}"] = d
    return out


def coef_summary(coefs, feat_names, targets):
    """Mean standardized coefficient over outer folds for selected targets, top-4 by |coef|."""
    out = {}
    for t in targets:
        ti = LAB.index(t)
        cs = np.array([coefs[(k, ti)][0] for k in range(5) if (k, ti) in coefs])
        Cs = [coefs[(k, ti)][1] for k in range(5) if (k, ti) in coefs]
        mean = cs.mean(0)
        order = np.argsort(-np.abs(mean))[:4]
        own = feat_names.index(f"{t}")
        out[t] = {"chosen_C": Cs, "own_coef": float(mean[own]), "top": [(feat_names[o], float(mean[o])) for o in order]}
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth-csv", default="data/weak-labels/nartaa_rsna-knee-hpo-assets/train_labels_v1.parquet.csv")
    ap.add_argument("--out", type=Path, default=Path("outputs/analysis/stacking-e4-20261013"))
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    t0 = time.time()
    fams = {n: load_val(p) for n, p in FAMILY_PATHS.items()}
    common = fams["own224"].index
    for n in fams:
        common = common.intersection(fams[n].index)
    print("val rows per family", {n: len(f) for n, f in fams.items()}, "common", len(common))
    fold = fams["c384r"].loc[common, "fold"].values
    for n in fams:
        assert (fams[n].loc[common, "fold"].values == fold).all(), f"fold assignment differs for {n}"
    Y, M = load_truth(a.truth_csv, common)
    P = {n: fams[n].loc[common, LAB].values.astype(float) for n in fams}
    P["blend2"] = 0.5 * (P["c384r"] + P["r2c224"])
    P["blend3"] = (P["own224"] + P["c384r"] + P["r2c224"]) / 3

    scores = {}   # name -> score matrix
    for n in ("own224", "c384r", "r2c224", "blend2", "blend3"):
        scores[f"id:{n}"] = P[n]
    # stack configurations: name -> (list of prob matrices as inputs, identity reference)
    cfg = {
        "own224": ([P["own224"]], "id:own224"),
        "c384r": ([P["c384r"]], "id:c384r"),
        "r2c224": ([P["r2c224"]], "id:r2c224"),
        "blend2": ([P["blend2"]], "id:blend2"),
        "c384r+r2c224": ([P["c384r"], P["r2c224"]], "id:blend2"),
        "all3": ([P["own224"], P["c384r"], P["r2c224"]], "id:blend3"),
    }
    coefs_store = {}
    for name, (probs, _) in cfg.items():
        X = feats_logit(probs)
        cf = {}
        scores[f"lr:{name}"] = nested_stack(X, Y, M, fold, "lr", None, coefs=cf)
        coefs_store[name] = cf
        for C in C_GRID:
            scores[f"lr@{C}:{name}"] = nested_stack(X, Y, M, fold, "lr", C)
        print("done", name, round(time.time() - t0), "s", flush=True)
    for name in ("own224", "blend2", "all3"):
        scores[f"gbm:{name}"] = nested_stack(feats_diff(cfg[name][0]), Y, M, fold, "gbm")
        print("done gbm", name, round(time.time() - t0), "s", flush=True)

    res = {"n_studies": int(len(common)), "folds": [int(np.sum(fold == k)) for k in range(5)],
           "label_support": {t: {"n": int(M[:, j].sum()), "pos": int(Y[M[:, j] > 0, j].sum())} for j, t in enumerate(LAB)},
           "pooled": {}}
    for k, S in scores.items():
        r = report(Y, S, M)
        res["pooled"][k] = {"macro_auc": r["macro_auc_defined"], "per_target": r["per_target"], "undefined": r["undefined_targets"]}

    pairs = {}
    for name, (_, ident) in cfg.items():
        pairs[f"lr:{name}"] = (f"lr:{name}", ident)
    for name in ("own224", "blend2", "all3"):
        pairs[f"gbm:{name}"] = (f"gbm:{name}", cfg[name][1])
    pairs["id:blend2_vs_own224"] = ("id:blend2", "id:own224")
    pairs["id:blend2_vs_c384r"] = ("id:blend2", "id:c384r")
    pairs["lr:blend2_vs_id:blend2"] = ("lr:blend2", "id:blend2")
    pairs["lr:all3_vs_id:blend2"] = ("lr:all3", "id:blend2")
    pairs["lr:c384r+r2c224_vs_id:blend2"] = ("lr:c384r+r2c224", "id:blend2")
    keep_names = {x for p in pairs.values() for x in p}
    res["bootstrap"] = paired_bootstrap(Y, M, {k: scores[k] for k in keep_names}, pairs, a.draws, a.seed)
    res["bootstrap_pair_defs"] = {k: list(v) for k, v in pairs.items()}
    print("bootstrap done", round(time.time() - t0), "s", flush=True)

    ix = {n: i for i, n in enumerate(LAB)}
    res["pair_diagnostics"] = {}
    for name in ("own224", "c384r", "blend2", "all3"):
        probs, ident = cfg[name]
        Lid = logit(scores[ident])
        res["pair_diagnostics"][name] = pair_diagnostics(Y, M, scores[ident], scores[f"lr:{name}"], Lid)
    fn = lambda probs_n: [f"{t}" if n == 0 else f"{t}#{n}" for n in range(probs_n) for t in LAB]
    res["coefs"] = {}
    for name in ("own224", "blend2"):
        res["coefs"][name] = coef_summary(coefs_store[name], fn(len(cfg[name][0])), ["Lateral OA", "PF OA", "Medial OA", "Contusion", "Fracture", "Synovitis", "Effusion"])
    # linear separability probe: how well is each target predicted from the OTHER 11 outputs only (own224 + blend2)?
    probe = {}
    for name in ("own224", "blend2"):
        X = feats_logit(cfg[name][0])
        S = np.zeros_like(scores["id:own224"])
        for t in range(len(LAB)):
            Xt = np.delete(X, t, axis=1)
            S[:, t] = _others_only(Xt, Y, M, fold, t)
        probe[name] = report(Y, S, M)["per_target"]
    res["others_only_probe"] = probe
    res["runtime_s"] = round(time.time() - t0, 1)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out / "stacking_e4.json", "w"), indent=1)
    np.savez_compressed(a.out / "scores.npz", ids=np.array(common), fold=fold, **{k.replace(":", "_"): v for k, v in scores.items()})
    print_summary(res)


def _others_only(Xt, Y, M, fold, t):
    """OOF score for target t from the 11 other outputs (own column removed), nested over folds, C=0.03."""
    out = np.zeros(len(Xt))
    for k in np.unique(fold):
        tr = np.where((fold != k) & (M[:, t] > 0))[0]
        te = np.where(fold == k)[0]
        out[te] = _score(make_model("lr", 0.03).fit(Xt[tr], Y[tr, t]), Xt[te])
    return out


def print_summary(res):
    pooled, bs = res["pooled"], res["bootstrap"]
    print("pooled macro:")
    for k, v in pooled.items():
        if "@" in k:
            continue
        print(f"  {k:22s} {v['macro_auc']:.4f}")
    for k, v in bs.items():
        m = v["macro"]
        print(f"  d[{k}] {m['mean']:+.4f}  CI [{m['ci95'][0]:+.4f},{m['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
