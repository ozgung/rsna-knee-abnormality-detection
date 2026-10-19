"""Lane B local analysis (CPU): VM records.jsonl -> deployed R6/R7 probabilities (stackers applied offline), mapping check,
out-of-fold columns, the Lane A decision table (paired study bootstrap) and gold-58 read-outs.

  python3 scripts/laneb/analyze_oof.py --records R --out DIR [--parity-r6 DIR --parity-r7 DIR] [--final]

Probabilities follow compact_infer v4c: per model set S, z = mean over S of the per-model logits (R6 flip: the combined
0.5 (z + P z_m) logits; plain: the plain pass), p = sigmoid(z); Effusion from the bundle's fluid stacker (z, log1p volume, missing),
Baker's from its state stacker (z + the mean state read-out over S).  OOF column = S = {the study's hash-fold model}.
Gold-58 = S = all four fold models (gold is out-of-fold for every model).  Truth for the pool: labels_v1_blend >= 0.5.
Writes IDs only under --out (ignored outputs/); prints aggregates only."""
import argparse, csv, hashlib, json, math, sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/ozgun/Dev/rsna-knee-abnormality-detection")
sys.path.insert(0, str(ROOT))
from knee.paired_validation import _auc_draws, resample_counts  # noqa: E402

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
          "Baker's", "Contusion", "Fracture"]
BUNDLE_DIRS = {"r6": ROOT / "outputs/research/public-0958-20261017/r6-v1-assets",
               "r7": ROOT / "outputs/analysis/lanea-20261019/r7-v1-assets"}
BLEND = ROOT / "data/weak-labels/nartaa_rsna-knee-hpo-assets/labels_v1_blend.parquet.csv"
TRAIN = ROOT / "data/raw/train.csv"
DRAWS, SEED = 2000, 42


def sigmoid(z):
    z = np.asarray(z, np.float64)
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def state_layout(spec):
    cols, j = [], 0
    for l in LABELS:
        k = len(spec["categories"][l])
        cols.append({"class": l, "categories": list(spec["categories"][l]), "start": j, "stop": j + k}); j += k
    for b, labs in spec["binaries"].items():
        for l in labs:
            cols.append({"binary": b, "class": l, "column": j}); j += 1
    return {"columns": cols, "width": j}


def load_stackers(d):
    spec = json.load(open(d / "state_spec.json"))
    lay = state_layout(spec)
    fl = json.load(open(d / "fluid_stacker.json"))
    assert fl["feature"]["combine"] == "logit" and [float(x) for x in fl["feature"]["weights"]] == [1.0]
    fluid = {n: {k: float(r[k]) for k in ("b0", "b_z", "b_lv", "b_miss")} | {"vol": r["volume_column"]}
             for n, r in fl["classes"].items() if r.get("adopt")}
    st = json.load(open(d / "state_stacker.json"))
    assert st["kind"] == "state_stacker_v1" and st["layout"]["columns"] == lay["columns"]
    blocks = {c["class"]: c for c in lay["columns"] if "categories" in c}
    bins = [c for c in lay["columns"] if "binary" in c]
    state = {}
    for n, r in st["classes"].items():
        if not r.get("adopt"):
            continue
        b = blocks[n]
        want = ["z"] + ["state:" + c for c in b["categories"]] + ["bin:" + c["binary"] for c in bins if c["class"] == n]
        cols = [None] + list(range(b["start"], b["stop"])) + [c["column"] for c in bins if c["class"] == n]
        assert not r.get("volume_column")
        keep = [i for i, f in enumerate(want) if f not in set(r.get("dropped") or [])]
        want, cols = [want[i] for i in keep], [cols[i] for i in keep]
        assert list(r["features"]) == want
        state[n] = {"cols": cols, "feat": want, "coef": np.asarray(r["coef"], np.float64), "b0": float(r["b0"])}
    assert not set(fluid) & set(state)
    return {"fluid": fluid, "state": state, "width": lay["width"]}


STK = {k: load_stackers(v) for k, v in BUNDLE_DIRS.items()}


def deployed(rec, bundle, models, flip):
    """compact_infer v4c probabilities of one study from the recorded per-model outputs (None when a model is missing)."""
    g = rec.get(bundle) or {}
    if any(str(k) not in g for k in models):
        return None
    zk, sk = ("zt", "stt") if flip else ("z", "st")
    if flip and "zt" not in g[str(models[0])]:
        return None
    zs = np.asarray([g[str(k)][zk] for k in models], np.float32)
    z = zs.mean(0).astype(np.float64)                      # score(): zs.mean(0) in float32, the stackers read it as float64
    p = sigmoid(z)
    stv = np.asarray([g[str(k)][sk] for k in models], np.float64).mean(0)
    s = STK[bundle]
    fl = rec.get("fluid") or {}
    for n, co in s["fluid"].items():
        v = fl.get(co["vol"])
        miss = v is None or not math.isfinite(float(v)) or float(v) < 0
        lv = 0.0 if miss else float(np.log1p(float(v)))
        j = LABELS.index(n)
        p[j] = float(sigmoid(co["b0"] + co["b_z"] * z[j] + co["b_lv"] * lv + co["b_miss"] * (1.0 if miss else 0.0)))
    for n, co in s["state"].items():
        j = LABELS.index(n)
        x = np.asarray([z[j] if f == "z" else stv[c] for f, c in zip(co["feat"], co["cols"])], np.float64)
        p[j] = float(sigmoid(co["b0"] + float(x @ co["coef"])))
    return p


def ranks(p):
    """per-column average rank / n in (0, 1] (the entries' rank blend; ties averaged)."""
    def rankdata(v):
        order = np.argsort(v, kind="stable"); sv = v[order]
        r = np.empty(len(v)); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
        return r
    return np.stack([rankdata(p[:, j]) / len(p) for j in range(p.shape[1])], 1)


def macro_auc(y, p):
    a = _auc_draws(y, p, np.ones((1, len(y)), dtype=np.int64))[0]
    return float(np.nanmean(a)), a


def boot_table(y, cols, pairs, groups, draws=DRAWS, seed=SEED):
    counts = resample_counts(groups, draws=draws, seed=seed)
    per = {n: _auc_draws(y, p, counts) for n, p in cols.items()}             # [draws, 12]
    out = {}
    for a, b in pairs:
        ok = np.isfinite(per[a]).all(1) & np.isfinite(per[b]).all(1)
        d = per[a][ok].mean(1) - per[b][ok].mean(1)
        point = macro_auc(y, cols[a])[0] - macro_auc(y, cols[b])[0]
        out["%s - %s" % (a, b)] = {"delta": point, "ci95": np.quantile(d, [.025, .975]).tolist(), "sd": float(d.std()),
                                   "p_gt0": float((d > 0).mean()), "valid_draws": int(ok.sum())}
    return out


def load_records(path):
    recs = {}
    for line in open(path):
        r = json.loads(line)
        recs[r["uid"]] = r
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parity-r6"); ap.add_argument("--parity-r7")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs = load_records(a.records)
    blend = {r["StudyInstanceUID"]: [float(r[l]) for l in LABELS] for r in csv.DictReader(open(BLEND, encoding="utf-8"))}
    gold = {}
    for r in csv.DictReader(open(TRAIN, encoding="utf-8")):
        if all(r[l] != "" for l in LABELS):
            gold[r["StudyInstanceUID"]] = [float(r[l]) for l in LABELS]
    ok = {u: r for u, r in recs.items() if not r.get("error")}
    res = {"records": len(recs), "errors": len(recs) - len(ok), "error_kinds": {}}
    for r in recs.values():
        if r.get("error"):
            k = r["error"][:40]; res["error_kinds"][k] = res["error_kinds"].get(k, 0) + 1
    pool_ok = [u for u in ok if u not in gold]
    res["pool_scored"] = len(pool_ok); res["gold_scored"] = sum(u in gold for u in ok)
    # fold rule sanity: the recorded fold equals sha1(uid)[:8] % 4
    assert all(r["fold"] == int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4 for u, r in recs.items())

    # ---- mapping check: 'all'-mode pool studies, each fold model on its own hash fold vs the other three folds
    allm = [u for u in pool_ok if ok[u]["mode"] in ("all", "map")]
    mc = {"studies": len(allm)}
    if allm:
        y = np.asarray([[1.0 if v >= 0.5 else 0.0 for v in blend[u]] for u in allm])
        fo = np.asarray([ok[u]["fold"] for u in allm])
        for bundle, flip, tag in (("r6", False, "R6 plain"), ("r7", False, "R7 plain")):
            rows = []
            for k in range(4):
                p = np.stack([deployed(ok[u], bundle, [k], flip) for u in allm])
                own, _ = macro_auc(y[fo == k], p[fo == k])
                oth, _ = macro_auc(y[fo != k], p[fo != k])
                rows.append({"model": k, "own_fold_n": int((fo == k).sum()), "own_fold_auc": own, "other_folds_n": int((fo != k).sum()),
                             "other_folds_auc": oth, "gap": oth - own})
            mc[tag] = rows
        # agreement between fold models on a study: in-sample pairs vs a held-out member
        mc["verdict"] = "CONFIRMED" if all(r["gap"] > 0.02 for t in ("R6 plain", "R7 plain") for r in mc[t]) else "NOT_CONFIRMED"
    res["mapping_check"] = mc

    # ---- out-of-fold columns on the pool
    configs = {"R6 plain": ("r6", False), "R6 flip": ("r6", True), "R7 plain": ("r7", False), "R7 flip": ("r7", True)}
    pool = [u for u in sorted(pool_ok) if all(deployed(ok[u], b, [ok[u]["fold"]], f) is not None for b, f in configs.values()
                                             if not (f and b == "r7")) and u in blend]
    has_r7f = bool(pool) and all(deployed(ok[u], "r7", [ok[u]["fold"]], True) is not None for u in pool)
    cols = {}
    for name, (b, f) in configs.items():
        if name == "R7 flip" and not has_r7f:
            continue
        cols[name] = np.stack([deployed(ok[u], b, [ok[u]["fold"]], f) for u in pool]) if pool else np.zeros((0, 12))
    y = np.asarray([[1.0 if v >= 0.5 else 0.0 for v in blend[u]] for u in pool])
    with open(out / "oof_columns.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["StudyInstanceUID", "fold"] + ["%s|%s" % (c, l) for c in cols for l in LABELS])
        for i, u in enumerate(pool):
            w.writerow([u, ok[u]["fold"]] + ["%.9g" % cols[c][i, j] for c in cols for j in range(12)])
    if not pool:
        res["decision_table"] = None
    rk = {n: ranks(p) for n, p in cols.items()} if pool else {}
    if pool:
        cand = {"dual 1/2 rank(R6 plain) + 1/2 rank(R7 plain)": 0.5 * rk["R6 plain"] + 0.5 * rk["R7 plain"],
                "R6 flip": rk["R6 flip"], "R6 plain": rk["R6 plain"], "R7 plain": rk["R7 plain"],
                "dual-flip 1/2 rank(R6 flip) + 1/2 rank(R7 plain)": 0.5 * rk["R6 flip"] + 0.5 * rk["R7 plain"]}
        if "R7 flip" in rk:
            cand["R7 flip"] = rk["R7 flip"]
            cand["dual-both-flip 1/2 rank(R6 flip) + 1/2 rank(R7 flip)"] = 0.5 * rk["R6 flip"] + 0.5 * rk["R7 flip"]
        tab = {"n": len(pool), "truth": "labels_v1_blend >= 0.5", "fold_counts": np.bincount([ok[u]["fold"] for u in pool], minlength=4).tolist(),
               "support_pos": y.sum(0).astype(int).tolist(), "macro": {}, "per_target": {}}
        for n, p in cand.items():
            m, per = macro_auc(y, p)
            tab["macro"][n] = m; tab["per_target"][n] = dict(zip(LABELS, per.tolist()))
        D, F, P = list(cand)[0], "R6 flip", "R6 plain"
        pairs = [(D, F), (D, P), (F, P), ("R7 plain", P), (list(cand)[4], F)]
        if "R7 flip" in cand:
            pairs.append((list(cand)[6], F))
        tab["paired_bootstrap"] = boot_table(y, cand, pairs, pool) if len(pool) >= 50 else None
        tab["bootstrap"] = {"draws": DRAWS, "seed": SEED, "unit": "study", "macro": "mean of 12 per-target AUCs; draws with an undefined target dropped"}
        dd = tab["paired_bootstrap"]["%s - %s" % (D, F)] if tab["paired_bootstrap"] else None
        if dd:
            tab["rule"] = {"final1_adopt_P6": bool(dd["delta"] >= 0.0003 and dd["ci95"][0] > 0),
                           "final2_noninferior": bool(dd["delta"] >= -0.0003),
                           "text": "adopt P6 for final 1 only if delta(dual - R6 flip) >= +0.0003 with CI excluding 0; final 2 if delta >= -0.0003"}
        res["decision_table"] = tab

    # ---- gold-58: all four fold models (out-of-fold for every model); read once per artifact
    gu = sorted(u for u in ok if u in gold)
    if gu:
        yg = np.asarray([gold[u] for u in gu])
        gcols = {n: np.stack([deployed(ok[u], b, [0, 1, 2, 3], f) for u in gu]) for n, (b, f) in configs.items()
                 if all(deployed(ok[u], b, [0, 1, 2, 3], f) is not None for u in gu)}
        grk = {n: ranks(p) for n, p in gcols.items()}
        gc = {"R6 plain": grk["R6 plain"], "R6 flip": grk["R6 flip"], "R7 plain": grk["R7 plain"],
              "dual 1/2 rank(R6 plain) + 1/2 rank(R7 plain)": 0.5 * grk["R6 plain"] + 0.5 * grk["R7 plain"]}
        if "R7 flip" in grk:
            gc["R7 flip"] = grk["R7 flip"]
        res["gold58"] = {"n": len(gu), "macro": {n: macro_auc(yg, p)[0] for n, p in gc.items()},
                         "per_target": {n: dict(zip(LABELS, macro_auc(yg, p)[1].tolist())) for n, p in gc.items()},
                         "per_fold_model_R6_flip": [macro_auc(yg, np.stack([deployed(ok[u], "r6", [k], True) for u in gu]))[0] for k in range(4)],
                         "per_fold_model_R7_plain": [macro_auc(yg, np.stack([deployed(ok[u], "r7", [k], False) for u in gu]))[0] for k in range(4)],
                         "exposure": "gold-58 read for Lane B sanity (pipeline check); no selection on it"}
        with open(out / "gold58_columns.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["StudyInstanceUID"] + ["%s|%s" % (c, l) for c in gcols for l in LABELS])
            for i, u in enumerate(gu):
                w.writerow([u] + ["%.9g" % gcols[c][i, j] for c in gcols for j in range(12)])
        # parity against compact_infer main() predictions.csv (source unchanged, all four models)
        for tag, d, b, f in (("r6_flip", a.parity_r6, "r6", True), ("r7_plain", a.parity_r7, "r7", False)):
            if not d:
                continue
            ref = {r["StudyInstanceUID"]: [float(r[l]) for l in LABELS] for r in csv.DictReader(open(Path(d) / "predictions.csv"))}
            common = [u for u in gu if u in ref]
            diff = np.abs(np.stack([deployed(ok[u], b, [0, 1, 2, 3], f) for u in common]) - np.asarray([ref[u] for u in common]))
            res.setdefault("parity", {})[tag] = {"studies": len(common), "max_abs_dp": float(diff.max()),
                                                 "max_abs_dp_per_target": dict(zip(LABELS, diff.max(0).tolist()))}
    json.dump(res, open(out / ("summary_final.json" if a.final else "summary.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in res if k not in ("decision_table",)}, indent=1)[:6000])
    if res.get("decision_table"):
        tab = res["decision_table"]
        print(json.dumps({k: tab[k] for k in ("n", "fold_counts", "macro", "paired_bootstrap", "rule") if k in tab}, indent=1))


if __name__ == "__main__":
    main()
