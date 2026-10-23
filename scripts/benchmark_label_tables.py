"""Benchmark public report-derived label tables against the 58 gold studies.

CPU-only analysis. For every table: harmonize columns to the 12 official targets,
run a gold-leak check first, then score per-target AUC vs the gold-58 labels in
``train.csv`` with a study-level bootstrap, calibration bins, simple ensembles,
and pairwise agreement on non-gold rows (overall and by report script).

The ``Report`` column of ``train.csv`` is read only to classify the dominant
Unicode script of each study (Latin/Greek/Cyrillic/...); the text is never
stored, printed, or written. Only aggregate numbers are written to ``--out-dir``.

Needs pandas and numpy. Parquet tables use pandas+pyarrow when available;
otherwise they are converted once to a sibling ``*.parquet.csv`` by shelling out
to ``--parquet-python`` (a Python that has pyarrow).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import subprocess
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
ID = "StudyInstanceUID"
SEED = 20261008
N_BOOT = 2000
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
# Candidate label tables: name -> (dataset ref, file, loader options).
# ``dir`` is relative to --tables-dir (None = tables-dir itself); ``prefix``/``suffix``
# are stripped from column names before matching the 12 targets.
TABLES = [
    ("steven_v4_blend", "stevenleehans/rsna-knee-llm-report-labels", None, "llm_labels_v4_blend.csv", {}),
    ("local_llm_v2", "local (llm_labels_v2.csv, provenance unrecorded)", None, "llm_labels_v2.csv", {}),
    ("local_llm_full", "local (llm_labels_full.csv, provenance unrecorded)", None, "llm_labels_full.csv", {}),
    ("pilkwang_v2", "pilkwang/rsna-knee-llm-labels", "pilkwang_rsna-knee-llm-labels", "report_labels_v2.csv", {}),
    ("lixin_sol56", "lixin73/rsna-knee-llm-report-labels-sol56", "lixin73_rsna-knee-llm-report-labels-sol56", "labels_llm_gpt56sol.csv", {}),
    ("laymond_qwen3_8b", "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels", "laymond_rsna-knee-abnormality-qwen3-8b-weak-labels", "qwen_knee_weak_labels.csv", {"suffix": "__label"}),
    ("nartaa_labels_v0raw", "nartaa/rsna-knee-hpo-assets", "nartaa_rsna-knee-hpo-assets", "labels_v0raw.parquet", {}),
    ("nartaa_labels_v1_gemlow", "nartaa/rsna-knee-hpo-assets", "nartaa_rsna-knee-hpo-assets", "labels_v1_gemlow.parquet", {}),
    ("nartaa_labels_v1_blend", "nartaa/rsna-knee-hpo-assets", "nartaa_rsna-knee-hpo-assets", "labels_v1_blend.parquet", {}),
    ("nartaa_labels_v1_perlabel", "nartaa/rsna-knee-hpo-assets", "nartaa_rsna-knee-hpo-assets", "labels_v1_perlabel.parquet", {}),
    ("nartaa_train_labels_v0", "nartaa/rsna-knee-hpo-assets", "nartaa_rsna-knee-hpo-assets", "train_labels_v0.parquet", {"prefix": "y_"}),
    ("nartaa_train_labels_v1", "nartaa/rsna-knee-hpo-assets", "nartaa_rsna-knee-hpo-assets", "train_labels_v1.parquet", {"prefix": "y_"}),
    ("jev_hybrid_scores", "riadmohamed42/jev-knee-labels", "riadmohamed42_jev-knee-labels", "HYBRID_labels_scores.csv", {}),
    ("jev_final2_scores", "riadmohamed42/jev-knee-labels", "riadmohamed42_jev-knee-labels", "FINAL2_labels_scores.csv", {}),
    ("rayanbabur_soft", "rayanbabur/rsna-knee-2026-calibrated-soft-targets", "rayanbabur_rsna-knee-2026-calibrated-soft-targets", "train_targets_soft.csv", {}),
    ("dread_llm_soft", "dreaddevelopment/rsna-knee-labels", "dreaddevelopment_rsna-knee-labels", "labels_llm_soft.csv", {}),
]
# Tables whose upstream construction used gold-58 labels (evidence: the JEV README says the
# per-target source choice used the 58-study gold set and the ensemble weights used 67 gold studies).
GOLD_INFORMED = {
    "jev_hybrid_scores": "per-target steven/pilkwang source chosen on gold-58; ensemble weights from 67 gold studies",
    "jev_final2_scores": "ensemble member weights from AUC edge on 67 gold studies",
}
# One table per distinct labeller family (predeclared ensemble) and the independent-labeller agreement core.
FAMILY = ["steven_v4_blend", "pilkwang_v2", "nartaa_train_labels_v1", "lixin_sol56", "laymond_qwen3_8b"]
RECOMMENDED = ["steven_v4_blend", "nartaa_train_labels_v1", "pilkwang_v2"]
CORE_FAMILY = ["steven_v4_blend", "pilkwang_v2", "nartaa_labels_v1_gemlow", "lixin_sol56", "laymond_qwen3_8b", "dread_llm_soft"]
# Gold-evaluation files and a weights file: consistency-checked, not scored as label tables.
AUX_FILES = [
    ("rayanbabur/rsna-knee-2026-calibrated-soft-targets", "rayanbabur_rsna-knee-2026-calibrated-soft-targets", "gold_benchmark_58.csv", "gold_set"),
    ("riadmohamed42/jev-knee-labels", "riadmohamed42_jev-knee-labels", "gold74_eval_set.csv", "gold_set"),
    ("rayanbabur/rsna-knee-2026-calibrated-soft-targets", "rayanbabur_rsna-knee-2026-calibrated-soft-targets", "train_targets_weights.csv", "weights"),
]


def norm_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return {"bakerscyst": "bakers", "baker": "bakers", "bakercyst": "bakers"}.get(key, key)


TARGET_KEYS = {norm_key(t): t for t in TARGETS}


def harmonize(columns, prefix: str = "", suffix: str = "") -> dict[str, str]:
    """Map original column -> official target name (prefix/suffix stripped first)."""
    mapping: dict[str, str] = {}
    for col in columns:
        c = str(col)
        if prefix:
            if not c.startswith(prefix):
                continue
            c = c[len(prefix):]
        if suffix:
            if not c.endswith(suffix):
                continue
            c = c[: -len(suffix)]
        target = TARGET_KEYS.get(norm_key(c))
        if target is not None and target not in mapping.values():
            mapping[col] = target
    return mapping


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: Path, usecols_fn, parquet_python: str):
    """Read CSV/parquet; return (DataFrame, route)."""
    if path.suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
            return df[[c for c in df.columns if usecols_fn(c)]], "parquet"
        except ImportError:
            alt = path.with_suffix(".parquet.csv")
            if not alt.exists():
                code = ("import sys,pyarrow.parquet as pq,pyarrow.csv as pc;"
                        "pc.write_csv(pq.read_table(sys.argv[1]),sys.argv[2])")
                subprocess.run([parquet_python, "-I", "-c", code, str(path), str(alt)], check=True)
            return pd.read_csv(alt, usecols=usecols_fn, low_memory=False), "parquet->csv(pyarrow in " + Path(parquet_python).parent.parent.name + ")"
    return pd.read_csv(path, usecols=usecols_fn, low_memory=False), "csv"


# ---------------------------------------------------------------- statistics
def auc_pairwise(pred: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """Rank AUC with ties counted as half; optional integer/real study weights."""
    ok = ~np.isnan(pred)
    pred, y = pred[ok], y[ok]
    w = np.ones(len(pred)) if w is None else w[ok]
    pos, neg = y == 1, y == 0
    wp, wn = w[pos], w[neg]
    if wp.sum() <= 0 or wn.sum() <= 0:
        return float("nan")
    diff = pred[pos][:, None] - pred[neg][None, :]
    cmp = (diff > 0).astype(float) + 0.5 * (diff == 0)
    return float((wp[:, None] * wn[None, :] * cmp).sum() / (wp.sum() * wn.sum()))


def comparison_tensor(pred: np.ndarray, y: np.ndarray):
    """Per target: (n,n) matrix C[i,j] = credit if i beats j, with class masks, for fast weighted AUC."""
    n, k = pred.shape
    tensors = []
    for t in range(k):
        p = pred[:, t]
        diff = p[:, None] - p[None, :]
        c = (diff > 0).astype(float) + 0.5 * (diff == 0)
        valid = ~np.isnan(p)
        pos = (y[:, t] == 1) & valid
        neg = (y[:, t] == 0) & valid
        c = np.where(np.isnan(diff), 0.0, c)
        tensors.append((c, pos.astype(float), neg.astype(float)))
    return tensors


def boot_macro(tensors, draws: np.ndarray):
    """Macro AUC per draw; draws = (B, n) study weights. Returns (macros, skipped_mask)."""
    B = draws.shape[0]
    aucs = np.full((B, len(tensors)), np.nan)
    for t, (c, pos, neg) in enumerate(tensors):
        wp = draws * pos[None, :]
        wn = draws * neg[None, :]
        num = np.einsum("bi,ij,bj->b", wp, c, wn)
        den = wp.sum(1) * wn.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            aucs[:, t] = np.where(den > 0, num / den, np.nan)
    skipped = np.isnan(aucs).any(axis=1)
    return aucs.mean(axis=1), skipped


def ci_from(macros: np.ndarray, skipped: np.ndarray):
    good = macros[~skipped]
    if len(good) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(good, 2.5)), float(np.percentile(good, 97.5))


def pearson_masked(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def dominant_script(text: str) -> str:
    counts: Counter = Counter()
    for ch in text:
        if ch.isalpha():
            try:
                counts[unicodedata.name(ch).split()[0]] += 1
            except ValueError:
                counts["UNKNOWN"] += 1
    return counts.most_common(1)[0][0] if counts else "NONE"


def calibration(pred: np.ndarray, y: np.ndarray):
    out = {}
    for t, name in enumerate(TARGETS):
        rows = []
        p, g = pred[:, t], y[:, t]
        for i, (lo, hi) in enumerate(BINS):
            m = (p >= lo) & ((p < hi) if i < len(BINS) - 1 else (p <= hi))
            n = int(m.sum())
            rows.append({"bin": f"[{lo:g},{hi:g}{']' if i == len(BINS) - 1 else ')'}", "n": n,
                         "mean_table": float(p[m].mean()) if n else None,
                         "gold_pos_rate": float(g[m].mean()) if n else None})
        out[name] = rows
    return out


def clean(obj):
    """Make nested structures JSON-safe (NaN -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--tables-dir", type=Path, default=ROOT / "data/weak-labels")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/analysis/label-benchmark-20261009")
    ap.add_argument("--parquet-python", default="~/miniconda3/bin/python",
                    help="Python with pyarrow, used only if pandas cannot read parquet")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- gold-58 and report-script classification (text never retained)
    train = pd.read_csv(args.data_dir / "train.csv", low_memory=False)
    all_ids = train[ID].astype(str).tolist()
    id_index = {s: i for i, s in enumerate(all_ids)}
    script = np.array([dominant_script(str(r)) if isinstance(r, str) else "NONE" for r in train["Report"]])
    train = train.drop(columns=["Report"])
    is_gold = train[TARGETS].notna().all(axis=1).to_numpy()
    gold_idx = np.flatnonzero(is_gold)
    nongold_idx = np.flatnonzero(~is_gold)
    gold_y = train.loc[is_gold, TARGETS].to_numpy(dtype=float)
    assert len(gold_idx) == 58 and set(np.unique(gold_y)) <= {0.0, 1.0}
    script_counts = {k: int(v) for k, v in Counter(script).most_common()}
    script_counts_nongold = {k: int(v) for k, v in Counter(script[nongold_idx]).most_common()}
    n_all = len(all_ids)

    # ---- load and harmonize tables onto the train.csv order (NaN where absent)
    mats: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for name, ref, sub, fname, opts in TABLES:
        path = (args.tables_dir / sub / fname) if sub else (args.tables_dir / fname)
        info = {"ref": ref, "file": fname, "path_exists": path.exists()}
        if not path.exists():
            info["error"] = "missing"
            meta[name] = info
            print(f"[skip] {name}: file missing ({path})")
            continue
        if path.suffix == ".csv":
            header = pd.read_csv(path, nrows=0).columns
            keep = set(harmonize(header, opts.get("prefix", ""), opts.get("suffix", ""))) | {ID}
            df, route = read_table(path, lambda c: c in keep, args.parquet_python)
        else:
            df, route = read_table(path, lambda c: True, args.parquet_python)
        mapping = harmonize(df.columns, opts.get("prefix", ""), opts.get("suffix", ""))
        missing_t = [t for t in TARGETS if t not in mapping.values()]
        if missing_t:
            raise SystemExit(f"{name}: missing targets {missing_t}")
        df = df[[ID] + list(mapping)].rename(columns=mapping)
        df[ID] = df[ID].astype(str)
        dups = int(df[ID].duplicated().sum())
        df = df.drop_duplicates(ID, keep="first")
        for t in TARGETS:
            df[t] = pd.to_numeric(df[t], errors="coerce")
        known = df[ID].isin(id_index)
        mat = np.full((n_all, len(TARGETS)), np.nan)
        rows = df.loc[known]
        mat[[id_index[s] for s in rows[ID]]] = rows[TARGETS].to_numpy(dtype=float)
        mats[name] = mat
        info.update({"load_route": route, "sha256": sha256(path), "rows": int(len(df)), "duplicate_ids_dropped": dups,
                     "ids_not_in_train": int((~known).sum()), "nan_cells": int(rows[TARGETS].isna().sum().sum())})
        meta[name] = info

    # ---- auxiliary gold sets / weights: consistency only
    aux = {}
    for ref, sub, fname, kind in AUX_FILES:
        path = args.tables_dir / sub / fname
        if not path.exists():
            aux[fname] = {"ref": ref, "error": "missing"}
            continue
        df = pd.read_csv(path, low_memory=False)
        info = {"ref": ref, "kind": kind, "sha256": sha256(path), "rows": int(len(df))}
        if kind == "gold_set":
            df[ID] = df[ID].astype(str)
            g = df[df[ID].isin(id_index)]
            idx = [id_index[s] for s in g[ID]]
            info["overlap_with_train_ids"] = int(len(g))
            info["overlap_with_gold58"] = int(sum(is_gold[i] for i in idx))
            gm = {t: t for t in TARGETS if t in g.columns}
            sel = [i for i, s in zip(idx, g[ID]) if is_gold[i]]
            if sel:
                gsub = g[[is_gold[i] for i in idx]]
                official = train.loc[sel, TARGETS].to_numpy(dtype=float)
                other = gsub[TARGETS].to_numpy(dtype=float)
                info["frac_cells_equal_to_official_on_gold58"] = float(np.nanmean(other == official))
            info["extra_rows_not_in_gold58"] = int(len(g) - info["overlap_with_gold58"])
        else:
            info["note"] = "sample weights, not labels; not scored"
        aux[fname] = info

    # ---- per-table statistics and leak check
    results: dict[str, dict] = {}
    for name, mat in mats.items():
        gm = mat[gold_idx]
        covered = ~np.isnan(gm).all(axis=1)
        n_cov = int(covered.sum())
        allv = mat[~np.isnan(mat)]
        r = {
            "rows": meta[name]["rows"],
            "gold_rows_present": n_cov,
            "gold_coverage": n_cov / 58,
            "value_min": float(allv.min()), "value_max": float(allv.max()),
            "frac_exact_0.25": float(np.mean(allv == 0.25)), "frac_exact_0.5": float(np.mean(allv == 0.5)),
            "silence_by_target": {t: {"frac_0.25": float(np.nanmean(mat[:, k] == 0.25)),
                                      "frac_0.5": float(np.nanmean(mat[:, k] == 0.5)),
                                      "nan_frac": float(np.isnan(mat[:, k]).mean())}
                                  for k, t in enumerate(TARGETS)},
        }
        ng = mat[nongold_idx]
        ng_vals = ng[~np.isnan(ng)]
        r["nongold_rows_present"] = int((~np.isnan(ng).all(axis=1)).sum())
        r["frac_nongold_cells_near_binary"] = float(np.mean(np.minimum(np.abs(ng_vals), np.abs(ng_vals - 1)) <= 0.01))
        if n_cov < 29:
            r["status"] = "unscorable_no_gold_rows" if n_cov == 0 else "unscorable_low_gold_coverage"
            r["gold_leak_suspected"] = None
            results[name] = r
            continue
        gv = gm[covered]
        gy = gold_y[covered]
        valid = ~np.isnan(gv)
        near = np.abs(gv - gy) <= 0.01
        frac_near = float(near[valid].mean())
        per_auc = [auc_pairwise(gm[:, k], gold_y[:, k]) for k in range(len(TARGETS))]
        macro = float(np.nanmean(per_auc))
        n99 = int(np.sum(np.array(per_auc) >= 0.99))
        gold_dist = float(np.abs(gv[valid] - 0.5).mean())
        ng_dist = float(np.abs(ng_vals - 0.5).mean())
        frac_gold_binary = float(np.mean(np.minimum(np.abs(gv[valid]), np.abs(gv[valid] - 1)) <= 0.01))
        leak = bool(frac_near >= 0.9 or n99 >= 9)
        r.update({
            "frac_gold_cells_within_0.01_of_official": frac_near,
            "n_targets_auc_ge_0.99": n99,
            "gold_leak_suspected": leak,
            "mean_abs_dist_from_0.5_gold_rows": gold_dist,
            "mean_abs_dist_from_0.5_nongold_rows": ng_dist,
            "frac_gold_cells_near_binary": frac_gold_binary,
            "partial_leak_hint": bool(gold_dist - ng_dist > 0.10),
            "per_target_auc": dict(zip(TARGETS, per_auc)),
            "macro_auc": macro,
            "status": "LEAKED" if leak else "ok",
            "calibration": calibration(gm, gold_y),
        })
        results[name] = r

    # ---- lineage: identical values on non-gold or gold rows (keep first in table order)
    order = [n for n, *_ in TABLES if n in mats]

    def same(a, b, idx, min_cells):
        A, B = mats[a][idx], mats[b][idx]
        both = ~(np.isnan(A) | np.isnan(B))
        return bool(both.sum() >= min_cells and np.allclose(A[both], B[both], atol=1e-9))

    scorable = [n for n in order if not results[n]["status"].startswith("unscorable")]
    ok_tables = [n for n in order if results[n]["status"] == "ok"]
    ident_nongold = {a: next((b for b in order[:i] if same(a, b, nongold_idx, 1000)), None) for i, a in enumerate(order)}
    ident_gold = {a: next((b for b in scorable[:scorable.index(a)] if same(a, b, gold_idx, 29 * 12)), None) if a in scorable else None
                  for a in order}
    ens_dup: dict[str, str | None] = {}
    for a in ok_tables:  # duplicates for ensembling: identical on non-gold or gold rows to an earlier scorable table
        ens_dup[a] = next((b for b in ok_tables[:ok_tables.index(a)] if ens_dup[b] is None
                           and (same(a, b, nongold_idx, 1000) or same(a, b, gold_idx, 29 * 12))), None)
    for n in order:
        results[n]["identical_nongold_to"] = ident_nongold[n]
        results[n]["identical_gold_rows_to"] = ident_gold[n]
        results[n]["gold_informed_upstream"] = GOLD_INFORMED.get(n)
        if results[n]["status"].startswith("unscorable"):  # inherit a score when the non-gold rows are identical
            via = next((b for b in ok_tables if same(n, b, nongold_idx, 1000)), None)
            results[n]["lineage_scored_via"] = via

    # ---- bootstrap (shared draws)
    rng = np.random.default_rng(SEED)
    draw_idx = rng.integers(0, 58, size=(N_BOOT, 58))
    draws = np.stack([np.bincount(row, minlength=58) for row in draw_idx]).astype(float)
    ci_tensors = {}
    boot_macros = {}
    for name in scorable:
        r = results[name]
        tens = comparison_tensor(mats[name][gold_idx], gold_y)
        macros, skipped = boot_macro(tens, draws)
        lo, hi = ci_from(macros, skipped)
        r.update({"ci95_lo": lo, "ci95_hi": hi, "boot_skipped": int(skipped.sum()),
                  "boot_mean": float(macros[~skipped].mean()), "boot_se": float(macros[~skipped].std(ddof=1))})
        ci_tensors[name] = tens
        boot_macros[name] = (macros, skipped)
    for n in order:
        via = results[n].get("lineage_scored_via")
        if via:
            results[n]["lineage_macro_auc"] = results[via]["macro_auc"]
            results[n]["lineage_ci95"] = [results[via]["ci95_lo"], results[via]["ci95_hi"]]

    # ---- ensembles of non-leaked, scorable tables (members deduplicated unless stated)
    eligible = [n for n in ok_tables if ens_dup[n] is None]
    ranked = sorted(eligible, key=lambda n: -results[n]["macro_auc"])
    uninformed = [n for n in ranked if n not in GOLD_INFORMED]
    baseline = "steven_v4_blend"
    ensembles = {}

    def ensemble(label, members, note):
        pred = np.nanmean(np.stack([mats[m][gold_idx] for m in members]), axis=0)
        per_auc = [auc_pairwise(pred[:, k], gold_y[:, k]) for k in range(len(TARGETS))]
        macros, skipped = boot_macro(comparison_tensor(pred, gold_y), draws)
        lo, hi = ci_from(macros, skipped)
        bm, bs = boot_macros[baseline]
        ok = ~(skipped | bs)
        d = macros[ok] - bm[ok]
        ensembles[label] = {"members": members, "note": note, "macro_auc": float(np.nanmean(per_auc)),
                            "ci95_lo": lo, "ci95_hi": hi, "boot_skipped": int(skipped.sum()),
                            "boot_se": float(macros[~skipped].std(ddof=1)),
                            "paired_diff_vs_" + baseline: {"mean_diff": float(d.mean()),
                                                           "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]},
                            "per_target_auc": dict(zip(TARGETS, per_auc)), "calibration": calibration(pred, gold_y)}

    ensemble("mean_top3_nonleaked", ranked[:3], "requested; members chosen by gold-58 macro AUC, so optimistic; includes gold-informed tables")
    ensemble("mean_all_nonleaked_distinct", eligible, "requested; deduplicated on identical rows")
    ensemble("mean_all_nonleaked_incl_duplicates", ok_tables, "requested literally; duplicates double-weight their lineage")
    ensemble("mean_top3_gold_uninformed", uninformed[:3], "top 3 by gold-58 AUC after dropping tables whose upstream used gold; still selection-biased")
    ensemble("mean_all_gold_uninformed", uninformed, "no selection on gold; excludes gold-informed tables")
    ensemble("mean_one_per_labeller_family", [m for m in FAMILY if m in ok_tables],
             "predeclared (not gold-selected): steven v4, pilkwang v2, nartaa Gemini re-read, lixin GPT, laymond Qwen")
    ensemble("mean_steven_gemlow_pilkwang", [m for m in RECOMMENDED if m in ok_tables],
             "recommended triple: three distinct labellers, none gold-informed; chosen by reasoning, then scored once")

    # ---- pairwise agreement on non-gold rows
    scripts_ng = script[nongold_idx]
    subsets = {"all": np.ones(len(nongold_idx), bool)}
    for sc in ("LATIN", "GREEK", "CYRILLIC"):
        subsets[sc.lower()] = scripts_ng == sc
    subsets["other_script"] = ~np.isin(scripts_ng, ["LATIN", "GREEK", "CYRILLIC"])
    ng_mats = {n: mats[n][nongold_idx] for n in order}
    agree_rows = []
    pair_study = {}  # (a, b) -> per-study mean disagreement over targets, for non-identical pairs
    for a, b in itertools.combinations(order, 2):
        A, B = ng_mats[a], ng_mats[b]
        valid = ~(np.isnan(A) | np.isnan(B))
        if valid.sum() == 0:
            continue
        dis = ((A < 0.5) & (B > 0.5)) | ((A > 0.5) & (B < 0.5))
        dis = np.where(valid, dis, np.nan)  # NaN where either value is undefined
        identical = bool(same(a, b, nongold_idx, 1000))
        for sub, mask in subsets.items():
            if mask.sum() == 0:
                continue
            Am, Bm, Dm = A[mask], B[mask], dis[mask]
            for k, t in enumerate(TARGETS):
                agree_rows.append({"table_a": a, "table_b": b, "subset": sub, "target": t,
                                   "n_studies": int((~np.isnan(Dm[:, k])).sum()),
                                   "pearson": pearson_masked(Am[:, k], Bm[:, k]),
                                   "disagree_frac": float(np.nanmean(Dm[:, k])) if (~np.isnan(Dm[:, k])).any() else np.nan,
                                   "identical_tables": identical})
            agree_rows.append({"table_a": a, "table_b": b, "subset": sub, "target": "ALL", "n_studies": int(mask.sum()),
                               "pearson": float(np.nanmean([pearson_masked(Am[:, k], Bm[:, k]) for k in range(len(TARGETS))])),
                               "disagree_frac": float(np.nanmean(Dm)) if (~np.isnan(Dm)).any() else np.nan,
                               "identical_tables": identical})
        if not identical:
            pair_study[(a, b)] = dis  # (studies, targets), NaN where either value is undefined
    agree = pd.DataFrame(agree_rows)
    agree.to_csv(args.out_dir / "agreement.csv", index=False, float_format="%.6f")

    def script_summary(pairs):
        """Mean disagreement/Pearson over the given non-identical pairs, by script subset, plus Greek/Cyrillic vs Latin."""
        pairset = set(pairs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # studies where a table has no value
            per_target = np.nanmean(np.stack([pair_study[p] for p in pairs]), axis=0)  # (studies, targets)
            study_dis = np.nanmean(per_target, axis=1)
        sel = agree[(agree.target == "ALL") & agree.apply(lambda r: (r.table_a, r.table_b) in pairset, axis=1)]
        out = {"n_pairs": len(pairs)}
        brng = np.random.default_rng(SEED + 1)

        def boot_diff(v, lat):
            ok_v, ok_l = v[~np.isnan(v)], lat[~np.isnan(lat)]
            d = [ok_v[brng.integers(0, len(ok_v), len(ok_v))].mean() - ok_l[brng.integers(0, len(ok_l), len(ok_l))].mean()
                 for _ in range(N_BOOT)]
            return float(ok_v.mean() - ok_l.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

        lat = study_dis[subsets["latin"]]
        for sub, mask in subsets.items():
            if mask.sum() == 0:
                continue
            s = sel[sel.subset == sub]
            out[sub] = {"n_studies": int(mask.sum()), "mean_pairwise_disagree_frac": float(s.disagree_frac.mean()),
                        "mean_pairwise_pearson": float(s.pearson.mean()), "per_study_mean_disagree": float(np.nanmean(study_dis[mask]))}
            if sub in ("greek", "cyrillic") and mask.sum() >= 5:
                out[sub]["diff_vs_latin"], out[sub]["diff_vs_latin_ci95"] = boot_diff(study_dis[mask], lat)
                out[sub]["pearson_diff_vs_latin"] = float(out[sub]["mean_pairwise_pearson"] - out["latin"]["mean_pairwise_pearson"])
        out["per_target_disagree"] = {
            t: {sub: float(np.nanmean(per_target[subsets[sub], k])) for sub in ("latin", "greek", "cyrillic")}
            | {f"{sub}_minus_latin": boot_diff(per_target[subsets[sub], k], per_target[subsets["latin"], k])
               for sub in ("greek", "cyrillic")}
            for k, t in enumerate(TARGETS)}
        return out

    core_pairs = [p for p in pair_study if p[0] in CORE_FAMILY and p[1] in CORE_FAMILY]
    agreement_summary = {
        "all_distinct_pairs": script_summary(list(pair_study)),
        "core_family_pairs": script_summary(core_pairs),
        "core_family": CORE_FAMILY,
    }

    # ---- write outputs
    tab_rows = []
    for name in order:
        r, m = results[name], meta[name]
        row = {"table": name, "kind": "table", "ref": m["ref"], "file": m["file"], "rows": r["rows"],
               "gold_rows_present": r["gold_rows_present"], "gold_coverage": r["gold_coverage"],
               "status": r["status"], "gold_leak_suspected": r.get("gold_leak_suspected"),
               "partial_leak_hint": r.get("partial_leak_hint"),
               "gold_informed_upstream": r.get("gold_informed_upstream"),
               "identical_nongold_to": ident_nongold[name], "identical_gold_rows_to": ident_gold[name],
               "ensemble_duplicate_of": ens_dup.get(name),
               "frac_gold_within_0.01_of_official": r.get("frac_gold_cells_within_0.01_of_official"),
               "n_targets_auc_ge_0.99": r.get("n_targets_auc_ge_0.99"),
               "gold_mean_abs_dist_0.5": r.get("mean_abs_dist_from_0.5_gold_rows"),
               "nongold_mean_abs_dist_0.5": r.get("mean_abs_dist_from_0.5_nongold_rows"),
               "value_min": r["value_min"], "value_max": r["value_max"],
               "frac_exact_0.25": r["frac_exact_0.25"], "frac_exact_0.5": r["frac_exact_0.5"],
               "macro_auc": r.get("macro_auc"), "ci95_lo": r.get("ci95_lo"), "ci95_hi": r.get("ci95_hi"),
               "boot_skipped": r.get("boot_skipped"),
               "lineage_scored_via": r.get("lineage_scored_via"), "lineage_macro_auc": r.get("lineage_macro_auc")}
        for t in TARGETS:
            row[f"auc_{t}"] = r.get("per_target_auc", {}).get(t)
        tab_rows.append(row)
    for label, e in ensembles.items():
        row = {"table": f"ENSEMBLE:{label}", "kind": "ensemble", "ref": "+".join(e["members"]), "file": "", "status": "ok",
               "macro_auc": e["macro_auc"], "ci95_lo": e["ci95_lo"], "ci95_hi": e["ci95_hi"], "boot_skipped": e["boot_skipped"]}
        for t in TARGETS:
            row[f"auc_{t}"] = e["per_target_auc"][t]
        tab_rows.append(row)
    pd.DataFrame(tab_rows).to_csv(args.out_dir / "per_table.csv", index=False, float_format="%.6f")

    summary = {
        "script": "scripts/benchmark_label_tables.py", "seed": SEED, "bootstrap_draws": N_BOOT,
        "targets": TARGETS, "gold_studies": 58,
        "gold_prevalence": dict(zip(TARGETS, gold_y.mean(0).round(4).tolist())),
        "script_counts_all_studies": script_counts, "script_counts_nongold": script_counts_nongold,
        "script_counts_gold": {k: int(v) for k, v in Counter(script[gold_idx]).most_common()},
        "leak_rule": "gold_leak_suspected if frac gold cells within 0.01 of official >= 0.9 or >= 9/12 per-target AUC >= 0.99",
        "leak_rule_caveat": "natively binary tables (lixin, laymond) are scored by accuracy on rule (a); accuracy 0.8 is not a leak",
        "table_meta": meta, "aux_files": aux, "tables": results,
        "ensemble_duplicates": {k: v for k, v in ens_dup.items() if v},
        "non_leaked_rank_by_macro_auc": ranked, "ensembles": ensembles,
        "agreement": {**agreement_summary,
                      "note": "non-gold rows only; pairs of tables with identical non-gold rows excluded from means"},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(clean(summary), indent=1))

    # ---- console digest (aggregate numbers only)
    print(f"{'table':28s} {'rows':>5s} {'gold':>4s} {'status':>28s} {'macro':>6s} {'CI95':>13s} skip")
    for name in order:
        r = results[name]
        ci = f"[{r['ci95_lo']:.3f},{r['ci95_hi']:.3f}]" if "ci95_lo" in r else "-"
        mac = f"{r['macro_auc']:.3f}" if "macro_auc" in r else "-"
        extra = f" via {r['lineage_scored_via']} ({r['lineage_macro_auc']:.3f})" if r.get("lineage_scored_via") else ""
        print(f"{name:28s} {r['rows']:5d} {r['gold_rows_present']:4d} {r['status']:>28s} {mac:>6s} {ci:>13s} {r.get('boot_skipped', '-')}{extra}")
    for label, e in ensembles.items():
        dd = e["paired_diff_vs_" + baseline]
        print(f"ENSEMBLE {label}: macro {e['macro_auc']:.3f} [{e['ci95_lo']:.3f},{e['ci95_hi']:.3f}] "
              f"diff_vs_{baseline} {dd['mean_diff']:+.3f} [{dd['ci95'][0]:+.3f},{dd['ci95'][1]:+.3f}] members={e['members']}")
    for grp in ("all_distinct_pairs", "core_family_pairs"):
        g = agreement_summary[grp]
        print(f"agreement[{grp}] pairs={g['n_pairs']}", {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in g[k].items()}
                                                         for k in ("latin", "greek", "cyrillic")})
        for t, v in g["per_target_disagree"].items():
            print(f"   {t:17s} latin {v['latin']:.3f} greek {v['greek']:.3f} ({v['greek_minus_latin'][0]:+.3f} [{v['greek_minus_latin'][1][0]:+.3f},{v['greek_minus_latin'][1][1]:+.3f}])"
                  f" cyr {v['cyrillic']:.3f} ({v['cyrillic_minus_latin'][0]:+.3f} [{v['cyrillic_minus_latin'][1][0]:+.3f},{v['cyrillic_minus_latin'][1][1]:+.3f}])")
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
