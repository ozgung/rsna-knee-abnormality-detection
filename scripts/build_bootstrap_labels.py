"""Track B2: build an OOF soft-bootstrapped training label table and check it on gold-58 first.

CPU-only. Inputs: base soft label table (non-gold studies), the nartaa ``train_labels_v1`` table
(``m_<finding>`` addressed-by-report masks), OOF prediction files ``oof_fold{k}.csv``
(StudyInstanceUID, 12 finding probabilities, ``set`` in {val, gold}) and ``train.csv``.

Only StudyInstanceUID and the 12 label columns of ``train.csv`` are read; the ``Report`` column is
never loaded. Gold-58 studies (all 12 targets filled) are never written to the output tables; gold
labels are used only by the descriptive teacher-side check, which is not used for selection.

Mixing (predeclared in docs/claude-astra-handoff.md section 3, Track B):
    addressed cell (m=1): y1 = (1-alpha_addressed)*y_base + alpha_addressed*p_oof
    silent cell    (m=0): y1 = (1-alpha_silent)*y_base    + alpha_silent*p_oof
    then clip to [clip_lo, clip_hi]; fractions are kept (never rounded).
Variant ``_synovitisprior``: silent Synovitis cells replace y_base by an effusion-conditioned prior
(0.15 if base Effusion < 0.3, 0.45 if 0.3-0.7, 0.60 if > 0.7) before mixing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_label_tables import (  # noqa: E402  (reuse, do not re-derive)
    BINS, ID, N_BOOT, SEED, TARGETS, boot_macro, calibration, ci_from, clean, comparison_tensor,
    harmonize, sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "data/weak-labels/nartaa_rsna-knee-hpo-assets/labels_v1_blend.parquet.csv"
DEFAULT_TRUTH = ROOT / "data/weak-labels/nartaa_rsna-knee-hpo-assets/train_labels_v1.parquet.csv"
SYN = TARGETS.index("Synovitis")
EFF = TARGETS.index("Effusion")
# Predeclared effusion-conditioned synovitis prior (handoff section 3, Track B2).
SYN_PRIOR_LOW, SYN_PRIOR_MID, SYN_PRIOR_HIGH = 0.15, 0.45, 0.60
GRID = (0.3, 0.5, 0.7, 0.9)
DESCRIPTIVE_NOTE = ("DESCRIPTIVE ONLY, NOT FOR SELECTION: gold-58 has SE ~0.02, so the alpha grid and all "
                    "gold numbers here are a sanity/calibration check; alphas stay at the predeclared values.")


# ---------------------------------------------------------------- statistics
def rank_auc(pred: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC (ties = 0.5); NaN predictions dropped; NaN if a class is empty."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    ok = ~(np.isnan(pred) | np.isnan(y))
    pred, y = pred[ok], y[ok]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = pd.Series(pred).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def masked_auc_report(pred: np.ndarray, base: np.ndarray, mask: np.ndarray) -> dict:
    """Per-target AUC of ``pred`` vs ``base >= 0.5`` on addressed (mask == 1) cells, plus macro."""
    per = {}
    for k, t in enumerate(TARGETS):
        sel = mask[:, k] == 1
        y = (base[sel, k] >= 0.5).astype(float)
        per[t] = {"auc": rank_auc(pred[sel, k], y), "n_addressed": int(sel.sum()),
                  "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    aucs = [v["auc"] for v in per.values()]
    undefined = [t for t, v in per.items() if np.isnan(v["auc"])]
    return {"per_target": per, "macro_auc": float(np.nanmean(aucs)) if not all(np.isnan(aucs)) else float("nan"),
            "undefined_targets": undefined, "macro_policy": "nanmean over targets with both classes"}


def make_draws(n: int = 58, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    return np.stack([np.bincount(r, minlength=n) for r in idx]).astype(float)


def gold_report(pred: np.ndarray, gold_y: np.ndarray, draws: np.ndarray) -> dict:
    """Per-target AUC + support, macro AUC and study-bootstrap 95% CI for one prediction matrix."""
    per = {}
    for k, t in enumerate(TARGETS):
        per[t] = {"auc": rank_auc(pred[:, k], gold_y[:, k]), "n_pos": int((gold_y[:, k] == 1).sum()),
                  "n_neg": int((gold_y[:, k] == 0).sum())}
    macros, skipped = boot_macro(comparison_tensor(pred, gold_y), draws)
    lo, hi = ci_from(macros, skipped)
    aucs = [v["auc"] for v in per.values()]
    return {"macro_auc": float(np.nanmean(aucs)), "ci95": [lo, hi], "per_target": per,
            "bootstrap_skipped_draws": int(skipped.sum()), "_macros": macros, "_skipped": skipped}


def paired_delta(a: dict, b: dict) -> dict:
    """Bootstrap CI of macro(a) - macro(b) on the shared draws."""
    skip = a["_skipped"] | b["_skipped"]
    d = (a["_macros"] - b["_macros"])[~skip]
    return {"delta": a["macro_auc"] - b["macro_auc"],
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) >= 10 else [None, None]}


# ---------------------------------------------------------------- mixing
def synovitis_prior(effusion_base: np.ndarray) -> np.ndarray:
    e = np.asarray(effusion_base, float)
    return np.where(e < 0.3, SYN_PRIOR_LOW, np.where(e <= 0.7, SYN_PRIOR_MID, SYN_PRIOR_HIGH))


def mix(y_base: np.ndarray, mask: np.ndarray, p_oof: np.ndarray, alpha_addressed: float, alpha_silent: float,
        clip: tuple[float, float], synovitis_effusion_prior: bool = False) -> np.ndarray:
    """Soft-bootstrap mixing, (n, 12) arrays in TARGETS order. Fractions are kept, never rounded."""
    y_base = np.asarray(y_base, float).copy()
    if synovitis_effusion_prior:
        silent_syn = np.asarray(mask)[:, SYN] == 0
        y_base[silent_syn, SYN] = synovitis_prior(y_base[silent_syn, EFF])
    alpha = np.where(np.asarray(mask) == 1, alpha_addressed, alpha_silent)
    return np.clip((1 - alpha) * y_base + alpha * np.asarray(p_oof, float), clip[0], clip[1])


# ---------------------------------------------------------------- loading
def expand_oof_paths(items) -> list[Path]:
    paths: list[Path] = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            paths.extend(sorted(p.glob("oof_fold*.csv")))
        else:
            paths.append(p)
    if not paths:
        raise SystemExit("no OOF files found")
    return paths


def fold_of(path: Path):
    m = re.search(r"oof_fold(\d+)", path.name)
    return int(m.group(1)) if m else None


def load_oof(paths, require_folds: int | None = None):
    """Return (val_df indexed by ID, gold_df indexed by ID averaged over folds, folds list, n_gold_files)."""
    vals, golds, folds = [], [], []
    for p in paths:
        df = pd.read_csv(p)
        miss = [c for c in [ID, "set"] + TARGETS if c not in df.columns]
        if miss:
            raise ValueError(f"{p}: missing columns {miss}")
        bad = set(df["set"].unique()) - {"val", "gold"}
        if bad:
            raise ValueError(f"{p}: unexpected set values {sorted(bad)}")
        f = fold_of(Path(p))
        if f is not None:
            folds.append(f)
        df[ID] = df[ID].astype(str)
        vals.append(df[df["set"] == "val"][[ID] + TARGETS])
        golds.append(df[df["set"] == "gold"][[ID] + TARGETS])
    if len(set(folds)) != len(folds):
        raise ValueError(f"duplicate fold files: {sorted(folds)}")
    if require_folds is not None and set(folds) != set(range(require_folds)):
        raise ValueError(f"--require-folds {require_folds}: found folds {sorted(folds)}")
    val = pd.concat(vals, ignore_index=True)
    dup = val[ID][val[ID].duplicated()]
    if len(dup):
        raise ValueError(f"{len(dup)} duplicated val studies across OOF files (each non-gold study must appear once)")
    if val[TARGETS].isna().any().any():
        raise ValueError("NaN predictions in OOF val rows")
    gold = pd.concat(golds, ignore_index=True)
    if gold[TARGETS].isna().any().any():
        raise ValueError("NaN predictions in OOF gold rows")
    gold = gold.groupby(ID, sort=False)[TARGETS].mean()
    return val.set_index(ID), gold, sorted(folds)


def load_gold_ids_and_labels(train_csv: Path):
    """Only StudyInstanceUID + the 12 targets are read; the Report column is never loaded."""
    tr = pd.read_csv(train_csv, usecols=[ID] + TARGETS)
    tr[ID] = tr[ID].astype(str)
    g = tr[tr[TARGETS].notna().all(axis=1)].reset_index(drop=True)
    return g[ID].tolist(), g[TARGETS].to_numpy(dtype=float)


def load_wide(path: Path, prefix: str = "") -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    mapping = harmonize(header, prefix)
    missing = [t for t in TARGETS if t not in mapping.values()]
    if missing:
        raise ValueError(f"{path.name}: missing targets {missing} (prefix {prefix!r})")
    df = pd.read_csv(path, usecols=[ID] + list(mapping)).rename(columns=mapping)
    df[ID] = df[ID].astype(str)
    if df[ID].duplicated().any():
        raise ValueError(f"{path.name}: duplicated study IDs")
    return df.set_index(ID)[TARGETS].apply(pd.to_numeric, errors="coerce")


# ---------------------------------------------------------------- table building (no gold labels used)
def build_tables(base: pd.DataFrame, mask: pd.DataFrame, oof_val: pd.DataFrame, gold_ids,
                 alpha_addressed: float, alpha_silent: float, clip: tuple[float, float], allow_partial: bool = False):
    """Return (y1, y1_synovitisprior, info). Gold studies are excluded and a gold ID in OOF val aborts.

    Takes no gold labels, so the output cannot depend on them.
    """
    gold_ids = set(gold_ids)
    leaked = gold_ids & set(oof_val.index)
    if leaked:
        raise ValueError(f"{len(leaked)} gold studies appear among OOF val rows")
    if set(oof_val.index) - set(base.index):
        raise ValueError("OOF val rows contain studies absent from the base table")
    base_ng = base[~base.index.isin(gold_ids)]
    ids = [s for s in base_ng.index if s in oof_val.index]
    if len(ids) < len(base_ng) and not allow_partial:
        raise ValueError(f"OOF covers {len(ids)} of {len(base_ng)} non-gold base studies; use --allow-partial to proceed")
    if not ids:
        raise ValueError("no overlapping studies between OOF and base table")
    if mask.reindex(ids).isna().any().any():
        raise ValueError("addressed mask missing/NaN for some output studies")
    if base_ng.loc[ids].isna().any().any():
        raise ValueError("NaN in base table for output studies")
    yb = base_ng.loc[ids, TARGETS].to_numpy(float)
    m = mask.loc[ids, TARGETS].to_numpy(float)
    p = oof_val.loc[ids, TARGETS].to_numpy(float)
    out = {}
    for name, flag in (("y1", False), ("y1_synovitisprior", True)):
        out[name] = pd.DataFrame(mix(yb, m, p, alpha_addressed, alpha_silent, clip, flag), index=pd.Index(ids, name=ID),
                                 columns=TARGETS)
    assert not (set(out["y1"].index) & gold_ids)
    info = {"rows": len(ids), "base_nongold_rows": int(len(base_ng)), "base_rows_dropped_as_gold": int(len(base) - len(base_ng)),
            "silent_fraction_per_target": {t: float((m[:, k] == 0).mean()) for k, t in enumerate(TARGETS)}}
    return out["y1"], out["y1_synovitisprior"], info


# ---------------------------------------------------------------- gold-side descriptive check
def teacher_side_check(p_gold: np.ndarray, base_gold: np.ndarray, m_gold: np.ndarray, gold_y: np.ndarray,
                       alpha_addressed: float, alpha_silent: float, clip: tuple[float, float]) -> dict:
    """Descriptive gold-58 comparison of base, OOF and mixed values. Not a selection metric."""
    draws = make_draws(len(gold_y))
    base_r = gold_report(base_gold, gold_y, draws)
    oof_r = gold_report(p_gold, gold_y, draws)
    mixed = mix(base_gold, m_gold, p_gold, alpha_addressed, alpha_silent, clip)
    mixed_r = gold_report(mixed, gold_y, draws)
    syn = mix(base_gold, m_gold, p_gold, alpha_addressed, alpha_silent, clip, True)
    syn_r = gold_report(syn, gold_y, draws)
    grid = []
    for a_a in GRID:
        for a_s in GRID:
            r = gold_report(mix(base_gold, m_gold, p_gold, a_a, a_s, clip), gold_y, draws)
            grid.append({"alpha_addressed": a_a, "alpha_silent": a_s, "macro_auc": r["macro_auc"], "ci95": r["ci95"]})
    strip = lambda r: {k: v for k, v in r.items() if not k.startswith("_")}  # noqa: E731
    return {
        "note": DESCRIPTIVE_NOTE,
        "base_on_gold": strip(base_r), "oof_on_gold": strip(oof_r),
        "mixed_on_gold": strip(mixed_r), "mixed_synovitisprior_on_gold": strip(syn_r),
        "mixed_minus_base_paired": paired_delta(mixed_r, base_r),
        "mixed_synovitisprior_minus_base_paired": paired_delta(syn_r, base_r),
        "alpha_grid_descriptive_not_for_selection": grid,
        "calibration_base": calibration(base_gold, gold_y),
        "calibration_mixed": calibration(mixed, gold_y),
        "calibration_bins": [list(b) for b in BINS],
        "gold_mask_addressed_fraction": {t: float(m_gold[:, k].mean()) for k, t in enumerate(TARGETS)},
    }


# ---------------------------------------------------------------- output
def write_parquet(csv_paths: list[Path]) -> str:
    """Parquet via pandas+pyarrow in this interpreter, else via pyarrow alone in another interpreter."""
    try:
        import pyarrow  # noqa: F401
        for p in csv_paths:
            pd.read_csv(p, dtype={ID: str}).to_parquet(p.with_suffix(".parquet"), index=False)
        return "pandas+pyarrow (current interpreter)"
    except ImportError:
        pass
    code = ("import sys,pyarrow as pa,pyarrow.csv as pc,pyarrow.parquet as pq;"
            "t=pc.read_csv(sys.argv[1],convert_options=pc.ConvertOptions(column_types={'StudyInstanceUID':pa.string()}));"
            "pq.write_table(t,sys.argv[2])")
    for cand in (ROOT / ".venv/bin/python", Path("/Users/ozgun/miniconda3/bin/python")):
        if not cand.exists():
            continue
        try:
            for p in csv_paths:
                subprocess.run([str(cand), "-I", "-c", code, str(p), str(p.with_suffix(".parquet"))], check=True,
                               capture_output=True)
            return f"pyarrow.csv->parquet via {cand}"
        except (subprocess.CalledProcessError, OSError):
            continue
    return "csv only (no interpreter with pyarrow found)"


def fmt_ci(r: dict) -> str:
    lo, hi = r["ci95"]
    return f"{r['macro_auc']:.4f} [{lo:.4f}, {hi:.4f}]" if lo is not None and not np.isnan(lo) else f"{r['macro_auc']:.4f} [n/a]"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--oof-dir", nargs="+", required=True, help="directory(ies) with oof_fold*.csv, or the CSV files")
    ap.add_argument("--base-table", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--truth", type=Path, default=DEFAULT_TRUTH, help="train_labels_v1 CSV (m_<finding> masks, y_ for gold rows)")
    ap.add_argument("--train-csv", type=Path, default=ROOT / "data/raw/train.csv")
    ap.add_argument("--alpha-addressed", type=float, default=0.5)
    ap.add_argument("--alpha-silent", type=float, default=0.8)
    ap.add_argument("--clip", type=float, nargs=2, default=(0.02, 0.98), metavar=("LO", "HI"))
    ap.add_argument("--require-folds", type=int, default=None)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    clip = (a.clip[0], a.clip[1])
    t0 = time.time()

    paths = expand_oof_paths(a.oof_dir)
    oof_val, oof_gold, folds = load_oof(paths, a.require_folds)
    gold_ids, gold_y = load_gold_ids_and_labels(a.train_csv)
    if len(gold_ids) != 58:
        raise SystemExit(f"expected 58 gold studies in train.csv, found {len(gold_ids)}")
    if set(oof_gold.index) != set(gold_ids):
        raise SystemExit("OOF gold rows do not match the 58 gold studies of train.csv")
    base = load_wide(a.base_table)
    truth_y, truth_m = load_wide(a.truth, "y_"), load_wide(a.truth, "m_")

    y1, y1_syn, info = build_tables(base, truth_m, oof_val, gold_ids, a.alpha_addressed, a.alpha_silent, clip, a.allow_partial)

    # ---- 2. OOF model quality (val rows vs base thresholded at 0.5 on addressed cells)
    ids = list(y1.index)
    q = masked_auc_report(oof_val.loc[ids, TARGETS].to_numpy(float), base.loc[ids, TARGETS].to_numpy(float),
                          truth_m.loc[ids, TARGETS].to_numpy(float))
    print(f"[OOF quality] folds={folds} val rows={len(ids)}  masked macro AUC vs base>=0.5: {q['macro_auc']:.4f}")
    for t, v in q["per_target"].items():
        print(f"   {t:17s} AUC {v['auc']:.4f}  addressed={v['n_addressed']} pos={v['n_pos']} neg={v['n_neg']}")

    # ---- 4. teacher-side check on gold (gold labels used only here)
    gold_order = gold_ids
    p_gold = oof_gold.loc[gold_order, TARGETS].to_numpy(float)
    if base.index.isin(gold_order).all():
        base_gold, base_gold_src = base.loc[gold_order, TARGETS].to_numpy(float), "base table gold rows"
    else:
        base_gold, base_gold_src = truth_y.loc[gold_order, TARGETS].to_numpy(float), "truth file (train_labels_v1) y_ gold rows"
    m_gold = truth_m.loc[gold_order, TARGETS].to_numpy(float)
    if np.isnan(base_gold).any() or np.isnan(m_gold).any():
        raise SystemExit("NaN in base/mask values for gold rows")
    near = float((np.abs(base_gold - gold_y) <= 0.01).mean())
    gold = teacher_side_check(p_gold, base_gold, m_gold, gold_y, a.alpha_addressed, a.alpha_silent, clip)
    gold["base_gold_source"] = base_gold_src
    gold["base_gold_cells_within_0.01_of_official"] = near
    gold["base_gold_leak_suspected"] = bool(near >= 0.9)
    print(f"[gold-58] base gold source: {base_gold_src}; cells within 0.01 of official: {near:.3f}")
    for key, label in (("base_on_gold", "base table"), ("oof_on_gold", "OOF (fold-avg)"), ("mixed_on_gold", "mixed (chosen alphas)"),
                       ("mixed_synovitisprior_on_gold", "mixed + synovitis prior")):
        print(f"   {label:26s} macro AUC {fmt_ci(gold[key])}")
    d = gold["mixed_minus_base_paired"]
    print(f"   paired mixed-base delta {d['delta']:+.4f} CI {d['ci95']}")
    print("   per-target gold AUC (support pos/neg): " + "; ".join(
        f"{t} {v['auc']:.3f} ({v['n_pos']}/{v['n_neg']})" for t, v in gold["oof_on_gold"]["per_target"].items()))
    print(f"   {DESCRIPTIVE_NOTE}")
    print("   alpha_addr  alpha_silent  macro AUC (descriptive)")
    for g in gold["alpha_grid_descriptive_not_for_selection"]:
        print(f"     {g['alpha_addressed']:.1f}        {g['alpha_silent']:.1f}         {g['macro_auc']:.4f}")
    print("   calibration (5 bins, macro-pooled over targets): bin n base_mean mixed_mean gold_pos_rate")
    for i, (lo, hi) in enumerate(BINS):
        cb = [gold["calibration_base"][t][i] for t in TARGETS]
        cm = [gold["calibration_mixed"][t][i] for t in TARGETS]
        nb, nm = sum(c["n"] for c in cb), sum(c["n"] for c in cm)
        wm = lambda cs, f: (sum(c["n"] * c[f] for c in cs if c["n"]) / max(sum(c["n"] for c in cs), 1))  # noqa: E731
        print(f"     [{lo:g},{hi:g}) base n={nb} mean={wm(cb, 'mean_table'):.3f} pos={wm(cb, 'gold_pos_rate'):.3f} | "
              f"mixed n={nm} mean={wm(cm, 'mean_table'):.3f} pos={wm(cm, 'gold_pos_rate'):.3f}")

    # ---- 5. outputs
    a.out_dir.mkdir(parents=True, exist_ok=True)
    f1, f2 = a.out_dir / "labels_bootstrap_v1.csv", a.out_dir / "labels_bootstrap_v1_synovitisprior.csv"
    y1.reset_index().to_csv(f1, index=False)
    y1_syn.reset_index().to_csv(f2, index=False)
    parquet_route = write_parquet([f1, f2])
    summary = {
        "script": "scripts/build_bootstrap_labels.py", "date_run_unix": int(t0), "runtime_s": round(time.time() - t0, 2),
        "alpha_addressed": a.alpha_addressed, "alpha_silent": a.alpha_silent, "clip": list(clip),
        "synovitis_prior": {"effusion<0.3": SYN_PRIOR_LOW, "0.3<=effusion<=0.7": SYN_PRIOR_MID, "effusion>0.7": SYN_PRIOR_HIGH,
                            "provenance": "predeclared in docs/claude-astra-handoff.md section 3 Track B (B2)"},
        "folds_used": folds, "allow_partial": a.allow_partial, "require_folds": a.require_folds,
        "row_counts": {**info, "oof_val_rows": int(len(oof_val)), "gold_studies": len(gold_ids),
                       "output_rows": int(len(y1)), "complete": bool(len(y1) == info["base_nongold_rows"])},
        "gold_in_output": 0,
        "inputs_sha256": {**{str(p.name): sha256(p) for p in paths}, "base_table": sha256(a.base_table),
                          "truth": sha256(a.truth), "train_csv": sha256(a.train_csv)},
        "oof_quality_masked_vs_base": q, "gold_teacher_side_check": gold,
        "output_value_range": [float(y1.to_numpy().min()), float(y1.to_numpy().max())],
        "parquet": parquet_route,
        "bootstrap": {"seed": SEED, "draws": N_BOOT, "unit": "study"},
    }
    (a.out_dir / "summary.json").write_text(json.dumps(clean(summary), indent=2))
    print(f"[out] {len(y1)} rows -> {a.out_dir}  (parquet: {parquet_route})")


if __name__ == "__main__":
    main()
