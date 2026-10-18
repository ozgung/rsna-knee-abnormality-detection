#!/usr/bin/env python3
"""Score a blinded LLM relabeling pilot against gold-58, public tables and itself.

Inputs (``--pilot-dir``): ``pilot_key.json`` from ``relabel_pilot_prepare.py`` and labeler files
``labels_<labeler>_batch_XX.json``, each a JSON object hid -> {finding: {"p": float, "state": str}}.

Reports (aggregates only; the receipt and printout contain no hid, UID or report text):

(a) gold macro and per-finding AUC with positive counts for each labeler, the labeler mean,
    and the comparison tables (steven_v4, nartaa labels_v1_blend) on the same gold rows;
(b) directional agreement (p >= 0.5) on non-gold rows where steven, gemlow (if present) and
    pilkwang all agree;
(c) inter-rater agreement for every labeler pair: state agreement and |p_a - p_b| <= 0.2;
(d) (c) and (b) broken down by detected language;
(e) go/no-go checks (thresholds in ``DEFAULT_GATES``). Every check must pass, and every labeler must
    cover every pilot hid with valid entries, for a GO; a check that cannot be computed makes
    the decision INCOMPLETE unless another check already failed.

Macro policy: unweighted mean over findings whose AUC is defined (both classes present among
scored rows); undefined findings are listed. Labeler mean = per-cell mean over labelers that
returned a valid value. Gates compare the labeler mean with steven on the same rows.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
STATES = {"pos", "neg", "not_mentioned", "uncertain"}
DEFAULT_PILOT = ROOT / "outputs" / "labels" / "relabel-pilot-20261013"
WL = ROOT / "data" / "weak-labels"
DEFAULT_TABLES = {
    "steven_v4": WL / "llm_labels_v4_blend.csv",
    "nartaa_v1_blend": WL / "nartaa_rsna-knee-hpo-assets" / "labels_v1_blend.parquet.csv",
    "gemlow": WL / "nartaa_rsna-knee-hpo-assets" / "labels_v1_gemlow.parquet.csv",
    "pilkwang": WL / "pilkwang_rsna-knee-llm-labels" / "report_labels_v2.csv",
}
GOLD_COMPARE = ["steven_v4", "nartaa_v1_blend"]
CONSENSUS = ["steven_v4", "gemlow", "pilkwang"]
DEFAULT_GATES = {
    "gold_macro_margin": 0.01,        # mean macro >= steven macro - 0.01
    "per_finding_min_pos": 12,        # findings with >= 12 gold positives ...
    "per_finding_margin": 0.05,       # ... must not trail steven by more than 0.05
    "directional_min": 0.90,
    "inter_rater_min": 0.95,          # applies to state agreement and to p-within-0.2
    "p_tolerance": 0.2,
    "language_gap_max": 0.03,         # minority language vs English, both inter-rater metrics
    "language_min_reports": 20,       # languages with fewer pilot reports are reported, not gated
}
FILE_RE = re.compile(r"^labels_(?P<labeler>.+)_batch_(?P<batch>\d+)\.json$")


def _norm(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return {"bakerscyst": "bakers", "baker": "bakers", "bakercyst": "bakers"}.get(key, key)


TARGET_KEYS = {_norm(t): t for t in TARGETS}


# ---------------------------------------------------------------- statistics
def auc(y, p):
    """Rank AUC with ties counted as half; NaN predictions dropped; None if undefined."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~(np.isnan(p) | np.isnan(y))
    y, p = y[ok], p[ok]
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def score_auc(y: np.ndarray, pred: np.ndarray) -> dict:
    per, pos_n, n_rows = {}, {}, {}
    for k, t in enumerate(TARGETS):
        ok = ~np.isnan(pred[:, k])
        per[t] = auc(y[ok, k], pred[ok, k])
        pos_n[t] = int(np.nansum(y[ok, k]))
        n_rows[t] = int(ok.sum())
    defined = [v for v in per.values() if v is not None]
    return {"macro": float(np.mean(defined)) if defined else None, "per_finding": per,
            "positives": pos_n, "rows": n_rows, "undefined": [t for t, v in per.items() if v is None]}


def _frac(mask_ok: np.ndarray, hits: np.ndarray):
    n = int(mask_ok.sum())
    return (float(hits[mask_ok].mean()) if n else None), n


# ---------------------------------------------------------------- loading
def load_key(pilot_dir: Path) -> dict:
    key = json.loads((pilot_dir / "pilot_key.json").read_text(encoding="utf-8"))
    return key["entries"]


def load_labelers(pilot_dir: Path, entries: dict) -> tuple[dict, dict]:
    """Return (labels[labeler][hid] = (p[12], state[12]), integrity counts per labeler)."""
    labels: dict[str, dict] = {}
    integ: dict[str, dict] = {}
    for path in sorted(pilot_dir.glob("labels_*_batch_*.json")):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        name, batch = m.group("labeler"), int(m.group("batch"))
        lab = labels.setdefault(name, {})
        c = integ.setdefault(name, {"files": 0, "unknown_hids": 0, "duplicate_hids": 0, "batch_mismatch": 0,
                                    "invalid_cells": 0, "missing_cells": 0, "unparseable_files": 0})
        c["files"] += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError
        except ValueError:
            c["unparseable_files"] += 1
            continue
        for hid, findings in data.items():
            if hid not in entries:
                c["unknown_hids"] += 1
                continue
            if hid in lab:
                c["duplicate_hids"] += 1
            if entries[hid]["batch"] != batch:
                c["batch_mismatch"] += 1
            p = np.full(len(TARGETS), np.nan)
            s: list = [None] * len(TARGETS)
            got = {}
            if isinstance(findings, dict):
                for fname, val in findings.items():
                    t = TARGET_KEYS.get(_norm(fname))
                    if t is not None:
                        got[t] = val
            for k, t in enumerate(TARGETS):
                val = got.get(t)
                if val is None:
                    c["missing_cells"] += 1
                    continue
                pv = val.get("p") if isinstance(val, dict) else None
                sv = val.get("state") if isinstance(val, dict) else None
                if (isinstance(pv, (int, float)) and not isinstance(pv, bool) and math.isfinite(pv)
                        and 0.0 <= pv <= 1.0 and sv in STATES):
                    p[k], s[k] = float(pv), sv
                else:
                    c["invalid_cells"] += 1
            lab[hid] = (p, s)
    for name, c in integ.items():
        c["hids_labeled"] = len(labels[name])
        c["hids_expected"] = len(entries)
        full = [h for h in entries if h in labels[name] and not np.isnan(labels[name][h][0]).any()]
        c["hids_complete_valid"] = len(full)
        c["complete"] = len(full) == len(entries) and c["unknown_hids"] == 0 and c["unparseable_files"] == 0
    return labels, integ


def load_table(path: Path | None):
    """Return {uid: p[12]} using the exact target column names, or None if unavailable."""
    if path is None or not Path(path).exists():
        return None
    import pandas as pd

    head = pd.read_csv(path, nrows=0).columns
    cols = {}
    for c in head:
        t = TARGET_KEYS.get(_norm(c))
        if t is not None and t not in cols.values():
            cols[c] = t
    df = pd.read_csv(path, usecols=[ID] + list(cols), dtype={ID: str}).rename(columns=cols)
    if any(t not in df.columns for t in TARGETS):
        raise ValueError(f"{path} lacks some targets")
    mat = df[TARGETS].to_numpy(dtype=float)
    return dict(zip(df[ID], mat))


def read_gold_truth(train_csv: Path, uids: list[str]) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(train_csv, usecols=[ID] + TARGETS, dtype={ID: str}).set_index(ID)
    sub = df.loc[uids, TARGETS]
    if sub.isna().any().any():
        raise ValueError("a pilot gold row has missing targets")
    return sub.to_numpy(dtype=float)


def matrix(hids: list[str], source: dict, keyfn=lambda h: h) -> np.ndarray:
    out = np.full((len(hids), len(TARGETS)), np.nan)
    for i, h in enumerate(hids):
        v = source.get(keyfn(h))
        if v is not None:
            out[i] = v[0] if isinstance(v, tuple) else v
    return out


def states(hids: list[str], lab: dict) -> np.ndarray:
    out = np.full((len(hids), len(TARGETS)), None, dtype=object)
    for i, h in enumerate(hids):
        if h in lab:
            out[i] = lab[h][1]
    return out


def mean_of(mats: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(mats)
    cnt = (~np.isnan(stack)).sum(0)
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, np.nansum(stack, 0) / np.maximum(cnt, 1), np.nan)


# ---------------------------------------------------------------- main scoring
def pair_agreement(pa, pb, sa, sb, tol) -> dict:
    ok = ~(np.isnan(pa) | np.isnan(pb))
    st, n = _frac(ok, sa == sb)
    pw, _ = _frac(ok, np.abs(np.nan_to_num(pa) - np.nan_to_num(pb)) <= tol + 1e-12)
    per = {}
    for k, t in enumerate(TARGETS):
        s_k, n_k = _frac(ok[:, k], (sa == sb)[:, k])
        p_k, _ = _frac(ok[:, k], (np.abs(np.nan_to_num(pa[:, k]) - np.nan_to_num(pb[:, k])) <= tol + 1e-12))
        per[t] = {"state": s_k, "p_within": p_k, "cells": n_k}
    return {"state": st, "p_within": pw, "cells": n, "per_finding": per}


def directional(pred: np.ndarray, cons: np.ndarray, cons_ok: np.ndarray) -> dict:
    ok = cons_ok & ~np.isnan(pred)
    hit = (np.nan_to_num(pred) >= 0.5) == cons
    agg, n = _frac(ok, hit)
    per = {t: dict(zip(("agreement", "cells"), _frac(ok[:, k], hit[:, k]))) for k, t in enumerate(TARGETS)}
    return {"agreement": agg, "cells": n, "per_finding": per}


def score(pilot_dir: Path, train_csv: Path, tables: dict, pair: tuple[str, str] | None = None,
          gates: dict | None = None) -> dict:
    GATES = {**DEFAULT_GATES, **(gates or {})}
    pilot_dir = Path(pilot_dir)
    entries = load_key(pilot_dir)
    labels, integ = load_labelers(pilot_dir, entries)
    names = sorted(labels)
    hids = sorted(entries)
    uid = {h: entries[h]["uid"] for h in hids}
    lang = {h: entries[h]["lang"] for h in hids}
    gold_h = [h for h in hids if entries[h]["gold"]]
    non_h = [h for h in hids if not entries[h]["gold"]]
    tabs = {n: load_table(tables.get(n)) for n in set(GOLD_COMPARE + CONSENSUS)}
    tol = GATES["p_tolerance"]
    receipt: dict = {"labelers": names, "n_reports": len(hids), "n_gold": len(gold_h), "n_nongold": len(non_h),
                     "integrity": integ, "tables_available": {n: t is not None for n, t in tabs.items()},
                     "gates_config": GATES}

    # (a) gold
    y = read_gold_truth(train_csv, [uid[h] for h in gold_h]) if gold_h else np.zeros((0, len(TARGETS)))
    lab_gold = {n: matrix(gold_h, labels[n]) for n in names}
    mean_gold = mean_of(list(lab_gold.values())) if names else np.full_like(y, np.nan)
    gold = {n: score_auc(y, m) for n, m in lab_gold.items()}
    gold["labeler_mean"] = score_auc(y, mean_gold)
    tab_gold = {}
    for n in GOLD_COMPARE:
        if tabs[n] is None:
            gold[n] = None
            continue
        tab_gold[n] = matrix(gold_h, tabs[n], keyfn=lambda h: uid[h])
        res = score_auc(y, tab_gold[n])
        res["gold_rows_present"] = int((~np.isnan(tab_gold[n])).any(1).sum())
        gold[n] = res
    receipt["gold_positives"] = dict(zip(TARGETS, y.sum(0).astype(int).tolist()))
    receipt["gold"] = gold

    # steven on exactly the rows the labeler mean scored (gate basis)
    gate_cmp = None
    if "steven_v4" in tab_gold and names:
        st = np.where(np.isnan(mean_gold), np.nan, tab_gold["steven_v4"])
        gate_cmp = {"labeler_mean": gold["labeler_mean"], "steven_v4_same_rows": score_auc(y, st)}

    # (b) directional vs consensus on non-gold
    avail = [n for n in CONSENSUS if tabs[n] is not None]
    receipt["consensus_tables"] = avail
    lab_non = {n: matrix(non_h, labels[n]) for n in names}
    direction = {}
    if len(avail) >= 2 and non_h:
        mats = [matrix(non_h, tabs[n], keyfn=lambda h: uid[h]) for n in avail]
        present = np.all([~np.isnan(m) for m in mats], axis=0)
        dirs = [np.nan_to_num(m) >= 0.5 for m in mats]
        cons_ok = present & np.all([d == dirs[0] for d in dirs], axis=0)
        cons = dirs[0]
        receipt["consensus_cells"] = int(cons_ok.sum())
        receipt["consensus_positive_cells"] = int((cons_ok & cons).sum())
        srcs = dict(lab_non)
        if names:
            srcs["labeler_mean"] = mean_of(list(lab_non.values()))
        for n, m in srcs.items():
            res = directional(m, cons, cons_ok)
            res["by_language"] = {}
            for lg in sorted(set(lang[h] for h in non_h)):
                rows = np.array([lang[h] == lg for h in non_h])
                ok = cons_ok & rows[:, None] & ~np.isnan(m)
                a, c = _frac(ok, (np.nan_to_num(m) >= 0.5) == cons)
                res["by_language"][lg] = {"agreement": a, "cells": c, "reports": int(rows.sum())}
            direction[n] = res
    receipt["directional"] = direction

    # (c) + (d) inter-rater
    pairs = list(combinations(names, 2))
    primary = tuple(sorted(pair)) if pair else (pairs[0] if pairs else None)
    inter = {}
    lang_counts = {lg: sum(1 for h in hids if lang[h] == lg) for lg in sorted(set(lang.values()))}
    for a, b in pairs:
        pa, pb = matrix(hids, labels[a]), matrix(hids, labels[b])
        sa, sb = states(hids, labels[a]), states(hids, labels[b])
        res = pair_agreement(pa, pb, sa, sb, tol)
        res["by_language"] = {}
        for lg in lang_counts:
            rows = np.array([lang[h] == lg for h in hids])
            r = pair_agreement(pa[rows], pb[rows], sa[rows], sb[rows], tol)
            res["by_language"][lg] = {"state": r["state"], "p_within": r["p_within"], "cells": r["cells"],
                                      "reports": lang_counts[lg]}
        inter[f"{a}|{b}"] = res
    receipt["pilot_language_counts"] = lang_counts
    receipt["inter_rater"] = inter
    receipt["primary_pair"] = "|".join(primary) if primary else None

    # (e) gates
    checks = {}

    def chk(name, value, threshold, ok, detail=None):
        checks[name] = {"value": value, "threshold": threshold, "pass": ok, **({"detail": detail} if detail else {})}

    if gate_cmp and gate_cmp["labeler_mean"]["macro"] is not None and gate_cmp["steven_v4_same_rows"]["macro"] is not None:
        mm, sm = gate_cmp["labeler_mean"]["macro"], gate_cmp["steven_v4_same_rows"]["macro"]
        chk("gold_macro_vs_steven", mm, sm - GATES["gold_macro_margin"], mm >= sm - GATES["gold_macro_margin"])
        worst, fails = None, []
        for t in TARGETS:
            if receipt["gold_positives"][t] < GATES["per_finding_min_pos"]:
                continue
            a_, s_ = gate_cmp["labeler_mean"]["per_finding"][t], gate_cmp["steven_v4_same_rows"]["per_finding"][t]
            if a_ is None or s_ is None:
                fails.append(t)
                continue
            d = a_ - s_
            worst = d if worst is None else min(worst, d)
            if d < -GATES["per_finding_margin"]:
                fails.append(t)
        eligible = [t for t in TARGETS if receipt["gold_positives"][t] >= GATES["per_finding_min_pos"]]
        # vacuously true when no finding reaches the positive-count floor (recorded in detail)
        chk("per_finding_vs_steven", worst, -GATES["per_finding_margin"], not fails,
            {"eligible_findings": eligible, "failing_findings": fails})
    else:
        chk("gold_macro_vs_steven", None, None, None)
        chk("per_finding_vs_steven", None, None, None)
    dm = direction.get("labeler_mean", {}).get("agreement")
    chk("directional_agreement", dm, GATES["directional_min"], None if dm is None else dm >= GATES["directional_min"])
    pr = inter.get(receipt["primary_pair"]) if primary else None
    for metric in ("state", "p_within"):
        v = pr[metric] if pr else None
        chk(f"inter_rater_{metric}", v, GATES["inter_rater_min"], None if v is None else v >= GATES["inter_rater_min"])
    if pr and pr["by_language"].get("en", {}).get("cells"):
        en = pr["by_language"]["en"]
        gaps, fails, ok_all = {}, [], True
        for lg, r in pr["by_language"].items():
            if lg == "en" or r["reports"] < GATES["language_min_reports"] or not r["cells"]:
                continue
            g = min(r["state"] - en["state"], r["p_within"] - en["p_within"])
            gaps[lg] = g
            if g < -GATES["language_gap_max"]:
                fails.append(lg)
                ok_all = False
        chk("minority_language_gap", min(gaps.values()) if gaps else None, -GATES["language_gap_max"],
            ok_all if gaps else None, {"gaps_vs_en": gaps, "failing_languages": fails})
    else:
        chk("minority_language_gap", None, -GATES["language_gap_max"], None)
    coverage_ok = bool(names) and all(integ[n]["complete"] for n in names)
    results = [c["pass"] for c in checks.values()]
    decision = ("NO-GO" if any(r is False for r in results)
                else "INCOMPLETE" if any(r is None for r in results) or not coverage_ok else "GO")
    receipt["gate_basis"] = gate_cmp
    receipt["checks"] = checks
    receipt["coverage_complete"] = coverage_ok
    receipt["decision"] = decision
    return receipt


def _f(v):
    return "n/a" if v is None else f"{v:.3f}"


def summary_lines(r: dict) -> list[str]:
    out = [f"reports {r['n_reports']} (gold {r['n_gold']}, non-gold {r['n_nongold']}); labelers: {', '.join(r['labelers']) or 'none'}"]
    for n, c in r["integrity"].items():
        out.append(f"  {n}: files {c['files']}, complete-valid {c['hids_complete_valid']}/{c['hids_expected']}, "
                   f"invalid {c['invalid_cells']}, missing {c['missing_cells']}, unknown {c['unknown_hids']}")
    out.append("(a) gold macro AUC (per finding: AUC/pos)")
    for n, g in r["gold"].items():
        if g is None:
            out.append(f"  {n}: unavailable")
            continue
        per = " ".join(f"{t}={_f(g['per_finding'][t])}/{g['positives'][t]}" for t in TARGETS)
        extra = f" gold rows present {g['gold_rows_present']}" if "gold_rows_present" in g else ""
        out.append(f"  {n}: macro {_f(g['macro'])}{extra} | {per}")
    out.append(f"(b) directional agreement on {r.get('consensus_cells', 0)} consensus cells ({', '.join(r['consensus_tables'])})")
    for n, d in r["directional"].items():
        langs = " ".join(f"{lg}={_f(v['agreement'])}" for lg, v in d["by_language"].items())
        out.append(f"  {n}: {_f(d['agreement'])} | {langs}")
    out.append("(c/d) inter-rater state / p-within-0.2")
    for n, d in r["inter_rater"].items():
        langs = " ".join(f"{lg}={_f(v['state'])}/{_f(v['p_within'])}" for lg, v in d["by_language"].items())
        out.append(f"  {n}: {_f(d['state'])}/{_f(d['p_within'])} over {d['cells']} cells | {langs}")
    out.append("(e) checks")
    for n, c in r["checks"].items():
        out.append(f"  {n}: value {_f(c['value'])} threshold {_f(c['threshold'])} pass {c['pass']}")
    out.append(f"coverage complete: {r['coverage_complete']}; DECISION: {r['decision']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--train-csv", type=Path, default=ROOT / "data" / "raw" / "train.csv")
    for n, p in DEFAULT_TABLES.items():
        ap.add_argument(f"--{n.replace('_', '-')}", type=Path, default=p, dest=n)
    ap.add_argument("--pair", default=None, help="primary labeler pair 'a,b' for the inter-rater gate")
    ap.add_argument("--out", type=Path, default=None, help="receipt path (default <pilot-dir>/score_receipt.json)")
    args = ap.parse_args(argv)
    tables = {n: getattr(args, n) for n in DEFAULT_TABLES}
    r = score(args.pilot_dir, args.train_csv, tables, tuple(args.pair.split(",")) if args.pair else None)
    out = args.out or args.pilot_dir / "score_receipt.json"
    out.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    print("\n".join(summary_lines(r)))
    print(f"receipt: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
