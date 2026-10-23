"""Lane F: compact-slot mixing grid from out-of-fold evidence.  For w in the grid, column = w * rank(own) + (1 - w) * rank(R6 flip),
per target, ranks over the scored population (pool: 4,345 studies, single hash-fold model per study for both columns; gold-58: own
4-fold mean logit vs Lane B's 4-model R6 flip).  Paired study bootstrap (2,000 draws, seed 42) of each w against w = 0.5, plus
own vs R6 flip vs R7 plain as in the decision table.  Truth: pool labels_v1_blend >= 0.5 (report-derived; caveat), gold train.csv.

  python3 scripts/lanef/mix_grid.py --oof OOF_DIR --out OOF_DIR/grid.json [--laneb DIR] [--w 0.3,0.5,0.6,0.7,0.8,1.0]
Prints aggregates only."""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/lanef"))
from analyze_oof import LABELS, BLEND, TRAIN, MAIN, ranks, macro_auc, boot_table, sigmoid  # noqa: E402


def read_cols(path, prefix):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["StudyInstanceUID"]] = np.asarray([float(r["%s|%s" % (prefix, l)]) for l in LABELS])
    return out


def grid_table(y, own, r6, r7, groups, ws, label):
    rk_own, rk_r6 = ranks(own), ranks(r6)
    cols = {"own": own, "R6 flip": r6}
    if r7 is not None:
        cols["R7 plain"] = r7
    for w in ws:
        cols["w=%.1f" % w] = w * rk_own + (1 - w) * rk_r6
    pairs = [("w=%.1f" % w, "w=0.5") for w in ws if abs(w - 0.5) > 1e-9] + [("own", "R6 flip")] + ([("own", "R7 plain")] if r7 is not None else [])
    pairs += [("w=0.5", "R6 flip")]
    tab = {"population": label, "n": len(y), "macro": {n: macro_auc(y, p)[0] for n, p in cols.items()},
           "per_target": {n: dict(zip(LABELS, macro_auc(y, p)[1].tolist())) for n, p in cols.items()},
           "paired_vs_w0.5_and_references": boot_table(y, cols, pairs, groups)}
    return tab


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oof", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--laneb", default=str(MAIN / "outputs/laneb-20261019/analysis"))
    ap.add_argument("--w", default="0.3,0.5,0.6,0.7,0.8,1.0")
    ap.add_argument("--name", default="own", help="column prefix of the own OOF columns (analyze_oof.py --name)")
    ap.add_argument("--blend", default=str(BLEND)); ap.add_argument("--train", default=str(TRAIN))
    a = ap.parse_args()
    ws = [float(x) for x in a.w.split(",")]
    oof = Path(a.oof)
    own = read_cols(oof / "oof_columns.csv", a.name)
    lb6 = read_cols(Path(a.laneb) / "oof_columns.csv", "R6 flip")
    lb7 = read_cols(Path(a.laneb) / "oof_columns.csv", "R7 plain")
    blend = {}
    with open(a.blend, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            blend[r["StudyInstanceUID"]] = [float(r[l]) for l in LABELS]
    pool = sorted(u for u in own if u in lb6 and u in blend)
    y = np.asarray([[1.0 if v >= 0.5 else 0.0 for v in blend[u]] for u in pool])
    res = {"grid_w": ws, "formula": "w * rank(own) + (1 - w) * rank(R6 flip), per-target average ranks over the scored population",
           "bootstrap": {"draws": 2000, "seed": 42, "unit": "study"},
           "pool": grid_table(y, np.stack([own[u] for u in pool]), np.stack([lb6[u] for u in pool]), np.stack([lb7[u] for u in pool]), pool, ws,
                              "pool out-of-fold, single hash-fold model per study for own and for R6 flip; truth labels_v1_blend >= 0.5")}
    # gold-58: own = 4-fold mean logit from gold58_columns.csv; R6 flip = Lane B's 4-model column
    gown = {}
    with open(oof / "gold58_columns.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gown[r["StudyInstanceUID"]] = sigmoid([float(r["mean logit|%s" % l]) for l in LABELS])
    g6 = read_cols(Path(a.laneb) / "gold58_columns.csv", "R6 flip")
    g7 = read_cols(Path(a.laneb) / "gold58_columns.csv", "R7 plain")
    gold = {}
    with open(a.train, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if all(r[l] not in ("", None) for l in LABELS):
                gold[r["StudyInstanceUID"]] = [float(r[l]) for l in LABELS]
    gu = sorted(u for u in gown if u in g6 and u in gold)
    yg = np.asarray([gold[u] for u in gu])
    res["gold58"] = grid_table(yg, np.stack([gown[u] for u in gu]), np.stack([g6[u] for u in gu]), np.stack([g7[u] for u in gu]), gu, ws,
                               "gold-58, own = 4-fold mean logit (exposed), R6 flip = Lane B 4-model; n = 58, wide intervals")
    res["caveat"] = ("pool truth is report-derived and labels v3 share its report sources, so own's margin on the pool is partly label-family "
                     "alignment; the deployed compact also averages four fold models while the pool columns are single-model, so the "
                     "pool grid is an upper-side read of own's share; gold-58 is the only image-truth read and is n = 58")
    json.dump(res, open(a.out, "w"), indent=1)
    for pop in ("pool", "gold58"):
        t = res[pop]
        print(pop, "n", t["n"], {k: round(v, 4) for k, v in t["macro"].items()})
        for k, v in t["paired_vs_w0.5_and_references"].items():
            print("  %-28s %+.4f [%+.4f, %+.4f]" % (k, v["delta"], *v["ci95"]))


if __name__ == "__main__":
    main()
