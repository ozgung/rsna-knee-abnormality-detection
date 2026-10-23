"""Lane D2: parity of the A100 teacher (p2_teacher.py) against the Lane D Kaggle T4 teacher rows on the same exams.
Gate (fixed before the comparison): every arm genuine on every sampled exam, max |delta p| <= 0.01 per arm (Lane C's
measured A100-vs-T4 tolerance for compact_infer), and per-finding Spearman >= 0.99.  Prints aggregates only.
Usage: python p2_parity.py <A100 out dir> <Kaggle teacher dir> <out json>"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fm_targets as T


def a100_rows(d, arm):
    out = {}
    sub = {"old": "old", "crop224": "crop224", "r6": "r6"}[arm]
    for p in sorted(Path(d).glob("part*-%s/%s/predictions.npz" % (arm, sub))):
        z = np.load(p, allow_pickle=False)
        for u, pr, st in zip(z["study_ids"], z["probabilities"], z["statuses"]):
            if str(st) == "genuine":
                out[str(u)] = np.asarray(pr, np.float64)
    return out


def main(a100, kag, outp):
    from scipy.stats import spearmanr
    rec, ok = {}, True
    for arm in T.ARMS:
        a, k = a100_rows(a100, arm), T.arm_rows([kag], arm)
        ids = sorted(set(a) & set(k))
        A, K = np.stack([a[u] for u in ids]), np.stack([k[u] for u in ids])
        d = np.abs(A - K)
        sp = [float(spearmanr(A[:, j], K[:, j])[0]) for j in range(12)]
        r = {"a100_genuine": len(a), "common": len(ids), "max_abs": float(d.max()), "mean_abs": float(d.mean()),
             "spearman_min": min(sp)}
        r["pass"] = r["max_abs"] <= 0.01 and r["spearman_min"] >= 0.99
        rec[arm] = r; ok &= r["pass"]
    rec["status"] = "PASS" if ok else "FAIL"
    json.dump(rec, open(outp, "w"), indent=1); print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
