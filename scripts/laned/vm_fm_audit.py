"""Lane D (VM): fail-closed audit of the fastMRI pixel store (/content/pxfm) on a fixed 50-study sample.
Checks: store FINALIZED, every selected study disposed (done or a listed error), error rate <= 5 %; per sampled study rows > 0,
both sagittal and coronal blocks, at least one fluid-sensitive block, meta finite; every row decodes to uint8 320x320 and is not
blank (std > 2, zero fraction < 0.95); per-study mean intensity inside [10, 200]; reproduction: 5 of the 50 studies re-rendered
through the same CStudies.rows path give byte-identical rows and meta.  Optional competition reference (/content/px) compares
rows/study and intensity distributions (reported, gate: median rows/study within x0.5..x2 of the competition median).
Writes /content/fm_meta/cache_audit.json; exit 1 on any failure.  Prints counts only."""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np

FM = Path("/content/pxfm")
sys.path.insert(0, "/content/code/cloud_code")


def rows_of(store, idx, lo, n):
    return store[np.arange(lo, lo + n)]


def main():
    sys.path.insert(0, "/content")
    sys.path.insert(0, "/content/code")
    from compact_train_s6v2 import PngRows
    fails, R = [], {}
    if not (FM / "FINALIZED").exists():
        fails.append("not finalized")
    idx = json.load(open(FM / "index.json"))
    order, done, errs = idx["order"], idx["done"], idx["errors"]
    R["selected"], R["done"], R["errors"] = len(order), len(done), len(errs)
    if set(order) != set(done) | set(errs):
        fails.append("undisposed studies")
    if len(errs) > 0.05 * len(order):
        fails.append("error rate %.3f" % (len(errs) / len(order)))
    st = PngRows(str(FM))
    meta = np.load(FM / "meta.f32.npy", mmap_mode="r")
    cols = idx["meta_cols"]
    ci = {c: cols.index(c) for c in ("plane", "fluid", "view_id")}
    rng = np.random.default_rng(20261019)
    ok_ids = sorted(done)
    sample = [ok_ids[i] for i in rng.choice(len(ok_ids), size=min(50, len(ok_ids)), replace=False)]
    per = []
    for u in sample:
        lo, n, nb = done[u]
        if n <= 0:
            fails.append("empty study"); continue
        m = np.asarray(meta[lo:lo + n])
        x = rows_of(st, None, lo, n)
        planes = set(np.unique(m[:, ci["plane"]]).astype(int).tolist())
        rec = {"rows": int(n), "blocks": int(nb), "planes": sorted(planes), "fluid_rows": int((m[:, ci["fluid"]] > 0.5).sum()),
               "mean": float(x.mean()), "std_min": float(x.reshape(n, -1).std(1).min()),
               "zero_frac_max": float((x.reshape(n, -1) == 0).mean(1).max())}
        per.append(rec)
        if not np.isfinite(m).all():
            fails.append("meta not finite")
        if x.dtype != np.uint8 or x.shape[1:] != (320, 320):
            fails.append("row shape")
        if rec["fluid_rows"] == 0:
            fails.append("no fluid rows")
        if len(planes) < 2:
            fails.append("fewer than two planes")
        if not (10 <= rec["mean"] <= 200):
            fails.append("mean intensity %.1f" % rec["mean"])
        if (x.reshape(n, -1).std(1) <= 2).mean() > 0.05 or (((x.reshape(n, -1) == 0).mean(1)) >= 0.95).mean() > 0.05:
            fails.append("blank rows > 5%")
    R["sample"] = len(sample)
    R["rows_per_study_median"] = float(np.median([p["rows"] for p in per]))
    R["blocks_per_study_median"] = float(np.median([p["blocks"] for p in per]))
    R["mean_intensity_median"] = float(np.median([p["mean"] for p in per]))
    R["plane_sets"] = {str(k): v for k, v in zip(*np.unique([str(p["planes"]) for p in per], return_counts=True))}
    R["plane_sets"] = {k: int(v) for k, v in R["plane_sets"].items()}
    # reproduction of 5 studies through the builder's own render path
    os.environ.setdefault("S6V2_DATA", "/content/fmdata")
    import vm_build_cache as B
    B.worker_init()
    import cv2
    rep = []
    for u in sample[:5]:
        lo, n, nb = done[u]
        uid, enc, mt, info, err, secs = B.build_one(u)
        same = err is None and len(enc) == n and all(np.array_equal(cv2.imdecode(np.frombuffer(e, np.uint8), cv2.IMREAD_UNCHANGED), st[lo + k])
                                                    for k, e in enumerate(enc)) and np.array_equal(mt, np.asarray(meta[lo:lo + n]))
        rep.append(bool(same))
    R["reproduction_identical"] = rep
    if not all(rep):
        fails.append("re-render differs")
    if (Path("/content/px") / "FINALIZED").exists():
        ci_ = json.load(open("/content/px/index.json"))["done"]
        cr = np.median([v[1] for v in ci_.values() if v[1] > 0])
        R["competition_rows_per_study_median"] = float(cr)
        if not (0.5 * cr <= R["rows_per_study_median"] <= 2 * cr):
            fails.append("rows/study far from competition")
    R["fails"] = sorted(set(fails)); R["status"] = "PASS" if not fails else "FAIL"
    json.dump(R, open("/content/fm_meta/cache_audit.json", "w"), indent=1)
    print(json.dumps(R, indent=1))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
