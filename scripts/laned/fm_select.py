"""Lane D (local): label fastMRI series with the validated header classifier, pick usable exams, write the series table.
Plane comes from ImageOrientationPatient when present (geometric truth) and is cross-checked against the header classifier;
Fluid_Sensitive = Fat_Suppression = the classifier's fat-sat/fluid head (one head; they are equal on every competition series).
Usage: python fm_select.py <fm_series_headers.csv> <model.pkl> <out dir> [max exams]
Writes <out>/fm_train_series.csv (competition schema) and <out>/fm_select_summary.json (aggregate counts only)."""
import json, pickle, re, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import series_classifier as SC


def iop_plane(v):
    try:
        x = [float(t) for t in str(v).split("|")]
        n = np.cross(np.array(x[:3]), np.array(x[3:6]))
        return ["Sagittal", "Coronal", "Axial"][int(np.argmax(np.abs(n)))], float(np.max(np.abs(n)) / (np.linalg.norm(n) + 1e-9))
    except Exception:
        return None, 0.0


def main(hdr, model, out, cap=0):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    h = pd.read_csv(hdr, dtype=str, keep_default_na=False)
    S = {"series_read": len(h), "studies_read": int(h.StudyInstanceUID.nunique())}
    d = h.SeriesDescription.str.upper()
    it = h.ImageType.str.upper()
    n = pd.to_numeric(h.n_files, errors="coerce").fillna(0)
    nf = pd.to_numeric(h.NumberOfFrames.replace("", "1"), errors="coerce").fillna(1)
    drop = {"not_mr": h.Modality.str.upper().ne("MR"), "localizer": it.str.contains("LOCALIZER") | d.str.contains(r"LOC|SCOUT|SURVEY|CAL\b|3-?PL"),
            "few_files": n < 10, "multiframe": nf > 1, "derived": it.str.startswith("DERIVED")}
    keep = np.ones(len(h), bool)
    for k, m in drop.items():
        S["drop_" + k] = int((m & keep).sum()); keep &= ~m.values
    h = h[keep].reset_index(drop=True)
    # Siemens prescan-normalised duplicates ("<desc>_FIL") of a series that is also present unfiltered: keep the unfiltered one
    # (the competition series have no such duplicates)
    base = h.SeriesDescription.str.upper().str.replace(r"_FIL$", "", regex=True)
    have = set(zip(h.StudyInstanceUID, h.SeriesDescription.str.upper()))
    fil = h.SeriesDescription.str.upper().str.endswith("_FIL") & np.array([(u, b) in have for u, b in zip(h.StudyInstanceUID, base)])
    S["drop_fil_duplicate"] = int(fil.sum())
    h = h[~fil.values].reset_index(drop=True)
    for c in SC.NUM + ["n_files"]:
        if c in h:
            h[c] = pd.to_numeric(h[c], errors="coerce")
    bundle = pickle.load(open(model, "rb"))
    pm, fs, pconf, pfs = SC.predict(h.replace({"": np.nan}), bundle)
    geo = [iop_plane(v) for v in h.ImageOrientationPatient]
    h["plane_model"], h["fs"], h["plane_iop"], h["iop_purity"] = pm, fs, [g[0] for g in geo], [g[1] for g in geo]
    has = h.plane_iop.notna()
    S["iop_present"] = int(has.sum())
    S["plane_model_vs_iop_agreement"] = float((h.plane_model[has] == h.plane_iop[has]).mean()) if has.any() else None
    S["oblique_lt_0.8"] = int((h.iop_purity[has] < 0.8).sum())
    h["plane"] = np.where(has, h.plane_iop, h.plane_model)
    S["fs_rate"] = float(np.mean(fs)); S["fs_prob_uncertain_0.2_0.8"] = int(((pfs > 0.2) & (pfs < 0.8)).sum())
    S["series_kept"] = len(h)
    S["kept_by_plane_fs"] = {"%s|%d" % k: int(v) for k, v in h.groupby(["plane", "fs"]).size().items()}
    g = h.groupby("StudyInstanceUID")
    ok = g.apply(lambda x: (x.plane == "Sagittal").any() and (x.plane == "Coronal").any() and len(x) >= 3 and (x.fs == 1).any())
    studies = sorted(ok[ok].index)
    S["studies_usable"] = len(studies); S["studies_rejected"] = int((~ok).sum())
    studies.sort(key=lambda u: hashlib.sha1(u.encode()).hexdigest())      # deterministic, content-free order
    if cap and len(studies) > cap:
        studies = studies[:cap]
    S["studies_selected"] = len(studies)
    t = h[h.StudyInstanceUID.isin(set(studies))]
    S["series_selected"] = len(t); S["series_per_study"] = float(len(t) / max(1, len(studies)))
    S["manufacturers"] = t.Manufacturer.value_counts().to_dict()
    S["field_strength"] = t.MagneticFieldStrength.round(1).astype(str).value_counts().to_dict()
    o = pd.DataFrame({"StudyInstanceUID": t.StudyInstanceUID, "SeriesInstanceUID": t.SeriesInstanceUID,
                      "Fluid_Sensitive": t.fs.astype(int), "Fat_Suppression": t.fs.astype(int), "Anatomical_Plane": t.plane})
    o.to_csv(out / "fm_train_series.csv", index=False)
    json.dump(S, open(out / "fm_select_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
