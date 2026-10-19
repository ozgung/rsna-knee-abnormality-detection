"""Header -> (Anatomical_Plane, Fluid_Sensitive/Fat_Suppression) classifier for external knee DICOM series.

Trained and validated on the competition train series headers (labels from train_series.csv via the site-prior
header table). In the competition labels Fluid_Sensitive == Fat_Suppression on every train series, so one binary
head ("fs") predicts both. Features use only header fields that exist in any MR DICOM (no IDs, no pixels).

  fit:      python series_classifier.py fit <series_headers.csv> <out_model.pkl> <out_report.json>
  predict:  imported by the VM extractor (predict(df) -> plane, fs, probabilities)
"""
import json, re, sys
import numpy as np

NUM = ["EchoTime", "RepetitionTime", "InversionTime", "FlipAngle", "EchoTrainLength", "PixelBandwidth", "SliceThickness",
       "SpacingBetweenSlices", "Rows", "Columns", "n_files", "PercentPhaseFieldOfView", "MagneticFieldStrength",
       "NumberOfAverages", "ImagingFrequency"]
TXT = ["SeriesDescription", "ProtocolName", "ScanOptions", "SequenceVariant", "ScanningSequence", "MRAcquisitionType",
       "InPlanePhaseEncodingDirection", "Manufacturer"]
PLANES = ["Axial", "Coronal", "Sagittal"]


def _tok(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    s = str(v).upper()
    if s.startswith("DUMMY"):
        return ["<DUMMY>"]
    return [t for t in re.split(r"[^A-Z0-9]+", s) if t]


def _num(v):
    try:
        return float(str(v).split("|")[0])
    except (TypeError, ValueError):
        return np.nan


def featurize(df, vocab=None):
    toks = []
    for _, r in df.iterrows():
        t = set()
        for c in TXT:
            for w in _tok(r.get(c)):
                t.add(c[:4] + ":" + w)
                if c in ("SeriesDescription", "ProtocolName"):  # sub-word cues inside fused descriptions
                    for k in ("SAG", "COR", "AX", "TRA", "FS", "FAT", "STIR", "SPAIR", "SPIR", "TIRM", "PD", "T1", "T2", "WE", "DIXON"):
                        if k in w:
                            t.add("sub:" + k)
        toks.append(t)
    if vocab is None:
        from collections import Counter
        cnt = Counter(w for t in toks for w in t)
        vocab = sorted(w for w, n in cnt.items() if n >= 20)
    idx = {w: i for i, w in enumerate(vocab)}
    X = np.full((len(df), len(NUM) + 6 + len(vocab)), np.nan, dtype=np.float32)
    for j, c in enumerate(NUM):
        X[:, j] = [_num(v) for v in df[c]] if c in df else np.nan
    am = [str(v).split("|") if isinstance(v, str) else [] for v in df.get("AcquisitionMatrix", [None] * len(df))]
    for k in range(4):
        X[:, len(NUM) + k] = [_num(a[k]) if len(a) == 4 else np.nan for a in am]
    ps = [str(v).split("|") if isinstance(v, str) else [] for v in df.get("PixelSpacing", [None] * len(df))]
    X[:, len(NUM) + 4] = [_num(a[0]) if a else np.nan for a in ps]
    X[:, len(NUM) + 5] = X[:, NUM.index("Rows")] * X[:, len(NUM) + 4]  # FOV mm
    base = len(NUM) + 6
    X[:, base:] = 0
    for i, t in enumerate(toks):
        for w in t:
            if w in idx:
                X[i, base + idx[w]] = 1
    return X, vocab


def _model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=31, random_state=0)


def fit(csv, out_pkl, out_json):
    import pandas as pd, pickle
    from sklearn.model_selection import GroupKFold
    h = pd.read_csv(csv, low_memory=False)
    h = h[h.split == "train"].reset_index(drop=True)
    assert (h.fluid == h.fatsat).all(), "fluid != fatsat somewhere; one head is not enough"
    yp = h.plane.map({p: i for i, p in enumerate(PLANES)}).values
    yf = h.fluid.astype(int).values
    groups = h.StudyInstanceUID.values
    oof_p, oof_f = np.zeros((len(h), 3)), np.zeros(len(h))
    for tr, va in GroupKFold(n_splits=5).split(h, groups=groups):
        X, vocab = featurize(h.iloc[tr])          # vocabulary fitted on training folds only
        Xv, _ = featurize(h.iloc[va], vocab)
        oof_p[va] = _model().fit(X, yp[tr]).predict_proba(Xv)
        oof_f[va] = _model().fit(X, yf[tr]).predict_proba(Xv)[:, 1]
    pp, pf = oof_p.argmax(1), (oof_f >= 0.5).astype(int)
    dummy = h.SeriesDescription.fillna("").str.upper().str.startswith("DUMMY") | h.SeriesDescription.isna()
    sie = h.Manufacturer.fillna("").str.upper().str.contains("SIEMENS")
    def acc(m):
        m = np.asarray(m)
        return {"n": int(m.sum()), "plane": float((pp == yp)[m].mean()), "fs": float((pf == yf)[m].mean()),
                "both": float(((pp == yp) & (pf == yf))[m].mean())}
    rep = {"series": len(h), "studies": int(h.StudyInstanceUID.nunique()), "cv": "GroupKFold(5) by study, vocab per fold",
           "all": acc(np.ones(len(h), bool)), "dummy_desc": acc(dummy), "siemens": acc(sie), "siemens_named": acc(sie & ~dummy)}
    X, vocab = featurize(h)
    mp, mf = _model().fit(X, yp), _model().fit(X, yf)
    with open(out_pkl, "wb") as f:
        pickle.dump({"vocab": vocab, "plane": mp, "fs": mf, "planes": PLANES}, f)
    json.dump(rep, open(out_json, "w"), indent=1)
    print(json.dumps(rep, indent=1))


def predict(df, bundle):
    X, _ = featurize(df, bundle["vocab"])
    pp = bundle["plane"].predict_proba(X)
    pf = bundle["fs"].predict_proba(X)[:, 1]
    return [bundle["planes"][i] for i in pp.argmax(1)], (pf >= 0.5).astype(int), pp.max(1), pf


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(*sys.argv[2:5])
