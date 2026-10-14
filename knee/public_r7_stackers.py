"""Port of heliosli R7 (compact_infer v4c) study-level stacker application.

Source: heliosli "0.957 latest dataset score in one click", embedded compact_infer.py
sha256 8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d
(fluid_stacker_load, fluid_stack, state_layout, state_stacker_load, apply_stacker,
apply_state_stacker). Read as data and re-implemented; the downloaded code is not
imported here. Arithmetic uses the same numpy/math expressions in the same order so
results are bit-identical; the worker cross-checks the fluid path against the
pinned source at runtime.

Deliberate difference: heliosli keeps the model probability when the stacker code
raises. Here any exception propagates (fatal, as for every unknown source error in
our recovery runtime). A non-finite stacker logit still keeps the model probability,
exactly as in the source.
"""
import hashlib
import json
import math

FLUID_SIZE = 384
FLUID_REGION = {"eff_sag_p99_open": 4, "bak_sag_p99_open": 5}
# Adopted classes of the pinned R7 v1 stacker JSONs (manifest bundle 8c7b666e27dc0e8f).
FLUID_ADOPTED = ("Effusion",)
STATE_ADOPTED = ("Baker's",)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def load_fluid(raw, labels, n_groups=1):
    """fluid_stacker_load + the main() checks: adopted classes -> coefficients."""
    st = json.loads(raw.decode("utf-8"))
    feat = st["feature"]
    _require(feat.get("combine") == "logit", "FLUID_COMBINE_LOGIT")
    w = [float(x) for x in feat["weights"]]
    _require(len(w) == n_groups and len(feat["lines"]) == n_groups, "FLUID_MODEL_LINES")
    _require(abs(sum(w) - 1.0) <= 1e-9, "FLUID_LINE_WEIGHTS_SUM")
    cls = {}
    for name, r in st["classes"].items():
        if not r.get("adopt"):
            continue
        _require(name in labels, "FLUID_UNKNOWN_CLASS")
        _require(r["volume_column"] in FLUID_REGION and FLUID_REGION[r["volume_column"]] == int(r["view_id"]), "FLUID_D2_COLUMN")
        co = {k: float(r[k]) for k in ("b0", "b_z", "b_lv", "b_miss")}
        _require(all(math.isfinite(v) for v in co.values()), "FLUID_FINITE_COEFFICIENTS")
        wc = [float(x) for x in (feat.get("weights_per_class") or {}).get(name, w)]
        _require(len(wc) == n_groups and abs(sum(wc) - 1.0) <= 1e-9, "FLUID_CLASS_LINE_WEIGHTS")
        # one model group: the stacker must have been fitted on the deployed line mix [1.0]
        _require(n_groups != 1 or max(abs(x - y) for x, y in zip(wc, [1.0])) <= 1e-12, "FLUID_DEPLOYED_LINE_MIX")
        cls[name] = dict(co, volume_column=r["volume_column"], weights=wc)
    return {"sha256": sha256(raw), "combine": feat["combine"], "weights": w, "classes": cls}


def state_layout(spec, labels):
    """compact_train_v97.state_layout with LABELS passed in."""
    bad = [l for l in labels if len(spec["prior"][l]) != len(spec["categories"][l])]
    _require(not bad, "STATE_SPEC_PRIOR_LENGTH")
    cols, j = [], 0
    for l in labels:
        k_ = len(spec["categories"][l])
        cols.append({"class": l, "categories": list(spec["categories"][l]), "start": j, "stop": j + k_})
        j += k_
    for b, labs in spec["binaries"].items():
        for l in labs:
            cols.append({"binary": b, "class": l, "column": j})
            j += 1
    return {"columns": cols, "width": j}


def load_state(raw, labels, layout, spec_md5):
    """state_stacker_load: adopted classes -> features, state columns and coefficients."""
    st = json.loads(raw.decode("utf-8"))
    _require(st.get("kind") == "state_stacker_v1", "STATE_STACKER_KIND")
    feat = st["feature"]
    _require([float(x) for x in feat.get("weights", [])] == [1.0] and len(feat.get("lines", [])) == 1, "STATE_ONE_MODEL_GROUP")
    _require(st["layout"]["columns"] == layout["columns"] and int(st["layout"]["width"]) == layout["width"], "STATE_LAYOUT")
    _require(st.get("state_spec_md5") == spec_md5, "STATE_SPEC_MD5")
    blocks = {c_["class"]: c_ for c_ in layout["columns"] if "categories" in c_}
    bins = [c_ for c_ in layout["columns"] if "binary" in c_]
    cls = {}
    for name, r in st["classes"].items():
        if not r.get("adopt"):
            continue
        _require(name in labels, "STATE_UNKNOWN_CLASS")
        b_ = blocks[name]
        want = ["z"] + ["state:" + c for c in b_["categories"]] + ["bin:" + c_["binary"] for c_ in bins if c_["class"] == name]
        cols = [None] + list(range(b_["start"], b_["stop"])) + [c_["column"] for c_ in bins if c_["class"] == name]
        if r.get("volume_column"):
            _require(r["volume_column"] in FLUID_REGION, "STATE_VOLUME_COLUMN")
            want += ["lv", "miss"]
            cols += [None, None]
        keep = [i for i, f in enumerate(want) if f not in set(r.get("dropped") or [])]
        want, cols = [want[i] for i in keep], [cols[i] for i in keep]
        _require(list(r["features"]) == want, "STATE_FEATURES")
        coef = [float(x) for x in r["coef"]]
        _require(len(coef) == len(want) and all(math.isfinite(x) for x in coef + [float(r["b0"])]), "STATE_COEFFICIENTS")
        cls[name] = {"features": want, "columns": cols, "coef": coef, "b0": float(r["b0"]), "volume_column": r.get("volume_column")}
    return {"sha256": sha256(raw), "classes": cls, "needs_volume": any(v["volume_column"] for v in cls.values())}


def check_pair(fluid, state, *, size=FLUID_SIZE, views="v4"):
    """main(): a class is replaced by at most one stacker; volume terms need the 384/v4 rows."""
    _require(not set(fluid["classes"]) & set(state["classes"]), "ONE_STACKER_PER_CLASS")
    _require(size == FLUID_SIZE and views == "v4", "FLUID_SIZE_VIEWS")
    return {"fluid": sorted(fluid["classes"]), "state": sorted(state["classes"])}


def _volume(record, column):
    """(miss, lv): missing / non-finite / negative volume -> (True, 0.0), else log1p(mL)."""
    import numpy as np
    v = None if record is None else record.get(column)
    miss = v is None or not math.isfinite(float(v)) or float(v) < 0
    return miss, (0.0 if miss else float(np.log1p(float(v))))


def sigmoid(z):
    """The source's stable two-branch logistic (Python floats)."""
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def fluid_logits(fluid, labels, zmix, record):
    """fluid_stack: {class: stacker logit}; zmix = float64 fold-mean logits [12]."""
    import numpy as np
    z = np.asarray(zmix, np.float64)
    out = {}
    for name, co in fluid["classes"].items():
        miss, lv = _volume(record, co["volume_column"])
        x = np.array([z[labels.index(name)], lv, 1.0 if miss else 0.0], np.float64)
        out[name] = co["b0"] + float(x @ np.array([co["b_z"], co["b_lv"], co["b_miss"]], np.float64))
    return out


def state_logits(state, labels, zf, stv, record):
    """apply_state_stacker inner loop: {class: (stacker logit, feature vector)}."""
    import numpy as np
    out = {}
    for name, co in state["classes"].items():
        j = labels.index(name)
        x = []
        for f, col in zip(co["features"], co["columns"]):
            if f == "z":
                x.append(float(zf[j]))
            elif f in ("lv", "miss"):
                miss, lv = _volume(record, co["volume_column"])
                x.append(lv if f == "lv" else (1.0 if miss else 0.0))
            else:
                x.append(float(stv[col]))
        out[name] = (co["b0"] + float(np.dot(np.asarray(x, np.float64), np.asarray(co["coef"], np.float64))), x)
    return out


def apply(p_model, zmean, state_mean, record, fluid, state, labels):
    """Fluid stacker, then state stacker, on one study's model probabilities.

    p_model: float64 sigmoid of the float32 fold-mean logits; zmean: that fold-mean
    logit as float64; state_mean: float64 mean over fold models of state_readout.
    Returns (probabilities float64 [12], per-class receipt).
    """
    import numpy as np
    p = np.array(p_model, dtype=np.float64, copy=True)
    _require(p.shape == (len(labels),) and np.isfinite(p).all(), "MODEL_PROBABILITIES")
    receipt = {}
    for kind, values in (("fluid", {k: (v, None) for k, v in fluid_logits(fluid, labels, zmean, record).items()}),
                         ("state", state_logits(state, labels, zmean, state_mean, record))):
        for name, (z_out, x) in values.items():
            j = labels.index(name)
            before = float(p[j])
            finite = math.isfinite(z_out)
            if finite:
                p[j] = sigmoid(z_out)
            receipt[name] = {"stacker": kind, "z_feature": float(np.asarray(zmean, np.float64)[j]), "z_out": z_out if finite else None,
                             "p_model": before, "p_out": float(p[j]), "fallback": None if finite else "non-finite stacker logit",
                             **({"features": x} if x is not None else {"volume_missing": _volume(record, fluid["classes"][name]["volume_column"])[0]})}
    return p, receipt


def quantize9(values):
    """compact_infer v3b+ predictions.csv precision: '%.9g' parsed back as float64."""
    import numpy as np
    return np.asarray([[float('%.9g' % float(v)) for v in row] for row in values], dtype=np.float64)
