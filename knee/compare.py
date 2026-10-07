"""Bounded head comparisons on one fixed feature cache; select on weak BCE only.

The existing baseline checkpoint is read-only. Every run requires a fresh output
folder, and private weak predictions stay there. This module never reads gold
labels or gold metrics to select a model. The underlying trainer may report them.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import torch

from knee.experiment import environment, train
from knee.features import StudyHead
from knee.validation import file_sha256


CANDIDATES = (("attention_seed43", "attention", 43),
              ("attention_seed44", "attention", 44),
              ("meanmax_seed42", "meanmax", 42))


def weighted_bce(truth, probabilities):
    """Same target-macro confidence weighting as training; 0.5 has zero weight."""
    truth = np.asarray(truth, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 2 or truth.shape != probabilities.shape or not len(truth):
        raise ValueError("Weak-label/prediction shape mismatch")
    if (not np.isfinite(truth).all() or not np.isfinite(probabilities).all() or
            (truth < 0).any() or (truth > 1).any() or
            (probabilities < 0).any() or (probabilities > 1).any()):
        raise ValueError("Weak labels and probabilities must be finite and in [0,1]")
    weights = np.abs(truth - .5) * 2
    support = weights.sum(0)
    valid = support > 0
    if not valid.any():
        raise ValueError("Weak holdout has no supervised targets")
    # Float64 avoids float32 sigmoid saturation during comparison. Clip only at
    # machine precision; the same convention is applied to every candidate.
    p = np.clip(probabilities, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
    loss = -truth * np.log(p) - (1 - truth) * np.log1p(-p)
    return float(((loss * weights).sum(0)[valid] / support[valid]).mean())


def select_candidates(truth, predictions, *, ensembles=False, tolerance=1e-5):
    """Compare singles, optionally all pairs and all models, using weak BCE only.

    The mapping's first entry is the protected baseline. Ties within tolerance
    favor fewer models, then earlier entries. No per-target or fitted weights.
    """
    if not predictions or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Need predictions and a nonnegative finite tolerance")
    names = list(predictions)
    members = [(name,) for name in names]
    if ensembles:
        members.extend(combinations(names, 2))
        if len(names) > 2:
            members.append(tuple(names))
    comparisons = []
    for group in members:
        values = np.stack([predictions[name] for name in group]).mean(0)
        comparisons.append({"members": list(group), "weights": [1 / len(group)] * len(group),
                            "weak_bce": weighted_bce(truth, values)})
    selected = comparisons[0]
    for item in comparisons[1:]:
        improves = item["weak_bce"] < selected["weak_bce"] - tolerance
        simpler_tie = (abs(item["weak_bce"] - selected["weak_bce"]) <= tolerance and
                       len(item["members"]) < len(selected["members"]))
        if improves or simpler_tie:
            selected = item
    return comparisons, dict(selected)


def weak_view(cache_path):
    """Load only weak evaluation inputs; never access cached gold labels."""
    with np.load(cache_path, allow_pickle=False) as cache:
        ids, targets, splits = cache["ids"], cache["targets"].tolist(), cache["splits"]
        masks = cache["masks"]
        if len(ids) != len(set(ids.tolist())):
            raise ValueError("Duplicate cache IDs")
        indices = np.flatnonzero((splits == "weak_validation") & masks.any((1, 2)))
        if not len(indices):
            raise ValueError("No usable weak-validation studies")
        groups = cache["groups"]
        partition_groups = {role: set(groups[splits == role]) for role in ("train", "weak_validation", "gold")}
        if any(partition_groups[a] & partition_groups[b] for a, b in combinations(partition_groups, 2)):
            raise ValueError("Report-group leakage between cache partitions")
        return {"ids": ids[indices], "targets": targets, "features": cache["features"][indices],
                "masks": masks[indices], "truth": cache["weak"][indices]}


def predict_weak(checkpoint_path, view, fingerprint, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["feature_sha256"] != fingerprint:
        raise ValueError("Checkpoint was trained on a different feature cache")
    if checkpoint["targets"] != view["targets"]:
        raise ValueError("Checkpoint target order mismatch")
    state, config = checkpoint["state_dict"], checkpoint["config"]
    model = StudyHead(state["mean"], state["std"], state["prior_logits"].sigmoid(),
                      hidden=config["hidden"], kind=config["kind"], dropout=config["dropout"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    blocks = []
    with torch.inference_mode():
        for offset in range(0, len(view["ids"]), 64):
            features = torch.from_numpy(view["features"][offset:offset + 64].astype(np.float32)).to(device)
            masks = torch.from_numpy(view["masks"][offset:offset + 64]).to(device)
            blocks.append(model(features, masks).cpu().double().sigmoid().numpy())
    values = np.concatenate(blocks)
    weighted_bce(view["truth"], values)  # Validate probabilities before persisting.
    return values


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def compare(cache_path, extraction_path, baseline_dir, output_dir, *, device="cuda",
            max_seconds=300, max_epochs=80, patience=12, ensembles=False):
    if device not in ("cpu", "cuda"):
        raise ValueError("Comparison supports explicit cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    if not 0 < max_seconds <= 300 or not 1 <= max_epochs <= 80 or not 1 <= patience <= 80:
        raise ValueError("Budget must be <=300 seconds, <=80 epochs and positive")
    started = time.monotonic()
    cache_path, extraction_path = Path(cache_path).resolve(), Path(extraction_path).resolve()
    baseline_path = Path(baseline_dir).resolve() / "head.pt"
    out = Path(output_dir).resolve()
    fingerprint = file_sha256(cache_path)
    extraction = json.loads(extraction_path.read_text())
    if extraction.get("features_sha256") != fingerprint:
        raise ValueError("Extraction record does not match feature cache")
    view = weak_view(cache_path)
    baseline_sha = file_sha256(baseline_path)
    predictions = {"baseline": predict_weak(baseline_path, view, fingerprint, device)}
    # Fail on reuse before training or touching any previous candidate artifacts.
    out.mkdir(parents=True, exist_ok=False)
    checkpoints = {"baseline": str(baseline_path)}
    statuses = {"baseline": {"status": "preserved", "checkpoint_sha256": baseline_sha}}
    support = (np.abs(view["truth"] - .5) * 2).sum(0)
    report = {
        "status": "running", "feature_sha256": fingerprint, "extraction_sha256": file_sha256(extraction_path),
        "environment": environment(), "device": device,
        "selection_metric": "target-macro confidence-weighted BCE on fixed weak-label holdout only",
        "selection_uses_gold": False, "selection_tolerance": 1e-5,
        "ensembles": "equal probabilities: all pairs and all models" if ensembles else "disabled",
        "budgets": {"seconds_per_new_head": max_seconds, "max_epochs": max_epochs, "patience": patience,
                    "new_heads": len(CANDIDATES), "total_training_seconds_bound": max_seconds * len(CANDIDATES),
                    "note": "Trainer checks budget at epoch boundaries; setup, scoring and finalization are additional."},
        "weak_validation_studies": len(view["ids"]), "targets": view["targets"],
        "undefined_weak_targets": [target for target, value in zip(view["targets"], support) if value == 0],
        "candidates": statuses, "checkpoints": checkpoints,
        "limitation": "Repeated candidate selection on this small weak-label holdout may overfit it; public weak-label gold independence remains unverified.",
    }

    def snapshot(final=False):
        comparisons, selected = select_candidates(view["truth"], predictions, ensembles=ensembles and final)
        report.update(comparisons=comparisons, selected=selected, runtime_seconds=time.monotonic() - started)
        _write(out / "comparison.json", report)

    def save_predictions(name):
        np.savez_compressed(out / f"{name}_weak_predictions.npz", ids=view["ids"], targets=np.asarray(view["targets"]),
                            predictions=predictions[name])

    save_predictions("baseline")
    snapshot()
    for name, kind, seed in CANDIDATES:
        candidate_dir = out / name
        statuses[name] = {"status": "running", "kind": kind, "seed": seed}
        snapshot()
        try:
            train(cache_path, extraction_path, candidate_dir, kind=kind, seed=seed,
                  max_epochs=max_epochs, patience=patience, max_seconds=max_seconds, device=device)
            checkpoint_path = candidate_dir / "head.pt"
            values = predict_weak(checkpoint_path, view, fingerprint, device)
            predictions[name] = values
            checkpoints[name] = str(checkpoint_path)
            statuses[name].update(status="completed", checkpoint_sha256=file_sha256(checkpoint_path))
            save_predictions(name)
        except Exception as exc:
            predictions.pop(name, None)
            checkpoints.pop(name, None)
            statuses[name].update(status="failed", error_type=type(exc).__name__)
            print(json.dumps({"event": "comparison_candidate_failed", "candidate": name,
                              "error_type": type(exc).__name__}), flush=True)
        snapshot()
    if file_sha256(baseline_path) != baseline_sha:
        raise RuntimeError("Protected baseline changed during comparison")
    report["status"] = "completed_with_failures" if any(v["status"] == "failed" for v in statuses.values()) else "completed"
    snapshot(final=True)
    selected = report["selected"]
    averaged = np.stack([predictions[name] for name in selected["members"]]).mean(0)
    np.savez_compressed(out / "selected_weak_predictions.npz", ids=view["ids"], targets=np.asarray(view["targets"]), predictions=averaged)
    print(json.dumps({"event": "comparison_done", "status": report["status"], "selected": selected,
                      "runtime_seconds": report["runtime_seconds"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "extraction", "baseline-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--ensembles", action="store_true")
    args = parser.parse_args()
    compare(args.cache, args.extraction, args.baseline_dir, args.output_dir, device=args.device,
            max_seconds=args.max_seconds, max_epochs=args.max_epochs, patience=args.patience, ensembles=args.ensembles)


if __name__ == "__main__":
    main()
