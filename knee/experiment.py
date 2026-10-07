"""Bounded feature extraction and head training, with explicit provenance."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import time
import unicodedata

import numpy as np
import torch

from knee.features import StudyHead, encode_volume, fit_feature_stats, load_encoder, masked_loss
from knee.validation import evaluate_predictions, file_sha256, make_gold_holdout, read_csv, read_schema


def environment():
    packages = {}
    for name in ("numpy", "torch", "transformers", "pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "code_revision": os.environ.get("KNEE_CODE_REVISION", "unversioned"),
            "source_sha256": os.environ.get("KNEE_SOURCE_SHA256", "unversioned")}


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def report_group(row):
    text = " ".join(unicodedata.normalize("NFKC", row.get("Report", "")).casefold().split())
    return hashlib.sha256((text or row["StudyInstanceUID"]).encode()).hexdigest()


def choose_studies(rows, assignments, train_limit, weak_limit, seed=42):
    groups = defaultdict(list)
    selected, splits = [], []
    for row in rows:
        uid = row["StudyInstanceUID"]
        if assignments[uid] == "validation":
            selected.append(row)
            splits.append("gold")
        elif assignments[uid] == "train":
            groups[report_group(row)].append(row)
    # Hash-order groups before assignment: no label-dependent split decisions.
    order = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    counts = Counter()
    for group in order:
        members = groups[group]
        split = "weak_validation" if counts["weak_validation"] + len(members) <= weak_limit else "train"
        limit = weak_limit if split == "weak_validation" else train_limit
        if counts[split] + len(members) > limit:
            continue
        selected.extend(members)
        splits.extend([split] * len(members))
        counts[split] += len(members)
        if counts["train"] == train_limit and counts["weak_validation"] == weak_limit:
            break
    return selected, np.asarray(splits)


def exclude_related(splits, groups, patients):
    """Transitive report/patient components cannot cross held-out boundaries."""
    result = np.asarray(splits, dtype="U32").copy()
    parent = list(range(len(result)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i in range(len(result)):
        for key in [("report", groups[i]), *(("patient", p) for p in patients[i])]:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    gold_groups = {find(i) for i,s in enumerate(result) if s == "gold"}
    weak_groups = {find(i) for i,s in enumerate(result) if s == "weak_validation"}
    for i,s in enumerate(result):
        if s != "gold" and find(i) in gold_groups:
            result[i] = "excluded"
        elif s == "train" and find(i) in weak_groups:
            result[i] = "excluded"
    return result


def extract(data_dir, encoder_dir, weak_path, output_dir, *, train_limit=2000, weak_limit=300,
            image_size=224, slices_per_plane=8, crop_mm=160, max_seconds=5400):
    from knee.dicom import PREPROCESSING_VERSION, StudyLoadError, StudyVolume, load_study
    if not torch.cuda.is_available():
        raise RuntimeError("Feature extraction explicitly requires CUDA")
    torch.set_num_threads(2)
    started = time.monotonic()
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    id_column, targets = read_schema(data_dir / "sample_submission.csv")
    columns, rows = read_csv(data_dir / "train.csv")
    patient_columns = [c for c in columns if c.lower().replace("_", "").replace(" ", "") in ("patientid", "patientidentifier")]
    if len(patient_columns) > 1:
        raise ValueError("Ambiguous patient identity fields require an explicit audit")
    assignments, split_audit = make_gold_holdout(rows, targets, id_column,
                                               patient_column=patient_columns[0] if patient_columns else None)
    selected, splits = choose_studies(rows, assignments, train_limit, weak_limit)
    _, series_rows = read_csv(data_dir / "train_series.csv")
    by_study = defaultdict(list)
    for row in series_rows:
        by_study[row[id_column]].append(row)
    _, weak_rows = read_csv(weak_path)
    weak = {row[id_column]: [float(row[t]) for t in targets] for row in weak_rows}
    if len(weak) != len(weak_rows) or not all(row[id_column] in weak for row in selected):
        raise ValueError("Weak-label coverage/uniqueness failure")
    uids = np.asarray([row[id_column] for row in selected])
    labels = np.asarray([weak[uid] for uid in uids], np.float32)
    if not np.isfinite(labels).all() or (labels < 0).any() or (labels > 1).any():
        raise ValueError("Invalid weak probabilities")
    gold = np.asarray([[float(row[t]) if row[t].strip() else np.nan for t in targets] for row in selected], np.float32)
    config = {"image_size": image_size, "slices_per_plane": slices_per_plane, "crop_mm": crop_mm,
              "preprocessing_version": PREPROCESSING_VERSION, "encoder": "metaresearch/dinov2/PyTorch/small/1",
              "features": "CLS concatenated with mean patch token; grayscale repeated RGB; ImageNet mean/std",
              "seed": 42, "train_limit": train_limit, "weak_limit": weak_limit, "max_seconds": max_seconds}
    x = np.zeros((len(uids), 3, slices_per_plane, 768), np.float16)
    masks = np.zeros(x.shape[:3], bool)
    patients = []
    totals, errors = Counter(), Counter()
    model = load_encoder(encoder_dir, "cuda")
    def get_volume(i):
        try:
            return load_study(data_dir / "train_series", uids[i], by_study[uids[i]],
                              image_size=image_size, slices_per_plane=slices_per_plane, crop_mm=crop_mm)
        except StudyLoadError as exc:
            return StudyVolume(np.zeros((3, slices_per_plane, image_size, image_size), np.float32),
                               np.zeros((3, slices_per_plane), bool), np.zeros(3, bool), exc.audit)
    print(json.dumps({"event": "extraction_start", "studies": len(uids), "splits": dict(Counter(splits)), "config": config}), flush=True)
    completed = 0
    # Small prefetch queue bounds resident decoded pixels and filesystem pressure.
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {i: pool.submit(get_volume, i) for i in range(min(4, len(uids)))}
        for i in range(len(uids)):
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("Extraction exceeded predeclared wall-clock budget")
            volume = pending.pop(i).result()
            if i + 4 < len(uids):
                pending[i + 4] = pool.submit(get_volume, i + 4)
            x[i] = encode_volume(model, volume, "cuda")
            masks[i] = volume.slice_mask
            patients.append(list(volume.patient_hashes))
            totals.update(volume.audit["counts"])
            errors.update(volume.audit["errors"])
            totals["missing_planes"] += volume.audit["missing_planes"]
            totals["empty_studies"] += int(not volume.plane_mask.any())
            completed = i + 1
            if completed % 25 == 0 or completed == len(uids):
                elapsed = time.monotonic() - started
                print(json.dumps({"event": "extraction_progress", "completed": completed,
                                  "total": len(uids), "seconds": round(elapsed, 1),
                                  "estimated_total_seconds": round(elapsed / completed * len(uids)),
                                  "empty_studies": totals["empty_studies"]}), flush=True)
            if completed == 25 and totals["empty_studies"] > 2:
                raise RuntimeError("Pilot decoding failure rate exceeds 8%; fix pipeline before scale")
    # Patient hashes observed in DICOM participate in exclusion when available.
    splits = exclude_related(splits, [report_group(r) for r in selected], patients)
    removed = int((splits == "excluded").sum())
    if (splits == "train").sum() < 100:
        raise RuntimeError("Too few training studies after identity exclusions")
    cache = output_dir / "features.npz"
    np.savez_compressed(cache, features=x, masks=masks, weak=labels, gold=gold, ids=uids,
                        splits=splits, groups=np.asarray([report_group(r) for r in selected]), targets=np.asarray(targets))
    json_write(output_dir / "patients-private.json", patients)
    report = {"run_id": "dino-frozen-20261008", "status": "completed", "environment": environment(),
              "config": config, "metadata_fingerprints": {f: file_sha256(data_dir / f) for f in ("train.csv", "train_series.csv", "sample_submission.csv")},
              "weak_labels_sha256": file_sha256(weak_path), "features_sha256": file_sha256(cache),
              "split": split_audit, "selected_splits": dict(Counter(splits)), "patient_exclusions": removed,
              "patient_identity_coverage": sum(bool(p) for p in patients), "counts": dict(totals), "errors": dict(errors),
              "runtime_seconds": time.monotonic() - started, "feature_bytes": cache.stat().st_size,
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
              "label_provenance_limitation": "Public v1 labeler prompt development independence from gold is unverified; gold AUC is exploratory."}
    json_write(output_dir / "extraction.json", report)
    print(json.dumps({k:report[k] for k in ("status", "selected_splits", "counts", "errors", "runtime_seconds", "feature_bytes")}), flush=True)
    return report


def train(cache_path, extraction_path, output_dir, *, kind="attention", seed=42, max_epochs=80,
          patience=12, max_seconds=900, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    started = time.monotonic()
    data = np.load(cache_path, allow_pickle=False)
    x, mask, splits = data["features"], data["masks"], data["splits"]
    y = data["weak"]
    train_idx = np.flatnonzero((splits == "train") & mask.any((1, 2)))
    val_idx = np.flatnonzero((splits == "weak_validation") & mask.any((1, 2)))
    gold_idx = np.flatnonzero(splits == "gold")
    if not len(train_idx) or not len(val_idx) or not len(gold_idx):
        raise ValueError("All three evaluation/training partitions required")
    if set(data["groups"][train_idx]) & set(data["groups"][val_idx]):
        raise ValueError("Report-group leakage")
    confidence = np.abs(y - .5) * 2
    support = confidence[train_idx].sum(0)
    if (support == 0).any():
        raise ValueError("A training target has no supervision")
    prior = (y[train_idx] * confidence[train_idx]).sum(0) / support
    mean, std = fit_feature_stats(x, mask, train_idx)
    model = StudyHead(mean, std, prior, kind=kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.02)
    tx = torch.from_numpy(x.astype(np.float32)).to(device)
    tm = torch.from_numpy(mask).to(device)
    ty = torch.from_numpy(y).to(device)
    tw = torch.from_numpy(confidence).to(device)
    best, best_epoch, best_state, history = float("inf"), -1, None, []
    rng = np.random.default_rng(seed)
    stop_reason = "max_epochs"
    for epoch in range(max_epochs):
        if time.monotonic() - started > max_seconds:
            stop_reason = "time_budget"
            break
        model.train()
        losses = []
        for batch in np.array_split(rng.permutation(train_idx), max(1, int(np.ceil(len(train_idx) / 64)))):
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(tx[batch], tm[batch]), ty[batch], tw[batch])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            val_loss = float(masked_loss(model(tx[val_idx], tm[val_idx]), ty[val_idx], tw[val_idx]))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "weak_validation_loss": val_loss})
        if val_loss < best - 1e-5:
            best, best_epoch = val_loss, epoch
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(json.dumps({"event": "training", "kind":kind, "seed":seed, **history[-1]}), flush=True)
        if epoch - best_epoch >= patience:
            stop_reason = "early_stopping"
            break
    if best_state is None:
        raise RuntimeError("No trained checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predictions = model(tx[gold_idx], tm[gold_idx]).sigmoid().cpu().numpy()
    ids, targets = data["ids"], data["targets"].tolist()
    gold = {ids[i]:[None if np.isnan(v) else float(v) for v in data["gold"][i]] for i in gold_idx}
    scored = {ids[i]:predictions[j].tolist() for j,i in enumerate(gold_idx)}
    metrics = evaluate_predictions(gold, scored, targets)
    baseline = evaluate_predictions(gold, {uid:prior.tolist() for uid in gold}, targets)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    extraction = json.loads(Path(extraction_path).read_text())
    config = {"kind":kind, "hidden":128, "dropout":.2, "seed":seed, "lr":.001, "weight_decay":.02,
              "max_epochs":max_epochs, "patience":patience, "max_seconds":max_seconds, "batch_size":64}
    # Tensors and primitive types only: compatible with torch weights_only=True.
    torch.save({"state_dict":best_state, "targets":targets, "config":config,
                "preprocessing":extraction["config"], "feature_sha256":file_sha256(cache_path)}, out / "head.pt")
    report = {"status":"completed", "config":config, "environment":environment(),
              "data_sha256":file_sha256(cache_path), "extraction":extraction,
              "selection_metric":"confidence-weighted BCE on weak-label holdout; gold never selects epochs",
              "best_epoch":best_epoch+1, "weak_validation_loss":best, "stop_reason":stop_reason,
              "runtime_seconds":time.monotonic()-started, "history":history, "gold_metrics":metrics,
              "prevalence_baseline":baseline, "training_priors":dict(zip(targets,map(float,prior))),
              "training_count":len(train_idx), "weak_validation_count":len(val_idx),
              "gold_empty_studies":int((~mask[gold_idx].any((1,2))).sum())}
    json_write(out / "training.json", report)
    np.savez_compressed(out / "gold_predictions.npz", ids=ids[gold_idx], predictions=predictions)
    print(json.dumps({"event":"training_done", "kind":kind, "seed":seed,
                      "best_epoch":best_epoch+1, "gold_macro_auc":metrics["macro_all_targets_auc"],
                      "weak_validation_loss":best, "seconds":report["runtime_seconds"]}), flush=True)
    return report
