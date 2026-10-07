"""Fail-closed, bounded reproduction checks for an explicitly audited public reader.

The public source is supplied at build time, never vendored or executed wholesale.
This is a rehearsal tool, not training or a competition submission entry point.
"""
from __future__ import annotations

import ast
import base64
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
import threading
import time
import types

SOURCE_SHA256 = "493c7b0d44fe5244b02024448a93f5393cdb073781a068951badb3ffd53fcbde"
FASTREAD_SHA256 = "5a59b6f82e846bcdfd452aaee6ff518cdd9d0b04aab03691b45c3f70a5198cb0"
WEIGHT_SHA256 = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
WEIGHT_NAME = "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"
SOURCE_URL = "https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror"
WEIGHT_DATASET = "nartaa/rsna-knee-publication-swa-weights-20261007"
CONSTANTS = {"INPUT_RES", "INPUT_MODE", "IMG", "CROP_MM", "SLOTS", "MAXS", "K_EVAL",
             "NORM", "LAB", "_MEAN", "_STD", "SLOT_SIZES", "SAGITTAL_SLOTS", "_WLUT"}
DEFINITIONS = {"build_backbone", "RaptorClassifier", "_eval_centers", "_slot_index", "_mirror_tri",
               "infer_pair_probs", "eval_windows", "gpu_windows", "infer_probs", "_make_reader",
               "_pick_series_for_slot", "build_study"}
RECIPE = {"image_size": 384, "crop_mm": 140.0, "input_res": 320, "input_mode": "crop",
          "slots": [["Sagittal", 1, 26], ["Sagittal", 0, 22], ["Coronal", 1, 18],
                    ["Coronal", 0, 12], ["Axial", -1, 18]],
          "windows": 94, "normalization": "imagenet", "fastread": "study_lut",
          "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
          "tta": "equal original and anatomical-mirror probabilities; CUDA fp16",
          "short_series": "public repeated sampling preserved",
          "slot_boundaries": "public cross-slot windows preserved"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def audit_source(source: str) -> tuple[str, str, dict]:
    """Extract only reviewed definitions from exactly the inspected source bytes."""
    if sha256(source.encode()) != SOURCE_SHA256:
        raise ValueError("Public source differs from audited snapshot; review before changing pin")
    tree = ast.parse(source)
    selected, found, payload = [], set(), None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in DEFINITIONS:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names == {"_FASTREAD_B64"}:
                payload = base64.b64decode(ast.literal_eval(node.value), validate=True)
            elif names and names <= CONSTANTS:
                selected.append(node)
                found.update(names)
    if found != CONSTANTS | DEFINITIONS or payload is None:
        raise ValueError("Audited definition inventory differs")
    if sha256(payload) != FASTREAD_SHA256:
        raise ValueError("Embedded decoder differs from audited payload")
    # Imports are ours; no public top-level guards, model loading, file writing,
    # main(), error-to-prior wrappers, or base64 loader are carried into execution.
    prefix = ("import glob\nfrom concurrent.futures import ThreadPoolExecutor\n"
              "import numpy as np\nimport torch\nfrom torch import nn\n"
              "from torch.nn import functional as F\nHDR_THREADS=4\nPX_THREADS=4\n")
    module = prefix + ast.unparse(ast.Module(body=selected, type_ignores=[])) + "\n"
    compile(module, "<audited-public-definitions>", "exec")
    compile(payload, "<audited-fastread>", "exec")
    manifest = {"source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
                "decoded_fastread_sha256": FASTREAD_SHA256,
                "extracted_module_sha256": sha256(module.encode()),
                "checkpoint_sha256": WEIGHT_SHA256, "checkpoint_file": WEIGHT_NAME,
                "weight_dataset": WEIGHT_DATASET, "recipe": RECIPE,
                "attribution": "Danial Zakaria (nartaa), based on Dread Development's Raptor inference",
                "purpose": "bounded offline reproduction; no training or score-based selection",
                "permissions": "Source/checkpoint eligibility is audited separately; this tool grants no license."}
    return module, payload.decode(), manifest


def load_definitions(module_source, decoder_source, manifest):
    if (manifest.get("source_sha256") != SOURCE_SHA256 or
            manifest.get("checkpoint_sha256") != WEIGHT_SHA256 or manifest.get("recipe") != RECIPE):
        raise ValueError("Reference manifest contract mismatch")
    if sha256(module_source.encode()) != manifest["extracted_module_sha256"]:
        raise ValueError("Extracted source hash mismatch")
    if sha256(decoder_source.encode()) != FASTREAD_SHA256:
        raise ValueError("Decoder source hash mismatch")
    public, decoder = types.ModuleType("audited_public"), types.ModuleType("audited_fastread")
    exec(compile(module_source, "<audited-public-definitions>", "exec"), public.__dict__)
    exec(compile(decoder_source, "<audited-fastread>", "exec"), decoder.__dict__)
    # Avoid the public helper raising process-wide file descriptor limits.
    decoder._KEEP_MAX = 256
    return public, decoder


def bounded_ids(ids, *, count=24, seed=20261008):
    """Stable ID-only sampling; no reports, target values or gold membership read."""
    ids = [str(x) for x in ids]
    if len(ids) != len(set(ids)) or any(not x or "/" in x or "\\" in x or x in {".", ".."} for x in ids):
        raise ValueError("Study IDs must be unique nonempty path components")
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 24 or len(ids) < count:
        raise ValueError("Rehearsal requires 1 to 24 available training studies")
    return sorted(ids, key=lambda s: (sha256(f"{seed}:{s}".encode()), s))[:count]


def slot_plan(public, study, records):
    used, plan, substitutions = set(), [], 0
    for plane, fluid, count in public.SLOTS:
        row = public._pick_series_for_slot(records.get(study, []), plane, fluid, used)
        if row is not None:
            series = str(row["SeriesInstanceUID"])
            if not series or "/" in series or "\\" in series or series in {".", ".."}:
                raise ValueError("Invalid series path component")
            used.add(series)
            substitutions += int(fluid in (0, 1) and int(row.get("Fluid_Sensitive", 0) or 0) != fluid)
        plan.append((row, count))
    return plan, substitutions


def validate_volume(volume, mask, plan, *, image_size=384):
    import numpy as np
    size = sum(count for _, count in plan)
    if volume.shape != (size, image_size, image_size) or volume.dtype != np.uint8:
        raise ValueError("Reference volume shape/type mismatch")
    mask = np.asarray(mask)
    if mask.shape != (size,) or not np.isin(mask, [0, 1]).all():
        raise ValueError("Invalid reference mask")
    nonzero = volume.reshape(size, -1).sum(1) > 0
    if not np.array_equal(mask.astype(bool), nonzero) or not nonzero.any():
        raise ValueError("Empty study or pixel/mask inconsistency")
    offset = 0
    for row, count in plan:
        valid = nonzero[offset:offset + count]
        if row is not None and not valid.all():
            raise ValueError("Selected series has unexpectedly zero-filled slices")
        if row is None and valid.any():
            raise ValueError("Absent series unexpectedly contains pixels")
        offset += count
    return {"valid_slices": int(nonzero.sum()), "missing_slots": sum(row is None for row, _ in plan)}


def validate_predictions(values, *, rows=1, targets=12):
    import numpy as np
    values = np.asarray(values)
    expected = (targets,) if rows == 1 else (rows, targets)
    if values.shape != expected or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid reference probability shape/range")
    flat = values.reshape(-1, targets)
    if (np.all(flat == .5, axis=1) | np.all(flat == 0, axis=1) | np.all(flat == 1, axis=1)).any():
        raise ValueError("Constant fallback-like prediction rejected")
    return values


def install_decoder_guards(decoder):
    """Keep exact successful arithmetic; prevent its caught failures becoming zeros."""
    counters, lock = Counter(), threading.Lock()
    original_records, original_result, original_px = decoder._records, decoder._result, decoder._Ctx.px
    original_make_pyd = decoder._make_pyd

    def records(rows):
        if any(not row[3] for row in rows):
            raise ValueError("Fast decoder header failure")
        return original_records(rows)

    def result(selected, got, med_ps):
        if any(value[0] is None for value in got.values()):
            raise ValueError("Fast decoder pixel failure")
        return original_result(selected, got, med_ps)

    def pixels(self, data, info):
        value = original_px(self, data, info)
        if value[0] is None:
            raise ValueError("Fast decoder swallowed pixel failure")
        with lock:
            counters["decoded_unique_slices"] += 1
        return value

    def make_pyd():
        header, pixel = original_make_pyd()
        def read_header(*args):
            with lock:
                counters["pydicom_header_paths"] += 1
            return header(*args)
        def read_pixel(*args):
            with lock:
                counters["pydicom_pixel_paths"] += 1
            return pixel(*args)
        return read_header, read_pixel
    decoder._records, decoder._result, decoder._Ctx.px = records, result, pixels
    decoder._make_pyd = make_pyd
    return counters


def reference_reader(public):
    """Pydicom reference with errors recorded before public build_study catches them."""
    import glob
    import numpy as np
    import pydicom
    _, read_pixel, resize = public._make_reader()
    failures = []
    def order(directory):
        rows = []
        for path in glob.glob(directory + "/*.dcm"):
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                orient = np.asarray(ds.ImageOrientationPatient, dtype=float)
                position = np.asarray(ds.ImagePositionPatient, dtype=float)
                spacing = np.asarray(ds.PixelSpacing, dtype=float)
                if (orient.shape != (6,) or position.shape != (3,) or spacing.shape != (2,) or
                        not all(np.isfinite(x).all() for x in (orient, position, spacing)) or (spacing <= 0).any()):
                    raise ValueError
                coordinate = float(np.dot(position, np.cross(orient[:3], orient[3:])))
                rows.append((coordinate, path, float(spacing[0])))
            except Exception:
                raise ValueError("Reference header failure; no geometry fallback permitted") from None
        if not rows:
            raise ValueError("Selected series has no DICOM files")
        rows.sort(key=lambda item: item[0])
        return [(path, spacing) for _, path, spacing in rows], float(np.median([r[2] for r in rows]))
    def pixels(path):
        try:
            value = read_pixel(path)
            if value.ndim != 2 or not np.isfinite(value).all():
                raise ValueError
            return value
        except Exception:
            failures.append("pixel_failure")
            raise ValueError("Reference pixel failure") from None
    return (order, pixels, resize), failures


def load_model(public, path, device):
    import gc
    import timm
    import torch
    path = Path(path)
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != WEIGHT_SHA256:
            raise ValueError("Reference checkpoint hash mismatch")
    # Never inherit the public weights_only=False pickle execution path.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("arch") != RECIPE["arch"] or int(checkpoint.get("res", 0)) != 384:
        raise ValueError("Reference checkpoint architecture/resolution mismatch")
    public.timm = timm
    backbone = public.build_backbone(RECIPE["arch"], pretrained=False)
    model = public.RaptorClassifier(backbone, F_dim=backbone.num_features)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    del checkpoint
    gc.collect()
    return model


def _csv(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def rehearse(root, checkpoint_path, public_source, decoder_source, manifest, output_dir, *,
             train_count=24, seed=20261008, max_seconds=3600, device="cuda"):
    import importlib.metadata
    import numpy as np
    import torch
    import warnings
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Reference qualification explicitly requires available CUDA")
    if not 0 < max_seconds <= 3600:
        raise ValueError("Reference rehearsal must be bounded to at most 3600 seconds")
    started = time.monotonic()
    root, out = Path(root), Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    public, decoder = load_definitions(public_source, decoder_source, manifest)
    # Pydicom diagnostics can include source values; failures are recorded below
    # as sanitized aggregate codes rather than third-party warnings.
    warnings.filterwarnings("ignore", module=r"pydicom(?:\..*)?")
    header, visible_rows = _csv(root / "test.csv")
    if header != ["StudyInstanceUID"] or len(visible_rows) != 3:
        raise ValueError("Bounded rehearsal requires exactly the three visible test studies")
    with (root / "sample_submission.csv").open(newline="") as handle:
        expected_columns = next(csv.reader(handle))
    if expected_columns != ["StudyInstanceUID", *public.LAB]:
        raise ValueError("Official target order differs from the reference")
    records = {}
    for split in ("train", "test"):
        _, rows = _csv(root / f"{split}_series.csv")
        grouped = {}
        for row in rows:
            grouped.setdefault(row["StudyInstanceUID"], []).append(row)
        records[split] = grouped
    selected_train = bounded_ids(records["train"], count=train_count, seed=seed)
    visible_ids = [row["StudyInstanceUID"] for row in visible_rows]
    bounded_ids(visible_ids, count=3, seed=seed)  # Validate, but preserve official order.
    jobs = [("test", sid) for sid in visible_ids] + [("train", sid) for sid in selected_train]
    counters = install_decoder_guards(decoder)
    fast = decoder.make_builder("study_lut", 384, public.SLOTS, hdr_threads=4, px_threads=4)
    reference, failures = reference_reader(public)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(4)
    model = load_model(public, checkpoint_path, device)
    torch.cuda.reset_peak_memory_stats()
    setup_seconds = time.monotonic() - started
    def package_version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    report = {"status": "running", "manifest_sha256": sha256(canonical_bytes(manifest)),
              "manifest": manifest, "source_revision": __import__("os").environ.get("KNEE_CODE_REVISION"),
              "source_bundle_sha256": __import__("os").environ.get("KNEE_SOURCE_SHA256"),
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "packages": {name: package_version(name) for name in
                                           ("torch", "timm", "numpy", "pydicom", "opencv-python", "opencv-python-headless")},
                              "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda},
              "sampling": {"seed": seed, "train_count": train_count, "test_count": 3,
                           "uses_labels": False, "selection": "SHA256(seed:StudyInstanceUID) order",
                           "ordered_ids_sha256": sha256(canonical_bytes(jobs))},
              "bounds": {"seconds": max_seconds, "threads": 4, "concurrent_studies": 1},
              "studies": [], "prediction_fallbacks": 0, "setup_seconds": setup_seconds}
    visible_predictions, timings = [], []
    def save():
        (out / "rehearsal.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        for ordinal, (split, sid) in enumerate(jobs):
            if time.monotonic() - started >= max_seconds:
                raise TimeoutError("Reference rehearsal time bound exceeded")
            plan, substitutions = slot_plan(public, sid, records[split])
            for row, _ in plan:
                if row is not None and not (root / f"{split}_series" / sid / row["SeriesInstanceUID"]).is_dir():
                    raise ValueError("Selected series directory is absent")
            before = time.monotonic()
            volume, mask = fast(sid, records[split], str(root / f"{split}_series"))
            fast_seconds = time.monotonic() - before
            counts = validate_volume(volume, mask, plan)
            reference_before = time.monotonic()
            reference_volume, reference_mask = public.build_study(sid, records[split], str(root / f"{split}_series"), reference)
            if failures:
                raise ValueError("Reference decoder swallowed an error")
            validate_volume(reference_volume, reference_mask, plan)
            if not np.array_equal(volume, reference_volume) or not np.array_equal(mask, reference_mask):
                raise ValueError("Fast decoder differs from pydicom reference pixels/mask")
            reference_seconds = time.monotonic() - reference_before
            del reference_volume, reference_mask
            windows, window_seconds = [], 0.0
            for mirrored in (False, True):
                before = time.monotonic()
                gpu = public.gpu_windows(volume, mask, 94, 320, device, mirror=mirrored)
                torch.cuda.synchronize()
                window_seconds += time.monotonic() - before
                cpu = public.eval_windows(volume, mask, 94, 320, mirror=mirrored)
                if not torch.equal(cpu, gpu.cpu()):
                    raise ValueError("CPU and GPU windows are not bit-identical")
                del cpu
                windows.append(gpu)
            torch.cuda.synchronize()
            before = time.monotonic()
            probabilities = validate_predictions(public.infer_pair_probs(model, *windows, device))
            torch.cuda.synchronize()
            inference_seconds = time.monotonic() - before
            repeated = validate_predictions(public.infer_pair_probs(model, *windows, device))
            repeat_error = float(np.max(np.abs(probabilities - repeated)))
            if repeat_error > 1e-6:
                raise ValueError("Repeated predictions exceed 1e-6 tolerance")
            if split == "test":
                visible_predictions.append(probabilities)
            production_seconds = fast_seconds + window_seconds + inference_seconds
            timings.append(production_seconds)
            report["studies"].append({"ordinal": ordinal, "split": split, **counts,
                                      "sequence_preference_substitutions": substitutions,
                                      "fast_seconds": fast_seconds, "reference_seconds": reference_seconds,
                                      "windows_seconds": window_seconds, "inference_seconds": inference_seconds,
                                      "production_seconds": production_seconds,
                                      "decoder_pixel_max_error": 0, "window_max_error": 0,
                                      "repeat_prediction_max_error": repeat_error})
            del volume, mask, windows, gpu
            report["runtime_seconds"] = time.monotonic() - started
            save()
            print(json.dumps({"event": "reference_study_verified", "ordinal": ordinal,
                              "split": split, "production_seconds": production_seconds}), flush=True)
        values = validate_predictions(np.stack(visible_predictions), rows=3)
        with (out / "visible-predictions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(expected_columns)
            writer.writerows([[sid, *p.tolist()] for sid, p in zip(visible_ids, values)])
        written_header, written_rows = _csv(out / "visible-predictions.csv")
        if written_header != expected_columns or [r["StudyInstanceUID"] for r in written_rows] != visible_ids:
            raise ValueError("Visible prediction coverage/order mismatch")
        report.update(status="passed", runtime_seconds=time.monotonic() - started,
                      decoder_counts=dict(counters),
                      estimated_1300_seconds_mean=setup_seconds + float(np.mean(timings)) * 1300,
                      estimated_1300_seconds_p95=setup_seconds + float(np.percentile(timings, 95)) * 1300,
                      runtime_estimate_note="Sequential measured sample extrapolation, not hidden-test runtime; excludes qualification-only parity/repeats.",
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                      peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                      output_disk_bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
                      disk_free_bytes=shutil.disk_usage(out).free,
                      visible_csv_sha256=sha256((out / "visible-predictions.csv").read_bytes()))
        report["estimated_1300_seconds_p95_with_25_percent_margin"] = report["estimated_1300_seconds_p95"] * 1.25
        if report["estimated_1300_seconds_p95_with_25_percent_margin"] >= 9 * 3600:
            raise ValueError("Measured runtime estimate lacks required notebook headroom")
        save()
    except Exception as exc:
        report.update(status="failed", failure_type=type(exc).__name__,
                      runtime_seconds=time.monotonic() - started, decoder_counts=dict(counters))
        save()
        raise RuntimeError("Reference rehearsal failed; inspect sanitized aggregate report") from None
    return report
