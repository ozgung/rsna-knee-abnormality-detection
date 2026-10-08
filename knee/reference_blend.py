"""Fixed equal-rank blend of two hash-pinned public references; no fitting.

Only the inspected ConvNeXt reader and plain, non-OAI Raptor are supported.
ML imports stay inside runtime functions so contract/ranking checks work offline.
"""
from __future__ import annotations

import ast
from collections import Counter
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import time
import types
import warnings

from knee.public_reader_contract import INPUT_POLICY_VERSION, require_reader_rehearsal

from knee.reference import (RAPTOR_WEIGHT_DATASET, RAPTOR_WEIGHT_NAME, canonical_bytes,
                            infer_checked, load_definitions, load_model, reference_reader,
                            sha256, slot_plan, validate_predictions, validate_volume)

CONVNEXT_DATASET = "goodpjw2008/rsna-knee-2-5d-convnext-reader"
CONVNEXT_URL = "https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944"
CONVNEXT_SOURCE_HASHES = {
    "preprocess.py": "4491963e4d2156625c9cf79051f4a25c32d2d761f1f07df3117a439ed8029ae2",
    "knee.py": "8e9fd05dd94e9be70d4b4a6cbdf5201c219b5e6eb3d57c07bb8992c8dbe72ae7",
}
CONVNEXT_ENTRY_SHA256 = "450feb0a5090d427ee754bfba6695804d978b5687f0fc1fb13ad9878300db863"
RAPTOR_PLAIN_SHA256 = "89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded"
DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
# Exact successful Python 3.12 install from the standalone ConvNeXt rehearsal log.
CODEC_WHEELS = (
    "pylibjpeg-2.1.0-py3-none-any.whl",
    "pylibjpeg_libjpeg-2.4.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
    "pylibjpeg_openjpeg-2.6.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
)
RECIPE = {"version": 1, "components": ["convnext-three-fold-mean-probability", "plain-raptor"],
          "weights": [0.5, 0.5], "ranking": "per target over all actual test IDs; average ties; rank/n",
          "selection": "predeclared; no labels or leaderboard-based weight choice", "tta": "none"}


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def select_codec_wheels(directory):
    """Select the pinned cp312/pure-Python codecs, ignoring other Python ABIs.

    Reject another compatible wheel for any codec, including a second version or
    platform build. Only these three paths are ever passed to offline pip.
    """
    compatible = {}
    for path in Path(directory).glob("pylibjpeg*.whl"):
        parts = path.name[:-4].split("-")
        if path.is_file() and len(parts) >= 5 and {"py3", "cp312"}.intersection(parts[-3].split(".")):
            compatible[path.name] = path
    if set(compatible) != set(CODEC_WHEELS):
        raise ValueError("Missing, duplicate or unexpected compatible ConvNeXt codec wheel")
    return [compatible[name] for name in CODEC_WHEELS]


def extract_reader(entry):
    """Reuse exact reviewed functions, excluding top-level installs/model loads/jobs."""
    if sha256(entry.encode()) != CONVNEXT_ENTRY_SHA256:
        raise ValueError("ConvNeXt entry differs from the reviewed rehearsal wrapper")
    names = {"count_event", "fatal_series_error", "load_selected_series", "prepare_study", "predict", "run"}
    nodes = [n for n in ast.parse(entry).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("ConvNeXt definition inventory changed")
    return ast.unparse(ast.Module(body=nodes, type_ignores=[])) + "\n"


def qualify_components(cnn_contract, cnn_report, raptor_contract, raptor_report, manifest):
    """Require successful independent component rehearsals before even packaging."""
    if (cnn_contract.get("rehearsal") is not True or cnn_report.get("rehearsal") is not True or
            cnn_report.get("status") != "PASS" or cnn_contract.get("sources") != CONVNEXT_SOURCE_HASHES or
            cnn_report.get("source_hashes") != CONVNEXT_SOURCE_HASHES or
            cnn_contract.get("entry_sha256") != CONVNEXT_ENTRY_SHA256 or
            cnn_contract.get("input_policy_version") != INPUT_POLICY_VERSION or
            cnn_contract.get("revision") != cnn_report.get("code_revision")):
        raise ValueError("ConvNeXt successful rehearsal/source contract required")
    weights = require_reader_rehearsal(cnn_report, CONVNEXT_ENTRY_SHA256, CONVNEXT_SOURCE_HASHES)
    if (raptor_contract.get("source") != manifest or raptor_report.get("manifest") != manifest or
            manifest.get("profile") != "raptor" or manifest.get("checkpoint_sha256") != RAPTOR_PLAIN_SHA256 or
            manifest.get("checkpoint_file") != RAPTOR_WEIGHT_NAME or
            manifest.get("weight_dataset") != RAPTOR_WEIGHT_DATASET or
            raptor_report.get("status") != "passed" or raptor_report.get("prediction_fallbacks") != 0 or
            raptor_report.get("manifest_sha256") != sha256(canonical_bytes(manifest)) or
            not _digest(raptor_report.get("source_bundle_sha256")) or
            raptor_contract.get("private") is not True or raptor_contract.get("internet") is not False):
        raise ValueError("Plain Raptor successful rehearsal/source contract required")
    sampling, studies = raptor_report.get("sampling", {}), raptor_report.get("studies", [])
    if (sampling.get("uses_labels") is not False or sampling.get("test_count") != 3 or
            sampling.get("train_count", 0) < 16 or
            sampling.get("train_count") != raptor_contract.get("train_count") or
            sampling.get("seed") != raptor_contract.get("seed") or
            len(studies) != sampling["train_count"] + 3 or
            any(s.get("valid_slices", 0) <= 0 or s.get("decoder_pixel_max_error") != 0 or
                not 0 <= s.get("window_max_error", math.inf) <= 3e-5 or
                not 0 <= s.get("repeat_prediction_max_error", math.inf) <= 1e-6 for s in studies)):
        raise ValueError("Raptor rehearsal sampling/parity guards failed")
    # ConvNeXt report has a sample mean, Raptor a p95. Keep that distinction explicit.
    cnn_seconds = cnn_report.get("estimated_1300_studies_seconds", math.inf)
    raptor_seconds = raptor_report.get("estimated_1300_seconds_p95", math.inf)
    estimated = 1.25 * (cnn_seconds + raptor_seconds)
    if not math.isfinite(estimated) or not 0 < estimated < 8 * 3600:
        raise ValueError("Component runtime projections do not fit the eight-hour blend budget")
    return {"convnext_weights": weights, "raptor_manifest": manifest,
            "estimated_1300_seconds_with_margin": estimated,
            "estimate_note": "1.25 * (ConvNeXt sample mean projection + Raptor sample p95 projection); different source environments; not a hidden-test measurement",
            "original_environments": {"convnext": cnn_report.get("versions"), "raptor": raptor_report.get("environment")}}


def probability_matrix(values, rows):
    """Component validation rejects constant fallback rows; rank output uses its own guard."""
    import numpy as np
    result = np.asarray(values)
    if result.shape != (rows, 12):
        raise ValueError("Component must have one ordered row and twelve targets per study")
    validate_predictions(result[0] if rows == 1 else result, rows=rows)
    return result


def equal_rank_blend(first, second):
    """Average of two target-wise percentile ranks, with average ranks for exact ties.

    Raw probabilities are untouched. A one-study cohort ranks to one; a wholly tied
    target ranks to (n+1)/(2n). No softmax or per-study class normalization is applied.
    """
    if len(first) != len(second) or not len(first):
        raise ValueError("Component row counts must match and be nonempty")
    rows = len(first)
    a, b = probability_matrix(first, rows), probability_matrix(second, rows)
    import numpy as np
    result = np.zeros((rows, 12), dtype=np.float64)
    for component in (a, b):
        for target in range(12):
            order = np.argsort(component[:, target], kind="stable")
            sorted_values = component[order, target]
            start = 0
            while start < rows:
                end = start + 1
                while end < rows and sorted_values[end] == sorted_values[start]:
                    end += 1
                result[order[start:end], target] += 0.5 * ((start + 1 + end) / (2.0 * rows))
                start = end
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Invalid blended rank probabilities")
    return result


def read_test_contract(root, *, rehearsal):
    """Use real test.csv IDs, never sample_submission row coverage or gold files."""
    root = Path(root)
    with (root / "sample_submission.csv").open(newline="") as handle:
        columns = next(csv.reader(handle))
    if len(columns) != 13 or columns[0] != "StudyInstanceUID" or len(set(columns)) != 13:
        raise ValueError("Official twelve-target schema is invalid")
    with (root / "test.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [columns[0]]:
            raise ValueError("Unexpected test ID schema")
        ids = [row[columns[0]] for row in reader]
    def safe(value):
        return isinstance(value, str) and value and value not in {".", ".."} and "/" not in value and "\\" not in value
    if not ids or len(ids) != len(set(ids)) or not all(safe(value) for value in ids):
        raise ValueError("Actual test IDs must be unique nonempty path components")
    if rehearsal and len(ids) != 3:
        raise ValueError("Visible-only qualification requires exactly three test studies")
    with (root / "test_series.csv").open(newline="") as handle:
        records = {}
        for row in csv.DictReader(handle):
            if not safe(row[columns[0]]) or not safe(row["SeriesInstanceUID"]):
                raise ValueError("Invalid series metadata path component")
            records.setdefault(row[columns[0]], []).append(row)
    if set(records) != set(ids):
        raise ValueError("Actual test/series ID coverage differs")
    return columns, ids, records


def write_probabilities(path, columns, ids, values):
    """Validate every row before exclusive creation; supports legitimate rank endpoints."""
    if len(columns) != 13 or columns[0] != "StudyInstanceUID" or len(set(columns)) != 13:
        raise ValueError("Prediction target schema differs")
    if not ids or len(set(ids)) != len(ids) or len(values) != len(ids):
        raise ValueError("Prediction coverage differs")
    rows = [[float(x) for x in row] for row in values]
    if any(len(row) != 12 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in row) for row in rows):
        raise ValueError("Predictions require twelve finite probabilities in range")
    with Path(path).open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows([sid, *row] for sid, row in zip(ids, rows))
    return sha256(Path(path).read_bytes())


def _source_module(name, source):
    module = types.ModuleType(name)
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    return module


def _convnext(root, assets, ids, labels, source, entry, weights, rehearsal, deadline):
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    for name, digest in CONVNEXT_SOURCE_HASHES.items():
        if sha256(source[name].encode()) != digest:
            raise ValueError("ConvNeXt source hash changed")
    extracted = extract_reader(entry)
    prep = _source_module("audited_blend_preprocess", source["preprocess.py"])
    model_code = _source_module("audited_blend_model", source["knee.py"])
    if labels != model_code.LABELS:
        raise ValueError("ConvNeXt target order differs from competition")
    models = []
    for i in range(3):
        path = Path(assets) / f"cnxt_v0_fold{i}.pt"
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != weights[path.name]:
                raise ValueError("ConvNeXt checkpoint differs from successful rehearsal")
        ck = torch.load(path, map_location="cpu", weights_only=True)
        args = ck["args"]
        if args["backbone"] != "convnext_tiny.fb_in22k_ft_in1k" or args["res"] != 256 or args["k_eval"] != 12:
            raise ValueError("ConvNeXt architecture/resolution/window contract changed")
        model = model_code.KneeNet(args["backbone"].split(".")[0], pretrained=False, img_size=args["res"])
        model.load_state_dict(ck["model"], strict=True)
        models.append((model.cuda().to(memory_format=torch.channels_last).eval(), args["res"]))
        del ck, model
    audit = {"studies": 0, "selected_series": 0, "missing_slots": 0, "decode_failures": 0, "zero_studies": 0,
             "input_policy_version": INPUT_POLICY_VERSION, "resource_errors": 0, "partial_series": 0,
             "failed_series": 0, "short_series": 0, "zero_decoded_series": 0, "series_error_types": {}}
    namespace = {"np": np, "pd": pd, "torch": torch, "time": time, "Path": Path,
                 "root": Path(root), "prep": prep, "model_code": model_code, "models": models,
                 "audit": audit, "labels": labels, "cv2": cv2, "REHEARSAL": False, "t0": time.monotonic()}
    exec(compile(extracted, "<audited-convnext-entry-functions>", "exec"), namespace)
    # This is the exact independently rehearsed traversal/preparation/three-fold mean.
    prediction, durations = namespace["run"]("test", ids)
    if prediction.index.tolist() != ids or prediction.columns.tolist() != labels:
        raise ValueError("ConvNeXt prediction ID/target order differs")
    values = probability_matrix(prediction.to_numpy(), len(ids)).copy()
    deadline()
    production_audit = dict(audit)
    error, rerun_values = None, None
    if rehearsal:
        # Invoke its standalone path again: it enumerates/rereads raw DICOM. No fixture/cache.
        repeated, _ = namespace["run"]("test", ids)
        rerun_values = probability_matrix(repeated.to_numpy(), len(ids)).copy()
        error = float(np.max(np.abs(values - rerun_values)))
        if error > 1e-6:
            raise ValueError("ConvNeXt independent raw reread parity failed")
    namespace.clear()
    models.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return values, {**production_audit, "study_seconds": durations, "raw_rerun_max_abs_error": error}, rerun_values


def _raptor(root, assets, ids, labels, records, source, manifest, rehearsal, deadline):
    import numpy as np
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    public, _ = load_definitions(source, "", manifest)
    if labels != public.LAB or manifest["checkpoint_sha256"] != RAPTOR_PLAIN_SHA256:
        raise ValueError("Plain Raptor identity or target order mismatch")
    model = load_model(public, Path(assets) / manifest["checkpoint_file"], "cuda", manifest)
    reader, failures = reference_reader(public)
    counters, durations, result, rerun_values = Counter(), [], [], []
    error = None
    for ordinal, sid in enumerate(ids):
        deadline()
        started = time.monotonic()
        plan, substitutions = slot_plan(public, sid, records)
        cache = {}
        def cached(path):
            if path not in cache:
                cache[path] = reader[1](path)
            return cache[path]
        volume, mask = public.build_study(sid, records, str(Path(root) / "test_series"), (reader[0], cached, reader[2]))
        cache.clear()
        if failures:
            raise ValueError("Raptor decoder swallowed a pixel failure")
        counts = validate_volume(volume, mask, plan, image_size=public.IMG)
        windows = public.eval_windows(volume, mask, public.K_EVAL, 384).cuda()
        probability = validate_predictions(infer_checked(public, model, [windows], "cuda", mirror=False))
        torch.cuda.synchronize()
        durations.append(time.monotonic() - started)
        result.append(probability)
        counters.update(counts)
        counters["sequence_preference_substitutions"] += substitutions
        if rehearsal:
            # Separate public reference traversal with an uncached strict reader.
            reference_volume, reference_mask = public.build_study(sid, records, str(Path(root) / "test_series"), reader)
            if failures or not np.array_equal(volume, reference_volume) or not np.array_equal(mask, reference_mask):
                raise ValueError("Raptor independent raw reread pixel/mask parity failed")
            qwindows = public.eval_windows(reference_volume, reference_mask, public.K_EVAL, 384).cuda()
            q = validate_predictions(infer_checked(public, model, [qwindows], "cuda", mirror=False))
            rerun_values.append(q)
            error = max(error or 0, float(np.max(np.abs(probability - q))))
            if error > 1e-6:
                raise ValueError("Raptor independent raw reread prediction parity failed")
            del reference_volume, reference_mask, qwindows
        del volume, mask, windows
        if (ordinal + 1) % 8 == 0:
            print(json.dumps({"component": "raptor", "completed_studies": ordinal + 1, "total_studies": len(ids)}), flush=True)
    del model, public
    gc.collect()
    torch.cuda.empty_cache()
    return (probability_matrix(np.stack(result), len(ids)), {**dict(counters), "studies": len(ids),
            "decode_failures": 0, "zero_studies": 0, "study_seconds": durations, "raw_rerun_max_abs_error": error},
            probability_matrix(np.stack(rerun_values), len(ids)) if rehearsal else None)


def run_blend(root, cnn_assets, raptor_assets, cnn_sources, cnn_entry, raptor_source, contract, out, *, codec_hashes):
    """Sequential CUDA inference; full execution requires a common-environment receipt."""
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Blend explicitly requires CUDA; no device fallback")
    if contract["recipe"] != RECIPE or not 0 < contract["max_seconds"] <= 8 * 3600:
        raise ValueError("Fixed blend recipe/runtime contract changed")
    if not contract["rehearsal"] and not _digest(contract.get("blend_rehearsal_sha256")):
        raise ValueError("Full blend entry requires its successful common-environment rehearsal")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started >= contract["max_seconds"]:
            raise TimeoutError("Blend runtime bound exhausted; no partial submission")
    report = {"status": "running", "rehearsal": contract["rehearsal"],
              "qualification_sha256": contract["qualification_sha256"], "contract": contract,
              "codec_hashes": codec_hashes, "components": {}, "prediction_fallbacks": 0}
    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        (out / "blend-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        # Decoder diagnostics can include source values; retain only our aggregate failures.
        warnings.filterwarnings("ignore", module=r"pydicom(?:\..*)?")
        torch.set_num_threads(4)
        torch.manual_seed(20261008)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        columns, ids, records = read_test_contract(root, rehearsal=contract["rehearsal"])
        report["test_studies"] = len(ids)
        report["ordered_ids_sha256"] = sha256(canonical_bytes(ids))
        report["environment"] = {"python": sys.version, "gpu": torch.cuda.get_device_name(0),
                                 "packages": {p: importlib.metadata.version(p) for p in ("torch", "timm", "numpy", "pandas", "pydicom", "pylibjpeg")}}
        cnn, report["components"]["convnext"], cnn_rerun = _convnext(root, cnn_assets, ids, columns[1:], cnn_sources,
            cnn_entry, contract["qualification"]["convnext_weights"], contract["rehearsal"], deadline)
        report["convnext_csv_sha256"] = write_probabilities(out / "convnext-raw.csv", columns, ids, cnn)
        save()
        raptor, report["components"]["raptor"], raptor_rerun = _raptor(root, raptor_assets, ids, columns[1:], records,
            raptor_source, contract["qualification"]["raptor_manifest"], contract["rehearsal"], deadline)
        report["raptor_csv_sha256"] = write_probabilities(out / "raptor-raw.csv", columns, ids, raptor)
        blend = equal_rank_blend(cnn, raptor)
        repeated = equal_rank_blend(cnn_rerun, raptor_rerun) if contract["rehearsal"] else equal_rank_blend(cnn.copy(), raptor.copy())
        if not np.array_equal(blend, repeated):
            raise ValueError("Rank blend repetition differs")
        report["blend_repeat_max_abs_error"] = 0.0
        production = np.array(report["components"]["convnext"]["study_seconds"]) + np.array(report["components"]["raptor"]["study_seconds"])
        report["estimated_1300_seconds_mean_with_margin"] = 1.25 * float(production.mean()) * 1300
        report["runtime_estimate_note"] = "Sequential sample mean with 25% margin; visible-only sample includes warm-up and is not hidden-test timing"
        if contract["rehearsal"] and report["estimated_1300_seconds_mean_with_margin"] >= 8 * 3600:
            raise ValueError("Common-environment runtime estimate exceeds the eight-hour cap")
        deadline()
        name = "visible-blend.csv" if contract["rehearsal"] else "submission.csv"
        report["blend_csv_sha256"] = write_probabilities(out / name, columns, ids, blend)
        report.update(status="PASS", peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            disk_free_bytes=shutil.disk_usage(out).free,
            output_disk_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()))
        save()
        return report
    except Exception as exc:
        report.update(status="FAIL", failure_type=type(exc).__name__)
        save()
        raise RuntimeError("Blend failed; inspect sanitized aggregate audit") from None
