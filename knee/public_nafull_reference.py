from __future__ import annotations

import ast

import base64

from collections import Counter

import csv

import hashlib

import json

from pathlib import Path

import platform

import re

import resource

import shutil

import sys

import threading

import time

import types

SOURCE_SHA256 = "493c7b0d44fe5244b02024448a93f5393cdb073781a068951badb3ffd53fcbde"

FASTREAD_SHA256 = "5a59b6f82e846bcdfd452aaee6ff518cdd9d0b04aab03691b45c3f70a5198cb0"

WEIGHT_SHA256 = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"

RAPTOR_SOURCE_SHA256 = "859aae12ecea593b45ab0a5f6463fa573cd3083f340c161e920f10fbba29e81b"

RAPTOR_WEIGHT_DATASET = "dreaddevelopment/raptor-knee-widedense"

RAPTOR_WEIGHT_NAME = "raptor_ft_coatnet_v4_full.pt"

RECIPE = {"image_size": 384, "crop_mm": 140.0, "input_res": 320, "input_mode": "crop",
          "slots": [["Sagittal", 1, 26], ["Sagittal", 0, 22], ["Coronal", 1, 18],
                    ["Coronal", 0, 12], ["Axial", -1, 18]],
          "windows": 94, "normalization": "imagenet", "fastread": "study_lut",
          "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
          "tta": "equal original and anatomical-mirror probabilities; CUDA fp16",
          "short_series": "public repeated sampling preserved",
          "slot_boundaries": "public cross-slot windows preserved"}

RAPTOR_RECIPE = {"image_size": 336, "crop_mm": 140.0, "input_res": 384, "input_mode": "resize",
                 "slots": [["Sagittal", 1, 18], ["Sagittal", 0, 14], ["Coronal", 1, 12],
                           ["Coronal", 0, 8], ["Axial", -1, 12]],
                 "windows": 42, "normalization": "imagenet", "fastread": "memoized_pydicom",
                 "slice_span": [0.06, 0.94], "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
                 "tta": "none; CUDA fp16; no silent fp32 fallback",
                 "short_series": "public repeated sampling preserved",
                 "slot_boundaries": "public cross-slot windows preserved"}

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def load_definitions(module_source, decoder_source, manifest):
    mirror = manifest.get("profile", "mirror") == "mirror"
    if (manifest.get("profile", "mirror") not in {"mirror", "raptor"} or
            manifest.get("source_sha256") != (SOURCE_SHA256 if mirror else RAPTOR_SOURCE_SHA256) or
            (mirror and manifest.get("checkpoint_sha256") != WEIGHT_SHA256) or
            not re.fullmatch(r"[0-9a-f]{64}", manifest.get("checkpoint_sha256", "")) or
            manifest.get("recipe") not in ([RECIPE] if mirror else [RAPTOR_RECIPE, {**RAPTOR_RECIPE, "windows": 62}])):
        raise ValueError("Reference manifest contract mismatch")
    if sha256(module_source.encode()) != manifest["extracted_module_sha256"]:
        raise ValueError("Extracted source hash mismatch")
    if (mirror and sha256(decoder_source.encode()) != FASTREAD_SHA256) or (not mirror and decoder_source):
        raise ValueError("Decoder source hash mismatch")
    public, decoder = types.ModuleType("audited_public"), types.ModuleType("audited_fastread")
    exec(compile(module_source, "<audited-public-definitions>", "exec"), public.__dict__)
    if not mirror:
        from knee.raptor_input import propagate_fatal, transform_exception_handlers
        protected, contract = transform_exception_handlers(module_source)
        if manifest.get('exception_policy') != contract:
            raise ValueError('Raptor protected exception policy differs')
        public._raptor_propagate_fatal = propagate_fatal
        public._raptor_protected_source = protected
        exec(compile(protected, '<raptor-resource-policy>', 'exec'), public.__dict__)
        if public.K_EVAL != 42:
            raise ValueError("Pinned public Raptor source must retain its original 42 windows")
        # Explicit evaluation-only variant; preserve all public scientific definitions.
        public.K_EVAL = manifest["recipe"]["windows"]
        return public, None
    from knee.raptor_input import propagate_fatal, transform_exception_handlers
    from knee.mirror_input import protect_decoder
    protected, policy = transform_exception_handlers(module_source)
    if manifest.get('exception_policy') != policy:
        raise ValueError('Mirror public exception policy differs')
    public._raptor_propagate_fatal = propagate_fatal
    public._raptor_protected_source = protected
    exec(compile(protected, '<mirror-public-resource-policy>', 'exec'), public.__dict__)
    protected_decoder, policy = protect_decoder(decoder_source)
    if manifest.get('decoder_exception_policy') != policy:
        raise ValueError('Mirror decoder exception policy differs')
    decoder._mirror_propagate_fatal = propagate_fatal
    decoder._mirror_protected_source = protected_decoder
    exec(compile(protected_decoder, "<audited-fastread-protected>", "exec"), decoder.__dict__)
    # Avoid the public helper raising process-wide file descriptor limits.
    decoder._KEEP_MAX = 256
    return public, decoder

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

def load_model(public, path, device, manifest):
    import gc
    import timm
    import torch
    path = Path(path)
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != manifest["checkpoint_sha256"]:
            raise ValueError("Reference checkpoint hash mismatch")
    if manifest.get('profile') == 'mirror':
        from knee.mirror_input import CHECKPOINT_BYTES
        if path.stat().st_size != CHECKPOINT_BYTES:
            raise ValueError('Mirror checkpoint size mismatch')
    # Never inherit the public weights_only=False pickle execution path.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("arch") != manifest["recipe"]["arch"] or int(checkpoint.get("res", 0)) != 384:
        raise ValueError("Reference checkpoint architecture/resolution mismatch")
    if "lab" in checkpoint and list(checkpoint["lab"]) != public.LAB:
        raise ValueError("Reference checkpoint target order mismatch")
    public.timm = timm
    backbone = public.build_backbone(manifest["recipe"]["arch"], pretrained=False)
    model = public.RaptorClassifier(backbone, F_dim=backbone.num_features)
    if manifest.get('profile') == 'mirror' and (not checkpoint.get('model') or
            not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in checkpoint['model'].values())):
        raise ValueError('Mirror checkpoint tensors nonfinite or invalid')
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    del checkpoint
    gc.collect()
    return model

def infer_checked(public, model, windows, device, *, mirror):
    import torch
    if mirror:
        return public.infer_pair_probs(model, *windows, device)
    if any(layer.training for layer in model.modules()):
        raise ValueError("Reference inference unexpectedly in training mode")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        logits = model(windows[0].unsqueeze(0)).float()
        if not torch.isfinite(logits).all():
            raise ValueError("Reference logits are nonfinite")
        return logits.sigmoid()[0].cpu().numpy()
