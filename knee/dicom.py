"""Bounded, deterministic MRI preprocessing shared by training and inference.

Only :func:`load_study` requires numpy, pydicom and torch. Inputs are official
``*_series.csv`` records; ``root`` is the corresponding ``*_series`` directory.
No report or target information participates in selection or normalization.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping
import warnings


PREPROCESSING_VERSION = "dicom-physical-v1"
PLANES = ("sagittal", "coronal", "axial")
_REQUIRED = ("StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive",
             "Fat_Suppression", "Anatomical_Plane")


@dataclass
class StudyVolume:
    """Images [plane, slice, height, width] in [0,1], with explicit padding masks.

    ``patient_hashes`` must remain in private artifacts: hashes permit linkage,
    and their presence does not establish that PatientID is useful for grouping.
    ``audit`` contains only aggregate counts, settings, and sanitized error codes.
    """

    images: Any
    slice_mask: Any
    plane_mask: Any
    audit: dict[str, Any]
    patient_hashes: tuple[str, ...] = field(default_factory=tuple, repr=False)


class StudyLoadError(ValueError):
    """No plane could be loaded; audit contains sanitized failure counts."""

    def __init__(self, audit: dict[str, Any]):
        super().__init__("Study has no usable DICOM planes; inspect aggregate audit")
        self.audit = audit


class _SeriesError(Exception):
    pass


def _component(value: Any) -> str:
    value = str(value).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("Study and series identifiers must be single path components")
    return value


def _flag(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _series_path(root: Path, study: str, series: str) -> Path:
    direct = root / study / series
    if direct.is_dir():
        return direct
    # Also accept a competition root, but never arbitrarily choose a split.
    matches = [root / split / study / series for split in ("train_series", "test_series")
               if (root / split / study / series).is_dir()]
    if len(matches) != 1:
        raise _SeriesError("missing_or_ambiguous_series_directory")
    return matches[0]


def _vector(value: Any, length: int, code: str, np: Any) -> Any:
    try:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (length,) or not np.isfinite(result).all():
            raise ValueError
        return result
    except (TypeError, ValueError):
        raise _SeriesError(code) from None


def _geometry(ds: Any, np: Any) -> tuple[Any, Any, Any, tuple[int, int]]:
    orient = _vector(getattr(ds, "ImageOrientationPatient", None), 6,
                     "missing_or_invalid_orientation", np)
    width, height = orient[:3], orient[3:]
    if (abs(np.linalg.norm(width) - 1) > 0.02 or
            abs(np.linalg.norm(height) - 1) > 0.02 or abs(width @ height) > 0.02):
        raise _SeriesError("nonorthonormal_orientation")
    position = _vector(getattr(ds, "ImagePositionPatient", None), 3,
                       "missing_or_invalid_position", np)
    spacing = _vector(getattr(ds, "PixelSpacing", None), 2,
                      "missing_or_invalid_spacing", np)
    if (spacing <= 0).any() or (spacing > 20).any():
        raise _SeriesError("missing_or_invalid_spacing")
    try:
        shape = (int(ds.Rows), int(ds.Columns))
    except (AttributeError, TypeError, ValueError):
        raise _SeriesError("invalid_pixel_dimensions") from None
    if min(shape) < 1 or max(shape) > 4096:
        raise _SeriesError("invalid_pixel_dimensions")
    if int(getattr(ds, "NumberOfFrames", 1)) != 1:
        raise _SeriesError("multiframe_not_supported")
    return orient, position, spacing, shape


def _canonical(image: Any, spacing: Any, orient: Any, plane: int, np: Any) -> tuple[Any, Any]:
    # Horizontal axis points posterior for sagittal and patient-left otherwise;
    # vertical axis points inferior for sagittal/coronal and posterior for axial.
    target_height = np.asarray((0, 1, 0) if plane == 2 else (0, 0, -1))
    target_width = np.asarray((0, 1, 0) if plane == 0 else (1, 0, 0))
    width, height = orient[:3], orient[3:]
    direct = abs(width @ target_width) + abs(height @ target_height)
    swapped = abs(height @ target_width) + abs(width @ target_height)
    if swapped > direct:
        image, spacing = image.T, spacing[::-1]
        width, height = height, width
    if width @ target_width < 0:
        image = image[:, ::-1]
    if height @ target_height < 0:
        image = image[::-1]
    return image, spacing


def _physical_crop(image: Any, spacing: Any, crop_mm: float, np: Any) -> Any:
    target_h, target_w = (max(1, int(round(crop_mm / float(s)))) for s in spacing)
    # Bound allocations even when corrupted spacing claims submicron pixels.
    if max(target_h, target_w) > 4096:
        raise _SeriesError("physical_crop_exceeds_bound")
    h, w = image.shape
    result = np.zeros((target_h, target_w), dtype=np.float32)
    copy_h, copy_w = min(h, target_h), min(w, target_w)
    src_y, src_x = (h - copy_h) // 2, (w - copy_w) // 2
    dst_y, dst_x = (target_h - copy_h) // 2, (target_w - copy_w) // 2
    result[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = image[
        src_y:src_y + copy_h, src_x:src_x + copy_w]
    return result


def _load_series(path: Path, plane: int, count: int, image_size: int, crop_mm: float,
                 max_files: int, counters: Counter, np: Any, pydicom: Any) -> tuple[Any, tuple[str, ...]]:
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".dcm")
    if not files:
        raise _SeriesError("no_dicom_files")
    if len(files) > max_files:
        raise _SeriesError("series_exceeds_header_bound")
    headers = []
    for file in files:
        try:
            # Suppress third-party warnings which can include paths/identifiers.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = pydicom.dcmread(file, stop_before_pixels=True)
            geometry = _geometry(ds, np)
        except _SeriesError:
            raise
        except Exception:
            raise _SeriesError("header_read_failed") from None
        headers.append((file, ds, geometry))
        counters["headers_read"] += 1

    orient, _, spacing, shape = headers[0][2]
    normal = np.cross(orient[:3], orient[3:])
    axis = int(np.argmax(np.abs(normal)))
    if axis != plane or abs(normal[axis]) < 0.7:
        raise _SeriesError("metadata_geometry_plane_mismatch")
    # Increasing patient coordinate, independent of acquisition reversal.
    normal = normal * (1 if normal[axis] >= 0 else -1)
    located = []
    for file, ds, (other_orient, pos, other_spacing, other_shape) in headers:
        if (orient[:3] @ other_orient[:3] < math.cos(math.radians(3)) or
                orient[3:] @ other_orient[3:] < math.cos(math.radians(3))):
            raise _SeriesError("inconsistent_orientation")
        if not np.allclose(spacing, other_spacing, rtol=0.01, atol=1e-4) or shape != other_shape:
            raise _SeriesError("inconsistent_pixel_geometry")
        located.append((float(pos @ normal), file, ds))
    located.sort(key=lambda item: (item[0], str(item[1])))
    unique = []
    for item in located:
        if unique and abs(item[0] - unique[-1][0]) < 1e-3:
            counters["duplicate_positions_removed"] += 1
        else:
            unique.append(item)
    # Even coverage without replacement; fewer slices means explicit padding.
    selected = np.rint(np.linspace(0, len(unique) - 1, min(count, len(unique)))).astype(int)
    arrays, valid_arrays, photos, patient_hashes = [], [], [], set()
    for index in selected:
        _, file, header = unique[int(index)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = pydicom.dcmread(file)
                photo = str(getattr(ds, "PhotometricInterpretation", "")).upper()
                if photo not in {"MONOCHROME1", "MONOCHROME2"}:
                    raise _SeriesError("unsupported_photometric_interpretation")
                counters["pixel_decode_attempts"] += 1
                pixels = ds.pixel_array
            if pixels.ndim == 3 and pixels.shape[0] == 1:
                pixels = pixels[0]
            if pixels.ndim != 2 or pixels.shape != shape:
                raise _SeriesError("unsupported_pixel_shape")
            valid = np.ones(pixels.shape, dtype=bool)
            if hasattr(ds, "PixelPaddingValue"):
                pad = float(ds.PixelPaddingValue)
                pad_end = float(getattr(ds, "PixelPaddingRangeLimit", pad))
                valid &= ~((pixels >= min(pad, pad_end)) & (pixels <= max(pad, pad_end)))
            slope, intercept = float(getattr(ds, "RescaleSlope", 1)), float(getattr(ds, "RescaleIntercept", 0))
            if not math.isfinite(slope) or not math.isfinite(intercept) or slope == 0:
                raise _SeriesError("invalid_rescale")
            pixels = pixels.astype(np.float32) * slope + intercept
            if not np.isfinite(pixels).all():
                raise _SeriesError("nonfinite_pixels")
            if not valid.any():
                raise _SeriesError("all_padding_slice")
            arrays.append(pixels)
            valid_arrays.append(valid)
            photos.append(photo)
            patient = str(getattr(header, "PatientID", "")).strip()
            if patient:
                patient_hashes.add(hashlib.sha256(("knee-patient-v1:" + patient).encode()).hexdigest())
            counters["slices_decoded"] += 1
        except _SeriesError:
            raise
        except Exception:
            raise _SeriesError("pixel_decode_failed") from None
    if len(set(photos)) != 1:
        raise _SeriesError("inconsistent_photometric_interpretation")
    values = np.concatenate([a[v] for a, v in zip(arrays, valid_arrays)])
    low, high = np.percentile(values, [0.5, 99.5])
    if not high > low:
        raise _SeriesError("constant_intensity_series")
    crops = []
    for pixels, valid in zip(arrays, valid_arrays):
        pixels = np.clip((pixels - low) / (high - low), 0, 1).astype(np.float32)
        if photos[0] == "MONOCHROME1":
            pixels = 1 - pixels
        pixels[~valid] = 0
        pixels, canonical_spacing = _canonical(pixels, spacing, orient, plane, np)
        crops.append(_physical_crop(pixels, canonical_spacing, crop_mm, np))
    import torch
    import torch.nn.functional as F

    batch = torch.from_numpy(np.stack(crops))[:, None]
    resized = F.interpolate(batch, size=(image_size, image_size), mode="bilinear",
                            align_corners=False, antialias=True)[:, 0].numpy()
    return np.clip(resized, 0, 1), tuple(sorted(patient_hashes))


def load_study(root: str | Path, study_id: Any, series_rows: Iterable[Mapping[str, Any]],
               image_size: int = 160, slices_per_plane: int = 8, crop_mm: float = 160,
               *, max_files_per_series: int = 1024, max_series_per_plane: int = 3) -> StudyVolume:
    """Load one deterministic fluid-sensitive-preferred series in each plane.

    Series preference is fluid-sensitive first, then fat suppression, then UID.
    Up to ``max_series_per_plane`` candidates are attempted, in that order.
    Headers are scanned before evenly sampling at most ``slices_per_plane``
    unique physical positions; pixels are decoded only for those positions.
    A corrupt series is rejected as a unit, with a sanitized reason in audit.
    Missing or failed planes and padded slices are zero and explicitly masked.
    Normalization uses only the selected study pixels, never dataset statistics.

    Unsupported enhanced/multiframe DICOM and missing geometry fail explicitly.
    Masks must be used during model pooling; black slices are not validity masks.
    """
    for name, value, maximum in (("image_size", image_size, 1024),
                                 ("slices_per_plane", slices_per_plane, 64),
                                 ("max_files_per_series", max_files_per_series, 4096),
                                 ("max_series_per_plane", max_series_per_plane, 20)):
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    if not isinstance(crop_mm, (float, int)) or not math.isfinite(crop_mm) or not 0 < crop_mm <= 500:
        raise ValueError("crop_mm must be finite and in (0, 500]")
    import numpy as np
    import pydicom

    study, root = _component(study_id), Path(root)
    candidates: list[dict[str, int]] = [{}, {}, {}]
    counters, errors = Counter(), Counter()
    for row in series_rows:
        if any(key not in row for key in _REQUIRED):
            raise ValueError("Series records must contain all official series metadata columns")
        if str(row["StudyInstanceUID"]).strip() != study:
            continue
        name = str(row["Anatomical_Plane"]).strip().lower()
        if name not in PLANES:
            counters["unknown_plane_rows"] += 1
            continue
        series = _component(row["SeriesInstanceUID"])
        score = 2 * _flag(row["Fluid_Sensitive"]) + _flag(row["Fat_Suppression"])
        plane = PLANES.index(name)
        candidates[plane][series] = max(score, candidates[plane].get(series, -1))
    images = np.zeros((3, slices_per_plane, image_size, image_size), dtype=np.float32)
    mask = np.zeros((3, slices_per_plane), dtype=bool)
    patient_hashes = set()
    for plane, choices in enumerate(candidates):
        for series in sorted(choices, key=lambda s: (-choices[s], s))[:max_series_per_plane]:
            counters["series_attempted"] += 1
            try:
                path = _series_path(root, study, series)
                volume, hashes = _load_series(path, plane, slices_per_plane, image_size, float(crop_mm),
                                             max_files_per_series, counters, np, pydicom)
            except _SeriesError as exc:
                errors[str(exc)] += 1
                continue
            except OSError:
                errors["series_io_failed"] += 1
                continue
            images[plane, :len(volume)] = volume
            mask[plane, :len(volume)] = True
            patient_hashes.update(hashes)
            counters["series_used"] += 1
            break
    plane_mask = mask.any(axis=1)
    audit = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "planes": list(PLANES), "image_size": image_size,
        "slices_per_plane": slices_per_plane, "crop_mm": float(crop_mm),
        "max_files_per_series": max_files_per_series,
        "max_series_per_plane": max_series_per_plane,
        "candidate_series": sum(map(len, candidates)),
        "valid_slices": int(mask.sum()), "missing_planes": int((~plane_mask).sum()),
        "patient_id_hash_count": len(patient_hashes),
        "counts": dict(sorted(counters.items())), "errors": dict(sorted(errors.items())),
    }
    if not plane_mask.any():
        raise StudyLoadError(audit)
    return StudyVolume(images, mask, plane_mask, audit, tuple(sorted(patient_hashes)))
