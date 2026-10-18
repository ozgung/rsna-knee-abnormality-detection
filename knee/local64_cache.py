"""Private local64 cache contracts and fail-closed shard bookkeeping.

The renderer is supplied by the separately pinned public S6 source closure. This
module owns only the cache schema, historical membership, slot indexing, privacy,
and K0 evidence aggregation; it does not redefine DICOM geometry.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

from knee import public_training_smoke as smoke
from knee import source96_cache as base

VERSION = "local64-s6-anchor-384crop352-v1"
DEPTH = 64
STORAGE = 352
RENDER_SIZE = 384
SPAN = (0.06, 0.94)
SLOTS = (("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
         ("Coronal", 0, 8), ("Axial", -1, 12))
VIEWS = {"sagittal": {"plane": "Sagittal", "anchor": "joint", "crop_mm": 100.0},
         "coronal": {"plane": "Coronal", "anchor": "joint", "crop_mm": 130.0},
         "axial": {"plane": "Axial", "anchor": "patella", "crop_mm": 100.0}}
TRAIN_COUNT = 4345
GOLD_COUNT = 58
MASK_SCHEMA = "sagittal_joint_mask_native_projected_then_same_geometry_nearest"
RAW_BYTES = DEPTH * STORAGE * STORAGE + DEPTH + STORAGE * STORAGE
MAX_FILE_BYTES = RAW_BYTES + 65536
DISK_RESERVE = 512 * 1024**2

safe_id, make_shards, membership, filename, safe_file, atomic_json = (
    base.safe_id, base.make_shards, base.membership, base.filename, base.safe_file, base.atomic_json)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def slug(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) or len(value) > 50:
        raise ValueError("Invalid Kaggle slug")
    return value


def partition_population(training_ids, gold_ids):
    """Keep historical training order and scoring-only gold membership disjoint."""
    train = list(training_ids)
    gold = list(gold_ids)
    if len(train) != TRAIN_COUNT or len(set(train)) != TRAIN_COUNT:
        raise ValueError("Expected 4,345 unique historical training studies")
    if len(gold) != GOLD_COUNT or len(set(gold)) != GOLD_COUNT:
        raise ValueError("Expected 58 unique scoring studies")
    if set(train) & set(gold):
        raise ValueError("Gold scoring membership overlaps training")
    shards = [[train[n] for n in indices] for indices in make_shards(train, 4)]
    if [len(s) for s in shards] != [1087, 1086, 1086, 1086]:
        raise ValueError("Historical four-shard sizes differ")
    return shards, gold


def selected_frame_indices(frame_count, slot_count):
    """Exact Dread 6–94% linspace/round frame picks, including repeated short stacks."""
    import numpy as np
    if type(frame_count) is not int or frame_count < 1 or type(slot_count) is not int or slot_count <= 0:
        raise ValueError("Invalid frame/slot count")
    lo = int(frame_count * SPAN[0])
    hi = max(int(frame_count * SPAN[1]) - 1, lo)
    return np.linspace(lo, hi, slot_count).round().astype(np.int64).clip(0, frame_count - 1)


def center_crop_rendered(image):
    """Apply the recipe's exact 384→352 central crop to a rendered plane or stack."""
    import numpy as np
    a = np.asarray(image)
    if a.shape[-2:] != (RENDER_SIZE, RENDER_SIZE):
        raise ValueError("Expected 384×384 renderer output")
    return np.ascontiguousarray(a[..., 16:368, 16:368])


def align_binary_mask(mask, geometry, *, cv2_module=None):
    """Render a native binary mask through the image geometry and crop with nearest sampling."""
    import numpy as np
    if cv2_module is None:
        import cv2 as cv2_module
    native = np.asarray(mask)
    if native.ndim != 2 or native.dtype not in (np.uint8, np.bool_) or not np.isin(native, (0, 1)).all():
        raise ValueError("Native joint mask must be a binary plane")
    if tuple(native.shape) != tuple(geometry.native_shape):
        raise ValueError("Joint mask and image geometry native shapes differ")
    x0, y0, x1, y1 = map(int, geometry.crop_xyxy)
    px, py, _, _ = map(int, geometry.valid_xyxy)
    oh, ow = map(int, geometry.resized_shape)
    resized = cv2_module.resize(native[y0:y1, x0:x1].astype(np.uint8), (ow, oh), interpolation=cv2_module.INTER_NEAREST)
    canvas = np.zeros((RENDER_SIZE, RENDER_SIZE), dtype=np.uint8)
    canvas[py:py + oh, px:px + ow] = resized
    return center_crop_rendered(canvas)


def validate_slots(slots):
    if not isinstance(slots, list) or len(slots) != len(SLOTS):
        raise ValueError("Slot metadata must contain the five pinned slot groups")
    total = 0
    for row, (plane, fluid, count) in zip(slots, SLOTS):
        if (not isinstance(row, dict) or row.get("plane") != plane or row.get("fluid_sensitive") != fluid or
                row.get("count") != count or type(row.get("selected")) is not bool):
            raise ValueError("Slot order/plane/sensitivity/count differs")
        if row["selected"]:
            if not isinstance(row.get("series_key"), str) or not re.fullmatch(r"[0-9a-f]{24}", row["series_key"]):
                raise ValueError("Selected slot lacks opaque series key")
            if type(row.get("frame_count")) is not int or row["frame_count"] < 1:
                raise ValueError("Selected slot frame count is invalid")
        elif row.get("series_key") is not None or row.get("frame_count") != 0:
            raise ValueError("Missing slot must not carry a series mapping")
        total += count
    if total != DEPTH:
        raise ValueError("Slot counts do not cover 64 slices")


def map_dread_files_to_native(dread_files, native_files):
    """Map Dread's ordered raw DICOM files onto S6 native-cache rows by exact basename."""
    import numpy as np
    raw = [Path(p).name for p in dread_files]
    native = [Path(p).name for p in native_files]
    if not raw or len(raw) != len(set(raw)) or len(native) != len(set(native)):
        raise ValueError("DICOM frame inventory is empty or ambiguous")
    by_name = {name: i for i, name in enumerate(native)}
    if set(raw) != set(native):
        raise ValueError("Dread/native DICOM frame membership differs")
    return np.asarray([by_name[name] for name in raw], dtype=np.int64)


def frame_reference_fallback(geometry_by_series, selected_series, anchor):
    """Fail closed for cross-series joint placement without one registered FoR."""
    if anchor is None or anchor.get("joint_lps") is None:
        return "anchor_unavailable"
    refs = set()
    for sid in selected_series:
        geometry = geometry_by_series.get(sid)
        ref = geometry.get("frame_of_reference_uid") if isinstance(geometry, dict) else None
        if not isinstance(ref, str) or not ref:
            return "frame_of_reference_missing"
        refs.add(ref)
    if len(refs) > 1:
        return "incompatible_frame_of_reference"
    return None


def local_anchor(anchor_record, series_id, *, frame_fallback_reason=None):
    """Return normalized in-plane anchor and an explicit learned/fallback label."""
    import numpy as np
    if frame_fallback_reason:
        return (.5, .5), "field_center", frame_fallback_reason
    entry = (anchor_record or {}).get("series", {}).get(series_id)
    if not entry or not entry.get("in_field", True) or not entry.get("joint"):
        return (.5, .5), "field_center", "anchor_not_in_field"
    row, col = map(float, entry["joint"])
    if not np.isfinite((row, col)).all() or not (0 <= row <= 1 and 0 <= col <= 1):
        return (.5, .5), "field_center", "invalid_projected_anchor"
    method = str((anchor_record or {}).get("method") or "")
    if method.startswith("skm_tea_joint"):
        return (row, col), "learned", None
    if method == "slab_prior":
        return (row, col), "slab_prior", None
    return (.5, .5), "field_center", "unknown_anchor_method"


def compose_slots(slot_rows, rendered_by_series):
    """Compose exact slot order from selected-series local planes; absent slots stay masked."""
    import numpy as np
    validate_slots(slot_rows)
    volume = np.zeros((DEPTH, STORAGE, STORAGE), dtype=np.uint8)
    mask = np.zeros((DEPTH,), dtype=np.uint8)
    offset = 0
    records = []
    for slot, (plane, fluid, count) in zip(slot_rows, SLOTS):
        if slot["selected"]:
            key = slot["series_key"]
            source = rendered_by_series.get(key)
            if source is None:
                raise ValueError("Selected Dread slot has no exact rendered series")
            image_stack = np.asarray(source["images"])
            frame_indices = np.asarray(source["dread_to_native"], dtype=np.int64)
            if image_stack.ndim != 3 or image_stack.shape[1:] != (STORAGE, STORAGE) or image_stack.dtype != np.uint8:
                raise ValueError("Rendered series must be uint8[N,352,352]")
            chosen = selected_frame_indices(len(frame_indices), count)
            native = frame_indices[chosen]
            if native.min(initial=0) < 0 or native.max(initial=-1) >= len(image_stack):
                raise ValueError("Dread frame mapping points outside native image stack")
            frames = image_stack[native]
            volume[offset:offset + count] = frames
            # This synthetic composer has no source-reader receipt, so it can
            # only represent the slot-presence mask. Runtime uses the exact
            # mask returned by unmodified Dread prepare_study.
            mask[offset:offset + count] = 1
            records.append({"plane": plane, "fluid_sensitive": fluid, "count": count,
                            "selected": True, "series_key": key,
                            "frame_count": len(frame_indices), "selected_frame_indices": chosen.tolist(),
                            "native_frame_indices": native.tolist(),
                            "anchor_path": source.get("anchor_path"),
                            "fallback_reason": source.get("fallback_reason")})
        else:
            records.append({"plane": plane, "fluid_sensitive": fluid, "count": count,
                            "selected": False, "series_key": None, "frame_count": 0,
                            "selected_frame_indices": [], "native_frame_indices": []})
        offset += count
    return volume, mask, records


def validate_arrays(volume, mask, joint_mask):
    import numpy as np
    if volume.shape != (DEPTH, STORAGE, STORAGE) or volume.dtype != np.uint8:
        raise ValueError("local64 volume must be uint8[64,352,352]")
    if mask.shape != (DEPTH,) or mask.dtype != np.uint8 or not np.isin(mask, (0, 1)).all():
        raise ValueError("local64 missing-plane mask must be uint8[64] in {0,1}")
    # Availability is inherited from the pinned Dread source preparation path.
    # A valid selected frame can be dark after a different local crop, so local
    # pixel content must never rewrite the authoritative source mask.
    if not mask.any():
        raise ValueError("Entirely empty study cannot qualify")
    if not volume[mask.astype(bool)].any():
        raise ValueError("Local crop has no usable pixels on any available source frame")
    if joint_mask.shape != (STORAGE, STORAGE) or joint_mask.dtype != np.uint8 or not np.isin(joint_mask, (0, 1)).all():
        raise ValueError("Sagittal joint mask must be uint8[352,352] in {0,1}")
    return True


def validate_slot_mask(mask, slots):
    """Bind the authoritative Dread availability mask to its five selected slots."""
    import numpy as np
    validate_slots(slots)
    expected = np.concatenate([np.full(count, int(row["selected"]), dtype=np.uint8)
                               for row, (_, _, count) in zip(slots, SLOTS)])
    if mask.shape != (DEPTH,) or not np.array_equal(mask, expected):
        raise ValueError("Dread availability mask and selected slot metadata differ")
    return True


def read_npz(path):
    """Read the bounded on-disk schema before allocating arrays."""
    import numpy as np
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("Invalid local64 cache file")
    limits = {"volume.npy": DEPTH * STORAGE * STORAGE + 4096,
              "mask.npy": DEPTH + 4096, "joint_mask.npy": STORAGE * STORAGE + 4096}
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        if len(infos) != 3 or {i.filename for i in infos} != set(limits):
            raise ValueError("Unexpected local64 archive members")
        for info in infos:
            if info.file_size > limits[info.filename]:
                raise ValueError("Local64 member exceeds size bound")
            with z.open(info) as stream:
                if np.lib.format.read_magic(stream) != (1, 0):
                    raise ValueError("Unexpected NPY format")
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                expected = {"volume.npy": (DEPTH, STORAGE, STORAGE), "mask.npy": (DEPTH,),
                            "joint_mask.npy": (STORAGE, STORAGE)}[info.filename]
                if shape != expected or fortran or dtype != np.dtype("uint8") or info.file_size != stream.tell() + int(np.prod(shape)):
                    raise ValueError("Local64 NPY header differs from contract")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"volume", "mask", "joint_mask"}:
            raise ValueError("Unexpected local64 NPZ keys")
        volume, mask, joint = z["volume"], z["mask"], z["joint_mask"]
    validate_arrays(volume, mask, joint)
    return volume, mask, joint


def array_hashes(volume, mask, joint_mask):
    import numpy as np
    validate_arrays(volume, mask, joint_mask)
    return {"volume_sha256": smoke.digest(volume.tobytes()), "mask_sha256": smoke.digest(mask.tobytes()),
            "joint_mask_sha256": smoke.digest(joint_mask.tobytes())}


def save_npz(path, volume, mask, joint_mask):
    import numpy as np
    path = Path(path)
    hashes = array_hashes(volume, mask, joint_mask)
    if path.is_symlink() or path.with_name(path.name + ".tmp").is_symlink():
        raise ValueError("Cache symlinks forbidden")
    tmp = path.with_name(path.name + ".tmp")
    if path.exists():
        old = read_npz(path)
        if array_hashes(*old) != hashes:
            raise ValueError("Existing immutable cache file differs")
        return {**hashes, "file_sha256": smoke.file_sha(path), "bytes": path.stat().st_size}
    try:
        with tmp.open("wb") as f:
            np.savez_compressed(f, volume=volume, mask=mask, joint_mask=joint_mask)
        arrays = read_npz(tmp)
        if array_hashes(*arrays) != hashes:
            raise ValueError("NPZ save/reload hash differs")
        tmp.replace(path)
    except BaseException:
        if tmp.exists() and not tmp.is_symlink():
            tmp.unlink()
        raise
    return {**hashes, "file_sha256": smoke.file_sha(path), "bytes": path.stat().st_size}


def private_study_entry(ordinal, study_id, file_name, file_receipt, slots, anchor, side, joint_mask_meta):
    validate_slots(slots)
    if type(ordinal) is not int or ordinal < 0 or file_name != filename(ordinal):
        raise ValueError("Study index/name binding differs")
    if not isinstance(anchor, dict) or anchor.get("path") not in ("learned", "slab_prior", "field_center"):
        raise ValueError("Anchor path must be explicit")
    if not isinstance(side, dict) or type(side.get("flipped")) is not bool:
        raise ValueError("Side decision/flip metadata missing")
    if not isinstance(joint_mask_meta, dict) or joint_mask_meta.get("schema") != MASK_SCHEMA:
        raise ValueError("Joint-mask alignment metadata missing")
    return {"ordinal": ordinal, "StudyInstanceUID": study_id, "file": file_name,
            **file_receipt, "slots": slots, "anchor": anchor, "side": side,
            "joint_mask": joint_mask_meta}


def validate_private_entry(entry):
    """Schema check used by metadata collection; private identifiers stay in the private index."""
    if not isinstance(entry, dict) or type(entry.get("ordinal")) is not int:
        raise ValueError("Malformed private index entry")
    uid = entry.get("StudyInstanceUID")
    if not isinstance(uid, str) or not uid or "/" in uid or "\\" in uid:
        raise ValueError("Malformed private study ID")
    validate_slots(entry.get("slots"))
    if entry.get("anchor", {}).get("path") not in ("learned", "slab_prior", "field_center"):
        raise ValueError("Malformed anchor path")
    if entry.get("joint_mask", {}).get("schema") != MASK_SCHEMA:
        raise ValueError("Malformed joint-mask schema")
    for slot in entry["slots"]:
        if slot.get("requested_fluid_sensitive", slot.get("fluid_sensitive")) != slot.get("fluid_sensitive"):
            raise ValueError("Requested slot sensitivity metadata differs")
        if slot["selected"] and not ("source_fluid_sensitive" in slot and "source_fat_suppression" in slot):
            raise ValueError("Selected source series sensitivity metadata missing")


def k0_aggregate(entries, *, visual_entries=None, montage_review=None, expected_binding=None):
    """Compute full-cache placement rates separately from the frozen32 visual review."""
    rows = list(entries)
    visual = list(visual_entries or [])
    if not rows:
        raise ValueError("K0 needs full-cache study entries")
    for row in [*rows, *visual]:
        validate_private_entry(row)
    if len({row["StudyInstanceUID"] for row in rows}) != len(rows):
        raise ValueError("Duplicate full-cache K0 study")
    selected_sag = {}
    for row in rows:
        for slot in row["slots"][:2]:
            if slot["selected"]:
                selected_sag[slot["series_key"]] = slot.get("anchor_path", row["anchor"]["path"])
    effective_studies = sum(row["anchor"].get("path") == "learned" for row in rows)
    model_studies = sum(str(row["anchor"].get("bank_method", "")).startswith("skm_tea_joint") for row in rows)
    learned_series = sum(path == "learned" for path in selected_sag.values())
    field_center = sum(path == "field_center" for path in selected_sag.values())
    full_bounds = {
        "effective_learned_anchor_studies_ge_60pct": effective_studies / len(rows) >= .60,
        "sagittal_field_center_le_30pct": bool(selected_sag) and field_center / len(selected_sag) <= .30,
    }
    visual_uids = [row["StudyInstanceUID"] for row in visual]
    visual_ok = len(visual) == 32 and len(set(visual_uids)) == 32
    wrong_values = [row["side"].get("wrong_side") for row in visual]
    axis_values = [row["anchor"].get("axis_error") for row in visual]
    trunc_values = [row["anchor"].get("sagittal_truncated") for row in visual]
    axial_values = [row["anchor"].get("axial_patella_retained") for row in visual]
    wrong_known = visual_ok and all(type(v) is bool for v in wrong_values)
    axis_known = visual_ok and all(type(v) is bool for v in axis_values)
    trunc_known = visual_ok and all(type(v) is bool for v in trunc_values)
    axial_selected = [row["anchor"].get("axial_patella_retained") for row in visual
                      if any(slot["selected"] for slot in row["slots"][4:5])]
    axial_known = visual_ok and all(
        type(row["anchor"].get("axial_patella_retained")) is bool if row["slots"][4]["selected"]
        else row["anchor"].get("axial_patella_retained") is None for row in visual)
    axial_all_retained = axial_known and all(v is True for v in axial_selected)
    wrong_side = sum(v is True for v in wrong_values)
    axis_errors = sum(v is True for v in axis_values)
    truncated = sum(v is True for v in trunc_values)
    visual_bounds = {
        "wrong_side_zero": wrong_known and wrong_side == 0,
        "axis_errors_zero": axis_known and axis_errors == 0,
        "truncated_sagittal_le_2_of_32": visual_ok and trunc_known and truncated <= 2,
        "axial_patella_retained_when_selected": visual_ok and axial_known and axial_all_retained,
    }
    reviewed = (isinstance(montage_review, dict) and isinstance(expected_binding, dict) and
                montage_review.get("status") == "REVIEWED" and bool(montage_review.get("reviewer")) and
                bool(montage_review.get("date")) and not montage_review.get("unknowns") and
                all(montage_review.get(key) == value for key, value in expected_binding.items()))
    status = "PASS" if visual_ok and all(full_bounds.values()) and all(visual_bounds.values()) and reviewed else "UNKNOWN"
    return {"status": status, "full_cache_study_count": len(rows),
            "effective_learned_anchor_studies": effective_studies,
            "auxiliary_model_path_studies": model_studies,
            "selected_sagittal_series": len(selected_sag), "effective_learned_sagittal_series": learned_series,
            "sagittal_field_center_fallbacks": field_center,
            "frozen32_visual_studies": len(visual), "wrong_side": wrong_side,
            "axis_errors": axis_errors, "truncated_sagittal": truncated,
            "selected_axial_patella_retained": sum(v is True for v in axial_selected),
            "selected_axial_series": len(axial_selected),
            "review_unknowns": {"wrong_side": not wrong_known, "axis_errors": not axis_known,
                                "truncated_sagittal": not trunc_known, "axial_patella_retained": not axial_known},
            "full_cache_bounds": full_bounds, "frozen32_visual_bounds": visual_bounds,
            "montage_review": montage_review or {"status": "PENDING"},
            "anatomy_pass_claimed": status == "PASS"}


def disk_required(studies):
    if type(studies) is not int or studies < 0:
        raise ValueError("Invalid remaining study count")
    return studies * MAX_FILE_BYTES + 2 * MAX_FILE_BYTES + DISK_RESERVE
