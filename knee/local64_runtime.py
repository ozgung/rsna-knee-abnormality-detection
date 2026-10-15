"""CPU-only S6-native/Dread-slot local64 cache runtime.

This adapter uses the accepted S6 source closure for native reads, side choice,
anchors, image intensity mapping and view geometry. It maps Dread-selected raw
frame filenames to native-cache rows exactly and never averages/reorders frames
by an assumed slab index.
"""
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

from knee import local64_cache as cache
from knee import public_training_smoke as smoke
from knee.reference import slot_plan

UID = "StudyInstanceUID"
SERIES_COLUMNS = (UID, "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane")


def _read_series(path, selected_ids):
    records = {}
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if not set(SERIES_COLUMNS) <= set(reader.fieldnames or ()):
            raise ValueError("Official series metadata schema differs")
        for row in reader:
            if row[UID] in selected_ids:
                records.setdefault(row[UID], []).append(row)
    if set(records) != set(selected_ids):
        raise ValueError("Selected study metadata coverage differs")
    return records


def _stage_image_only_inputs(raw_root, staged_root, selected_ids):
    """Stage only selected IDs and public image metadata; train.csv contains IDs only."""
    raw_root, staged_root = Path(raw_root), Path(staged_root)
    if staged_root.is_symlink():
        raise ValueError("Image-only staging root must not be a symlink")
    # Only rebuild our derived image-only links/metadata. Never follow links to
    # immutable raw DICOMs; completed NPZ/index checkpoints live outside this tree.
    if staged_root.exists():
        if not staged_root.is_dir() or {p.name for p in staged_root.iterdir()} != {"train.csv", "train_series.csv", "train_series"}:
            raise ValueError("Unexpected image-only staging inventory")
        shutil.rmtree(staged_root)
    ordered_ids = list(selected_ids)
    selected = set(ordered_ids)
    staged_root.mkdir(parents=True)
    with (raw_root / "train_series.csv").open(newline="") as src:
        reader = csv.DictReader(src)
        if not set(SERIES_COLUMNS) <= set(reader.fieldnames or ()):
            raise ValueError("Official series metadata schema differs")
        rows = [{name: row[name] for name in SERIES_COLUMNS} for row in reader if row[UID] in selected]
    pairs = [(row[UID], row["SeriesInstanceUID"]) for row in rows]
    if {uid for uid, _ in pairs} != selected or len(pairs) != len(set(pairs)):
        raise ValueError("Selected image metadata coverage or uniqueness differs")
    with (staged_root / "train.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow([UID]); w.writerows([[uid] for uid in ordered_ids])
    with (staged_root / "train_series.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SERIES_COLUMNS); w.writeheader(); w.writerows(rows)
    images = staged_root / "train_series"; images.mkdir()
    inventory = []
    for uid, sid in pairs:
        src = raw_root / "train_series" / uid / sid
        if not src.is_dir() or not src.resolve().is_relative_to((raw_root / "train_series" / uid).resolve()):
            raise ValueError("Selected image directory escapes raw study folder")
        dst = images / uid / sid; dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src.resolve(), target_is_directory=True)
        if not any(dst.glob("*.dcm")):
            raise ValueError("Selected image series has no DICOM files")
        inventory.append([uid, sid, str(dst.resolve())])
    return {"studies": len(selected), "series": len(rows),
            "id_csv_sha256": smoke.file_sha(staged_root / "train.csv"),
            "series_csv_sha256": smoke.file_sha(staged_root / "train_series.csv"),
            "image_paths_sha256": smoke.digest(smoke.canonical(inventory)),
            "label_values_staged": False, "report_text_staged": False}


def _anchor_model_state(dataset):
    h = hashlib.sha256()
    modules = [("anchor", dataset.bank.model if dataset.bank else None), ("side", dataset.side_net)]
    for role, model in modules:
        if model is None:
            h.update(cache.canonical([role, None])); continue
        state = {"parameter/" + n: t for n, t in model.named_parameters()}
        state.update({"buffer/" + n: t for n, t in model.named_buffers()})
        for name, tensor in sorted(state.items()):
            value = tensor.detach().cpu().contiguous().numpy()
            h.update(cache.canonical([role, name, str(value.dtype), list(value.shape)])); h.update(value.tobytes())
    return h.hexdigest()


def _attach_finite_hooks(dataset):
    import torch
    hooks = []
    for module in (dataset.bank.model if dataset.bank else None, dataset.side_net):
        if module is None:
            continue
        def finite(_module, _inputs, output):
            values = output if isinstance(output, (tuple, list)) else (output,)
            for value in values:
                if torch.is_tensor(value) and not torch.isfinite(value).all().item():
                    raise ValueError("Nonfinite CPU anchor/side model output")
        hooks.append(module.register_forward_hook(finite))
    return hooks


def _native_series_map(study):
    result = {}
    for s in study["series"]:
        sid = s.get("SeriesInstanceUID")
        files = s.get("files")
        if not isinstance(sid, str) or sid in result or not isinstance(files, list):
            raise ValueError("Native series inventory lacks a unique file mapping")
        result[sid] = s
    return result


def _geometry_map(work):
    out = {}
    for path in (Path(work) / "geometry").glob("*.json"):
        obj = json.loads(path.read_bytes())
        out[path.stem] = obj
    return out


def _point_in_retained(point, geometry, *, crop=352):
    """Whether normalized native [row,col] point survives 384→352 and letterboxing."""
    if point is None:
        return False
    row, col = map(float, point)
    h, w = map(int, geometry.native_shape)
    x0, y0, x1, y1 = map(float, geometry.crop_xyxy)
    px, py, px1, py1 = map(float, geometry.valid_xyxy)
    off = (geometry.size - crop) // 2
    ix, iy = col * max(w - 1, 1), row * max(h - 1, 1)
    return (x0 <= ix <= x1 and y0 <= iy <= y1 and px + (ix - x0) * (px1 - px) / (x1 - x0) >= off and
            px + (ix - x0) * (px1 - px) / (x1 - x0) < off + crop and
            py + (iy - y0) * (py1 - py) / (y1 - y0) >= off and
            py + (iy - y0) * (py1 - py) / (y1 - y0) < off + crop)


def _render_one(series, native_index, center, crop_mm, *, view_geometry, normalize_intensity,
                select_display_window, cv2, volume):
    spacing = series["spacing_mm"][native_index]
    geometry = view_geometry(volume.shape[1:], spacing, cache.RENDER_SIZE, crop_mm=crop_mm, center=center)
    x0, y0, x1, y1 = map(int, geometry.crop_xyxy)
    px, py, _, _ = map(int, geometry.valid_xyxy)
    oh, ow = map(int, geometry.resized_shape)
    window = series.get("_display_window")
    if window is None:
        window = select_display_window(series, "series_companded")
        series["_display_window"] = window
    pixels = cv2.resize(volume[native_index, y0:y1, x0:x1].astype("float32"), (ow, oh), interpolation=cv2.INTER_AREA)
    pixels = normalize_intensity(pixels, window, "series_companded")
    out = __import__("numpy").zeros((cache.RENDER_SIZE, cache.RENDER_SIZE), dtype="uint8")
    out[py:py + oh, px:px + ow] = __import__("numpy").rint(__import__("numpy").clip(pixels, 0, 1) * 255).astype("uint8")
    return cache.center_crop_rendered(out), geometry


def _retained_native_rectangle(geometry):
    """Native-coordinate rectangle that survives both letterbox and 384→352 crop."""
    x0, y0, x1, y1 = map(float, geometry.crop_xyxy)
    px, py, px1, py1 = map(float, geometry.valid_xyxy)
    crop_lo, crop_hi = 16., 368.
    left, top, right, bottom = max(px, crop_lo), max(py, crop_lo), min(px1, crop_hi), min(py1, crop_hi)
    if right <= left or bottom <= top or px1 <= px or py1 <= py:
        return None
    return (x0 + (left - px) * (x1 - x0) / (px1 - px),
            y0 + (top - py) * (y1 - y0) / (py1 - py),
            x0 + (right - px) * (x1 - x0) / (px1 - px),
            y0 + (bottom - py) * (y1 - y0) / (py1 - py))


def _montage_pages(uid, diagnostic_frames, destination, *, cv2):
    """Write paginated matched native-source/crop-outline and local352 diagnostic views."""
    import numpy as np
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(uid.encode()).hexdigest()[:24]
    pages = []
    for page_index in range((len(diagnostic_frames) + 7) // 8):
        page = np.zeros((4 * 368, 2 * 720, 3), dtype=np.uint8)
        for local_index, record in enumerate(diagnostic_frames[page_index * 8:(page_index + 1) * 8]):
            row, col = divmod(local_index, 2)
            x, y = col * 720, row * 368
            source = cv2.cvtColor(record["source"], cv2.COLOR_GRAY2BGR)
            rendered = cv2.cvtColor(record["rendered"], cv2.COLOR_GRAY2BGR)
            page[y + 16:y + 368, x:x + 352] = source
            page[y + 16:y + 368, x + 360:x + 712] = rendered
            cv2.putText(page, "native source + retained crop outline", (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, .30, (255, 255, 255), 1)
            label = (f"{record['plane']} slot{record['slot']} frame{record['dread_frame']} native{record['native_index']} "
                     f"{record['frame_key']} {record['crop_mm']}mm {record['anchor_path']}")
            cv2.putText(page, label[:100], (x + 360, y + 12), cv2.FONT_HERSHEY_SIMPLEX, .30, (255, 255, 255), 1)
            bounds = record.get("retained_native_rectangle")
            if bounds:
                h, w = record["native_shape"]
                x0, y0, x1, y1 = bounds
                rect = (round(x0 * 352 / max(w, 1)), round(y0 * 352 / max(h, 1)),
                        round(x1 * 352 / max(w, 1)), round(y1 * 352 / max(h, 1)))
                cv2.rectangle(page, (x + rect[0], y + 16 + rect[1]),
                              (x + rect[2], y + 16 + rect[3]), (0, 255, 0), 2)
        name = f"{key}-page{page_index:02d}.png"
        path = destination / name
        if not cv2.imwrite(str(path), page):
            raise ValueError("Failed to write private K0 montage page")
        pages.append({"file": name, "sha256": smoke.file_sha(path), "bytes": path.stat().st_size,
                      "first_frame_ordinal": page_index * 8, "frame_count": min(8, len(diagnostic_frames) - page_index * 8)})
    return {"study_key": key, "frame_count": len(diagnostic_frames), "pages": pages}


def _render_slots(uid, data_root, work, dataset, public, metadata, *, code, montage_dir=None):
    import cv2
    import numpy as np
    import px_cache as PX
    import training.native_data as ND
    from training.input_provenance import view_geometry
    from training.intensity_mapping import normalize_intensity
    cv2.setNumThreads(1)
    raw_records = metadata[uid]
    volume = np.zeros((cache.DEPTH, cache.STORAGE, cache.STORAGE), dtype=np.uint8)
    # Keep Dread's original availability semantics. Local crop darkness is not
    # a missing source frame and must not change the sampling mask.
    _source_volume, mask, source_counts = _dread_source_inputs(public, uid, {uid: raw_records}, data_root)
    del _source_volume
    slots = []
    rendered_meta = {}
    centers_by_id = {}
    diagnostic_frames = []
    offset = 0
    side_note = dataset._native_and_geometry(uid, work)
    # _native_and_geometry performs the pinned image-side check and (if needed) a fresh native+geometry render.
    # Read the final study after that operation so all file/index bindings match its selected side.
    native_json = json.loads((Path(work) / "studies" / (hashlib.sha256(uid.encode()).hexdigest()[:24] + ".json")).read_bytes())
    native_by_id = _native_series_map(native_json)
    anchor = dataset.bank.record(uid, native_json, geo_dir=str(Path(work) / "geometry")) if dataset.bank else None
    planned, substitutions = slot_plan(public, uid, {uid: raw_records})
    selected_ids = [str(row["SeriesInstanceUID"]) for row, _ in planned if row is not None]
    geo_by_id = {}
    for sid in selected_ids:
        gp = Path(work) / "geometry" / (hashlib.sha256((uid + "/" + sid).encode()).hexdigest()[:24] + ".json")
        if gp.is_file(): geo_by_id[sid] = json.loads(gp.read_bytes())
    frame_reason = cache.frame_reference_fallback(geo_by_id, selected_ids, anchor)
    series_by_uid = {row["SeriesInstanceUID"]: row for row in raw_records}
    anchor_counts = Counter()
    for slot_index, ((row, count), (plane, fluid, expected_count)) in enumerate(zip(planned, cache.SLOTS)):
        if count != expected_count:
            raise ValueError("Dread slot plan count differs")
        if row is None:
            slots.append({"plane": plane, "fluid_sensitive": fluid, "count": count,
                          "requested_fluid_sensitive": fluid, "source_fluid_sensitive": None,
                          "source_fat_suppression": None,
                          "selected": False, "series_key": None, "frame_count": 0,
                          "selected_frame_indices": [], "native_frame_indices": []})
            offset += count
            continue
        sid = str(row["SeriesInstanceUID"])
        native = native_by_id.get(sid)
        if native is None:
            raise ValueError("Exact Dread-selected series missing from S6 native cache")
        raw_row = series_by_uid.get(sid)
        if raw_row is None or raw_row.get("Anatomical_Plane") != plane:
            raise ValueError("Selected Dread series metadata differs")
        raw_path = Path(data_root) / "train_series" / uid / sid
        raw_files = _dread_ordered_paths(public, raw_path)
        native_names = native["files"]
        mapping = cache.map_dread_files_to_native(raw_files, native_names)
        native_path = Path(native["path"])
        if not native_path.exists():
            native_path = Path(work) / "series" / Path(str(native["path"]).replace("\\", "/")).name
        native_volume = np.load(native_path, mmap_mode="r", allow_pickle=False)
        if native_volume.ndim != 3 or len(native_volume) != len(native_names) or len(native_volume) != len(native.get("positions", [])):
            raise ValueError("S6 native volume/files/positions alignment differs")
        center, path, fallback_reason = cache.local_anchor(anchor, sid, frame_fallback_reason=frame_reason)
        if plane == "Axial":
            entry = (anchor or {}).get("series", {}).get(sid, {})
            if frame_reason:
                center, path = (.5, .5), "field_center"
            elif entry.get("patella"):
                center = tuple(map(float, entry["patella"]))
            elif entry.get("joint"):
                # Exact accepted PX.series_views(v4, Axial) fallback: patella
                # center when segmented, otherwise joint shifted -40 mm anterior.
                import px_cache as PX
                center = PX.anchored(tuple(map(float, entry["joint"])),
                    native["spacing_mm"][0], native["native_shape"][-2:], -40., 0.)
            else:
                center, path, fallback_reason = (.5, .5), "field_center", "patella_anchor_unavailable"
        size = len(raw_files)
        selected = cache.selected_frame_indices(size, count)
        native_indices = mapping[selected]
        rendered = np.zeros((len(native_volume), cache.STORAGE, cache.STORAGE), dtype=np.uint8)
        crop_mm = cache.VIEWS[{"Sagittal": "sagittal", "Coronal": "coronal", "Axial": "axial"}[plane]]["crop_mm"]
        local_geometries = {}
        for native_index in sorted(set(map(int, native_indices.tolist()))):
            image, geometry = _render_one(native, native_index, center, crop_mm, view_geometry=view_geometry,
                normalize_intensity=normalize_intensity, select_display_window=ND.select_display_window,
                cv2=cv2, volume=native_volume)
            rendered[native_index] = image
            local_geometries[native_index] = geometry
        if montage_dir is not None:
            window = native.get("_display_window")
            chosen_native = list(map(int, native_indices.tolist()))
            for dread_frame, native_index in enumerate(chosen_native):
                source = normalize_intensity(native_volume[native_index].astype("float32"), window, "series_companded")
                source = np.rint(np.clip(source, 0, 1) * 255).astype("uint8")
                source = cv2.resize(source, (352, 352), interpolation=cv2.INTER_AREA)
                geom = local_geometries[native_index]
                diagnostic_frames.append({"plane": plane, "slot": slot_index, "dread_frame": int(selected[dread_frame]),
                    "native_index": native_index, "frame_key": hashlib.sha256(Path(raw_files[int(selected[dread_frame])]).name.encode()).hexdigest()[:12],
                    "crop_mm": crop_mm, "anchor_path": path, "source": source, "rendered": rendered[native_index],
                    "retained_native_rectangle": _retained_native_rectangle(geom), "native_shape": tuple(native_volume.shape[1:])})
        stack = rendered[native_indices]
        volume[offset:offset + count] = stack
        # Availability mask comes from Dread's unmodified prepare_study above.
        anchor_counts[path] += 1
        key = hashlib.sha256(sid.encode()).hexdigest()[:24]
        slots.append({"plane": plane, "fluid_sensitive": fluid, "count": count, "selected": True,
                      "requested_fluid_sensitive": fluid,
                      "source_fluid_sensitive": raw_row.get("Fluid_Sensitive"),
                      "source_fat_suppression": raw_row.get("Fat_Suppression"),
                      "series_key": key, "frame_count": len(raw_files),
                      "selected_frame_indices": selected.tolist(), "native_frame_indices": native_indices.tolist(),
                      "anchor_path": path, "fallback_reason": fallback_reason,
                      "series_frame_names_sha256": smoke.digest(smoke.canonical([Path(p).name for p in raw_files]))})
        rendered_meta[sid] = {"images": rendered, "dread_to_native": mapping,
                              "anchor_path": path, "fallback_reason": fallback_reason,
                              "side": native.get("side"), "geometry": local_geometries,
                              "native_indices": native_indices, "selected_indices": selected,
                              "frame_names": [Path(p).name for p in raw_files], "crop_mm": crop_mm}
        centers_by_id[sid] = center
        offset += count
        del native_volume, rendered
    joint_mask = np.zeros((cache.STORAGE, cache.STORAGE), dtype=np.uint8)
    mask_diagnostic = None
    mask_meta = {"schema": cache.MASK_SCHEMA, "available": False,
                 "reason": "no_selected_sagittal_series", "source_series_key": None,
                 "native_frame_index": None, "slot_index": None, "crop_mm": None}
    sag_rows = [(i, slot) for i, slot in enumerate(slots[:2]) if slot["selected"]]
    if sag_rows and anchor is not None:
        slot_index, sagittal_slot = sag_rows[0]
        sid = next(sid for sid in selected_ids if hashlib.sha256(sid.encode()).hexdigest()[:24] == sagittal_slot["series_key"])
        native = native_by_id[sid]
        geo_path = Path(work) / "geometry" / (hashlib.sha256((uid + "/" + sid).encode()).hexdigest()[:24] + ".json")
        geo = json.loads(geo_path.read_bytes())
        segs, segment_reason = dataset.bank.segment(native, geo)
        if segs:
            center_native = (len(native["files"]) - 1) / 2
            mask_index, probabilities = min(segs, key=lambda item: abs(item[0] - center_native))
            native_mask = (np.max(probabilities[1:4], axis=0) > .5).astype(np.uint8)
            seg_spacing = native["spacing_mm"][int(mask_index)]
            mask_geometry = view_geometry(native["native_shape"][-2:], seg_spacing, cache.RENDER_SIZE,
                                          crop_mm=cache.VIEWS["sagittal"]["crop_mm"],
                                          center=centers_by_id[sid])
            joint_mask = cache.align_binary_mask(native_mask, mask_geometry, cv2_module=cv2)
            selected_native = set(map(int, sagittal_slot["native_frame_indices"]))
            mapped_slot = slot_index if int(mask_index) in selected_native else None
            mask_meta = {"schema": cache.MASK_SCHEMA, "available": True,
                         "reason": None if mapped_slot is not None else "diagnostic_slice_not_in_selected_dread_frames",
                         "source_series_key": sagittal_slot["series_key"], "native_frame_index": int(mask_index),
                         "slot_index": mapped_slot, "crop_mm": 100.0,
                         "render_crop_xyxy": list(mask_geometry.crop_xyxy),
                         "valid_xyxy": list(mask_geometry.valid_xyxy), "center_crop_xyxy": [16, 16, 368, 368],
                         "segment_reason": segment_reason}
            mask_meta["selected_cache_frame_indices"] = [
                sum(s["count"] for s in slots[:slot_index]) + i
                for i, n in enumerate(sagittal_slot["native_frame_indices"]) if n == int(mask_index)]
            if montage_dir is not None:
                native_path = Path(native["path"])
                if not native_path.exists():
                    native_path = Path(work) / "series" / native_path.name
                native_volume = np.load(native_path, mmap_mode="r", allow_pickle=False)
                diagnostic, _ = _render_one(native, int(mask_index), centers_by_id[sid], 100.,
                    view_geometry=view_geometry, normalize_intensity=normalize_intensity,
                    select_display_window=ND.select_display_window, cv2=cv2, volume=native_volume)
                plain = cv2.cvtColor(diagnostic, cv2.COLOR_GRAY2BGR)
                overlay = plain.copy()
                overlay[joint_mask.astype(bool)] = (
                    .5 * overlay[joint_mask.astype(bool)] + np.array([0, 127.5, 0])).astype(np.uint8)
                destination = Path(montage_dir); destination.mkdir(parents=True, exist_ok=True)
                artifact = destination / "joint-mask-diagnostic.png"
                if not cv2.imwrite(str(artifact), np.concatenate([plain, overlay], axis=1)):
                    raise ValueError("Could not write joint-mask diagnostic")
                mask_diagnostic = {"file": artifact.name, "sha256": smoke.file_sha(artifact),
                    "bytes": artifact.stat().st_size, "frame_count": 0,
                    "role": "single_segmentation_diagnostic_not_dense_channels",
                    "native_frame_index": int(mask_index), "source_series_key": sagittal_slot["series_key"]}
                mask_meta["diagnostic"] = mask_diagnostic
                del native_volume
        else:
            mask_meta.update(reason="segmentation_unavailable:" + str(segment_reason),
                             source_series_key=sagittal_slot["series_key"])
    cache.validate_arrays(volume, mask, joint_mask)
    cache.validate_slot_mask(mask, slots)
    primary_sag_path = next((slot.get("anchor_path") for slot in slots[:2] if slot["selected"]), "field_center")
    bank_method = (anchor or {}).get("method")
    sagittal_entry = next((anchor.get("series", {}).get(sid) for sid in selected_ids
                           if any(slot.get("series_key") == hashlib.sha256(sid.encode()).hexdigest()[:24]
                                  for slot in slots[:2])), None) if anchor else None
    anchor_summary = {"path": primary_sag_path, "bank_method": bank_method,
                      "field_center_reasons": sorted({slot.get("fallback_reason") for slot in slots if slot.get("fallback_reason")}),
                      "axis_error": None, "sagittal_truncated": None,
                      "axial_patella_retained": None, "frame_reference_fallback": frame_reason,
                      "joint_lps_available": bool(anchor and anchor.get("joint_lps")),
                      "sagittal_anchor_in_field": None if not sagittal_entry else bool(sagittal_entry.get("in_field"))}
    axial_meta = next((rendered_meta[sid] for sid in selected_ids if sid in rendered_meta and
                       next((s.get("Anatomical_Plane") for s in raw_records if s["SeriesInstanceUID"] == sid), None) == "Axial"), None)
    axial_retained = None
    if axial_meta and anchor and anchor.get("series"):
        axial_sid = next(sid for sid in selected_ids if sid in rendered_meta and
                         next((s.get("Anatomical_Plane") for s in raw_records if s["SeriesInstanceUID"] == sid), None) == "Axial")
        pat = anchor["series"].get(axial_sid, {}).get("patella")
        g = next(iter(axial_meta["geometry"].values()), None)
        if g is not None and pat is not None and not axial_meta["fallback_reason"]:
            axial_retained = _point_in_retained(pat, g)
    anchor_summary["axial_patella_retained"] = axial_retained
    final_sides = {s.get("side") for s in native_json["series"]}
    final_side = next(iter(final_sides)) if len(final_sides) == 1 else None
    side = {"side": final_side, "original_side": None if not side_note else side_note.get("side"),
            "status": None if not side_note else side_note.get("status"),
            "new_side": None if not side_note else side_note.get("new_side"),
            "flipped": bool(side_note and side_note.get("flipped")),
            "score": None if not side_note else side_note.get("score"),
            "wrong_side": None}
    montage_receipt = _montage_pages(uid, diagnostic_frames, montage_dir, cv2=cv2) if montage_dir is not None else None
    if mask_diagnostic is not None:
        montage_receipt["pages"].append(mask_diagnostic)
    return volume, mask, joint_mask, slots, anchor_summary, side, mask_meta, {
        "selected_slots": sum(slot["selected"] for slot in slots),
        "missing_slots": sum(not slot["selected"] for slot in slots),
        "empty_selected_slots": sum(slot["selected"] and not mask[sum(s["count"] for s in slots[:i]):sum(s["count"] for s in slots[:i + 1])].any()
                                     for i, slot in enumerate(slots)),
        "side_flipped": int(side["flipped"]), "anchor_paths": dict(anchor_counts),
        "raw_frame_mapping_exact": True, "fluid_substitutions": int(substitutions),
        "source_prepare_study": source_counts, "montage": montage_receipt}


def _index_rows(root, doc, contract_sha, ordinals, ids):
    if doc.get("version") != cache.VERSION or doc.get("contract_sha256") != contract_sha:
        raise ValueError("Resume index contract differs")
    entries = doc.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Private index entries missing")
    seen = {}
    for row in entries:
        cache.validate_private_entry(row)
        n = row["ordinal"]
        if n not in ordinals or n in seen or row["StudyInstanceUID"] != ids[n] or row["file"] != cache.filename(n):
            raise ValueError("Private cache index membership/order differs")
        path = cache.safe_file(root, row["file"])
        if smoke.file_sha(path) != row.get("file_sha256"):
            raise ValueError("Resume cache file identity differs")
        volume, mask, joint = cache.read_npz(path)
        hashes = cache.array_hashes(volume, mask, joint)
        if any(hashes[k] != row.get(k) for k in hashes):
            raise ValueError("Private cache raw-array hashes differ")
        cache.validate_slot_mask(mask, row["slots"])
        row["verified_file_sha256"] = smoke.file_sha(path)
        seen[n] = row
    return seen


def validate_runtime_population(cfg):
    """Verify the same full identity contract for every shard, staging gold only on zero."""
    shard_index = cfg["shard_index"]
    if not 0 <= shard_index < 4 or cfg["shard_count"] != 4:
        raise ValueError("local64 uses exactly four training shards")
    ids = cfg["training_ids"]
    all_gold = cfg.get("all_gold_ids", [])
    if len(all_gold) != cache.GOLD_COUNT or len(set(all_gold)) != cache.GOLD_COUNT or set(all_gold) & set(ids):
        raise ValueError("Full gold identity list must be complete and disjoint")
    gold_ids = cfg.get("gold_ids", [])
    if len(ids) != cache.TRAIN_COUNT or len(gold_ids) not in (0, cache.GOLD_COUNT):
        raise ValueError("Runtime membership differs from pinned training/gold counts")
    contract = cfg["contract"]
    frozen_hashes = cfg.get("frozen32_uid_sha256", [])
    if (not isinstance(frozen_hashes, list) or len(frozen_hashes) != 32 or len(set(frozen_hashes)) != 32 or
            frozen_hashes != contract.get("frozen32", {}).get("uid_sha256")):
        raise ValueError("Frozen32 visual membership hash list differs from the cache contract")
    if (contract.get("version") != cache.VERSION or
            smoke.digest(smoke.canonical(ids)) != contract.get("training_membership_sha256") or
            smoke.digest(smoke.canonical(all_gold)) != contract.get("gold_membership_sha256")):
        raise ValueError("Runtime membership does not match the immutable cache contract")
    members = cache.make_shards(ids, 4)[shard_index]
    expected_part = contract.get("partitions", [])[shard_index]
    if (expected_part.get("studies") != len(members) or
            expected_part.get("ordinals_sha256") != smoke.digest(smoke.canonical(members)) or
            expected_part.get("ordered_ids_sha256") != smoke.digest(smoke.canonical([ids[n] for n in members]))):
        raise ValueError("Runtime shard partition differs from the immutable contract")
    if gold_ids != (all_gold if shard_index == 0 else []):
        raise ValueError("Only shard zero may cache the complete scoring-only gold population")
    return ids, gold_ids, shard_index, members


def make_runtime_config(*, contract, training_ids, all_gold_ids, **paths):
    """One producer for bootstrap/run identity fields; no model or image I/O."""
    shard = contract["shard_index"]
    cfg = dict(paths, contract=contract, training_ids=list(training_ids), all_gold_ids=list(all_gold_ids),
               gold_ids=list(all_gold_ids) if shard == 0 else [], shard_index=shard, shard_count=4,
               frozen32_uid_sha256=contract["frozen32"]["uid_sha256"])
    validate_runtime_population(cfg)
    return cfg


def _dread_source_inputs(public, uid, records, data_root):
    from knee.raptor_input import prepare_study
    return prepare_study(public, uid, records, str(Path(data_root) / "train_series"), cache_pixels=True)


def _dread_ordered_paths(public, series_root):
    """Pinned Dread order returns (filename, pixel-spacing) pairs, never bare paths."""
    order, _pixel, _resize = public._make_reader()
    rows, _median_spacing = order(str(series_root))
    if any(not isinstance(row, (tuple, list)) or len(row) != 2 or not isinstance(row[0], str) for row in rows):
        raise ValueError("Pinned Dread ordered-frame API differs")
    return [row[0] for row in rows]


def run(config):
    """Build one CPU training shard and, on shard0, an isolated 58-study gold cache."""
    import numpy as np
    import torch
    import cv2
    from knee.fresh_reader_setup import time_limit
    cfg = dict(config)
    if torch.cuda.is_available():
        raise ValueError("Explicit CPU-only cache runtime received a CUDA-visible device")
    torch.set_num_threads(1); cv2.setNumThreads(1)
    raw_root = Path(cfg["data_root"]); code = Path(cfg["code"]); out = Path(cfg["out"])
    ids, gold_ids, shard_index, members = validate_runtime_population(cfg)
    contract = cfg["contract"]
    frozen_hashes = cfg["frozen32_uid_sha256"]
    gold_members = list(range(len(gold_ids)))
    if out.is_symlink(): raise ValueError("Cache root symlink forbidden")
    out.mkdir(parents=True, exist_ok=True)
    contract_sha = cache.digest(contract)
    runfile = out / "run-contract.json"
    if runfile.exists():
        if json.loads(runfile.read_bytes()) != contract: raise ValueError("Run contract differs")
    elif any(out.iterdir()): raise ValueError("Unbound nonempty output path")
    else: cache.atomic_json(runfile, contract)
    indexfile = out / "index-private.json"
    if indexfile.exists(): entries = _index_rows(out, json.loads(indexfile.read_bytes()), contract_sha, members, ids)
    else: entries = {}
    counters = Counter(); started = time.monotonic()
    frozen_done = {hashlib.sha256(row["StudyInstanceUID"].encode()).hexdigest()
                   for row in entries.values() if row.get("k0_montage")}
    for row in entries.values():
        montage = row.get("k0_montage")
        if montage:
            if montage.get("frame_count") != sum(slot["count"] for slot in row["slots"] if slot["selected"]):
                raise ValueError("Resumed K0 montage frame coverage differs")
            for page in montage.get("pages", []):
                path = out / "k0-montage" / ("shard" + str(shard_index)) / montage["study_key"] / page["file"]
                if not path.is_file() or smoke.file_sha(path) != page.get("sha256"):
                    raise ValueError("Resumed K0 montage artifact hash differs")
    report = {"status": "RUNNING", "phase": "setup", "version": cache.VERSION,
              "contract_sha256": contract_sha, "shard_index": shard_index,
              "shard_count": 4, "completed_studies": len(entries), "requested_studies": len(members),
              "private_outputs": True, "aggregate_contains_ids": False,
              "environment": {"python": sys.version.split()[0], "device": "cpu", "torch_threads": torch.get_num_threads(),
                              "opencv_threads": cv2.getNumThreads(),
                              "packages": {p: importlib.metadata.version(p) for p in
                                  ("torch", "numpy", "pydicom", "opencv-python", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg")}},
              "counters": {}, "counters_scope": "current invocation; per-study source counts retained in private index"}
    cache.atomic_json(out / "receipt.json", report)
    try:
        free_disk = shutil.disk_usage(out).free
        report["minimum_free_disk_bytes"] = free_disk
        required_disk = cache.disk_required(len(members) - len(entries) + len(gold_ids))
        report["required_free_disk_upper_bound_bytes"] = required_disk
        if free_disk < required_disk:
            raise ValueError("Insufficient disk for the bounded remaining cache")
        from knee.reference import load_definitions
        public, decoder = load_definitions(cfg["dread_module"], "", cfg["dread_manifest"])
        if decoder is not None: raise ValueError("Unexpected Dread decoder payload")
        if (smoke.digest(cfg["dread_module"].encode()) != contract["reader"]["module_sha256"] or
                smoke.digest(smoke.canonical(cfg["dread_manifest"])) != contract["reader"]["manifest_sha256"] or
                cfg["dread_manifest"].get("source_sha256") != contract["reader"]["source_sha256"] or
                [list(x) for x in public.SLOTS] != [list(x) for x in cache.SLOTS] or
                public.IMG != 336 or public.MAXS != cache.DEPTH):
            raise ValueError("Dread renderer source/slot contract differs")
        expected_env = contract["qualified_environment"]
        actual_packages = {p: importlib.metadata.version(p) for p in expected_env["packages"]}
        if (sys.version.split()[0] != expected_env["python"] or expected_env.get("device") != "cpu" or
                actual_packages != expected_env["packages"]):
            raise ValueError("Qualified Python/CPU dependency environment differs")
        if (smoke.file_sha(cfg["anatomy_path"]) != contract["input_pins"]["anatomy_sha256"] or
                smoke.file_sha(cfg["side_model_path"]) != contract["input_pins"]["side_cls_sha256"]):
            raise ValueError("Pinned anatomy/side checkpoint differs")
        sys.path.insert(0, str(code)); os.environ["RSNA_CODE_ROOT"] = str(code)
        import compact_infer as CI
        active_ids = [ids[n] for n in members] + gold_ids
        data_root = out / "images-only"
        stage_receipt = _stage_image_only_inputs(raw_root, data_root, active_ids)
        metadata = _read_series(data_root / "train_series.csv", set(active_ids))
        dataset = CI.CStudies(str(data_root), "train", [], out / "scratch", str(code), 384, 1, "v4", 3,
                              cfg["anatomy_path"], 3.5, cfg["side_model_path"], 2.0, 0.0, 20.0)
        dataset._setup()
        torch.set_num_threads(1)
        if dataset.bank is None or dataset.bank.model is None or dataset.side_net is None:
            raise ValueError("Pinned CPU anchor and side models must both be initialized")
        for model in (dataset.bank.model, dataset.side_net):
            tensors = [*model.parameters(), *model.buffers()]
            if not tensors or any(tensor.device.type != "cpu" for tensor in tensors):
                raise ValueError("Pinned auxiliary model is absent or not on CPU")
        hooks = _attach_finite_hooks(dataset)
        state_before = _anchor_model_state(dataset)
        gold_root = out / "gold58" if gold_ids else None
        if gold_root is not None:
            gold_root.mkdir(exist_ok=True)
            gold_contract = {**contract, "population": "scoring-only-gold58", "training_membership_sha256": None,
                             "gold_ids_sha256": smoke.digest(smoke.canonical(gold_ids))}
            cache.atomic_json(gold_root / "run-contract.json", gold_contract)
            gold_sha = cache.digest(gold_contract)
            gold_index = gold_root / "index-private.json"
            gold_entries = _index_rows(gold_root, json.loads(gold_index.read_bytes()), gold_sha, gold_members, gold_ids) if gold_index.exists() else {}
        else: gold_entries = {}
        with time_limit(max(1, 12 * 3600 - (time.monotonic() - started)), "actual CPU platform12h limit"):
            todo = [(n, ids[n], out, entries, members, indexfile) for n in members if n not in entries]
            if gold_root is not None:
                todo.extend((n, gold_ids[n], gold_root, gold_entries, gold_members, gold_root / "index-private.json")
                            for n in gold_members if n not in gold_entries)
            for n, uid, target, dest_entries, ordinals, idx_path in todo:
                if n in dest_entries: continue
                work = out / "scratch" / ("s_" + hashlib.sha256(uid.encode()).hexdigest()[:16])
                work.parent.mkdir(parents=True, exist_ok=True)
                is_frozen32 = hashlib.sha256(uid.encode()).hexdigest() in set(frozen_hashes)
                montage_dir = (out / "k0-montage" / ("shard" + str(shard_index)) /
                               hashlib.sha256(uid.encode()).hexdigest()[:24]) if is_frozen32 else None
                try:
                    (volume, mask, joint, slots, anchor, side, joint_meta, study_counts) = _render_slots(
                        uid, data_root, work, dataset, public, metadata, code=code, montage_dir=montage_dir)
                finally:
                    CI.FORCED_SIDE.pop(uid, None)
                    shutil.rmtree(work, ignore_errors=True)
                file_receipt = cache.save_npz(cache.safe_file(target, cache.filename(n)), volume, mask, joint)
                row = cache.private_study_entry(n, uid, cache.filename(n), file_receipt, slots, anchor, side, joint_meta)
                # Keep only the first two sagittal slot paths attached to each selected series.
                row["slots"] = slots
                row["source_prepare_study"] = study_counts["source_prepare_study"]
                if is_frozen32:
                    row["k0_montage"] = study_counts["montage"]
                    frozen_done.add(hashlib.sha256(uid.encode()).hexdigest())
                cache.validate_private_entry(row)
                dest_entries[n] = row
                counters.update({k: v for k, v in study_counts.items() if type(v) is int})
                counters.update({"source_" + k: v for k, v in study_counts["source_prepare_study"].items()
                                 if type(v) is int})
                cache.atomic_json(idx_path, {"version": cache.VERSION, "contract_sha256":
                    cache.digest(contract if target == out else gold_contract),
                    "entries": [dest_entries[x] for x in ordinals if x in dest_entries]})
                report.update(status="RUNNING", phase="decode", completed_studies=len(entries),
                              elapsed_seconds=time.monotonic() - started, counters=dict(counters),
                              private_index_sha256=smoke.file_sha(indexfile),
                              minimum_free_disk_bytes=min(report["minimum_free_disk_bytes"], shutil.disk_usage(out).free),
                              peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) *
                                  (1 if sys.platform == "darwin" else 1024))
                cache.atomic_json(out / "receipt.json", report)
                del volume, mask, joint
            if _anchor_model_state(dataset) != state_before:
                raise ValueError("CPU auxiliary model state changed during cache render")
            for hook in hooks: hook.remove()
        if len(entries) != len(members): raise ValueError("Training shard coverage incomplete")
        expected_frozen = contract["frozen32"]["counts_by_shard"][shard_index]
        if len(frozen_done) != expected_frozen:
            raise ValueError("Frozen32 montage coverage differs from the pinned partition")
        if gold_root is not None:
            if len(gold_entries) != cache.GOLD_COUNT: raise ValueError("Scoring-only gold coverage incomplete")
            cache.atomic_json(gold_root / "receipt.json", {
                "status": "PASS", "population": "scoring-only-gold58", "completed_studies": len(gold_entries),
                "contract_sha256": cache.digest(gold_contract),
                "private_index_sha256": smoke.file_sha(gold_root / "index-private.json"),
                "training_promotion_allowed": False, "aggregate_contains_ids": False})
        report.update(status="PASS", phase="complete", completed_studies=len(entries),
                      training_shard_count=len(members), gold_studies=len(gold_entries),
                      image_only_staging=stage_receipt,
                      elapsed_seconds=time.monotonic() - started, counters=dict(counters),
                      private_index_sha256=smoke.file_sha(indexfile),
                      model_state_before=state_before, model_state_after=_anchor_model_state(dataset),
                      k0={"status": "UNKNOWN", "montage_review": "PENDING", "anatomy_pass_claimed": False,
                          "frozen32_studies_montaged": len(frozen_done),
                          "montage_pages": sum(len(row["k0_montage"]["pages"]) for row in entries.values() if "k0_montage" in row)},
                      minimum_free_disk_bytes=min(report["minimum_free_disk_bytes"], shutil.disk_usage(out).free),
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) *
                          (1 if sys.platform == "darwin" else 1024))
        cache.atomic_json(out / "receipt.json", report)
        return report
    except BaseException as error:
        cache.atomic_json(out / "error-private.json", {
            "failure_type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "private": True})
        report.update(status="ERROR", failure_type=type(error).__name__, elapsed_seconds=time.monotonic() - started,
                      counters=dict(counters), k0={"status": "UNKNOWN", "anatomy_pass_claimed": False})
        cache.atomic_json(out / "receipt.json", report)
        raise RuntimeError("local64 cache failed; preserve all private outputs") from None
