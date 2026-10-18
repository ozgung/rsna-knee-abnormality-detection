import tempfile
import unittest
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_CV2 = importlib.util.find_spec("cv2") is not None
if HAVE_NUMPY:
    import numpy as np

from knee import local64_cache as cache
from knee import local64_loader as loader
from knee import local64_runtime as runtime


@unittest.skipUnless(HAVE_NUMPY, "NumPy required")
class Local64ContractTests(unittest.TestCase):
    def test_historical_partition_and_gold_are_disjoint(self):
        train = [f"train-{i:04d}" for i in range(cache.TRAIN_COUNT)]
        gold = [f"gold-{i:03d}" for i in range(cache.GOLD_COUNT)]
        shards, scoring = cache.partition_population(train, gold)
        self.assertEqual([len(s) for s in shards], [1087, 1086, 1086, 1086])
        self.assertEqual(sorted(x for shard in shards for x in shard), train)
        self.assertTrue(all(shard == [x for x in train if x in set(shard)] for shard in shards))
        self.assertEqual(scoring, gold)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            cache.partition_population(train, [*gold[:-1], train[0]])

    def test_slot_order_frame_selection_and_missing_masks(self):
        expected = [18, 14, 12, 8, 12]
        self.assertEqual([slot[2] for slot in cache.SLOTS], expected)
        self.assertEqual(cache.selected_frame_indices(1, 18).tolist(), [0] * 18)
        picks = cache.selected_frame_indices(100, 18)
        self.assertEqual(int(picks[0]), 6)
        self.assertEqual(int(picks[-1]), 93)
        rows = []
        for plane, fluid, count in cache.SLOTS:
            rows.append({"plane": plane, "fluid_sensitive": fluid, "count": count,
                         "selected": False, "series_key": None, "frame_count": 0})
        v, m, result = cache.compose_slots(rows, {})
        self.assertEqual(v.shape, (64, 352, 352))
        self.assertEqual(int(m.sum()), 0)
        self.assertTrue(all(not r["selected"] for r in result))
        with self.assertRaisesRegex(ValueError, "Entirely empty"):
            cache.validate_arrays(v, m, np.zeros((352, 352), np.uint8))

    def test_native_frame_mapping_is_exact_and_orientation_is_not_reflipped(self):
        mapped = cache.map_dread_files_to_native(
            ["s/a.dcm", "s/b.dcm", "s/c.dcm"], ["c.dcm", "a.dcm", "b.dcm"])
        self.assertEqual(mapped.tolist(), [1, 2, 0])
        with self.assertRaisesRegex(ValueError, "membership differs"):
            cache.map_dread_files_to_native(["a.dcm", "missing.dcm"], ["a.dcm", "b.dcm"])
        record = {"method": "skm_tea_joint", "series": {"sid": {"joint": [.4, .7], "in_field": True}}}
        # Left/right canonicalization remains owned by the pinned renderer. The adapter
        # preserves its already-renderer-oriented coordinates without an extra flip.
        self.assertEqual(cache.local_anchor(record, "sid"), ((.4, .7), "learned", None))
        center, path, reason = cache.local_anchor(record, "sid", frame_fallback_reason="incompatible_frame_of_reference")
        self.assertEqual((center, path, reason), ((.5, .5), "field_center", "incompatible_frame_of_reference"))

    def test_center_crop_and_mask_alignment(self):
        image = np.arange(384 * 384, dtype=np.uint32).reshape(384, 384)
        crop = cache.center_crop_rendered(image)
        self.assertEqual(crop.shape, (352, 352))
        self.assertEqual(int(crop[0, 0]), int(image[16, 16]))
        self.assertEqual(int(crop[-1, -1]), int(image[367, 367]))

        class CV2:
            INTER_NEAREST = 0

            @staticmethod
            def resize(array, size, interpolation):
                h, w = size[1], size[0]
                return array[np.minimum((np.arange(h) * array.shape[0] / h).astype(int), array.shape[0] - 1)[:, None],
                             np.minimum((np.arange(w) * array.shape[1] / w).astype(int), array.shape[1] - 1)[None, :]]

        geometry = type("Geometry", (), {"native_shape": (8, 8), "crop_xyxy": (0, 0, 8, 8),
                           "valid_xyxy": (0, 0, 384, 384), "resized_shape": (384, 384)})()
        mask = np.zeros((8, 8), np.uint8)
        mask[2:6, 3:7] = 1
        aligned = cache.align_binary_mask(mask, geometry, cv2_module=CV2)
        self.assertEqual(aligned.shape, (352, 352))
        self.assertTrue(aligned.any())
        self.assertEqual(set(np.unique(aligned)), {0, 1})

    def test_npz_schema_privacy_and_k0_never_falsely_passes(self):
        volume = np.zeros((64, 352, 352), np.uint8)
        mask = np.zeros((64,), np.uint8)
        joint = np.zeros((352, 352), np.uint8)
        volume[0, 20, 20] = 1
        mask[0] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.npz"
            receipt = cache.save_npz(path, volume, mask, joint)
            self.assertEqual(cache.array_hashes(*cache.read_npz(path)),
                             {k: receipt[k] for k in ("volume_sha256", "mask_sha256", "joint_mask_sha256")})
            with np.load(path, allow_pickle=False) as z:
                self.assertEqual(set(z.files), {"volume", "mask", "joint_mask"})

        slots = [{"plane": p, "fluid_sensitive": f, "count": n, "selected": False,
                  "series_key": None, "frame_count": 0} for p, f, n in cache.SLOTS]
        entries = []
        for i in range(32):
            study_slots = [{**slot, "selected": True, "series_key": f"{i:024x}",
                            "frame_count": 10, "anchor_path": "field_center",
                            "requested_fluid_sensitive": slot["fluid_sensitive"],
                            "source_fluid_sensitive": slot["fluid_sensitive"],
                            "source_fat_suppression": "synthetic"} if j == 0 else slot
                           for j, slot in enumerate(slots)]
            entries.append({"ordinal": i, "StudyInstanceUID": f"private-{i}", "file": cache.filename(i),
                "slots": study_slots, "anchor": {"path": "field_center", "bank_method": "skm_tea_joint",
                                                   "axial_patella_retained": True},
                "side": {"flipped": False, "wrong_side": False},
                "joint_mask": {"schema": cache.MASK_SCHEMA}})
        aggregate = cache.k0_aggregate(entries, visual_entries=entries)
        self.assertEqual(aggregate["status"], "UNKNOWN")
        self.assertFalse(aggregate["anatomy_pass_claimed"])
        reviewed = {"status": "REVIEWED", "reviewer": "synthetic"}
        aggregate = cache.k0_aggregate(entries, visual_entries=entries, montage_review=reviewed)
        self.assertEqual(aggregate["status"], "UNKNOWN")  # review lacks exact evidence bindings/date
        self.assertFalse(aggregate["full_cache_bounds"]["effective_learned_anchor_studies_ge_60pct"])
        self.assertEqual(aggregate["auxiliary_model_path_studies"], 32)
        self.assertEqual(aggregate["sagittal_field_center_fallbacks"], 32)

    def test_k0_full_cache_study_rate_and_frozen32_visual_bounds_are_separate(self):
        full, visual = [], []
        for i in range(100):
            slots = [{"plane": p, "fluid_sensitive": f, "count": n,
                      "selected": j == 0, "series_key": f"{i:024x}" if j == 0 else None,
                      "frame_count": 10 if j == 0 else 0,
                      "anchor_path": "learned" if i < 70 else "field_center",
                      "requested_fluid_sensitive": f,
                      "source_fluid_sensitive": f,
                      "source_fat_suppression": "synthetic"} if j == 0 else {
                          "plane": p, "fluid_sensitive": f, "count": n,
                          "selected": False, "series_key": None, "frame_count": 0
                      } for j, (p, f, n) in enumerate(cache.SLOTS)]
            row = {"ordinal": i, "StudyInstanceUID": f"private-{i}", "file": cache.filename(i),
                   "slots": slots, "anchor": {"path": "learned" if i < 70 else "field_center",
                       "bank_method": "skm_tea_joint", "sagittal_truncated": False,
                       "axis_error": False, "axial_patella_retained": None},
                   "side": {"flipped": False, "wrong_side": False},
                   "joint_mask": {"schema": cache.MASK_SCHEMA}}
            full.append(row)
            if i < 32: visual.append(row)
        binding = {"source_bundle_sha256": "a" * 64, "asset_sha256": "b" * 64,
                   "contract_sha256": "c" * 64, "montage_sha256": "d" * 64}
        review = {"status": "REVIEWED", "reviewer": "reviewer", "date": "2026-10-15",
                  "unknowns": [], **binding}
        result = cache.k0_aggregate(full, visual_entries=visual, montage_review=review,
                                    expected_binding=binding)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["effective_learned_anchor_studies"], 70)
        self.assertEqual(result["frozen32_visual_studies"], 32)
        self.assertTrue(result["frozen32_visual_bounds"]["axial_patella_retained_when_selected"])

    def test_dark_local_crop_does_not_change_authoritative_availability_mask(self):
        slots = [{"plane": plane, "fluid_sensitive": f, "count": n,
                  "selected": i == 0, "series_key": "a" * 24 if i == 0 else None,
                  "frame_count": 1 if i == 0 else 0}
                 for i, (plane, f, n) in enumerate(cache.SLOTS)]
        dark = np.zeros((1, 352, 352), np.uint8)
        volume, mask, _ = cache.compose_slots(slots, {
            "a" * 24: {"images": dark, "dread_to_native": np.array([0])}})
        self.assertEqual(int(mask.sum()), 18)
        self.assertFalse(volume.any())
        with self.assertRaisesRegex(ValueError, "no usable pixels"):
            cache.validate_arrays(volume, mask, np.zeros((352, 352), np.uint8))
        volume[0, 20, 20] = 1
        self.assertTrue(cache.validate_arrays(volume, mask, np.zeros((352, 352), np.uint8)))

    @unittest.skipUnless(HAVE_TORCH, "PyTorch required")
    def test_local_loader_hash_binding_and_pinned_384_window_contract(self):
        volume = np.zeros((64, 352, 352), np.uint8)
        mask = np.zeros(64, np.uint8); mask[0] = 1
        volume[0, 20, 20] = 1
        joint = np.zeros((352, 352), np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / cache.filename(0)
            receipt = cache.save_npz(file, volume, mask, joint)
            entry = {"ordinal": 0, "file": cache.filename(0), **receipt}
            proxy = loader.Local64Proxy(root, [entry])
            loaded = proxy[0]
            self.assertEqual(loaded[0].shape, (64, 352, 352))
            self.assertFalse(loaded[0].flags.writeable)
            class Public:
                def eval_windows(self, got_volume, got_mask, k, resolution):
                    self.asserted = (got_volume.shape, got_mask.shape, k, resolution)
                    import torch
                    return torch.zeros((k, 3, resolution, resolution))
            public = Public()
            windows = loader.eval_windows(public, *loaded[:2], k=42, resolution=384)
            self.assertEqual(tuple(windows.shape), (42, 3, 384, 384))
            self.assertEqual(public.asserted, ((64, 352, 352), (64,), 42, 384))
            with self.assertRaisesRegex(ValueError, "pinned to 384"):
                loader.eval_windows(public, *loaded[:2], resolution=352)

    @unittest.skipUnless(HAVE_CV2, "OpenCV required")
    def test_render_slots_binds_dread_frames_to_native_pixels_and_keeps_source_mask(self):
        import hashlib
        import json
        import numpy as np

        uid, sid = "synthetic-study", "synthetic-series"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_root = root / "raw"
            series_dir = raw_root / "train_series" / uid / sid
            series_dir.mkdir(parents=True)
            raw_names = [f"frame-{i:02d}.dcm" for i in range(30)]
            for name in raw_names:
                (series_dir / name).write_bytes(b"synthetic")
            work = root / "work"
            (work / "studies").mkdir(parents=True)
            (work / "geometry").mkdir()
            (work / "series").mkdir()
            native_file = work / "series" / "native.npy"
            native_volume = np.zeros((30, 48, 64), dtype=np.uint16)
            for i in range(30):
                native_volume[i, 5:43, 8:56] = 100 + i
            np.save(native_file, native_volume)
            native_names = list(reversed(raw_names))
            native = {"SeriesInstanceUID": sid, "files": native_names,
                      "positions": list(range(30)), "spacing_mm": [[1.0, 1.0]] * 30,
                      "native_shape": [30, 48, 64], "path": "/missing/native.npy",
                      "side": "left"}
            geometry_path = work / "geometry" / hashlib.sha256((uid + "/" + sid).encode()).hexdigest()[:24]
            geometry_path = geometry_path.with_suffix(".json")
            geometry_path.write_text(json.dumps({"frame_of_reference_uid": "for-synthetic"}))

            native_inventory = [native]
            def side_check(got_uid, got_work):
                study = {"series": native_inventory}
                path = Path(got_work) / "studies" / (hashlib.sha256(got_uid.encode()).hexdigest()[:24] + ".json")
                path.write_text(json.dumps(study))
                return {"side": "right", "new_side": "left", "status": "corrected", "flipped": True}

            probabilities = np.ones((3, 48, 64), dtype=np.float32)
            bank = SimpleNamespace(
                record=lambda *_args, **_kwargs: {"method": "skm_tea_joint_v4", "joint_lps": [0, 0, 0],
                    "series": {sid: {"joint": [.5, .5], "in_field": True}}},
                segment=lambda *_args, **_kwargs: ([(2, probabilities)], None))
            dataset = SimpleNamespace(_native_and_geometry=side_check, bank=bank)

            class Public:
                IMG = 336
                MAXS = 64
                SLOTS = cache.SLOTS
                @staticmethod
                def _make_reader():
                    def order(path):
                        return [(str(p), 1.0) for p in sorted(Path(path).glob("*.dcm"))], 1.0
                    return order, None, None

            planned = [({"SeriesInstanceUID": sid}, 18), *((None, count) for _, _, count in cache.SLOTS[1:])]
            training = ModuleType("training")
            training.__path__ = []
            nd = ModuleType("training.native_data")
            nd.select_display_window = lambda *_args: (0.0, 255.0)
            provenance = ModuleType("training.input_provenance")
            def view_geometry(native_shape, spacing, size, *, crop_mm, center):
                return SimpleNamespace(native_shape=tuple(native_shape), crop_xyxy=(0, 0, native_shape[1], native_shape[0]),
                    valid_xyxy=(0, 0, size, size), resized_shape=(size, size), size=size)
            provenance.view_geometry = view_geometry
            intensity = ModuleType("training.intensity_mapping")
            intensity.normalize_intensity = lambda pixels, window, _mode: np.clip(
                (pixels - window[0]) / (window[1] - window[0]), 0, 1)
            raptor_input = ModuleType("knee.raptor_input")
            px_cache = ModuleType("px_cache")
            source_volume = np.ones((64, 336, 336), dtype=np.uint8)
            source_mask = np.zeros(64, dtype=np.uint8); source_mask[:18] = 1
            raptor_input.prepare_study = lambda *_args, **_kwargs: (source_volume, source_mask, {"source": "pinned-mask"})
            with patch.dict("sys.modules", {"training": training, "training.native_data": nd,
                "training.input_provenance": provenance, "training.intensity_mapping": intensity,
                "knee.raptor_input": raptor_input, "px_cache": px_cache}), patch.object(runtime, "slot_plan", return_value=(planned, 1)):
                metadata = {uid: [{"SeriesInstanceUID": sid, "Anatomical_Plane": "Sagittal",
                    "Fluid_Sensitive": "1", "Fat_Suppression": "FS"}]}
                volume, mask, joint, slots, anchor, side, joint_meta, _counts = runtime._render_slots(
                    uid, raw_root, work, dataset, Public, metadata, code=root, montage_dir=root / "montage")
                # An unselected native series may supply the bank's joint/slab prior.
                extra_sid = sid + '-unselected-anchor-source'
                native_inventory.append({**native, "SeriesInstanceUID": extra_sid})
                extra_geo = work / "geometry" / (hashlib.sha256((uid + "/" + extra_sid).encode()).hexdigest()[:24] + ".json")
                for reference, reason in [("unregistered-frame", "incompatible_frame_of_reference"),
                                           (None, "frame_of_reference_missing")]:
                    extra_geo.write_text(json.dumps({"frame_of_reference_uid": reference}))
                    blocked = runtime._render_slots(uid, raw_root, work, dataset, Public, metadata, code=root)
                    self.assertEqual(blocked[4]["path"], "field_center")
                    self.assertEqual(blocked[4]["frame_reference_fallback"], reason)
                    self.assertEqual(blocked[4]["frame_reference_series_count"], 2)
                    self.assertEqual(blocked[4]["frame_reference_scope"], "all_native_anchor_sources_and_targets")

            expected_raw_picks = cache.selected_frame_indices(30, 18)
            expected_native_picks = np.asarray([native_names.index(raw_names[i]) for i in expected_raw_picks])
            self.assertEqual(volume.shape, (64, 352, 352))
            self.assertEqual(mask.tolist(), source_mask.tolist())
            self.assertTrue(volume[:18].any())
            self.assertEqual(slots[0]["selected_frame_indices"], expected_raw_picks.tolist())
            self.assertEqual(slots[0]["native_frame_indices"], expected_native_picks.tolist())
            self.assertEqual(slots[0]["requested_fluid_sensitive"], 1)
            self.assertEqual(slots[0]["source_fluid_sensitive"], "1")
            self.assertEqual(slots[0]["source_fat_suppression"], "FS")
            self.assertEqual(anchor["path"], "learned")
            self.assertEqual(side["side"], "left")
            self.assertEqual(side["original_side"], "right")
            self.assertTrue(side["flipped"])
            self.assertTrue(joint.any())
            self.assertEqual(joint_meta["schema"], cache.MASK_SCHEMA)
            self.assertEqual(_counts["montage"]["frame_count"], 18)
            self.assertEqual(sum(p["frame_count"] for p in _counts["montage"]["pages"]), 18)
            self.assertEqual(len(_counts["montage"]["pages"]), 4)  # three frame pages plus one diagnostic
            self.assertEqual(joint_meta["diagnostic"]["role"], "single_segmentation_diagnostic_not_dense_channels")
            for page in _counts["montage"]["pages"]:
                from knee.public_training_smoke import file_sha
                self.assertEqual(file_sha(root / "montage" / page["file"]), page["sha256"])


if __name__ == "__main__":
    unittest.main()
