import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from knee.reference import (CONSTANTS, DEFINITIONS, RECIPE, audit_source, bounded_ids,
                            install_decoder_guards, load_definitions, slot_plan,
                            validate_predictions, validate_volume)


class ReferenceContractTests(unittest.TestCase):
    def test_sampling_is_id_only_stable_bounded_and_order_independent(self):
        ids = [f"synthetic-{i}" for i in range(50)]
        chosen = bounded_ids(ids)
        self.assertEqual(len(chosen), 24)
        self.assertEqual(chosen, bounded_ids(list(reversed(ids))))
        self.assertNotEqual(chosen, bounded_ids(ids, seed=43))
        for values, count in [(["a", "a"], 1), (["../private"], 1), (ids, 65), (ids, True)]:
            with self.subTest(values=values, count=count), self.assertRaises(ValueError):
                bounded_ids(values, count=count)

    def test_unreviewed_source_is_rejected_before_execution(self):
        with self.assertRaisesRegex(ValueError, "audited snapshot"):
            audit_source("raise RuntimeError('must never execute')")

    def test_decoder_guard_prevents_silent_failed_headers_and_pixels(self):
        class Context:
            def px(self, data, info):
                return (None, None) if data is None else (data, None)
        decoder = SimpleNamespace(_records=lambda rows: rows,
                                  _result=lambda selected, got, spacing: got,
                                  _Ctx=Context,
                                  _make_pyd=lambda: (lambda *args: (0, "unused", 1, True), lambda data: data))
        counts = install_decoder_guards(decoder)
        with self.assertRaisesRegex(ValueError, "header failure"):
            decoder._records([(0, "unused", 1, False)])
        with self.assertRaisesRegex(ValueError, "pixel failure"):
            decoder._result([], {0: (None, None)}, 1)
        with self.assertRaisesRegex(ValueError, "swallowed"):
            Context().px(None, None)
        self.assertEqual(Context().px(b"pixels", None), (b"pixels", None))
        _, px = decoder._make_pyd()
        self.assertEqual(px(b"pixels"), b"pixels")
        self.assertEqual(counts["decoded_unique_slices"], 1)
        self.assertEqual(counts["pydicom_pixel_paths"], 1)

    def test_slot_plan_records_substitution_and_preserves_absent_slots(self):
        def choose(rows, plane, fluid, used):
            candidates = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
            return next((r for r in candidates if int(r["Fluid_Sensitive"]) == fluid), candidates[0] if candidates else None)
        public = SimpleNamespace(SLOTS=[("Sagittal", 1, 2), ("Sagittal", 0, 2), ("Axial", -1, 2)],
                                 _pick_series_for_slot=choose)
        rows = [{"SeriesInstanceUID": f"s{i}", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": "1"} for i in range(2)]
        plan, substitutions = slot_plan(public, "study", {"study": rows})
        self.assertEqual(substitutions, 1)
        self.assertEqual([r["SeriesInstanceUID"] if r else None for r, _ in plan], ["s0", "s1", None])

    def test_builder_contract_is_private_offline_attributed_and_no_extra_models(self):
        import importlib.util
        import sys
        directory = Path(__file__).resolve().parents[1] / "scripts"
        with patch.object(sys, "path", [str(directory), *sys.path]):
            spec = importlib.util.spec_from_file_location("reference_builder", directory / "build_reference_rehearsal.py")
            builder = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(builder)
        manifest = {"attribution": "Synthetic source author", "source_url": "https://example.com/reference",
                    "weight_dataset": builder.WEIGHT_DATASET, "checkpoint_file":"synthetic.pt", "checkpoint_sha256":"a"*64}
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.py"
            source.write_text("# synthetic input")
            destination = Path(temp) / "build"
            publisher = Path(temp) / "checkpoint-manifest.json"
            publisher.write_text(json.dumps({"accuracy":{"filename":"synthetic.pt","sha256":"a"*64}}))
            with patch.object(builder, "audit_source", return_value=("# audited definitions\n", "# audited decoder\n", manifest)):
                receipt = builder.build_reference(source, destination, "reference-test", publisher_manifest=publisher)
            metadata = json.loads((destination / "kernel-metadata.json").read_text())
            self.assertTrue(metadata["is_private"])
            self.assertFalse(metadata["enable_internet"])
            self.assertEqual(metadata["model_sources"], [])
            self.assertEqual(metadata["dataset_sources"], [builder.WEIGHT_DATASET, builder.CONVNEXT_DATASET])
            notebook = json.loads((destination / "notebook.ipynb").read_text())
            self.assertIn("Synthetic source author", "".join(notebook["cells"][0]["source"]))
            compile("".join(notebook["cells"][1]["source"]), "<generated>", "exec")
            self.assertLess(receipt["notebook_bytes"], 900000)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "optional numpy unavailable")
class ReferenceArrayTests(unittest.TestCase):
    def test_missing_slots_are_allowed_but_decode_holes_and_empty_studies_are_not(self):
        import numpy as np
        plan = [({"SeriesInstanceUID": "synthetic"}, 2), (None, 2)]
        volume = np.zeros((4, 3, 3), np.uint8)
        volume[:2] = 1
        mask = np.array([1, 1, 0, 0], np.uint8)
        self.assertEqual(validate_volume(volume, mask, plan, image_size=3), {"valid_slices": 2, "missing_slots": 1})
        volume[1] = 0
        mask[1] = 0
        with self.assertRaisesRegex(ValueError, "unexpectedly zero-filled"):
            validate_volume(volume, mask, plan, image_size=3)
        with self.assertRaisesRegex(ValueError, "Empty study"):
            validate_volume(volume * 0, mask * 0, plan, image_size=3)

    def test_output_guards_reject_nonfinite_wrong_shape_and_prior_fallbacks(self):
        import numpy as np
        good = np.linspace(.05, .95, 12)
        np.testing.assert_array_equal(validate_predictions(good), good)
        for bad in (np.ones(12) * .5, np.zeros(12), np.ones(12), np.ones(11),
                    np.full(12, np.nan), np.full(12, 1.1), np.full(12, -.1)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                validate_predictions(bad)


@unittest.skipUnless(os.environ.get("KNEE_REFERENCE_SOURCE") and importlib.util.find_spec("torch"),
                     "requires optional audited external source and torch")
class ReferenceSnapshotTests(unittest.TestCase):
    def test_exact_snapshot_extraction_and_window_tta_parity(self):
        import numpy as np
        import torch
        torch.set_num_threads(2)
        source = Path(os.environ["KNEE_REFERENCE_SOURCE"]).read_text()
        extracted, decoder, manifest = audit_source(source)
        self.assertNotIn("_FASTREAD_B64", extracted)
        self.assertNotIn("def main(", extracted)
        self.assertNotIn("weights_only=False", extracted)
        public, _ = load_definitions(extracted, decoder, manifest)
        self.assertEqual(public.INPUT_RES, RECIPE["input_res"])
        self.assertEqual(public.K_EVAL, RECIPE["windows"])
        # Distinct depth/horizontal values detect the two anatomical mirror axes.
        volume = (np.arange(96, dtype=np.uint8)[:, None, None] +
                  np.arange(384, dtype=np.uint16)[None, None, :] % 128).astype(np.uint8)
        volume = np.broadcast_to(volume, (96, 384, 384)).copy()
        mask = np.ones(96, np.uint8)
        for mirrored in (False, True):
            cpu = public.eval_windows(volume, mask, 94, 320, mirror=mirrored)
            lut = public.gpu_windows(volume, mask, 94, 320, "cpu", mirror=mirrored)
            self.assertTrue(torch.equal(cpu, lut))
            del cpu, lut
        sample = torch.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
        mirrored = public._mirror_tri(sample, [1, 50], 96)
        self.assertTrue(torch.equal(mirrored[0], sample[0].flip(0)))
        self.assertTrue(torch.equal(mirrored[1], sample[1].flip(-1)))


@unittest.skipUnless(os.environ.get("KNEE_RAPTOR_SOURCE") and importlib.util.find_spec("torch"),
                     "requires optional audited Raptor source and torch")
class RaptorSnapshotTests(unittest.TestCase):
    def test_62_windows_is_the_only_change_and_requires_matching_manifest(self):
        import copy
        import numpy as np
        import torch
        torch.set_num_threads(2)
        source = Path(os.environ["KNEE_RAPTOR_SOURCE"]).read_text()
        old_source, decoder, old = audit_source(source, profile="raptor", checkpoint_sha256="a" * 64)
        new_source, _, new = audit_source(source, profile="raptor", checkpoint_sha256="a" * 64, raptor_windows=62)
        self.assertEqual(old_source, new_source)
        expected = copy.deepcopy(old)
        expected["recipe"]["windows"] = 62
        self.assertEqual(new, expected)
        before, _ = load_definitions(old_source, decoder, old)
        after, _ = load_definitions(new_source, decoder, new)
        self.assertEqual(before.K_EVAL, 42)
        self.assertEqual(after.K_EVAL, 62)
        volume = np.broadcast_to(np.arange(64, dtype=np.uint8)[:, None, None], (64, 8, 8)).copy()
        mask = np.ones(64, np.uint8)
        self.assertEqual(after.eval_windows(volume, mask, after.K_EVAL, 8).shape, (62, 3, 8, 8))
        # The variant must retain exact existing arithmetic for an explicit 42 request.
        self.assertTrue(torch.equal(before.eval_windows(volume, mask, 42, 8), after.eval_windows(volume, mask, 42, 8)))
        bad = copy.deepcopy(new)
        bad["recipe"]["slice_span"] = [0.02, 0.98]
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            load_definitions(new_source, decoder, bad)
        for count in (43, 64, True, 62.0):
            with self.assertRaisesRegex(ValueError, "42/62"):
                audit_source(source, profile="raptor", checkpoint_sha256="a" * 64, raptor_windows=count)

    def test_non_oai_profile_requires_checkpoint_pin_and_excludes_fp32_fallback(self):
        import numpy as np
        import torch
        from knee.reference import RAPTOR_RECIPE, raptor_gpu_windows
        source = Path(os.environ["KNEE_RAPTOR_SOURCE"]).read_text()
        with self.assertRaisesRegex(ValueError, "checkpoint SHA256"):
            audit_source(source, profile="raptor")
        extracted, decoder, manifest = audit_source(source, profile="raptor", checkpoint_sha256="a" * 64)
        self.assertEqual(decoder, "")
        self.assertNotIn("def infer_probs", extracted)
        self.assertNotIn("torch.load", extracted)
        self.assertEqual(manifest["recipe"], RAPTOR_RECIPE)
        public, fast = load_definitions(extracted, decoder, manifest)
        self.assertIsNone(fast)
        self.assertEqual(public.MAXS, 64)
        self.assertEqual(public.K_EVAL, 42)
        rng = np.random.default_rng(42)
        volume = rng.integers(0, 256, (64, 336, 336), dtype=np.uint8)
        mask = np.ones(64, np.uint8)
        expected = public.eval_windows(volume, mask, 42, 384)
        actual = raptor_gpu_windows(public, volume, mask, "cpu")
        self.assertLessEqual(float((expected - actual).abs().max()), 3e-5)
        self.assertEqual(expected.shape, (42, 3, 384, 384))


if __name__ == "__main__":
    unittest.main()
