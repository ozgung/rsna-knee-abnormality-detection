import copy
from contextlib import ExitStack
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from knee.reference import canonical_bytes, sha256
from knee.reference_blend import (CODEC_WHEELS, CONVNEXT_ENTRY_SHA256, CONVNEXT_SOURCE_HASHES,
    RAPTOR_PLAIN_SHA256, RAPTOR_WEIGHT_DATASET, RAPTOR_WEIGHT_NAME, RECIPE,
    equal_rank_blend, extract_reader, qualify_components, read_test_contract,
    run_blend, select_codec_wheels, write_probabilities)


def component_receipts():
    manifest = {"profile": "raptor", "checkpoint_sha256": RAPTOR_PLAIN_SHA256,
                "checkpoint_file": RAPTOR_WEIGHT_NAME, "weight_dataset": RAPTOR_WEIGHT_DATASET}
    cc = {"rehearsal": True, "revision": "synthetic-revision", "sources": CONVNEXT_SOURCE_HASHES,
          "entry_sha256": CONVNEXT_ENTRY_SHA256}
    cr = {"rehearsal": True, "status": "PASS", "code_revision": cc["revision"],
          "source_hashes": CONVNEXT_SOURCE_HASHES,
          "weight_hashes": {f"cnxt_v0_fold{i}.pt": str(i) * 64 for i in range(3)},
          "studies": 19, "test_studies": 3, "decode_failures": 0, "zero_studies": 0,
          "repeat_max_abs_error": 0, "cache_parity_max_abs_error": 0,
          "estimated_1300_studies_seconds": 7000}
    rc = {"source": manifest, "private": True, "internet": False, "train_count": 16, "seed": 42}
    rr = {"status": "passed", "manifest": manifest, "manifest_sha256": sha256(canonical_bytes(manifest)),
          "prediction_fallbacks": 0, "source_bundle_sha256": "a" * 64,
          "sampling": {"train_count": 16, "test_count": 3, "seed": 42, "uses_labels": False},
          "studies": [{"valid_slices": 64, "decoder_pixel_max_error": 0,
                       "window_max_error": 0, "repeat_prediction_max_error": 0} for _ in range(19)],
          "estimated_1300_seconds_p95": 5000}
    return cc, cr, rc, rr, manifest


def schema_fixture(root, ids=("test-z", "test-a", "test-m")):
    labels = [f"target-{i}" for i in range(12)]
    with (root / "sample_submission.csv").open("w", newline="") as handle:
        # Deliberately different sample rows: actual test.csv controls coverage.
        csv.writer(handle).writerows([["StudyInstanceUID", *labels], ["old-visible-row", *([0] * 12)]])
    with (root / "test.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows([["StudyInstanceUID"], *[[sid] for sid in ids]])
    with (root / "test_series.csv").open("w", newline="") as handle:
        csv.writer(handle).writerows([["StudyInstanceUID", "SeriesInstanceUID"],
                                     *[[sid, f"series-{i}"] for i, sid in enumerate(ids)]])
    (root / "train.csv").write_text("MUST NOT BE READ OR USED FOR SELECTION")
    return ["StudyInstanceUID", *labels], list(ids)


class BlendContractTests(unittest.TestCase):
    def test_codec_selection_ignores_cp311_but_rejects_missing_or_duplicate_cp312(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in CODEC_WHEELS:
                (root / name).write_bytes(b"synthetic wheel")
                if "cp312" in name:
                    (root / name.replace("cp312", "cp311")).write_bytes(b"different ABI")
            self.assertEqual(len(list(root.glob("*.whl"))), 5)
            selected = select_codec_wheels(root)
            self.assertEqual([p.name for p in selected], list(CODEC_WHEELS))
            self.assertTrue(all("cp311" not in p.name for p in selected))
            # A second compatible version must not be silently chosen by pip.
            duplicate = root / CODEC_WHEELS[1].replace("2.4.0", "2.5.0")
            duplicate.write_bytes(b"unexpected compatible version")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                select_codec_wheels(root)
            duplicate.unlink()
            (root / CODEC_WHEELS[2]).unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                select_codec_wheels(root)

    def test_both_rehearsals_required_with_exact_sources_plain_weight_and_runtime(self):
        qualified = qualify_components(*component_receipts())
        self.assertEqual(qualified["estimated_1300_seconds_with_margin"], 15000)
        cases = [(0, "entry_sha256", "a" * 64), (1, "status", "FAIL"),
                 (1, "decode_failures", 1), (1, "zero_studies", 1),
                 (1, "repeat_max_abs_error", None), (1, "cache_parity_max_abs_error", .01),
                 (1, "estimated_1300_studies_seconds", 24000),
                 (3, "prediction_fallbacks", 1), (3, "status", "running"),
                 (3, "manifest_sha256", "f" * 64)]
        for index, key, value in cases:
            data = copy.deepcopy(component_receipts())
            data[index][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                qualify_components(*data)
        data = copy.deepcopy(component_receipts())
        data[3]["sampling"]["uses_labels"] = True
        with self.assertRaisesRegex(ValueError, "sampling"):
            qualify_components(*data)
        data = copy.deepcopy(component_receipts())
        data[4]["checkpoint_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            qualify_components(*data)

    def test_unreviewed_reader_cannot_execute_or_package(self):
        with self.assertRaisesRegex(ValueError, "reviewed rehearsal"):
            extract_reader("raise RuntimeError('never execute this')")

    def test_schema_uses_actual_ids_in_order_and_visible_rehearsal_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns, ids = schema_fixture(root)
            actual_columns, actual_ids, records = read_test_contract(root, rehearsal=True)
            self.assertEqual((actual_columns, actual_ids), (columns, ids))
            self.assertEqual(set(records), set(ids))
            schema_fixture(root, ("new-hidden-id",))
            self.assertEqual(read_test_contract(root, rehearsal=False)[1], ["new-hidden-id"])
            with self.assertRaisesRegex(ValueError, "three"):
                read_test_contract(root, rehearsal=True)
            schema_fixture(root, ("duplicate", "duplicate", "another"))
            with self.assertRaisesRegex(ValueError, "unique"):
                read_test_contract(root, rehearsal=True)

    def test_writer_rejects_bad_predictions_before_creating_file_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns, ids = schema_fixture(root)
            output = root / "predictions.csv"
            with self.assertRaisesRegex(ValueError, "finite"):
                write_probabilities(output, columns, ids, [[float("nan")] * 12] * 3)
            self.assertFalse(output.exists())
            write_probabilities(output, columns, ids, [[1.] * 12] * 3)
            with output.open() as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], columns)
            self.assertEqual([r[0] for r in rows[1:]], ids)
            with self.assertRaises(FileExistsError):
                write_probabilities(output, columns, ids, [[1.] * 12] * 3)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "optional numpy unavailable")
class RankTests(unittest.TestCase):
    def test_average_ties_not_order_ties_or_probability_average(self):
        import numpy as np
        first = np.repeat(np.array([.1, .1, .9])[:, None], 12, axis=1)
        second = np.repeat(np.array([.7, .2, .2])[:, None], 12, axis=1)
        a, b = first.copy(), second.copy()
        result = equal_rank_blend(first, second)
        np.testing.assert_array_equal(result[:, 0], [.75, .5, .75])
        np.testing.assert_array_equal(first, a)
        np.testing.assert_array_equal(second, b)
        permutation = [2, 0, 1]
        np.testing.assert_array_equal(equal_rank_blend(first[permutation], second[permutation]), result[permutation])

    def test_one_study_rank_boundary_and_component_fallback_rejection(self):
        import numpy as np
        probabilities = np.linspace(.1, .9, 12)[None]
        np.testing.assert_array_equal(equal_rank_blend(probabilities, probabilities), np.ones((1, 12)))
        for bad in (np.full((1, 12), .5), np.zeros((1, 12)), np.full((1, 12), np.nan),
                    np.full((1, 12), 1.2), np.ones((1, 11))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                equal_rank_blend(probabilities, bad)
        with self.assertRaises(ValueError):
            equal_rank_blend(probabilities, np.repeat(probabilities, 2, axis=0))


def load_builder():
    directory = Path(__file__).resolve().parents[1] / "scripts"
    with patch.object(sys, "path", [str(directory), *sys.path]):
        spec = importlib.util.spec_from_file_location("blend_builder", directory / "build_reference_blend.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class BlendReceiptTests(unittest.TestCase):
    def test_full_entry_requires_matching_common_environment_parity_receipt(self):
        builder = load_builder()
        component = {"studies": 3, "decode_failures": 0, "zero_studies": 0, "raw_rerun_max_abs_error": 0}
        report = {"status": "PASS", "rehearsal": True, "qualification_sha256": "a" * 64,
                  "test_studies": 3, "prediction_fallbacks": 0, "blend_repeat_max_abs_error": 0,
                  "estimated_1300_seconds_mean_with_margin": 17000,
                  "components": {"convnext": component, "raptor": component},
                  "codec_hashes": {name: str(i) * 64 for i, name in enumerate(CODEC_WHEELS)}}
        self.assertEqual(builder.require_blend_rehearsal(report, "a" * 64), report["codec_hashes"])
        with self.assertRaisesRegex(ValueError, "matching"):
            builder.require_blend_rehearsal(report, "b" * 64)
        changed = copy.deepcopy(report)
        changed["components"]["raptor"]["raw_rerun_max_abs_error"] = .001
        with self.assertRaisesRegex(ValueError, "reread"):
            builder.require_blend_rehearsal(changed, "a" * 64)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch unavailable")
class BlendRuntimeGuardTests(unittest.TestCase):
    def contract(self):
        return {"recipe": RECIPE, "max_seconds": 3600, "rehearsal": True,
                "qualification_sha256": "a" * 64,
                "qualification": {"convnext_weights": {}, "raptor_manifest": {}}}

    def environment(self):
        stack = ExitStack()
        for name, result in (("is_available", True), ("get_device_name", "synthetic CUDA"),
                             ("reset_peak_memory_stats", None), ("max_memory_allocated", 1),
                             ("max_memory_reserved", 1)):
            stack.enter_context(patch("torch.cuda." + name, return_value=result))
        stack.enter_context(patch("knee.reference_blend.importlib.metadata.version", return_value="synthetic"))
        return stack

    def test_component_failure_writes_sanitized_audit_but_no_partial_submission(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory, self.environment():
            root = Path(directory)
            schema_fixture(root)
            values = np.repeat(np.linspace(.1, .9, 12)[None], 3, axis=0)
            info = {"study_seconds": [1., 1., 1.]}
            with patch("knee.reference_blend._convnext", return_value=(values, info, values.copy())), \
                 patch("knee.reference_blend._raptor", side_effect=ValueError("sensitive-study-value")), \
                 self.assertRaisesRegex(RuntimeError, "sanitized"):
                run_blend(root, root, root, {}, "", "", self.contract(), root / "out", codec_hashes={})
            self.assertFalse((root / "out/submission.csv").exists())
            self.assertFalse((root / "out/visible-blend.csv").exists())
            report = (root / "out/blend-audit.json").read_text()
            self.assertNotIn("sensitive-study-value", report)
            self.assertEqual(json.loads(report)["status"], "FAIL")

    def test_success_preserves_components_and_repeats_fixed_blend(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory, self.environment():
            root = Path(directory)
            columns, ids = schema_fixture(root)
            first = np.repeat(np.array([.1, .1, .9])[:, None], 12, axis=1)
            second = np.repeat(np.array([.7, .2, .2])[:, None], 12, axis=1)
            info = {"study_seconds": [1., 1., 1.], "studies": 3, "decode_failures": 0,
                    "zero_studies": 0, "raw_rerun_max_abs_error": 0}
            with patch("knee.reference_blend._convnext", return_value=(first, copy.deepcopy(info), first.copy())), \
                 patch("knee.reference_blend._raptor", return_value=(second, copy.deepcopy(info), second.copy())):
                report = run_blend(root, root, root, {}, "", "", self.contract(), root / "out", codec_hashes={})
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["blend_repeat_max_abs_error"], 0)
            for filename, values in (("convnext-raw.csv", first), ("raptor-raw.csv", second),
                                     ("visible-blend.csv", equal_rank_blend(first, second))):
                with (root / "out" / filename).open() as handle:
                    rows = list(csv.reader(handle))
                self.assertEqual(rows[0], columns)
                self.assertEqual([row[0] for row in rows[1:]], ids)
                np.testing.assert_array_equal(np.array([row[1:] for row in rows[1:]], dtype=float), values)
            self.assertFalse((root / "out/submission.csv").exists())

    def test_tiny_component_drift_that_changes_ranking_fails_qualification(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory, self.environment():
            root = Path(directory)
            schema_fixture(root)
            values = np.repeat(np.array([.1, .1 + 1e-8, .9])[:, None], 12, axis=1)
            rerun = values.copy()
            rerun[[0, 1]] = rerun[[1, 0]]
            self.assertLess(float(np.max(np.abs(values - rerun))), 1e-6)
            info = {"study_seconds": [1., 1., 1.], "studies": 3, "decode_failures": 0,
                    "zero_studies": 0, "raw_rerun_max_abs_error": 1e-8}
            with patch("knee.reference_blend._convnext", return_value=(values, info, rerun)), \
                 patch("knee.reference_blend._raptor", return_value=(values, info, values)), \
                 self.assertRaisesRegex(RuntimeError, "sanitized"):
                run_blend(root, root, root, {}, "", "", self.contract(), root / "out", codec_hashes={})
            self.assertFalse((root / "out/visible-blend.csv").exists())


@unittest.skipUnless(os.environ.get("KNEE_REFERENCE_PARENT"), "requires optional local audited sources/receipts")
class BlendSnapshotTests(unittest.TestCase):
    def test_real_receipts_build_private_offline_notebook_without_prediction_fixtures(self):
        builder = load_builder()
        root = Path(os.environ["KNEE_REFERENCE_PARENT"])
        options = dict(convnext_source_dir=root / "outputs/research/2026-10-08/convnext-audit",
            convnext_entry=root / "scripts/public_reader_entry.py",
            convnext_contract=root / "outputs/kaggle/public-reader-rehearsal/build-contract.json",
            convnext_rehearsal=root / "outputs/kaggle/public-reader-rehearsal-result/reader-audit.json",
            raptor_source=root / "outputs/research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py",
            raptor_contract=root / "outputs/kaggle/raptor-rehearsal/reference-contract.json",
            raptor_rehearsal=root / "outputs/kaggle/raptor-rehearsal-result/reference-rehearsal/rehearsal.json")
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "package"
            receipt = builder.build_blend(dest, "synthetic-private-build", **options)
            metadata = json.loads((dest / "kernel-metadata.json").read_text())
            self.assertTrue(metadata["is_private"])
            self.assertFalse(metadata["enable_internet"])
            self.assertEqual(metadata["model_sources"], [])
            self.assertEqual(metadata["kernel_sources"], [])
            self.assertEqual(len(metadata["dataset_sources"]), 2)
            self.assertLess(receipt["notebook_bytes"], 900000)
            notebook = json.loads((dest / "notebook.ipynb").read_text())
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), "<generated-blend>", "exec")
            with self.assertRaisesRegex(ValueError, "successful blend rehearsal"):
                builder.build_blend(Path(directory) / "full", "full-build", submission=True, **options)
            self.assertFalse((Path(directory) / "full").exists())


if __name__ == "__main__":
    unittest.main()
