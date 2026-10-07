"""Local packaging checks; no data upload, model loading or remote execution."""
import builtins
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_ensemble_kaggle.py"
sys.path.insert(0, str(SCRIPT.parent))
try:
    spec = importlib.util.spec_from_file_location("knee_ensemble_packager_tests", SCRIPT)
    builder = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(builder)
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
        builder = None
finally:
    sys.path.pop(0)


@unittest.skipIf(builder is None, "Optional packaging dependency unavailable")
class EnsemblePackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.baseline = self.root / "baseline"
        self.comparisons = self.root / "comparisons"
        self.assets = self.root / "upload"
        self.destination = self.root / "notebook"
        self.marker = "SYNTHETIC_PRIVATE_PAYLOAD_MUST_NOT_SHIP"
        candidates = {}
        for name in ("baseline", "attention_seed43", "attention_seed44", "meanmax_seed42"):
            directory = self.baseline / "attention" if name == "baseline" else self.comparisons / name
            directory.mkdir(parents=True)
            content = ("synthetic-checkpoint-" + name).encode()
            (directory / "head.pt").write_bytes(content)
            (directory / "gold_predictions.npz").write_text(self.marker)
            (directory / "weak_predictions.npz").write_text(self.marker)
            (directory / "private-report.txt").write_text(self.marker)
            candidates[name] = {"checkpoint_sha256": hashlib.sha256(content).hexdigest()}
        (self.baseline / "features.npz").write_text(self.marker)
        self.report = {
            "status": "completed", "selection_uses_gold": False, "selection_tolerance": 1e-5,
            "selected": {"members": ["baseline", "attention_seed43"], "weights": [.5, .5], "weak_bce": .4},
            "comparisons": [{"members": ["baseline"], "weak_bce": .5}],
            "feature_sha256": "f" * 64, "targets": [f"t{i}" for i in range(12)], "candidates": candidates,
        }
        self.write_report()

    def write_report(self):
        (self.comparisons / "comparison.json").write_text(json.dumps(self.report))

    def package(self, rehearsal=False):
        with contextlib.redirect_stdout(io.StringIO()):
            builder.build_selected(self.baseline, self.comparisons, self.destination, "synthetic-ensemble",
                                   assets_dir=self.assets, dataset_source="ozgung/synthetic-private-heads",
                                   rehearsal=rehearsal)

    @contextlib.contextmanager
    def forbid_prediction_reads(self):
        original_builtin, original_io = builtins.open, io.open
        def guard(original):
            def wrapped(file, *args, **kwargs):
                if isinstance(file, (str, bytes, os.PathLike)) and Path(os.fsdecode(file)).suffix == ".npz":
                    raise AssertionError("Packager attempted to touch private prediction/cache data")
                return original(file, *args, **kwargs)
            return wrapped
        with patch("builtins.open", side_effect=guard(original_builtin)), patch("io.open", side_effect=guard(original_io)):
            yield

    def assert_safe_package(self):
        selected_files = {name + ".pt" for name in self.report["selected"]["members"]}
        self.assertEqual({path.name for path in self.assets.iterdir()}, selected_files | {"dataset-metadata.json"})
        selection = json.loads((self.destination / "selection.json").read_text())
        self.assertEqual(set(selection["asset_sha256"]), selected_files)
        for name, digest in selection["asset_sha256"].items():
            self.assertEqual(hashlib.sha256((self.assets / name).read_bytes()).hexdigest(), digest)
        metadata = json.loads((self.destination / "kernel-metadata.json").read_text())
        self.assertTrue(metadata["is_private"])
        self.assertFalse(metadata["enable_internet"])
        self.assertEqual(metadata["dataset_sources"], ["ozgung/synthetic-private-heads"])
        self.assertEqual(metadata["kernel_sources"], ["ozgung/knee-dino-train-20261008"])
        text = (self.destination / "notebook.ipynb").read_text()
        self.assertNotIn(self.marker, text)
        self.assertNotIn("synthetic-checkpoint-", text)
        self.assertLess(len(text.encode()), 1_000_000)
        notebook = json.loads(text)
        source = "".join(cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
                         for cell in notebook["cells"] if cell["cell_type"] == "code")
        compile(source, "packaged-notebook", "exec")
        return source

    def test_submission_never_touches_predictions_or_unselected_assets(self):
        with self.forbid_prediction_reads():
            self.package()
        self.assert_safe_package()

    def test_rehearsal_also_packages_only_weights_without_local_prediction_reads(self):
        with self.forbid_prediction_reads():
            self.package(rehearsal=True)
        source = self.assert_safe_package()
        # The reference is generated inside the private Kaggle runtime from its
        # existing training attachment, never supplied as a dataset asset.
        self.assertIn("features.npz", source)
        self.assertNotIn("selected_root/'expected-gold.npz'", source)

    def test_selected_checkpoint_hash_mismatch_fails_before_any_assets_written(self):
        (self.comparisons / "attention_seed43/head.pt").write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.package()
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.assets.exists() and any(self.assets.iterdir()))

    def test_stale_gold_or_other_upload_content_is_rejected_without_overwriting(self):
        for rehearsal in (False, True):
            for filename in ("expected-gold.npz", "private-report.txt", "unselected.pt"):
                with self.subTest(rehearsal=rehearsal, filename=filename):
                    self.assets = self.root / f"stale-{rehearsal}-{filename}"
                    self.assets.mkdir()
                    poison = self.assets / filename
                    poison.write_text(self.marker)
                    with self.assertRaises(ValueError):
                        self.package(rehearsal=rehearsal)
                    self.assertEqual(poison.read_text(), self.marker)
                    self.assertEqual({p.name for p in self.assets.iterdir()}, {filename})
        self.assertFalse(self.destination.exists())

    def test_selection_can_exclude_baseline_and_rejects_gold_selected_manifest(self):
        self.report["selected"] = {"members": ["attention_seed43"], "weights": [1.], "weak_bce": .4}
        self.write_report()
        with self.forbid_prediction_reads():
            self.package()
        self.assert_safe_package()
        self.assertFalse((self.assets / "baseline.pt").exists())
        self.report["selection_uses_gold"] = True
        self.write_report()
        self.assets = self.root / "rejected-gold-selection"
        self.destination = self.root / "rejected-notebook"
        with self.assertRaisesRegex(ValueError, "gold-independent"):
            self.package()
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.assets.exists() and any(self.assets.iterdir()))


if __name__ == "__main__":
    unittest.main()
