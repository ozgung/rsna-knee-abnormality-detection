import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

from knee import public_r6_profile as profile


class PublicR6ProfileTests(unittest.TestCase):
    def test_frozen_recipe_and_identity(self):
        self.assertEqual(profile.BUNDLE_ID, "fe055406fad6a12f")
        self.assertEqual(profile.BUNDLE_TAG, "cnx_r6fm_c352")
        self.assertTrue(profile.FLIP_TTA)
        self.assertEqual((profile.R384_WEIGHT, profile.R224_WEIGHT), (0.65, 0.35))
        self.assertEqual(dict(profile.RAPTOR_WEIGHTS), {
            "Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6,
            "Synovitis": 0.8, "Baker's": 0.35,
        })
        self.assertEqual(profile.DEFAULT_RAPTOR_WEIGHT, 0.5)
        with self.assertRaises(TypeError):
            profile.RAPTOR_WEIGHTS["Medial OA"] = 0.0
        with self.assertRaises(TypeError):
            profile.ASSET_SHA256["side_cls.pt"] = "0" * 64

    def test_streaming_hash_verifies_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"abc123" * 400_000  # Exercises multiple streaming reads.
            target = root / "asset.bin"
            target.write_bytes(payload)
            pins = MappingProxyType({"asset.bin": hashlib.sha256(payload).hexdigest()})
            self.assertEqual(profile._verify_files(root, pins, label="TEST")["asset.bin"], pins["asset.bin"])
            target.write_bytes(payload + b"changed")
            with self.assertRaisesRegex(ValueError, "TEST_SHA256_MISMATCH"):
                profile._verify_files(root, pins, label="TEST")

    def test_missing_and_nonregular_assets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = MappingProxyType({"expected.bin": "0" * 64})
            with self.assertRaisesRegex(ValueError, "ASSET_MISSING_OR_UNREADABLE"):
                profile._verify_files(root, pins, label="ASSET")
            (root / "expected.bin").mkdir()
            with self.assertRaisesRegex(ValueError, "ASSET_NOT_REGULAR_DIRECT_FILE"):
                profile._verify_files(root, pins, label="ASSET")

    def test_wrong_manifest_bytes_and_symlink_asset_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "MANIFEST_SHA256_MISMATCH"):
                profile.validate_r6_assets(root)
            target = root / "real.bin"
            target.write_bytes(b"synthetic")
            (root / "asset.bin").symlink_to(target)
            pins = MappingProxyType({"asset.bin": hashlib.sha256(target.read_bytes()).hexdigest()})
            with self.assertRaisesRegex(ValueError, "ASSET_NOT_REGULAR_DIRECT_FILE"):
                profile._verify_files(root, pins, label="ASSET")

    def test_wrong_bundle_identity_is_rejected_even_with_matching_manifest_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # A synthetic R7-like manifest lets the identity guard be exercised
            # independently of the fixed production manifest digest.
            assets = {"asset.bin": "a" * 64}
            manifest = json.dumps({"bundle_id": "r7", "tag": "r7", "sha256": assets}).encode()
            (root / "manifest.json").write_bytes(manifest)
            with patch.object(profile, "MANIFEST_SHA256", hashlib.sha256(manifest).hexdigest()), \
                 patch.object(profile, "ASSET_SHA256", MappingProxyType(assets)):
                with self.assertRaisesRegex(ValueError, "MANIFEST_R6_IDENTITY_MISMATCH"):
                    profile.validate_r6_assets(root)

    def test_public_validator_accepts_and_rejects_assembled_synthetic_bundle(self):
        asset_bytes = {name: f"synthetic:{name}".encode() for name in profile.ASSET_SHA256}
        source_bytes = {name: f"synthetic-source:{name}".encode() for name in profile.SOURCE_SHA256}
        asset_pins = MappingProxyType({name: hashlib.sha256(data).hexdigest()
                                       for name, data in asset_bytes.items()})
        source_pins = MappingProxyType({name: hashlib.sha256(data).hexdigest()
                                        for name, data in source_bytes.items()})

        def fixture(base: Path, *, closure=None):
            base.mkdir(parents=True, exist_ok=True)
            assets = base / "assets"
            sources = base / "sources"
            assets.mkdir()
            sources.mkdir()
            for name, data in asset_bytes.items():
                (assets / name).write_bytes(data)
            for name, data in source_bytes.items():
                (sources / name).write_bytes(data)
            manifest = json.dumps({
                "bundle_id": profile.BUNDLE_ID,
                "tag": profile.BUNDLE_TAG,
                "sha256": dict(closure if closure is not None else asset_pins),
            }, sort_keys=True).encode()
            (assets / "manifest.json").write_bytes(manifest)
            return assets, sources, manifest

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            assets, sources, manifest = fixture(base)
            with patch.object(profile, "ASSET_SHA256", asset_pins), \
                 patch.object(profile, "SOURCE_SHA256", source_pins), \
                 patch.object(profile, "MANIFEST_SHA256", hashlib.sha256(manifest).hexdigest()):
                result = profile.validate_r6_assets(assets)
                self.assertEqual(result["assets_sha256"], dict(asset_pins))
                self.assertIsNone(result["sources_sha256"])
                result_with_sources = profile.validate_r6_assets(assets, source_dir=sources)
                self.assertEqual(result_with_sources["sources_sha256"], dict(source_pins))

                model = assets / "E_lrprior_s0_f0.pt"
                model.write_bytes(b"modified-model")
                with self.assertRaisesRegex(ValueError, "ASSET_SHA256_MISMATCH"):
                    profile.validate_r6_assets(assets)
                model.write_bytes(asset_bytes[model.name])
                model.unlink()
                with self.assertRaisesRegex(ValueError, "ASSET_MISSING_OR_UNREADABLE"):
                    profile.validate_r6_assets(assets)
                model.write_bytes(asset_bytes[model.name])

                worker = sources / "compact_infer.py"
                worker.write_bytes(b"modified-worker")
                with self.assertRaisesRegex(ValueError, "SOURCE_SHA256_MISMATCH"):
                    profile.validate_r6_assets(assets, source_dir=sources)
                worker.write_bytes(source_bytes[worker.name])

                wrong_closure = dict(asset_pins)
                wrong_closure["side_cls.pt"] = "0" * 64
                bad_assets, _, bad_manifest = fixture(base / "bad-closure", closure=wrong_closure)
                with patch.object(profile, "MANIFEST_SHA256", hashlib.sha256(bad_manifest).hexdigest()):
                    with self.assertRaisesRegex(ValueError, "MANIFEST_ASSET_CLOSURE_MISMATCH"):
                        profile.validate_r6_assets(bad_assets)

    def test_source_closure_rejects_a_changed_worker_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "worker.py").write_bytes(b"worker-v1")
            pins = MappingProxyType({"worker.py": hashlib.sha256(b"worker-v1").hexdigest()})
            self.assertEqual(profile._verify_files(source, pins, label="SOURCE")["worker.py"], pins["worker.py"])
            (source / "worker.py").write_bytes(b"worker-v2")
            with self.assertRaisesRegex(ValueError, "SOURCE_SHA256_MISMATCH"):
                profile._verify_files(source, pins, label="SOURCE")


if __name__ == "__main__":
    unittest.main()
