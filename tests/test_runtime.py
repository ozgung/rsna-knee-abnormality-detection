import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knee.runtime import competition_root, prepare_codecs, training_artifacts


class RuntimeTests(unittest.TestCase):
    def test_attachment_discovery_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "notebooks/ozgung/knee-dino-train-20261008/artifacts"
            (first / "attention").mkdir(parents=True)
            (first / "extraction.json").touch()
            (first / "attention/head.pt").touch()
            self.assertEqual(training_artifacts(root), first)
            second = root / "knee-dino-train-20261008/artifacts"
            (second / "attention").mkdir(parents=True)
            (second / "extraction.json").touch()
            (second / "attention/head.pt").touch()
            with self.assertRaises(ValueError):
                training_artifacts(root)

    def test_wheel_hash_mismatch_fails_before_install(self):
        with tempfile.TemporaryDirectory() as tmp, patch("knee.runtime.subprocess.run") as run:
            root = Path(tmp)
            hashes = {}
            for i in range(4):
                p = root / f"codec{i}.whl"
                p.write_bytes(b"fixture")
                hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(prepare_codecs(root, expected_hashes=hashes), hashes)
            self.assertIn("--no-index", run.call_args.args[0])
            run.reset_mock()
            next(root.glob("*.whl")).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                prepare_codecs(root, expected_hashes=hashes)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
