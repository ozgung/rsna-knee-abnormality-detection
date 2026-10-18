import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from knee.s6_solo_pair import COMMON, CONTROL_DATASET, CONTROL_SHA256, SOURCE_SHA256, TARGETS, V2_DATASET, main, stage_checkpoint, validate_pair


def descriptors():
    base = {"targets": list(TARGETS), "dataset_slug": CONTROL_DATASET, "dataset_version": 1,
            "checkpoint_size": 4, "common": copy.deepcopy(COMMON)}
    return ({**base, "variant": "control", "checkpoint_filename": "E_prior_s42_fall.pt", "checkpoint_sha256": CONTROL_SHA256},
            {**base, "dataset_slug": V2_DATASET, "dataset_version": 9, "variant": "v2",
             "checkpoint_filename": "E_prior_s42_v2all.pt", "checkpoint_sha256": "b" * 64,
             "runtime_alias": "E_prior_s17_fold_fresh.pt"})


class S6SoloPairTests(unittest.TestCase):
    def test_pair_accepts_independent_versions_and_keeps_evidence_pending(self):
        a, b = descriptors()
        plan = validate_pair(a, b)
        self.assertFalse(plan["seal_available"])
        self.assertTrue(plan["published_control_descriptor_matches_pin"])
        self.assertFalse(plan["descriptor_hashes_verified_against_files"])
        self.assertFalse(plan["source_bytes_verified"])
        self.assertEqual(plan["control"]["dataset_version"], 1)
        self.assertEqual(plan["v2"]["dataset_version"], 9)
        self.assertIn("CPU/state/constructor receipt", plan["pending"])

    def test_pending_receipts_are_not_required_or_mistaken_for_verified(self):
        a, b = descriptors()
        a.update(training_provenance=None, cpu_receipt=None)
        b.update(training_provenance=None, cpu_receipt=None)
        self.assertFalse(validate_pair(a, b)["seal_available"])

    def test_rejects_wrong_order_version_alias_and_shared_contract(self):
        cases = [("targets", ["wrong", *TARGETS[1:]]), ("dataset_version", True),
                 ("dataset_version", 0), ("dataset_version", "latest"),
                 ("checkpoint_sha256", "not-a-hash"), ("checkpoint_size", True),
                 ("runtime_alias", "../E_prior_s17_fold.pt"),
                 ("runtime_alias", "E_prior_s42_fbad\x00.pt"),
                 ("runtime_alias", "E_prior_s42_fbad\n.pt"),
                 ("runtime_alias", 42), ("checkpoint_filename", "wrong.pt"),
                 ("common", {**COMMON, "recovery": "fallback"})]
        for field, value in cases:
            with self.subTest(field=field):
                a, b = descriptors(); b[field] = value
                with self.assertRaises(ValueError): validate_pair(a, b)
        a, b = descriptors()
        with self.assertRaisesRegex(ValueError, "source hash"):
            validate_pair(a, b, source=b"wrong")

    def test_published_control_and_dataset_identity_are_not_arbitrary_descriptors(self):
        for field, value in [("checkpoint_sha256", "a" * 64), ("dataset_slug", "wrong/control")]:
            a, b = descriptors(); a[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "control descriptor"):
                validate_pair(a, b)
        a, b = descriptors(); b["dataset_slug"] = "wrong/v2"
        with self.assertRaisesRegex(ValueError, "V2 dataset"):
            validate_pair(a, b)

    def test_every_shared_inference_field_is_frozen(self):
        for key in COMMON:
            a, b = descriptors(); b["common"] = {**b["common"], key: None}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "shared source/A/recovery/inference"):
                validate_pair(a, b)

    def test_stage_verifies_bytes_and_creates_identical_one_file_alias(self):
        a, b = descriptors(); payload = b"tiny synthetic checkpoint"
        b["checkpoint_filename"] = "E_prior_s42_v2all.pt"
        b["checkpoint_sha256"] = hashlib.sha256(payload).hexdigest(); b["checkpoint_size"] = len(payload)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / b["checkpoint_filename"]; source.write_bytes(payload)
            dest = root / "isolated"
            receipt = stage_checkpoint(b, source, dest)
            alias = dest / b["runtime_alias"]
            self.assertEqual(alias.read_bytes(), payload)
            self.assertEqual(receipt["source_sha256"], receipt["alias_sha256"])
            self.assertEqual(sorted(p.name for p in dest.iterdir()), [b["runtime_alias"]])
            self.assertFalse(receipt["seal_available"])

    def test_stage_rejects_mismatch_unsafe_source_and_existing_or_extra_directory(self):
        _, b = descriptors(); payload = b"payload"
        b["checkpoint_sha256"] = hashlib.sha256(payload).hexdigest(); b["checkpoint_size"] = len(payload)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / b["checkpoint_filename"]; source.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "do not match"):
                stage_checkpoint(b, source, root / "dest")
            source.write_bytes(payload)
            b["runtime_alias"] = "../escape.pt"
            with self.assertRaises(ValueError): stage_checkpoint(b, source, root / "dest")
            b["runtime_alias"] = "E_prior_s42_fall_v2.pt"
            existing = root / "existing"; existing.mkdir(); (existing / "extra").write_text("x")
            with self.assertRaisesRegex(ValueError, "already exists"):
                stage_checkpoint(b, source, existing)

    def test_failed_copy_identity_cleans_only_the_new_destination(self):
        _, b = descriptors(); payload = b"payload"
        digest = hashlib.sha256(payload).hexdigest()
        b.update(checkpoint_sha256=digest, checkpoint_size=len(payload))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / b["checkpoint_filename"]; source.write_bytes(payload)
            dest = root / "fresh"
            with patch("knee.s6_solo_pair._file_identity", side_effect=[(digest, len(payload)), ("0" * 64, len(payload))]):
                with self.assertRaisesRegex(ValueError, "staged copy"):
                    stage_checkpoint(b, source, dest)
            self.assertFalse(dest.exists())
            self.assertEqual(source.read_bytes(), payload)

    def test_existing_empty_destination_and_symlinks_are_preserved(self):
        _, b = descriptors(); payload = b"payload"
        b.update(checkpoint_sha256=hashlib.sha256(payload).hexdigest(), checkpoint_size=len(payload))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / b["checkpoint_filename"]; source.write_bytes(payload)
            dest = root / "empty"; dest.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                stage_checkpoint(b, source, dest)
            self.assertTrue(dest.is_dir())
            alias = root / "symlink"; alias.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                stage_checkpoint(b, alias, root / "unused")

    def test_plan_contract_does_not_share_mutable_global_and_cli_is_fresh(self):
        a, b = descriptors(); plan = validate_pair(a, b)
        plan["common_contract"]["weights"]["B"] = 99
        self.assertEqual(COMMON["weights"]["B"], 1)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); ap = root / "a.json"; bp = root / "b.json"; out = root / "plan.json"
            ap.write_text(json.dumps(a)); bp.write_text(json.dumps(b))
            main([str(ap), str(bp), "--output", str(out)])
            with self.assertRaises(FileExistsError): main([str(ap), str(bp), "--output", str(out)])


if __name__ == "__main__": unittest.main()
