import tempfile
import unittest
import importlib.util
from pathlib import Path

from knee.public_ws3_prefetch import (
    bind_evaluation_input_reference,
    reference_from_receipt,
    run_paired_r7_phases,
)


def synthetic_receipt(studies=39, fingerprints=8):
    rows = [
        {"id": f"synthetic-{i:02d}", "input": {f"f{j}": f"sha-{i}-{j}" for j in range(fingerprints)}}
        for i in range(studies)
    ]
    return {"workers": {"s6": [{"records": rows[:20]}, {"records": rows[20:]}]}}


class Ws3PrefetchAdapterTests(unittest.TestCase):
    def test_reference_requires_complete_unique_39_by_8_receipt(self):
        reference = reference_from_receipt(synthetic_receipt())
        self.assertEqual(len(reference), 39)
        self.assertTrue(all(len(row) == 8 for row in reference.values()))
        with self.assertRaisesRegex(ValueError, "V7_R7_STUDY_COUNT"):
            reference_from_receipt(synthetic_receipt(studies=38))
        with self.assertRaisesRegex(ValueError, "V7_R7_FINGERPRINT_SCHEMA"):
            reference_from_receipt(synthetic_receipt(fingerprints=7))

    def test_binding_is_rehearsal_only_and_does_not_mutate_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            # Exercise the pure binding copy with a patched receipt loader so the
            # test never needs real evaluation IDs or a private output artifact.
            from unittest.mock import patch

            reference = reference_from_receipt(synthetic_receipt())
            cfg = {"mode": "rehearsal", "assignment": [0, 1]}
            with patch("knee.public_ws3_prefetch.load_v7_input_reference", return_value=reference):
                bound = bind_evaluation_input_reference(cfg, path)
            self.assertIsNot(bound, cfg)
            self.assertNotIn("evaluation_input_reference", cfg)
            self.assertEqual(bound["evaluation_input_reference"], reference)
            with patch("knee.public_ws3_prefetch.load_v7_input_reference", return_value=reference):
                with self.assertRaisesRegex(ValueError, "REHEARSAL_ONLY"):
                    bind_evaluation_input_reference({"mode": "entry"}, path)

    def test_paired_launch_preserves_cfg_and_changes_only_prefetch_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            serial_out, prefetch_out = root / "serial", root / "prefetch"
            serial_out.mkdir()
            prefetch_out.mkdir()
            cfg = {
                "mode": "rehearsal",
                "assignment": [["a", "b"], ["c"]],
                "boards": [{"uuid": "board-a"}, {"uuid": "board-b"}],
                "evaluation_input_reference": {f"id-{i}": {"f": str(i)} for i in range(39)},
                "state": "bound",
            }
            calls = []

            def fake_launch(phase_cfg, out, arm):
                calls.append((phase_cfg, out, arm))
                return ([{"status": "PASS"}], {"peaks": {"host_rss_bytes": 123}})

            result = run_paired_r7_phases(cfg, serial_out, prefetch_out, launch=fake_launch)
            self.assertEqual([(x[0]["r7_prefetch"], x[2]) for x in calls], [(False, "s6"), (True, "s6")])
            self.assertEqual(calls[0][0]["assignment"], calls[1][0]["assignment"])
            self.assertEqual(calls[0][0]["boards"], calls[1][0]["boards"])
            self.assertEqual(calls[0][0]["evaluation_input_reference"], calls[1][0]["evaluation_input_reference"])
            self.assertNotIn("r7_prefetch", cfg)
            self.assertEqual(result["paired_identity"]["evaluation_studies"], 39)

    def test_pair_rejects_entry_mode_and_reused_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "one").mkdir()
            with self.assertRaisesRegex(ValueError, "REHEARSAL_ONLY"):
                run_paired_r7_phases({"mode": "entry"}, path / "one", path / "one", launch=lambda *a: None)
            cfg = {"mode": "rehearsal", "evaluation_input_reference": {}}
            with self.assertRaisesRegex(ValueError, "DISTINCT"):
                run_paired_r7_phases(cfg, path / "one", path / "one", launch=lambda *a: None)

    def test_p3_speed_and_combined_rss_gates(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "compare_r7_prefetch_p3.py"
        spec = importlib.util.spec_from_file_location("compare_r7_prefetch_p3", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        serial = {"peaks": {"host_rss_bytes": 18_000_000_000}}
        prefetch = {"peaks": {"host_rss_bytes": 21_000_000_000}}
        passing = module.p3_runtime_gates(.84, serial, prefetch)
        self.assertTrue(passing["speed"]["pass"])
        self.assertTrue(passing["combined_rss"]["pass"])
        slow = module.p3_runtime_gates(.86, serial, prefetch)
        self.assertFalse(slow["speed"]["pass"])
        memory = module.p3_runtime_gates(.84, serial, {"peaks": {"host_rss_bytes": 22_000_000_001}})
        self.assertFalse(memory["combined_rss"]["pass"])


if __name__ == "__main__":
    unittest.main()
