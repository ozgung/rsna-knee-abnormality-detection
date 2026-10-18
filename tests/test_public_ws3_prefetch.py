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
        for invalid in (0., -0.1, float("nan"), float("inf")):
            self.assertFalse(module.p3_runtime_gates(invalid, serial, prefetch)["speed"]["pass"])
        memory = module.p3_runtime_gates(.84, serial, {"peaks": {"host_rss_bytes": 22_000_000_001}})
        self.assertFalse(memory["combined_rss"]["pass"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional numpy arrays unavailable")
    def test_genuine_ledger_and_optional_feature_nan_boundaries(self):
        import numpy as np

        script = Path(__file__).resolve().parents[1] / "scripts" / "compare_r7_prefetch_p3.py"
        spec = importlib.util.spec_from_file_location("compare_r7_prefetch_p3_finite", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ledger = {
            "study_ids": np.asarray(["s0"]),
            "statuses": np.asarray(["genuine"]),
            "probabilities": np.zeros((1, 12)),
            "logits": np.zeros((1, 4, 12)),
        }
        self.assertTrue(module.finite_genuine_ledger(ledger))
        self.assertTrue(module.same_physical_runtime(
            {"environment": {"physical_uuid": "uuid", "torch": "2.10", "packages": {"timm": "1"}}},
            {"environment": {"physical_uuid": "uuid", "torch": "2.10", "packages": {"timm": "1"}}},
        ))
        self.assertFalse(module.same_physical_runtime(
            {"environment": {"physical_uuid": "uuid", "torch": "2.10", "packages": {"timm": "1"}}},
            {"environment": {"physical_uuid": "uuid", "torch": "2.9", "packages": {"timm": "1"}}},
        ))
        ledger["probabilities"][0, 0] = np.nan
        self.assertFalse(module.finite_genuine_ledger(ledger))
        ledger["probabilities"][0, 0] = 0.
        ledger["logits"][0, 0, 0] = np.inf
        self.assertFalse(module.finite_genuine_ledger(ledger))

        row = {"status": "genuine", "model_probabilities": [0.] * 12, "probabilities": [0.] * 12,
               "stackers": {"optional": {"z_feature": 0., "z_out": 0., "p_model": 0., "p_out": 0.,
                                            "features": [0., float("nan")]}}}
        self.assertTrue(module.finite_worker_output({"records": [row], "sentinels": []}))
        row["probabilities"][0] = float("nan")
        self.assertFalse(module.finite_worker_output({"records": [row], "sentinels": []}))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional numpy arrays unavailable")
    def test_full_compare_separates_mechanics_from_speed_and_checks_npz_hash(self):
        import numpy as np
        from unittest.mock import patch

        script = Path(__file__).resolve().parents[1] / "scripts" / "compare_r7_prefetch_p3.py"
        spec = importlib.util.spec_from_file_location("compare_r7_prefetch_p3_full", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from knee.public_r7_prefetch_candidate import VERSION

        with tempfile.TemporaryDirectory() as tmp:
            import hashlib
            import json

            folder = Path(tmp)
            np.savez(folder / "predictions.npz", study_ids=np.asarray(["synthetic"]),
                     statuses=np.asarray(["genuine"]), probabilities=np.zeros((1, 12)), logits=np.zeros((1, 4, 12)))
            npz_path = folder / "predictions.npz"
            expected_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
            (folder / "receipt.json").write_text(json.dumps({"ledger_sha256": expected_sha}))
            loaded_receipt, _, actual_sha = module.load_worker(folder)
            self.assertEqual(loaded_receipt["ledger_sha256"], actual_sha)

        all_ids = [f"s{i:02d}" for i in range(39)]
        reference_inputs = {sid: {f"f{j}": f"fp-{sid}-{j}" for j in range(8)} for sid in all_ids}
        v7_workers = []
        pair_loads = {}
        for board, ids in enumerate((all_ids[:20], all_ids[20:])):
            v7_workers.append({"board": board, "assigned": ids, "state_before": {"state": "same"},
                               "records": [{"id": sid, "input": reference_inputs[sid]} for sid in ids]})

        def worker(board, ids, prefetched, run_name, wall):
            records = []
            for sid in ids:
                row = {"id": sid, "status": "genuine", "windows": 4, "input": reference_inputs[sid],
                       "anchor_method": "synthetic", "anchor_reasons": [], "side": {}, "native_scratch_bytes": 0,
                       "field_center_fallbacks": 0, "fluid": {}, "source_diagnostics": {},
                       "pilot_input_parity": True, "model_probabilities": [0.] * 12,
                       "probabilities": [0.] * 12, "stackers": {}, "four_model_seconds": 1.,
                       "prefetch_wait_seconds": 0.}
                records.append(row)
            state = {"state": "same"}
            receipt = {"status": "PASS", "version": VERSION, "slot": "r7", "board": board,
                       "prefetch_enabled": prefetched, "compile": {"requested": True, "active": True,
                           "fallback": False, "runtime_fallback_study": None, "decision": "compiled"},
                       "assigned": ids, "state_before": state, "state_after": state, "zero_failed": True,
                       "fp32_recoveries": 0, "evaluation_wall": {"seconds": wall, "sum_production_seconds": wall,
                                                                     "studies": len(ids)},
                       "peak_rss_bytes": 8_000_000_000, "peak_cuda_bytes": 1, "peak_cuda_reserved_bytes": 2,
                       "ledger_sha256": "a" * 64,
                       "environment": {"physical_uuid": f"uuid-{board}", "torch": "2.10.0",
                                       "packages": {"timm": "1.0.26"}},
                       "evaluation_input_parity": {"checked": len(ids)}, "records": records, "sentinels": [],
                       "prefetch": ({"max_concurrent_stages": 1, "alive_at_exit": False,
                                     "prefetched": len(ids)-1, "submitted": len(ids)-1,
                                     "contexts": [{"main": "default", "producer": "default"}]}
                                    if prefetched else {"submitted": 0})}
            ledger = {"study_ids": np.asarray(ids), "statuses": np.asarray(["genuine"] * len(ids)),
                      "probabilities": np.zeros((len(ids), 12)), "logits": np.zeros((len(ids), 4, 12))}
            pair_loads[f"{run_name}{board}"] = (receipt, ledger, "a" * 64)

        for board, ids in enumerate((all_ids[:20], all_ids[20:])):
            worker(board, ids, False, "serial", 100.)
            worker(board, ids, True, "prefetch", 80.)
        v7 = {"workers": {"s6": v7_workers}}
        phase = {"peaks": {"host_rss_bytes": 21_000_000_000}}
        with patch.object(module, "load_worker", side_effect=lambda path: pair_loads[str(path)]):
            result = module.compare(["serial0", "serial1"], ["prefetch0", "prefetch1"], v7, "separate",
                                    expected_sentinels=0, serial_phase=phase, prefetch_phase=phase)
        self.assertTrue(result["mechanics_pass"])
        self.assertTrue(result["mirror_speed_pass"])
        self.assertTrue(result["pass"])

        # Failing speed alone leaves parity/resources intact so the parent may
        # record an output-preserving no-gain result and drop the mirror.
        for board, ids in enumerate((all_ids[:20], all_ids[20:])):
            worker(board, ids, False, "serial", 100.)
            worker(board, ids, True, "prefetch", 90.)
        with patch.object(module, "load_worker", side_effect=lambda path: pair_loads[str(path)]):
            no_gain = module.compare(["serial0", "serial1"], ["prefetch0", "prefetch1"], v7, "separate",
                                     expected_sentinels=0, serial_phase=phase, prefetch_phase=phase)
        self.assertTrue(no_gain["mechanics_pass"])
        self.assertFalse(no_gain["mirror_speed_pass"])
        self.assertFalse(no_gain["pass"])

        # A receipt/NPZ hash mismatch is a mechanics failure even at passing speed.
        for board, ids in enumerate((all_ids[:20], all_ids[20:])):
            worker(board, ids, False, "serial", 100.)
            worker(board, ids, True, "prefetch", 80.)
        pair_loads["prefetch0"] = (*pair_loads["prefetch0"][:2], "b" * 64)
        with patch.object(module, "load_worker", side_effect=lambda path: pair_loads[str(path)]):
            bad_hash = module.compare(["serial0", "serial1"], ["prefetch0", "prefetch1"], v7, "separate",
                                      expected_sentinels=0, serial_phase=phase, prefetch_phase=phase)
        self.assertFalse(bad_hash["mechanics_pass"])
        self.assertTrue(bad_hash["mirror_speed_pass"])


if __name__ == "__main__":
    unittest.main()
