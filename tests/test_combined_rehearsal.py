"""Synthetic contract regressions, with no real model or qualification claim."""
import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from knee.combined_rehearsal import BASE_PROJECTION, EvidenceError, check_packet, digest
from knee.public_s6_qualification import LABELS, UID

HEADER = [UID] + LABELS


class PacketFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / "packet.json"
        self.artifacts = {}
        self.ids = ["synthetic-%03d" % i for i in range(39)]
        self.state = "a" * 64
        self.csv_rows = [[sid] + ["0.250000000"] * 12 for sid in self.ids]
        self.put_csv("official", self.csv_rows[:3])
        self.put_json("registry", {"study_ids": self.ids, "visible_ids": self.ids[:3]})
        self.put_csv("baseline", self.csv_rows)
        self.put_csv("combined", self.csv_rows)
        self.put_csv("final", self.csv_rows[:3])
        self.put_csv("base_visible", self.csv_rows[:3])
        for key in ("source", "reader", "weights", "provenance", "arithmetic", "interface", "repeat_policy",
                    "predictions", "masks", "ids", "split", "labels", "teacher", "control", "best_epoch"):
            self.put(key, b"synthetic evidence\n")
        self.put_json("dataset", {"study_count": 10})
        self.put_json("swa", {"study_count": 10, "full_data": True, "fixed_swa_epochs": [1, 2, 3],
                              "weights_sha256": self.sha("weights")})
        self.put_json("evaluation", dict(zip(
            ("predictions", "masks", "ids", "split", "labels", "teacher_predictions", "control_predictions", "best_epoch_predictions"),
            ("predictions", "masks", "ids", "split", "labels", "teacher", "control", "best_epoch")), rank_weight=.3))
        self.gate = {"delta": .02, "ci_lower": .01, "transfer_delta": .003, "transfer_ci_lower": .002,
                     "arm_auc": .81, "swa_delta": .02, "best_epoch_delta": .015,
                     "crop_blend_auc": .92, "full_blend_auc": .91}
        self.put_json("gates", {"evaluation_sha256": self.sha("evaluation"), "targets": {"PF OA": self.gate}})
        self.handover = {"schema_version": 1, "arm": "ax-crop", "mode": "full_data", "input_kind": "images",
                         "source": "source", "reader_config": "reader", "weights": "weights", "dataset_manifest": "dataset",
                         "initializer_provenance": "provenance", "swa_registration": "swa", "evaluation_inputs": "evaluation",
                         "arithmetic_contract": "arithmetic", "worker_interface": "interface", "tensor_digest": self.state,
                         "repeat_policy": "repeat_policy",
                         "supported_targets": ["PF OA"], "gate_sha256": self.sha("gates")}
        self.put_json("handover", self.handover)
        self.put_json("missing", {"arm": "ax-crop", "plane": "axial", "unavailable_reason": "missing_plane",
                                  "output": "preserve_base_strings", "other_errors": "fatal"})
        self.records = []
        probabilities = [.25] * 12
        output_sha = digest(json.dumps(probabilities, separators=(",", ":")).encode())
        for sid in self.ids:
            self.records.append({"study_id": sid, "state": "present", "reason": None, "error": None,
                                 "input_kind": "images", "input_sha256": "b" * 64, "state_before": self.state,
                                 "repeat_input_sha256": "b" * 64,
                                 "state_after": self.state, "probabilities": probabilities,
                                 "repeat_probabilities": probabilities, "plane_present_mask": [True, False],
                                 "output_sha256": output_sha, "repeat_output_sha256": output_sha})
        self.put_json("records", self.records)
        self.projection = {"conservative_seconds": 1000}
        self.put_json("base_projection", self.projection)
        self.put_json("base_inputs", {"cfg": {}, "phases": {}, "reports": {}, "setup": 1, "merge_seconds": 1})
        self.put_json("base_receipt", {"status": "PASS", "projection": self.projection})
        self.put_json("base_binding", {"synthetic": True})
        self.measurement = {"scope": "combined_same_session", "start_seconds": 5, "end_seconds": 105,
                            "elapsed_seconds": 100, "segments": [{"name": "whole", "start_seconds": 5, "end_seconds": 105}],
                            "peaks": {"host_rss_bytes": 100, "cuda_bytes": 100, "output_disk_bytes": 100},
                            "limits": {"host_rss_bytes": 1000, "cuda_bytes": 1000, "output_disk_bytes": 1000},
                            "minimum_disk_free_bytes": 1000}
        self.put_json("measurement", self.measurement)
        self.put_json("projection_policy", {"base_function": BASE_PROJECTION, "measurement_scope": "combined_same_session",
                                            "owner_handover": True, "formula": "synthetic prospective formula",
                                            "evaluator_source": "source"})
        self.put_json("projection_result", {"conservative_seconds": 2000, "policy_sha256": self.sha("projection_policy"),
                                            "measurement_sha256": self.sha("measurement"),
                                            "base_projection_sha256": self.sha("base_projection")})
        self.component = {"name": "ax-crop", "targets": ["PF OA"], "weight": .3, "handover": "handover",
                          "gates": "gates", "missing_plane_policy": "missing", "records": "records"}
        self.packet = {"schema_version": 1, "purpose": "acceptance", "evidence_kind": "synthetic", "candidate": "A_prime",
                       "artifacts": self.artifacts, "official_sample": "official", "study_registry": "registry",
                       "components": [self.component], "base_predictions": "baseline", "combined_predictions": "combined",
                       "final_csv": "final", "base": {"receipt": "base_receipt", "binding": "base_binding", "csv": "base_visible",
                       "projection_inputs": "base_inputs", "projection_result": "base_projection"},
                       "runtime": {"measurement": "measurement", "combined_projection_policy": "projection_policy",
                                   "combined_projection_result": "projection_result"}}
        self.save()

    def put(self, key, data):
        path = self.root / (key + ".artifact")
        path.write_bytes(data)
        self.artifacts[key] = {"path": path.name, "sha256": digest(data)}

    def sha(self, key):
        return self.artifacts[key]["sha256"]

    def put_json(self, key, value):
        self.put(key, json.dumps(value, allow_nan=False).encode())

    def put_csv(self, key, records, header=HEADER):
        out = io.StringIO(newline="")
        csv.writer(out).writerows([header, *records])
        self.put(key, out.getvalue().encode())

    def save(self):
        self.path.write_text(json.dumps(self.packet))

    def rebind_gates(self, targets):
        self.put_json("gates", {"evaluation_sha256": self.sha("evaluation"), "targets": targets})
        self.handover["gate_sha256"] = self.sha("gates")
        self.put_json("handover", self.handover)

    def add_raptor_column_gate(self):
        self.put("raptor_column", b"synthetic Raptor fold0 predictions\n")
        evaluation = json.loads((self.root / "evaluation.artifact").read_text())
        evaluation["raptor_column_predictions"] = "raptor_column"
        self.put_json("evaluation", evaluation)
        self.gate["raptor_column_blend"] = {
            "reference": "raptor_fold0_column", "target": "Lateral Meniscus",
            "rank_weight": .3, "predictions_sha256": self.sha("raptor_column"),
            "ci_lower": .0017,
        }

    def add_prefetch(self):
        paired = {"input_kind": "images", "assets_sha256": self.state, "assignment_sha256": self.state,
                  "compiled_control_sha256": self.state, "device_identity_sha256": self.state,
                  "records": [{"study_id": sid, "input_sha256": self.state, "outputs": [.25] * 12,
                               "intermediates": {"logits": [[.1] * 12]}} for sid in self.ids]}
        self.put_json("serial", paired)
        self.put_json("prefetch", paired)
        self.put_json("parity_policy", {"owner_handover": True, "output_atol": .001, "intermediate_atol": {"logits": .001}})
        observations = {
            "ownership": {"scratch_gate_scope": "global_quiescent", "single_producer": True, "reset_barrier_count": 39},
            "shutdown": {"live_threads_after_teardown": 0, "cancellation_aware": True,
                         "baseexception_preserved": True, "unknown_cuda_error": "fatal"},
            "cuda_precision": {"warmup": "serial", "device_explicit": True, "stream_explicit": True,
                               "auxiliary_autocast": False, "handoff": "completed_cpu_only"},
            "paired_actual39": {"study_count": 39, "compiled_control": "identical", "input_hash_only": False},
            "wall_and_resources": {"serial_elapsed_seconds": 110, "prefetch_elapsed_seconds": 100},
        }
        conditions = {}
        for name, value in observations.items():
            self.put_json(name, {"condition": name, "observations": value, "evidence_artifacts": ["serial", "prefetch"]})
            conditions[name] = name
        self.packet["prefetch"] = {"serial": "serial", "prefetch": "prefetch", "parity_policy": "parity_policy",
                                   "astra_conditions": conditions}
        self.save()


class CombinedRehearsalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.f = PacketFixture(self.temporary.name)

    def rejected(self, code):
        self.f.save()
        with self.assertRaisesRegex(EvidenceError, "^" + code + "$"):
            check_packet(self.f.path)

    def test_synthetic_receipts_never_qualify_even_with_pass_flag(self):
        self.f.packet["status"] = "PASS"
        self.f.packet["entry_qualified"] = True
        self.f.save()
        result = check_packet(self.f.path)
        self.assertEqual(result["status"], "PREPARATION_ONLY")
        self.assertFalse(result["entry_qualified"])
        self.assertFalse(result["scientific_qualified"])
        self.assertIn("SYNTHETIC_EVIDENCE_CANNOT_QUALIFY", result["blockers"])

    def test_preparation_requires_explicit_undelivered_contracts(self):
        self.f.packet["purpose"] = "preparation"
        self.f.packet["pending"] = ["worker_interface", "rank_and_missing_plane_arithmetic", "combined_projection_policy"]
        self.f.save()
        self.assertEqual(check_packet(self.f.path)["status"], "PREPARATION_ONLY")
        self.f.packet["pending"].pop()
        self.rejected("PREPARATION_BLOCKERS_REQUIRED")

    def test_tampered_file_is_rejected_without_repin(self):
        (self.f.root / "weights.artifact").write_bytes(b"changed")
        self.rejected("ARTIFACT_TAMPERED")

    def test_wrong_header_and_study_order(self):
        self.f.put_csv("combined", self.f.csv_rows, HEADER[:1] + list(reversed(HEADER[1:])))
        self.rejected("OFFICIAL_HEADER_ORDER")
        self.f.put_csv("combined", list(reversed(self.f.csv_rows)))
        self.rejected("CSV_STUDY_ORDER_COVERAGE")

    def test_duplicate_and_missing_studies(self):
        self.f.put_csv("combined", self.f.csv_rows[:-1] + [self.f.csv_rows[0]])
        self.rejected("CSV_DUPLICATE_OR_EMPTY_ID")
        self.f.put_csv("combined", self.f.csv_rows[:-1])
        self.rejected("CSV_STUDY_ORDER_COVERAGE")

    def test_nonfinite_or_out_of_range_probabilities(self):
        for value in ("nan", "inf", "-0.1", "1.1", "not-a-number"):
            records = copy.deepcopy(self.f.csv_rows)
            records[0][1] = value
            self.f.put_csv("combined", records)
            self.rejected("CSV_PROBABILITY")

    def test_protected_columns_preserve_strings_not_only_floats(self):
        records = copy.deepcopy(self.f.csv_rows)
        records[0][1] = "0.25"
        self.f.put_csv("combined", records)
        self.rejected("PROTECTED_OR_MISSING_COLUMN_CHANGED")

    def test_missing_plane_preserves_base_target_strings(self):
        self.f.records[0].update(state="unavailable", reason="missing_plane", plane="axial", probabilities=None,
                                plane_present_mask=[False, False])
        self.f.put_json("records", self.f.records)
        self.f.save()
        self.assertFalse(check_packet(self.f.path)["entry_qualified"])
        records = copy.deepcopy(self.f.csv_rows)
        records[0][HEADER.index("PF OA")] = "0.3"
        self.f.put_csv("combined", records)
        self.rejected("PROTECTED_OR_MISSING_COLUMN_CHANGED")

    def test_fatal_and_unknown_unavailable_do_not_become_fallback(self):
        for error in ("resource", "programming", "unknown"):
            self.f.records[0]["error"] = error
            self.f.put_json("records", self.f.records)
            self.rejected("NEW_ARM_FAILURE_FATAL")
        self.f.records[0].update(error=None, state="unavailable", reason="empty_output", plane="axial", probabilities=None)
        self.f.put_json("records", self.f.records)
        self.rejected("UNDOCUMENTED_UNAVAILABLE_STATE")

    def test_all39_state_mutation_repeat_and_image_only(self):
        for field, value, code in (("state_after", "c" * 64, "COMPONENT_STATE_MUTATION"),
                                   ("repeat_output_sha256", "c" * 64, "ALL39_REPEAT_OUTPUT_HASH"),
                                   ("input_kind", "reports", "IMAGE_ONLY_RECORD")):
            records = copy.deepcopy(self.f.records)
            records[-1][field] = value
            self.f.put_json("records", records)
            self.rejected(code)

    def test_duplicate_component_records(self):
        self.f.put_json("records", self.f.records[:-1] + [self.f.records[0]])
        self.rejected("COMPONENT_ALL39_ORDER_COVERAGE")

    def test_missing_or_contradictory_plane_masks(self):
        for mask, code in ((None, "PLANE_MASK_REQUIRED"), ([0, 1], "PLANE_MASK_REQUIRED"),
                           ([False, False], "UNKNOWN_COMPONENT_STATE")):
            records = copy.deepcopy(self.f.records)
            records[0]["plane_present_mask"] = mask
            self.f.put_json("records", records)
            self.rejected(code)

    def test_actual_repeat_values_are_bound_independently_of_first_output(self):
        records = copy.deepcopy(self.f.records)
        records[-1]["repeat_probabilities"] = [.3] * 12
        self.f.put_json("records", records)
        self.rejected("ALL39_REPEAT_OUTPUT_HASH")

    def test_no_new_arm_numerical_repeat_tolerance_is_invented(self):
        records = copy.deepcopy(self.f.records)
        records[-1]["repeat_probabilities"] = [.25001] * 12
        records[-1]["repeat_output_sha256"] = digest(json.dumps(records[-1]["repeat_probabilities"], separators=(",", ":")).encode())
        self.f.put_json("records", records)
        self.f.save()
        result = check_packet(self.f.path)
        self.assertIn("ARM_REPEAT_TOLERANCE_REPLAY_NOT_DELIVERED", result["blockers"])
        self.assertFalse(result["entry_qualified"])

    def test_official_sample_cannot_redefine_target_names(self):
        self.f.put_csv("official", self.f.csv_rows[:3], HEADER[:1] + list(reversed(HEADER[1:])))
        self.rejected("OFFICIAL_HEADER_ORDER")

    def test_weights_and_target_admission(self):
        self.f.component["weight"] = .31
        self.rejected("FIXED_COMPONENT_WEIGHT")
        self.f.component["weight"] = .3
        self.f.component["targets"] = ["ACL"]
        self.rejected("PRIMARY_TARGET_ADMISSION")

    def test_unqualified_scientific_and_transfer_gates(self):
        changes = (("delta", .009, "PRIMARY_DELTA_GATE"), ("ci_lower", 0, "POSITIVE_CI_GATE"),
                   ("arm_auc", .79, "ARM_ALONE_GATE"), ("best_epoch_delta", -.01, "EPOCH_SIGN_GATE"),
                   ("full_blend_auc", .93, "CROP_CONTROL_GATE"), ("transfer_delta", -.001, "TRANSFER_GATE"),
                   ("transfer_ci_lower", .001, "ENTRY_INTEGRATION_SCREEN"))
        for field, value, code in changes:
            g = dict(self.f.gate, **{field: value})
            self.f.rebind_gates({"PF OA": g})
            self.rejected(code)

    def test_sagittal_optional_targets_need_primary_and_own_gates(self):
        targets = ["Lateral Meniscus", "ACL"]
        self.f.component.update(name="sag-crop", targets=targets)
        self.f.handover.update(arm="sag-crop", supported_targets=targets)
        self.f.put_json("missing", {"arm": "sag-crop", "plane": "sagittal", "unavailable_reason": "missing_plane",
                                    "output": "preserve_base_strings", "other_errors": "fatal"})
        self.f.add_raptor_column_gate()
        self.f.rebind_gates(dict.fromkeys(targets, self.f.gate))
        self.f.save()
        self.assertFalse(check_packet(self.f.path)["entry_qualified"])
        self.f.rebind_gates({targets[0]: self.f.gate, "ACL": dict(self.f.gate, ci_lower=0)})
        self.rejected("POSITIVE_CI_GATE")
        self.f.rebind_gates({targets[0]: dict(self.f.gate, delta=.009), "ACL": self.f.gate})
        self.rejected("PRIMARY_DELTA_GATE")
        self.f.component["targets"] = ["ACL"]
        self.rejected("PRIMARY_TARGET_ADMISSION")

    def sagittal_packet(self):
        self.f.component.update(name="sag-crop", targets=["Lateral Meniscus"])
        self.f.handover.update(arm="sag-crop", supported_targets=["Lateral Meniscus"])
        self.f.put_json("missing", {"arm": "sag-crop", "plane": "sagittal", "unavailable_reason": "missing_plane",
                                   "output": "preserve_base_strings", "other_errors": "fatal"})
        self.f.add_raptor_column_gate()
        self.f.rebind_gates({"Lateral Meniscus": self.f.gate})
        self.f.save()

    def test_a_prime_raptor_column_gate_at_boundary_stays_preparation(self):
        self.sagittal_packet()
        result = check_packet(self.f.path)
        self.assertEqual(result["status"], "PREPARATION_ONLY")
        self.assertFalse(result["scientific_qualified"])

    def test_a_prime_cannot_substitute_indistil_ci_for_host_column(self):
        self.sagittal_packet()
        for host in (None, dict(self.f.gate["raptor_column_blend"], reference="in-distil")):
            gate = dict(self.f.gate, raptor_column_blend=host)
            self.f.rebind_gates({"Lateral Meniscus": gate})
            self.rejected("RAPTOR_COLUMN_REFERENCE_REQUIRED")

    def test_a_prime_raptor_column_ci_below_boundary_is_rejected(self):
        self.sagittal_packet()
        gate = dict(self.f.gate, raptor_column_blend=dict(self.f.gate["raptor_column_blend"], ci_lower=.001699))
        self.f.rebind_gates({"Lateral Meniscus": gate})
        self.rejected("RAPTOR_COLUMN_A_PRIME_GATE")

    def test_a_prime_raptor_column_predictions_must_be_bound(self):
        self.sagittal_packet()
        gate = dict(self.f.gate, raptor_column_blend=dict(self.f.gate["raptor_column_blend"], predictions_sha256="c" * 64))
        self.f.rebind_gates({"Lateral Meniscus": gate})
        self.rejected("RAPTOR_COLUMN_PREDICTION_BINDING")

    def test_b1_prime_retains_its_screen_without_a_prime_host_gate(self):
        self.sagittal_packet()
        self.f.packet["candidate"] = "B1_prime"
        gate = dict(self.f.gate, transfer_ci_lower=.0072)
        gate.pop("raptor_column_blend")
        self.f.rebind_gates({"Lateral Meniscus": gate})
        self.f.save()
        self.assertEqual(check_packet(self.f.path)["status"], "PREPARATION_ONLY")
        self.f.rebind_gates({"Lateral Meniscus": dict(gate, transfer_ci_lower=.007199)})
        self.rejected("ENTRY_INTEGRATION_SCREEN")

    def test_missing_evaluation_mask_artifact_is_rejected(self):
        evaluation = json.loads((self.f.root / "evaluation.artifact").read_text())
        evaluation.pop("masks")
        self.f.put_json("evaluation", evaluation)
        self.f.rebind_gates({"PF OA": self.f.gate})
        self.rejected("ARTIFACT_REFERENCE_REQUIRED")

    def test_full_data_and_weights_registration(self):
        self.f.handover["mode"] = "fold0"
        self.f.put_json("handover", self.f.handover)
        self.rejected("FULL_DATA_IMAGE_ONLY_HANDOVER")
        self.f.handover["mode"] = "full_data"
        self.f.put_json("handover", self.f.handover)
        self.f.put_json("swa", {"study_count": 10, "full_data": True, "fixed_swa_epochs": [1, 2, 3], "weights_sha256": "c" * 64})
        self.rejected("FULL_DATA_WEIGHT_BINDING")

    def test_runtime_omission_resource_and_reconciliation(self):
        self.f.packet["runtime"].pop("combined_projection_policy")
        self.rejected("ARTIFACT_REFERENCE_REQUIRED")
        self.f.packet["runtime"]["combined_projection_policy"] = "projection_policy"
        self.f.measurement["peaks"]["cuda_bytes"] = 1001
        self.f.put_json("measurement", self.f.measurement)
        self.rejected("RESOURCE_LIMIT_EXCEEDED")
        self.f.measurement["peaks"]["cuda_bytes"] = 100
        self.f.measurement["segments"][0]["start_seconds"] = 6
        self.f.put_json("measurement", self.f.measurement)
        self.rejected("FULL_WALL_SEGMENT_COVERAGE")

    def test_projection_binding_and_8_3_hour_gate(self):
        result = {"conservative_seconds": 8.3 * 3600 + 1, "policy_sha256": self.f.sha("projection_policy"),
                  "measurement_sha256": self.f.sha("measurement"), "base_projection_sha256": self.f.sha("base_projection")}
        self.f.put_json("projection_result", result)
        self.rejected("COMBINED_8_3_HOUR_GATE")
        result["conservative_seconds"] = 2000
        result["measurement_sha256"] = "c" * 64
        self.f.put_json("projection_result", result)
        self.rejected("PROJECTION_BINDING")

    def test_elapsed_preserves_actual_nine_hour_ceiling(self):
        self.f.measurement.update(end_seconds=5 + 8.5 * 3600, elapsed_seconds=8.5 * 3600)
        self.f.measurement["segments"][0]["end_seconds"] = self.f.measurement["end_seconds"]
        self.f.put_json("measurement", self.f.measurement)
        self.f.put_json("projection_result", {"conservative_seconds": 2000, "policy_sha256": self.f.sha("projection_policy"),
                                              "measurement_sha256": self.f.sha("measurement"),
                                              "base_projection_sha256": self.f.sha("base_projection")})
        self.f.save()
        self.assertEqual(check_packet(self.f.path)["status"], "PREPARATION_ONLY")
        self.f.measurement.update(end_seconds=5 + 9 * 3600 + 1, elapsed_seconds=9 * 3600 + 1)
        self.f.measurement["segments"][0]["end_seconds"] = self.f.measurement["end_seconds"]
        self.f.put_json("measurement", self.f.measurement)
        self.f.put_json("projection_result", {"conservative_seconds": 2000, "policy_sha256": self.f.sha("projection_policy"),
                                              "measurement_sha256": self.f.sha("measurement"),
                                              "base_projection_sha256": self.f.sha("base_projection")})
        self.rejected("COMBINED_9_HOUR_RUNTIME_LIMIT")

    def test_existing_base_verifier_is_lazy_and_rejects(self):
        with patch("knee.combined_rehearsal.importlib.import_module", side_effect=AssertionError("must remain lazy")):
            check_packet(self.f.path)
        module = types.SimpleNamespace(require_rehearsal=lambda *args: (_ for _ in ()).throw(ValueError("private-ID")))
        with patch("knee.combined_rehearsal.importlib.import_module", return_value=module):
            with self.assertRaisesRegex(EvidenceError, "^EXISTING_BASE_VERIFIER_REJECTED$"):
                check_packet(self.f.path, verify_base=True)

    def test_base_projection_replays_existing_function_but_never_qualifies(self):
        calls = []
        module = types.SimpleNamespace(require_rehearsal=lambda *args: calls.append(args),
                                       projection=lambda **kwargs: self.f.projection)
        with patch("knee.combined_rehearsal.importlib.import_module", return_value=module):
            result = check_packet(self.f.path, verify_base=True)
        self.assertEqual(len(calls), 1)
        self.assertIn("existing_base_rehearsal_and_projection", result["checks"])
        self.assertFalse(result["entry_qualified"])
        module.projection = lambda **kwargs: {"conservative_seconds": 2001}
        with patch("knee.combined_rehearsal.importlib.import_module", return_value=module):
            with self.assertRaisesRegex(EvidenceError, "^FROZEN_BASE_PROJECTION_REPLAY$"):
                check_packet(self.f.path, verify_base=True)

    def test_prefetch_actual39_intermediates_and_all_astra_packet(self):
        self.f.add_prefetch()
        result = check_packet(self.f.path)
        self.assertIn("PREFETCH_ASTRA_OWNER_ACCEPTANCE_NOT_REPLAYED", result["blockers"])
        evidence = json.loads((self.f.root / "prefetch.artifact").read_text())
        evidence["records"][-1]["intermediates"]["logits"][0][0] = .5
        self.f.put_json("prefetch", evidence)
        self.rejected("PREFETCH_ACTUAL_VALUE_PARITY")

    def test_prefetch_fingerprints_or_visible_only_are_insufficient(self):
        self.f.add_prefetch()
        evidence = json.loads((self.f.root / "prefetch.artifact").read_text())
        evidence["records"] = evidence["records"][:3]
        self.f.put_json("prefetch", evidence)
        self.rejected("PREFETCH_ACTUAL_ALL39_REQUIRED")
        self.f.add_prefetch()
        self.f.packet["prefetch"]["astra_conditions"].pop("ownership")
        self.rejected("ALL_ASTRA_CONDITIONS_REQUIRED")

    def test_prefetch_cannot_widen_frozen_output_tolerance(self):
        self.f.add_prefetch()
        self.f.put_json("parity_policy", {"owner_handover": True, "output_atol": 100, "intermediate_atol": {"logits": .001}})
        self.rejected("PREFETCH_EXPLICIT_FROZEN_OUTPUT_TOLERANCE")

    def test_file_mutation_during_base_check_is_rejected(self):
        def mutate(*_):
            (self.f.root / "weights.artifact").write_bytes(b"late mutation")
        module = types.SimpleNamespace(require_rehearsal=mutate, projection=lambda **kwargs: self.f.projection)
        with patch("knee.combined_rehearsal.importlib.import_module", return_value=module):
            with self.assertRaisesRegex(EvidenceError, "^ARTIFACT_CHANGED_DURING_CHECK$"):
                check_packet(self.f.path, verify_base=True)

    def test_cli_preparation_exit_is_two_and_does_not_print_ids(self):
        script = Path(__file__).resolve().parents[1] / "scripts/check_combined_rehearsal.py"
        result = subprocess.run([sys.executable, str(script), str(self.f.path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "PREPARATION_ONLY")
        self.assertNotIn(self.f.ids[0], result.stdout)


if __name__ == "__main__":
    unittest.main()
