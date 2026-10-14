"""Prospective A-prime evidence contract; never a model runner or qualifier.

Only stdlib imports on the default path. Bound evidence is checked, not trusted
because it says PASS. The existing base verifier is imported only on request.
Arm arithmetic, projection and scientific replay are intentionally not invented.
"""
import csv
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import re
from knee.public_s6_qualification import LABELS, UID

SCHEMA_VERSION = 1
BASE_MODULE = "knee.public_r7_recovery_entry"
BASE_PROJECTION = BASE_MODULE + ".projection"
GATE_SECONDS = 8.3 * 3600
MAX_ELAPSED_SECONDS = 9 * 3600
R7_MAX_OUTPUT_ATOL = 1e-3
ARM_TARGETS = {"ax-crop": ("PF OA",),
               "sag-crop": ("Lateral Meniscus", "ACL", "Medial Meniscus")}
PLANES = {"ax-crop": "axial", "sag-crop": "sagittal"}
SHA = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceError(ValueError):
    """Stable code only: errors must not disclose private IDs or report text."""


def require(condition, code):
    if not condition:
        raise EvidenceError(code)


def number(value, code, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), code)
    if positive:
        require(value > 0, code)
    return value


def digest(data):
    return hashlib.sha256(data).hexdigest()


def decode_json(data):
    def invalid_constant(_):
        raise EvidenceError("NONFINITE_JSON")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    try:
        return json.loads(data, parse_constant=invalid_constant,
                          object_pairs_hook=unique_keys)
    except (ValueError, UnicodeError) as exc:
        if isinstance(exc, EvidenceError):
            raise
        raise EvidenceError("INVALID_JSON") from None


class Artifacts:
    """Snapshot hash-bound files and recheck them before returning a result."""

    def __init__(self, root, declarations):
        require(isinstance(declarations, dict) and declarations, "ARTIFACT_MAP_REQUIRED")
        self.data, self.paths, self.shas = {}, {}, {}
        for key, spec in declarations.items():
            require(isinstance(key, str) and isinstance(spec, dict), "ARTIFACT_DECLARATION")
            require(set(spec) == {"path", "sha256"}, "ARTIFACT_DECLARATION")
            require(isinstance(spec["path"], str) and spec["path"], "ARTIFACT_PATH")
            require(isinstance(spec["sha256"], str) and SHA.fullmatch(spec["sha256"]), "ARTIFACT_SHA")
            path = (root / spec["path"]).resolve()
            try:
                data = path.read_bytes()
            except OSError:
                raise EvidenceError("ARTIFACT_UNREADABLE") from None
            require(digest(data) == spec["sha256"], "ARTIFACT_TAMPERED")
            self.data[key], self.paths[key], self.shas[key] = data, path, spec["sha256"]

    def get(self, key):
        require(isinstance(key, str) and key in self.data, "ARTIFACT_REFERENCE_REQUIRED")
        return self.data[key]

    def json(self, key):
        return decode_json(self.get(key))

    def sha(self, key):
        self.get(key)
        return self.shas[key]

    def unchanged(self):
        for key, path in self.paths.items():
            try:
                actual = digest(path.read_bytes())
            except OSError:
                raise EvidenceError("ARTIFACT_CHANGED_DURING_CHECK") from None
            require(actual == self.shas[key], "ARTIFACT_CHANGED_DURING_CHECK")


def rows(artifacts, key, header=None, ids=None):
    try:
        result = list(csv.reader(io.StringIO(artifacts.get(key).decode("utf-8-sig"), newline=""), strict=True))
    except (UnicodeError, csv.Error):
        raise EvidenceError("INVALID_CSV") from None
    require(result and len(result[0]) == 13 and len(set(result[0])) == 13, "CSV_HEADER")
    require(all(len(row) == 13 for row in result), "CSV_WIDTH")
    if header is not None:
        require(result[0] == header, "OFFICIAL_HEADER_ORDER")
    actual_ids = [row[0] for row in result[1:]]
    require(actual_ids and all(actual_ids) and len(set(actual_ids)) == len(actual_ids), "CSV_DUPLICATE_OR_EMPTY_ID")
    if ids is not None:
        require(actual_ids == ids, "CSV_STUDY_ORDER_COVERAGE")
    for row in result[1:]:
        for token in row[1:]:
            try:
                value = float(token)
            except ValueError:
                raise EvidenceError("CSV_PROBABILITY") from None
            require(math.isfinite(value) and 0 <= value <= 1, "CSV_PROBABILITY")
    return result


def declarations(packet, header):
    components = packet.get("components")
    require(isinstance(components, list) and components, "COMPONENTS_REQUIRED")
    used, names = set(), set()
    for component in components:
        require(isinstance(component, dict), "COMPONENT_DECLARATION")
        name = component.get("name")
        require(name in ARM_TARGETS and name not in names, "COMPONENT_NAME")
        names.add(name)
        targets = component.get("targets")
        require(isinstance(targets, list) and targets and len(set(targets)) == len(targets), "COMPONENT_TARGETS")
        require(targets[0] == ARM_TARGETS[name][0] and set(targets) <= set(ARM_TARGETS[name]), "PRIMARY_TARGET_ADMISSION")
        require(set(targets) <= set(header[1:]) and not used.intersection(targets), "DISJOINT_OFFICIAL_TARGETS")
        require(type(component.get("weight")) in (int, float) and component["weight"] == .3, "FIXED_COMPONENT_WEIGHT")
        used.update(targets)
    return components, used


def component_evidence(component, artifacts, ids, candidate):
    """Check supplied receipts; independent scientific/worker replay stays blocked."""
    name, targets = component["name"], component["targets"]
    handover = artifacts.json(component.get("handover"))
    require(type(handover.get("schema_version")) is int and handover["schema_version"] == 1 and handover.get("arm") == name,
            "OWNER_HANDOVER_SCHEMA")
    require(handover.get("mode") == "full_data" and handover.get("input_kind") == "images",
            "FULL_DATA_IMAGE_ONLY_HANDOVER")
    for field in ("source", "reader_config", "weights", "dataset_manifest", "initializer_provenance",
                  "swa_registration", "evaluation_inputs", "arithmetic_contract", "worker_interface", "repeat_policy"):
        artifacts.get(handover.get(field))
    require(SHA.fullmatch(str(handover.get("tensor_digest", ""))), "TENSOR_DIGEST")
    require(handover.get("supported_targets") == targets, "HANDOVER_TARGETS")
    require(handover.get("gate_sha256") == artifacts.sha(component.get("gates")), "GATE_HANDOVER_BINDING")
    registration = artifacts.json(handover["swa_registration"])
    dataset = artifacts.json(handover["dataset_manifest"])
    require(type(dataset.get("study_count")) is int and dataset["study_count"] > 0
            and registration.get("study_count") == dataset["study_count"]
            and registration.get("full_data") is True, "FULL_DATA_REGISTRATION")
    epochs = registration.get("fixed_swa_epochs")
    require(isinstance(epochs, list) and len(epochs) == 3 and len(set(epochs)) == 3
            and all(type(epoch) is int and epoch >= 0 for epoch in epochs), "FIXED_SWA_REGISTRATION")
    require(registration.get("weights_sha256") == artifacts.sha(handover["weights"]), "FULL_DATA_WEIGHT_BINDING")
    evaluation = artifacts.json(handover["evaluation_inputs"])
    for field in ("predictions", "masks", "ids", "split", "labels", "teacher_predictions", "control_predictions", "best_epoch_predictions"):
        artifacts.get(evaluation.get(field))
    require(evaluation.get("rank_weight") == .3, "EVALUATION_FIXED_WEIGHT")
    gates = artifacts.json(component["gates"])
    require(gates.get("evaluation_sha256") == artifacts.sha(handover["evaluation_inputs"]), "GATE_EVALUATION_BINDING")
    require(set(gates.get("targets", {})) == set(targets), "PER_TARGET_GATES_REQUIRED")
    screen = .0017 if candidate == "A_prime" else .0072
    for index, target in enumerate(targets):
        g = gates["targets"][target]
        delta = number(g.get("delta"), "SCIENTIFIC_DELTA")
        require(number(g.get("ci_lower"), "SCIENTIFIC_CI") > 0, "POSITIVE_CI_GATE")
        require(number(g.get("transfer_delta"), "TRANSFER_DELTA") >= 0, "TRANSFER_GATE")
        if index == 0:
            require(delta >= .010, "PRIMARY_DELTA_GATE")
            require(number(g.get("arm_auc"), "ARM_AUC") >= .80 and g["arm_auc"] <= 1, "ARM_ALONE_GATE")
            require(number(g.get("swa_delta"), "SWA_DELTA") > 0
                    and number(g.get("best_epoch_delta"), "BEST_EPOCH_DELTA") > 0, "EPOCH_SIGN_GATE")
            crop = number(g.get("crop_blend_auc"), "CROP_AUC")
            full = number(g.get("full_blend_auc"), "FULL_AUC")
            require(0 <= full < crop <= 1, "CROP_CONTROL_GATE")
            require(number(g.get("transfer_ci_lower"), "INTEGRATION_CI") >= screen, "ENTRY_INTEGRATION_SCREEN")
    policy = artifacts.json(component.get("missing_plane_policy"))
    require(policy == {"arm": name, "plane": PLANES[name], "unavailable_reason": "missing_plane",
                       "output": "preserve_base_strings", "other_errors": "fatal"}, "MISSING_PLANE_POLICY")
    records = artifacts.json(component.get("records"))
    require(isinstance(records, list) and len(records) == 39
            and all(isinstance(r, dict) for r in records)
            and [r.get("study_id") for r in records] == ids, "COMPONENT_ALL39_ORDER_COVERAGE")
    unavailable = set()
    for ordinal, record in enumerate(records):
        require(record.get("error") is None, "NEW_ARM_FAILURE_FATAL")
        require(record.get("input_kind") == "images", "IMAGE_ONLY_RECORD")
        mask = record.get("plane_present_mask")
        require(isinstance(mask, list) and mask and all(type(value) is bool for value in mask), "PLANE_MASK_REQUIRED")
        for field in ("input_sha256", "state_before", "state_after"):
            require(SHA.fullmatch(str(record.get(field, ""))), "RECORD_HASH_REQUIRED")
        require(record["state_before"] == record["state_after"] == handover["tensor_digest"], "COMPONENT_STATE_MUTATION")
        if record.get("state") == "unavailable":
            require(record.get("reason") == "missing_plane" and record.get("plane") == PLANES[name]
                    and record.get("probabilities") is None and not any(mask), "UNDOCUMENTED_UNAVAILABLE_STATE")
            unavailable.add(ordinal)
        else:
            require(record.get("state") == "present" and record.get("reason") is None and any(mask), "UNKNOWN_COMPONENT_STATE")
            probabilities = record.get("probabilities")
            require(isinstance(probabilities, list) and len(probabilities) == 12, "COMPONENT_PROBABILITIES")
            require(all(0 <= number(p, "COMPONENT_PROBABILITIES") <= 1 for p in probabilities), "COMPONENT_PROBABILITIES")
            canonical = json.dumps(probabilities, separators=(",", ":"), allow_nan=False).encode()
            repeat = record.get("repeat_probabilities")
            require(isinstance(repeat, list) and len(repeat) == 12
                    and all(0 <= number(p, "REPEAT_PROBABILITIES") <= 1 for p in repeat), "REPEAT_PROBABILITIES")
            require(record.get("repeat_input_sha256") == record["input_sha256"], "REPEAT_INPUT_HASH")
            repeat_canonical = json.dumps(repeat, separators=(",", ":"), allow_nan=False).encode()
            require(record.get("output_sha256") == digest(canonical)
                    and record.get("repeat_output_sha256") == digest(repeat_canonical), "ALL39_REPEAT_OUTPUT_HASH")
    return unavailable


def resources(packet, artifacts):
    runtime = packet.get("runtime", {})
    measurement = artifacts.json(runtime.get("measurement"))
    require(measurement.get("scope") == "combined_same_session", "COMBINED_MEASUREMENT_REQUIRED")
    start = number(measurement.get("start_seconds"), "TIMING_START")
    end = number(measurement.get("end_seconds"), "TIMING_END")
    elapsed = number(measurement.get("elapsed_seconds"), "TIMING_ELAPSED", True)
    require(math.isclose(end - start, elapsed, rel_tol=0, abs_tol=1e-6), "WALL_TIMING_RECONCILIATION")
    segments = measurement.get("segments")
    require(isinstance(segments, list) and segments, "FULL_WALL_SEGMENTS_REQUIRED")
    cursor = start
    for segment in segments:
        require(isinstance(segment.get("name"), str) and segment["name"], "TIMING_SEGMENT_NAME")
        a = number(segment.get("start_seconds"), "SEGMENT_START")
        b = number(segment.get("end_seconds"), "SEGMENT_END")
        require(math.isclose(a, cursor, rel_tol=0, abs_tol=1e-6) and b > a, "FULL_WALL_SEGMENT_COVERAGE")
        cursor = b
    require(math.isclose(cursor, end, rel_tol=0, abs_tol=1e-6), "FULL_WALL_SEGMENT_COVERAGE")
    for field in ("host_rss_bytes", "cuda_bytes", "output_disk_bytes"):
        peak = number(measurement.get("peaks", {}).get(field), "MEASURED_RESOURCE_REQUIRED", True)
        limit = number(measurement.get("limits", {}).get(field), "RESOURCE_LIMIT_REQUIRED", True)
        require(peak <= limit, "RESOURCE_LIMIT_EXCEEDED")
    require(number(measurement.get("minimum_disk_free_bytes"), "DISK_FREE_REQUIRED", True) > 0, "DISK_FREE_REQUIRED")
    policy = artifacts.json(runtime.get("combined_projection_policy"))
    require(policy.get("base_function") == BASE_PROJECTION
            and policy.get("measurement_scope") == "combined_same_session"
            and policy.get("owner_handover") is True, "EXPLICIT_COMBINED_PROJECTION_POLICY")
    require(isinstance(policy.get("formula"), str) and policy["formula"], "COMBINED_PROJECTION_FORMULA_REQUIRED")
    artifacts.get(policy.get("evaluator_source"))
    result = artifacts.json(runtime.get("combined_projection_result"))
    require(result.get("policy_sha256") == artifacts.sha(runtime["combined_projection_policy"])
            and result.get("measurement_sha256") == artifacts.sha(runtime["measurement"])
            and result.get("base_projection_sha256") == artifacts.sha(packet["base"].get("projection_result")), "PROJECTION_BINDING")
    projected = number(result.get("conservative_seconds"), "PROJECTION_SECONDS", True)
    require(projected <= GATE_SECONDS, "COMBINED_8_3_HOUR_GATE")
    require(elapsed <= MAX_ELAPSED_SECONDS, "COMBINED_9_HOUR_RUNTIME_LIMIT")
    return projected


def numeric_parity(left, right, atol):
    """Compare actual nested numerical intermediates, never their fingerprints."""
    if isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right) and left, "PREFETCH_INTERMEDIATE_SHAPE")
        for a, b in zip(left, right):
            numeric_parity(a, b, atol)
    else:
        require(abs(number(left, "PREFETCH_FINITE_VALUES")
                    - number(right, "PREFETCH_FINITE_VALUES")) <= atol, "PREFETCH_ACTUAL_VALUE_PARITY")


def prefetch_evidence(spec, artifacts, ids, runtime):
    """Prospective paired-output/Astra packet; owner acceptance is still needed."""
    require(isinstance(spec, dict), "PREFETCH_PACKET_REQUIRED")
    policy = artifacts.json(spec.get("parity_policy"))
    require(policy.get("owner_handover") is True, "PREFETCH_BOUND_PARITY_POLICY")
    output_atol = number(policy.get("output_atol"), "PREFETCH_OUTPUT_TOLERANCE")
    tolerances = policy.get("intermediate_atol")
    require(0 <= output_atol <= R7_MAX_OUTPUT_ATOL and isinstance(tolerances, dict) and tolerances,
            "PREFETCH_EXPLICIT_FROZEN_OUTPUT_TOLERANCE")
    for atol in tolerances.values():
        require(number(atol, "PREFETCH_INTERMEDIATE_TOLERANCE") >= 0, "PREFETCH_EXPLICIT_TOLERANCES")
    serial, concurrent = artifacts.json(spec.get("serial")), artifacts.json(spec.get("prefetch"))
    for evidence in (serial, concurrent):
        records = evidence.get("records")
        require(isinstance(records, list) and len(records) == 39
                and [r.get("study_id") for r in records] == ids, "PREFETCH_ACTUAL_ALL39_REQUIRED")
        for field in ("assets_sha256", "assignment_sha256", "compiled_control_sha256", "device_identity_sha256"):
            require(SHA.fullmatch(str(evidence.get(field, "")))
                    and evidence[field] == serial.get(field), "PREFETCH_PAIRED_BINDING")
        require(evidence.get("input_kind") == "images", "PREFETCH_IMAGE_ONLY")
    for a, b in zip(serial["records"], concurrent["records"]):
        require(SHA.fullmatch(str(a.get("input_sha256", "")))
                and a["input_sha256"] == b.get("input_sha256"), "PREFETCH_INPUT_PARITY")
        require(isinstance(a.get("outputs"), list) and len(a["outputs"]) == 12
                and isinstance(b.get("outputs"), list) and len(b["outputs"]) == 12, "PREFETCH_ACTUAL_OUTPUTS_REQUIRED")
        require(all(0 <= number(p, "PREFETCH_PROBABILITIES") <= 1 for p in a["outputs"] + b["outputs"]), "PREFETCH_PROBABILITIES")
        numeric_parity(a["outputs"], b["outputs"], output_atol)
        require(isinstance(a.get("intermediates"), dict) and isinstance(b.get("intermediates"), dict)
                and set(a["intermediates"]) == set(b["intermediates"]) == set(tolerances),
                "PREFETCH_ACTUAL_INTERMEDIATES_REQUIRED")
        for field, atol in tolerances.items():
            numeric_parity(a["intermediates"][field], b["intermediates"][field], atol)
    requirements = {
        "ownership": {"scratch_gate_scope": "global_quiescent", "single_producer": True, "reset_barrier_count": 39},
        "shutdown": {"live_threads_after_teardown": 0, "cancellation_aware": True,
                     "baseexception_preserved": True, "unknown_cuda_error": "fatal"},
        "cuda_precision": {"warmup": "serial", "device_explicit": True, "stream_explicit": True,
                           "auxiliary_autocast": False, "handoff": "completed_cpu_only"},
        "paired_actual39": {"study_count": 39, "compiled_control": "identical", "input_hash_only": False},
        "wall_and_resources": {},
    }
    conditions = spec.get("astra_conditions")
    require(isinstance(conditions, dict) and set(conditions) == set(requirements), "ALL_ASTRA_CONDITIONS_REQUIRED")
    for name, expected in requirements.items():
        condition = artifacts.json(conditions[name])
        require(condition.get("condition") == name, "ASTRA_CONDITION_IDENTITY")
        references = condition.get("evidence_artifacts")
        require(isinstance(references, list) and references, "ASTRA_CONDITION_EVIDENCE_REQUIRED")
        for key in references:
            artifacts.get(key)
        observations = condition.get("observations", {})
        require(all(key in observations and type(observations[key]) is type(value)
                    and observations[key] == value for key, value in expected.items()), "ASTRA_CONDITION_OBSERVATIONS")
    wall = artifacts.json(conditions["wall_and_resources"])["observations"]
    serial_time = number(wall.get("serial_elapsed_seconds"), "PREFETCH_SERIAL_WALL", True)
    paired_time = number(wall.get("prefetch_elapsed_seconds"), "PREFETCH_PAIRED_WALL", True)
    actual_time = artifacts.json(runtime["measurement"])["elapsed_seconds"]
    require(paired_time == actual_time and paired_time < serial_time, "PREFETCH_ACTUAL_WALL_IMPROVEMENT")


def base_proof(packet, artifacts):
    """No simplified base proof and no caller-provided verification callback."""
    base = packet["base"]
    receipt, binding = artifacts.json(base.get("receipt")), artifacts.json(base.get("binding"))
    inputs = artifacts.json(base.get("projection_inputs"))
    require(set(inputs) == {"cfg", "phases", "reports", "setup", "merge_seconds"}, "BASE_PROJECTION_INPUTS")
    expected = artifacts.json(base.get("projection_result"))
    try:
        module = importlib.import_module(BASE_MODULE)
    except ImportError:
        raise EvidenceError("BASE_VERIFIER_UNAVAILABLE") from None
    try:
        module.require_rehearsal(receipt, binding, artifacts.paths[base["csv"]])
        actual = module.projection(**inputs)
    except Exception:
        raise EvidenceError("EXISTING_BASE_VERIFIER_REJECTED") from None
    require(actual == expected == receipt.get("projection"), "FROZEN_BASE_PROJECTION_REPLAY")


def check_packet(path, *, verify_base=False):
    """Return aggregate evidence checks and blockers; never entry qualification.

    preparation packets declare pending owner artifacts; acceptance packets must
    supply all listed evidence. Even a complete real packet requires independent
    scientific, arm-worker, rank and combined-projection replay by the owner.
    """
    path = Path(path).resolve()
    try:
        packet_bytes = path.read_bytes()
    except OSError:
        raise EvidenceError("PACKET_UNREADABLE") from None
    packet = decode_json(packet_bytes)
    require(isinstance(packet, dict) and type(packet.get("schema_version")) is int
            and packet["schema_version"] == SCHEMA_VERSION, "PACKET_SCHEMA_VERSION")
    require(packet.get("purpose") in ("preparation", "acceptance"), "PACKET_PURPOSE")
    require(packet.get("evidence_kind") in ("synthetic", "real"), "EVIDENCE_KIND_REQUIRED")
    require(packet.get("candidate") in ("A_prime", "B1_prime"), "CANDIDATE_REQUIRED")
    artifacts = Artifacts(path.parent, packet.get("artifacts"))
    official = rows(artifacts, packet.get("official_sample"), [UID] + LABELS)
    registry = artifacts.json(packet.get("study_registry"))
    ids = registry.get("study_ids")
    require(isinstance(ids, list) and len(ids) == 39 and all(isinstance(i, str) and i for i in ids)
            and len(set(ids)) == 39, "FROZEN_ALL39_REGISTRY")
    visible_ids = [row[0] for row in official[1:]]
    require(len(visible_ids) == 3 and registry.get("visible_ids") == visible_ids
            and set(visible_ids) <= set(ids), "OFFICIAL_VISIBLE_REGISTRY")
    components, changed = declarations(packet, official[0])
    checks = ["artifact_sha256", "official_schema_and_registry", "fixed_disjoint_target_declarations"]
    blockers = ["SCIENTIFIC_EVIDENCE_REPLAY_NOT_DELIVERED", "ARM_WORKER_AND_RANK_REPLAY_NOT_DELIVERED",
                "ARM_REPEAT_TOLERANCE_REPLAY_NOT_DELIVERED", "COMBINED_PROJECTION_REPLAY_NOT_DELIVERED"]
    if packet["evidence_kind"] == "synthetic":
        blockers.append("SYNTHETIC_EVIDENCE_CANNOT_QUALIFY")
    if packet["purpose"] == "preparation":
        require(not verify_base, "BASE_VERIFY_REQUIRES_ACCEPTANCE_PACKET")
        pending = packet.get("pending")
        require(isinstance(pending, list) and all(isinstance(x, str) and x for x in pending)
                and {"worker_interface", "rank_and_missing_plane_arithmetic", "combined_projection_policy"} <= set(pending),
                "PREPARATION_BLOCKERS_REQUIRED")
        blockers.extend("PENDING_" + item.upper() for item in pending)
    else:
        base = packet.get("base", {})
        for field in ("receipt", "binding", "csv", "projection_inputs", "projection_result"):
            artifacts.get(base.get(field))
        baseline = rows(artifacts, packet.get("base_predictions"), official[0], ids)
        combined = rows(artifacts, packet.get("combined_predictions"), official[0], ids)
        final = rows(artifacts, packet.get("final_csv"), official[0], visible_ids)
        unavailable = {}
        for component in components:
            absent = component_evidence(component, artifacts, ids, packet["candidate"])
            for target in component["targets"]:
                unavailable[target] = absent
        for ordinal, (before, after) in enumerate(zip(baseline[1:], combined[1:])):
            for column, target in enumerate(official[0][1:], 1):
                if target not in changed or ordinal in unavailable[target]:
                    require(before[column] == after[column], "PROTECTED_OR_MISSING_COLUMN_CHANGED")
        by_id = {row[0]: row for row in combined[1:]}
        require(all(row == by_id[row[0]] for row in final[1:]), "FINAL_VISIBLE_STRING_PARITY")
        projected = resources(packet, artifacts)
        if packet.get("prefetch") is not None:
            prefetch_evidence(packet["prefetch"], artifacts, ids, packet["runtime"])
            checks.append("actual39_prefetch_parity_and_all_astra_receipt_structure")
            blockers.append("PREFETCH_ASTRA_OWNER_ACCEPTANCE_NOT_REPLAYED")
            blockers.append("PREFETCH_INTERMEDIATE_POLICY_NOT_ACCEPTED")
        if verify_base:
            base_proof(packet, artifacts)
            checks.append("existing_base_rehearsal_and_projection")
        else:
            blockers.append("EXISTING_BASE_VERIFIER_NOT_RUN")
        checks.extend(["component_handover_and_gate_receipt_structure", "all39_image_only_state_and_repeat_hashes",
                       "protected_csv_strings_and_visible_output", "bound_combined_resource_and_projection_receipts"])
    artifacts.unchanged()
    require(path.read_bytes() == packet_bytes, "PACKET_CHANGED_DURING_CHECK")
    return {"schema_version": SCHEMA_VERSION, "packet_sha256": digest(packet_bytes),
            "status": "PREPARATION_ONLY", "purpose": packet["purpose"], "evidence_kind": packet["evidence_kind"],
            "entry_qualified": False, "scientific_qualified": False, "checks": checks,
            "blockers": sorted(set(blockers)), "study_count": 39, "component_count": len(components),
            "conservative_seconds": projected if packet["purpose"] == "acceptance" else None}
