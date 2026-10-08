"""Dependency-free qualification for the source-compatible ConvNeXt input policy."""
import math

INPUT_POLICY_VERSION = "convnext-partial-series-v2"
TRAIN_STUDIES = 64
STRESS_CASES = frozenset({"healthy_missing_planes", "usable_partial_volume",
    "exception_masked_without_retry", "short_masked_without_retry", "all_empty_fails",
    "swallowed_memory_still_fatal", "swallowed_timeout_still_fatal"})


def require_reader_rehearsal(report, entry_sha256, source_hashes, weights=None):
    """Reject old/synthetic-incomplete receipts; allowed partial inputs are counted."""
    expected_names = {f"cnxt_v0_fold{i}.pt" for i in range(3)}
    actual_weights = report.get("weight_hashes", {})
    stress = report.get("stress_checks", {})
    valid_digest = lambda value: (isinstance(value, str) and len(value) == 64 and
                                   all(c in "0123456789abcdef" for c in value))
    if (report.get("status") != "PASS" or report.get("rehearsal") is not True or
            report.get("entry_sha256") != entry_sha256 or report.get("source_hashes") != source_hashes or
            report.get("input_policy_version") != INPUT_POLICY_VERSION or
            report.get("train_studies_requested") != TRAIN_STUDIES or report.get("test_studies") != 3 or
            report.get("studies") != TRAIN_STUDIES + 3 or not valid_digest(report.get("sample_fingerprint")) or
            set(actual_weights) != expected_names or not all(valid_digest(v) for v in actual_weights.values()) or
            (weights is not None and actual_weights != weights)):
        raise ValueError("Matching expanded ConvNeXt rehearsal/source/weight contract required")
    if (stress.get("status") != "PASS" or stress.get("uses_raw_data") is not False or
            stress.get("uses_labels") is not False or stress.get("cases") != dict.fromkeys(STRESS_CASES, "PASS")):
        raise ValueError("ConvNeXt production-path stress checks did not pass")
    if (any(report.get(key) != 0 for key in ("zero_studies", "resource_errors")) or
            any(not isinstance(report.get(key), (int, float)) or not math.isfinite(report[key]) or
                not 0 <= report[key] <= 1e-6 for key in ("repeat_max_abs_error", "cache_parity_max_abs_error")) or
            not 0 < report.get("elapsed_seconds", math.inf) <= 3500):
        raise ValueError("ConvNeXt rehearsal integrity/parity/runtime guards failed")
    for key in ("decode_failures", "partial_series", "failed_series", "short_series", "zero_decoded_series",
                "missing_slots", "selected_series"):
        if not isinstance(report.get(key), int) or isinstance(report[key], bool) or report[key] < 0:
            raise ValueError("ConvNeXt aggregate input counters missing or invalid")
    return actual_weights
