"""Small rehearsal adapter for the lane 3b R7 prefetch P3 pairing.

This module binds the archived v7 evaluation fingerprints to rehearsal config and
runs the candidate worker's serial-control and prefetch phases in separate output
directories. It does not change the worker or the frozen builders.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

V7_RECEIPT_SHA256 = "e10075cf8bf53d74b5a3750e31bd81d003e7714453d4ca55047d363d96d0cc07"
EXPECTED_STUDIES = 39
EXPECTED_FINGERPRINTS = 8


def reference_from_receipt(receipt: dict[str, Any], expected_studies: int = EXPECTED_STUDIES) -> dict[str, dict[str, Any]]:
    """Return the all-study fingerprint map, rejecting incomplete/duplicate ledgers."""
    workers = receipt.get("workers", {}).get("s6")
    if not isinstance(workers, list) or len(workers) != 2:
        raise ValueError("V7_R7_WORKERS")
    reference: dict[str, dict[str, Any]] = {}
    for worker in workers:
        records = worker.get("records")
        if not isinstance(records, list):
            raise ValueError("V7_R7_RECORDS")
        for row in records:
            study_id, fingerprints = row.get("id"), row.get("input")
            if not isinstance(study_id, str) or not study_id or study_id in reference:
                raise ValueError("V7_R7_STUDY_IDS")
            if not isinstance(fingerprints, dict) or len(fingerprints) != EXPECTED_FINGERPRINTS:
                raise ValueError("V7_R7_FINGERPRINT_SCHEMA")
            reference[study_id] = fingerprints
    if len(reference) != expected_studies:
        raise ValueError("V7_R7_STUDY_COUNT")
    return reference


def load_v7_input_reference(receipt_path: str | Path) -> dict[str, dict[str, Any]]:
    """Read only the pinned v7 receipt and bind its 39 × 8 evaluation fingerprints."""
    raw = Path(receipt_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != V7_RECEIPT_SHA256:
        raise ValueError("EXACT_V7_RECEIPT")
    try:
        receipt = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("V7_RECEIPT_JSON") from exc
    return reference_from_receipt(receipt)


def bind_evaluation_input_reference(
    cfg: dict[str, Any], receipt_path: str | Path
) -> dict[str, Any]:
    """Return a rehearsal-only config copy carrying the pinned v7 fingerprint map."""
    if cfg.get("mode") != "rehearsal":
        raise ValueError("EVALUATION_INPUT_PARITY_REHEARSAL_ONLY")
    bound = dict(cfg)
    reference = load_v7_input_reference(receipt_path)
    prior = bound.get("evaluation_input_reference")
    if prior is not None and prior != reference:
        raise ValueError("EVALUATION_INPUT_REFERENCE_CONFLICT")
    bound["evaluation_input_reference"] = reference
    return bound


def run_paired_r7_phases(
    cfg: dict[str, Any],
    serial_out: str | Path,
    prefetch_out: str | Path,
    *,
    launch: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run same-config serial control then prefetch R7 phases in fresh directories.

    Pass the already v7-bound rehearsal config. The output directories must exist
    and be empty, matching ``candidate.launch``'s resource-sampling contract.
    The returned phase receipts should be written alongside the run evidence and
    supplied to the P3 comparison script.
    """
    if cfg.get("mode") != "rehearsal":
        raise ValueError("R7_SERIAL_CONTROL_REHEARSAL_ONLY")
    if not isinstance(cfg.get("evaluation_input_reference"), dict):
        raise ValueError("EVALUATION_INPUT_REFERENCE_REQUIRED")
    if launch is None:
        from knee import public_r7_prefetch_candidate as candidate

        launch = candidate.launch

    serial_path, prefetch_path = Path(serial_out), Path(prefetch_out)
    if serial_path.resolve() == prefetch_path.resolve():
        raise ValueError("P3_OUTPUTS_MUST_BE_DISTINCT")
    for path in (serial_path, prefetch_path):
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError("P3_OUTPUT_DIRECTORY_MUST_BE_EMPTY")

    serial_cfg = {**cfg, "r7_prefetch": False}
    prefetch_cfg = {**cfg, "r7_prefetch": True}
    serial = launch(serial_cfg, serial_path, "s6")
    prefetched = launch(prefetch_cfg, prefetch_path, "s6")
    return {
        "serial": serial,
        "prefetch": prefetched,
        "serial_config": serial_cfg,
        "prefetch_config": prefetch_cfg,
        "paired_identity": {
            "assignment": cfg.get("assignment"),
            "boards": cfg.get("boards"),
            "evaluation_studies": len(cfg["evaluation_input_reference"]),
            "worker_module": "knee.public_r7_prefetch_candidate",
        },
    }
