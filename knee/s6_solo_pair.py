"""Validate and stage an unsealed S6 solo checkpoint pair."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

SOURCE_SHA256 = "6e87350d5ae092b7a889546ca13517303523bc469f7e611ffe945913434d22fe"
CONTROL_SHA256 = "0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12"
CONTROL_DATASET = "nartaa/rsna-knee-s6fall-weights"
V2_DATASET = "ozgung/knee-s6v2-v2all-20261016"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
COMMON = {"source_sha256": SOURCE_SHA256, "weights": {"A": 0, "B": 1},
          "embedded_a_source_sha256": "bcd864cbf349882e876959b49cb348cbb89b4c62fc204d4e4261049c5e3a79c5",
          "precision": "fp16", "views": "v4", "image_size": 320, "crop_size": 288,
          "slices": 3, "slice_spacing_mm": 3.5, "eval_stride": 1, "chunk_size": 192,
          "workers_per_gpu": 2, "recovery": "captured-publisher-controller-v1"}
ALIAS_RE = re.compile(r"E_prior_s[0-9]+_f[A-Za-z0-9_.-]*\.pt")
PENDING = ["CPU/state/constructor receipt", "GPU rehearsal", "complete training provenance", "final packet sealing"]


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _hash(value, field):
    _need(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"{field} must be lowercase SHA256")
    return value


def _file_identity(path):
    h, size = hashlib.sha256(), 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            size += len(block); h.update(block)
    return h.hexdigest(), size


def validate_pair(control, v2, *, source=None):
    """Validate descriptor structure; this does not authenticate descriptor claims."""
    _need(isinstance(control, dict) and isinstance(v2, dict), "descriptors must be objects")
    for name, d, variant, filename in (("control", control, "control", "E_prior_s42_fall.pt"),
                                        ("v2", v2, "v2", "E_prior_s42_v2all.pt")):
        _need(d.get("variant") == variant, f"{name} variant mismatch")
        _need(d.get("targets") == TARGETS, f"{name} canonical target order mismatch")
        _need(isinstance(d.get("dataset_slug"), str) and d["dataset_slug"], f"{name} dataset_slug required")
        _need(type(d.get("dataset_version")) is int and d["dataset_version"] > 0, f"{name} dataset_version must be positive integer")
        _hash(d.get("checkpoint_sha256"), f"{name}.checkpoint_sha256")
        _need(type(d.get("checkpoint_size")) is int and d["checkpoint_size"] > 0, f"{name} checkpoint_size must be positive")
        _need(d.get("checkpoint_filename") == filename, f"{name} checkpoint filename mismatch")
        _need(d.get("common") == COMMON, f"{name} shared source/A/recovery/inference contract mismatch")
    _need(control["checkpoint_sha256"] != v2["checkpoint_sha256"], "control and V2 checkpoint identities must differ")
    _need(control["checkpoint_sha256"] == CONTROL_SHA256 and control["dataset_slug"] == CONTROL_DATASET,
          "published control descriptor pin mismatch")
    _need(v2["dataset_slug"] == V2_DATASET, "V2 dataset mismatch")
    _need(isinstance(v2.get("runtime_alias"), str) and ALIAS_RE.fullmatch(v2["runtime_alias"]),
          "V2 alias must be safe basename matching E_prior_s*_f*.pt")
    if source is not None:
        _need(hashlib.sha256(source).hexdigest() == SOURCE_SHA256, "archived common source hash mismatch")
    return {"status": "UNSEALED_PREPARATION_ONLY", "seal_available": False, "pending": list(PENDING),
            "descriptor_hashes_verified_against_files": False, "published_control_descriptor_matches_pin": True,
            "source_bytes_verified": source is not None,
            "source_sha256": SOURCE_SHA256, "common_contract": copy.deepcopy(COMMON),
            "control": {k: control[k] for k in ("dataset_slug", "dataset_version", "checkpoint_sha256", "checkpoint_size", "checkpoint_filename")},
            "v2": {k: v2[k] for k in ("dataset_slug", "dataset_version", "checkpoint_sha256", "checkpoint_size", "checkpoint_filename")}
                  | {"runtime_alias": v2["runtime_alias"]}}


def stage_checkpoint(descriptor, checkpoint_path, fresh_model_dir):
    """Verify and copy one file; directory ownership is claimed exclusively.

    This checks bytes only. It does not deserialize or qualify a model.
    """
    src, dest = Path(checkpoint_path), Path(fresh_model_dir)
    _need(src.is_file() and not src.is_symlink(), "checkpoint must be a regular non-symlink file")
    _need(src.name == descriptor.get("checkpoint_filename"), "checkpoint source filename mismatch")
    alias = descriptor.get("runtime_alias", src.name)
    _need(isinstance(alias, str) and ALIAS_RE.fullmatch(alias), "unsafe or invalid runtime alias")
    expected_hash = _hash(descriptor.get("checkpoint_sha256"), "checkpoint_sha256")
    expected_size = descriptor.get("checkpoint_size")
    _need(type(expected_size) is int and expected_size > 0, "checkpoint_size must be positive")
    actual_hash, actual_size = _file_identity(src)
    _need((actual_hash, actual_size) == (expected_hash, expected_size), "checkpoint bytes do not match descriptor hash/size")
    _need(not dest.exists() and not dest.is_symlink(), "fresh model directory already exists")
    _need(dest.parent.is_dir(), "model directory parent must already exist")
    # mkdir must fail if another preparation claimed the destination after the
    # existence check. Never replace an existing (even empty) model directory.
    dest.mkdir()
    try:
        target = dest / alias
        shutil.copyfile(src, target)
        copied_hash, copied_size = _file_identity(target)
        _need((copied_hash, copied_size) == (actual_hash, actual_size), "staged copy identity mismatch")
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return {"status": "STAGED_UNSEALED", "source_file": src.name, "alias_file": alias,
            "source_sha256": actual_hash, "alias_sha256": copied_hash, "source_size": actual_size,
            "alias_size": copied_size, "identical_bytes": True, "model_directory": str(dest),
            "seal_available": False, "pending": list(PENDING)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("control", type=Path); p.add_argument("v2", type=Path)
    p.add_argument("--source", type=Path); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    plan = validate_pair(json.loads(a.control.read_text()), json.loads(a.v2.read_text()), source=a.source.read_bytes() if a.source else None)
    with a.output.open("x", encoding="utf-8") as f:
        f.write(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": plan["status"], "seal_available": False, "output": str(a.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
