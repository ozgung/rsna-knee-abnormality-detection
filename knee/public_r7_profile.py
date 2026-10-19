"""Frozen, dependency-free identity and asset checks for heliosli's R7 bundle run through compact_infer v4c (plain pass).

The pins are the ones already qualified for A-prime in knee.public_r7_recovery_entry (dataset
heliosli/rsna-knee-compact-r7cnxfsss version 1, bundle 8c7b666e27dc0e8f, manifest c4bb0f73..., stacker/spec
SHA256, trainer MD5, flip_tta False); the four fold checkpoints and side net are pinned by the manifest's own
SHA256 closure. The seven worker sources are byte-identical to R6's (same compact_infer/compact_train v4c).
Identity only: it does not qualify the reader, runtime, or a result.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType

from knee.public_r6_profile import SOURCE_SHA256, _checked_file, _verify_files
from knee.public_r7_recovery_entry import (R7_ASSET_SHA, R7_BUNDLE_ID, R7_DATASET, R7_DATASET_VERSION, R7_MANIFEST_SHA,
                                           R7_SOURCE_SHA, R7_TRAINER_MD5)

PROFILE_ID = "public-r7-8c7b666e-plain-v1"
BUNDLE_ID = R7_BUNDLE_ID
BUNDLE_TAG = "cnx_r7"
DATASET = R7_DATASET
DATASET_VERSION = R7_DATASET_VERSION
MANIFEST_SHA256 = R7_MANIFEST_SHA
TRAINER_MD5 = R7_TRAINER_MD5
FLIP_TTA = False
ASSET_SHA256 = MappingProxyType({
    "E_lrprior_s0_f0.pt": "ca8a76e024828793e0663e5a2a13461de17e820bed6789eafe3f155072d3612f",
    "E_lrprior_s0_f1.pt": "fb432cf544f218608dbaecf7a8dd3c8b6a26b34a3dfe2dc198172c3e222d5910",
    "E_lrprior_s0_f2.pt": "a9718ae111d6307492e724eb1767f0b5a77397d9a4d82c50c1e146b046523bf7",
    "E_lrprior_s0_f3.pt": "073c22c71a9946c1457e37e56fec145f073a921da106b55a10884da01a40fc94",
    "side_cls.pt": "ab623e2b619db4917639220fed15202e8c8aedd07c8e298a66a2afc0f153650a",
    **R7_ASSET_SHA,
})
# R7's compact_infer/compact_train/probe_dual pins (A-prime) equal R6's: one source closure serves both bundles.
if any(SOURCE_SHA256[k] != v for k, v in R7_SOURCE_SHA.items()):
    raise ValueError("R7_R6_SHARED_SOURCE_CLOSURE")


def validate_r7_assets(asset_dir: str | os.PathLike[str], *, source_dir: str | os.PathLike[str] | None = None) -> dict[str, object]:
    """Verify the R7 v1 manifest identity (A-prime's gate) and every pinned asset; optionally the seven worker sources."""
    from knee.public_r7_recovery_entry import r7_manifest_gate
    root = Path(asset_dir)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("ASSET_ROOT_NOT_DIRECTORY")
    manifest_bytes = _checked_file(root, "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("MANIFEST_SHA256_MISMATCH")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("MANIFEST_INVALID_JSON") from exc
    if manifest.get("bundle_id") != BUNDLE_ID or manifest.get("tag") != BUNDLE_TAG:
        raise ValueError("MANIFEST_R7_IDENTITY_MISMATCH")
    r7_manifest_gate(manifest)
    declared = manifest.get("sha256")
    if not isinstance(declared, dict) or any(declared.get(k) != v for k, v in ASSET_SHA256.items()):
        raise ValueError("MANIFEST_ASSET_CLOSURE_MISMATCH")
    assets = _verify_files(root, ASSET_SHA256, label="ASSET")
    sources = None
    if source_dir is not None:
        source_root = Path(source_dir)
        if not source_root.is_dir() or source_root.is_symlink():
            raise ValueError("SOURCE_ROOT_NOT_DIRECTORY")
        sources = _verify_files(source_root, SOURCE_SHA256, label="SOURCE")
    return {"profile_id": PROFILE_ID, "bundle_id": BUNDLE_ID, "manifest_sha256": MANIFEST_SHA256,
            "assets_sha256": assets, "sources_sha256": sources}
