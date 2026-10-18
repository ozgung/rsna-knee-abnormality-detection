"""Frozen, dependency-free identity and asset checks for the public R6 recipe.

This module identifies the captured R6 bundle and its public inference recipe.
It does not qualify the reader, model execution, runtime, or competition result.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from types import MappingProxyType


PROFILE_ID = "public-r6-fe055406-fliptta-v1"
BUNDLE_ID = "fe055406fad6a12f"
BUNDLE_TAG = "cnx_r6fm_c352"
MANIFEST_SHA256 = "6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648"

# MappingProxyType and tuples prevent callers from changing the recipe in place.
ASSET_SHA256 = MappingProxyType({
    "E_lrprior_s0_f0.pt": "59cc88a197bc80c7fa5faa89e69ae7bec04f69cab5259a23a0d0ceb9431e53f6",
    "E_lrprior_s0_f1.pt": "3d9dd8d6a1667935ea25e85b2fcfc80c7ae7dd37fe1bb587f8dad9d720fd8066",
    "E_lrprior_s0_f2.pt": "8d32fa2c57efc8e1c9159cf2b22f38cee0f05dce46aab2f000095847f537b1ea",
    "E_lrprior_s0_f3.pt": "ea147aac4c30fe5a616b966018446e04bd6815c9965f2ac9b0a4702fa9f32710",
    "state_spec.json": "3914242b0c404aee625ea65801f14f60adfee326c1ddd6fb96c48ed7050b7ff3",
    "side_cls.pt": "ab623e2b619db4917639220fed15202e8c8aedd07c8e298a66a2afc0f153650a",
    "fluid_stacker.json": "9fbd6f652470c5ef131f997edb4e358038937dd45e6f6b65a4afed80e10cf35d",
    "state_stacker.json": "b05147bb463a389a66a4a0d7ff0555efd2bde6f486718f79e9d7fa3d90b24522",
})
SOURCE_SHA256 = MappingProxyType({
    "px_cache.py": "b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632",
    "compact_train.py": "04df7e2e154b2c2521696908f272940c63ce17e9b6337b85847fee64504eeaba",
    "compact_infer.py": "8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d",
    "probe_infer.py": "a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980",
    "probe_dual.py": "a6155eb08701f9bb746f838bc8ef8e2129c85f76a5c58e379692ffe2e76bbef0",
    "knee_anchor_bank_v2.py": "e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309",
    "side_common.py": "bbcc5d027834e6a6fd48c4a8cf228e25661b7522c792587c840fe4d7cd5c0566",
})
FLIP_TTA = True
R384_WEIGHT = 0.65
R224_WEIGHT = 0.35
DEFAULT_RAPTOR_WEIGHT = 0.5
RAPTOR_WEIGHTS = MappingProxyType({
    "Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6,
    "Synovitis": 0.8, "Baker's": 0.35,
})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_file(root: Path, name: str) -> Path:
    """Return a direct, regular child file; never follow asset symlinks."""
    path = root / name
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"ASSET_MISSING_OR_UNREADABLE:{name}") from exc
    if not path.parent.samefile(root) or path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"ASSET_NOT_REGULAR_DIRECT_FILE:{name}")
    return path


def _verify_files(root: Path, pins: MappingProxyType, *, label: str) -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, expected in pins.items():
        path = _checked_file(root, name)
        digest = _sha256_file(path)
        if digest != expected:
            raise ValueError(f"{label}_SHA256_MISMATCH:{name}")
        actual[name] = digest
    return actual


def validate_r6_assets(asset_dir: str | os.PathLike[str], *, source_dir: str | os.PathLike[str] | None = None) -> dict[str, object]:
    """Verify captured manifest identity and every pinned R6 asset.

    The optional source directory checks the seven captured worker-source files.
    Paths and content are caller supplied; pins and recipe are fixed here.
    """
    root = Path(asset_dir)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("ASSET_ROOT_NOT_DIRECTORY")
    manifest_path = _checked_file(root, "manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("MANIFEST_SHA256_MISMATCH")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("MANIFEST_INVALID_JSON") from exc
    if manifest.get("bundle_id") != BUNDLE_ID or manifest.get("tag") != BUNDLE_TAG:
        raise ValueError("MANIFEST_R6_IDENTITY_MISMATCH")
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
    return {
        "profile_id": PROFILE_ID,
        "bundle_id": BUNDLE_ID,
        "manifest_sha256": MANIFEST_SHA256,
        "assets_sha256": assets,
        "sources_sha256": sources,
    }
