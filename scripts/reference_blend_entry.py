"""Appended by build_reference_blend after the pinned source and receipt contract."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from knee.reference import canonical_bytes, sha256
from knee.reference_blend import CONVNEXT_DATASET, RAPTOR_WEIGHT_DATASET, run_blend, select_codec_wheels

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
qualification = BLEND_CONTRACT["qualification"]
if sha256(canonical_bytes(qualification)) != BLEND_CONTRACT["qualification_sha256"]:
    raise ValueError("Blend qualification fingerprint differs")
if sys.version_info[:2] != (3, 12):
    raise RuntimeError("Blend requires the qualified Python 3.12 image, not mixed-ABI codec wheels")
for package in ("torch", "timm"):
    if importlib.metadata.version(package) != qualification["common_environment"][package]:
        raise RuntimeError("Blend model environment changed; qualify it again")

def expired(signum, frame):
    raise TimeoutError("Eight-hour/visible-rehearsal wall-clock cap exhausted")

signal.signal(signal.SIGALRM, expired)
signal.alarm(BLEND_CONTRACT["max_seconds"])

def attachment(dataset, required):
    owner, slug = dataset.split("/")
    candidates = [Path(f"/kaggle/input/datasets/{owner}/{slug}"), Path(f"/kaggle/input/{slug}")]
    matches = [path for path in candidates if (path / required).is_file()]
    if len(matches) != 1:
        raise ValueError("Required reference attachment missing or ambiguous")
    return matches[0]

cnn = attachment(CONVNEXT_DATASET, "cnxt_v0_fold0.pt")
raptor = attachment(RAPTOR_WEIGHT_DATASET, qualification["raptor_manifest"]["checkpoint_file"])
roots = [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
         Path("/kaggle/input/rsna-knee-abnormality-detection")]
roots = [root for root in roots if (root / "test.csv").is_file()]
if len(roots) != 1:
    raise ValueError("Competition attachment missing or ambiguous")
# Same three offline codecs as the passing ConvNeXt entry. Qualification records
# their hashes; full entry requires the exact same payloads before pip executes.
wheels = select_codec_wheels(cnn)
codec_hashes = {path.name: hashlib.file_digest(path.open("rb"), "sha256").hexdigest() for path in wheels}
if BLEND_CONTRACT["expected_codec_hashes"] and codec_hashes != BLEND_CONTRACT["expected_codec_hashes"]:
    raise ValueError("Offline codec payload differs from common-environment rehearsal")
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *map(str, wheels)], check=True)
out = Path("/kaggle/working/reference-blend")
report = run_blend(roots[0], cnn, raptor, CONVNEXT_SOURCES, CONVNEXT_ENTRY, RAPTOR_SOURCE,
                   BLEND_CONTRACT, out, codec_hashes=codec_hashes)
if not BLEND_CONTRACT["rehearsal"]:
    target = Path("/kaggle/working/submission.csv")
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing submission")
    shutil.copyfile(out / "submission.csv", target)
signal.alarm(0)
print(json.dumps({"event": "reference_blend_done", "status": report["status"],
                  "rehearsal": report["rehearsal"], "test_studies": report["test_studies"],
                  "elapsed_seconds": report["elapsed_seconds"], "qualification_sha256": report["qualification_sha256"]}), flush=True)
