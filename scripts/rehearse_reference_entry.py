"""Entry appended after the audited source/manifest by build_reference_rehearsal."""
import hashlib
import json
import os
from pathlib import Path

from knee.reference import canonical_bytes, rehearse, sha256
from knee.runtime import competition_root, prepare_codecs, training_artifacts

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[key] = "1"
artifacts = training_artifacts()
prepare_codecs(artifacts / "wheels", expected_hashes=json.loads((artifacts / "wheel-hashes.json").read_text()))
owner, slug = REFERENCE_MANIFEST["weight_dataset"].split("/")
locations = [Path(f"/kaggle/input/datasets/{owner}/{slug}"), Path(f"/kaggle/input/{slug}")]
matches = [p for p in locations if (p / REFERENCE_MANIFEST["checkpoint_file"]).is_file()]
if len(matches) != 1:
    raise ValueError("Reference weights attachment missing or ambiguous")
weights_root = matches[0]
publisher = REFERENCE_MANIFEST.get("publisher_manifest")
if publisher:
    if sha256((weights_root / publisher["filename"]).read_bytes()) != publisher["sha256"]:
        raise ValueError("Publisher manifest hash mismatch")
Path("/kaggle/working/reference-contract.json").write_text(json.dumps(REFERENCE_CONTRACT, indent=2))
report = rehearse(competition_root(), weights_root / REFERENCE_MANIFEST["checkpoint_file"],
                 PUBLIC_SOURCE, DECODER_SOURCE, REFERENCE_MANIFEST,
                 "/kaggle/working/reference-rehearsal", train_count=REFERENCE_CONTRACT["train_count"],
                 seed=REFERENCE_CONTRACT["seed"], max_seconds=REFERENCE_CONTRACT["max_seconds"])
print(json.dumps({"event": "reference_rehearsal_done", "status": report["status"],
                  "contract_sha256": sha256(canonical_bytes(REFERENCE_CONTRACT)),
                  "runtime_seconds": report["runtime_seconds"],
                  "estimated_1300_seconds_p95": report["estimated_1300_seconds_p95"]}), flush=True)
