"""Entry appended after the audited source/manifest by build_reference_rehearsal."""
import hashlib
import time
REFERENCE_STARTED = time.monotonic()
import json
import os
import signal
import subprocess
import sys
import importlib.metadata
from pathlib import Path

from knee.reference import canonical_bytes, rehearse, sha256
from knee.runtime import competition_root, prepare_codecs, training_artifacts

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[key] = "1"
if REFERENCE_MANIFEST['profile'] in ('raptor', 'mirror'):
    from knee.reference_blend import CONVNEXT_DATASET, select_codec_wheels
    if REFERENCE_MANIFEST['profile'] == 'raptor':
        from knee.raptor_input import policy_contract
    else:
        from knee.mirror_input import policy_contract
    if (REFERENCE_CONTRACT.get('input_policy') != policy_contract() or sys.version_info[:2] != (3, 12) or
            importlib.metadata.version('torch') != '2.10.0+cu128' or importlib.metadata.version('timm') != '1.0.26'):
        raise ValueError('Reference production policy/environment changed; rebuild and qualify')
    def expired(signum, frame):
        raise TimeoutError('Reference rehearsal platform/session time exhausted')
    signal.signal(signal.SIGALRM, expired)
    remaining = REFERENCE_CONTRACT['max_seconds'] - (time.monotonic() - REFERENCE_STARTED)
    if remaining <= 0: raise TimeoutError('Reference setup exhausted runtime')
    signal.setitimer(signal.ITIMER_REAL, remaining)
    codec_owner, codec_slug = CONVNEXT_DATASET.split('/')
    codec_roots = [Path(f'/kaggle/input/datasets/{codec_owner}/{codec_slug}'), Path(f'/kaggle/input/{codec_slug}')]
    codec_roots = [p for p in codec_roots if (p/'pylibjpeg-2.1.0-py3-none-any.whl').is_file()]
    if len(codec_roots) != 1: raise ValueError('Expected one Python 3.12 codec attachment')
    wheels = select_codec_wheels(codec_roots[0])
    if REFERENCE_MANIFEST['profile'] == 'mirror':
        from knee.validation import file_sha256
        if {p.name:file_sha256(p) for p in wheels} != REFERENCE_CONTRACT['codec_hashes']:
            raise ValueError('Mirror offline codec bytes changed')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *map(str, wheels)], check=True)
else:
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
if REFERENCE_MANIFEST['profile'] in ('raptor', 'mirror'): signal.setitimer(signal.ITIMER_REAL, 0)
if REFERENCE_MANIFEST['profile'] == 'mirror':
    from knee.mirror_input import require_rehearsal
    report['inference_rehearsal_seconds'] = report['runtime_seconds']
    report['runtime_seconds'] = time.monotonic() - REFERENCE_STARTED
    require_rehearsal(REFERENCE_CONTRACT, report, bundle_sha256=os.environ['KNEE_SOURCE_SHA256'])
    Path('/kaggle/working/reference-rehearsal/rehearsal.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(json.dumps({"event": "reference_rehearsal_done", "status": report["status"],
                  "contract_sha256": sha256(canonical_bytes(REFERENCE_CONTRACT)),
                  "runtime_seconds": report["runtime_seconds"],
                  "estimated_1300_seconds_p95": report["estimated_1300_seconds_p95"]}), flush=True)
