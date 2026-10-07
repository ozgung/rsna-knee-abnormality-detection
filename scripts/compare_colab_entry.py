"""Run in a bounded Colab session after uploading /content/knee-input.zip."""
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

root = Path("/content/knee-comparison")
root.mkdir(exist_ok=False)
with zipfile.ZipFile("/content/knee-input.zip") as archive:
    manifest = json.loads(archive.read("manifest.json"))
    # Only the explicit manifest's regular files are materialized.
    for name, expected_hash in manifest["files"].items():
        path = root / name
        if path.is_absolute() and not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unexpected archive path")
        content = archive.read(name)
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError("Uploaded input hash mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
sys.path.insert(0, str(root / "project"))
os.environ["KNEE_CODE_REVISION"] = manifest["code_revision"]
os.environ["KNEE_SOURCE_SHA256"] = manifest["source_sha256"]
from knee.compare import compare

artifacts = root / "artifacts"
out = root / "comparisons"
report = compare(artifacts / "features.npz", artifacts / "extraction.json", artifacts / "attention",
                 out, device="cuda", max_seconds=300, max_epochs=80, patience=12, ensembles=True)
with zipfile.ZipFile("/content/knee-comparison-results.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(out.rglob("*")):
        if path.is_file():
            archive.write(path, str(path.relative_to(root)))
    archive.writestr("input-manifest.json", json.dumps(manifest, indent=2))
print(json.dumps({"event":"comparison_artifacts_ready", "status":report["status"],
                  "selected":report["selected"], "archive":"/content/knee-comparison-results.zip"}), flush=True)
