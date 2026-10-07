"""Package only code, compact features and the protected baseline for Colab."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def build(artifacts, destination):
    root = Path(__file__).resolve().parents[1]
    artifacts, destination = Path(artifacts), Path(destination)
    files = {f"project/knee/{p.name}": p for p in sorted((root / "knee").glob("*.py"))}
    for name in ("features.npz", "extraction.json", "attention/head.pt", "attention/training.json"):
        files["artifacts/" + name] = artifacts / name
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    extraction = json.loads((artifacts / "extraction.json").read_text())
    if hashes["artifacts/features.npz"] != extraction["features_sha256"]:
        raise ValueError("Feature cache fingerprint differs from extraction record")
    sources = {str(p.relative_to(root)): p.read_text() for p in sorted((root / "knee").glob("*.py"))}
    manifest = {"code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "source_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                "files": hashes, "training_seconds_per_head": 300, "maximum_new_heads": 3,
                "note": "Private compact features/labels only; no raw images, reports or credentials."}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    print(json.dumps({"archive": str(destination), "bytes":destination.stat().st_size,
                      "sha256":hashlib.sha256(destination.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts")
    parser.add_argument("destination")
    args = parser.parse_args()
    build(args.artifacts, args.destination)
