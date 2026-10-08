"""Package inspected public ConvNeXt source with strict offline inference checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knee.public_reader_contract import INPUT_POLICY_VERSION, TRAIN_STUDIES, require_reader_rehearsal

SOURCE_HASHES = {
    "preprocess.py": "4491963e4d2156625c9cf79051f4a25c32d2d761f1f07df3117a439ed8029ae2",
    "knee.py": "8e9fd05dd94e9be70d4b4a6cbdf5201c219b5e6eb3d57c07bb8992c8dbe72ae7",
}


def build(source_dir, destination, slug, *, rehearsal=True, weight_hashes=None, rehearsal_report=None):
    root = Path(__file__).resolve().parents[1]
    source_dir, destination = Path(source_dir), Path(destination)
    sources = {}
    for name, digest in SOURCE_HASHES.items():
        value = (source_dir / name).read_bytes()
        if hashlib.sha256(value).hexdigest() != digest:
            raise ValueError(f"Unreviewed public source: {name}")
        compile(value, name, "exec")
        sources[name] = value.decode()
    if not rehearsal and (not weight_hashes or set(weight_hashes) != {f"cnxt_v0_fold{i}.pt" for i in range(3)}):
        raise ValueError("Submission requires all three rehearsed checkpoint hashes")
    entry = (root / "scripts/public_reader_entry.py").read_text()
    entry_digest = hashlib.sha256(entry.encode()).hexdigest()
    report_digest = None
    if not rehearsal:
        if rehearsal_report is None:
            raise ValueError("Submission requires a matching expanded rehearsal report")
        report_bytes = Path(rehearsal_report).read_bytes()
        require_reader_rehearsal(json.loads(report_bytes), entry_digest, SOURCE_HASHES, weight_hashes)
        report_digest = hashlib.sha256(report_bytes).hexdigest()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    bootstrap = "from pathlib import Path\nimport json\n"
    bootstrap += f"REHEARSAL = {rehearsal!r}\nEXPECTED_WEIGHTS = {weight_hashes or {}!r}\n"
    bootstrap += f"CODE_REVISION = {revision!r}\nSOURCE_HASHES = {SOURCE_HASHES!r}\n"
    bootstrap += f"ENTRY_SHA256 = {entry_digest!r}\n"
    bootstrap += f"sources = {sources!r}\n"
    bootstrap += "source_dir = Path('/kaggle/working/audited_reader')\nsource_dir.mkdir(exist_ok=True)\n"
    bootstrap += "for name, content in sources.items():\n    (source_dir / name).write_text(content)\n"
    bootstrap += entry
    compile(bootstrap, "reader-notebook", "exec")
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}}, "cells": [
        {"cell_type": "markdown", "id": "attribution", "metadata": {}, "source": [
            "# Audited public ConvNeXt reader\n",
            "Model and preprocessing: goodpjw2008, Apache-2.0. Source: https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944\n",
            "Own wrapper retains usable partial series, records masked unusable slots, and rejects empty studies or resource failures. Checkpoint hashes, actual test coverage and bounded offline rehearsal are enforced. No fitting or gold-based selection.\n"]},
        {"cell_type": "code", "id": "reader-run", "metadata": {}, "execution_count": None, "outputs": [], "source": bootstrap.splitlines(True)}]}
    meta = {"id": f"ozgung/{slug}", "title": slug.replace("-", " ").title(), "code_file": "notebook.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader"],
            "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": [],
            "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "notebook.ipynb").write_text(json.dumps(nb, indent=2))
    (destination / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    (destination / "build-contract.json").write_text(json.dumps({"revision": revision, "sources": SOURCE_HASHES,
        "entry_sha256": entry_digest, "rehearsal": rehearsal, "input_policy_version": INPUT_POLICY_VERSION,
        "train_studies_requested": TRAIN_STUDIES, "rehearsal_report_sha256": report_digest,
        "expected_weights": weight_hashes or {}}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source_dir"); ap.add_argument("destination"); ap.add_argument("slug")
    ap.add_argument("--submission", action="store_true")
    ap.add_argument("--weight-hashes")
    ap.add_argument("--rehearsal-report", help="Required current expanded reader-audit.json for full entry")
    args = ap.parse_args()
    hashes = json.loads(Path(args.weight_hashes).read_text()) if args.weight_hashes else None
    build(args.source_dir, args.destination, args.slug, rehearsal=not args.submission, weight_hashes=hashes,
          rehearsal_report=args.rehearsal_report)
