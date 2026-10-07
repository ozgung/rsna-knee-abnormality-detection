"""Bundle project sources into a private, reproducible Kaggle notebook."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def build(destination, slug, entry, *, internet=False, kernel_sources=(), weak_labels=True, gpu=True):
    root = Path(__file__).resolve().parents[1]
    files = {str(p.relative_to(root)): p.read_text() for p in sorted((root / "knee").glob("*.py"))}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    bootstrap = (
        "import os, sys, json\nfrom pathlib import Path\n"
        "project = Path('/kaggle/working/project')\n"
        f"sources = {files!r}\n"
        "for name, content in sources.items():\n"
        "    path = project / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(content)\n"
        "sys.path.insert(0, str(project))\n"
        f"os.environ['KNEE_CODE_REVISION'] = {revision!r}\n"
        f"os.environ['KNEE_SOURCE_SHA256'] = {digest!r}\n"
        + entry
    )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"}}, "cells": [
        {"cell_type": "markdown", "id": "description", "metadata": {}, "source": [
            "# Knee MRI: reproducible image-only baseline\n",
            "Private research run. DINOv2 pretrained encoder (Meta, Apache-2.0); "
            "report-derived training labels by Steven Lee Hans (CC0). "
            "Study-level splits do not prove patient disjointness where identity is absent.\n"]},
        {"cell_type": "code", "id": "run", "metadata": {}, "execution_count": None, "outputs": [], "source": bootstrap.splitlines(True)}]}
    (destination / "notebook.ipynb").write_text(json.dumps(nb, indent=2))
    meta = {"id": f"ozgung/{slug}", "title": slug.replace("-", " ").title(),
            "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": internet,
            "dataset_sources": ["stevenleehans/rsna-knee-llm-report-labels"] if weak_labels else [],
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "kernel_sources": list(kernel_sources), "model_sources": ["metaresearch/dinov2/PyTorch/small/1"]}
    (destination / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    parser.add_argument("slug")
    parser.add_argument("entry_file")
    parser.add_argument("--internet", action="store_true")
    parser.add_argument("--kernel-source", action="append", default=[])
    parser.add_argument("--no-weak-labels", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    print(build(args.destination, args.slug, Path(args.entry_file).read_text(), internet=args.internet,
                kernel_sources=args.kernel_source, weak_labels=not args.no_weak_labels, gpu=not args.cpu))
