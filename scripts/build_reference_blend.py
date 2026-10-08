"""Package one predeclared 50/50 rank blend; this command never uploads or runs it.

Example (all receipts/source snapshots are ignored local inputs):
  python3.12 scripts/build_reference_blend.py outputs/kaggle/reference-blend-rehearsal \
    knee-reference-blend-rehearsal-20261008 \
    --convnext-source-dir outputs/research/2026-10-08/convnext-audit \
    --convnext-entry scripts/public_reader_entry.py \
    --convnext-contract outputs/kaggle/public-reader-rehearsal/build-contract.json \
    --convnext-rehearsal outputs/kaggle/public-reader-rehearsal-result/reader-audit.json \
    --raptor-source outputs/research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py \
    --raptor-contract outputs/kaggle/raptor-rehearsal/reference-contract.json \
    --raptor-rehearsal outputs/kaggle/raptor-rehearsal-result/reference-rehearsal/rehearsal.json

After a passing visible-only run, add --submission --blend-rehearsal PATH/blend-audit.json.
The builder requires that common-environment receipt before creating a full entry.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knee.reference import audit_source, canonical_bytes, sha256
from knee.reference_blend import (CONVNEXT_DATASET, CONVNEXT_SOURCE_HASHES, CONVNEXT_URL,
    DOCKER_IMAGE, RAPTOR_PLAIN_SHA256, RAPTOR_WEIGHT_DATASET, RECIPE, extract_reader, qualify_components)
from build_kaggle import build


def require_blend_rehearsal(report, fingerprint):
    if (report.get("status") != "PASS" or report.get("rehearsal") is not True or
            report.get("qualification_sha256") != fingerprint or report.get("test_studies") != 3 or
            report.get("prediction_fallbacks") != 0 or report.get("blend_repeat_max_abs_error") != 0 or
            not 0 < report.get("estimated_1300_seconds_mean_with_margin", float("inf")) < 8 * 3600):
        raise ValueError("Successful matching common-environment blend rehearsal required")
    for name in ("convnext", "raptor"):
        component = report.get("components", {}).get(name, {})
        if (component.get("studies") != 3 or component.get("decode_failures") != 0 or
                component.get("zero_studies") != 0 or
                not 0 <= component.get("raw_rerun_max_abs_error", float("inf")) <= 1e-6):
            raise ValueError("Common-environment component reread parity did not pass")
    hashes = report.get("codec_hashes", {})
    if len(hashes) != 3 or any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in hashes.values()):
        raise ValueError("Common-environment rehearsal codec hashes missing")
    return hashes


def build_blend(destination, slug, *, convnext_source_dir, convnext_entry, convnext_contract,
                convnext_rehearsal, raptor_source, raptor_contract, raptor_rehearsal,
                submission=False, blend_rehearsal=None, max_seconds=None):
    root = Path(__file__).resolve().parents[1]
    paths = {"convnext_contract": convnext_contract, "convnext_rehearsal": convnext_rehearsal,
             "raptor_contract": raptor_contract, "raptor_rehearsal": raptor_rehearsal}
    raw = {k: Path(v).read_bytes() for k, v in paths.items()}
    receipts = {k: json.loads(v) for k, v in raw.items()}
    cnn_sources = {name: (Path(convnext_source_dir) / name).read_text() for name in CONVNEXT_SOURCE_HASHES}
    if any(sha256(value.encode()) != CONVNEXT_SOURCE_HASHES[name] for name, value in cnn_sources.items()):
        raise ValueError("ConvNeXt source snapshot differs from reviewed source")
    entry = Path(convnext_entry).read_text()
    extract_reader(entry)  # Fail before packaging any unreviewed top-level program.
    public, decoder, manifest = audit_source(Path(raptor_source).read_text(), profile="raptor",
                                            checkpoint_sha256=RAPTOR_PLAIN_SHA256)
    if decoder:
        raise ValueError("Plain Raptor must not have an opaque decoder payload")
    qualification = qualify_components(receipts["convnext_contract"], receipts["convnext_rehearsal"],
        receipts["raptor_contract"], receipts["raptor_rehearsal"], manifest)
    qualification["evidence_sha256"] = {name: sha256(value) for name, value in raw.items()}
    qualification["runtime_sha256"] = {name: sha256((root / name).read_bytes()) for name in
        ("knee/reference.py", "knee/reference_blend.py", "scripts/reference_blend_entry.py")}
    qualification["recipe"] = RECIPE
    qualification["common_environment"] = {"python": [3, 12], "docker_image": DOCKER_IMAGE,
        "torch": "2.10.0+cu128", "timm": "1.0.26"}
    fingerprint = sha256(canonical_bytes(qualification))
    max_seconds = max_seconds if max_seconds is not None else (28000 if submission else 3600)
    if not isinstance(max_seconds, int) or isinstance(max_seconds, bool) or not 0 < max_seconds <= (28800 if submission else 3600):
        raise ValueError("Blend entry exceeds its declared time bound")
    expected_codecs, rehearsal_hash = {}, None
    if submission:
        if blend_rehearsal is None:
            raise ValueError("Submission packaging requires a successful blend rehearsal receipt")
        data = Path(blend_rehearsal).read_bytes()
        expected_codecs = require_blend_rehearsal(json.loads(data), fingerprint)
        rehearsal_hash = sha256(data)
    elif blend_rehearsal is not None:
        raise ValueError("Blend receipt is only an input for full entry packaging")
    contract = {"qualification": qualification, "qualification_sha256": fingerprint, "recipe": RECIPE,
        "rehearsal": not submission, "max_seconds": max_seconds,
        "blend_rehearsal_sha256": rehearsal_hash, "expected_codec_hashes": expected_codecs,
        "private": True, "internet": False}
    runtime = (f"CONVNEXT_SOURCES={cnn_sources!r}\nCONVNEXT_ENTRY={entry!r}\n"
        f"RAPTOR_SOURCE={public!r}\nBLEND_CONTRACT={contract!r}\n"
        + (root / "scripts/reference_blend_entry.py").read_text())
    compile(runtime, "<blend-entry>", "exec")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Use a new destination to preserve previous build receipts")
    digest = build(destination, slug, runtime, internet=False, weak_labels=False, kernel_sources=[])
    meta_path = destination / "kernel-metadata.json"
    metadata = json.loads(meta_path.read_text())
    metadata.update(dataset_sources=[CONVNEXT_DATASET, RAPTOR_WEIGHT_DATASET], model_sources=[],
                    docker_image=DOCKER_IMAGE)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    notebook_path = destination / "notebook.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook["metadata"]["language_info"]["version"] = "3.12"
    notebook["cells"][0]["source"] = [
        "# Predeclared equal-rank reference blend\n",
        f"ConvNeXt: goodpjw2008, Apache-2.0. [Source]({CONVNEXT_URL}). Three folds, unchanged mean probabilities.\n",
        f"Plain Raptor: Dread Development / Johnathan Wagner, CC0 weights. [Source]({manifest['source_url']}).\n",
        "Fixed 50/50 average of per-target percentile ranks with average ties. No fitting, calibration, "
        "label-based selection, OAI checkpoint, or leaderboard-dependent weights.\n",
        "Private offline run. Both source definitions and weights are pinned to successful bounded rehearsals. "
        "Visible-only qualification independently rereads each component in the shared Python 3.12 environment.\n",
        f"Qualification SHA256: `{fingerprint}`. Runtime cap: {max_seconds}s.\n"]
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n")
    if notebook_path.stat().st_size > 900000:
        raise ValueError("Notebook exceeds conservative source size limit")
    (destination / "blend-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    receipt = {"bundle_sha256": digest, "qualification_sha256": fingerprint,
               "contract_sha256": sha256(canonical_bytes(contract)), "notebook_bytes": notebook_path.stat().st_size}
    (destination / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("slug")
    for option in ("convnext-source-dir", "convnext-entry", "convnext-contract", "convnext-rehearsal",
                   "raptor-source", "raptor-contract", "raptor-rehearsal"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--blend-rehearsal", type=Path)
    parser.add_argument("--max-seconds", type=int)
    print(json.dumps(build_blend(**vars(parser.parse_args())), indent=2))
