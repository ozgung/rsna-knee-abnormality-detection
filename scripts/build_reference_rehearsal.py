"""Build a private internet-off reference rehearsal from the reviewed source snapshot.

Example (paths are local ignored inputs/outputs):
  python3.12 scripts/build_reference_rehearsal.py \
    outputs/research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py \
    outputs/kaggle/reference-rehearsal knee-reference-rehearsal-20261008 \
    --profile raptor --checkpoint-sha256 ACTUAL_AUDITED_SHA256

No upload, model download, execution or submission occurs in this builder.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knee.reference import audit_source, canonical_bytes, sha256, WEIGHT_DATASET
from knee.raptor_input import TRAIN_COUNT, MAX_SECONDS, policy_contract
from knee.reference_blend import CONVNEXT_DATASET, DOCKER_IMAGE
from build_kaggle import build


def build_reference(source_path, destination, slug, *, train_count=None, seed=20261008,
                    max_seconds=None, publisher_manifest=None, profile="mirror", checkpoint_sha256=None):
    raptor = profile == 'raptor'
    train_count = train_count if train_count is not None else (TRAIN_COUNT if raptor else 24)
    max_seconds = max_seconds if max_seconds is not None else (MAX_SECONDS if raptor else 3600)
    if (not isinstance(train_count, int) or isinstance(train_count, bool) or
            (raptor and train_count != TRAIN_COUNT) or (not raptor and not 1 <= train_count <= 24) or
            not isinstance(max_seconds, int) or isinstance(max_seconds, bool) or
            not 0 < max_seconds <= (MAX_SECONDS if raptor else 3600)):
        raise ValueError("Reference rehearsal exceeds its bounded study/time contract")
    module, decoder, manifest = audit_source(Path(source_path).read_text(), profile=profile,
                                            checkpoint_sha256=checkpoint_sha256)
    if publisher_manifest is not None:
        publisher_manifest = Path(publisher_manifest)
        manifest["publisher_manifest"] = {"filename": publisher_manifest.name,
                                          "sha256": sha256(publisher_manifest.read_bytes())}
    contract = {"source": manifest, "train_count": train_count, "seed": seed,
                "max_seconds": max_seconds, "private": True, "internet": False}
    if raptor:
        contract['input_policy'] = policy_contract()
    entry = (f"PUBLIC_SOURCE={module!r}\nDECODER_SOURCE={decoder!r}\n"
             f"REFERENCE_MANIFEST={manifest!r}\nREFERENCE_CONTRACT={contract!r}\n"
             + Path(__file__).with_name("rehearse_reference_entry.py").read_text())
    digest = build(destination, slug, entry, internet=False, weak_labels=False,
                   kernel_sources=[] if raptor else ["ozgung/knee-dino-train-20261008"])
    destination = Path(destination)
    path = destination / "kernel-metadata.json"
    metadata = json.loads(path.read_text())
    metadata["dataset_sources"], metadata["model_sources"] = [manifest["weight_dataset"]], []
    if raptor:
        metadata['dataset_sources'].append(CONVNEXT_DATASET)
        metadata['docker_image'] = DOCKER_IMAGE
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    notebook_path = destination / "notebook.ipynb"
    notebook = json.loads(notebook_path.read_text())
    if raptor:
        notebook['metadata']['language_info']['version'] = '3.12'
    notebook["cells"][0]["source"] = [
        "# Bounded offline CoAtNet reference rehearsal\n",
        f"Attribution: {manifest['attribution']}. [Public inference source]({manifest['source_url']}).\n",
        "Private reproducibility check, not training or a competition submission. "
        "Three visible studies plus an ID-only deterministic sample; no label-based selection.\n",
        "The selected profile's public successful preprocessing and TTA setting are preserved. "
        "Raptor retains public partial inputs with explicit counters and separate strict healthy comparisons; "
        "resource failures, empty studies and constant-prior predictions are rejected.\n" if raptor else
        "Caught decode failures and constant-prior predictions are rejected.\n",
        f"Reference contract SHA256: `{sha256(canonical_bytes(contract))}`.\n"]
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n")
    if notebook_path.stat().st_size > 900_000:
        raise ValueError("Generated notebook exceeds conservative source-size bound")
    (destination / "reference-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    receipt = {"bundle_sha256": digest, "contract_sha256": sha256(canonical_bytes(contract)),
               "notebook_bytes": notebook_path.stat().st_size, "source_audit": manifest}
    (destination / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_path", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("slug")
    parser.add_argument("--profile", choices=("raptor", "mirror"), required=True)
    parser.add_argument("--checkpoint-sha256", help="Required for raptor; hash from independent checkpoint audit")
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--max-seconds", type=int)
    parser.add_argument("--publisher-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_reference(args.source_path, args.destination, args.slug,
                                    train_count=args.train_count, seed=args.seed,
                                    max_seconds=args.max_seconds,
                                    publisher_manifest=args.publisher_manifest,
                                    profile=args.profile, checkpoint_sha256=args.checkpoint_sha256), indent=2))
