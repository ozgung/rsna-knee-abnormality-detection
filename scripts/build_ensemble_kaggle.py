"""Package weak-selected heads as a private dataset for offline Kaggle inference."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from build_kaggle import build


def build_selected(baseline_artifacts, comparisons, destination, slug, *, assets_dir, dataset_source, rehearsal=False):
    baseline_artifacts, comparisons = Path(baseline_artifacts), Path(comparisons)
    assets_dir = Path(assets_dir)
    if not dataset_source.startswith("ozgung/") or len(dataset_source.split("/")) != 2:
        raise ValueError("Expected a private dataset in the authenticated account")
    assets_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((comparisons / "comparison.json").read_text())
    if report["status"] != "completed" or report["selection_uses_gold"]:
        raise ValueError("Require a completed gold-independent comparison")
    selected = report["selected"]
    members, weights = selected["members"], selected["weights"]
    allowed = {"baseline", "attention_seed43", "attention_seed44", "meanmax_seed42"}
    if not members or len(set(members)) != len(members) or not set(members) <= allowed:
        raise ValueError("Unexpected selected members")
    if len(weights) != len(members) or any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1):
        raise ValueError("Invalid selected weights")
    baseline_loss = next(c["weak_bce"] for c in report["comparisons"] if c["members"] == ["baseline"])
    if selected["weak_bce"] >= baseline_loss - report["selection_tolerance"]:
        raise ValueError("Selected candidate does not improve the protected baseline")
    assets, hashes, expected, ids = {}, {}, [], None
    for name in members:
        directory = baseline_artifacts / "attention" if name == "baseline" else comparisons / name
        content = (directory / "head.pt").read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != report["candidates"][name]["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint hash mismatch")
        assets[name + ".pt"] = content
        hashes[name + ".pt"] = digest
        if rehearsal:
            with np.load(directory / "gold_predictions.npz", allow_pickle=False) as data:
                if ids is None:
                    ids = data["ids"].copy()
                elif not np.array_equal(ids, data["ids"]):
                    raise ValueError("Selected gold prediction ID order differs")
                expected.append(data["predictions"].astype(np.float64))
    if rehearsal:
        buffer = io.BytesIO()
        prediction = sum(w * p for w, p in zip(weights, expected))
        np.savez_compressed(buffer, ids=ids, predictions=prediction, targets=report["targets"])
        content = buffer.getvalue()
        assets["expected-gold.npz"] = content
        hashes["expected-gold.npz"] = hashlib.sha256(content).hexdigest()
    selection = {"members": members, "weights": weights, "weak_bce": selected["weak_bce"],
                 "baseline_weak_bce": baseline_loss, "feature_sha256": report["feature_sha256"],
                 "comparison_sha256": hashlib.sha256((comparisons / "comparison.json").read_bytes()).hexdigest(),
                 "asset_sha256": hashes}
    unexpected = {p.name for p in assets_dir.iterdir()} - set(assets) - {"dataset-metadata.json", "expected-gold.npz"}
    if unexpected:
        raise ValueError("Unexpected files in private asset upload directory")
    for name, content in assets.items():
        (assets_dir / name).write_bytes(content)
    (assets_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": "Knee Own Heads 20261008", "id": dataset_source,
        "licenses": [{"name": "other"}],
        "description": "Private trained head artifacts for the owner's competition entries. No raw images, report text or credentials."}, indent=2))
    owner, dataset_slug = dataset_source.split("/")
    entry = (
        "import hashlib, json, os\nfrom pathlib import Path\n"
        "from knee.runtime import competition_root, infer, prepare_codecs, training_artifacts\n"
        "from knee.features import find_encoder\n"
        "os.environ['HF_HUB_OFFLINE']='1'\nos.environ['TRANSFORMERS_OFFLINE']='1'\n"
        f"selection={selection!r}\n"
        f"locations=[Path('/kaggle/input/datasets/{owner}/{dataset_slug}'),Path('/kaggle/input/{dataset_slug}')]\n"
        "matches=[p for p in locations if (p/(selection['members'][0]+'.pt')).is_file()]\n"
        "if len(matches)!=1: raise ValueError('Selected head dataset missing or ambiguous')\n"
        "selected_root=matches[0]\n"
        "for name,digest in selection['asset_sha256'].items():\n"
        "    if hashlib.sha256((selected_root/name).read_bytes()).hexdigest()!=digest:\n"
        "        raise ValueError('Selected artifact hash mismatch')\n"
        "Path('/kaggle/working/selection.json').write_text(json.dumps(selection,indent=2))\n"
        "head_paths=[selected_root/(name+'.pt') for name in selection['members']]\n"
        "artifacts=training_artifacts()\n"
        "prepare_codecs(artifacts/'wheels',expected_hashes=json.loads((artifacts/'wheel-hashes.json').read_text()))\n"
        "root,encoder=competition_root(),find_encoder('/kaggle/input')\n"
    )
    if rehearsal:
        entry += ("from knee.rehearsal import rehearse\n"
                  "rehearse(root,encoder,head_paths,selected_root/'expected-gold.npz',"
                  "'/kaggle/working/ensemble-rehearsal',head_weights=selection['weights'])\n")
    else:
        entry += ("infer(root,encoder,head_paths[0],'/kaggle/working/submission.csv',"
                  "additional_head_paths=head_paths[1:],head_weights=selection['weights'])\n")
    digest = build(destination, slug, entry, internet=False,
                   kernel_sources=["ozgung/knee-dino-train-20261008"], weak_labels=False)
    metadata_path = Path(destination, "kernel-metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["dataset_sources"] = [dataset_source]
    metadata_path.write_text(json.dumps(metadata, indent=2))
    Path(destination, "selection.json").write_text(json.dumps(selection, indent=2))
    print(json.dumps({"source_sha256": digest, "selected": selected,
                      "notebook_bytes": Path(destination, "notebook.ipynb").stat().st_size}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_artifacts")
    parser.add_argument("comparisons")
    parser.add_argument("destination")
    parser.add_argument("slug")
    parser.add_argument("--rehearsal", action="store_true")
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--dataset-source", required=True)
    args = parser.parse_args()
    build_selected(args.baseline_artifacts, args.comparisons, args.destination, args.slug, rehearsal=args.rehearsal, assets_dir=args.assets_dir, dataset_source=args.dataset_source)
