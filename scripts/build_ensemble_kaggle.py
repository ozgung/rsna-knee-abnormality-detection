"""Embed the weak-selected small heads into a private offline Kaggle notebook."""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import zlib

import numpy as np
from build_kaggle import build


def build_selected(baseline_artifacts, comparisons, destination, slug, *, rehearsal=False):
    baseline_artifacts, comparisons = Path(baseline_artifacts), Path(comparisons)
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
        assets[name + ".pt"] = base64.b64encode(zlib.compress(content)).decode()
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
        np.savez_compressed(buffer, ids=ids, predictions=prediction)
        content = buffer.getvalue()
        assets["expected-gold.npz"] = base64.b64encode(zlib.compress(content)).decode()
        hashes["expected-gold.npz"] = hashlib.sha256(content).hexdigest()
    selection = {"members": members, "weights": weights, "weak_bce": selected["weak_bce"],
                 "baseline_weak_bce": baseline_loss, "feature_sha256": report["feature_sha256"],
                 "comparison_sha256": hashlib.sha256((comparisons / "comparison.json").read_bytes()).hexdigest(),
                 "asset_sha256": hashes}
    entry = (
        "import base64, hashlib, json, os, zlib\nfrom pathlib import Path\n"
        "from knee.runtime import competition_root, infer, prepare_codecs, training_artifacts\n"
        "from knee.features import find_encoder\n"
        "os.environ['HF_HUB_OFFLINE']='1'\nos.environ['TRANSFORMERS_OFFLINE']='1'\n"
        f"assets={assets!r}\nselection={selection!r}\n"
        "selected_root=Path('/kaggle/working/selected-heads')\nselected_root.mkdir(exist_ok=True)\n"
        "for name, encoded in assets.items():\n"
        "    content=zlib.decompress(base64.b64decode(encoded))\n"
        "    if hashlib.sha256(content).hexdigest()!=selection['asset_sha256'][name]:\n"
        "        raise ValueError('Embedded selected artifact hash mismatch')\n"
        "    (selected_root/name).write_bytes(content)\n"
        "(selected_root/'selection.json').write_text(json.dumps(selection,indent=2))\n"
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
    args = parser.parse_args()
    build_selected(args.baseline_artifacts, args.comparisons, args.destination, args.slug, rehearsal=args.rehearsal)
