"""Entry code bundled by build_kaggle.py; run on attached Kaggle data only."""
import json
from pathlib import Path
from knee.runtime import competition_root, prepare_codecs
from knee.features import find_encoder

artifacts = Path("/kaggle/working/artifacts")
artifacts.mkdir(exist_ok=True)
wheel_hashes = prepare_codecs(artifacts / "wheels", download=True)
from knee.experiment import extract, train

root = competition_root()
encoder = find_encoder("/kaggle/input")
label_candidates = [
    Path("/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_full.csv"),
    Path("/kaggle/input/rsna-knee-llm-report-labels/llm_labels_full.csv"),
]
labels = [p for p in label_candidates if p.exists()]
if len(labels) != 1:
    raise RuntimeError("Public v1 weak label file missing or ambiguous")
(artifacts / "wheel-hashes.json").write_text(json.dumps(wheel_hashes, indent=2))
extract(root, encoder, labels[0], artifacts, train_limit=2000, weak_limit=300, max_seconds=5400)
train(artifacts / "features.npz", artifacts / "extraction.json", artifacts / "attention",
      kind="attention", seed=42, max_epochs=80, patience=12, max_seconds=900)
