"""Offline gold-image parity and repeated visible-test rehearsal; no fitting."""
import csv
import json
import os
from pathlib import Path
import numpy as np
from knee.runtime import competition_root, infer, prepare_codecs, training_artifacts
from knee.features import find_encoder
from knee.validation import evaluate_predictions, label_matrix, read_csv, read_schema

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
artifacts = training_artifacts()
prepare_codecs(artifacts / "wheels",
               expected_hashes=json.loads((artifacts / "wheel-hashes.json").read_text()))
root, encoder = competition_root(), find_encoder("/kaggle/input")
id_column, targets = read_schema(root / "sample_submission.csv")
expected = np.load(artifacts / "attention/gold_predictions.npz", allow_pickle=False)
ids = expected["ids"].tolist()
gold_root = Path("/kaggle/working/gold-rehearsal")
gold_root.mkdir(exist_ok=True)
# The inference function sees the same test-only contract as the real submission.
(gold_root / "test_series").symlink_to(root / "train_series", target_is_directory=True)
with (gold_root / "test.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([id_column])
    w.writerows([[uid] for uid in ids])
with (gold_root / "sample_submission.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([id_column, *targets])
    w.writerows([[uid, *([.5] * len(targets))] for uid in ids])
columns, series = read_csv(root / "train_series.csv")
selected = set(ids)
with (gold_root / "test_series.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=columns)
    w.writeheader()
    w.writerows([r for r in series if r[id_column] in selected])
gold_report = infer(gold_root, encoder, artifacts / "attention/head.pt", gold_root / "predictions.csv")
_, output_rows = read_csv(gold_root / "predictions.csv")
predictions = {r[id_column]: [float(r[t]) for t in targets] for r in output_rows}
observed = np.asarray([predictions[uid] for uid in ids])
max_error = float(np.max(np.abs(observed - expected["predictions"])))
if not np.allclose(observed, expected["predictions"], atol=2e-4, rtol=2e-4):
    raise RuntimeError(f"Offline predictions disagree with training validation: {max_error}")
_, train_rows = read_csv(root / "train.csv")
gold = label_matrix([r for r in train_rows if r[id_column] in selected], targets, id_column)
metrics = evaluate_predictions(gold, predictions, targets)
visible = Path("/kaggle/working/visible-rehearsal")
visible.mkdir(exist_ok=True)
visible_report = infer(root, encoder, artifacts / "attention/head.pt", visible / "submission.csv", repeat=2)
report = {"status": "passed", "gold_prediction_max_error": max_error,
          "gold_metrics": metrics, "gold_resources": gold_report,
          "visible_test_resources": visible_report,
          "note": "Internet disabled by notebook metadata; gold labels used only for final evaluation."}
(Path("/kaggle/working") / "rehearsal.json").write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps({"event": "rehearsal_passed", "gold_prediction_max_error": max_error,
                  "gold_macro_auc": metrics["macro_all_targets_auc"]}), flush=True)
