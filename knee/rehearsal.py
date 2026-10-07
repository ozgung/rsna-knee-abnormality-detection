"""Offline image-only prediction parity and repeated visible-test rehearsal.

No fitting occurs. Importing this module needs only the standard library; NumPy
is loaded when the rehearsal runs. Caller supplies offline environment controls.
"""
import csv
import json
from pathlib import Path

from knee.ensemble import validate_weights
from knee.runtime import infer
from knee.validation import evaluate_predictions, label_matrix, read_csv, read_schema


SERIES_FIELDS = ("SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane")


def _csv(path, columns, rows):
    with path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def rehearse(data_dir, encoder_dir, head_paths, expected_predictions_path, output_dir, head_weights=None):
    """Compare raw gold images to frozen predictions, then repeat visible inference.

    Gold inference receives ID-only test.csv, constant sample probabilities and
    whitelisted image-series fields. Reports and labels are never copied into its
    metadata directory. Its temporary image-tree symlink is removed on success
    or failure. Outputs are private artifacts and existing runs are not replaced.
    """
    import numpy as np

    root, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    paths = [Path(path) for path in head_paths]
    weights = validate_weights(len(paths), head_weights)
    id_column, targets = read_schema(root / "sample_submission.csv")
    with np.load(expected_predictions_path, allow_pickle=False) as expected:
        ids = expected["ids"].tolist()
        values = np.asarray(expected["predictions"], dtype=np.float64)
        if "targets" in expected and expected["targets"].tolist() != targets:
            raise ValueError("Expected prediction target order mismatch")
    if (not ids or any(not isinstance(uid, str) or not uid.strip() for uid in ids) or
            len(ids) != len(set(ids))):
        raise ValueError("Expected prediction IDs must be nonempty and unique")
    if values.shape != (len(ids), len(targets)) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Expected predictions must have the exact shape and finite probabilities")
    selected = set(ids)
    columns, series = read_csv(root / "train_series.csv")
    series_columns = [id_column, *SERIES_FIELDS]
    if not set(series_columns) <= set(columns):
        raise ValueError("Missing required image-series metadata fields")
    selected_series = [row for row in series if row[id_column] in selected]
    if {row[id_column] for row in selected_series} != selected:
        raise ValueError("Gold studies lack image-series metadata")
    output.mkdir(parents=True, exist_ok=True)
    gold_root, visible_root = output / "gold-rehearsal", output / "visible-rehearsal"
    if gold_root.exists() or visible_root.exists() or (output / "rehearsal.json").exists():
        raise FileExistsError("Rehearsal outputs already exist; use a fresh output directory")
    gold_root.mkdir()
    _csv(gold_root / "test.csv", [id_column], [[uid] for uid in ids])
    _csv(gold_root / "sample_submission.csv", [id_column, *targets],
         [[uid, *([.5] * len(targets))] for uid in ids])
    _csv(gold_root / "test_series.csv", series_columns,
         [[row[column] for column in series_columns] for row in selected_series])
    image_link = gold_root / "test_series"
    image_link.symlink_to(root / "train_series", target_is_directory=True)
    try:
        gold_report = infer(gold_root, encoder_dir, paths[0], gold_root / "predictions.csv",
                            additional_head_paths=paths[1:], head_weights=weights)
    finally:
        # Collected notebook output must never retain a link to the raw corpus.
        image_link.unlink(missing_ok=True)
    _, output_rows = read_csv(gold_root / "predictions.csv")
    predictions = label_matrix(output_rows, targets, id_column)
    if set(predictions) != selected:
        raise ValueError("Gold prediction ID coverage differs from expected predictions")
    observed = np.asarray([predictions[uid] for uid in ids], dtype=np.float64)
    if not np.isfinite(observed).all():
        raise ValueError("Gold predictions contain missing or nonfinite probabilities")
    max_error = float(np.max(np.abs(observed - values)))
    if not np.allclose(observed, values, atol=2e-4, rtol=2e-4):
        raise RuntimeError(f"Offline predictions disagree with frozen validation predictions: {max_error}")
    # Gold labels enter only after image-only inference and parity have passed.
    _, train_rows = read_csv(root / "train.csv")
    gold = label_matrix([row for row in train_rows if row[id_column] in selected], targets, id_column)
    metrics = evaluate_predictions(gold, predictions, targets)
    visible_root.mkdir()
    visible_report = infer(root, encoder_dir, paths[0], visible_root / "submission.csv", repeat=2,
                           additional_head_paths=paths[1:], head_weights=weights)
    report = {"status": "passed", "gold_prediction_max_error": max_error,
              "gold_metrics": metrics, "gold_resources": gold_report,
              "visible_test_resources": visible_report,
              "note": "No fitting; gold labels used only after prediction parity. Caller is responsible for offline execution."}
    with (output / "rehearsal.json").open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event": "rehearsal_passed", "gold_prediction_max_error": max_error,
                      "gold_macro_auc": metrics["macro_all_targets_auc"]}), flush=True)
    return report
