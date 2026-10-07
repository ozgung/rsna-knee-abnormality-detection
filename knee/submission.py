"""Submission writer: derive exact target names and row order from official sample."""
import csv
import math
from pathlib import Path


def write_submission(sample_path, predictions, output_path):
    with Path(sample_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    if len(header) != 13 or len(set(header)) != 13:
        raise ValueError("Expected one ID column and 12 distinct target columns")
    if any(len(row) != 13 for row in rows):
        raise ValueError("Malformed sample row")
    ids = [row[0] for row in rows]
    if not ids or any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Sample needs nonempty, unique study IDs")
    if set(ids) != set(predictions):
        raise ValueError("Prediction IDs must match sample exactly")
    validated = []
    for study_id in ids:
        values = [float(value) for value in predictions[study_id]]
        if len(values) != 12 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Each study needs 12 finite probabilities in [0, 1]")
        validated.append([study_id, *values])
    # Exclusive creation avoids accidentally overwriting a prior submission.
    with Path(output_path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(validated)
