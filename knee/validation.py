"""Dependency-free metadata audit, gold holdout and masked AUC.

CLI output contains aggregates only. Split IDs belong in ignored outputs.
"""
import argparse
import csv
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("Missing or duplicate CSV columns")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed CSV row")
    return columns, rows


def _index(rows, id_column):
    if any(id_column not in row for row in rows):
        raise ValueError("Missing ID column")
    ids = [row[id_column] for row in rows]
    if any(not value.strip() for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("IDs must be nonempty and unique")
    return dict(zip(ids, rows))


def read_schema(sample_path):
    columns, rows = read_csv(sample_path)
    if len(columns) != 13 or not rows:
        raise ValueError("Sample must have one ID column and 12 targets, with rows")
    _index(rows, columns[0])
    return columns[0], columns[1:]


def _number(value, *, binary=False):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("Labels/probabilities must be finite and in [0, 1]")
    if binary and number not in (0.0, 1.0):
        raise ValueError("Evaluation labels must be binary or missing")
    return number


def label_matrix(rows, targets, id_column):
    """Return ID-keyed labels; blanks remain None, never implicit negatives."""
    indexed = _index(rows, id_column)
    if any(target not in row for row in rows for target in targets):
        raise ValueError("Missing target column")
    return {key: [_number(row[target]) for target in targets] for key, row in indexed.items()}


def _report_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def make_gold_holdout(rows, targets, id_column, patient_column=None, report_column="Report"):
    """Hold out every gold study; exclude related weak rows transitively.

    Group on patient identity and exact normalized reports. Missing patient IDs
    and empty reports never connect unrelated studies. Paraphrases are not found.
    """
    labels = label_matrix(rows, targets, id_column)
    if patient_column and any(patient_column not in row for row in rows):
        raise ValueError("Missing patient identity column")
    parent = {key: key for key in labels}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    patients, reports = {}, {}
    for row in rows:
        key = row[id_column]
        for value, seen in (
            (row.get(patient_column, "").strip() if patient_column else "", patients),
            (_report_key(row.get(report_column, "")), reports),
        ):
            if value:
                if value in seen:
                    parent[find(key)] = find(seen[value])
                else:
                    seen[value] = key
    gold = {key for key, values in labels.items() if any(value is not None for value in values)}
    gold_groups = {find(key) for key in gold}
    assignments = {
        key: "validation" if key in gold else
        "excluded_related_to_validation" if find(key) in gold_groups else "train"
        for key in sorted(labels)
    }
    return assignments, {
        "policy": "all trusted-label studies held out; patient/report components touching gold excluded from training",
        "grouping": "patient and normalized-report duplicates" if patient_column else "study and normalized-report duplicates",
        "patient_column": patient_column,
        "missing_patient_identity": sum(not row[patient_column].strip() for row in rows) if patient_column else len(rows),
        "limitation": "Near-duplicate reports are not detected. " + (
            "Rows missing patient identity have no patient-level guarantee." if patient_column else
            "No patient identity supplied; repeat patients with distinct reports cannot be detected."
        ),
        "counts": dict(Counter(assignments.values())),
        "group_count": len({find(key) for key in labels}),
        "validation_group_count": len(gold_groups),
        "fingerprint_sha256": hashlib.sha256(json.dumps(assignments, sort_keys=True).encode()).hexdigest(),
    }


def _auc(pairs):
    positives = sum(label == 1 for label, _ in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(pairs, key=lambda pair: pair[1])
    positive_ranks, start = 0.0, 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        mean_rank = (start + 1 + end) / 2
        positive_ranks += mean_rank * sum(label == 1 for label, _ in ordered[start:end])
        start = end
    return (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate_predictions(labels, predictions, targets):
    """Require exact ID coverage; mask missing truth, list undefined target AUCs."""
    if not labels or set(labels) != set(predictions):
        raise ValueError("Prediction IDs must match nonempty evaluation labels exactly")
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Targets must be nonempty and unique")
    truth, scores = {}, {}
    for key in labels:
        if len(labels[key]) != len(targets) or len(predictions[key]) != len(targets):
            raise ValueError("Target width mismatch")
        truth[key] = [_number(value, binary=True) for value in labels[key]]
        scores[key] = [_number(value) for value in predictions[key]]
        if any(value is None for value in scores[key]):
            raise ValueError("Predictions may not be missing")
    per_target = {}
    for index, target in enumerate(targets):
        pairs = [(truth[key][index], scores[key][index]) for key in truth if truth[key][index] is not None]
        positive = sum(label == 1 for label, _ in pairs)
        per_target[target] = {"known": len(pairs), "positive": positive,
                              "negative": len(pairs) - positive,
                              "missing": len(truth) - len(pairs), "auc": _auc(pairs)}
    defined = [row["auc"] for row in per_target.values() if row["auc"] is not None]
    undefined = [target for target, row in per_target.items() if row["auc"] is None]
    mean = sum(defined) / len(defined) if defined else None
    return {
        "studies": len(labels), "per_target": per_target, "undefined_targets": undefined,
        "macro_all_targets_auc": mean if not undefined else None,
        "macro_defined_targets_auc": mean,
        "aggregation_policy": "Primary mean requires all targets defined; secondary mean excludes and lists targets without both classes. Missing truth labels are masked.",
    }


def fit_prevalence(labels, targets, training_ids):
    """Fit explicit training IDs only, allowing soft labels; no-label prior is null."""
    training_ids = list(training_ids)
    if len(training_ids) != len(set(training_ids)) or not set(training_ids) <= set(labels):
        raise ValueError("Training IDs must be unique and present in labels")
    selected = [labels[key] for key in training_ids]
    if any(len(row) != len(targets) for row in selected):
        raise ValueError("Target width mismatch")
    selected = [[_number(value) for value in row] for row in selected]
    result = {}
    for index, target in enumerate(targets):
        known = [row[index] for row in selected if row[index] is not None]
        result[target] = {"known_training_labels": len(known), "probability": sum(known) / len(known) if known else None}
    return result


def _identity_columns(columns):
    return [column for column in columns if column.lower().replace("_", "").replace(" ", "")
            in {"patientid", "patientidentifier", "subjectid", "patientinstanceuid"}]


def _coverage(series, ids, id_column):
    studies = {row[id_column] for row in series}
    counts = Counter(row[id_column] for row in series)
    planes = defaultdict(set)
    for row in series:
        planes[row[id_column]].add(row.get("Anatomical_Plane", ""))
    return {"series_rows": len(series), "studies": len(studies),
            "missing_expected_studies": len(ids - studies), "unexpected_studies": len(studies - ids),
            "series_per_study": {"min": min(counts.values(), default=0), "max": max(counts.values(), default=0),
                                 "mean": len(series) / len(studies) if studies else None},
            "plane_study_counts": dict(Counter(plane for values in planes.values() for plane in values)),
            "studies_by_plane_count": dict(sorted(Counter(len(values) for values in planes.values()).items()))}


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit_metadata(train_path, train_series_path, test_series_path, sample_path):
    paths = {"train": train_path, "train_series": train_series_path, "test_series": test_series_path, "sample_submission": sample_path}
    tables = {name: read_csv(path) for name, path in paths.items()}
    id_column, targets = read_schema(sample_path)
    train_columns, train = tables["train"]
    train_series, test_series, sample = [tables[name][1] for name in ("train_series", "test_series", "sample_submission")]
    labels = label_matrix(train, targets, id_column)
    for values in labels.values():
        for value in values:
            _number(value, binary=True)
    train_ids, test_ids = set(labels), set(_index(sample, id_column))
    for series in (train_series, test_series):
        _index(series, "SeriesInstanceUID")
        if any(not row.get(id_column, "").strip() for row in series):
            raise ValueError("Missing series study identity")
    reports = Counter(_report_key(row.get("Report", "")) for row in train)
    reports.pop("", None)
    patient_columns = _identity_columns(train_columns)
    if len(patient_columns) > 1:
        raise ValueError("Multiple patient identity fields; choose one explicitly before splitting")
    # Do not ignore patient identity present only in a series table.
    if any(_identity_columns(tables[name][0]) for name in ("train_series", "test_series")) and not patient_columns:
        raise ValueError("Patient identity exists in series metadata; join it to studies before splitting")
    split, split_summary = make_gold_holdout(train, targets, id_column, patient_columns[0] if patient_columns else None)
    heldout = {key: labels[key] for key, role in split.items() if role == "validation"}
    neutral = evaluate_predictions(heldout, {key: [0.5] * len(targets) for key in heldout}, targets) if heldout else None
    return {
        "schema": {"id_column": id_column, "targets": targets},
        "files": {name: {"sha256": file_sha256(path), "columns": tables[name][0], "rows": len(tables[name][1]),
                         "missing": {col: sum(not row[col].strip() for row in tables[name][1]) for col in tables[name][0]},
                         "patient_identity_columns": _identity_columns(tables[name][0])} for name, path in paths.items()},
        "label_studies": {"any_known": sum(any(v is not None for v in row) for row in labels.values()),
                          "fully_known": sum(all(v is not None for v in row) for row in labels.values()),
                          "all_missing": sum(all(v is None for v in row) for row in labels.values())},
        "report_duplicates": {"normalization": "Unicode NFKC, casefold, collapsed whitespace; exact matches only",
                              "groups": sum(count > 1 for count in reports.values()),
                              "studies": sum(count for count in reports.values() if count > 1)},
        "overlap": {"train_test_studies": len(train_ids & test_ids),
                    "train_test_series": len({r["SeriesInstanceUID"] for r in train_series} & {r["SeriesInstanceUID"] for r in test_series})},
        "train_series_coverage": _coverage(train_series, train_ids, id_column),
        "test_series_coverage": _coverage(test_series, test_ids, id_column),
        "split": split_summary,
        "training_prevalence": fit_prevalence(labels, targets, [key for key, role in split.items() if role == "train"]),
        "neutral_reference": neutral,
        "neutral_reference_note": "Constant 0.5 ranking reference only; no prevalence estimated from held-out gold. Public test rows are examples, not hidden-test size.",
    }, split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit, split = audit_metadata(*(args.data_dir / name for name in ("train.csv", "train_series.csv", "test_series.csv", "sample_submission.csv")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    with (args.output_dir / "split.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([audit["schema"]["id_column"], "split"])
        writer.writerows(split.items())
    print(json.dumps({"label_studies": audit["label_studies"], "split": audit["split"], "overlap": audit["overlap"], "neutral_macro_auc": audit["neutral_reference"]["macro_all_targets_auc"] if audit["neutral_reference"] else None}, indent=2))


if __name__ == "__main__":
    main()
