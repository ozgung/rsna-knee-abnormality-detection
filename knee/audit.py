"""Inspect a user-provided CSV without printing report text or downloading data."""
import argparse
import csv
import json
from collections import Counter


def audit(path, id_column=None):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("Missing or duplicate column names")
        if id_column and id_column not in columns:
            raise ValueError(f"Missing ID column: {id_column}")
        rows, ids, missing = 0, set(), Counter()
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed CSV row")
            rows += 1
            missing.update(k for k, v in row.items() if not v.strip())
            if id_column and row[id_column].strip():
                ids.add(row[id_column])
    return {"columns": columns, "rows": rows, "missing": dict(missing),
            "unique_nonempty_ids": len(ids) if id_column else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--id-column")
    args = parser.parse_args()
    print(json.dumps(audit(args.csv_path, args.id_column), indent=2))
