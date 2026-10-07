import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from knee.validation import (
    audit_metadata, evaluate_predictions, fit_prevalence, label_matrix,
    make_gold_holdout, read_csv, read_schema,
)


class HoldoutTests(unittest.TestCase):
    def row(self, key, report="", patient="", label=""):
        return {"study": key, "Report": report, "patient": patient, "finding": label}

    def test_transitive_patient_and_report_components_excluded(self):
        rows = [self.row("gold", "Alpha report", "p1", "1"),
                self.row("same_patient", "Other report", "p1"),
                self.row("same_report", "  OTHER\nREPORT ", "p2"),
                self.row("next_patient", "Different report", "p2"),
                self.row("clean", "Unrelated report", "p3")]
        split, summary = make_gold_holdout(rows, ["finding"], "study", "patient")
        self.assertEqual(split["gold"], "validation")
        self.assertEqual(split["clean"], "train")
        for key in ("same_patient", "same_report", "next_patient"):
            self.assertEqual(split[key], "excluded_related_to_validation")
        self.assertEqual(summary["counts"], {"validation": 1, "train": 1, "excluded_related_to_validation": 3})
        reversed_split, reversed_summary = make_gold_holdout(rows[::-1], ["finding"], "study", "patient")
        self.assertEqual(split, reversed_split)
        self.assertEqual(summary["fingerprint_sha256"], reversed_summary["fingerprint_sha256"])

    def test_empty_report_or_identity_does_not_join_unrelated_rows(self):
        rows = [self.row("gold", label="0"), self.row("weak1"), self.row("weak2")]
        split, _ = make_gold_holdout(rows, ["finding"], "study", "patient")
        self.assertEqual(split, {"gold": "validation", "weak1": "train", "weak2": "train"})

    def test_unicode_normalization_and_all_gold_retained(self):
        rows = [self.row("g1", "ＡＢＣ", label="1"), self.row("g2", "abc", label="0"),
                self.row("w1", "ABC"), self.row("clean", "xyz")]
        split, summary = make_gold_holdout(rows, ["finding"], "study")
        self.assertEqual(split["g1"], "validation")
        self.assertEqual(split["g2"], "validation")
        self.assertEqual(split["w1"], "excluded_related_to_validation")
        self.assertEqual(summary["validation_group_count"], 1)
        self.assertIn("No patient identity", summary["limitation"])

    def test_partial_gold_is_validation_and_missing_labels_remain_missing(self):
        rows = [{"study": "a", "x": "", "y": "0"}, {"study": "b", "x": "", "y": ""}]
        labels = label_matrix(rows, ["x", "y"], "study")
        self.assertEqual(labels["a"], [None, 0])
        self.assertEqual(make_gold_holdout(rows, ["x", "y"], "study")[0], {"a": "validation", "b": "train"})

    def test_duplicate_ids_and_missing_patient_column_rejected(self):
        with self.assertRaises(ValueError):
            label_matrix([self.row("a"), self.row("a")], ["finding"], "study")
        with self.assertRaises(ValueError):
            make_gold_holdout([self.row("a")], ["finding"], "study", "wrong")


class EvaluationTests(unittest.TestCase):
    def test_tie_aware_auc_and_missing_truth(self):
        labels = {"a": [1, 0], "b": [0, 1], "c": [1, None], "d": [0, None]}
        preds = {"a": [0.5, 0.1], "b": [0.5, 0.9], "c": [0.9, 0.4], "d": [0.1, 0.4]}
        result = evaluate_predictions(labels, preds, ["x", "y"])
        self.assertEqual(result["per_target"]["x"]["auc"], 0.875)
        self.assertEqual(result["per_target"]["y"]["auc"], 1)
        self.assertEqual(result["per_target"]["y"]["missing"], 2)
        self.assertEqual(result["macro_all_targets_auc"], 0.9375)

    def test_undefined_targets_are_listed_and_primary_withheld(self):
        labels = {"a": [1, 1, None], "b": [0, 1, None]}
        preds = {key: [0.5] * 3 for key in labels}
        result = evaluate_predictions(labels, preds, ["mixed", "one_class", "missing"])
        self.assertIsNone(result["macro_all_targets_auc"])
        self.assertEqual(result["macro_defined_targets_auc"], 0.5)
        self.assertEqual(result["undefined_targets"], ["one_class", "missing"])
        self.assertEqual(result["per_target"]["one_class"]["positive"], 2)
        self.assertEqual(result["per_target"]["one_class"]["negative"], 0)
        self.assertEqual(result["per_target"]["missing"]["known"], 0)

    def test_exact_ids_width_binary_truth_and_finite_predictions(self):
        for labels, predictions in [
            ({"a": [0]}, {"b": [0.5]}), ({"a": [0]}, {"a": [0.5], "extra": [0.5]}),
            ({"a": [0]}, {"a": []}), ({"a": [0.2]}, {"a": [0.5]}),
            ({"a": [0]}, {"a": [math.nan]}), ({"a": [0]}, {"a": [math.inf]}),
            ({"a": [None]}, {"a": [None]}), ({"a": [0]}, {"a": [1.1]}),
        ]:
            with self.subTest(labels=labels, predictions=predictions), self.assertRaises(ValueError):
                evaluate_predictions(labels, predictions, ["x"])

    def test_training_prevalence_ignores_heldout_and_masks_missing(self):
        labels = {"train1": [0, None], "train2": [0.6, None], "heldout": [1, 1]}
        result = fit_prevalence(labels, ["x", "y"], ["train1", "train2"])
        self.assertEqual(result["x"], {"known_training_labels": 2, "probability": 0.3})
        self.assertEqual(result["y"], {"known_training_labels": 0, "probability": None})
        with self.assertRaises(ValueError):
            fit_prevalence(labels, ["x", "y"], ["missing"])
        with self.assertRaises(ValueError):
            fit_prevalence(labels, ["x", "y"], ["train1", "train1"])


class MetadataTests(unittest.TestCase):
    def write(self, root, name, columns, rows):
        path = root / name
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def test_complete_audit_uses_official_order_and_emits_no_private_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = [f"target_{i}" for i in range(12)][::-1]
            columns = ["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"]
            train = self.write(root, "train.csv", ["StudyInstanceUID", "Report", *targets], [
                ["private_gold_a", "Sensitive report alpha", *([1] * 12)],
                ["private_gold_b", "Sensitive report beta", *([0] * 12)],
                ["private_related", "SENSITIVE REPORT ALPHA", *([""] * 12)],
                ["private_train", "Sensitive report gamma", *([""] * 12)],
            ])
            series = self.write(root, "series.csv", columns, [
                [key, f"series_{i}", "SAG"] for i, key in enumerate(
                    ["private_gold_a", "private_gold_b", "private_related", "private_train"])])
            test = self.write(root, "test.csv", columns, [["private_test", "test_series", "COR"]])
            sample = self.write(root, "sample.csv", ["StudyInstanceUID", *targets], [["private_test", *([0.5] * 12)]])
            audit, split = audit_metadata(train, series, test, sample)
            self.assertEqual(audit["schema"]["targets"], targets)
            self.assertEqual(audit["label_studies"], {"any_known": 2, "fully_known": 2, "all_missing": 2})
            self.assertEqual(audit["report_duplicates"]["groups"], 1)
            self.assertEqual(audit["split"]["counts"]["train"], 1)
            self.assertEqual(audit["neutral_reference"]["macro_all_targets_auc"], 0.5)
            self.assertEqual(audit["train_series_coverage"]["missing_expected_studies"], 0)
            self.assertEqual(audit["overlap"], {"train_test_studies": 0, "train_test_series": 0})
            for prior in audit["training_prevalence"].values():
                self.assertIsNone(prior["probability"])
            self.assertEqual(len(audit["files"]["train"]["sha256"]), 64)
            serialized = json.dumps(audit)
            for private_value in ["Sensitive report", "SENSITIVE REPORT", *split.keys(), "private_test"]:
                self.assertNotIn(private_value, serialized)

    def test_rejects_duplicate_headers_malformed_rows_and_bad_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = root / "bad.csv"
            for content in ("a,a\n1,2\n", "a,b\n1\n", "a,b\n1,2,3\n"):
                bad.write_text(content)
                with self.assertRaises(ValueError):
                    read_csv(bad)
            bad.write_text("a,b\n1,2\n")
            with self.assertRaises(ValueError):
                read_schema(bad)


if __name__ == "__main__":
    unittest.main()
