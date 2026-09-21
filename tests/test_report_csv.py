"""Tests for the CSV export of scorecards (plan.md Phase 1, item 1.1)."""

import csv
import os
import tempfile
import unittest


def card(model, runtime="ollama", engine="llama.cpp", **extra):
    return {
        "model": model,
        "runtime": runtime,
        "engine": engine,
        "composite_score": 90.0,
        "coding_pass_rate": 100.0,
        "reasoning_accuracy": 66.7,
        "avg_eval_tok_sec": 100.0,
        "avg_ttft_sec": 0.1,
        "peak_vram_mb": 5000.0,
        "total_runs": 3,
        **extra,
    }


class CsvExportTests(unittest.TestCase):
    def test_csv_export_writes_one_row_per_scorecard_with_expected_columns(self):
        # Import inside the test so a missing module reads as "not found" / red, not as a collection error.
        from benchrig.reporting.csv_export import SCORECARD_CSV_COLUMNS, write_scorecards_csv

        scorecards = [
            card("phi4-mini:latest", runtime="ollama"),
            card("deepseek-r1:14b", runtime="foundry", engine="onnx"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scorecards.csv")
            write_scorecards_csv(scorecards, path)
            with open(path, newline="") as fh:
                rows = list(csv.reader(fh))

        self.assertEqual(rows[0], list(SCORECARD_CSV_COLUMNS))
        self.assertEqual(len(rows), 1 + len(scorecards))
        # Each data row has one cell per header column.
        for row in rows[1:]:
            self.assertEqual(len(row), len(SCORECARD_CSV_COLUMNS))

    def test_csv_export_returns_path(self):
        from benchrig.reporting.csv_export import write_scorecards_csv

        scorecards = [card("phi4-mini:latest")]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scorecards.csv")
            returned = write_scorecards_csv(scorecards, path)
        self.assertEqual(returned, path)

    def test_csv_export_handles_empty_scorecards(self):
        from benchrig.reporting.csv_export import SCORECARD_CSV_COLUMNS, write_scorecards_csv

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scorecards.csv")
            write_scorecards_csv([], path)
            with open(path, newline="") as fh:
                rows = list(csv.reader(fh))
        # Header only, no data rows.
        self.assertEqual(rows, [list(SCORECARD_CSV_COLUMNS)])


if __name__ == "__main__":
    unittest.main()
