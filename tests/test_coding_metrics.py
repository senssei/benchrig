"""Tests for executable-test evidence and disaggregated coding metrics (spec.md Phase 14, I15)."""

import csv
import os
import tempfile
import unittest

from benchrig.core.runner import BenchmarkRunner
from benchrig.reporting.csv_export import SCORECARD_CSV_COLUMNS, write_scorecards_csv
from benchrig.reporting.markdown import generate_markdown_report


class CodingMetricsTests(unittest.TestCase):
    def test_csv_columns_contain_disaggregated_coding_metrics(self):
        """SCORECARD_CSV_COLUMNS must include coding_task_pass_rate, coding_tasks_passed, coding_task_count, etc."""
        self.assertIn("coding_task_pass_rate", SCORECARD_CSV_COLUMNS)
        self.assertIn("coding_tasks_passed", SCORECARD_CSV_COLUMNS)
        self.assertIn("coding_task_count", SCORECARD_CSV_COLUMNS)
        self.assertIn("coding_assertions_passed", SCORECARD_CSV_COLUMNS)
        self.assertIn("coding_assertion_count", SCORECARD_CSV_COLUMNS)

    def test_runner_scorecard_disaggregates_task_and_assertion_metrics(self):
        """compute_model_scorecard must separate task-level pass rate from assertion-level pass rate."""
        runner = BenchmarkRunner(client=None, config={})
        results = [
            # Task 1: 3/3 assertions passed -> task passed
            {
                "model": "m",
                "suite": "coding",
                "task_id": "c1",
                "passed": True,
                "passed_tests": 3,
                "total_tests": 3,
                "pass_ratio": 1.0,
            },
            # Task 2: 1/3 assertions passed -> task failed
            {
                "model": "m",
                "suite": "coding",
                "task_id": "c2",
                "passed": False,
                "passed_tests": 1,
                "total_tests": 3,
                "pass_ratio": 0.33,
            },
        ]
        sc = runner.compute_model_scorecard("m", results)

        # 1 of 2 tasks passed = 50.0%
        self.assertEqual(sc["coding_task_count"], 2)
        self.assertEqual(sc["coding_tasks_passed"], 1)
        self.assertAlmostEqual(sc["coding_task_pass_rate"], 50.0, places=1)

        # 4 of 6 assertions passed = 66.7%
        self.assertEqual(sc["coding_assertion_count"], 6)
        self.assertEqual(sc["coding_assertions_passed"], 4)
        self.assertAlmostEqual(sc["coding_pass_rate"], 66.7, places=1)

    def test_csv_round_trip_disaggregated_coding_metrics(self):
        """write_scorecards_csv correctly serializes disaggregated coding metrics."""
        scorecard = {
            "model": "m",
            "runtime": "ollama",
            "engine": "llama.cpp",
            "composite_score": 80.0,
            "coding_pass_rate": 66.7,
            "coding_task_pass_rate": 50.0,
            "coding_tasks_passed": 1,
            "coding_task_count": 2,
            "coding_assertions_passed": 4,
            "coding_assertion_count": 6,
            "reasoning_accuracy": 100.0,
            "avg_eval_tok_sec": 50.0,
            "avg_ttft_sec": 0.1,
            "peak_vram_mb": 1000.0,
            "total_runs": 1,
            "eval_tok_sec_floored": False,
            "requested_device": "gpu",
            "observed_device": "cuda",
            "cpu_fallback": False,
            "prefill_duration_sec": 0.05,
            "decode_duration_sec": 0.20,
            "prefill_provenance": "engine",
            "peak_rss_mb": 200.0,
            "rss_coverage": "client_only",
            "context_tokens": 2048,
        }
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "out.csv")
            write_scorecards_csv([scorecard], csv_path)
            with open(csv_path) as fh:
                rows = list(csv.reader(fh))
            task_rate_idx = rows[0].index("coding_task_pass_rate")
            tasks_passed_idx = rows[0].index("coding_tasks_passed")
            assert_count_idx = rows[0].index("coding_assertion_count")
            self.assertEqual(rows[1][task_rate_idx], "50.0")
            self.assertEqual(rows[1][tasks_passed_idx], "1")
            self.assertEqual(rows[1][assert_count_idx], "6")

    def test_markdown_report_displays_task_and_assertion_counts(self):
        """Markdown report highlights and/or tables display task and assertion pass counts."""
        scorecard = {
            "model": "m",
            "runtime": "ollama",
            "engine": "llama.cpp",
            "composite_score": 80.0,
            "coding_pass_rate": 66.7,
            "coding_task_pass_rate": 50.0,
            "coding_tasks_passed": 1,
            "coding_task_count": 2,
            "coding_assertions_passed": 4,
            "coding_assertion_count": 6,
            "reasoning_accuracy": 100.0,
            "avg_eval_tok_sec": 50.0,
            "avg_ttft_sec": 0.1,
            "peak_vram_mb": 1000.0,
            "total_runs": 1,
            "eval_tok_sec_floored": False,
            "requested_device": "gpu",
            "observed_device": "cuda",
            "cpu_fallback": False,
        }
        report = generate_markdown_report([scorecard], [], {"platform": "Linux"})
        self.assertIn("Tasks Passed", report)
        self.assertIn("1/2", report)


if __name__ == "__main__":
    unittest.main()
