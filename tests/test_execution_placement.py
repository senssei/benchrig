"""Tests for execution placement and CPU fallback reporting (spec.md Phase 14, I11)."""

import csv
import os
import tempfile
import unittest

from benchrig.core.runner import BenchmarkRunner
from benchrig.reporting.csv_export import SCORECARD_CSV_COLUMNS, write_scorecards_csv
from benchrig.reporting.markdown import generate_markdown_report


class ExecutionPlacementTests(unittest.TestCase):
    def test_csv_columns_contain_placement_fields(self):
        """SCORECARD_CSV_COLUMNS must include requested_device, observed_device, and cpu_fallback."""
        self.assertIn("requested_device", SCORECARD_CSV_COLUMNS)
        self.assertIn("observed_device", SCORECARD_CSV_COLUMNS)
        self.assertIn("cpu_fallback", SCORECARD_CSV_COLUMNS)

    def test_csv_round_trip_placement_fields(self):
        """write_scorecards_csv correctly serializes requested_device, observed_device, cpu_fallback."""
        scorecard = {
            "model": "m",
            "runtime": "r",
            "engine": "e",
            "composite_score": 10.0,
            "coding_pass_rate": 0.5,
            "reasoning_accuracy": 0.5,
            "avg_eval_tok_sec": 10.0,
            "avg_ttft_sec": 0.1,
            "peak_vram_mb": 100.0,
            "total_runs": 1,
            "eval_tok_sec_floored": False,
            "requested_device": "cuda",
            "observed_device": "CPU",
            "cpu_fallback": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "out.csv")
            write_scorecards_csv([scorecard], csv_path)
            with open(csv_path) as fh:
                rows = list(csv.reader(fh))
            req_idx = rows[0].index("requested_device")
            obs_idx = rows[0].index("observed_device")
            fall_idx = rows[0].index("cpu_fallback")
            self.assertEqual(rows[1][req_idx], "cuda")
            self.assertEqual(rows[1][obs_idx], "CPU")
            self.assertEqual(rows[1][fall_idx], "True")

    def test_runner_scorecard_detects_cpu_fallback(self):
        """Runner must set cpu_fallback=True when requested GPU but observed CPU."""
        runner = BenchmarkRunner(client=None, config={})
        results = [
            {
                "model": "test-model",
                "suite": "speed",
                "eval_tok_per_sec": 10.0,
                "eval_count": 50,
                "prompt_tok_per_sec": 15.0,
                "prompt_eval_count": 20,
                "ttft_sec": 0.5,
                "requested_device": "cuda",
                "observed_device": "CPU",
                "cpu_fallback": True,
            }
        ]
        sc = runner.compute_model_scorecard("test-model", results)
        self.assertEqual(sc.get("requested_device"), "cuda")
        self.assertEqual(sc.get("observed_device"), "CPU")
        self.assertTrue(sc.get("cpu_fallback"))

    def test_runner_scorecard_records_unknown_placement(self):
        """Runner must record observed_device as 'unknown' when backend provides no evidence."""
        runner = BenchmarkRunner(client=None, config={})
        results = [
            {
                "model": "test-model",
                "suite": "speed",
                "eval_tok_per_sec": 10.0,
                "eval_count": 50,
                "prompt_tok_per_sec": 15.0,
                "prompt_eval_count": 20,
                "ttft_sec": 0.5,
            }
        ]
        sc = runner.compute_model_scorecard("test-model", results)
        self.assertEqual(sc.get("observed_device"), "unknown")
        self.assertFalse(sc.get("cpu_fallback"))

    def test_markdown_report_renders_cpu_fallback_warning(self):
        """Markdown leaderboard must render fallback warning when cpu_fallback is True."""
        scorecards = [
            {
                "model": "fallback-model",
                "runtime": "foundry",
                "engine": "ONNX Runtime",
                "composite_score": 50.0,
                "coding_pass_rate": 0.0,
                "reasoning_accuracy": 0.0,
                "avg_eval_tok_sec": 5.0,
                "avg_ttft_sec": 1.0,
                "peak_vram_mb": 0.0,
                "requested_device": "cuda",
                "observed_device": "CPU",
                "cpu_fallback": True,
            }
        ]
        report = generate_markdown_report(
            scorecards,
            raw_results=[],
            system_specs={"platform": "Linux", "gpu_type": "nvidia"},
        )
        self.assertIn("CPU Fallback", report)
        self.assertNotIn("✅ 100% VRAM", report)

    def test_markdown_report_renders_unknown_device_without_assuming_vram(self):
        """Markdown report must not claim 100% VRAM when placement is unknown."""
        scorecards = [
            {
                "model": "unknown-model",
                "runtime": "ollama",
                "engine": "llama.cpp",
                "composite_score": 60.0,
                "coding_pass_rate": 0.0,
                "reasoning_accuracy": 0.0,
                "avg_eval_tok_sec": 20.0,
                "avg_ttft_sec": 0.3,
                "peak_vram_mb": 0.0,
                "requested_device": "gpu",
                "observed_device": "unknown",
                "cpu_fallback": False,
            }
        ]
        report = generate_markdown_report(
            scorecards,
            raw_results=[],
            system_specs={"platform": "Linux", "gpu_type": "nvidia"},
        )
        self.assertIn("Unknown", report)
        self.assertNotIn("✅ 100% VRAM", report)

    def test_markdown_report_renders_device_placement_column(self):
        """Leaderboard table must render Device / Placement column showing observed device."""
        scorecards = [
            {
                "model": "m",
                "runtime": "ollama",
                "engine": "llama.cpp",
                "composite_score": 60.0,
                "coding_pass_rate": 0.0,
                "reasoning_accuracy": 0.0,
                "avg_eval_tok_sec": 20.0,
                "avg_ttft_sec": 0.3,
                "peak_vram_mb": 0.0,
                "requested_device": "gpu",
                "observed_device": "CUDA (GPU)",
                "cpu_fallback": False,
            }
        ]
        report = generate_markdown_report(scorecards, [], {"platform": "Linux"})
        self.assertIn("Device / Placement", report)
        self.assertIn("✅ CUDA (GPU)", report)

    def test_ollama_client_inspects_running_models_for_placement(self):
        """OllamaClient.generate() queries get_running_models() to detect CPU fallback when metrics omit device."""
        from unittest.mock import MagicMock, patch

        from benchrig.core.client import OllamaClient

        client = OllamaClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Simulated Ollama streaming chunks without device in final_metrics
        mock_response.iter_lines.return_value = [
            b'{"model":"phi4","response":"hello","done":false}',
            b'{"model":"phi4","response":"","done":true,"eval_count":1,"eval_duration":100000000}',
        ]
        with patch.object(client, "_make_request", return_value=mock_response):
            with patch.object(
                client,
                "get_running_models",
                return_value=[{"name": "phi4:latest", "size": 1000000, "size_vram": 0}],
            ):
                result = client.generate("phi4", "prompt")
                self.assertEqual(result.get("observed_device"), "cpu")
                self.assertTrue(result.get("cpu_fallback"))


if __name__ == "__main__":
    unittest.main()
