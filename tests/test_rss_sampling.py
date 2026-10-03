"""Tests for host RSS sampling and fixed context attribution (spec.md Phase 14, I13)."""

import os
import unittest
from unittest.mock import patch

from benchrig.core.hardware import HardwareSampler
from benchrig.core.runner import BenchmarkRunner
from benchrig.reporting.csv_export import SCORECARD_CSV_COLUMNS


class RssSamplingTests(unittest.TestCase):
    def test_get_process_rss_mb_returns_positive_float_for_current_process(self):
        """get_process_rss_mb(os.getpid()) must return a positive float representing host RSS in MB."""
        from benchrig.core.hardware import get_process_rss_mb

        rss = get_process_rss_mb(os.getpid())
        self.assertIsInstance(rss, float)
        self.assertGreater(rss, 0.0)

    def test_hardware_sampler_samples_client_only_rss(self):
        """When no server PID is provided, HardwareSampler reports rss_coverage='client_only' and positive peak_rss_mb."""
        sampler = HardwareSampler(interval_sec=0.01)
        sampler.start()
        sampler._stop_event.wait(0.03)
        summary = sampler.stop()

        self.assertIn("peak_rss_mb", summary)
        self.assertIn("rss_coverage", summary)
        self.assertEqual(summary["rss_coverage"], "client_only")
        self.assertGreater(summary["peak_rss_mb"], 0.0)

    def test_hardware_sampler_samples_client_and_server_rss(self):
        """When server_pid is provided, HardwareSampler reports rss_coverage='client_and_server'."""
        with patch(
            "benchrig.core.hardware.get_process_rss_mb", side_effect=lambda pid: 150.0 if pid == 99999 else 50.0
        ):
            sampler = HardwareSampler(interval_sec=0.01, server_pid=99999)
            sampler.start()
            sampler._stop_event.wait(0.03)
            summary = sampler.stop()

            self.assertEqual(summary["rss_coverage"], "client_and_server")
            self.assertAlmostEqual(summary["peak_rss_mb"], 200.0, places=1)

    def test_runner_records_fixed_context_workload_parameters(self):
        """Runner _base_record must record context_tokens, prompt_tokens, and max_output_tokens."""
        runner = BenchmarkRunner(client=None, config={})
        resp = {
            "model": "m",
            "success": True,
            "eval_count": 20,
            "prompt_eval_count": 50,
            "ttft_sec": 0.1,
            "eval_tok_per_sec": 10.0,
        }
        options = {"num_ctx": 2048, "num_predict": 128}
        hw = {"peak_rss_mb": 120.0, "rss_coverage": "client_only"}
        rec = runner._base_record("speed", "s1", "Scenario 1", "m", resp, hw, options)

        self.assertEqual(rec["context_tokens"], 2048)
        self.assertEqual(rec["prompt_tokens"], 50)
        self.assertEqual(rec["max_output_tokens"], 128)

    def test_scorecard_aggregates_rss_and_context(self):
        """compute_model_scorecard must aggregate peak_rss_mb, rss_coverage, and context_tokens."""
        runner = BenchmarkRunner(client=None, config={})
        results = [
            {
                "model": "m",
                "suite": "speed",
                "eval_tok_per_sec": 20.0,
                "eval_count": 10,
                "prompt_eval_count": 20,
                "ttft_sec": 0.1,
                "context_tokens": 1024,
                "hardware": {"peak_rss_mb": 180.0, "rss_coverage": "client_only"},
            },
            {
                "model": "m",
                "suite": "speed",
                "eval_tok_per_sec": 20.0,
                "eval_count": 10,
                "prompt_eval_count": 20,
                "ttft_sec": 0.1,
                "context_tokens": 2048,
                "hardware": {"peak_rss_mb": 250.0, "rss_coverage": "client_and_server"},
            },
        ]
        sc = runner.compute_model_scorecard("m", results)

        self.assertAlmostEqual(sc["peak_rss_mb"], 250.0, places=1)
        self.assertEqual(sc["rss_coverage"], "client_and_server")
        self.assertIn("context_tokens", sc)
        self.assertIn("peak_rss_mb", SCORECARD_CSV_COLUMNS)
        self.assertIn("rss_coverage", SCORECARD_CSV_COLUMNS)

    def test_get_process_rss_prefers_getrusage_for_own_pid_without_subprocess(self):
        """get_process_rss_mb must use getrusage for os.getpid() without spawning ps subprocess."""
        from unittest.mock import patch

        from benchrig.core.hardware import get_process_rss_mb

        with patch("sys.platform", "darwin"):
            with patch("builtins.open", side_effect=FileNotFoundError):
                with patch("subprocess.check_output") as mock_ps:
                    rss = get_process_rss_mb(os.getpid())
                    # Should not invoke ps subprocess for current process
                    mock_ps.assert_not_called()
                    self.assertGreaterEqual(rss, 0.0)


if __name__ == "__main__":
    unittest.main()
