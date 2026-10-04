"""Tests for warm-up and KV cache reuse protocol (spec.md Phase 14, I14)."""

import unittest

from benchrig.cli import build_parser
from benchrig.core.client import BaseRuntimeClient
from benchrig.core.runner import BenchmarkRunner


class FakeWarmupClient(BaseRuntimeClient):
    name = "fake"
    display_name = "Fake"
    engine_name = "fake-engine"

    def __init__(self, responses=None):
        super().__init__(base_url="http://fake")
        self.responses = list(responses or [])
        self.call_count = 0

    def generate(self, model, prompt, system=None, options=None, measure_ttft=True):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
        else:
            resp = {
                "success": True,
                "response": "ok",
                "eval_count": 10,
                "eval_tok_per_sec": 50.0,
                "prompt_eval_count": 20,
                "ttft_sec": 0.1,
            }
        self.call_count += 1
        return dict(resp)


class WarmupProtocolTests(unittest.TestCase):
    def test_cli_parser_registers_warmup_runs_flag(self):
        """CLI parser must accept --warmup-runs and default to 1."""
        parser = build_parser()
        args_default = parser.parse_args([])
        self.assertEqual(args_default.warmup_runs, 1)

        args_custom = parser.parse_args(["--warmup-runs", "3"])
        self.assertEqual(args_custom.warmup_runs, 3)

    def test_warmup_runs_strictly_excluded_from_scorecard_aggregates(self):
        """compute_model_scorecard must exclude records with phase='warmup' from averages."""
        runner = BenchmarkRunner(client=None, config={})
        results = [
            {
                "model": "m",
                "suite": "speed",
                "phase": "warmup",
                "eval_tok_per_sec": 5.0,  # slow warmup
                "eval_count": 10,
                "prompt_eval_count": 20,
                "ttft_sec": 2.0,
            },
            {
                "model": "m",
                "suite": "speed",
                "phase": "measured",
                "eval_tok_per_sec": 100.0,  # steady state
                "eval_count": 10,
                "prompt_eval_count": 20,
                "ttft_sec": 0.1,
            },
        ]
        sc = runner.compute_model_scorecard("m", results)
        # Average eval speed must be exactly 100.0, NOT 52.5
        self.assertAlmostEqual(sc["avg_eval_tok_sec"], 100.0, places=1)
        self.assertAlmostEqual(sc["avg_ttft_sec"], 0.1, places=2)

    def test_runner_executes_warmup_runs_and_tags_phase(self):
        """Runner configured with warmup_runs=1 executes warm-up and tags phase='warmup'."""
        client = FakeWarmupClient()
        runner = BenchmarkRunner(client=client, config={}, warmup_runs=1)
        scenarios = [{"id": "s1", "name": "Scenario 1", "prompt": "test prompt"}]
        results = runner.run_speed_suite("m", scenarios)

        # Should have 2 results: 1 warmup + 1 measured
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["phase"], "warmup")
        self.assertEqual(results[1]["phase"], "measured")

    def test_cache_mode_attribution_verified_vs_unverified(self):
        """cache_mode is stamped as 'prefix_cached' when verified by telemetry, or 'unverified' when unconfirmed."""
        runner = BenchmarkRunner(client=None, config={})
        # Verified prefix cache hit: prompt_eval_count is 0 on repeated prefix
        resp_verified = {
            "model": "m",
            "success": True,
            "eval_count": 10,
            "prompt_eval_count": 0,
            "ttft_sec": 0.02,
            "eval_tok_per_sec": 50.0,
            "prefix_cache_hit": True,
        }
        rec_verified = runner._base_record("speed", "s", "S", "m", resp_verified, {}, {})
        self.assertEqual(rec_verified.get("cache_mode"), "prefix_cached")

        # Unconfirmed prefix cache hit
        resp_unconfirmed = {
            "model": "m",
            "success": True,
            "eval_count": 10,
            "prompt_eval_count": 50,
            "ttft_sec": 0.1,
            "eval_tok_per_sec": 50.0,
            "repeat_workload": True,
        }
        rec_unconfirmed = runner._base_record("speed", "s", "S", "m", resp_unconfirmed, {}, {})
        self.assertEqual(rec_unconfirmed.get("cache_mode"), "unverified")

    def test_markdown_detailed_tables_exclude_warmup_records(self):
        """Detailed scenario tables in Markdown report must strictly exclude phase='warmup' records."""
        from benchrig.reporting.markdown import generate_markdown_report

        scorecards = [
            {
                "model": "m",
                "runtime": "ollama",
                "engine": "llama.cpp",
                "composite_score": 100.0,
                "coding_pass_rate": 100.0,
                "reasoning_accuracy": 100.0,
                "avg_eval_tok_sec": 50.0,
                "avg_ttft_sec": 0.1,
                "peak_vram_mb": 1000.0,
                "total_runs": 1,
            }
        ]
        raw_results = [
            {
                "model": "m",
                "suite": "coding",
                "name": "Nested Dict",
                "phase": "warmup",
                "passed": False,
                "passed_tests": 0,
                "total_tests": 4,
                "eval_tok_per_sec": 10.0,
            },
            {
                "model": "m",
                "suite": "coding",
                "name": "Nested Dict",
                "phase": "measured",
                "passed": True,
                "passed_tests": 4,
                "total_tests": 4,
                "eval_tok_per_sec": 50.0,
            },
        ]
        report = generate_markdown_report(scorecards, raw_results, {"platform": "Linux"})
        # Should not display the warmup failure
        self.assertNotIn("❌ FAIL", report)
        self.assertIn("✅ PASS", report)
        self.assertEqual(report.count("Nested Dict"), 1)

    def test_cache_mode_is_unverified_without_backend_evidence(self):
        """Scenario order alone cannot establish runtime or prefix cache state."""
        client = FakeWarmupClient()
        runner = BenchmarkRunner(client=client, config={}, warmup_runs=1)
        scenarios = [{"id": "s1", "name": "Scenario 1", "prompt": "test prompt"}]
        results = runner.run_speed_suite("m", scenarios)
        self.assertEqual(results[0]["phase"], "warmup")
        self.assertEqual(results[0]["cache_mode"], "unverified")
        self.assertEqual(results[1]["phase"], "measured")
        self.assertEqual(results[1]["cache_mode"], "unverified")


if __name__ == "__main__":
    unittest.main()


def test_first_scenario_after_model_warmup_is_not_cold():
    runner = BenchmarkRunner(FakeWarmupClient(), {})
    runner.warmup("m")
    record = runner._base_record("speed", "s", "S", "m", {"success": True}, {})
    assert record["runtime_state"] == "warm"
    assert record["prefix_cache_state"] == "unverified"
    assert record["cache_mode"] == "unverified"


def test_zero_prefill_count_without_cache_evidence_is_not_verified():
    runner = BenchmarkRunner(None, {})
    record = runner._base_record("speed", "s", "S", "m", {"repeat_workload": True, "prompt_eval_count": 0}, {})
    assert record["cache_mode"] == "unverified"


def test_runtime_reload_and_cache_evidence_are_independent():
    runner = BenchmarkRunner(FakeWarmupClient(), {})
    runner.warmup("m")
    response = {"success": True, "load_time_sec": 0.25, "runtime_initialized": True, "prefix_cache_hit": False}
    record = runner._base_record("speed", "s", "S", "m", response, {})
    assert record["runtime_state"] == "cold"
    assert record["prefix_cache_state"] == "miss"
    assert record["cache_mode"] == "cold"


def test_missing_load_timing_is_unavailable():
    record = BenchmarkRunner(None, {})._base_record("speed", "s", "S", "m", {}, {})
    assert record["load_time_sec"] is None
    assert record["load_provenance"] == "unavailable"


def test_positive_load_duration_does_not_prove_runtime_initialization():
    runner = BenchmarkRunner(FakeWarmupClient(), {})
    runner.warmup("m")
    record = runner._base_record("speed", "s", "S", "m", {"success": True, "load_time_sec": 0.001}, {})
    assert record["runtime_state"] == "warm"
