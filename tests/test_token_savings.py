"""Unit tests for token savings calculation and reporting in Ollama BenchRig."""

import unittest
from unittest.mock import MagicMock
from core.runner import BenchmarkRunner
from reporting.display import display_token_savings
from reporting.markdown import generate_markdown_report


class TestTokenSavings(unittest.TestCase):
    """Test suite for token and cost savings calculations."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.runner = BenchmarkRunner(
            client=self.mock_client,
            config={
                "benchmark": {
                    "composite_weights": {"coding": 0.4, "reasoning": 0.3, "performance": 0.3}
                }
            }
        )
        self.runner.specs = {"memory_type": "Unified Memory (UMA)", "gpu_type": "apple_silicon"}

    def test_compute_scorecard_token_savings(self):
        raw_results = [
            {
                "model": "qwen2.5-coder:7b",
                "suite": "coding",
                "eval_count": 250,
                "prompt_eval_count": 500,
                "eval_tok_per_sec": 45.0,
                "prompt_tok_per_sec": 300.0,
                "ttft_sec": 0.05,
                "total_tests": 5,
                "passed_tests": 5,
                "pass_ratio": 1.0,
                "hardware": {"vram_peak_mb": 4500, "vram_warning": False},
            },
            {
                "model": "qwen2.5-coder:7b",
                "suite": "reasoning",
                "eval_count": 350,
                "prompt_eval_count": 200,
                "eval_tok_per_sec": 40.0,
                "prompt_tok_per_sec": 280.0,
                "ttft_sec": 0.06,
                "correct": True,
                "hardware": {"vram_peak_mb": 4600, "vram_warning": False},
            },
        ]

        scorecard = self.runner.compute_model_scorecard("qwen2.5-coder:7b", raw_results)

        self.assertIsNotNone(scorecard)
        self.assertEqual(scorecard["total_eval_tokens"], 600)
        self.assertEqual(scorecard["total_prompt_tokens"], 700)
        self.assertEqual(scorecard["total_tokens_saved"], 1300)

        # Expected cost: 700 * 0.000003 ($0.0021) + 600 * 0.000015 ($0.009) = $0.0111
        expected_cost = round((700 * 0.000003) + (600 * 0.000015), 4)
        self.assertAlmostEqual(scorecard["est_cost_saved_usd"], expected_cost, places=4)

    def test_display_token_savings_runs_cleanly(self):
        scorecards = [
            {
                "model": "qwen2.5-coder:7b",
                "total_prompt_tokens": 1500,
                "total_eval_tokens": 800,
                "total_tokens_saved": 2300,
                "est_cost_saved_usd": 0.0165,
            }
        ]
        # Should execute without throwing any exception
        display_token_savings(scorecards)

    def test_markdown_report_includes_savings_table(self):
        scorecards = [
            {
                "model": "qwen2.5-coder:7b",
                "composite_score": 95.0,
                "coding_pass_rate": 100.0,
                "reasoning_accuracy": 100.0,
                "avg_eval_tok_sec": 45.0,
                "avg_prompt_tok_sec": 300.0,
                "avg_ttft_sec": 0.05,
                "peak_vram_mb": 4500,
                "total_vram_mb": 36864,
                "vram_warning": False,
                "total_prompt_tokens": 1500,
                "total_eval_tokens": 800,
                "total_tokens_saved": 2300,
                "est_cost_saved_usd": 0.0165,
            }
        ]
        specs = {"platform_short": "Apple Silicon (Metal)", "memory_type": "UMA", "gpu_type": "apple_silicon"}

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            temp_path = f.name

        content = generate_markdown_report(scorecards, [], specs, output_path=temp_path)
        self.assertIn("Cloud Token & Cost Savings", content)
        self.assertIn("2,300", content)
        self.assertIn("$0.0165", content)
        self.assertIn("✅ $0.00", content)


if __name__ == "__main__":
    unittest.main()
