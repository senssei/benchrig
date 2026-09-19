"""Comprehensive unit tests covering Microsoft Foundry Server Runtime integration and cross-runtime benchmarking."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from benchmark import resolve_target_models
from core.client import (
    BaseRuntimeClient,
    FoundryClient,
    OllamaClient,
    create_runtime_client,
)
from core.runner import BenchmarkRunner
from reporting.display import display_leaderboard, display_scenario_result
from reporting.markdown import generate_markdown_report


class TestFoundryClient(unittest.TestCase):
    """Test suite for Microsoft Foundry Server Runtime client."""

    def setUp(self):
        self.client = FoundryClient(
            base_url="http://localhost:5272/v1",
            timeout_sec=30,
            auto_detect_port=False,
            cli_path="foundry",
        )

    def test_client_attributes(self):
        """Verify standard runtime attributes."""
        self.assertEqual(self.client.name, "foundry")
        self.assertEqual(self.client.display_name, "MS Foundry")
        self.assertEqual(self.client.engine_name, "ONNX Runtime GenAI")
        self.assertEqual(self.client.base_url, "http://localhost:5272/v1")

    def test_endpoint_resolution(self):
        """Verify URL path construction handles base URL variations."""
        c1 = FoundryClient(base_url="http://localhost:5272/v1", auto_detect_port=False)
        self.assertEqual(c1._get_api_endpoint("chat/completions"), "http://localhost:5272/v1/chat/completions")
        self.assertEqual(c1._get_api_endpoint("models"), "http://localhost:5272/v1/models")

        c2 = FoundryClient(base_url="http://localhost:5272", auto_detect_port=False)
        self.assertEqual(c2._get_api_endpoint("chat/completions"), "http://localhost:5272/v1/chat/completions")

    @patch("requests.get")
    def test_is_reachable_success(self, mock_get):
        """Verify is_reachable returns True when models endpoint responds."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        self.assertTrue(self.client.is_reachable())
        mock_get.assert_called()

    @patch("requests.get")
    def test_is_reachable_failure(self, mock_get):
        """Verify is_reachable returns False when connection fails."""
        mock_get.side_effect = Exception("Connection refused")
        self.assertFalse(self.client.is_reachable())

    @patch("requests.get")
    def test_list_installed_models_rest(self, mock_get):
        """Verify parsing of OpenAI-compatible /v1/models response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"id": "phi-4", "object": "model"},
                {"id": "qwen2.5-coder-7b", "object": "model"},
            ]
        }
        mock_get.return_value = mock_resp

        models = self.client.list_installed_models()
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["name"], "phi-4")
        self.assertEqual(models[0]["runtime"], "foundry")
        self.assertEqual(models[0]["details"]["engine"], "ONNX Runtime GenAI")
        self.assertEqual(models[1]["name"], "qwen2.5-coder-7b")

    @patch("requests.post")
    def test_generate_streaming_success(self, mock_post):
        """Verify streaming SSE chat completion captures TTFT and token speeds."""
        # Simulated SSE chunks from Foundry Local
        sse_lines = [
            b'data: {"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant","content":""}}]}',
            b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"def add(a, b):\\n"}}]}',
            b'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"    return a + b"}}]}',
            b'data: {"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":18,"total_tokens":30}}',
            b"data: [DONE]",
        ]

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_lines.return_value = sse_lines
        mock_post.return_value = mock_resp

        result = self.client.generate(
            model="phi-4",
            prompt="Write an add function",
            system="You are an expert Python engineer.",
            options={"temperature": 0.2, "num_predict": 64},
            measure_ttft=True,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["model"], "phi-4")
        self.assertEqual(result["runtime"], "foundry")
        self.assertEqual(result["engine"], "ONNX Runtime GenAI")
        self.assertEqual(result["response"], "def add(a, b):\n    return a + b")
        self.assertEqual(result["prompt_eval_count"], 12)
        self.assertEqual(result["eval_count"], 18)
        self.assertGreater(result["ttft_sec"], 0.0)
        self.assertGreater(result["eval_tok_per_sec"], 0.0)

    @patch("requests.post")
    def test_generate_non_streaming(self, mock_post):
        """Verify non-streaming chat completion parsing."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "42"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2},
        }
        mock_post.return_value = mock_resp

        result = self.client.generate(
            model="phi-4",
            prompt="What is the answer?",
            measure_ttft=False,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["response"], "42")
        self.assertEqual(result["eval_count"], 2)
        self.assertEqual(result["prompt_eval_count"], 8)

    @patch("requests.post")
    def test_generate_error_handling(self, mock_post):
        """Verify error handling returns structured failure dictionary."""
        mock_post.side_effect = Exception("HTTP 500 Internal Server Error")

        result = self.client.generate(
            model="phi-4",
            prompt="Hello",
            measure_ttft=True,
        )

        self.assertFalse(result["success"])
        self.assertIn("500", result["error"])
        self.assertEqual(result["eval_tok_per_sec"], 0.0)
        self.assertEqual(result["runtime"], "foundry")


class TestFactoryAndRunnerIntegration(unittest.TestCase):
    """Test suite for factory function, multi-runtime runner, and model resolver."""

    def test_create_runtime_client_foundry(self):
        """Verify create_runtime_client produces FoundryClient with config values."""
        conf = {
            "foundry": {
                "base_url": "http://127.0.0.1:8080/v1",
                "timeout_sec": 120,
                "auto_detect_port": False,
                "cli_path": "foundry",
            }
        }
        client = create_runtime_client("foundry", conf)
        self.assertIsInstance(client, FoundryClient)
        self.assertEqual(client.base_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(client.timeout_sec, 120)

    def test_create_runtime_client_ollama(self):
        """Verify create_runtime_client produces OllamaClient."""
        conf = {"ollama": {"base_url": "http://localhost:11434", "timeout_sec": 90}}
        client = create_runtime_client("ollama", conf)
        self.assertIsInstance(client, OllamaClient)
        self.assertEqual(client.base_url, "http://localhost:11434")

    def test_runner_scorecard_with_runtime_field(self):
        """Verify BenchmarkRunner attaches runtime and engine to scorecards."""
        mock_foundry = MagicMock(spec=BaseRuntimeClient)
        mock_foundry.name = "foundry"
        mock_foundry.display_name = "MS Foundry"
        mock_foundry.engine_name = "ONNX Runtime GenAI"

        runner = BenchmarkRunner(
            client=mock_foundry,
            config={"benchmark": {"composite_weights": {"coding": 0.4, "reasoning": 0.3, "performance": 0.3}}},
        )
        runner.specs = {"memory_type": "VRAM", "gpu_type": "nvidia", "gpu_vram_total_mb": "12288"}

        raw_results = [
            {
                "model": "phi-4",
                "runtime": "foundry",
                "engine": "ONNX Runtime GenAI",
                "suite": "coding",
                "eval_count": 200,
                "prompt_eval_count": 100,
                "eval_tok_per_sec": 65.0,
                "prompt_tok_per_sec": 350.0,
                "ttft_sec": 0.04,
                "total_tests": 4,
                "passed_tests": 4,
                "pass_ratio": 1.0,
                "hardware": {"vram_peak_mb": 4200, "vram_warning": False},
            }
        ]

        scorecard = runner.compute_model_scorecard("phi-4", raw_results, runtime="foundry")
        self.assertIsNotNone(scorecard)
        self.assertEqual(scorecard["runtime"], "foundry")
        self.assertEqual(scorecard["engine"], "ONNX Runtime GenAI")
        self.assertEqual(scorecard["coding_pass_rate"], 100.0)
        self.assertEqual(scorecard["avg_eval_tok_sec"], 65.0)

    def test_resolve_target_models_prefixes(self):
        """Verify resolve_target_models handles explicit runtime prefixes."""
        clients = {
            "ollama": MagicMock(spec=BaseRuntimeClient),
            "foundry": MagicMock(spec=BaseRuntimeClient),
        }
        raw_arg = "ollama:qwen2.5-coder:7b,foundry:phi-4,foundry:mistral-7b"
        resolved = resolve_target_models(raw_arg, selected_runtime="all", clients=clients)

        self.assertEqual(
            resolved,
            [
                ("ollama", "qwen2.5-coder:7b"),
                ("foundry", "phi-4"),
                ("foundry", "mistral-7b"),
            ],
        )

    def test_resolve_target_models_single_runtime(self):
        """Verify flat model names inherit selected_runtime."""
        clients = {
            "foundry": MagicMock(spec=BaseRuntimeClient),
        }
        raw_arg = "phi-4,phi-4-mini"
        resolved = resolve_target_models(raw_arg, selected_runtime="foundry", clients=clients)
        self.assertEqual(resolved, [("foundry", "phi-4"), ("foundry", "phi-4-mini")])


class TestCrossRuntimeReporting(unittest.TestCase):
    """Test suite verifying cross-runtime terminal UI and Markdown comparison reports."""

    def test_display_leaderboard_cross_runtime(self):
        """Verify display_leaderboard renders both Ollama and MS Foundry badges."""
        scorecards = [
            {
                "model": "qwen2.5-coder:7b",
                "runtime": "ollama",
                "engine": "llama.cpp",
                "composite_score": 93.0,
                "coding_pass_rate": 100.0,
                "reasoning_accuracy": 90.0,
                "avg_eval_tok_sec": 85.0,
                "avg_prompt_tok_sec": 320.0,
                "avg_ttft_sec": 0.05,
                "peak_vram_mb": 4500.0,
                "total_vram_mb": 12288.0,
                "gpu_type": "nvidia",
                "vram_warning": False,
            },
            {
                "model": "phi-4",
                "runtime": "foundry",
                "engine": "ONNX Runtime GenAI",
                "composite_score": 91.5,
                "coding_pass_rate": 100.0,
                "reasoning_accuracy": 85.0,
                "avg_eval_tok_sec": 78.0,
                "avg_prompt_tok_sec": 290.0,
                "avg_ttft_sec": 0.06,
                "peak_vram_mb": 4200.0,
                "total_vram_mb": 12288.0,
                "gpu_type": "nvidia",
                "vram_warning": False,
            },
        ]
        specs = {"platform_short": "Linux (NVIDIA RTX CUDA)", "gpu_type": "nvidia", "gpu_vram_total_mb": "12288"}
        # Must execute cleanly without exception
        display_leaderboard(scorecards, specs=specs)

    def test_display_scenario_result_with_runtime(self):
        """Verify display_scenario_result formats runtime tag."""
        res_foundry = {
            "suite": "coding",
            "model": "phi-4",
            "runtime": "foundry",
            "name": "Fibonacci Test",
            "passed": True,
            "passed_tests": 4,
            "total_tests": 4,
            "eval_tok_per_sec": 75.0,
            "hardware": {"vram_peak_mb": 4100},
        }
        display_scenario_result(res_foundry)

    def test_markdown_cross_runtime_comparison_table(self):
        """Verify Markdown report outputs the cross-engine comparison table when multiple runtimes exist."""
        scorecards = [
            {
                "model": "qwen2.5-coder:7b",
                "runtime": "ollama",
                "engine": "llama.cpp",
                "composite_score": 93.0,
                "coding_pass_rate": 100.0,
                "reasoning_accuracy": 90.0,
                "avg_eval_tok_sec": 85.0,
                "avg_prompt_tok_sec": 320.0,
                "avg_ttft_sec": 0.05,
                "peak_vram_mb": 4500.0,
                "total_vram_mb": 12288.0,
                "total_prompt_tokens": 1500,
                "total_eval_tokens": 800,
                "total_tokens_saved": 2300,
                "est_cost_saved_usd": 0.0165,
                "vram_warning": False,
            },
            {
                "model": "phi-4",
                "runtime": "foundry",
                "engine": "ONNX Runtime GenAI",
                "composite_score": 91.5,
                "coding_pass_rate": 100.0,
                "reasoning_accuracy": 85.0,
                "avg_eval_tok_sec": 78.0,
                "avg_prompt_tok_sec": 290.0,
                "avg_ttft_sec": 0.06,
                "peak_vram_mb": 4200.0,
                "total_vram_mb": 12288.0,
                "total_prompt_tokens": 1400,
                "total_eval_tokens": 750,
                "total_tokens_saved": 2150,
                "est_cost_saved_usd": 0.0154,
                "vram_warning": False,
            },
        ]
        specs = {
            "platform": "Linux 6.8 (x86_64)",
            "platform_short": "Linux (NVIDIA CUDA)",
            "cpu_model": "AMD Ryzen 9",
            "cpu_cores": "16",
            "ram_total_gb": "32.0",
            "gpu_name": "NVIDIA GeForce RTX",
            "gpu_type": "nvidia",
            "gpu_vram_total_mb": "12288",
            "memory_type": "VRAM",
            "driver_version": "CUDA 12.4",
        }

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            temp_path = f.name

        try:
            content = generate_markdown_report(scorecards, [], specs, output_path=temp_path)
            # Verify columns
            self.assertIn("| Runtime | Engine |", content)
            self.assertIn("`MS Foundry`", content)
            self.assertIn("`Ollama`", content)
            self.assertIn("ONNX Runtime GenAI", content)
            self.assertIn("llama.cpp", content)
            # Verify cross-engine comparison section
            self.assertIn("Engine Architecture Comparison", content)
            self.assertIn("Average Decode Speed", content)
            self.assertIn("Average Time to First Token", content)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_1to1_comparison_report_generation(self):
        """Verify generate_1to1_comparison_report produces head-to-head analysis."""
        from reporting.display import display_1to1_comparison
        from reporting.markdown import generate_1to1_comparison_report

        sc_a = {
            "model": "phi3:mini",
            "runtime": "ollama",
            "engine": "llama.cpp",
            "composite_score": 31.1,
            "coding_pass_rate": 17.6,
            "avg_eval_tok_sec": 94.0,
            "avg_prompt_tok_sec": 2639.8,
            "avg_ttft_sec": 0.21,
            "peak_vram_mb": 11135.0,
            "total_tokens_saved": 1659,
            "est_cost_saved_usd": 0.0199,
        }
        sc_b = {
            "model": "Phi-3.5-mini-instruct-generic-cpu",
            "runtime": "foundry",
            "engine": "ONNX Runtime GenAI",
            "composite_score": 16.4,
            "coding_pass_rate": 29.4,
            "avg_eval_tok_sec": 9.2,
            "avg_prompt_tok_sec": 47.8,
            "avg_ttft_sec": 2.08,
            "peak_vram_mb": 7136.0,
            "total_tokens_saved": 2100,
            "est_cost_saved_usd": 0.0268,
        }
        res_a = [
            {
                "test_id": "code_lru",
                "name": "LRU Cache",
                "passed": True,
                "passed_tests": 2,
                "total_tests": 2,
                "eval_tok_per_sec": 65.9,
            }
        ]
        res_b = [
            {
                "test_id": "code_lru",
                "name": "LRU Cache",
                "passed": False,
                "passed_tests": 0,
                "total_tests": 2,
                "eval_tok_per_sec": 8.7,
                "sandbox_error": "SyntaxError",
            }
        ]
        specs = {
            "platform_short": "WSL2 (NVIDIA)",
            "gpu_name": "NVIDIA GeForce RTX 5070",
            "gpu_vram_total_mb": "12227",
            "cpu_model": "AMD Ryzen 7",
            "cpu_cores": "16",
            "ram_total_gb": "32.0",
            "driver_version": "616.92",
        }

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            temp_path = f.name

        try:
            content = generate_1to1_comparison_report(
                sc_a, sc_b, res_a, res_b, specs, pair_name="Phi-3.5 Mini vs Phi-3 Mini", output_path=temp_path
            )
            self.assertIn("1:1 Model Comparison", content)
            self.assertIn("phi3:mini", content)
            self.assertIn("Phi-3.5-mini-instruct-generic-cpu", content)
            self.assertIn("`Ollama` is **10.2x faster**", content)
            self.assertIn("LRU Cache", content)

            # Also ensure terminal display runs without error
            display_1to1_comparison(sc_a, sc_b, res_a, res_b)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_client_options_alignment_num_ctx_and_seed(self):
        """Verify OllamaClient injects default_num_ctx and FoundryClient forwards seed."""
        ollama = OllamaClient(default_num_ctx=4096)
        self.assertEqual(ollama.default_num_ctx, 4096)

        foundry = FoundryClient(auto_detect_port=False, default_max_tokens=4096)
        self.assertEqual(foundry.default_max_tokens, 4096)


if __name__ == "__main__":
    unittest.main()
