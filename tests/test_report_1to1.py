"""Tests for the 1:1 comparison: choosing the pair, filtering records, scenario status and runtime labels."""

import os
import tempfile
import unittest
from unittest.mock import patch

from rich.console import Console

from benchrig import cli
from benchrig.reporting.common import scenario_outcome, status_markdown, status_rich
from benchrig.reporting.display import display_1to1_comparison


def card(model, runtime="ollama", engine="llama.cpp", **extra):
    return {
        "model": model,
        "runtime": runtime,
        "engine": engine,
        "composite_score": 90.0,
        "coding_pass_rate": 100.0,
        "reasoning_accuracy": 66.7,
        "avg_eval_tok_sec": 100.0,
        "avg_prefill_eff_tok_sec": 1000.0,
        "avg_ttft_sec": 0.1,
        "peak_vram_mb": 5000.0,
        **extra,
    }


def rec(model, runtime, suite, test_id, **extra):
    return {
        "model": model,
        "runtime": runtime,
        "suite": suite,
        "test_id": test_id,
        "name": test_id,
        "eval_tok_per_sec": 100.0,
        **extra,
    }


class SelectPairTests(unittest.TestCase):
    def test_counterpart_is_found_by_name_not_by_position(self):
        cards = [
            card("Phi-4-mini-instruct-cuda-gpu", "prism", "ONNX Runtime GenAI"),
            card("mistral:7b"),
            card("phi3:mini"),
            card("phi4-mini:latest"),
        ]
        ollama, other = cli.select_pair(cards)
        self.assertEqual((ollama["model"], other["model"]), ("phi4-mini:latest", "Phi-4-mini-instruct-cuda-gpu"))

    def test_size_tags_are_not_confused(self):
        cards = [
            card("qwen2.5-coder:3b"),
            card("qwen2.5-coder:14b"),
            card("qwen2.5-coder:7b"),
            card("qwen2.5-coder-7b-instruct-generic-cpu-4:v4", "foundry", "ONNX Runtime GenAI"),
        ]
        self.assertEqual(cli.select_pair(cards)[0]["model"], "qwen2.5-coder:7b")

    def test_ollama_proxy_prefix_is_ignored(self):
        cards = [
            card("llama3.1:8b"),
            card("qwen2.5-coder:3b"),
            card("ollama:qwen2.5-coder:3b", "prism", "Ollama (llama.cpp)"),
        ]
        self.assertEqual(cli.select_pair(cards)[0]["model"], "qwen2.5-coder:3b")

    def test_without_a_match_the_first_ollama_scorecard_is_used(self):
        cards = [card("mistral:7b"), card("gemma3:12b"), card("totally-different", "prism", "ONNX Runtime GenAI")]
        self.assertEqual(cli.select_pair(cards)[0]["model"], "mistral:7b")


class RecordsForTests(unittest.TestCase):
    def test_only_the_models_own_records_on_its_runtime(self):
        results = [
            rec("phi4-mini:latest", "ollama", "coding", "c1"),
            rec("phi3:mini", "ollama", "coding", "c1"),
            rec("phi4-mini:latest", "prism", "coding", "c1"),
            {"model": "phi4-mini:latest", "suite": "coding", "test_id": "old"},  # record from before `runtime` existed
        ]
        picked = cli.records_for(results, card("phi4-mini:latest"))
        self.assertEqual([r["test_id"] for r in picked], ["c1", "old"])
        self.assertTrue(all(r["model"] == "phi4-mini:latest" for r in picked))
        self.assertEqual(len(cli.records_for(results, card("phi4-mini:latest", "prism"))), 1)


class ScenarioOutcomeTests(unittest.TestCase):
    def test_each_suite_is_read_from_its_own_fields(self):
        coding = rec("m", "ollama", "coding", "c", passed=False, passed_tests=1, total_tests=5, success=False)
        reasoning = rec("m", "ollama", "reasoning", "r", correct=True, success=True)
        speed = rec("m", "ollama", "speed", "s", success=True)
        self.assertEqual(status_markdown(coding), "❌ FAIL (1/5)")
        self.assertEqual(status_markdown(reasoning), "✅ PASS")
        self.assertEqual(status_markdown(speed), "✅ OK")
        self.assertEqual(status_markdown(rec("m", "ollama", "speed", "s", success=False)), "❌ ERROR")

    def test_no_record_is_not_a_failure(self):
        self.assertIsNone(scenario_outcome({}).ok)
        self.assertEqual(status_markdown({}), "—")
        self.assertIn("n/a", status_rich({}))

    def test_speed_scenarios_are_never_reported_as_zero_of_zero(self):
        for text in (
            status_markdown(rec("m", "o", "speed", "s", success=True)),
            status_rich(rec("m", "o", "context", "c", success=True)),
        ):
            self.assertNotIn("0/0", text)

    def test_truncated_answers_are_marked(self):
        coding = rec("m", "ollama", "coding", "c", passed=False, passed_tests=0, total_tests=4, truncated=True)
        self.assertIn("cut off", status_markdown(coding))
        self.assertIn("cut off", status_rich(coding))


class PrismReportTests(unittest.TestCase):
    SPECS = {"platform": "Linux", "gpu_name": "GPU", "gpu_vram_total_mb": "12000", "memory_type": "VRAM"}

    def setUp(self):
        self.cards = [
            card("Phi-4-mini-instruct-cuda-gpu", "prism", "ONNX Runtime GenAI", vram_model_mb=10424.0),
            card("phi3:mini", vram_model_mb=5421.0),
            card("phi4-mini:latest", vram_model_mb=3732.0),
        ]
        self.results = [
            rec(
                "phi3:mini",
                "ollama",
                "coding",
                "flatten",
                passed=False,
                passed_tests=0,
                total_tests=4,
                sandbox_error="NameError: newcur_key",
            ),
            rec("phi4-mini:latest", "ollama", "coding", "flatten", passed=True, passed_tests=4, total_tests=4),
            rec("phi4-mini:latest", "ollama", "speed", "raw_speed", success=True),
            rec(
                "Phi-4-mini-instruct-cuda-gpu", "prism", "coding", "flatten", passed=True, passed_tests=4, total_tests=4
            ),
            rec("Phi-4-mini-instruct-cuda-gpu", "prism", "speed", "raw_speed", success=True),
        ]

    def test_markdown_report_is_about_the_right_models_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmp, patch("benchrig.cli.console", Console(quiet=True)):
            cli.show_1to1_comparison(self.cards, self.results, self.SPECS, tmp)
            with open(os.path.join(tmp, "1TO1_COMPARISON_REPORT.md"), encoding="utf-8") as f:
                content = f.read()
        self.assertIn("phi4-mini:latest (Ollama) vs Phi-4-mini-instruct-cuda-gpu (Prism)", content)
        self.assertNotIn("MS Foundry", content)
        self.assertNotIn("newcur_key", content)  # phi3:mini's error must not appear under phi4-mini
        self.assertNotIn("0/0", content)
        self.assertIn("✅ OK", content)  # the speed scenario

    def test_terminal_report_names_both_runtimes(self):
        console = Console(record=True, width=200)
        with patch("benchrig.reporting.display.console", console):
            display_1to1_comparison(self.cards[2], self.cards[0], self.results[1:3], self.results[3:], pair_name="pair")
        text = console.export_text()
        self.assertIn("Ollama", text)
        self.assertIn("Prism", text)
        self.assertNotIn("Foundry", text)
        self.assertNotIn("0/0", text)


if __name__ == "__main__":
    unittest.main()
