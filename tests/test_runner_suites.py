"""Unit tests for BenchmarkRunner suite result records (offline, with a fake runtime client)."""

import unittest
from unittest.mock import patch

from benchrig.core.client import BaseRuntimeClient
from benchrig.core.runner import BenchmarkRunner

COMMON_KEYS = {
    "suite",
    "test_id",
    "name",
    "model",
    "runtime",
    "engine",
    "success",
    "eval_tok_per_sec",
    "prompt_tok_per_sec",
    "ttft_sec",
    "prompt_eval_count",
    "hardware",
}


class FakeClient(BaseRuntimeClient):
    name = "fake"
    display_name = "Fake"
    engine_name = "fake-engine"

    def __init__(self, response: str = "42"):
        super().__init__(base_url="http://fake")
        self.response = response
        self.prompts = []

    def generate(self, model, prompt, system=None, options=None, measure_ttft=True):
        self.prompts.append((prompt, options))
        return {
            "success": True,
            "response": self.response,
            "eval_count": 10,
            "eval_tok_per_sec": 50.0,
            "prompt_eval_count": 5,
            "prompt_tok_per_sec": 100.0,
            "ttft_sec": 0.1,
            "total_time_sec": 1.0,
        }


class RunnerSuiteTests(unittest.TestCase):
    def setUp(self):
        patches = [
            patch("benchrig.core.runner.time.sleep"),
            patch("benchrig.core.runner.get_system_specs", return_value={}),
            patch("benchrig.core.runner.HardwareSampler"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.client = FakeClient()
        self.runner = BenchmarkRunner(client=self.client, config={})

    def test_speed_suite_record(self):
        (rec,) = self.runner.run_speed_suite("m", [{"id": "s1", "name": "Speed", "prompt": "hi"}])
        self.assertTrue(COMMON_KEYS <= rec.keys())
        self.assertEqual((rec["suite"], rec["runtime"], rec["engine"]), ("speed", "fake", "fake-engine"))
        self.assertEqual(rec["eval_count"], 10)
        self.assertEqual(rec["total_time_sec"], 1.0)

    def test_record_uses_engine_and_device_reported_with_the_response(self):
        """A multi-engine server (Prism) reports per model which engine and device served it; reports must show that."""
        original = self.client.generate

        def generate(*args, **kwargs):
            return {**original(*args, **kwargs), "engine": "Ollama (llama.cpp)", "device": "cuda"}

        self.client.generate = generate
        (rec,) = self.runner.run_speed_suite("m", [{"id": "s1", "name": "Speed", "prompt": "hi"}])
        self.assertEqual((rec["engine"], rec["device"]), ("Ollama (llama.cpp)", "cuda"))

    def test_record_omits_device_when_not_reported(self):
        (rec,) = self.runner.run_speed_suite("m", [{"id": "s1", "name": "Speed", "prompt": "hi"}])
        self.assertNotIn("device", rec)

    def test_coding_suite_scores_sandbox_result(self):
        self.client.response = "```python\ndef add(a, b):\n    return a + b\n```"
        scenario = {
            "id": "c1",
            "name": "Add",
            "prompt": "write add",
            "test_assertions": ["assert add(1, 2) == 3", "assert add(2, 2) == 5"],
        }
        (rec,) = self.runner.run_coding_suite("m", [scenario])
        self.assertTrue(COMMON_KEYS <= rec.keys())
        self.assertEqual((rec["passed_tests"], rec["total_tests"]), (1, 2))
        self.assertFalse(rec["success"])

    def test_reasoning_suite_marks_correct_answer(self):
        scenario = {"id": "r1", "name": "Math", "prompt": "6*7?", "expected_answer": "42", "check_type": "numeric"}
        (rec,) = self.runner.run_reasoning_suite("m", [scenario])
        self.assertTrue(rec["correct"] and rec["success"])
        self.assertIn("has_think_tags", rec)

    def test_polish_suite_omits_think_fields(self):
        scenario = {"id": "p1", "name": "PL", "prompt": "?", "expected_answer": "42"}
        (rec,) = self.runner.run_polish_suite("m", [scenario])
        self.assertTrue(rec["correct"])
        self.assertNotIn("has_think_tags", rec)

    def test_context_suite_record_and_options(self):
        scenario = {"name": "512", "context_size": 512, "prompt_multiplier": 2, "instruction": "Summarize."}
        (rec,) = self.runner.run_context_suite("m", [scenario])
        self.assertEqual(rec["test_id"], "context_512")
        self.assertEqual(rec["context_size"], 512)
        # Context records intentionally do not contribute generated-token counts.
        self.assertNotIn("eval_count", rec)
        prompt, options = self.client.prompts[-1]
        self.assertTrue(prompt.endswith("Question: Summarize."))
        self.assertEqual(options["num_ctx"], 512)

    def test_scorecard_percentages_and_empty_input(self):
        self.assertEqual(self.runner.compute_model_scorecard("missing", []), {})
        results = [
            {
                "model": "m",
                "runtime": "fake",
                "suite": "coding",
                "passed_tests": 3,
                "total_tests": 4,
                "eval_tok_per_sec": 60.0,
                "prompt_tok_per_sec": 1.0,
                "ttft_sec": 0.5,
            },
        ]
        sc = self.runner.compute_model_scorecard("m", results)
        self.assertEqual(sc["coding_pass_rate"], 75.0)
        self.assertEqual(sc["reasoning_accuracy"], 0.0)
        # 0.4 * 75 + 0.3 * 0 + 0.3 * 100
        self.assertEqual(sc["composite_score"], 60.0)


if __name__ == "__main__":
    unittest.main()
