"""The context suite hides a fact in the filler and asks for it, so it measures whether the model uses the context."""

import json
import os
import tempfile
import unittest
from importlib.resources import files
from unittest.mock import patch

from rich.console import Console

from benchrig.core.runner import CONTEXT_FILLER_PARAGRAPH, build_context_prompt
from benchrig.reporting.display import display_scenario_result
from benchrig.reporting.markdown import generate_markdown_report
from tests.test_measurement_methodology import FakeClient, RunnerTestCase

SCENARIOS = json.loads(files("benchrig").joinpath("data/scenarios/context_scaling.json").read_text(encoding="utf-8"))
NEEDLE = "The access code for vault 7 is 4921."


class PromptTests(unittest.TestCase):
    def test_the_needle_is_in_the_prompt_once_at_a_sentence_boundary(self):
        prompt = build_context_prompt(2048, "What is the code?", needle=NEEDLE, depth=0.5)
        self.assertEqual(prompt.count(NEEDLE), 1)
        before = prompt.split(NEEDLE)[0]
        self.assertTrue(before.endswith(". "), "the fact must start a sentence, not cut one in half")
        self.assertTrue(prompt.endswith("Question: What is the code?"))

    def test_depth_moves_the_needle(self):
        early = build_context_prompt(2048, "Q", needle=NEEDLE, depth=0.1).index(NEEDLE)
        middle = build_context_prompt(2048, "Q", needle=NEEDLE, depth=0.5).index(NEEDLE)
        late = build_context_prompt(2048, "Q", needle=NEEDLE, depth=0.9).index(NEEDLE)
        self.assertLess(early, middle)
        self.assertLess(middle, late)

    def test_without_a_needle_the_prompt_is_just_filler(self):
        prompt = build_context_prompt(512, "Q")
        self.assertNotIn("access code", prompt)
        self.assertTrue(prompt.startswith(CONTEXT_FILLER_PARAGRAPH[:40]))

    def test_the_size_still_follows_the_window(self):
        self.assertGreater(
            len(build_context_prompt(8192, "Q", needle=NEEDLE)), len(build_context_prompt(512, "Q", needle=NEEDLE)) * 10
        )


class ScenarioFileTests(unittest.TestCase):
    def test_every_step_hides_its_own_fact(self):
        self.assertEqual([s["context_size"] for s in SCENARIOS], [512, 1024, 2048, 4096, 8192])
        for sc in SCENARIOS:
            self.assertIn(sc["expected"], sc["needle"], sc["name"])
            self.assertNotIn(sc["expected"], sc["instruction"], "the question must not give the answer away")
        self.assertEqual(len({s["expected"] for s in SCENARIOS}), len(SCENARIOS), "reused codes could be memorised")


class SuiteTests(RunnerTestCase):
    STEP = [
        {"context_size": 512, "name": "c", "instruction": "What is the code?", "needle": NEEDLE, "expected": "4921"}
    ]

    def test_finding_the_fact_is_a_success(self):
        (rec,) = self.runner(FakeClient(response="4921")).run_context_suite("m", self.STEP)
        self.assertTrue(rec["retrieved"])
        self.assertTrue(rec["success"])

    def test_a_wrong_or_lookalike_answer_is_a_miss(self):
        for answer in ("1234", "49210", "I do not know", ""):
            (rec,) = self.runner(FakeClient(response=answer)).run_context_suite("m", self.STEP)
            self.assertFalse(rec["retrieved"], answer)
            self.assertFalse(rec["success"], answer)

    def test_the_answer_may_be_wrapped_in_words(self):
        (rec,) = self.runner(FakeClient(response="The access code is 4921.")).run_context_suite("m", self.STEP)
        self.assertTrue(rec["retrieved"])

    def test_a_failed_request_is_a_miss_and_keeps_its_error(self):
        client = FakeClient(success=False, error="model runner has unexpectedly stopped", response="")
        (rec,) = self.runner(client).run_context_suite("m", self.STEP)
        self.assertFalse(rec["retrieved"])
        self.assertIn("unexpectedly stopped", rec["error"])

    def test_a_step_without_a_fact_only_measures_speed(self):
        step = [{"context_size": 512, "name": "c", "instruction": "Summarize."}]
        (rec,) = self.runner(FakeClient()).run_context_suite("m", step)
        self.assertNotIn("retrieved", rec)
        self.assertTrue(rec["success"])

    def test_the_scorecard_reports_the_share_of_facts_found(self):
        runner = self.runner(FakeClient(response="4921"))
        found = runner.run_context_suite("m", self.STEP * 3)
        runner.client.response["response"] = "nope"
        missed = runner.run_context_suite("m", self.STEP)
        self.assertEqual(runner.compute_model_scorecard("m", found + missed)["context_retrieval_pct"], 75.0)

    def test_no_facts_asked_means_no_percentage(self):
        step = [{"context_size": 512, "name": "c", "instruction": "Summarize."}]
        runner = self.runner(FakeClient())
        self.assertIsNone(
            runner.compute_model_scorecard("m", runner.run_context_suite("m", step))["context_retrieval_pct"]
        )


class ReportTests(unittest.TestCase):
    BASE = {
        "suite": "context",
        "model": "m",
        "runtime": "ollama",
        "name": "Context 2k",
        "context_size": 2048,
        "eval_tok_per_sec": 100.0,
        "hardware": {},
    }

    def show(self, record):
        console = Console(record=True, width=200)
        with patch("benchrig.reporting.display.console", console):
            display_scenario_result(record)
        return console.export_text()

    def test_a_missed_fact_is_not_shown_as_a_request_error(self):
        text = self.show({**self.BASE, "success": False, "retrieved": False})
        self.assertIn("missed the fact", text)
        self.assertNotIn("ERROR", text)

    def test_a_found_fact(self):
        self.assertIn("found the fact", self.show({**self.BASE, "success": True, "retrieved": True}))

    def test_a_failed_request_is_an_error(self):
        text = self.show({**self.BASE, "success": False, "retrieved": False, "error": "connection refused"})
        self.assertIn("ERROR", text)
        self.assertIn("connection refused", text)

    def test_a_wrong_reasoning_answer_is_a_fail_not_an_error(self):
        text = self.show(
            {"suite": "reasoning", "model": "m", "name": "Boxes", "success": False, "correct": False, "hardware": {}}
        )
        self.assertIn("FAIL", text)
        self.assertNotIn("ERROR", text)

    def test_an_older_record_with_no_verdict_is_still_an_error(self):
        self.assertIn(
            "ERROR", self.show({"suite": "speed", "model": "m", "name": "S", "success": False, "hardware": {}})
        )

    def test_markdown_context_table_lists_whether_the_fact_was_found(self):
        card = {
            "model": "m",
            "runtime": "ollama",
            "engine": "llama.cpp",
            "composite_score": 1.0,
            "coding_pass_rate": 0.0,
            "reasoning_accuracy": 0.0,
            "avg_eval_tok_sec": 1.0,
            "avg_ttft_sec": 0.1,
            "peak_vram_mb": 1.0,
            "total_prompt_tokens": 0,
            "total_eval_tokens": 0,
            "total_tokens_saved": 0,
            "est_cost_saved_usd": 0.0,
        }
        results = [
            {**self.BASE, "success": True, "retrieved": True},
            {**self.BASE, "success": False, "retrieved": False},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            content = generate_markdown_report([card], results, {}, output_path=os.path.join(tmp, "r.md"))
        self.assertIn("Found the fact", content)
        self.assertIn("✅", content)
        self.assertIn("❌", content)


if __name__ == "__main__":
    unittest.main()
