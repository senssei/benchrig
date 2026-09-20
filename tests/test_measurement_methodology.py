"""Tests for measurement methodology: context warm-up, prefill definition, VRAM baseline, truncation and reasoning checks."""

import json
import unittest
from importlib.resources import files
from unittest.mock import MagicMock, patch

from benchrig.core.client import BaseRuntimeClient, OllamaClient
from benchrig.core.reasoning_parser import evaluate_reasoning_answer, extract_thinking_and_answer
from benchrig.core.runner import (
    CONTEXT_CHARS_PER_TOKEN,
    CONTEXT_WARMUP_PROMPT,
    DEFAULT_THINKING_TOKEN_MULTIPLIER,
    BenchmarkRunner,
    build_context_prompt,
)
from benchrig.core.sandbox import extract_python_code, has_complete_code_block

SCENARIOS = json.loads(files("benchrig").joinpath("data/scenarios/reasoning.json").read_text(encoding="utf-8"))
BOXES = next(sc for sc in SCENARIOS if sc["id"] == "reasoning_three_boxes")  # the real scenario, not a copy


def evaluate(text, scenario=BOXES):
    return evaluate_reasoning_answer(
        text, scenario["expected_answer"], scenario["check_type"], scenario["accepted_patterns"]
    )


class ReasoningEvaluatorTests(unittest.TestCase):
    def test_answer_format_does_not_matter(self):
        one_line = "Box 1: Oranges, Box 2: Apples and Oranges, Box 3: Apples"
        lines = "Box 1: Oranges\nBox 2: Apples and Oranges\nBox 3: Apples"
        bullets = "Conclusion:\n- **Box 1**: Oranges\n- **Box 2**: Apples and Oranges\n- **Box 3**: Apples"
        for text in (one_line, lines, bullets):
            self.assertTrue(evaluate(text)["correct"], text)

    def test_wrong_answers_are_wrong(self):
        wrong = [
            "Box 1: Apples\nBox 2: Oranges\nBox 3: Apples and Oranges",
            "Final answer: Box 1: Oranges, Box 2: Apples, Box 3: Oranges and Apples.",  # a real mistral:7b answer
            "**Box 1: Apples, Box 2: Oranges, Box 3: Oranges**",  # a real qwen2.5-coder:7b answer
            "**Final Answer:**\nBox 1: Oranges, Box 2: Apples, Box 3: Apples",  # a real deepseek-r1:14b answer
        ]
        for text in wrong:
            self.assertFalse(evaluate(text)["correct"], text)

    def test_correct_words_scattered_through_the_reasoning_do_not_count(self):
        """Words from the right answer appear in a long derivation, but the stated conclusion is wrong."""
        reasoning = (
            "Box 1 is labeled Apples so it holds oranges or both. Box 2 might be apples and oranges. Box 3 is labeled "
            "Apples and Oranges and we drew an apple so box 3 has apples. "
            + "Let me double-check the logic carefully. "
            * 12
        )
        self.assertFalse(evaluate(reasoning + "Final: Box 1: Apples, Box 2: Oranges, Box 3: Both.")["correct"])

    def test_accepts_a_conclusion_written_with_extra_words(self):
        text = "So: Box 1 contains Oranges. Box 2 contains Oranges and Apples. Box 3 contains Apples."
        self.assertTrue(evaluate(text)["correct"])

    def test_latex_answers_match_plain_expected_values(self):
        venn = next(sc for sc in SCENARIOS if sc["id"] == "reasoning_venn_probability")
        for text in (
            "**Final Answer:**\n\\[\n\\boxed{\\dfrac{1}{6}}\n\\]",
            "The probability is \\frac{5}{30} = \\frac{1}{6}",
            "1/6",
        ):
            self.assertTrue(evaluate(text, venn)["correct"], text)
        self.assertFalse(evaluate("The probability is \\frac{1}{5}", venn)["correct"])

    def test_unterminated_think_is_not_an_answer(self):
        """The trace mentions the right words, but the model never answered (token budget ran out while thinking)."""
        text = "<think>Maybe box 1: oranges, box 2: apples and oranges, box 3: apples. But wait, let me re-check"
        result = evaluate(text)
        self.assertFalse(result["correct"])
        self.assertTrue(result["truncated_thinking"])

    def test_answer_after_closed_think_counts(self):
        text = "<think>hmm</think>Box 1: Oranges\nBox 2: Apples and Oranges\nBox 3: Apples"
        result = evaluate(text)
        self.assertTrue(result["correct"])
        self.assertFalse(result["truncated_thinking"])

    def test_correct_words_only_inside_think_do_not_count(self):
        text = "<think>Box 1: Oranges, Box 2: Apples and Oranges, Box 3: Apples</think>I am not sure."
        self.assertFalse(evaluate(text)["correct"])

    def test_bare_closing_tag(self):
        parsed = extract_thinking_and_answer("reasoning here</think>final answer")
        self.assertEqual(
            (parsed["thinking"], parsed["answer"], parsed["truncated_thinking"]),
            ("reasoning here", "final answer", False),
        )

    def test_answer_excerpt_keeps_the_end_of_a_long_answer(self):
        result = evaluate("x" * 1000 + " Box 1: Oranges, Box 2: Apples and Oranges, Box 3: Apples")
        self.assertTrue(result["answer_excerpt"].endswith("Box 3: Apples"))
        self.assertLessEqual(len(result["answer_excerpt"]), 403)


class CodeExtractionTests(unittest.TestCase):
    def test_code_inside_think_is_ignored(self):
        text = "<think>```python\ndef wrong(): pass\n```</think>```python\ndef right(): return 1\n```"
        self.assertEqual(extract_python_code(text), "def right(): return 1")

    def test_unterminated_think_yields_no_code(self):
        self.assertEqual(extract_python_code("<think>```python\ndef half(): pass\n```"), "")

    def test_complete_code_block_detection(self):
        self.assertTrue(has_complete_code_block("```python\ndef f(): pass\n```"))
        self.assertFalse(has_complete_code_block("```python\ndef f(): pa"))
        self.assertFalse(has_complete_code_block("<think>```python\ndef f(): pass\n```</think>"))


class FakeClient(BaseRuntimeClient):
    name = "fake"
    display_name = "Fake"
    engine_name = "fake-engine"

    def __init__(self, reloads=False, source="client_ttft", **response):
        super().__init__(base_url="http://fake")
        self.reloads_on_context_change = reloads
        self.prefill_source = source
        self.calls = []
        self.response = {
            "success": True,
            "response": "ok",
            "eval_count": 10,
            "eval_tok_per_sec": 50.0,
            "prompt_eval_count": 200,
            "prompt_tok_per_sec": 9999.0,
            "ttft_sec": 0.1,
            "total_time_sec": 1.0,
            **response,
        }

    def generate(self, model, prompt, system=None, options=None, measure_ttft=True):
        self.calls.append({"prompt": prompt, "options": options or {}, "measure_ttft": measure_ttft})
        return dict(self.response)


def make_runner(client, config=None):
    patches = [
        patch("benchrig.core.runner.time.sleep"),
        patch("benchrig.core.runner.get_system_specs", return_value={}),
        patch("benchrig.core.runner.HardwareSampler"),
    ]
    started = [p.start() for p in patches]
    started[2].return_value.stop.side_effect = lambda: {}  # a fresh, empty hardware summary per request
    runner = BenchmarkRunner(client=client, config=config or {})
    runner._patches = patches
    return runner


class RunnerTestCase(unittest.TestCase):
    def runner(self, client, config=None):
        runner = make_runner(client, config)
        self.addCleanup(lambda: [p.stop() for p in runner._patches])
        return runner


class ContextSuiteTests(RunnerTestCase):
    SCENARIOS = [
        {"context_size": 512, "name": "512", "instruction": "Summarize.", "fill_ratio": 0.75},
        {"context_size": 2048, "name": "2k", "instruction": "Summarize.", "fill_ratio": 0.75},
    ]

    def test_build_context_prompt_size_follows_the_window(self):
        sizes = [len(build_context_prompt(n, "Q?")) for n in (512, 1024, 2048, 8192)]
        self.assertEqual(sizes, sorted(sizes))
        expected_chars = 8192 * 0.75 * CONTEXT_CHARS_PER_TOKEN
        self.assertAlmostEqual(sizes[-1], expected_chars, delta=100)
        self.assertTrue(build_context_prompt(512, "Q?").endswith("Question: Q?"))

    def test_ollama_like_client_is_warmed_up_before_each_timed_request(self):
        client = FakeClient(reloads=True)
        self.runner(client).run_context_suite("m", self.SCENARIOS)
        self.assertEqual(len(client.calls), 5)  # a warm-up and a timed request per step, then one restoring the context
        for warm, timed, ctx in ((0, 1, 512), (2, 3, 2048)):
            self.assertEqual(client.calls[warm]["prompt"], CONTEXT_WARMUP_PROMPT)
            self.assertEqual(client.calls[warm]["options"]["num_ctx"], ctx)
            self.assertEqual(client.calls[warm]["options"]["num_predict"], 1)
            self.assertEqual(client.calls[timed]["options"]["num_ctx"], ctx)
            self.assertNotEqual(client.calls[timed]["prompt"], CONTEXT_WARMUP_PROMPT)

    def test_other_clients_are_not_warmed_up(self):
        client = FakeClient(reloads=False)
        self.runner(client).run_context_suite("m", self.SCENARIOS)
        self.assertEqual(len(client.calls), 2)

    def test_the_model_is_left_at_the_default_context_so_the_next_suite_does_not_reload(self):
        """The last step leaves the model at num_ctx 2048/8192; without this the next suite's first request paid the reload."""
        client = FakeClient(reloads=True)
        runner = self.runner(client, {"execution_alignment_1to1": {"context_tokens": 4096}})
        runner.run_context_suite("m", self.SCENARIOS)
        last = client.calls[-1]
        self.assertEqual(last["prompt"], CONTEXT_WARMUP_PROMPT)
        self.assertEqual(last["options"]["num_ctx"], 4096)
        self.assertFalse(last["measure_ttft"])

    def test_without_a_configured_default_the_restoring_request_uses_the_clients_own(self):
        client = FakeClient(reloads=True)
        self.runner(client).run_context_suite("m", self.SCENARIOS)
        self.assertNotIn("num_ctx", client.calls[-1]["options"])  # OllamaClient supplies its default_num_ctx

    def test_the_restoring_request_is_not_a_measured_result(self):
        client = FakeClient(reloads=True)
        records = self.runner(client).run_context_suite("m", self.SCENARIOS)
        self.assertEqual(len(records), len(self.SCENARIOS))

    def test_warmup_request_is_not_sampled(self):
        client = FakeClient(reloads=True)
        runner = self.runner(client)
        with patch.object(runner, "_generate_with_telemetry", wraps=runner._generate_with_telemetry) as timed:
            runner.run_context_suite("m", self.SCENARIOS)
        self.assertEqual(timed.call_count, 2)  # only the timed requests go through the hardware sampler

    def test_record_carries_actual_prompt_tokens_and_load_time(self):
        client = FakeClient(reloads=True, load_time_sec=0.0, prompt_eval_count=391)
        (rec, _) = self.runner(client).run_context_suite("m", self.SCENARIOS)
        self.assertEqual((rec["context_size"], rec["prompt_tokens_actual"], rec["load_time_sec"]), (512, 391, 0.0))

    def test_real_ollama_client_declares_the_reload_behaviour(self):
        self.assertTrue(OllamaClient.reloads_on_context_change)
        self.assertEqual(OllamaClient.prefill_source, "server")
        self.assertFalse(BaseRuntimeClient.reloads_on_context_change)


class PrefillTests(RunnerTestCase):
    def test_effective_prefill_uses_the_same_definition_for_every_client(self):
        server = FakeClient(source="server", prompt_eval_count=200, ttft_sec=0.1, prompt_tok_per_sec=4000.0)
        client_side = FakeClient(source="client_ttft", prompt_eval_count=200, ttft_sec=0.1, prompt_tok_per_sec=2000.0)
        sc = [{"id": "s", "name": "S", "prompt": "hi"}]
        (a,) = self.runner(server).run_speed_suite("m", sc)
        (b,) = self.runner(client_side).run_speed_suite("m", sc)
        self.assertEqual(a["prefill_eff_tok_per_sec"], 2000.0)
        self.assertEqual(b["prefill_eff_tok_per_sec"], 2000.0)
        self.assertEqual((a["prefill_source"], b["prefill_source"]), ("server", "client_ttft"))
        self.assertEqual((a["prompt_tok_per_sec"], b["prompt_tok_per_sec"]), (4000.0, 2000.0))  # engine value is kept

    def test_zero_ttft_gives_zero_not_a_division_error(self):
        (rec,) = self.runner(FakeClient(ttft_sec=0.0)).run_speed_suite("m", [{"id": "s", "name": "S", "prompt": "hi"}])
        self.assertEqual(rec["prefill_eff_tok_per_sec"], 0.0)

    def test_scorecard_averages_effective_prefill(self):
        runner = self.runner(FakeClient(prompt_eval_count=100, ttft_sec=0.1))
        recs = runner.run_speed_suite(
            "m", [{"id": "a", "name": "A", "prompt": "x"}, {"id": "b", "name": "B", "prompt": "y"}]
        )
        self.assertEqual(runner.compute_model_scorecard("m", recs)["avg_prefill_eff_tok_sec"], 1000.0)


class VramBaselineTests(RunnerTestCase):
    def provider(self, readings):
        provider = MagicMock()
        provider.read_gpu.side_effect = [(mb, 12000.0, 0.0, 0.0, 0.0) for mb in readings]
        return provider

    def test_baseline_waits_until_the_reading_settles(self):
        runner = self.runner(FakeClient())
        provider = self.provider([9000, 7000, 6100, 6120, 6119])  # a model is still unloading, then stable
        with patch.object(runner, "_create_sampler", return_value=MagicMock(provider=provider)):
            self.assertEqual(runner.measure_vram_baseline(), 6120)
        self.assertEqual(runner.vram_baseline_mb, 6120)

    def test_model_memory_is_peak_minus_baseline_and_never_negative(self):
        runner = self.runner(FakeClient())
        runner.vram_baseline_mb = 6000.0
        sampler = MagicMock()
        sampler.stop.side_effect = [{"vram_peak_mb": 9500.0}, {"vram_peak_mb": 5000.0}]
        with patch.object(runner, "_create_sampler", return_value=sampler):
            _, first = runner._generate_with_telemetry("m", "p", {})
            _, second = runner._generate_with_telemetry("m", "p", {})
        self.assertEqual((first["vram_baseline_mb"], first["vram_model_mb"]), (6000.0, 3500.0))
        self.assertEqual(second["vram_model_mb"], 0.0)

    def test_no_baseline_no_model_memory_fields(self):
        runner = self.runner(FakeClient())
        sampler = MagicMock()
        sampler.stop.return_value = {"vram_peak_mb": 100.0}
        with patch.object(runner, "_create_sampler", return_value=sampler):
            _, hw = runner._generate_with_telemetry("m", "p", {})
        self.assertNotIn("vram_model_mb", hw)

    def test_scorecard_reports_baseline_and_model_memory(self):
        runner = self.runner(FakeClient())
        runner.vram_baseline_mb = 6000.0
        sampler = MagicMock()
        sampler.stop.return_value = {"vram_peak_mb": 9500.0, "vram_total_mb": 12000.0}
        with patch.object(runner, "_create_sampler", return_value=sampler):
            recs = runner.run_speed_suite("m", [{"id": "a", "name": "A", "prompt": "x"}])
        card = runner.compute_model_scorecard("m", recs)
        self.assertEqual(
            (card["peak_vram_mb"], card["vram_baseline_mb"], card["vram_model_mb"]), (9500.0, 6000.0, 3500.0)
        )


class TruncationAndBudgetTests(RunnerTestCase):
    CODING = [
        {
            "id": "c",
            "name": "Add",
            "prompt": "write add",
            "options": {"num_predict": 512},
            "test_assertions": ["assert add(1, 2) == 3"],
        }
    ]

    def test_finish_reason_length_marks_the_record_truncated(self):
        client = FakeClient(finish_reason="length", response="<think>still thinking")
        (rec,) = self.runner(client).run_coding_suite("m", self.CODING)
        self.assertTrue(rec["truncated"])
        self.assertIn("truncated", rec["sandbox_error"])
        self.assertFalse(rec["success"])

    def test_engines_without_finish_reason_are_judged_by_the_token_budget(self):
        client = FakeClient(eval_count=512, response="```python\ndef add(a, b):\n    return a")
        (rec,) = self.runner(client).run_coding_suite("m", self.CODING)
        self.assertTrue(rec["truncated"])

    def test_normal_stop_is_not_truncated(self):
        client = FakeClient(finish_reason="stop", response="```python\ndef add(a, b):\n    return a + b\n```")
        (rec,) = self.runner(client).run_coding_suite("m", self.CODING)
        self.assertFalse(rec["truncated"])
        self.assertTrue(rec["success"])

    def test_truncated_but_complete_code_keeps_the_real_test_result(self):
        client = FakeClient(finish_reason="length", response="```python\ndef add(a, b):\n    return 0\n```\nAnd then")
        (rec,) = self.runner(client).run_coding_suite("m", self.CODING)
        self.assertNotIn("truncated at", rec["sandbox_error"].lower())

    def test_thinking_models_get_a_larger_budget(self):
        client = FakeClient(finish_reason="stop", response="```python\ndef add(a, b):\n    return a + b\n```")
        self.runner(client).run_coding_suite("deepseek-r1:14b", self.CODING)
        self.assertEqual(client.calls[-1]["options"]["num_predict"], 512 * DEFAULT_THINKING_TOKEN_MULTIPLIER)

    def test_other_models_keep_the_scenario_budget(self):
        client = FakeClient(finish_reason="stop", response="```python\ndef add(a, b):\n    return a + b\n```")
        self.runner(client).run_coding_suite("phi3:mini", self.CODING)
        self.assertEqual(client.calls[-1]["options"]["num_predict"], 512)

    def test_budget_multiplier_and_patterns_come_from_config(self):
        cfg = {"benchmark": {"thinking_models": ["mymodel"], "thinking_token_multiplier": 2}}
        client = FakeClient(finish_reason="stop")
        runner = self.runner(client, cfg)
        runner.run_coding_suite("MyModel:7b", self.CODING)
        self.assertEqual(client.calls[-1]["options"]["num_predict"], 1024)
        runner.run_coding_suite("deepseek-r1:14b", self.CODING)  # no longer matched by the overridden list
        self.assertEqual(client.calls[-1]["options"]["num_predict"], 512)

    def test_speed_and_context_suites_are_not_scaled(self):
        client = FakeClient(finish_reason="stop")
        runner = self.runner(client)
        runner.run_speed_suite(
            "deepseek-r1:14b", [{"id": "s", "name": "S", "prompt": "x", "options": {"num_predict": 100}}]
        )
        self.assertEqual(client.calls[-1]["options"]["num_predict"], 100)

    def test_fixed_length_suites_are_never_reported_as_truncated(self):
        """Speed and context ask for a fixed number of tokens, so hitting the limit there is by design."""
        client = FakeClient(finish_reason="length", eval_count=512)
        runner = self.runner(client)
        speed = runner.run_speed_suite("m", [{"id": "s", "name": "S", "prompt": "x", "options": {"num_predict": 512}}])
        context = runner.run_context_suite("m", ContextSuiteTests.SCENARIOS)
        self.assertFalse(any(r["truncated"] for r in speed + context))
        self.assertEqual(speed[0]["finish_reason"], "length")  # still recorded
        self.assertEqual(runner.compute_model_scorecard("m", speed + context)["truncated_runs"], 0)

    def test_scorecard_counts_truncated_runs(self):
        client = FakeClient(finish_reason="length", response="<think>...")
        runner = self.runner(client)
        recs = runner.run_coding_suite("m", self.CODING * 2)
        self.assertEqual(runner.compute_model_scorecard("m", recs)["truncated_runs"], 2)

    def test_reasoning_record_exposes_audit_fields(self):
        scenario = {"id": "r", "name": "Boxes", "prompt": "puzzle", **BOXES, "options": {"num_predict": 1024}}
        client = FakeClient(finish_reason="stop", response="Box 1: Oranges\nBox 2: Apples and Oranges\nBox 3: Apples")
        (rec,) = self.runner(client).run_reasoning_suite("m", [scenario])
        self.assertTrue(rec["correct"])
        self.assertIn("Box 3: Apples", rec["answer_excerpt"])
        self.assertFalse(rec["truncated_thinking"])


class ReportTests(unittest.TestCase):
    CARD = {
        "model": "deepseek-r1:14b",
        "runtime": "ollama",
        "engine": "llama.cpp",
        "composite_score": 50.0,
        "coding_pass_rate": 10.0,
        "reasoning_accuracy": 33.3,
        "avg_eval_tok_sec": 60.0,
        "avg_prompt_tok_sec": 4373.0,
        "avg_prefill_eff_tok_sec": 812.0,
        "avg_ttft_sec": 0.5,
        "peak_vram_mb": 10800.0,
        "vram_model_mb": 4700.0,
        "truncated_runs": 3,
        "vram_warning": False,
        "total_prompt_tokens": 0,
        "total_eval_tokens": 0,
        "total_tokens_saved": 0,
        "est_cost_saved_usd": 0.0,
    }

    def test_markdown_leaderboard_shows_effective_prefill_model_memory_and_truncation(self):
        import os
        import tempfile

        from benchrig.reporting.markdown import generate_markdown_report

        with tempfile.TemporaryDirectory() as tmp:
            content = generate_markdown_report([self.CARD], [], {}, output_path=os.path.join(tmp, "r.md"))
        self.assertIn("Prefill, eff. (t/s)", content)
        self.assertIn("812.0 t/s", content)  # the effective value, not the engine-reported 4373
        self.assertNotIn("4373.0 t/s", content)
        self.assertIn("Model", content)
        self.assertIn("4700 MB", content)
        self.assertIn("⚠ 3 cut", content)
        self.assertIn("whole GPU", content)

    def test_older_scorecards_without_the_new_fields_still_render(self):
        import os
        import tempfile

        from benchrig.reporting.markdown import generate_markdown_report

        old = {
            k: v
            for k, v in self.CARD.items()
            if k not in ("avg_prefill_eff_tok_sec", "vram_model_mb", "truncated_runs")
        }
        with tempfile.TemporaryDirectory() as tmp:
            content = generate_markdown_report([old], [], {}, output_path=os.path.join(tmp, "r.md"))
        self.assertIn("4373.0 t/s", content)  # falls back to the engine-reported value
        self.assertNotIn("N cut", content)  # no truncation footnote for runs that were not cut off


class ClientFinishReasonTests(unittest.TestCase):
    @patch("requests.post")
    def test_ollama_reports_done_reason(self, mock_post):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"response": "x", "done": True, "done_reason": "length", "eval_count": 5}
        mock_post.return_value = resp
        result = OllamaClient().generate("m", "p", measure_ttft=False)
        self.assertEqual(result["finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()


class RepetitionTests(RunnerTestCase):
    ALIGNED = {"execution_alignment_1to1": {"temperature": 0.1, "top_p": 0.9, "seed": 42, "context_tokens": 4096}}
    SPEED = [{"id": "s", "name": "S", "prompt": "hello", "options": {"num_predict": 8}}]

    def test_first_repetition_is_sent_unchanged_and_later_ones_are_marked(self):
        client = FakeClient()
        runner = self.runner(client)
        for run in (0, 1, 2):
            runner.run_index = run
            runner.run_speed_suite("m", self.SPEED)
        self.assertEqual([c["prompt"] for c in client.calls], ["hello", "(request 2)\nhello", "(request 3)\nhello"])

    def test_the_marker_changes_the_first_token_so_a_prefix_cache_cannot_hit(self):
        client = FakeClient()
        runner = self.runner(client)
        runner.run_index = 1
        runner.run_speed_suite("m", self.SPEED)
        self.assertNotEqual(client.calls[0]["prompt"][:5], "hello"[:5])

    def test_warmup_is_never_marked(self):
        client = FakeClient()
        runner = self.runner(client)
        runner.run_index = 2
        runner.warmup("m")
        self.assertEqual(client.calls[0]["prompt"], "Hello, respond with OK.")

    def test_aligned_sampling_is_the_default_and_the_seed_follows_the_repetition(self):
        client = FakeClient()
        runner = self.runner(client, self.ALIGNED)
        for run in (0, 2):
            runner.run_index = run
            runner.run_speed_suite("m", self.SPEED)
        first, third = (c["options"] for c in client.calls)
        self.assertEqual((first["seed"], third["seed"]), (42, 44))
        self.assertEqual((first["top_p"], first["temperature"], first["num_ctx"]), (0.9, 0.1, 4096))

    def test_a_scenarios_own_options_win_over_the_aligned_defaults(self):
        client = FakeClient()
        scenario = [{"id": "s", "name": "S", "prompt": "x", "options": {"temperature": 0.7, "num_ctx": 2048}}]
        self.runner(client, self.ALIGNED).run_speed_suite("m", scenario)
        options = client.calls[0]["options"]
        self.assertEqual((options["temperature"], options["num_ctx"], options["top_p"]), (0.7, 2048, 0.9))

    def test_without_alignment_config_only_the_think_default_is_added(self):
        client = FakeClient()
        self.runner(client).run_speed_suite("m", self.SPEED)
        self.assertEqual(client.calls[0]["options"], {"num_predict": 8, "think": False})

    def test_context_suite_uses_the_aligned_defaults_but_keeps_its_own_window(self):
        client = FakeClient()
        runner = self.runner(client, self.ALIGNED)
        runner.run_context_suite("m", ContextSuiteTests.SCENARIOS[:1])
        options = client.calls[0]["options"]
        self.assertEqual((options["num_ctx"], options["seed"], options["top_p"]), (512, 42, 0.9))

    def test_records_carry_their_repetition_and_the_scorecard_reports_the_spread(self):
        speeds = iter([100.0, 140.0, 120.0])

        class Varying(FakeClient):
            def generate(self, *args, **kwargs):
                self.response["eval_tok_per_sec"] = next(speeds)
                return super().generate(*args, **kwargs)

        runner = self.runner(Varying())
        records = []
        for run in range(3):
            runner.run_index = run
            records += runner.run_speed_suite("m", self.SPEED)
        self.assertEqual([r["run"] for r in records], [0, 1, 2])
        card = runner.compute_model_scorecard("m", records)
        self.assertEqual(card["runs"], 3)
        self.assertEqual(card["spread"]["avg_eval_tok_sec"], [100.0, 140.0])
        self.assertEqual(card["avg_eval_tok_sec_mean"], 120.0)  # plain mean of the three
        self.assertEqual(card["avg_eval_tok_sec"], 117.8)  # token-weighted: equal token counts, so the harmonic mean

    def test_a_single_run_has_no_spread(self):
        runner = self.runner(FakeClient())
        card = runner.compute_model_scorecard("m", runner.run_speed_suite("m", self.SPEED))
        self.assertEqual((card["runs"], card["spread"]), (1, None))

    def test_spread_lines_for_the_reports(self):
        from benchrig.reporting.common import spread_lines

        card = {
            "model": "m",
            "runs": 3,
            "spread": {
                k: [1.0, 2.0]
                for k in (
                    "composite_score",
                    "coding_pass_rate",
                    "reasoning_accuracy",
                    "avg_eval_tok_sec",
                    "avg_ttft_sec",
                )
            },
        }
        (line,) = spread_lines([card, {"model": "single", "spread": None}])
        self.assertIn("3 runs", line)
        self.assertIn("composite 1.0-2.0", line)


class DirtyBaselineTests(RunnerTestCase):
    """A runtime that still holds a model when the baseline is read would make peak minus baseline too small."""

    def baseline_with(self, loaded):
        client = FakeClient()
        client.get_running_models = lambda: loaded
        runner = self.runner(client)
        provider = MagicMock()
        provider.read_gpu.return_value = (5000.0, 12000.0, 0.0, 0.0, 0.0)
        with patch.object(runner, "_create_sampler", return_value=MagicMock(provider=provider)):
            runner.measure_vram_baseline()
        return runner

    def test_a_loaded_model_at_baseline_time_marks_the_baseline_dirty(self):
        self.assertTrue(self.baseline_with([{"name": "previous-model"}]).vram_baseline_dirty)

    def test_no_loaded_model_means_a_clean_baseline(self):
        self.assertFalse(self.baseline_with([]).vram_baseline_dirty)

    def test_a_runtime_that_cannot_list_models_is_not_penalised(self):
        client = FakeClient()

        def broken():
            raise OSError("down")

        client.get_running_models = broken
        runner = self.runner(client)
        provider = MagicMock()
        provider.read_gpu.return_value = (5000.0, 12000.0, 0.0, 0.0, 0.0)
        with patch.object(runner, "_create_sampler", return_value=MagicMock(provider=provider)):
            runner.measure_vram_baseline()
        self.assertFalse(runner.vram_baseline_dirty)

    def test_a_dirty_baseline_gives_no_model_memory_figure(self):
        runner = self.baseline_with([{"name": "previous-model"}])
        sampler = MagicMock()
        sampler.stop.return_value = {"vram_peak_mb": 9000.0, "vram_total_mb": 12000.0}
        with patch.object(runner, "_create_sampler", return_value=sampler):
            recs = runner.run_speed_suite("m", [{"id": "s", "name": "S", "prompt": "x"}])
        self.assertTrue(recs[0]["hardware"]["vram_baseline_dirty"])
        self.assertNotIn("vram_model_mb", recs[0]["hardware"])
        card = runner.compute_model_scorecard("m", recs)
        self.assertIsNone(card["vram_model_mb"])
        self.assertTrue(card["vram_baseline_dirty"])
        self.assertEqual(card["peak_vram_mb"], 9000.0)  # the whole-GPU peak is still reported

    def test_a_clean_baseline_still_gives_the_figure(self):
        runner = self.baseline_with([])
        sampler = MagicMock()
        sampler.stop.return_value = {"vram_peak_mb": 9000.0, "vram_total_mb": 12000.0}
        with patch.object(runner, "_create_sampler", return_value=sampler):
            recs = runner.run_speed_suite("m", [{"id": "s", "name": "S", "prompt": "x"}])
        card = runner.compute_model_scorecard("m", recs)
        self.assertEqual(card["vram_model_mb"], 4000.0)
        self.assertFalse(card["vram_baseline_dirty"])

    def test_reports_explain_the_missing_figure(self):
        import os
        import tempfile

        from benchrig.reporting.markdown import generate_markdown_report

        card = ReportTests.CARD | {"vram_model_mb": None, "vram_baseline_dirty": True}
        with tempfile.TemporaryDirectory() as tmp:
            content = generate_markdown_report([card], [], {}, output_path=os.path.join(tmp, "r.md"))
        self.assertIn("still loaded", content)
        self.assertIn("| - |", content)

    def test_prism_reports_its_active_model_as_running(self):
        from benchrig.core.client import PrismClient

        client = PrismClient()
        with patch.object(PrismClient, "_health", return_value={"active_model": "phi-4-mini"}):
            self.assertEqual(client.get_running_models(), [{"name": "phi-4-mini"}])
        with patch.object(PrismClient, "_health", return_value={"active_model": None}):
            self.assertEqual(client.get_running_models(), [])
        with patch.object(PrismClient, "_health", return_value={}):
            self.assertEqual(client.get_running_models(), [])


class ExtraMetricsTests(RunnerTestCase):
    def test_decode_speed_is_weighted_by_tokens(self):
        runner = self.runner(FakeClient())
        long_run = {
            "model": "m",
            "runtime": "fake",
            "suite": "speed",
            "eval_count": 512,
            "eval_tok_per_sec": 100.0,
            "prompt_tok_per_sec": 1.0,
            "ttft_sec": 0.1,
        }
        short_run = {**long_run, "eval_count": 16, "eval_tok_per_sec": 200.0}
        card = runner.compute_model_scorecard("m", [long_run, short_run])
        self.assertEqual(card["avg_eval_tok_sec"], 101.5)  # 528 tokens / (5.12 s + 0.08 s), not the mean of 100 and 200
        self.assertEqual(card["avg_eval_tok_sec_mean"], 150.0)

    def test_records_without_token_counts_fall_back_to_the_mean(self):
        runner = self.runner(FakeClient())
        old = {
            "model": "m",
            "runtime": "fake",
            "suite": "speed",
            "eval_tok_per_sec": 80.0,
            "prompt_tok_per_sec": 1.0,
            "ttft_sec": 0.1,
        }
        self.assertEqual(runner.compute_model_scorecard("m", [old])["avg_eval_tok_sec"], 80.0)

    def test_warmup_records_the_cold_start_and_puts_it_on_every_record(self):
        client = FakeClient(total_time_sec=4.2)
        runner = self.runner(client)
        runner.warmup("m")
        (rec,) = runner.run_speed_suite("m", [{"id": "s", "name": "S", "prompt": "x"}])
        self.assertEqual(rec["cold_start_sec"], 4.2)
        self.assertEqual(runner.compute_model_scorecard("m", [rec])["cold_start_sec"], 4.2)

    def test_a_failed_warmup_reports_no_cold_start(self):
        runner = self.runner(FakeClient(success=False, total_time_sec=9.0))
        runner.warmup("m")
        (rec,) = runner.run_speed_suite("m", [{"id": "s", "name": "S", "prompt": "x"}])
        self.assertNotIn("cold_start_sec", rec)

    def test_gpu_fit_comes_from_the_running_models_list(self):
        client = FakeClient()
        client.get_running_models = lambda: [
            {"name": "deepseek-r1:14b", "size": 9_470_000_000, "size_vram": 9_470_000_000}
        ]
        runner = self.runner(client)
        runner.warmup("deepseek-r1:14b")
        self.assertEqual(runner.gpu_fit_pct, 100.0)
        client.get_running_models = lambda: [{"name": "big:70b", "size": 40_000_000_000, "size_vram": 10_000_000_000}]
        self.assertEqual(runner.measure_gpu_fit("big:70b"), 25.0)

    def test_gpu_fit_is_none_when_the_runtime_does_not_say(self):
        runner = self.runner(FakeClient())
        runner.warmup("m")  # BaseRuntimeClient.get_running_models() is empty
        self.assertIsNone(runner.gpu_fit_pct)
        client = FakeClient()
        client.get_running_models = lambda: [{"name": "other", "size": 1, "size_vram": 1}]
        self.assertIsNone(self.runner(client).measure_gpu_fit("m"))

        def broken():
            raise OSError("down")

        client.get_running_models = broken
        self.assertIsNone(self.runner(client).measure_gpu_fit("m"))

    def test_gpu_fit_reaches_the_scorecard(self):
        client = FakeClient()
        client.get_running_models = lambda: [{"name": "m", "size": 100, "size_vram": 60}]
        runner = self.runner(client)
        runner.warmup("m")
        card = runner.compute_model_scorecard(
            "m", runner.run_speed_suite("m", [{"id": "s", "name": "S", "prompt": "x"}])
        )
        self.assertEqual(card["gpu_fit_pct"], 60.0)

    def test_energy_is_tokens_per_second_over_watts(self):
        runner = self.runner(FakeClient(eval_tok_per_sec=120.0))
        sampler = MagicMock()
        sampler.stop.return_value = {"power_avg_w": 60.0, "vram_peak_mb": 1.0}
        with patch.object(runner, "_create_sampler", return_value=sampler):
            recs = runner.run_speed_suite("m", [{"id": "s", "name": "S", "prompt": "x"}])
        self.assertEqual(recs[0]["tokens_per_joule"], 2.0)
        self.assertEqual(runner.compute_model_scorecard("m", recs)["tokens_per_joule"], 2.0)

    def test_no_power_reading_means_no_energy_figure(self):
        runner = self.runner(FakeClient())
        (rec,) = runner.run_speed_suite("m", [{"id": "s", "name": "S", "prompt": "x"}])
        self.assertNotIn("tokens_per_joule", rec)
        self.assertIsNone(runner.compute_model_scorecard("m", [rec])["tokens_per_joule"])


class EfficiencyReportTests(unittest.TestCase):
    CARD = ReportTests.CARD | {
        "cold_start_sec": 4.2,
        "gpu_fit_pct": 100.0,
        "tokens_per_joule": 2.1,
        "context_retrieval_pct": 80.0,
    }

    def test_markdown_shows_the_efficiency_table(self):
        import os
        import tempfile

        from benchrig.reporting.markdown import generate_markdown_report

        with tempfile.TemporaryDirectory() as tmp:
            content = generate_markdown_report([self.CARD], [], {}, output_path=os.path.join(tmp, "r.md"))
        self.assertIn("Start-up, GPU Fit & Efficiency", content)
        self.assertIn("| `deepseek-r1:14b` | 4.2 s | 100% | 2.10 | 80% |", content)

    def test_scorecards_without_any_of_the_figures_add_no_table(self):
        from benchrig.reporting.common import efficiency_rows

        self.assertEqual(efficiency_rows([ReportTests.CARD]), [])
        self.assertEqual(efficiency_rows([{**ReportTests.CARD, "gpu_fit_pct": 55.0}])[0][2], "55%")

    def test_the_terminal_leaderboard_shows_it_too(self):
        from rich.console import Console

        from benchrig.reporting.display import display_leaderboard

        console = Console(record=True, width=220)
        with patch("benchrig.reporting.display.console", console):
            display_leaderboard([self.CARD], specs={"memory_type": "VRAM", "gpu_type": "nvidia"})
        text = console.export_text()
        self.assertIn("Start-up, GPU fit", text)
        self.assertIn("4.2 s", text)
