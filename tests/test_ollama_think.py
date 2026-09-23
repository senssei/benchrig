"""Ollama's `think` option: sending it, reading the thinking stream, and the per-suite defaults."""

import json
import unittest
from unittest.mock import MagicMock, patch

from benchrig.core.client import OllamaClient
from benchrig.core.runner import DEFAULT_THINKING_TOKEN_MULTIPLIER
from tests.test_measurement_methodology import FakeClient, RunnerTestCase


def stream(*chunks):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_lines.return_value = [json.dumps(c).encode() for c in chunks]
    return resp


DONE = {
    "done": True,
    "eval_count": 5,
    "eval_duration": 1_000_000_000,
    "prompt_eval_count": 3,
    "prompt_eval_duration": 10_000_000,
}


def fake_ollama(capabilities, generate_response):
    """requests.post stand-in: `/api/show` lists capabilities, `/api/generate` returns `generate_response`."""
    calls = {"show": 0, "generate": []}

    def post(url, json=None, **kwargs):
        if url.endswith("/api/show"):
            calls["show"] += 1
            resp = MagicMock(status_code=200)
            resp.json.return_value = {"capabilities": capabilities}
            return resp
        calls["generate"].append(json)
        return generate_response

    return post, calls


class ThinkPayloadTests(unittest.TestCase):
    def test_think_is_sent_at_top_level_for_a_model_that_can_think(self):
        post, calls = fake_ollama(["completion", "thinking"], stream({"response": "x"}, DONE))
        with patch("requests.post", side_effect=post):
            OllamaClient().generate("deepseek-r1:14b", "hi", options={"think": False, "num_predict": 8})
        payload = calls["generate"][0]
        self.assertIs(payload["think"], False)
        self.assertNotIn("think", payload["options"])
        self.assertEqual(payload["options"]["num_predict"], 8)

    def test_think_is_not_sent_to_a_model_that_cannot_think(self):
        post, calls = fake_ollama(["completion"], stream({"response": "x"}, DONE))
        with patch("requests.post", side_effect=post):
            OllamaClient().generate("phi4-mini:latest", "hi", options={"think": False})
        self.assertNotIn("think", calls["generate"][0])
        self.assertNotIn("think", calls["generate"][0]["options"])

    def test_no_think_option_means_the_models_default(self):
        post, calls = fake_ollama(["thinking"], stream({"response": "x"}, DONE))
        with patch("requests.post", side_effect=post):
            OllamaClient().generate("deepseek-r1:14b", "hi")
        self.assertNotIn("think", calls["generate"][0])

    def test_capabilities_are_looked_up_once_per_model(self):
        post, calls = fake_ollama(["thinking"], stream({"response": "x"}, DONE))
        client = OllamaClient()
        with patch("requests.post", side_effect=post):
            client.generate("deepseek-r1:14b", "a", options={"think": True})
            client.generate("deepseek-r1:14b", "b", options={"think": True})
        self.assertEqual(calls["show"], 1)

    def test_an_unreachable_show_endpoint_means_no_think(self):
        with patch("requests.post", side_effect=OSError("down")):
            self.assertFalse(OllamaClient().supports_thinking("m"))


class ThinkingStreamTests(unittest.TestCase):
    def generate(self, chunks, times):
        post, _ = fake_ollama(["thinking"], stream(*chunks))
        with (
            patch("requests.post", side_effect=post),
            patch("benchrig.core.client.time.perf_counter", side_effect=times),
        ):
            return OllamaClient().generate("deepseek-r1:14b", "hi", options={"think": True})

    def test_ttft_is_the_first_token_of_any_kind_and_the_answer_is_reported_separately(self):
        # Fixture values for ``times`` account for the additional ``time.perf_counter``
        # calls introduced by both ``OllamaClient._make_request`` and ``supports_thinking``
        # (each goes through ``_logged_request`` and consumes 2 perf_counter readings).
        # On a fresh OllamaClient() the consume order is:
        #   0..1 = supports_thinking HTTP hook (started_at + duration)
        #   2    = start_wall_time
        #   3..4 = generate HTTP hook (started_at + duration)
        #   5    = first thinking token, 6 = first answer token, 7 = end_wall_time
        # ttft_sec = first_thinking - start_wall_time = 3.0 - 2.0 = 1.0
        # answer_ttft_sec = first_answer - start_wall_time = 7.0 - 2.0 = 5.0
        result = self.generate(
            [{"thinking": "let me"}, {"thinking": " see"}, {"response": "42"}, DONE],
            [0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 7.0, 8.0],
        )
        self.assertEqual(result["ttft_sec"], 1.0)
        self.assertEqual(result["answer_ttft_sec"], 5.0)
        self.assertEqual(result["thinking_chars"], len("let me see"))
        self.assertEqual(result["response"], "42")  # the thinking is not part of the answer
        self.assertIs(result["think"], True)

    def test_a_response_that_never_left_the_thinking_phase(self):
        """The budget ran out while thinking: there is no answer token, but the latency is still the first token.

        Same consume order as ``test_ttft_is_the_first_token_of_any_kind_...`` but
        only one thinking chunk and no answer chunk:
          0..1 = supports_thinking hook, 2 = start_wall_time, 3..4 = generate hook,
          5 = first thinking token, 6 = end_wall_time.
        ttft_sec = first_thinking - start_wall_time = 3.0 - 1.0 = 2.0.
        """
        result = self.generate(
            [{"thinking": "hmm"}, {**DONE, "done_reason": "length"}],
            [0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 10.0],
        )
        self.assertEqual(result["ttft_sec"], 2.0)
        self.assertIsNone(result["answer_ttft_sec"])
        self.assertEqual(result["response"], "")
        self.assertEqual(result["finish_reason"], "length")

    def test_a_model_that_does_not_think_reports_no_thinking(self):
        post, _ = fake_ollama([], stream({"response": "hi"}, DONE))
        # Phase 11 follow-up: this test does NOT call `supports_thinking` (the option
        # dict has no `think` key, so the body skips it), so the fixture has 5 readings:
        #   0=start_wall_time, 1..2=generate HTTP hook (start + duration),
        #   3=first answer token, 4=end.
        with (
            patch("requests.post", side_effect=post),
            patch(
                "benchrig.core.client.time.perf_counter",
                side_effect=[0.0, 0.5, 0.6, 1.0, 2.0],
            ),
        ):
            result = OllamaClient().generate("phi4-mini:latest", "hi")
        self.assertEqual((result["ttft_sec"], result["answer_ttft_sec"], result["thinking_chars"]), (1.0, 1.0, 0))
        self.assertIsNone(result["think"])

    def test_non_streaming_reads_the_thinking_field(self):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"response": "42", "thinking": "abcd", **DONE}
        post, _ = fake_ollama(["thinking"], resp)
        with patch("requests.post", side_effect=post):
            result = OllamaClient().generate("deepseek-r1:14b", "hi", measure_ttft=False)
        self.assertEqual(result["thinking_chars"], 4)


class SuiteDefaultsTests(RunnerTestCase):
    def options_sent(self, runner_method, suite_arg, model="deepseek-r1:14b", config=None, scenario=None):
        client = FakeClient(finish_reason="stop")
        runner = self.runner(client, config)
        getattr(runner, runner_method)(model, [scenario or suite_arg])
        return client.calls[-1]["options"]

    def test_speed_and_context_do_not_think_by_default(self):
        speed = {"id": "s", "name": "S", "prompt": "x", "options": {"num_predict": 8}}
        self.assertIs(self.options_sent("run_speed_suite", speed)["think"], False)
        client = FakeClient()
        self.runner(client).run_context_suite(
            "m", [{"context_size": 512, "name": "c", "instruction": "Q", "fill_ratio": 0.75}]
        )
        self.assertIs(client.calls[-1]["options"]["think"], False)

    def test_answer_suites_use_the_models_own_default(self):
        coding = {"id": "c", "name": "C", "prompt": "x", "options": {"num_predict": 100}, "test_assertions": []}
        self.assertNotIn("think", self.options_sent("run_coding_suite", coding))

    def test_config_sets_think_per_suite_and_a_thinking_model_keeps_its_budget(self):
        coding = {"id": "c", "name": "C", "prompt": "x", "options": {"num_predict": 100}, "test_assertions": []}
        options = self.options_sent("run_coding_suite", coding, config={"benchmark": {"think": {"coding": False}}})
        self.assertIs(options["think"], False)
        # still multiplied: deepseek-r1 with thinking off was measured running into the base budget
        self.assertEqual(options["num_predict"], 100 * DEFAULT_THINKING_TOKEN_MULTIPLIER)

    def test_a_model_that_is_not_a_thinking_model_keeps_the_scenario_budget(self):
        coding = {"id": "c", "name": "C", "prompt": "x", "options": {"num_predict": 100}, "test_assertions": []}
        options = self.options_sent("run_coding_suite", coding, model="phi4-mini:latest")
        self.assertEqual(options["num_predict"], 100)

    def test_a_scenarios_own_think_wins(self):
        speed = {"id": "s", "name": "S", "prompt": "x", "options": {"num_predict": 8, "think": True}}
        self.assertIs(self.options_sent("run_speed_suite", speed)["think"], True)

    def test_records_show_the_think_setting_and_the_thinking_time(self):
        client = FakeClient(think=True, thinking_chars=900, ttft_sec=0.4, answer_ttft_sec=12.4)
        (rec,) = self.runner(client).run_speed_suite("m", [{"id": "s", "name": "S", "prompt": "x"}])
        self.assertEqual(
            (rec["think"], rec["thinking_chars"], rec["answer_ttft_sec"], rec["think_time_sec"]),
            (True, 900, 12.4, 12.0),
        )

    def test_records_of_non_thinking_runs_stay_compact(self):
        (rec,) = self.runner(FakeClient()).run_speed_suite("m", [{"id": "s", "name": "S", "prompt": "x"}])
        for key in ("think", "thinking_chars", "answer_ttft_sec", "think_time_sec"):
            self.assertNotIn(key, rec)


if __name__ == "__main__":
    unittest.main()
