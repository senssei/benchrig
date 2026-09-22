"""A persistent 'insufficient_resources' failure (model does not fit in available VRAM) must not be
retried scenario-by-scenario, suite-by-suite: the first such failure short-circuits the rest of the
model's scenarios. Before this fix, `_run_suite` called `generate()` for every remaining scenario in
every remaining suite, each one retrying 503s for several seconds, spamming the same error dozens of
times for a condition that cannot resolve itself mid-run (see plan.md, manual-run complaint 2026-09-22).
"""

import unittest
from unittest.mock import patch

from benchrig.core.client import BaseRuntimeClient
from benchrig.core.runner import BenchmarkRunner

INSUFFICIENT_VRAM_ERROR = (
    "prism-local kept responding 503 (4 attempts): insufficient_resources: Insufficient resources to "
    "load model 'qwen2.5-coder-7b-instruct-generic-cpu-4:v4': Insufficient VRAM to load model on CUDA: "
    "9057.7 MB free < 9295.4 MB required (7759.4 MB model footprint + 1536.0 MB reserve)."
)


class AlwaysCapacityFailClient(BaseRuntimeClient):
    """Every call fails with the same persistent (non-transient) capacity error."""

    name = "fake"
    display_name = "Fake"
    engine_name = "fake-engine"

    def __init__(self):
        super().__init__(base_url="http://fake")
        self.calls = 0

    def generate(self, model, prompt, system=None, options=None, measure_ttft=True):
        self.calls += 1
        return {
            "success": False,
            "error": INSUFFICIENT_VRAM_ERROR,
            "model": model,
            "runtime": self.name,
            "engine": self.engine_name,
            "response": "",
            "eval_tok_per_sec": 0.0,
            "prompt_tok_per_sec": 0.0,
            "ttft_sec": 0.0,
            "total_time_sec": 0.01,
        }


class TransientBusyThenOkClient(BaseRuntimeClient):
    """A `server_busy` (queue full) failure is transient: it must NOT short-circuit remaining scenarios."""

    name = "fake"
    display_name = "Fake"
    engine_name = "fake-engine"

    def __init__(self):
        super().__init__(base_url="http://fake")
        self.calls = 0

    def generate(self, model, prompt, system=None, options=None, measure_ttft=True):
        self.calls += 1
        if self.calls == 1:
            return {
                "success": False,
                "error": "prism-local kept responding 503 (4 attempts): server_busy: queue full",
                "model": model,
                "runtime": self.name,
                "engine": self.engine_name,
                "response": "",
                "eval_tok_per_sec": 0.0,
                "prompt_tok_per_sec": 0.0,
                "ttft_sec": 0.0,
                "total_time_sec": 0.01,
            }
        return {
            "success": True,
            "response": "ok",
            "eval_count": 1,
            "eval_tok_per_sec": 1.0,
            "prompt_eval_count": 1,
            "prompt_tok_per_sec": 1.0,
            "ttft_sec": 0.01,
            "total_time_sec": 0.01,
        }


def _scenarios(n, prefix="s"):
    return [{"id": f"{prefix}{i}", "name": f"Scenario {i}", "prompt": "hi"} for i in range(n)]


class CapacityShortCircuitTests(unittest.TestCase):
    def setUp(self):
        patches = [
            patch("benchrig.core.runner.time.sleep"),
            patch("benchrig.core.runner.get_system_specs", return_value={}),
            patch("benchrig.core.runner.HardwareSampler"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_persistent_capacity_failure_stops_remaining_scenarios_in_the_suite(self):
        client = AlwaysCapacityFailClient()
        runner = BenchmarkRunner(client=client, config={})

        results = runner.run_coding_suite("m", _scenarios(5))

        self.assertEqual(len(results), 1, "only the first (failing) scenario should have run")
        self.assertEqual(client.calls, 1)
        self.assertFalse(results[0]["success"])

    def test_persistent_capacity_failure_skips_later_suites_for_the_same_model(self):
        client = AlwaysCapacityFailClient()
        runner = BenchmarkRunner(client=client, config={})

        runner.run_coding_suite("m", _scenarios(3))
        self.assertEqual(client.calls, 1)

        # A later suite for the SAME model (same runner instance, as benchrig/cli.py reuses it across
        # suites) must not retry at all: the condition cannot have changed mid-run.
        results = runner.run_reasoning_suite("m", _scenarios(3, prefix="r"))
        self.assertEqual(results, [])
        self.assertEqual(client.calls, 1, "no new HTTP call should have been made for the next suite")

    def test_transient_busy_failure_does_not_short_circuit(self):
        client = TransientBusyThenOkClient()
        runner = BenchmarkRunner(client=client, config={})

        results = runner.run_speed_suite("m", _scenarios(3))

        self.assertEqual(len(results), 3, "a transient/queue-full failure must not stop the remaining scenarios")
        self.assertEqual(client.calls, 3)


if __name__ == "__main__":
    unittest.main()
