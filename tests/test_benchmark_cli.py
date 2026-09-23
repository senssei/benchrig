"""Unit tests for benchmark.py CLI helpers (target resolution, pair lookup, comparison selection)."""

import unittest
from unittest.mock import MagicMock, patch

from benchrig import cli as benchmark
from benchrig.core.client import BaseRuntimeClient


def fake_client(installed, reachable=True):
    client = MagicMock()
    client.is_reachable.return_value = reachable
    client.list_installed_models.return_value = [{"name": n} for n in installed]
    return client


class GeneratingFakeClient(BaseRuntimeClient):
    """A minimal real client (not a MagicMock) so `evaluate_model` can run its real BenchmarkRunner end to end."""

    name = "fake"
    display_name = "Fake"
    engine_name = "fake-engine"

    def __init__(self):
        super().__init__(base_url="http://fake")

    def generate(self, model, prompt, system=None, options=None, measure_ttft=True):
        return {
            "success": True,
            "response": "42",
            "eval_count": 1,
            "eval_tok_per_sec": 1.0,
            "prompt_eval_count": 1,
            "prompt_tok_per_sec": 1.0,
            "ttft_sec": 0.01,
            "total_time_sec": 0.01,
        }


class ResolveTargetModelsTests(unittest.TestCase):
    def setUp(self):
        self.clients = {
            "ollama": fake_client(["qwen2.5-coder:7b", "phi3:mini"]),
            "foundry": fake_client(["phi-4"]),
            "onnx-gpu": fake_client([], reachable=False),
        }

    def test_installed_uses_selected_runtime_only(self):
        targets = benchmark.resolve_target_models("installed", "ollama", self.clients)
        self.assertEqual(targets, [("ollama", "qwen2.5-coder:7b"), ("ollama", "phi3:mini")])

    def test_installed_all_skips_unreachable_runtimes(self):
        targets = benchmark.resolve_target_models("all", "all", self.clients)
        self.assertEqual(
            targets,
            [("ollama", "qwen2.5-coder:7b"), ("ollama", "phi3:mini"), ("foundry", "phi-4")],
        )

    def test_prefixes_are_normalized_and_model_tags_preserved(self):
        targets = benchmark.resolve_target_models(
            "ollama:qwen2.5-coder:7b, ms-foundry:phi-4, onnx:phi-4-cuda", "ollama", self.clients
        )
        self.assertEqual(
            targets,
            [("ollama", "qwen2.5-coder:7b"), ("foundry", "phi-4"), ("onnx-gpu", "phi-4-cuda")],
        )

    def test_unprefixed_model_tag_is_not_mistaken_for_runtime(self):
        targets = benchmark.resolve_target_models("phi3:mini", "foundry", self.clients)
        self.assertEqual(targets, [("foundry", "phi3:mini")])

    def test_unprefixed_with_all_finds_installing_runtime_or_defaults_to_ollama(self):
        targets = benchmark.resolve_target_models("phi-4,unknown", "all", self.clients)
        self.assertEqual(targets, [("foundry", "phi-4"), ("ollama", "unknown")])


class PairTests(unittest.TestCase):
    CONFIG = {
        "model_pairs_1to1": [
            {"id": "p", "name": "Pair", "ollama": "o", "foundry": "f", "onnx": "x", "recommended_suite": "coding"}
        ]
    }

    def test_full_pair_runs_both_engines(self):
        targets, suite = benchmark.resolve_pair_targets(self.CONFIG, "p", "ollama", has_baseline=False)
        self.assertEqual(targets, [("ollama", "o"), ("foundry", "f")])
        self.assertEqual(suite, "coding")

    def test_baseline_skips_ollama(self):
        self.assertEqual(
            benchmark.resolve_pair_targets(self.CONFIG, "p", "foundry", has_baseline=True)[0], [("foundry", "f")]
        )
        self.assertEqual(
            benchmark.resolve_pair_targets(self.CONFIG, "p", "onnx-gpu", has_baseline=True)[0], [("onnx-gpu", "x")]
        )

    def test_unknown_pair_exits(self):
        with self.assertRaises(SystemExit):
            benchmark.resolve_pair_targets(self.CONFIG, "nope", "ollama", has_baseline=False)


class ParserTests(unittest.TestCase):
    def test_suite_choices_follow_registry(self):
        parser = benchmark.build_parser()
        for suite in benchmark.SUITES:
            self.assertEqual(parser.parse_args(["--suite", suite]).suite, suite)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--suite", "bogus"])

    def test_every_suite_has_a_runner_method_and_scenario_file(self):
        import os

        from benchrig.core.runner import BenchmarkRunner

        for filename, method, _ in benchmark.SUITES.values():
            self.assertTrue(hasattr(BenchmarkRunner, method), method)
            self.assertTrue(os.path.exists(os.path.join(benchmark.resolve_scenarios_dir(), filename)), filename)


class PullRecommendedModelsTests(unittest.TestCase):
    def test_onnx_gpu_handles_gracefully(self):
        clients = {
            "onnx-gpu": fake_client(["Phi-4-mini-instruct-cuda-gpu"], reachable=True),
        }
        config = {"recommended_models": {}}
        # Should execute without raising any exception
        benchmark.pull_recommended_models(clients, config, target_runtime="onnx-gpu")

    def test_empty_recommended_handles_gracefully(self):
        clients = {
            "ollama": fake_client([], reachable=True),
        }
        config = {"recommended_models": {}}
        benchmark.pull_recommended_models(clients, config, target_runtime="ollama")


class TotalScenarioStepsTests(unittest.TestCase):
    """spec.md/plan.md Phase 8, item 8.2: the overall progress bar's unit count."""

    def test_counts_one_unit_per_target_run_and_scenario(self):
        scenarios = {"speed": [{"id": "s1"}, {"id": "s2"}], "coding": [{"id": "c1"}]}
        self.assertEqual(benchmark._total_scenario_steps(scenarios, runs=2, num_targets=3), 18)  # 3 scenarios * 2 * 3

    def test_zero_scenarios_is_zero_not_a_crash(self):
        self.assertEqual(benchmark._total_scenario_steps({}, runs=1, num_targets=5), 0)


class EvaluateModelProgressTests(unittest.TestCase):
    """Review finding: evaluate_model's on_progress wiring (Phase 8, item 8.2) had no test at all."""

    def setUp(self):
        patches = [
            patch("benchrig.core.runner.time.sleep"),
            patch("benchrig.core.runner.get_system_specs", return_value={}),
            patch("benchrig.cli.time.sleep"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        sampler_patch = patch("benchrig.core.runner.HardwareSampler")
        mock_sampler_cls = sampler_patch.start()
        self.addCleanup(sampler_patch.stop)
        mock_sampler = mock_sampler_cls.return_value
        mock_sampler.provider.read_gpu.return_value = (0.0, 0.0, 0.0, 0.0, 0.0)
        mock_sampler.provider.read_gpu_foreign_memory_mb.return_value = 0.0
        mock_sampler.stop.return_value = {}
        self.client = GeneratingFakeClient()

    def test_on_progress_fires_once_per_scenario_across_suites_and_runs(self):
        scenarios = {
            "speed": [{"id": "s1", "name": "Speed 1", "prompt": "hi"}, {"id": "s2", "name": "Speed 2", "prompt": "hi"}],
        }
        seen = []
        results = benchmark.evaluate_model(
            "fake", self.client, "m", config={}, suites=["speed"], scenarios=scenarios, runs=2, on_progress=seen.append
        )
        self.assertEqual(len(seen), 4)  # 2 scenarios * 2 runs
        self.assertEqual(seen, results)

    def test_on_progress_is_optional(self):
        scenarios = {"speed": [{"id": "s1", "name": "Speed 1", "prompt": "hi"}]}
        results = benchmark.evaluate_model(
            "fake", self.client, "m", config={}, suites=["speed"], scenarios=scenarios, runs=1
        )
        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
