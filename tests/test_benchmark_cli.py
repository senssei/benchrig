"""Unit tests for benchmark.py CLI helpers (target resolution, pair lookup, comparison selection)."""

import unittest
from unittest.mock import MagicMock

import benchmark


def fake_client(installed, reachable=True):
    client = MagicMock()
    client.is_reachable.return_value = reachable
    client.list_installed_models.return_value = [{"name": n} for n in installed]
    return client


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

        from core.runner import BenchmarkRunner

        for filename, method, _ in benchmark.SUITES.values():
            self.assertTrue(hasattr(BenchmarkRunner, method), method)
            self.assertTrue(os.path.exists(os.path.join(benchmark.SCENARIOS_DIR, filename)), filename)


if __name__ == "__main__":
    unittest.main()
