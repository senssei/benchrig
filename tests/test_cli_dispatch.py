"""Phase 12.4 — split ``benchrig/cli.py`` into per-command modules.

After the package restructure:

- The implementation lives in ``benchrig.cli.run``, ``benchrig.cli.check``,
  ``benchrig.cli.compare``, ``benchrig.cli.report`` and ``benchrig.cli._common``.
- ``benchrig.cli.__init__`` is a thin dispatcher that re-exports every symbol tests
  currently import via ``from benchrig.cli import X`` or patch via
  ``patch("benchrig.cli.X", ...)`` so the existing test suite (``test_benchmark_cli``,
  ``test_cli_logging``, ``test_cli_report_flags``, ``test_cli_capacity_skip_notice``,
  ``test_packaging``, ``test_foundry_runtime``, ``test_warmup_protocol``,
  ``test_report_1to1``) keeps passing unchanged.
- ``benchrig --help`` is byte-identical (the argparse configuration is moved as-is).
- ``main()`` routes each top-level command to the right subcommand module's callee.

Imports of the new submodules happen inside each test body so a missing module
fails only that test (not the whole file's collection, which would otherwise be
classified as "not found" by ``scripts/sdlc_check.py --red``).
"""

import importlib
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

# Pinned help text from BEFORE the split (captured 2026-10-03 with the
# monolithic ``benchrig/cli.py``). Any drift in this string after the split means the
# argparse configuration was rewritten instead of moved verbatim — the spec says
# ``--help`` must be byte-identical.
HELP_SNAPSHOT = """\
usage: benchrig [-h] [--version] [--config CONFIG]
                [--scenarios-dir SCENARIOS_DIR]
                [--runtime {ollama,foundry,onnx-gpu,prism,all}] [--check]
                [--pull-recommended] [--models MODELS]
                [--suite {all,speed,coding,reasoning,polish,context}]
                [--runs RUNS] [--warmup-runs WARMUP_RUNS]
                [--output-dir OUTPUT_DIR] [--compare COMPARE]
                [--baseline BASELINE] [--pair PAIR] [--csv CSV]
                [--chart CHART] [--log-level {DEBUG,INFO,WARNING,ERROR}]

Local LLM Benchmark & Load Testing Rig (Ollama, MS Foundry, and Direct ONNX
GenAI CUDA)

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --config CONFIG       Config file (default: $BENCHRIG_CONFIG, ./config.yaml,
                        then the bundled default)
  --scenarios-dir SCENARIOS_DIR
                        Directory with scenario JSON files (default:
                        ./scenarios, then the bundled scenarios)
  --runtime {ollama,foundry,onnx-gpu,prism,all}
                        Inference runtime selection (ollama, foundry, onnx-
                        gpu, prism, all; default: from config or ollama)
  --check               Run environment diagnostics across runtimes and
                        accelerators
  --pull-recommended    Pull recommended models for selected runtime(s)
  --models MODELS       Models to benchmark (comma-separated, runtime-prefixed
                        e.g. 'foundry:phi-4', or 'installed')
  --suite {all,speed,coding,reasoning,polish,context}
                        Test suite selection (all, coding, reasoning, speed,
                        context, polish)
  --runs RUNS           Number of repetitions per test (default: 1)
  --warmup-runs WARMUP_RUNS
                        Number of warm-up iterations per scenario before
                        measurement (default: 1)
  --output-dir OUTPUT_DIR
                        Directory to save benchmark results
  --compare COMPARE     Path to benchmark JSON run to load, analyze, and
                        display 1:1 cross-engine comparison without running
                        models
  --baseline BASELINE   Path to cached benchmark JSON run providing Ollama
                        baseline so Ollama is never re-run
  --pair PAIR           1:1 model comparison pair ID from config.yaml (e.g.
                        'phi_mini', 'qwen_coder_7b')
  --csv CSV             Write a CSV of the run's scorecards to this path
                        (linked from the Markdown report)
  --chart CHART         Write a PNG chart (one bar per scorecard) to this
                        path; requires the [charts] extra (embedded in the
                        Markdown report)
  --log-level {DEBUG,INFO,WARNING,ERROR}
                        Structured-log level on stderr (env:
                        BENCHRIG_LOG_LEVEL). Default WARNING.
"""

# Every symbol a test in the suite imports or patches via ``benchrig.cli.X``. The
# dispatcher must re-export each; if any one is missing, an ImportError surfaces
# the moment a downstream test (e.g. ``test_cli_logging``) runs.
REEXPORTS = (
    # Public entry points
    "main",
    "build_parser",
    # Run path
    "run_benchmarks",
    "evaluate_model",
    "build_scorecards",
    # Check / pull-recommended path
    "run_system_check",
    "pull_recommended_models",
    "resolve_target_models",
    # Compare path
    "run_compare_mode",
    "load_baseline",
    "show_1to1_comparison",
    # Report path
    "save_outputs",
    # Shared helpers
    "_resolve_log_level",
    "_warn_slow_provider",
    "_suite_skip_notice",
    "_bootstrap_cuda_env",
    # Cross-module callables other tests reach into
    "display_system_banner",
    "display_leaderboard",
    "display_token_savings",
    "get_system_specs",
    # Module-level references tests patch directly
    "console",
    "glob",
    "logging",
    "time",
)


class SubmoduleStructureTests(unittest.TestCase):
    """Each subcommand module owns its slice; the dispatcher re-exports them."""

    def test_run_submodule_owns_run_benchmarks(self):
        from benchrig.cli.run import run_benchmarks

        self.assertTrue(callable(run_benchmarks))

    def test_run_submodule_owns_evaluate_model(self):
        from benchrig.cli.run import evaluate_model

        self.assertTrue(callable(evaluate_model))

    def test_check_submodule_owns_run_system_check(self):
        from benchrig.cli.check import run_system_check

        self.assertTrue(callable(run_system_check))

    def test_check_submodule_owns_pull_recommended_models(self):
        from benchrig.cli.check import pull_recommended_models

        self.assertTrue(callable(pull_recommended_models))

    def test_compare_submodule_owns_run_compare_mode(self):
        from benchrig.cli.compare import run_compare_mode

        self.assertTrue(callable(run_compare_mode))

    def test_compare_submodule_owns_load_baseline(self):
        from benchrig.cli.compare import load_baseline

        self.assertTrue(callable(load_baseline))

    def test_report_submodule_owns_save_outputs(self):
        from benchrig.cli.report import save_outputs

        self.assertTrue(callable(save_outputs))

    def test_common_submodule_owns_resolve_log_level(self):
        from benchrig.cli._common import _resolve_log_level

        self.assertTrue(callable(_resolve_log_level))

    def test_common_submodule_owns_load_config(self):
        from benchrig.cli._common import load_config

        self.assertTrue(callable(load_config))

    def test_common_submodule_owns_suites_constant(self):
        from benchrig.cli._common import SUITES

        # Suite name -> (scenario file, runner method, progress message).
        self.assertIn("coding", SUITES)
        self.assertIn("reasoning", SUITES)
        self.assertIn("speed", SUITES)


class BackwardCompatReexportsTests(unittest.TestCase):
    """``from benchrig.cli import X`` must keep working for every symbol tests pin today."""

    def test_main_entry_point_resolves(self):
        cli = importlib.import_module("benchrig.cli")
        self.assertTrue(callable(cli.main))

    def test_reexports_resolve(self):
        cli = importlib.import_module("benchrig.cli")
        missing = [name for name in REEXPORTS if not hasattr(cli, name)]
        self.assertEqual(missing, [], f"missing re-exports on benchrig.cli: {missing}")

    def test_console_and_glob_are_the_module_objects_tests_patch(self):
        # tests/test_packaging.py:68 patches ``benchrig.cli.glob.glob``; if
        # ``benchrig.cli.glob`` is not the stdlib ``glob`` module, that patch
        # would silently target the wrong object.
        import glob as _glob
        import logging as _logging
        import time as _time

        cli = importlib.import_module("benchrig.cli")
        self.assertIs(cli.glob, _glob)
        self.assertIs(cli.logging, _logging)
        self.assertIs(cli.time, _time)


class HelpTextTests(unittest.TestCase):
    """``benchrig --help`` is byte-identical to the pre-split snapshot."""

    def test_help_is_byte_identical_to_pre_split_snapshot(self):
        from benchrig.cli import build_parser

        self.assertEqual(build_parser().format_help(), HELP_SNAPSHOT)


class DispatchRoutingTests(unittest.TestCase):
    """``main()`` routes each top-level command to the right subcommand callee."""

    def _enter(self, patches):
        """Enter all patches at once; return (stack, mocks) so the caller can assert on them."""
        stack = ExitStack()
        self.addCleanup(stack.close)
        mocks = [stack.enter_context(p) for p in patches]
        return stack, mocks

    def _bootstrap_mocks(self):
        # Bypass the heavy startup: CUDA env, config, runtime clients, structured-log setup.
        return [
            patch("benchrig.cli._bootstrap_cuda_env"),
            patch("benchrig.cli.load_config", return_value={}),
            patch("benchrig.cli.create_runtime_client", return_value=MagicMock()),
            patch("benchrig.cli.setup_logging"),
        ]

    def test_main_routes_check_to_run_system_check(self):
        patches = self._bootstrap_mocks()
        mock_check = patch("benchrig.cli.run_system_check")
        stack, _ = self._enter(patches)
        mock_obj = stack.enter_context(mock_check)
        with patch("sys.argv", ["benchrig", "--check"]):
            from benchrig.cli import main

            main()
        mock_obj.assert_called_once()

    def test_main_routes_pull_recommended(self):
        patches = self._bootstrap_mocks()
        mock_pull = patch("benchrig.cli.pull_recommended_models")
        stack, _ = self._enter(patches)
        mock_obj = stack.enter_context(mock_pull)
        with patch("sys.argv", ["benchrig", "--pull-recommended"]):
            from benchrig.cli import main

            main()
        mock_obj.assert_called_once()

    def test_main_routes_compare_to_run_compare_mode(self):
        patches = self._bootstrap_mocks()
        mock_compare = patch("benchrig.cli.run_compare_mode")
        stack, _ = self._enter(patches)
        mock_obj = stack.enter_context(mock_compare)
        with patch("sys.argv", ["benchrig", "--compare", "/tmp/r.json"]):
            from benchrig.cli import main

            main()
        mock_obj.assert_called_once()

    def test_main_routes_benchmark_to_run_benchmarks(self):
        patches = self._bootstrap_mocks()
        mock_run = patch("benchrig.cli.run_benchmarks")
        stack, _ = self._enter(patches)
        mock_obj = stack.enter_context(mock_run)
        with patch("sys.argv", ["benchrig"]):
            from benchrig.cli import main

            main()
        mock_obj.assert_called_once()


class EntryPointContractTests(unittest.TestCase):
    """The console-script entry point (``benchrig = "benchrig.cli:main"``) still resolves."""

    def test_entry_point_string_is_benchrig_cli_main(self):
        # tests/test_packaging.py:22 also pins this; listed here so the package-restructure
        # contract is self-documenting in this test file.
        from benchrig import __version__

        cli = importlib.import_module("benchrig.cli")
        self.assertTrue(callable(cli.main))
        # The pyproject entry point is string-equal to "benchrig.cli:main".
        self.assertEqual(cli.__name__, "benchrig.cli")
        self.assertTrue(hasattr(cli, "__version__") or cli.__name__ == "benchrig.cli")
        del __version__  # silence unused-import lint


class PatchCompatibilityTests(unittest.TestCase):
    def test_source_command_patch_intercepts_main(self):
        from benchrig import cli

        for argv, target in (
            (["--check"], "check.run_system_check"),
            (["--compare", "/tmp/run.json"], "compare.run_compare_mode"),
            (["--pull-recommended"], "check.pull_recommended_models"),
            ([], "run.run_benchmarks"),
        ):
            with self.subTest(target=target), ExitStack() as stack:
                stack.enter_context(patch("benchrig.cli._bootstrap_cuda_env"))
                stack.enter_context(patch("benchrig.cli.load_config", return_value={}))
                stack.enter_context(patch("benchrig.cli.create_runtime_client", return_value=MagicMock()))
                stack.enter_context(patch("benchrig.cli.setup_logging"))
                command = stack.enter_context(patch("benchrig.cli." + target))
                stack.enter_context(patch("sys.argv", ["benchrig", *argv]))
                cli.main()
                command.assert_called_once()

    def test_package_patch_intercepts_check_hardware_discovery(self):
        from benchrig import cli

        with patch("benchrig.cli.get_system_specs", return_value={}) as specs, patch("benchrig.cli.console"):
            cli.run_system_check({})
        specs.assert_called_once()

    def test_benchmark_routes_csv_and_chart_to_report_module(self):
        from benchrig import cli

        args = cli.build_parser().parse_args(["--csv", "/tmp/out.csv", "--chart", "/tmp/out.png"])
        with ExitStack() as stack:
            stack.enter_context(patch("benchrig.cli.resolve_target_models", return_value=[("ollama", "test")]))
            stack.enter_context(patch("benchrig.cli.get_system_specs", return_value={}))
            stack.enter_context(patch("benchrig.cli.console"))
            stack.enter_context(patch("benchrig.cli.display_system_banner"))
            stack.enter_context(patch("benchrig.cli.display_leaderboard"))
            stack.enter_context(patch("benchrig.cli.display_token_savings"))
            stack.enter_context(patch("benchrig.cli.evaluate_model", return_value=[]))
            stack.enter_context(patch("benchrig.cli.build_scorecards", return_value=[{"runtime": "ollama"}]))
            outputs = stack.enter_context(patch("benchrig.cli.report.save_outputs", return_value=("a.md", "a.json")))
            progress = stack.enter_context(patch("benchrig.cli.run.Progress"))
            cli.run_benchmarks(args, {}, {"ollama": MagicMock()}, "ollama")
            self.assertIs(progress.call_args.kwargs["console"], cli.console)
        outputs.assert_called_once()
        self.assertEqual(outputs.call_args.kwargs["csv_path"], "/tmp/out.csv")
        self.assertEqual(outputs.call_args.kwargs["chart_path"], "/tmp/out.png")

    def test_package_console_patch_reaches_report_output(self):
        import tempfile
        from pathlib import Path

        from benchrig import cli

        with tempfile.TemporaryDirectory() as tmp, patch("benchrig.cli.console") as console:
            cli.save_outputs(tmp, {}, [], [], 0, csv_path=str(Path(tmp) / "out.csv"))
        console.print.assert_called_once()
