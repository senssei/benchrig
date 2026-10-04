"""``benchrig`` console-script entry point and thin dispatcher.

Phase 12.4: the monolithic ``benchrig/cli.py`` (1028 lines) was split into per-command modules
in the ``benchrig/cli/`` package. This ``__init__`` keeps the public surface stable: every
symbol tests previously imported via ``from benchrig.cli import X`` or patched via
``patch("benchrig.cli.X", ...)`` is re-exported here, so the existing test suite
(``test_benchmark_cli``, ``test_cli_logging``, ``test_cli_report_flags``,
``test_cli_capacity_skip_notice``, ``test_packaging``, ``test_foundry_runtime``,
``test_warmup_protocol``, ``test_report_1to1``) keeps working.

``main()`` dispatches by argparse flag:

- ``--compare PATH`` → ``benchrig.cli.compare.run_compare_mode``
- ``--check`` → ``benchrig.cli.check.run_system_check``
- ``--pull-recommended`` → ``benchrig.cli.check.pull_recommended_models``
- (default) → ``benchrig.cli.run.run_benchmarks``

The ``argparse.ArgumentParser`` configuration is byte-identical to the pre-split version
(pinned by ``tests/test_cli_dispatch.py::HelpTextTests``); ``benchrig --help`` renders the
same.
"""

from __future__ import annotations

import argparse

# Built-in modules and ``console`` are intentionally re-exported on this package's namespace
# because tests patch them via ``benchrig.cli.X``. (e.g. ``test_packaging.py:68`` patches
# ``benchrig.cli.glob.glob``; ``test_report_1to1.py:147`` patches ``benchrig.cli.console``.)
import glob  # noqa: F401  — re-exported for tests that patch ``benchrig.cli.glob``
import logging  # noqa: F401  — re-exported for tests that patch ``benchrig.cli.logging``
import os
import sys
import time  # noqa: F401  — re-exported for tests that patch ``benchrig.cli.time``
import uuid

from benchrig import __version__
from benchrig.cli._common import (
    _VALID_LOG_LEVELS,
    BUNDLED_DATA,
    CONFIG_ENV_VAR,
    CONFIG_FILENAME,
    RUNTIME_CHOICES,
    SCENARIOS_DIR,  # noqa: F401  — re-exported; some docs/tests reference the bundled default
    SUITES,
    _bootstrap_cuda_env,
    _resolve_log_level,
    _suite_skip_notice,
    _total_scenario_steps,
    _warn_slow_provider,
    console,
    load_config,
    resolve_config_path,
    resolve_pair_targets,
    resolve_scenarios_dir,
    resolve_target_models,
)
from benchrig.cli.check import _check_prism, pull_recommended_models, run_system_check
from benchrig.cli.compare import (
    load_baseline,
    records_for,
    run_compare_mode,
    select_pair,
    show_1to1_comparison,
)
from benchrig.cli.report import save_outputs
from benchrig.cli.run import build_scorecards, evaluate_model, run_benchmarks
from benchrig.core.client import BaseRuntimeClient, create_runtime_client
from benchrig.core.hardware import get_system_specs
from benchrig.core.logging import setup_logging

# Display helpers used by ``run_benchmarks`` and tests; re-exported so
# ``patch("benchrig.cli.display_system_banner")`` still works.
from benchrig.reporting.display import (  # noqa: F401  — re-exported on benchrig.cli
    display_leaderboard,
    display_system_banner,
    display_token_savings,
)

# Retain the original exports so package overrides take precedence, while changes
# to the owning module are also visible to the dispatcher and command helpers.
_ORIGINAL_EXPORTS = {name: value for name, value in globals().copy().items() if callable(value)}
_ORIGINAL_EXPORTS["console"] = console


def _lookup(name):
    original = _ORIGINAL_EXPORTS[name]
    exported = globals()[name]
    if exported is not original:
        return exported
    owner = sys.modules.get(getattr(original, "__module__", ""))
    return getattr(owner, name, original)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchrig",
        description="Local LLM Benchmark & Load Testing Rig (Ollama, MS Foundry, and Direct ONNX GenAI CUDA)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config",
        default=None,
        help=f"Config file (default: ${CONFIG_ENV_VAR}, ./{CONFIG_FILENAME}, then the bundled default)",
    )
    parser.add_argument(
        "--scenarios-dir",
        default=None,
        help="Directory with scenario JSON files (default: ./scenarios, then the bundled scenarios)",
    )
    parser.add_argument(
        "--runtime",
        default=None,
        choices=RUNTIME_CHOICES,
        help="Inference runtime selection (ollama, foundry, onnx-gpu, prism, all; default: from config or ollama)",
    )
    parser.add_argument(
        "--check", action="store_true", help="Run environment diagnostics across runtimes and accelerators"
    )
    parser.add_argument(
        "--pull-recommended", action="store_true", help="Pull recommended models for selected runtime(s)"
    )
    parser.add_argument(
        "--models",
        default="installed",
        help="Models to benchmark (comma-separated, runtime-prefixed e.g. 'foundry:phi-4', or 'installed')",
    )
    parser.add_argument(
        "--suite",
        default="all",
        choices=["all", *SUITES],
        help="Test suite selection (all, coding, reasoning, speed, context, polish, tool_use)",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions per test (default: 1)")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warm-up iterations per scenario before measurement (default: 1)",
    )
    parser.add_argument("--output-dir", default="results", help="Directory to save benchmark results")
    parser.add_argument(
        "--compare",
        default=None,
        help="Path to benchmark JSON run to load, analyze, and display 1:1 cross-engine comparison without running models",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to cached benchmark JSON run providing Ollama baseline so Ollama is never re-run",
    )
    parser.add_argument(
        "--pair",
        default=None,
        help="1:1 model comparison pair ID from config.yaml (e.g. 'phi_mini', 'qwen_coder_7b')",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Write a CSV of the run's scorecards to this path (linked from the Markdown report)",
    )
    parser.add_argument(
        "--chart",
        default=None,
        help="Write a PNG chart (one bar per scorecard) to this path; requires the [charts] extra "
        "(embedded in the Markdown report)",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=list(_VALID_LOG_LEVELS),
        default=os.environ.get("BENCHRIG_LOG_LEVEL", "WARNING").upper(),
        help="Structured-log level on stderr (env: BENCHRIG_LOG_LEVEL). Default WARNING.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Stable per-invocation run_id; threaded through every JSON log record so an operator
    # downstream can filter a single run out of a noisy shared log target.
    args.run_id = uuid.uuid4().hex
    args.argv = sys.argv[1:]

    # Resolve --log-level and configure the JSON stderr handler. Done before any branch so
    # `compare` / `check` paths also produce structured logs (only `run_benchmarks` emits
    # `run.started` / `run.completed`, per spec.md §Phase 11).
    log_level = _resolve_log_level(args.log_level)
    setup_logging(level=log_level, run_id=args.run_id)

    _bootstrap_cuda_env()
    config = load_config(args.config)

    effective_suite = args.suite
    if args.pair and effective_suite == "all":
        pair = next(
            (p for p in config.get("model_pairs_1to1", []) if args.pair in (p.get("id"), p.get("name"))),
            {},
        )
        effective_suite = pair.get("recommended_suite") or effective_suite
    if effective_suite == "tool_use" and not (args.check or args.pull_recommended or args.compare):
        from benchrig.cli._common import load_scenario_file
        from benchrig.core.tool_use import validate_scenarios

        path = os.path.join(resolve_scenarios_dir(args.scenarios_dir), "tool_use.json")
        try:
            validate_scenarios(load_scenario_file(path), path)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))

    clients: dict[str, BaseRuntimeClient] = {
        name: create_runtime_client(name, config) for name in ("ollama", "foundry", "onnx-gpu", "prism")
    }

    if args.compare:
        _lookup("run_compare_mode")(args.compare, args.output_dir, clients)
        return

    selected_runtime = args.runtime or config.get("benchmark", {}).get("default_runtime", "ollama")

    if args.check:
        _lookup("run_system_check")(clients)
        return
    if args.pull_recommended:
        _lookup("pull_recommended_models")(clients, config, target_runtime=selected_runtime)
        return

    _lookup("run_benchmarks")(args, config, clients, selected_runtime)


if __name__ == "__main__":
    main()


__all__ = [
    # Entry points
    "main",
    "build_parser",
    # Constants
    "BUNDLED_DATA",
    "CONFIG_ENV_VAR",
    "CONFIG_FILENAME",
    "RUNTIME_CHOICES",
    "SCENARIOS_DIR",
    "SUITES",
    "_VALID_LOG_LEVELS",
    # Command modules re-exported for the public surface
    "run_benchmarks",
    "run_system_check",
    "pull_recommended_models",
    "run_compare_mode",
    "resolve_target_models",
    "resolve_config_path",
    "resolve_pair_targets",
    "resolve_scenarios_dir",
    "_resolve_log_level",
    "_warn_slow_provider",
    "_suite_skip_notice",
    "_total_scenario_steps",
    "_bootstrap_cuda_env",
    "_check_prism",
    "evaluate_model",
    "build_scorecards",
    "save_outputs",
    "show_1to1_comparison",
    "load_baseline",
    "select_pair",
    "records_for",
    "display_system_banner",
    "display_leaderboard",
    "display_token_savings",
    "get_system_specs",
    "create_runtime_client",
    "setup_logging",
    "console",
    "glob",
    "logging",
    "time",
]
