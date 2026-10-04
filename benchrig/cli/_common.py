"""Shared constants and cross-command helpers for ``benchrig.cli`` submodules.

Phase 12.4: previously this lived at the top of the monolithic ``benchrig/cli.py``. The package
restructure keeps cross-cutting helpers here (config loading, log-level validation, CUDA
bootstrap, target resolution) so the per-command modules can import them without circular
dependencies.
"""

from __future__ import annotations

import argparse  # noqa: F401  — re-exported via benchrig.cli for tests that patch argparse behaviour
import glob
import json
import os
import re
import site
import sys
from importlib import resources

import yaml
from rich.console import Console

import benchrig.cli as _cli_pkg
from benchrig.core.client import BaseRuntimeClient, create_runtime_client
from benchrig.core.hardware import get_system_specs
from benchrig.core.logging import setup_logging
from benchrig.core.runtimes import warning_for_provider
from benchrig.reporting.display import console

CONFIG_FILENAME = "config.yaml"
CONFIG_ENV_VAR = "BENCHRIG_CONFIG"
# Bundled defaults ship inside the wheel; `./config.yaml` and `./scenarios` in the working directory override them.
BUNDLED_DATA = resources.files("benchrig") / "data"
SCENARIOS_DIR = str(BUNDLED_DATA / "scenarios")

# Suite name -> (scenario file, BenchmarkRunner method, progress message). Order is execution order.
SUITES: dict[str, tuple[str, str, str]] = {
    "speed": ("speed.json", "run_speed_suite", "Running speed & throughput tests..."),
    "coding": ("coding.json", "run_coding_suite", "Running coding tests (isolated unit test subprocess)..."),
    "reasoning": ("reasoning.json", "run_reasoning_suite", "Running reasoning & logic tests..."),
    "polish": ("polish.json", "run_polish_suite", "Running Polish multilingual NLP tests..."),
    "context": ("context_scaling.json", "run_context_suite", "Running context scaling suite (512 - 8k tokens)..."),
}

RUNTIME_CHOICES = ["ollama", "foundry", "onnx-gpu", "prism", "all"]

# Accepted `runtime:model` prefixes, normalized to canonical runtime names.
RUNTIME_PREFIXES = {
    "ollama": "ollama",
    "foundry": "foundry",
    "ms-foundry": "foundry",
    "onnx-gpu": "onnx-gpu",
    "onnx": "onnx-gpu",
    "prism": "prism",
    "prism-local": "prism",
}

# Accepted log levels for the `--log-level` flag and `BENCHRIG_LOG_LEVEL` env var. The default
# level (WARNING) keeps the JSON stderr silent during normal runs; operators set the level to
# DEBUG or INFO via the flag/env when they need a diagnostic trail.
_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

_GENERIC_CPU_SUFFIX_RE = re.compile(r"-generic-cpu(?::\d+)?$", re.IGNORECASE)


def _resolve_log_level(value: str | None) -> str:
    """Validate and normalise a CLI-supplied log level. Exits 2 on an invalid value.

    A non-empty whitespace-only string is treated as no-value (falls back to WARNING) so a
    stray `BENCHRIG_LOG_LEVEL=` in the environment does not crash the CLI.
    """
    candidate = (value or "").strip().upper() or "WARNING"
    if candidate not in _VALID_LOG_LEVELS:
        Console(stderr=True).print(
            f"[bold red]invalid --log-level:[/] [yellow]{candidate!r}[/]\n"
            f"  valid values: {', '.join(_VALID_LOG_LEVELS)}"
        )
        sys.exit(2)
    return candidate


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def resolve_config_path(explicit: str | None = None) -> str | None:
    """Pick the config file: --config, then $BENCHRIG_CONFIG, then ./config.yaml, then the bundled default."""
    for candidate in (explicit, os.environ.get(CONFIG_ENV_VAR), CONFIG_FILENAME):
        if candidate and os.path.isfile(candidate):
            return candidate
    bundled = BUNDLED_DATA / CONFIG_FILENAME
    return str(bundled) if bundled.is_file() else None


def load_config(config_path: str | None = None) -> dict:
    """Load the configuration yaml file (see `resolve_config_path` for the lookup order)."""
    path = _cli_pkg._lookup("resolve_config_path")(config_path)
    if path:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def resolve_scenarios_dir(explicit: str | None = None) -> str:
    """Pick the scenarios directory: --scenarios-dir, then ./scenarios, then the bundled scenarios."""
    for candidate in (explicit, "scenarios"):
        if candidate and os.path.isdir(candidate):
            return candidate
    return SCENARIOS_DIR


def load_scenario_file(filepath: str) -> list[dict]:
    """Load scenarios from JSON file."""
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    return []


def _warn_slow_provider(targets: list[tuple[str, str]], specs: dict) -> None:
    """Print the slow-provider warning once per run when a target uses generic-cpu on a CUDA host.

    Targets are ``(runtime, model)`` pairs. The provider is read from the model alias suffix
    (``-generic-cpu``); the host GPU type comes from ``specs["gpu_type"]``. The warning is informational:
    the run continues and exits 0 (spec.md I4).
    """
    host_gpu_type = specs.get("gpu_type", "")
    warned = False
    for _runtime, model in targets:
        if _GENERIC_CPU_SUFFIX_RE.search(model):
            msg = warning_for_provider(provider="generic-cpu", host_gpu_type=host_gpu_type)
            if msg and not warned:
                _cli_pkg._lookup("console").print(f"[bold yellow]⚠ {msg}[/]\n")
                warned = True


def _suite_skip_notice(runner, suite_results: list[dict], requested_scenarios: list[dict]) -> str | None:
    """The reason to show the operator when a suite produced no results despite having scenarios to run.

    `None` when the suite actually ran (or had nothing to run in the first place, e.g. an empty scenario
    file) — only a *silent* empty result, caused by `BenchmarkRunner`'s capacity short-circuit (plan.md
    item 4.6), gets a message. Without this, `--suite all` prints "Running coding tests..." followed by
    nothing, with no indication that the model simply does not fit in available VRAM.
    """
    if suite_results or not requested_scenarios:
        return None
    return runner.capacity_exhausted_reason


def _total_scenario_steps(scenarios: dict[str, list[dict]], runs: int, num_targets: int) -> int:
    """Units for the overall progress bar: one per (target x run x scenario) (plan.md Phase 8, item 8.2)."""
    return sum(len(v) for v in scenarios.values()) * runs * num_targets


def load_json_or_exit(path: str, description: str) -> dict:
    """Load a JSON file, or print an error and exit if it does not exist."""
    if not os.path.exists(path):
        _cli_pkg._lookup("console").print(f"[bold red]{description} not found:[/] [yellow]{path}[/]")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_ollama(entry: dict) -> bool:
    """True if a scorecard or result record was produced by the Ollama runtime."""
    return "ollama" in str(entry.get("runtime", "")).lower()


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def _installed_names(client: BaseRuntimeClient) -> list[str]:
    return [m.get("name") for m in client.list_installed_models()]


def resolve_target_models(
    raw_models_arg: str,
    selected_runtime: str,
    clients: dict[str, BaseRuntimeClient],
) -> list[tuple[str, str]]:
    """
    Resolve model target list into tuples of (runtime_name, model_name).
    Supports:
    - 'installed' / 'all'
    - 'qwen2.5-coder:7b' (inherits selected_runtime)
    - 'ollama:qwen2.5-coder:7b,foundry:phi-4' (explicit prefixes)
    """
    targets: list[tuple[str, str]] = []

    if raw_models_arg in ("installed", "all"):
        runtimes = ["ollama", "foundry", "onnx-gpu", "prism"] if selected_runtime == "all" else [selected_runtime]
        for rt in runtimes:
            client = clients.get(rt)
            if client and client.is_reachable():
                names = _installed_names(client)
                if rt == "prism":
                    # Prism also proxies Ollama models as `ollama:<name>`; those are benchmarked natively via the ollama
                    # runtime (ask for them explicitly with `prism:ollama:<name>` to measure the proxy).
                    names = [n for n in names if not str(n).lower().startswith("ollama:")]
                targets.extend((rt, name) for name in names if name)
        return targets

    for item in (i.strip() for i in raw_models_arg.split(",")):
        if not item:
            continue

        prefix, _, model_name = item.partition(":")
        if model_name and prefix in RUNTIME_PREFIXES:
            targets.append((RUNTIME_PREFIXES[prefix], model_name))
        elif selected_runtime == "all":
            # Unprefixed model with runtime 'all': use every runtime that already has it installed.
            found = []
            for rt in ("onnx-gpu", "prism", "foundry", "ollama"):
                client = clients.get(rt)
                if client and client.is_reachable():
                    if any(item == inst or item in str(inst) for inst in _installed_names(client)):
                        found.append((rt, item))
            targets.extend(found or [("ollama", item)])
        else:
            targets.append((selected_runtime, item))

    return targets


def resolve_pair_targets(
    config: dict, pair_id: str, selected_runtime: str, has_baseline: bool
) -> tuple[list[tuple[str, str]], str | None]:
    """
    Resolve a `--pair` id from config into (targets, recommended_suite).

    With a cached Ollama baseline only the other engine is run.
    """
    pairs = config.get("model_pairs_1to1", [])
    pair = next((p for p in pairs if pair_id in (p.get("id"), p.get("name"))), None)
    if not pair:
        _cli_pkg._lookup("console").print(
            f"[bold red]Unknown 1:1 pair ID:[/] {pair_id}. Available: {[p.get('id') for p in pairs]}"
        )
        sys.exit(1)

    _cli_pkg._lookup("console").print(f"[bold cyan]Selected 1:1 Pair:[/] [bold]{pair.get('name')}[/]")
    if has_baseline:
        if selected_runtime == "onnx-gpu":
            targets = [("onnx-gpu", pair.get("onnx", pair["foundry"]))]
        elif selected_runtime == "prism":
            targets = [("prism", pair.get("prism", pair.get("onnx", pair["foundry"])))]
        else:
            targets = [("foundry", pair["foundry"])]
    else:
        targets = [("ollama", pair["ollama"]), ("foundry", pair["foundry"])]
    return targets, pair.get("recommended_suite")


# ---------------------------------------------------------------------------
# Process setup
# ---------------------------------------------------------------------------


def _bootstrap_cuda_env() -> None:
    """Ensure CUDA dynamic linker paths and library symlinks are configured before ONNX runtime initialization."""
    if os.environ.get("_ONNX_CUDA_BOOTSTRAPPED") == "1":
        return

    site_dirs = [*site.getsitepackages(), site.getusersitepackages()]
    venv_nvidia = [d for sp in site_dirs for d in glob.glob(os.path.join(sp, "nvidia", "*", "lib"))]
    ollama_cuda13 = "/usr/local/lib/ollama/cuda_v13"

    # Some wheels ship libcufft.so.11 while onnxruntime expects .so.12
    for d in venv_nvidia:
        if "cufft" in d:
            so11 = os.path.join(d, "libcufft.so.11")
            so12 = os.path.join(d, "libcufft.so.12")
            if os.path.exists(so11) and not os.path.exists(so12):
                try:
                    os.symlink("libcufft.so.11", so12)
                except OSError:
                    pass

    existing_dirs = [d for d in [ollama_cuda13, *venv_nvidia] if os.path.isdir(d)]
    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if not all(d in cur_ld for d in existing_dirs):
        os.environ["LD_LIBRARY_PATH"] = ":".join(existing_dirs + ([cur_ld] if cur_ld else []))
        os.environ["_ONNX_CUDA_BOOTSTRAPPED"] = "1"
        try:
            # LD_LIBRARY_PATH is only read at process start, so re-exec ourselves.
            os.execv(sys.executable, [sys.executable, "-m", "benchrig", *sys.argv[1:]])
        except OSError:
            pass


__all__ = [
    "CONFIG_FILENAME",
    "CONFIG_ENV_VAR",
    "BUNDLED_DATA",
    "SCENARIOS_DIR",
    "SUITES",
    "RUNTIME_CHOICES",
    "RUNTIME_PREFIXES",
    "_VALID_LOG_LEVELS",
    "_resolve_log_level",
    "resolve_config_path",
    "load_config",
    "resolve_scenarios_dir",
    "load_scenario_file",
    "_warn_slow_provider",
    "_suite_skip_notice",
    "_total_scenario_steps",
    "load_json_or_exit",
    "is_ollama",
    "_installed_names",
    "resolve_target_models",
    "resolve_pair_targets",
    "_bootstrap_cuda_env",
    "create_runtime_client",
    "get_system_specs",
    "setup_logging",
    "console",
    "warning_for_provider",
]
