#!/usr/bin/env python3
"""
Local LLM Benchmark & Load Testing Rig (macOS Apple Silicon Metal / Linux NVIDIA CUDA).
Profiles and ranks local models across Ollama and Microsoft Foundry Server Runtimes
by coding quality, reasoning, generation speed, and memory efficiency.
"""

import argparse
import glob
import json
import os
import re
import site
import sys
import time
from datetime import datetime
from importlib import resources
from typing import Any

import yaml

from benchrig import __version__
from benchrig.core.client import BaseRuntimeClient, create_runtime_client
from benchrig.core.hardware import get_system_specs
from benchrig.core.runner import BenchmarkRunner
from benchrig.core.runtimes import runtime_label, warning_for_provider
from benchrig.reporting import write_scorecards_chart, write_scorecards_csv
from benchrig.reporting.display import (
    console,
    display_1to1_comparison,
    display_leaderboard,
    display_scenario_result,
    display_system_banner,
    display_token_savings,
)
from benchrig.reporting.markdown import (
    generate_1to1_comparison_report,
    generate_markdown_report,
)

CONFIG_FILENAME = "config.yaml"
CONFIG_ENV_VAR = "BENCHRIG_CONFIG"
# Bundled defaults ship inside the wheel; `./config.yaml` and `./scenarios` in the working directory override them.
BUNDLED_DATA = resources.files("benchrig") / "data"
SCENARIOS_DIR = str(BUNDLED_DATA / "scenarios")

# Suite name -> (scenario file, BenchmarkRunner method, progress message). Order is execution order.
SUITES: dict[str, tuple[str, str, str]] = {
    "speed": ("speed.json", "run_speed_suite", "Running speed & throughput tests..."),
    "coding": ("coding.json", "run_coding_suite", "Running coding tests (automated unit test sandbox)..."),
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


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load the configuration yaml file (see `resolve_config_path` for the lookup order)."""
    path = resolve_config_path(config_path)
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


def load_scenario_file(filepath: str) -> list[dict[str, Any]]:
    """Load scenarios from JSON file."""
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    return []


_GENERIC_CPU_SUFFIX_RE = re.compile(r"-generic-cpu(?::\d+)?$", re.IGNORECASE)


def _warn_slow_provider(targets: list[tuple[str, str]], specs: dict[str, Any]) -> None:
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
                console.print(f"[bold yellow]⚠ {msg}[/]\n")
                warned = True


def _suite_skip_notice(
    runner: BenchmarkRunner, suite_results: list[dict[str, Any]], requested_scenarios: list[dict[str, Any]]
) -> str | None:
    """The reason to show the operator when a suite produced no results despite having scenarios to run.

    `None` when the suite actually ran (or had nothing to run in the first place, e.g. an empty scenario
    file) — only a *silent* empty result, caused by `BenchmarkRunner`'s capacity short-circuit (plan.md
    item 4.6), gets a message. Without this, `--suite all` prints "Running coding tests..." followed by
    nothing, with no indication that the model simply does not fit in available VRAM.
    """
    if suite_results or not requested_scenarios:
        return None
    return runner.capacity_exhausted_reason


def load_json_or_exit(path: str, description: str) -> dict[str, Any]:
    """Load a JSON file, or print an error and exit if it does not exist."""
    if not os.path.exists(path):
        console.print(f"[bold red]{description} not found:[/] [yellow]{path}[/]")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_ollama(entry: dict[str, Any]) -> bool:
    """True if a scorecard or result record was produced by the Ollama runtime."""
    return "ollama" in str(entry.get("runtime", "")).lower()


# ---------------------------------------------------------------------------
# --check: environment diagnostics
# ---------------------------------------------------------------------------


def _check_ollama(client: BaseRuntimeClient | None) -> None:
    if client and client.is_reachable():
        console.print(
            f"  [green]✔ Ollama REST API:[/] Available (Version: {client.get_version()}, llama.cpp) at {client.base_url}"
        )
        installed = client.list_installed_models()
        console.print(f"     Installed Ollama models ({len(installed)}):")
        for m in installed:
            size_gb = m.get("size", 0) / (1024**3)
            details = m.get("details", {})
            console.print(
                f"     • [cyan]{m.get('name', ''):<24}[/] "
                f"({size_gb:.1f} GB, {details.get('parameter_size', 'N/A')}, {details.get('quantization_level', 'N/A')})"
            )
    else:
        url = client.base_url if client else "http://localhost:11434"
        console.print(f"  [yellow]⚠ Ollama REST API:[/] Unreachable at {url}")


def _check_foundry(client: BaseRuntimeClient | None) -> None:
    if client and client.is_reachable():
        console.print(
            f"  [green]✔ MS Foundry Server REST API:[/] Available "
            f"(Version: {client.get_version()}, ONNX Runtime GenAI) at {client.base_url}"
        )
        installed = client.list_installed_models()
        console.print(f"     Available Foundry models ({len(installed)}):")
        for m in installed:
            details = m.get("details", {})
            console.print(
                f"     • [blue]{m.get('name', ''):<24}[/] "
                f"({details.get('parameter_size', 'ONNX')}, {details.get('quantization_level', 'ONNX')})"
            )
    else:
        url = client.base_url if client else "http://localhost:5272/v1"
        console.print(f"  [yellow]⚠ MS Foundry Server REST API:[/] Unreachable at {url}")


def _check_prism(client: BaseRuntimeClient | None) -> None:
    if client and client.is_reachable():
        models = client.list_installed_models()
        console.print(f"  [green]✔ Prism Server REST API:[/] Available ({client.get_version()}) at {client.base_url}")
        console.print(f"     Available Prism models ({len(models)}):")
        for m in models:
            console.print(f"     • [green]{m.get('name', ''):<44}[/] ({m.get('details', {}).get('engine', 'N/A')})")
    else:
        url = client.base_url if client else "http://127.0.0.1:5272/v1"
        console.print(f"  [yellow]ℹ Prism Server:[/] Not running at {url} (pip install prism-local && prism serve)")


def _check_onnx(client: BaseRuntimeClient | None) -> None:
    if client and client.is_reachable():
        cuda_ok = getattr(client, "is_cuda_available", lambda: False)()
        cuda_status = "[bold green]CUDA Enabled[/]" if cuda_ok else "[yellow]CPU[/]"
        console.print(
            f"  [green]✔ ONNX GenAI Direct Engine:[/] Available (Version: {client.get_version()}, {cuda_status})"
        )
        installed = client.list_installed_models()
        console.print(f"     Available Direct ONNX Models ({len(installed)}):")
        for m in installed:
            quant = m.get("details", {}).get("quantization_level", "ONNX")
            console.print(f"     • [magenta]{m.get('name', ''):<28}[/] ({quant})")
    else:
        console.print('  [yellow]ℹ ONNX GenAI Direct Engine:[/] Not installed (pip install "benchrig[onnx-gpu]")')


def _check_accelerator(specs: dict[str, Any]) -> None:
    gpu_type = specs.get("gpu_type")
    gpu_name = specs.get("gpu_name")
    if gpu_type == "apple_silicon":
        console.print(
            f"\n  [green]✔ Apple Silicon Metal Accelerator:[/] Detected {gpu_name} "
            f"({specs.get('ram_total_gb')} GB Unified Memory)"
        )
    elif gpu_type == "nvidia":
        console.print(
            f"\n  [green]✔ NVIDIA GPU Access:[/] Detected {gpu_name} ({specs.get('gpu_vram_total_mb')} MB VRAM)"
        )
    elif gpu_name != "None":
        console.print(f"\n  [green]✔ Hardware Accelerator:[/] Detected {gpu_name}")
    else:
        console.print("\n  [yellow]⚠ No dedicated hardware accelerator detected. Inference will run on CPU.[/]")


def run_system_check(clients: dict[str, BaseRuntimeClient]) -> None:
    """Run diagnostics on Ollama, MS Foundry, platform, and GPU/Metal accelerator."""
    specs = get_system_specs(client=clients.get("ollama"))
    display_system_banner(specs)

    console.print("\n[bold cyan]🔍 Checking environment readiness across runtimes:[/]")
    _check_ollama(clients.get("ollama"))
    _check_foundry(clients.get("foundry"))
    _check_onnx(clients.get("onnx-gpu"))
    _check_prism(clients.get("prism"))
    _check_accelerator(specs)


# ---------------------------------------------------------------------------
# --pull-recommended
# ---------------------------------------------------------------------------


def _print_pull_progress(data: dict[str, Any]) -> None:
    status = data.get("status", "")
    completed = data.get("completed", 0)
    total = data.get("total", 0)
    if total > 0:
        pct = completed / total * 100
        print(
            f"\r    {status}: {pct:.1f}% ({completed // (1024 * 1024)}MB / {total // (1024 * 1024)}MB)",
            end="",
            flush=True,
        )
    else:
        print(f"\r    {status}", end="", flush=True)


def _recommended_models_for(rec_config: Any, runtime: str) -> list[str]:
    """Read recommended models for a runtime (supports the legacy flat list, meaning Ollama)."""
    if isinstance(rec_config, dict):
        return rec_config.get(runtime, [])
    if runtime == "ollama" and isinstance(rec_config, list):
        return rec_config
    return []


def pull_recommended_models(clients: dict[str, BaseRuntimeClient], config: dict[str, Any], target_runtime: str) -> None:
    """Pull / download recommended models from active runtime registries."""
    rec_config = config.get("recommended_models", {})

    if target_runtime == "onnx-gpu":
        console.print("\n[bold yellow]━━━ Direct ONNX GenAI (onnx-gpu) Model Management ━━━[/]")
        client = clients.get("onnx-gpu")
        installed = [m.get("name") for m in client.list_installed_models()] if client else []
        if installed:
            console.print(
                f"  [green]✔ Direct ONNX model(s) already installed in `models/`:[/] [bold cyan]{', '.join(installed)}[/]"
            )
        else:
            console.print("  [yellow]ℹ No ONNX models found in `models/` or Foundry cache.[/]")

        console.print(
            "\n  [dim]Direct ONNX models are local checkpoint directories (HuggingFace), not managed by a daemon pull registry.[/]"
        )
        console.print("  [bold cyan]To download official ONNX CUDA weights via HuggingFace:[/] ")
        console.print("    [bold white]huggingface-cli download microsoft/Phi-4-mini-instruct-onnx[/] \\")
        console.print("    [bold white]  --include 'gpu/gpu-int4-rtn-block-32/*'[/] \\")
        console.print("    [bold white]  --local-dir models/Phi-4-mini-instruct-cuda-gpu[/]\n")
        console.print("  [bold cyan]To run benchmarks with the installed model:[/] ")
        console.print("    [bold white]benchrig --runtime onnx-gpu --suite coding[/]\n")
        return

    runtimes_to_pull = ["ollama", "foundry"] if target_runtime == "all" else [target_runtime]

    for rt in runtimes_to_pull:
        client = clients.get(rt)
        models = _recommended_models_for(rec_config, rt)
        if not client or not models:
            console.print(f"\n[yellow]⚠ No recommended models configured for runtime '{rt}'.[/]")
            continue

        console.print(
            f"\n[bold yellow]━━━ Checking recommended models for {client.display_name} ({client.engine_name}) ━━━[/]"
        )
        installed = [m.get("name") for m in client.list_installed_models()]

        for model in models:
            if any(model == inst or model in str(inst) for inst in installed):
                console.print(f"  [green]✔ Model {model} is already cached/installed.[/]")
                continue

            console.print(f"  [bold yellow]⬇ Downloading model {model}...[/]")
            success = client.pull_model(model, stream_callback=_print_pull_progress)
            print()
            if success:
                console.print(f"  [bold green]✔ Model {model} acquired successfully![/]")
            else:
                console.print(f"  [bold red]✖ Error acquiring model {model}.[/]")

    if target_runtime == "all":
        client_onnx = clients.get("onnx-gpu")
        installed_onnx = [m.get("name") for m in client_onnx.list_installed_models()] if client_onnx else []
        console.print("\n[bold yellow]━━━ Direct ONNX GenAI (onnx-gpu) Checkpoints ━━━[/]")
        if installed_onnx:
            console.print(
                f"  [green]✔ Direct ONNX model(s) available in `models/`:[/] [cyan]{', '.join(installed_onnx)}[/]"
            )
        else:
            console.print("  [dim]No Direct ONNX models found in `models/`. Download via huggingface-cli.[/]")


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
    config: dict[str, Any], pair_id: str, selected_runtime: str, has_baseline: bool
) -> tuple[list[tuple[str, str]], str | None]:
    """
    Resolve a `--pair` id from config into (targets, recommended_suite).

    With a cached Ollama baseline only the other engine is run.
    """
    pairs = config.get("model_pairs_1to1", [])
    pair = next((p for p in pairs if pair_id in (p.get("id"), p.get("name"))), None)
    if not pair:
        console.print(f"[bold red]Unknown 1:1 pair ID:[/] {pair_id}. Available: {[p.get('id') for p in pairs]}")
        sys.exit(1)

    console.print(f"[bold cyan]Selected 1:1 Pair:[/] [bold]{pair.get('name')}[/]")
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
# 1:1 cross-engine comparison
# ---------------------------------------------------------------------------


def _model_key(name: object) -> str:
    """Comparable form of a model name: no `ollama:` proxy prefix, no `:latest`, only lowercase letters and digits.

    `qwen2.5-coder:7b` and `qwen2.5-coder-7b-instruct-generic-cpu` share the key `qwen25coder7b`, while the size tag
    that tells 3b from 14b is kept.
    """
    text = str(name).lower()
    text = text.removeprefix("ollama:").removesuffix(":latest")
    return re.sub(r"[^a-z0-9]", "", text)


def select_pair(scorecards: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pick the two scorecards a 1:1 report compares: the first non-Ollama one and its Ollama counterpart.

    The counterpart is the Ollama scorecard whose model name is contained in the other's (or the reverse), the longest match
    winning; with no match it falls back to the first Ollama scorecard. Before this, the first Ollama scorecard in the list
    was used, which was only the right model when it happened to come first.
    """
    sc_other = next((sc for sc in scorecards if not is_ollama(sc)), scorecards[1])
    other_key = _model_key(sc_other.get("model"))
    candidates = [sc for sc in scorecards if is_ollama(sc)]
    matches = [
        sc for sc in candidates if (key := _model_key(sc.get("model"))) and (key in other_key or other_key in key)
    ]
    sc_ollama = (
        max(matches, key=lambda sc: len(_model_key(sc.get("model")))) if matches else (candidates or scorecards)[0]
    )
    return sc_ollama, sc_other


def records_for(results: list[dict[str, Any]], scorecard: dict[str, Any]) -> list[dict[str, Any]]:
    """The result records of one model on one runtime (a run can hold many models, and several runtimes for one name)."""
    runtime = scorecard.get("runtime", "ollama")
    return [r for r in results if r.get("model") == scorecard.get("model") and r.get("runtime", "ollama") == runtime]


def show_1to1_comparison(
    scorecards: list[dict[str, Any]],
    results: list[dict[str, Any]],
    specs: dict[str, Any],
    output_dir: str,
) -> None:
    """Display and write the head-to-head report between a non-Ollama scorecard and its Ollama counterpart."""
    sc_ollama, sc_other = select_pair(scorecards)
    res_ollama = records_for(results, sc_ollama)
    res_other = records_for(results, sc_other)

    other_label = runtime_label(sc_other.get("runtime"), default=str(sc_other.get("runtime", "other")))
    pair_title = f"{sc_ollama.get('model')} (Ollama) vs {sc_other.get('model')} ({other_label})"
    display_1to1_comparison(sc_ollama, sc_other, res_ollama, res_other, pair_name=pair_title)

    report_path = f"{output_dir}/1TO1_COMPARISON_REPORT.md"
    generate_1to1_comparison_report(
        sc_ollama, sc_other, res_ollama, res_other, specs, pair_name=pair_title, output_path=report_path
    )
    console.print(f"[bold green]✔ 1:1 comparison report written to:[/] [cyan]{report_path}[/]\n")


def run_compare_mode(compare_path: str, output_dir: str, clients: dict[str, BaseRuntimeClient]) -> None:
    """Offline analysis of a cached benchmark run (no models are executed)."""
    data = load_json_or_exit(compare_path, "Benchmark file")
    scorecards = data.get("scorecards", [])
    results = data.get("results", [])
    specs = data.get("specs") or get_system_specs(client=clients.get("ollama"))

    display_system_banner(specs)
    display_leaderboard(scorecards, specs=specs)
    display_token_savings(scorecards)

    if len(scorecards) >= 2:
        show_1to1_comparison(scorecards, results, specs, output_dir)


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


def load_baseline(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load cached Ollama scorecards and results so Ollama models are never re-run."""
    data = load_json_or_exit(path, "Baseline file")
    scorecards = [sc for sc in data.get("scorecards", []) if is_ollama(sc)]
    results = [r for r in data.get("results", []) if is_ollama(r)]
    console.print(f"[bold green]✔ Baseline loaded ({len(scorecards)} models) from:[/] [cyan]{path}[/]")
    console.print("[dim]Ollama models will NOT be re-executed; baseline data will be merged directly.[/]")
    return scorecards, results


def evaluate_model(
    runtime_name: str,
    client: BaseRuntimeClient,
    model: str,
    config: dict[str, Any],
    suites: list[str],
    scenarios: dict[str, list[dict[str, Any]]],
    runs: int,
) -> list[dict[str, Any]]:
    """Load, warm up, benchmark, and unload one model; return its raw result records."""
    console.print(
        f"\n[bold yellow]━━━ [{client.display_name}: {model}] Starting evaluation ({client.engine_name}) ━━━[/]"
    )
    runner = BenchmarkRunner(client=client, config=config)
    # Before the model is loaded, so reports can separate the model's memory from whatever else uses the GPU.
    baseline_mb = runner.measure_vram_baseline()
    if baseline_mb > 0:
        console.print(f"  [dim]GPU memory in use before loading: {baseline_mb:.0f} MB[/]")

    console.print(f"  [dim]Ensuring model {model} is loaded ({client.display_name})...[/]")
    client.load_model(model)

    rt_config = config.get(runtime_name, {})
    if rt_config.get("warmup", True):
        runner.warmup(model)

    results: list[dict[str, Any]] = []
    for run_idx in range(runs):
        runner.run_index = run_idx
        if runs > 1:
            console.print(f"  [dim]Run {run_idx + 1}/{runs}[/]")
        for suite in suites:
            if not scenarios.get(suite):
                continue
            _, method_name, message = SUITES[suite]
            console.print(f"  [bold]• {message}[/]")
            suite_results = getattr(runner, method_name)(model, scenarios[suite])
            skip_reason = _suite_skip_notice(runner, suite_results, scenarios[suite])
            if skip_reason:
                console.print(f"  [yellow]⚠ Skipped — model does not fit in available VRAM: {skip_reason}[/]")
            for r in suite_results:
                display_scenario_result(r)
            results.extend(suite_results)

    # Unload after the test so the next model starts from clean memory
    if rt_config.get("unload_after_test", True):
        console.print(f"  [dim]Unloading memory for model {model} ({client.display_name})...[/]")
        client.unload_model(model)
        time.sleep(1.0)

    return results


def build_scorecards(
    config: dict[str, Any],
    ollama_client: BaseRuntimeClient,
    targets: list[tuple[str, str]],
    results: list[dict[str, Any]],
    baseline_scorecards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score each target model and append baseline scorecards not superseded by a fresh run."""
    scorer = BenchmarkRunner(client=ollama_client, config=config)
    scorecards = [
        sc for runtime, model in targets if (sc := scorer.compute_model_scorecard(model, results, runtime=runtime))
    ]
    existing = {(sc.get("runtime"), sc.get("model")) for sc in scorecards}
    scorecards.extend(sc for sc in baseline_scorecards if (sc.get("runtime"), sc.get("model")) not in existing)
    return scorecards


def save_outputs(
    output_dir: str,
    specs: dict[str, Any],
    scorecards: list[dict[str, Any]],
    results: list[dict[str, Any]],
    total_duration: float,
    csv_path: str | None = None,
    chart_path: str | None = None,
) -> tuple[str, str]:
    """Write raw JSON, latest-run JSON and markdown summary; return (markdown_path, raw_json_path).

    When ``csv_path``/``chart_path`` are given (from ``--csv``/``--chart``), also write those
    artifacts and link/embed them in the Markdown report (plan.md Phase 1, items 1.1-1.3).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"{output_dir}/runs", exist_ok=True)
    raw_json_path = f"{output_dir}/runs/benchmark_{timestamp}.json"

    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "specs": specs,
                "scorecards": scorecards,
                "results": results,
                "total_duration_sec": total_duration,
            },
            f,
            indent=2,
        )
    with open(f"{output_dir}/latest.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "scorecards": scorecards}, f, indent=2)

    md_report_path = f"{output_dir}/LATEST_SUMMARY.md"
    md_dir = os.path.dirname(md_report_path) or "."

    if csv_path:
        write_scorecards_csv(scorecards, csv_path)
        console.print(f"  [dim]CSV written to {csv_path}[/]")
    if chart_path:
        write_scorecards_chart(scorecards, chart_path)
        console.print(f"  [dim]Chart written to {chart_path}[/]")

    generate_markdown_report(
        scorecards,
        results,
        specs,
        output_path=md_report_path,
        chart_path=os.path.relpath(chart_path, md_dir) if chart_path else None,
        csv_path=os.path.relpath(csv_path, md_dir) if csv_path else None,
    )
    return md_report_path, raw_json_path


def run_benchmarks(
    args: argparse.Namespace,
    config: dict[str, Any],
    clients: dict[str, BaseRuntimeClient],
    selected_runtime: str,
) -> None:
    """Resolve targets, execute the selected suites, then display and persist the results."""
    baseline_scorecards: list[dict[str, Any]] = []
    baseline_results: list[dict[str, Any]] = []
    if args.baseline:
        baseline_scorecards, baseline_results = load_baseline(args.baseline)

    suite = args.suite
    if args.pair:
        targets, recommended_suite = resolve_pair_targets(
            config, args.pair, selected_runtime, has_baseline=bool(args.baseline)
        )
        if suite == "all" and recommended_suite:
            suite = recommended_suite
    else:
        targets = resolve_target_models(args.models, selected_runtime, clients)

    if not targets and not baseline_scorecards:
        console.print("[bold red]No models to benchmark! Run with --check, --pull-recommended, or specify --models.[/]")
        onnx_client = clients.get("onnx-gpu")
        if selected_runtime == "onnx-gpu" and (not onnx_client or not onnx_client.is_reachable()):
            console.print(
                "[dim]Hint: Direct ONNX GenAI is not active in this Python environment. "
                'Try: pip install "benchrig[onnx-gpu]"[/]'
            )
        elif selected_runtime == "prism":
            console.print("[dim]Hint: Start the Prism server first: pip install prism-local && prism serve[/]")
        else:
            console.print(
                "[dim]Hint: If testing MS Foundry, ensure Foundry Local server is running ('foundry service start').[/]"
            )
        sys.exit(1)

    specs = get_system_specs(client=clients["ollama"])
    display_system_banner(specs)

    suites_to_run = list(SUITES) if suite == "all" else [suite]
    scenarios_dir = resolve_scenarios_dir(args.scenarios_dir)
    scenarios = {name: load_scenario_file(os.path.join(scenarios_dir, SUITES[name][0])) for name in suites_to_run}

    console.print(f"\n[bold green]🚀 Starting benchmark for models:[/] {', '.join(f'{rt}:{m}' for rt, m in targets)}")
    console.print(f"[bold cyan]Selected test suites:[/] {', '.join(suites_to_run)}\n")
    # Warn once per run when a target uses the slow generic-cpu execution provider on a CUDA host (spec.md I4).
    _warn_slow_provider(targets, specs)

    raw_results: list[dict[str, Any]] = []
    total_start = time.time()
    for runtime_name, model in targets:
        client = clients.get(runtime_name)
        if not client:
            console.print(f"[bold red]Unknown runtime: {runtime_name}[/]")
            continue
        if not client.is_reachable():
            console.print(
                f"[bold yellow]⚠ Skipping {runtime_name}:{model} — {client.display_name} server is not responding.[/]"
            )
            continue
        raw_results.extend(evaluate_model(runtime_name, client, model, config, suites_to_run, scenarios, args.runs))
    total_duration = time.time() - total_start

    all_results = baseline_results + raw_results
    scorecards = build_scorecards(config, clients["ollama"], targets, all_results, baseline_scorecards)
    if not scorecards:
        console.print("[bold yellow]No benchmark results were recorded.[/]")
        return

    console.print("\n")
    display_leaderboard(scorecards, specs=specs)
    display_token_savings(scorecards)

    has_ollama = any(is_ollama(sc) for sc in scorecards)
    has_other = any(not is_ollama(sc) for sc in scorecards)
    if has_ollama and has_other:
        show_1to1_comparison(scorecards, all_results, specs, args.output_dir)

    md_report_path, raw_json_path = save_outputs(
        args.output_dir,
        specs,
        scorecards,
        all_results,
        total_duration,
        csv_path=args.csv,
        chart_path=args.chart,
    )

    tokens_saved = sum(sc.get("total_tokens_saved", 0) for sc in scorecards)
    cost_saved = sum(sc.get("est_cost_saved_usd", 0.0) for sc in scorecards)
    console.print(f"\n[bold green]✔ Benchmark completed in {total_duration:.1f}s![/]")
    console.print(
        f"  [cyan]Cloud Tokens Saved:[/] [bold green]{tokens_saved:,}[/] ([bold]~${cost_saved:.4f} USD[/] equivalent)"
    )
    console.print(f"  [cyan]Markdown Report:[/] [bold]{md_report_path}[/]")
    console.print(f"  [cyan]Raw JSON Data:[/] [bold]{raw_json_path}[/]\n")


# ---------------------------------------------------------------------------
# Process setup & CLI
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
        help="Test suite selection (all, coding, reasoning, speed, context, polish)",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions per test (default: 1)")
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _bootstrap_cuda_env()
    config = load_config(args.config)

    clients: dict[str, BaseRuntimeClient] = {
        name: create_runtime_client(name, config) for name in ("ollama", "foundry", "onnx-gpu", "prism")
    }

    if args.compare:
        run_compare_mode(args.compare, args.output_dir, clients)
        return

    selected_runtime = args.runtime or config.get("benchmark", {}).get("default_runtime", "ollama")

    if args.check:
        run_system_check(clients)
    elif args.pull_recommended:
        pull_recommended_models(clients, config, target_runtime=selected_runtime)
    else:
        run_benchmarks(args, config, clients, selected_runtime)


if __name__ == "__main__":
    main()
