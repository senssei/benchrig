"""Main benchmark orchestration (``benchrig`` without ``--check``/``--compare``/``--pull-recommended``).

Phase 12.4: carved out of the monolithic ``benchrig/cli.py``. ``benchrig.cli:main`` dispatches to
``run_benchmarks`` for the default benchmark path; ``run.py`` calls into ``compare.py`` for
the 1:1 cross-engine report and ``report.py`` for the Markdown/CSV/chart artifacts.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Callable
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Package alias so cross-module helpers are looked up via ``benchrig.cli.X`` at call time
# (tests patch ``benchrig.cli.X``, not the source ``_common.X``; using the re-export keeps
# the patch effective).
import benchrig.cli as _cli_pkg
from benchrig.cli._common import (
    DEFAULT_SUITES,
    SUITES,
    is_ollama,
    load_scenario_file,
)
from benchrig.core.client import BaseRuntimeClient
from benchrig.core.runner import BenchmarkRunner
from benchrig.reporting.display import (
    display_scenario_result,
)


def evaluate_model(
    runtime_name: str,
    client: BaseRuntimeClient,
    model: str,
    config: dict[str, Any],
    suites: list[str],
    scenarios: dict[str, list[dict[str, Any]]],
    runs: int,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    warmup_runs: int = 0,
) -> list[dict[str, Any]]:
    """Load, warm up, benchmark, and unload one model; return its raw result records.

    ``on_progress``, when given, is called with each scenario's result record as soon as it completes
    (before the whole suite finishes), so a caller can advance an overall progress bar in real time.
    """
    _cli_pkg._lookup("console").print(
        f"\n[bold yellow]━━━ [{client.display_name}: {model}] Starting evaluation ({client.engine_name}) ━━━[/]"
    )

    def on_result(record: dict[str, Any]) -> None:
        if record.get("phase") == "warmup":
            return
        display_scenario_result(record)
        if on_progress:
            on_progress(record)

    runner = BenchmarkRunner(client=client, config=config, on_result=on_result, warmup_runs=warmup_runs)
    # Before the model is loaded, so reports can separate the model's memory from whatever else uses the GPU.
    baseline_mb = runner.measure_vram_baseline()
    if baseline_mb > 0:
        _cli_pkg._lookup("console").print(f"  [dim]GPU memory in use before loading: {baseline_mb:.0f} MB[/]")

    _cli_pkg._lookup("console").print(f"  [dim]Ensuring model {model} is loaded ({client.display_name})...[/]")
    tool_only = suites == ["tool_use"]
    if not tool_only or getattr(client, "native_tool_channel", False):
        client.load_model(model)

    rt_config = config.get(runtime_name, {})
    if not tool_only and rt_config.get("warmup", True):
        runner.warmup(model)

    results: list[dict[str, Any]] = []
    for run_idx in range(runs):
        runner.run_index = run_idx
        if runs > 1:
            _cli_pkg._lookup("console").print(f"  [dim]Run {run_idx + 1}/{runs}[/]")
        for suite in suites:
            if not scenarios.get(suite):
                continue
            _, method_name, message = SUITES[suite]
            _cli_pkg._lookup("console").print(f"  [bold]• {message}[/]")
            suite_results = getattr(runner, method_name)(model, scenarios[suite])
            skip_reason = _cli_pkg._lookup("_suite_skip_notice")(runner, suite_results, scenarios[suite])
            if skip_reason:
                _cli_pkg._lookup("console").print(
                    f"  [yellow]⚠ Skipped — model does not fit in available VRAM: {skip_reason}[/]"
                )
            results.extend(suite_results)

    # Unload after the test so the next model starts from clean memory
    if rt_config.get("unload_after_test", True):
        _cli_pkg._lookup("console").print(f"  [dim]Unloading memory for model {model} ({client.display_name})...[/]")
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


def validate_cache_probe(args, selected_runtime: str, scenarios: list[dict]) -> str:
    """Validate a probe without connecting to a model or runtime."""
    if args.suite != "speed" or args.pair or args.baseline or args.compare or args.check or args.pull_recommended:
        raise ValueError("--cache-probe requires --suite speed and cannot use pair/baseline/compare/check/pull modes")
    if args.runs <= 0 or selected_runtime == "all":
        raise ValueError("--cache-probe requires positive --runs and one runtime")
    model = args.models.strip()
    if not model or model in ("installed", "all") or "," in model:
        raise ValueError("--cache-probe requires one explicit --models value")
    if ":" in model and model.split(":", 1)[0] in ("ollama", "foundry", "onnx-gpu", "prism"):
        runtime, model = model.split(":", 1)
        if runtime != selected_runtime or not model:
            raise ValueError("--cache-probe model prefix must match selected runtime")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("cache probe needs a nonempty speed.json scenario list")
    for scenario in scenarios:
        if not isinstance(scenario, dict) or not all(
            isinstance(scenario.get(k), str) and scenario[k] for k in ("id", "name", "prompt")
        ):
            raise ValueError("cache probe scenarios require id, name and prompt")
        options = scenario.get("options")
        if not isinstance(options, dict) or type(options.get("num_ctx")) is not int or options["num_ctx"] <= 0:
            raise ValueError("cache probe requires explicit positive num_ctx in every speed scenario")
    return model


def run_cache_probe(args, config: dict, clients: dict, selected_runtime: str) -> None:
    """Run an explicitly selected paired probe and persist evidence without scorecards."""
    client = clients[selected_runtime]
    model = validate_cache_probe(args, selected_runtime, args.cache_probe_scenarios)
    runner = BenchmarkRunner(client, config)
    specs = _cli_pkg._lookup("get_system_specs")(client=client)
    started = time.monotonic()
    try:
        if not client.load_model(model):
            raise ValueError(f"model startup failed: {selected_runtime}:{model}")
        results = runner.run_cache_probe(model, args.cache_probe_scenarios, runs=args.runs)
    except Exception as exc:
        results = [
            {
                "suite": "cache_probe",
                "model": model,
                "runtime": selected_runtime,
                "phase": "startup",
                "success": False,
                "error": f"startup/probe failure: {exc}",
                "runtime_state": "unverified",
                "prefix_cache_state": "unverified",
                "comparable": False,
                "comparison_reasons": ["startup/probe failure"],
            }
        ]
    finally:
        client.unload_model(model)
    markdown, raw_json = _cli_pkg._lookup("save_outputs")(
        args.output_dir,
        specs,
        [],
        results,
        time.monotonic() - started,
        csv_path=args.csv,
        chart_path=args.chart,
        config=config,
    )
    _cli_pkg._lookup("console").print(f"Cache probe evidence: {raw_json}\nReport: {markdown}")


def run_benchmarks(
    args: argparse.Namespace,
    config: dict[str, Any],
    clients: dict[str, BaseRuntimeClient],
    selected_runtime: str,
) -> None:
    """Resolve targets, execute the selected suites, then display and persist the results."""
    log = logging.getLogger("benchrig")

    baseline_scorecards: list[dict[str, Any]] = []
    baseline_results: list[dict[str, Any]] = []
    if args.baseline:
        baseline_scorecards, baseline_results = _cli_pkg._lookup("load_baseline")(args.baseline)

    suite = args.suite
    if args.pair:
        targets, recommended_suite = _cli_pkg._lookup("resolve_pair_targets")(
            config, args.pair, selected_runtime, has_baseline=bool(args.baseline)
        )
        if suite == "all" and recommended_suite:
            suite = recommended_suite
    else:
        targets = _cli_pkg._lookup("resolve_target_models")(args.models, selected_runtime, clients)

    # ``run.started`` (Phase 11 event) fires after target resolution with `num_models` filled
    # in. Logged via JSON stderr; the rich UX on stdout (the green "Starting benchmark for" line
    # below) is unchanged.
    log.info(
        "run.started",
        extra={
            "event": "run.started",
            "argv": getattr(args, "argv", []),
            "runtime": selected_runtime,
            "num_models": len(targets),
            "runs": args.runs,
        },
    )

    started_monotonic = time.monotonic()
    total_duration_sec = 0.0
    scorecards: list[dict[str, Any]] = []
    try:
        if not targets and not baseline_scorecards:
            _cli_pkg._lookup("console").print(
                "[bold red]No models to benchmark! Run with --check, --pull-recommended, or specify --models.[/]"
            )
            onnx_client = clients.get("onnx-gpu")
            if selected_runtime == "onnx-gpu" and (not onnx_client or not onnx_client.is_reachable()):
                _cli_pkg._lookup("console").print(
                    "[dim]Hint: Direct ONNX GenAI is not active in this Python environment. "
                    'Try: pip install "benchrig[onnx-gpu]"[/]'
                )
            elif selected_runtime == "prism":
                _cli_pkg._lookup("console").print(
                    "[dim]Hint: Start the Prism server first: pip install prism-local && prism serve[/]"
                )
            else:
                _cli_pkg._lookup("console").print(
                    "[dim]Hint: If testing MS Foundry, ensure Foundry Local server is running ('foundry service start').[/]"
                )
            sys.exit(1)

        specs = _cli_pkg._lookup("get_system_specs")(client=clients["ollama"])
        _cli_pkg._lookup("display_system_banner")(specs)

        suites_to_run = list(DEFAULT_SUITES) if suite == "all" else [suite]
        scenarios_dir = _cli_pkg._lookup("resolve_scenarios_dir")(args.scenarios_dir)
        scenarios = {name: load_scenario_file(os.path.join(scenarios_dir, SUITES[name][0])) for name in suites_to_run}

        _cli_pkg._lookup("console").print(
            f"\n[bold green]🚀 Starting benchmark for models:[/] {', '.join(f'{rt}:{m}' for rt, m in targets)}"
        )
        _cli_pkg._lookup("console").print(f"[bold cyan]Selected test suites:[/] {', '.join(suites_to_run)}\n")
        # Warn once per run when a target uses the slow generic-cpu execution provider on a CUDA host (spec.md I4).
        _cli_pkg._lookup("_warn_slow_provider")(targets, specs)

        # One unit per (model x run x scenario); models skipped for capacity leave the bar short of 100%, which is fine —
        # it still shows real progress instead of nothing until a whole suite finishes (plan.md progress-bar item).
        total_steps = _cli_pkg._lookup("_total_scenario_steps")(scenarios, args.runs, len(targets))

        raw_results: list[dict[str, Any]] = []
        total_start = time.time()
        with Progress(
            TextColumn("[bold blue]{task.fields[label]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=_cli_pkg._lookup("console"),
        ) as progress:
            task = progress.add_task("benchmark", total=total_steps or None, label="Benchmarking")
            for runtime_name, model in targets:
                client = clients.get(runtime_name)
                if not client:
                    _cli_pkg._lookup("console").print(f"[bold red]Unknown runtime: {runtime_name}[/]")
                    continue
                adapter_unsupported = suite == "tool_use" and not getattr(client, "native_tool_channel", False)
                if not adapter_unsupported and not client.is_reachable():
                    _cli_pkg._lookup("console").print(
                        f"[bold yellow]⚠ Skipping {runtime_name}:{model} — {client.display_name} server is not responding.[/]"
                    )
                    continue
                progress.update(task, label=f"{runtime_name}:{model}")
                raw_results.extend(
                    _cli_pkg._lookup("evaluate_model")(
                        runtime_name,
                        client,
                        model,
                        config,
                        suites_to_run,
                        scenarios,
                        args.runs,
                        on_progress=lambda _record: progress.advance(task),
                        warmup_runs=args.warmup_runs,
                    )
                )
        total_duration_sec = time.time() - total_start
        scorecards = _cli_pkg._lookup("build_scorecards")(
            config, clients["ollama"], targets, baseline_results + raw_results, baseline_scorecards
        )
        if not scorecards:
            _cli_pkg._lookup("console").print("[bold yellow]No benchmark results were recorded.[/]")
            return

        all_results = baseline_results + raw_results
        _cli_pkg._lookup("console").print("\n")
        _cli_pkg._lookup("display_leaderboard")(scorecards, specs=specs)
        _cli_pkg._lookup("display_token_savings")(scorecards)

        legacy_cards = [sc for sc in scorecards if sc.get("composite_score") is not None]
        has_ollama = any(is_ollama(sc) for sc in legacy_cards)
        has_other = any(not is_ollama(sc) for sc in legacy_cards)
        if has_ollama and has_other:
            _cli_pkg._lookup("show_1to1_comparison")(legacy_cards, all_results, specs, args.output_dir)

        md_report_path, raw_json_path = _cli_pkg._lookup("save_outputs")(
            args.output_dir,
            specs,
            scorecards,
            all_results,
            total_duration_sec,
            csv_path=args.csv,
            chart_path=args.chart,
            config=config,
        )

        tokens_saved = sum(sc.get("total_tokens_saved", 0) for sc in scorecards)
        cost_saved = sum(sc.get("est_cost_saved_usd", 0.0) for sc in scorecards)
        _cli_pkg._lookup("console").print(f"\n[bold green]✔ Benchmark completed in {total_duration_sec:.1f}s![/]")
        _cli_pkg._lookup("console").print(
            f"  [cyan]Cloud Tokens Saved:[/] [bold green]{tokens_saved:,}[/] ([bold]~${cost_saved:.4f} USD[/] equivalent)"
        )
        _cli_pkg._lookup("console").print(f"  [cyan]Markdown Report:[/] [bold]{md_report_path}[/]")
        _cli_pkg._lookup("console").print(f"  [cyan]Raw JSON Data:[/] [bold]{raw_json_path}[/]\n")
    finally:
        # Phase 11: ``run.completed`` always fires - including on a crash mid-benchmark.
        models_ok = len({(sc.get("runtime"), sc.get("model")) for sc in scorecards})
        log.info(
            "run.completed",
            extra={
                "event": "run.completed",
                "total_duration_sec": round(time.monotonic() - started_monotonic, 3),
                "models_ok": models_ok,
                "models_skipped": max(0, len(targets) - models_ok),
            },
        )


__all__ = ["run_benchmarks", "evaluate_model", "build_scorecards"]
