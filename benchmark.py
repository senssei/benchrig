#!/usr/bin/env python3
"""
Local LLM Benchmark & Load Testing Rig (macOS Apple Silicon Metal / Linux NVIDIA CUDA).
Profiles and ranks local models across Ollama and Microsoft Foundry Server Runtimes
by coding quality, reasoning, generation speed, and memory efficiency.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple
import yaml

from core.client import (
    BaseRuntimeClient,
    FoundryClient,
    OllamaClient,
    create_runtime_client,
)
from core.hardware import get_system_specs
from core.runner import BenchmarkRunner
from reporting.display import (
    console,
    display_1to1_comparison,
    display_leaderboard,
    display_scenario_result,
    display_system_banner,
    display_token_savings,
)
from reporting.markdown import (
    generate_1to1_comparison_report,
    generate_markdown_report,
)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration yaml file."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_scenario_file(filepath: str) -> List[Dict[str, Any]]:
    """Load scenarios from JSON file."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def run_system_check(clients: Dict[str, BaseRuntimeClient]):
    """Run diagnostics on Ollama, MS Foundry, platform, and GPU/Metal accelerator."""
    ollama_client = clients.get("ollama")
    specs = get_system_specs(client=ollama_client)
    display_system_banner(specs)

    console.print("\n[bold cyan]🔍 Checking environment readiness across runtimes:[/]")

    # 1. Ollama status
    if ollama_client and ollama_client.is_reachable():
        ver = ollama_client.get_version()
        console.print(f"  [green]✔ Ollama REST API:[/] Available (Version: {ver}, llama.cpp) at {ollama_client.base_url}")
        installed = ollama_client.list_installed_models()
        console.print(f"     Installed Ollama models ({len(installed)}):")
        for m in installed:
            name = m.get("name", "")
            size_gb = m.get("size", 0) / (1024**3)
            param_sz = m.get("details", {}).get("parameter_size", "N/A")
            quant = m.get("details", {}).get("quantization_level", "N/A")
            console.print(f"     • [cyan]{name:<24}[/] ({size_gb:.1f} GB, {param_sz}, {quant})")
    else:
        url = ollama_client.base_url if ollama_client else "http://localhost:11434"
        console.print(f"  [yellow]⚠ Ollama REST API:[/] Unreachable at {url}")

    # 2. MS Foundry status
    foundry_client = clients.get("foundry")
    if foundry_client and foundry_client.is_reachable():
        ver = foundry_client.get_version()
        console.print(f"  [green]✔ MS Foundry Server REST API:[/] Available (Version: {ver}, ONNX Runtime GenAI) at {foundry_client.base_url}")
        installed = foundry_client.list_installed_models()
        console.print(f"     Available Foundry models ({len(installed)}):")
        for m in installed:
            name = m.get("name", "")
            param_sz = m.get("details", {}).get("parameter_size", "ONNX")
            quant = m.get("details", {}).get("quantization_level", "ONNX")
            console.print(f"     • [blue]{name:<24}[/] ({param_sz}, {quant})")
    else:
        url = foundry_client.base_url if foundry_client else "http://localhost:5272/v1"
    # 3. Direct ONNX GenAI GPU status
    onnx_client = clients.get("onnx-gpu")
    if onnx_client and onnx_client.is_reachable():
        ver = onnx_client.get_version()
        cuda_ok = getattr(onnx_client, "is_cuda_available", lambda: False)()
        cuda_status = "[bold green]CUDA Enabled[/]" if cuda_ok else "[yellow]CPU[/]"
        console.print(f"  [green]✔ ONNX GenAI Direct Engine:[/] Available (Version: {ver}, {cuda_status})")
        installed_onnx = onnx_client.list_installed_models()
        console.print(f"     Available Direct ONNX Models ({len(installed_onnx)}):")
        for m in installed_onnx:
            name = m.get("name", "")
            quant = m.get("details", {}).get("quantization_level", "ONNX")
            console.print(f"     • [magenta]{name:<28}[/] ({quant})")
    else:
        console.print("  [yellow]ℹ ONNX GenAI Direct Engine:[/] Not installed (pip install onnxruntime-genai-cuda)")

    # 4. Accelerator / GPU
    gpu_type = specs.get("gpu_type")
    if gpu_type == "apple_silicon":
        console.print(f"\n  [green]✔ Apple Silicon Metal Accelerator:[/] Detected {specs.get('gpu_name')} ({specs.get('ram_total_gb')} GB Unified Memory)")
    elif gpu_type == "nvidia":
        console.print(f"\n  [green]✔ NVIDIA GPU Access:[/] Detected {specs.get('gpu_name')} ({specs.get('gpu_vram_total_mb')} MB VRAM)")
    elif specs.get("gpu_name") != "None":
        console.print(f"\n  [green]✔ Hardware Accelerator:[/] Detected {specs.get('gpu_name')}")
    else:
        console.print("\n  [yellow]⚠ No dedicated hardware accelerator detected. Inference will run on CPU.[/]")


def pull_recommended_models(clients: Dict[str, BaseRuntimeClient], config: Dict[str, Any], target_runtime: str):
    """Pull / download recommended models from active runtime registries."""
    rec_config = config.get("recommended_models", {})

    runtimes_to_pull = ["ollama", "foundry"] if target_runtime == "all" else [target_runtime]

    for rt in runtimes_to_pull:
        client = clients.get(rt)
        if not client:
            continue

        if isinstance(rec_config, dict):
            models = rec_config.get(rt, [])
        elif rt == "ollama" and isinstance(rec_config, list):
            models = rec_config
        else:
            models = []

        if not models:
            continue

        console.print(f"\n[bold yellow]━━━ Checking recommended models for {client.display_name} ({client.engine_name}) ━━━[/]")
        installed = [m.get("name") for m in client.list_installed_models()]

        for model in models:
            if any(model == inst or model in str(inst) for inst in installed):
                console.print(f"  [green]✔ Model {model} is already cached/installed.[/]")
                continue

            console.print(f"  [bold yellow]⬇ Downloading model {model}...[/]")

            def progress_cb(data):
                status = data.get("status", "")
                completed = data.get("completed", 0)
                total = data.get("total", 0)
                if total > 0:
                    pct = (completed / total) * 100
                    print(f"\r    {status}: {pct:.1f}% ({completed//(1024*1024)}MB / {total//(1024*1024)}MB)", end="", flush=True)
                else:
                    print(f"\r    {status}", end="", flush=True)

            success = client.pull_model(model, stream_callback=progress_cb)
            print()
            if success:
                console.print(f"  [bold green]✔ Model {model} acquired successfully![/]")
            else:
                console.print(f"  [bold red]✖ Error acquiring model {model}.[/]")


def resolve_target_models(
    raw_models_arg: str,
    selected_runtime: str,
    clients: Dict[str, BaseRuntimeClient],
) -> List[Tuple[str, str]]:
    """
    Resolve model target list into tuples of (runtime_name, model_name).
    Supports:
    - 'installed' / 'all'
    - 'qwen2.5-coder:7b' (inherits selected_runtime)
    - 'ollama:qwen2.5-coder:7b,foundry:phi-4' (explicit prefixes)
    """
    targets: List[Tuple[str, str]] = []

    if raw_models_arg in ("installed", "all"):
        rts = ["ollama", "foundry", "onnx-gpu"] if selected_runtime == "all" else [selected_runtime]
        for rt in rts:
            client = clients.get(rt)
            if client and client.is_reachable():
                for m in client.list_installed_models():
                    name = m.get("name")
                    if name:
                        targets.append((rt, name))
        return targets

    # Comma-separated list
    items = [item.strip() for item in raw_models_arg.split(",") if item.strip()]
    for item in items:
        if ":" in item and any(item.startswith(f"{prefix}:") for prefix in ("ollama", "foundry", "ms-foundry", "onnx-gpu", "onnx")):
            prefix, model_name = item.split(":", 1)
            if prefix in ("onnx-gpu", "onnx"):
                rt = "onnx-gpu"
            elif prefix in ("foundry", "ms-foundry"):
                rt = "foundry"
            else:
                rt = "ollama"
            targets.append((rt, model_name))
        else:
            # Check if runtime is 'all' - look up in clients
            if selected_runtime == "all":
                found_in_any = False
                for rt in ("onnx-gpu", "foundry", "ollama"):
                    client = clients.get(rt)
                    if client and client.is_reachable():
                        installed = [m.get("name") for m in client.list_installed_models()]
                        if any(item == inst or item in str(inst) for inst in installed):
                            targets.append((rt, item))
                            found_in_any = True
                if not found_in_any:
                    # Default to ollama if not found
                    targets.append(("ollama", item))
            else:
                targets.append((selected_runtime, item))

    return targets


def _bootstrap_cuda_env():
    """Ensure CUDA dynamic linker paths and library symlinks are configured before ONNX runtime initialization."""
    if os.environ.get("_ONNX_CUDA_BOOTSTRAPPED") == "1":
        return

    import glob
    project_root = os.path.abspath(os.path.dirname(__file__))
    venv_nvidia = glob.glob(os.path.join(project_root, ".venv/lib/python*/site-packages/nvidia/*/lib"))
    ollama_cuda13 = "/usr/local/lib/ollama/cuda_v13"

    for d in venv_nvidia:
        if "cufft" in d:
            so11 = os.path.join(d, "libcufft.so.11")
            so12 = os.path.join(d, "libcufft.so.12")
            if os.path.exists(so11) and not os.path.exists(so12):
                try:
                    os.symlink("libcufft.so.11", so12)
                except Exception:
                    pass

    candidate_dirs = [ollama_cuda13] + venv_nvidia
    existing_dirs = [d for d in candidate_dirs if os.path.isdir(d)]

    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if not all(d in cur_ld for d in existing_dirs):
        new_ld = ":".join(existing_dirs) + (":" + cur_ld if cur_ld else "")
        os.environ["LD_LIBRARY_PATH"] = new_ld
        os.environ["_ONNX_CUDA_BOOTSTRAPPED"] = "1"
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            pass


def main():
    _bootstrap_cuda_env()
    parser = argparse.ArgumentParser(
        description="Local LLM Benchmark & Load Testing Rig (Ollama, MS Foundry, and Direct ONNX GenAI CUDA)"
    )
    parser.add_argument(
        "--runtime",
        type=str,
        default=None,
        choices=["ollama", "foundry", "onnx-gpu", "all"],
        help="Inference runtime selection (ollama, foundry, onnx-gpu, all; default: from config or ollama)",
    )
    parser.add_argument(
        "--check", action="store_true", help="Run environment diagnostics across runtimes and accelerators"
    )
    parser.add_argument(
        "--pull-recommended", action="store_true", help="Pull recommended models for selected runtime(s)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="installed",
        help="Models to benchmark (comma-separated, runtime-prefixed e.g. 'foundry:phi-4', or 'installed')",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "coding", "reasoning", "speed", "context", "polish"],
        help="Test suite selection (all, coding, reasoning, speed, context, polish)",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of repetitions per test (default: 1)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Path to benchmark JSON run to load, analyze, and display 1:1 cross-engine comparison without running models",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to cached benchmark JSON run providing Ollama baseline so Ollama is never re-run",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="1:1 model comparison pair ID from config.yaml (e.g. 'phi_mini', 'qwen_coder_7b')",
    )

    args = parser.parse_args()

    config = load_config("config.yaml")

    # Instantiate runtime clients
    clients: Dict[str, BaseRuntimeClient] = {
        "ollama": create_runtime_client("ollama", config),
        "foundry": create_runtime_client("foundry", config),
        "onnx-gpu": create_runtime_client("onnx-gpu", config),
    }

    # 0. Compare mode (offline evaluation & reporting from cached run)
    if args.compare:
        compare_path = args.compare
        if not os.path.exists(compare_path):
            console.print(f"[bold red]Benchmark file not found:[/] [yellow]{compare_path}[/]")
            sys.exit(1)
        with open(compare_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scorecards = data.get("scorecards", [])
        results = data.get("results", [])
        specs = data.get("specs") or get_system_specs(client=clients.get("ollama"))

        display_system_banner(specs)
        display_leaderboard(scorecards, specs=specs)
        display_token_savings(scorecards)

        # 1:1 Head-to-Head Comparison display
        if len(scorecards) >= 2:
            sc_ollama = next(
                (sc for sc in scorecards if "ollama" in str(sc.get("runtime", "")).lower()),
                scorecards[0],
            )
            sc_other = next(
                (sc for sc in scorecards if "ollama" not in str(sc.get("runtime", "")).lower()),
                scorecards[1],
            )

            res_ollama = [
                r for r in results
                if r.get("model") == sc_ollama.get("model") or "ollama" in str(r.get("runtime", "")).lower()
            ]
            res_other = [
                r for r in results
                if r.get("model") == sc_other.get("model") or str(r.get("runtime", "")).lower() == str(sc_other.get("runtime", "")).lower()
            ]

            other_rt_label = "ONNX GPU" if "onnx" in str(sc_other.get("runtime", "")).lower() else "MS Foundry"
            pair_title = f"{sc_ollama.get('model')} (Ollama) vs {sc_other.get('model')} ({other_rt_label})"
            display_1to1_comparison(
                sc_ollama,
                sc_other,
                res_ollama,
                res_other,
                pair_name=pair_title,
            )

            rep_path = f"{args.output_dir}/1TO1_COMPARISON_REPORT.md"
            generate_1to1_comparison_report(
                sc_ollama,
                sc_other,
                res_ollama,
                res_other,
                specs,
                pair_name=pair_title,
                output_path=rep_path,
            )
            console.print(f"[bold green]✔ 1:1 Model comparison report written to:[/] [cyan]{rep_path}[/]\n")
        return

    # Determine active runtime
    default_rt = config.get("benchmark", {}).get("default_runtime", "ollama")
    selected_runtime = args.runtime or default_rt

    # 1. System check mode
    if args.check:
        run_system_check(clients)
        return

    # 2. Pull recommended mode
    if args.pull_recommended:
        pull_recommended_models(clients, config, target_runtime=selected_runtime)
        return

    # 3. Baseline & Pair resolution
    target_models: List[Tuple[str, str]] = []
    baseline_scorecards: List[Dict[str, Any]] = []
    baseline_results: List[Dict[str, Any]] = []

    if args.baseline:
        if os.path.exists(args.baseline):
            with open(args.baseline, "r", encoding="utf-8") as f:
                b_data = json.load(f)
            baseline_scorecards = [
                sc for sc in b_data.get("scorecards", [])
                if "ollama" in str(sc.get("runtime", "")).lower()
            ]
            baseline_results = [
                r for r in b_data.get("results", [])
                if "ollama" in str(r.get("runtime", "")).lower()
            ]
            console.print(f"[bold green]✔ Baseline loaded ({len(baseline_scorecards)} models) from:[/] [cyan]{args.baseline}[/]")
            console.print("[dim]Ollama models will NOT be re-executed; baseline data will be merged directly.[/]")
        else:
            console.print(f"[bold red]Baseline file not found:[/] [yellow]{args.baseline}[/]")
            sys.exit(1)

    if args.pair:
        pairs = config.get("model_pairs_1to1", [])
        matched_pair = next(
            (p for p in pairs if p.get("id") == args.pair or p.get("name") == args.pair),
            None,
        )
        if not matched_pair:
            avail = [p.get("id") for p in pairs]
            console.print(f"[bold red]Unknown 1:1 pair ID:[/] {args.pair}. Available: {avail}")
            sys.exit(1)

        console.print(f"[bold cyan]Selected 1:1 Pair:[/] [bold]{matched_pair.get('name')}[/]")
        if args.baseline:
            target_rt = "onnx-gpu" if selected_runtime == "onnx-gpu" else "foundry"
            target_model_name = matched_pair.get("onnx", matched_pair["foundry"]) if target_rt == "onnx-gpu" else matched_pair["foundry"]
            target_models = [(target_rt, target_model_name)]
        else:
            target_models = [("ollama", matched_pair["ollama"]), ("foundry", matched_pair["foundry"])]
        if args.suite == "all" and matched_pair.get("recommended_suite"):
            args.suite = matched_pair.get("recommended_suite")
    else:
        target_models = resolve_target_models(
            raw_models_arg=args.models,
            selected_runtime=selected_runtime,
            clients=clients,
        )

    if not target_models and not baseline_scorecards:
        console.print("[bold red]No models to benchmark! Run with --check, --pull-recommended, or specify --models.[/]")
        console.print("[dim]Hint: If testing MS Foundry, ensure Foundry Local server is running ('foundry service start').[/]")
        sys.exit(1)


    specs = get_system_specs(client=clients["ollama"])
    display_system_banner(specs)

    # Load scenarios
    speed_scenarios = load_scenario_file("scenarios/speed.json")
    coding_scenarios = load_scenario_file("scenarios/coding.json")
    reasoning_scenarios = load_scenario_file("scenarios/reasoning.json")
    context_scenarios = load_scenario_file("scenarios/context_scaling.json")
    polish_scenarios = load_scenario_file("scenarios/polish.json")

    suites_to_run = (
        ["speed", "coding", "reasoning", "polish", "context"]
        if args.suite == "all"
        else [args.suite]
    )

    models_display = [f"{rt}:{m}" for rt, m in target_models]
    console.print(f"\n[bold green]🚀 Starting benchmark for models:[/] {', '.join(models_display)}")
    console.print(f"[bold cyan]Selected test suites:[/] {', '.join(suites_to_run)}\n")

    all_raw_results: List[Dict[str, Any]] = []
    total_start = time.time()

    for runtime_name, model in target_models:
        client = clients.get(runtime_name)
        if not client:
            console.print(f"[bold red]Unknown runtime: {runtime_name}[/]")
            continue

        if not client.is_reachable():
            console.print(f"[bold yellow]⚠ Skipping {runtime_name}:{model} — {client.display_name} server is not responding.[/]")
            continue

        console.print(f"\n[bold yellow]━━━ [{client.display_name}: {model}] Starting evaluation ({client.engine_name}) ━━━[/]")

        # Ensure model is placed into memory in daemon
        console.print(f"  [dim]Ensuring model {model} is loaded ({client.display_name})...[/]")
        client.load_model(model)

        runner = BenchmarkRunner(
            client=client,
            config=config,
            progress_callback=lambda st, d: None,
        )

        # Warmup if configured
        rt_config = config.get(runtime_name, {})
        if rt_config.get("warmup", True):
            runner.warmup(model)

        for run_idx in range(args.runs):
            if args.runs > 1:
                console.print(f"  [dim]Run {run_idx + 1}/{args.runs}[/]")

            if "speed" in suites_to_run and speed_scenarios:
                console.print(f"  [bold]• Running speed & throughput tests...[/]")
                res = runner.run_speed_suite(model, speed_scenarios)
                for r in res:
                    display_scenario_result(r)
                all_raw_results.extend(res)

            if "coding" in suites_to_run and coding_scenarios:
                console.print(f"  [bold]• Running coding tests (automated unit test sandbox)...[/]")
                res = runner.run_coding_suite(model, coding_scenarios)
                for r in res:
                    display_scenario_result(r)
                all_raw_results.extend(res)

            if "reasoning" in suites_to_run and reasoning_scenarios:
                console.print(f"  [bold]• Running reasoning & logic tests...[/]")
                res = runner.run_reasoning_suite(model, reasoning_scenarios)
                for r in res:
                    display_scenario_result(r)
                all_raw_results.extend(res)

            if "polish" in suites_to_run and polish_scenarios:
                console.print(f"  [bold]• Running Polish multilingual NLP tests...[/]")
                res = runner.run_polish_suite(model, polish_scenarios)
                for r in res:
                    display_scenario_result(r)
                all_raw_results.extend(res)

            if "context" in suites_to_run and context_scenarios:
                console.print(f"  [bold]• Running context scaling suite (512 - 8k tokens)...[/]")
                res = runner.run_context_suite(model, context_scenarios)
                for r in res:
                    display_scenario_result(r)
                all_raw_results.extend(res)

        # Unload model after test to guarantee clean memory for next model
        if rt_config.get("unload_after_test", True):
            console.print(f"  [dim]Unloading memory for model {model} ({client.display_name})...[/]")
            client.unload_model(model)
            time.sleep(1.0)

    total_duration = time.time() - total_start

    # Merge baseline results if provided
    if baseline_results:
        all_raw_results = baseline_results + all_raw_results

    # Compute scorecards
    scorecards = []
    # Create a dummy runner for scorecard calculations
    dummy_runner = BenchmarkRunner(client=clients["ollama"], config=config)

    for runtime_name, model in target_models:
        sc = dummy_runner.compute_model_scorecard(model, all_raw_results, runtime=runtime_name)
        if sc:
            scorecards.append(sc)

    # Append baseline scorecards
    if baseline_scorecards:
        existing_keys = {(sc.get("runtime"), sc.get("model")) for sc in scorecards}
        for b_sc in baseline_scorecards:
            if (b_sc.get("runtime"), b_sc.get("model")) not in existing_keys:
                scorecards.append(b_sc)

    if not scorecards:
        console.print("[bold yellow]No benchmark results were recorded.[/]")
        return

    # Display results
    console.print("\n")
    display_leaderboard(scorecards, specs=specs)
    display_token_savings(scorecards)

    # If cross-engine models exist, display 1:1 comparison
    runtimes = set(str(sc.get("runtime", "")).lower() for sc in scorecards)
    has_other = any("ollama" not in rt for rt in runtimes)
    if len(scorecards) >= 2 and (any("ollama" in rt for rt in runtimes) and has_other):
        sc_ollama = next((sc for sc in scorecards if "ollama" in str(sc.get("runtime", "")).lower()), scorecards[0])
        sc_other = next((sc for sc in scorecards if "ollama" not in str(sc.get("runtime", "")).lower()), scorecards[1])
        res_ollama = [
            r for r in all_raw_results
            if r.get("model") == sc_ollama.get("model") or "ollama" in str(r.get("runtime", "")).lower()
        ]
        res_other = [
            r for r in all_raw_results
            if r.get("model") == sc_other.get("model") or str(r.get("runtime", "")).lower() == str(sc_other.get("runtime", "")).lower()
        ]
        other_rt_label = "ONNX GPU" if "onnx" in str(sc_other.get("runtime", "")).lower() else "MS Foundry"
        pair_title = f"{sc_ollama.get('model')} (Ollama) vs {sc_other.get('model')} ({other_rt_label})"
        display_1to1_comparison(sc_ollama, sc_other, res_ollama, res_other, pair_name=pair_title)
        comp_md_path = f"{args.output_dir}/1TO1_COMPARISON_REPORT.md"
        generate_1to1_comparison_report(
            sc_ollama, sc_other, res_ollama, res_other, specs, pair_name=pair_title, output_path=comp_md_path
        )
        console.print(f"  [cyan]1:1 Comparison Report:[/] [bold]{comp_md_path}[/]")


    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"{args.output_dir}/runs", exist_ok=True)
    raw_json_path = f"{args.output_dir}/runs/benchmark_{timestamp}.json"
    latest_json_path = f"{args.output_dir}/latest.json"

    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "specs": specs,
            "scorecards": scorecards,
            "results": all_raw_results,
            "total_duration_sec": total_duration,
        }, f, indent=2)

    with open(latest_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "scorecards": scorecards,
        }, f, indent=2)

    md_report_path = f"{args.output_dir}/LATEST_SUMMARY.md"
    generate_markdown_report(scorecards, all_raw_results, specs, output_path=md_report_path)

    total_tokens_saved_all = sum(sc.get("total_tokens_saved", 0) for sc in scorecards)
    total_cost_saved_all = sum(sc.get("est_cost_saved_usd", 0.0) for sc in scorecards)

    console.print(f"\n[bold green]✔ Benchmark completed in {total_duration:.1f}s![/]")
    console.print(f"  [cyan]Cloud Tokens Saved:[/] [bold green]{total_tokens_saved_all:,}[/] ([bold]~${total_cost_saved_all:.4f} USD[/] equivalent)")
    console.print(f"  [cyan]Markdown Report:[/] [bold]{md_report_path}[/]")
    console.print(f"  [cyan]Raw JSON Data:[/] [bold]{raw_json_path}[/]\n")


if __name__ == "__main__":
    main()
