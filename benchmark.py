#!/usr/bin/env python3
"""
Ollama Benchmark & Load Testing Rig (macOS Apple Silicon Metal / Linux NVIDIA CUDA).
Profiles and ranks local models by coding quality, reasoning, generation speed, and memory efficiency.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List
import yaml

from core.client import OllamaClient
from core.hardware import get_system_specs
from core.runner import BenchmarkRunner
from reporting.display import (
    console,
    display_leaderboard,
    display_scenario_result,
    display_system_banner,
    display_token_savings,
)
from reporting.markdown import generate_markdown_report


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


def run_system_check(client: OllamaClient):
    """Run diagnostics on Ollama, platform, and GPU/Metal accelerator."""
    specs = get_system_specs(client=client)
    display_system_banner(specs)

    console.print("\n[bold cyan]🔍 Checking environment readiness:[/]")
    # Ollama status
    if client.is_reachable():
        ver = client.get_version()
        console.print(f"  [green]✔ Ollama REST API:[/] Available (Version: {ver}) at {client.base_url}")
    else:
        console.print(f"  [bold red]✖ Ollama REST API:[/] Unreachable at {client.base_url}")
        sys.exit(1)

    # Installed models
    installed = client.list_installed_models()
    console.print(f"  [green]✔ Installed Ollama models ({len(installed)}):[/]")
    for m in installed:
        name = m.get("name", "")
        size_gb = m.get("size", 0) / (1024**3)
        param_sz = m.get("details", {}).get("parameter_size", "N/A")
        quant = m.get("details", {}).get("quantization_level", "N/A")
        console.print(f"     • [cyan]{name:<24}[/] ({size_gb:.1f} GB, {param_sz}, {quant})")

    # Accelerator / GPU
    gpu_type = specs.get("gpu_type")
    if gpu_type == "apple_silicon":
        console.print(f"  [green]✔ Apple Silicon Metal Accelerator:[/] Detected {specs.get('gpu_name')} ({specs.get('ram_total_gb')} GB Unified Memory)")
    elif gpu_type == "nvidia":
        console.print(f"  [green]✔ NVIDIA GPU Access:[/] Detected {specs.get('gpu_name')} ({specs.get('gpu_vram_total_mb')} MB VRAM)")
    elif specs.get("gpu_name") != "None":
        console.print(f"  [green]✔ Hardware Accelerator:[/] Detected {specs.get('gpu_name')}")
    else:
        console.print("  [yellow]⚠ No dedicated hardware accelerator detected. Inference will run on CPU.[/]")


def pull_recommended_models(client: OllamaClient, models: List[str]):
    """Pull missing recommended models from Ollama."""
    installed = [m.get("name") for m in client.list_installed_models()]
    for model in models:
        # Check if already installed
        if any(model == inst or model in inst for inst in installed):
            console.print(f"[green]✔ Model {model} is already installed.[/]")
            continue

        console.print(f"\n[bold yellow]⬇ Pulling model {model}... This may take several minutes.[/]")

        def progress_cb(data):
            status = data.get("status", "")
            completed = data.get("completed", 0)
            total = data.get("total", 0)
            if total > 0:
                pct = (completed / total) * 100
                print(f"\r  {status}: {pct:.1f}% ({completed//(1024*1024)}MB / {total//(1024*1024)}MB)", end="", flush=True)
            else:
                print(f"\r  {status}", end="", flush=True)

        success = client.pull_model(model, stream_callback=progress_cb)
        print()
        if success:
            console.print(f"[bold green]✔ Model {model} pulled successfully![/]")
        else:
            console.print(f"[bold red]✖ Error pulling model {model}.[/]")


def main():
    parser = argparse.ArgumentParser(
        description="Ollama Benchmark & Load Testing Rig (macOS Apple Silicon Metal / Linux NVIDIA CUDA)"
    )
    parser.add_argument(
        "--check", action="store_true", help="Run environment diagnostics (Ollama, GPU/Metal, models)"
    )
    parser.add_argument(
        "--pull-recommended", action="store_true", help="Pull recommended models from Ollama library"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="installed",
        help="Models to benchmark (comma-separated or 'installed', default: all installed)",
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

    args = parser.parse_args()

    config = load_config("config.yaml")
    base_url = config.get("ollama", {}).get("base_url", "http://localhost:11434")
    timeout = config.get("ollama", {}).get("timeout_sec", 180)
    client = OllamaClient(base_url=base_url, timeout_sec=timeout)

    # 1. System check mode
    if args.check:
        run_system_check(client)
        return

    # 2. Pull recommended mode
    if args.pull_recommended:
        recommended = config.get("recommended_models", ["llama3.1:8b", "mistral:7b"])
        pull_recommended_models(client, recommended)
        return

    # 3. Setup models to test
    installed = client.list_installed_models()
    installed_names = [m.get("name") for m in installed]

    if args.models == "installed" or args.models == "all":
        target_models = installed_names
    else:
        target_models = [m.strip() for m in args.models.split(",") if m.strip()]

    if not target_models:
        console.print("[bold red]No models to benchmark! Run with --check or specify --models.[/]")
        sys.exit(1)

    specs = get_system_specs(client=client)
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

    console.print(f"\n[bold green]🚀 Starting benchmark for models:[/] {', '.join(target_models)}")
    console.print(f"[bold cyan]Selected test suites:[/] {', '.join(suites_to_run)}\n")

    all_raw_results: List[Dict[str, Any]] = []
    runner = BenchmarkRunner(
        client=client,
        config=config,
        progress_callback=lambda st, d: None,
    )

    total_start = time.time()

    for model in target_models:
        console.print(f"\n[bold yellow]━━━ Starting tests for model: {model} ━━━[/]")

        # Warmup if configured
        if config.get("ollama", {}).get("warmup", True):
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
        if config.get("ollama", {}).get("unload_after_test", True):
            console.print(f"  [dim]Unloading memory for model {model}...[/]")
            client.unload_model(model)
            time.sleep(1.0)

    total_duration = time.time() - total_start

    # Compute scorecards
    scorecards = []
    for model in target_models:
        sc = runner.compute_model_scorecard(model, all_raw_results)
        if sc:
            scorecards.append(sc)

    # Display results
    console.print("\n")
    display_leaderboard(scorecards, specs=specs)
    display_token_savings(scorecards)

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
