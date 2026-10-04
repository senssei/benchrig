"""``--check`` (environment diagnostics) and ``--pull-recommended`` paths.

Phase 12.4: this module was carved out of the monolithic ``benchrig/cli.py`` so each command
group lives in its own module. ``benchrig/cli/__init__.py`` re-exports the public entry points
so tests that do ``from benchrig.cli import run_system_check`` or
``patch("benchrig.cli.run_system_check")`` keep working unchanged.
"""

from __future__ import annotations

from typing import Any

import benchrig.cli as _cli_pkg
from benchrig.core.client import BaseRuntimeClient


def _recommended_models_for(rec_config: Any, runtime: str) -> list[str]:
    """Read recommended models for a runtime (supports the legacy flat list, meaning Ollama)."""
    if isinstance(rec_config, dict):
        return rec_config.get(runtime, [])
    if runtime == "ollama" and isinstance(rec_config, list):
        return rec_config
    return []


def _check_ollama(client: BaseRuntimeClient | None) -> None:
    if client and client.is_reachable():
        _cli_pkg._lookup("console").print(
            f"  [green]✔ Ollama REST API:[/] Available (Version: {client.get_version()}, llama.cpp) at {client.base_url}"
        )
        installed = client.list_installed_models()
        _cli_pkg._lookup("console").print(f"     Installed Ollama models ({len(installed)}):")
        for m in installed:
            size_gb = m.get("size", 0) / (1024**3)
            details = m.get("details", {})
            _cli_pkg._lookup("console").print(
                f"     • [cyan]{m.get('name', ''):<24}[/] "
                f"({size_gb:.1f} GB, {details.get('parameter_size', 'N/A')}, {details.get('quantization_level', 'N/A')})"
            )
    else:
        url = client.base_url if client else "http://localhost:11434"
        _cli_pkg._lookup("console").print(f"  [yellow]⚠ Ollama REST API:[/] Unreachable at {url}")


def _check_foundry(client: BaseRuntimeClient | None) -> None:
    if client and client.is_reachable():
        _cli_pkg._lookup("console").print(
            f"  [green]✔ MS Foundry Server REST API:[/] Available "
            f"(Version: {client.get_version()}, ONNX Runtime GenAI) at {client.base_url}"
        )
        installed = client.list_installed_models()
        _cli_pkg._lookup("console").print(f"     Available Foundry models ({len(installed)}):")
        for m in installed:
            details = m.get("details", {})
            _cli_pkg._lookup("console").print(
                f"     • [blue]{m.get('name', ''):<24}[/] "
                f"({details.get('parameter_size', 'ONNX')}, {details.get('quantization_level', 'ONNX')})"
            )
    else:
        url = client.base_url if client else "http://localhost:5272/v1"
        _cli_pkg._lookup("console").print(f"  [yellow]⚠ MS Foundry Server REST API:[/] Unreachable at {url}")


def _check_prism(client: BaseRuntimeClient | None) -> None:
    if client and client.is_reachable():
        models = client.list_installed_models()
        _cli_pkg._lookup("console").print(
            f"  [green]✔ Prism Server REST API:[/] Available ({client.get_version()}) at {client.base_url}"
        )
        _cli_pkg._lookup("console").print(f"     Available Prism models ({len(models)}):")
        for m in models:
            _cli_pkg._lookup("console").print(
                f"     • [green]{m.get('name', ''):<44}[/] ({m.get('details', {}).get('engine', 'N/A')})"
            )
    else:
        url = client.base_url if client else "http://127.0.0.1:5272/v1"
        _cli_pkg._lookup("console").print(
            f"  [yellow]ℹ Prism Server:[/] Not running at {url} (pip install prism-local && prism serve)"
        )


def _check_onnx(client: BaseRuntimeClient | None) -> None:
    if client and client.is_reachable():
        cuda_ok = getattr(client, "is_cuda_available", lambda: False)()
        cuda_status = "[bold green]CUDA Enabled[/]" if cuda_ok else "[yellow]CPU[/]"
        _cli_pkg._lookup("console").print(
            f"  [green]✔ ONNX GenAI Direct Engine:[/] Available (Version: {client.get_version()}, {cuda_status})"
        )
        installed = client.list_installed_models()
        _cli_pkg._lookup("console").print(f"     Available Direct ONNX Models ({len(installed)}):")
        for m in installed:
            quant = m.get("details", {}).get("quantization_level", "ONNX")
            _cli_pkg._lookup("console").print(f"     • [magenta]{m.get('name', ''):<28}[/] ({quant})")
    else:
        _cli_pkg._lookup("console").print(
            '  [yellow]ℹ ONNX GenAI Direct Engine:[/] Not installed (pip install "benchrig[onnx-gpu]")'
        )


def _check_accelerator(specs: dict[str, Any]) -> None:
    gpu_type = specs.get("gpu_type")
    gpu_name = specs.get("gpu_name")
    if gpu_type == "apple_silicon":
        _cli_pkg._lookup("console").print(
            f"\n  [green]✔ Apple Silicon Metal Accelerator:[/] Detected {gpu_name} "
            f"({specs.get('ram_total_gb')} GB Unified Memory)"
        )
    elif gpu_type == "nvidia":
        _cli_pkg._lookup("console").print(
            f"\n  [green]✔ NVIDIA GPU Access:[/] Detected {gpu_name} ({specs.get('gpu_vram_total_mb')} MB VRAM)"
        )
    elif gpu_name != "None":
        _cli_pkg._lookup("console").print(f"\n  [green]✔ Hardware Accelerator:[/] Detected {gpu_name}")
    else:
        _cli_pkg._lookup("console").print(
            "\n  [yellow]⚠ No dedicated hardware accelerator detected. Inference will run on CPU.[/]"
        )


def run_system_check(clients: dict[str, BaseRuntimeClient]) -> None:
    """Run diagnostics on Ollama, MS Foundry, platform, and GPU/Metal accelerator."""
    specs = _cli_pkg._lookup("get_system_specs")(client=clients.get("ollama"))
    _cli_pkg._lookup("display_system_banner")(specs)

    _cli_pkg._lookup("console").print("\n[bold cyan]🔍 Checking environment readiness across runtimes:[/]")
    _check_ollama(clients.get("ollama"))
    _check_foundry(clients.get("foundry"))
    _check_onnx(clients.get("onnx-gpu"))
    _cli_pkg._lookup("_check_prism")(clients.get("prism"))
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


def pull_recommended_models(clients: dict[str, BaseRuntimeClient], config: dict[str, Any], target_runtime: str) -> None:
    """Pull / download recommended models from active runtime registries."""
    rec_config = config.get("recommended_models", {})

    if target_runtime == "onnx-gpu":
        _cli_pkg._lookup("console").print("\n[bold yellow]━━━ Direct ONNX GenAI (onnx-gpu) Model Management ━━━[/]")
        client = clients.get("onnx-gpu")
        installed = [m.get("name") for m in client.list_installed_models()] if client else []
        if installed:
            _cli_pkg._lookup("console").print(
                f"  [green]✔ Direct ONNX model(s) already installed in `models/`:[/] [bold cyan]{', '.join(installed)}[/]"
            )
        else:
            _cli_pkg._lookup("console").print("  [yellow]ℹ No ONNX models found in `models/` or Foundry cache.[/]")

        _cli_pkg._lookup("console").print(
            "\n  [dim]Direct ONNX models are local checkpoint directories (HuggingFace), not managed by a daemon pull registry.[/]"
        )
        _cli_pkg._lookup("console").print("  [bold cyan]To download official ONNX CUDA weights via HuggingFace:[/] ")
        _cli_pkg._lookup("console").print(
            "    [bold white]huggingface-cli download microsoft/Phi-4-mini-instruct-onnx[/] \\"
        )
        _cli_pkg._lookup("console").print("    [bold white]  --include 'gpu/gpu-int4-rtn-block-32/*'[/] \\")
        _cli_pkg._lookup("console").print("    [bold white]  --local-dir models/Phi-4-mini-instruct-cuda-gpu[/]\n")
        _cli_pkg._lookup("console").print("  [bold cyan]To run benchmarks with the installed model:[/] ")
        _cli_pkg._lookup("console").print("    [bold white]benchrig --runtime onnx-gpu --suite coding[/]\n")
        return

    runtimes_to_pull = ["ollama", "foundry"] if target_runtime == "all" else [target_runtime]

    for rt in runtimes_to_pull:
        client = clients.get(rt)
        models = _recommended_models_for(rec_config, rt)
        if not client or not models:
            _cli_pkg._lookup("console").print(f"\n[yellow]⚠ No recommended models configured for runtime '{rt}'.[/]")
            continue

        _cli_pkg._lookup("console").print(
            f"\n[bold yellow]━━━ Checking recommended models for {client.display_name} ({client.engine_name}) ━━━[/]"
        )
        installed = [m.get("name") for m in client.list_installed_models()]

        for model in models:
            if any(model == inst or model in str(inst) for inst in installed):
                _cli_pkg._lookup("console").print(f"  [green]✔ Model {model} is already cached/installed.[/]")
                continue

            _cli_pkg._lookup("console").print(f"  [bold yellow]⬇ Downloading model {model}...[/]")
            success = client.pull_model(model, stream_callback=_print_pull_progress)
            print()
            if success:
                _cli_pkg._lookup("console").print(f"  [bold green]✔ Model {model} acquired successfully![/]")
            else:
                _cli_pkg._lookup("console").print(f"  [bold red]✖ Error acquiring model {model}.[/]")

    if target_runtime == "all":
        client_onnx = clients.get("onnx-gpu")
        installed_onnx = [m.get("name") for m in client_onnx.list_installed_models()] if client_onnx else []
        _cli_pkg._lookup("console").print("\n[bold yellow]━━━ Direct ONNX GenAI (onnx-gpu) Checkpoints ━━━[/]")
        if installed_onnx:
            _cli_pkg._lookup("console").print(
                f"  [green]✔ Direct ONNX model(s) available in `models/`:[/] [cyan]{', '.join(installed_onnx)}[/]"
            )
        else:
            _cli_pkg._lookup("console").print(
                "  [dim]No Direct ONNX models found in `models/`. Download via huggingface-cli.[/]"
            )


__all__ = [
    "run_system_check",
    "pull_recommended_models",
    "_check_ollama",
    "_check_foundry",
    "_check_prism",
    "_check_onnx",
    "_check_accelerator",
    "_print_pull_progress",
]
