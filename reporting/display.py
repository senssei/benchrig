"""Terminal UI and formatted reports using Rich."""

from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def display_system_banner(specs: Dict[str, str]):
    """Display system hardware specs banner."""
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(style="cyan", justify="right")
    grid.add_column(style="white")
    grid.add_column(style="cyan", justify="right")
    grid.add_column(style="white")

    gpu_type = specs.get("gpu_type", "")
    mem_label = "UMA Memory:" if gpu_type == "apple_silicon" else "VRAM Total:"
    mem_val = f"{specs.get('gpu_vram_total_mb', 'N/A')} MB ({specs.get('memory_type', 'VRAM')})"

    grid.add_row(
        "GPU / Accelerator:", f"[bold green]{specs.get('gpu_name', 'N/A')}[/]",
        mem_label, f"[bold]{mem_val}[/]"
    )
    grid.add_row(
        "CPU:", f"{specs.get('cpu_model', 'N/A')} ({specs.get('cpu_cores', 'N/A')} vCPUs)",
        "System RAM:", f"{specs.get('ram_total_gb', 'N/A')} GB"
    )
    grid.add_row(
        "Driver / Metal:", f"{specs.get('driver_version', 'N/A')}",
        "Platform:", f"[bold]{specs.get('platform', 'Unknown')}[/]"
    )

    console.print(
        Panel(
            grid,
            title="[bold blue]Ollama BenchRig - Hardware Profile[/]",
            border_style="blue",
        )
    )


def display_leaderboard(scorecards: List[Dict[str, Any]], specs: Optional[Dict[str, str]] = None):
    """Display the final ranked leaderboard in terminal."""
    if not scorecards:
        return

    specs = specs or {}
    platform_desc = specs.get("platform_short") or (
        "Apple Silicon (Metal)" if scorecards[0].get("gpu_type") == "apple_silicon" else "GPU / CUDA"
    )

    # Sort by composite score descending
    ranked = sorted(scorecards, key=lambda x: x.get("composite_score", 0), reverse=True)

    table = Table(
        title=f"🏆 Ollama Benchmark Leaderboard ({platform_desc})",
        header_style="bold magenta",
        show_header=True,
    )

    # Determine dynamic memory column header
    total_mb_str = specs.get("gpu_vram_total_mb") or str(ranked[0].get("total_vram_mb", 0))
    is_mac = specs.get("gpu_type") == "apple_silicon" or ranked[0].get("gpu_type") == "apple_silicon"
    mem_kind = "UMA" if is_mac else "VRAM"

    try:
        total_gb = float(total_mb_str) / 1024.0
        fit_col_title = f"{total_gb:.0f}GB {mem_kind} Fit" if total_gb >= 1.0 else f"{mem_kind} Fit"
    except (ValueError, TypeError):
        fit_col_title = f"{mem_kind} Fit"

    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Composite", justify="right", style="bold yellow")
    table.add_column("Coding Pass", justify="right", style="green")
    table.add_column("Reasoning", justify="right", style="blue")
    table.add_column("Eval Speed", justify="right", style="magenta")
    table.add_column("Prefill Speed", justify="right")
    table.add_column("Avg TTFT", justify="right")
    table.add_column(f"Peak {mem_kind}", justify="right")
    table.add_column(fit_col_title, justify="center")

    for idx, sc in enumerate(ranked, start=1):
        medal = "🥇 " if idx == 1 else ("🥈 " if idx == 2 else ("🥉 " if idx == 3 else f"#{idx} "))
        fit_status = (
            "[red]⚠️ Spill[/]" if sc.get("vram_warning") else ("[green]✅ 100% Metal[/]" if is_mac else "[green]✅ 100% GPU[/]")
        )

        table.add_row(
            medal,
            f"[bold]{sc['model']}[/]",
            f"{sc['composite_score']:.1f}",
            f"{sc['coding_pass_rate']:.1f}%",
            f"{sc['reasoning_accuracy']:.1f}%",
            f"{sc['avg_eval_tok_sec']:.1f} t/s",
            f"{sc['avg_prompt_tok_sec']:.1f} t/s",
            f"{sc['avg_ttft_sec']:.2f}s",
            f"{sc['peak_vram_mb']:.0f} MB",
            fit_status,
        )

    console.print(table)


def display_scenario_result(res: Dict[str, Any]):
    """Print one-line summary of scenario execution."""
    suite = res.get("suite", "")
    model = res.get("model", "")
    name = res.get("name", "")
    tok_s = res.get("eval_tok_per_sec", 0.0)
    vram = res.get("hardware", {}).get("vram_peak_mb", 0.0)

    if suite == "coding":
        status = "[bold green]PASS[/]" if res.get("passed") else "[bold red]FAIL[/]"
        tests = f"{res.get('passed_tests')}/{res.get('total_tests')} tests"
        console.print(f"  [{model}] {name} -> {status} ({tests}) | {tok_s:.1f} t/s | Mem: {vram:.0f}MB")
    elif suite == "reasoning":
        status = "[bold green]PASS[/]" if res.get("correct") else "[bold red]FAIL[/]"
        think_info = " [dim](<think> tag)[/]" if res.get("has_think_tags") else ""
        console.print(f"  [{model}] {name} -> {status}{think_info} | {tok_s:.1f} t/s | Mem: {vram:.0f}MB")
    elif suite == "polish":
        status = "[bold green]PASS[/]" if res.get("correct") else "[bold red]FAIL[/]"
        console.print(f"  [{model}] {name} -> {status} | {tok_s:.1f} t/s | Mem: {vram:.0f}MB")
    elif suite == "context":
        ctx = res.get("context_size", 0)
        p_tok = res.get("prompt_tok_per_sec", 0.0)
        console.print(f"  [{model}] {name} ({ctx} ctx) -> Prefill: {p_tok:.1f} t/s | Decode: {tok_s:.1f} t/s | Mem: {vram:.0f}MB")
    else:
        console.print(f"  [{model}] {name} -> {tok_s:.1f} t/s | Mem: {vram:.0f}MB")


def display_token_savings(scorecards: List[Dict[str, Any]]):
    """Render Rich table detailing local tokens processed and estimated cloud cost savings."""
    if not scorecards:
        return

    table = Table(
        title="⚡ Cloud Token & Cost Savings (via Local Coder Offloading)",
        header_style="bold green",
        show_header=True,
    )

    table.add_column("Model", style="cyan", justify="left")
    table.add_column("Prompt Ingested", justify="right", style="dim")
    table.add_column("Completion Tokens", justify="right")
    table.add_column("Total Tokens Saved", justify="right", style="bold green")
    table.add_column("Est. Cloud API Savings*", justify="right", style="bold yellow")
    table.add_column("Local Cost", justify="center", style="bold green")

    total_prompt = 0
    total_eval = 0
    total_saved = 0
    total_cost = 0.0

    for sc in scorecards:
        p_tok = sc.get("total_prompt_tokens", 0)
        e_tok = sc.get("total_eval_tokens", 0)
        s_tok = sc.get("total_tokens_saved", 0)
        c_usd = sc.get("est_cost_saved_usd", 0.0)

        total_prompt += p_tok
        total_eval += e_tok
        total_saved += s_tok
        total_cost += c_usd

        table.add_row(
            f"[bold]{sc['model']}[/]",
            f"{p_tok:,}",
            f"{e_tok:,}",
            f"{s_tok:,}",
            f"${c_usd:.4f}",
            "✅ $0.00",
        )

    if len(scorecards) > 1:
        table.add_section()
        table.add_row(
            "[bold]Total Offloaded[/]",
            f"[bold]{total_prompt:,}[/]",
            f"[bold]{total_eval:,}[/]",
            f"[bold]{total_saved:,}[/]",
            f"[bold]${total_cost:.4f}[/]",
            "[bold green]✅ $0.00[/]",
        )

    console.print(table)
    console.print(
        "[dim]* Est. savings compared against standard frontier coding models (Claude 3.5 Sonnet / GPT-4o: $3.00/1M prompt, $15.00/1M completion).[/]\n"
    )
