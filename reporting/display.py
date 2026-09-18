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
        title=f"🏆 Local LLM Benchmark Leaderboard ({platform_desc})",
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
    table.add_column("Runtime", justify="center")
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
        rt = str(sc.get("runtime", "ollama")).lower()
        if "onnx" in rt:
            runtime_badge = "[bold magenta]ONNX GPU[/]"
        elif "foundry" in rt:
            runtime_badge = "[bold blue]MS Foundry[/]"
        else:
            runtime_badge = "[bold cyan]Ollama[/]"

        table.add_row(
            medal,
            f"[bold]{sc['model']}[/]",
            runtime_badge,
            f"{sc['composite_score']:.1f}",
            f"{sc['coding_pass_rate']:.1f}%",
            f"{sc['reasoning_accuracy']:.1f}%",
            f"{sc['avg_eval_tok_sec']:.1f} t/s",
            f"{sc.get('avg_prompt_tok_sec', sc.get('avg_prompt_eval_tok_sec', 0.0)):.1f} t/s",
            f"{sc['avg_ttft_sec']:.2f}s",
            f"{sc['peak_vram_mb']:.0f} MB",
            fit_status,
        )

    console.print(table)


def display_scenario_result(res: Dict[str, Any]):
    """Print one-line summary of scenario execution."""
    suite = res.get("suite", "")
    model = res.get("model", "")
    rt = res.get("runtime", "")
    if "onnx" in rt.lower():
        rt_name = "ONNX GenAI"
    elif "foundry" in rt.lower():
        rt_name = "MS Foundry"
    else:
        rt_name = "Ollama"
    rt_prefix = f"[{rt_name}: " if rt else "["
    model_tag = f"{rt_prefix}{model}]" if rt else f"[{model}]"
    name = res.get("name", "")
    tok_s = res.get("eval_tok_per_sec", 0.0)
    vram = res.get("hardware", {}).get("vram_peak_mb", 0.0)

    if not res.get("success", True):
        err = res.get("error") or res.get("sandbox_error") or "Request failed"
        console.print(f"  {model_tag} {name} -> [bold red]ERROR[/] ({err}) | Mem: {vram:.0f}MB")
        return

    if suite == "coding":
        status = "[bold green]PASS[/]" if res.get("passed") else "[bold red]FAIL[/]"
        tests = f"{res.get('passed_tests')}/{res.get('total_tests')} tests"
        console.print(f"  {model_tag} {name} -> {status} ({tests}) | {tok_s:.1f} t/s | Mem: {vram:.0f}MB")
    elif suite == "reasoning":
        status = "[bold green]PASS[/]" if res.get("correct") else "[bold red]FAIL[/]"
        think_info = " [dim](<think> tag)[/]" if res.get("has_think_tags") else ""
        console.print(f"  {model_tag} {name} -> {status}{think_info} | {tok_s:.1f} t/s | Mem: {vram:.0f}MB")
    elif suite == "polish":
        status = "[bold green]PASS[/]" if res.get("correct") else "[bold red]FAIL[/]"
        console.print(f"  {model_tag} {name} -> {status} | {tok_s:.1f} t/s | Mem: {vram:.0f}MB")
    elif suite == "context":
        ctx = res.get("context_size", 0)
        p_tok = res.get("prompt_tok_per_sec", 0.0)
        console.print(f"  {model_tag} {name} ({ctx} ctx) -> Prefill: {p_tok:.1f} t/s | Decode: {tok_s:.1f} t/s | Mem: {vram:.0f}MB")
    else:
        console.print(f"  {model_tag} {name} -> {tok_s:.1f} t/s | Mem: {vram:.0f}MB")


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


def display_1to1_comparison(
    sc_a: Dict[str, Any],
    sc_b: Dict[str, Any],
    results_a: List[Dict[str, Any]],
    results_b: List[Dict[str, Any]],
    pair_name: str = "1:1 Architecture Comparison",
):
    """Display a side-by-side terminal comparison between two 1:1 models across runtimes."""
    mod_a = sc_a.get("model", "Model A")
    rt_a = "Ollama (llama.cpp)" if sc_a.get("runtime") == "ollama" else "Model A"
    mod_b = sc_b.get("model", "Model B")
    rt_b = "MS Foundry (ONNX Runtime)" if sc_b.get("runtime") == "foundry" else "Model B"

    # 1. Summary comparison table
    table = Table(
        title=f"⚖️ 1:1 Cross-Engine Model Comparison: {pair_name}",
        header_style="bold yellow",
        show_header=True,
    )
    table.add_column("Evaluation Metric", style="cyan", justify="left")
    table.add_column(f"{mod_a}\n[dim]({rt_a})[/]", justify="right", style="white")
    table.add_column(f"{mod_b}\n[dim]({rt_b})[/]", justify="right", style="white")
    table.add_column("Delta / Advantage", justify="center", style="bold")

    # Speed metrics
    spd_a = sc_a.get("avg_eval_tok_sec", 0.0)
    spd_b = sc_b.get("avg_eval_tok_sec", 0.0)
    spd_ratio = (
        f"Ollama {spd_a / spd_b:.1f}x faster"
        if spd_b > 0 and spd_a >= spd_b
        else (f"Foundry {spd_b / spd_a:.1f}x faster" if spd_a > 0 else "N/A")
    )
    table.add_row("Decode Speed (tok/s)", f"{spd_a:.1f} t/s", f"{spd_b:.1f} t/s", f"[green]{spd_ratio}[/]")

    pref_a = sc_a.get("avg_prompt_tok_sec", 0.0)
    pref_b = sc_b.get("avg_prompt_tok_sec", 0.0)
    pref_ratio = (
        f"Ollama {pref_a / pref_b:.1f}x faster"
        if pref_b > 0 and pref_a >= pref_b
        else (f"Foundry {pref_b / pref_a:.1f}x faster" if pref_a > 0 else "N/A")
    )
    table.add_row("Prefill Speed (tok/s)", f"{pref_a:.1f} t/s", f"{pref_b:.1f} t/s", f"[green]{pref_ratio}[/]")

    ttft_a = sc_a.get("avg_ttft_sec", 0.0)
    ttft_b = sc_b.get("avg_ttft_sec", 0.0)
    ttft_adv = (
        f"Ollama {ttft_b / ttft_a:.1f}x lower"
        if ttft_a > 0 and ttft_a <= ttft_b
        else (f"Foundry {ttft_a / ttft_b:.1f}x lower" if ttft_b > 0 else "N/A")
    )
    table.add_row("Avg TTFT (Latency)", f"{ttft_a:.2f}s", f"{ttft_b:.2f}s", f"[cyan]{ttft_adv}[/]")

    code_a = sc_a.get("coding_pass_rate", 0.0)
    code_b = sc_b.get("coding_pass_rate", 0.0)
    code_delta = (
        f"[green]Foundry +{code_b - code_a:.1f}%[/]"
        if code_b > code_a
        else (f"[green]Ollama +{code_a - code_b:.1f}%[/]" if code_a > code_b else "Equal")
    )
    table.add_row("Coding Pass Rate", f"{code_a:.1f}%", f"{code_b:.1f}%", code_delta)

    vram_a = sc_a.get("peak_vram_mb", 0.0)
    vram_b = sc_b.get("peak_vram_mb", 0.0)
    table.add_row(
        "Peak Memory Usage",
        f"{vram_a:.0f} MB (VRAM)",
        f"{vram_b:.0f} MB (RAM/VRAM)",
        "[dim]GPU vs CPU/RAM[/]",
    )

    comp_a = sc_a.get("composite_score", 0.0)
    comp_b = sc_b.get("composite_score", 0.0)
    comp_lead = (
        f"Ollama (+{comp_a - comp_b:.1f})"
        if comp_a >= comp_b
        else f"Foundry (+{comp_b - comp_a:.1f})"
    )
    table.add_row("Composite Score", f"{comp_a:.1f}/100", f"{comp_b:.1f}/100", f"[bold yellow]{comp_lead}[/]")

    console.print("\n")
    console.print(table)

    # 2. Scenario-by-scenario test table
    tests_a = {r.get("test_id"): r for r in results_a}
    tests_b = {r.get("test_id"): r for r in results_b}
    all_test_ids = list(dict.fromkeys(list(tests_a.keys()) + list(tests_b.keys())))

    if all_test_ids:
        t_table = Table(
            title="🔬 Scenario-by-Scenario Assertion Breakdown",
            header_style="bold magenta",
            show_header=True,
        )
        t_table.add_column("Scenario Name", style="white", justify="left")
        t_table.add_column(f"{mod_a}\n[dim]Status (Pass/Total)[/]", justify="center")
        t_table.add_column(f"{mod_b}\n[dim]Status (Pass/Total)[/]", justify="center")
        t_table.add_column("Speed Comparison", justify="right", style="dim")

        for tid in all_test_ids:
            ra = tests_a.get(tid, {})
            rb = tests_b.get(tid, {})
            tname = ra.get("name") or rb.get("name") or tid

            status_a = (
                f"[green]PASS ({ra.get('passed_tests', 0)}/{ra.get('total_tests', 0)})[/]"
                if ra.get("passed")
                else f"[red]FAIL ({ra.get('passed_tests', 0)}/{ra.get('total_tests', 0)})[/]"
            )
            status_b = (
                f"[green]PASS ({rb.get('passed_tests', 0)}/{rb.get('total_tests', 0)})[/]"
                if rb.get("passed")
                else f"[red]FAIL ({rb.get('passed_tests', 0)}/{rb.get('total_tests', 0)})[/]"
            )
            spd_str = f"{ra.get('eval_tok_per_sec', 0.0):.1f} vs {rb.get('eval_tok_per_sec', 0.0):.1f} t/s"

            t_table.add_row(tname, status_a, status_b, spd_str)

        console.print(t_table)
        console.print("\n")
