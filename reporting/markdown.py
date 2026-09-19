"""Markdown report generator for Ollama benchmark results."""

import os
from datetime import datetime
from typing import Any


def generate_markdown_report(
    scorecards: list[dict[str, Any]],
    raw_results: list[dict[str, Any]],
    system_specs: dict[str, str],
    output_path: str = "results/LATEST_SUMMARY.md",
) -> str:
    """Generate comprehensive Markdown report with tables and recommendations."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ranked = sorted(scorecards, key=lambda x: x.get("composite_score", 0), reverse=True)

    # Detect category winners
    best_overall = ranked[0] if ranked else None
    best_coding = max(scorecards, key=lambda x: x.get("coding_pass_rate", 0), default=None)
    best_reasoning = max(scorecards, key=lambda x: x.get("reasoning_accuracy", 0), default=None)
    fastest_speed = max(scorecards, key=lambda x: x.get("avg_eval_tok_sec", 0), default=None)

    platform_label = system_specs.get("platform_short") or system_specs.get("platform", "Local LLM")
    gpu_type = system_specs.get("gpu_type", "")
    is_mac = gpu_type == "apple_silicon"
    mem_kind = "UMA" if is_mac else "VRAM"

    lines = [
        f"# 📊 Local LLM Benchmark Report ({platform_label})",
        "",
        f"**Test Date**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`  ",
        f"**GPU / Accelerator**: `{system_specs.get('gpu_name', 'Unknown')}` ({system_specs.get('gpu_vram_total_mb', '0')} MB {system_specs.get('memory_type', 'VRAM')})  ",
        f"**CPU**: `{system_specs.get('cpu_model', 'Unknown')}` ({system_specs.get('cpu_cores', 'Unknown')} threads/cores)  ",
        f"**System RAM**: `{system_specs.get('ram_total_gb', 'Unknown')} GB`  ",
        f"**Environment**: `{system_specs.get('platform', 'Unknown')}` (`{system_specs.get('driver_version', 'N/A')}`)  ",
        "",
        "---",
        "",
        f"## 🏆 Highlights & Recommendations ({platform_label})",
        "",
    ]

    if best_overall:
        lines.append(
            f"- 🥇 **Overall Leader (Composite Score)**: **`{best_overall['model']}`** [{best_overall.get('runtime', 'ollama')}] (Score: **{best_overall['composite_score']:.1f}/100**)"
        )
    if best_coding:
        lines.append(
            f"- 💻 **Top Coding Performer (Unit Tests Pass Rate)**: **`{best_coding['model']}`** [{best_coding.get('runtime', 'ollama')}] (Passed Tests: **{best_coding['coding_pass_rate']:.1f}%**)"
        )
    if best_reasoning:
        lines.append(
            f"- 🧠 **Top Reasoning Performer (Accuracy)**: **`{best_reasoning['model']}`** [{best_reasoning.get('runtime', 'ollama')}] (Accuracy: **{best_reasoning['reasoning_accuracy']:.1f}%**)"
        )
    if fastest_speed:
        lines.append(
            f"- ⚡ **Fastest Generation Speed**: **`{fastest_speed['model']}`** [{fastest_speed.get('runtime', 'ollama')}] (**{fastest_speed['avg_eval_tok_sec']:.1f} tok/s**, TTFT: {fastest_speed['avg_ttft_sec']:.2f}s)"
        )

    if is_mac:
        lines.extend(
            [
                "",
                "> [!TIP]",
                "> **Apple Silicon Unified Memory (UMA) & Metal Performance Tip:**",
                "> Thanks to Unified Memory Architecture (UMA), Apple Silicon M-Series processors avoid PCIe bandwidth bottlenecks by allowing the Metal GPU direct access to system memory.",
                "> Depending on your Mac's RAM configuration:",
                "> - 7B-8B models (Q4_K_M ~4.5 GB) run smoothly on any Mac (even 8GB-16GB configurations).",
                "> - 14B models (Q4_K_M ~9 GB) and 32B models (~20 GB) fit comfortably on 24GB, 36GB, or larger unified memory setups, offering zero token fees and complete local privacy.",
                "",
                "---",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "> [!TIP]",
                "> **NVIDIA VRAM Sizing & Optimization Tip:**",
                "> 7B-8B models (as well as 12B models under moderate context windows) fit entirely into 12GB of VRAM, providing immediate zero-latency decoding. For larger models or extended context windows (KV cache), approaching the VRAM capacity may cause layer offloading to system RAM.",
                "",
                "---",
                "",
            ]
        )

    lines.extend(
        [
            "## 📈 Leaderboard",
            "",
            f"| Rank | Model | Runtime | Engine | Composite Score | Coding (Pass %) | Reasoning (%) | Decode Speed (t/s) | Prefill Speed (t/s) | Avg TTFT | Peak {mem_kind} | {mem_kind} Status |",
            "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
    )

    for idx, sc in enumerate(ranked, start=1):
        medal = "🥇 " if idx == 1 else ("🥈 " if idx == 2 else ("🥉 " if idx == 3 else f"{idx}."))
        vram_status = (
            "⚠️ Near Memory Limit" if sc.get("vram_warning") else ("✅ 100% Metal" if is_mac else "✅ 100% VRAM")
        )
        rt = sc.get("runtime", "ollama")
        rt_display = "MS Foundry" if "foundry" in str(rt).lower() else "Ollama"
        engine_display = sc.get("engine", "llama.cpp" if rt_display == "Ollama" else "ONNX Runtime")
        lines.append(
            f"| {medal} | **`{sc['model']}`** | `{rt_display}` | {engine_display} | **{sc['composite_score']:.1f}** | {sc['coding_pass_rate']:.1f}% | {sc['reasoning_accuracy']:.1f}% | {sc['avg_eval_tok_sec']:.1f} t/s | {sc['avg_prompt_tok_sec']:.1f} t/s | {sc['avg_ttft_sec']:.2f}s | {sc['peak_vram_mb']:.0f} MB | {vram_status} |"
        )

    # Token & Cloud Cost Savings Breakdown
    lines.extend(
        [
            "",
            "---",
            "",
            "## ⚡ Cloud Token & Cost Savings (via Local Coder Offloading)",
            "",
            "By executing coding evaluations and assistant routines on local accelerators, cloud API quota consumption is completely eliminated:",
            "",
            "| Model | Prompt Tokens Offloaded | Tokens Generated | Total Cloud Tokens Saved | Est. Cloud API Savings* | Effective Cloud Spend |",
            "|:---|:---:|:---:|:---:|:---:|:---:|",
        ]
    )
    for sc in ranked:
        p_tok = sc.get("total_prompt_tokens", 0)
        e_tok = sc.get("total_eval_tokens", 0)
        s_tok = sc.get("total_tokens_saved", 0)
        c_usd = sc.get("est_cost_saved_usd", 0.0)
        lines.append(
            f"| **`{sc['model']}`** | {p_tok:,} | {e_tok:,} | **{s_tok:,}** | **${c_usd:.4f}** | **✅ $0.00** |"
        )
    # Cross-runtime comparison if multiple runtimes evaluated
    runtimes_present = {sc.get("runtime", "ollama") for sc in scorecards}
    if len(runtimes_present) > 1:
        lines.extend(
            [
                "",
                "---",
                "",
                "## ⚖️ Engine Architecture Comparison: Ollama (`llama.cpp`) vs MS Foundry (`ONNX Runtime GenAI`)",
                "",
                "| Metric | Ollama (`llama.cpp`) | MS Foundry (`ONNX Runtime`) |",
                "|:---|:---:|:---:|",
            ]
        )
        ollama_scs = [sc for sc in scorecards if sc.get("runtime") == "ollama"]
        foundry_scs = [sc for sc in scorecards if "foundry" in str(sc.get("runtime", "")).lower()]
        avg_ollama_speed = (
            sum(sc.get("avg_eval_tok_sec", 0) for sc in ollama_scs) / len(ollama_scs) if ollama_scs else 0.0
        )
        avg_foundry_speed = (
            sum(sc.get("avg_eval_tok_sec", 0) for sc in foundry_scs) / len(foundry_scs) if foundry_scs else 0.0
        )
        avg_ollama_ttft = sum(sc.get("avg_ttft_sec", 0) for sc in ollama_scs) / len(ollama_scs) if ollama_scs else 0.0
        avg_foundry_ttft = (
            sum(sc.get("avg_ttft_sec", 0) for sc in foundry_scs) / len(foundry_scs) if foundry_scs else 0.0
        )
        avg_ollama_pass = (
            sum(sc.get("coding_pass_rate", 0) for sc in ollama_scs) / len(ollama_scs) if ollama_scs else 0.0
        )
        avg_foundry_pass = (
            sum(sc.get("coding_pass_rate", 0) for sc in foundry_scs) / len(foundry_scs) if foundry_scs else 0.0
        )
        lines.extend(
            [
                f"| **Average Decode Speed** | {avg_ollama_speed:.1f} t/s | {avg_foundry_speed:.1f} t/s |",
                f"| **Average Time to First Token (TTFT)** | {avg_ollama_ttft:.2f}s | {avg_foundry_ttft:.2f}s |",
                f"| **Average Coding Pass Rate** | {avg_ollama_pass:.1f}% | {avg_foundry_pass:.1f}% |",
                f"| **Evaluated Models** | {len(ollama_scs)} | {len(foundry_scs)} |",
            ]
        )

    # Detailed coding breakdown
    coding_tests = [r for r in raw_results if r.get("suite") == "coding"]
    if coding_tests:
        lines.extend(
            [
                "",
                "---",
                "",
                "## 💻 Detailed Coding Task Results",
                "",
                f"| Model | Task | Status | Passed Assertions | Decode Speed | Peak {mem_kind} | Notes / Errors |",
                "|:---|:---|:---:|:---:|:---:|:---:|:---|",
            ]
        )
        for r in coding_tests:
            status_badge = "✅ PASS" if r.get("passed") else "❌ FAIL"
            tests_str = f"{r.get('passed_tests', 0)}/{r.get('total_tests', 0)}"
            err = r.get("sandbox_error", "") or "-"
            if len(err) > 50:
                err = err[:47] + "..."
            lines.append(
                f"| `{r['model']}` | {r['name']} | {status_badge} | {tests_str} | {r.get('eval_tok_per_sec', 0):.1f} t/s | {r.get('hardware', {}).get('vram_peak_mb', 0):.0f} MB | `{err}` |"
            )

    # Detailed reasoning breakdown
    reasoning_tests = [r for r in raw_results if r.get("suite") == "reasoning"]
    if reasoning_tests:
        lines.extend(
            [
                "",
                "---",
                "",
                "## 🧠 Detailed Reasoning Task Results",
                "",
                "| Model | Task | Status | <think> Mode | Decode Speed | Output / Extracted Answer |",
                "|:---|:---|:---:|:---:|:---:|:---|",
            ]
        )
        for r in reasoning_tests:
            status_badge = "✅ CORRECT" if r.get("correct") else "❌ INCORRECT"
            think_badge = f"Yes ({r.get('think_chars', 0)} chars)" if r.get("has_think_tags") else "No"
            ans = str(r.get("extracted_answer", "")).replace("\n", " ")
            if len(ans) > 40:
                ans = ans[:37] + "..."
            lines.append(
                f"| `{r['model']}` | {r['name']} | {status_badge} | {think_badge} | {r.get('eval_tok_per_sec', 0):.1f} t/s | `{ans}` |"
            )

    # Detailed Polish NLP breakdown
    polish_tests = [r for r in raw_results if r.get("suite") == "polish"]
    if polish_tests:
        lines.extend(
            [
                "",
                "---",
                "",
                "## 🌐 Multilingual / Polish NLP Results",
                "",
                "| Model | Task | Status | Decode Speed | Output / Extracted Answer |",
                "|:---|:---|:---:|:---:|:---|",
            ]
        )
        for r in polish_tests:
            status_badge = "✅ CORRECT" if r.get("correct") else "❌ INCORRECT"
            ans = str(r.get("extracted_answer", "")).replace("\n", " ")
            if len(ans) > 50:
                ans = ans[:47] + "..."
            lines.append(
                f"| `{r['model']}` | {r['name']} | {status_badge} | {r.get('eval_tok_per_sec', 0):.1f} t/s | `{ans}` |"
            )

    # Context scaling breakdown
    context_tests = [r for r in raw_results if r.get("suite") == "context"]
    if context_tests:
        lines.extend(
            [
                "",
                "---",
                "",
                f"## 📐 Context Scaling & {mem_kind} Saturation (512 - 8192 tokens)",
                "",
                f"| Model | Context Window | Prefill Speed | Decode Speed | TTFT | Peak {mem_kind} | {mem_kind} Utilization (%) |",
                "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
            ]
        )
        for r in context_tests:
            v_peak = r.get("hardware", {}).get("vram_peak_mb", 0)
            v_pct = r.get("hardware", {}).get("vram_peak_pct", 0)
            lines.append(
                f"| `{r['model']}` | {r.get('context_size', 0)} tok | {r.get('prompt_tok_per_sec', 0):.1f} t/s | {r.get('eval_tok_per_sec', 0):.1f} t/s | {r.get('ttft_sec', 0):.2f}s | {v_peak:.0f} MB | {v_pct:.1f}% |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "*Generated automatically by Ollama BenchRig.*",
            "",
        ]
    )

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content


def generate_1to1_comparison_report(
    scorecard_a: dict[str, Any],
    scorecard_b: dict[str, Any],
    raw_results_a: list[dict[str, Any]],
    raw_results_b: list[dict[str, Any]],
    system_specs: dict[str, str],
    pair_name: str = "Phi-3.5 Mini vs Phi-3 Mini (3.8B)",
    output_path: str = "results/1TO1_COMPARISON_REPORT.md",
) -> str:
    """Generate comprehensive 1:1 cross-engine comparison Markdown report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    platform_label = system_specs.get("platform_short") or system_specs.get("platform", "Local LLM")
    mod_a = scorecard_a.get("model", "Model A")
    rt_a = "Ollama" if scorecard_a.get("runtime") == "ollama" else scorecard_a.get("runtime", "A")
    eng_a = scorecard_a.get("engine", "llama.cpp")

    mod_b = scorecard_b.get("model", "Model B")
    rt_b = "MS Foundry" if scorecard_b.get("runtime") == "foundry" else scorecard_b.get("runtime", "B")
    eng_b = scorecard_b.get("engine", "ONNX Runtime GenAI")

    spd_a = scorecard_a.get("avg_eval_tok_sec", 0.0)
    spd_b = scorecard_b.get("avg_eval_tok_sec", 0.0)
    pref_a = scorecard_a.get("avg_prompt_tok_sec", 0.0)
    pref_b = scorecard_b.get("avg_prompt_tok_sec", 0.0)
    ttft_a = scorecard_a.get("avg_ttft_sec", 0.0)
    ttft_b = scorecard_b.get("avg_ttft_sec", 0.0)
    code_a = scorecard_a.get("coding_pass_rate", 0.0)
    code_b = scorecard_b.get("coding_pass_rate", 0.0)
    vram_a = scorecard_a.get("peak_vram_mb", 0.0)
    vram_b = scorecard_b.get("peak_vram_mb", 0.0)

    lines = [
        f"# ⚖️ 1:1 Model Comparison: `{mod_a}` vs `{mod_b}`",
        "",
        f"**Benchmark Pair**: {pair_name}  ",
        f"**Platform**: `{platform_label}` (`{system_specs.get('driver_version', 'N/A')}`)  ",
        f"**Hardware**: `{system_specs.get('gpu_name', 'Unknown')}` ({system_specs.get('gpu_vram_total_mb', '0')} MB VRAM) | `{system_specs.get('cpu_model', 'Unknown')}` ({system_specs.get('cpu_cores', 'Unknown')} vCPUs) | `{system_specs.get('ram_total_gb', 'Unknown')} GB RAM`  ",
        f"**Report Generated**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`  ",
        "",
        "---",
        "",
        "## 📊 Executive Summary & Head-to-Head Scorecard",
        "",
        f"| Metric | `{mod_a}` [{rt_a} / {eng_a}] | `{mod_b}` [{rt_b} / {eng_b}] | Delta / Advantage |",
        "|:---|:---:|:---:|:---:|",
        f"| **Decode Speed (Generation)** | **{spd_a:.1f} tok/s** | **{spd_b:.1f} tok/s** | "
        + (
            f"`{rt_a}` is **{spd_a / spd_b:.1f}x faster**"
            if spd_b > 0 and spd_a >= spd_b
            else f"`{rt_b}` is **{spd_b / spd_a:.1f}x faster**"
        )
        + " |",
        f"| **Prompt Prefill Speed** | **{pref_a:.1f} tok/s** | **{pref_b:.1f} tok/s** | "
        + (
            f"`{rt_a}` is **{pref_a / pref_b:.1f}x faster**"
            if pref_b > 0 and pref_a >= pref_b
            else f"`{rt_b}` is **{pref_b / pref_a:.1f}x faster**"
        )
        + " |",
        f"| **Time to First Token (TTFT)** | **{ttft_a:.2f}s** | **{ttft_b:.2f}s** | "
        + (
            f"`{rt_a}` has **{ttft_b / ttft_a:.1f}x lower latency**"
            if ttft_a > 0 and ttft_a <= ttft_b
            else f"`{rt_b}` has lower latency"
        )
        + " |",
        f"| **Coding Unit Test Pass Rate** | **{code_a:.1f}%** | **{code_b:.1f}%** | "
        + (
            f"`{rt_b}` leads by **+{code_b - code_a:.1f}%**"
            if code_b > code_a
            else f"`{rt_a}` leads by **+{code_a - code_b:.1f}%**"
        )
        + " |",
        f"| **Peak Memory Footprint** | **{vram_a:.0f} MB** (VRAM) | **{vram_b:.0f} MB** (RAM/VRAM) | Dedicated VRAM vs System RAM |",
        f"| **Composite Benchmark Score** | **{scorecard_a.get('composite_score', 0):.1f}/100** | **{scorecard_b.get('composite_score', 0):.1f}/100** | "
        + (
            f"`{rt_a}` (+{scorecard_a.get('composite_score', 0) - scorecard_b.get('composite_score', 0):.1f})"
            if scorecard_a.get("composite_score", 0) >= scorecard_b.get("composite_score", 0)
            else f"`{rt_b}` (+{scorecard_b.get('composite_score', 0) - scorecard_a.get('composite_score', 0):.1f})"
        )
        + " |",
        "",
        "---",
        "",
        "## 🔬 Scenario-by-Scenario Task Breakdown",
        "",
        f"| Scenario Name | `{mod_a}` [{rt_a}] Status | `{mod_b}` [{rt_b}] Status | Speed Comparison | Error / Diagnostic Notes |",
        "|:---|:---:|:---:|:---:|:---|",
    ]

    tests_a = {r.get("test_id"): r for r in raw_results_a}
    tests_b = {r.get("test_id"): r for r in raw_results_b}
    all_test_ids = list(dict.fromkeys(list(tests_a.keys()) + list(tests_b.keys())))

    for tid in all_test_ids:
        ra = tests_a.get(tid, {})
        rb = tests_b.get(tid, {})
        tname = ra.get("name") or rb.get("name") or tid

        status_a = (
            f"✅ PASS ({ra.get('passed_tests', 0)}/{ra.get('total_tests', 0)})"
            if ra.get("passed")
            else f"❌ FAIL ({ra.get('passed_tests', 0)}/{ra.get('total_tests', 0)})"
        )
        status_b = (
            f"✅ PASS ({rb.get('passed_tests', 0)}/{rb.get('total_tests', 0)})"
            if rb.get("passed")
            else f"❌ FAIL ({rb.get('passed_tests', 0)}/{rb.get('total_tests', 0)})"
        )
        spd_str = f"{ra.get('eval_tok_per_sec', 0.0):.1f} vs {rb.get('eval_tok_per_sec', 0.0):.1f} t/s"

        err_notes = []
        if ra.get("sandbox_error"):
            err_notes.append(f"**{rt_a}**: `{str(ra['sandbox_error'])[:60]}...`")
        if rb.get("sandbox_error"):
            err_notes.append(f"**{rt_b}**: `{str(rb['sandbox_error'])[:60]}...`")
        err_str = "<br>".join(err_notes) if err_notes else "Clean execution"

        lines.append(f"| **{tname}** | {status_a} | {status_b} | {spd_str} | {err_str} |")

    is_b_gpu = (
        "cuda" in mod_b.lower()
        or "gpu" in rt_b.lower()
        or "cuda" in str(scorecard_b.get("engine", "")).lower()
        or spd_b > 50
    )
    lines.extend(
        [
            "",
            "---",
            "",
            "## 💡 Architectural Insights & Trade-offs",
            "",
            "1. **Execution Provider & Acceleration**:",
            f"   - **{rt_a} (`{eng_a}`)**: Evaluated via CUDA kernels on host GPU ({vram_a:.0f} MB peak VRAM, {spd_a:.1f} tok/s decode, {ttft_a:.2f}s TTFT).",
            f"   - **{rt_b} (`{eng_b}`)**: Evaluated via {'CUDA Execution Provider (GPU)' if is_b_gpu else 'CPU Execution Provider'} ({vram_b:.0f} MB peak memory, {spd_b:.1f} tok/s decode, {ttft_b:.2f}s TTFT).",
            "",
            "2. **Coding Quality Nuance**:",
            f"   - `{mod_b}` achieved {code_b:.1f}% test pass rate across evaluated unit test sandbox suites.",
            f"   - `{mod_a}` achieved {code_a:.1f}% test pass rate across evaluated unit test sandbox suites.",
            "",
            "3. **Zero Token Cost Economics**:",
            "   - Both engines ran completely locally on host hardware with zero API token spend.",
            f"   - Total offloaded tokens: **{scorecard_a.get('total_tokens_saved', 0) + scorecard_b.get('total_tokens_saved', 0):,} tokens** (~${scorecard_a.get('est_cost_saved_usd', 0.0) + scorecard_b.get('est_cost_saved_usd', 0.0):.4f} USD frontier cloud equivalent saved).",
            "",
            "---",
            "",
            "*Generated automatically by Ollama BenchRig 1:1 Cross-Engine Analyzer.*",
            "",
        ]
    )

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content
