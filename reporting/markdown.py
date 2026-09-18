"""Markdown report generator for Ollama benchmark results."""

import os
from datetime import datetime
from typing import Any, Dict, List


def generate_markdown_report(
    scorecards: List[Dict[str, Any]],
    raw_results: List[Dict[str, Any]],
    system_specs: Dict[str, str],
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
        f"# 📊 Ollama Model Benchmark Report ({platform_label})",
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
        lines.append(f"- 🥇 **Overall Leader (Composite Score)**: **`{best_overall['model']}`** (Score: **{best_overall['composite_score']:.1f}/100**)")
    if best_coding:
        lines.append(f"- 💻 **Top Coding Performer (Unit Tests Pass Rate)**: **`{best_coding['model']}`** (Passed Tests: **{best_coding['coding_pass_rate']:.1f}%**)")
    if best_reasoning:
        lines.append(f"- 🧠 **Top Reasoning Performer (Accuracy)**: **`{best_reasoning['model']}`** (Accuracy: **{best_reasoning['reasoning_accuracy']:.1f}%**)")
    if fastest_speed:
        lines.append(f"- ⚡ **Fastest Generation Speed**: **`{fastest_speed['model']}`** (**{fastest_speed['avg_eval_tok_sec']:.1f} tok/s**, TTFT: {fastest_speed['avg_ttft_sec']:.2f}s)")

    if is_mac:
        lines.extend([
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
        ])
    else:
        lines.extend([
            "",
            "> [!TIP]",
            "> **NVIDIA VRAM Sizing & Optimization Tip:**",
            "> 7B-8B models (as well as 12B models under moderate context windows) fit entirely into 12GB of VRAM, providing immediate zero-latency decoding. For larger models or extended context windows (KV cache), approaching the VRAM capacity may cause layer offloading to system RAM.",
            "",
            "---",
            "",
        ])

    lines.extend([
        "## 📈 Leaderboard",
        "",
        f"| Rank | Model | Composite Score | Coding (Pass %) | Reasoning (%) | Decode Speed (t/s) | Prefill Speed (t/s) | Avg TTFT | Peak {mem_kind} | {mem_kind} Status |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for idx, sc in enumerate(ranked, start=1):
        medal = "🥇 " if idx == 1 else ("🥈 " if idx == 2 else ("🥉 " if idx == 3 else f"{idx}."))
        vram_status = (
            "⚠️ Near Memory Limit" if sc.get("vram_warning") else ("✅ 100% Metal" if is_mac else "✅ 100% VRAM")
        )
        lines.append(
            f"| {medal} | **`{sc['model']}`** | **{sc['composite_score']:.1f}** | {sc['coding_pass_rate']:.1f}% | {sc['reasoning_accuracy']:.1f}% | {sc['avg_eval_tok_sec']:.1f} t/s | {sc['avg_prompt_tok_sec']:.1f} t/s | {sc['avg_ttft_sec']:.2f}s | {sc['peak_vram_mb']:.0f} MB | {vram_status} |"
        )

    # Detailed coding breakdown
    coding_tests = [r for r in raw_results if r.get("suite") == "coding"]
    if coding_tests:
        lines.extend([
            "",
            "---",
            "",
            "## 💻 Detailed Coding Task Results",
            "",
            f"| Model | Task | Status | Passed Assertions | Decode Speed | Peak {mem_kind} | Notes / Errors |",
            "|:---|:---|:---:|:---:|:---:|:---:|:---|",
        ])
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
        lines.extend([
            "",
            "---",
            "",
            "## 🧠 Detailed Reasoning Task Results",
            "",
            "| Model | Task | Status | <think> Mode | Decode Speed | Output / Extracted Answer |",
            "|:---|:---|:---:|:---:|:---:|:---|",
        ])
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
        lines.extend([
            "",
            "---",
            "",
            "## 🌐 Multilingual / Polish NLP Results",
            "",
            "| Model | Task | Status | Decode Speed | Output / Extracted Answer |",
            "|:---|:---|:---:|:---:|:---|",
        ])
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
        lines.extend([
            "",
            "---",
            "",
            f"## 📐 Context Scaling & {mem_kind} Saturation (512 - 8192 tokens)",
            "",
            f"| Model | Context Window | Prefill Speed | Decode Speed | TTFT | Peak {mem_kind} | {mem_kind} Utilization (%) |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])
        for r in context_tests:
            v_peak = r.get("hardware", {}).get("vram_peak_mb", 0)
            v_pct = r.get("hardware", {}).get("vram_peak_pct", 0)
            lines.append(
                f"| `{r['model']}` | {r.get('context_size', 0)} tok | {r.get('prompt_tok_per_sec', 0):.1f} t/s | {r.get('eval_tok_per_sec', 0):.1f} t/s | {r.get('ttft_sec', 0):.2f}s | {v_peak:.0f} MB | {v_pct:.1f}% |"
            )

    lines.extend([
        "",
        "---",
        "",
        "*Generated automatically by Ollama BenchRig.*",
        "",
    ])

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content
