"""``--compare`` path: offline analysis of a cached benchmark run.

Phase 12.4: carved out of the monolithic ``benchrig/cli.py``. ``benchrig.cli:main`` dispatches
to ``run_compare_mode`` when ``--compare`` is supplied; ``run.py`` also calls
``show_1to1_comparison`` after a benchmark run that produced both an Ollama and a non-Ollama
scorecard.
"""

from __future__ import annotations

import re
from typing import Any

import benchrig.cli as _cli_pkg
from benchrig.cli._common import is_ollama, load_json_or_exit
from benchrig.core.client import BaseRuntimeClient
from benchrig.core.runtimes import runtime_label
from benchrig.reporting.display import (
    display_1to1_comparison,
)
from benchrig.reporting.markdown import generate_1to1_comparison_report


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
    sc_ollama, sc_other = _cli_pkg._lookup("select_pair")(scorecards)
    res_ollama = _cli_pkg._lookup("records_for")(results, sc_ollama)
    res_other = _cli_pkg._lookup("records_for")(results, sc_other)

    other_label = runtime_label(sc_other.get("runtime"), default=str(sc_other.get("runtime", "other")))
    pair_title = f"{sc_ollama.get('model')} (Ollama) vs {sc_other.get('model')} ({other_label})"
    display_1to1_comparison(sc_ollama, sc_other, res_ollama, res_other, pair_name=pair_title)

    report_path = f"{output_dir}/1TO1_COMPARISON_REPORT.md"
    generate_1to1_comparison_report(
        sc_ollama, sc_other, res_ollama, res_other, specs, pair_name=pair_title, output_path=report_path
    )
    _cli_pkg._lookup("console").print(f"[bold green]✔ 1:1 comparison report written to:[/] [cyan]{report_path}[/]\n")


def run_compare_mode(compare_path: str, output_dir: str, clients: dict[str, BaseRuntimeClient]) -> None:
    """Offline analysis of a cached benchmark run (no models are executed)."""
    data = load_json_or_exit(compare_path, "Benchmark file")
    scorecards = data.get("scorecards", [])
    results = data.get("results", [])
    specs = data.get("specs") or _cli_pkg._lookup("get_system_specs")(client=clients.get("ollama"))

    _cli_pkg._lookup("display_system_banner")(specs)
    _cli_pkg._lookup("display_leaderboard")(scorecards, specs=specs)
    _cli_pkg._lookup("display_token_savings")(scorecards)

    if len(scorecards) >= 2:
        _cli_pkg._lookup("show_1to1_comparison")(scorecards, results, specs, output_dir)


def load_baseline(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load cached Ollama scorecards and results so Ollama models are never re-run."""
    data = load_json_or_exit(path, "Baseline file")
    scorecards = [sc for sc in data.get("scorecards", []) if is_ollama(sc)]
    results = [r for r in data.get("results", []) if is_ollama(r)]
    _cli_pkg._lookup("console").print(
        f"[bold green]✔ Baseline loaded ({len(scorecards)} models) from:[/] [cyan]{path}[/]"
    )
    _cli_pkg._lookup("console").print(
        "[dim]Ollama models will NOT be re-executed; baseline data will be merged directly.[/]"
    )
    return scorecards, results


__all__ = [
    "run_compare_mode",
    "show_1to1_comparison",
    "load_baseline",
    "select_pair",
    "records_for",
    "_model_key",
]
