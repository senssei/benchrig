"""Output writers: Markdown, CSV, and chart artifacts (``--csv`` / ``--chart``).

Phase 12.4: carved out of the monolithic ``benchrig/cli.py``. ``save_outputs`` is called by
``run.py::run_benchmarks`` after a benchmark run; the Markdown/CSV/chart producers it wraps
live in ``benchrig.reporting.*`` and are unchanged by the package restructure.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import benchrig.cli as _cli_pkg
from benchrig.reporting import write_scorecards_chart, write_scorecards_csv
from benchrig.reporting.markdown import generate_markdown_report


def save_outputs(
    output_dir: str,
    specs: dict[str, Any],
    scorecards: list[dict[str, Any]],
    results: list[dict[str, Any]],
    total_duration: float,
    csv_path: str | None = None,
    chart_path: str | None = None,
    config: dict[str, Any] | None = None,
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
        _cli_pkg._lookup("console").print(f"  [dim]CSV written to {csv_path}[/]")
    if chart_path:
        write_scorecards_chart(scorecards, chart_path)
        _cli_pkg._lookup("console").print(f"  [dim]Chart written to {chart_path}[/]")

    generate_markdown_report(
        scorecards,
        results,
        specs,
        output_path=md_report_path,
        chart_path=os.path.relpath(chart_path, md_dir) if chart_path else None,
        csv_path=os.path.relpath(csv_path, md_dir) if csv_path else None,
        config=config,
    )
    return md_report_path, raw_json_path


__all__ = ["save_outputs"]
