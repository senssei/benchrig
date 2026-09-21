"""PNG chart export of benchmark scorecards (plan.md Phase 1, item 1.2).

``matplotlib`` is imported lazily inside the figure-building function so that ``benchrig --help`` and the rest of
the CLI still load when the ``[charts]`` extra is not installed (spec.md I3). A missing ``matplotlib`` raises a
clear ``RuntimeError`` and never writes a partial file.
"""

from __future__ import annotations

import os

# Type-only import: does NOT pull matplotlib at module load.
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.figure


_PNG_BAR_HEIGHT_KEY = "composite_score"


def _bar_label(scorecard: dict[str, Any]) -> str:
    """Bar label: ``<model> (<runtime>)``. Runtime falls back to ``"unknown"`` if missing."""
    runtime = scorecard.get("runtime") or "unknown"
    return f"{scorecard['model']} ({runtime})"


def make_scorecard_figure(scorecards: list[dict[str, Any]]) -> matplotlib.figure.Figure:
    """Build a matplotlib Figure with one bar per scorecard (bar height = composite_score).

    The function imports ``matplotlib`` lazily; it raises ``RuntimeError`` with an actionable message when
    matplotlib is not installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend; safe in any environment
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --chart (install with `pip install -e .[charts]`)") from exc

    labels = [_bar_label(sc) for sc in scorecards]
    heights = [float(sc.get(_PNG_BAR_HEIGHT_KEY, 0) or 0) for sc in scorecards]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(scorecards)), 4.5))
    ax.bar(range(len(scorecards)), heights, tick_label=labels)
    ax.set_ylabel("composite score")
    ax.set_ylim(0, 100)
    ax.set_title("BenchRig scorecards")
    fig.tight_layout()
    return fig


def write_scorecards_chart(scorecards: list[dict[str, Any]], path: str) -> str:
    """Render a chart of ``scorecards`` to ``path`` as PNG. Returns ``path``."""
    fig = make_scorecard_figure(scorecards)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Ensure the file does not exist as a half-written stub if savefig fails.
    if os.path.exists(path):
        os.remove(path)
    try:
        fig.savefig(path, format="png")
    except Exception:
        # Defensive: if matplotlib leaves a partial file, remove it.
        if os.path.exists(path):
            os.remove(path)
        raise
    return path
