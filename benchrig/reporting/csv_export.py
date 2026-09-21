"""CSV export of benchmark scorecards (plan.md Phase 1, item 1.1).

Each scorecard is reduced to a fixed set of columns so the CSV stays comparable across runs and across
changes to the scorecard dict. The columns are deliberately a small subset of the scorecard fields: the
JSON in ``results/runs/<ts>.json`` keeps the full record; the CSV is for spreadsheets.
"""

from __future__ import annotations

import csv
import os
from typing import Any

# Ordered list of scorecard keys that go into the CSV. Adding a column here is a spec change (I1 in
# spec.md); removing or renaming a column is a breaking change for downstream spreadsheets.
SCORECARD_CSV_COLUMNS: tuple[str, ...] = (
    "model",
    "runtime",
    "engine",
    "composite_score",
    "coding_pass_rate",
    "reasoning_accuracy",
    "avg_eval_tok_sec",
    "avg_ttft_sec",
    "peak_vram_mb",
    "total_runs",
)


def _row_for(scorecard: dict[str, Any]) -> list[str]:
    """Render one scorecard as a CSV row, in the order of ``SCORECARD_CSV_COLUMNS``."""
    out: list[str] = []
    for key in SCORECARD_CSV_COLUMNS:
        value = scorecard.get(key)
        out.append("" if value is None else str(value))
    return out


def write_scorecards_csv(scorecards: list[dict[str, Any]], path: str) -> str:
    """Write ``scorecards`` to ``path`` as CSV. Returns ``path`` so callers can chain it."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(SCORECARD_CSV_COLUMNS))
        for sc in scorecards:
            writer.writerow(_row_for(sc))
    return path
