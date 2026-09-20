"""Helpers shared by the terminal and Markdown reports."""

from typing import Any, NamedTuple

# Suites that only measure speed or memory: there is nothing to pass or fail, the request either worked or it did not.
UNSCORED_SUITES = ("speed", "context")


class Outcome(NamedTuple):
    """How one scenario went: `ok` is None when the runtime has no record for it."""

    ok: bool | None
    detail: str  # "4/5" for tests with assertions, otherwise ""
    truncated: bool  # the response stopped at the token limit before it was complete
    scored: bool  # False for speed/context scenarios (OK/ERROR instead of PASS/FAIL)


def scenario_outcome(record: dict[str, Any]) -> Outcome:
    """Reduce a result record to an outcome, whatever suite it came from.

    Coding records carry `passed_tests`/`total_tests`, reasoning and polish records carry `correct`, and speed/context
    records only `success`; reading `passed_tests` from all of them made every non-coding scenario look like "FAIL (0/0)".
    """
    if not record:
        return Outcome(None, "", False, True)
    truncated = bool(record.get("truncated"))
    # A context scenario that asks about a hidden fact is scored (PASS/FAIL); one that only measures speed is not.
    scored = record.get("suite") not in UNSCORED_SUITES or "retrieved" in record
    if record.get("total_tests"):
        return Outcome(
            bool(record.get("passed")), f"{record.get('passed_tests', 0)}/{record['total_tests']}", truncated, scored
        )
    if "correct" in record:
        return Outcome(bool(record["correct"]), "", truncated, scored)
    return Outcome(bool(record.get("success")), "", truncated, scored)


def _label(outcome: Outcome) -> str:
    if outcome.ok is None:
        return "n/a"
    word = ("PASS" if outcome.ok else "FAIL") if outcome.scored else ("OK" if outcome.ok else "ERROR")
    return f"{word} ({outcome.detail})" if outcome.detail else word


def status_markdown(record: dict[str, Any]) -> str:
    """Status cell for the Markdown report."""
    outcome = scenario_outcome(record)
    if outcome.ok is None:
        return "—"
    mark = "✅" if outcome.ok else "❌"
    return f"{mark} {_label(outcome)}" + (" ✂ cut off" if outcome.truncated else "")


def status_rich(record: dict[str, Any]) -> str:
    """Status cell for the terminal report (rich markup)."""
    outcome = scenario_outcome(record)
    if outcome.ok is None:
        return "[dim]n/a[/]"
    colour = "green" if outcome.ok else "red"
    return f"[{colour}]{_label(outcome)}[/]" + (" [yellow]✂ cut off[/]" if outcome.truncated else "")


def spread_lines(scorecards: list[dict[str, Any]]) -> list[str]:
    """One line per scorecard that was measured over several repetitions: min-max of each headline metric."""
    lines = []
    for sc in scorecards:
        spread = sc.get("spread")
        if not spread:
            continue

        def span(key: str, unit: str = "", digits: int = 1, spread: dict = spread) -> str:
            low, high = spread[key]
            return f"{low:.{digits}f}{unit}" if low == high else f"{low:.{digits}f}-{high:.{digits}f}{unit}"

        lines.append(
            f"`{sc['model']}` ({sc.get('runs', '?')} runs): composite {span('composite_score')}, coding {span('coding_pass_rate', '%')}, "
            f"reasoning {span('reasoning_accuracy', '%')}, decode {span('avg_eval_tok_sec', ' t/s')}, TTFT {span('avg_ttft_sec', 's', 2)}"
        )
    return lines


EFFICIENCY_HEADERS = ("Model", "Cold start", "GPU fit", "Tokens / J", "Facts found in context")
EFFICIENCY_NOTE = (
    "Cold start = the first request, which includes loading the model when it was not resident. GPU fit = share of the model in "
    "GPU memory (Ollama only; below 100% part of it runs on the CPU). Tokens / J = generated tokens per joule of GPU power "
    "(average power over each request, whole GPU, so it includes other processes)."
)


def efficiency_rows(scorecards: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    """Rows for the start-up / fit / efficiency table; empty when no scorecard has any of these figures."""

    def fmt(value: Any, template: str) -> str:
        return "-" if value is None else template.format(value)

    rows = []
    for sc in scorecards:
        figures = (
            sc.get("cold_start_sec"),
            sc.get("gpu_fit_pct"),
            sc.get("tokens_per_joule"),
            sc.get("context_retrieval_pct"),
        )
        if all(v is None for v in figures):
            continue
        rows.append(
            (
                str(sc["model"]),
                fmt(figures[0], "{:.1f} s"),
                fmt(figures[1], "{:.0f}%"),
                fmt(figures[2], "{:.2f}"),
                fmt(figures[3], "{:.0f}%"),
            )
        )
    return rows
