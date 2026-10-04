import csv
import io
from unittest.mock import MagicMock, patch

from benchrig.core.runner import BenchmarkRunner


def tool_scorecard(unsupported=False):
    with patch("benchrig.core.runner.get_system_specs", return_value={}):
        runner = BenchmarkRunner(MagicMock(), {})
    turn = {
        "expect": {"tool": "read_file", "args": {}},
        "passed": True,
        "selection_correct": True,
        "arguments_correct": True,
        "structured_call": True,
        "tool_status": "ok",
        "request_latency_sec": 0.2,
    }
    return runner.compute_model_scorecard(
        "model",
        [
            {
                "model": "model",
                "runtime": "ollama",
                "engine": "llama.cpp",
                "suite": "tool_use",
                "tool_status": "unsupported" if unsupported else "ok",
                "success": not unsupported,
                "turns": [turn],
            }
        ],
    )


def test_tool_suite_is_opt_in_and_all_is_unchanged():
    from benchrig.cli import build_parser
    from benchrig.cli._common import DEFAULT_SUITES, SUITES

    assert build_parser().parse_args(["--suite", "tool_use"]).suite == "tool_use"
    assert "tool_use" in SUITES
    assert "tool_use" not in DEFAULT_SUITES
    assert set(DEFAULT_SUITES) == {"speed", "coding", "reasoning", "polish", "context"}


def test_reports_render_counts_and_unavailable_rates(tmp_path):
    from rich.console import Console

    from benchrig.reporting.display import display_leaderboard
    from benchrig.reporting.markdown import generate_markdown_report

    cards = [tool_scorecard(), tool_scorecard(True)]
    text = generate_markdown_report(cards, [], {}, str(tmp_path / "report.md"))
    assert "Tool-use" in text
    assert "1/1" in text and "n/a" in text and "unsupported" in text
    out = io.StringIO()
    with patch("benchrig.reporting.display.console", Console(file=out, width=220)):
        display_leaderboard(cards)
    assert "Tool-use" in out.getvalue()
    assert "unsupported" in out.getvalue()


def test_tool_only_scorecards_have_no_fabricated_composite():
    sc = tool_scorecard()
    assert sc["composite_score"] is None
    assert sc["peak_vram_mb"] is None
    assert sc["tool_task_pass_rate"] == 100


def test_csv_appends_tool_columns_preserving_existing_columns(tmp_path):
    from benchrig.reporting.csv_export import SCORECARD_CSV_COLUMNS, write_scorecards_csv

    assert SCORECARD_CSV_COLUMNS.index("tool_status") > SCORECARD_CSV_COLUMNS.index("coding_assertion_count")
    write_scorecards_csv([tool_scorecard(True)], str(tmp_path / "out.csv"))
    with (tmp_path / "out.csv").open() as f:
        row = next(csv.DictReader(f))
    assert row["tool_task_pass_rate"] == ""
    assert row["composite_score"] == ""
    assert row["tool_status"] == "unsupported"


def test_charts_distinguish_tool_success_from_composite():
    import matplotlib.pyplot as plt

    from benchrig.reporting.charts import make_scorecard_figure

    fig = make_scorecard_figure([tool_scorecard(), tool_scorecard(True)])
    try:
        assert len(fig.axes) == 1
        assert "tool" in fig.axes[0].get_ylabel().lower()
        assert len(fig.axes[0].patches) == 1
        assert any("n/a" in t.get_text() for t in fig.axes[0].texts)
    finally:
        plt.close(fig)


def test_progress_counts_scenario_repetitions_not_turns_or_warmups():
    from benchrig.cli import evaluate_model
    from tests.test_tool_use_runner import make_runner, ok
    from tests.test_tool_use_scenarios import scenario

    runner, client = make_runner([ok(), ok()], warmups=1)
    client.display_name = "Ollama"
    runner.measure_vram_baseline = MagicMock(return_value=0)
    steps = []

    def runner_factory(**kwargs):
        runner.on_result = kwargs["on_result"]
        return runner

    with (
        patch("benchrig.cli.run.BenchmarkRunner", side_effect=runner_factory),
        patch("benchrig.cli.run.display_scenario_result"),
        patch("benchrig.cli.time.sleep"),
    ):
        result = evaluate_model(
            "ollama",
            client,
            "model",
            {"ollama": {"warmup": False, "unload_after_test": False}},
            ["tool_use"],
            {"tool_use": [scenario()]},
            1,
            on_progress=steps.append,
            warmup_runs=1,
        )
    assert len(result) == 2
    assert len(steps) == 1


def test_pair_tool_suite_validates_before_creating_clients(tmp_path):
    import pytest

    from benchrig.cli import main

    (tmp_path / "tool_use.json").write_text('[{"id":"bad","tools":[]}]')
    config = {"model_pairs_1to1": [{"id": "tools", "recommended_suite": "tool_use"}]}
    with (
        patch("sys.argv", ["benchrig", "--pair", "tools", "--scenarios-dir", str(tmp_path)]),
        patch("benchrig.cli.load_config", return_value=config),
        patch("benchrig.cli._bootstrap_cuda_env"),
        patch("benchrig.cli.create_runtime_client") as create,
        pytest.raises(SystemExit) as exc,
    ):
        main()
    assert exc.value.code == 2
    create.assert_not_called()
