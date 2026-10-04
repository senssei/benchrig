import pytest


def response(calls=None, text="", **extra):
    return {
        "tool_status": "ok",
        "success": True,
        "tool_calls": calls or [],
        "response": text,
        "structured_call": bool(calls),
        "parse_errors": [],
        "json_in_content": False,
        **extra,
    }


@pytest.mark.parametrize(
    "actual,expected,passed",
    [
        ({"k": 1}, {"k": 1}, True),
        ({"k": "1"}, {"k": 1}, False),
        ({"k": True}, {"k": 1}, False),
        ({"k": 1.0}, {"k": 1}, False),
        ({"k": 1, "x": 2}, {"k": 1}, False),
        ({"b": 2, "a": 1}, {"a": 1, "b": 2}, True),
        ({"ids": [2, 1]}, {"ids": [1, 2]}, False),
        ({"path": "a.txt"}, {"path": {"type": "string", "contains": "txt"}}, True),
    ],
)
def test_argument_matching_is_typed_and_rejects_extras(actual, expected, passed):
    from benchrig.core.tool_use import score_turn

    r = response([{"name": "read_file", "arguments": actual}])
    assert score_turn({"tool": "read_file", "args": expected}, r)["passed"] is passed


def test_refusal_and_text_json_fail_abstention():
    from benchrig.core.tool_use import score_turn

    expect = {"tool": None, "args": {}, "answer": "4"}
    assert score_turn(expect, response(text=" 4 "))["passed"]
    assert not score_turn(expect, response(text="I cannot help"))["passed"]
    assert not score_turn(expect, response(text="4", json_in_content=True))["passed"]


def test_rates_use_planned_denominators_and_exclude_unsupported_and_warmups():
    from benchrig.core.tool_use import aggregate_tools

    good = {
        "expect": {"tool": "read_file", "args": {}},
        "passed": True,
        "selection_correct": True,
        "arguments_correct": True,
        "structured_call": True,
    }
    skipped = {
        "expect": {"tool": "read_file", "args": {}},
        "tool_status": "not_run_due_to_prior_failure",
        "passed": False,
    }
    rows = [
        {"tool_status": "ok", "success": False, "turns": [good, skipped], "phase": "measured"},
        {"tool_status": "unsupported", "turns": [good], "phase": "measured"},
        {"tool_status": "ok", "success": True, "turns": [good], "phase": "warmup"},
    ]
    sc = aggregate_tools(rows)
    assert sc["tool_selection_accuracy"] == 50
    assert sc["tool_selection_count"] == 2
    assert sc["tool_task_pass_rate"] == 0
    assert sc["tool_unsupported_count"] == 1


def test_zero_denominators_are_unavailable():
    from benchrig.core.tool_use import aggregate_tools

    assert aggregate_tools([{"tool_status": "unsupported", "turns": []}])["tool_task_pass_rate"] is None


def test_tool_evidence_does_not_change_legacy_composite():
    from unittest.mock import MagicMock, patch

    from benchrig.core.runner import BenchmarkRunner

    with patch("benchrig.core.runner.get_system_specs", return_value={}):
        runner = BenchmarkRunner(MagicMock(), {})
    legacy = {
        "model": "m",
        "runtime": "ollama",
        "suite": "speed",
        "eval_tok_per_sec": 10,
        "eval_count": 10,
        "hardware": {"vram_peak_mb": 10},
    }
    tool = {
        "model": "m",
        "runtime": "ollama",
        "engine": "llama.cpp",
        "suite": "tool_use",
        "tool_status": "ok",
        "success": True,
        "turns": [],
        "hardware": {"vram_peak_mb": 999999},
        "eval_tok_per_sec": 999999,
        "eval_count": 999999,
    }
    before = runner.compute_model_scorecard("m", [legacy])
    after = runner.compute_model_scorecard("m", [legacy, tool])
    for key in before:
        assert after[key] == before[key]
    assert after["tool_task_count"] == 1
    only = runner.compute_model_scorecard("m", [tool])
    assert only["composite_score"] is None
    assert only["avg_eval_tok_sec"] is None
