from unittest.mock import MagicMock, patch

from benchrig.core.runner import BenchmarkRunner
from tests.test_tool_use_scenarios import scenario


def make_runner(responses, warmups=0):
    client = MagicMock(name="client")
    client.name = "ollama"
    client.engine_name = "llama.cpp"
    client.chat_tools.side_effect = responses
    with patch("benchrig.core.runner.get_system_specs", return_value={}):
        runner = BenchmarkRunner(client, {}, warmup_runs=warmups)
    runner._create_sampler = MagicMock()
    runner._create_sampler.return_value.stop.return_value = {}
    return runner, client


def ok(path="a.txt", call_id="call-actual"):
    return {
        "tool_status": "ok",
        "success": True,
        "response": "",
        "tool_calls": [{"id": call_id, "name": "read_file", "arguments": {"path": path}}],
        "structured_call": True,
        "raw_tool_calls": [],
        "parse_errors": [],
        "request_latency_sec": 0.1,
        "finish_reason": "tool_calls",
    }


def test_continuation_uses_actual_call_id_and_fixture_result():
    sc = scenario()
    sc["turns"][0]["tool_result"] = "b.txt"
    sc["turns"].append({"expect": {"tool": "read_file", "args": {"path": "b.txt"}}})
    runner, client = make_runner([ok(), ok("b.txt")])
    result = runner.run_tool_use_suite("model", [sc])[0]
    assert result["success"]
    assert client.chat_tools.call_args_list[1].kwargs["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-actual",
        "tool_name": "read_file",
        "content": "b.txt",
    }


def test_failed_turn_stops_followups_but_preserves_planned_count():
    sc = scenario()
    sc["turns"][0]["tool_result"] = "b.txt"
    sc["turns"].append({"expect": {"tool": "read_file", "args": {"path": "b.txt"}}})
    runner, client = make_runner([ok("wrong.txt")])
    result = runner.run_tool_use_suite("model", [sc])[0]
    assert not result["success"]
    assert len(result["turns"]) == 2
    assert result["turns"][1]["tool_status"] == "not_run_due_to_prior_failure"
    assert client.chat_tools.call_count == 1


def test_unsupported_stops_only_tool_requests_for_that_model():
    runner, client = make_runner([{"tool_status": "unsupported", "success": False, "error": "no tools"}])
    result = runner.run_tool_use_suite("model", [scenario(), {**scenario(), "id": "second"}])
    assert [r["tool_status"] for r in result] == ["unsupported", "unsupported"]
    runner.run_index = 1
    runner.run_tool_use_suite("model", [scenario()])
    assert client.chat_tools.call_count == 1
    assert runner.capacity_exhausted_reason is None


def test_warmups_and_repetitions_use_fresh_conversations():
    runner, client = make_runner([ok(), ok(), ok(), ok()], warmups=1)
    rows = runner.run_tool_use_suite("model", [scenario()])
    runner.run_index = 1
    rows += runner.run_tool_use_suite("model", [scenario()])
    assert [r["phase"] for r in rows] == ["warmup", "measured", "warmup", "measured"]
    assert all(len(c.kwargs["messages"]) == 1 for c in client.chat_tools.call_args_list)


def test_fixture_tools_never_execute_model_selected_actions():
    runner, client = make_runner([ok("/etc/passwd")])
    with patch("builtins.open", side_effect=AssertionError("real tool executed")):
        assert not runner.run_tool_use_suite("model", [scenario()])[0]["success"]
    client.generate.assert_not_called()


def test_effective_settings_and_placement_are_recorded():
    runner, client = make_runner([{**ok(), "requested_device": "gpu", "observed_device": "CPU", "cpu_fallback": True}])
    row = runner.run_tool_use_suite("model", [scenario()])[0]
    assert row["options"]["temperature"] == 0
    assert row["turns"][0]["observed_device"] == "CPU"


def test_capacity_suppressed_turns_are_not_reported_as_requests():
    from benchrig.core.tool_use import aggregate_tools

    runner, client = make_runner([])
    runner._capacity_exhausted_reason = "insufficient_resources"
    rows = runner.run_tool_use_suite("model", [scenario()])
    assert rows[0]["attempted_turn_count"] == 0
    assert aggregate_tools(rows)["tool_attempted_turn_count"] == 0
    assert aggregate_tools(rows)["tool_selection_count"] == 1
    client.chat_tools.assert_not_called()
