"""Hermetic evidence for the opt-in paired cache experiment."""

import json
from unittest.mock import MagicMock, patch

import pytest

from benchrig.core.runner import BenchmarkRunner

SCENARIO = {
    "id": "p",
    "name": "Probe",
    "prompt": "Identical prefix",
    "options": {"num_ctx": 4096, "num_predict": 16, "temperature": 0, "seed": 42},
}


def response(**extra):
    return {
        "success": True,
        "response": "ok",
        "prompt_eval_count": 20,
        "eval_count": 2,
        "ttft_sec": 0.1,
        "load_time_sec": 0,
        "load_provenance": "engine",
        "observed_context_tokens": 4096,
        "observed_cache_type": "q8",
        **extra,
    }


def make_runner():
    client = MagicMock(name="client")
    client.name = "ollama"
    client.engine_name = "llama.cpp"
    client.requested_device = "unknown"
    client.prefill_source = "unavailable"
    with patch("benchrig.core.runner.get_system_specs", return_value={}):
        return BenchmarkRunner(client, {}), client


def test_probe_reuses_identical_prompt_and_effective_settings():
    runner, client = make_runner()
    with patch.object(
        runner,
        "_generate_with_telemetry",
        side_effect=[(response(), {}), (response(prefix_cache_hit=False), {}), (response(prefix_cache_hit=True), {})],
    ) as send:
        records = runner.run_cache_probe("m", [SCENARIO], runs=1)
    pair = [r for r in records if r["phase"] == "measured"]
    assert len(pair) == 2
    assert send.call_args_list[1] == send.call_args_list[2]
    assert pair[0]["pair_id"] == pair[1]["pair_id"]
    assert [r["request_position"] for r in pair] == [1, 2]
    assert pair[0]["prompt_sha256"] == pair[1]["prompt_sha256"]
    assert pair[0]["comparable"] and pair[1]["comparable"]
    client.unload_model.assert_not_called()


def test_unavailable_cache_settings_prevent_controlled_comparison():
    runner, _ = make_runner()
    with patch.object(runner, "_generate_with_telemetry", return_value=(response(observed_cache_type=None), {})):
        records = runner.run_cache_probe("m", [SCENARIO])
    pair = [r for r in records if r["phase"] == "measured"]
    assert all(not r["comparable"] for r in pair)
    assert "cache type unavailable" in pair[0]["comparison_reasons"]


def test_probe_does_not_affect_legacy_scores():
    runner, _ = make_runner()
    assert (
        runner.compute_model_scorecard(
            "m", [{"model": "m", "suite": "cache_probe", "phase": "measured", "eval_count": 10000}]
        )
        == {}
    )


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--suite", "speed"],
        ["--suite", "speed", "--models", "m", "--runtime", "all"],
        ["--suite", "speed", "--models", "m", "--runs", "0"],
        ["--suite", "speed", "--models", "m", "--check"],
    ],
)
def test_probe_validates_before_model_requests(extra, tmp_path):
    from benchrig.cli import main

    (tmp_path / "speed.json").write_text(json.dumps([SCENARIO]))
    with (
        patch("sys.argv", ["benchrig", "--cache-probe", "--scenarios-dir", str(tmp_path), *extra]),
        patch("benchrig.cli._bootstrap_cuda_env"),
        patch("benchrig.cli.load_config", return_value={}),
        patch("benchrig.cli.create_runtime_client") as create,
        pytest.raises(SystemExit) as exc,
    ):
        main()
    assert exc.value.code == 2
    create.assert_not_called()


def test_probe_validates_context_before_clients(tmp_path, capsys):
    from benchrig.cli import main

    (tmp_path / "speed.json").write_text(json.dumps([{**SCENARIO, "options": {"num_ctx": 0}}]))
    with (
        patch(
            "sys.argv",
            ["benchrig", "--cache-probe", "--suite", "speed", "--models", "m", "--scenarios-dir", str(tmp_path)],
        ),
        patch("benchrig.cli._bootstrap_cuda_env"),
        patch("benchrig.cli.load_config", return_value={}),
        patch("benchrig.cli.create_runtime_client") as create,
        pytest.raises(SystemExit) as exc,
    ):
        main()
    assert exc.value.code == 2
    create.assert_not_called()
    assert "num_ctx" in capsys.readouterr().err


def test_cache_probe_report_separates_load_prefill_decode_and_ttft(tmp_path):
    from benchrig.reporting.markdown import generate_markdown_report

    runner, _ = make_runner()
    with patch.object(
        runner,
        "_generate_with_telemetry",
        return_value=(response(prefill_duration_sec=0.05, decode_duration_sec=0.2, prefill_provenance="engine"), {}),
    ):
        records = runner.run_cache_probe("m", [SCENARIO])
    report = generate_markdown_report([], records, {}, str(tmp_path / "report.md"))
    assert "Cache probe" in report
    for label in ("Load (s)", "Prefill (s)", "Decode (s)", "TTFT (s)"):
        assert label in report
    assert "0.050" in report and "0.200" in report
    assert "Composite" not in report
    assert "q8" in report and "4096" in report


def test_cache_probe_report_exposes_unknown_settings_and_failed_pairs(tmp_path):
    from benchrig.reporting.markdown import generate_markdown_report

    runner, _ = make_runner()
    with patch.object(
        runner,
        "_generate_with_telemetry",
        return_value=(
            response(
                success=False, error="transport failed", observed_cache_type=None, load_time_sec=None, ttft_sec=None
            ),
            {},
        ),
    ):
        records = runner.run_cache_probe("m", [SCENARIO])
    report = generate_markdown_report([], records, {}, str(tmp_path / "report.md"))
    assert "cache type unavailable" in report
    assert "transport failed" in report
    assert "unavailable" in report
    assert "unverified" in report


def test_probe_does_not_replace_legacy_artifacts(tmp_path):
    from benchrig.cli.report import save_outputs

    (tmp_path / "latest.json").write_text('{"scorecards":[{"model":"legacy"}]}')
    (tmp_path / "LATEST_SUMMARY.md").write_text("legacy report")
    runner, _ = make_runner()
    with patch.object(runner, "_generate_with_telemetry", return_value=(response(), {})):
        records = runner.run_cache_probe("m", [SCENARIO])
    save_outputs(str(tmp_path), {}, [], records, 1.0)
    assert json.loads((tmp_path / "latest.json").read_text())["scorecards"][0]["model"] == "legacy"
    assert (tmp_path / "LATEST_SUMMARY.md").read_text() == "legacy report"
    assert (tmp_path / "CACHE_PROBE_SUMMARY.md").is_file()


def test_startup_failure_keeps_diagnostic_artifacts(tmp_path):
    from benchrig.cli import build_parser
    from benchrig.cli.run import run_cache_probe

    args = build_parser().parse_args(
        ["--cache-probe", "--suite", "speed", "--models", "m", "--output-dir", str(tmp_path)]
    )
    args.cache_probe_scenarios = [SCENARIO]
    _, client = make_runner()
    client.load_model.return_value = False
    with (
        patch("benchrig.cli.get_system_specs", return_value={}),
        patch("benchrig.core.runner.get_system_specs", return_value={}),
    ):
        run_cache_probe(args, {}, {"ollama": client}, "ollama")
    artifact = next((tmp_path / "runs").glob("cache_probe_*.json"))
    result = json.loads(artifact.read_text())["results"][0]
    assert result["success"] is False
    assert "startup" in result["error"]
    assert "startup" in (tmp_path / "CACHE_PROBE_SUMMARY.md").read_text()
    client.generate.assert_not_called()
