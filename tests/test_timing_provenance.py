"""Tests for timing breakdown and provenance reporting (spec.md Phase 14, I12)."""

import csv
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from benchrig.core.client import FoundryClient, OllamaClient
from benchrig.core.runner import BenchmarkRunner
from benchrig.reporting.csv_export import SCORECARD_CSV_COLUMNS, write_scorecards_csv


def make_clock(readings):
    it = iter(readings)
    last = readings[-1] if readings else 0.0
    return lambda: next(it, last)


class TimingProvenanceTests(unittest.TestCase):
    def test_csv_columns_contain_timing_provenance_fields(self):
        """SCORECARD_CSV_COLUMNS must include prefill_duration_sec, decode_duration_sec, and prefill_provenance."""
        self.assertIn("prefill_duration_sec", SCORECARD_CSV_COLUMNS)
        self.assertIn("decode_duration_sec", SCORECARD_CSV_COLUMNS)
        self.assertIn("prefill_provenance", SCORECARD_CSV_COLUMNS)

    def test_openai_compatible_client_does_not_label_ttft_as_engine_prefill(self):
        """OpenAI-compatible client without engine prefill timing sets prefill_duration_sec=None and provenance='unavailable'."""
        client = FoundryClient(base_url="http://fake-foundry:8000/v1")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            b'data: {"choices": [{"delta": {"content": " world"}}]}',
            b"data: [DONE]",
        ]

        with patch("benchrig.core.client.requests.post", return_value=mock_response):
            with patch(
                "benchrig.core.client.time.perf_counter", side_effect=make_clock([0.0, 0.01, 0.02, 0.1, 0.15, 0.2])
            ):
                res = client.generate("test-model", "Say hi", measure_ttft=True)

        self.assertTrue(res["success"])
        self.assertAlmostEqual(res["ttft_sec"], 0.1, places=2)
        # Invariant I12: TTFT is never labelled as engine prefill duration
        self.assertIsNone(res.get("prefill_duration_sec"))
        self.assertEqual(res.get("prefill_provenance"), "unavailable")
        self.assertIn("prefill_eff_tok_sec", res)
        self.assertGreater(res["prefill_eff_tok_sec"], 0.0)

    def test_ollama_client_records_engine_prefill_duration_and_provenance(self):
        """Ollama client with prompt_eval_duration sets prefill_provenance='engine'."""
        client = OllamaClient(base_url="http://fake-ollama:11434")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b'{"response": "Hello", "done": false}',
            b'{"response": " world", "done": true, "prompt_eval_count": 20, "prompt_eval_duration": 50000000, "eval_count": 2, "eval_duration": 100000000, "total_duration": 150000000}',
        ]

        with patch("benchrig.core.client.requests.post", return_value=mock_response):
            with patch("benchrig.core.client.time.perf_counter", side_effect=make_clock([0.0, 0.08, 0.18, 0.2])):
                res = client.generate("test-model", "Say hi", measure_ttft=True)

        self.assertTrue(res["success"])
        self.assertEqual(res.get("prefill_provenance"), "engine")
        self.assertIsNotNone(res.get("prefill_duration_sec"))
        self.assertAlmostEqual(res["prefill_duration_sec"], 0.05, places=3)
        self.assertAlmostEqual(res["prompt_tok_per_sec"], 400.0, places=1)

    def test_runner_scorecard_aggregates_timing_provenance(self):
        """BenchmarkRunner scorecard aggregates prefill_duration_sec and prefill_provenance."""
        runner = BenchmarkRunner(client=None, config={})
        results = [
            {
                "model": "m",
                "suite": "speed",
                "eval_tok_per_sec": 50.0,
                "eval_count": 10,
                "prompt_eval_count": 20,
                "ttft_sec": 0.1,
                "prefill_duration_sec": 0.04,
                "decode_duration_sec": 0.18,
                "prefill_provenance": "engine",
                "prefill_eff_tok_per_sec": 200.0,
            }
        ]
        sc = runner.compute_model_scorecard("m", results)
        self.assertEqual(sc.get("prefill_provenance"), "engine")
        self.assertAlmostEqual(sc.get("prefill_duration_sec"), 0.04, places=2)
        self.assertAlmostEqual(sc.get("decode_duration_sec"), 0.18, places=2)

    def test_csv_round_trip_timing_provenance(self):
        """write_scorecards_csv serializes prefill_duration_sec, decode_duration_sec, and prefill_provenance."""
        scorecard = {
            "model": "m",
            "runtime": "ollama",
            "engine": "llama.cpp",
            "composite_score": 10.0,
            "coding_pass_rate": 0.5,
            "reasoning_accuracy": 0.5,
            "avg_eval_tok_sec": 10.0,
            "avg_ttft_sec": 0.1,
            "peak_vram_mb": 100.0,
            "total_runs": 1,
            "eval_tok_sec_floored": False,
            "requested_device": "gpu",
            "observed_device": "cuda",
            "cpu_fallback": False,
            "prefill_duration_sec": 0.05,
            "decode_duration_sec": 0.20,
            "prefill_provenance": "engine",
        }
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "out.csv")
            write_scorecards_csv([scorecard], csv_path)
            with open(csv_path) as fh:
                rows = list(csv.reader(fh))
            prefill_dur_idx = rows[0].index("prefill_duration_sec")
            decode_dur_idx = rows[0].index("decode_duration_sec")
            prov_idx = rows[0].index("prefill_provenance")
            self.assertEqual(rows[1][prefill_dur_idx], "0.05")
            self.assertEqual(rows[1][decode_dur_idx], "0.2")
            self.assertEqual(rows[1][prov_idx], "engine")


if __name__ == "__main__":
    unittest.main()


def test_missing_load_and_cache_evidence_in_clients():
    from benchrig.core.client import FoundryClient, OllamaClient

    for client in (FoundryClient(base_url="http://fake/v1"), OllamaClient(base_url="http://fake")):
        response = MagicMock(status_code=200)
        response.iter_lines.return_value = (
            [b'{"response":"ok","done":true}']
            if client.name == "ollama"
            else [b'data: {"choices":[{"delta":{"content":"ok"}}]}', b"data: [DONE]"]
        )
        with patch("benchrig.core.client.requests.post", return_value=response):
            result = client.generate("m", "p")
        assert result["load_time_sec"] is None
        assert result["load_provenance"] == "unavailable"
        assert result.get("prefix_cache_hit") is None


def test_backend_cache_evidence_retains_raw_telemetry():
    client = FoundryClient(base_url="http://fake/v1")
    response = MagicMock(status_code=200)
    response.iter_lines.return_value = [
        b'data: {"choices":[{"delta":{"content":"ok"}}]}',
        b'data: {"choices":[],"telemetry":{"prefix_cache_hit":true,"cache_type":"q8","context_tokens":4096,"load_time_sec":0.0}}',
        b"data: [DONE]",
    ]
    with patch("benchrig.core.client.requests.post", return_value=response):
        result = client.generate("m", "p")
    assert result["prefix_cache_hit"] is True
    assert result["observed_cache_type"] == "q8"
    assert result["observed_context_tokens"] == 4096
    assert result["raw_metrics"]["telemetry"]["prefix_cache_hit"] is True


def test_native_partial_cache_telemetry_is_retained():
    client = OllamaClient(base_url="http://fake")
    response = MagicMock(status_code=200)
    response.iter_lines.return_value = [
        b'{"response":"ok","done":true,"context_tokens":4096,"cache_type":"q8","prompt_eval_cached_count":10}'
    ]
    with patch("benchrig.core.client.requests.post", return_value=response):
        result = client.generate("m", "p")
    assert result["raw_metrics"]["server_metrics"]["prompt_eval_cached_count"] == 10
    assert result["cache_evidence"]["prompt_eval_cached_count"] == 10
    assert result["prefix_cache_hit"] is None


def test_native_prefill_without_token_is_not_client_ttft():
    client = OllamaClient(base_url="http://fake")
    response = MagicMock(status_code=200)
    response.iter_lines.return_value = [b'{"done":true,"prompt_eval_duration":123000000}']
    with patch("benchrig.core.client.requests.post", return_value=response):
        result = client.generate("m", "p")
    assert result["ttft_sec"] is None
    assert result["ttft_provenance"] == "unavailable"
    assert result["prefill_duration_sec"] == 0.123


def test_openai_response_without_first_token_has_no_ttft():
    client = FoundryClient(base_url="http://fake/v1")
    response = MagicMock(status_code=200)
    response.iter_lines.return_value = [
        b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":0}}',
        b"data: [DONE]",
    ]
    with patch("benchrig.core.client.requests.post", return_value=response):
        result = client.generate("m", "p")
    assert result["ttft_sec"] is None
    assert result["ttft_provenance"] == "unavailable"


def test_null_ttft_does_not_enter_scorecard_mean_or_spread():
    runner = BenchmarkRunner(None, {})
    records = [
        {"model": "m", "suite": "speed", "success": True, "eval_tok_per_sec": 1.0, "ttft_sec": ttft, "run": run}
        for run, ttft in enumerate([None, 0.2])
    ]
    card = runner.compute_model_scorecard("m", records)
    assert card["avg_ttft_sec"] == 0.2
    assert card["spread"]["avg_ttft_sec"] == [0.2, 0.2]
    assert runner.compute_model_scorecard("m", records[:1])["avg_ttft_sec"] is None


def test_null_ttft_is_rendered_unavailable_in_reports(tmp_path):
    from benchrig.reporting.display import display_1to1_comparison, display_leaderboard
    from benchrig.reporting.markdown import generate_1to1_comparison_report, generate_markdown_report

    runner = BenchmarkRunner(None, {})
    cards = [
        runner.compute_model_scorecard(
            "m", [{"model": "m", "runtime": rt, "suite": "speed", "eval_tok_per_sec": 1.0, "ttft_sec": None}]
        )
        for rt in ("ollama", "foundry")
    ]
    # Supply complete summary fields to isolate rendering from aggregation.
    for card in cards:
        card["avg_ttft_sec"] = None
        card["avg_prefill_eff_tok_sec"] = 1.0
    report = generate_markdown_report(
        cards, [{"suite": "context", "model": "m", "ttft_sec": None}], {}, str(tmp_path / "summary.md")
    )
    assert "n/a" in report
    assert "n/a" in generate_1to1_comparison_report(cards[0], cards[1], [], [], {}, str(tmp_path / "comparison.md"))
    display_leaderboard(cards)
    display_1to1_comparison(cards[0], cards[1], [], [])


def test_failed_request_has_unavailable_ttft(caplog):
    caplog.set_level(50, logger="benchrig")
    client = FoundryClient(base_url="http://fake/v1")
    with patch("benchrig.core.client.requests.post", side_effect=OSError("transport failed")):
        result = client.generate("m", "p")
    assert result["success"] is False
    assert result["ttft_sec"] is None
