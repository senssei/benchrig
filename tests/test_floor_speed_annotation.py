"""Tests for Phase 10 floor-driven decode-speed annotation (plan.md §Phase 10).

Covers:
  - Client-level: ``eval_tok_sec_floored`` flag on the response dict, set when the
    0.001 s measurement floor engages (``eval_count > 0`` and ``eval_duration_sec_raw
    < 0.0015``). NOT set when the real wall-clock duration was above the floor.
  - Markdown rendering: tilde prefix in the leaderboard row when the scorecard's
    aggregated flag is True; no prefix when False.
  - CSV export: new ``eval_tok_sec_floored`` boolean column at the end of
    ``SCORECARD_CSV_COLUMNS``; round-trips through the writer.
"""

import csv
import os
import tempfile
import unittest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Per-record flag (set in client.py)
# ---------------------------------------------------------------------------


class EvalTokSecFlooredFlagTests(unittest.TestCase):
    """Verify ``eval_tok_sec_floored`` is set on the response dict when the
    0.001 s floor engages, and not set when the real measurement was above the
    floor. The Prism path is exercised; the floor and its flag exist only on
    the Foundry/Prism path (Ollama reports its own eval_duration).
    """

    def _stream_response(self, eval_count, eval_dur_ns, prompt_eval_count=10, prompt_eval_dur_ns=10_000_000):
        """Build a stream whose chunk count matches ``eval_count``.

        Prism/FoundryClient counts `total_stream_chunks` from the stream; we use that to
        drive the eval_count so the resulting ``eval_tok_sec`` floor arithmetic lines up.
        """
        import json as _json

        mock = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        mock.status_code = 200
        mock.raise_for_status.return_value = None
        chunks = []
        for i in range(eval_count):
            chunk = {
                "id": f"chunk-{i}",
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": "ok"}, "index": 0}],
            }
            chunks.append(f"data: {_json.dumps(chunk)}\n\n".encode())
        chunks.append(b"data: [DONE]\n\n")
        mock.iter_lines.return_value = chunks
        mock.json.return_value = {
            "choices": [{"message": {"content": "ok" * eval_count}}],
            "usage": {
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count,
                "total_tokens": prompt_eval_count + eval_count,
            },
        }
        return mock

    def test_eval_tok_sec_floored_true_when_floor_engages(self):
        """A streamed response where the wall-clock eval window is sub-millisecond sets the flag."""
        from unittest.mock import MagicMock

        from benchrig.core.client import PrismClient

        client = object.__new__(PrismClient)
        client.base_url = "http://127.0.0.1:5272/v1"
        client.name = "prism"
        client.engine_name = "ONNX Runtime GenAI"
        client.timeout_sec = 30
        client.default_num_ctx = 4096
        client.default_max_tokens = None
        client.api_key = None
        client._engines = {}
        client._request_kwargs = MagicMock(return_value={})
        client.unload_model = lambda *_a, **_k: True
        client._make_request = PrismClient._make_request.__get__(client, PrismClient)

        mock_resp = self._stream_response(eval_count=3, eval_dur_ns=500_000)
        import itertools

        # Tighter increments (0.0001) keep the eval window strictly below the 0.0015 floor
        # threshold so the flag is set.
        with patch("benchrig.core.client.requests.post", return_value=mock_resp):
            with patch(
                "benchrig.core.client.time.perf_counter",
                side_effect=itertools.count(0.0, 0.0001),
            ):
                result = client.generate("phi4", "What is the access code for vault 7? Answer with only the number.")

        self.assertTrue(
            result.get("eval_tok_sec_floored"),
            f"3 tokens in < 1ms must flag the floor; eval_count={result.get('eval_count')} "
            f"eval_tok_per_sec={result.get('eval_tok_per_sec')} raw_metrics={result.get('raw_metrics')}",
        )
        self.assertEqual(result["eval_count"], 3)

    def test_eval_tok_sec_floored_false_when_real(self):
        """A streamed response with a real wall-clock eval window does NOT set the flag."""
        from unittest.mock import MagicMock

        from benchrig.core.client import PrismClient

        client = object.__new__(PrismClient)
        client.base_url = "http://127.0.0.1:5272/v1"
        client.name = "prism"
        client.engine_name = "ONNX Runtime GenAI"
        client.timeout_sec = 30
        client.default_num_ctx = 4096
        client.default_max_tokens = None
        client.api_key = None
        client._engines = {}
        client._request_kwargs = MagicMock(return_value={})
        client.unload_model = lambda *_a, **_k: True
        client._make_request = PrismClient._make_request.__get__(client, PrismClient)

        mock_resp = self._stream_response(eval_count=12, eval_dur_ns=12_000_000)
        # 5 ms increments — well above the 0.0015 s floor threshold so the flag is NOT set.
        import itertools

        with patch("benchrig.core.client.requests.post", return_value=mock_resp):
            with patch(
                "benchrig.core.client.time.perf_counter",
                side_effect=itertools.count(0.0, 0.005),
            ):
                result = client.generate("phi4", "Write a short paragraph about BenchRig.")

        self.assertFalse(
            result.get("eval_tok_sec_floored", False),
            "5 ms eval window must NOT flag the floor; result keys: {list(result.keys())}",
        )
        self.assertEqual(result["eval_count"], 12)


# ---------------------------------------------------------------------------
# Markdown rendering (tilde prefix)
# ---------------------------------------------------------------------------


def _scorecard(**extra):
    base = {
        "model": "fixture-model",
        "runtime": "prism",
        "engine": "ONNX Runtime GenAI",
        "composite_score": 50.0,
        "coding_pass_rate": 0.0,
        "reasoning_accuracy": 50.0,
        "avg_eval_tok_sec": 3000.0,
        "avg_ttft_sec": 0.1,
        "peak_vram_mb": 8000.0,
        "total_runs": 1,
    }
    base.update(extra)
    return base


class MarkdownFloorSpeedAnnotationTests(unittest.TestCase):
    def test_leaderboard_renders_tilde_when_floored(self):
        """When `eval_tok_sec_floored=True`, the leaderboard row prints `~3000.0 t/s`."""
        from benchrig.reporting.markdown import generate_markdown_report

        with tempfile.TemporaryDirectory() as tmp:
            md_path = os.path.join(tmp, "report.md")
            generate_markdown_report(
                scorecards=[_scorecard(eval_tok_sec_floored=True)],
                raw_results=[],
                system_specs={},
                output_path=md_path,
            )
            content = open(md_path).read()
            self.assertIn(
                "~3000.0 t/s",
                content,
                "floored speed must render with tilde prefix",
            )

    def test_leaderboard_renders_plain_when_real(self):
        """When `eval_tok_sec_floored=False` (or absent), no tilde."""
        from benchrig.reporting.markdown import generate_markdown_report

        with tempfile.TemporaryDirectory() as tmp:
            md_path = os.path.join(tmp, "report.md")
            generate_markdown_report(
                scorecards=[_scorecard(eval_tok_sec_floored=False)],
                raw_results=[],
                system_specs={},
                output_path=md_path,
            )
            content = open(md_path).read()
            self.assertIn("3000.0 t/s", content)
            self.assertNotIn("~3000.0", content)


# ---------------------------------------------------------------------------
# CSV export (new column)
# ---------------------------------------------------------------------------


class CsvEvalTokSecFlooredColumnTests(unittest.TestCase):
    def test_csv_columns_includes_eval_tok_sec_floored(self):
        """`SCORECARD_CSV_COLUMNS` ends with the new boolean column."""
        from benchrig.reporting.csv_export import SCORECARD_CSV_COLUMNS

        self.assertEqual(SCORECARD_CSV_COLUMNS[-1], "eval_tok_sec_floored")

    def test_csv_round_trips_floored_true(self):
        """`write_scorecards_csv` writes `True` for the new column on a floored scorecard."""
        from benchrig.reporting.csv_export import write_scorecards_csv

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "out.csv")
            write_scorecards_csv(
                [_scorecard(eval_tok_sec_floored=True)],
                csv_path,
            )
            with open(csv_path) as fh:
                rows = list(csv.reader(fh))
            self.assertEqual(rows[0][-1], "eval_tok_sec_floored")
            self.assertEqual(rows[1][-1], "True")

    def test_csv_round_trips_floored_false(self):
        """`write_scorecards_csv` writes `False` for the new column on a non-floored scorecard."""
        from benchrig.reporting.csv_export import write_scorecards_csv

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "out.csv")
            write_scorecards_csv(
                [_scorecard(eval_tok_sec_floored=False)],
                csv_path,
            )
            with open(csv_path) as fh:
                rows = list(csv.reader(fh))
            self.assertEqual(rows[1][-1], "False")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Review fixes (Phase 10 review, 2026-09-23)
# ---------------------------------------------------------------------------


class RunnerPropagationTests(unittest.TestCase):
    """The per-record flag must survive the runner: response -> record -> scorecard (review finding 1)."""

    def _runner(self, flagged):
        from benchrig.core.client import BaseRuntimeClient
        from benchrig.core.runner import BenchmarkRunner

        class Fake(BaseRuntimeClient):
            name = "fake"
            display_name = "Fake"
            engine_name = "fake-engine"

            def generate(self, model, prompt, system=None, options=None, measure_ttft=True):
                return {
                    "success": True,
                    "response": "1234",
                    "eval_count": 3,
                    "eval_tok_per_sec": 3000.0,
                    "eval_tok_sec_floored": flagged,
                    "prompt_eval_count": 5,
                    "prompt_tok_per_sec": 100.0,
                    "ttft_sec": 0.1,
                    "total_time_sec": 1.0,
                }

        with (
            patch("benchrig.core.runner.time.sleep"),
            patch("benchrig.core.runner.get_system_specs", return_value={}),
            patch("benchrig.core.runner.HardwareSampler"),
        ):
            runner = BenchmarkRunner(client=Fake(base_url="http://fake"), config={})
            records = runner.run_speed_suite("m", [{"id": "s1", "name": "Speed", "prompt": "hi"}])
        return runner, records

    def test_record_and_scorecard_carry_the_floor_flag(self):
        runner, records = self._runner(True)
        self.assertTrue(records[0]["eval_tok_sec_floored"])
        self.assertTrue(runner.compute_model_scorecard("m", records)["eval_tok_sec_floored"])

    def test_record_and_scorecard_are_not_flagged_when_the_response_is_not(self):
        runner, records = self._runner(False)
        self.assertFalse(records[0].get("eval_tok_sec_floored", False))
        self.assertFalse(runner.compute_model_scorecard("m", records)["eval_tok_sec_floored"])


class FloorBoundaryTests(unittest.TestCase):
    """Boundary values of the raw eval window on the Foundry/Prism path (review findings 2 and 3, 16)."""

    def _client(self):
        from unittest.mock import MagicMock

        from benchrig.core.client import PrismClient

        client = object.__new__(PrismClient)
        client.base_url = "http://127.0.0.1:5272/v1"
        client.name = "prism"
        client.engine_name = "ONNX Runtime GenAI"
        client.timeout_sec = 30
        client.default_num_ctx = 4096
        client.default_max_tokens = None
        client.api_key = None
        client._engines = {}
        client._request_kwargs = MagicMock(return_value={})
        client.unload_model = lambda *_a, **_k: True
        return client

    def _stream(self, n=3):
        import json as _json
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status.return_value = None
        chunks = [
            f"data: {_json.dumps({'choices': [{'delta': {'content': 'ok'}, 'index': 0}]})}\n\n".encode()
            for _ in range(n)
        ]
        chunks.append(b"data: [DONE]\n\n")
        mock.iter_lines.return_value = chunks
        return mock

    def _generate_with_clock(self, ticks):
        """Run a streamed generate() whose perf_counter reads come from ``ticks`` (last value repeats)."""
        client = self._client()
        state = {"i": 0}

        def clock():
            v = ticks[min(state["i"], len(ticks) - 1)]
            state["i"] += 1
            return v

        with patch("benchrig.core.client.requests.post", return_value=self._stream()):
            with patch("benchrig.core.client.time.perf_counter", side_effect=clock):
                return client.generate("phi4", "hi")

    def test_zero_length_eval_window_with_tokens_is_flagged(self):
        """Raw window of exactly 0.0 (first token and end at the same instant) engages the floor: flag it."""
        result = self._generate_with_clock([0.0])
        self.assertEqual(result["eval_count"], 3)
        self.assertEqual(result["eval_tok_per_sec"], 3000.0)
        self.assertTrue(result["eval_tok_sec_floored"])

    def test_window_at_the_threshold_is_not_flagged(self):
        """0.0015 s is not below the threshold: real measurement, no tilde."""
        # The eval window is first-token -> end, i.e. it grows only with the final clock read. Count the reads
        # with a constant clock first, so the test does not depend on how many the request path makes.
        reads = []
        with patch("benchrig.core.client.requests.post", return_value=self._stream()):
            with patch("benchrig.core.client.time.perf_counter", side_effect=lambda: reads.append(0) or 0.0):
                self._client().generate("phi4", "hi")
        result = self._generate_with_clock([0.0] * (len(reads) - 1) + [0.0015])
        self.assertEqual(result["eval_tok_per_sec"], 2000.0)  # 3 tokens / 0.0015 s: a real measurement
        self.assertFalse(result["eval_tok_sec_floored"])

    def test_non_streaming_instant_response_is_flagged(self):
        from unittest.mock import MagicMock

        client = self._client()
        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status.return_value = None
        mock.json.return_value = {
            "choices": [{"message": {"content": "abc"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        with patch("benchrig.core.client.requests.post", return_value=mock):
            with patch("benchrig.core.client.time.perf_counter", return_value=0.0):
                result = client.generate("phi4", "hi", measure_ttft=False)
        self.assertEqual(result["eval_count"], 3)
        self.assertTrue(result["eval_tok_sec_floored"])

    def test_ollama_server_reported_short_window_is_not_flagged(self):
        """Ollama has no 0.001 s floor: a server-reported 1.2 ms window is a measurement, not an artifact."""
        import json as _json
        from unittest.mock import MagicMock

        from benchrig.core.client import OllamaClient

        client = OllamaClient(base_url="http://127.0.0.1:11434")
        mock = MagicMock()
        mock.status_code = 200
        mock.raise_for_status.return_value = None
        final = {
            "response": "abc",
            "done": True,
            "eval_count": 3,
            "eval_duration": 1_200_000,
            "prompt_eval_count": 5,
            "prompt_eval_duration": 5_000_000,
            "load_duration": 0,
            "total_duration": 7_000_000,
        }
        mock.iter_lines.return_value = [_json.dumps(final).encode()]
        mock.json.return_value = final
        with patch("benchrig.core.client.requests.post", return_value=mock):
            result = client.generate("m", "hi", measure_ttft=False)
        self.assertEqual(result["eval_count"], 3)
        self.assertFalse(result["eval_tok_sec_floored"])


class OlderRunJsonTests(unittest.TestCase):
    """Scorecards from runs that predate the flag must render and export without it (review finding 16)."""

    def test_csv_exports_a_scorecard_that_lacks_the_key(self):
        from benchrig.reporting.csv_export import SCORECARD_CSV_COLUMNS, write_scorecards_csv

        sc = _scorecard()
        sc.pop("eval_tok_sec_floored", None)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "o.csv")
            write_scorecards_csv([sc], path)
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0].keys()), list(SCORECARD_CSV_COLUMNS))
        self.assertEqual(rows[0]["eval_tok_sec_floored"], "")  # unknown for a run that predates the flag
