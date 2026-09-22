"""Tests for the Prism (prism-local) runtime: client, CLI wiring and report labels."""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import requests

from benchrig import cli
from benchrig.core.client import PrismClient, create_runtime_client
from benchrig.core.runtimes import runtime_label


def response(payload, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


class PrismClientTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRISM_API_KEY", None)
            self.client = PrismClient()

    def test_defaults_and_identity(self):
        self.assertEqual(self.client.name, "prism")
        self.assertEqual(self.client.display_name, "Prism")
        self.assertEqual(self.client.base_url, "http://127.0.0.1:5272/v1")
        self.assertFalse(self.client.auto_detect_port)
        self.assertFalse(self.client._managed_by_foundry_cli())

    def test_never_discovers_a_foundry_daemon(self):
        with patch("os.path.isfile", return_value=True), patch("builtins.open", side_effect=AssertionError("read")):
            client = PrismClient(base_url="http://127.0.0.1:5272/v1")
        self.assertEqual(client.base_url, "http://127.0.0.1:5272/v1")

    def test_api_key_from_argument_env_and_header(self):
        self.assertEqual(self.client._request_kwargs(), {})
        with patch.dict(os.environ, {"PRISM_API_KEY": "from-env"}):
            self.assertEqual(PrismClient().api_key, "from-env")
        keyed = PrismClient(api_key="secret")
        self.assertEqual(keyed._request_kwargs(), {"headers": {"Authorization": "Bearer secret"}})

    @patch("requests.get")
    def test_requests_carry_the_bearer_token(self, mock_get):
        mock_get.return_value = response({"data": [{"id": "phi-4-mini", "owned_by": "ONNX Runtime GenAI"}]})
        keyed = PrismClient(api_key="secret")
        self.assertTrue(keyed.is_reachable())
        keyed.list_installed_models()
        for call in mock_get.call_args_list:
            self.assertEqual(call.kwargs["headers"], {"Authorization": "Bearer secret"})

    @patch("requests.get")
    def test_get_version_reports_active_device(self, mock_get):
        mock_get.return_value = response({"status": "ok", "active_model": "m", "active_device": "cuda"})
        self.assertEqual(self.client.get_version(), "Prism (active device: cuda)")
        mock_get.return_value = response({"status": "ok", "active_model": None, "active_device": None})
        self.assertEqual(self.client.get_version(), "Prism (idle)")
        mock_get.side_effect = OSError("down")
        self.assertEqual(self.client.get_version(), "Prism")

    @patch("requests.get")
    @patch("requests.post")
    def test_generate_records_the_device_that_ran_the_request(self, mock_post, mock_get):
        mock_post.return_value = response(
            {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "telemetry": {"device": "cuda"},
            }
        )
        result = self.client.generate("phi-4-mini", "hi", measure_ttft=False)
        self.assertTrue(result["success"])
        self.assertEqual(result["device"], "cuda")
        self.assertEqual(result["runtime"], "prism")
        mock_get.assert_not_called()  # telemetry was enough, no /health round trip

    @patch("requests.get")
    @patch("requests.post")
    def test_generate_falls_back_to_health_for_the_device(self, mock_post, mock_get):
        mock_post.return_value = response(
            {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        )
        mock_get.return_value = response({"status": "ok", "active_device": "cpu"})
        self.assertEqual(self.client.generate("m", "hi", measure_ttft=False)["device"], "cpu")

    @patch("requests.get")
    @patch("requests.post")
    def test_generate_does_not_guess_the_device_of_ollama_served_models(self, mock_post, mock_get):
        mock_post.return_value = response(
            {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        )
        mock_get.return_value = response({"status": "ok", "active_device": "cuda"})  # stale: last ONNX model's device
        result = self.client.generate("ollama:qwen2.5-coder:3b", "hi", measure_ttft=False)
        self.assertTrue(result["success"])
        self.assertNotIn("device", result)
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_load_model_checks_listing_without_touching_the_foundry_cli(self, mock_get):
        mock_get.return_value = response({"data": [{"id": "phi-4-mini"}]})
        with patch("benchrig.core.client.subprocess.run") as run:
            self.assertTrue(self.client.load_model("phi-4-mini"))
            self.assertFalse(self.client.load_model("missing"))
            self.assertTrue(self.client.unload_model("phi-4-mini"))
        run.assert_not_called()

    @patch("benchrig.core.client.shutil.which", return_value=None)
    def test_pull_without_prism_cli_reports_an_error(self, _which):
        events = []
        self.assertFalse(self.client.pull_model("phi-4-mini", stream_callback=events.append))
        self.assertEqual(events[-1]["status"], "error")

    @patch("benchrig.core.client.subprocess.run")
    @patch("benchrig.core.client.shutil.which", return_value="/usr/bin/prism")
    def test_pull_runs_prism_pull(self, _which, run):
        run.return_value = MagicMock(returncode=0, stderr="")
        self.assertTrue(self.client.pull_model("phi-4-mini"))
        self.assertEqual(run.call_args.args[0], ["/usr/bin/prism", "pull", "phi-4-mini"])

    def test_factory_reads_the_prism_config_section(self):
        client = create_runtime_client(
            "prism", {"prism": {"base_url": "http://10.0.0.5:9000/v1", "api_key": "k", "timeout_sec": 7}}
        )
        self.assertIsInstance(client, PrismClient)
        self.assertEqual((client.base_url, client.api_key, client.timeout_sec), ("http://10.0.0.5:9000/v1", "k", 7))
        self.assertIsInstance(create_runtime_client("prism-local", {}), PrismClient)


class UnloadModelTests(unittest.TestCase):
    """Phase 5 (spec.md I7): `PrismClient.unload_model` calls `POST /v1/unload` on prism-local HEAD (6d6467a),
    instead of the inherited `FoundryClient.unload_model` no-op (which only acts when discovered via the
    `foundry` CLI, never true for Prism)."""

    def setUp(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRISM_API_KEY", None)
            self.client = PrismClient()

    @patch("requests.post")
    def test_unload_model_posts_to_v1_unload(self, mock_post):
        mock_post.return_value = response({"unloaded": True, "model": "phi-4-mini"})
        self.assertTrue(self.client.unload_model("phi-4-mini"))
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args.args[0], "http://127.0.0.1:5272/v1/unload")

    @patch("requests.post")
    def test_unload_model_sends_the_bearer_token_when_api_key_is_set(self, mock_post):
        mock_post.return_value = response({"unloaded": False, "model": None})
        keyed = PrismClient(api_key="secret")
        self.assertTrue(keyed.unload_model("phi-4-mini"))
        self.assertEqual(mock_post.call_args.kwargs["headers"], {"Authorization": "Bearer secret"})

    @patch("requests.post", side_effect=OSError("connection refused"))
    def test_unload_model_is_best_effort_on_transport_error(self, mock_post):
        self.assertTrue(self.client.unload_model("phi-4-mini"))
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_unload_model_is_best_effort_on_404_from_an_older_prism_server(self, mock_post):
        mock_post.return_value = response({"error": {"message": "not found"}}, status=404)
        mock_post.return_value.raise_for_status.side_effect = requests.HTTPError("404")
        self.assertTrue(self.client.unload_model("phi-4-mini"))
        mock_post.assert_called_once()


class UsageTests(unittest.TestCase):
    """Token counts are the server's when it reports `usage`, and marked as estimates when it does not."""

    def stream(self, *chunks):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.iter_lines.return_value = [f"data: {json.dumps(c)}".encode() for c in chunks] + [b"data: [DONE]"]
        return resp

    def generate(self, *chunks):
        with (
            patch("requests.post", return_value=self.stream(*chunks)),
            patch("requests.get", return_value=response({})),
        ):
            return PrismClient().generate("m", "one two three four", measure_ttft=True)

    def test_usage_and_telemetry_from_the_last_chunk_are_used(self):
        result = self.generate(
            {"choices": [{"delta": {"content": "a"}}]},
            {"choices": [{"delta": {"content": "b"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}, "telemetry": {"device": "cuda"}},
        )
        self.assertEqual((result["prompt_eval_count"], result["eval_count"]), (7, 2))
        self.assertFalse(result["usage_estimated"])
        self.assertEqual(result["device"], "cuda")  # from the stream itself, no /health round trip needed

    def test_a_server_that_sends_no_usage_is_marked_as_estimated(self):
        result = self.generate(
            {"choices": [{"delta": {"content": "a"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        )
        self.assertTrue(result["usage_estimated"])
        self.assertEqual(result["prompt_eval_count"], int(4 * 1.3))

    def test_the_flag_reaches_records_scorecards_and_reports(self):
        import os
        import tempfile

        from benchrig.reporting.markdown import generate_markdown_report
        from tests.test_measurement_methodology import FakeClient, make_runner

        runner = make_runner(FakeClient(usage_estimated=True))
        records = runner.run_speed_suite("m", [{"id": "s", "name": "S", "prompt": "x"}])
        self.assertTrue(records[0]["usage_estimated"])
        card = runner.compute_model_scorecard("m", records)
        self.assertTrue(card["usage_estimated"])
        with tempfile.TemporaryDirectory() as tmp:
            content = generate_markdown_report(
                [{**card, "peak_vram_mb": 1.0}], [], {}, output_path=os.path.join(tmp, "r.md")
            )
        self.assertIn("Estimated token counts for: `m`", content)

    def test_exact_counts_add_nothing(self):
        from tests.test_measurement_methodology import FakeClient, make_runner

        runner = make_runner(FakeClient())
        records = runner.run_speed_suite("m", [{"id": "s", "name": "S", "prompt": "x"}])
        self.assertNotIn("usage_estimated", records[0])
        self.assertFalse(runner.compute_model_scorecard("m", records)["usage_estimated"])

    def test_device_comes_from_telemetry_not_from_export_label(self):
        """Regression for plan Phase 4 item 4.2: device must come from the response's telemetry.device
        (the device that actually ran the request), never from `exported_for` (a user-supplied label
        that can differ). The Prism server can put `exported_for` in any payload field; the parser
        ignores it.
        """
        # Stream mimics a Prism 0.2.0 response where the model was exported for CPU and runs on CUDA.
        result = self.generate(
            {"choices": [{"delta": {"content": "a"}}]},
            {"choices": [{"delta": {"content": "b"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                "telemetry": {"device": "CUDA (GPU)"},
                # `exported_for` belongs to /v1/models, not to the chat completion response, but a
                # future change could leak it here; the parser must still ignore it.
                "exported_for": "cpu",
            },
        )
        self.assertEqual(result["device"], "CUDA (GPU)", "device must come from telemetry, not exported_for")
        self.assertFalse(result["usage_estimated"])


class Retry503Tests(unittest.TestCase):
    """Phase 4 item 4.5: prism-local returns 503 with Retry-After: 30 for `server_busy` and
    `insufficient_resources` (prism/server.py:669-671). Benchrig retries up to 3× with the
    server-provided delay (capped at 60 s) and raises PrismBusyError after that. Other 5xx are NOT
    retried."""

    def _response(self, status, *, body=None, retry_after=None):
        resp = MagicMock()
        resp.status_code = status
        resp.raise_for_status = MagicMock(
            side_effect=requests.HTTPError(f"{status} simulated", response=resp) if status >= 400 else None
        )
        resp.json.return_value = body if body is not None else {}
        resp.text = json.dumps(body) if body is not None else ""
        resp.headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
        resp.iter_lines.return_value = (
            [f"data: {json.dumps(c)}".encode() for c in body.get("chunks", [])] + [b"data: [DONE]"]
            if status == 200 and body
            else []
        )
        return resp

    def test_prism_client_retries_on_503_with_retry_after(self):
        from benchrig.core.client import PrismClient

        ok_body = {
            "chunks": [
                {"choices": [{"delta": {"content": "a"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ],
        }
        busy_body = {"error": {"code": "server_busy", "message": "queue full"}}
        responses = [
            self._response(503, body=busy_body, retry_after=1),
            self._response(200, body=ok_body),
        ]
        with (
            patch("requests.post", side_effect=responses),
            patch("requests.get", return_value=response({})),
            patch("time.sleep") as mocked_sleep,
        ):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PRISM_API_KEY", None)
                client = PrismClient()
                result = client.generate("m", "hi", measure_ttft=False)
        self.assertEqual(result["success"], True)
        self.assertEqual(mocked_sleep.call_count, 1, "must sleep exactly once between the two POSTs")
        # The sleep argument is the parsed Retry-After (1).
        self.assertEqual(mocked_sleep.call_args.args[0], 1)

    def test_prism_client_gives_up_after_three_retries(self):
        from benchrig.core.client import PrismClient

        busy_body = {"error": {"code": "insufficient_resources", "message": "load lock held by another process"}}
        responses = [
            self._response(503, body=busy_body, retry_after=0)
            for _ in range(4)  # initial + 3 retries = 4 POSTs total
        ]
        with (
            patch("requests.post", side_effect=responses),
            patch("requests.get", return_value=response({})),
            patch("time.sleep"),
        ):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PRISM_API_KEY", None)
                client = PrismClient()
                # The benchrig generate() swallows exceptions into a failure result (matches
                # existing behaviour for HTTPError). The retry helper DID exhaust its budget
                # and the failure result carries the PrismBusyError reason so the user can
                # distinguish "queue full" from "load lock held" from a real 500.
                result = client.generate("m", "hi", measure_ttft=False)
        self.assertFalse(result["success"], "after 4×503 the run must be marked failed")
        self.assertIn("insufficient_resources", str(result.get("error", "")))
        self.assertIn("load lock", str(result.get("error", "")))

    def test_post_with_503_retry_raises_prism_busy_after_budget(self):
        """The helper itself raises PrismBusyError when its budget runs out — exercised directly,
        without the swallow in generate()."""
        from benchrig.core.client import PrismBusyError, _post_with_503_retry

        busy_body = {"error": {"code": "insufficient_resources", "message": "load lock held"}}
        responses = [self._response(503, body=busy_body, retry_after=0) for _ in range(4)]
        with (
            patch("requests.post", side_effect=responses),
            patch("time.sleep"),
        ):
            with self.assertRaises(PrismBusyError) as ctx:
                _post_with_503_retry("http://x/y")
        self.assertEqual(ctx.exception.attempts, 4)
        self.assertIn("insufficient_resources", ctx.exception.reason)
        self.assertIn("load lock held", ctx.exception.reason)

    def test_prism_client_does_not_retry_on_500(self):
        from benchrig.core.client import PrismClient

        resp500 = self._response(500, body={"error": "oops"}, retry_after=1)
        with (
            patch("requests.post", return_value=resp500),
            patch("requests.get", return_value=response({})),
            patch("time.sleep") as mocked_sleep,
        ):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PRISM_API_KEY", None)
                client = PrismClient()
                # 500 is not retried; the benchrig generate() swallows the HTTPError into a
                # failure result (matches existing behaviour). The point of the test is that
                # NO sleep was triggered — i.e. the retry helper did NOT engage.
                result = client.generate("m", "hi", measure_ttft=False)
        self.assertFalse(result["success"], "500 produces a failure result, not a success")
        self.assertEqual(mocked_sleep.call_count, 0, "500 must NOT trigger a retry")
        self.assertEqual(resp500.raise_for_status.call_count, 1)


class CliWiringTests(unittest.TestCase):
    def fake_client(self, names, reachable=True):
        client = MagicMock()
        client.is_reachable.return_value = reachable
        client.list_installed_models.return_value = [{"name": n} for n in names]
        return client

    def test_parser_accepts_prism_runtime(self):
        self.assertEqual(cli.build_parser().parse_args(["--runtime", "prism"]).runtime, "prism")

    def test_prefixes_route_to_prism(self):
        clients = {"prism": self.fake_client(["phi-4-mini"])}
        self.assertEqual(cli.resolve_target_models("prism:phi-4-mini", "ollama", clients), [("prism", "phi-4-mini")])
        self.assertEqual(
            cli.resolve_target_models("prism-local:phi-4-mini", "ollama", clients), [("prism", "phi-4-mini")]
        )
        self.assertEqual(
            cli.resolve_target_models("prism:ollama:qwen2.5-coder:3b", "ollama", clients),
            [("prism", "ollama:qwen2.5-coder:3b")],
        )

    def test_installed_skips_ollama_models_proxied_by_prism(self):
        clients = {"prism": self.fake_client(["phi-4-mini-cuda", "ollama:qwen2.5-coder:3b", "qwen3-0.6b"])}
        targets = cli.resolve_target_models("installed", "prism", clients)
        self.assertEqual(targets, [("prism", "phi-4-mini-cuda"), ("prism", "qwen3-0.6b")])

    def test_all_runtimes_includes_prism(self):
        clients = {
            "ollama": self.fake_client(["phi3:mini"]),
            "prism": self.fake_client(["phi-4-mini", "ollama:phi3:mini"]),
        }
        targets = cli.resolve_target_models("installed", "all", clients)
        self.assertIn(("ollama", "phi3:mini"), targets)
        self.assertIn(("prism", "phi-4-mini"), targets)
        self.assertNotIn(("prism", "ollama:phi3:mini"), targets)

    def test_pair_with_baseline_uses_prism_model_then_falls_back(self):
        config = {"model_pairs_1to1": [{"id": "p", "ollama": "o", "foundry": "f", "prism": "x", "onnx": "y"}]}
        self.assertEqual(cli.resolve_pair_targets(config, "p", "prism", has_baseline=True)[0], [("prism", "x")])
        del config["model_pairs_1to1"][0]["prism"]
        self.assertEqual(cli.resolve_pair_targets(config, "p", "prism", has_baseline=True)[0], [("prism", "y")])

    def test_check_reports_a_prism_server(self):
        client = self.fake_client(["phi-4-mini"])
        client.base_url = "http://127.0.0.1:5272/v1"
        client.get_version.return_value = "Prism (idle)"
        client.list_installed_models.return_value = [
            {"name": "phi-4-mini", "details": {"engine": "ONNX Runtime GenAI"}}
        ]
        cli._check_prism(client)
        cli._check_prism(None)  # not running: prints a hint, never raises

    def test_bundled_config_has_a_prism_section(self):
        config = cli.load_config()
        self.assertEqual(config["prism"]["base_url"], "http://127.0.0.1:5272/v1")


class LabelTests(unittest.TestCase):
    def test_runtime_labels(self):
        self.assertEqual(runtime_label("prism"), "Prism")
        self.assertEqual(runtime_label("foundry"), "MS Foundry")
        self.assertEqual(runtime_label("onnx-gpu"), "ONNX GenAI")
        self.assertEqual(runtime_label("ollama"), "Ollama")
        self.assertEqual(runtime_label(None), "Ollama")
        self.assertEqual(runtime_label("weird", default="?"), "?")


if __name__ == "__main__":
    unittest.main()
