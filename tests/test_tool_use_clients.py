import json
from unittest.mock import MagicMock, patch

import pytest

from benchrig.core.client import FoundryClient, OllamaClient, PrismClient

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
MESSAGES = [{"role": "user", "content": "Read a.txt"}]


def reply(ollama=False, args=None):
    fn = {
        "name": "read_file",
        "arguments": args if args is not None else ({"path": "a.txt"} if ollama else '{"path":"a.txt"}'),
    }
    message = {"content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": fn}]}
    return (
        {"message": message, "done_reason": "stop", "eval_count": 8}
        if ollama
        else {
            "choices": [{"message": message, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        }
    )


def response(body, status=200):
    r = MagicMock(status_code=status)
    r.json.return_value = body
    r.text = json.dumps(body)
    if status >= 400:
        import requests

        r.raise_for_status.side_effect = requests.HTTPError(r.text)
    return r


def test_ollama_uses_chat_and_object_arguments():
    client = OllamaClient()
    with patch.object(client, "_make_request", return_value=response(reply(True))) as post:
        result = client.chat_tools("model", MESSAGES, TOOLS, {"temperature": 0})
    assert post.call_args.args[0].endswith("/api/chat")
    assert post.call_args.kwargs["json"]["stream"] is False
    assert result["tool_calls"][0]["arguments"] == {"path": "a.txt"}
    assert result["tool_status"] == "ok"


@pytest.mark.parametrize("cls", [FoundryClient, PrismClient])
def test_foundry_and_prism_decode_argument_strings(cls):
    client = cls(base_url="http://localhost:1234/v1", api_key="fixture-token")
    with patch.object(client, "_make_request", return_value=response(reply())) as post:
        result = client.chat_tools("model", MESSAGES, TOOLS, {})
    assert result["tool_calls"][0]["id"] == "call_1"
    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer fixture-token"}
    assert result["usage_estimated"] is False


def test_direct_onnx_returns_unsupported_without_generation():
    from benchrig.core.onnx_client import OnnxGenAiClient

    client = OnnxGenAiClient.__new__(OnnxGenAiClient)
    with patch.object(client, "generate") as generate:
        assert client.chat_tools("model", MESSAGES, TOOLS, {})["tool_status"] == "unsupported"
    generate.assert_not_called()


@pytest.mark.parametrize("args", ["{", "[]", '{"k":1,"k":2}', '{"k":NaN}', '{"k":Infinity}'])
def test_malformed_arguments_preserve_raw_evidence(args):
    client = FoundryClient(base_url="http://localhost:1234/v1")
    with patch.object(client, "_make_request", return_value=response(reply(args=args))):
        result = client.chat_tools("model", MESSAGES, TOOLS, {})
    assert result["parse_errors"]
    assert result["raw_tool_calls"][0]["function"]["arguments"] == args
    assert result["structured_call"] is True


def test_content_json_is_not_a_structured_call():
    client = OllamaClient()
    with patch.object(
        client,
        "_make_request",
        return_value=response({"message": {"content": '{"name":"read_file","arguments":{"path":"a.txt"}}'}}),
    ):
        result = client.chat_tools("model", MESSAGES, TOOLS, {})
    assert result["json_in_content"] is True
    assert result["tool_calls"] == []


@pytest.mark.parametrize(
    "text,status",
    [("model does not support tools", "unsupported"), ("bad request", "error"), ("route not found", "error")],
)
def test_only_explicit_capability_rejection_is_unsupported(text, status):
    client = OllamaClient()
    with patch.object(client, "_make_request", return_value=response({"error": text}, 400)):
        result = client.chat_tools("model", MESSAGES, TOOLS, {})
    assert result["tool_status"] == status
    assert result["http_status"] == 400


def test_chat_tools_preserves_auth_loading_and_retry_contracts():
    client = FoundryClient(base_url="http://localhost:1234/v1", api_key="fixture-token")
    with (
        patch.object(
            client, "_make_request", side_effect=[response({"error": "model is not loaded"}, 400), response(reply())]
        ) as post,
        patch.object(client, "load_model", return_value=True) as load,
    ):
        assert client.chat_tools("model", MESSAGES, TOOLS, {})["tool_status"] == "ok"
    load.assert_called_once_with("model")
    assert post.call_count == 2
    client = PrismClient()
    with patch("benchrig.core.client._post_with_503_retry", return_value=response(reply())) as retry:
        assert client.chat_tools("model", MESSAGES, TOOLS, {})["tool_status"] == "ok"
    retry.assert_called_once()


def test_nonstreaming_latency_is_not_ttft():
    client = OllamaClient()
    with patch.object(client, "_make_request", return_value=response(reply(True))):
        result = client.chat_tools("model", MESSAGES, TOOLS, {})
    assert result["request_latency_sec"] >= 0
    assert result["ttft_sec"] is None
    assert result["decode_duration_sec"] is None


def test_numeric_overflow_is_rejected_in_arguments():
    client = FoundryClient(base_url="http://localhost:1234/v1")
    with patch.object(client, "_make_request", return_value=response(reply(args='{"k":1e999}'))):
        result = client.chat_tools("model", MESSAGES, TOOLS, {})
    assert result["parse_errors"]


def test_server_usage_and_telemetry_are_retained():
    body = reply()
    body["telemetry"] = {"prompt_eval_duration_sec": 0.5, "device": "CPU", "cache_hit": True}
    client = FoundryClient(base_url="http://localhost:1234/v1")
    with patch.object(client, "_make_request", return_value=response(body)):
        result = client.chat_tools("model", MESSAGES, TOOLS, {})
    assert result["prefill_duration_sec"] == 0.5
    assert result["raw_metrics"]["telemetry"] == body["telemetry"]
    assert result["raw_metrics"]["reported_usage"] == body["usage"]


@pytest.mark.parametrize(
    "usage", [{"prompt_tokens": "bad", "completion_tokens": -8}, {"prompt_tokens": True, "completion_tokens": 8}]
)
def test_malformed_usage_is_a_response_error(usage):
    body = reply()
    body["usage"] = usage
    client = FoundryClient(base_url="http://localhost:1234/v1")
    with patch.object(client, "_make_request", return_value=response(body)):
        result = client.chat_tools("model", MESSAGES, TOOLS, {})
    assert result["tool_status"] == "error"
    assert result["success"] is False


def test_request_latency_excludes_prism_health_probe():
    clock = [0.0]
    client = PrismClient()

    def send(*args, **kwargs):
        clock[0] = 0.1
        return response(reply())

    def health():
        clock[0] = 3.1
        return {"active_device": "CPU"}

    with (
        patch("benchrig.core.client.time.perf_counter", side_effect=lambda: clock[0]),
        patch.object(client, "_make_request", side_effect=send),
        patch.object(client, "_engine_for", return_value="ONNX"),
        patch.object(client, "_health", side_effect=health),
    ):
        result = client.chat_tools("model", MESSAGES, TOOLS, {})
    assert result["request_latency_sec"] == 0.1


def test_native_arguments_reject_nested_nonfinite_numbers():
    from benchrig.core.tool_use import normalize_calls

    calls, errors = normalize_calls([{"function": {"name": "f", "arguments": {"nested": [float("inf")]}}}], True)
    assert calls == [] and errors
