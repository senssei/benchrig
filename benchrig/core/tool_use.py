"""Validation and evidence for inert, fixture-based tool-use benchmarks."""

from __future__ import annotations

import math
from typing import Any

JSON_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def _type_matches(value: Any, kind: str) -> bool:
    if kind == "number":
        return type(value) in (int, float) and math.isfinite(value)
    return type(value) is {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "boolean": bool,
        "null": type(None),
    }.get(kind)


def _schema_valid(schema: Any) -> bool:
    if not isinstance(schema, dict) or not isinstance(schema.get("type"), str) or schema.get("type") not in JSON_TYPES:
        return False
    kind = schema["type"]
    allowed = {"type", "description", "enum"}
    if kind == "object":
        allowed |= {"properties", "required", "additionalProperties"}
        props, required = schema.get("properties"), schema.get("required", [])
        if not isinstance(props, dict) or not all(isinstance(k, str) and _schema_valid(v) for k, v in props.items()):
            return False
        if not isinstance(required, list) or any(not isinstance(k, str) or k not in props for k in required):
            return False
        if len(required) != len(set(required)) or schema.get("additionalProperties") is not False:
            return False
    if kind == "array":
        allowed.add("items")
        if not _schema_valid(schema.get("items")):
            return False
    if set(schema) - allowed:
        return False
    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum or any(not _type_matches(v, kind) for v in enum):
            return False
    return True


def conforms(value: Any, schema: dict) -> bool:
    if not _type_matches(value, schema["type"]):
        return False
    if "enum" in schema and not any(type(value) is type(v) and value == v for v in schema["enum"]):
        return False
    if schema["type"] == "object":
        props = schema["properties"]
        return (
            set(schema.get("required", [])) <= value.keys()
            and value.keys() <= props.keys()
            and all(conforms(v, props[k]) for k, v in value.items())
        )
    if schema["type"] == "array":
        return all(conforms(v, schema["items"]) for v in value)
    return True


def validate_scenarios(scenarios: Any, source: str = "tool_use.json") -> None:
    """Reject undefined expectations before any runtime or model request."""

    def fail(sid, field):
        raise ValueError(f"{source}:{sid}: invalid {field}")

    if not isinstance(scenarios, list) or not scenarios:
        fail("?", "scenario array")
    seen = set()
    for sc in scenarios:
        if not isinstance(sc, dict):
            fail("?", "scenario object")
        sid = sc.get("id", "?")
        for key in ("id", "name", "category", "prompt"):
            if not isinstance(sc.get(key), str) or not sc[key].strip():
                fail(sid, key)
        if sid in seen:
            fail(sid, "duplicate id")
        seen.add(sid)
        if sc.get("check_type") != "tool_call" or not isinstance(sc.get("options"), dict):
            fail(sid, "check_type/options")
        options = sc["options"]
        for key, value in options.items():
            if key in ("temperature", "top_p", "repetition_penalty") and not _type_matches(value, "number"):
                fail(sid, f"options.{key}")
            if key in ("seed", "num_predict", "max_tokens", "num_ctx", "top_k") and type(value) is not int:
                fail(sid, f"options.{key}")
        if not isinstance(sc.get("tools"), list) or not sc["tools"]:
            fail(sid, "tools")
        schemas = {}
        for tool in sc["tools"]:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                fail(sid, "tools.function")
            fn = tool.get("function")
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
                fail(sid, "tools.function.name")
            schema = fn.get("parameters")
            if not _schema_valid(schema) or schema["type"] != "object" or fn["name"] in schemas:
                fail(sid, "tools.function.parameters")
            schemas[fn["name"]] = schema
        turns = sc.get("turns")
        if not isinstance(turns, list) or not turns:
            fail(sid, "turns")
        for i, turn in enumerate(turns):
            if not isinstance(turn, dict) or not isinstance(turn.get("expect"), dict):
                fail(sid, f"turns.{i}.expect")
            exp = turn["expect"]
            if "tool" not in exp or not isinstance(exp.get("args"), dict):
                fail(sid, f"turns.{i}.expect")
            name = exp["tool"]
            if name is None:
                if exp["args"] or not isinstance(exp.get("answer"), str) or not exp["answer"].strip():
                    fail(sid, f"turns.{i}.answer")
            else:
                if not isinstance(name, str) or name not in schemas:
                    fail(sid, f"turns.{i}.tool")
                args = dict(exp["args"])
                for key, value in args.items():
                    if isinstance(value, dict) and "contains" in value:
                        if (
                            set(value) != {"type", "contains"}
                            or value["type"] != "string"
                            or not isinstance(value["contains"], str)
                        ):
                            fail(sid, f"turns.{i}.matcher")
                        args[key] = value["contains"]
                if not conforms(args, schemas[name]):
                    fail(sid, f"turns.{i}.args")
            if i < len(turns) - 1 and (name is None or not isinstance(turn.get("tool_result"), str)):
                fail(sid, f"turns.{i}.tool_result")


def strict_json(text: str) -> Any:
    """Decode call arguments without silently accepting duplicate keys or NaN."""
    import json

    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate argument key: {key}")
            out[key] = value
        return out

    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite_float)


def content_has_call(content: str) -> bool:
    import re

    text = content.strip()
    fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence[1]
    try:
        value = strict_json(text)
    except (ValueError, TypeError):
        return False

    def shaped(item):
        return isinstance(item, dict) and (
            ("name" in item and "arguments" in item)
            or ("tool" in item and "args" in item)
            or "tool_calls" in item
            or (isinstance(item.get("function"), dict) and "name" in item["function"])
        )

    return any(shaped(v) for v in value) if isinstance(value, list) else shaped(value)


def normalize_calls(raw: Any, native: bool) -> tuple[list[dict], list[str]]:
    calls, errors = [], []
    if not isinstance(raw, list):
        return [], ["tool_calls must be an array"]
    for i, call in enumerate(raw):
        try:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise ValueError("missing function object")
            fn = call["function"]
            if not isinstance(fn.get("name"), str) or not fn["name"]:
                raise ValueError("missing function name")
            arguments = fn.get("arguments")
            if native:
                import json

                json.dumps(arguments, allow_nan=False)
                if not isinstance(arguments, dict):
                    raise ValueError("native arguments must be an object")
            else:
                if not isinstance(arguments, str):
                    raise ValueError("arguments must be a JSON string")
                arguments = strict_json(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("arguments must decode to an object")
            calls.append({"id": call.get("id"), "name": fn["name"], "arguments": arguments})
        except (ValueError, TypeError) as exc:
            errors.append(f"call {i}: {exc}")
    return calls, errors


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict) and set(expected) == {"type", "contains"}:
        return type(actual) is str and expected["type"] == "string" and expected["contains"] in actual
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_equal(a, b) for a, b in zip(actual, expected, strict=True))
    return actual == expected


def score_turn(expect: dict, response: dict, schema: dict | None = None) -> dict:
    """Score the expected channel, call cardinality, schema and literal arguments."""
    available = response.get("tool_status") == "ok" and response.get("success") and not response.get("truncated")
    calls = response.get("tool_calls", [])
    valid = bool(available and not response.get("parse_errors"))
    selected = bool(valid and len(calls) == 1 and calls[0].get("name") == expect["tool"] and expect["tool"] is not None)
    arguments = bool(
        selected
        and _equal(calls[0].get("arguments"), expect["args"])
        and (schema is None or conforms(calls[0]["arguments"], schema))
    )
    abstention = bool(
        valid
        and expect["tool"] is None
        and not calls
        and not response.get("structured_call")
        and not response.get("json_in_content")
        and response.get("response", "").strip() == expect["answer"].strip()
    )
    return {
        "passed": arguments if expect["tool"] is not None else abstention,
        "selection_correct": selected,
        "arguments_correct": arguments,
        "abstention_correct": abstention,
        "false_call": bool(
            expect["tool"] is None and (response.get("structured_call") or response.get("json_in_content"))
        ),
    }


def aggregate_tools(records: list[dict]) -> dict:
    """Aggregate planned turn denominators; warm-ups and unsupported are ineligible."""
    measured = [r for r in records if r.get("phase") != "warmup"]
    eligible = [r for r in measured if r.get("tool_status") != "unsupported"]
    turns = [t for r in eligible for t in r.get("turns", [])]
    call_turns = [t for t in turns if t["expect"]["tool"] is not None]
    null_turns = [t for t in turns if t["expect"]["tool"] is None]
    tasks_passed = sum(bool(r.get("success")) for r in eligible)
    out = {
        "tool_tasks_passed": tasks_passed,
        "tool_task_count": len(eligible),
        "tool_unsupported_count": sum(r.get("tool_status") == "unsupported" for r in measured),
        "tool_error_count": sum(t.get("tool_status") == "error" for t in turns),
        "tool_skipped_turn_count": sum(t.get("tool_status") == "not_run_due_to_prior_failure" for t in turns),
        "tool_attempted_turn_count": sum(
            t.get("tool_status") in ("ok", "error") and "request_latency_sec" in t for t in turns
        ),
        "tool_planned_turn_count": len(turns),
        "tool_content_json_count": sum(bool(t.get("json_in_content")) for t in turns),
        "tool_task_pass_rate": tasks_passed / len(eligible) * 100 if eligible else None,
    }
    for name, rows, flag in (
        ("selection", call_turns, "selection_correct"),
        ("argument", call_turns, "arguments_correct"),
        ("abstention", null_turns, "abstention_correct"),
        ("false_call", null_turns, "false_call"),
        ("structured_call", call_turns, "structured_call"),
    ):
        numerator = sum(bool(t.get(flag)) for t in rows)
        out[f"tool_{name}_passed"] = numerator
        out[f"tool_{name}_count"] = len(rows)
        key = f"tool_{name}_rate" if name in ("false_call", "structured_call") else f"tool_{name}_accuracy"
        out[key] = numerator / len(rows) * 100 if rows else None
    latencies = [
        t["request_latency_sec"]
        for t in turns
        if t.get("tool_status") == "ok" and t.get("request_latency_sec") is not None
    ]
    out["tool_request_latency_sec"] = sum(latencies) / len(latencies) if latencies else None
    out["tool_status"] = (
        "unsupported"
        if measured and not eligible
        else ("partial" if out["tool_unsupported_count"] else "error" if out["tool_error_count"] else "ok")
    )
    return out
