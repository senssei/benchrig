import copy
import json
from collections import Counter
from importlib.resources import files

import pytest


def scenario():
    return {
        "id": "case",
        "name": "Case",
        "category": "selection",
        "prompt": "Read a.txt",
        "check_type": "tool_call",
        "options": {"temperature": 0, "seed": 42},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "turns": [{"expect": {"tool": "read_file", "args": {"path": "a.txt"}}}],
    }


def test_bundled_categories_have_three_cases_each():
    from benchrig.core.tool_use import validate_scenarios

    cases = json.loads(files("benchrig").joinpath("data/scenarios/tool_use.json").read_text())
    validate_scenarios(cases, "tool_use.json")
    assert len(cases) >= 18
    counts = Counter(c["category"] for c in cases)
    assert set(counts) == {"selection", "arguments", "abstention", "multistep", "polish", "retrieval"}
    assert min(counts.values()) >= 3


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(id=""),
        lambda c: c["turns"][0]["expect"].update(tool="missing"),
        lambda c: c["tools"][0]["function"]["parameters"].update(required=["absent"]),
        lambda c: c["turns"][0]["expect"].update(args={"path": 123}),
    ],
)
def test_invalid_scenario_is_rejected_before_requests(mutate):
    from benchrig.core.tool_use import validate_scenarios

    c = scenario()
    mutate(c)
    with pytest.raises(ValueError, match="custom.json"):
        validate_scenarios([c], "custom.json")


def test_multistep_requires_fixture_result():
    from benchrig.core.tool_use import validate_scenarios

    c = scenario()
    c["turns"].append(copy.deepcopy(c["turns"][0]))
    with pytest.raises(ValueError, match="tool_result"):
        validate_scenarios([c])


def test_abstention_requires_answer():
    from benchrig.core.tool_use import validate_scenarios

    c = scenario()
    c["turns"][0]["expect"] = {"tool": None, "args": {}}
    with pytest.raises(ValueError, match="answer"):
        validate_scenarios([c])


def test_typed_matchers_are_validated():
    from benchrig.core.tool_use import validate_scenarios

    c = scenario()
    c["turns"][0]["expect"]["args"] = {"path": {"type": "string", "contains": "txt"}}
    validate_scenarios([c])
    c["turns"][0]["expect"]["args"]["path"]["type"] = "integer"
    with pytest.raises(ValueError, match="matcher"):
        validate_scenarios([c])
