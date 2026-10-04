"""Keep the Python 3.10 product CI install independent of Python 3.11 SDLC tools."""

import re
from pathlib import Path

from packaging.requirements import Requirement


def test_kit_dependency_excludes_python310_and_includes_python311():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    requirement = Requirement(re.search(r'"(local-sdlc-kit[^"\n]+)"', text).group(1))
    selected310 = requirement.marker is None or requirement.marker.evaluate({"python_version": "3.10"})
    selected311 = requirement.marker is None or requirement.marker.evaluate({"python_version": "3.11"})
    assert not selected310, "Python 3.10 CI must not install the Python 3.11-only SDLC kit"
    assert selected311
