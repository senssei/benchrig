"""Guards against docs drifting from the code: flags, runtimes, config sections and env vars must be documented."""

import re
import unittest
from pathlib import Path

from benchrig import cli

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def read(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


class DocsCoverageTests(unittest.TestCase):
    def test_every_cli_option_is_documented(self):
        text = read("cli.md")
        options = [
            o for a in cli.build_parser()._actions for o in a.option_strings if o.startswith("--") and o != "--help"
        ]
        self.assertGreater(len(options), 10)
        for option in options:
            self.assertTrue(f"`{option}`" in text, f"{option} is missing from docs/cli.md")

    def test_every_runtime_is_documented(self):
        text = read("runtimes.md")
        for runtime in cli.RUNTIME_CHOICES:
            self.assertTrue(f"`{runtime}`" in text, f"runtime {runtime} is missing from docs/runtimes.md")

    def test_every_config_section_is_documented(self):
        text = read("configuration.md")
        for section in cli.load_config():
            self.assertTrue(f"`{section}`" in text, f"config section {section} is missing from docs/configuration.md")

    def test_every_benchmark_option_is_documented(self):
        text = read("configuration.md")
        for key in cli.load_config()["benchmark"]:
            self.assertTrue(f"`{key}" in text, f"benchmark.{key} is missing from docs/configuration.md")

    def test_environment_variables_are_documented(self):
        text = read("configuration.md")
        for var in (cli.CONFIG_ENV_VAR, "BENCHRIG_MODEL_DIRS", "PRISM_API_KEY"):
            self.assertTrue(var in text, f"{var} is missing from docs/configuration.md")

    def test_every_page_is_in_the_nav_and_every_nav_entry_exists(self):
        nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8").split("\nnav:\n", 1)[1]
        listed = set(re.findall(r":\s*(\S+\.md)\s*$", nav, re.MULTILINE))
        pages = {p.relative_to(DOCS).as_posix() for p in DOCS.rglob("*.md")}
        self.assertEqual(listed, pages)


if __name__ == "__main__":
    unittest.main()
