"""Packaging guards: version, bundled data, and config lookup order."""

import os
import re
import unittest
from unittest.mock import patch

from benchrig import __version__, cli

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class PackagingTests(unittest.TestCase):
    def test_version_is_semver(self):
        self.assertRegex(__version__, r"^\d+\.\d+\.\d+([abrc.\-+]\w*)*$")

    def test_console_script_points_at_main(self):
        with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
            match = re.search(r'^benchrig = "([\w.]+):(\w+)"', f.read(), re.MULTILINE)  # tomllib needs Python 3.11
        self.assertIsNotNone(match)
        module, func = match.groups()
        self.assertEqual(module, "benchrig.cli")
        self.assertTrue(callable(getattr(cli, func)))

    def test_bundled_config_and_scenarios_exist(self):
        self.assertTrue((cli.BUNDLED_DATA / "config.yaml").is_file())
        for filename, _, _ in cli.SUITES.values():
            self.assertTrue((cli.BUNDLED_DATA / "scenarios" / filename).is_file(), filename)

    def test_changelog_mentions_current_version(self):
        with open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8") as f:
            self.assertRegex(f.read(), rf"## \[{re.escape(__version__)}\]")


class ConfigLookupTests(unittest.TestCase):
    def test_bundled_default_is_used_outside_a_checkout(self):
        with patch.dict(os.environ, {}, clear=False), patch("os.path.isfile", side_effect=lambda p: False):
            os.environ.pop(cli.CONFIG_ENV_VAR, None)
            self.assertEqual(cli.resolve_config_path(), str(cli.BUNDLED_DATA / "config.yaml"))

    def test_explicit_path_wins_over_env_and_cwd(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            explicit = os.path.join(tmp, "a.yaml")
            other = os.path.join(tmp, "b.yaml")
            for path, body in ((explicit, "benchmark: {default_runs: 7}\n"), (other, "benchmark: {default_runs: 9}\n")):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(body)
            with patch.dict(os.environ, {cli.CONFIG_ENV_VAR: other}):
                self.assertEqual(cli.load_config(explicit)["benchmark"]["default_runs"], 7)
                self.assertEqual(cli.load_config()["benchmark"]["default_runs"], 9)

    def test_scenarios_dir_falls_back_to_bundled(self):
        with patch("os.path.isdir", side_effect=lambda p: False):
            self.assertEqual(cli.resolve_scenarios_dir(), cli.SCENARIOS_DIR)


class CliEntryTests(unittest.TestCase):
    def test_version_flag(self):
        with patch("sys.argv", ["benchrig", "--version"]), self.assertRaises(SystemExit) as ctx:
            cli.main()
        self.assertEqual(ctx.exception.code, 0)

    def test_cuda_bootstrap_reexecs_as_module(self):
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("benchrig.cli.glob.glob", return_value=["/fake/nvidia/cublas/lib"]),
            patch("os.path.isdir", return_value=True),
            patch("os.execv") as execv,
            patch("sys.argv", ["/x/benchrig", "--check"]),
        ):
            os.environ.pop("_ONNX_CUDA_BOOTSTRAPPED", None)
            cli._bootstrap_cuda_env()
            os.environ.pop("_ONNX_CUDA_BOOTSTRAPPED", None)
        args = execv.call_args.args[1]
        self.assertEqual(args[1:], ["-m", "benchrig", "--check"])


if __name__ == "__main__":
    unittest.main()
