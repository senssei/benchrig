"""Tests for foreign-process VRAM baseline measurement (plan.md Phase 2, item 2.2)."""

import os
import unittest
from unittest.mock import MagicMock, patch

from benchrig.core.hardware import BaseHardwareProvider, LinuxNvidiaProvider


class ForeignGpuMemoryTests(unittest.TestCase):
    def test_base_provider_returns_zero_foreign_memory(self):
        """Apple Silicon and other non-NVIDIA providers have no concept of foreign GPU processes."""
        provider = BaseHardwareProvider()
        self.assertEqual(provider.read_gpu_foreign_memory_mb(own_pids={os.getpid()}), 0.0)

    def test_linux_nvidia_sums_non_own_pids_from_query_compute_apps(self):
        """`nvidia-smi --query-compute-apps=pid,used_memory` lists foreign-process GPU memory."""
        provider = LinuxNvidiaProvider()
        # CSV: pid, used_memory (MiB). Pretend PID 1111 is ours (own) and 2222/3333 are foreign.
        fake_csv = "1111, 100\n2222, 1500\n3333, 200\n"
        with patch.object(LinuxNvidiaProvider, "_run_nvidia_smi", return_value=fake_csv):
            foreign = provider.read_gpu_foreign_memory_mb(own_pids={1111})
        self.assertEqual(foreign, 1700.0, "expected only 1500+200 MiB from foreign PIDs")

    def test_linux_nvidia_returns_zero_when_no_foreign_processes(self):
        provider = LinuxNvidiaProvider()
        with patch.object(LinuxNvidiaProvider, "_run_nvidia_smi", return_value="1234, 100\n"):
            self.assertEqual(provider.read_gpu_foreign_memory_mb(own_pids={1234}), 0.0)

    def test_linux_nvidia_returns_zero_when_query_returns_empty(self):
        provider = LinuxNvidiaProvider()
        with patch.object(LinuxNvidiaProvider, "_run_nvidia_smi", return_value=""):
            self.assertEqual(provider.read_gpu_foreign_memory_mb(own_pids={os.getpid()}), 0.0)

    def test_linux_nvidia_returns_zero_on_subprocess_failure(self):
        provider = LinuxNvidiaProvider()
        with patch.object(LinuxNvidiaProvider, "_run_nvidia_smi", return_value=None):
            self.assertEqual(provider.read_gpu_foreign_memory_mb(own_pids={os.getpid()}), 0.0)


class ScorecardDirtyMBForwardingTests(unittest.TestCase):
    """`compute_model_scorecard` must carry `vram_baseline_dirty_mb` from per-record
    `hardware` to the scorecard root, not only into the per-record dict.

    Regression guard for the Sept 22 run (`results/runs/runs/benchmark_20260922_214037.json`),
    where every scorecard serialized `vram_baseline_dirty_mb: null` even though each record's
    per-record `hardware` carried the field. Spec.md Phase 2 (plan.md item 2.2) defines the
    invariant: the field must be readable from the scorecard.
    """

    @staticmethod
    def _stub_client() -> MagicMock:
        c = MagicMock()
        c.engine_name = "ONNX Runtime GenAI"
        c.display_name = "Prism"
        c.name = "prism"
        return c

    @staticmethod
    def _result(
        *,
        hardware_vram_baseline_dirty_mb: float | None = 0.0,
        hardware_vram_baseline_mb: float = 1677.0,
        hardware_vram_peak_mb: float = 8084.0,
        eval_count: int = 100,
        eval_tok_per_sec: float = 100.0,
    ) -> dict:
        hardware = {
            "vram_baseline_mb": hardware_vram_baseline_mb,
            "vram_baseline_dirty": False,
            "vram_peak_mb": hardware_vram_peak_mb,
            "vram_total_mb": 12227.0,
        }
        if hardware_vram_baseline_dirty_mb is not None:
            hardware["vram_baseline_dirty_mb"] = hardware_vram_baseline_dirty_mb
        return {
            "model": "fixture-model",
            "runtime": "prism",
            "engine": "ONNX Runtime GenAI",
            "suite": "speed",
            "test_id": "speed_short",
            "name": "Short prompt",
            "eval_tok_per_sec": eval_tok_per_sec,
            "prompt_tok_per_sec": 1000.0,
            "prefill_eff_tok_per_sec": 1000.0,
            "prefill_source": "client_ttft",
            "run": 0,
            "ttft_sec": 0.1,
            "prompt_eval_count": 100,
            "load_time_sec": 0.0,
            "finish_reason": "stop",
            "truncated": False,
            "eval_count": eval_count,
            "hardware": hardware,
        }

    def test_scorecard_carries_zero_foreign_memory_as_zero_not_null(self):
        """The Sept 22 case: every record carries `0.0`, the scorecard must read `0.0`, not `None`."""
        from benchrig.core.runner import BenchmarkRunner

        runner = BenchmarkRunner(client=self._stub_client(), config={})
        record = self._result()
        record["hardware"]["vram_baseline_dirty_mb"] = 0.0
        scorecard = runner.compute_model_scorecard("fixture-model", [record], _split_runs=False)
        self.assertIn(
            "vram_baseline_dirty_mb",
            scorecard,
            "scorecard root must carry vram_baseline_dirty_mb (per-record has it; aggregate must too)",
        )
        self.assertEqual(
            scorecard["vram_baseline_dirty_mb"],
            0.0,
            "vram_baseline_dirty_mb=0.0 on every record must serialise as 0.0, not None/null",
        )

    def test_scorecard_carries_nonzero_foreign_memory(self):
        """When at least one record has a real foreign footprint, the scorecard must report it (not drop it)."""
        from benchrig.core.runner import BenchmarkRunner

        runner = BenchmarkRunner(client=self._stub_client(), config={})
        record_with_foreign = self._result()
        record_with_foreign["hardware"]["vram_baseline_dirty_mb"] = 4800.0
        record_without_foreign = self._result()
        record_without_foreign["hardware"]["vram_baseline_dirty_mb"] = 0.0
        scorecard = runner.compute_model_scorecard(
            "fixture-model", [record_with_foreign, record_without_foreign], _split_runs=False
        )
        self.assertEqual(
            scorecard["vram_baseline_dirty_mb"],
            4800.0,
            "max of per-record vram_baseline_dirty_mb must reach the scorecard root",
        )

    def test_scorecard_returns_none_when_no_record_carries_the_field(self):
        """Defensive: a model whose records pre-date Phase 2.2 must not crash aggregation."""
        from benchrig.core.runner import BenchmarkRunner

        runner = BenchmarkRunner(client=self._stub_client(), config={})
        record = self._result()
        # No `vram_baseline_dirty_mb` key in hardware at all.
        scorecard = runner.compute_model_scorecard("fixture-model", [record], _split_runs=False)
        # Either the field is absent (older format) or it is None. The exact value is not
        # asserted — only that aggregation did not raise.
        self.assertIsInstance(scorecard, dict)


class RunnerVramBaselineDirtyTests(unittest.TestCase):
    def test_measure_vram_baseline_records_dirty_foreign_memory(self):
        """`BenchmarkRunner.measure_vram_baseline` sets `vram_baseline_dirty_mb` to the foreign portion."""
        # Import inside the test so a missing module reads as "not found" / red.
        from benchrig.core.runner import BenchmarkRunner

        mock_client = MagicMock()
        mock_client.get_running_models.return_value = []  # not dirty by the boolean path
        mock_client.engine_name = "llama.cpp"
        mock_client.display_name = "Ollama"
        mock_client.is_reachable.return_value = True

        provider = MagicMock(spec=BaseHardwareProvider)
        # Two settled readings: 5000 MiB total (own ~200 + foreign ~4800).
        provider.read_gpu.return_value = (5000.0, 12000.0, 0.0, 0.0, 0.0)
        provider.read_gpu_foreign_memory_mb.return_value = 4800.0
        provider.get_warning_threshold_mb.return_value = 10560.0

        runner = BenchmarkRunner(client=mock_client, config={})
        # Replace the provider the runner builds inside measure_vram_baseline.
        with patch("benchrig.core.runner.HardwareSampler") as mock_sampler_cls:
            mock_sampler_cls.return_value.provider = provider
            runner.measure_vram_baseline()

        self.assertEqual(runner.vram_baseline_mb, 5000.0, "total baseline = own + foreign")
        self.assertEqual(runner.vram_baseline_dirty_mb, 4800.0, "foreign baseline is the noisy portion")
        self.assertFalse(runner.vram_baseline_dirty, "boolean dirty stays False when only foreign processes use VRAM")

    def test_measure_vram_baseline_returns_zero_when_no_foreign(self):
        from benchrig.core.runner import BenchmarkRunner

        mock_client = MagicMock()
        mock_client.get_running_models.return_value = []
        mock_client.engine_name = "llama.cpp"
        mock_client.display_name = "Ollama"

        provider = MagicMock(spec=BaseHardwareProvider)
        provider.read_gpu.return_value = (200.0, 12000.0, 0.0, 0.0, 0.0)
        provider.read_gpu_foreign_memory_mb.return_value = 0.0
        provider.get_warning_threshold_mb.return_value = 10560.0

        runner = BenchmarkRunner(client=mock_client, config={})
        with patch("benchrig.core.runner.HardwareSampler") as mock_sampler_cls:
            mock_sampler_cls.return_value.provider = provider
            runner.measure_vram_baseline()

        self.assertEqual(runner.vram_baseline_mb, 200.0)
        self.assertEqual(runner.vram_baseline_dirty_mb, 0.0)


if __name__ == "__main__":
    unittest.main()
