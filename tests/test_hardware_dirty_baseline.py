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
