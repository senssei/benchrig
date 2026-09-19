"""Comprehensive unit tests covering hardware providers, telemetry parsers, reporting, and runner scorecard computation."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from benchrig.core.hardware import (
    BaseHardwareProvider,
    DarwinAppleSiliconProvider,
    GenericCPUProvider,
    HardwareSampler,
    get_system_specs,
)
from benchrig.reporting.display import display_leaderboard, display_system_banner
from benchrig.reporting.markdown import generate_markdown_report


class TestHardwareProviders(unittest.TestCase):
    """Test suite for hardware providers and telemetry parsers."""

    def test_parse_vm_stat_apple_silicon(self):
        """Verify vm_stat output is correctly parsed into used RAM in GB."""
        sample_vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                27244.
Pages active:                            1376882.
Pages inactive:                           982310.
Pages speculative:                         28123.
Pages throttled:                               0.
Pages wired down:                         382109.
Pages purgeable:                           59102.
"Translation faults":                 123456789.
Pages copy-on-write:                    9876543.
Pages zero filled:                     45678901.
Pages reactivated:                         1234.
Pages purged:                               567.
File-backed pages:                        12345.
Anonymous pages:                          56789.
Pages stored in compressor:               120450.
Pages occupied by compressor:              60225.
Decompressions:                           23456.
Compressions:                             78901.
"""
        used_gb = DarwinAppleSiliconProvider.parse_vm_stat(sample_vm_stat)
        self.assertAlmostEqual(used_gb, 27.76, delta=0.05)

    def test_parse_ioreg_gpu_utilization(self):
        """Verify IOAccelerator utilization parsing from ioreg."""
        sample_ioreg = """
+-o AppleM3Max  <class AGXAccelerator, id 0x100000456, registered, matched, active, busy 0 (2 ms), retain 107>
  {
    "PerformanceStatistics" = {"Alloc system memory"=536870912,"Device Utilization %"=42,"In use system memory"=536870912}
  }
"""
        util = DarwinAppleSiliconProvider.parse_ioreg_gpu(sample_ioreg)
        self.assertEqual(util, 42.0)

        # Compact syntax check
        sample_compact = '"Device Utilization %"=88'
        self.assertEqual(DarwinAppleSiliconProvider.parse_ioreg_gpu(sample_compact), 88.0)

        # Empty check
        self.assertEqual(DarwinAppleSiliconProvider.parse_ioreg_gpu(""), 0.0)

    @patch.object(DarwinAppleSiliconProvider, "_query_hw_memsize", return_value=38654705664)  # 36 GB
    @patch.object(DarwinAppleSiliconProvider, "_query_cpu_model", return_value="Apple M3 Max")
    @patch.object(DarwinAppleSiliconProvider, "_query_gpu_name", return_value="Apple M3 Max (30 GPU cores)")
    def test_darwin_apple_silicon_specs(self, mock_gpu, mock_cpu, mock_mem):
        """Verify get_specs format on Apple Silicon."""
        provider = DarwinAppleSiliconProvider()
        specs = provider.get_specs()

        self.assertEqual(specs["gpu_type"], "apple_silicon")
        self.assertEqual(specs["cpu_model"], "Apple M3 Max")
        self.assertEqual(specs["gpu_name"], "Apple M3 Max (30 GPU cores)")
        self.assertEqual(specs["ram_total_gb"], "36.0")
        self.assertEqual(specs["gpu_vram_total_mb"], "36864")
        self.assertEqual(specs["memory_type"], "Unified Memory (UMA)")
        self.assertIn("Apple Silicon", specs["platform_short"])

    def test_warning_threshold_calculation(self):
        """Verify dynamic auto threshold and custom threshold calculation."""
        provider = BaseHardwareProvider()

        # Explicit custom setting
        self.assertEqual(provider.get_warning_threshold_mb(11000, total_mb=12227), 11000.0)

        # Auto setting: 88% of 12227 MB (NVIDIA)
        auto_nvidia = provider.get_warning_threshold_mb("auto", total_mb=12227)
        self.assertAlmostEqual(auto_nvidia, 12227 * 0.88, delta=1.0)

        # Auto setting on Apple Silicon 36GB (36864 MB)
        auto_mac = provider.get_warning_threshold_mb(None, total_mb=36864)
        self.assertAlmostEqual(auto_mac, 36864 * 0.88, delta=1.0)

    def test_generic_cpu_provider(self):
        """Verify fallback generic CPU provider."""
        provider = GenericCPUProvider()
        specs = provider.get_specs()
        self.assertEqual(specs["gpu_type"], "cpu")
        self.assertEqual(specs["gpu_name"], "None (CPU Inference)")
        self.assertEqual(provider.read_gpu(), (0.0, 0.0, 0.0, 0.0, 0.0))

    def test_current_platform_specs(self):
        """Verify get_system_specs executes successfully on host machine."""
        specs = get_system_specs()
        self.assertIsInstance(specs, dict)
        self.assertIn("platform", specs)
        self.assertIn("cpu_model", specs)
        self.assertIn("gpu_name", specs)
        self.assertIn("ram_total_gb", specs)
        self.assertIn("gpu_vram_total_mb", specs)

    def test_hardware_sampler_mock(self):
        """Verify HardwareSampler aggregation with mock provider."""
        mock_provider = MagicMock(spec=BaseHardwareProvider)
        mock_provider.read_gpu.return_value = (4500.0, 16000.0, 75.0, 50.0, 35.0)
        mock_provider.read_ram_used_gb.return_value = 8.5
        mock_provider.get_warning_threshold_mb.return_value = 14000.0

        sampler = HardwareSampler(interval_sec=0.01, provider=mock_provider)
        sampler.start()
        import time

        time.sleep(0.05)
        metrics = sampler.stop()

        self.assertGreater(metrics["vram_peak_mb"], 0)
        self.assertEqual(metrics["vram_total_mb"], 16000.0)
        self.assertFalse(metrics["vram_warning"])
        self.assertEqual(metrics["warning_threshold_mb"], 14000.0)


class TestReportingCrossPlatform(unittest.TestCase):
    """Verify display and markdown reports render correctly for both Mac and Linux."""

    def test_display_system_banner_apple_silicon(self):
        """Verify banner displays without error with Apple Silicon specs."""
        mac_specs = {
            "platform": "macOS 14.5 (arm64)",
            "platform_short": "macOS Apple Silicon (Metal)",
            "cpu_model": "Apple M3 Max",
            "cpu_cores": "14",
            "ram_total_gb": "36.0",
            "gpu_name": "Apple M3 Max (30 GPU cores)",
            "gpu_type": "apple_silicon",
            "gpu_vram_total_mb": "36864",
            "memory_type": "Unified Memory (UMA)",
            "driver_version": "Metal 3 (Darwin 23.5.0)",
        }
        # Should execute cleanly without exceptions
        display_system_banner(mac_specs)

    def test_display_leaderboard_apple_silicon(self):
        """Verify leaderboard renders with UMA headers."""
        scorecards = [
            {
                "model": "qwen2.5-coder:7b",
                "composite_score": 92.5,
                "coding_pass_rate": 100.0,
                "reasoning_accuracy": 90.0,
                "avg_eval_tok_sec": 95.0,
                "avg_prompt_tok_sec": 300.0,
                "avg_ttft_sec": 0.05,
                "peak_vram_mb": 4500.0,
                "total_vram_mb": 36864.0,
                "memory_type": "Unified Memory (UMA)",
                "gpu_type": "apple_silicon",
                "vram_warning": False,
            }
        ]
        mac_specs = {
            "platform_short": "macOS Apple Silicon (Metal)",
            "gpu_type": "apple_silicon",
            "gpu_vram_total_mb": "36864",
        }
        display_leaderboard(scorecards, specs=mac_specs)

    def test_markdown_generation_apple_silicon(self):
        """Verify Markdown generator includes Apple Silicon UMA recommendations."""
        scorecards = [
            {
                "model": "qwen2.5-coder:7b",
                "composite_score": 92.5,
                "coding_pass_rate": 100.0,
                "reasoning_accuracy": 90.0,
                "avg_eval_tok_sec": 95.0,
                "avg_prompt_tok_sec": 300.0,
                "avg_ttft_sec": 0.05,
                "peak_vram_mb": 4500.0,
                "total_vram_mb": 36864.0,
                "vram_warning": False,
            }
        ]
        mac_specs = {
            "platform": "macOS 14.5 (arm64)",
            "platform_short": "macOS Apple Silicon (Metal)",
            "cpu_model": "Apple M3 Max",
            "cpu_cores": "14",
            "ram_total_gb": "36.0",
            "gpu_name": "Apple M3 Max (30 GPU cores)",
            "gpu_type": "apple_silicon",
            "gpu_vram_total_mb": "36864",
            "memory_type": "Unified Memory (UMA)",
            "driver_version": "Metal 3",
        }

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            out_path = f.name

        try:
            report = generate_markdown_report(scorecards, [], mac_specs, output_path=out_path)
            self.assertIn("Apple Silicon", report)
            self.assertIn("Unified Memory", report)
            self.assertIn("Metal", report)
            self.assertIn("UMA", report)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)


if __name__ == "__main__":
    unittest.main()
