"""Cross-platform Hardware monitoring module supporting macOS Apple Silicon (Metal/UMA) and Linux/WSL2 (NVIDIA CUDA)."""

import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


def find_nvidia_smi() -> Optional[str]:
    """Find path to nvidia-smi (including WSL standard locations)."""
    locations = [
        shutil.which("nvidia-smi"),
        "/usr/lib/wsl/lib/nvidia-smi",
        "/usr/bin/nvidia-smi",
    ]
    for loc in locations:
        if loc and os.path.exists(loc):
            return loc
    return None


class BaseHardwareProvider:
    """Base hardware interface for platform-specific specs and metrics."""

    def get_specs(self) -> Dict[str, str]:
        raise NotImplementedError

    def read_gpu(self) -> Tuple[float, float, float, float, float]:
        """Read (vram_used_mb, vram_total_mb, util_pct, temp_c, power_w)."""
        return 0.0, 0.0, 0.0, 0.0, 0.0

    def read_ram_used_gb(self) -> float:
        """Read currently used system RAM in GB."""
        return 0.0

    def get_warning_threshold_mb(self, custom_setting: Optional[Any] = None, total_mb: float = 0.0) -> float:
        """Calculate VRAM/UMA warning threshold in MB."""
        if isinstance(custom_setting, (int, float)) and custom_setting > 0:
            return float(custom_setting)
        # Default auto threshold: 88% of available memory
        if total_mb > 0:
            return total_mb * 0.88
        return 0.0


class DarwinAppleSiliconProvider(BaseHardwareProvider):
    """Hardware provider for macOS Apple Silicon (M1/M2/M3/M4) with Unified Memory & Metal."""

    def __init__(self, client: Optional[Any] = None):
        self.client = client
        self._total_mem_bytes = self._query_hw_memsize()
        self._total_mem_mb = round(self._total_mem_bytes / (1024 * 1024), 1)

    def _query_hw_memsize(self) -> int:
        """Query total physical memory in bytes via sysctl."""
        try:
            res = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and res.stdout.strip():
                return int(res.stdout.strip())
        except Exception:
            pass
        return 0

    def _query_cpu_model(self) -> str:
        """Get CPU model name via sysctl."""
        try:
            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
        # Fallback to platform processor
        proc = platform.processor()
        return proc if proc else "Apple Silicon"

    def _query_gpu_name(self, cpu_model: str) -> str:
        """Query GPU device name and core count from system_profiler or CPU name."""
        try:
            res = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
            )
            if res.returncode == 0 and res.stdout:
                # Find chipset or model
                m_chip = re.search(r"Chipset Model:\s*(.+)", res.stdout)
                m_cores = re.search(r"Total Number of Cores:\s*(\d+)", res.stdout)
                chip_name = m_chip.group(1).strip() if m_chip else cpu_model
                cores_str = f" ({m_cores.group(1)} GPU cores)" if m_cores else ""
                return f"{chip_name}{cores_str}"
        except Exception:
            pass
        return f"{cpu_model} (Metal GPU)"

    def get_specs(self) -> Dict[str, str]:
        cpu_model = self._query_cpu_model()
        gpu_name = self._query_gpu_name(cpu_model)
        mac_ver = platform.mac_ver()[0] or "macOS"
        arch = platform.machine() or "arm64"

        ram_total_gb = f"{self._total_mem_bytes / (1024**3):.1f}" if self._total_mem_bytes > 0 else "0.0"

        return {
            "platform": f"macOS {mac_ver} ({arch})",
            "platform_short": f"macOS Apple Silicon (Metal)",
            "cpu_model": cpu_model,
            "cpu_cores": str(os.cpu_count() or "Unknown"),
            "ram_total_gb": ram_total_gb,
            "gpu_name": gpu_name,
            "gpu_type": "apple_silicon",
            "gpu_vram_total_mb": str(int(self._total_mem_mb)),
            "memory_type": "Unified Memory (UMA)",
            "driver_version": f"Metal 3 (Darwin {platform.release()})",
            "cuda_version": "N/A (Metal)",
        }

    @staticmethod
    def parse_vm_stat(vm_stat_output: str, total_ram_bytes: int = 0) -> float:
        """
        Parse vm_stat output to calculate currently used RAM in GB.
        Formula: (wired + active + occupied_by_compressor) * page_size
        """
        if not vm_stat_output:
            return 0.0

        # Determine page size (typically 16384 on Apple Silicon)
        page_size = 16384
        m_page = re.search(r"page size of (\d+) bytes", vm_stat_output, re.IGNORECASE)
        if m_page:
            page_size = int(m_page.group(1))

        def get_count(pattern: str) -> int:
            m = re.search(pattern, vm_stat_output)
            return int(m.group(1)) if m else 0

        wired = get_count(r"Pages wired down:\s+(\d+)")
        active = get_count(r"Pages active:\s+(\d+)")
        compressed = get_count(r"Pages occupied by compressor:\s+(\d+)")

        used_bytes = (wired + active + compressed) * page_size
        if used_bytes > 0:
            return round(used_bytes / (1024**3), 2)

        # Fallback if specific pages not found: total - free
        free_pages = get_count(r"Pages free:\s+(\d+)")
        if total_ram_bytes > 0 and free_pages > 0:
            used = total_ram_bytes - (free_pages * page_size)
            return round(max(0.0, used / (1024**3)), 2)

        return 0.0

    def read_ram_used_gb(self) -> float:
        """Execute vm_stat and parse used RAM."""
        try:
            res = subprocess.run(
                ["vm_stat"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=1.0,
            )
            if res.returncode == 0:
                return self.parse_vm_stat(res.stdout, self._total_mem_bytes)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def parse_ioreg_gpu(ioreg_output: str) -> float:
        """Parse Device Utilization % from ioreg output."""
        if not ioreg_output:
            return 0.0
        m = re.search(r'"Device Utilization %"\s*=\s*(\d+)', ioreg_output)
        if m:
            return float(m.group(1))
        # Alternative syntax without spaces: "Device Utilization %"=12
        m_compact = re.search(r'"Device Utilization %"=(\d+)', ioreg_output)
        if m_compact:
            return float(m_compact.group(1))
        return 0.0

    def _read_gpu_utilization(self) -> float:
        """Query IOAccelerator for real-time GPU load percentage."""
        try:
            res = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=1.0,
            )
            if res.returncode == 0 and res.stdout:
                return self.parse_ioreg_gpu(res.stdout)
        except Exception:
            pass
        return 0.0

    def _read_ollama_vram_mb(self) -> float:
        """Read model VRAM size in MB allocated in Metal by Ollama via /api/ps."""
        if self.client and hasattr(self.client, "get_running_models"):
            try:
                models = self.client.get_running_models()
                vram_bytes = sum(m.get("size_vram", m.get("size", 0)) for m in models)
                if vram_bytes > 0:
                    return round(vram_bytes / (1024 * 1024), 1)
            except Exception:
                pass
        return 0.0

    def read_gpu(self) -> Tuple[float, float, float, float, float]:
        """
        Return (vram_used_mb, vram_total_mb, util_pct, temp_c, power_w).
        On Apple Silicon, VRAM is allocated from Unified Memory.
        """
        vram_used = self._read_ollama_vram_mb()
        util = self._read_gpu_utilization()
        # Power & Temp on macOS require root powermetrics, graceful 0.0 for non-root
        return vram_used, self._total_mem_mb, util, 0.0, 0.0


class LinuxNvidiaProvider(BaseHardwareProvider):
    """Hardware provider for Linux/WSL2 with NVIDIA GPUs (nvidia-smi)."""

    def __init__(self, smi_path: Optional[str] = None):
        self.smi_path = smi_path or find_nvidia_smi()

    def get_specs(self) -> Dict[str, str]:
        specs = {
            "platform": "Linux",
            "platform_short": "Linux (NVIDIA)",
            "cpu_model": "Unknown",
            "cpu_cores": str(os.cpu_count() or "Unknown"),
            "ram_total_gb": "0.0",
            "gpu_name": "None",
            "gpu_type": "nvidia",
            "gpu_vram_total_mb": "0",
            "memory_type": "Dedicated VRAM",
            "cuda_version": "Unknown",
            "driver_version": "Unknown",
        }

        # Platform check (WSL vs Native Linux)
        release = platform.uname().release
        if os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop") or "microsoft" in release.lower():
            specs["platform"] = f"WSL2 Linux ({release})"
            specs["platform_short"] = "WSL2 (NVIDIA)"
        else:
            specs["platform"] = f"Linux ({release})"

        # CPU info
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        specs["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

        # RAM info
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        specs["ram_total_gb"] = f"{kb / (1024 * 1024):.1f}"
                        break
        except Exception:
            pass

        # GPU info
        if self.smi_path:
            try:
                res = subprocess.run(
                    [
                        self.smi_path,
                        "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader,nounits",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                    timeout=3,
                )
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 3:
                    specs["gpu_name"] = parts[0]
                    specs["gpu_vram_total_mb"] = parts[1]
                    specs["driver_version"] = parts[2]
            except Exception:
                pass

        return specs

    def read_gpu(self) -> Tuple[float, float, float, float, float]:
        if not self.smi_path:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        try:
            res = subprocess.run(
                [
                    self.smi_path,
                    "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=1.0,
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                used = float(parts[0])
                total = float(parts[1])
                util = float(parts[2]) if len(parts) > 2 and parts[2] != "[N/A]" else 0.0
                temp = float(parts[3]) if len(parts) > 3 and parts[3] != "[N/A]" else 0.0
                power = float(parts[4]) if len(parts) > 4 and parts[4] != "[N/A]" else 0.0
                return used, total, util, temp, power
        except Exception:
            pass
        return 0.0, 0.0, 0.0, 0.0, 0.0

    def read_ram_used_gb(self) -> float:
        try:
            total_kb = 0
            avail_kb = 0
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
            if total_kb > 0 and avail_kb > 0:
                used_kb = total_kb - avail_kb
                return round(used_kb / (1024 * 1024), 2)
        except Exception:
            pass
        return 0.0


class GenericCPUProvider(BaseHardwareProvider):
    """Fallback provider for generic environments without dedicated GPU."""

    def get_specs(self) -> Dict[str, str]:
        return {
            "platform": f"{platform.system()} ({platform.machine()})",
            "platform_short": f"{platform.system()} (CPU)",
            "cpu_model": platform.processor() or "Generic CPU",
            "cpu_cores": str(os.cpu_count() or "Unknown"),
            "ram_total_gb": "0.0",
            "gpu_name": "None (CPU Inference)",
            "gpu_type": "cpu",
            "gpu_vram_total_mb": "0",
            "memory_type": "System RAM",
            "driver_version": "N/A",
            "cuda_version": "N/A",
        }


def get_hardware_provider(client: Optional[Any] = None) -> BaseHardwareProvider:
    """Detect platform and return appropriate hardware provider."""
    # Check macOS Apple Silicon
    if sys.platform == "darwin":
        if platform.machine() in ("arm64", "aarch64"):
            return DarwinAppleSiliconProvider(client=client)
        # Intel Mac fallback
        return DarwinAppleSiliconProvider(client=client)

    # Check Linux NVIDIA
    smi = find_nvidia_smi()
    if smi:
        return LinuxNvidiaProvider(smi_path=smi)

    return GenericCPUProvider()


def get_system_specs(client: Optional[Any] = None) -> Dict[str, str]:
    """Get system CPU, RAM, and GPU specifications based on active platform."""
    provider = get_hardware_provider(client=client)
    return provider.get_specs()


class HardwareSampler:
    """Threaded hardware metrics sampler during benchmark execution."""

    def __init__(
        self,
        interval_sec: float = 0.15,
        client: Optional[Any] = None,
        provider: Optional[BaseHardwareProvider] = None,
        warning_threshold_setting: Optional[Any] = None,
    ):
        self.interval_sec = interval_sec
        self.client = client
        self.provider = provider or get_hardware_provider(client=client)
        self.warning_threshold_setting = warning_threshold_setting

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.vram_samples: List[float] = []
        self.gpu_util_samples: List[float] = []
        self.power_samples: List[float] = []
        self.temp_samples: List[float] = []
        self.ram_used_samples: List[float] = []

        self.start_vram_mb: float = 0.0
        self.vram_total_mb: float = 0.0

    def _worker(self):
        while not self._stop_event.is_set():
            v_used, v_total, util, temp, power = self.provider.read_gpu()
            r_used = self.provider.read_ram_used_gb()

            if v_total > 0:
                self.vram_total_mb = v_total
            if v_used > 0:
                self.vram_samples.append(v_used)
            if util > 0:
                self.gpu_util_samples.append(util)
            if temp > 0:
                self.temp_samples.append(temp)
            if power > 0:
                self.power_samples.append(power)
            if r_used > 0:
                self.ram_used_samples.append(r_used)

            time.sleep(self.interval_sec)

    def start(self):
        """Start sampling in background."""
        self.vram_samples.clear()
        self.gpu_util_samples.clear()
        self.power_samples.clear()
        self.temp_samples.clear()
        self.ram_used_samples.clear()
        self._stop_event.clear()

        # baseline
        v_used, v_total, _, _, _ = self.provider.read_gpu()
        self.start_vram_mb = v_used
        self.vram_total_mb = v_total

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop sampling and return aggregated summary metrics."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

        end_vram, v_total_final, _, _, _ = self.provider.read_gpu()
        if v_total_final > 0:
            self.vram_total_mb = v_total_final

        peak_vram = max(self.vram_samples) if self.vram_samples else end_vram
        min_vram = min(self.vram_samples) if self.vram_samples else self.start_vram_mb
        avg_util = (
            sum(self.gpu_util_samples) / len(self.gpu_util_samples)
            if self.gpu_util_samples
            else 0.0
        )
        peak_util = max(self.gpu_util_samples) if self.gpu_util_samples else 0.0
        avg_power = (
            sum(self.power_samples) / len(self.power_samples)
            if self.power_samples
            else 0.0
        )
        peak_temp = max(self.temp_samples) if self.temp_samples else 0.0
        peak_ram = max(self.ram_used_samples) if self.ram_used_samples else 0.0

        vram_pct_used = (
            (peak_vram / self.vram_total_mb * 100.0)
            if self.vram_total_mb > 0
            else 0.0
        )

        warning_threshold = self.provider.get_warning_threshold_mb(
            self.warning_threshold_setting,
            total_mb=self.vram_total_mb,
        )
        is_warning = (peak_vram >= warning_threshold) if warning_threshold > 0 else False

        return {
            "vram_start_mb": round(self.start_vram_mb, 1),
            "vram_peak_mb": round(peak_vram, 1),
            "vram_end_mb": round(end_vram, 1),
            "vram_delta_mb": round(max(0.0, peak_vram - self.start_vram_mb), 1),
            "vram_total_mb": round(self.vram_total_mb, 1),
            "vram_peak_pct": round(vram_pct_used, 1),
            "gpu_util_avg_pct": round(avg_util, 1),
            "gpu_util_peak_pct": round(peak_util, 1),
            "power_avg_w": round(avg_power, 1),
            "temp_peak_c": round(peak_temp, 1),
            "ram_peak_gb": round(peak_ram, 1),
            "vram_warning": is_warning,
            "warning_threshold_mb": round(warning_threshold, 1),
        }
