"""GPU / system telemetry without heavy dependencies.

Strategy (first that works wins):

1. **NVML via pynvml** (if installed - optional accelerant)
2. **nvidia-smi --query-gpu** subprocess (ships with every NVIDIA driver)
3. **PyTorch CUDA** for device name / VRAM totals
4. *No GPU* - CPU-only fallback keeps the UI functional.

The module is Qt-free: the UI polls :func:`query_gpu` from a QTimer, and the
processing worker logs samples.  Nothing here ever raises to the caller.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from typing import List, Optional

from .logger import get_logger

log = get_logger("gpu_monitor")

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_NVIDIA_SMI = shutil.which("nvidia-smi")


@dataclass
class GPUStats:
    index: int = 0
    name: str = "No NVIDIA GPU detected"
    total_vram_mb: int = 0
    used_vram_mb: int = 0
    free_vram_mb: int = 0
    utilization: int = 0          # percent
    temperature_c: int = 0
    power_w: float = 0.0
    driver: str = ""
    cuda_version: str = ""
    available: bool = False
    source: str = "none"          # nvml | smi | torch | none

    @property
    def vram_percent(self) -> float:
        if not self.total_vram_mb:
            return 0.0
        return min(100.0, 100.0 * self.used_vram_mb / self.total_vram_mb)

    @property
    def label(self) -> str:
        if not self.available:
            return "CPU mode"
        return f"{self.name} - {self.used_vram_mb}/{self.total_vram_mb} MB"


@dataclass
class SystemStats:
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    cpu_name: str = ""
    cpu_count: int = 0
    os_name: str = ""


_CACHE_LOCK = threading.Lock()
_STATIC_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def nvidia_smi_available() -> bool:
    return bool(_NVIDIA_SMI)


def torch_cuda_available() -> bool:
    try:
        import torch  # local import - torch is optional for UI-only tests
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def torch_device_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return ""


def detect_gpus() -> List[GPUStats]:
    """Enumerate NVIDIA GPUs (cached for process lifetime)."""
    with _CACHE_LOCK:
        cached = _STATIC_CACHE.get("gpus")
        if cached is not None:
            return [GPUStats(**g) for g in cached]

    gpus: List[GPUStats] = []

    # 1) pynvml (fastest, richest)
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode(errors="ignore")
        try:
            cuda_v = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda_str = f"{cuda_v // 1000}.{(cuda_v % 1000) // 10}"
        except Exception:
            cuda_str = ""
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode(errors="ignore")
            gpus.append(GPUStats(
                index=i, name=name,
                total_vram_mb=int(mem.total / 1048576),
                used_vram_mb=int(mem.used / 1048576),
                free_vram_mb=int(mem.free / 1048576),
                driver=str(driver), cuda_version=cuda_str,
                available=True, source="nvml"))
        pynvml.nvmlShutdown()
    except Exception:
        gpus = []

    # 2) nvidia-smi
    if not gpus and _NVIDIA_SMI:
        try:
            cmd = [_NVIDIA_SMI,
                   "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                   "utilization.gpu,temperature.gpu,power.draw,driver_version",
                   "--format=csv,noheader,nounits"]
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 creationflags=CREATE_NO_WINDOW,
                                 timeout=10).stdout.strip()
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 9:
                    continue
                try:
                    total = int(float(parts[2]))
                    used = int(float(parts[3]))
                    free = int(float(parts[4]))
                except ValueError:
                    continue
                gpus.append(GPUStats(
                    index=int(parts[0]), name=parts[1],
                    total_vram_mb=total, used_vram_mb=used, free_vram_mb=free,
                    utilization=_safe_int(parts[5]),
                    temperature_c=_safe_int(parts[6]),
                    power_w=_safe_float(parts[7]),
                    driver=parts[8], available=True, source="smi"))
        except (OSError, subprocess.SubprocessError, ValueError):
            gpus = []

    # 3) torch
    if not gpus and torch_cuda_available():
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            total = int(props.total_memory / 1048576)
            gpus.append(GPUStats(
                index=0, name=torch.cuda.get_device_name(0),
                total_vram_mb=total, used_vram_mb=0, free_vram_mb=total,
                available=True, source="torch"))
        except Exception:
            gpus = []

    if not gpus:
        gpus = [GPUStats(available=False, source="none")]

    with _CACHE_LOCK:
        _STATIC_CACHE["gpus"] = [g.__dict__.copy() for g in gpus]
    return gpus


def query_gpu(index: int = 0, live: bool = True) -> GPUStats:
    """Return current GPU stats.  ``live=False`` returns static info only."""
    base = detect_gpus()[0] if index == 0 else (
        detect_gpus()[index] if index < len(detect_gpus()) else detect_gpus()[0])
    if not live or not base.available:
        return base

    # nvidia-smi gives utilisation/temperature live values cheaply
    if _NVIDIA_SMI:
        try:
            cmd = [_NVIDIA_SMI, f"--id={base.index}",
                   "--query-gpu=memory.used,memory.free,utilization.gpu,"
                   "temperature.gpu,power.draw",
                   "--format=csv,noheader,nounits"]
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 creationflags=CREATE_NO_WINDOW,
                                 timeout=5).stdout.strip()
            parts = [p.strip() for p in out.splitlines()[0].split(",")]
            if len(parts) >= 5:
                base.used_vram_mb = _safe_int(parts[0])
                base.free_vram_mb = _safe_int(parts[1])
                base.utilization = _safe_int(parts[2])
                base.temperature_c = _safe_int(parts[3])
                base.power_w = _safe_float(parts[4])
        except (OSError, subprocess.SubprocessError, IndexError, ValueError):
            pass

    if base.used_vram_mb == 0 and base.available:
        try:
            import torch
            if torch.cuda.is_available():
                base.used_vram_mb = int(torch.cuda.memory_allocated(0) / 1048576)
                base.free_vram_mb = base.total_vram_mb - base.used_vram_mb
        except Exception:
            pass
    return base


def system_stats() -> SystemStats:
    stats = SystemStats()
    try:
        import psutil
        stats.cpu_percent = psutil.cpu_percent(interval=0.15)
        vm = psutil.virtual_memory()
        stats.ram_total_gb = vm.total / 1073741824
        stats.ram_used_gb = vm.used / 1073741824
        stats.ram_percent = vm.percent
        stats.cpu_count = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    except Exception:
        try:
            stats.cpu_count = os.cpu_count() or 1
        except Exception:
            stats.cpu_count = 1
    if "cpu_name" not in _STATIC_CACHE:
        _STATIC_CACHE["cpu_name"] = _query_cpu_name()
        _STATIC_CACHE["os_name"] = f"{os.name} {os.version if hasattr(os, 'version') else ''}".strip()
    stats.cpu_name = _STATIC_CACHE.get("cpu_name", "")
    stats.os_name = _STATIC_CACHE.get("os_name", os.name)
    return stats


def recommend_tile_size(total_vram_mb: int, fallback: int = 256) -> int:
    """VRAM-aware Real-ESRGAN tile size (0 when the GPU handles full frames).

    These thresholds are conservative and leave headroom for the OOM retry
    ladder implemented in the engines.
    """
    if total_vram_mb <= 0:
        return 128
    if total_vram_mb >= 16000:
        return 0          # full-frame inference (with auto OOM fallback)
    if total_vram_mb >= 10000:
        return 512
    if total_vram_mb >= 6000:
        return 384
    if total_vram_mb >= 4000:
        return 256
    if total_vram_mb >= 2500:
        return 192
    return fallback if fallback < 192 else 128


def recommend_batch_frames(total_vram_mb: int) -> int:
    if total_vram_mb <= 0:
        return 1
    if total_vram_mb >= 16000:
        return 8
    if total_vram_mb >= 10000:
        return 6
    if total_vram_mb >= 6000:
        return 4
    if total_vram_mb >= 4000:
        return 2
    return 1


def nvidia_smi_json() -> Optional[dict]:
    """Raw nvidia-smi JSON (diagnostics page / bug reports)."""
    if not _NVIDIA_SMI:
        return None
    try:
        out = subprocess.run([_NVIDIA_SMI, "--query-gpu=name,memory.total",
                              "--format=json"],
                             capture_output=True, text=True,
                             creationflags=CREATE_NO_WINDOW, timeout=10).stdout
        return json.loads(out)
    except Exception:
        return None


# ---------------------------------------------------------------------------
def _safe_int(value: str) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: str) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _query_cpu_name() -> str:
    if os.name == "nt":
        try:
            return (os.environ.get("PROCESSOR_IDENTIFIER") or "CPU").strip()
        except Exception:
            return "CPU"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "CPU"
