"""Device selection and GPU memory management.

Responsibilities
----------------
* decide whether inference runs on CUDA or CPU (config override honoured)
* pick a sensible dtype (fp16 on CUDA, fp32 on CPU)
* expose VRAM numbers to the processing pipeline (tile sizes, batch counts)
* provide :meth:`DeviceManager.inference_guard`, a context manager that
  catches CUDA OOM, clears the cache and lets callers retry with smaller
  tiles instead of crashing the whole job
* keep exactly one heavyweight model resident at a time (see
  ``ai.model_manager``); :meth:`release` frees VRAM between stages

The module never imports torch at import time so that the UI can start on a
machine without PyTorch installed and show a helpful message instead.
"""
from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from utils.logger import get_logger

log = get_logger("device")


class GPUMemoryError(RuntimeError):
    """Raised when a tile/batch cannot fit even after retries."""


@dataclass
class DeviceInfo:
    device: str = "cpu"            # "cuda" or "cpu"
    torch_dtype: str = "float32"   # "float16" | "float32" | "bfloat16"
    name: str = "CPU"
    total_vram_mb: int = 0
    cuda_version: str = ""
    torch_version: str = ""
    cudnn_version: str = ""

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def label(self) -> str:
        if self.is_cuda:
            return f"{self.name} (CUDA {self.cuda_version}, {self.total_vram_mb} MB)"
        return f"CPU ({self.name})"


class DeviceManager:
    """Process-wide inference device policy."""

    _instance: Optional["DeviceManager"] = None

    def __init__(self, preferred: str = "auto", precision: str = "fp16") -> None:
        self.preferred = preferred or "auto"
        self.precision = precision or "fp16"
        self._torch = None
        self._info: Optional[DeviceInfo] = None
        self._lock = threading.RLock()
        self._resident: dict = {}          # slot -> (owner_id, payload)
        self._oom_count = 0

    # -- singleton -----------------------------------------------------
    @classmethod
    def instance(cls, preferred: str = "auto", precision: str = "fp16"
                 ) -> "DeviceManager":
        if cls._instance is None:
            cls._instance = cls(preferred, precision)
        else:
            cls._instance.preferred = preferred or cls._instance.preferred
            cls._instance.precision = precision or cls._instance.precision
        return cls._instance

    # -- torch access --------------------------------------------------
    def torch(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch  # noqa: F401
            self._torch = torch
        except Exception as exc:  # pragma: no cover - torch is a hard dep
            raise RuntimeError(
                "PyTorch is not installed. Install the CUDA build first:\n"
                "  pip install torch==2.4.1 torchvision==0.19.1 "
                "--index-url https://download.pytorch.org/whl/cu121"
            ) from exc
        return self._torch

    @property
    def torch_available(self) -> bool:
        try:
            self.torch()
            return True
        except RuntimeError:
            return False

    # -- discovery -----------------------------------------------------
    def info(self, refresh: bool = False) -> DeviceInfo:
        if self._info is not None and not refresh:
            return self._info
        torch = self.torch()
        want_cuda = self.preferred in ("auto", "cuda", "gpu")
        cuda_ok = bool(torch.cuda.is_available()) if want_cuda else False
        forced_cpu = self.preferred == "cpu"

        if forced_cpu or not cuda_ok:
            info = DeviceInfo(device="cpu", torch_dtype="float32", name=_cpu_name(),
                              torch_version=torch.__version__)
            if want_cuda and not cuda_ok and not forced_cpu:
                log.warning("CUDA requested but unavailable - falling back to CPU")
        else:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            dtype = "bfloat16" if (self.precision == "bf16" and
                                   props.major >= 8) else (
                "float16" if self.precision in ("fp16", "float16") else "float32")
            info = DeviceInfo(
                device="cuda", torch_dtype=dtype, name=props.name,
                total_vram_mb=int(props.total_memory / 1048576),
                cuda_version=str(torch.version.cuda or ""),
                torch_version=torch.__version__,
                cudnn_version=str(torch.backends.cudnn.version() or ""),
            )
            log.info("Using CUDA device: %s (%d MB, %s)",
                     info.name, info.total_vram_mb, info.torch_dtype)
        self._info = info
        return info

    # -- tensors -------------------------------------------------------
    @property
    def device(self):
        return self.torch().device(self.info().device)

    @property
    def dtype(self):
        torch = self.torch()
        return {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}[self.info().torch_dtype]

    def to(self, obj: Any) -> Any:
        """Move module/tensor to the configured device & dtype."""
        if hasattr(obj, "to"):
            if self.info().is_cuda:
                return obj.to(self.device, dtype=self.dtype)
            return obj.to(self.device)
        return obj

    # -- VRAM ----------------------------------------------------------
    def vram_total_mb(self) -> int:
        return self.info().total_vram_mb

    def vram_free_mb(self) -> int:
        if not self.info().is_cuda:
            return 0
        torch = self.torch()
        free, total = torch.cuda.mem_get_info()
        return int(free / 1048576)

    def vram_used_mb(self) -> int:
        return self.vram_total_mb() - self.vram_free_mb()

    def tile_size(self, default: int = 0) -> int:
        """VRAM-aware tile size; 0 means 'full frame' inference."""
        from utils.gpu_monitor import recommend_tile_size
        if self.info().is_cuda:
            return recommend_tile_size(self.vram_total_mb(), default or 256)
        return 300 if default == 0 else default  # CPU: moderate tiles

    def batch_frames(self, default: int = 4) -> int:
        from utils.gpu_monitor import recommend_batch_frames
        if self.info().is_cuda:
            return recommend_batch_frames(self.vram_total_mb())
        return max(1, min(default, 2))

    # -- model residency ----------------------------------------------
    def can_load(self, key: str, required_mb: int = 0) -> bool:
        """Heuristic check that loading ``key`` will not immediately OOM."""
        if not self.info().is_cuda:
            return True
        free = self.vram_free_mb()
        return free > required_mb * 1.05

    def resident_id(self, slot: str) -> Optional[str]:
        with self._lock:
            entry = self._resident.get(slot)
            return entry[0] if entry else None

    def mark_resident(self, slot: str, owner_id: str, payload: Any = None) -> None:
        with self._lock:
            self._resident[slot] = (owner_id, payload)

    def release(self, slot: Optional[str] = None) -> None:
        """Drop cached references and empty the CUDA cache."""
        with self._lock:
            if slot is None:
                self._resident.clear()
            else:
                self._resident.pop(slot, None)
        gc.collect()
        if self._info is not None and self._info.is_cuda:
            try:
                self.torch().cuda.empty_cache()
                self.torch().cuda.ipc_collect()
            except Exception:
                pass

    def empty_cache(self) -> None:
        if self._info is not None and self._info.is_cuda:
            try:
                self.torch().cuda.empty_cache()
            except Exception:
                pass

    # -- OOM guard -----------------------------------------------------
    def inference_guard(self) -> "OOMGuard":
        return OOMGuard(self)


class OOMGuard:
    """Context manager turning CUDA OOM into a recoverable condition.

    Usage::

        with dev.inference_guard() as guard:
            out = model(x)
        if guard.oom:
            ... retry with a smaller tile ...
    """

    def __init__(self, dev: DeviceManager) -> None:
        self.dev = dev
        self.oom = False
        self.error: Optional[BaseException] = None

    def __enter__(self) -> "OOMGuard":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            return False
        oom_types = (RuntimeError, MemoryError)
        if isinstance(exc, oom_types):
            msg = str(exc).lower()
            if isinstance(exc, MemoryError) or "out of memory" in msg or \
                    "cuda error" in msg or "not enough memory" in msg:
                self.oom = True
                self.error = exc
                self.dev._oom_count += 1
                log.warning("GPU memory error (#%d): %s", self.dev._oom_count, exc)
                self.dev.release()
                return True          # swallowed - caller decides to retry
        return False


class TiledInferenceMixin:
    """Tile-splitting inference loop shared by the SR engines.

    ``runner(tensor)`` receives a uint8/float tensor tile in [0,1] of shape
    (1,3,h,w) and must return an upscaled float tensor (1,3,H,W) in [0,1].
    Tile borders are blended by cropping ``pad`` pixels, so seams are invisible
    and VRAM use stays bounded.
    """

    def tiled_infer(self, img, runner: Callable, scale: int, tile: int,
                    tile_pad: int = 16, progress: Optional[Callable[[float], None]] = None,
                    oom_retry: bool = True):
        _, _c, h, w = img.shape
        if tile <= 0 or (tile >= h and tile >= w):
            if progress:
                progress(1.0)
            return runner(img)

        out = img.new_zeros(1, 3, h * scale, w * scale)
        weight = img.new_zeros(1, 1, h * scale, w * scale)
        tiles_y = (h + tile - 1) // tile
        tiles_x = (w + tile - 1) // tile
        total = tiles_y * tiles_x
        idx = 0
        current = tile

        def warn(msg: str) -> None:
            # ``self.log`` is optional so the mixin also works standalone
            logger = getattr(self, "log", None) or log
            logger.warning(msg)

        y = 0
        while y < h:
            x = 0
            while x < w:
                # guard: remaining strip smaller than tile
                th = min(current, h - y)
                tw = min(current, w - x)
                py0 = max(0, y - tile_pad)
                px0 = max(0, x - tile_pad)
                py1 = min(h, y + th + tile_pad)
                px1 = min(w, x + tw + tile_pad)
                patch = img[:, :, py0:py1, px0:px1]

                try:
                    if hasattr(self, "dev") and self.dev.info().is_cuda:
                        with self.dev.inference_guard() as guard:
                            sr = runner(patch)
                        if guard.oom:
                            raise GPUMemoryError(str(guard.error))
                    else:
                        sr = runner(patch)
                except GPUMemoryError:
                    if not oom_retry or current <= 64:
                        raise
                    current = max(64, current // 2)
                    warn(f"GPU memory exhausted - reducing tile size to {current}px "
                         "and retrying")
                    return self.tiled_infer(img, runner, scale, current,
                                            tile_pad, progress, oom_retry=False)

                # crop padding back off
                cw0 = (x - px0) * scale
                ch0 = (y - py0) * scale
                cw = tw * scale
                ch = th * scale
                out[:, :, y * scale:(y + th) * scale, x * scale:(x + tw) * scale] += \
                    sr[:, :, ch0:ch0 + ch, cw0:cw0 + cw]
                weight[:, :, y * scale:(y + th) * scale, x * scale:(x + tw) * scale] += 1.0
                del sr, patch
                idx += 1
                if progress:
                    progress(idx / total)
                x += tw
            y += th
        weight = weight.clamp(min=1e-6)
        return out / weight


def _cpu_name() -> str:
    try:
        from utils.gpu_monitor import system_stats
        return system_stats().cpu_name or "CPU"
    except Exception:
        return "CPU"
