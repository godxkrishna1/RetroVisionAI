"""RIFE frame-interpolation engine.

``interpolate_pair(img0, img1, t)`` returns one synthetic frame at time ``t``
(0..1) using real IFNet inference.  The video pipeline calls it for
midpoints (2x) or thirds (3x) between consecutive source frames.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from utils.logger import get_logger

from .archs.rife import IFNet
from .device_manager import DeviceManager, GPUMemoryError

log = get_logger("rife")


class RIFEEngine:
    def __init__(self, device: DeviceManager, model_manager) -> None:
        self.dev = device
        self.models = model_manager
        self._lock = threading.RLock()
        self._net = None
        self._key: Optional[str] = None
        self.ensemble = True
        self.frames_generated = 0

    @property
    def loaded(self) -> bool:
        return self._net is not None

    def load(self, model_key: str = "rife47") -> bool:
        with self._lock:
            if self._net is not None and self._key == model_key:
                return True
            path = self.models.path_for(model_key)
            if not path.exists():
                raise FileNotFoundError(
                    f"RIFE weights missing ({path.name}). Download them in "
                    "AI Model Manager before enabling motion interpolation.")
            self.unload()
            torch = self.dev.torch()
            net = IFNet(arch_ver="4.7")
            state = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = {k: v for k, v in state.items()
                     if not k.startswith("step_counter")}
            net.load_state_dict(state, strict=True)
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            if self.dev.info().is_cuda:
                self.dev.to(net)
            self._net = net
            self._key = model_key
            self.dev.mark_resident("rife", model_key, net)
            log.info("RIFE %s loaded (%s)", model_key, self.dev.info().label)
            return True

    def unload(self) -> None:
        with self._lock:
            if self._net is not None:
                del self._net
                self._net = None
                self._key = None
                self.dev.release("rife")

    # ------------------------------------------------------------------
    def interpolate_pair(self, img0_rgb: np.ndarray, img1_rgb: np.ndarray,
                         timestep: float = 0.5) -> np.ndarray:
        """Generate the frame at ``timestep`` between two RGB uint8 frames.

        Frames may differ in size (e.g. after an SR pass changed dimensions);
        ``img1`` is resized to ``img0`` which is correct because they belong to
        the same sequence.
        """
        if self._net is None:
            raise RuntimeError("RIFE is not loaded")
        if img0_rgb.shape != img1_rgb.shape:
            img1_rgb = _resize_like(img1_rgb, img0_rgb)
        torch = self.dev.torch()

        a = torch.from_numpy(img0_rgb.astype(np.float32) / 255.0)
        b = torch.from_numpy(img1_rgb.astype(np.float32) / 255.0)
        a = a.permute(2, 0, 1).unsqueeze(0).contiguous()
        b = b.permute(2, 0, 1).unsqueeze(0).contiguous()
        if self.dev.info().is_cuda:
            a = a.to(self.dev.device, dtype=self.dev.dtype)
            b = b.to(self.dev.device, dtype=self.dev.dtype)

        with self.dev.inference_guard() as guard:
            with torch.no_grad():
                mid = self._net(a, b, timestep, fastmode=True,
                                ensemble=self.ensemble)
        if guard.oom:
            raise GPUMemoryError(str(guard.error))

        mid = mid.float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        self.frames_generated += 1
        return (mid * 255.0).round().astype(np.uint8)

    def warmup(self) -> bool:
        if self._net is None:
            return False
        try:
            z = np.zeros((64, 64, 3), np.uint8)
            self.interpolate_pair(z, z, 0.5)
            return True
        except Exception as exc:  # pragma: no cover
            log.warning("RIFE warmup failed: %s", exc)
            return False


def _resize_like(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.resize(img, (ref.shape[1], ref.shape[0]),
                      interpolation=cv2.INTER_LINEAR)
