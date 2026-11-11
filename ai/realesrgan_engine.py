"""Real-ESRGAN super-resolution engine (real model inference).

Public API
----------
``engine = RealESRGANEngine(device_manager, model_manager)``
``engine.load(model_key)``                   # single resident engine
``engine.upscale(np_rgb_uint8) -> np_rgb_uint8``   # full-frame inference

Internals
---------
* Frames are fed as float32 RGB tensors in [0, 1].
* Tile size is VRAM-aware; on CUDA OOM the tile size is halved and the frame
  is retried automatically (down to 64 px) - no crash, no fake output.
* Only one SR model is resident at a time: loading another key frees the old
  weights first (explicit VRAM policy from the spec).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from utils.logger import get_logger

from .archs.rrdbnet import RRDBNet, SRVGGNetCompact
from .device_manager import DeviceManager, GPUMemoryError

log = get_logger("realesrgan")

_ARCH_BY_KEY = {
    "realesrgan_x4plus": dict(kind="rrdb", scale=4, blocks=23),
    "realesrgan_x2plus": dict(kind="rrdb", scale=2, blocks=23),
    "realesrgan_anime6b": dict(kind="rrdb", scale=4, blocks=6),
    "realesr_animevideo": dict(kind="srvgg", scale=4, conv=16),
    "realesr_general_x4v3": dict(kind="srvgg", scale=4, conv=32),
}


class RealESRGANEngine:
    def __init__(self, device: DeviceManager, model_manager) -> None:
        self.dev = device
        self.models = model_manager
        self._lock = threading.RLock()
        self._net = None
        self._key: Optional[str] = None
        self._scale = 4
        self._tile = 0
        self._tile_pad = 16
        self.tiles_processed = 0
        self.frames_done = 0

    # ------------------------------------------------------------------
    @property
    def loaded_key(self) -> Optional[str]:
        return self._key

    @property
    def scale(self) -> int:
        return self._scale

    def is_loaded(self, key: str) -> bool:
        return self._net is not None and self._key == key

    def load(self, model_key: str, tile: int = -1) -> int:
        """Load ``model_key`` (freeing any other SR model first). Returns scale."""
        with self._lock:
            if self._net is not None and self._key == model_key and tile < 0:
                return self._scale
            arch = _ARCH_BY_KEY.get(model_key)
            if arch is None:
                raise ValueError(f"{model_key} is not a Real-ESRGAN model")
            path = self.models.path_for(model_key)
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file missing: {path.name}. Open AI Model Manager "
                    "and download it first.")

            self._release()
            torch = self.dev.torch()
            if arch["kind"] == "rrdb":
                net = RRDBNet(num_in_ch=3, num_out_ch=3, scale=arch["scale"],
                              num_feat=64, num_block=arch["blocks"],
                              num_grow_ch=32)
            else:
                net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                      num_conv=arch["conv"],
                                      upscale=arch["scale"], act_type="prelu")

            state = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                state = state.get("params_ema") or state.get("params") or state
            net.load_state_dict(state, strict=True)
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)

            if self.dev.info().is_cuda:
                self.dev.to(net)
            self._net = net
            self._key = model_key
            self._scale = int(arch["scale"])
            self._tile = tile if tile >= 0 else self.dev.tile_size()
            if self._tile == 0 and self.dev.info().is_cuda and \
                    self.dev.vram_total_mb() < 16000:
                self._tile = 512
            self.dev.mark_resident("sr", model_key, net)
            log.info("Loaded %s (scale x%d, tile %s, %s)", model_key, self._scale,
                     self._tile or "full-frame", self.dev.info().label)
            return self._scale

    def set_tile(self, tile: int) -> None:
        self._tile = max(0, int(tile))

    def set_tile_pad(self, pad: int) -> None:
        self._tile_pad = max(0, int(pad))

    def _release(self) -> None:
        if self._net is not None:
            del self._net
            self._net = None
            self._key = None
            self.dev.release("sr")
            log.info("Super-resolution model released from memory")

    def unload(self) -> None:
        with self._lock:
            self._release()

    # ------------------------------------------------------------------
    def _run_tile(self, tile_tensor):
        """Model forward for one tile (float32/16 NCHW in [0,1])."""
        with self.dev.torch().no_grad():
            out = self._net(tile_tensor)
            return out.clamp_(0, 1)

    def upscale(self, frame_rgb: np.ndarray,
                progress: Optional[callable] = None) -> np.ndarray:
        """Upscale one HxWx3 uint8 RGB frame -> (H*scale)x(W*scale)x3 uint8."""
        if self._net is None:
            raise RuntimeError("No super-resolution model loaded")
        torch = self.dev.torch()
        t0 = time.time()

        arr = frame_rgb.astype(np.float32) / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
        if self.dev.info().is_cuda:
            img = img.to(self.dev.device, dtype=self.dev.dtype)
        else:
            img = img.float()

        tile = self._tile
        out = self._tiled(img, tile, progress)
        out = out.float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = (out * 255.0).round().astype(np.uint8)
        self.frames_done += 1
        log.debug("upscale %dx%d -> %dx%d in %.2fs (tile=%s)",
                  frame_rgb.shape[1], frame_rgb.shape[0],
                  result.shape[1], result.shape[0], time.time() - t0, tile or "full")
        return result

    def _tiled(self, img, tile: int, progress=None):
        torch = self.dev.torch()
        _, _, h, w = img.shape
        scale = self._scale

        if tile <= 0 or (tile >= h and tile >= w):
            return self._guarded_run(img)

        out = torch.zeros(1, 3, h * scale, w * scale, device=img.device,
                          dtype=torch.float32)
        weight = torch.zeros(1, 1, h * scale, w * scale, device=img.device,
                             dtype=torch.float32)
        pad = self._tile_pad
        tiles_y = (h + tile - 1) // tile
        tiles_x = (w + tile - 1) // tile
        total = max(1, tiles_y * tiles_x)
        idx = 0
        current = tile

        y = 0
        while y < h:
            x = 0
            while x < w:
                th, tw = min(current, h - y), min(current, w - x)
                py0, px0 = max(0, y - pad), max(0, x - pad)
                py1, px1 = min(h, y + th + pad), min(w, x + tw + pad)
                patch = img[:, :, py0:py1, px0:px1]
                try:
                    sr = self._guarded_run(patch)
                except GPUMemoryError:
                    if current <= 64:
                        raise
                    current = max(64, current // 2)
                    log.warning("OOM - retrying frame with tile size %d", current)
                    del out, weight
                    return self._tiled(img, current, progress)

                cw0, ch0 = (x - px0) * scale, (y - py0) * scale
                cw, ch = tw * scale, th * scale
                out[:, :, y * scale:(y + th) * scale,
                    x * scale:(x + tw) * scale] += sr[:, :, ch0:ch0 + ch, cw0:cw0 + cw]
                weight[:, :, y * scale:(y + th) * scale,
                       x * scale:(x + tw) * scale] += 1.0
                del sr, patch
                idx += 1
                self.tiles_processed += 1
                if progress:
                    progress(idx / total)
                x += tw
            y += th
        return out / weight.clamp(min=1.0)

    def _guarded_run(self, tensor):
        if not self.dev.info().is_cuda:
            return self._run_tile(tensor)
        with self.dev.inference_guard() as guard:
            out = self._run_tile(tensor)
        if guard.oom:
            raise GPUMemoryError(str(guard.error))
        return out

    # ------------------------------------------------------------------
    def warmup(self, size: int = 64) -> bool:
        """Run one tiny forward pass so the first real frame is not slow."""
        if self._net is None:
            return False
        try:
            dummy = np.zeros((size, size, 3), dtype=np.uint8)
            dummy[:, :, 0] = 40
            self.upscale(dummy)
            return True
        except Exception as exc:  # pragma: no cover
            log.warning("Warmup failed: %s", exc)
            return False
