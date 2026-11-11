"""GFPGAN v1.4 face-restoration engine.

Pipeline per frame
------------------
1. detect faces with OpenCV's Haar cascade (ships with opencv-python, no
   extra model download, works on stylised 80s/90s footage);
2. for each face: crop with margin -> resize to 512 -> GFPGAN forward ->
   resize back -> blend with the original crop using ``strength``;
3. paste back with a soft (feathered) mask so edges never show.

``strength`` (0..1) controls the blend between the original and restored
face, matching the UI slider.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

from .archs.gfpganv1 import GFPGANv1Clean
from .device_manager import DeviceManager, GPUMemoryError

log = get_logger("gfpgan")

# GFPGAN v1.4 checkpoint geometry (verified against params_ema shapes):
# 512px output, channel_multiplier=2, different_w, SFT on half the channels.
GFPGAN_V14_CONFIG = dict(
    out_size=512, num_style_feat=512, channel_multiplier=2, fix_decoder=False,
    num_mlp=8, input_is_latent=True, different_w=True, narrow=1, sft_half=True,
)


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0

    @property
    def area(self) -> int:
        return self.w * self.h


class GFPGANEngine:
    def __init__(self, device: DeviceManager, model_manager) -> None:
        self.dev = device
        self.models = model_manager
        self._lock = threading.RLock()
        self._net = None
        self._cascade: Optional[cv2.CascadeClassifier] = None
        self.strength = 0.8
        self.min_face = 32
        self.faces_restored = 0
        self.frames_with_faces = 0

    # ------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._net is not None

    def load(self, model_key: str = "gfpgan_v14") -> bool:
        with self._lock:
            if self._net is not None:
                return True
            path = self.models.path_for(model_key)
            if not path.exists():
                raise FileNotFoundError(
                    f"GFPGAN weights missing ({path.name}). Download them in "
                    "AI Model Manager - face restoration needs the 333 MB model.")
            torch = self.dev.torch()
            net = GFPGANv1Clean(**GFPGAN_V14_CONFIG)
            state = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "params_ema" in state:
                state = state["params_ema"]
            net.load_state_dict(state, strict=True)
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            if self.dev.info().is_cuda:
                self.dev.to(net)
            self._net = net
            self.dev.mark_resident("face", model_key, net)
            self._load_cascade()
            log.info("GFPGAN v1.4 loaded (%s)", self.dev.info().label)
            return True

    def _load_cascade(self) -> None:
        if self._cascade is not None:
            return
        candidates = []
        try:
            base = cv2.data.haarcascades  # type: ignore[attr-defined]
            candidates.append(base + "haarcascade_frontalface_default.xml")
            candidates.append(base + "haarcascade_frontalface_alt2.xml")
        except AttributeError:
            pass
        for c in candidates:
            clf = cv2.CascadeClassifier(c)
            if not clf.empty():
                self._cascade = clf
                log.info("Face detector ready: %s", c.rsplit("/", 1)[-1])
                return
        log.warning("No Haar cascade found - face restoration disabled")
        self._cascade = None

    def unload(self) -> None:
        with self._lock:
            if self._net is not None:
                del self._net
                self._net = None
                self.dev.release("face")
                log.info("GFPGAN released from memory")

    # ------------------------------------------------------------------
    def detect_faces(self, frame_rgb: np.ndarray) -> List[Face]:
        if self._cascade is None:
            self._load_cascade()
        if self._cascade is None:
            return []
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape[:2]
        min_size = max(24, min(self.min_face, min(h, w) // 3))
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=5, minSize=(min_size, min_size),
            flags=cv2.CASCADE_SCALE_IMAGE)
        result = [Face(int(x), int(y), int(fw), int(fh)) for x, y, fw, fh in faces]

        # Prefer the largest faces; cap per frame to keep processing sane
        result.sort(key=lambda f: f.area, reverse=True)
        return result[:6]

    def restore_frame(self, frame_rgb: np.ndarray,
                      strength: Optional[float] = None,
                      detected: Optional[List[Face]] = None) -> Tuple[np.ndarray, int]:
        """Restore all detected faces in one RGB uint8 frame.

        Returns ``(frame, n_faces_restored)``.
        """
        if self._net is None:
            raise RuntimeError("GFPGAN is not loaded")
        strength = self.strength if strength is None else float(strength)
        if strength <= 0:
            return frame_rgb, 0
        faces = detected if detected is not None else self.detect_faces(frame_rgb)
        if not faces:
            return frame_rgb, 0

        out = frame_rgb.copy()
        count = 0
        for face in faces:
            try:
                restored = self._restore_face(out, face, strength)
                if restored is not None:
                    out = restored
                    count += 1
                    self.faces_restored += 1
            except GPUMemoryError:
                log.warning("OOM while restoring a face - skipping this face")
                self.dev.release()
            except Exception as exc:  # a single bad face must not kill the frame
                log.warning("Face restoration failed: %s", exc)
        if count:
            self.frames_with_faces += 1
        return out, count

    # ------------------------------------------------------------------
    def _restore_face(self, frame: np.ndarray, face: Face,
                      strength: float) -> Optional[np.ndarray]:
        torch = self.dev.torch()
        h, w = frame.shape[:2]
        # generous margin so hair/chin are included, like GFPGAN's aligner
        margin = 0.32
        cx, cy = face.x + face.w / 2.0, face.y + face.h / 2.0
        side = max(face.w, face.h) * (1 + 2 * margin)
        x0 = int(max(0, cx - side / 2))
        y0 = int(max(0, cy - side / 2))
        x1 = int(min(w, cx + side / 2))
        y1 = int(min(h, cy + side / 2))
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None

        crop = frame[y0:y1, x0:x1]
        crop_rgb = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)

        tensor = torch.from_numpy(crop_rgb.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        if self.dev.info().is_cuda:
            tensor = tensor.to(self.dev.device, dtype=self.dev.dtype)

        with self.dev.inference_guard() as guard:
            with torch.no_grad():
                output, _ = self._net(tensor, return_rgb=False,
                                      randomize_noise=False)
        if guard.oom:
            raise GPUMemoryError(str(guard.error))

        output = output.float().clamp(-1, 1)
        output = (output + 1) / 2
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        restored = (output * 255.0).round().astype(np.uint8)

        # back to the crop size
        restored = cv2.resize(restored, (x1 - x0, y1 - y0),
                              interpolation=cv2.INTER_LANCZOS4)

        # blend original & restored with a feathered ellipse mask
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        mcx, mcy = (x1 - x0) / 2, (y1 - y0) / 2
        cv2.ellipse(mask, (int(mcx), int(mcy)),
                    (int((x1 - x0) * 0.46), int((y1 - y0) * 0.46)),
                    0, 0, 360, 1.0, -1)
        blur = max(3, (min(x1 - x0, y1 - y0) // 8) | 1)
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        mask = (mask * max(0.0, min(1.0, strength)))[..., None]

        blended = (restored.astype(np.float32) * mask +
                   crop.astype(np.float32) * (1 - mask)).round().astype(np.uint8)
        result = frame.copy()
        result[y0:y1, x0:x1] = blended
        return result

    # ------------------------------------------------------------------
    def warmup(self) -> bool:
        if self._net is None:
            return False
        try:
            self._restore_face(np.zeros((512, 512, 3), np.uint8),
                               Face(128, 128, 256, 256), 1.0)
            return True
        except Exception as exc:  # pragma: no cover
            log.warning("GFPGAN warmup failed: %s", exc)
            return False
