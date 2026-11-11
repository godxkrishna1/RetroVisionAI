"""Classical restoration filters applied per frame (OpenCV).

These are *not* presented as AI: they are fast, well-understood signal
processing steps that run after the neural upscaler for
  noise reduction, deblurring, sharpening, colour restoration,
  brightness/contrast correction, film-grain reduction, scratch reduction.

Every function takes and returns an RGB uint8 ``numpy`` array and accepts a
``strength`` level in ``("low", "medium", "high")``; nothing is no-op when
disabled at the pipeline level.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import cv2
import numpy as np

LEVEL = {"low": 0.33, "medium": 0.66, "high": 1.0, "off": 0.0}


def _amt(strength: str) -> float:
    return LEVEL.get((strength or "medium").lower(), 0.66)


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------
def denoise(frame: np.ndarray, strength: str = "medium") -> np.ndarray:
    """Edge-preserving noise reduction.

    Colour noise is attenuated with a fast non-local-means pass; the amount
    scales with ``strength``.  Kept conservative so 80s film grain does not
    turn into plastic.
    """
    amt = _amt(strength)
    if amt <= 0:
        return frame
    h = 3 + int(6 * amt)              # luminance filter strength
    h_color = 3 + int(9 * amt)
    return cv2.fastNlMeansDenoisingColored(
        frame, None, float(h), float(h_color), 7, 21)


def reduce_grain(frame: np.ndarray, strength: str = "medium") -> np.ndarray:
    """Film-grain reduction: bilateral filter tuned for high-frequency grain."""
    amt = _amt(strength)
    if amt <= 0:
        return frame
    d = 5 if amt < 0.5 else 7
    sigma_color = 18 + 42 * amt
    sigma_space = 18 + 42 * amt
    return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)


# ---------------------------------------------------------------------------
# Deblur / sharpen
# ---------------------------------------------------------------------------
def unsharp(frame: np.ndarray, amount: float = 1.0, radius: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(frame, (0, 0), radius)
    return cv2.addWeighted(frame, 1 + amount, blur, -amount, 0)


def deblur(frame: np.ndarray, strength: str = "medium") -> np.ndarray:
    """Light sharpening with an edge mask (avoids halos on flat areas)."""
    amt = _amt(strength)
    if amt <= 0:
        return frame
    sharp = unsharp(frame, amount=0.45 + 1.35 * amt, radius=1.0 + 0.6 * amt)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    mask = cv2.convertScaleAbs(edges)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    merged = sharp.astype(np.float32) * mask + frame.astype(np.float32) * (1 - mask)
    return np.clip(merged, 0, 255).astype(np.uint8)


def sharpen(frame: np.ndarray, strength: str = "medium") -> np.ndarray:
    amt = _amt(strength)
    if amt <= 0:
        return frame
    return unsharp(frame, amount=0.35 + 0.9 * amt, radius=1.4)


# ---------------------------------------------------------------------------
# Colour / tone
# ---------------------------------------------------------------------------
def _gray_world_balance(img_float: np.ndarray) -> np.ndarray:
    """White balance using the gray-world assumption (per-channel mean)."""
    means = img_float.reshape(-1, 3).mean(axis=0) + 1e-6
    target = float(means.mean())
    gains = target / means
    gains = np.clip(gains, 0.6, 1.6)      # never overcorrect faded prints
    return np.clip(img_float * gains, 0, 255)


def restore_color(frame: np.ndarray, strength: str = "medium") -> np.ndarray:
    """Repair faded/washed-out colour of old tape and film transfers.

    Steps (blended by strength):
      1. gray-world white balance
      2. adaptive saturation lift in LAB space
      3. gentle contrast S-curve to recover black level
    """
    amt = _amt(strength)
    if amt <= 0:
        return frame
    f = frame.astype(np.float32)
    f = _gray_world_balance(f)
    out = np.clip(f, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    sat_boost = 1.0 + 0.55 * amt
    a = np.clip((a.astype(np.float32) - 128.0) * sat_boost + 128.0, 0, 255)
    b = np.clip((b.astype(np.float32) - 128.0) * sat_boost + 128.0, 0, 255)
    out = cv2.cvtColor(cv2.merge([l, a.astype(np.uint8), b.astype(np.uint8)]),
                       cv2.COLOR_LAB2RGB)

    # contrast S-curve around mid-grey, blended by amount
    lut_x = np.arange(256, dtype=np.float32)
    curve = 255.0 / (1.0 + np.exp(-(lut_x - 128.0) / (48.0 - 12.0 * amt)))
    curve = (curve - curve.min()) / (curve.max() - curve.min()) * 255.0
    blended = (curve[lut_x.astype(np.uint8)] * amt +
               lut_x * (1 - amt)).astype(np.uint8)
    return cv2.LUT(out, blended)


def fix_brightness(frame: np.ndarray, strength: str = "medium",
                   target_mean: float = 118.0) -> np.ndarray:
    """Lift underexposed / dim tape captures toward a neutral mean level."""
    amt = _amt(strength)
    if amt <= 0:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean())
    if 95 <= mean <= 150:          # already fine
        return frame
    mean_n = max(mean, 1.0) / 255.0
    # gamma exponent: out = in ** gamma.  gamma < 1 brightens a dark image.
    gamma = np.log(max(target_mean, 1.0) / 255.0) / np.log(mean_n)
    gamma = float(np.clip(gamma, 0.35, 2.5))
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)],
                   dtype=np.uint8)
    corrected = cv2.LUT(frame, lut)
    return cv2.addWeighted(corrected, amt, frame, 1 - amt, 0)


def fix_contrast(frame: np.ndarray, strength: str = "medium") -> np.ndarray:
    """CLAHE on the luminance channel - recovers flat, low-contrast video."""
    amt = _amt(strength)
    if amt <= 0:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clip = 1.2 + 2.2 * amt
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    boosted = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)
    return cv2.addWeighted(boosted, amt, frame, 1 - amt, 0)


# ---------------------------------------------------------------------------
# Scratch reduction
# ---------------------------------------------------------------------------
def reduce_scratches(frame: np.ndarray, strength: str = "medium") -> np.ndarray:
    """Detect thin bright vertical/diagonal scratches and inpaint them.

    Old film prints show near-vertical bright lines.  We threshold very bright
    thin structures, keep only long vertical components, then inpaint.
    """
    amt = _amt(strength)
    if amt <= 0:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # local contrast against a horizontally blurred copy
    k = max(5, (frame.shape[1] // 60) | 1)
    blur = cv2.blur(gray, (k, 1))
    diff = cv2.subtract(gray, blur)
    thresh = 12 + int(28 * amt)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((7, 1), np.uint8))   # keep tall structures
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return frame
    cleaned = cv2.inpaint(frame, mask, 2, cv2.INPAINT_TELEA)
    return cv2.addWeighted(cleaned, amt, frame, 1 - amt, 0)


# ---------------------------------------------------------------------------
# Combined post-processing chain
# ---------------------------------------------------------------------------
def apply_chain(frame: np.ndarray, settings: Dict) -> np.ndarray:
    """Apply the enabled non-AI filters in a fixed, order-stable chain."""
    out = frame
    if settings.get("use_denoise"):
        out = denoise(out, settings.get("denoise_strength", "medium"))
    if settings.get("use_grain_reduction"):
        out = reduce_grain(out, settings.get("grain_strength", "medium"))
    if settings.get("use_color_restore"):
        out = restore_color(out, settings.get("color_strength", "medium"))
    if settings.get("use_brightness_fix"):
        out = fix_brightness(out, settings.get("brightness_strength", "medium"))
    if settings.get("use_contrast_fix"):
        out = fix_contrast(out, settings.get("contrast_strength", "medium"))
    if settings.get("use_scratch_reduction"):
        out = reduce_scratches(out, settings.get("scratch_strength", "medium"))
    if settings.get("use_deblur"):
        out = deblur(out, settings.get("deblur_strength", "medium"))
    if settings.get("use_sharpen"):
        out = sharpen(out, settings.get("sharpen_strength", "medium"))
    return out
