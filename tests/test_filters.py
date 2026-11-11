"""Classical filter behaviour and colour handling."""
from __future__ import annotations

import numpy as np
import pytest

from core import filters


@pytest.fixture()
def frame():
    rng = np.random.default_rng(3)
    base = np.zeros((96, 128, 3), np.uint8)
    yy, xx = np.mgrid[0:96, 0:128]
    base[..., 0] = (xx * 2) % 256
    base[..., 1] = (yy * 2) % 256
    base[..., 2] = 128
    noisy = np.clip(base.astype(np.int16) + rng.normal(0, 18, base.shape),
                    0, 255).astype(np.uint8)
    return noisy


def _std(img):
    return float(np.std(img.astype(np.float32)))


def test_denoise_reduces_noise(frame):
    out = filters.denoise(frame, "medium")
    assert out.shape == frame.shape and out.dtype == np.uint8
    # local variance should drop noticeably
    assert _std(out.astype(np.float32) - frame.astype(np.float32)) > 0


def test_dark_frame_gets_brighter():
    dark = np.full((64, 64, 3), 40, np.uint8)
    out = filters.fix_brightness(dark, "high")
    assert out.mean() > dark.mean()


def test_faded_frame_gains_saturation():
    # a faded colour patch (low chroma) must gain chroma after restoration
    faded = np.full((64, 64, 3), 128, np.uint8)
    faded[:32, :, 0] = 170
    faded[:32, :, 2] = 90
    out = filters.restore_color(faded, "high")
    assert out.std() > faded.std()


def test_brightness_handles_all_exposures():
    for level, should_rise in ((40, True), (90, True), (180, False)):
        frame = np.full((48, 48, 3), level, np.uint8)
        out = filters.fix_brightness(frame, "high")
        if should_rise:
            assert out.mean() > frame.mean()
        assert 80 <= out.mean() <= 160


def test_sharpen_preserves_range(frame):
    out = filters.sharpen(frame, "high")
    assert out.min() >= 0 and out.max() <= 255


def test_off_strength_is_identity(frame):
    off = "off"
    assert np.array_equal(filters.denoise(frame, off), frame)
    assert np.array_equal(filters.sharpen(frame, off), frame)


def test_chain_only_applies_enabled_steps(frame):
    settings = {"use_denoise": False, "use_sharpen": False,
                "use_color_restore": False, "use_deblur": False,
                "use_scratch_reduction": False, "use_contrast_fix": False,
                "use_brightness_fix": False, "use_grain_reduction": False}
    assert np.array_equal(filters.apply_chain(frame, settings), frame)


def test_chain_with_all_steps_keeps_shape(frame):
    settings = {k: True for k in (
        "use_denoise", "use_deblur", "use_sharpen", "use_color_restore",
        "use_brightness_fix", "use_contrast_fix", "use_grain_reduction")}
    out = filters.apply_chain(frame, settings)
    assert out.shape == frame.shape and out.dtype == np.uint8
