"""Engine tests. Weight-dependent tests run when real weights are available."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_torch

from ai.device_manager import DeviceManager, GPUMemoryError, OOMGuard


@requires_torch
def test_device_manager_cpu_mode():
    dev = DeviceManager("cpu", "fp32")
    info = dev.info(refresh=True)
    assert info.device == "cpu"
    assert info.torch_dtype == "float32"
    assert dev.tile_size() > 0
    assert dev.batch_frames() >= 1


@requires_torch
def test_oom_guard_swallows_memory_errors():
    dev = DeviceManager("cpu", "fp32")
    dev.info()
    with dev.inference_guard() as guard:
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert guard.oom is True
    assert isinstance(guard.error, RuntimeError)


@requires_torch
def test_oom_guard_passes_normal_exceptions():
    dev = DeviceManager("cpu", "fp32")
    dev.info()
    with pytest.raises(ValueError):
        with dev.inference_guard() as guard:
            raise ValueError("not an oom")
    assert guard.oom is False


@requires_torch
def test_tiled_inference_stitches_exactly():
    """Tiled path must reproduce the untiled result on a simple op."""
    import torch
    from ai.device_manager import TiledInferenceMixin

    class Engine(TiledInferenceMixin):
        pass

    eng = Engine()

    def runner(tile):
        return torch.nn.functional.interpolate(tile, scale_factor=2, mode="nearest")

    img = torch.rand(1, 3, 40, 56)
    full = runner(img)
    tiled = eng.tiled_infer(img, runner, scale=2, tile=16, tile_pad=4)
    assert tiled.shape == full.shape
    assert torch.allclose(full, tiled, atol=1e-5)


# ---------------------------------------------------------------------------
# Real-weight inference (skipped when weights are absent)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def manager(models_dir):
    from ai.model_manager import ModelManager
    return ModelManager(models_dir)


def _gradient(h=32, w=48):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([(xx * 5) % 256, (yy * 4) % 256, ((xx + yy) * 3) % 256],
                    axis=-1).astype(np.uint8)


@pytest.mark.parametrize("key,scale", [
    ("realesrgan_x2plus", 2),
    ("realesrgan_x4plus", 4),
])
def test_realesrgan_upscales(manager, key, scale):
    if not manager.is_installed(key):
        pytest.skip(f"{key} weights not available")
    from ai.realesrgan_engine import RealESRGANEngine
    dev = DeviceManager("cpu", "fp32")
    engine = RealESRGANEngine(dev, manager)
    assert engine.load(key, tile=0) == scale
    src = _gradient()
    out = engine.upscale(src)
    assert out.shape == (src.shape[0] * scale, src.shape[1] * scale, 3)
    assert out.dtype == np.uint8
    # structure must be preserved, not garbage
    assert abs(float(out.mean()) - float(src.mean())) < 40
    engine.unload()


def test_rife_interpolates(manager):
    if not manager.is_installed("rife47"):
        pytest.skip("rife47 weights not available")
    from ai.rife_engine import RIFEEngine
    dev = DeviceManager("cpu", "fp32")
    engine = RIFEEngine(dev, manager)
    engine.load("rife47")
    a = _gradient(64, 64)
    b = np.roll(a, 8, axis=1)
    mid = engine.interpolate_pair(a, b, 0.5)
    assert mid.shape == a.shape and mid.dtype == np.uint8
    # midpoint must be closer to both inputs than they are to each other
    d_a = np.abs(mid.astype(int) - a.astype(int)).mean()
    d_ab = np.abs(a.astype(int) - b.astype(int)).mean()
    assert d_a <= d_ab
    engine.unload()


def test_gfpgan_restores_face_crop(manager):
    if not manager.is_installed("gfpgan_v14"):
        pytest.skip("gfpgan weights not available")
    pytest.importorskip("cv2")
    from ai.gfpgan_engine import Face, GFPGANEngine
    dev = DeviceManager("cpu", "fp32")
    engine = GFPGANEngine(dev, manager)
    engine.load("gfpgan_v14")
    frame = np.full((200, 240, 3), 90, np.uint8)
    frame[40:140, 60:160] = (150, 120, 100)
    out = engine._restore_face(frame, Face(60, 40, 100, 100), 0.9)
    assert out is not None and out.shape == frame.shape
    engine.unload()
