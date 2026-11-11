"""Shared pytest fixtures.

Tests that need FFmpeg or torch are skipped automatically when the dependency
is missing so the suite also runs on a bare dev machine.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from utils.ffmpeg import FFmpeg  # noqa: E402


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _have_ffmpeg() -> bool:
    try:
        FFmpeg().locate()
        return True
    except Exception:
        return False


HAVE_TORCH = _have_torch()
HAVE_FFMPEG = _have_ffmpeg()

requires_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
requires_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """4 s 320x240 @ 25 fps H.264 clip with a 440 Hz AAC tone."""
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not installed")
    import subprocess

    target = tmp_path_factory.mktemp("media") / "sample_av.mp4"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=44100",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
        "-shortest", str(target),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return target


@pytest.fixture()
def temp_config(tmp_path):
    from utils.config import Config

    cfg = Config(tmp_path / "config.json")
    cfg.ensure_dirs()
    return cfg


@pytest.fixture(scope="session")
def models_dir() -> Path:
    """Directory holding real model weights, if the environment has any."""
    env = os.environ.get("RV_MODELS_DIR")
    candidates = [Path(env)] if env else []
    candidates += [Path("/tmp/rv_models"), ROOT / "models"]
    for c in candidates:
        if c and c.exists() and any(p.stat().st_size > 1_000_000
                                    for p in c.glob("*.pth")):
            return c
    pytest.skip("no model weights available")


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
