"""End-to-end pipeline test.

Runs a real 1-second restoration (super-resolution + filters + encode) on a
generated clip.  Requires FFmpeg, torch and Real-ESRGAN weights.
"""
from __future__ import annotations

import pytest

from conftest import HAVE_FFMPEG, HAVE_TORCH, requires_ffmpeg, requires_torch

pytestmark = pytest.mark.skipif(
    not (HAVE_FFMPEG and HAVE_TORCH), reason="needs ffmpeg + torch")


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    import subprocess
    target = tmp_path_factory.mktemp("tiny") / "tiny.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", str(target)], check=True, capture_output=True, timeout=60)
    return target


def test_full_pipeline_produces_video(tiny_video, models_dir, tmp_path):
    from ai.device_manager import DeviceManager
    from ai.model_manager import ModelManager
    from core.job import EnhancementSettings, JobSpec
    from core.pipeline import VideoPipeline
    from utils.ffmpeg import FFmpeg, EncodeSettings

    managers = ModelManager(models_dir)
    if not managers.is_installed("realesr_general_x4v3"):
        pytest.skip("no upscaler weights available")

    events = []
    spec = JobSpec(
        input_path=str(tiny_video),
        enhancement=EnhancementSettings(
            target_resolution="720p", ai_model="realesr-general-x4v3",
            use_super_resolution=True, use_face_restoration=False,
            use_denoise=False, use_color_restore=False),
        encode=EncodeSettings(codec="h264", encoder="cpu", crf=30,
                              preset="ultrafast"),
        output_path=str(tmp_path / "out.mp4"))

    pipeline = VideoPipeline(
        FFmpeg(), DeviceManager("cpu", "fp32"), managers,
        tmp_path / "temp", progress_cb=lambda p: events.append(p),
        resume=False)
    result = pipeline.run(spec)

    assert result.status == "completed", result.error
    out = tmp_path / "out.mp4"
    assert out.exists() and out.stat().st_size > 1000
    info = FFmpeg().probe(out)
    assert info.width == 960 and info.height == 720   # 4:3 preserved in 720p box
    assert info.has_audio, "original audio must be preserved"
    assert result.frames_processed == pytest.approx(10, abs=1)
    assert any(p.stage == "Completed" for p in events)
    # progress must be monotonic within the restore stage
    percents = [p.percent for p in events if p.stage == "Restoring frames"]
    assert percents == sorted(percents)


def test_pipeline_reports_missing_model(tiny_video, tmp_path):
    from ai.device_manager import DeviceManager
    from ai.model_manager import ModelManager
    from core.job import EnhancementSettings, JobSpec
    from core.pipeline import VideoPipeline
    from utils.ffmpeg import FFmpeg, EncodeSettings

    empty = ModelManager(tmp_path / "models")
    spec = JobSpec(
        input_path=str(tiny_video),
        enhancement=EnhancementSettings(target_resolution="720p",
                                        use_super_resolution=True),
        encode=EncodeSettings(encoder="cpu", preset="ultrafast"),
        output_path=str(tmp_path / "never.mp4"))
    result = VideoPipeline(FFmpeg(), DeviceManager("cpu", "fp32"), empty,
                           tmp_path / "temp", resume=False).run(spec)
    assert result.status == "failed"
    assert "model" in result.error.lower()
    assert not (tmp_path / "never.mp4").exists()
