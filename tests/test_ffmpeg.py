"""FFprobe parsing, target sizing and real FFmpeg encode/decode round-trips."""
from __future__ import annotations

import pytest

from conftest import requires_ffmpeg
from utils.ffmpeg import EncodeSettings, FFmpeg, FFmpegError, VideoInfo, _parse_fraction
from utils.file_manager import human_duration, human_size, unique_path


def test_parse_fraction():
    assert _parse_fraction("25/1") == 25.0
    assert _parse_fraction("30000/1001") == pytest.approx(29.97, abs=0.01)
    assert _parse_fraction("0/0") == 0.0
    assert _parse_fraction("N/A") == 0.0
    assert _parse_fraction("bogus") == 0.0


def test_human_helpers():
    assert human_size(0) == "0 B"
    assert human_size(1536).startswith("1.5 KB")
    assert human_size(5 * 1024 ** 3).startswith("5.0 GB")
    assert human_duration(0) == "00:00"
    assert human_duration(75) == "01:15"
    assert human_duration(3661) == "01:01:01"


def test_unique_path(tmp_path):
    (tmp_path / "clip.mp4").write_text("x")
    second = unique_path(tmp_path, "clip")
    assert second.name == "clip_1.mp4"


@requires_ffmpeg
def test_probe_reads_streams(sample_video):
    info = FFmpeg().probe(sample_video)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(25, abs=0.01)
    assert info.nb_frames == pytest.approx(75, abs=2)
    assert info.duration == pytest.approx(3, abs=0.2)
    assert info.video_codec == "h264"
    assert info.has_audio and info.audio_codec == "aac"
    assert info.container == "mov" or "mp4" in info.container


@requires_ffmpeg
def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(FFmpegError):
        FFmpeg().probe(tmp_path / "nope.mp4")


@requires_ffmpeg
def test_extract_frames_and_rebuild(sample_video, tmp_path):
    ff = FFmpeg()
    frames_dir = tmp_path / "frames"
    seen = []
    count = ff.extract_frames(sample_video, frames_dir,
                              progress=lambda done, pct: seen.append((done, pct)),
                              total_frames=75)
    assert count == pytest.approx(75, abs=2)
    assert seen and seen[-1][1] == 100.0
    assert len(list(frames_dir.glob("*.png"))) == count

    audio = ff.extract_audio(sample_video, tmp_path / "audio")
    assert audio is not None and audio.stat().st_size > 0

    out = tmp_path / "rebuilt.mp4"
    ff.build_video(frames_dir / "%08d.png", 25.0, out,
                   settings=EncodeSettings(codec="h264", encoder="cpu", crf=28,
                                           preset="ultrafast"),
                   audio=str(audio), total_frames=count, duration=count / 25)
    assert out.exists() and out.stat().st_size > 1000
    rebuilt = ff.probe(out)
    assert rebuilt.width == 320 and rebuilt.height == 240
    assert rebuilt.has_audio
    assert rebuilt.duration == pytest.approx(count / 25, abs=0.3)


@requires_ffmpeg
def test_thumbnail(sample_video, tmp_path):
    ff = FFmpeg()
    thumb = ff.make_thumbnail(sample_video, tmp_path / "thumb.jpg")
    assert thumb is not None and thumb.exists()


@requires_ffmpeg
def test_list_encoders_contains_libx264():
    ff = FFmpeg()
    assert "libx264" in ff.list_encoders()


def test_video_info_roundtrip():
    info = VideoInfo(path="x.mp4", width=1920, height=1080, duration=10.0, fps=25)
    assert info.resolution == "1920x1080"
    assert info.aspect == pytest.approx(16 / 9)
    clone = VideoInfo.from_dict(info.to_dict())
    assert clone.width == 1920 and clone.fps == 25


def test_cancel_event_stops_encode(tmp_path):
    """A set cancel flag must abort extraction quickly, not hang."""
    import threading
    ff = FFmpeg()
    if not ff.available:
        pytest.skip("ffmpeg missing")
    ev = threading.Event()
    ev.set()
    with pytest.raises(FFmpegError):
        ff._run(["ffmpeg", "-f", "lavfi", "-i", "testsrc2=d=5",
                 "-f", "null", "-"], None, ev, "cancel test")
