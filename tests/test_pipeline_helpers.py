"""Pure-logic tests for the pipeline (no GPU/AI required)."""
from __future__ import annotations

import pytest

from core.job import EnhancementSettings, JobSpec
from core.pipeline import ResumeState, target_size
from utils.ffmpeg import EncodeSettings, VideoInfo
from utils.file_manager import JobWorkspace


class _Info(VideoInfo):
    pass


@pytest.mark.parametrize("target,expected", [
    ("1080p", (1440, 1080)),     # 4:3 into 16:9 box -> pillarboxed size
    ("720p", (960, 720)),
    ("2k", (1920, 1440)),
    ("4k", (2880, 2160)),
])
def test_target_size_preserves_aspect(target, expected):
    info = VideoInfo(path="x", width=320, height=240)
    assert target_size(info, target) == expected


def test_target_size_original():
    info = VideoInfo(path="x", width=1281, height=721)   # odd numbers
    assert target_size(info, "original") == (1280, 720)


def test_target_size_widescreen():
    info = VideoInfo(path="x", width=1920, height=1080)
    assert target_size(info, "1080p") == (1920, 1080)
    assert target_size(info, "4k") == (3840, 2160)


def test_enhancement_settings_validation():
    enh = EnhancementSettings(target_resolution="nonsense",
                              denoise_strength="extreme", face_strength=2.5,
                              interpolation_factor=7)
    warnings = enh.validate()
    assert enh.target_resolution == "1080p"
    assert enh.denoise_strength == "medium"
    assert enh.face_strength == 1.0
    assert enh.interpolation_factor == 2
    assert isinstance(warnings, list)


def test_enhancement_ai_steps():
    enh = EnhancementSettings(use_super_resolution=True,
                              use_face_restoration=True,
                              use_interpolation=True,
                              use_temporal_restore=False)
    assert enh.ai_steps() == ["superres", "face", "interpolation"]
    assert enh.filters_enabled() in (True, False)


def test_job_spec_defaults():
    spec = JobSpec(input_path="/videos/a.mp4",
                   enhancement=EnhancementSettings(),
                   encode=EncodeSettings(),
                   output_path="/out/a_1080p.mp4")
    assert spec.label == "a"
    assert spec.job_id.startswith("job_")
    data = spec.to_dict()
    assert data["enhancement"]["target_resolution"] == "1080p"


def test_resume_state_roundtrip(tmp_path):
    ws = JobWorkspace(tmp_path, "job_test")
    state = ResumeState(ws, enabled=True)
    assert state.load() == 0
    state.save(42, "Restoring frames")
    assert ResumeState(ws, enabled=True).load() == 42
    state.clear()
    assert ResumeState(ws, enabled=True).load() == 0


def test_resume_disabled_writes_nothing(tmp_path):
    ws = JobWorkspace(tmp_path, "job_off")
    state = ResumeState(ws, enabled=False)
    state.save(10, "stage")
    assert not state.file.exists()


def test_job_workspace_cleanup_only_inside_temp(tmp_path):
    ws = JobWorkspace(tmp_path, "job_cleanup")
    (ws.frames_out / "00000001.png").write_bytes(b"x")
    ws.cleanup()
    assert not ws.root.exists()


def test_job_workspace_keep(tmp_path):
    ws = JobWorkspace(tmp_path, "job_keep")
    ws.cleanup(keep=True)
    assert ws.root.exists()
