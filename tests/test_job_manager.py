"""Job queue behaviour (worker replaced by a stub - no GPU needed)."""
from __future__ import annotations

import time

import pytest

from core.job import EnhancementSettings, JobResult, JobSpec
from core.job_manager import JobManager
from utils.ffmpeg import EncodeSettings


class _Ctx:
    def __init__(self, tmp_path):
        from utils.config import Config
        self.config = Config(tmp_path / "config.json")
        self.config.root = tmp_path
        self.ffmpeg = None
        self.model_manager = None


def _spec(tmp_path, name="clip.mp4"):
    return JobSpec(input_path=str(tmp_path / name),
                   enhancement=EnhancementSettings(),
                   encode=EncodeSettings(),
                   output_path=str(tmp_path / f"out_{name}"))


@pytest.fixture()
def manager(qapp, tmp_path):
    return JobManager(_Ctx(tmp_path), tmp_path / "history.jsonl")


def test_enqueue_and_snapshot(manager, tmp_path):
    spec = _spec(tmp_path)
    manager.queue.append(spec)          # avoid starting a real worker
    assert manager.queue_snapshot()[0] is spec
    assert manager.remove(spec.job_id) is True
    assert manager.queue_snapshot() == []


def test_history_persists_across_instances(qapp, tmp_path):
    mgr = JobManager(_Ctx(tmp_path), tmp_path / "history.jsonl")
    result = JobResult(job_id="job_x", status="completed",
                       output_path=str(tmp_path / "out.mp4"),
                       frames_processed=120, duration_s=5.0,
                       input_info={"width": 320, "height": 240, "duration": 4.0})
    spec = _spec(tmp_path)
    mgr.current = spec
    mgr._record(result)

    reloaded = JobManager(_Ctx(tmp_path), tmp_path / "history.jsonl")
    records = reloaded.history_records()
    assert len(records) == 1
    assert records[0]["input_name"] == "clip.mp4"
    assert records[0]["resolution"] == "320x240"
    assert reloaded.stats()["completed"] == 1


def test_clear_history(manager, tmp_path):
    mgr = JobManager(_Ctx(tmp_path), tmp_path / "history.jsonl")
    mgr.current = _spec(tmp_path)
    mgr._record(JobResult(job_id="job_y", status="failed", error="boom"))
    assert mgr.stats()["failed"] == 1
    mgr.clear_history()
    assert mgr.history_records() == []
