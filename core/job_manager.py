"""Job queue, background worker threads and persistent history.

* One job runs at a time (GPU work is serialised by design - it keeps VRAM
  predictable and progress honest).
* The worker is a ``QThread`` running :class:`PipelineWorker`; all
  communication with the UI uses Qt signals emitted from the worker thread.
* History is a JSON-lines file in the config folder, written atomically.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QThread, Signal

from ai.device_manager import DeviceManager
from core.job import JobProgress, JobResult, JobSpec, JobStage
from core.pipeline import VideoPipeline
from utils.ffmpeg import FFmpeg
from utils.logger import get_logger

log = get_logger("jobs")


class PipelineWorker(QThread):
    """Runs one :class:`VideoPipeline` away from the UI thread."""

    progress = Signal(object)         # JobProgress
    stage_changed = Signal(str, int, int)
    finished_ok = Signal(object)      # JobResult
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, spec: JobSpec, ffmpeg: FFmpeg, device: DeviceManager,
                 model_manager, temp_root: Path, options: Dict) -> None:
        super().__init__()
        self.spec = spec
        self.cancel_event = threading.Event()
        self._ffmpeg = ffmpeg
        self._device = device
        self._models = model_manager
        self._temp_root = Path(temp_root)
        self._options = options
        self.pipeline: Optional[VideoPipeline] = None

    def cancel(self) -> None:
        log.info("Cancel requested for %s", self.spec.job_id)
        self.cancel_event.set()

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            self.pipeline = VideoPipeline(
                self._ffmpeg, self._device, self._models, self._temp_root,
                cancel_event=self.cancel_event,
                progress_cb=self.progress.emit,
                stage_cb=self.stage_changed.emit,
                resume=self._options.get("resume", True),
                keep_temp=self._options.get("keep_temp", False),
                tile_size=self._options.get("tile_size", -1),
                tile_pad=self._options.get("tile_pad", 16),
                batch_frames=self._options.get("batch_frames", 4))
            result = self.pipeline.run(self.spec)
            if result.status == "completed":
                self.finished_ok.emit(result)
            elif result.status == "cancelled":
                self.cancelled.emit()
            else:
                self.failed.emit(result.error or "Processing failed")
        except Exception as exc:  # pragma: no cover - safety net
            log.exception("Worker crashed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class JobManager(QObject):
    """Owns the queue, the active worker and the on-disk history."""

    job_started = Signal(object)          # JobSpec
    job_progress = Signal(object)         # JobProgress
    job_stage = Signal(str, int, int)
    job_finished = Signal(object)         # JobResult
    job_failed = Signal(object, str)      # (JobSpec, error)
    job_cancelled = Signal(object)
    queue_changed = Signal()

    def __init__(self, ctx, history_path: Path) -> None:
        super().__init__()
        self.ctx = ctx
        self.history_path = Path(history_path)
        self.queue: deque[JobSpec] = deque()
        self.current: Optional[JobSpec] = None
        self.worker: Optional[PipelineWorker] = None
        self.last_result: Optional[JobResult] = None
        self.paused = False
        self._recent: List[dict] = []
        self._load_history()

    # ------------------------------------------------------------------
    # Queue API
    # ------------------------------------------------------------------
    def enqueue(self, spec: JobSpec) -> None:
        self.queue.append(spec)
        log.info("Queued job %s (%s)", spec.job_id, spec.label)
        self.queue_changed.emit()
        self._maybe_start_next()

    def enqueue_many(self, specs: List[JobSpec]) -> None:
        for spec in specs:
            self.queue.append(spec)
        log.info("Queued %d jobs", len(specs))
        self.queue_changed.emit()
        self._maybe_start_next()

    def remove(self, job_id: str) -> bool:
        for spec in list(self.queue):
            if spec.job_id == job_id:
                self.queue.remove(spec)
                self.queue_changed.emit()
                return True
        return False

    def clear_queue(self) -> None:
        self.queue.clear()
        self.queue_changed.emit()

    @property
    def busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def queue_snapshot(self) -> List[JobSpec]:
        items = ([self.current] if self.current else []) + list(self.queue)
        return items

    def start(self) -> None:
        """Start processing (called after enqueue; explicit for UI control)."""
        self._maybe_start_next()

    def pause(self) -> None:
        self.paused = True
        log.info("Queue paused")
        self.queue_changed.emit()

    def resume(self) -> None:
        self.paused = False
        log.info("Queue resumed")
        self.queue_changed.emit()
        self._maybe_start_next()

    # ------------------------------------------------------------------
    def _maybe_start_next(self) -> None:
        if self.paused:
            return
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.queue:
            self.current = None
            return

        spec = self.queue.popleft()
        self.current = spec
        self.queue_changed.emit()
        opts = {
            "resume": bool(self.ctx.config.get("processing.resume_enabled", True)),
            "keep_temp": bool(self.ctx.config.get("processing.keep_temp_frames", False)),
            "tile_size": int(self.ctx.config.get("processing.tile_size", 0))
            if int(self.ctx.config.get("processing.tile_size", 0)) > 0 else -1,
            "tile_pad": int(self.ctx.config.get("processing.tile_pad", 16)),
            "batch_frames": int(self.ctx.config.get("processing.max_batch_frames", 4)),
        }
        device = DeviceManager.instance(
            self.ctx.config.get("processing.device", "auto"),
            self.ctx.config.get("processing.precision", "fp16"))
        worker = PipelineWorker(spec, self.ctx.ffmpeg, device,
                               self.ctx.model_manager, self.ctx.config.temp_dir,
                               opts)
        worker.progress.connect(self.job_progress)
        worker.stage_changed.connect(self.job_stage)
        worker.finished_ok.connect(self._on_success)
        worker.failed.connect(lambda msg: self._on_failure(spec, msg))
        worker.cancelled.connect(lambda: self._on_cancelled(spec))
        worker.finished.connect(lambda: self._on_worker_done(spec))
        self.worker = worker
        self.job_started.emit(spec)
        log.info("Starting job %s: %s", spec.job_id, spec.label)
        worker.start()

    def _on_success(self, result: JobResult) -> None:
        self.last_result = result
        self._record(result)
        self.job_finished.emit(result)

    def _on_failure(self, spec: JobSpec, message: str) -> None:
        result = JobResult(job_id=spec.job_id, status="failed", error=message,
                           input_info={}, started_at=time.time(),
                           finished_at=time.time())
        self.last_result = result
        self._record(result, spec)
        self.job_failed.emit(spec, message)

    def _on_cancelled(self, spec: JobSpec) -> None:
        result = JobResult(job_id=spec.job_id, status="cancelled",
                           started_at=time.time(), finished_at=time.time())
        self.last_result = result
        self._record(result, spec)
        self.job_cancelled.emit(spec)

    def _on_worker_done(self, spec: JobSpec) -> None:
        self.current = None
        self.worker = None
        self.queue_changed.emit()
        self._maybe_start_next()

    # ------------------------------------------------------------------
    def cancel_current(self) -> None:
        if self.worker is not None:
            self.worker.cancel()

    def cancel_all(self) -> None:
        self.queue.clear()
        if self.worker is not None:
            self.worker.cancel()
        self.queue_changed.emit()

    # ------------------------------------------------------------------
    # History (JSON lines, newest last)
    # ------------------------------------------------------------------
    def _load_history(self) -> None:
        self._recent = []
        if not self.history_path.exists():
            return
        try:
            for line in self.history_path.read_text("utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self._recent.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError as exc:
            log.warning("Could not read history: %s", exc)

    def _record(self, result: JobResult, spec: Optional[JobSpec] = None) -> None:
        spec = spec or self.current
        if spec is None:
            return
        inp = result.input_info or {}
        record = {
            "job_id": result.job_id,
            "status": result.status,
            "input_path": spec.input_path,
            "input_name": Path(spec.input_path).name,
            "output_path": result.output_path,
            "resolution": f"{inp.get('width', 0)}x{inp.get('height', 0)}",
            "duration": _fmt_duration(inp.get("duration", 0)),
            "output_duration": float((result.output_info or {}).get("duration") or inp.get("duration") or 0),
            "frames": result.frames_processed,
            "processing_seconds": round(result.duration_s, 1),
            "finished_at": time.strftime("%Y-%m-%d %H:%M"),
            "settings": spec.enhancement.to_dict(),
            "encode": {"codec": spec.encode.codec, "quality": spec.encode.quality},
            "error": result.error,
        }
        self._recent.append(record)
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            log.warning("Could not append history: %s", exc)

    def history_records(self, limit: int = 100) -> List[dict]:
        return list(reversed(self._recent[-limit:]))

    def clear_history(self) -> None:
        self._recent.clear()
        try:
            self.history_path.unlink(missing_ok=True)
        except OSError:
            pass

    def stats(self) -> Dict:
        completed = [r for r in self._recent if r.get("status") == "completed"]
        return {
            "total": len(self._recent),
            "completed": len(completed),
            "failed": sum(1 for r in self._recent if r.get("status") == "failed"),
            "frames": sum(int(r.get("frames") or 0) for r in completed),
            "seconds": sum(float(r.get("output_duration") or 0) for r in completed),
        }

    def shutdown(self) -> None:
        """Stop the worker cleanly (called on app close)."""
        if self.worker is not None:
            self.worker.cancel()
            if not self.worker.wait(15000):
                log.warning("Worker did not stop in time")


def _fmt_duration(seconds: float) -> str:
    try:
        seconds = int(float(seconds or 0))
    except (TypeError, ValueError):
        return "-"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
