"""Application context: one place that owns long-lived services.

Passed to every page so they can reach the config, FFmpeg facade, model
manager and job manager without globals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ai.device_manager import DeviceManager
from ai.model_manager import ModelManager
from core.job_manager import JobManager
from utils.config import Config
from utils.ffmpeg import FFmpeg
from utils.logger import get_logger

log = get_logger("context")


class AppContext:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.ffmpeg = FFmpeg(
            config.get("paths.ffmpeg_path", ""),
            config.get("paths.ffprobe_path", ""))
        self.model_manager = ModelManager(config.models_dir)
        self.device = DeviceManager.instance(
            config.get("processing.device", "auto"),
            config.get("processing.precision", "fp16"))

        history_path = config.root / "logs" / "history.jsonl"
        self.job_manager = JobManager(self, history_path)

        # page-to-page coordination (e.g. "open this file in the enhancer")
        self.pending_video: Optional[str] = None
        self.last_output: Optional[str] = None

    # -- convenience ---------------------------------------------------
    def ensure_dirs(self) -> None:
        for d in (self.config.models_dir, self.config.output_dir,
                  self.config.temp_dir, self.config.logs_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

    def ffmpeg_status(self) -> tuple[bool, str]:
        try:
            self.ffmpeg.locate()
            enc = self.ffmpeg.available_encoders()
            hw = [e.name for e in enc if e.hardware and e.available]
            return True, (f"FFmpeg {self.ffmpeg.version()}" +
                          (f"  ·  NVENC: {', '.join(hw)}" if hw else "  ·  CPU encoding"))
        except Exception as exc:
            return False, str(exc)

    def shutdown(self) -> None:
        try:
            self.job_manager.shutdown()
        except Exception:
            pass
        self.config.save()
        log.info("Application context shut down")
