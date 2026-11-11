"""Filesystem helpers: temp dirs, safe cleanup, paths, human-readable sizes.

Rules enforced here (see development rules):
  * original videos are never touched - we only ever read them;
  * temporary files live inside a job-scoped folder and are removed safely;
  * cleanup never follows symlinks outside known roots.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, Optional

from .logger import get_logger

log = get_logger("file_manager")

VIDEO_EXTENSIONS = (
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg", ".webm", ".flv",
)

RESOLUTION_PRESETS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def human_size(num_bytes: float) -> str:
    """Format a byte count as e.g. ``1.4 GB``."""
    try:
        num_bytes = float(num_bytes or 0)
    except (TypeError, ValueError):
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(num_bytes)} B"
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"  # pragma: no cover


def human_duration(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS`` / ``MM:SS``."""
    try:
        seconds = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "00:00"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def human_eta(seconds: Optional[float]) -> str:
    return "--:--" if seconds is None else human_duration(seconds)


def is_video_file(path: Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def unique_path(directory: Path, stem: str, suffix: str = ".mp4") -> Path:
    """Return a non-existing path ``dir/stem.mp4``, ``dir/stem_1.mp4``, ..."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}{suffix}"
    n = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{n}{suffix}"
        n += 1
    return candidate


# ---------------------------------------------------------------------------
# Temporary workspace management
# ---------------------------------------------------------------------------
class JobWorkspace:
    """A sandboxed temp folder owned by one processing job.

    Layout::

        <temp_dir>/<job_id>/
            frames_in/      extracted source frames (only when needed)
            frames_out/     processed frames (written by the renderer)
            audio/          extracted audio stream
            work/           mux outputs, resume state

    ``cleanup()`` removes the folder but can be told to keep it for debugging.
    """

    def __init__(self, temp_root: Path, job_id: Optional[str] = None,
                 keep_frames: bool = False) -> None:
        self.temp_root = Path(temp_root)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.job_id = job_id or f"job_{uuid.uuid4().hex[:10]}"
        self.root = self.temp_root / self.job_id
        self.frames_in = self.root / "frames_in"
        self.frames_out = self.root / "frames_out"
        self.audio_dir = self.root / "audio"
        self.work_dir = self.root / "work"
        self.keep_frames = keep_frames
        for d in (self.root, self.frames_in, self.frames_out,
                  self.audio_dir, self.work_dir):
            d.mkdir(parents=True, exist_ok=True)
        log.debug("Job workspace created: %s", self.root)

    # -- state helpers -----------------------------------------------------
    def state_file(self, name: str) -> Path:
        return self.work_dir / name

    def disk_usage_mb(self) -> float:
        total = 0
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for f in filenames:
                try:
                    total += (Path(dirpath) / f).stat().st_size
                except OSError:
                    pass
        return total / (1024 * 1024)

    def cleanup(self, keep: Optional[bool] = None) -> None:
        """Delete the workspace.  Never raises."""
        keep = self.keep_frames if keep is None else keep
        if keep:
            log.info("Keeping job workspace for inspection: %s", self.root)
            return
        # Guard: only delete folders that live inside temp_root
        try:
            self.root.resolve().relative_to(self.temp_root.resolve())
        except ValueError:  # pragma: no cover - defensive
            log.error("Refusing to delete %s (outside temp root)", self.root)
            return
        shutil.rmtree(self.root, ignore_errors=True)
        log.debug("Job workspace removed: %s", self.root)


def ensure_free_space(path: Path, min_free_gb: float = 2.0) -> bool:
    """Check that ``path``'s volume has at least ``min_free_gb`` gigabytes."""
    try:
        usage = shutil.disk_usage(str(path))
    except OSError:
        return True  # cannot determine -> do not block the user
    free_gb = usage.free / (1024 ** 3)
    return free_gb >= float(min_free_gb)


def free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(str(path)).free / (1024 ** 3)
    except OSError:
        return 0.0


def sweep_stale_temp(temp_root: Path, max_age_hours: float = 48.0) -> int:
    """Delete leftover ``job_*`` folders older than ``max_age_hours``.

    Returns the number of folders removed.  Called once at start-up so a
    crashed session does not fill the disk forever.
    """
    import time

    temp_root = Path(temp_root)
    if not temp_root.exists():
        return 0
    removed = 0
    cutoff = time.time() - max_age_hours * 3600
    for entry in temp_root.iterdir():
        if not entry.is_dir() or not entry.name.startswith("job_"):
            continue
        try:
            latest = max((p.stat().st_mtime for p in entry.rglob("*")),
                         default=entry.stat().st_mtime)
        except OSError:
            continue
        if latest < cutoff:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    if removed:
        log.info("Swept %d stale temp workspace(s)", removed)
    return removed


def list_videos(folder: Path) -> Iterable[Path]:
    """Yield supported video files directly inside ``folder``."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        (p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )


def make_temp_file(prefix: str = "rv_", suffix: str = ".tmp") -> Path:
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return Path(name)
