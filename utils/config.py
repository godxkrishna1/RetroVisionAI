"""Configuration loading / saving for RetroVision AI.

The app keeps a single JSON configuration file (``config.json`` next to the
executable during development, or next to the EXE when frozen).  Missing keys
are filled from :data:`DEFAULTS` so older config files keep working after an
upgrade.
"""
from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------
# Defaults (mirrors config.json - that file is the human-editable master)
# --------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    "app": {
        "name": "RetroVision AI",
        "version": "1.0.0",
        "first_run_complete": False,
        "theme": "dark",
        "language": "en",
    },
    "paths": {
        "models_dir": "models",
        "output_dir": "output",
        "temp_dir": "temp",
        "logs_dir": "logs",
        "ffmpeg_path": "",
        "ffprobe_path": "",
    },
    "processing": {
        "device": "auto",            # auto | cuda | cpu
        "tile_size": 0,              # 0 = auto (VRAM aware)
        "tile_pad": 16,
        "precision": "fp16",         # fp16 | fp32
        "cpu_threads": 0,            # 0 = auto
        "keep_temp_frames": False,
        "resume_enabled": True,
        "max_batch_frames": 8,
        "vram_headroom_mb": 1024,
        "temp_drive_min_free_gb": 2.0,
    },
    "enhancement": {
        "target_resolution": "1080p",     # original|720p|1080p|2k|4k
        "ai_model": "realesrgan_x2plus",
        "use_super_resolution": True,
        "use_face_restoration": False,
        "face_strength": 0.8,
        "use_denoise": True,
        "denoise_strength": "medium",
        "use_deblur": False,
        "deblur_strength": "medium",
        "use_sharpen": False,
        "sharpen_strength": "medium",
        "use_color_restore": True,
        "color_strength": "medium",
        "use_brightness_fix": False,
        "brightness_strength": "medium",
        "use_contrast_fix": False,
        "contrast_strength": "medium",
        "use_grain_reduction": False,
        "grain_strength": "medium",
        "use_scratch_reduction": False,
        "scratch_strength": "medium",
        "use_interpolation": False,
        "interpolation_factor": 2,
        "interpolation_model": "rife47",
        "use_temporal_restore": False,
        "temporal_window": 5,
    },
    "export": {
        "codec": "h264",             # h264 | h265
        "encoder": "auto",           # auto | nvenc | cpu
        "quality": "high",           # low | medium | high | lossless
        "crf": 18,
        "bitrate_kbps": 0,           # 0 = use CRF
        "preset": "slow",
        "pixel_format": "yuv420p",
        "audio_codec": "aac",
        "audio_bitrate_kbps": 192,
        "copy_audio_when_possible": True,
        "container": "mp4",
    },
    "ui": {
        "window_width": 1440,
        "window_height": 900,
        "sidebar_collapsed": False,
        "show_welcome_on_start": True,
        "recent_projects_limit": 8,
    },
    "gpu": {"monitor_interval_ms": 1000, "enable_cuda_memory_pool": True},
    "downloads": {"retries": 3, "chunk_size_kb": 256, "timeout_seconds": 30},
}

SUPPORTED_VIDEO_EXTENSIONS = (
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg", ".webm", ".flv",
)

TARGET_RESOLUTIONS = {
    "original": None,
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
}

RESOLUTION_LABELS = {
    "original": "Original",
    "720p": "720p HD",
    "1080p": "1080p Full HD",
    "2k": "2K QHD",
    "4k": "4K UHD",
}

INTENSITIES = ("low", "medium", "high")


def app_root() -> Path:
    """Directory that holds the application (script dir or PyInstaller dir)."""
    import sys

    if getattr(sys, "frozen", False):  # PyInstaller
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``base`` deep-merged with ``override`` (override wins)."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Thread-safe application configuration backed by a JSON file."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.root = app_root()
        self.path = Path(path) if path else self.root / "config.json"
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = copy.deepcopy(DEFAULTS)
        self.load()

    # -- persistence -------------------------------------------------------
    def load(self) -> None:
        with self._lock:
            if self.path.exists():
                try:
                    with open(self.path, "r", encoding="utf-8") as fh:
                        user = json.load(fh)
                    if not isinstance(user, dict):
                        raise ValueError("config root must be an object")
                    self._data = _deep_merge(DEFAULTS, user)
                except (json.JSONDecodeError, OSError, ValueError):
                    # Corrupt config: keep defaults, but preserve the bad file
                    try:
                        self.path.replace(self.path.with_suffix(".json.bak"))
                    except OSError:
                        pass
                    self._data = copy.deepcopy(DEFAULTS)
            else:
                self._data = copy.deepcopy(DEFAULTS)
                self.save()

    def save(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".json.tmp")
            try:
                tmp.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(self._data, fh, indent=2)
                os.replace(tmp, self.path)  # atomic on Windows & POSIX
            except OSError:
                # Never crash the UI because the config could not be written.
                pass

    # -- generic access ----------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        """Read a value with a dotted key, e.g. ``get("export.crf")``."""
        with self._lock:
            node: Any = self._data
            for part in dotted.split("."):
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return copy.deepcopy(node)

    def set(self, dotted: str, value: Any, autosave: bool = True) -> None:
        with self._lock:
            parts = dotted.split(".")
            node = self._data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
                if not isinstance(node, dict):  # pragma: no cover - defensive
                    raise TypeError(f"config path {dotted!r} is not a mapping")
            node[parts[-1]] = value
            if autosave:
                self.save()

    def update(self, mapping: Dict[str, Any], autosave: bool = True) -> None:
        with self._lock:
            self._data = _deep_merge(self._data, mapping)
            if autosave:
                self.save()

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    def reset(self, autosave: bool = True) -> None:
        with self._lock:
            self._data = copy.deepcopy(DEFAULTS)
            if autosave:
                self.save()

    # -- resolved paths ----------------------------------------------------
    def resolve_path(self, dotted: str) -> Path:
        raw = str(self.get(dotted, "") or "")
        p = Path(raw) if raw else Path(DEFAULTS["paths"][dotted.split(".")[-1]])
        return p if p.is_absolute() else (self.root / p)

    def ensure_dirs(self) -> None:
        """Create every configured folder (idempotent)."""
        for d in (self.models_dir, self.output_dir, self.temp_dir, self.logs_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

    @property
    def models_dir(self) -> Path:
        return self.resolve_path("paths.models_dir")

    @property
    def output_dir(self) -> Path:
        return self.resolve_path("paths.output_dir")

    @property
    def temp_dir(self) -> Path:
        return self.resolve_path("paths.temp_dir")

    @property
    def logs_dir(self) -> Path:
        return self.resolve_path("paths.logs_dir")
