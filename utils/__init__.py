"""Shared utilities: config, logging, ffmpeg wrapper, GPU telemetry, files."""
from .config import Config, DEFAULTS, TARGET_RESOLUTIONS, RESOLUTION_LABELS, \
    INTENSITIES, SUPPORTED_VIDEO_EXTENSIONS, app_root
from .file_manager import (JobWorkspace, ensure_free_space, free_gb, human_duration,
                           human_eta, human_size, is_video_file, sweep_stale_temp,
                           unique_path)
from .ffmpeg import (FFmpeg, FFmpegError, FFmpegNotFoundError, EncodeSettings,
                     VideoInfo, get_ffmpeg)
from .logger import get_logger, setup_logging, add_listener, remove_listener, \
    recent_lines
from .gpu_monitor import (GPUStats, SystemStats, detect_gpus, query_gpu,
                          system_stats, recommend_tile_size,
                          recommend_batch_frames)

__all__ = [
    "Config", "DEFAULTS", "TARGET_RESOLUTIONS", "RESOLUTION_LABELS",
    "INTENSITIES", "SUPPORTED_VIDEO_EXTENSIONS", "app_root",
    "JobWorkspace", "ensure_free_space", "free_gb", "human_duration",
    "human_eta", "human_size", "is_video_file", "sweep_stale_temp", "unique_path",
    "FFmpeg", "FFmpegError", "FFmpegNotFoundError", "EncodeSettings", "VideoInfo",
    "get_ffmpeg",
    "get_logger", "setup_logging", "add_listener", "remove_listener",
    "recent_lines",
    "GPUStats", "SystemStats", "detect_gpus", "query_gpu", "system_stats",
    "recommend_tile_size", "recommend_batch_frames",
]
