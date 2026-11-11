"""Video analysis: FFprobe metadata + thumbnail prep."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from utils.ffmpeg import FFmpeg, FFmpegError, VideoInfo
from utils.logger import get_logger

log = get_logger("analyzer")


class VideoAnalyzer:
    """Wraps FFprobe to produce a :class:`VideoInfo` plus UI helpers."""

    def __init__(self, ffmpeg: FFmpeg) -> None:
        self.ffmpeg = ffmpeg

    def analyze(self, path: str | Path) -> VideoInfo:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        info = self.ffmpeg.probe(path)
        log.info("Analyzed %s: %dx%d @ %.3f fps, %.1fs, %d frames, codec %s, "
                 "audio %s", path.name, info.width, info.height, info.fps,
                 info.duration, info.nb_frames, info.video_codec or "-",
                 info.audio_codec or "none")
        if info.nb_frames <= 0:
            log.warning("Stream reports no frame count - it will be counted "
                        "during extraction")
        return info

    def thumbnail(self, path: str | Path, out_dir: Path,
                  at: Optional[float] = None) -> Optional[Path]:
        path = Path(path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if at is None:
            # 1 s in, but never past the middle of very short clips
            try:
                info = self.ffmpeg.probe(path)
                at = min(1.0, max(0.0, info.duration * 0.25))
            except FFmpegError:
                at = 0.5
        out = out_dir / f"{path.stem}_thumb.jpg"
        return self.ffmpeg.make_thumbnail(path, out, at=at, width=480)

    def is_supported(self, path: str | Path) -> bool:
        from utils.config import SUPPORTED_VIDEO_EXTENSIONS
        return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS

    def validate(self, path: str | Path) -> Tuple[bool, str]:
        """Quick validation used before adding a job to the queue."""
        path = Path(path)
        if not path.exists():
            return False, "File does not exist"
        try:
            info = self.ffmpeg.probe(path)
        except FFmpegError as exc:
            return False, f"Cannot read video: {exc}"
        if info.width < 16 or info.height < 16:
            return False, f"Resolution too small ({info.width}x{info.height})"
        if info.duration <= 0 and info.nb_frames <= 0:
            return False, "Duration unknown and no frames reported"
        return True, "ok"
