"""Thin, well-behaved wrapper around the FFmpeg / FFprobe executables.

Everything the application does with video goes through this module:

  * locating ffmpeg/ffprobe (bundled ``bin/`` folder, config override, PATH)
  * probing media files into :class:`VideoInfo`
  * extracting frames as a PNG sequence, keeping every source frame
  * extracting the original audio stream untouched when possible
  * encoding an image sequence back to H.264/H.265 (NVENC or libx26x)
  * generating thumbnails / preview frames

The wrapper always runs FFmpeg with ``-progress pipe:1`` and parses that
machine-readable output, so progress reported to the UI is real.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .logger import get_logger

log = get_logger("ffmpeg")

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _has_nvidia_adapter() -> bool:
    """True when an NVIDIA display adapter exists on this machine.

    Uses cheap platform checks (no imports of torch, no subprocess on POSIX).
    """
    if os.name == "nt":
        try:
            res = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True, text=True, timeout=12,
                creationflags=CREATE_NO_WINDOW)
            return "nvidia" in (res.stdout or "").lower()
        except (OSError, subprocess.SubprocessError):
            return False
    # Linux / macOS: presence of the driver library is a good enough hint
    for candidate in ("/proc/driver/nvidia/version", "/dev/nvidia0",
                      "/dev/nvidiactl"):
        if os.path.exists(candidate):
            return True
    return False


class FFmpegError(RuntimeError):
    """Raised when FFmpeg / FFprobe fail."""


class FFmpegNotFoundError(FFmpegError):
    pass


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class VideoInfo:
    """Everything the UI needs to know about a source video."""

    path: str
    width: int = 0
    height: int = 0
    duration: float = 0.0
    fps: float = 0.0
    fps_raw: str = ""
    nb_frames: int = 0
    video_codec: str = ""
    video_bitrate: int = 0
    pix_fmt: str = ""
    audio_codec: str = ""
    audio_channels: int = 0
    audio_sample_rate: int = 0
    audio_bitrate: int = 0
    container: str = ""
    size_bytes: int = 0
    rotation: int = 0
    has_audio: bool = False
    color_transfer: str = ""
    color_primaries: str = ""
    is_vfr: bool = False
    field_order: str = "progressive"
    stream_index: int = 0

    # -- convenience -------------------------------------------------------
    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}" if self.width else "unknown"

    @property
    def mp(self) -> float:
        return self.width * self.height / 1_000_000.0

    @property
    def aspect(self) -> float:
        return (self.width / self.height) if self.height else 16 / 9

    @property
    def duration_seconds(self) -> float:
        if self.duration > 0:
            return self.duration
        if self.fps > 0 and self.nb_frames:
            return self.nb_frames / self.fps
        return 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "VideoInfo":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class EncoderInfo:
    name: str
    available: bool
    hardware: bool
    label: str = ""


@dataclass
class EncodeSettings:
    """Video/audio encoder configuration used when muxing the result."""

    codec: str = "h264"              # h264 | h265
    encoder: str = "auto"            # auto | nvenc | cpu
    quality: str = "high"            # low | medium | high | lossless
    crf: Optional[int] = None        # None -> derived from quality
    bitrate_kbps: int = 0            # >0 overrides CRF
    preset: str = "slow"
    pix_fmt: str = "yuv420p"
    audio_codec: str = "aac"
    audio_bitrate_kbps: int = 192
    copy_audio: bool = True
    extra_args: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Locating the binaries
# ---------------------------------------------------------------------------
def _bundled_candidates(name: str) -> List[Path]:
    exe = f"{name}.exe" if os.name == "nt" else name
    here = Path(__file__).resolve().parent.parent
    roots = [here, Path.cwd()]
    if getattr(sys, "frozen", False):
        roots.insert(0, Path(sys.executable).resolve().parent)
        roots.insert(1, Path(getattr(sys, "_MEIPASS", here)))
    cands: List[Path] = []
    for root in roots:
        cands.extend([
            root / "bin" / exe,
            root / "ffmpeg" / "bin" / exe,
            root / exe,
        ])
    return cands


class FFmpeg:
    """Facade around the two executables (module users share one instance)."""

    def __init__(self, ffmpeg_path: str = "", ffprobe_path: str = "") -> None:
        self.ffmpeg = ffmpeg_path or ""
        self.ffprobe = ffprobe_path or ""
        self._version: Optional[str] = None
        self._encoder_cache: Optional[List[str]] = None
        self._nvenc_ok: Dict[str, bool] = {}
        self._lock = threading.Lock()

    # -- discovery ---------------------------------------------------------
    def locate(self, force: bool = False) -> Tuple[str, str]:
        """Find ffmpeg & ffprobe.  Raises :class:`FFmpegNotFoundError`."""
        with self._lock:
            if self.ffmpeg and self.ffprobe and not force \
                    and Path(self.ffmpeg).exists() and Path(self.ffprobe).exists():
                return self.ffmpeg, self.ffprobe

            ffmpeg = self.ffmpeg if self.ffmpeg and Path(self.ffmpeg).exists() else None
            ffprobe = self.ffprobe if self.ffprobe and Path(self.ffprobe).exists() else None

            if not ffmpeg:
                for c in _bundled_candidates("ffmpeg"):
                    if c.exists():
                        ffmpeg = str(c)
                        break
            if not ffprobe:
                for c in _bundled_candidates("ffprobe"):
                    if c.exists():
                        ffprobe = str(c)
                        break
            if not ffmpeg:
                ffmpeg = shutil.which("ffmpeg")
            if not ffprobe:
                ffprobe = shutil.which("ffprobe")
            if not ffmpeg or not ffprobe:
                raise FFmpegNotFoundError(
                    "FFmpeg was not found.\n\n"
                    "Install it (winget install Gyan.FFmpeg) or place "
                    "ffmpeg.exe and ffprobe.exe in the application's bin/ folder, "
                    "then set the path in Settings."
                )
            self.ffmpeg, self.ffprobe = str(ffmpeg), str(ffprobe)
            log.info("FFmpeg found: %s", self.ffmpeg)
            log.info("FFprobe found: %s", self.ffprobe)
            return self.ffmpeg, self.ffprobe

    @property
    def available(self) -> bool:
        try:
            self.locate()
            return True
        except FFmpegError:
            return False

    def version(self) -> str:
        """Return e.g. ``7.1 (ffmpeg)`` or ``unknown``."""
        if self._version is not None:
            return self._version
        try:
            ffmpeg, _ = self.locate()
            out = subprocess.run([ffmpeg, "-version"], capture_output=True,
                                 text=True, creationflags=CREATE_NO_WINDOW,
                                 timeout=20).stdout
            m = re.search(r"ffmpeg version (\S+)", out)
            self._version = m.group(1) if m else "unknown"
        except (FFmpegError, OSError, subprocess.SubprocessError):
            self._version = "unknown"
        return self._version

    def _supports(self, option: str) -> bool:
        """Check whether the ffmpeg build knows a given CLI option."""
        key = f"_opt_{option}"
        cached = getattr(self, key, None)
        if cached is not None:
            return cached
        try:
            ffmpeg, _ = self.locate()
            out = subprocess.run([ffmpeg, "-hide_banner", "-h", "long"],
                                 capture_output=True, text=True,
                                 creationflags=CREATE_NO_WINDOW, timeout=20).stdout
            ok = option in out
        except (FFmpegError, OSError, subprocess.SubprocessError):
            ok = False
        setattr(self, key, ok)
        return ok

    # -- encoders ----------------------------------------------------------
    def list_encoders(self) -> List[str]:
        if self._encoder_cache is not None:
            return self._encoder_cache
        encoders: List[str] = []
        try:
            ffmpeg, _ = self.locate()
            out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                                 capture_output=True, text=True,
                                 creationflags=CREATE_NO_WINDOW, timeout=20).stdout
            for line in out.splitlines():
                m = re.match(r"\s*[VAS][.FSXBD]{0,6}\s+(\S+)", line)
                if m:
                    encoders.append(m.group(1))
        except (FFmpegError, OSError, subprocess.SubprocessError) as exc:
            log.warning("Could not list encoders: %s", exc)
        self._encoder_cache = encoders
        return encoders

    def nvenc_works(self, codec: str = "h264") -> bool:
        """Functional test: can this machine really encode with NVENC?

        Encoder presence in ``-encoders`` is not enough - the driver and GPU
        must support it.  We run a 0.1 s black-frame encode once and cache it.
        """
        if codec in self._nvenc_ok:
            return self._nvenc_ok[codec]
        name = {"h264": "h264_nvenc", "h265": "hevc_nvenc"}.get(codec, "h264_nvenc")
        ok = False
        # Cheap pre-check: without an NVIDIA adapter present there is no point
        # shelling out (and no reason to log a scary-looking driver error).
        if not _has_nvidia_adapter():
            self._nvenc_ok[codec] = False
            log.debug("NVENC skipped - no NVIDIA adapter detected")
            return False
        if name in self.list_encoders():
            try:
                ffmpeg, _ = self.locate()
                cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                       "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1:r=30",
                       "-frames:v", "2", "-c:v", name, "-f", "null", "-"]
                res = subprocess.run(cmd, capture_output=True, text=True,
                                     creationflags=CREATE_NO_WINDOW, timeout=40)
                ok = res.returncode == 0
                if not ok:
                    log.info("NVENC %s unusable: %s", name,
                             (res.stderr or "").strip().splitlines()[-1:] or "")
            except (FFmpegError, OSError, subprocess.SubprocessError) as exc:
                log.info("NVENC probe failed: %s", exc)
        self._nvenc_ok[codec] = ok
        return ok

    def available_encoders(self) -> List[EncoderInfo]:
        encs = self.list_encoders()
        out: List[EncoderInfo] = []
        for codec, cpu_name, hw_name, label in (
            ("h264", "libx264", "h264_nvenc", "H.264"),
            ("h265", "libx265", "hevc_nvenc", "H.265 / HEVC"),
        ):
            if hw_name in encs:
                out.append(EncoderInfo(hw_name, self.nvenc_works(codec), True,
                                       f"{label} (NVENC)"))
            if cpu_name in encs:
                out.append(EncoderInfo(cpu_name, True, False, f"{label} (CPU)"))
        return out

    # -- probing -----------------------------------------------------------
    def probe(self, path: str | Path) -> VideoInfo:
        """Read stream metadata with FFprobe."""
        path = Path(path)
        if not path.exists():
            raise FFmpegError(f"File not found: {path}")
        _, ffprobe = self.locate()
        cmd = [ffprobe, "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", str(path)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 creationflags=CREATE_NO_WINDOW, timeout=60)
        except subprocess.SubprocessError as exc:
            raise FFmpegError(f"FFprobe failed: {exc}") from exc
        if res.returncode != 0:
            raise FFmpegError(f"FFprobe error: {res.stderr.strip()[:400]}")
        try:
            data = json.loads(res.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise FFmpegError("FFprobe returned invalid JSON") from exc

        info = VideoInfo(path=str(path), size_bytes=path.stat().st_size)
        fmt = data.get("format", {}) or {}
        info.container = (fmt.get("format_name") or "").split(",")[0]
        try:
            info.duration = float(fmt.get("duration") or 0.0)
        except (TypeError, ValueError):
            info.duration = 0.0
        try:
            info.video_bitrate = int(float(fmt.get("bit_rate") or 0))
        except (TypeError, ValueError):
            info.video_bitrate = 0

        streams = data.get("streams", []) or []
        vstreams = [s for s in streams if s.get("codec_type") == "video"]
        astreams = [s for s in streams if s.get("codec_type") == "audio"]
        # Ignore embedded cover art (m4v / png attached_pic)
        vstreams = [s for s in vstreams
                    if not (s.get("disposition") or {}).get("attached_pic")]
        if not vstreams:
            raise FFmpegError("No video stream found in file")
        v = vstreams[0]

        info.width = int(v.get("width") or 0)
        info.height = int(v.get("height") or 0)
        info.video_codec = v.get("codec_name") or ""
        info.pix_fmt = v.get("pix_fmt") or ""
        info.stream_index = int(v.get("index") or 0)
        info.field_order = v.get("field_order") or "progressive"
        info.color_transfer = v.get("color_transfer") or ""
        info.color_primaries = v.get("color_primaries") or ""

        # Rotation from side-data (phones) - display matrix
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                try:
                    info.rotation = int(float(sd["rotation"]))
                except (TypeError, ValueError):
                    pass
        if not info.rotation:
            try:
                info.rotation = int(float((v.get("tags") or {}).get("rotate", 0)))
            except (TypeError, ValueError):
                pass

        # Frame rate
        info.fps_raw = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/0"
        info.fps = _parse_fraction(info.fps_raw)
        if info.fps <= 0:
            info.fps = _parse_fraction(v.get("r_frame_rate") or "0/0") or 25.0

        # Frame count (may be absent in some containers)
        for candidate in (v.get("nb_frames"), v.get("nb_read_frames")):
            try:
                if candidate not in (None, "N/A", ""):
                    info.nb_frames = int(float(candidate))
                    if info.nb_frames > 0:
                        break
            except (TypeError, ValueError):
                continue
        if info.nb_frames <= 0 and info.fps > 0 and info.duration > 0:
            info.nb_frames = int(round(info.duration * info.fps))

        # VFR detection: avg vs r frame rate far apart
        r = _parse_fraction(v.get("r_frame_rate") or "0/0")
        if r > 0 and info.fps > 0 and abs(r - info.fps) / max(info.fps, 1e-6) > 0.02:
            info.is_vfr = True

        if astreams:
            a = astreams[0]
            info.has_audio = True
            info.audio_codec = a.get("codec_name") or ""
            info.audio_channels = int(a.get("channels") or 0)
            info.audio_sample_rate = int(a.get("sample_rate") or 0)
            try:
                info.audio_bitrate = int(float(a.get("bit_rate") or 0))
            except (TypeError, ValueError):
                info.audio_bitrate = 0

        if info.duration <= 0 and info.fps > 0 and info.nb_frames > 0:
            info.duration = info.nb_frames / info.fps
        return info

    # -- frame extraction --------------------------------------------------
    def extract_frames(
        self,
        video: str | Path,
        out_dir: str | Path,
        progress: Optional[Callable[[int, Optional[float]], None]] = None,
        cancel: Optional[threading.Event] = None,
        start: float = 0.0,
        duration: Optional[float] = None,
        total_frames: int = 0,
        fps: Optional[float] = None,
        image_format: str = "png",
    ) -> int:
        """Extract frames to ``out_dir/%08d.png`` keeping *every* frame.

        Returns the number of frames written.  Raises :class:`FFmpegError`
        on failure and :class:`ProcessingCancelled` semantics via return when
        the cancel event is set (returns -1).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg, _ = self.locate()

        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", str(video)]
        if duration and duration > 0:
            cmd += ["-t", f"{duration:.3f}"]
        # passthrough preserves every source frame (no duplicating / dropping)
        if self._supports("fps_mode"):
            cmd += ["-fps_mode", "passthrough"]
        else:  # ffmpeg < 5.1
            cmd += ["-vsync", "0"]
        if fps:
            cmd += ["-r", f"{fps:.6f}"]
        # Full-range RGB output so AI models see the same pixels we render
        cmd += ["-pix_fmt", "rgb24", "-q:v", "1",
                str(out_dir / f"%08d.{image_format}")]

        written = {"n": 0}
        last = {"pct": -1}

        def on_progress(key: str, value: str) -> None:
            if key == "frame":
                try:
                    written["n"] = int(value)
                except ValueError:
                    pass
                if progress and total_frames > 0:
                    pct = min(100, int(written["n"] * 100 / total_frames))
                    if pct != last["pct"]:
                        last["pct"] = pct
                        progress(written["n"], pct * 100.0)
            elif key == "progress" and value == "end":
                if progress and total_frames > 0:
                    progress(written["n"], 100.0)

        self._run(cmd, on_progress, cancel, "frame extraction")
        count = len(list(out_dir.glob(f"*.{image_format}")))
        if count == 0:
            raise FFmpegError("Frame extraction produced no frames")
        log.info("Extracted %d frames -> %s", count, out_dir)
        return count

    # -- audio -------------------------------------------------------------
    def extract_audio(self, video: str | Path, out_dir: str | Path,
                      copy_stream: bool = True) -> Optional[Path]:
        """Extract the original audio stream.

        Tries a lossless *stream copy* first (keeps the exact original
        audio); falls back to PCM WAV if the container refuses.
        Returns ``None`` when the video has no audio track.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg, _ = self.locate()
        try:
            info = self.probe(video)
        except FFmpegError:
            return None
        if not info.has_audio:
            log.info("Source has no audio stream - skipping audio extraction")
            return None

        if copy_stream:
            ext = {"aac": "m4a", "mp3": "mp3", "opus": "opus", "vorbis": "ogg",
                   "flac": "flac", "pcm_s16le": "wav", "ac3": "ac3",
                   "eac3": "eac3"}.get(info.audio_codec, "mka")
            target = out_dir / f"audio_original.{ext}"
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-y", "-i", str(video), "-map", "0:a:0", "-vn", "-c:a",
                   "copy", str(target)]
            try:
                self._run(cmd, None, None, "audio extraction (copy)")
                if target.exists() and target.stat().st_size > 0:
                    log.info("Audio copied losslessly -> %s", target)
                    return target
            except FFmpegError as exc:
                log.warning("Stream-copy audio failed (%s); using WAV", exc)

        target = out_dir / "audio_original.wav"
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-i", str(video), "-map", "0:a:0", "-vn", "-c:a", "pcm_s16le",
               "-ar", str(info.audio_sample_rate or 48000), str(target)]
        self._run(cmd, None, None, "audio extraction (wav)")
        if target.exists() and target.stat().st_size > 0:
            return target
        log.warning("Audio extraction produced nothing")
        return None

    # -- encoding ----------------------------------------------------------
    def build_video(
        self,
        frames_pattern: str | Path,
        fps: float,
        output: str | Path,
        settings: Optional[EncodeSettings] = None,
        audio: Optional[str | Path] = None,
        total_frames: int = 0,
        duration: float = 0.0,
        progress: Optional[Callable[[int, Optional[float]], None]] = None,
        cancel: Optional[threading.Event] = None,
        image_format: str = "png",
    ) -> Path:
        """Encode a numbered image sequence (+ optional audio) to a video."""
        settings = settings or EncodeSettings()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg, _ = self.locate()

        codec = settings.codec.lower()
        hw = settings.encoder == "nvenc" or (
            settings.encoder == "auto" and self.nvenc_works(codec))
        if hw:
            venc = "h264_nvenc" if codec == "h264" else "hevc_nvenc"
        else:
            venc = "libx264" if codec == "h264" else "libx265"
            if venc not in self.list_encoders():
                fallback = "libx264" if "libx264" in self.list_encoders() else None
                if not fallback:
                    raise FFmpegError("No usable H.264/H.265 encoder in FFmpeg build")
                log.warning("%s missing - falling back to %s", venc, fallback)
                venc = fallback

        crf = settings.crf if settings.crf is not None else {
            "low": 26, "medium": 22, "high": 19, "lossless": 0,
        }.get(settings.quality, 19)

        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-framerate", f"{fps:.6f}",
               "-i", str(Path(frames_pattern))]
        if audio:
            cmd += ["-i", str(audio)]
            cmd += ["-map", "0:v:0", "-map", "1:a:0"]
        else:
            cmd += ["-map", "0:v:0"]

        cmd += ["-c:v", venc]
        if hw:
            cmd += ["-preset", "p5" if settings.preset in ("slow", "slower") else "p4",
                    "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
        else:
            preset = settings.preset if settings.preset in (
                "ultrafast", "superfast", "veryfast", "faster", "fast",
                "medium", "slow", "slower", "veryslow") else "medium"
            cmd += ["-preset", preset]
            if crf == 0:
                cmd += ["-qp", "0"]          # true lossless for x264/x265
            else:
                cmd += ["-crf", str(crf)]
        if settings.bitrate_kbps and settings.bitrate_kbps > 0:
            cmd += ["-maxrate", f"{settings.bitrate_kbps}k",
                    "-bufsize", f"{settings.bitrate_kbps * 2}k"]
        cmd += ["-pix_fmt", settings.pix_fmt or "yuv420p"]

        if audio:
            if settings.copy_audio and Path(str(audio)).suffix.lower() in (
                    ".m4a", ".aac", ".mp3") and output.suffix.lower() in (
                    ".mp4", ".mkv", ".mov"):
                cmd += ["-c:a", "copy"]
            else:
                cmd += ["-c:a", settings.audio_codec or "aac",
                        "-b:a", f"{settings.audio_bitrate_kbps or 192}k"]
            cmd += ["-shortest"]
        cmd += list(settings.extra_args)
        cmd.append(str(output))

        last = {"pct": -1}

        def on_progress(key: str, value: str) -> None:
            if key in ("out_time_us", "out_time_ms") and duration > 0:
                try:
                    us = float(value)
                except ValueError:
                    return
                secs = us / 1_000_000.0 if key == "out_time_us" else us / 1000.0
                pct = min(99, max(0, int(secs * 100 / duration)))
                if progress and pct != last["pct"]:
                    last["pct"] = pct
                    frames = int(secs * fps)
                    progress(frames, pct * 100.0)

        self._run(cmd, on_progress, cancel, "video encoding")
        if not output.exists() or output.stat().st_size == 0:
            raise FFmpegError("Encoder produced an empty file")
        if progress:
            progress(total_frames or 0, 100.0)
        log.info("Rendered %s (%s, %.1f MB)", output, venc,
                 output.stat().st_size / 1048576)
        return output

    # -- misc helpers ------------------------------------------------------
    def make_thumbnail(self, video: str | Path, out_path: str | Path,
                       at: float = 1.0, width: int = 480) -> Optional[Path]:
        """Save a single preview frame as JPEG."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ffmpeg, _ = self.locate()
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-y", "-ss", f"{max(0.0, at):.2f}", "-i", str(video),
                   "-frames:v", "1", "-vf", f"scale={width}:-2",
                   "-q:v", "3", str(out_path)]
            self._run(cmd, None, None, "thumbnail")
            return out_path if out_path.exists() else None
        except FFmpegError as exc:
            log.warning("Thumbnail failed: %s", exc)
            return None

    def grab_frames(self, video: str | Path, out_dir: str | Path,
                    times: List[float], width: int = 640) -> List[Path]:
        """Extract frames at specific timestamps (used by the comparison view)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result: List[Path] = []
        ffmpeg, _ = self.locate()
        for i, t in enumerate(times):
            target = out_dir / f"grab_{i:03d}.jpg"
            try:
                cmd = [ffmpeg, "-hide_banner", "-loglevel", "error",
                       "-nostdin", "-y", "-ss", f"{max(0.0, t):.3f}",
                       "-i", str(video), "-frames:v", "1",
                       "-vf", f"scale={width}:-2", "-q:v", "3", str(target)]
                self._run(cmd, None, None, "frame grab")
                if target.exists():
                    result.append(target)
            except FFmpegError as exc:
                log.debug("grab_frames failed at %.2fs: %s", t, exc)
        return result

    def count_frames(self, video: str | Path) -> int:
        """Count *decoded* frames - accurate but linear.  Used as a fallback."""
        _, ffprobe = self.locate()
        cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
               "-count_frames", "-show_entries", "stream=nb_read_frames",
               "-of", "csv=p=0", str(video)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 creationflags=CREATE_NO_WINDOW, timeout=600)
            return int((res.stdout or "0").strip().split(",")[0] or 0)
        except (subprocess.SubprocessError, ValueError):
            return 0

    # -- process plumbing --------------------------------------------------
    def _run(self, cmd: List[str], progress_cb, cancel: Optional[threading.Event],
             stage: str = "") -> None:
        """Run FFmpeg, stream ``-progress`` output, support cancellation."""
        full = [cmd[0], "-progress", "pipe:1", "-nostats"] + cmd[1:]
        log.debug("FFmpeg command: %s", " ".join(full))
        try:
            proc = subprocess.Popen(
                full, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, creationflags=CREATE_NO_WINDOW)
        except OSError as exc:
            raise FFmpegError(f"Could not start FFmpeg: {exc}") from exc

        stderr_tail: List[str] = []

        def drain_stderr() -> None:
            try:
                for line in proc.stderr:  # type: ignore[union-attr]
                    line = line.rstrip()
                    if line:
                        stderr_tail.append(line)
                        if len(stderr_tail) > 60:
                            del stderr_tail[0]
            except (ValueError, OSError):
                pass

        t = threading.Thread(target=drain_stderr, daemon=True)
        t.start()

        cancelled = False
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    proc.terminate()
                    break
                line = line.strip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if progress_cb:
                    try:
                        progress_cb(key.strip(), value.strip())
                    except Exception:  # progress callback must not kill encode
                        log.exception("progress callback error")
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
        finally:
            t.join(timeout=5)
            if proc.poll() is None:
                proc.kill()

        if cancelled:
            raise FFmpegError(f"Cancelled during {stage}")
        if proc.returncode not in (0, None):
            tail = "\n".join(stderr_tail[-12:])
            raise FFmpegError(f"FFmpeg failed during {stage} (code {proc.returncode}):\n{tail}")


# ---------------------------------------------------------------------------
def _parse_fraction(text: str) -> float:
    try:
        if not text or text in ("0/0", "N/A"):
            return 0.0
        if "/" in text:
            num, den = text.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(text)
    except (ValueError, ZeroDivisionError):
        return 0.0


_SINGLETON: Optional[FFmpeg] = None


def get_ffmpeg(ffmpeg_path: str = "", ffprobe_path: str = "") -> FFmpeg:
    """Process-wide FFmpeg facade."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = FFmpeg(ffmpeg_path, ffprobe_path)
    elif ffmpeg_path or ffprobe_path:
        if ffmpeg_path:
            _SINGLETON.ffmpeg = ffmpeg_path
        if ffprobe_path:
            _SINGLETON.ffprobe = ffprobe_path
    return _SINGLETON
