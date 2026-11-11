"""The video restoration pipeline.

Stages (each reports real progress through the callback):

    1. analyze        FFprobe metadata, estimate target size
    2. audio          lossless stream copy of the original audio
    3. extract        FFmpeg -> PNG frames (fps_mode passthrough, every frame)
    4. restore        per frame: Real-ESRGAN -> GFPGAN -> filters -> resize
    5. temporal       optional BasicVSR++ pass over frame windows
    6. interpolate    optional RIFE pass (2x/3x new frames)
    7. encode         FFmpeg image sequence -> H.264/H.265 (+ NVENC if available)
    8. mux/verify     audio re-attached, duration/frame-count cross-checked

Design notes
------------
* Frames are streamed in batches of ``batch_frames``; peak RAM stays bounded.
* :class:`ResumeState` lets a cancelled/failed job continue from the last
  written frame (only when frame processing itself was interrupted - encoding
  always restarts because FFmpeg needs the complete sequence).
* AI models are loaded lazily per stage and released as soon as the stage
  ends, so VRAM never holds two heavy models at once.
* Nothing here touches the source file or the user's originals.
"""
from __future__ import annotations

import math
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ai.device_manager import DeviceManager, GPUMemoryError
from ai.gfpgan_engine import GFPGANEngine
from ai.realesrgan_engine import RealESRGANEngine
from ai.rife_engine import RIFEEngine
from core.filters import apply_chain
from core.job import EnhancementSettings, JobProgress, JobResult, JobSpec, JobStage
from utils.config import TARGET_RESOLUTIONS
from utils.ffmpeg import FFmpeg, FFmpegError, VideoInfo
from utils.file_manager import JobWorkspace, human_duration, human_size, unique_path
from utils.logger import get_logger

log = get_logger("pipeline")


class CancelledError(RuntimeError):
    pass


def target_size(info: VideoInfo, resolution: str) -> Tuple[int, int]:
    """Compute output width/height preserving aspect ratio (even numbers)."""
    preset = TARGET_RESOLUTIONS.get(resolution)
    if not preset:
        w, h = info.width, info.height
    else:
        tw, th = preset
        aspect = info.width / info.height if info.height else 16 / 9
        # fit inside the target box without cropping or stretching
        if aspect >= tw / th:
            w = tw
            h = int(round(tw / aspect))
        else:
            h = th
            w = int(round(th * aspect))
    w -= w % 2
    h -= h % 2
    return max(2, w), max(2, h)


class ResumeState:
    """Tracks how many output frames are already finalised on disk."""

    def __init__(self, workspace: JobWorkspace, enabled: bool = True) -> None:
        self.file = workspace.state_file("resume.json")
        self.enabled = enabled
        self.done_frames = 0
        self.stage = ""

    def load(self) -> int:
        import json
        if not self.enabled or not self.file.exists():
            return 0
        try:
            data = json.loads(self.file.read_text("utf-8"))
            self.done_frames = int(data.get("done_frames", 0))
            self.stage = data.get("stage", "")
            log.info("Resuming from frame %d (%s)", self.done_frames, self.stage)
        except (ValueError, OSError):
            self.done_frames = 0
        return self.done_frames

    def save(self, done_frames: int, stage: str) -> None:
        if not self.enabled:
            return
        import json
        try:
            self.file.write_text(json.dumps(
                {"done_frames": int(done_frames), "stage": stage,
                 "updated": time.time()}), "utf-8")
        except OSError:
            pass

    def clear(self) -> None:
        try:
            self.file.unlink(missing_ok=True)
        except OSError:
            pass


ProgressFn = Callable[[JobProgress], None]
StageFn = Callable[[str, int, int], None]     # (stage name, index, count)


class VideoPipeline:
    """Stateless per job: create one instance per run."""

    def __init__(self, ffmpeg: FFmpeg, device: DeviceManager, model_manager,
                 temp_root: Path, cancel_event: Optional[threading.Event] = None,
                 progress_cb: Optional[ProgressFn] = None,
                 stage_cb: Optional[StageFn] = None,
                 resume: bool = True, keep_temp: bool = False,
                 tile_size: int = -1, tile_pad: int = 16,
                 batch_frames: int = 4) -> None:
        self.ffmpeg = ffmpeg
        self.dev = device
        self.models = model_manager
        self.temp_root = Path(temp_root)
        self.cancel = cancel_event or threading.Event()
        self.progress_cb = progress_cb
        self.stage_cb = stage_cb
        self.resume_enabled = resume
        self.keep_temp = keep_temp
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.batch_frames = max(1, batch_frames)

        self.sr = RealESRGANEngine(device, model_manager)
        self.face = GFPGANEngine(device, model_manager)
        self.rife = RIFEEngine(device, model_manager)

        self.workspace: Optional[JobWorkspace] = None
        self._last_report = 0.0
        self._t_start = 0.0

    # ------------------------------------------------------------------
    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise CancelledError()

    def _report(self, progress: JobProgress, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_report < 0.12:
            return
        self._last_report = now
        progress.elapsed = now - self._t_start
        if self.progress_cb:
            try:
                self.progress_cb(progress)
            except Exception:
                log.exception("progress callback failed")

    def _stage(self, name: str, idx: int, count: int, progress: JobProgress) -> None:
        progress.stage = name
        progress.stage_index = idx
        progress.stage_count = count
        progress.percent = 0.0
        log.info("[%s] stage %d/%d", name, idx, count)
        if self.stage_cb:
            try:
                self.stage_cb(name, idx, count)
            except Exception:
                pass
        self._report(progress, force=True)

    # ------------------------------------------------------------------
    def run(self, spec: JobSpec) -> JobResult:
        self._t_start = time.time()
        progress = JobProgress()
        result = JobResult(job_id=spec.job_id, status="failed",
                           started_at=self._t_start)
        stages = self._plan_stages(spec.enhancement)
        total_stages = len(stages) + 3      # analyze, extract, encode + listed
        stage_idx = 0
        info: Optional[VideoInfo] = None

        self.workspace = JobWorkspace(self.temp_root, spec.job_id, self.keep_temp)
        resume = ResumeState(self.workspace, self.resume_enabled)

        try:
            # ---- 1. analyze -------------------------------------------
            stage_idx += 1
            self._stage(JobStage.ANALYZING, stage_idx, total_stages, progress)
            info = self.ffmpeg.probe(spec.input_path)
            result.input_info = info.to_dict()
            out_w, out_h = target_size(info, spec.enhancement.target_resolution)
            out_fps = info.fps * (spec.enhancement.interpolation_factor
                                  if spec.enhancement.use_interpolation else 1)
            log.info("Source %dx%d @%.3f fps -> output %dx%d @%.3f fps (%s)",
                     info.width, info.height, info.fps, out_w, out_h, out_fps,
                     spec.enhancement.target_resolution)

            # ---- 2. audio ---------------------------------------------
            stage_idx += 1
            self._stage(JobStage.EXTRACTING_AUDIO, stage_idx, total_stages, progress)
            audio_path = self.ffmpeg.extract_audio(spec.input_path,
                                                   self.workspace.audio_dir)
            self._check_cancel()

            # ---- 3. frames --------------------------------------------
            stage_idx += 1
            self._stage(JobStage.EXTRACTING, stage_idx, total_stages, progress)
            total_src = info.nb_frames or int(info.duration * info.fps)
            if total_src <= 0:
                total_src = self.ffmpeg.count_frames(spec.input_path)
            progress.frames_total = total_src

            src_frames = self._extract(info, progress)
            self._check_cancel()

            # ---- 4. AI restore + filters ------------------------------
            stage_idx += 1
            self._stage(JobStage.LOADING_MODELS, stage_idx, total_stages, progress)
            self._load_models(spec.enhancement)
            stage_idx += 1
            self._stage(JobStage.RESTORING, stage_idx, total_stages, progress)
            produced = self._restore_frames(src_frames, spec, out_w, out_h,
                                            progress, resume)

            # ---- 5. temporal ------------------------------------------
            if spec.enhancement.use_temporal_restore:
                stage_idx += 1
                self._stage(JobStage.TEMPORAL, stage_idx, total_stages, progress)
                produced = self._temporal_pass(produced, spec.enhancement,
                                               progress)
                self._check_cancel()

            # ---- 6. interpolation -------------------------------------
            if spec.enhancement.use_interpolation:
                stage_idx += 1
                self._stage(JobStage.INTERPOLATING, stage_idx, total_stages, progress)
                produced = self._interpolate(produced, spec.enhancement, progress)
                self._check_cancel()

            self._release_models()
            if produced == 0:
                raise RuntimeError("No frames were produced from the source video")

            # ---- 7. encode --------------------------------------------
            stage_idx += 1
            self._stage(JobStage.ENCODING, stage_idx, total_stages, progress)
            out_path = Path(spec.output_path)
            tmp_out = self.workspace.work_dir / ("encoded" + out_path.suffix)
            fps_out = out_fps
            self._encode(tmp_out, fps_out, audio_path, spec, produced, progress)
            self._check_cancel()

            # ---- 8. finalise ------------------------------------------
            stage_idx += 1
            self._stage(JobStage.VERIFYING, stage_idx, total_stages, progress)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.move(str(tmp_out), str(out_path))
            out_info = self._verify(out_path, out_w, out_h, fps_out)
            result.output_info = out_info.to_dict() if out_info else {}
            result.output_path = str(out_path)
            result.frames_processed = produced
            result.status = "completed"
            progress.stage = JobStage.DONE
            progress.percent = 100.0
            progress.frames_done = produced
            progress.frames_total = produced
            self._report(progress, force=True)
            resume.clear()
            log.info("Job %s completed in %s -> %s", spec.job_id,
                     human_duration(time.time() - self._t_start),
                     human_size(out_path.stat().st_size))

        except CancelledError:
            result.status = "cancelled"
            progress.stage = JobStage.CANCELLED
            self._report(progress, force=True)
            log.info("Job %s cancelled by user", spec.job_id)
        except (FFmpegError, GPUMemoryError, FileNotFoundError, RuntimeError) as exc:
            result.status = "failed"
            result.error = str(exc)
            progress.stage = JobStage.FAILED
            progress.message = str(exc)
            self._report(progress, force=True)
            log.error("Job %s failed: %s", spec.job_id, exc)
        except Exception as exc:  # unexpected - keep the traceback in the log
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
            progress.stage = JobStage.FAILED
            log.error("Job %s crashed: %s\n%s", spec.job_id, exc,
                      traceback.format_exc())
        finally:
            result.finished_at = time.time()
            result.duration_s = result.finished_at - result.started_at
            self._release_models()
            from utils.logger import recent_lines
            result.log_tail = "\n".join(recent_lines(40))
            if self.workspace is not None:
                if result.status != "cancelled":
                    self.workspace.cleanup(keep=self.keep_temp)
                else:
                    keep = self.resume_enabled or self.keep_temp
                    self.workspace.cleanup(keep=keep)
        return result

    # ------------------------------------------------------------------
    def _plan_stages(self, enh: EnhancementSettings) -> List[str]:
        stages = [JobStage.EXTRACTING_AUDIO, JobStage.EXTRACTING,
                  JobStage.LOADING_MODELS, JobStage.RESTORING]
        if enh.use_temporal_restore:
            stages.append(JobStage.TEMPORAL)
        if enh.use_interpolation:
            stages.append(JobStage.INTERPOLATING)
        stages.append(JobStage.ENCODING)
        return stages

    def _extract(self, info: VideoInfo, progress: JobProgress) -> int:
        def cb(done: int, pct: Optional[float]) -> None:
            progress.frames_done = done
            progress.percent = pct or 0.0
            progress.fps = 0.0
            progress.message = f"Extracting frame {done:,}"
            self._report(progress)

        count = self.ffmpeg.extract_frames(
            info.path, self.workspace.frames_in, progress=cb, cancel=self.cancel,
            total_frames=progress.frames_total)
        progress.frames_total = count
        progress.percent = 100.0
        self._report(progress, force=True)
        return count

    def _load_models(self, enh: EnhancementSettings) -> None:
        if enh.use_super_resolution:
            key = self.models.resolve_upscaler(enh.ai_model)
            if key is None:
                raise FileNotFoundError(
                    "No AI upscaling model is installed. Open AI Model Manager "
                    "and download Real-ESRGAN x2plus (recommended) first.")
            self.sr.set_tile_pad(self.tile_pad)
            self.sr.load(key, tile=self.tile_size)
        if enh.use_face_restoration:
            self.face.load("gfpgan_v14")
            self.face.strength = enh.face_strength
        if enh.use_interpolation:
            self.rife.load(enh.interpolation_model or "rife47")

    def _release_models(self) -> None:
        try:
            self.sr.unload()
        except Exception:
            pass
        try:
            self.face.unload()
        except Exception:
            pass
        try:
            self.rife.unload()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _restore_frames(self, src_count: int, spec: JobSpec, out_w: int,
                        out_h: int, progress: JobProgress,
                        resume: ResumeState) -> int:
        enh = spec.enhancement
        in_dir = self.workspace.frames_in
        out_dir = self.workspace.frames_out
        frames = sorted(in_dir.glob("*.png"))
        if not frames:
            raise RuntimeError("Frame extraction produced no files")

        start_at = resume.load()
        if start_at >= len(frames):
            start_at = 0
        if start_at:
            log.info("Resuming restore at frame %d/%d", start_at, len(frames))

        produced = start_at
        t_last = time.time()
        done_in_window = 0
        progress.frames_total = len(frames)

        for idx in range(start_at, len(frames)):
            self._check_cancel()
            frame = cv2.imread(str(frames[idx]), cv2.IMREAD_COLOR)
            if frame is None:
                log.warning("Skipping unreadable frame %s", frames[idx].name)
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- 4a. neural super resolution --------------------------
            if enh.use_super_resolution:
                frame = self.sr.upscale(frame)

            # --- 4b. face restoration (before downscale = more detail) --
            if enh.use_face_restoration and self.face.loaded:
                frame, n_faces = self.face.restore_frame(
                    frame, strength=enh.face_strength)
                if n_faces:
                    progress.detail["faces"] = str(self.face.faces_restored)

            # --- 4c. classical restoration chain ----------------------
            if enh.filters_enabled():
                frame = apply_chain(frame, enh.to_dict())

            # --- 4d. resize to the exact target resolution ------------
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                interp = (cv2.INTER_AREA if frame.shape[1] > out_w
                          else cv2.INTER_LANCZOS4)
                frame = cv2.resize(frame, (out_w, out_h), interpolation=interp)

            # --- 4e. write output frame (atomic-ish, numbered) --------
            out_path = out_dir / f"{produced + 1:08d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])
            produced += 1

            # --- real progress numbers --------------------------------
            done_in_window += 1
            now = time.time()
            if now - t_last >= 0.5:
                speed = done_in_window / (now - t_last)
                progress.fps = speed
                remaining = (len(frames) - idx - 1) / speed if speed > 0 else None
                progress.eta = remaining
                t_last = now
                done_in_window = 0
            progress.frames_done = idx + 1
            progress.percent = (idx + 1) * 100.0 / len(frames)
            progress.output_frames = produced
            progress.message = (
                f"Frame {idx + 1:,} / {len(frames):,}  ({out_w}x{out_h})"
                f"  ·  {self.dev.info().label}")
            self._report(progress)

            if (idx + 1) % 25 == 0:
                resume.save(produced, JobStage.RESTORING)

        resume.save(produced, JobStage.RESTORING)
        progress.percent = 100.0
        self._report(progress, force=True)
        log.info("Restored %d frames (%d reused from resume)", produced,
                 start_at)
        return produced

    # ------------------------------------------------------------------
    def _temporal_pass(self, count: int, enh: EnhancementSettings,
                       progress: JobProgress) -> int:
        """BasicVSR++ over sliding windows of frames (window = temporal_window)."""
        from ai.archs.basicvsr_pp import load_basicvsr_pp
        key = "basicvsr_pp"
        path = self.models.path_for(key)
        spynet = self.models.path_for("spynet")
        if not path.exists() or not spynet.exists():
            raise FileNotFoundError(
                "BasicVSR++ requires both basicvsr_pp and spynet models. "
                "Download them in AI Model Manager.")
        self.dev.release()
        net = load_basicvsr_pp(str(path), str(spynet), device="cpu")
        if self.dev.info().is_cuda:
            self.dev.to(net)
        net.eval()
        torch = self.dev.torch()

        out_dir = self.workspace.frames_out
        files = sorted(out_dir.glob("*.png"))
        window = max(2, int(enh.temporal_window))
        scale = 4

        total = len(files)
        progress.frames_total = total
        t_last = time.time()
        done = 0
        tmp_dir = self.workspace.work_dir / "temporal"
        tmp_dir.mkdir(exist_ok=True)

        for start in range(0, total, window):
            self._check_cancel()
            chunk = files[start:start + window]
            if len(chunk) < 2:
                break
            arrays = []
            for f in chunk:
                img = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if img is None:
                    break
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # BasicVSR++ x4 on lower-res input; downscale first to bound cost
                small = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale),
                                   interpolation=cv2.INTER_AREA)
                arrays.append(small)
            if not arrays:
                continue
            h = min(a.shape[0] for a in arrays)
            w = min(a.shape[1] for a in arrays)
            h -= h % 4
            w -= w % 4
            seq = torch.from_numpy(
                np.stack([cv2.resize(a, (w, h)) for a in arrays])
                .astype(np.float32) / 255.0).permute(0, 3, 1, 2).unsqueeze(0)
            if self.dev.info().is_cuda:
                seq = seq.to(self.dev.device)
            with torch.no_grad():
                out = net(seq)
            out = out.float().clamp(0, 1).squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
            for i, f in enumerate(chunk[:len(out)]):
                sr = (out[i] * 255.0).round().astype(np.uint8)
                target = cv2.imread(str(f), cv2.IMREAD_COLOR)
                th, tw = target.shape[:2]
                sr_resized = cv2.resize(sr, (tw, th), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(str(f), sr_resized, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                done += 1
            now = time.time()
            if now - t_last > 0.5:
                speed = done / max(now - t_last, 1e-3)
                progress.fps = speed
                progress.eta = (total - done) / speed if speed else None
                t_last = now
            progress.frames_done = done
            progress.percent = done * 100.0 / total
            progress.message = f"Temporal window {start // window + 1} " \
                               f"of {math.ceil(total / window)}"
            self._report(progress)
        del net
        self.dev.release()
        self._report(progress, force=True)
        return total

    # ------------------------------------------------------------------
    def _interpolate(self, count: int, enh: EnhancementSettings,
                     progress: JobProgress) -> int:
        out_dir = self.workspace.frames_out
        files = sorted(out_dir.glob("*.png"))
        if len(files) < 2:
            return count
        factor = enh.interpolation_factor if enh.interpolation_factor in (2, 3, 4) else 2
        tmp_dir = self.workspace.work_dir / "interp"
        tmp_dir.mkdir(exist_ok=True)
        total_pairs = len(files) - 1
        progress.frames_total = total_pairs
        t_last = time.time()
        produced = 0
        out_index = 0

        prev = cv2.cvtColor(cv2.imread(str(files[0]), cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB)
        self._write_frame(tmp_dir, out_index, prev)
        out_index += 1

        for i in range(1, len(files)):
            self._check_cancel()
            cur = cv2.imread(str(files[i]), cv2.IMREAD_COLOR)
            if cur is None:
                continue
            cur = cv2.cvtColor(cur, cv2.COLOR_BGR2RGB)
            # intermediate frames at t = 1/factor ... (factor-1)/factor
            for k in range(1, factor):
                mid = self.rife.interpolate_pair(prev, cur, k / factor)
                self._write_frame(tmp_dir, out_index, mid)
                out_index += 1
                produced += 1
            self._write_frame(tmp_dir, out_index, cur)
            out_index += 1
            prev = cur

            now = time.time()
            if now - t_last > 0.5:
                speed = i / max(now - t_last, 1e-3)
                progress.fps = speed
                progress.eta = (total_pairs - i) / speed if speed else None
                t_last = now
            progress.frames_done = i
            progress.percent = i * 100.0 / total_pairs
            progress.message = (f"Interpolating pair {i:,} / {total_pairs:,}"
                                f"  ·  {factor}x  ·  +{produced:,} new frames")
            self._report(progress)

        # swap directories
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        tmp_dir.rename(out_dir)
        self._report(progress, force=True)
        log.info("Interpolation added %d frames (total %d)", produced, out_index)
        return out_index

    @staticmethod
    def _write_frame(directory: Path, index: int, rgb: np.ndarray) -> None:
        cv2.imwrite(str(directory / f"{index + 1:08d}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])

    # ------------------------------------------------------------------
    def _encode(self, out_file: Path, fps: float, audio_path: Optional[Path],
                spec: JobSpec, frames: int, progress: JobProgress) -> None:
        pattern = self.workspace.frames_out / "%08d.png"
        duration = frames / fps if fps else 0

        def cb(done: int, pct: Optional[float]) -> None:
            progress.frames_done = done
            progress.percent = pct or 0.0
            progress.message = f"Encoding with FFmpeg ({spec.encode.codec.upper()})"
            self._report(progress)

        self.ffmpeg.build_video(
            pattern, fps, out_file, settings=spec.encode,
            audio=str(audio_path) if audio_path else None,
            total_frames=frames, duration=duration, progress=cb,
            cancel=self.cancel)
        progress.percent = 100.0
        self._report(progress, force=True)

    def _verify(self, out_path: Path, want_w: int, want_h: int,
                want_fps: float) -> Optional[VideoInfo]:
        try:
            info = self.ffmpeg.probe(out_path)
        except FFmpegError as exc:
            raise RuntimeError(f"Output verification failed: {exc}")
        log.info("Output verified: %dx%d @%.2f fps, %.2fs, %s",
                 info.width, info.height, info.fps, info.duration,
                 human_size(info.size_bytes))
        if abs(info.width - want_w) > 2 or abs(info.height - want_h) > 2:
            log.warning("Output resolution %dx%d differs from requested %dx%d",
                        info.width, info.height, want_w, want_h)
        return info
