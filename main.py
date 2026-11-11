#!/usr/bin/env python3
"""RetroVision AI - entry point.

    python main.py            start the desktop application
    python main.py --check    run environment diagnostics (no GUI)

The application is fully local: no network calls except the explicit model
downloads you trigger in the AI Model Manager.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

# Make the project importable when launched from any cwd
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_NAME = "RetroVision AI"
APP_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Diagnostics (usable headless, also tests the same code paths as the GUI)
# ---------------------------------------------------------------------------
def run_checks() -> int:
    from utils.config import Config
    from utils.logger import get_logger, setup_logging
    from utils.ffmpeg import FFmpeg
    from ai.model_manager import ModelManager
    from ai.device_manager import DeviceManager
    from utils.gpu_monitor import query_gpu, system_stats

    config = Config()
    config.ensure_dirs()
    setup_logging(config.logs_dir)
    log = get_logger("checks")
    ok = True
    print(f"{APP_NAME} {APP_VERSION} - environment check\n" + "-" * 52)

    print(f"Python        : {sys.version.split()[0]} ({sys.implementation.name})")
    print(f"Project root  : {ROOT}")
    print(f"Config        : {config.path}")

    try:
        import PySide6
        print(f"PySide6       : {PySide6.__version__}")
    except Exception as exc:
        ok = False
        print(f"PySide6       : MISSING ({exc})")

    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"PyTorch       : {torch.__version__}   CUDA: {cuda}")
        if cuda:
            print(f"CUDA device   : {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"PyTorch       : not available ({exc})")

    try:
        import cv2
        print(f"OpenCV        : {cv2.__version__}")
    except Exception:
        ok = False
        print("OpenCV        : MISSING")

    ff = FFmpeg(config.get("paths.ffmpeg_path", ""),
                config.get("paths.ffprobe_path", ""))
    try:
        ff.locate()
        print(f"FFmpeg        : {ff.version()}")
        encs = [e.name for e in ff.available_encoders() if e.available]
        print(f"Encoders      : {', '.join(encs) or 'none'}")
    except Exception as exc:
        ok = False
        print(f"FFmpeg        : MISSING ({exc})")

    gpu = query_gpu()
    print(f"GPU           : {gpu.name if gpu.available else 'CPU only'}"
          + (f" ({gpu.total_vram_mb} MB)" if gpu.available else ""))
    stats = system_stats()
    print(f"CPU / RAM     : {stats.cpu_count} threads / {stats.ram_total_gb:.1f} GB")

    mm = ModelManager(config.models_dir)
    print(f"Models folder : {mm.models_dir}")
    for status in mm.model_status():
        mark = "OK " if status.installed else "-- "
        print(f"  [{mark}] {status.title:<38} {status.size_bytes/1048576:7.1f} MB"
              f"   {status.license}")

    dev = DeviceManager.instance(config.get("processing.device", "auto"),
                                 config.get("processing.precision", "fp16"))
    try:
        info = dev.info()
        print(f"Device        : {info.label}")
    except Exception as exc:
        print(f"Device        : unavailable ({exc})")

    print("-" * 52)
    print("RESULT        :", "ready" if ok else "problems found")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
def run_gui(files: list[str]) -> int:
    # Qt 6 high-DPI handling is automatic; keep the app crisp on scaled displays
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon, QPixmap
    from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen

    from app.context import AppContext
    from app.main_window import MainWindow
    from app.theme import build_stylesheet
    from utils.config import Config
    from utils.file_manager import sweep_stale_temp
    from utils.logger import get_logger, setup_logging

    config = Config()
    config.ensure_dirs()
    setup_logging(config.logs_dir)
    log = get_logger("main")
    log.info("Starting %s %s", APP_NAME, APP_VERSION)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("RetroVision")
    app.setStyle("Fusion")
    app.setStyleSheet(build_stylesheet())

    icon_path = ROOT / "assets" / "app_icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    splash = None
    splash_path = ROOT / "assets" / "app_icon_256.png"
    if splash_path.exists():
        pix = QPixmap(str(splash_path))
        splash = QSplashScreen(pix.scaled(220, 220, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation))
        splash.showMessage("Loading RetroVision AI...",
                           Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                           Qt.GlobalColor.white)
        splash.show()
        app.processEvents()

    try:
        sweep_stale_temp(config.temp_dir, max_age_hours=48)
        ctx = AppContext(config)
        window = MainWindow(ctx)
        window.show()
        if splash is not None:
            splash.finish(window)

        # CLI files open straight in the enhancer
        for f in files:
            if Path(f).exists():
                window._open_in_enhancer(f)
                break

        # first-run guidance
        if not config.get("app.first_run_complete", False):
            installed = [s for s in ctx.model_manager.model_status() if s.installed]
            if not installed:
                QMessageBox.information(
                    window, "Welcome to RetroVision AI",
                    "No AI models are installed yet.\n\n"
                    "Open the AI Model Manager, download Real-ESRGAN x2plus "
                    "(the recommended upscaler), and you are ready to restore "
                    "your first video.\n\n"
                    "Models are downloaded from their official repositories "
                    "and verified with SHA-256.")
            config.set("app.first_run_complete", True)

        return app.exec()
    except Exception as exc:
        log.exception("Fatal error during startup")
        QMessageBox.critical(None, "RetroVision AI - startup error",
                             f"{type(exc).__name__}: {exc}\n\n"
                             f"See {config.logs_dir / 'retrovision.log'} for details.")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} {APP_VERSION}")
    parser.add_argument("files", nargs="*", help="video file(s) to open")
    parser.add_argument("--check", action="store_true",
                        help="run environment diagnostics and exit")
    parser.add_argument("--version", action="store_true", help="print version")
    parser.add_argument("--quick", metavar="VIDEO",
                        help="headless: enhance VIDEO and exit (no GUI)")
    parser.add_argument("--preset", default=None,
                        choices=["original", "720p", "1080p", "2k", "4k"],
                        help="target resolution for --quick (default: from config)")
    parser.add_argument("--output", default=None,
                        help="output file for --quick")
    parser.add_argument("--no-sr", action="store_true",
                        help="--quick: skip AI super-resolution (filters only)")
    parser.add_argument("--quiet", action="store_true",
                        help="--quick: only print the final result")
    args = parser.parse_args(argv)

    if args.version:
        print(f"{APP_NAME} {APP_VERSION}")
        return 0
    if args.check:
        return run_checks()
    if args.quick:
        return run_quick(args.quick, args.preset, args.output,
                         not args.no_sr, args.quiet)
    return run_gui(args.files)


def run_quick(video: str, preset: str | None, output: str | None,
              super_res: bool = True, quiet: bool = False) -> int:
    """Headless one-file enhancement.  Prints real pipeline progress."""
    from pathlib import Path

    from utils.config import Config
    from utils.logger import setup_logging

    from ai.device_manager import DeviceManager
    from ai.model_manager import ModelManager
    from core.job import EnhancementSettings, JobSpec
    from core.pipeline import VideoPipeline, target_size
    from utils.ffmpeg import EncodeSettings, FFmpeg

    config = Config()
    config.ensure_dirs()
    setup_logging(config.logs_dir, console=False)

    source = Path(video).expanduser()
    if not source.exists():
        print(f"error: no such file: {source}", file=sys.stderr)
        return 2

    ffmpeg = FFmpeg()
    if not ffmpeg.available:
        print("error: FFmpeg was not found. Install it or set paths.ffmpeg_path.",
              file=sys.stderr)
        return 3
    try:
        info = ffmpeg.probe(source)
    except Exception as exc:                       # noqa: BLE001
        print(f"error: cannot read video: {exc}", file=sys.stderr)
        return 4

    models = ModelManager(config.models_dir)
    enh_dict = config.get("enhancement", {}) or {}
    enh = EnhancementSettings.from_dict(enh_dict) if hasattr(
        EnhancementSettings, "from_dict") else EnhancementSettings(**{
            k: v for k, v in enh_dict.items()
            if k in EnhancementSettings.__dataclass_fields__})
    enh.use_super_resolution = bool(super_res)
    if preset:
        enh.target_resolution = preset
    if enh.use_super_resolution and models.missing_for([enh.ai_model]):
        print(f"error: model '{enh.ai_model}' is not installed. "
              f"Run the app once and use AI Model Manager.", file=sys.stderr)
        return 5

    if output:
        out_path = Path(output).expanduser()
    else:
        w, h = target_size(info, enh.target_resolution)
        out_path = source.with_name(f"{source.stem}_{enh.target_resolution}_{w}x{h}_restored.mp4")

    device = DeviceManager(config.get("processing.device", "auto"),
                           config.get("processing.precision", "fp32"))
    encode = EncodeSettings(
        codec=config.get("export.codec", "h264"),
        encoder=config.get("export.encoder", "auto"),
        quality=config.get("export.quality", "high"),
        crf=int(config.get("export.crf", 19)),
        preset=config.get("export.preset", "medium"),
        audio_codec=config.get("export.audio_codec", "aac"),
        audio_bitrate_kbps=int(config.get("export.audio_bitrate_kbps", 192)))

    spec = JobSpec(input_path=str(source), enhancement=enh, encode=encode,
                   output_path=str(out_path))

    last = {"stage": ""}

    def on_progress(p):
        if quiet:
            return
        if p.stage != last["stage"]:
            last["stage"] = p.stage
            print(f"\n[{p.stage}]", end="", flush=True)
        if p.total:
            print(f"\r{p.stage}: {p.done}/{p.total} frames "
                  f"({p.percent:5.1f}%)", end="", flush=True)

    print(f"RetroVision AI {APP_VERSION} - headless")
    print(f"  source : {source}")
    print(f"  output : {out_path}")
    print(f"  target : {enh.target_resolution} -> "
          f"{target_size(info, enh.target_resolution)}")
    print(f"  device : {device.info().device.upper()}")

    pipeline = VideoPipeline(ffmpeg, device, models, config.temp_dir,
                             progress_cb=on_progress,
                             resume=bool(config.get("processing.resume_enabled", True)))
    try:
        result = pipeline.run(spec)
    except KeyboardInterrupt:
        print("\ncancelled", file=sys.stderr)
        return 130

    if result.status != "completed":
        print(f"\nerror: {result.error}", file=sys.stderr)
        return 6

    if not quiet:
        print()
    print(f"done: {out_path}  ({result.frames_processed} frames, "
          f"{result.duration_s:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
