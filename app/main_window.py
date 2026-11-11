"""Main window: sidebar navigation + stacked pages + status bar."""
from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QStackedWidget,
                               QStatusBar, QWidget)

from utils.config import app_root
from utils.file_manager import human_size
from utils.gpu_monitor import query_gpu
from utils.logger import get_logger

from .about_page import AboutPage
from .batch_page import BatchPage
from .dashboard import DashboardPage
from .gpu_page import GPUPage
from .history_page import HistoryPage
from .model_page import ModelManagerPage
from .settings_page import SettingsPage
from .sidebar import Sidebar
from .theme import Palette, build_stylesheet
from .video_enhancer import VideoEnhancerPage

log = get_logger("main_window")


class MainWindow(QMainWindow):
    def __init__(self, ctx) -> None:
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle("RetroVision AI - Video Restoration Studio")
        icon_path = app_root() / "assets" / "app_icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.setStyleSheet(build_stylesheet())
        self.resize(int(ctx.config.get("ui.window_width", 1440)),
                    int(ctx.config.get("ui.window_height", 900)))
        self.setMinimumSize(1180, 760)

        self._build()
        self._connect()
        self._start_status_timer()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        central = QWidget()
        central.setObjectName("Root")
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.sidebar = Sidebar()
        layout.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.pages: Dict[str, QWidget] = {
            "dashboard": DashboardPage(self.ctx),
            "enhancer": VideoEnhancerPage(self.ctx),
            "batch": BatchPage(self.ctx),
            "models": ModelManagerPage(self.ctx),
            "gpu": GPUPage(self.ctx),
            "history": HistoryPage(self.ctx),
            "settings": SettingsPage(self.ctx),
            "about": AboutPage(self.ctx),
        }
        self.page_index: Dict[str, int] = {}
        for key, page in self.pages.items():
            self.page_index[key] = self.stack.addWidget(page)

        # ---- status bar ------------------------------------------
        bar = QStatusBar()
        bar.setSizeGripEnabled(False)
        self.setStatusBar(bar)
        self.status_gpu = QLabel("GPU: detecting...")
        self.status_ffmpeg = QLabel("")
        self.status_job = QLabel("")
        self.status_models = QLabel("")
        for w in (self.status_gpu, self.status_ffmpeg, self.status_models,
                  self.status_job):
            w.setStyleSheet(f"color: {Palette.TEXT_MUTED}; font-size: 11px;")
            bar.addWidget(w)
        bar.addPermanentWidget(QLabel("  "))
        version = QLabel(f"v1.0.0")
        version.setStyleSheet(f"color: {Palette.TEXT_MUTED}; font-size: 11px;")
        bar.addPermanentWidget(version)

    # ------------------------------------------------------------------
    def _connect(self) -> None:
        self.sidebar.navigate.connect(self.navigate)
        dash: DashboardPage = self.pages["dashboard"]      # type: ignore
        dash.video_dropped.connect(self._open_in_enhancer)
        dash.quick_enhance.connect(self._quick_enhance)
        dash.navigate.connect(self.navigate)

        enh: VideoEnhancerPage = self.pages["enhancer"]     # type: ignore
        jm = self.ctx.job_manager
        jm.job_started.connect(lambda spec: self.status_job.setText(
            f"Processing: {spec.label}"))
        jm.job_finished.connect(lambda r: self.status_job.setText(
            "Job finished"))
        jm.job_failed.connect(lambda s, m: self.status_job.setText("Job failed"))
        jm.queue_changed.connect(self._refresh_queue_status)

        dashboard = self.pages["dashboard"]
        jm.job_finished.connect(lambda _r: dashboard.refresh_all())  # type: ignore

    def _start_status_timer(self) -> None:
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self._refresh_status)
        self.timer.start()
        self._refresh_status()

    def _refresh_status(self) -> None:
        gpu = query_gpu(live=False)
        if gpu.available:
            self.status_gpu.setText(f"GPU: {gpu.name[:38]}  ·  {gpu.total_vram_mb} MB")
            self.sidebar.set_gpu(gpu.name[:22],
                                 f"{gpu.total_vram_mb} MB VRAM", True)
        else:
            self.status_gpu.setText("GPU: none (CPU mode)")
            self.sidebar.set_gpu("CPU mode", "no NVIDIA GPU", False)
        ok, msg = self.ctx.ffmpeg_status()
        self.status_ffmpeg.setText(("  ·  " + msg) if ok else "  ·  FFmpeg missing")
        statuses = self.ctx.model_manager.model_status()
        installed = sum(1 for s in statuses if s.installed)
        self.status_models.setText(f"  ·  Models: {installed}/{len(statuses)}")
        self._refresh_queue_status()

    def _refresh_queue_status(self) -> None:
        jm = self.ctx.job_manager
        if jm.busy and jm.current is not None:
            self.status_job.setText(f"  ·  Running: {jm.current.label}  "
                                    f"({len(jm.queue)} queued)")
        elif jm.queue:
            self.status_job.setText(f"  ·  {len(jm.queue)} job(s) queued")
        else:
            self.status_job.setText("")

    # ------------------------------------------------------------------
    def navigate(self, key: str) -> None:
        if key not in self.page_index:
            return
        idx = self.page_index[key]
        self.stack.setCurrentIndex(idx)
        self.sidebar.set_active(key)
        page = self.pages[key]
        refresh = getattr(page, "refresh", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                log.exception("Page refresh failed for %s", key)
        page_refresh_all = getattr(page, "refresh_all", None)
        if callable(page_refresh_all):
            try:
                page_refresh_all()
            except Exception:
                log.exception("Page refresh_all failed for %s", key)

    def _open_in_enhancer(self, path: str) -> None:
        self.navigate("enhancer")
        enh: VideoEnhancerPage = self.pages["enhancer"]   # type: ignore
        enh.load_video(path)

    def _quick_enhance(self, path: str, settings: dict) -> None:
        self.navigate("enhancer")
        enh: VideoEnhancerPage = self.pages["enhancer"]   # type: ignore
        if enh.load_video(path):
            enh.apply_settings(settings)
            enh.start_job()

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        jm = self.ctx.job_manager
        if jm.busy:
            from PySide6.QtWidgets import QMessageBox
            answer = QMessageBox.question(
                self, "Processing in progress",
                "A video is still being processed. Cancel it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            jm.cancel_all()
        self.ctx.config.update({
            "ui": {"window_width": self.width(), "window_height": self.height()}})
        for page in self.pages.values():
            shutdown = getattr(page, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
        self.ctx.shutdown()
        super().closeEvent(event)
