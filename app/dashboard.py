"""Dashboard page: hero upload, KPIs, GPU card, models, recent projects."""
from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from utils import gpu_monitor
from utils.file_manager import free_gb, human_size
from utils.logger import get_logger

from .theme import (Badge, Card, Divider, GradientBar, MetricRow, Palette,
                    RingGauge, StatCard, draw_icon_pixmap, make_icon)
from .widgets import (NoticeBar, PageHeader, RecentProjectRow, SectionTitle,
                      VideoDropZone)

log = get_logger("dashboard")

QUICK_PRESETS = [
    ("VHS Cleanup", "Denoise + colour restore + 1080p", "realesrgan_x2plus",
     {"use_denoise": True, "use_color_restore": True, "use_grain_reduction": True,
      "use_sharpen": True}, "1080p"),
    ("Concert Footage", "Faces + colour + 2K output", "realesrgan_x2plus",
     {"use_face_restoration": True, "use_color_restore": True, "use_deblur": True},
     "2k"),
    ("Archive Master", "Max quality 4K + interpolation", "realesrgan_x4plus",
     {"use_face_restoration": True, "use_temporal_restore": True,
      "use_interpolation": True}, "4k"),
    ("Quick Preview", "Fast 720p pass to check settings", "realesrgan_x2plus",
     {"use_denoise": True}, "720p"),
]


class DashboardPage(QWidget):
    video_dropped = Signal(str)          # start enhancing this file
    quick_enhance = Signal(str, dict)    # (path, settings overrides)
    navigate = Signal(str)

    def __init__(self, ctx, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._recent_rows: List[RecentProjectRow] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(26, 20, 26, 24)
        root.setSpacing(13)

        # ---------------------------------------------------------- header
        header = PageHeader(
            "Dashboard",
            "Restore vintage footage to crisp HD with local AI - no cloud, no login.")
        self.import_btn = QPushButton("  Import Video")
        self.import_btn.setObjectName("Primary")
        self.import_btn.setIcon(make_icon("upload", "#FFFFFF"))
        self.import_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.import_btn.clicked.connect(self._import_video)
        header.add_action(self.import_btn)
        root.addWidget(header)

        self.notice = NoticeBar()
        root.addWidget(self.notice)

        # ------------------------------------------------------------ hero
        hero = Card(object_name="HeroCard", padding=20, spacing=12)
        hero_row = QHBoxLayout()
        hero_row.setSpacing(22)
        left = QVBoxLayout()
        left.setSpacing(8)
        kick = QLabel("LOCAL  ·  GPU ACCELERATED  ·  OPEN SOURCE MODELS")
        kick.setStyleSheet(
            f"color: #C4B5FD; font-size: 10px; font-weight: 800; letter-spacing: 1.8px;")
        left.addWidget(kick)
        h1 = QLabel("Bring your old videos back to life.")
        h1.setStyleSheet("font-size: 27px; font-weight: 800;")
        left.addWidget(h1)
        sub = QLabel("Real-ESRGAN upscaling, GFPGAN face repair, RIFE interpolation "
                     "and BasicVSR++ temporal restoration - all local.")
        sub.setWordWrap(True)
        sub.setObjectName("Dim")
        sub.setMaximumWidth(600)
        left.addWidget(sub)
        left.addSpacing(6)
        self.drop_zone = VideoDropZone(subtitle="or click to browse  ·  MP4, AVI, "
                                                  "MOV, MKV, WMV, MPEG, WEBM, FLV")
        self.drop_zone.setMinimumHeight(136)
        self.drop_zone.file_selected.connect(self._on_file)
        left.addWidget(self.drop_zone)
        hero_row.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(10)
        self.gauge = RingGauge(diameter=132)
        gw = QHBoxLayout()
        gw.addStretch(1)
        gw.addWidget(self.gauge)
        gw.addStretch(1)
        right.addLayout(gw)
        self.hero_gpu = QLabel("Detecting GPU...")
        self.hero_gpu.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hero_gpu.setStyleSheet("font-weight: 700; font-size: 12px;")
        right.addWidget(self.hero_gpu)
        self.hero_vram = QLabel("")
        self.hero_vram.setObjectName("Muted")
        self.hero_vram.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hero_vram.setStyleSheet("font-size: 11px;")
        right.addWidget(self.hero_vram)
        right.addSpacing(4)
        gpu_btn = QPushButton("Open GPU Monitor")
        gpu_btn.setObjectName("Ghost")
        gpu_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        gpu_btn.clicked.connect(lambda: self.navigate.emit("gpu"))
        right.addWidget(gpu_btn)
        right.addStretch(1)
        hero_row.addLayout(right, 2)
        hero.add_layout(hero_row)
        root.addWidget(hero)

        # ------------------------------------------------------------ KPIs
        kpis = QGridLayout()
        kpis.setHorizontalSpacing(14)
        kpis.setVerticalSpacing(14)
        self.kpi_processed = StatCard("film", "Videos processed", "0",
                                      "all time", Palette.PURPLE)
        self.kpi_hours = StatCard("history", "Footage restored", "0m",
                                  "total output duration", Palette.BLUE)
        self.kpi_models = StatCard("cpu", "AI models installed", "0",
                                   "of 6 available", Palette.CYAN)
        self.kpi_storage = StatCard("folder", "Output folder free", "-",
                                    "on output drive", Palette.GREEN)
        for i, card in enumerate((self.kpi_processed, self.kpi_hours,
                                  self.kpi_models, self.kpi_storage)):
            kpis.addWidget(card, 0, i)
        root.addLayout(kpis)

        # -------------------------------------------------- quick + recent
        mid = QHBoxLayout()
        mid.setSpacing(16)

        quick = Card(padding=18, spacing=10)
        quick.add_layout(self._card_header(
            "Quick enhancement", "One-click recipes using the loaded defaults"))
        self.quick_grid = QGridLayout()
        self.quick_grid.setSpacing(10)
        self._build_quick_buttons()
        quick.add_layout(self.quick_grid)
        mid.addWidget(quick, 3)

        recent = Card(padding=18, spacing=10)
        rh = self._card_header("Recent projects", "Double-click to open the result")
        see_all = QPushButton("View all")
        see_all.setObjectName("Link")
        see_all.setCursor(Qt.CursorShape.PointingHandCursor)
        see_all.clicked.connect(lambda: self.navigate.emit("history"))
        rh.addWidget(see_all)
        recent.add_layout(rh)
        self.recent_container = QVBoxLayout()
        self.recent_container.setSpacing(8)
        recent.add_layout(self.recent_container)
        self.recent_empty = QLabel("Nothing processed yet - import a video to begin.")
        self.recent_empty.setObjectName("Muted")
        self.recent_empty.setStyleSheet("padding: 16px 0;")
        self.recent_container.addWidget(self.recent_empty)
        recent.body().addStretch(1)
        mid.addWidget(recent, 4)
        root.addLayout(mid)

        # -------------------------------------------------- models strip
        models_card = Card(padding=18, spacing=10)
        models_card.add_layout(self._card_header(
            "Installed AI models", "Managed locally in the models folder"))
        mh = QHBoxLayout()
        mh.setSpacing(10)
        self.model_badges = QHBoxLayout()
        self.model_badges.setSpacing(8)
        mh.addLayout(self.model_badges)
        mh.addStretch(1)
        manage = QPushButton("Manage models")
        manage.setObjectName("Ghost")
        manage.setCursor(Qt.CursorShape.PointingHandCursor)
        manage.clicked.connect(lambda: self.navigate.emit("models"))
        mh.addWidget(manage)
        models_card.add_layout(mh)
        self.models_hint = QLabel("")
        self.models_hint.setObjectName("Muted")
        self.models_hint.setStyleSheet("font-size: 11px;")
        models_card.add(self.models_hint)
        root.addWidget(models_card)
        root.addStretch(1)

        # GPU refresh timer (live utilisation + VRAM)
        self._gpu_timer = QTimer(self)
        self._gpu_timer.setInterval(int(ctx.config.get("gpu.monitor_interval_ms", 1000)))
        self._gpu_timer.timeout.connect(self._refresh_gpu)
        self._gpu_timer.start()
        self.refresh_all()

    # ------------------------------------------------------------------
    @staticmethod
    def _card_header(title: str, subtitle: str = "") -> QHBoxLayout:
        row = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("H3")
        col.addWidget(t)
        if subtitle:
            s = QLabel(subtitle)
            s.setObjectName("Muted")
            s.setStyleSheet("font-size: 11px;")
            col.addWidget(s)
        row.addLayout(col)
        row.addStretch(1)
        return row

    def _build_quick_buttons(self) -> None:
        while self.quick_grid.count():
            item = self.quick_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, (name, desc, model, overrides, resolution) in enumerate(QUICK_PRESETS):
            btn = QPushButton()
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("preset_name", name)
            btn.setStyleSheet("text-align: left; padding: 10px 12px;")
            btn.setIcon(make_icon("bolt", Palette.TEXT, Palette.PURPLE))
            btn.setText(f"  {name}\n  {desc}")
            btn.setMinimumHeight(48)
            btn.clicked.connect(lambda _=False, n=name, m=model, o=dict(overrides),
                                r=resolution: self._run_preset(n, m, o, r))
            self.quick_grid.addWidget(btn, i // 2, i % 2)

    def _run_preset(self, name: str, model: str, overrides: dict,
                    resolution: str) -> None:
        path = getattr(self, "_pending_path", "")
        if not path:
            self.notice.show_message(
                f"Import a video first, then click \u201c{name}\u201d to apply the preset.",
                "info")
            self._import_video()
            return
        settings = {
            "ai_model": model,
            "target_resolution": resolution,
            "use_super_resolution": True,
            **overrides,
        }
        log.info("Quick preset '%s' applied to %s", name, path)
        self.quick_enhance.emit(path, settings)

    # ------------------------------------------------------------------
    def _import_video(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        from utils.config import SUPPORTED_VIDEO_EXTENSIONS
        patterns = " ".join(f"*{e}" for e in SUPPORTED_VIDEO_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Import video", "", f"Video files ({patterns});;All files (*.*)")
        if path:
            self._on_file(path)

    def _on_file(self, path: str) -> None:
        self._pending_path = path
        self.video_dropped.emit(path)

    # ------------------------------------------------------------------
    def refresh_all(self) -> None:
        self._refresh_gpu()
        self._refresh_kpis()
        self._refresh_recent()
        self._refresh_models()

    def _refresh_gpu(self) -> None:
        try:
            gpu = gpu_monitor.query_gpu()
        except Exception:
            gpu = gpu_monitor.GPUStats()
        if gpu.available:
            self.gauge.set_value(gpu.utilization)
            name = gpu.name
            if len(name) > 34:
                name = name[:32] + "\u2026"
            self.hero_gpu.setText(name)
            self.hero_vram.setText(
                f"{gpu.used_vram_mb} / {gpu.total_vram_mb} MB VRAM"
                f"  ·  {gpu.temperature_c}\u00b0C")
        else:
            self.gauge.set_value(0)
            self.hero_gpu.setText("CPU mode")
            self.hero_vram.setText("No NVIDIA GPU detected - processing will be slow")

    def _refresh_kpis(self) -> None:
        jm = getattr(self.ctx, "job_manager", None)
        records = []
        if jm is not None:
            try:
                records = jm.history_records(limit=500)
            except Exception:
                records = []
        completed = [r for r in records if r.get("status") == "completed"]
        self.kpi_processed.set_value(str(len(completed)),
                                     f"{len(records)} jobs in history")
        total_sec = sum(float(r.get("output_duration") or 0) for r in completed)
        if total_sec >= 3600:
            self.kpi_hours.set_value(f"{total_sec / 3600:.1f}h")
        else:
            self.kpi_hours.set_value(f"{total_sec / 60:.0f}m")

        mm = getattr(self.ctx, "model_manager", None)
        installed = 0
        total = 0
        if mm is not None:
            try:
                statuses = mm.model_status()
                total = len(statuses)
                installed = sum(1 for s in statuses if s.installed)
            except Exception:
                pass
        self.kpi_models.set_value(f"{installed}", f"of {total or 6} available")

        try:
            gb = free_gb(self.ctx.config.output_dir)
            self.kpi_storage.set_value(f"{gb:.0f} GB",
                                       str(self.ctx.config.output_dir))
        except Exception:
            self.kpi_storage.set_value("-", "output folder")

    def _refresh_recent(self) -> None:
        # clear old rows
        for row in self._recent_rows:
            row.setParent(None)
            row.deleteLater()
        self._recent_rows.clear()
        jm = getattr(self.ctx, "job_manager", None)
        records = []
        if jm is not None:
            try:
                records = jm.history_records(limit=5)
            except Exception:
                records = []
        self.recent_empty.setVisible(not records)
        for rec in records:
            row = RecentProjectRow(rec)
            row.open_requested.connect(self._open_video)
            row.folder_requested.connect(self._open_folder)
            self.recent_container.addWidget(row)
            self._recent_rows.append(row)

    def _refresh_models(self) -> None:
        while self.model_badges.count():
            item = self.model_badges.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        mm = getattr(self.ctx, "model_manager", None)
        if mm is None:
            self.models_hint.setText("Model manager not initialised.")
            return
        try:
            statuses = mm.model_status()
        except Exception as exc:
            self.models_hint.setText(f"Could not read model list: {exc}")
            return
        installed = [s for s in statuses if s.installed]
        for s in statuses:
            b = Badge(s.short_name, "green" if s.installed else "purple")
            b.setToolTip(f"{s.title}\n"
                         f"{'Installed' if s.installed else 'Not installed'}"
                         f" · {human_size(s.size_bytes)}")
            self.model_badges.addWidget(b)
        missing = [s.title for s in statuses if not s.installed]
        if missing:
            self.models_hint.setText(
                f"{len(installed)} of {len(statuses)} models installed. "
                f"Missing: {', '.join(missing[:4])}"
                + (" ..." if len(missing) > 4 else "")
                + "  -  open the AI Model Manager to download.")
        else:
            self.models_hint.setText("All models installed and ready.")

    # ------------------------------------------------------------------
    @staticmethod
    def _open_video(path: str) -> None:
        if path and os.path.exists(path):
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])

    @staticmethod
    def _open_folder(path: str) -> None:
        folder = os.path.dirname(path) if path else ""
        if folder and os.path.isdir(folder):
            if os.name == "nt":
                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", path])
            else:
                subprocess.Popen(["xdg-open", folder])

    def notify(self, text: str, kind: str = "info") -> None:
        self.notice.show_message(text, kind)
