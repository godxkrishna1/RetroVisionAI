"""Batch Processing page: queue many files with one shared configuration."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QHeaderView,
                               QLabel, QMessageBox, QProgressBar, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from core.job import EnhancementSettings, JobSpec
from utils.config import RESOLUTION_LABELS
from utils.file_manager import human_size, unique_path
from utils.logger import get_logger

from .theme import Badge, Card, Palette, make_icon
from .widgets import NoticeBar, PageHeader, ProgressPanel

log = get_logger("batch")


class BatchPage(QWidget):
    navigate = Signal(str)

    def __init__(self, ctx, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.files: List[str] = []
        self._build()
        self.ctx.job_manager.queue_changed.connect(self.refresh_queue)
        self.ctx.job_manager.job_progress.connect(self._on_progress)
        self.ctx.job_manager.job_finished.connect(lambda _r: self.refresh_queue())
        self.ctx.job_manager.job_failed.connect(lambda *_: self.refresh_queue())

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 20)
        root.setSpacing(14)

        header = PageHeader("Batch Processing",
                            "Queue a folder of clips and restore them overnight "
                            "with one configuration.")
        add_btn = QPushButton("  Add files")
        add_btn.setObjectName("Primary")
        add_btn.setIcon(make_icon("upload", "#FFFFFF"))
        add_btn.clicked.connect(self.add_files)
        header.add_action(add_btn)
        folder_btn = QPushButton("Add folder")
        folder_btn.setObjectName("Ghost")
        folder_btn.clicked.connect(self.add_folder)
        header.add_action(folder_btn)
        root.addWidget(header)

        self.notice = NoticeBar()
        root.addWidget(self.notice)

        cfg = Card(padding=16, spacing=10)
        row = QHBoxLayout()
        row.setSpacing(12)
        lbl = QLabel("Target resolution")
        lbl.setObjectName("Muted")
        row.addWidget(lbl)
        self.res_combo = QComboBox()
        for key in ("original", "720p", "1080p", "2k", "4k"):
            self.res_combo.addItem(RESOLUTION_LABELS[key], key)
        self.res_combo.setCurrentIndex(2)
        row.addWidget(self.res_combo)

        lbl2 = QLabel("AI model")
        lbl2.setObjectName("Muted")
        row.addWidget(lbl2)
        self.model_combo = QComboBox()
        row.addWidget(self.model_combo)
        row.addStretch(1)
        self.count_label = QLabel("0 files queued")
        self.count_label.setObjectName("Muted")
        row.addWidget(self.count_label)
        cfg.add_layout(row)

        row2 = QHBoxLayout()
        row2.setSpacing(10)
        self.use_enhancer_settings = QComboBox()
        self.use_enhancer_settings.addItem(
            "Use the enhancement settings from the Video Enhancer page", True)
        self.use_enhancer_settings.addItem(
            "Batch defaults (super-res + denoise + colour restore)", False)
        row2.addWidget(self.use_enhancer_settings, 1)
        cfg.add_layout(row2)
        root.addWidget(cfg)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["FILE", "SOURCE", "DURATION", "STATUS", "OUTPUT"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        root.addWidget(self.table, 1)

        self.progress_panel = ProgressPanel()
        root.addWidget(self.progress_panel)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.start_btn = QPushButton("  Start batch")
        self.start_btn.setObjectName("Primary")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setIcon(make_icon("play", "#FFFFFF"))
        self.start_btn.clicked.connect(self.start_batch)
        actions.addWidget(self.start_btn, 2)
        self.pause_btn = QPushButton("Pause queue")
        self.pause_btn.setObjectName("Ghost")
        self.pause_btn.setMinimumHeight(40)
        self.pause_btn.clicked.connect(self.toggle_pause)
        actions.addWidget(self.pause_btn, 1)
        self.cancel_btn = QPushButton("Cancel current")
        self.cancel_btn.setObjectName("Danger")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.clicked.connect(self.ctx.job_manager.cancel_current)
        actions.addWidget(self.cancel_btn, 1)
        remove_btn = QPushButton("Remove selected")
        remove_btn.setObjectName("Ghost")
        remove_btn.setMinimumHeight(40)
        remove_btn.clicked.connect(self.remove_selected)
        actions.addWidget(remove_btn, 1)
        clear_btn = QPushButton("Clear list")
        clear_btn.setObjectName("Ghost")
        clear_btn.setMinimumHeight(40)
        clear_btn.clicked.connect(self.clear_list)
        actions.addWidget(clear_btn, 1)
        root.addLayout(actions)

        self.refresh_models()
        self.refresh_queue()

    # ------------------------------------------------------------------
    def refresh_models(self) -> None:
        self.model_combo.clear()
        for status in self.ctx.model_manager.model_status():
            if status.category != "upscale":
                continue
            self.model_combo.addItem(status.title + ("" if status.installed
                                                     else "  -  not installed"),
                                     status.key)

    # ------------------------------------------------------------------
    def add_files(self, paths: list[str] | None = None) -> None:
        """Add files to the queue - opens a dialog when no paths are supplied."""
        if paths:
            self._add([str(p) for p in paths])
            return
        from utils.config import SUPPORTED_VIDEO_EXTENSIONS
        patterns = " ".join(f"*{e}" for e in SUPPORTED_VIDEO_EXTENSIONS)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add videos to batch", "",
            f"Video files ({patterns});;All files (*.*)")
        self._add(paths)

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add a folder of videos")
        if not folder:
            return
        from utils.file_manager import list_videos
        paths = [str(p) for p in list_videos(Path(folder))]
        if not paths:
            self.notice.show_message("No supported videos found in that folder.",
                                     "warn")
            return
        self._add(paths)

    def _add(self, paths: List[str]) -> None:
        added = 0
        for p in paths:
            if p in self.files:
                continue
            try:
                info = self.ctx.ffmpeg.probe(p)
            except Exception as exc:
                log.warning("Skipping %s: %s", p, exc)
                continue
            self.files.append(p)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(Path(p).name))
            self.table.setItem(row, 1, QTableWidgetItem(f"{info.width}x{info.height}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{info.duration:.1f}s"))
            status_item = QTableWidgetItem("Ready")
            self.table.setItem(row, 3, status_item)
            self.table.setItem(row, 4, QTableWidgetItem(""))
            added += 1
        if added:
            self.notice.show_message(f"Added {added} file(s) to the batch.", "success")
        self.count_label.setText(f"{len(self.files)} files queued")

    def remove_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()},
                      reverse=True)
        for row in rows:
            self.table.removeRow(row)
            if 0 <= row < len(self.files):
                self.files.pop(row)
        self.count_label.setText(f"{len(self.files)} files queued")

    def clear_list(self) -> None:
        self.table.setRowCount(0)
        self.files.clear()
        self.count_label.setText("0 files queued")

    # ------------------------------------------------------------------
    def _build_settings(self, index: int) -> EnhancementSettings:
        if self.use_enhancer_settings.currentData():
            cfg = self.ctx.config
            enh = EnhancementSettings.from_dict(cfg.get("enhancement", {}))
        else:
            enh = EnhancementSettings()
            enh.use_super_resolution = True
            enh.use_denoise = True
            enh.use_color_restore = True
        enh.target_resolution = self.res_combo.currentData()
        key = self.model_combo.currentData()
        if key:
            enh.ai_model = key
        return enh

    def start_batch(self) -> None:
        if not self.files:
            self.notice.show_message("Add at least one video to the batch.", "warn")
            return
        missing = self.ctx.model_manager.missing_for([self.model_combo.currentData()])
        if missing:
            self.notice.show_message(
                "The selected AI model is not installed - download it in "
                "AI Model Manager first.", "warn")
            return
        out_dir = self.ctx.config.output_dir
        specs = []
        for i, path in enumerate(self.files):
            enh = self._build_settings(i)
            stem = Path(path).stem
            out_path = unique_path(out_dir, f"{stem}_{enh.target_resolution}_restored")
            spec = JobSpec(
                input_path=path, enhancement=enh,
                encode=self._encode_settings(), output_path=str(out_path),
                priority=0, label=f"{stem} -> {enh.target_resolution}")
            specs.append(spec)
            self.table.setItem(i, 3, QTableWidgetItem("Queued"))
        self.ctx.job_manager.enqueue_many(specs)
        self.ctx.job_manager.resume()
        self.notice.show_message(
            f"Batch started: {len(specs)} job(s). You can keep working - "
            "progress updates live below.", "success")

    def _encode_settings(self):
        from utils.ffmpeg import EncodeSettings
        cfg = self.ctx.config
        return EncodeSettings(
            codec=cfg.get("export.codec", "h264"),
            encoder=cfg.get("export.encoder", "auto"),
            quality=cfg.get("export.quality", "high"),
            crf=int(cfg.get("export.crf", 19)),
            preset=cfg.get("export.preset", "slow"))

    def toggle_pause(self) -> None:
        jm = self.ctx.job_manager
        if jm.paused:
            jm.resume()
            self.pause_btn.setText("Pause queue")
        else:
            jm.pause()
            self.pause_btn.setText("Resume queue")

    # ------------------------------------------------------------------
    def refresh_queue(self) -> None:
        jm = self.ctx.job_manager
        active = jm.current
        queue_ids = {s.job_id for s in jm.queue}
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 3)
            if row < len(self.files):
                path = self.files[row]
                if active is not None and getattr(active,'input_path',None) == path:
                    item.setText("Processing")
                elif any(s.input_path == path for s in jm.queue):
                    item.setText("Queued")

    def _on_progress(self, p) -> None:
        from utils.file_manager import human_duration
        self.progress_panel.set_stage(p.stage)
        self.progress_panel.set_progress(
            p.frames_done, p.frames_total, p.percent,
            speed=f"{p.fps:.2f} fps" if p.fps else "",
            elapsed=human_duration(p.elapsed),
            eta=human_duration(p.eta) if p.eta else "--:--")
