"""Settings page: paths, processing, export defaults, maintenance."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QScrollArea, QSpinBox, QVBoxLayout,
                               QWidget)

from utils.file_manager import free_gb, human_size, sweep_stale_temp
from utils.logger import get_logger

from .theme import Card, Palette, make_icon
from .widgets import NoticeBar, PageHeader

log = get_logger("settings_page")


class SettingsPage(QWidget):
    def __init__(self, ctx, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._build()
        self.load_values()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(26, 22, 26, 20)
        outer.setSpacing(14)

        header = PageHeader("Settings",
                            "Application preferences. Everything is stored in "
                            "config.json next to the app.")
        save_btn = QPushButton("  Save settings")
        save_btn.setObjectName("Primary")
        save_btn.setIcon(make_icon("check", "#FFFFFF"))
        save_btn.clicked.connect(self.save_values)
        header.add_action(save_btn)
        outer.addWidget(header)
        self.notice = NoticeBar()
        outer.addWidget(self.notice)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.form_lay = QVBoxLayout(content)
        self.form_lay.setContentsMargins(0, 0, 8, 0)
        self.form_lay.setSpacing(14)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # ---- FFmpeg ---------------------------------------------------
        ff_card = Card(padding=18, spacing=10)
        ff_card.add(self._title("FFMPEG", "Video decode / encode backend"))
        self.ffmpeg_status = QLabel("")
        self.ffmpeg_status.setWordWrap(True)
        ff_card.add(self.ffmpeg_status)
        form = QFormLayout()
        form.setSpacing(9)
        self.ffmpeg_path = QLineEdit()
        self.ffmpeg_path.setPlaceholderText("Auto-detect (PATH or bin/ffmpeg.exe)")
        form.addRow("ffmpeg.exe", self._with_browse(self.ffmpeg_path, "file"))
        self.ffprobe_path = QLineEdit()
        self.ffprobe_path.setPlaceholderText("Auto-detect")
        form.addRow("ffprobe.exe", self._with_browse(self.ffprobe_path, "file"))
        ff_card.add_layout(form)
        detect = QPushButton("Detect FFmpeg")
        detect.setObjectName("Ghost")
        detect.clicked.connect(self._detect_ffmpeg)
        ff_card.add(detect)
        self.form_lay.addWidget(ff_card)

        # ---- paths ----------------------------------------------------
        path_card = Card(padding=18, spacing=10)
        path_card.add(self._title("FOLDERS", "Models, outputs and temporary files"))
        form2 = QFormLayout()
        form2.setSpacing(9)
        self.models_dir = QLineEdit()
        form2.addRow("Models", self._with_browse(self.models_dir, "dir"))
        self.output_dir = QLineEdit()
        form2.addRow("Output", self._with_browse(self.output_dir, "dir"))
        self.temp_dir = QLineEdit()
        form2.addRow("Temporary", self._with_browse(self.temp_dir, "dir"))
        path_card.add_layout(form2)
        self.disk_info = QLabel("")
        self.disk_info.setObjectName("Muted")
        path_card.add(self.disk_info)
        self.form_lay.addWidget(path_card)

        # ---- processing ----------------------------------------------
        proc_card = Card(padding=18, spacing=10)
        proc_card.add(self._title("PROCESSING", "Device and memory policy"))
        form3 = QFormLayout()
        form3.setSpacing(9)
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (GPU when available)", "auto")
        self.device_combo.addItem("NVIDIA GPU (CUDA)", "cuda")
        self.device_combo.addItem("CPU only", "cpu")
        form3.addRow("Inference device", self.device_combo)
        self.precision_combo = QComboBox()
        self.precision_combo.addItem("FP16 (fast, half VRAM)", "fp16")
        self.precision_combo.addItem("FP32 (max compatibility)", "fp32")
        form3.addRow("Precision", self.precision_combo)
        self.precision_hint = QLabel("")
        self.precision_hint.setObjectName("Muted")
        self.precision_hint.setStyleSheet("font-size: 11px;")
        self.precision_hint.setWordWrap(True)
        form3.addRow("", self.precision_hint)
        self.device_combo.currentIndexChanged.connect(
            lambda *_: self._update_precision_hint())
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 16)
        form3.addRow("Frames per batch", self.batch_spin)
        self.resume_check = QCheckBox("Allow resuming an interrupted job")
        form3.addRow("", self.resume_check)
        self.keep_temp_check = QCheckBox("Keep temporary frames after processing")
        form3.addRow("", self.keep_temp_check)
        proc_card.add_layout(form3)
        self.form_lay.addWidget(proc_card)

        # ---- maintenance --------------------------------------------
        maint = Card(padding=18, spacing=10)
        maint.add(self._title("MAINTENANCE", "Housekeeping - nothing happens "
                                             "automatically without your click"))
        row = QHBoxLayout()
        row.setSpacing(8)
        clean_temp = QPushButton("Clean temporary files")
        clean_temp.setObjectName("Ghost")
        clean_temp.clicked.connect(self._clean_temp)
        row.addWidget(clean_temp)
        open_logs = QPushButton("Open logs folder")
        open_logs.setObjectName("Ghost")
        open_logs.clicked.connect(self._open_logs)
        row.addWidget(open_logs)
        reset_btn = QPushButton("Reset all settings")
        reset_btn.setObjectName("Danger")
        reset_btn.clicked.connect(self._reset)
        row.addWidget(reset_btn)
        row.addStretch(1)
        maint.add_layout(row)
        self.temp_info = QLabel("")
        self.temp_info.setObjectName("Muted")
        maint.add(self.temp_info)
        self.form_lay.addWidget(maint)
        self.form_lay.addStretch(1)

    @staticmethod
    def _title(title: str, subtitle: str = "") -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("CardTitle")
        lay.addWidget(t)
        if subtitle:
            s = QLabel(subtitle)
            s.setObjectName("Muted")
            s.setStyleSheet("font-size: 11px;")
            lay.addWidget(s)
        return w

    def _with_browse(self, line_edit: QLineEdit, kind: str) -> QWidget:
        holder = QWidget()
        lay = QHBoxLayout(holder)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(line_edit, 1)
        btn = QPushButton("Browse")
        btn.setObjectName("Ghost")
        # The global button style adds 16px side padding, so leave room for the
        # label instead of hard-cropping it.
        btn.setFixedWidth(96)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)

        def browse() -> None:
            if kind == "dir":
                path = QFileDialog.getExistingDirectory(self, "Choose folder",
                                                        line_edit.text() or "")
            else:
                path, _ = QFileDialog.getOpenFileName(self, "Choose executable",
                                                      line_edit.text() or "",
                                                      "Executables (*.exe);;All files (*)")
            if path:
                line_edit.setText(path)

        btn.clicked.connect(browse)
        lay.addWidget(btn)
        return holder

    # ------------------------------------------------------------------
    def load_values(self) -> None:
        cfg = self.ctx.config
        self.ffmpeg_path.setText(cfg.get("paths.ffmpeg_path", ""))
        self.ffprobe_path.setText(cfg.get("paths.ffprobe_path", ""))
        self.models_dir.setText(str(cfg.models_dir))
        self.output_dir.setText(str(cfg.output_dir))
        self.temp_dir.setText(str(cfg.temp_dir))
        idx = self.device_combo.findData(cfg.get("processing.device", "auto"))
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        idx = self.precision_combo.findData(cfg.get("processing.precision", "fp16"))
        if idx >= 0:
            self.precision_combo.setCurrentIndex(idx)
        self._update_precision_hint()
        self.batch_spin.setValue(int(cfg.get("processing.max_batch_frames", 4)))
        self.resume_check.setChecked(bool(cfg.get("processing.resume_enabled", True)))
        self.keep_temp_check.setChecked(bool(cfg.get("processing.keep_temp_frames", False)))
        self._detect_ffmpeg(quiet=True)
        self._update_disk_info()

    def _update_precision_hint(self) -> None:
        """Explain what precision actually does on the selected device."""
        try:
            cuda = bool(self.ctx.device_manager.info().device == "cuda")
        except Exception:                                  # noqa: BLE001
            cuda = False
        want = (self.device_combo.currentData() or "auto") != "cpu"
        if not cuda or not want:
            self.precision_hint.setText(
                "CPU inference always runs in FP32 - half precision is only "
                "used on an NVIDIA GPU.")
        else:
            self.precision_hint.setText(
                "FP16 roughly halves VRAM use on Turing or newer GPUs; "
                "FP32 is the safe default.")

    def _update_disk_info(self) -> None:
        try:
            gb = free_gb(Path(self.output_dir.text() or "."))
            self.disk_info.setText(f"Output drive free space: {gb:.1f} GB")
            temp_gb = free_gb(Path(self.temp_dir.text() or "."))
            temp_root = Path(self.temp_dir.text() or "./temp")
            size = sum(p.stat().st_size for p in temp_root.rglob("*")
                       if p.is_file()) if temp_root.exists() else 0
            self.temp_info.setText(
                f"Temporary folder: {human_size(size)} in use  ·  "
                f"{temp_gb:.1f} GB free")
        except Exception:
            pass

    def _detect_ffmpeg(self, quiet: bool = False) -> None:
        ok, message = self.ctx.ffmpeg_status()
        color = Palette.GREEN if ok else Palette.RED
        self.ffmpeg_status.setText(
            f"<span style='color:{color}; font-weight:700'>"
            f"{'FFmpeg ready' if ok else 'FFmpeg missing'}</span>  ·  {message}")
        if not quiet:
            self.notice.show_message(
                "FFmpeg detected." if ok else
                "FFmpeg not found - install it or set the path above.",
                "success" if ok else "error")

    # ------------------------------------------------------------------
    def save_values(self) -> None:
        cfg = self.ctx.config
        cfg.update({
            "paths": {
                "ffmpeg_path": self.ffmpeg_path.text().strip(),
                "ffprobe_path": self.ffprobe_path.text().strip(),
                "models_dir": self.models_dir.text().strip() or "models",
                "output_dir": self.output_dir.text().strip() or "output",
                "temp_dir": self.temp_dir.text().strip() or "temp",
            },
            "processing": {
                "device": self.device_combo.currentData(),
                "precision": self.precision_combo.currentData(),
                "max_batch_frames": self.batch_spin.value(),
                "resume_enabled": self.resume_check.isChecked(),
                "keep_temp_frames": self.keep_temp_check.isChecked(),
            },
        })
        # apply live
        self.ctx.ffmpeg.ffmpeg = cfg.get("paths.ffmpeg_path", "")
        self.ctx.ffmpeg.ffprobe = cfg.get("paths.ffprobe_path", "")
        self.ctx.model_manager.models_dir = cfg.models_dir
        self.ctx.device.preferred = cfg.get("processing.device", "auto")
        self.ctx.device.precision = cfg.get("processing.precision", "fp16")
        self.ctx.device.info(refresh=True)
        self._detect_ffmpeg(quiet=True)
        self._update_disk_info()
        self.notice.show_message("Settings saved.", "success")

    def _clean_temp(self) -> None:
        temp = self.ctx.config.temp_dir
        removed = sweep_stale_temp(temp, max_age_hours=0)
        self._update_disk_info()
        self.notice.show_message(
            f"Removed {removed} temporary workspace(s).", "success")

    def _open_logs(self) -> None:
        import os
        import subprocess
        import sys
        logs = str(self.ctx.config.logs_dir)
        try:
            if os.name == "nt":
                os.startfile(logs)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", logs])
            else:
                subprocess.Popen(["xdg-open", logs])
        except Exception as exc:
            self.notice.show_message(f"Could not open logs: {exc}", "error")

    def _reset(self) -> None:
        answer = QMessageBox.question(
            self, "Reset settings",
            "Restore every setting to its default value? Downloaded models "
            "and output videos are not affected.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.ctx.config.reset()
            self.load_values()
            self.notice.show_message("Settings reset to defaults.", "success")

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._update_disk_info()
