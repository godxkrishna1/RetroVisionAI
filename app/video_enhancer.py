"""Video Enhancer page: load a video, configure restoration, run, view results."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QProgressBar, QPushButton, QRadioButton, QScrollArea,
                               QSizePolicy, QSpinBox, QSplitter, QVBoxLayout,
                               QWidget)

from core.job import EnhancementSettings, JobProgress, JobResult, JobSpec, JobStage
from core.pipeline import target_size
from utils.config import INTENSITIES, RESOLUTION_LABELS, TARGET_RESOLUTIONS
from utils.ffmpeg import EncodeSettings
from utils.file_manager import human_duration, human_size, is_video_file, unique_path
from utils.logger import get_logger

from .theme import (Badge, Card, Divider, MetricRow, Palette, add_shadow,
                    draw_icon_pixmap, make_icon)
from .video_preview import PreviewPanel
from .widgets import (NoticeBar, PageHeader, ProgressPanel, VideoDropZone,
                      VideoSummaryCard, make_combo)

log = get_logger("enhancer")

RESOLUTION_ORDER = ["original", "720p", "1080p", "2k", "4k"]
ENHANCEMENT_TOGGLES = [
    # (attr, label, tooltip, needs_model)
    ("use_super_resolution", "AI Super Resolution",
     "Real-ESRGAN neural upscaling - the core restoration engine", True),
    ("use_face_restoration", "Face Restoration",
     "GFPGAN v1.4 repairs faces, eyes and skin (requires the 333 MB model)", True),
    ("use_denoise", "Noise Reduction",
     "Non-local-means denoise for tape hiss / sensor noise", False),
    ("use_deblur", "Deblurring", "Edge-aware deconvolution-style sharpening", False),
    ("use_sharpen", "Sharpness Enhancement", "Unsharp mask for soft transfers", False),
    ("use_color_restore", "Color Restoration",
     "White balance, saturation and black-level recovery", False),
    ("use_brightness_fix", "Brightness Correction", "Lift dim, underexposed footage", False),
    ("use_contrast_fix", "Contrast Enhancement", "CLAHE contrast recovery", False),
    ("use_grain_reduction", "Film Grain Reduction", "Bilateral grain smoothing", False),
    ("use_scratch_reduction", "Scratch Reduction",
     "Detect and inpaint thin film scratches (best effort)", False),
    ("use_interpolation", "Motion Interpolation",
     "RIFE 4.x - generate in-between frames for smoother motion (needs model)", True),
    ("use_temporal_restore", "Temporal Restoration (BasicVSR++)",
     "Advanced multi-frame restoration - slow, needs both models", True),
]


class EnhancementToggle(QFrame):
    """Toggle row with label, tooltip and an intensity combo."""

    changed = Signal()

    def __init__(self, attr: str, label: str, tooltip: str,
                 intensities: bool = False, parent: Optional[QWidget] = None
                 ) -> None:
        super().__init__(parent)
        self.attr = attr
        self.setObjectName("CardFlat")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(8)
        from PySide6.QtWidgets import QCheckBox
        self.check = QCheckBox(label)
        self.check.setToolTip(tooltip)
        self.check.setMinimumWidth(180)
        self.check.toggled.connect(self._on_toggle)
        lay.addWidget(self.check, 1)
        self.combo: Optional[QComboBox] = None
        if intensities:
            self.combo = make_combo()
            self.combo.addItems(["Low", "Medium", "High"])
            self.combo.setCurrentText("Medium")
            self.combo.setFixedWidth(92)
            self.combo.currentTextChanged.connect(lambda _: self.changed.emit())
            lay.addWidget(self.combo)
        self.model_badge = QLabel("")
        self.model_badge.setVisible(False)
        lay.addWidget(self.model_badge)

    def _on_toggle(self, checked: bool) -> None:
        bg = "rgba(139,92,246,26)" if checked else Palette.SURFACE_2
        border = "rgba(139,92,246,115)" if checked else Palette.BORDER_SOFT
        self.setStyleSheet(
            f"QFrame#CardFlat {{ background: {bg};"
            f" border: 1px solid {border}; border-radius: 12px; }}")
        if self.combo is not None:
            self.combo.setEnabled(checked)
        self.changed.emit()

    # -- state ---------------------------------------------------------
    def value(self) -> bool:
        return self.check.isChecked()

    def set_value(self, enabled: bool) -> None:
        self.check.setChecked(enabled)
        self._on_toggle(enabled)

    def intensity(self) -> str:
        return (self.combo.currentText().lower() if self.combo else "medium")

    def set_intensity(self, level: str) -> None:
        if self.combo:
            self.combo.setCurrentText((level or "medium").capitalize())


class ResolutionSelector(QWidget):
    """Four-radio resolution picker with computed output size hint."""

    changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self._buttons: Dict[str, QRadioButton] = {}
        for i, key in enumerate(RESOLUTION_ORDER):
            rb = QRadioButton(RESOLUTION_LABELS[key])
            rb.setObjectName("ResRadio")
            rb.setCursor(Qt.CursorShape.PointingHandCursor)
            self.group.addButton(rb)
            self._buttons[key] = rb
            grid.addWidget(rb, i // 3, i % 3)
            rb.toggled.connect(lambda checked, k=key: checked and self.changed.emit(k))
        self._buttons["1080p"].setChecked(True)
        lay.addLayout(grid)
        self.hint = QLabel("")
        self.hint.setObjectName("Muted")
        self.hint.setStyleSheet("font-size: 11px;")
        lay.addWidget(self.hint)

    def current(self) -> str:
        for key, rb in self._buttons.items():
            if rb.isChecked():
                return key
        return "1080p"

    def set_current(self, key: str) -> None:
        self._buttons.get(key, self._buttons["1080p"]).setChecked(True)

    def update_hint(self, src_w: int, src_h: int) -> None:
        if not src_w or not src_h:
            self.hint.setText("")
            return
        from utils.ffmpeg import VideoInfo
        fake = VideoInfo(path="", width=src_w, height=src_h)
        w, h = target_size(fake, self.current())
        presets = TARGET_RESOLUTIONS
        label = RESOLUTION_LABELS[self.current()]
        if presets.get(self.current()):
            self.hint.setText(
                f"Output: {w} x {h} ({label})  ·  source {src_w} x {src_h}  ·  "
                f"aspect preserved")
        else:
            self.hint.setText(f"Output: {w} x {h} (unchanged resolution)")


class VideoEnhancerPage(QWidget):
    """The main workspace page."""

    request_preview = Signal(str, str)      # (original, output)

    def __init__(self, ctx, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.info = None
        self._thumb_path: Optional[str] = None
        self._job_started: Optional[float] = None
        self._last_frames_done = 0
        self._frame_times: List[float] = []
        self._toggle_widgets: Dict[str, EnhancementToggle] = {}
        self._last_failed = False
        self._build()
        self._load_settings_from_config()
        self._connect_jobs()
        self._update_model_availability()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 20)
        root.setSpacing(14)

        header = PageHeader("Video Enhancer",
                            "Import footage, choose the restoration stack and export "
                            "in 720p, 1080p, 2K or 4K.")
        self.status_badge = Badge("IDLE", "purple")
        header.add_action(self.status_badge)
        self.import_btn = QPushButton("  Import Video")
        self.import_btn.setObjectName("Primary")
        self.import_btn.setIcon(make_icon("upload", "#FFFFFF"))
        self.import_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.import_btn.clicked.connect(self.browse_video)
        header.add_action(self.import_btn)
        root.addWidget(header)

        self.notice = NoticeBar()
        root.addWidget(self.notice)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # ---------------- left: source + settings ----------------------
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(12)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(470)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        splitter.addWidget(left_scroll)

        self.drop_zone = VideoDropZone(title="Import a video to restore",
                                       subtitle="Drag & drop or click  ·  MP4, AVI, "
                                                "MOV, MKV, WMV, MPEG, WEBM, FLV")
        self.drop_zone.file_selected.connect(self.load_video)
        left_lay.addWidget(self.drop_zone)

        self.summary = VideoSummaryCard()
        left_lay.addWidget(self.summary)

        res_card = Card(padding=16, spacing=8)
        res_card.add(self._section_label("OUTPUT RESOLUTION"))
        self.resolution = ResolutionSelector()
        self.resolution.changed.connect(lambda _: self._refresh_preview_estimate())
        res_card.add(self.resolution)
        left_lay.addWidget(res_card)

        ai_card = Card(padding=16, spacing=8)
        ai_card.add(self._section_label("AI MODELS"))
        form = QFormLayout()
        form.setSpacing(9)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.model_combo = make_combo()
        self.model_combo.currentIndexChanged.connect(
            lambda _: self._refresh_preview_estimate())
        form.addRow("Upscaler", self.model_combo)
        self.interp_combo = make_combo()
        self.interp_combo.addItem("RIFE 4.7", "rife47")
        self.interp_combo.addItem("RIFE 4.9", "rife49")
        form.addRow("Interpolation model", self.interp_combo)
        self.face_strength = QDoubleSpinBox()
        self.face_strength.setRange(0.0, 1.0)
        self.face_strength.setSingleStep(0.05)
        self.face_strength.setValue(0.8)
        form.addRow("Face strength", self.face_strength)
        self.temporal_window = QSpinBox()
        self.temporal_window.setRange(2, 15)
        self.temporal_window.setValue(5)
        form.addRow("BasicVSR++ window", self.temporal_window)
        ai_card.add_layout(form)
        self.model_hint = QLabel("")
        self.model_hint.setObjectName("Muted")
        self.model_hint.setWordWrap(True)
        self.model_hint.setStyleSheet("font-size: 11px;")
        ai_card.add(self.model_hint)
        left_lay.addWidget(ai_card)

        enh_card = Card(padding=16, spacing=8)
        enh_card.add(self._section_label("ENHANCEMENT PIPELINE"))
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, (attr, label, tip, needs_model) in enumerate(ENHANCEMENT_TOGGLES):
            intensities = attr not in ("use_super_resolution", "use_face_restoration",
                                       "use_interpolation", "use_temporal_restore")
            wdg = EnhancementToggle(attr, label, tip, intensities=intensities)
            wdg.changed.connect(self._settings_changed)
            self._toggle_widgets[attr] = wdg
            wdg._on_toggle(wdg.value())
            grid.addWidget(wdg, i, 0)
        enh_card.add_layout(grid)
        left_lay.addWidget(enh_card)

        export_card = Card(padding=16, spacing=8)
        export_card.add(self._section_label("EXPORT"))
        form2 = QFormLayout()
        form2.setSpacing(9)
        self.codec_combo = make_combo()
        self.codec_combo.addItem("MP4 · H.264", "h264")
        self.codec_combo.addItem("MP4 · H.265 / HEVC", "h265")
        form2.addRow("Codec", self.codec_combo)
        self.encoder_combo = make_combo()
        self.encoder_combo.addItem("Auto (NVENC if available)", "auto")
        self.encoder_combo.addItem("NVIDIA NVENC (GPU)", "nvenc")
        self.encoder_combo.addItem("CPU (libx264 / libx265)", "cpu")
        form2.addRow("Encoder", self.encoder_combo)
        self.quality_combo = make_combo()
        for label, key in (("High quality (CRF 19)", "high"),
                           ("Balanced (CRF 22)", "medium"),
                           ("Small file (CRF 26)", "low"),
                           ("Lossless (very large)", "lossless")):
            self.quality_combo.addItem(label, key)
        form2.addRow("Quality", self.quality_combo)
        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)
        self.crf_spin.setValue(19)
        self.crf_spin.setToolTip("0 = lossless. Lower = better quality, bigger file.")
        form2.addRow("Custom CRF", self.crf_spin)
        self.quality_combo.currentIndexChanged.connect(self._sync_crf)
        self.audio_check = make_combo()
        self.audio_check.addItem("Preserve original audio (copy)", True)
        self.audio_check.addItem("Re-encode AAC 192 kbps", False)
        form2.addRow("Audio", self.audio_check)

        out_row = QHBoxLayout()
        self.output_edit = QLineEdit(str(self.ctx.config.output_dir))
        self.output_edit.setReadOnly(True)
        browse = QPushButton("Choose...")
        browse.setObjectName("Ghost")
        browse.clicked.connect(self._choose_output_dir)
        out_row.addWidget(self.output_edit, 1)
        out_row.addWidget(browse)
        form2.addRow("Output folder", out_row)
        export_card.add_layout(form2)
        self.estimate_label = QLabel("")
        self.estimate_label.setObjectName("Muted")
        self.estimate_label.setWordWrap(True)
        self.estimate_label.setStyleSheet("font-size: 11px;")
        export_card.add(self.estimate_label)
        left_lay.addWidget(export_card)
        left_lay.addStretch(1)

        # ---------------- right: preview + controls -------------------
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(12)

        self.preview = PreviewPanel(self.ctx)
        right_lay.addWidget(self.preview, 1)

        control = Card(padding=16, spacing=10)
        self.progress_panel = ProgressPanel()
        self.progress_panel.setStyleSheet(
            f"QFrame#CardFlat {{ background: {Palette.SURFACE_2};"
            f" border: 1px solid {Palette.BORDER_SOFT}; border-radius: 12px; }}")
        control.add(self.progress_panel)
        btns = QHBoxLayout()
        btns.setSpacing(8)
        self.start_btn = QPushButton("  Start Enhancement")
        self.start_btn.setObjectName("Primary")
        self.start_btn.setIcon(make_icon("play", "#FFFFFF"))
        self.start_btn.setMinimumHeight(42)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self.start_job)
        btns.addWidget(self.start_btn, 2)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("Danger")
        self.cancel_btn.setMinimumHeight(42)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_job)
        btns.addWidget(self.cancel_btn, 1)
        self.retry_btn = QPushButton("Retry")
        self.retry_btn.setObjectName("Ghost")
        self.retry_btn.setMinimumHeight(42)
        self.retry_btn.setEnabled(False)
        self.retry_btn.clicked.connect(self.start_job)
        btns.addWidget(self.retry_btn, 1)
        control.add_layout(btns)
        right_lay.addWidget(control)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([560, 760])

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("CardTitle")
        return lbl

    # ------------------------------------------------------------------
    def _connect_jobs(self) -> None:
        jm = self.ctx.job_manager
        jm.job_started.connect(self._on_job_started)
        jm.job_progress.connect(self._on_progress)
        jm.job_stage.connect(self._on_stage)
        jm.job_finished.connect(self._on_finished)
        jm.job_failed.connect(self._on_failed)
        jm.job_cancelled.connect(self._on_cancelled)
        jm.queue_changed.connect(self._refresh_buttons)

    # ------------------------------------------------------------------
    def load_video(self, path: str) -> bool:
        path = str(path)
        if not is_video_file(Path(path)):
            self.notice.show_message(
                "Unsupported file type. Supported: MP4, AVI, MOV, MKV, WMV, "
                "MPEG, MPG, WEBM, FLV.", "warn")
            return False
        try:
            info = self.ctx.ffmpeg.probe(path)
        except Exception as exc:
            self.notice.show_message(f"Could not read this video: {exc}", "error")
            return False
        self.info = info
        thumb_dir = self.ctx.config.temp_dir / "thumbs"
        thumb = None
        try:
            thumb = self.ctx.ffmpeg.make_thumbnail(path, thumb_dir / f"{Path(path).stem}.jpg")
        except Exception:
            pass
        self._thumb_path = str(thumb) if thumb else None
        self.summary.set_video(info, self._thumb_path)
        self.drop_zone.setVisible(False)
        self.resolution.update_hint(info.width, info.height)
        self.preview.set_source(path, info)
        self.ctx.pending_video = path
        self.notice.setVisible(False)
        self._refresh_preview_estimate()
        self._refresh_buttons()
        log.info("Loaded %s for enhancement", Path(path).name)
        return True

    def browse_video(self) -> None:
        from utils.config import SUPPORTED_VIDEO_EXTENSIONS
        patterns = " ".join(f"*{e}" for e in SUPPORTED_VIDEO_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Import video", "", f"Video files ({patterns});;All files (*.*)")
        if path:
            self.load_video(path)

    def _choose_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose output folder", str(self.ctx.config.output_dir))
        if folder:
            self.ctx.config.set("paths.output_dir", folder)
            self.output_edit.setText(folder)

    # ------------------------------------------------------------------
    def _settings_changed(self) -> None:
        self._refresh_preview_estimate()
        self._update_model_availability()

    def _sync_crf(self) -> None:
        mapping = {"high": 19, "medium": 22, "low": 26, "lossless": 0}
        self.crf_spin.setValue(mapping.get(self.quality_combo.currentData(), 19))

    def _refresh_preview_estimate(self) -> None:
        # may be called by toggle signals while the page is still building
        if not hasattr(self, "estimate_label"):
            return
        if self.info is None:
            self.estimate_label.setText("")
            return
        self.resolution.update_hint(self.info.width, self.info.height)
        from utils.ffmpeg import VideoInfo
        w, h = target_size(self.info, self.resolution.current())
        fps = self.info.fps
        if self._toggle_widgets["use_interpolation"].value():
            factor = 2
            fps *= factor
        frames = int(round(self.info.duration_seconds * fps))
        mp = w * h / 1_000_000
        # rough size estimate at CRF 19 (~ H.264 bitrate heuristic)
        bitrate_mbps = max(2.0, mp * 0.55)
        est_mb = bitrate_mbps * self.info.duration_seconds / 8
        self.estimate_label.setText(
            f"{w} x {h}  ·  {fps:.2f} fps  ·  ~{frames:,} output frames  ·  "
            f"estimated {human_size(est_mb * 1048576)} at CRF 19")

    def _update_model_availability(self) -> None:
        if not hasattr(self, "model_hint") or \
                "use_temporal_restore" not in self._toggle_widgets:
            return
        mm = self.ctx.model_manager
        # upscaler combo shows installed + not-installed (marked)
        current = self.model_combo.currentData()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for status in mm.model_status():
            if status.category != "upscale":
                continue
            label = status.title + ("" if status.installed else "  -  not installed")
            self.model_combo.addItem(label, status.key)
        if current:
            idx = self.model_combo.findData(current)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        self.model_combo.blockSignals(False)

        missing = []
        if self._toggle_widgets["use_face_restoration"].value() and \
                not mm.is_installed("gfpgan_v14"):
            missing.append("GFPGAN v1.4")
        if self._toggle_widgets["use_interpolation"].value() and \
                not (mm.is_installed(self.interp_combo.currentData())):
            missing.append(self.interp_combo.currentText())
        if self._toggle_widgets["use_temporal_restore"].value() and \
                not (mm.is_installed("basicvsr_pp") and mm.is_installed("spynet")):
            missing.append("BasicVSR++ + SPyNet")
        key = self.model_combo.currentData()
        if key and not mm.is_installed(key):
            missing.append(self.model_combo.currentText().split("  -")[0])
        if missing:
            self.model_hint.setText("Not installed: " + ", ".join(missing) +
                                    "  -  open AI Model Manager to download.")
            self.model_hint.setStyleSheet("color: #FCD34D; font-size: 11px;")
        else:
            self.model_hint.setText("All selected models are installed.")
            self.model_hint.setStyleSheet(
                f"color: {Palette.TEXT_MUTED}; font-size: 11px;")

    # ------------------------------------------------------------------
    def collect_settings(self) -> EnhancementSettings:
        t = self._toggle_widgets
        enh = EnhancementSettings(
            target_resolution=self.resolution.current(),
            ai_model=self.model_combo.currentData() or "realesrgan_x2plus",
            use_super_resolution=t["use_super_resolution"].value(),
            use_face_restoration=t["use_face_restoration"].value(),
            face_strength=self.face_strength.value(),
            use_denoise=t["use_denoise"].value(),
            denoise_strength=t["use_denoise"].intensity(),
            use_deblur=t["use_deblur"].value(),
            deblur_strength=t["use_deblur"].intensity(),
            use_sharpen=t["use_sharpen"].value(),
            sharpen_strength=t["use_sharpen"].intensity(),
            use_color_restore=t["use_color_restore"].value(),
            color_strength=t["use_color_restore"].intensity(),
            use_brightness_fix=t["use_brightness_fix"].value(),
            brightness_strength=t["use_brightness_fix"].intensity(),
            use_contrast_fix=t["use_contrast_fix"].value(),
            contrast_strength=t["use_contrast_fix"].intensity(),
            use_grain_reduction=t["use_grain_reduction"].value(),
            grain_strength=t["use_grain_reduction"].intensity(),
            use_scratch_reduction=t["use_scratch_reduction"].value(),
            scratch_strength=t["use_scratch_reduction"].intensity(),
            use_interpolation=t["use_interpolation"].value(),
            interpolation_factor=2,
            interpolation_model=self.interp_combo.currentData() or "rife47",
            use_temporal_restore=t["use_temporal_restore"].value(),
            temporal_window=self.temporal_window.value())
        enh.validate()
        return enh

    def apply_settings(self, settings: Dict) -> None:
        """Apply a preset dict (used by the dashboard quick actions)."""
        res = settings.get("target_resolution")
        if res:
            self.resolution.set_current(res)
        model = settings.get("ai_model")
        if model:
            idx = self.model_combo.findData(model)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        for attr, widget in self._toggle_widgets.items():
            if attr in settings:
                widget.set_value(bool(settings[attr]))
        self._refresh_preview_estimate()

    def _load_settings_from_config(self) -> None:
        cfg = self.ctx.config
        self.resolution.set_current(cfg.get("enhancement.target_resolution", "1080p"))
        for attr, widget in self._toggle_widgets.items():
            widget.set_value(bool(cfg.get(f"enhancement.{attr}",
                                          widget.value())))
            if widget.combo is not None:
                strength_key = {
                    "use_denoise": "denoise_strength",
                    "use_deblur": "deblur_strength",
                    "use_sharpen": "sharpen_strength",
                    "use_color_restore": "color_strength",
                    "use_brightness_fix": "brightness_strength",
                    "use_contrast_fix": "contrast_strength",
                    "use_grain_reduction": "grain_strength",
                    "use_scratch_reduction": "scratch_strength",
                }.get(attr)
                if strength_key:
                    widget.set_intensity(cfg.get(f"enhancement.{strength_key}", "medium"))
        model_idx = self.model_combo.findData(
            cfg.get("enhancement.ai_model", "realesrgan_x2plus"))
        if model_idx >= 0:
            self.model_combo.setCurrentIndex(model_idx)
        self.face_strength.setValue(float(cfg.get("enhancement.face_strength", 0.8)))
        self.temporal_window.setValue(int(cfg.get("enhancement.temporal_window", 5)))
        idx = self.interp_combo.findData(cfg.get("enhancement.interpolation_model", "rife47"))
        if idx >= 0:
            self.interp_combo.setCurrentIndex(idx)
        codec_idx = self.codec_combo.findData(cfg.get("export.codec", "h264"))
        if codec_idx >= 0:
            self.codec_combo.setCurrentIndex(codec_idx)
        enc_idx = self.encoder_combo.findData(cfg.get("export.encoder", "auto"))
        if enc_idx >= 0:
            self.encoder_combo.setCurrentIndex(enc_idx)
        q_idx = self.quality_combo.findData(cfg.get("export.quality", "high"))
        if q_idx >= 0:
            self.quality_combo.setCurrentIndex(q_idx)
        self.crf_spin.setValue(int(cfg.get("export.crf", 19)))

    def save_settings_to_config(self) -> None:
        enh = self.collect_settings()
        self.ctx.config.update({"enhancement": enh.to_dict(),
                                "export": {
                                    "codec": self.codec_combo.currentData(),
                                    "encoder": self.encoder_combo.currentData(),
                                    "quality": self.quality_combo.currentData(),
                                    "crf": self.crf_spin.value()}})

    # ------------------------------------------------------------------
    def build_encode_settings(self) -> EncodeSettings:
        quality = self.quality_combo.currentData()
        audio_copy = bool(self.audio_check.currentData())
        return EncodeSettings(
            codec=self.codec_combo.currentData(),
            encoder=self.encoder_combo.currentData(),
            quality=quality,
            crf=self.crf_spin.value(),
            bitrate_kbps=0,
            preset="slow" if quality in ("high", "lossless") else "medium",
            pix_fmt="yuv420p",
            audio_codec="aac",
            audio_bitrate_kbps=192,
            copy_audio=audio_copy)

    def start_job(self) -> None:
        if self.info is None:
            self.notice.show_message("Import a video first.", "warn")
            return
        if self.ctx.job_manager.busy:
            self.notice.show_message("Another job is running - wait for it to finish "
                                     "or cancel it first.", "warn")
            return
        enh = self.collect_settings()
        warnings = enh.validate()
        missing = []
        if enh.use_super_resolution:
            missing += self.ctx.model_manager.missing_for([enh.ai_model])
        if enh.use_face_restoration:
            missing += self.ctx.model_manager.missing_for(["gfpgan_v14"])
        if enh.use_interpolation:
            missing += self.ctx.model_manager.missing_for([enh.interpolation_model])
        if enh.use_temporal_restore:
            missing += self.ctx.model_manager.missing_for(["basicvsr_pp"])
        # de-duplicate while preserving order
        seen_keys, unique = set(), []
        for spec_m in missing:
            if spec_m.key not in seen_keys:
                seen_keys.add(spec_m.key)
                unique.append(spec_m)
        missing = unique
        if missing:
            names = ", ".join(sorted({m.title for m in missing}))
            self.notice.show_message(
                f"Required model(s) not installed: {names}. Open AI Model "
                "Manager to download them (one click, official source).", "warn")
            return

        out_dir = Path(self.ctx.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self.info.path).stem
        from utils.ffmpeg import VideoInfo
        w, h = target_size(self.info, enh.target_resolution)
        suffix = f"{enh.target_resolution}_{w}x{h}"
        out_path = unique_path(out_dir, f"{stem}_{suffix}_restored")
        spec = JobSpec(input_path=self.info.path, enhancement=enh,
                       encode=self.build_encode_settings(),
                       output_path=str(out_path),
                       label=f"{stem} -> {enh.target_resolution}")
        self.save_settings_to_config()
        self.ctx.job_manager.enqueue(spec)
        self.ctx.job_manager.start()
        if warnings:
            self.notice.show_message(" ".join(warnings), "info")

    def cancel_job(self) -> None:
        self.ctx.job_manager.cancel_current()

    # ------------------------------------------------------------------
    def _refresh_buttons(self) -> None:
        busy = self.ctx.job_manager.busy
        self.start_btn.setEnabled(not busy and self.info is not None)
        self.cancel_btn.setEnabled(busy)
        self.retry_btn.setEnabled(not busy and self._last_failed)
        self.import_btn.setEnabled(not busy)
        self.status_badge.setText("RUNNING" if busy else
                                  ("READY" if self.info else "IDLE"))
        self.status_badge.set_kind("green" if self.info and not busy else
                                   ("purple" if busy else "purple"))

    _last_failed = False

    def _on_job_started(self, spec) -> None:
        self._job_started = time.time()
        self._frame_times = []
        self._last_failed = False
        self.progress_panel.set_stage(JobStage.QUEUED)
        self.progress_panel.set_progress(0, 0, 0.0)
        self.preview.set_processing(True)
        self._refresh_buttons()

    def _on_stage(self, stage: str, index: int, count: int) -> None:
        self.progress_panel.set_stage(f"{stage}  ({index}/{count})")

    def _on_progress(self, p: JobProgress) -> None:
        from utils.file_manager import human_duration as hd
        speed = f"{p.fps:.2f} fps" if p.fps else "-"
        self.progress_panel.set_stage(p.stage + (
            f"  ·  {p.message}" if p.message else ""))
        self.progress_panel.set_progress(
            p.frames_done, p.frames_total, p.percent, speed=speed,
            elapsed=hd(p.elapsed),
            eta=hd(p.eta) if p.eta is not None else "--:--")

    def _on_finished(self, result: JobResult) -> None:
        self._last_failed = False
        self.preview.set_processing(False)
        self.ctx.last_output = result.output_path
        if result.output_path:
            self.preview.set_output(result.output_path)
        self.status_badge.setText("DONE")
        self.status_badge.set_kind("green")
        self.notice.show_message(
            f"Restoration complete - saved to {result.output_path}", "success")
        self._refresh_buttons()

    def _on_failed(self, spec, message: str) -> None:
        self._last_failed = True
        self.preview.set_processing(False)
        self.status_badge.setText("FAILED")
        self.status_badge.set_kind("red")
        self.notice.show_message(f"Processing failed: {message}", "error")
        self._refresh_buttons()

    def _on_cancelled(self, spec) -> None:
        self.preview.set_processing(False)
        self.status_badge.setText("CANCELLED")
        self.status_badge.set_kind("amber")
        self.notice.show_message(
            "Job cancelled. Temporary frames are kept so you can retry and "
            "resume from the last finished frame.", "info")
        self._refresh_buttons()

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        try:
            self.save_settings_to_config()
        except Exception:
            pass
