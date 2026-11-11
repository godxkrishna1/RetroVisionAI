"""Composite widgets shared across pages."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from .theme import (Badge, Card, Divider, MetricRow, Palette, StatCard,
                    add_shadow, draw_icon_pixmap, make_icon)


def make_combo(adjust_to_contents: bool = True) -> "QComboBox":
    """A combo box that never forces the surrounding card to grow.

    Qt's default size policy lets long item text dictate the minimum width of a
    whole form.  We size to contents for a sane default *hint* but allow the
    widget to shrink, and let long text elide via the view's own width.
    """
    from PySide6.QtWidgets import QComboBox, QSizePolicy

    combo = QComboBox()
    combo.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        if adjust_to_contents else QComboBox.SizeAdjustPolicy.AdjustToContents)
    combo.setMinimumContentsLength(8)
    combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    combo.setMinimumWidth(120)
    return combo


class VideoDropZone(QFrame):
    """Dashed drop target that opens the file dialog on click."""

    file_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None,
                 title: str = "Drop a video here",
                 subtitle: str = "or click to browse  ·  MP4, AVI, MOV, MKV, WMV, MPEG, WEBM, FLV") -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(168)
        self.setObjectName("DropZone")
        self.setStyleSheet(
            f"QFrame#DropZone {{ border: 2px dashed {Palette.BORDER};"
            f" border-radius: 16px; background: rgba(139,92,246,10); }}"
            f"QFrame#DropZone:hover {{ border-color: {Palette.PURPLE};"
            f" background: rgba(139,92,246,23); }}")
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(6)
        icon = QLabel()
        icon.setPixmap(draw_icon_pixmap("upload", 40, Palette.TEXT_DIM, Palette.PURPLE))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(icon)
        self.title = QLabel(title)
        self.title.setObjectName("H3")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.title)
        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName("Muted")
        self.subtitle.setStyleSheet("font-size: 11px;")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.subtitle)

    # -- interactions -----------------------------------------------------
    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._pick_file()
        super().mouseReleaseEvent(event)

    def _pick_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        from utils.config import SUPPORTED_VIDEO_EXTENSIONS
        patterns = " ".join(f"*{ext}" for ext in SUPPORTED_VIDEO_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a video", "", f"Video files ({patterns});;All files (*.*)")
        if path:
            self.file_selected.emit(path)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                f"QFrame#DropZone {{ border: 2px solid {Palette.PURPLE};"
                "border-radius: 16px; background: rgba(139,92,246,36); }}")

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self.setStyleSheet(
            f"QFrame#DropZone {{ border: 2px dashed {Palette.BORDER};"
            " border-radius: 16px; background: rgba(139,92,246,10); }}")

    def dropEvent(self, event) -> None:  # noqa: N802
        self.dragLeaveEvent(event)
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.file_selected.emit(path)
                break


class VideoSummaryCard(Card):
    """Thumbnail + technical metadata for the loaded file."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, padding=16, spacing=10)
        self.setMinimumWidth(330)
        self._thumb = QLabel()
        self._thumb.setFixedHeight(150)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet(
            f"background: {Palette.BG_ALT}; border-radius: 10px;"
            f"color: {Palette.TEXT_MUTED};")
        self._thumb.setText("No video loaded")
        self._layout.addWidget(self._thumb)

        self.title = QLabel("No video loaded")
        self.title.setObjectName("H3")
        self.title.setWordWrap(True)
        self._layout.addWidget(self.title)
        self.path_label = QLabel("")
        self.path_label.setObjectName("Muted")
        self.path_label.setStyleSheet("font-size: 10px;")
        self.path_label.setWordWrap(True)
        self._layout.addWidget(self.path_label)
        self._layout.addWidget(Divider())

        grid = QVBoxLayout()
        grid.setSpacing(6)
        self._rows = {}
        fields = [
            ("resolution", "Resolution"), ("duration", "Duration"),
            ("fps", "Frame rate"), ("frames", "Frames"),
            ("size", "File size"), ("codec", "Video codec"),
            ("audio", "Audio"), ("container", "Container"),
            ("aspect", "Aspect ratio"),
        ]
        for key, label in fields:
            row = MetricRow(label)
            self._rows[key] = row
            grid.addLayout(row)
        self._layout.addLayout(grid)
        self._layout.addStretch(1)

    def set_video(self, info, thumb_path: Optional[str] = None) -> None:
        from utils.file_manager import human_duration, human_size
        if info is None:
            self.clear()
            return
        if thumb_path:
            pix = QPixmap(thumb_path)
            if not pix.isNull():
                self._thumb.setPixmap(pix.scaled(
                    400, 150, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
                self._thumb.setText("")
        self.title.setText(f"{info.path.rsplit('/', 1)[-1].rsplit(chr(92), 1)[-1]}")
        self.path_label.setText(info.path)
        self._rows["resolution"].set_value(f"{info.width} x {info.height}")
        self._rows["duration"].set_value(human_duration(info.duration))
        self._rows["fps"].set_value(f"{info.fps:.2f} fps" +
                                    ("  (VFR)" if info.is_vfr else ""))
        self._rows["frames"].set_value(f"{info.nb_frames:,}")
        self._rows["size"].set_value(human_size(info.size_bytes))
        self._rows["codec"].set_value((info.video_codec or "-").upper())
        if info.has_audio:
            self._rows["audio"].set_value(
                f"{(info.audio_codec or '-').upper()} · {info.audio_channels}ch")
        else:
            self._rows["audio"].set_value("none")
        self._rows["container"].set_value((info.container or "-").upper())
        try:
            ratio = info.width / info.height if info.height else 0
        except ZeroDivisionError:
            ratio = 0
        pretty = {1.333: "4:3", 1.778: "16:9", 1.5: "3:2", 1.25: "5:4",
                  2.39: "2.39:1", 1.85: "1.85:1", 0.5625: "9:16"}.get(
            round(ratio, 3), f"{ratio:.2f}:1" if ratio else "-")
        self._rows["aspect"].set_value(pretty)

    def clear(self) -> None:
        self._thumb.setPixmap(QPixmap())
        self._thumb.setText("No video loaded")
        self.title.setText("No video loaded")
        self.path_label.setText("")
        for row in self._rows.values():
            row.set_value("-")


class SectionTitle(QWidget):
    def __init__(self, title: str, subtitle: str = "",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 4)
        lay.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("H2")
        lay.addWidget(t)
        if subtitle:
            s = QLabel(subtitle)
            s.setObjectName("Muted")
            lay.addWidget(s)


class PageHeader(QWidget):
    """Big page title + optional action buttons on the right."""

    def __init__(self, title: str, subtitle: str = "",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(12)
        col = QVBoxLayout()
        col.setSpacing(3)
        self.title = QLabel(title)
        self.title.setObjectName("H1")
        col.addWidget(self.title)
        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName("Muted")
        col.addWidget(self.subtitle)
        self._lay.addLayout(col)
        self._lay.addStretch(1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        self._lay.addLayout(self.actions)

    def add_action(self, widget: QWidget) -> QWidget:
        self.actions.addWidget(widget)
        return widget


class RecentProjectRow(QFrame):
    """One row in the Dashboard 'Recent projects' list."""

    open_requested = Signal(str)
    folder_requested = Signal(str)

    def __init__(self, record: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.record = record
        self.setObjectName("CardFlat")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 9)
        lay.setSpacing(12)

        icon = QLabel()
        icon.setPixmap(draw_icon_pixmap("film", 22, Palette.PURPLE, Palette.PURPLE))
        icon.setFixedSize(34, 34)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("background: rgba(139,92,246,31); border-radius: 9px;")
        lay.addWidget(icon)

        col = QVBoxLayout()
        col.setSpacing(1)
        name = QLabel(record.get("input_name") or "Untitled")
        name.setStyleSheet("font-weight: 700; font-size: 12px;")
        name.setMaximumWidth(360)
        sub = QLabel(f"{record.get('resolution', '-')} · {record.get('duration', '-')}"
                     f" · {record.get('finished_at', '')}")
        sub.setObjectName("Muted")
        sub.setStyleSheet("font-size: 10px;")
        col.addWidget(name)
        col.addWidget(sub)
        lay.addLayout(col, 1)

        status = (record.get("status") or "unknown").lower()
        kind = {"completed": "green", "failed": "red", "cancelled": "amber",
                "running": "purple"}.get(status, "purple")
        lay.addWidget(Badge(status.upper(), kind))

        open_btn = QPushButton("Open video")
        open_btn.setObjectName("Ghost")
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        open_btn.clicked.connect(lambda: self.open_requested.emit(
            record.get("output_path", "")))
        lay.addWidget(open_btn)

        folder_btn = QPushButton()
        folder_btn.setObjectName("Ghost")
        folder_btn.setIcon(make_icon("folder", Palette.TEXT_DIM))
        folder_btn.setFixedSize(28, 28)
        folder_btn.setToolTip("Show in folder")
        folder_btn.clicked.connect(lambda: self.folder_requested.emit(
            record.get("output_path", "")))
        lay.addWidget(folder_btn)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.open_requested.emit(self.record.get("output_path", ""))
        super().mouseDoubleClickEvent(event)


class NoticeBar(QFrame):
    """Inline info / warning / error banner used at the top of pages."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("CardFlat")
        self.setVisible(False)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(10)
        self.icon = QLabel()
        lay.addWidget(self.icon)
        self.label = QLabel("")
        self.label.setWordWrap(True)
        lay.addWidget(self.label, 1)
        self.close_btn = QPushButton("×")
        self.close_btn.setObjectName("Ghost")
        self.close_btn.setFixedSize(22, 22)
        self.close_btn.clicked.connect(lambda: self.setVisible(False))
        lay.addWidget(self.close_btn)

    def show_message(self, text: str, kind: str = "info") -> None:
        palette = {"info": (Palette.BLUE, "#93C5FD"),
                   "warn": (Palette.AMBER, "#FCD34D"),
                   "error": (Palette.RED, "#FCA5A5"),
                   "success": (Palette.GREEN, "#86EFAC")}
        border, fg = palette.get(kind, palette["info"])
        bg = _rgba(border, 0.13)
        edge = _rgba(border, 0.40)
        self.setStyleSheet(
            f"QFrame#CardFlat {{ background: {bg};"
            f" border: 1px solid {edge}; border-radius: 10px; }}")
        self.icon.setPixmap(draw_icon_pixmap(
            "info" if kind in ("info", "success") else "bolt", 16, fg, fg))
        self.label.setText(text)
        self.label.setStyleSheet(f"color: {fg}; font-size: 12px;")
        self.setVisible(True)


class ProgressPanel(QFrame):
    """Deterministic progress display - driven only by real counters."""

    cancel_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("CardFlat")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(7)

        top = QHBoxLayout()
        self.stage = QLabel("Idle")
        self.stage.setStyleSheet("font-weight: 700; font-size: 13px;")
        top.addWidget(self.stage)
        top.addStretch(1)
        self.percent = QLabel("0%")
        self.percent.setStyleSheet(
            f"font-weight: 800; font-size: 15px; color: {Palette.PURPLE};")
        top.addWidget(self.percent)
        lay.addLayout(top)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(10)
        lay.addWidget(self.bar)

        stats = QGridLayout()
        stats.setHorizontalSpacing(14)
        stats.setVerticalSpacing(4)
        self.frames = MetricRow("Frames")
        self.fps = MetricRow("Processing speed")
        self.elapsed = MetricRow("Elapsed")
        self.eta = MetricRow("Estimated remaining")
        stats.addLayout(self.frames, 0, 0)
        stats.addLayout(self.fps, 0, 1)
        stats.addLayout(self.elapsed, 1, 0)
        stats.addLayout(self.eta, 1, 1)
        lay.addLayout(stats)

    def set_stage(self, text: str) -> None:
        self.stage.setText(text)

    def set_progress(self, done: int, total: int, percent: Optional[float] = None,
                     speed: str = "", elapsed: str = "", eta: str = "") -> None:
        if percent is None:
            percent = (done / total * 100.0) if total else 0.0
        percent = max(0.0, min(100.0, percent))
        self.bar.setValue(int(percent * 10))
        self.percent.setText(f"{percent:.1f}%")
        self.frames.set_value(f"{done:,} / {total:,}" if total else f"{done:,}")
        if speed:
            self.fps.set_value(speed)
        if elapsed:
            self.elapsed.set_value(elapsed)
        self.eta.set_value(eta or "--:--")


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' + alpha (0..1) to a Qt-compatible rgba() string."""
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return hex_color
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{int(round(alpha * 255))})"
