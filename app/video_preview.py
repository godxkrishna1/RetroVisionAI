"""Video preview panel: original player, enhanced player and A/B comparison.

Playback strategy
-----------------
Qt Multimedia gives us real, frame-accurate playback of both files with the
*same* clock.  When Qt Multimedia is unavailable (some minimal Windows
installs lack the media back-end) we fall back to an FFmpeg-extracted frame
at the requested timestamp, refreshed by a timer - never a dead panel.

The comparison mode uses a draggable split with a wipe divider; both sides
are rendered from synchronised players so the wipe compares identical frames.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QPoint, QRect, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QSizePolicy, QSlider, QStackedLayout,
                               QVBoxLayout, QWidget)

from utils.file_manager import human_duration
from utils.logger import get_logger

from .theme import Card, Palette, draw_icon_pixmap, make_icon

log = get_logger("preview")

HAVE_QTMM = False
try:  # Qt Multimedia is part of PySide6-Essentials on Windows
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    HAVE_QTMM = True
except Exception as exc:  # pragma: no cover - headless CI
    log.warning("Qt Multimedia unavailable (%s) - using frame preview", exc)


class VideoSurface(QLabel):
    """Displays either a live QVideoWidget child or an FFmpeg-grabbed frame."""

    def __init__(self, label: str = "ORIGINAL", parent: Optional[QWidget] = None
                 ) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 200)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #05060B; border-radius: 10px;")
        self._tag = label
        self._pixmap: Optional[QPixmap] = None
        self.setText("Nothing loaded")

    def set_frame(self, image: QImage) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self._repaint_frame()

    def set_message(self, text: str) -> None:
        self._pixmap = None
        self.setText(text)

    def _repaint_frame(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.size() * self.devicePixelRatioF(),
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled)
        self.setText("")

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._repaint_frame()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        rect = QRect(10, 10, 82, 24)
        p.drawRoundedRect(rect, 6, 6)
        p.setPen(QColor(236, 237, 245))
        f = p.font()
        f.setPointSize(8)
        f.setBold(True)
        p.setFont(f)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._tag)


class CompareWidget(QWidget):
    """Before/after wipe: original on the left, enhanced on the right."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._before: Optional[QPixmap] = None
        self._after: Optional[QPixmap] = None
        self._split = 0.5
        self._dragging = False
        self.setMinimumSize(400, 240)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.SplitHCursor)

    def set_images(self, before: Optional[QPixmap], after: Optional[QPixmap]) -> None:
        self._before = before
        self._after = after
        self.update()

    def set_split(self, value: float) -> None:
        self._split = max(0.0, min(1.0, value))
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._to_pos(event.position().x())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._dragging:
            self._to_pos(event.position().x())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._dragging = False

    def _to_pos(self, x: float) -> None:
        if self.width() > 0:
            self.set_split(x / self.width())

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#05060B"))
        if self._before is None or self._after is None:
            p.setPen(QColor(Palette.TEXT_MUTED))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Run an enhancement to compare before / after")
            return

        # same aspect -> scale both to the widget box
        box = self.rect()
        after_scaled = self._after.scaled(
            box.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        before_scaled = self._before.scaled(
            after_scaled.size(), Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

        x0 = (box.width() - after_scaled.width()) // 2
        y0 = (box.height() - after_scaled.height()) // 2
        target = QRect(x0, y0, after_scaled.width(), after_scaled.height())

        # right side: enhanced (full)
        p.drawPixmap(target, after_scaled)
        # left side: original, clipped to split
        split_x = target.left() + int(target.width() * self._split)
        clip = QRect(target.left(), target.top(), split_x - target.left(),
                     target.height())
        p.save()
        p.setClipRect(clip)
        p.drawPixmap(target, before_scaled)
        p.restore()

        # divider
        pen = QPen(QColor(Palette.PURPLE), 2)
        p.setPen(pen)
        p.drawLine(split_x, target.top(), split_x, target.bottom())
        handle_y = target.center().y()
        p.setBrush(QColor(Palette.PURPLE))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPoint(split_x, handle_y), 11, 11)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255))
        p.drawPolygon(QPoint(split_x - 5, handle_y - 3), QPoint(split_x - 1, handle_y),
                      QPoint(split_x - 5, handle_y + 3))
        p.drawPolygon(QPoint(split_x + 5, handle_y - 3), QPoint(split_x + 1, handle_y),
                      QPoint(split_x + 5, handle_y + 3))

        # labels
        p.setPen(QColor(236, 237, 245))
        f = p.font()
        f.setBold(True)
        f.setPointSize(8)
        p.setFont(f)
        p.setBrush(QColor(0, 0, 0, 150))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRect(target.left() + 10, target.top() + 10, 62, 22), 6, 6)
        p.drawRoundedRect(QRect(target.right() - 72, target.top() + 10, 62, 22), 6, 6)
        p.setPen(QColor(236, 237, 245))
        p.drawText(QRect(target.left() + 10, target.top() + 10, 62, 22),
                   Qt.AlignmentFlag.AlignCenter, "BEFORE")
        p.drawText(QRect(target.right() - 72, target.top() + 10, 62, 22),
                   Qt.AlignmentFlag.AlignCenter, "AFTER")


class PreviewPanel(Card):
    """Holds the player area, mode switcher and transport controls."""

    def __init__(self, ctx, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, padding=14, spacing=10)
        self.ctx = ctx
        self.source_path: Optional[str] = None
        self.output_path: Optional[str] = None
        self.info = None
        self._mode = "compare"           # original | enhanced | compare
        self._position = 0               # ms
        self._duration = 0
        self._playing = False

        # ---- top row: mode buttons ---------------------------------
        top = QHBoxLayout()
        top.setSpacing(6)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Before / After", "compare")
        self.mode_combo.addItem("Original only", "original")
        self.mode_combo.addItem("Enhanced only", "enhanced")
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        top.addWidget(self.mode_combo)
        top.addStretch(1)
        self.info_label = QLabel("")
        self.info_label.setObjectName("Muted")
        self.info_label.setStyleSheet("font-size: 11px;")
        top.addWidget(self.info_label)
        self._layout.addLayout(top)

        # ---- display stack -----------------------------------------
        self.stack = QStackedLayout()
        holder = QWidget()
        holder.setLayout(self.stack)
        holder.setMinimumHeight(300)
        self._layout.addWidget(holder, 1)

        self.compare = CompareWidget()
        self.stack.addWidget(self.compare)

        dual = QWidget()
        dual_lay = QHBoxLayout(dual)
        dual_lay.setContentsMargins(0, 0, 0, 0)
        dual_lay.setSpacing(8)
        self.original_view = VideoSurface("SOURCE")
        self.enhanced_view = VideoSurface("ENHANCED")
        dual_lay.addWidget(self.original_view, 1)
        dual_lay.addWidget(self.enhanced_view, 1)
        self.exit_dual = QPushButton("Show control")
        self.stack.addWidget(dual)

        self.players: dict[str, QWidget] = {}
        self._frame_timer_cache: dict[str, float] = {}

        # ---- transport ---------------------------------------------
        transport = QHBoxLayout()
        transport.setSpacing(8)
        self.play_btn = QPushButton()
        self.play_btn.setObjectName("Ghost")
        self.play_btn.setFixedSize(36, 36)
        self.play_btn.setIcon(make_icon("play", Palette.TEXT))
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_play)
        transport.addWidget(self.play_btn)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(self._seek)
        transport.addWidget(self.slider, 1)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setObjectName("Muted")
        self.time_label.setStyleSheet("font-size: 11px;")
        transport.addWidget(self.time_label)
        self._layout.addLayout(transport)

        self.status_label = QLabel("")
        self.status_label.setObjectName("Muted")
        self.status_label.setStyleSheet("font-size: 11px;")
        self._layout.addWidget(self.status_label)

        # ---- players (created lazily once a file exists) -----------
        self.player_src = None
        self.player_out = None
        self.audio_src = None
        self.audio_out = None
        if HAVE_QTMM:
            self._init_players()

        self._fallback_timer = QTimer(self)
        self._fallback_timer.setInterval(40)
        self._fallback_timer.timeout.connect(self._on_fallback_tick)
        self._set_mode("compare")

    # ------------------------------------------------------------------
    def _init_players(self) -> None:
        self.player_src = QMediaPlayer(self)
        self.player_out = QMediaPlayer(self)
        self.audio_src = QAudioOutput(self)
        self.audio_src.setMuted(True)          # only one soundtrack audible
        self.player_src.setAudioOutput(self.audio_src)
        self.audio_out = QAudioOutput(self)
        self.audio_out.setVolume(0.85)
        self.player_out.setAudioOutput(self.audio_out)

        for player, view in ((self.player_src, self.original_view),
                             (self.player_out, self.enhanced_view)):
            vw = QVideoWidget()
            vw.setStyleSheet("background: #05060B; border-radius: 10px;")
            vw.setMinimumSize(200, 150)
            layout = view.layout()
            if layout is None:
                layout = QVBoxLayout(view)
                layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(vw)
            player.setVideoOutput(vw)
            self.players[player.objectName()] = vw

        self.player_out.positionChanged.connect(self._on_position)
        self.player_out.durationChanged.connect(self._on_duration)
        self.player_out.playbackStateChanged.connect(self._on_state)

    # ------------------------------------------------------------------
    def set_source(self, path: str, info) -> None:
        self.source_path = path
        self.info = info
        self.output_path = None
        self._duration = int((info.duration or 0) * 1000) if info else 0
        if self.player_src is not None:
            self.player_src.setSource(QUrl.fromLocalFile(path))
            self.player_src.pause()
            self.player_src.setPosition(0)
        self.enhanced_view.set_message("Run an enhancement to see the result")
        self.play_btn.setEnabled(True)
        self._refresh_info()
        self._update_compare_from_source()
        self._start_fallback_if_needed()

    def set_output(self, path: str) -> None:
        self.output_path = path
        if self.player_out is not None:
            self.player_out.setSource(QUrl.fromLocalFile(path))
            self.player_out.pause()
            try:
                out_info = self.ctx.ffmpeg.probe(path)
                self._duration = int(out_info.duration * 1000)
            except Exception:
                pass
        self._refresh_info()
        self._update_compare_from_source()
        self._set_mode("compare")

    def set_processing(self, running: bool) -> None:
        self.status_label.setText(
            "Processing - live preview appears when the render finishes. "
            "Progress below is measured from real processed frames."
            if running else "")

    # ------------------------------------------------------------------
    def _refresh_info(self) -> None:
        parts = []
        if self.info is not None:
            parts.append(f"Source {self.info.width}x{self.info.height} · "
                         f"{self.info.fps:.2f} fps · "
                         f"{human_duration(self.info.duration)}")
        if self.output_path:
            try:
                oi = self.ctx.ffmpeg.probe(self.output_path)
                parts.append(f"Output {oi.width}x{oi.height} · {oi.fps:.2f} fps")
            except Exception:
                pass
        self.info_label.setText("   →   ".join(parts))
        self.time_label.setText(f"{human_duration(self._position / 1000)} / "
                                f"{human_duration(self._duration / 1000)}")

    def _mode_changed(self) -> None:
        self._set_mode(self.mode_combo.currentData())

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        if mode == "compare":
            self.stack.setCurrentIndex(0)
        elif mode == "original":
            self.stack.setCurrentIndex(1)
            self.original_view.setVisible(True)
            self.enhanced_view.setVisible(False)
        else:
            self.stack.setCurrentIndex(1)
            self.original_view.setVisible(False)
            self.enhanced_view.setVisible(True)
        idx = self.mode_combo.findData(mode)
        if idx >= 0 and self.mode_combo.currentIndex() != idx:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(idx)
            self.mode_combo.blockSignals(False)

    # ------------------------------------------------------------------
    def toggle_play(self) -> None:
        if self.player_out is None:
            self._toggle_fallback()
            return
        if self._playing:
            self.player_out.pause()
            self.player_src.pause()
            self._playing = False
        else:
            self.player_out.play()
            self.player_src.play()
            if self.player_src.position() != self.player_out.position():
                self.player_src.setPosition(self.player_out.position())
            self._playing = True
        self._update_play_icon()

    def _update_play_icon(self) -> None:
        self.play_btn.setIcon(make_icon("pause" if self._playing else "play",
                                        Palette.TEXT))

    def _on_position(self, pos: int) -> None:
        self._position = pos
        if self.player_src is not None and abs(self.player_src.position() - pos) > 120:
            self.player_src.setPosition(pos)
        if self._duration:
            self.slider.blockSignals(True)
            self.slider.setValue(int(pos * 1000 / self._duration))
            self.slider.blockSignals(False)
            self.time_label.setText(f"{human_duration(pos / 1000)} / "
                                    f"{human_duration(self._duration / 1000)}")
        if self._mode == "compare" and self.output_path and not HAVE_QTMM:
            self._update_compare_from_source()

    def _on_duration(self, dur: int) -> None:
        if dur > 0:
            self._duration = dur
        self._refresh_info()

    def _on_state(self, state) -> None:
        if self.player_out is None:
            return
        self._playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._update_play_icon()

    def _seek(self, value: int) -> None:
        if not self._duration:
            return
        ms = int(value * self._duration / 1000)
        self._position = ms
        if self.player_out is not None:
            self.player_out.setPosition(ms)
            self.player_src.setPosition(ms)
        if self._mode == "compare":
            self._update_compare_from_source()

    # ------------------------------------------------------------------
    # Frame-grab fallback (works without a Qt multimedia backend)
    # ------------------------------------------------------------------
    def _start_fallback_if_needed(self) -> None:
        if HAVE_QTMM:
            return
        if self.source_path:
            self.show_source_frame(0.0)

    def _toggle_fallback(self) -> None:
        if self._fallback_timer.isActive():
            self._fallback_timer.stop()
            self._playing = False
        else:
            self._fallback_timer.start(40)
            self._playing = True
            self._fallback_started = __import__("time").time()
            self._fallback_base = self._position
        self._update_play_icon()

    def _on_fallback_tick(self) -> None:
        import time
        elapsed = time.time() - getattr(self, "_fallback_started", time.time())
        pos = getattr(self, "_fallback_base", 0) + elapsed * 1000
        if self._duration and pos >= self._duration:
            self._fallback_timer.stop()
            self._playing = False
            self._update_play_icon()
            pos = self._duration
        self._position = int(pos)
        if self._duration:
            self.slider.blockSignals(True)
            self.slider.setValue(int(self._position * 1000 / self._duration))
            self.slider.blockSignals(False)
        self.time_label.setText(f"{human_duration(self._position / 1000)} / "
                                f"{human_duration(self._duration / 1000)}")
        self.show_source_frame(self._position / 1000)

    def show_source_frame(self, seconds: float) -> None:
        if not self.source_path:
            return
        # throttle grabs to ~4 fps
        import time
        now = time.time()
        if now - self._frame_timer_cache.get("t", 0) < 0.22 and \
                abs(seconds - self._frame_timer_cache.get("s", -1)) < 0.5:
            return
        self._frame_timer_cache.update(t=now, s=seconds)
        try:
            tmp = self.ctx.config.temp_dir / "preview"
            tmp.mkdir(parents=True, exist_ok=True)
            frames = self.ctx.ffmpeg.grab_frames(self.source_path, tmp,
                                                 [max(0.0, seconds)], width=720)
            if frames:
                img = QImage(str(frames[0]))
                if not img.isNull():
                    self.original_view.set_frame(img)
                    self.enhanced_view.set_frame(img) if not self.output_path else None
        except Exception as exc:
            log.debug("preview grab failed: %s", exc)

    def _update_compare_from_source(self) -> None:
        """Fill the wipe view with real frames (used when QtMM can't render)."""
        if not self.source_path:
            return
        try:
            tmp = self.ctx.config.temp_dir / "preview"
            tmp.mkdir(parents=True, exist_ok=True)
            t = max(0.0, self._position / 1000)
            before_frames = self.ctx.ffmpeg.grab_frames(self.source_path, tmp, [t], 640)
            after_frames = (self.ctx.ffmpeg.grab_frames(self.output_path, tmp, [t], 640)
                            if self.output_path else [])
            before = QPixmap(str(before_frames[0])) if before_frames else None
            after = QPixmap(str(after_frames[0])) if after_frames else None
            self.compare.set_images(before, after)
            if before is not None and HAVE_QTMM:
                self.original_view.set_frame(before.toImage())
            if after is not None and HAVE_QTMM:
                self.enhanced_view.set_frame(after.toImage())
        except Exception as exc:
            log.debug("compare update failed: %s", exc)
