"""AI Model Manager page: inspect, download, verify, enable/disable, remove."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from ai.model_manager import DownloadCancelled, ModelStatus
from ai.registry import LICENSE_SUMMARY, all_models
from utils.file_manager import human_size
from utils.logger import get_logger

from .theme import Badge, Card, Divider, Palette, make_icon
from .widgets import NoticeBar, PageHeader

log = get_logger("model_page")


class DownloadWorker(QThread):
    progress = Signal(str, int, int)       # key, done, total
    status = Signal(str, str)              # key, message
    finished_ok = Signal(str)
    failed = Signal(str, str)

    def __init__(self, manager, key: str) -> None:
        super().__init__()
        self.manager = manager
        self.key = key
        self.cancel_event = threading.Event()

    def run(self) -> None:
        try:
            self.manager.download(
                self.key,
                progress=lambda done, total: self.progress.emit(self.key, done, total),
                status=lambda msg: self.status.emit(self.key, msg),
                cancel=self.cancel_event,
                retries=3)
            self.finished_ok.emit(self.key)
        except DownloadCancelled:
            self.failed.emit(self.key, "Download cancelled")
        except Exception as exc:
            self.failed.emit(self.key, str(exc))


class VerifyWorker(QThread):
    progress = Signal(str, int, int)
    finished_ok = Signal(str, bool)

    def __init__(self, manager, key: str) -> None:
        super().__init__()
        self.manager = manager
        self.key = key

    def run(self) -> None:
        ok = self.manager.verify(
            self.key,
            progress=lambda done, total: self.progress.emit(self.key, done, total))
        self.finished_ok.emit(self.key, ok)


class ModelRow(QFrame):
    """One model card with license, size, state and actions."""

    request_download = Signal(str)
    request_remove = Signal(str)
    request_verify = Signal(str)
    toggled = Signal(str, bool)
    cancel = Signal(str)

    def __init__(self, status: ModelStatus, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.status = status
        self.setObjectName("CardFlat")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(12)

        info = QVBoxLayout()
        info.setSpacing(3)
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title = QLabel(status.title)
        title.setStyleSheet("font-weight: 800; font-size: 13px;")
        title_row.addWidget(title)
        self.state_badge = Badge("", "green")
        title_row.addWidget(self.state_badge)
        self.license_badge = Badge(status.license.split(" (")[0], "purple")
        title_row.addWidget(self.license_badge)
        title_row.addStretch(1)
        info.addLayout(title_row)

        desc = QLabel(status.description)
        desc.setWordWrap(True)
        desc.setObjectName("Muted")
        desc.setStyleSheet("font-size: 11px;")
        # Word-wrapped labels do not propagate height-for-width through nested
        # layouts inside a scroll area, so reserve exactly two lines: the card
        # then has a deterministic height and cannot squash its own text.
        desc.setFixedHeight(30)
        desc.setToolTip(status.description)
        desc.setAlignment(Qt.AlignmentFlag.AlignLeft |
                          Qt.AlignmentFlag.AlignTop)
        info.addWidget(desc)

        meta = QLabel(
            f"{status.category.upper()}  ·  v{status.version}  ·  "
            f"{status.size_bytes/1048576:.0f} MB  ·  {status.author}"
            + (f"  ·  {status.quality} quality / {status.speed}" if status.quality else ""))
        meta.setObjectName("Muted")
        meta.setStyleSheet("font-size: 10px;")
        info.addWidget(meta)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.setVisible(False)
        info.addWidget(self.bar)
        self.status_label = QLabel("")
        self.status_label.setObjectName("Muted")
        self.status_label.setStyleSheet("font-size: 10px;")
        self.status_label.setVisible(False)
        info.addWidget(self.status_label)
        info_holder = QWidget()
        info_holder.setLayout(info)
        info.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(info_holder, 1)

        actions = QVBoxLayout()
        actions.setSpacing(6)
        actions.setContentsMargins(0, 0, 0, 0)
        self.enable_check = QCheckBox("Enabled")
        self.enable_check.setChecked(status.enabled)
        self.enable_check.setEnabled(status.installed)
        self.enable_check.toggled.connect(
            lambda v: self.toggled.emit(self.status.key, v))
        actions.addWidget(self.enable_check)

        self.action_btn = QPushButton()
        self.action_btn.setMinimumWidth(104)
        self.action_btn.setFixedHeight(30)
        self.action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        actions.addWidget(self.action_btn)

        self.secondary_btn = QPushButton("Verify")
        self.secondary_btn.setObjectName("Ghost")
        self.secondary_btn.setFixedHeight(30)
        self.secondary_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.secondary_btn.clicked.connect(
            lambda: self.request_verify.emit(self.status.key))
        actions.addWidget(self.secondary_btn)
        actions.addStretch(1)
        act_holder = QWidget()
        act_holder.setLayout(actions)
        act_holder.setFixedWidth(132)
        # NOTE: no AlignTop here - an alignment flag would let the parent layout
        # ignore this column's minimum height and squash the card.
        lay.addWidget(act_holder, 0)

        self._mode = "installed" if status.installed else "missing"
        self.action_btn.clicked.connect(self._on_action)
        self._apply_mode()

    def _apply_mode(self) -> None:
        if self._mode == "installed":
            self.state_badge.setText("INSTALLED")
            self.state_badge.set_kind("green")
            self.action_btn.setText("Remove")
            self.action_btn.setObjectName("Ghost")
            self.secondary_btn.setVisible(True)
            self.enable_check.setEnabled(True)
        elif self._mode == "missing":
            self.state_badge.setText("NOT INSTALLED")
            self.state_badge.set_kind("amber")
            self.action_btn.setText("  Download")
            self.action_btn.setObjectName("Primary")
            self.action_btn.setIcon(make_icon("download", "#FFFFFF"))
            self.secondary_btn.setVisible(False)
            self.enable_check.setEnabled(False)
        elif self._mode == "downloading":
            self.state_badge.setText("DOWNLOADING")
            self.state_badge.set_kind("purple")
            self.action_btn.setText("Cancel")
            self.action_btn.setObjectName("Danger")
            self.secondary_btn.setVisible(False)
            self.bar.setVisible(True)
            self.status_label.setVisible(True)
        elif self._mode == "verifying":
            self.state_badge.setText("VERIFYING")
            self.state_badge.set_kind("purple")
            self.bar.setVisible(True)
            self.status_label.setVisible(True)

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self._apply_mode()

    def _on_action(self) -> None:
        if self._mode == "installed":
            self.request_remove.emit(self.status.key)
        elif self._mode == "missing":
            self.request_download.emit(self.status.key)
        elif self._mode in ("downloading", "verifying"):
            self.cancel.emit(self.status.key)


class ModelManagerPage(QWidget):
    def __init__(self, ctx, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.rows: Dict[str, ModelRow] = {}
        self.downloads: Dict[str, DownloadWorker] = {}
        self.verifiers: Dict[str, VerifyWorker] = {}
        self._build()
        self.refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 20)
        root.setSpacing(14)

        header = PageHeader(
            "AI Model Manager",
            "Models are downloaded from their official repositories - never "
            "silently. Checksums are verified before a model is enabled.")
        refresh_btn = QPushButton("  Refresh")
        refresh_btn.setObjectName("Ghost")
        refresh_btn.setIcon(make_icon("refresh", Palette.TEXT))
        refresh_btn.clicked.connect(self.refresh)
        header.add_action(refresh_btn)
        self.disk_label = QLabel("")
        self.disk_label.setObjectName("Muted")
        header.add_action(self.disk_label)
        root.addWidget(header)

        self.notice = NoticeBar()
        root.addWidget(self.notice)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        self.list_lay = QVBoxLayout(holder)
        self.list_lay.setContentsMargins(0, 0, 8, 0)
        self.list_lay.setSpacing(10)
        scroll.setWidget(holder)
        root.addWidget(scroll, 1)

        lic = Card(padding=16, spacing=6)
        lbl = QLabel("THIRD-PARTY LICENSES")
        lbl.setObjectName("CardTitle")
        lic.add(lbl)
        for name, license_name, url in LICENSE_SUMMARY:
            row = QLabel(f"{name}  ·  {license_name}  ·  {url}")
            row.setObjectName("Muted")
            row.setStyleSheet("font-size: 11px;")
            row.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
            lic.add(row)
        root.addWidget(lic)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self.setUpdatesEnabled(False)
        while self.list_lay.count():
            item = self.list_lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)      # detach now, delete later
                widget.deleteLater()
        self.rows.clear()
        self.setUpdatesEnabled(True)
        for status in self.ctx.model_manager.model_status():
            row = ModelRow(status)
            row.request_download.connect(self.download)
            row.request_remove.connect(self.remove)
            row.request_verify.connect(self.verify)
            row.toggled.connect(self.toggle)
            row.cancel.connect(self.cancel_download)
            self.list_lay.addWidget(row)
            self.rows[status.key] = row
        self.list_lay.addStretch(1)
        used = self.ctx.model_manager.disk_usage_bytes()
        n_installed = sum(1 for s in self.ctx.model_manager.model_status()
                          if s.installed)
        self.disk_label.setText(
            f"{n_installed}/{len(all_models())} installed  ·  "
            f"{human_size(used)} on disk")

    # ------------------------------------------------------------------
    def download(self, key: str) -> None:
        if key in self.downloads:
            return
        spec = [s for s in self.ctx.model_manager.model_status() if s.key == key]
        title = spec[0].title if spec else key
        answer = QMessageBox.question(
            self, "Download model",
            f"Download \u201c{title}\u201d from its official repository?\n\n"
            f"Size: {human_size(spec[0].size_bytes) if spec else '?'}\n"
            f"License: {spec[0].license if spec else '?'}\n\n"
            "The file is verified with SHA-256 after download.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        worker = DownloadWorker(self.ctx.model_manager, key)
        worker.progress.connect(self._on_progress)
        worker.status.connect(self._on_status)
        worker.finished_ok.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        self.downloads[key] = worker
        row = self.rows.get(key)
        if row:
            row.set_mode("downloading")
            row.status_label.setText("Starting...")
        worker.start()

    def cancel_download(self, key: str) -> None:
        worker = self.downloads.get(key)
        if worker:
            worker.cancel_event.set()
            if self.rows.get(key):
                self.rows[key].status_label.setText("Cancelling...")

    def _on_progress(self, key: str, done: int, total: int) -> None:
        row = self.rows.get(key)
        if row and total > 0:
            row.bar.setValue(int(done * 1000 / total))

    def _on_status(self, key: str, message: str) -> None:
        row = self.rows.get(key)
        if row:
            row.status_label.setText(message)

    def _on_done(self, key: str) -> None:
        self.downloads.pop(key, None)
        self.notice.show_message("Model installed and checksum verified.", "success")
        self.refresh()
        self._update_enhancer_models()

    def _on_failed(self, key: str, message: str) -> None:
        self.downloads.pop(key, None)
        self.notice.show_message(f"Download failed: {message}", "error")
        self.refresh()

    def remove(self, key: str) -> None:
        spec = [s for s in self.ctx.model_manager.model_status() if s.key == key]
        title = spec[0].title if spec else key
        answer = QMessageBox.warning(
            self, "Remove model",
            f"Delete \u201c{title}\u201d from disk? You can download it again "
            "at any time.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.ctx.model_manager.remove(key)
            self.refresh()
            self._update_enhancer_models()

    def verify(self, key: str) -> None:
        if key in self.verifiers:
            return
        worker = VerifyWorker(self.ctx.model_manager, key)
        row = self.rows.get(key)
        if row:
            row.set_mode("verifying")
            row.status_label.setText("Hashing file...")
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self._on_verified)
        self.verifiers[key] = worker
        worker.start()

    def _on_verified(self, key: str, ok: bool) -> None:
        self.verifiers.pop(key, None)
        self.notice.show_message(
            "Checksum verified - file matches the official release." if ok else
            "Checksum FAILED - delete and re-download this model.", 
            "success" if ok else "error")
        self.refresh()

    def toggle(self, key: str, enabled: bool) -> None:
        self.ctx.model_manager.set_enabled(key, enabled)
        log.info("Model %s %s", key, "enabled" if enabled else "disabled")

    def _update_enhancer_models(self) -> None:
        # the enhancer page rebuilds its combo next time it is shown
        pass

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh()
