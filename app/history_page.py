"""Processing History: searchable list of every finished job."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from utils.file_manager import human_duration
from utils.logger import get_logger

from .theme import Badge, Card, Palette, make_icon
from .widgets import NoticeBar, PageHeader

log = get_logger("history_page")

COLUMNS = ["DATE", "SOURCE", "RESOLUTION", "FRAMES", "TIME", "STATUS", "OUTPUT"]


class HistoryPage(QWidget):
    def __init__(self, ctx, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._records = []
        self._build()
        self.refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 20)
        root.setSpacing(14)

        header = PageHeader("Processing History",
                            "Every finished job with the settings used - stored "
                            "locally in logs/history.jsonl.")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by file name...")
        self.search.setFixedWidth(240)
        self.search.textChanged.connect(self._apply_filter)
        header.add_action(self.search)
        open_btn = QPushButton("Open output")
        open_btn.setObjectName("Ghost")
        open_btn.setIcon(make_icon("play", Palette.TEXT))
        open_btn.clicked.connect(self.open_selected)
        header.add_action(open_btn)
        folder_btn = QPushButton("Show folder")
        folder_btn.setObjectName("Ghost")
        folder_btn.clicked.connect(self.show_folder)
        header.add_action(folder_btn)
        clear_btn = QPushButton("Clear history")
        clear_btn.setObjectName("Danger")
        clear_btn.clicked.connect(self.clear_history)
        header.add_action(clear_btn)
        root.addWidget(header)

        self.stats_card = Card(padding=14, spacing=6)
        self.stats_label = QLabel("")
        self.stats_label.setObjectName("Dim")
        self.stats_card.add(self.stats_label)
        root.addWidget(self.stats_card)

        self.notice = NoticeBar()
        root.addWidget(self.notice)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        for i in (0, 2, 3, 4, 5):
            header_view.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        self.table.doubleClicked.connect(lambda _i: self.open_selected())
        root.addWidget(self.table, 1)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self._records = self.ctx.job_manager.history_records(limit=500)
        self._populate(self._records)
        stats = self.ctx.job_manager.stats()
        self.stats_label.setText(
            f"{stats['completed']} completed  ·  {stats['failed']} failed  ·  "
            f"{stats['frames']:,} frames rendered  ·  "
            f"{human_duration(stats['seconds'])} of footage")
        if not self._records:
            self.notice.show_message("No jobs yet - run an enhancement first.", "info")
        else:
            self.notice.setVisible(False)

    def _populate(self, records) -> None:
        self.table.setRowCount(0)
        for rec in records:
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [
                rec.get("finished_at", ""),
                rec.get("input_name", ""),
                rec.get("resolution", ""),
                f"{int(rec.get('frames') or 0):,}",
                human_duration(rec.get("processing_seconds") or 0),
                (rec.get("status") or "").upper(),
                Path(rec.get("output_path") or "").name,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col in (3, 4):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                          Qt.AlignmentFlag.AlignVCenter)
                if col == 5:
                    status = (rec.get("status") or "").lower()
                    item.setForeground({
                        "completed": Qt.GlobalColor.green,
                        "failed": Qt.GlobalColor.red,
                    }.get(status, Qt.GlobalColor.yellow))
                self.table.setItem(row, col, item)

    def _apply_filter(self, text: str) -> None:
        text = (text or "").lower()
        filtered = [r for r in self._records
                    if text in (r.get("input_name") or "").lower()]
        self._populate(filtered)

    def _selected_record(self) -> Optional[dict]:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        name = self.table.item(row, 1).text() if self.table.item(row, 1) else ""
        for rec in self._records:
            if rec.get("input_name") == name:
                return rec
        return None

    def open_selected(self) -> None:
        rec = self._selected_record()
        if not rec:
            self.notice.show_message("Select a row first.", "warn")
            return
        path = rec.get("output_path") or ""
        if not path or not os.path.exists(path):
            self.notice.show_message(
                "The output file no longer exists (was it moved or deleted?).",
                "warn")
            return
        self._open(path)

    def show_folder(self) -> None:
        rec = self._selected_record()
        if not rec:
            folder = str(self.ctx.config.output_dir)
        else:
            folder = os.path.dirname(rec.get("output_path") or "") or \
                     str(self.ctx.config.output_dir)
        if os.path.isdir(folder):
            self._open(folder)

    @staticmethod
    def _open(path: str) -> None:
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def clear_history(self) -> None:
        answer = QMessageBox.question(
            self, "Clear history",
            "Remove the history list? Output videos are NOT deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.ctx.job_manager.clear_history()
            self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh()
