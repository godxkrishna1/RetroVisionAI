"""Left navigation sidebar with brand, nav buttons and GPU status chip."""
from __future__ import annotations

from typing import Dict

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from .theme import Palette, draw_icon_pixmap, make_icon

NAV_GROUPS = [
    ("WORKSPACE", [
        ("dashboard", "Dashboard", "dashboard"),
        ("enhancer", "Video Enhancer", "wand"),
        ("batch", "Batch Processing", "layers"),
        ("models", "AI Model Manager", "cpu"),
    ]),
    ("MONITOR", [
        ("gpu", "GPU Monitor", "gpu"),
        ("history", "Processing History", "history"),
    ]),
    ("SYSTEM", [
        ("settings", "Settings", "settings"),
        ("about", "About", "info"),
    ]),
]
NAV_ITEMS = [item for _group, items in NAV_GROUPS for item in items]


class Sidebar(QFrame):
    navigate = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(232)
        self._buttons: Dict[str, QPushButton] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 18, 14, 14)
        root.setSpacing(4)

        # ---- brand -------------------------------------------------------
        brand = QHBoxLayout()
        brand.setContentsMargins(6, 0, 0, 0)
        brand.setSpacing(10)
        mark = QLabel()
        pm = draw_icon_pixmap("wizard_mark", 30, "#FFFFFF", Palette.PURPLE)
        from PySide6.QtGui import QPixmap
        from utils.config import app_root
        logo_path = app_root() / "assets" / "logo_mark.png"
        pix = QPixmap(str(logo_path))
        if not pix.isNull():
            mark.setPixmap(pix.scaled(34, 34, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
        else:
            mark.setPixmap(pm)
        brand.addWidget(mark)
        text_col = QVBoxLayout()
        text_col.setSpacing(0)
        title = QLabel("RetroVision")
        title.setObjectName("BrandTitle")
        sub = QLabel("A I   S T U D I O")
        sub.setObjectName("BrandSub")
        text_col.addWidget(title)
        text_col.addWidget(sub)
        brand.addLayout(text_col)
        brand.addStretch(1)
        root.addLayout(brand)
        root.addSpacing(22)

        for group, items in NAV_GROUPS:
            section = QLabel(group)
            section.setStyleSheet(
                f"color: {Palette.TEXT_MUTED}; font-size: 10px; font-weight: 700;"
                "letter-spacing: 1.6px; padding-left: 8px;")
            root.addWidget(section)
            root.addSpacing(6)
            for key, label, icon in items:
                btn = QPushButton(f"  {label}")
                btn.setObjectName("NavButton")
                btn.setCheckable(True)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setIcon(make_icon(icon, Palette.TEXT_DIM))
                btn.setIconSize(QSize(18, 18))
                btn.clicked.connect(lambda _=False, k=key: self.navigate.emit(k))
                self._buttons[key] = btn
                root.addWidget(btn)
            root.addSpacing(12)

        root.addStretch(1)

        # ---- GPU chip ----------------------------------------------------
        self.gpu_chip = QFrame()
        self.gpu_chip.setObjectName("CardFlat")
        chip_lay = QVBoxLayout(self.gpu_chip)
        chip_lay.setContentsMargins(12, 10, 12, 10)
        chip_lay.setSpacing(3)
        chip_top = QHBoxLayout()
        chip_top.setSpacing(6)
        dot = QLabel("\u25CF")
        dot.setStyleSheet(f"color: {Palette.TEXT_MUTED}; font-size: 10px;")
        self.gpu_name = QLabel("Detecting GPU...")
        self.gpu_name.setStyleSheet("font-size: 11px; font-weight: 700;")
        self.gpu_name.setSizePolicy(QSizePolicy.Policy.Ignored,
                                    QSizePolicy.Policy.Preferred)
        chip_top.addWidget(dot)
        chip_top.addWidget(self.gpu_name, 1)
        chip_lay.addLayout(chip_top)
        self.gpu_detail = QLabel("")
        self.gpu_detail.setObjectName("Muted")
        self.gpu_detail.setStyleSheet("font-size: 10px;")
        chip_lay.addWidget(self.gpu_detail)
        root.addWidget(self.gpu_chip)
        root.addSpacing(8)

        self.version_label = QLabel("v1.0.0  ·  local processing")
        self.version_label.setObjectName("Muted")
        self.version_label.setStyleSheet("font-size: 10px; padding-left: 6px;")
        root.addWidget(self.version_label)

        self.set_active("dashboard")

    # ------------------------------------------------------------------
    def set_active(self, key: str) -> None:
        for k, btn in self._buttons.items():
            btn.setChecked(k == key)
            from .theme import make_icon
            btn.setIcon(make_icon(
                dict((a, c) for a, _b, c in NAV_ITEMS)[k],
                Palette.TEXT if k == key else Palette.TEXT_DIM))
            btn.setStyleSheet("")

    def set_gpu(self, name: str, detail: str, active: bool) -> None:
        self.gpu_name.setText(name)
        self.gpu_detail.setText(detail)
        color = Palette.GREEN if active else Palette.TEXT_MUTED
        for child in self.gpu_chip.findChildren(QLabel):
            if child.text() == "\u25CF":
                child.setStyleSheet(f"color: {color}; font-size: 10px;")
                break
