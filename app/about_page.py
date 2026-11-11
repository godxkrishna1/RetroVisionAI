"""About page: product summary, engine versions, licenses."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from ai.registry import LICENSE_SUMMARY
from utils.config import app_root
from utils.gpu_monitor import detect_gpus

from .theme import Card, Divider, GradientBar, Palette, draw_icon_pixmap
from .widgets import PageHeader

VERSION = "1.0.0"


class AboutPage(QWidget):
    def __init__(self, ctx, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 20)
        root.setSpacing(14)

        root.addWidget(PageHeader("About RetroVision AI",
                                  "A free, offline, GPU-accelerated video "
                                  "restoration studio."))

        hero = Card(padding=22, spacing=12)
        row = QHBoxLayout()
        row.setSpacing(18)
        art = QLabel()
        logo = app_root() / "assets" / "app_icon_256.png"
        if logo.exists():
            art.setPixmap(QPixmap(str(logo)).scaled(
                128, 128, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        else:
            art.setPixmap(draw_icon_pixmap("wand", 96, Palette.PURPLE, Palette.PURPLE))
        row.addWidget(art)
        col = QVBoxLayout()
        col.setSpacing(4)
        title = QLabel(f"RetroVision AI {VERSION}")
        title.setStyleSheet("font-size: 24px; font-weight: 800;")
        col.addWidget(title)
        sub = QLabel("Restore 80s and 90s music videos, films and VHS recordings "
                     "to 1080p, 2K and 4K - entirely on your own machine.")
        sub.setObjectName("Dim")
        sub.setWordWrap(True)
        col.addWidget(sub)
        col.addSpacing(6)
        bar = GradientBar(3)
        bar.setFixedWidth(220)
        col.addWidget(bar)
        col.addSpacing(6)
        facts = QLabel(
            "No account · no cloud · no subscriptions · no telemetry. "
            "Your videos never leave your computer.")
        facts.setObjectName("Muted")
        facts.setWordWrap(True)
        col.addWidget(facts)
        row.addLayout(col, 1)
        hero.add_layout(row)
        root.addWidget(hero)

        engine_board = Card(padding=18, spacing=10)
        h = QLabel("RESTORATION ENGINES")
        h.setObjectName("CardTitle")
        engine_board.add(h)
        engines = [
            ("Real-ESRGAN", "Blind super-resolution, x2 / x4", "BSD-3-Clause"),
            ("GFPGAN v1.4", "Generative face restoration", "Apache-2.0"),
            ("RIFE 4.7 / 4.9", "Frame interpolation (2x-4x)", "MIT"),
            ("BasicVSR++", "Recurrent video super-resolution", "Apache-2.0"),
            ("Classical filters", "Denoise, colour, grain, scratches", "OpenCV · Apache-2.0"),
            ("FFmpeg", "Decode / encode / mux", "LGPL/GPL"),
        ]
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        for i, (name, desc, lic) in enumerate(engines):
            cell = QWidget()
            lay = QVBoxLayout(cell)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(1)
            n = QLabel(name)
            n.setStyleSheet("font-weight: 700;")
            lay.addWidget(n)
            d = QLabel(f"{desc}  ·  {lic}")
            d.setObjectName("Muted")
            d.setStyleSheet("font-size: 11px;")
            d.setWordWrap(True)
            lay.addWidget(d)
            grid.addWidget(cell, i // 3, i % 3)
        engine_board.add_layout(grid)
        root.addWidget(engine_board)

        licenses = Card(padding=18, spacing=6)
        lh = QLabel("THIRD-PARTY LICENSES")
        lh.setObjectName("CardTitle")
        licenses.add(lh)
        for name, lic, url in LICENSE_SUMMARY:
            lbl = QLabel(f"{name}  ·  {lic}")
            lbl.setObjectName("Muted")
            lbl.setStyleSheet("font-size: 11px;")
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            licenses.add(lbl)
        licenses.add(Divider())
        note = QLabel(
            "RetroVision AI bundles no model weights in the installer. Models "
            "are downloaded by you, from their official repositories, under "
            "their own licenses. FFmpeg is invoked as an external program.")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 11px;")
        licenses.add(note)
        root.addWidget(licenses)

        env = Card(padding=18, spacing=8)
        eh = QLabel("ENVIRONMENT")
        eh.setObjectName("CardTitle")
        env.add(eh)
        info = self.ctx.device.info()
        gpu = detect_gpus()[0]
        lines = [
            f"Inference device: {info.label}",
            f"PyTorch {info.torch_version or '-'}  ·  CUDA {info.cuda_version or '-'}",
            f"GPU: {gpu.name if gpu.available else 'none (CPU mode)'}",
            f"FFmpeg: {self.ctx.ffmpeg.version()}",
            f"Config folder: {self.ctx.config.root}",
        ]
        for line in lines:
            lbl = QLabel(line)
            lbl.setObjectName("Muted")
            lbl.setStyleSheet("font-size: 11px;")
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            env.add(lbl)
        root.addWidget(env)
        root.addStretch(1)
