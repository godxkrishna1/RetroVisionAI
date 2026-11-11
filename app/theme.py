"""Design tokens, stylesheet and small UI primitives.

Everything here is Qt-only (no app logic) so it can be unit-tested offscreen.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

from PySide6.QtCore import Qt, QRectF, QSize, Signal, QPointF
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QIcon, QLinearGradient,
                           QPainter, QPainterPath, QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import (QFrame, QGraphicsDropShadowEffect, QHBoxLayout,
                               QLabel, QPushButton, QSizePolicy, QVBoxLayout,
                               QWidget)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
class Palette:
    BG = "#0B0B12"
    BG_ALT = "#0F1019"
    SURFACE = "#141522"
    SURFACE_2 = "#1A1C2C"
    SURFACE_3 = "#22243A"
    BORDER = "#2A2D45"
    BORDER_SOFT = "#232539"
    TEXT = "#ECEDF5"
    TEXT_DIM = "#A6A9C0"
    TEXT_MUTED = "#6E7292"
    PURPLE = "#8B5CF6"
    PURPLE_DARK = "#6D3DE0"
    PURPLE_SOFT = "rgba(139,92,246,41)"
    BLUE = "#3B82F6"
    CYAN = "#22D3EE"
    GREEN = "#22C55E"
    AMBER = "#F59E0B"
    RED = "#EF4444"
    PINK = "#EC4899"

    GRADIENT = ("#8B5CF6", "#6366F1", "#3B82F6")


def qcolor(value: str, alpha: Optional[int] = None) -> QColor:
    c = QColor(value)
    if alpha is not None:
        c.setAlpha(alpha)
    return c


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
def build_stylesheet(palette: Palette = Palette) -> str:
    p = palette
    return f"""
* {{ outline: none; }}
QWidget {{
    color: {p.TEXT};
    font-family: "Segoe UI", "Inter", "Segoe UI Variable", system-ui, sans-serif;
    font-size: 13px;
}}
QMainWindow, QDialog {{ background: {p.BG}; }}
QWidget#Root {{ background: {p.BG}; }}
QToolTip {{
    background: {p.SURFACE_3}; color: {p.TEXT};
    border: 1px solid {p.BORDER}; border-radius: 6px; padding: 6px 8px;
}}

/* ---------------- Sidebar ---------------- */
QWidget#Sidebar {{ background: {p.BG_ALT}; border-right: 1px solid {p.BORDER_SOFT}; }}
QLabel#BrandTitle {{ font-size: 17px; font-weight: 800; letter-spacing: 0.5px; }}
QLabel#BrandSub {{ color: {p.TEXT_MUTED}; font-size: 10px; letter-spacing: 2.2px; }}
QPushButton#NavButton {{
    text-align: left; padding: 10px 14px; border-radius: 10px;
    color: {p.TEXT_DIM}; background: transparent; border: 1px solid transparent;
    font-size: 13px; font-weight: 600;
}}
QPushButton#NavButton:hover {{ background: {p.SURFACE}; color: {p.TEXT}; }}
QPushButton#NavButton:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(139,92,246,56), stop:1 rgba(59,130,246,26));
    color: {p.TEXT}; border: 1px solid rgba(139,92,246,89);
}}

/* ---------------- Cards & panels ---------------- */
QFrame#Card {{
    background: {p.SURFACE}; border: 1px solid {p.BORDER_SOFT};
    border-radius: 14px;
}}
QFrame#CardFlat {{
    background: {p.SURFACE_2}; border: 1px solid {p.BORDER_SOFT};
    border-radius: 12px;
}}
QFrame#HeroCard {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #1B1233, stop:0.55 #141733, stop:1 #101C2E);
    border: 1px solid rgba(139,92,246,110); border-radius: 18px;
}}
QFrame#Divider {{ background: {p.BORDER_SOFT}; max-height: 1px; border: none; }}

/* ---------------- Typography ---------------- */
QLabel#H1 {{ font-size: 26px; font-weight: 800; }}
QLabel#H2 {{ font-size: 19px; font-weight: 700; }}
QLabel#H3 {{ font-size: 15px; font-weight: 700; }}
QLabel#CardTitle {{ font-size: 12px; font-weight: 700; color: {p.TEXT_DIM};
                    letter-spacing: 1.2px; }}
QLabel#Muted {{ color: {p.TEXT_MUTED}; }}
QLabel#Dim {{ color: {p.TEXT_DIM}; }}
QLabel#StatValue {{ font-size: 24px; font-weight: 800; }}
QLabel#StatValueSmall {{ font-size: 18px; font-weight: 700; }}
QLabel#Badge {{ background: {p.PURPLE_SOFT}; color: #C4B5FD; border-radius: 8px;
                padding: 3px 9px; font-size: 11px; font-weight: 700; }}
QLabel#BadgeGreen {{ background: rgba(34,197,94,38); color: #86EFAC;
                     border-radius: 8px; padding: 3px 9px; font-size: 11px;
                     font-weight: 700; }}
QLabel#BadgeRed {{ background: rgba(239,68,68,38); color: #FCA5A5;
                   border-radius: 8px; padding: 3px 9px; font-size: 11px;
                   font-weight: 700; }}
QLabel#BadgeAmber {{ background: rgba(245,158,11,38); color: #FCD34D;
                     border-radius: 8px; padding: 3px 9px; font-size: 11px;
                     font-weight: 700; }}

/* ---------------- Buttons ---------------- */
QPushButton {{
    background: {p.SURFACE_2}; color: {p.TEXT}; border: 1px solid {p.BORDER};
    border-radius: 10px; padding: 9px 16px; font-weight: 600;
}}
QPushButton:hover {{ background: {p.SURFACE_3}; border-color: #3A3E5E; }}
QPushButton:pressed {{ background: #1C1E30; }}
QPushButton:disabled {{ color: {p.TEXT_MUTED}; background: #171826;
                        border-color: #22243A; }}
QPushButton#Primary {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {p.PURPLE}, stop:1 {p.BLUE});
    border: 1px solid rgba(255,255,255,31); color: white; font-weight: 700;
    padding: 10px 20px;
}}
QPushButton#Primary:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #9B71FF, stop:1 #4C8FF8);
}}
QPushButton#Primary:disabled {{ background: #2A2C44; color: {p.TEXT_MUTED}; }}
QPushButton#Danger {{ background: rgba(239,68,68,36); color: #FCA5A5;
                      border: 1px solid rgba(239,68,68,89); }}
QPushButton#Danger:hover {{ background: rgba(239,68,68,56); }}
QPushButton#Ghost {{ background: transparent; border: 1px solid {p.BORDER};
                     color: {p.TEXT_DIM}; }}
QPushButton#Ghost:hover {{ color: {p.TEXT}; border-color: {p.PURPLE}; }}
QPushButton#Link {{ background: transparent; border: none; color: #A78BFA;
                    padding: 4px 6px; text-align: left; }}
QPushButton#Link:hover {{ color: #C4B5FD; text-decoration: underline; }}

/* ---------------- Inputs ---------------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {p.BG_ALT}; border: 1px solid {p.BORDER}; border-radius: 9px;
    padding: 7px 10px; selection-background-color: {p.PURPLE_DARK};
    color: {p.TEXT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {p.PURPLE}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{
    background: {p.SURFACE_2}; border: 1px solid {p.BORDER};
    selection-background-color: {p.PURPLE_DARK}; outline: none; padding: 4px;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 16px; background: transparent; border: none;
}}

/* ---------------- Checkboxes / toggles ---------------- */
QCheckBox {{ spacing: 9px; color: {p.TEXT}; }}
QCheckBox::indicator {{
    width: 18px; height: 18px; border-radius: 5px;
    border: 1px solid {p.BORDER}; background: {p.BG_ALT};
}}
QCheckBox::indicator:checked {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 {p.PURPLE}, stop:1 {p.BLUE});
    border: 1px solid {p.PURPLE};
}}
QCheckBox::indicator:hover {{ border-color: {p.PURPLE}; }}

/* ---------------- Slider ---------------- */
QSlider::groove:horizontal {{
    height: 5px; border-radius: 3px; background: {p.SURFACE_3};
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {p.PURPLE}, stop:1 {p.BLUE}); border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: white; width: 14px; height: 14px; margin: -5px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: #E9D5FF; }}

/* ---------------- Progress ---------------- */
QProgressBar {{
    background: {p.SURFACE_3}; border: none; border-radius: 5px;
    height: 10px; text-align: center; color: transparent; font-size: 1px;
}}
QProgressBar::chunk {{
    border-radius: 5px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {p.PURPLE}, stop:0.6 {p.BLUE}, stop:1 {p.CYAN});
}}

/* ---------------- Tabs ---------------- */
QTabWidget::pane {{ border: 1px solid {p.BORDER_SOFT}; border-radius: 12px;
                    background: {p.SURFACE}; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {p.TEXT_MUTED}; padding: 9px 18px;
    border-radius: 9px; margin-right: 4px; font-weight: 600;
}}
QTabBar::tab:selected {{ background: {p.SURFACE_2}; color: {p.TEXT}; }}
QTabBar::tab:hover {{ color: {p.TEXT}; }}

/* ---------------- Scrollbars ---------------- */
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QWidget#ScrollContent {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #2E3150; border-radius: 5px;
                               min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #3C4068; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #2E3150; border-radius: 5px;
                                 min-width: 30px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ---------------- Tables / lists ---------------- */
QTableWidget, QListWidget, QTreeWidget {{
    background: {p.SURFACE}; border: 1px solid {p.BORDER_SOFT};
    border-radius: 10px; gridline-color: {p.BORDER_SOFT};
    alternate-background-color: #161726;
}}
QTableWidget::item, QListWidget::item {{ padding: 7px 8px; border-radius: 6px; }}
QTableWidget::item:selected, QListWidget::item:selected {{
    background: rgba(139,92,246,56); color: {p.TEXT};
}}
QHeaderView::section {{
    background: {p.SURFACE_2}; color: {p.TEXT_MUTED}; border: none;
    border-bottom: 1px solid {p.BORDER}; padding: 8px; font-weight: 700;
    font-size: 11px; letter-spacing: 0.6px;
}}
QTableCornerButton::section {{ background: {p.SURFACE_2}; border: none; }}

/* ---------------- Menus ---------------- */
QMenu {{ background: {p.SURFACE_2}; border: 1px solid {p.BORDER};
         border-radius: 10px; padding: 6px; }}
QMenu::item {{ padding: 7px 20px; border-radius: 7px; }}
QMenu::item:selected {{ background: rgba(139,92,246,64); }}
QMenu::separator {{ height: 1px; background: {p.BORDER}; margin: 5px 8px; }}

/* ---------------- Misc ---------------- */
QGroupBox {{
    border: 1px solid {p.BORDER_SOFT}; border-radius: 12px;
    margin-top: 14px; padding: 16px 12px 12px 12px; background: {p.SURFACE};
    font-weight: 700;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 6px;
                    color: {p.TEXT_DIM}; }}
QSplitter::handle {{ background: {p.BORDER_SOFT}; }}
QSplitter::handle:horizontal {{ width: 2px; }}
QStatusBar {{ background: {p.BG_ALT}; color: {p.TEXT_MUTED};
              border-top: 1px solid {p.BORDER_SOFT}; }}
QStatusBar::item {{ border: none; }}
QFileDialog {{ background: {p.BG}; }}
"""


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
def scale_font(widget: QWidget, size: int, weight: QFont.Weight = QFont.Weight.Normal,
               letter_spacing: float = 0.0) -> QFont:
    font = widget.font()
    font.setPointSize(size)
    font.setWeight(weight)
    if letter_spacing:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, letter_spacing)
    widget.setFont(font)
    return font


# ---------------------------------------------------------------------------
# Icons (drawn as vectors so no external assets are required)
# ---------------------------------------------------------------------------
def draw_icon_pixmap(name: str, size: int = 22, color: str = Palette.TEXT_DIM,
                     accent: str = Palette.PURPLE) -> QPixmap:
    pm = QPixmap(size * 2, size * 2)  # 2x for HiDPI
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size
    pen = QPen(QColor(color), max(1.4, s / 14))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    c = s / 2

    if name == "dashboard":
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            p.drawRoundedRect(QRectF(s * 0.16 + dx * s * 0.38, s * 0.16 + dy * s * 0.38,
                                     s * 0.28, s * 0.28), 3, 3)
    elif name == "wand":
        p.drawLine(QPointF(s * 0.22, s * 0.78), QPointF(s * 0.68, s * 0.32))
        p.drawLine(QPointF(s * 0.62, s * 0.22), QPointF(s * 0.78, s * 0.38))
        p.setPen(QPen(QColor(accent), max(1.4, s / 14)))
        p.drawLine(QPointF(s * 0.82, s * 0.16), QPointF(s * 0.82, s * 0.24))
        p.drawLine(QPointF(s * 0.78, s * 0.20), QPointF(s * 0.86, s * 0.20))
    elif name == "layers":
        for i, y in enumerate((0.30, 0.50, 0.70)):
            p.drawPolygon(QPolygonF([
                QPointF(c, s * (y - 0.10)), QPointF(s * 0.84, s * y),
                QPointF(c, s * (y + 0.10)), QPointF(s * 0.16, s * y)]))
    elif name == "cpu":
        p.drawRoundedRect(QRectF(s * 0.22, s * 0.22, s * 0.56, s * 0.56), 4, 4)
        p.drawRoundedRect(QRectF(s * 0.38, s * 0.38, s * 0.24, s * 0.24), 2, 2)
        for i in (0.3, 0.5, 0.7):
            p.drawLine(QPointF(s * i, s * 0.1), QPointF(s * i, s * 0.2))
            p.drawLine(QPointF(s * i, s * 0.8), QPointF(s * i, s * 0.9))
            p.drawLine(QPointF(s * 0.1, s * i), QPointF(s * 0.2, s * i))
            p.drawLine(QPointF(s * 0.8, s * i), QPointF(s * 0.9, s * i))
    elif name == "gpu":
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.28, s * 0.62, s * 0.44), 4, 4)
        p.drawEllipse(QPointF(s * 0.40, c), s * 0.13, s * 0.13)
        p.drawRect(QRectF(s * 0.28, s * 0.78, s * 0.5, s * 0.07))
        p.drawLine(QPointF(s * 0.80, s * 0.36), QPointF(s * 0.88, s * 0.36))
        p.drawLine(QPointF(s * 0.80, s * 0.50), QPointF(s * 0.88, s * 0.50))
        p.drawLine(QPointF(s * 0.80, s * 0.64), QPointF(s * 0.88, s * 0.64))
    elif name == "history":
        p.drawArc(QRectF(s * 0.16, s * 0.16, s * 0.68, s * 0.68), 40 * 16, 290 * 16)
        p.drawPolyline(QPolygonF([QPointF(s * 0.5, s * 0.30),
                                  QPointF(s * 0.5, s * 0.52),
                                  QPointF(s * 0.64, s * 0.60)]))
    elif name == "settings":
        p.drawEllipse(QPointF(c, c), s * 0.16, s * 0.16)
        p.drawEllipse(QPointF(c, c), s * 0.30, s * 0.30)
        for k in range(8):
            a = math.radians(k * 45)
            x1, y1 = c + math.cos(a) * s * 0.33, c + math.sin(a) * s * 0.33
            x2, y2 = c + math.cos(a) * s * 0.42, c + math.sin(a) * s * 0.42
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    elif name == "info":
        p.drawEllipse(QPointF(c, c), s * 0.38, s * 0.38)
        p.drawLine(QPointF(c, s * 0.46), QPointF(c, s * 0.68))
        p.drawPoint(QPointF(c, s * 0.34))
    elif name == "play":
        p.setBrush(QColor(color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(QPolygonF([QPointF(s * 0.34, s * 0.22),
                                 QPointF(s * 0.80, s * 0.50),
                                 QPointF(s * 0.34, s * 0.78)]))
    elif name == "pause":
        p.setBrush(QColor(color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(s * 0.30, s * 0.24, s * 0.13, s * 0.52), 2, 2)
        p.drawRoundedRect(QRectF(s * 0.57, s * 0.24, s * 0.13, s * 0.52), 2, 2)
    elif name == "stop":
        p.setBrush(QColor(color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(s * 0.30, s * 0.30, s * 0.40, s * 0.40), 3, 3)
    elif name == "upload":
        p.drawPolyline(QPolygonF([QPointF(c, s * 0.20), QPointF(c, s * 0.62)]))
        p.drawPolyline(QPolygonF([QPointF(s * 0.32, s * 0.40), QPointF(c, s * 0.20),
                                  QPointF(s * 0.68, s * 0.40)]))
        p.drawPolyline(QPolygonF([QPointF(s * 0.24, s * 0.70), QPointF(s * 0.24, s * 0.82),
                                  QPointF(s * 0.76, s * 0.82), QPointF(s * 0.76, s * 0.70)]))
    elif name == "folder":
        p.drawPolyline(QPolygonF([QPointF(s * 0.16, s * 0.74), QPointF(s * 0.16, s * 0.28),
                                  QPointF(s * 0.42, s * 0.28), QPointF(s * 0.50, s * 0.38),
                                  QPointF(s * 0.84, s * 0.38), QPointF(s * 0.84, s * 0.74),
                                  QPointF(s * 0.16, s * 0.74)]))
    elif name == "download":
        p.drawLine(QPointF(c, s * 0.20), QPointF(c, s * 0.62))
        p.drawPolyline(QPolygonF([QPointF(s * 0.32, s * 0.44), QPointF(c, s * 0.64),
                                  QPointF(s * 0.68, s * 0.44)]))
        p.drawPolyline(QPolygonF([QPointF(s * 0.24, s * 0.74), QPointF(s * 0.24, s * 0.84),
                                  QPointF(s * 0.76, s * 0.84), QPointF(s * 0.76, s * 0.74)]))
    elif name == "check":
        p.drawPolyline(QPolygonF([QPointF(s * 0.24, s * 0.52), QPointF(s * 0.44, s * 0.72),
                                  QPointF(s * 0.78, s * 0.30)]))
    elif name == "close":
        p.drawLine(QPointF(s * 0.28, s * 0.28), QPointF(s * 0.72, s * 0.72))
        p.drawLine(QPointF(s * 0.72, s * 0.28), QPointF(s * 0.28, s * 0.72))
    elif name == "film":
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.20, s * 0.72, s * 0.60), 3, 3)
        for i in (0.3, 0.5, 0.7):
            p.drawLine(QPointF(s * 0.14, s * i), QPointF(s * 0.24, s * i))
            p.drawLine(QPointF(s * 0.76, s * i), QPointF(s * 0.86, s * i))
    elif name == "refresh":
        p.drawArc(QRectF(s * 0.18, s * 0.18, s * 0.64, s * 0.64), 30 * 16, 280 * 16)
        p.drawPolyline(QPolygonF([QPointF(s * 0.66, s * 0.12), QPointF(s * 0.74, s * 0.26),
                                  QPointF(s * 0.58, s * 0.28)]))
    elif name == "bolt":
        p.setBrush(QColor(accent))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(QPolygonF([QPointF(s * 0.56, s * 0.10), QPointF(s * 0.28, s * 0.56),
                                 QPointF(s * 0.48, s * 0.56), QPointF(s * 0.42, s * 0.90),
                                 QPointF(s * 0.74, s * 0.42), QPointF(s * 0.52, s * 0.42)]))
    p.end()
    return pm


def make_icon(name: str, color: str = Palette.TEXT_DIM,
              accent: str = Palette.PURPLE, size: int = 22) -> QIcon:
    return QIcon(draw_icon_pixmap(name, size, color, accent))


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------
def add_shadow(widget: QWidget, blur: int = 28, color: str = "#000000",
               alpha: int = 120, offset: int = 6) -> None:
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(blur)
    c = QColor(color)
    c.setAlpha(alpha)
    eff.setColor(c)
    eff.setOffset(0, offset)
    widget.setGraphicsEffect(eff)


class Card(QFrame):
    """Rounded surface used for every panel on the pages."""

    def __init__(self, parent: Optional[QWidget] = None, object_name: str = "Card",
                 padding: int = 18, spacing: int = 12) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(spacing)

    def body(self) -> QVBoxLayout:
        return self._layout

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self._layout.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)


class TitleBarMixin:
    """Small helper to create consistent card headers."""

    @staticmethod
    def header_row(title: str, subtitle: str = "", badge: str = "") -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        col = QVBoxLayout()
        col.setSpacing(2)
        lbl = QLabel(title)
        lbl.setObjectName("H3")
        col.addWidget(lbl)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("Muted")
            sub.setStyleSheet("font-size: 11px;")
            col.addWidget(sub)
        row.addLayout(col)
        row.addStretch(1)
        if badge:
            b = QLabel(badge)
            b.setObjectName("Badge")
            row.addWidget(b)
        return row


class Badge(QLabel):
    def __init__(self, text: str = "", kind: str = "purple",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.set_kind(kind)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_kind(self, kind: str) -> None:
        self.setObjectName({
            "purple": "Badge", "green": "BadgeGreen",
            "red": "BadgeRed", "amber": "BadgeAmber",
        }.get(kind, "Badge"))


class StatCard(Card):
    """A dashboard KPI tile with a drawn icon chip."""

    def __init__(self, icon: str, title: str, value: str = "-",
                 subtitle: str = "", accent: str = Palette.PURPLE,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, padding=16, spacing=6)
        self._accent = accent
        top = QHBoxLayout()
        top.setSpacing(10)
        chip = QLabel()
        chip.setFixedSize(38, 38)
        chip.setPixmap(draw_icon_pixmap(icon, 20, Palette.TEXT, accent))
        chip.setStyleSheet(
            f"background: rgba(139,92,246,36); border-radius: 10px; padding: 7px;")
        chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(chip)
        name = QLabel(title.upper())
        name.setObjectName("CardTitle")
        top.addWidget(name, 1)
        self._layout.addLayout(top)

        self.value_label = QLabel(value)
        self.value_label.setObjectName("StatValue")
        self._layout.addWidget(self.value_label)

        self.sub_label = QLabel(subtitle)
        self.sub_label.setObjectName("Muted")
        self.sub_label.setStyleSheet("font-size: 11px;")
        self._layout.addWidget(self.sub_label)

    def set_value(self, value: str, subtitle: Optional[str] = None) -> None:
        self.value_label.setText(value)
        if subtitle is not None:
            self.sub_label.setText(subtitle)


class IconButton(QPushButton):
    def __init__(self, icon: str, tooltip: str = "", size: int = 34,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setIcon(make_icon(icon, Palette.TEXT_DIM))
        self.setIconSize(QSize(18, 18))
        self.setToolTip(tooltip)
        self.setObjectName("Ghost")


class GradientBar(QWidget):
    """Slim gradient accent used under the brand and in the hero card."""

    def __init__(self, height: int = 3, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.width() <= 0 or self.height() <= 0:
            return
        grad = QLinearGradient(0, 0, max(1, self.width()), 0)
        grad.setColorAt(0.0, QColor(Palette.PURPLE))
        grad.setColorAt(0.5, QColor(Palette.BLUE))
        grad.setColorAt(1.0, QColor(Palette.CYAN))
        p.setBrush(grad)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), 2, 2)


class MetricRow(QHBoxLayout):
    """Label on the left, value on the right - used in detail panels."""

    def __init__(self, label: str, value: str = "-") -> None:
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.key = QLabel(label)
        self.key.setObjectName("Muted")
        self.key.setStyleSheet("font-size: 12px;")
        self.val = QLabel(value)
        self.val.setStyleSheet("font-size: 12px; font-weight: 600;")
        self.val.setAlignment(Qt.AlignmentFlag.AlignRight |
                              Qt.AlignmentFlag.AlignVCenter)
        # Key keeps its natural width; the value absorbs the rest with a small
        # floor so short strings never collapse, and long paths are elided in
        # the middle by the owning row.
        self.key.setSizePolicy(QSizePolicy.Policy.Preferred,
                               QSizePolicy.Policy.Preferred)
        self.val.setSizePolicy(QSizePolicy.Policy.Expanding,
                               QSizePolicy.Policy.Preferred)
        self.val.setMinimumWidth(60)
        self.val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.val.setToolTip(value)
        self.addWidget(self.key, 0)
        self.addWidget(self.val, 1)
        self._full_text = value

    def set_value(self, value: str) -> None:
        self.val.setText(value)
        self.val.setToolTip(value)
        self._full_text = value



class Divider(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("Divider")
        self.setFixedHeight(1)


class EmptyState(QWidget):
    """Friendly placeholder with a drawn illustration."""

    def __init__(self, title: str, subtitle: str = "", icon: str = "film",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 30, 20, 30)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        art = QLabel()
        art.setPixmap(draw_icon_pixmap(icon, 56, Palette.TEXT_MUTED, Palette.PURPLE))
        art.setAlignment(Qt.AlignmentFlag.AlignCenter)
        art.setStyleSheet("padding: 10px;")
        lay.addWidget(art)
        t = QLabel(title)
        t.setObjectName("H3")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)
        if subtitle:
            s = QLabel(subtitle)
            s.setObjectName("Muted")
            s.setAlignment(Qt.AlignmentFlag.AlignCenter)
            s.setWordWrap(True)
            lay.addWidget(s)


class RingGauge(QWidget):
    """Circular percentage gauge used by the GPU monitor."""

    def __init__(self, parent: Optional[QWidget] = None, diameter: int = 120,
                 label: str = "%") -> None:
        super().__init__(parent)
        self._value = 0.0
        self._label = label
        self._diameter = diameter
        self.setMinimumSize(diameter, diameter)

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(100.0, float(value)))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        if side <= 24:
            return
        rect = QRectF((self.width() - side) / 2 + 8, (self.height() - side) / 2 + 8,
                      side - 16, side - 16)
        p.setPen(QPen(QColor(Palette.SURFACE_3), 10))
        p.drawArc(rect, 0, 360 * 16)
        grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0, QColor(Palette.PURPLE))
        grad.setColorAt(1, QColor(Palette.CYAN))
        pen = QPen(QColor(Palette.PURPLE), 10)
        pen.setBrush(grad)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, 90 * 16, -int(self._value / 100.0 * 360 * 16))
        p.setPen(QColor(Palette.TEXT))
        f = p.font()
        f.setPointSizeF(max(9, side / 9))
        f.setBold(True)
        p.setFont(f)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                   f"{self._value:.0f}{self._label}")
