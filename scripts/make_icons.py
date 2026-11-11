"""Generate the RetroVision AI application icon set (no external assets).

Run:  python scripts/make_icons.py
Outputs into assets/:
    app_icon.png       1024x1024 master
    app_icon_256.png   Windows EXE icon source
    app_icon.ico       multi-size Windows icon
    logo_mark.png      transparent small mark for the sidebar
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

# Must run on headless machines (CI runners): never try to open a window.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QGuiApplication, QImage, QLinearGradient,
                           QPainter, QPainterPath, QPen, QPolygonF)


def draw_master(size: int = 1024) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    s = float(size)
    # ---- rounded background with purple->blue diagonal gradient
    bg = QPainterPath()
    bg.addRoundedRect(QRectF(s * 0.03, s * 0.03, s * 0.94, s * 0.94),
                      s * 0.22, s * 0.22)
    grad = QLinearGradient(0, 0, s, s)
    grad.setColorAt(0.0, QColor("#2A1358"))
    grad.setColorAt(0.45, QColor("#171A38"))
    grad.setColorAt(1.0, QColor("#0A1B33"))
    p.fillPath(bg, grad)

    # subtle border
    p.setPen(QPen(QColor(139, 92, 246, 150), s * 0.006))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(bg)

    # ---- orbiting ring (clipped to the area above the wordmark)
    p.save()
    p.setClipRect(QRectF(s * 0.02, s * 0.02, s * 0.96, s * 0.72))
    ring_rect = QRectF(s * 0.15, s * 0.15, s * 0.70, s * 0.70)
    p.setPen(QPen(QColor(255, 255, 255, 30), s * 0.010))
    p.drawEllipse(ring_rect)
    ring_grad = QLinearGradient(s * 0.15, s * 0.15, s * 0.85, s * 0.85)
    ring_grad.setColorAt(0, QColor("#8B5CF6"))
    ring_grad.setColorAt(1, QColor("#22D3EE"))
    pen = QPen(QColor("#8B5CF6"), s * 0.019)
    pen.setBrush(ring_grad)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawArc(ring_rect, 55 * 16, 205 * 16)
    p.restore()

    # ---- film strip: body + sprocket holes flanking the frame
    strip = QRectF(s * 0.22, s * 0.29, s * 0.56, s * 0.42)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(10, 12, 24, 120))
    p.drawRoundedRect(strip, s * 0.035, s * 0.035)

    hole_w, hole_h = s * 0.026, s * 0.034
    p.setBrush(QColor(236, 237, 245, 105))
    for row in range(4):
        y = s * 0.315 + row * s * 0.100
        for x in (s * 0.233, s * 0.741):
            p.drawRoundedRect(QRectF(x, y, hole_w, hole_h), s * 0.006, s * 0.006)

    # ---- inner screen
    frame = QRectF(s * 0.285, s * 0.305, s * 0.43, s * 0.39)
    p.setPen(QPen(QColor(236, 237, 245, 230), s * 0.011))
    p.setBrush(QColor(6, 8, 18, 160))
    p.drawRoundedRect(frame, s * 0.032, s * 0.032)

    # ---- play triangle (gradient fill)
    tri = QPolygonF([QPointF(s * 0.435, s * 0.365), QPointF(s * 0.625, s * 0.50),
                     QPointF(s * 0.435, s * 0.635)])
    tri_grad = QLinearGradient(s * 0.435, s * 0.365, s * 0.63, s * 0.635)
    tri_grad.setColorAt(0, QColor("#A78BFA"))
    tri_grad.setColorAt(1, QColor("#38BDF8"))
    p.setBrush(tri_grad)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPolygon(tri)

    # ---- sparkles (AI)
    def spark(cx: float, cy: float, r: float, color: QColor) -> None:
        path = QPainterPath()
        path.moveTo(cx, cy - r)
        path.quadTo(cx + r * 0.18, cy - r * 0.18, cx + r, cy)
        path.quadTo(cx + r * 0.18, cy + r * 0.18, cx, cy + r)
        path.quadTo(cx - r * 0.18, cy + r * 0.18, cx - r, cy)
        path.quadTo(cx - r * 0.18, cy - r * 0.18, cx, cy - r)
        p.fillPath(path, color)

    spark(s * 0.150, s * 0.215, s * 0.052, QColor("#FDE68A"))
    spark(s * 0.845, s * 0.330, s * 0.032, QColor("#C4B5FD"))

    # ---- wordmark
    f = QFont("DejaVu Sans", int(s * 0.062))
    f.setWeight(QFont.Weight.Black)
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, s * 0.004)
    p.setFont(f)
    p.setPen(QColor(240, 241, 250, 240))
    p.drawText(QRectF(0, s * 0.735, s, s * 0.13),
               Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
               "RETROVISION")
    f2 = QFont("DejaVu Sans", int(s * 0.030))
    f2.setWeight(QFont.Weight.DemiBold)
    f2.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, s * 0.016)
    p.setFont(f2)
    p.setPen(QColor(167, 139, 250, 235))
    p.drawText(QRectF(0, s * 0.838, s, s * 0.07),
               Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, "A I   S T U D I O")

    p.end()
    return img


def main() -> int:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)
    out = Path(__file__).resolve().parent.parent / "assets"
    out.mkdir(parents=True, exist_ok=True)

    master = draw_master(1024)
    master.save(str(out / "app_icon.png"))
    master.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio,
                  Qt.TransformationMode.SmoothTransformation).save(
        str(out / "app_icon_256.png"))
    master.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio,
                  Qt.TransformationMode.SmoothTransformation).save(
        str(out / "logo_mark.png"))

    # Multi-resolution .ico (Qt can write ico; embed several sizes via PNG)
    img256 = master.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
    ok = img256.save(str(out / "app_icon.ico"), "ICO")
    if not ok:
        print("WARNING: ICO writer unavailable; PNG icons still generated")

    # Simple light-on-dark preview banner for README
    banner = QImage(1600, 500, QImage.Format.Format_ARGB32_Premultiplied)
    banner.fill(QColor("#0B0B12"))
    bp = QPainter(banner)
    bp.setRenderHint(QPainter.RenderHint.Antialiasing)
    bg = QLinearGradient(0, 0, 1600, 500)
    bg.setColorAt(0, QColor("#160C2E"))
    bg.setColorAt(0.6, QColor("#0E1330"))
    bg.setColorAt(1, QColor("#08131F"))
    bp.fillRect(0, 0, 1600, 500, bg)
    bp.drawImage(QRectF(90, 90, 320, 320), draw_master(1024))
    f = QFont("Segoe UI", 54)
    f.setWeight(QFont.Weight.Black)
    bp.setFont(f)
    bp.setPen(QColor("#F2F3FA"))
    bp.drawText(QRectF(460, 150, 1100, 100), Qt.AlignmentFlag.AlignLeft, "RetroVision AI")
    f2 = QFont("Segoe UI", 21)
    bp.setFont(f2)
    bp.setPen(QColor("#A6A9C0"))
    bp.drawText(QRectF(462, 250, 1050, 120), Qt.AlignmentFlag.AlignLeft,
                "Restore 80s & 90s video to crisp 1080p / 2K / 4K\n"
                "Real-ESRGAN  ·  GFPGAN  ·  RIFE  ·  BasicVSR++   |   Local GPU, 100% free")
    bp.end()
    banner.save(str(out / "banner.png"))

    print(f"Icons written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
