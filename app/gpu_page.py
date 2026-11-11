"""GPU Monitor: live VRAM, utilisation, temperature plus tuning advice."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QVBoxLayout, QWidget)

from utils import gpu_monitor
from utils.logger import get_logger

from .theme import Badge, Card, MetricRow, Palette, RingGauge, StatCard, make_icon
from .widgets import NoticeBar, PageHeader

log = get_logger("gpu_page")


class GPUPage(QWidget):
    def __init__(self, ctx, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._build()
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(26, 22, 26, 20)
        root.setSpacing(14)

        header = PageHeader("GPU Monitor", "Live telemetry from NVIDIA NVML / "
                                           "nvidia-smi and the PyTorch allocator.")
        self.status_badge = Badge("DETECTING", "purple")
        header.add_action(self.status_badge)
        root.addWidget(header)

        self.notice = NoticeBar()
        root.addWidget(self.notice)

        top = QHBoxLayout()
        top.setSpacing(14)

        gpu_card = Card(padding=18, spacing=12)
        row = QHBoxLayout()
        row.setSpacing(18)
        self.gauge = RingGauge(diameter=150)
        row.addWidget(self.gauge)
        col = QVBoxLayout()
        col.setSpacing(4)
        self.gpu_name = QLabel("Detecting...")
        self.gpu_name.setStyleSheet("font-size: 18px; font-weight: 800;")
        self.gpu_name.setWordWrap(True)
        col.addWidget(self.gpu_name)
        self.gpu_sub = QLabel("")
        self.gpu_sub.setObjectName("Muted")
        col.addWidget(self.gpu_sub)
        col.addSpacing(6)
        self.rows = {}
        for key, label in (("vram", "VRAM used"), ("temp", "Temperature"),
                           ("power", "Power draw"), ("driver", "Driver"),
                           ("cuda", "CUDA"), ("torch", "PyTorch")):
            row_w = MetricRow(label)
            self.rows[key] = row_w
            col.addLayout(row_w)
        row.addLayout(col, 1)
        gpu_card.add_layout(row)
        top.addWidget(gpu_card, 2)

        sys_card = Card(padding=18, spacing=10)
        t = QLabel("SYSTEM")
        t.setObjectName("CardTitle")
        sys_card.add(t)
        self.sys_rows = {}
        for key, label in (("cpu", "CPU"), ("cpu_use", "CPU usage"),
                           ("ram", "Memory"), ("ram_use", "Memory usage"),
                           ("os", "Platform"), ("device", "Inference device"),
                           ("encoder", "Preferred encoder")):
            row_w = MetricRow(label)
            self.sys_rows[key] = row_w
            sys_card.add_layout(row_w)
        sys_card.setMinimumWidth(430)
        sys_card.body().addStretch(1)
        top.addWidget(sys_card, 2)
        root.addLayout(top)

        vram_card = Card(padding=18, spacing=8)
        vh = QHBoxLayout()
        lbl = QLabel("VRAM ALLOCATION")
        lbl.setObjectName("CardTitle")
        vh.addWidget(lbl)
        vh.addStretch(1)
        self.vram_label = QLabel("")
        self.vram_label.setObjectName("Muted")
        vh.addWidget(self.vram_label)
        vram_card.add_layout(vh)
        self.vram_bar = QProgressBar()
        self.vram_bar.setRange(0, 1000)
        self.vram_bar.setTextVisible(False)
        self.vram_bar.setFixedHeight(12)
        vram_card.add(self.vram_bar)
        self.tuning_hint = QLabel("")
        self.tuning_hint.setWordWrap(True)
        self.tuning_hint.setObjectName("Muted")
        vram_card.add(self.tuning_hint)
        root.addWidget(vram_card)

        adv = Card(padding=18, spacing=10)
        ah = QHBoxLayout()
        lbl2 = QLabel("TUNING")
        lbl2.setObjectName("CardTitle")
        ah.addWidget(lbl2)
        ah.addStretch(1)
        rescan = QPushButton("  Rescan hardware")
        rescan.setObjectName("Ghost")
        rescan.setIcon(make_icon("refresh", Palette.TEXT))
        rescan.clicked.connect(self._rescan)
        ah.addWidget(rescan)
        adv.add_layout(ah)
        form_row = QHBoxLayout()
        form_row.setSpacing(10)
        l1 = QLabel("Inference device")
        l1.setObjectName("Muted")
        form_row.addWidget(l1)
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (GPU when available)", "auto")
        self.device_combo.addItem("NVIDIA GPU (CUDA)", "cuda")
        self.device_combo.addItem("CPU only", "cpu")
        self.device_combo.currentIndexChanged.connect(self._device_changed)
        form_row.addWidget(self.device_combo)
        form_row.addSpacing(16)
        l2 = QLabel("Tile size")
        l2.setObjectName("Muted")
        form_row.addWidget(l2)
        self.tile_combo = QComboBox()
        self.tile_combo.addItem("Auto (VRAM aware)", 0)
        for size in (128, 192, 256, 384, 512, 640, 768):
            self.tile_combo.addItem(f"{size} px", size)
        self.tile_combo.currentIndexChanged.connect(self._tile_changed)
        form_row.addWidget(self.tile_combo)
        form_row.addStretch(1)
        adv.add_layout(form_row)
        note = QLabel(
            "Smaller tiles use less VRAM and run slower. Auto picks a size from "
            "detected VRAM and halves it automatically if a frame runs out of "
            "memory.")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 11px;")
        adv.add(note)
        root.addWidget(adv)
        root.addStretch(1)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        gpu = gpu_monitor.query_gpu()
        sysstats = gpu_monitor.system_stats()
        info = self.ctx.device.info()

        if gpu.available:
            self.status_badge.setText("GPU ACTIVE")
            self.status_badge.set_kind("green")
            self.gauge.set_value(gpu.utilization)
            self.gpu_name.setText(gpu.name)
            self.gpu_sub.setText(f"{gpu.used_vram_mb:,} / {gpu.total_vram_mb:,} MB  ·  "
                                 f"{gpu.vram_percent:.0f}% used")
            self.rows["vram"].set_value(f"{gpu.used_vram_mb:,} / {gpu.total_vram_mb:,} MB")
            self.rows["temp"].set_value(f"{gpu.temperature_c} °C")
            self.rows["power"].set_value(f"{gpu.power_w:.0f} W")
            self.rows["driver"].set_value(gpu.driver or "-")
            self.rows["cuda"].set_value(gpu.cuda_version or info.cuda_version or "-")
            self.vram_bar.setValue(int(gpu.vram_percent * 10))
            self.vram_label.setText(f"{gpu.free_vram_mb:,} MB free")
            rec_tile = gpu_monitor.recommend_tile_size(gpu.total_vram_mb)
            rec_batch = gpu_monitor.recommend_batch_frames(gpu.total_vram_mb)
            self.tuning_hint.setText(
                f"Recommended: tile {'full frame' if rec_tile == 0 else str(rec_tile) + ' px'}"
                f"  ·  batch {rec_batch} frames  ·  fp16 inference. "
                "BasicVSR++ needs roughly 6 GB VRAM at 720p input.")
            self.notice.setVisible(False)
        else:
            self.status_badge.setText("CPU MODE")
            self.status_badge.set_kind("amber")
            self.gauge.set_value(0)
            self.gpu_name.setText("No NVIDIA GPU detected")
            self.gpu_sub.setText("Processing still works on CPU - expect roughly "
                                 "10-40x slower AI inference.")
            self.rows["vram"].set_value("-")
            self.rows["temp"].set_value("-")
            self.rows["power"].set_value("-")
            self.rows["driver"].set_value("-")
            self.rows["cuda"].set_value("-")
            self.vram_bar.setValue(0)
            self.vram_label.setText("")
            self.notice.show_message(
                "No CUDA device found. Install the NVIDIA driver and the CUDA "
                "build of PyTorch (see README) to enable GPU acceleration.", "warn")

        self.rows["torch"].set_value(info.torch_version or "-")
        self.sys_rows["cpu"].set_value((sysstats.cpu_name or "-")[:42])
        self.sys_rows["cpu_use"].set_value(f"{sysstats.cpu_percent:.0f}%  ·  "
                                           f"{sysstats.cpu_count} threads")
        self.sys_rows["ram"].set_value(f"{sysstats.ram_used_gb:.1f} / "
                                       f"{sysstats.ram_total_gb:.1f} GB")
        self.sys_rows["ram_use"].set_value(f"{sysstats.ram_percent:.0f}%")
        self.sys_rows["os"].set_value(sysstats.os_name or "-")
        device_text = info.label
        if len(device_text) > 38:
            # keep the row tidy; the full name is in the tooltip and the CPU row
            device_text = ("CPU" if info.device == "cpu" else info.name[:34])
            if info.device != "cpu":
                device_text = f"{info.name[:26]}... (CUDA {info.cuda_version})"
        self.sys_rows["device"].set_value(device_text)
        try:
            hw = self.ctx.ffmpeg.nvenc_works("h264")
            self.sys_rows["encoder"].set_value(
                "NVENC available" if hw else "libx264 (CPU)")
        except Exception:
            self.sys_rows["encoder"].set_value("-")

    def _rescan(self) -> None:
        gpu_monitor.detect_gpus.__globals__["_STATIC_CACHE"].clear()
        self.ctx.device.info(refresh=True)
        self.refresh()
        self.notice.show_message("Hardware rescan complete.", "success")

    def _device_changed(self) -> None:
        value = self.device_combo.currentData()
        self.ctx.config.set("processing.device", value)
        self.ctx.device.preferred = value
        self.ctx.device.info(refresh=True)
        self.refresh()

    def _tile_changed(self) -> None:
        size = int(self.tile_combo.currentData() or 0)
        self.ctx.config.set("processing.tile_size", size)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh()
