"""GPU telemetry never crashes, even without a GPU."""
from __future__ import annotations

from utils import gpu_monitor


def test_detect_returns_at_least_one_entry():
    gpus = gpu_monitor.detect_gpus()
    assert len(gpus) >= 1
    assert all(hasattr(g, "name") for g in gpus)


def test_query_gpu_never_raises():
    stats = gpu_monitor.query_gpu()
    assert 0 <= stats.utilization <= 100 or stats.utilization == 0
    assert stats.total_vram_mb >= 0


def test_system_stats():
    stats = gpu_monitor.system_stats()
    assert stats.cpu_count >= 1
    assert stats.ram_total_gb >= 0


def test_tile_recommendations_scale_with_vram():
    assert gpu_monitor.recommend_tile_size(0) <= 128
    assert gpu_monitor.recommend_tile_size(24000) == 0
    assert gpu_monitor.recommend_tile_size(2000) <= 192
    assert gpu_monitor.recommend_batch_frames(24000) >= 4
    assert gpu_monitor.recommend_batch_frames(2000) >= 1
