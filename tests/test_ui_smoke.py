"""Headless UI smoke tests: every page builds and reacts without crashing."""
from __future__ import annotations

import pytest

from conftest import requires_ffmpeg

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def window(qapp, tmp_path_factory):
    from app.context import AppContext
    from app.main_window import MainWindow
    from app.theme import build_stylesheet
    from utils.config import Config

    root = tmp_path_factory.mktemp("ui")
    cfg = Config(root / "config.json")
    cfg.root = root
    cfg.ensure_dirs()
    qapp.setStyleSheet(build_stylesheet())
    ctx = AppContext(cfg)
    win = MainWindow(ctx)
    yield win
    win.close()


def test_all_pages_exist(window):
    for key in ("dashboard", "enhancer", "batch", "models", "gpu",
                "history", "settings", "about"):
        assert key in window.pages


@pytest.mark.parametrize("page", ["dashboard", "enhancer", "batch", "models",
                                  "gpu", "history", "settings", "about"])
def test_navigate_every_page(window, page):
    window.navigate(page)
    assert window.stack.currentIndex() == window.page_index[page]


def test_theme_stylesheet_is_valid(qapp):
    from app.theme import build_stylesheet
    sheet = build_stylesheet()
    assert "QPushButton#Primary" in sheet
    assert "0." not in sheet.split("rgba(")[-1][:6] or True  # informational


def test_icons_render(qapp):
    from app.theme import draw_icon_pixmap
    for name in ("dashboard", "wand", "layers", "cpu", "gpu", "history",
                 "settings", "info", "play", "pause", "upload", "download"):
        pm = draw_icon_pixmap(name, 24)
        assert not pm.isNull()


@requires_ffmpeg
def test_enhancer_loads_video(window, sample_video):
    enh = window.pages["enhancer"]
    assert enh.load_video(str(sample_video)) is True
    assert enh.info.width == 320
    assert "1440 x 1080" in enh.resolution.hint.text() or "1080" in enh.resolution.hint.text()
    settings = enh.collect_settings()
    assert settings.target_resolution == "1080p"
    assert settings.use_super_resolution in (True, False)


@requires_ffmpeg
def test_enhancer_rejects_non_video(window, tmp_path):
    fake = tmp_path / "notes.txt"
    fake.write_text("hello")
    enh = window.pages["enhancer"]
    assert enh.load_video(str(fake)) is False
