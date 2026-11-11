"""Config defaults, merge behaviour, persistence and path resolution."""
from __future__ import annotations

import json

from utils.config import DEFAULTS, Config, app_root


def test_defaults_are_used_when_file_missing(tmp_path):
    cfg = Config(tmp_path / "config.json")
    assert cfg.get("export.codec") == "h264"
    assert cfg.get("enhancement.target_resolution") == "1080p"
    assert (tmp_path / "config.json").exists(), "config is persisted on first run"


def test_get_set_and_persistence(tmp_path):
    path = tmp_path / "config.json"
    cfg = Config(path)
    cfg.set("export.crf", 21)
    cfg.set("processing.device", "cpu")

    reloaded = Config(path)
    assert reloaded.get("export.crf") == 21
    assert reloaded.get("processing.device") == "cpu"


def test_partial_file_is_merged_with_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"export": {"crf": 30}}), "utf-8")
    cfg = Config(path)
    assert cfg.get("export.crf") == 30
    assert cfg.get("export.codec") == DEFAULTS["export"]["codec"]


def test_corrupt_file_falls_back_and_backs_up(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{ this is not json", "utf-8")
    cfg = Config(path)
    assert cfg.get("app.name") == "RetroVision AI"
    assert path.with_suffix(".json.bak").exists()


def test_unknown_key_returns_default(tmp_path):
    cfg = Config(tmp_path / "config.json")
    assert cfg.get("nope.nothing", "fallback") == "fallback"


def test_relative_paths_resolve_against_app_root(tmp_path, monkeypatch):
    from utils import config as config_module

    cfg = Config(tmp_path / "config.json")
    monkeypatch.setattr(config_module, "app_root", lambda: tmp_path)
    cfg.root = tmp_path
    assert cfg.output_dir == tmp_path / "output"
    cfg.set("paths.output_dir", str(tmp_path / "custom"))
    assert cfg.output_dir == tmp_path / "custom"


def test_ensure_dirs_creates_folders(tmp_path):
    cfg = Config(tmp_path / "config.json")
    cfg.root = tmp_path
    cfg.ensure_dirs()
    for name in ("models", "output", "temp", "logs"):
        assert (tmp_path / name).is_dir()


def test_update_deep_merges(tmp_path):
    cfg = Config(tmp_path / "config.json")
    cfg.update({"export": {"codec": "h265"}})
    assert cfg.get("export.codec") == "h265"
    assert cfg.get("export.quality") == DEFAULTS["export"]["quality"]


def test_app_root_is_project_dir():
    assert (app_root() / "main.py").exists()
