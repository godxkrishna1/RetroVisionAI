"""Model registry integrity and ModelManager behaviour (no network)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai import registry
from ai.model_manager import ModelManager, sha256_file


def test_registry_entries_are_complete():
    for spec in registry.all_models():
        assert spec.url.startswith("https://"), spec.key
        assert len(spec.sha256) == 64, spec.key
        assert spec.size_bytes > 0
        assert spec.license and spec.author and spec.description
        assert spec.category in ("upscale", "face", "interpolation",
                                 "temporal", "auxiliary")


def test_registry_keys_unique_and_lookup():
    keys = [s.key for s in registry.all_models()]
    assert len(keys) == len(set(keys))
    assert registry.get_model("gfpgan_v14").filename == "GFPGANv1.4.pth"
    assert registry.get_model("does_not_exist") is None


def test_dependency_resolution_adds_spynet():
    resolved = registry.resolve_with_dependencies(["basicvsr_pp"])
    keys = [s.key for s in resolved]
    assert "basicvsr_pp" in keys and "spynet" in keys


def test_upscale_models_have_scale():
    for spec in registry.upscale_models():
        assert spec.scale in (2, 4)


def test_status_reports_missing_models(tmp_path):
    mm = ModelManager(tmp_path / "models")
    statuses = mm.model_status()
    assert len(statuses) == len(registry.all_models())
    assert all(not s.installed for s in statuses)
    assert mm.resolve_upscaler("realesrgan_x2plus") is None


def test_detects_installed_file_with_correct_size(tmp_path):
    mm = ModelManager(tmp_path / "models")
    spec = registry.get_model("spynet")
    fake = mm.path_for("spynet")
    # Create a file of exactly the declared size (content ignored here)
    with open(fake, "wb") as fh:
        fh.truncate(spec.size_bytes)
    assert mm.is_installed("spynet")


def test_wrong_size_is_not_installed(tmp_path):
    mm = ModelManager(tmp_path / "models")
    mm.path_for("spynet").write_bytes(b"too small")
    assert mm.is_installed("spynet") is False


def test_enable_disable_persists(tmp_path):
    mm = ModelManager(tmp_path / "models")
    mm.set_enabled("rife47", False)
    assert mm.is_enabled("rife47") is False
    reloaded = ModelManager(tmp_path / "models")
    assert reloaded.is_enabled("rife47") is False


def test_manifest_written(tmp_path):
    mm = ModelManager(tmp_path / "models")
    mm.set_enabled("rife47", False)
    manifest = json.loads((tmp_path / "models" / "manifest.json").read_text())
    assert manifest["rife47"]["enabled"] is False


def test_sha256_file(tmp_path):
    target = tmp_path / "x.bin"
    target.write_bytes(b"abc" * 1000)
    progress = []
    digest = sha256_file(target, progress=lambda d, t: progress.append((d, t)))
    assert len(digest) == 64
    assert progress and progress[-1][0] == progress[-1][1] == 3000


def test_missing_for_includes_dependencies(tmp_path):
    mm = ModelManager(tmp_path / "models")
    missing = {m.key for m in mm.missing_for(["basicvsr_pp"])}
    assert {"basicvsr_pp", "spynet"} <= missing


def test_resolve_upscaler_falls_back(tmp_path):
    mm = ModelManager(tmp_path / "models")
    spec = registry.get_model("realesr_general_x4v3")
    with open(mm.path_for("realesr_general_x4v3"), "wb") as fh:
        fh.truncate(spec.size_bytes)
    assert mm.resolve_upscaler("realesrgan_x2plus") == "realesr_general_x4v3"
