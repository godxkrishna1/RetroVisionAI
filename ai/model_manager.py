"""Model lifecycle: status, verified downloads, enable/disable, removal.

Design rules
------------
* Never download silently - :meth:`download` is only called by an explicit
  user action in the Model Manager page.
* Every download is streamed to ``<file>.part`` then *atomically* renamed
  after the SHA-256 digest matches the one recorded in ``ai.registry``.
* A ``models/manifest.json`` records what we installed (version, digest,
  install time) so integrity can be re-checked later without re-hashing
  everything on every launch.
* Models can be disabled without deleting the weights.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from utils.logger import get_logger

from .registry import CATALOGUE, ModelSpec, all_models, get_model

log = get_logger("models")

ProgressCb = Callable[[int, int], None]        # (downloaded, total) bytes
StatusCb = Callable[[str], None]               # human readable status


class DownloadCancelled(Exception):
    pass


class ChecksumError(RuntimeError):
    pass


@dataclass
class ModelStatus:
    key: str
    title: str
    short_name: str
    category: str
    filename: str
    installed: bool
    enabled: bool
    size_bytes: int
    path: Path
    version: str
    license: str
    author: str
    description: str
    verified: bool = False
    corrupted: bool = False
    homepage: str = ""
    scale: int = 4
    quality: str = ""
    speed: str = ""


def sha256_file(path: Path, progress: Optional[ProgressCb] = None,
                cancel: Optional[threading.Event] = None,
                chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with open(path, "rb") as fh:
        while True:
            if cancel is not None and cancel.is_set():
                raise DownloadCancelled()
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
            done += len(block)
            if progress:
                progress(done, total)
    return digest.hexdigest()


class ModelManager:
    """Owns the ``models/`` folder."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.models_dir / "manifest.json"
        self._lock = threading.RLock()
        self._manifest: Dict[str, dict] = {}
        self._load_manifest()

    # -- manifest ------------------------------------------------------
    def _load_manifest(self) -> None:
        try:
            if self.manifest_path.exists():
                self._manifest = json.loads(self.manifest_path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            self._manifest = {}

    def _save_manifest(self) -> None:
        try:
            tmp = self.manifest_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._manifest, indent=2), "utf-8")
            os.replace(tmp, self.manifest_path)
        except OSError as exc:
            log.warning("Could not write model manifest: %s", exc)

    # -- queries -------------------------------------------------------
    def path_for(self, spec: ModelSpec | str) -> Path:
        key = spec if isinstance(spec, str) else spec.key
        sp = get_model(key)
        if sp is None:
            return self.models_dir / key
        return self.models_dir / sp.filename

    def is_installed(self, key: str) -> bool:
        spec = get_model(key)
        if spec is None:
            return False
        p = self.path_for(spec)
        return p.exists() and p.stat().st_size == spec.size_bytes

    def is_enabled(self, key: str) -> bool:
        return bool(self._manifest.get(key, {}).get("enabled", True))

    def set_enabled(self, key: str, enabled: bool) -> None:
        with self._lock:
            entry = self._manifest.setdefault(key, {})
            entry["enabled"] = bool(enabled)
            self._save_manifest()

    def model_status(self, key: Optional[str] = None) -> List[ModelStatus]:
        specs = [get_model(key)] if key else all_models()
        out: List[ModelStatus] = []
        for spec in specs:
            if spec is None:
                continue
            p = self.path_for(spec)
            installed = p.exists() and p.stat().st_size == spec.size_bytes
            entry = self._manifest.get(spec.key, {})
            out.append(ModelStatus(
                key=spec.key, title=spec.title, short_name=spec.short_name or spec.key,
                category=spec.category, filename=spec.filename,
                installed=installed, enabled=entry.get("enabled", True),
                size_bytes=spec.size_bytes, path=p, version=spec.version,
                license=spec.license, author=spec.author,
                description=spec.description,
                verified=bool(entry.get("sha256")) and entry.get("sha256") == spec.sha256,
                corrupted=entry.get("corrupted", False),
                homepage=spec.homepage, scale=spec.scale,
                quality=spec.quality, speed=spec.speed))
        return out

    def installed_keys(self) -> List[str]:
        return [k for k in CATALOGUE if self.is_installed(k) and self.is_enabled(k)]

    def available_upscalers(self) -> List[ModelStatus]:
        return [s for s in self.model_status()
                if s.category == "upscale" and s.installed and s.enabled]

    def resolve_upscaler(self, requested: str) -> Optional[str]:
        """Return an installed, enabled upscaler key (falling back gracefully)."""
        if requested and self.is_installed(requested) and self.is_enabled(requested):
            return requested
        for fallback in ("realesrgan_x2plus", "realesrgan_x4plus",
                         "realesr-general-x4v3", "realesrgan_anime6b",
                         "realesr_animevideo"):
            if self.is_installed(fallback) and self.is_enabled(fallback):
                if requested and requested != fallback:
                    log.warning("Model %s unavailable - using %s", requested, fallback)
                return fallback
        pooled = self.available_upscalers()
        return pooled[0].key if pooled else None

    # -- verification --------------------------------------------------
    def verify(self, key: str, progress: Optional[ProgressCb] = None,
               cancel: Optional[threading.Event] = None) -> bool:
        spec = get_model(key)
        if spec is None:
            return False
        p = self.path_for(spec)
        if not p.exists():
            return False
        if p.stat().st_size != spec.size_bytes:
            log.warning("%s has wrong size (%d vs %d)", p.name,
                        p.stat().st_size, spec.size_bytes)
            self._mark_corrupted(key, True)
            return False
        digest = sha256_file(p, progress, cancel)
        ok = digest == spec.sha256
        if not ok:
            log.error("Checksum mismatch for %s: %s != %s", p.name, digest, spec.sha256)
            self._mark_corrupted(key, True)
        else:
            with self._lock:
                entry = self._manifest.setdefault(key, {})
                entry.update({"sha256": digest, "verified_at": time.time(),
                              "corrupted": False, "version": spec.version})
                self._save_manifest()
        return ok

    def _mark_corrupted(self, key: str, corrupted: bool) -> None:
        with self._lock:
            entry = self._manifest.setdefault(key, {})
            entry["corrupted"] = corrupted
            self._save_manifest()

    # -- downloads -----------------------------------------------------
    def download(self, key: str, progress: Optional[ProgressCb] = None,
                 status: Optional[StatusCb] = None,
                 cancel: Optional[threading.Event] = None,
                 retries: int = 3) -> Path:
        """Download + verify one model.  Raises on failure.  Blocking."""
        import requests

        spec = get_model(key)
        if spec is None:
            raise KeyError(f"Unknown model: {key}")
        if self.is_installed(key):
            return self.path_for(spec)

        target = self.path_for(spec)
        part = target.with_suffix(target.suffix + ".part")
        headers = {"User-Agent": "RetroVisionAI/1.0 (model downloader)"}

        last_error: Optional[Exception] = None
        for attempt in range(1, max(1, retries) + 1):
            try:
                if status:
                    status(f"Connecting (attempt {attempt}/{retries})...")
                with requests.get(spec.url, stream=True, timeout=(15, 60),
                                  headers=headers, allow_redirects=True) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("Content-Length") or spec.size_bytes)
                    done = 0
                    t0 = time.time()
                    with open(part, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=1024 * 256):
                            if cancel is not None and cancel.is_set():
                                raise DownloadCancelled()
                            if not chunk:
                                continue
                            fh.write(chunk)
                            done += len(chunk)
                            if progress:
                                progress(done, total)
                            if status and time.time() - t0 > 0.4:
                                speed = done / max(time.time() - t0, 1e-3) / 1048576
                                status(f"Downloading - {done / 1048576:.1f} / "
                                       f"{total / 1048576:.1f} MB ({speed:.1f} MB/s)")
                                t0 = time.time()
                if status:
                    status("Verifying SHA-256 checksum...")
                digest = sha256_file(part)
                if digest != spec.sha256:
                    part.unlink(missing_ok=True)
                    raise ChecksumError(
                        f"Checksum mismatch for {spec.filename}\n"
                        f"expected {spec.sha256}\ngot      {digest}\n"
                        "The file was discarded - please retry.")
                os.replace(part, target)
                with self._lock:
                    self._manifest[key] = {
                        "sha256": digest, "version": spec.version,
                        "installed_at": time.time(), "enabled": True,
                        "corrupted": False, "source": spec.url,
                    }
                    self._save_manifest()
                if status:
                    status("Installed and verified")
                log.info("Model installed: %s (%s)", spec.filename, spec.display_size)
                return target
            except DownloadCancelled:
                part.unlink(missing_ok=True)
                raise
            except ChecksumError:
                raise
            except Exception as exc:  # network problems - retry
                last_error = exc
                part.unlink(missing_ok=True)
                log.warning("Download attempt %d/%d failed: %s",
                            attempt, retries, exc)
                if attempt < retries:
                    time.sleep(min(2 * attempt, 8))
        raise RuntimeError(f"Download failed after {retries} attempts: {last_error}")

    def remove(self, key: str) -> bool:
        """Delete a model file (user action from the manager page)."""
        spec = get_model(key)
        if spec is None:
            return False
        p = self.path_for(spec)
        try:
            p.unlink(missing_ok=True)
            with self._lock:
                self._manifest.pop(key, None)
                self._save_manifest()
            log.info("Model removed: %s", p.name)
            return True
        except OSError as exc:
            log.error("Could not remove %s: %s", p, exc)
            return False

    # -- misc ----------------------------------------------------------
    def disk_usage_bytes(self) -> int:
        total = 0
        for p in self.models_dir.glob("*.pth"):
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    def generate_checksums_file(self, out_path: Optional[Path] = None) -> Path:
        """Write models/SHA256SUMS for manual verification."""
        out = out_path or (self.models_dir / "SHA256SUMS.txt")
        lines = []
        for m in all_models():
            p = self.path_for(m)
            if p.exists():
                lines.append(f"{sha256_file(p)}  {m.filename}")
        out.write_text("\n".join(lines) + "\n", "utf-8")
        return out

    def missing_for(self, keys: List[str]) -> List[ModelSpec]:
        """Models (including dependencies) from ``keys`` not yet installed."""
        from .registry import resolve_with_dependencies
        out: List[ModelSpec] = []
        for spec in resolve_with_dependencies(keys):
            if not self.is_installed(spec.key):
                out.append(spec)
        return out
