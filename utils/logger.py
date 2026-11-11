"""Logging setup for RetroVision AI.

One rotating log file per run plus an in-memory ring buffer that the UI can
read to display live log lines.  A ``logging.Handler`` subclass forwards
records to any callables registered via :func:`add_listener` (the GUI hooks
this to a Qt signal).
"""
from __future__ import annotations

import collections
import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from typing import Callable, Deque, List, Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-26s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

_LOGGERS: dict = {}
_RING: Deque[str] = collections.deque(maxlen=4000)
_LISTENERS: List[Callable[[str, str, str], None]] = []
_LOCK = threading.Lock()
_LOG_DIR: Optional[Path] = None


class _RingBufferHandler(logging.Handler):
    """Keep recent log records in memory and fan out to registered listeners."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:  # pragma: no cover - formatting must never crash
            msg = record.getMessage()
        with _LOCK:
            _RING.append(msg)
            listeners = list(_LISTENERS)
        for fn in listeners:
            try:
                fn(record.levelname, record.name, record.getMessage())
            except Exception:  # pragma: no cover - listener errors are non-fatal
                pass


def add_listener(fn: Callable[[str, str, str], None]) -> None:
    """Register a live log listener ``fn(level, logger_name, message)``."""
    with _LOCK:
        if fn not in _LISTENERS:
            _LISTENERS.append(fn)


def remove_listener(fn: Callable[[str, str, str], None]) -> None:
    with _LOCK:
        if fn in _LISTENERS:
            _LISTENERS.remove(fn)


def recent_lines(limit: int = 500) -> List[str]:
    """Return the most recent formatted log lines (oldest first)."""
    with _LOCK:
        items = list(_RING)
    return items[-limit:]


def setup_logging(log_dir: Path, level: int = logging.INFO,
                  console: bool = True) -> Path:
    """Configure the root logger.  Safe to call more than once."""
    global _LOG_DIR
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    _LOG_DIR = log_dir
    log_file = log_dir / "retrovision.log"

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers when called again (e.g. tests)
    for h in list(root.handlers):
        if getattr(h, "_rv_handler", False):
            root.removeHandler(h)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=8 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(formatter)
    fh._rv_handler = True  # type: ignore[attr-defined]
    root.addHandler(fh)

    ring = _RingBufferHandler()
    ring.setFormatter(formatter)
    ring._rv_handler = True  # type: ignore[attr-defined]
    root.addHandler(ring)

    if console and not getattr(sys, "frozen", False):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(formatter)
        sh._rv_handler = True  # type: ignore[attr-defined]
        root.addHandler(sh)

    # Third-party noise control
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging initialised -> %s", log_file)
    return log_file


def get_logger(name: str) -> logging.Logger:
    """Cached logger factory (keeps ``retrovision.*`` hierarchy tidy)."""
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(f"rv.{name}" if not name.startswith("rv.") else name)
    _LOGGERS[name] = logger
    return logger
