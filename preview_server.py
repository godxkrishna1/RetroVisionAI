#!/usr/bin/env python3
"""Tiny static server for the RetroVision AI preview pack.

Serves docs/site (the screenshot tour) plus the real artefacts produced by the
pipeline, so the preview is a single browsable page:

    python preview_server.py            # http://127.0.0.1:8000
    python preview_server.py --port 8080
"""
from __future__ import annotations

import argparse
import mimetypes
import os
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "docs" / "site"
DOCS = ROOT / "docs"

mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("image/png", ".png")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE), **kwargs)

    def do_GET(self) -> None:                       # noqa: N802
        path = self.path.split("?")[0]
        # expose the generated videos / raw screenshots next to the tour
        if path.startswith("/media/"):
            return self._serve_from(DOCS / "media", path[len("/media/"):])
        if path.startswith("/shots/"):
            return self._serve_from(DOCS / "screenshots", path[len("/shots/"):])
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def _serve_from(self, base: Path, rel: str) -> None:
        target = (base / rel).resolve()
        if base.resolve() not in target.parents and target != base.resolve():
            self.send_error(403, "outside preview roots")
            return
        if not target.is_file():
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         mimetypes.guess_type(str(target))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(target.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(target, "rb") as fh:
            shutil.copyfileobj(fh, self.wfile)

    def log_message(self, fmt: str, *args) -> None:  # keep the console quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    (DOCS / "media").mkdir(exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"RetroVision AI preview  ->  http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
