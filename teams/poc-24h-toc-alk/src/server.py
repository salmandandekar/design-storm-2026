"""Same-origin static and prediction-refresh server."""
from __future__ import annotations

import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import Config
from .live_data import refresh


def serve(cfg: Config, port: int = 8765) -> None:
    root = cfg.path.parents[2]
    latest_path = (
        root / "teams" / "explainable-viz" / "viz-data" /
        "latest-prediction.json"
    )

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cache-Control", "no-store, must-revalidate")
            super().end_headers()

        def _json(self, status: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/predictions/latest":
                if not latest_path.exists():
                    self._json(404, {"status": "unavailable", "error": "model not trained"})
                    return
                self._json(200, json.loads(latest_path.read_text(encoding="utf-8")))
                return
            path_part, _, query = self.path.partition("?")
            relative = path_part.lstrip("/")
            if relative and not relative.endswith("/"):
                candidate = root / relative
                if not candidate.exists() and candidate.with_suffix(".html").is_file():
                    self.path = "/" + candidate.with_suffix(".html").relative_to(root).as_posix()
                    if query:
                        self.path += "?" + query
            super().do_GET()

        def do_POST(self):
            if self.path != "/api/predictions/refresh":
                self._json(404, {"status": "error", "error": "unknown endpoint"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1024:
                self._json(413, {"status": "error", "error": "request too large"})
                return
            if length:
                self.rfile.read(length)
            try:
                self._json(200, refresh(cfg))
            except Exception as exc:
                stale = None
                if latest_path.exists():
                    stale = json.loads(latest_path.read_text(encoding="utf-8"))
                    stale["status"] = "stale"
                    stale["refresh_error"] = str(exc)
                self._json(502, stale or {"status": "error", "error": str(exc)})

        def log_message(self, fmt, *args):
            super().log_message(fmt, *args)

    os.chdir(root)
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"Serving {root} at http://localhost:{port}/")
        print("Prediction refresh API enabled; source readings remain provisional.")
        httpd.serve_forever()
