"""Loopback-only, read-only HTTP viewer for a telemetry directory."""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .telemetry import SCHEMA


def make_server(directory, *, port=8765, mode="replay"):
    directory = Path(directory)
    metadata = json.loads((directory / "session.json").read_text())
    if metadata.get("schema") != SCHEMA or mode not in {"replay", "live"}:
        raise ValueError("Unsupported session or mode")
    assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
              "/style.css": ("style.css", "text/css")}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            try:
                if parsed.path in assets:
                    name, content_type = assets[parsed.path]
                    payload = files("decisionvla.dashboard").joinpath("static", name).read_bytes()
                elif parsed.path == "/api/state":
                    index = json.loads((directory / "index.json").read_text())
                    status_path = directory / "recorder-status.json"
                    status = json.loads(status_path.read_text()) if status_path.exists() else None
                    outcome_path = directory / "episodes.json"
                    outcomes = json.loads(outcome_path.read_text()) if outcome_path.exists() else {}
                    payload = json.dumps({"metadata": metadata, "mode": mode, **index,
                                          "recorder": status, "outcomes": outcomes}, allow_nan=False).encode()
                    content_type = "application/json"
                elif parsed.path == "/api/frame":
                    frame_id = parse_qs(parsed.query).get("id", [""])[0]
                    if not re.fullmatch(r"[0-9]{1,12}", frame_id):
                        self.send_error(400, "Invalid frame id")
                        return
                    payload = (directory / "frames" / f"{int(frame_id):06d}.json").read_bytes()
                    content_type = "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type + "; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; "
                                 "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(payload)
            except FileNotFoundError:
                self.send_error(404, "Frame unavailable")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mode", choices=["live", "replay"], default="replay")
    args = parser.parse_args()
    with make_server(args.session, port=args.port, mode=args.mode) as server:
        print(f"DecisionVLA observatory: http://127.0.0.1:{server.server_port} ({args.mode})", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
