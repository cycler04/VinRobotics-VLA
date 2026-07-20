"""Serve a local copy of episode videos to the Caption QA editor.

Run this on your OWN laptop (B or C) after downloading videos with
``download_videos.py``. The editor (served from the shared server on laptop A)
can then be told to load video bytes from here — ``http://localhost:8090`` — so
playback/seeking comes off your local disk instead of streaming over slow wifi.
Everything else (assignments, SRT text, saving) still talks to laptop A.

It maps ``GET /video/<video_id>`` to ``<root>/<video_id>.mp4`` and supports HTTP
range requests, so scrubbing in the editor works. Pure standard library — no
pip installs needed.

    python scripts/local_video_server.py ~/caption_videos          # port 8090
    python scripts/local_video_server.py ~/caption_videos --port 9000

Bound to localhost only (just your machine can reach it). Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

# Set once in main(); the directory holding ``<video_id>.mp4`` files.
VIDEO_ROOT: Path = Path()

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def _resolve(video_id: str) -> Path | None:
    """Map a request path to a file under VIDEO_ROOT, or None if it's unsafe or
    absent. We guard traversal *lexically* (reject absolute paths and ``..``)
    rather than by resolving the file — so a symlink farm (rsync without ``-L``)
    serves fine too, not just a real ``rsync -L`` copy. VIDEO_ROOT is already an
    absolute, resolved path, so a clean relative video_id can't escape it."""
    rel = Path(video_id)
    if rel.is_absolute() or any(part == ".." for part in rel.parts):
        return None  # e.g. /etc/passwd or ../../secret
    candidate = VIDEO_ROOT / f"{video_id}.mp4"
    return candidate if candidate.is_file() else None  # .is_file() follows symlinks


class VideoHandler(BaseHTTPRequestHandler):
    server_version = "CaptionQALocalVideo/1.0"
    # Keep-alive so the browser reuses one connection across the many small range
    # requests it makes while scrubbing. We always send an accurate Content-Length.
    protocol_version = "HTTP/1.1"

    def _send_cors(self) -> None:
        # The editor page is served from laptop A (a different origin), so allow
        # it to fetch from here. Media playback doesn't strictly require CORS,
        # but this avoids any edge cases.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")

    def do_OPTIONS(self) -> None:  # noqa: N802  (stdlib naming)
        self.send_response(204)
        self._send_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve(head_only=True)

    def do_GET(self) -> None:  # noqa: N802
        self._serve(head_only=False)

    def _serve(self, head_only: bool) -> None:
        path = urlsplit(self.path).path  # drop any ?res= query the editor appends
        if not path.startswith("/video/"):
            self.send_error(404, "Not found")
            return

        video_id = unquote(path[len("/video/"):])
        target = _resolve(video_id)
        if target is None:
            # Not downloaded locally — tell the editor so it can fall back to A.
            self.send_error(404, "Video not in local cache")
            return

        size = target.stat().st_size
        start, end = 0, size - 1
        status = 200
        range_header = self.headers.get("Range")
        if range_header:
            m = _RANGE_RE.search(range_header)
            if m:
                g1, g2 = m.group(1), m.group(2)
                if g1 == "" and g2:                 # bytes=-N  (last N bytes)
                    start = max(0, size - int(g2))
                else:
                    start = int(g1)
                    if g2:
                        end = min(int(g2), size - 1)
                if start > end or start >= size:
                    self.send_response(416)         # range not satisfiable
                    self.send_header("Content-Range", f"bytes */{size}")
                    self._send_cors()
                    self.end_headers()
                    return
                status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self._send_cors()
        self.end_headers()

        if head_only:
            return
        with open(target, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return  # browser closed the connection (seek/abort) — normal
                remaining -= len(chunk)

    def log_message(self, fmt: str, *args) -> None:
        # Quieter than the default; one concise line per request.
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_root", help="Folder holding downloaded <video_id>.mp4 files")
    parser.add_argument("--port", type=int, default=8090, help="Port to listen on (default 8090)")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default 127.0.0.1 = this machine only)",
    )
    args = parser.parse_args()

    global VIDEO_ROOT
    VIDEO_ROOT = Path(args.video_root).expanduser().resolve()
    if not VIDEO_ROOT.is_dir():
        sys.exit(f"Video root does not exist: {VIDEO_ROOT}")

    count = sum(1 for _ in VIDEO_ROOT.rglob("*.mp4"))
    server = ThreadingHTTPServer((args.host, args.port), VideoHandler)
    print(f"Serving {count} local videos from {VIDEO_ROOT}")
    print(f"Listening on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    print("In the editor, turn on 'Local videos' to use this server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
