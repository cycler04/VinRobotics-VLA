"""Serve a local copy of episode videos to the Caption QA editor (EgoVerse).

Run this on your OWN laptop after downloading videos. The reviewer (served from
the shared server at http://100.89.98.89:7871/reviewer) can then be told to
load video bytes from here — ``http://localhost:8090`` — so playback/seeking
comes off your local disk instead of streaming over slow wifi. Everything else
(assignments, SRT text, saving) still talks to laptop A.

It maps ``GET /video/<video_id>`` to ``<root>/<video_id>.mp4`` and supports HTTP
range requests, so scrubbing in the editor works. Pure standard library — no
pip installs needed.

    # Simplest: use the global command:
    open_cut
    # Or point it at a folder explicitly:
    python3 src/local_video_server_verse.py dataset/EgoVerse_Label/26ai.dungnn_assigned_videos_batch1of1 --port 8090

Bound to localhost only (just your machine can reach it). Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

# Set once in main(); the directory holding ``<video_id>.mp4`` files.
VIDEO_ROOT: Path = Path()
VIDEO_INDEX: dict[str, Path] = {}

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def build_video_index(root: Path) -> int:
    """Build an in-memory index mapping video identifiers (relative paths, stems, basenames)
    to actual Path objects on disk."""
    global VIDEO_INDEX
    VIDEO_INDEX.clear()
    unique_files: set[Path] = set()
    for path in root.rglob("*.mp4"):
        if path.is_file():
            unique_files.add(path)
            try:
                rel = path.relative_to(root)
                rel_str = str(rel)
                VIDEO_INDEX[rel_str] = path
                if rel_str.endswith(".mp4"):
                    VIDEO_INDEX[rel_str[:-4]] = path
            except ValueError:
                pass
            VIDEO_INDEX[path.name] = path
            VIDEO_INDEX[path.stem] = path
    return len(unique_files)


def _resolve(video_id: str) -> Path | None:
    """Map a request path to a file under VIDEO_ROOT, or None if it's unsafe or
    absent."""
    rel = Path(video_id)
    if rel.is_absolute() or any(part == ".." for part in rel.parts):
        return None

    if video_id in VIDEO_INDEX:
        return VIDEO_INDEX[video_id]
    if video_id.endswith(".mp4") and video_id[:-4] in VIDEO_INDEX:
        return VIDEO_INDEX[video_id[:-4]]
    if f"{video_id}.mp4" in VIDEO_INDEX:
        return VIDEO_INDEX[f"{video_id}.mp4"]

    candidate = VIDEO_ROOT / f"{video_id}.mp4"
    if candidate.is_file():
        return candidate
    candidate_raw = VIDEO_ROOT / video_id
    if candidate_raw.is_file():
        return candidate_raw
    return None


class VideoHandler(BaseHTTPRequestHandler):
    server_version = "CaptionQALocalVideo/1.0"
    # Keep-alive so the browser reuses one connection across the many small range
    # requests it makes while scrubbing. We always send an accurate Content-Length.
    protocol_version = "HTTP/1.1"

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type, Authorization")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges")

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
        if path in ("/", "/ping", "/health", "/status"):
            body = b'{"status":"ok","server":"CaptionQALocalVideo/1.0"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors()
            self.end_headers()
            if not head_only:
                self.wfile.write(body)
            return

        if not path.startswith("/video/"):
            sys.stderr.write(f"  [404 NOT_VIDEO_PATH] path={self.path}\n")
            self.send_error(404, "Not found")
            return

        video_id = unquote(path[len("/video/"):])
        target = _resolve(video_id)
        if target is None:
            # Not downloaded locally — tell the editor so it can fall back to A.
            sys.stderr.write(f"  [404 MISSING_VIDEO] raw_path={self.path} video_id='{video_id}'\n")
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


class DualStackHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that listens on IPv6 (::1) and IPv4 (127.0.0.1) simultaneously,
    preventing ERR_CONNECTION_REFUSED when Chrome resolves localhost to ::1."""
    address_family = socket.AF_INET6

    def server_bind(self) -> None:
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except (AttributeError, OSError):
            pass
        super().server_bind()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video_root",
        nargs="?",
        default=str(Path(__file__).resolve().parent),
        help="Folder holding downloaded <video_id>.mp4 files "
        "(default: the folder this script sits in)",
    )
    parser.add_argument("--port", type=int, default=8090, help="Port to listen on (default 8090)")
    parser.add_argument(
        "--host",
        default="",
        help="Bind address (default empty string = dual-stack IPv4/IPv6)",
    )
    args = parser.parse_args()

    global VIDEO_ROOT
    VIDEO_ROOT = Path(args.video_root).expanduser().resolve()
    if not VIDEO_ROOT.is_dir():
        sys.exit(f"Video root does not exist: {VIDEO_ROOT}")

    count = build_video_index(VIDEO_ROOT)
    try:
        server = DualStackHTTPServer((args.host, args.port), VideoHandler)
    except Exception:
        server = ThreadingHTTPServer((args.host or "127.0.0.1", args.port), VideoHandler)

    display_host = args.host if args.host else "127.0.0.1 / localhost"
    print(f"Serving {count} local videos from {VIDEO_ROOT}")
    print(f"Listening on http://{display_host}:{args.port}  (Ctrl+C to stop)")
    print("In the editor, turn on 'Local videos' to use this server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
