"""Run the local Task 2 replay using separate telemetry and response channels."""

import argparse
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from threading import BoundedSemaphore, Lock
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def child(args, **kwargs):
    """Reap a child even when a browser disconnects halfway through a stream."""
    with subprocess.Popen(args, **kwargs) as process:
        try:
            yield process
        finally:
            if process.poll() is None:
                process.kill()


def pipeline(trace, threshold, proxy, response):
    """Yield decisions live. Hold terminal success until both children exit."""
    with ExitStack() as stack:
        worker = stack.enter_context(child(
            [sys.executable, "-u", str(ROOT / "worker/replay.py"), str(trace)],
            stdout=subprocess.PIPE,
        ))
        firewall = stack.enter_context(child(
            [str(proxy), "--threshold", str(threshold), "--response", str(response)],
            stdin=worker.stdout, stdout=subprocess.PIPE,
        ))
        worker.stdout.close()
        terminal = None
        for line in firewall.stdout:
            if terminal is not None:
                raise ValueError("The proxy emitted data after the end record.")
            if json.loads(line)["type"] == "end":
                terminal = line
            else:
                yield line
        if firewall.wait() or terminal is None:
            raise ValueError("The firewall rejected the stream. See the terminal for details.")
        # A HALT owns only the observed prefix. The worker may already have
        # noticed an invalid suffix or may still be writing into the closed pipe.
        if json.loads(terminal)["status"] != "halted" and worker.wait():
            raise ValueError("The worker rejected the trace. See the terminal for details.")
        yield terminal


def build_proxy():
    print("Building the Rust proxy...", file=sys.stderr)
    build = subprocess.run(
        ["cargo", "build", "--locked", "--manifest-path", str(ROOT / "proxy/Cargo.toml"),
         "--message-format=json-render-diagnostics"],
        check=True, stdout=subprocess.PIPE, text=True, encoding="utf-8",
    )
    for line in build.stdout.splitlines():
        artifact = json.loads(line)
        if artifact.get("target", {}).get("name") == "firewall-proxy" and artifact.get("executable"):
            return Path(artifact["executable"])
    raise ValueError("Cargo did not report the firewall-proxy executable.")


class ReplayStore:
    """Publish only completed responses. Keep a bounded history for open tabs."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.completed = OrderedDict()
        self.lock = Lock()
        self.slots = BoundedSemaphore(4)

    def publish(self, run_id, response):
        with self.lock:
            self.completed[run_id] = response
            if len(self.completed) > 32:
                _, oldest = self.completed.popitem(last=False)
                oldest.unlink(missing_ok=True)

    def get(self, run_id):
        with self.lock:
            path = self.completed.get(run_id)
            return path.read_bytes() if path else None


class ReplayHandler(BaseHTTPRequestHandler):
    server_version = "ClinicalReplay/1"

    def __init__(self, *args, trace, threshold, proxy, store, **kwargs):
        self.trace = trace
        self.threshold = threshold
        self.proxy = proxy
        self.store = store
        super().__init__(*args, **kwargs)

    def valid_origin(self):
        port = self.server.server_port
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        return (host in {f"localhost:{port}", f"127.0.0.1:{port}"} and
                (origin is None or origin == f"http://{host}"))

    def headers_for(self, status, content_type, length=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if length is not None:
            self.send_header("Content-Length", str(length))

    def send_bytes(self, data, content_type):
        self.headers_for(200, content_type, len(data))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.valid_origin():
            return self.send_error(403)
        if self.path == "/api/config":
            config = {"trace_name": self.trace.name, "threshold": self.threshold}
            return self.send_bytes(json.dumps(config).encode(), "application/json")
        match = re.fullmatch(r"/api/replays/([0-9a-f]{32})/response", self.path)
        if match:
            data = self.store.get(match.group(1))
            if data is None:
                return self.send_error(404, "Response unavailable")
            return self.send_bytes(data, "text/plain; charset=utf-8")
        assets = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
            "/main.js": ("main.js", "text/javascript; charset=utf-8"),
            "/view.js": ("view.js", "text/javascript; charset=utf-8"),
            "/telemetry.js": ("telemetry.js", "text/javascript; charset=utf-8"),
        }
        if self.path not in assets:
            return self.send_error(404)
        name, content_type = assets[self.path]
        return self.send_bytes((ROOT / "web/dist" / name).read_bytes(), content_type)

    def do_POST(self):
        if not self.valid_origin():
            return self.send_error(403)
        if self.path != "/api/replays":
            return self.send_error(404)
        if self.headers.get("Content-Length", "0") != "0" or "Transfer-Encoding" in self.headers:
            return self.send_error(400, "Replay takes an empty request body")
        if not self.store.slots.acquire(blocking=False):
            return self.send_error(503, "Four replays are already running")
        run_id = uuid4().hex
        response = self.store.directory / run_id
        published = False
        try:
            self.headers_for(200, "application/x-ndjson; charset=utf-8")
            self.send_header("X-Replay-Id", run_id)
            self.end_headers()
            events = pipeline(self.trace, self.threshold, self.proxy, response)
            try:
                for line in events:
                    if json.loads(line)["type"] == "end":
                        self.store.publish(run_id, response)
                        published = True
                    self.wfile.write(line)
                    self.wfile.flush()
            finally:
                events.close()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except (OSError, ValueError, KeyError) as exc:
            self.log_error("Replay failed: %s", exc)
            error = {"type": "error", "message": "Replay failed. Check the server terminal."}
            self.wfile.write(json.dumps(error).encode() + b"\n")
        finally:
            if not published:
                response.unlink(missing_ok=True)
            self.store.slots.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="scored trace in the shared NDJSON format")
    parser.add_argument("--threshold", type=float, required=True, help="halt at or above this score")
    parser.add_argument("--proxy", type=Path, help="use a prebuilt firewall-proxy executable")
    parser.add_argument("--port", type=int, default=8000, help="loopback HTTP port (0 picks a free port)")
    parser.add_argument("--out", type=Path, help="write response.txt and telemetry.jsonl to a new directory")
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("threshold must be a finite number in [0, 1]")
    if not args.trace.is_file():
        parser.error("trace must be an existing file")
    if not 0 <= args.port <= 65535:
        parser.error("port must be in [0, 65535]")
    try:
        proxy = args.proxy.resolve(strict=True) if args.proxy else build_proxy()
        if args.out:
            args.out.mkdir(parents=True, exist_ok=False)
            try:
                with (args.out / "telemetry.jsonl").open("xb") as output:
                    for line in pipeline(args.trace, args.threshold, proxy, args.out / "response.txt"):
                        output.write(line)
            except BaseException:
                for name in ("telemetry.jsonl", "response.txt"):
                    (args.out / name).unlink(missing_ok=True)
                args.out.rmdir()
                raise
            print(f"Wrote replay to {args.out}")
            return
        with tempfile.TemporaryDirectory(prefix="firewall-") as tmp:
            handler = partial(ReplayHandler, trace=args.trace, threshold=args.threshold,
                              proxy=proxy, store=ReplayStore(tmp))
            with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
                print(f"Replay console: http://127.0.0.1:{server.server_port}/", flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Replay failed: {exc}\n")


if __name__ == "__main__":
    main()
