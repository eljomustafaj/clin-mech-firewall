"""Black-box checks against the compiled firewall and the production entry points."""

import http.client
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replay

FIXTURE = ROOT / "fixtures/synthetic.wire.jsonl"
BASELINE = ROOT / "artifacts/baseline/trace.wire.jsonl"


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


class ReplayIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proxy = (Path(os.environ["FIREWALL_PROXY"]).resolve()
                     if "FIREWALL_PROXY" in os.environ else replay.build_proxy())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def offline(self, trace, threshold, name="run"):
        output = self.directory / name
        process = subprocess.run(
            [sys.executable, str(ROOT / "scripts/replay.py"), str(trace),
             "--proxy", str(self.proxy), "--threshold", str(threshold), "--out", str(output)],
            capture_output=True, timeout=30,
        )
        return process, output

    def assert_replay(self, trace, threshold, telemetry, response):
        source = rows(trace)
        tokens = source[1:-1]
        trigger = next((i for i, token in enumerate(tokens) if token["risk"] >= threshold), None)
        allowed = tokens if trigger is None else tokens[:trigger]
        expected = tokens if trigger is None else tokens[:trigger + 1]
        self.assertEqual(response, "".join(t["text"] for t in allowed).encode("utf-8"))
        self.assertEqual(telemetry[0], dict(source[0], threshold=threshold))
        decisions = telemetry[1:-1]
        self.assertEqual(len(decisions), len(expected))
        for event, token in zip(decisions, expected):
            self.assertEqual(event, dict(token, type="decision", trace_id=source[0]["trace_id"],
                                         decision="HALT" if token["index"] == trigger else "ALLOW"))
        self.assertEqual(telemetry[-1], {
            "type": "end", "status": "complete" if trigger is None else "halted",
            "trace_id": source[0]["trace_id"], "token_count": len(expected),
        })

    def test_offline_boundaries(self):
        for i, threshold in enumerate([0, 0.75, 1]):
            with self.subTest(threshold=threshold):
                process, output = self.offline(FIXTURE, threshold, str(i))
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assert_replay(FIXTURE, threshold, rows(output / "telemetry.jsonl"),
                                   (output / "response.txt").read_bytes())

    def test_bad_trace_has_no_successful_artifacts(self):
        for i, suffix in enumerate([b"", b'{"type":"end","token_count":0}\nnot json\n']):
            with self.subTest(suffix=suffix):
                bad = self.directory / f"bad-{i}.jsonl"
                bad.write_bytes(FIXTURE.read_bytes().splitlines()[0] + b"\n" + suffix)
                process, output = self.offline(bad, 1)
                self.assertNotEqual(process.returncode, 0)
                self.assertFalse(output.exists())

    def test_worker_and_proxy_release_before_input_is_closed(self):
        q = queue.Queue()
        with replay.child([sys.executable, "-u", str(ROOT / "worker/replay.py"), "-"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as worker:
            out = self.directory / "response"
            with replay.child([str(self.proxy), "--threshold", "0.75", "--response", str(out)],
                              stdin=worker.stdout, stdout=subprocess.PIPE) as proxy:
                worker.stdout.close()
                def read_decisions():
                    for line in proxy.stdout:
                        q.put(json.loads(line))
                reader = threading.Thread(target=read_decisions, daemon=True)
                reader.start()
                prefix = FIXTURE.read_bytes().splitlines()[:2]
                worker.stdin.write(b"\n".join(prefix) + b"\n")
                worker.stdin.flush()
                self.assertEqual(q.get(timeout=10)["type"], "start")
                self.assertEqual(q.get(timeout=10)["decision"], "ALLOW")
                self.assertEqual(out.read_bytes(), b"The")
                # The writer remains open while the approved token is delivered.
                worker.stdin.write(b'{"type":"end","token_count":1}\n')
                worker.stdin.close()
                self.assertEqual(q.get(timeout=10)["status"], "complete")
                self.assertEqual(proxy.wait(timeout=10), 0)
                self.assertEqual(out.read_bytes(), b"The")
                reader.join(timeout=10)

    def serve(self, trace, threshold):
        store = replay.ReplayStore(self.directory)
        handler = replay.partial(replay.ReplayHandler, trace=trace, threshold=threshold,
                                 proxy=self.proxy, store=store)
        server = replay.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def http_replay(self, trace, threshold):
        base = self.serve(trace, threshold)
        with urlopen(Request(base + "/api/replays", method="POST"), timeout=15) as stream:
            self.assertEqual(stream.headers.get_content_type(), "application/x-ndjson")
            url = base + "/api/replays/" + stream.headers["X-Replay-Id"] + "/response"
            events = [json.loads(line) for line in stream]
        with urlopen(url, timeout=10) as response:
            self.assert_replay(trace, threshold, events, response.read())

    def test_http_response_is_separate_from_trusted_telemetry(self):
        self.http_replay(FIXTURE, 0.75)

    def test_failed_http_replay_cannot_publish_partial_response(self):
        bad = self.directory / "incomplete.jsonl"
        bad.write_bytes(b"\n".join(FIXTURE.read_bytes().splitlines()[:2]) + b"\n")
        base = self.serve(bad, 1)
        with urlopen(Request(base + "/api/replays", method="POST"), timeout=15) as stream:
            url = base + "/api/replays/" + stream.headers["X-Replay-Id"] + "/response"
            events = [json.loads(line) for line in stream]
        self.assertEqual(events[-1]["type"], "error")
        self.assertNotIn("end", [event["type"] for event in events])
        with self.assertRaises(HTTPError) as response:
            urlopen(url, timeout=10)
        self.assertEqual(response.exception.code, http.client.NOT_FOUND)

    @unittest.skipUnless(BASELINE.exists(), "The Task 1 trace is absent from this checkout")
    def test_real_baseline_first_halt(self):
        # Check the producer's archive/probe identity before using wire data as
        # the oracle for the independently observed HTTP response and decisions.
        regenerated = self.directory / "regenerated.jsonl"
        export = subprocess.run(
            [sys.executable, "-m", "eval.export_wire", str(BASELINE.with_name("trace.json")),
             "--probe", str(BASELINE.with_name("probe.json")), "--output", str(regenerated)],
            cwd=ROOT, capture_output=True, timeout=30,
        )
        self.assertEqual(export.returncode, 0, export.stderr)
        self.assertEqual(rows(regenerated), rows(BASELINE))
        tokens = rows(BASELINE)[1:-1]
        self.assertTrue(tokens, "the real trace must contain generated tokens")
        probe = json.loads(BASELINE.with_name("probe.json").read_bytes())
        threshold = float(os.environ.get("REPLAY_THRESHOLD", probe["threshold"]))
        self.assertTrue(any(token["risk"] >= threshold for token in tokens),
                        "the baseline acceptance run must exercise HALT")
        self.http_replay(BASELINE, threshold)


if __name__ == "__main__":
    unittest.main()
