"""Check that an archived trace and probe produce the agreed worker wire format."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from eval.export_wire import export_wire


class WireExportTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.directory = Path(workspace.name)
        self.probe = self.directory / "probe.json"
        self.archive = self.directory / "trace.json"
        self.output = self.directory / "nested" / "worker.jsonl"
        self.probe.write_bytes(json.dumps({
            "schema_version": "linear-probe.v1", "positive_label": "unsupported",
            "model_id": "test-model", "model_revision": "fixture-revision", "layer": 0,
            "hidden_size": 1, "weights": [1.0], "bias": 0, "threshold": 0.5,
        }).encode())
        self.digest = "sha256:" + hashlib.sha256(self.probe.read_bytes()).hexdigest()
        self.fragments = ["", " β", "\n結果", "🙂"]
        self.risks = [0.0, 0.125, 0.5, 1.0]
        self.trace = {
            "schema_version": "research-trace.v1", "trace_id": "test-trace",
            "model_id": "test-model", "model_revision": "fixture-revision", "layer": 0,
            "probe_id": self.digest, "prompt": "test prompt", "archive_only": 1,
            "tokens": [
                {"position": i, "token_id": 10 + i, "text": fragment, "risk_score": risk,
                 "archive_only": 2}
                for i, (fragment, risk) in enumerate(zip(self.fragments, self.risks))
            ],
        }
        self.archive.write_text(json.dumps(self.trace), encoding="utf-8")

    def test_cli_preserves_fragments_and_risks_in_wire_records(self):
        args = [sys.executable, "-m", "eval.export_wire", str(self.archive),
                "--probe", str(self.probe), "--output", str(self.output)]
        subprocess.run(args, cwd=Path(__file__).resolve().parents[2], check=True, capture_output=True)
        wire = self.output.read_bytes()
        self.assertIn("結果".encode(), wire)
        self.assertTrue(wire.endswith(b"\n"))
        self.assertNotIn(b"\r", wire)
        records = [json.loads(line) for line in wire.split(b"\n")[:-1]]
        self.assertEqual(records[0], {
            "type": "start", "schema_version": 1, "trace_id": "test-trace", "model_id": "test-model",
            "model_revision": "fixture-revision", "layer": 0, "probe_id": "research-linear-probe-v1",
            "probe_revision": self.digest,
        })
        self.assertEqual(records[1:-1], [
            {"type": "token", "index": i, "text": fragment, "risk": self.risks[i]}
            for i, fragment in enumerate(self.fragments)
        ])
        self.assertEqual(records[-1], {"type": "end", "token_count": len(self.fragments)})
        export_wire(self.archive, self.probe, self.output)
        self.assertEqual(self.output.read_bytes(), wire)

    def test_modified_probe_invalidates_export_and_preserves_previous_stream(self):
        export_wire(self.archive, self.probe, self.output)
        previous = self.output.read_bytes()
        self.probe.write_bytes(self.probe.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            export_wire(self.archive, self.probe, self.output)
        self.assertEqual(self.output.read_bytes(), previous)

    def test_invalid_index_and_unencodable_text_never_publish_a_partial_stream(self):
        for key, value in (("position", 2), ("text", "\ud800")):
            with self.subTest(key=key):
                trace = json.loads(json.dumps(self.trace))
                trace["tokens"][-1][key] = value
                self.archive.write_text(json.dumps(trace), encoding="utf-8")
                with self.assertRaises(ValueError):
                    export_wire(self.archive, self.probe, self.output)
                self.assertFalse(self.output.exists())
                self.assertFalse(list(self.output.parent.glob("*.tmp")))

    def test_cannot_overwrite_inputs(self):
        for path in (self.archive, self.probe):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "distinct"):
                export_wire(self.archive, self.probe, path)


if __name__ == "__main__":
    unittest.main()
