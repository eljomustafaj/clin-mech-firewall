import tempfile
import unittest
from pathlib import Path

from eval.export_chronology_replay import export
from eval.validate_trace import parse_json


class PublishedChronologyReplayTests(unittest.TestCase):
    def test_named_generation_recreates_wire_and_recorded_response(self):
        artifacts = Path(__file__).resolve().parents[2] / "artifacts"
        replay = artifacts / "temporal-v2/replay"
        source_row = "temporal-v1.test.annotated_extract.id_document.0-3.prior.prior_first.current-3.candidate-3"
        with tempfile.TemporaryDirectory() as work:
            folder = Path(work)
            first, _ = export(artifacts / "temporal-v2/generation", artifacts / "baseline/probe.json",
                              source_row, folder)
            self.assertEqual(first, 0)
            for name in ("policy.json", "trace.wire.jsonl"):
                self.assertEqual((folder / name).read_bytes(), (replay / name).read_bytes())
        events = [parse_json(line) for line in (replay / "observed/telemetry.jsonl").read_bytes().splitlines()]
        self.assertEqual([e["type"] for e in events], ["start", "decision", "end"])
        self.assertEqual((events[1]["index"], events[1]["text"], events[1]["risk"], events[1]["decision"]),
                         (0, " passport", 1.0, "HALT"))
        self.assertEqual((replay / "observed/response.txt").read_bytes(), b"")
