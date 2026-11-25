import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from worker import replay


START = {
    "type": "start", "schema_version": 1, "trace_id": "test",
    "model_id": "test-model", "model_revision": "m1", "layer": 0,
    "probe_id": "test-probe", "probe_revision": "p1",
}
TOKEN = {"type": "token", "index": 0, "text": "  café\n🩺", "risk": 0.5}
END = {"type": "end", "token_count": 1}
CLI = [sys.executable, "-u", str(Path(replay.__file__).resolve())]


def json_line(record):
    return (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")


class TraceTests(unittest.TestCase):
    def test_round_trip_preserves_unicode_whitespace_and_optional_features(self):
        records = [
            START, TOKEN,
            {**TOKEN, "index": 1, "text": "", "risk": 0, "sae_feature_ids": []},
            {**TOKEN, "index": 2, "text": " ", "risk": 1, "sae_feature_ids": [0, 14]},
            {**END, "token_count": 3},
        ]
        target = io.BytesIO()
        replay.write_trace(iter(records), target)
        lines = target.getvalue().split(b"\n")
        self.assertEqual(len(lines), len(records) + 1)
        self.assertEqual(lines[-1], b"")
        self.assertIn("café".encode(), lines[1])
        source = io.BytesIO(b"\r\n".join(lines[:-1]))  # Also accept EOF without final LF.
        self.assertEqual(list(replay.read_trace(source)), records)

    def test_new_tokens_flush_before_the_next_token_is_requested(self):
        class Sink(io.BytesIO):
            def flush(self):
                flushed.append(json.loads(self.getvalue().splitlines()[-1]))
                super().flush()

        flushed = []

        def scorer():
            yield START
            self.assertEqual(flushed, [START])
            yield TOKEN
            self.assertEqual(flushed, [START, TOKEN])
            yield END
            self.assertEqual(flushed, [START, TOKEN])

        replay.write_trace(scorer(), Sink())
        self.assertEqual(flushed, [START, TOKEN, END])

    def test_empty_trace_may_complete_with_zero_tokens(self):
        records = [START, {**END, "token_count": 0}]
        self.assertEqual(list(replay.validate_trace(records)), records)

    def test_identifiers_use_unicode_white_space(self):
        for value in ["\x1c", "\x1d", "\x1e", "\x1f", "\ufeff"]:
            with self.subTest(value=value):
                records = [{**START, "trace_id": value}, {**END, "token_count": 0}]
                self.assertEqual(list(replay.validate_trace(records)), records)
        with self.assertRaisesRegex(replay.TraceError, "trace_id"):
            list(replay.validate_trace([{**START, "trace_id": "\x85\u2000\u202f"}]))

    def test_last_token_count_remains_exact_in_the_browser(self):
        last_index = replay.MAX_SAFE_INTEGER - 1
        replay._validate_token({**TOKEN, "index": last_index}, last_index)
        with self.assertRaisesRegex(ValueError, "token count exceeds"):
            replay._validate_token({**TOKEN, "index": last_index + 1}, last_index + 1)

    def test_invalid_framing_has_line_number(self):
        for invalid in (b"[]", b"{", b"", b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":"\xff"}'):
            with self.subTest(invalid=invalid):
                source = io.BytesIO(json_line(START) + invalid + b"\n")
                records = replay.read_trace(source)
                self.assertEqual(next(records), START)
                with self.assertRaisesRegex(replay.TraceError, "line 2:"):
                    next(records)

    def test_invalid_sequences_fail_before_emitting_an_end(self):
        for records, line in [
            ([], 1), ([TOKEN], 1), ([START], 2), ([START, TOKEN], 3),
            ([START, START], 2), ([START, END], 2), ([START, TOKEN, TOKEN], 3),
            ([START, TOKEN, END, TOKEN], 4), ([START, TOKEN, END, END], 4),
        ]:
            with self.subTest(records=records):
                target = io.BytesIO()
                with self.assertRaisesRegex(replay.TraceError, f"line {line}:"):
                    replay.write_trace(records, target)
                emitted = [json.loads(raw)["type"] for raw in target.getvalue().splitlines()]
                self.assertNotIn("end", emitted)

    def test_field_validation(self):
        cases = [
            (0, {"schema_version": 2}), (0, {"schema_version": True}),
            (0, {"schema_version": 1.0}), (0, {"trace_id": "  "}),
            (0, {"model_revision": None}), (0, {"probe_revision": ""}),
            (0, {"layer": -1}), (0, {"layer": True}), (0, {"layer": 2**53}),
            (0, {"extra": 0}),
            (1, {"index": 1}), (1, {"index": 0.0}), (1, {"index": False}),
            (1, {"text": None}), (1, {"text": "\ud800"}), (1, {"risk": True}),
            (1, {"risk": 1.1}), (1, {"risk": -0.001}), (1, {"risk": "0.5"}),
            (1, {"risk": float("nan")}), (1, {"risk": float("inf")}),
            (1, {"sae_feature_ids": None}), (1, {"sae_feature_ids": {}}),
            (1, {"sae_feature_ids": [True]}), (1, {"sae_feature_ids": [2**53]}),
            (1, {"sae_feature_ids": [-1]}), (1, {"extra": 0}),
            (2, {"token_count": True}), (2, {"token_count": 0}), (2, {"extra": 0}),
        ]
        for position, changes in cases:
            with self.subTest(position=position, changes=changes):
                records = [START, TOKEN, END]
                records[position] = {**records[position], **changes}
                with self.assertRaisesRegex(replay.TraceError, f"line {position + 1}:"):
                    list(replay.validate_trace(records))
        for position, record in enumerate((START, TOKEN, END)):
            for key in record:
                with self.subTest(position=position, missing_key=key):
                    records = [START, TOKEN, END]
                    records[position] = {k: v for k, v in record.items() if k != key}
                    with self.assertRaisesRegex(replay.TraceError, f"line {position + 1}:"):
                        list(replay.validate_trace(records))

    def test_cli_reads_stdin_and_reports_truncation_on_stderr(self):
        data = b"".join(map(json_line, (START, TOKEN, END)))
        valid = subprocess.run([*CLI, "-"], input=data, capture_output=True, timeout=10)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertEqual(valid.stderr, b"")
        self.assertEqual(list(replay.read_trace(io.BytesIO(valid.stdout))), [START, TOKEN, END])
        invalid = subprocess.run([*CLI, "-"], input=json_line(START), capture_output=True, timeout=10)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn(b"expected end", invalid.stderr)
        self.assertEqual(json.loads(invalid.stdout), START)

    def test_cli_exits_cleanly_if_policy_closes_pipe(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "input.jsonl"
            with trace.open("wb") as source:
                source.write(json_line(START))
                for i in range(128):
                    source.write(json_line({**TOKEN, "index": i, "text": "x" * 8192}))
                source.write(json_line({**END, "token_count": 128}))
            with subprocess.Popen([*CLI, str(trace)], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
                try:
                    self.assertEqual(json.loads(proc.stdout.readline()), START)
                    proc.stdout.close()
                    self.assertEqual(proc.wait(timeout=10), 0)
                    self.assertEqual(proc.stderr.read(), b"")
                finally:
                    if proc.poll() is None:
                        proc.kill()


if __name__ == "__main__":
    unittest.main()
