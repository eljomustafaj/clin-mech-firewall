import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from eval.validate_trace import parse_json, validate_probe, validate_trace


class HandoffContractTests(unittest.TestCase):
    """Small contract fixtures, never model output or reported benchmark data."""

    def setUp(self):
        self.probe = {
            "schema_version": "linear-probe.v1",
            "positive_label": "unsupported",
            "model_id": "test-model",
            "model_revision": "test-revision",
            "layer": 1,
            "hidden_size": 2,
            "weights": [0.2, -1],
            "bias": 0,
            "threshold": 0.5,
        }
        self.trace = {
            "schema_version": "research-trace.v1",
            "trace_id": "test-trace",
            "model_id": "test-model",
            "model_revision": "test-revision",
            "layer": 1,
            "probe_id": "sha256:" + "0" * 64,
            "prompt": "test prompt",
            "tokens": [
                {"position": 0, "token_id": 12, "text": "", "risk_score": 0.0},
                {"position": 1, "token_id": 13, "text": " β", "risk_score": 1.0},
            ],
        }

    def write_probe(self, path):
        raw = json.dumps(self.probe, allow_nan=False).encode()
        path.write_bytes(raw)
        self.trace["probe_id"] = "sha256:" + hashlib.sha256(raw).hexdigest()

    def test_valid_fragment_stream_empty_text_and_extension(self):
        self.trace["tokens"][0]["extra"] = 4
        validate_trace(self.trace)

    def test_required_schema_fields_are_checked(self):
        schema = json.loads((Path(__file__).resolve().parents[2] / "contracts" /
                             "research-trace.v1.schema.json").read_text(encoding="utf-8"))
        for key in schema["required"]:
            with self.subTest(trace_key=key):
                data = dict(self.trace)
                del data[key]
                with self.assertRaises(ValueError):
                    validate_trace(data)
        for key in schema["properties"]["tokens"]["items"]["required"]:
            with self.subTest(token_key=key):
                data = copy.deepcopy(self.trace)
                del data["tokens"][0][key]
                with self.assertRaises(ValueError):
                    validate_trace(data)

    def test_rejects_invalid_scores_ids_and_order(self):
        for key, value in (
            ("risk_score", True), ("risk_score", float("nan")), ("risk_score", float("inf")),
            ("risk_score", 1.01), ("risk_score", -0.4), ("risk_score", "0.5"),
            ("token_id", -1), ("token_id", False), ("position", 3), ("position", False),
            ("text", None),
        ):
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.trace)
                data["tokens"][0][key] = value
                with self.assertRaises(ValueError):
                    validate_trace(data)
        self.trace["tokens"][1]["position"] = 0
        with self.assertRaisesRegex(ValueError, "position"):
            validate_trace(self.trace)

    def test_json_parser_rejects_nonstandard_and_ambiguous_input(self):
        for raw in ('{"v": NaN}', '{"v": Infinity}', '{"v": 1, "v": 2}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_json(raw)

    def test_checks_artifact_identity_and_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            self.write_probe(path)
            validate_probe(self.trace, path)
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_probe(self.trace, path)
            for key, value in (
                ("model_revision", "other"), ("layer", True), ("positive_label", "supported"),
                ("hidden_size", 3), ("weights", [1, False]), ("threshold", 2),
            ):
                with self.subTest(key=key, value=value):
                    original = self.probe[key]
                    self.probe[key] = value
                    self.write_probe(path)
                    with self.assertRaises(ValueError):
                        validate_probe(self.trace, path)
                    self.probe[key] = original


if __name__ == "__main__":
    unittest.main()
