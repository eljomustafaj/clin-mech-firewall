import hashlib
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from worker.inference import write_inference


PROBE = {
    "schema_version": "linear-probe.v1", "positive_label": "unsupported",
    "model_id": "test-model", "model_revision": "v1", "layer": 0,
    "hidden_size": 1, "weights": [0.1], "bias": 0, "threshold": 0.5,
}


class InferenceTests(unittest.TestCase):
    def test_live_scores_flush_as_generated_and_identify_exact_probe_bytes(self):
        raw = (json.dumps(PROBE, indent=2) + "\r\n").encode("utf-8")
        flushed = []

        class Sink(io.BytesIO):
            def flush(self):
                flushed.append(json.loads(self.getvalue().splitlines()[-1]))
                super().flush()

        def generate(prompt, probe, limit):
            self.assertEqual((prompt, probe, limit), ("A prompt", PROBE, 2))
            self.assertEqual(flushed[0]["probe_revision"], "sha256:" + hashlib.sha256(raw).hexdigest())
            self.assertEqual(flushed[0]["trace_id"], "live-test")
            yield {"position": 0, "token_id": 3, "text": " first", "risk_score": 0.125}
            self.assertEqual(flushed[-1], {"type": "token", "index": 0, "text": " first", "risk": 0.125})
            yield {"position": 1, "token_id": 9, "text": "", "risk_score": 0.75}

        model = SimpleNamespace(model_id="test-model", revision="v1", layer=0, hidden_size=1,
                                generate_tokens=generate)
        path = Path("probe.json")
        with patch.object(Path, "read_bytes", autospec=True, return_value=raw) as read:
            write_inference(model, "A prompt", path, Sink(), trace_id="live-test", max_new_tokens=2)
            read.assert_called_once_with(path)
        self.assertEqual(flushed[-2], {"type": "token", "index": 1, "text": "", "risk": 0.75})
        self.assertEqual(flushed[-1], {"type": "end", "token_count": 2})

    def test_probe_mismatch_fails_before_generation_or_output(self):
        model = SimpleNamespace(model_id="test-model", revision="v1", layer=0, hidden_size=1,
                                generate_tokens=lambda *args: self.fail("generator should not start"))
        for changes, message in [({"model_revision": "other"}, "model_revision"),
                                 ({"hidden_size": 2, "weights": [0.1, 0.2]}, "hidden_size")]:
            with self.subTest(changes=changes):
                target = io.BytesIO()
                with patch.object(Path, "read_bytes", return_value=json.dumps({**PROBE, **changes}).encode()):
                    with self.assertRaisesRegex(ValueError, message):
                        write_inference(model, "x", Path("probe.json"), target,
                                        trace_id="live-test", max_new_tokens=1)
                self.assertEqual(target.getvalue(), b"")


if __name__ == "__main__":
    unittest.main()
