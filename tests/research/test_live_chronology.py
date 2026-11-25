import json
from pathlib import Path
import tempfile
import unittest

from eval.evaluate_source import cue_score as offline_cue
from mech_interp.chronology import generate_chronology_tokens
from mech_interp.source_probe import locate_source


BASELINE = Path(__file__).resolve().parents[2] / "artifacts/baseline/probe.json"


class RecordedModel:
    """Expose archived raw-v1 events through the model's generator interface."""

    def __init__(self, events):
        self.events = events
        self.n_consumed = 0

    def generate_tokens(self, prompt, probe, max_new_tokens):
        for i, (text, risk) in enumerate(self.events[:max_new_tokens]):
            self.n_consumed += 1
            yield {"position": i, "token_id": i + 1, "text": text, "risk_score": risk}


class LiveChronologyTests(unittest.TestCase):
    def test_published_prefixes_reproduce_the_recorded_chronology_decisions(self):
        archive = BASELINE.parents[1] / "temporal-v2/generation/generations.jsonl"
        threshold = json.loads(BASELINE.read_bytes())["threshold"]
        for line in archive.read_bytes().splitlines():
            row = json.loads(line)
            with self.subTest(id=row["source_row"]):
                saved = row["tokens"]
                raw = RecordedModel([(t["text"], t["original_risk"]) for t in saved])
                events = list(generate_chronology_tokens(raw, row["prompt"], BASELINE, len(saved)))
                for event, token in zip(events, saved, strict=True):
                    self.assertEqual((event["risk_score"], event["cue"], event["match"]),
                                     (token["cue_risk"], token["cue_reason"], token["source_match"]))
                    self.assertEqual(event["flagged"], token["cue_risk"] >= threshold)
                stop = next((e["position"] for e in events if e["flagged"]), None)
                self.assertEqual(row["policy"]["chronology-words"]["first_halt"], stop)
                self.assertEqual(row["policy"]["chronology-words"]["released"],
                                 "".join(e["text"] for e in events[:stop]))

    def test_offline_and_streaming_agree_without_lookahead(self):
        prompt = "Older record: a red door.\nCurrent record: a blue door.\n\nWhich door?"
        model = RecordedModel([(" a", 0.31), (" red", 0.02), (" door", 0.12)])
        stream = generate_chronology_tokens(model, prompt, BASELINE, 3)
        prefix = ""
        for position, basis in enumerate(("unmatched", "prior-cue", "prior-cue")):
            event = next(stream)
            self.assertEqual(model.n_consumed, position + 1)
            prefix += event["text"]
            match = locate_source(prompt, prefix)
            risk, reason = offline_cue(prompt, match.start if match else None, model.events[position][1])
            self.assertEqual((event["risk_score"], event["cue"]), (risk, reason))
            self.assertEqual(reason, basis)
            self.assertEqual(event["flagged"], position > 0)
        self.assertEqual(prompt[event["match"]["start"]:event["match"]["end"]], "a red door")
        with self.assertRaises(StopIteration):
            next(stream)

    def test_original_inclusive_threshold_applies_even_on_conflicting_cues(self):
        threshold = json.loads(BASELINE.read_bytes())["threshold"]
        prompt = "Prior and current record: rose.\nToday: violet.\n\nWhich?"
        for text, expected in [(" rose", ("conflicting-cues", True)), (" violet", ("current-cue", False))]:
            event = next(generate_chronology_tokens(RecordedModel([(text, threshold)]), prompt, BASELINE, 1))
            self.assertEqual((event["cue"], event["flagged"]), expected)
            self.assertEqual(event["fallback_risk"], threshold)

    def test_fallback_digest_checked_before_reading_any_tokens(self):
        model = RecordedModel([(" new", 0.1)])
        with tempfile.TemporaryDirectory() as temp:
            altered = Path(temp) / "altered.json"
            altered.write_bytes(BASELINE.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "published v1"):
                next(generate_chronology_tokens(model, "Prior: new", altered, 1))
            self.assertEqual(model.n_consumed, 0)
