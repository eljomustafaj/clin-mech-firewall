import unittest
from pathlib import Path

import numpy as np

from eval.evaluate_source import SOURCE, TEXT, audit_source, cue_score, load_source_development, source_metrics, source_predictions
from eval.evaluate_temporal import audit, load_development


class SourceEvaluationTests(unittest.TestCase):
    def test_line_prefix_whole_word_and_conflicting_cues(self):
        cases = [
            ("Archived tab: blue\nToday: red", "blue", (1.0, "prior-cue")),
            ("Last note: blue\nThis encounter: red", "red", (0.0, "current-cue")),
            ("Prior tab confirmed: red", "red", (0.7, "conflicting-cues")),
            ("PreviouslyClosed: red (prior)", "red", (0.7, "no-cues")),
            ("Earlier: blue\nOther: red, previous", "red", (0.7, "no-cues")),
        ]
        for prompt, target, expected in cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(cue_score(prompt, prompt.index(target), .7), expected)
        self.assertEqual(cue_score("Historical: red", None, .25), (.25, "unmatched"))

    def test_matched_and_unmatched_with_different_thresholds(self):
        prompt = "Current: blue.\nPrior visit: red.\n\nQuestion: red?"
        rows = [dict(id=str(i), prompt=prompt) for i in range(2)]
        raw = []
        for i in range(2):
            raw.append(dict(id=str(i), pair_id="p", group_id="g", layout_id="l", field_id="f",
                            chart_order="current_first", label=1, alternative_location=("prior", "absent")[i],
                            flags={"frozen-v1": 0, "recalibrated-v1": 0},
                            scores={"frozen-v1": .6, "recalibrated-v1": .6}))
        source = {"weights": [2.0], "bias": 0, "hidden_size": 1, "threshold": .55}
        probes = {"frozen-v1": {"threshold": .9}, "recalibrated-v1": {"threshold": .99}}
        args = (rows, raw, [" red", " yellow"], np.array([0]), np.array([[1.0]]),
                np.full(2, .6), source, probes, "recalibrated-v1")
        pred = source_predictions(*args)
        self.assertEqual((pred[0]["flags"][SOURCE], pred[0]["flags"][TEXT]), (1, 1))
        self.assertEqual((pred[1]["flags"][SOURCE], pred[1]["flags"][TEXT]), (1, 0))
        self.assertIsNone(pred[1]["match"])
        self.assertEqual(pred[0]["match"]["start"], prompt.index("red"))
        self.assertAlmostEqual(pred[0]["scores"][SOURCE], 1 / (1 + np.exp(-2)))
        thresholds = {k: v["threshold"] for k, v in probes.items()} | {SOURCE: .55, TEXT: .9}
        metrics = source_metrics(pred, thresholds, "recalibrated-v1")
        self.assertEqual(metrics["coverage"], {"matched": 1, "unmatched": 1})
        self.assertEqual(metrics["paired"]["frozen-v1"][SOURCE]["prior_catch"]["newly_flagged"], 1)
        self.assertEqual(metrics["overall"][TEXT]["supported_false_alarm"]["rate"], None)


class PublishedTemporalSourceTests(unittest.TestCase):
    def test_reproduce_two_frozen_evaluations_without_model(self):
        base = Path(__file__).resolve().parents[2] / "artifacts/temporal-v2"
        raw_sha = "6b7dc3b52e77721705bd5fe4a9d7e2c6321783c2ce450f84605b9f74ec01b6ae"
        source_sha = "8d0f4e8d359affa07be74cb3a2e8c564296c21f654861068530833f9d436358c"
        sel, probes, rows = load_development(base / "dev", raw_sha)
        source = load_source_development(base / "source-dev", source_sha, base / "dev", raw_sha, sel, probes, rows)
        audit(base / "test", base / "dev", raw_sha, sel, probes, rows)
        audit_source(base / "source-test", base / "test", sel, probes, source, source_sha, raw_sha)
