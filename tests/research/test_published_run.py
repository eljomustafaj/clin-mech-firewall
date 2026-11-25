"""Audit the published pilot without loading a language model or fitting a probe."""

import hashlib
import math
import unittest
from pathlib import Path

import numpy as np

from eval.validate_trace import parse_json, validate_probe, validate_trace
from mech_interp.data import build_dataset


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "baseline"


def read_artifact(name):
    return parse_json((ARTIFACTS / name).read_bytes())


def read_rows(name):
    return [parse_json(line) for line in (ARTIFACTS / name).read_bytes().splitlines()]


class PublishedRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = read_rows("dataset.jsonl")
        cls.predictions = read_rows("predictions.jsonl")
        cls.probe = read_artifact("probe.json")
        cls.metrics = read_artifact("metrics.json")
        cls.manifest = read_artifact("manifest.json")

    def test_manifest_dataset_and_trace_are_from_same_run(self):
        files = self.manifest["sha256"]
        for name in ("dataset.jsonl", "predictions.jsonl", "activations.npz", "probe.json",
                     "metrics.json", "trace.json", "trace.wire.jsonl"):
            with self.subTest(name=name):
                checksum = hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest()
                self.assertEqual(checksum, files[name])
        self.assertEqual(self.dataset, build_dataset())
        trace = read_artifact("trace.json")
        validate_trace(trace)
        validate_probe(trace, ARTIFACTS / "probe.json")
        self.assertEqual(self.metrics["probe_id"], trace["probe_id"])
        source = next(r for r in self.dataset if r["id"] == trace["source_row"])
        self.assertEqual((source["split"], source["prompt"]), ("test", trace["prompt"]))
        self.assertEqual(self.manifest["trace_source"], trace["source_row"])

    def test_recompute_every_prediction_from_saved_activations(self):
        with np.load(ARTIFACTS / "activations.npz", allow_pickle=False) as cache:
            self.assertEqual(cache["row_ids"].tolist(), [row["id"] for row in self.dataset])
            vectors = cache["activations"]
        self.assertEqual(vectors.shape, (len(self.dataset), self.probe["hidden_size"]))
        self.assertTrue(np.isfinite(vectors).all())
        self.assertEqual(self.metrics["threshold"], self.probe["threshold"])
        weights = self.probe["weights"]
        for row, vector, pred in zip(self.dataset, vectors, self.predictions, strict=True):
            with self.subTest(row=row["id"]):
                for key in ("id", "label", "split", "group_id", "pair_id"):
                    self.assertEqual(pred[key], row[key])
                logit = math.fsum(float(a) * w for a, w in zip(vector, weights, strict=True))
                logit += self.probe["bias"]
                # Symmetric logistic form avoids exponentiating large logits.
                risk = 0.5 + 0.5 * math.tanh(logit / 2)
                self.assertAlmostEqual(risk, pred["risk_score"], places=10)
                self.assertEqual(pred["predicted_label"], int(pred["risk_score"] >= self.probe["threshold"]))
                target_absent = row["target_text"].casefold() not in row["prompt"].casefold()
                self.assertEqual(int(target_absent), pred["lexical_risk"])

    def test_recompute_benchmark_metrics_without_training_code(self):
        for split, score_key, threshold, report in (
            ("validation", "risk_score", self.probe["threshold"], "validation"),
            ("test", "risk_score", self.probe["threshold"], "test"),
            ("test", "lexical_risk", 0.5, "test_lexical_baseline"),
        ):
            with self.subTest(split=split, report=report):
                records = [r for r in self.predictions if r["split"] == split]
                truth = [r["label"] for r in records]
                flags = [r[score_key] >= threshold for r in records]
                tp = sum(y == 1 and flag for y, flag in zip(truth, flags))
                fp = sum(y == 0 and flag for y, flag in zip(truth, flags))
                fn, tn = truth.count(1) - tp, truth.count(0) - fp
                pos = [r[score_key] for r in records if r["label"] == 1]
                neg = [r[score_key] for r in records if r["label"] == 0]
                pair_wins = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg)
                expected = {
                    "n": len(records), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "auroc": pair_wins / (len(pos) * len(neg)), "f1": 2 * tp / (2 * tp + fp + fn),
                    "accuracy": (tp + tn) / len(records), "false_positive_rate": fp / (fp + tn),
                    "recall": tp / (tp + fn), "precision": tp / (tp + fp) if tp + fp else 0,
                }
                for name, value in expected.items():
                    self.assertAlmostEqual(self.metrics[report][name], value, places=12)
                split_counts = self.metrics["splits"][split]
                self.assertEqual(split_counts["groups"], len({r["group_id"] for r in records}))
                self.assertEqual(split_counts["pairs"], len({r["pair_id"] for r in records}))
                self.assertEqual(split_counts["rows"], len(records))

    def test_threshold_is_the_documented_validation_choice(self):
        records = [r for r in self.predictions if r["split"] == "validation"]
        positives = sum(r["label"] for r in records)
        candidates = {1.0, *(r["risk_score"] for r in records)}
        rankings = []
        for candidate in candidates:
            flagged = [r for r in records if r["risk_score"] >= candidate]
            true_positives = sum(r["label"] for r in flagged)
            f1 = 2 * true_positives / (len(flagged) + positives)
            rankings.append((f1, true_positives - len(flagged), candidate))
        self.assertEqual(max(rankings)[2], self.probe["threshold"])


if __name__ == "__main__":
    unittest.main()
