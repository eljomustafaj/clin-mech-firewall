"""Audit free-generation enrollment and policy decisions without loading the language model."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "temporal-v2"


def read_json(path):
    return json.loads(path.read_bytes())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


class PublishedTemporalGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = ROOT / "generation"
        cls.rows = read_jsonl(cls.folder / "generations.jsonl")
        cls.metrics = read_json(cls.folder / "metrics.json")

    def test_file_identity_and_stratified_enrollment(self):
        for name, claimed in self.metrics["sha256"].items():
            self.assertEqual(hashlib.sha256((self.folder / name).read_bytes()).hexdigest(), claimed)
        dataset = ROOT / "test" / "dataset.jsonl"
        self.assertEqual(self.metrics["test_dataset_sha256"], hashlib.sha256(dataset.read_bytes()).hexdigest())
        strata = defaultdict(dict)
        for row in read_jsonl(dataset):
            if row["label"] == 0 and row["alternative_location"] == "prior":
                strata[row["layout_id"], row["chart_order"]].setdefault(row["prompt"], row)
        expected = []
        for key in sorted(strata):
            ranked = sorted(strata[key], key=lambda prompt: hashlib.sha256(
                b"prior-visit-generation-v2\0" + prompt.encode("utf-8")).digest())
            expected.extend(strata[key][p] for p in ranked[:4])
        self.assertEqual(len(strata), 8)
        self.assertEqual(len(expected), self.metrics["n"])
        for actual, source in zip(self.rows, expected, strict=True):
            self.assertEqual(actual["source_row"], source["id"])
            for key in ("group_id", "layout_id", "chart_order", "prompt", "current_value", "prior_value"):
                self.assertEqual(actual[key], source[key])

    def test_every_reported_decision_and_count(self):
        selection = read_json(ROOT / "dev" / "selection.json")
        selected = selection["models"][selection["selected"]]
        baseline = selection["models"]["frozen-v1"]["threshold"]
        source = read_json(ROOT / "source-dev" / "source-probe.json")["threshold"]
        methods = {"original": ("original_risk", baseline), "selected": ("selected_risk", selected["threshold"]),
                   "source": ("source_risk", source), "chronology-words": ("cue_risk", baseline)}
        tallies = defaultdict(lambda: defaultdict(int))
        for row in self.rows:
            tokens = row["tokens"]
            self.assertTrue(1 <= len(tokens) <= self.metrics["max_tokens"])
            self.assertEqual([t["position"] for t in tokens], list(range(len(tokens))))
            self.assertEqual("".join(t["text"] for t in tokens), row["generation"])
            self.assertEqual(set(row["policy"]), methods.keys())
            group = row["opening"]
            tallies[group]["prompts"] += 1
            for name, (risk_key, threshold) in methods.items():
                flags = [token[risk_key] >= threshold for token in tokens]
                halt = flags.index(True) if any(flags) else None
                expected = row["policy"][name]
                self.assertEqual(expected["first_halt"], halt)
                self.assertEqual(expected["released"], "".join(t["text"] for t in tokens[:halt]))
                end = row["value_end_position"]
                self.assertEqual(end is None, group == "unclassified")
                stopped = (halt is not None and halt <= end) if end is not None else None
                self.assertEqual(expected["halt_by_value_end"], stopped)
                tallies[group][name, "halt_anywhere"] += int(halt is not None)
                tallies[group][name, "halt_by_value_end"] += int(bool(stopped))
        for group, claim in self.metrics["opening_counts"].items():
            self.assertEqual(claim["prompts"], tallies[group]["prompts"])
            for name in methods:
                for outcome, value in claim[name].items():
                    if group == "unclassified" and outcome == "halt_by_value_end":
                        self.assertIsNone(value)
                    else:
                        self.assertEqual(value, tallies[group][name, outcome])
