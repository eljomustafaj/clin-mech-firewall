"""Check chart labels, counterbalancing, held-out layouts, and serialization."""

import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from mech_interp.data import SOURCE, build_dataset


class ChartDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build_dataset()

    def test_counts_and_splits(self):
        self.assertEqual(Counter(row["split"] for row in self.rows),
                         {"train": 160, "validation": 40, "test": 40})
        self.assertEqual(len({row["id"] for row in self.rows}), len(self.rows))
        for key in ("group_id", "pair_id", "layout_id", "prompt"):
            with self.subTest(key=key):
                splits = defaultdict(set)
                for row in self.rows:
                    splits[row[key]].add(row["split"])
                self.assertTrue(all(len(partitions) == 1 for partitions in splits.values()))

    def test_pair_and_quartet_controls(self):
        pairs, quartets = defaultdict(list), defaultdict(list)
        for row in self.rows:
            pairs[row["pair_id"]].append(row)
            quartets[row["group_id"]].append(row)
        self.assertEqual(len(pairs), 120)
        self.assertEqual(len(quartets), 60)
        for rows in pairs.values():
            self.assertEqual({row["label"] for row in rows}, {0, 1})
            self.assertEqual(len({row["prompt"] for row in rows}), 1)
            self.assertEqual(len({row["completion"] for row in rows}), 2)
        for rows in quartets.values():
            self.assertEqual(len(rows), 4)
            supported = Counter(r["completion"] for r in rows if r["label"] == 0)
            unsupported = Counter(r["completion"] for r in rows if r["label"] == 1)
            self.assertEqual(supported, unsupported)

    def test_labels_are_grounded_in_source_and_spans(self):
        source = json.loads(SOURCE.read_text(encoding="utf-8"))
        layouts = {x["id"]: x for x in source["layouts"]}
        fields = {x["id"]: x for x in source["fields"]}
        counts = Counter()
        for r in self.rows:
            with self.subTest(row=r["id"]):
                self.assertEqual(r["source_version"], source["version"])
                self.assertIn(r["candidate_value"], fields[r["field_id"]]["values"])
                expected_chart = layouts[r["layout_id"]]["chart"].format(
                    field=fields[r["field_id"]]["field"],
                    current=r["current_value"], prior=r["prior_value"],
                )
                self.assertEqual(r["chart"], expected_chart)
                self.assertTrue(r["prompt"].startswith(expected_chart))
                self.assertEqual(r["label"], int(r["candidate_value"] != r["current_value"]))
                self.assertNotEqual(r["current_value"], r["prior_value"])
                self.assertEqual(r["completion"].strip().removesuffix("."), r["candidate_value"])
                target = r["completion"][r["target_start"]:r["target_end"]]
                self.assertTrue(target)
                self.assertEqual(target, r["target_text"])
                appears = target.casefold() in r["prompt"].casefold()
                self.assertEqual(appears, r["label"] == 0 or r["alternative_location"] == "prior")
                if r["label"] == 1:
                    counts[r["split"], r["field_id"], r["alternative_location"]] += 1
        for split in ("train", "validation", "test"):
            for field in fields:
                self.assertGreater(counts[split, field, "prior"], 0)
                self.assertEqual(counts[split, field, "prior"], counts[split, field, "absent"])

    def test_export_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "examples.jsonl"
            command = [sys.executable, "-m", "mech_interp.data", "--output", str(path)]
            subprocess.run(command, cwd=SOURCE.parents[1], check=True, capture_output=True)
            data = path.read_bytes()
            self.assertNotIn(b"\r", data)
            decoded = [json.loads(line) for line in data.splitlines()]
            self.assertEqual(decoded, self.rows)
            subprocess.run(command, cwd=SOURCE.parents[1], check=True, capture_output=True)
            self.assertEqual(path.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
