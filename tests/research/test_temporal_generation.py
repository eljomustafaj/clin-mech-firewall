import unittest

from eval.generate_temporal import classify_opening, compare_generation, sample_prompts, summarize


class TemporalGenerationTests(unittest.TestCase):
    def test_opening_is_anchored_and_article_insensitive(self):
        self.assertEqual(classify_opening("  The Blue file.", "a blue file", "red file"), ("current", 15))
        self.assertEqual(classify_opening("red file.", "the blue file", "a red file"), ("prior", 8))
        for text in ["I saw a red file.", "red files", "possibly blue", ""]:
            self.assertEqual(classify_opening(text, "blue file", "red file"), ("unclassified", None))
        self.assertEqual(classify_opening("red file", "red", "red file"), ("unclassified", None))

    def test_sampling_is_balanced_repeatable_and_independent_of_row_order(self):
        rows = [dict(prompt=f"p-{k}-{i}", layout_id="a", chart_order=k, label=0,
                     alternative_location="prior") for k in ["prior_first", "current_first"] for i in range(8)]
        selected = sample_prompts(rows + rows)
        self.assertEqual(selected, sample_prompts(list(reversed(rows))))
        self.assertEqual(len(selected), 8)
        self.assertEqual(len({r["prompt"] for r in selected}), 8)

    def test_generic_early_halt_and_late_halt_are_distinct(self):
        row = dict(id="a", group_id="g", layout_id="l", chart_order="prior-first", prompt="chart",
                   current_value="by bus", prior_value="by car")
        events = [dict(position=i, token_id=i, text=text, risk_score=risk)
                  for i, (text, risk) in enumerate([(" by", 0.7), (" car", 0.2), (".", 0.9)])]
        other = [dict(e, risk_score=(1.0 if e["position"] == 2 else 0.0)) for e in events]
        probes = {name: {"threshold": 0.7} for name in ("original", "selected")}
        result = compare_generation(row, {"original": events, "selected": other}, probes)
        self.assertEqual(result["opening"], "prior")
        self.assertEqual(result["value_end_position"], 1)
        self.assertEqual(result["policy"]["original"]["released"], "")
        self.assertEqual(result["policy"]["selected"]["released"], " by car")
        counts = summarize([result])["prior"]
        self.assertEqual(counts["original"]["halt_by_value_end"], 1)
        self.assertEqual(counts["selected"]["halt_by_value_end"], 0)
        with self.assertRaisesRegex(ValueError, "different generated tokens"):
            compare_generation(row, {"original": events, "selected": other[:-1]}, probes)
