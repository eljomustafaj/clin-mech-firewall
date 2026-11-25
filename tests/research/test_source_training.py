import unittest

import numpy as np

from mech_interp.source_probe import extract_sources, locate_source, score_cached


class SourceLocationTests(unittest.TestCase):
    def test_unique_longest_phrase_comes_from_chart_only(self):
        prompt = "Current entry: a blue folder.\nPrior entry: a red folder.\n\nIs the red folder current?"
        found = locate_source(prompt, "The answer is a RED folder.")
        self.assertEqual(prompt[found.start:found.end], "a red folder")
        self.assertEqual(found.words, 3)
        self.assertIsNone(locate_source(prompt, "the folder"))
        self.assertIsNone(locate_source(prompt, "the"))
        self.assertIsNone(locate_source(prompt, "older"))
        self.assertIsNone(locate_source(prompt, "Is the"))

    def test_unique_match_can_be_in_the_first_or_second_visit(self):
        prior, current = "Former: a blue file.", "Active: a red file."
        for order in ((prior, current), (current, prior)):
            prompt = "\n".join(order) + "\n\nAnswer:"
            hit = locate_source(prompt, "a blue file.")
            self.assertEqual(prompt[hit.start:hit.end], "a blue file")
            self.assertTrue(prompt[:hit.start].rstrip().endswith("Former:"))

    def test_extraction_deduplicates_repeated_prefix_source_matches(self):
        class Model:
            hidden_size = 2

            def encode_rows(self, inputs, batch_size):
                self.inputs = inputs
                return np.array([[r["target_start"], r["target_end"]] for r in inputs])

        model = Model()
        chart = "Current: green\nEarlier: red\n\nWhat color?"
        matches, positions, vectors = extract_sources(model, [chart] * 3, ["red", "the", "red."], 2)
        self.assertEqual(positions.tolist(), [0, 2])
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(vectors.shape, (2, 2))
        self.assertEqual(vectors[0].tolist(), [matches[0].start, matches[0].end])
        self.assertEqual(vectors[0].tolist(), vectors[1].tolist())

    def test_unmatched_scores_use_exact_original_risk(self):
        probe = {"hidden_size": 2, "weights": [0.0, 0.0], "bias": 0.0}
        v = np.array([[3.0, -2.0]])
        scores = score_cached(v, np.array([1]), [0.782, 0.1, 0.99], probe)
        np.testing.assert_array_equal(scores, [0.782, 0.5, 0.99])
        with self.assertRaisesRegex(ValueError, "positions"):
            score_cached(np.concatenate((v, v)), np.array([1, 1]), [0.0, 0.0], probe)
