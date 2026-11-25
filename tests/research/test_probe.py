import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from eval.run import grouped_intervals, pair_accuracy, split_indices, trace_identifier
from mech_interp.model import ResidualModel, last_span_token
from mech_interp.probe import choose_threshold, classification_metrics, fit_probe, score, sigmoid
from worker.inference import write_inference
from worker.replay import read_trace


class ProbeTests(unittest.TestCase):
    def test_weights_serialize_and_score_raw_activations(self):
        x = np.array([[0, 100], [1, 100], [4, 100], [5, 100]], dtype=np.float32)
        fitted = json.loads(json.dumps(fit_probe(x, [0, 0, 1, 1])))
        scores = score(x, fitted)
        np.testing.assert_array_equal(scores >= 0.5, [False, False, True, True])
        self.assertEqual(fitted["weights"][1], 0.0)
        np.testing.assert_allclose(score(x[1], fitted), scores[1])
        np.testing.assert_array_equal(sigmoid([-10000, 10000]), [0, 1])
        with self.assertRaises(ValueError):
            score([float("nan"), 0], fitted)
        with self.assertRaises(ValueError):
            score([0, 0, 0], fitted)

    def test_threshold_metrics_and_ranking_handle_ties(self):
        y = [0, 0, 1, 1]
        risks = [0.1, 0.3, 0.3, 0.9]
        threshold = choose_threshold(y, risks)
        self.assertEqual(threshold, 0.3)
        m = classification_metrics(y, risks, threshold)
        self.assertEqual((m["tp"], m["fp"], m["tn"], m["fn"]), (2, 1, 1, 0))
        self.assertEqual(m["auroc"], 0.875)
        self.assertEqual(m["f1"], 0.8)
        self.assertEqual(classification_metrics(y, [0.5] * 4, 0.5)["auroc"], 0.5)
        rows = [{"pair_id": "a", "group_id": "g", "label": 0},
                {"pair_id": "a", "group_id": "g", "label": 1}]
        self.assertEqual(pair_accuracy(rows, [0.5, 0.5]), 0.5)
        self.assertEqual(grouped_intervals(rows, np.array([0.1, 0.8]), 0.5, 3)["auroc"], [1, 1])

    def test_cross_split_group_rejected(self):
        rows = [{"id": str(i), "group_id": "shared", "pair_id": "pair",
                 "label": i, "split": s} for i, s in enumerate(["train", "test"])]
        with self.assertRaisesRegex(ValueError, "split"):
            split_indices(rows)

    def test_trace_identity_binds_model_probe_prompt_and_ordered_scores(self):
        payload = {"model_id": "model", "probe_id": "abc", "prompt": "room",
                   "tokens": [{"token_id": 3, "risk_score": 0.2}]}
        key = trace_identifier(payload)
        self.assertEqual(key, trace_identifier(dict(reversed(list(payload.items())))))
        for field, changed in (("model_id", "other"), ("probe_id", "def"), ("prompt", "arm"),
                               ("tokens", [{"token_id": 3, "risk_score": 0.21}])):
            with self.subTest(field=field):
                self.assertNotEqual(key, trace_identifier({**payload, field: changed}))

    def test_span_alignment_checks_both_edges(self):
        offsets = [(0, 2), (2, 5), (5, 7), (0, 0)]
        self.assertEqual(last_span_token(offsets, 1, 5), 1)
        self.assertEqual(last_span_token(offsets, 2, 6), 2)
        with self.assertRaises(ValueError):
            last_span_token(offsets, 4, 8)


class OfflineModelTests(unittest.TestCase):
    """A randomly initialized tiny decoder tests indexing only; it never supplies research results."""

    @classmethod
    def setUpClass(cls):
        cls.local = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.local.cleanup)
        directory = Path(cls.local.name)
        tok = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "room": 2, "green": 3,
                                   "blue": 4, ".": 5, "patient": 6}, unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="[EOS]")
        fast.save_pretrained(directory)
        torch.manual_seed(17)
        cfg = GPT2Config(vocab_size=7, n_layer=3, n_head=2, n_embd=16, n_positions=64,
                         bos_token_id=1, eos_token_id=1, pad_token_id=0)
        GPT2LMHeadModel(cfg).save_pretrained(directory)
        cls.decoder = ResidualModel("unit-test/tiny", "0" * 40, 1, "cpu", "float32", directory)
        cls.probe = {
            "schema_version": "linear-probe.v1", "model_id": "unit-test/tiny",
            "model_revision": "0" * 40, "layer": 1, "hidden_size": 16,
            "positive_label": "unsupported",
            "weights": np.linspace(-1.5, 1.5, 16).tolist(), "bias": -0.2,
        }

    def test_right_padding_and_span_end_match_single_sequence(self):
        rows = [{"id": "a", "prompt": "room", "completion": " green.",
                 "target_start": 1, "target_end": 6},
                {"id": "b", "prompt": "patient room", "completion": " blue.",
                 "target_start": 1, "target_end": 5}]
        matrix = self.decoder.encode_rows(rows, 2)
        for i, row in enumerate(rows):
            ids = self.decoder.tokenizer.encode(row["prompt"] + row["completion"])
            with torch.inference_mode():
                result = self.decoder.model(input_ids=torch.tensor([ids]), output_hidden_states=True)
            # Punctuation follows the claim; the unpadded claim-end is the penultimate ID.
            expected = result.hidden_states[2][0, -2].numpy()
            np.testing.assert_allclose(matrix[i], expected, atol=1e-6)

    def test_every_trace_score_uses_the_consumed_greedy_token(self):
        prompt = "patient room"
        events = list(self.decoder.generate_tokens(prompt, self.probe, 4))
        ids = self.decoder.tokenizer.encode(prompt)
        original = list(ids)
        for i, event in enumerate(events):
            with torch.inference_mode():
                output = self.decoder.model(input_ids=torch.tensor([ids]))
                self.assertEqual(event["token_id"], output.logits[0, -1].argmax().item())
                ids.append(event["token_id"])
                output = self.decoder.model(input_ids=torch.tensor([ids]), output_hidden_states=True)
            expected = score(output.hidden_states[2][0, -1].numpy(), self.probe)
            self.assertEqual(event["position"], i)
            self.assertAlmostEqual(event["risk_score"], float(expected), places=10)
        opts = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        self.assertEqual(self.decoder.tokenizer.decode(original, **opts)
                         + "".join(event["text"] for event in events),
                         self.decoder.tokenizer.decode(ids, **opts))

    def test_trace_refuses_weights_for_another_model_layer(self):
        wrong_layer = {**self.probe, "layer": 0}
        with self.assertRaisesRegex(ValueError, "layer"):
            list(self.decoder.generate_tokens("room", wrong_layer, 1))

    def test_live_worker_matches_the_loaded_decoder(self):
        prompt = "patient room"
        probe = {**self.probe, "threshold": 0.5}
        expected = list(self.decoder.generate_tokens(prompt, probe, 3))
        target = io.BytesIO()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            path.write_text(json.dumps(probe), encoding="utf-8")
            write_inference(self.decoder, prompt, path, target, trace_id="live-decoder", max_new_tokens=3)
        target.seek(0)
        start, *tokens, end = read_trace(target)
        self.assertEqual(start["trace_id"], "live-decoder")
        self.assertEqual(tokens, [
            {"type": "token", "index": t["position"], "text": t["text"], "risk": t["risk_score"]}
            for t in expected
        ])
        self.assertEqual(end, {"type": "end", "token_count": len(expected)})


if __name__ == "__main__":
    unittest.main()
