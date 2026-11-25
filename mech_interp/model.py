"""Causal-LM residual extraction and candidate-token scoring for the research baseline."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import score


def last_span_token(offsets, start, end):
    """Index of the last token overlapping a nonempty half-open character span."""
    if start < 0 or start >= end:
        raise ValueError("Expected a nonempty target span")
    matches = [i for i, (left, right) in enumerate(offsets)
               if right > left and left < end and right > start]
    if not matches or offsets[matches[0]][0] > start or offsets[matches[-1]][1] < end:
        raise ValueError(f"Tokenizer offsets do not cover the target span [{start}, {end})")
    return matches[-1]


class ResidualModel:
    """Use the HF decoder's intermediate hidden state, after the indicated block."""

    def __init__(self, model_id, revision, layer, device, dtype, model_path=None):
        source = str(model_path) if model_path else model_id
        load_args = {"revision": revision, "local_files_only": bool(model_path),
                     "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **load_args)
        if not self.tokenizer.is_fast:
            raise ValueError("An offset-aware fast tokenizer is required")
        self.model = AutoModelForCausalLM.from_pretrained(
            source, torch_dtype=getattr(torch, dtype), attn_implementation="eager",
            use_safetensors=True, **load_args,
        ).to(device).eval()
        depth = self.model.config.num_hidden_layers
        if not 0 <= layer < depth - 1:
            raise ValueError(f"Choose a non-final block, 0 through {depth - 2}; "
                             "the final hidden state can include the output normalization")
        self.layer = layer
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.hidden_size = self.model.config.hidden_size
        self.max_positions = getattr(self.model.config, "max_position_embeddings", None)

    def _check_length(self, length):
        if self.max_positions is not None and length > self.max_positions:
            raise ValueError(f"{length} tokens exceed the model context of {self.max_positions}")

    def encode_example(self, row):
        prompt, completion = row["prompt"], row["completion"]
        start, end = row["target_start"], row["target_end"]
        if not 0 <= start < end <= len(completion):
            raise ValueError(f"Invalid target offsets for {row['id']}")
        encoded = self.tokenizer(prompt + completion, return_offsets_mapping=True,
                                 add_special_tokens=False)
        target = last_span_token(encoded["offset_mapping"], len(prompt) + start, len(prompt) + end)
        # Causal attention guarantees that text after the target cannot affect this state.
        ids = encoded["input_ids"][:target + 1]
        self._check_length(len(ids))
        return ids

    @torch.inference_mode()
    def encode_rows(self, rows, batch_size):
        if not rows or batch_size < 1:
            raise ValueError("A nonempty dataset and positive batch size are required")
        features = []
        for offset in range(0, len(rows), batch_size):
            sequences = [self.encode_example(row) for row in rows[offset:offset + batch_size]]
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id or 0
            lengths = torch.tensor([len(ids) for ids in sequences], device=self.device)
            width = int(lengths.max().item())
            ids = torch.tensor([seq + [pad_id] * (width - len(seq)) for seq in sequences],
                               device=self.device)
            mask = torch.arange(width, device=self.device)[None, :] < lengths[:, None]
            output = self.model.base_model(input_ids=ids, attention_mask=mask.long(),
                                           output_hidden_states=True, use_cache=False,
                                           return_dict=True)
            rows_in_batch = torch.arange(len(sequences), device=self.device)
            states = output.hidden_states[self.layer + 1][rows_in_batch, lengths - 1]
            features.append(states.float().cpu().numpy())
            del output, states
        return np.concatenate(features)

    @torch.inference_mode()
    def generate_tokens(self, prompt, probe, max_new_tokens):
        """Greedy-generate, then consume and score each candidate before yielding its event."""
        if max_new_tokens < 1:
            raise ValueError("Request at least one generated token")
        expected = {"schema_version": "linear-probe.v1", "model_id": self.model_id,
                    "model_revision": self.revision, "layer": self.layer,
                    "hidden_size": self.hidden_size, "positive_label": "unsupported"}
        for key, value in expected.items():
            if probe.get(key) != value or type(probe[key]) is not type(value):
                raise ValueError(f"Probe metadata '{key}' does not match the active scorer")
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            raise ValueError("The trace prompt must tokenize to at least one token")
        self._check_length(len(ids) + max_new_tokens)
        eos = self.model.generation_config.eos_token_id
        stop_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        kwargs = {"use_cache": False, "return_dict": True}
        seq = torch.tensor([ids], device=self.device)
        candidate = self.model(input_ids=seq, **kwargs).logits[0, -1].argmax().item()
        decode_args = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        displayed = self.tokenizer.decode(ids, **decode_args)
        for position in range(max_new_tokens):
            ids.append(candidate)
            seq = torch.tensor([ids], device=self.device)
            output = self.model(input_ids=seq, output_hidden_states=True, **kwargs)
            state = output.hidden_states[self.layer + 1][0, -1].float().cpu().numpy()
            risk = float(score(state, probe))
            finished = candidate in stop_ids or position + 1 == max_new_tokens
            decoded = self.tokenizer.decode(ids, **decode_args)
            # A byte-level token can end mid-codepoint. Wait for subsequent bytes before
            # displaying that character. Flush the tokenizer's exact decode on the last token.
            if not finished:
                decoded = decoded.rstrip("\ufffd")
            if not decoded.startswith(displayed):
                raise ValueError("Tokenizer rewrote previously decoded text during streaming")
            yield {"position": position, "token_id": candidate, "text": decoded[len(displayed):],
                   "risk_score": risk}
            if finished:
                return
            displayed = decoded
            candidate = output.logits[0, -1].argmax().item()
            del output
