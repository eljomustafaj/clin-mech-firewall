"""Find a visible chart mention in a candidate prefix and score its causal residual."""

from dataclasses import asdict, dataclass
import hashlib
import json
import re

import numpy as np

from .model import last_span_token
from .probe import score
from .temporal_probe import BASELINE_SHA256


MATCH_VERSION = "source-suffix.v1"
WORD = re.compile(r"\b\w+(?:[-'’]\w+)*\b")
BLANK_LINE = re.compile(r"\r?\n[ \t]*\r?\n")
STOP = frozenset("""
    a an the this that these those and or but nor so as at by for from in into of on onto
    to up upon with without after before during over under between through per than then
    i me my mine we our ours us you your yours he him his she her hers it its they them
    their theirs who whom whose which what there here when where how is am are was were
    be been being have has had do does did can could may might must shall should will would
    not no yes only also very s t
""".split())


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    words: int


def _same_phrase(text, left, right):
    gap = text[left.end():right.start()]
    return gap.isspace() and "\n" not in gap and "\r" not in gap


def locate_source(prompt, candidate_prefix):
    """Longest unique suffix, up to five whole words in one phrase on one chart line.

    Ignore trailing punctuation and whitespace. Treat case as insignificant. A phrase
    needs at least one non-function word. The first blank line ends the chart. Multiple
    occurrences of the longest matching suffix make the source ambiguous.
    """
    chart = BLANK_LINE.split(prompt, maxsplit=1)[0]
    candidate_words = list(WORD.finditer(candidate_prefix))[-5:]
    if not candidate_words:
        return None
    chart_words = list(WORD.finditer(chart))
    folded_chart = [w.group().casefold() for w in chart_words]
    for count in range(len(candidate_words), 0, -1):
        suffix = candidate_words[-count:]
        if not all(_same_phrase(candidate_prefix, a, b) for a, b in zip(suffix, suffix[1:])):
            continue
        wanted = [w.group().casefold() for w in suffix]
        if all(w in STOP for w in wanted):
            continue
        hits = []
        for start in range(len(chart_words) - count + 1):
            if folded_chart[start:start + count] != wanted:
                continue
            span = chart_words[start:start + count]
            if all(_same_phrase(chart, a, b) for a, b in zip(span, span[1:])):
                hits.append(Match(span[0].start(), span[-1].end(), count))
                if len(hits) > 1:
                    return None
        if hits:
            return hits[0]
    return None


def candidate_prefix(model, row):
    """Benchmark adapter. target_end identifies the consumed token in the annotated answer."""
    prompt, completion, end = row["prompt"], row["completion"], row["target_end"]
    if not 0 < end <= len(completion):
        raise ValueError("Benchmark target_end falls outside completion")
    encoded = model.tokenizer(prompt + completion, return_offsets_mapping=True,
                              add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    absolute = len(prompt) + end
    index = last_span_token(offsets, absolute - 1, absolute)
    return completion[:offsets[index][1] - len(prompt)]


def extract_sources(model, prompts, prefixes, batch_size=8):
    """Find mentions from raw strings; return all matches, matched row indices and vectors."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    matches, indices, keys, unique = [], [], [], {}
    for i, (prompt, prefix) in enumerate(zip(prompts, prefixes, strict=True)):
        hit = locate_source(prompt, prefix)
        matches.append(hit)
        if hit is not None:
            key = (prompt, hit.start, hit.end)
            if key not in unique:
                unique[key] = len(unique)
            indices.append(i)
            keys.append(unique[key])
    if unique:
        # The original prompt fixes token boundaries. encode_rows truncates at the last
        # token overlapping this raw-text match, so later chart text cannot enter its state.
        inputs = [{"id": str(i), "prompt": "", "completion": text,
                   "target_start": start, "target_end": end}
                  for i, (text, start, end) in enumerate(unique)]
        unique_vectors = model.encode_rows(inputs, batch_size)
        vectors = unique_vectors[keys]
    else:
        vectors = np.empty((0, model.hidden_size), dtype=np.float32)
    return matches, np.asarray(indices, dtype=np.int64), vectors


def score_cached(vectors, matched_indices, fallback_risks, source_probe):
    """Use source-role scores only on unique matches; apply the existing score elsewhere."""
    result = np.array(fallback_risks, dtype=np.float64, copy=True)
    indices = np.asarray(matched_indices)
    vectors = np.asarray(vectors)
    if (result.ndim != 1 or not np.isfinite(result).all()
            or np.any((result < 0) | (result > 1))):
        raise ValueError("Expected finite fallback probabilities in [0, 1]")
    if (indices.ndim != 1 or (indices.size and not np.issubdtype(indices.dtype, np.integer))
            or np.any(indices < 0) or np.any(indices >= len(result)) or np.any(np.diff(indices) <= 0)
            or vectors.shape != (len(indices), source_probe["hidden_size"])):
        raise ValueError("Matched vectors must correspond to unique ascending row positions")
    if len(indices):
        result[indices] = score(vectors, source_probe)
    return result


def score_prefixes(model, prompts, prefixes, fallback_risks, source_probe, batch_size=8):
    """Return pre-release risks and auditable evidence offsets, without annotations."""
    expected = {"schema_version": "source-role-probe.v1", "match_version": MATCH_VERSION,
                "model_id": model.model_id, "model_revision": model.revision, "layer": model.layer,
                "hidden_size": model.hidden_size, "positive_label": "unsupported",
                "fallback_probe_sha256": BASELINE_SHA256}
    if any(source_probe.get(k) != val for k, val in expected.items()):
        raise ValueError("Source probe does not match the active model, matcher or fallback")
    matches, indices, vectors = extract_sources(model, prompts, prefixes, batch_size)
    if len(fallback_risks) != len(matches):
        raise ValueError("Supply one frozen v1 fallback risk for every prefix")
    risks = score_cached(vectors, indices, fallback_risks, source_probe)
    return [{"risk_score": float(risk), "fallback_risk": float(fallback),
             "match": asdict(hit) if hit is not None else None}
            for risk, fallback, hit in zip(risks, fallback_risks, matches, strict=True)]


class SourceRoleScorer:
    """Score actual model candidates before emitting their event to the worker."""

    def __init__(self, model, source_probe, fallback_probe):
        raw = (json.dumps(fallback_probe, indent=2, ensure_ascii=False, allow_nan=False)
               + "\n").encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != BASELINE_SHA256:
            raise ValueError("The source-role worker requires the original published v1 fallback")
        self.model = model
        self.probe = source_probe
        self.fallback = fallback_probe

    def generate_tokens(self, prompt, max_new_tokens):
        prefix = ""
        for token in self.model.generate_tokens(prompt, self.fallback, max_new_tokens):
            prefix += token["text"]
            info = score_prefixes(self.model, [prompt], [prefix], [token["risk_score"]], self.probe)[0]
            yield {**token, **info}
