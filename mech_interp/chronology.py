"""Apply frozen chart-line chronology cues to v1 scores as tokens are generated."""

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path


PRIOR_PATTERN = re.compile(r"\b(?:prior|previous|previously|earlier|older|past|last|historical|archived|closed|preceding|former)\b", re.I)
CURRENT_PATTERN = re.compile(r"\b(?:current|today|active|present|latest|now|ongoing|confirmed|this encounter|this visit|most recent|in progress)\b", re.I)


def cue_score(prompt, start, frozen_risk):
    """Score the prefix of a matching chart line; return score and decision basis."""
    if start is None:
        return frozen_risk, "unmatched"
    if type(start) is not int or not 0 <= start < len(prompt):
        raise ValueError("match cursor outside prompt")
    line_prefix = prompt[prompt.rfind("\n", 0, start) + 1:start]
    prior = PRIOR_PATTERN.search(line_prefix) is not None
    current = CURRENT_PATTERN.search(line_prefix) is not None
    if prior and not current:
        return 1.0, "prior-cue"
    if current and not prior:
        return 0.0, "current-cue"
    return frozen_risk, "conflicting-cues" if prior else "no-cues"


def generate_chronology_tokens(model, prompt, fallback_probe_path, max_new_tokens):
    """Yield one scored token at a time. The caller controls release and stopping."""
    from .source_probe import locate_source
    from .temporal_probe import BASELINE_SHA256

    raw = Path(fallback_probe_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASELINE_SHA256:
        raise ValueError("Chronology scoring requires the published v1 probe and threshold")
    probe = json.loads(raw)
    threshold = probe["threshold"]
    prefix = ""
    for token in model.generate_tokens(prompt, probe, max_new_tokens):
        prefix += token["text"]
        match = locate_source(prompt, prefix)
        fallback = token["risk_score"]
        risk, basis = cue_score(prompt, match.start if match else None, fallback)
        yield {**token, "risk_score": risk, "fallback_risk": fallback, "cue": basis,
               "match": asdict(match) if match is not None else None, "flagged": risk >= threshold}
