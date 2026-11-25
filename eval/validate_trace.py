"""Check the research handoff using only the Python standard library."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def parse_json(raw):
    def reject_constant(value):
        raise ValueError(f"invalid JSON numeric constant: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_trace(trace):
    """Raise ValueError for malformed fields, invalid scores, or skipped tokens."""
    _object(trace, "trace")
    if trace.get("schema_version") != "research-trace.v1":
        raise ValueError("unsupported trace schema_version")
    for name in ("model_id", "model_revision", "trace_id", "prompt"):
        if not isinstance(trace.get(name), str) or not trace[name]:
            raise ValueError(f"{name} must be a nonempty string")
    if type(trace.get("layer")) is not int or trace["layer"] < 0:
        raise ValueError("layer must be a nonnegative integer")
    if not isinstance(trace.get("probe_id"), str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", trace["probe_id"]
    ):
        raise ValueError("probe_id must be sha256: followed by 64 lowercase hex digits")
    if not isinstance(trace.get("tokens"), list) or not trace["tokens"]:
        raise ValueError("tokens must be a nonempty array")
    for position, token in enumerate(trace["tokens"]):
        label = f"tokens[{position}]"
        _object(token, label)
        if type(token.get("position")) is not int or token["position"] != position:
            raise ValueError(f"{label}.position must equal {position}")
        if type(token.get("token_id")) is not int or token["token_id"] < 0:
            raise ValueError(f"{label}.token_id must be a nonnegative integer")
        if not isinstance(token.get("text"), str):
            raise ValueError(f"{label}.text must be a string")
        score = token.get("risk_score")
        if not _finite_number(score) or not 0 <= score <= 1:
            raise ValueError(f"{label}.risk_score must be a finite number in [0, 1]")


def validate_probe(trace, path):
    """Match the published artifact byte-for-byte and check its scoring dimensions."""
    raw = path.read_bytes()
    if trace["probe_id"] != "sha256:" + hashlib.sha256(raw).hexdigest():
        raise ValueError("probe_id differs from the supplied file's SHA-256")
    probe = parse_json(raw)
    _object(probe, "probe")
    if probe.get("schema_version") != "linear-probe.v1" or probe.get("positive_label") != "unsupported":
        raise ValueError("expected linear-probe.v1 with positive_label=unsupported")
    for name in ("model_id", "model_revision", "layer"):
        if type(probe.get(name)) is not type(trace[name]) or probe[name] != trace[name]:
            raise ValueError(f"probe {name} differs from the trace")
    size = probe.get("hidden_size")
    weights = probe.get("weights")
    if type(size) is not int or size <= 0 or not isinstance(weights, list) or len(weights) != size:
        raise ValueError("probe weights must have exactly hidden_size > 0 elements")
    if not all(_finite_number(weight) for weight in weights) or not _finite_number(probe.get("bias")):
        raise ValueError("probe weights and bias must be finite numbers")
    threshold = probe.get("threshold")
    if not _finite_number(threshold) or not 0 <= threshold <= 1:
        raise ValueError("probe threshold must be a finite number in [0, 1]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="path to trace.json")
    parser.add_argument("--probe", type=Path, help="verify the exact probe.json used for the trace")
    args = parser.parse_args()
    try:
        trace = parse_json(args.trace.read_bytes())
        validate_trace(trace)
        if args.probe:
            validate_probe(trace, args.probe)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Invalid handoff: {exc}\n")
    print(f"Validated {trace['trace_id']} ({len(trace['tokens'])} tokens)")


if __name__ == "__main__":
    main()
