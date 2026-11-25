"""Verify one archived free generation and emit its chronology decisions as wire-v1 input."""

import argparse
import hashlib
import math
import tempfile
from dataclasses import asdict
from pathlib import Path

from eval.export_wire import scored_wire_records
from eval.run import digest, trace_identifier, write_json
from eval.validate_trace import parse_json, validate_trace
from mech_interp.chronology import CURRENT_PATTERN, PRIOR_PATTERN, cue_score
from mech_interp.source_probe import MATCH_VERSION, locate_source
from mech_interp.temporal_probe import BASELINE_SHA256
from worker.replay import write_trace


def export(generation, fallback_path, source_row, folder):
    metadata = parse_json((generation / "metrics.json").read_bytes())
    path = generation / "generations.jsonl"
    if digest(path) != metadata["sha256"][path.name] or digest(fallback_path) != BASELINE_SHA256:
        raise ValueError("Generation archive or fallback probe checksum mismatch")
    rows = [parse_json(line) for line in path.read_bytes().splitlines()]
    selected = [row for row in rows if row["source_row"] == source_row]
    if len(rows) != metadata["n"] or len(selected) != 1:
        raise ValueError("Expected one named source row in the complete generation archive")
    row = selected[0]
    probe = parse_json(fallback_path.read_bytes())
    family = "chart-chronology-words-v1"
    policy = {"family": family, "match_version": MATCH_VERSION, "line_scope": "matching line before mention",
              "prior_regex": PRIOR_PATTERN.pattern, "current_regex": CURRENT_PATTERN.pattern,
              "regex_flags": "IGNORECASE (Unicode)", "prior_only_risk": 1.0, "current_only_risk": 0.0,
              "otherwise": "original token risk", "fallback_probe_sha256": BASELINE_SHA256,
              "halt_rule": "risk >= threshold", "threshold": probe["threshold"]}
    prefix, events = "", []
    for token in row["tokens"]:
        prefix += token["text"]
        match = locate_source(row["prompt"], prefix)
        fallback = token["original_risk"]
        if type(fallback) not in (int, float) or not math.isfinite(fallback) or not 0 <= fallback <= 1:
            raise ValueError("Invalid archived fallback risk")
        risk, cue = cue_score(row["prompt"], match.start if match else None, fallback)
        if (token["source_match"] != (asdict(match) if match else None) or
                token["cue_reason"] != cue or token["cue_risk"] != risk):
            raise ValueError("Archived chronology evidence or risk differs from recomputation")
        events.append({k: token[k] for k in ("position", "text", "token_id")} | {"risk_score": risk})
    first = next((i for i, e in enumerate(events) if e["risk_score"] >= probe["threshold"]), None)
    decision = row["policy"]["chronology-words"]
    released = "".join(e["text"] for e in events[:first])
    if prefix != row["generation"] or first != decision["first_halt"] or released != decision["released"]:
        raise ValueError("Recorded generation or release decision differs")
    write_json(folder / "policy.json", policy)
    trace = {"schema_version": "research-trace.v1", "model_id": probe["model_id"],
             "model_revision": probe["model_revision"], "layer": probe["layer"],
             "probe_id": "sha256:" + digest(folder / "policy.json"),
             "source_row": source_row, "prompt": row["prompt"], "tokens": events}
    trace["trace_id"] = trace_identifier(trace)
    validate_trace(trace)
    with (folder / "trace.wire.jsonl").open("wb") as stream:
        write_trace(scored_wire_records(trace, events, probe_family=family), stream)
    return first, hashlib.sha256(released.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, default=Path("artifacts/temporal-v2/generation"))
    parser.add_argument("--fallback", type=Path, default=Path("artifacts/baseline/probe.json"))
    parser.add_argument("--row", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        stage = Path(temp) / "wire"
        stage.mkdir()
        first, released_sha = export(args.generation, args.fallback, args.row, stage)
        stage.rename(args.output)
    print(f"First HALT: {first}; expected response SHA-256: {released_sha}")


if __name__ == "__main__":
    main()
