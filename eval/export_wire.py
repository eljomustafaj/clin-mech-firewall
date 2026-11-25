"""Convert a validated research archive into Clad's start/token/end JSONL stream."""

import argparse
import json
import tempfile
from pathlib import Path

from eval.validate_trace import parse_json, validate_probe, validate_trace


def wire_records(trace):
    """Map the validated archive to worker records before any proxy decisions."""
    for field in ("trace_id", "model_id", "model_revision"):
        if not trace[field].strip():
            raise ValueError(f"wire {field} must contain non-whitespace text")
    max_safe_integer = 2**53 - 1
    if trace["layer"] > max_safe_integer or len(trace["tokens"]) > max_safe_integer:
        raise ValueError("wire layer and token count must be JavaScript safe integers")
    yield from scored_wire_records(trace, trace["tokens"])


def scored_wire_records(trace, tokens, *, probe_family="research-linear-probe-v1"):
    """Map archived or live candidate scores to v1 without buffering generated tokens."""
    yield {
        "type": "start",
        "schema_version": 1,
        "trace_id": trace["trace_id"],
        "model_id": trace["model_id"],
        "model_revision": trace["model_revision"],
        "layer": trace["layer"],
        "probe_id": probe_family,
        "probe_revision": trace["probe_id"],
    }
    count = 0
    for count, token in enumerate(tokens, 1):
        yield {
            "type": "token",
            "index": token["position"],
            "text": token["text"],
            "risk": token["risk_score"],
        }
    yield {"type": "end", "token_count": count}


def export_wire(archive: Path, probe: Path, output: Path) -> None:
    if output.resolve() in (archive.resolve(), probe.resolve()):
        raise ValueError("output must be distinct from the trace and probe inputs")
    trace = parse_json(archive.read_bytes())
    validate_trace(trace)
    validate_probe(trace, probe)

    output.parent.mkdir(parents=True, exist_ok=True)
    # Readers must see a complete stream, including its terminal record.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for record in wire_records(trace):
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="research-trace.v1 JSON archive")
    parser.add_argument("--probe", type=Path, required=True, help="exact fitted probe JSON")
    parser.add_argument("--output", type=Path, required=True, help="worker JSONL destination")
    args = parser.parse_args()
    try:
        export_wire(args.trace, args.probe, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Could not export worker trace: {exc}\n")
    print(f"Wrote worker trace to {args.output}")


if __name__ == "__main__":
    main()
