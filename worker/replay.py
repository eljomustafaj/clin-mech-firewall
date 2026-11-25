"""Stream a UTF-8 JSONL trace to the firewall, one record per flush."""

import argparse
import errno
import json
import os
import re
import stat
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, BinaryIO


MAX_SAFE_INTEGER = 2**53 - 1
IDENTIFIERS = ("trace_id", "model_id", "model_revision", "probe_id", "probe_revision")
# The Unicode White_Space property shared by the Rust proxy and browser.
IDENTIFIER_CONTENT = re.compile(
    r"[^\u0009-\u000d\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]"
)
START_FIELDS = {"type", "schema_version", "layer", *IDENTIFIERS}
TOKEN_FIELDS = {"type", "index", "text", "risk"}
END_FIELDS = {"type", "token_count"}


class TraceError(ValueError):
    """A trace record cannot be represented or consumed."""


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant {value}")


def _read_json_lines(source: BinaryIO) -> Iterator[dict[str, Any]]:
    """Decode JSON objects lazily from a binary stream, preserving token text."""
    for line_number, raw in enumerate(source, 1):
        try:
            record = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_invalid_constant
            )
            if not isinstance(record, dict):
                raise ValueError("expected a JSON object")
        except (ValueError, RecursionError) as exc:
            raise TraceError(f"line {line_number}: {exc}") from exc
        yield record


def _fields(record: dict[str, Any], required: set[str], optional: tuple[str, ...] = ()) -> None:
    missing = required.difference(record)
    extra = record.keys() - required.union(optional)
    if missing:
        raise ValueError(f"missing field {min(missing)!r}")
    if extra:
        raise ValueError(f"unexpected field {next(iter(extra))!r}")


def _integer(value: Any, field: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_SAFE_INTEGER:
        raise ValueError(f"{field} must be a nonnegative safe integer")
    return value


def _validate_start(record: dict[str, Any]) -> None:
    if record.get("type") != "start":
        raise ValueError("expected a start record")
    _fields(record, START_FIELDS)
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("expected integer schema_version 1")
    _integer(record["layer"], "layer")
    for key in IDENTIFIERS:
        value = record[key]
        if not isinstance(value, str) or not IDENTIFIER_CONTENT.search(value):
            raise ValueError(f"{key} must be a nonempty string")
        value.encode("utf-8")


def _validate_token(record: dict[str, Any], expected_index: int) -> None:
    _fields(record, TOKEN_FIELDS, ("sae_feature_ids",))
    if _integer(record["index"], "index") != expected_index:
        raise ValueError(f"expected token index {expected_index}")
    if expected_index >= MAX_SAFE_INTEGER:
        raise ValueError("token count exceeds the safe integer range")
    if not isinstance(record["text"], str):
        raise ValueError("text must be a string")
    record["text"].encode("utf-8")
    risk = record["risk"]
    if type(risk) not in (int, float) or not 0 <= risk <= 1:
        raise ValueError("risk must be a finite number in [0, 1]")
    if "sae_feature_ids" in record:
        features = record["sae_feature_ids"]
        if not isinstance(features, list):
            raise ValueError("sae_feature_ids must be an array")
        for feature_id in features:
            _integer(feature_id, "SAE feature ID")


def validate_trace(records: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield a v1 start and scored tokens immediately. Yield the end only at EOF."""
    count = 0
    line_number = 0
    end = None
    for line_number, record in enumerate(records, 1):
        try:
            if not isinstance(record, dict):
                raise ValueError("expected a JSON object")
            if end is not None:
                raise ValueError("unexpected record after end")
            if line_number == 1:
                _validate_start(record)
            elif record.get("type") == "token":
                _validate_token(record, count)
                count += 1
            elif record.get("type") == "end":
                _fields(record, END_FIELDS)
                if _integer(record["token_count"], "token_count") != count:
                    raise ValueError(f"expected token_count {count}")
                end = record
                continue
            else:
                raise ValueError("expected a token or end record")
        except ValueError as exc:
            raise TraceError(f"line {line_number}: {exc}") from exc
        yield record
    if end is None:
        expected = "end" if line_number else "start"
        raise TraceError(f"line {line_number + 1}: expected {expected} before EOF")
    yield end


def read_trace(source: BinaryIO) -> Iterator[dict[str, Any]]:
    """Read and validate one scored trace from a binary JSONL stream."""
    return validate_trace(_read_json_lines(source))


def write_trace(records: Iterable[dict[str, Any]], target: BinaryIO) -> None:
    """Validate and flush records from a live scorer or replay to a binary stream."""
    for record in validate_trace(records):
        line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        target.write(line + b"\n")
        target.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="path to a scored JSONL trace, or - for standard input")
    args = parser.parse_args()
    try:
        if args.trace == "-":
            write_trace(_read_json_lines(sys.stdin.buffer), sys.stdout.buffer)
        else:
            with Path(args.trace).open("rb") as source:
                write_trace(_read_json_lines(source), sys.stdout.buffer)
    except OSError as exc:
        if exc.errno == errno.EPIPE or (
            os.name == "nt"
            and exc.errno == errno.EINVAL
            and exc.filename is None
            and stat.S_ISFIFO(os.fstat(sys.stdout.fileno()).st_mode)
        ):
            # A terminal HALT can close the downstream reader between records.
            # Redirect stdout so the interpreter's final flush also succeeds.
            with open(os.devnull, "wb") as sink:
                os.dup2(sink.fileno(), sys.stdout.fileno())
            return 0
        print(f"worker: {exc}", file=sys.stderr)
        return 1
    except TraceError as exc:
        print(f"worker: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
