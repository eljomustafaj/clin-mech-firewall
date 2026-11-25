"""Send candidate scores from an already-loaded research model to the firewall."""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from eval.export_wire import scored_wire_records
from eval.validate_trace import parse_probe
from worker.replay import write_trace

if TYPE_CHECKING:
    from mech_interp.model import ResidualModel


def write_inference(
    model: "ResidualModel",
    prompt: str,
    probe_path: Path,
    target: BinaryIO,
    *,
    trace_id: str,
    max_new_tokens: int,
) -> None:
    """Flush live v1 records to target using the caller's run ID and model instance."""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a nonempty string")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    raw = Path(probe_path).read_bytes()
    source = {
        "trace_id": trace_id,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "layer": model.layer,
        "probe_id": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }
    probe = parse_probe(raw, source)
    if probe["hidden_size"] != model.hidden_size:
        raise ValueError("probe hidden_size differs from the loaded model")
    tokens = model.generate_tokens(prompt, probe, max_new_tokens)
    write_trace(scored_wire_records(source, tokens), target)
