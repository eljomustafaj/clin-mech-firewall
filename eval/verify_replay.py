"""Independently replay a published Qwen2 trace with HF generation and a block hook."""

import argparse
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from eval.validate_trace import parse_json, validate_probe, validate_trace
from mech_interp.snapshot import verify_model_source


@torch.inference_mode()
def verify(run_dir, model_path, tolerance):
    trace = parse_json((run_dir / "trace.json").read_bytes())
    manifest = parse_json((run_dir / "manifest.json").read_bytes())
    probe_path = run_dir / "probe.json"
    validate_trace(trace)
    validate_probe(trace, probe_path)
    probe = parse_json(probe_path.read_bytes())
    verify_model_source(trace["model_id"], trace["model_revision"], model_path)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(manifest["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = str(model_path) if model_path else trace["model_id"]
    load = {"revision": trace["model_revision"], "local_files_only": bool(model_path),
            "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(source, **load)
    model = AutoModelForCausalLM.from_pretrained(
        source, torch_dtype=getattr(torch, manifest["dtype"]), use_safetensors=True,
        attn_implementation="eager", **load,
    ).to(manifest["device"]).eval()
    if model.config.model_type != "qwen2":
        raise ValueError("This independent block-hook check supports Qwen2")
    input_ids = tokenizer(trace["prompt"], add_special_tokens=False,
                          return_tensors="pt").input_ids.to(manifest["device"])
    input_length = input_ids.shape[1]
    eos = model.generation_config.eos_token_id
    generation = GenerationConfig(
        do_sample=False, use_cache=False, eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else
                     (eos[0] if isinstance(eos, list) else eos),
        max_new_tokens=manifest["max_new_tokens"],
    )
    states = []

    def capture(_module, _args, output):
        residual = output[0] if isinstance(output, tuple) else output
        states.append(residual[0, -1].detach().float().cpu().tolist())

    hook = model.model.layers[trace["layer"]].register_forward_hook(capture)
    try:
        ids = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                             generation_config=generation)
        completion_ids = ids[0, input_length:].tolist()
        if completion_ids != [t["token_id"] for t in trace["tokens"]]:
            raise ValueError("Hugging Face greedy generation differs from the published token IDs")
        # The first scoring pass consumed only the prompt. The decoder still needs
        # one final pass to observe the last generated candidate's residual.
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        hook.remove()
    if len(states) != len(completion_ids) + 1:
        raise ValueError(f"Unexpected number of block activations: {len(states)}")
    decode = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
    prompt_text = tokenizer.decode(input_ids[0], **decode)
    actual = prompt_text
    max_error = 0.0
    for position, (token, state) in enumerate(zip(trace["tokens"], states[1:], strict=True)):
        logit = math.fsum(w * x for w, x in zip(probe["weights"], state, strict=True)) + probe["bias"]
        score = 0.5 + 0.5 * math.tanh(logit / 2)
        max_error = max(max_error, abs(score - token["risk_score"]))
        actual += token["text"]
        decoded = tokenizer.decode(ids[0, :input_length + position + 1], **decode)
        # A byte-level token may temporarily decode to an incomplete code point.
        if not decoded.endswith("\ufffd") and actual != decoded:
            raise ValueError(f"Incremental decode differs at token {position}")
    if actual != tokenizer.decode(ids[0], **decode):
        raise ValueError("The exact text decoded from the generated IDs differs from the trace")
    if max_error > tolerance:
        raise ValueError(f"Maximum block-hook score error {max_error:.8g} exceeds {tolerance:.8g}")
    return len(completion_ids), max_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="directory containing the published baseline")
    parser.add_argument("--model-path", type=Path, help="offline snapshot of its pinned Hub model")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="absolute risk-score tolerance")
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("tolerance must be finite and nonnegative")
    n, error = verify(args.run_dir, args.model_path, args.tolerance)
    print(f"Verified {n} greedy token IDs and decoded text; maximum block-hook score error {error:.8g}")


if __name__ == "__main__":
    main()
