"""Score identical freely generated answers with the original and development-selected probes."""

import argparse
import hashlib
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

from eval.export_wire import export_wire, scored_wire_records
from eval.run import digest, trace_identifier, write_json, write_jsonl
from eval.validate_trace import parse_json, validate_trace
from mech_interp.model import ResidualModel
from mech_interp.snapshot import verify_model_source
from mech_interp.temporal_data import SOURCE_HASHES, build_temporal_dataset
from worker.replay import write_trace


def sample_prompts(rows):
    """Four unique prior-decoy prompts per layout and order; selection never reads scores."""
    strata = defaultdict(dict)
    for row in rows:
        if row["label"] == 0 and row["alternative_location"] == "prior":
            key = (row["layout_id"], row["chart_order"])
            strata[key].setdefault(row["prompt"], row)
    selected = []
    for key in sorted(strata):
        prompts = strata[key]
        ordered = sorted(prompts, key=lambda p: hashlib.sha256(
            b"prior-visit-generation-v2\0" + p.encode("utf-8")).digest())
        if len(ordered) < 4:
            raise ValueError(f"Insufficient unique prompts in stratum {key}")
        selected.extend(prompts[p] for p in ordered[:4])
    if not selected:
        raise ValueError("No eligible held-out prompts")
    return selected


def classify_opening(text, current, prior):
    """Conservatively identify an explicit answer opening, with its exclusive end offset."""
    matches = []
    for label, value in (("current", current), ("prior", prior)):
        core = re.sub(r"^(?:an?|the)\s+", "", value.strip(), flags=re.IGNORECASE)
        match = re.match(r"^\s*(?:(?:an?|the)\s+)?" + re.escape(core) + r"(?!\w)",
                         text, flags=re.IGNORECASE)
        if match:
            matches.append((label, match.end()))
    return matches[0] if len(matches) == 1 else ("unclassified", None)


def decision(events, threshold, detail_end):
    first_halt = next((e["position"] for e in events if e["risk_score"] >= threshold), None)
    allowed = events if first_halt is None else events[:first_halt]
    return {
        "first_halt": first_halt,
        "released": "".join(e["text"] for e in allowed),
        "halt_by_value_end": None if detail_end is None else
        (first_halt is not None and first_halt <= detail_end),
    }


def compare_generation(row, events, probes):
    original, selected = events["original"], events["selected"]
    identity = ("position", "token_id", "text")
    if len(original) != len(selected) or any(
        any(left[k] != right[k] for k in identity) for left, right in zip(original, selected)
    ):
        raise ValueError("Frozen scorers received different generated tokens")
    text = "".join(e["text"] for e in original)
    label, end = classify_opening(text, row["current_value"], row["prior_value"])
    detail_end = None
    if end is not None:
        offset = 0
        for e in original:
            offset += len(e["text"])
            if offset >= end:
                detail_end = e["position"]
                break
    return {
        "source_row": row["id"], "group_id": row["group_id"], "layout_id": row["layout_id"],
        "chart_order": row["chart_order"], "prompt": row["prompt"], "current_value": row["current_value"],
        "prior_value": row["prior_value"], "generation": text, "opening": label,
        "value_end_position": detail_end,
        "tokens": [dict(position=a["position"], token_id=a["token_id"], text=a["text"],
                        original_risk=a["risk_score"], selected_risk=b["risk_score"])
                   for a, b in zip(original, selected, strict=True)],
        "policy": {k: decision(events[k], p["threshold"], detail_end) for k, p in probes.items()},
    }


def summarize(rows):
    summary = {}
    for label in ("current", "prior", "unclassified"):
        subset = [r for r in rows if r["opening"] == label]
        counts = {"prompts": len(subset)}
        for scorer in rows[0]["policy"]:
            policies = [r["policy"][scorer] for r in subset]
            counts[scorer] = {
                "halt_anywhere": sum(p["first_halt"] is not None for p in policies),
                "halt_by_value_end": None if label == "unclassified" else
                sum(p["halt_by_value_end"] for p in policies),
            }
        summary[label] = counts
    return summary


def heldout_rows(test_dir, selection_hash):
    manifest = parse_json((test_dir / "manifest.json").read_bytes())
    dataset = test_dir / "dataset.jsonl"
    if (manifest["selection_sha256"] != selection_hash or
            manifest["source_sha256"] != SOURCE_HASHES["test"] or
            manifest["sha256"]["dataset.jsonl"] != digest(dataset)):
        raise ValueError("Held-out data differ from the frozen primary evaluation")
    rows = [parse_json(line) for line in dataset.read_bytes().splitlines()]
    if rows != build_temporal_dataset("test"):
        raise ValueError("Expected the full canonical fresh test split")
    return rows


def add_source_scores(model, row, result, source, original_threshold):
    from eval.evaluate_source import cue_score
    from mech_interp.source_probe import score_prefixes

    tokens = result["tokens"]
    prefixes, text = [], ""
    for event in tokens:
        text += event["text"]
        prefixes.append(text)
    scored = score_prefixes(model, [row["prompt"]] * len(tokens), prefixes,
                            [e["original_risk"] for e in tokens], source)
    events, cue_events = [], []
    for token, evidence in zip(tokens, scored, strict=True):
        token["source_risk"] = evidence["risk_score"]
        token["source_match"] = evidence["match"]
        offset = evidence["match"]["start"] if evidence["match"] is not None else None
        cue_risk, reason = cue_score(row["prompt"], offset, token["original_risk"])
        token["cue_risk"], token["cue_reason"] = cue_risk, reason
        cue_events.append({"position": token["position"], "text": token["text"], "risk_score": cue_risk})
        events.append({"position": token["position"], "token_id": token["token_id"],
                       "text": token["text"], "risk_score": evidence["risk_score"]})
    result["policy"]["source"] = decision(events, source["threshold"], result["value_end_position"])
    result["policy"]["chronology-words"] = decision(cue_events, original_threshold, result["value_end_position"])
    return events


def generate(args, folder):
    import torch

    dev = args.development
    selection_file = dev / "selection.json"
    if digest(selection_file) != args.selection_sha256:
        raise ValueError("Selection digest differs from the frozen development run")
    selection = parse_json(selection_file.read_bytes())
    if digest(args.source_probe) != args.source_sha256:
        raise ValueError("Source-role probe differs from the registered freeze")
    source = parse_json(args.source_probe.read_bytes())
    rows = heldout_rows(args.test, args.selection_sha256)
    paths, probes = {}, {}
    for key, name in (("original", "frozen-v1"), ("selected", selection["selected"])):
        item = selection["models"][name]
        path = dev / item["file"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"Digest mismatch for {name}")
        paths[key] = path
        probes[key] = parse_json(path.read_bytes())
    p = probes["selected"]
    verify_model_source(p["model_id"], p["model_revision"], args.model_path)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ResidualModel(p["model_id"], p["model_revision"], p["layer"],
                          args.device, args.dtype, args.model_path)
    sample = sample_prompts(rows)
    results = []
    for i, row in enumerate(sample):
        events = {name: list(model.generate_tokens(row["prompt"], probe, 16))
                  for name, probe in probes.items()}
        result = compare_generation(row, events, probes)
        source_events = add_source_scores(model, row, result, source, probes["original"]["threshold"])
        results.append(result)
        if i == 0:
            trace = {"schema_version": "research-trace.v1", "model_id": p["model_id"],
                     "model_revision": p["model_revision"], "layer": p["layer"],
                     "probe_id": "sha256:" + digest(paths["selected"]), "source_row": row["id"],
                     "prompt": row["prompt"], "tokens": events["selected"]}
            trace["trace_id"] = trace_identifier(trace)
            write_json(folder / "trace.json", trace)
            export_wire(folder / "trace.json", paths["selected"], folder / "trace.wire.jsonl")
            source_trace = {**trace, "tokens": source_events, "probe_id": "sha256:" + args.source_sha256}
            del source_trace["trace_id"]
            source_trace["trace_id"] = trace_identifier(source_trace)
            validate_trace(source_trace)
            write_json(folder / "source.trace.json", source_trace)
            with (folder / "source.trace.wire.jsonl").open("wb") as stream:
                write_trace(scored_wire_records(source_trace, source_events,
                                                probe_family="research-source-role-v1"), stream)
    write_jsonl(folder / "generations.jsonl", results)
    write_json(folder / "metrics.json", {
        "selected": selection["selected"], "selection_sha256": args.selection_sha256,
        "source_probe_sha256": args.source_sha256, "test_dataset_sha256": digest(args.test / "dataset.jsonl"),
        "device": args.device, "dtype": args.dtype,
        "max_tokens": 16, "n": len(results), "opening_counts": summarize(results),
        "sha256": {path.name: digest(path) for path in sorted(folder.iterdir())},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True, help="completed fresh evaluation directory")
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--source-probe", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        stage = Path(tmp) / "completed"
        stage.mkdir()
        generate(args, stage)
        stage.rename(args.output)
    print(f"Wrote matched generation results to {args.output}")


if __name__ == "__main__":
    main()
