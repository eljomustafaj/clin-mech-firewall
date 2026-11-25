"""Reproduce the pilot dataset, frozen activations, fitted probe, evaluation and real trace."""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
import platform
import re
import tempfile

import numpy as np
import torch
import transformers

from eval.export_wire import export_wire
from mech_interp.data import build_dataset
from mech_interp.model import ResidualModel
from mech_interp.probe import choose_threshold, classification_metrics, fit_probe, score
from mech_interp.snapshot import DEFAULT_MODEL, DEFAULT_REVISION, verify_model_source


def write_json(path, payload):
    path.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
                      + "\n").encode("utf-8"))


def write_jsonl(path, rows):
    with path.open("wb") as out:
        for row in rows:
            out.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_identifier(trace_payload):
    """Identify a complete archive by its metadata and ordered, scored model outputs."""
    canonical = json.dumps(trace_payload, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "greedy-" + hashlib.sha256(canonical).hexdigest()


def split_indices(rows):
    """Reject cross-split groups and broken pairs before constructing training arrays."""
    groups, pairs, splits, ids = {}, defaultdict(list), defaultdict(list), set()
    for i, row in enumerate(rows):
        group, split = row["group_id"], row["split"]
        if row["id"] in ids or groups.setdefault(group, split) != split:
            raise ValueError("Duplicate row ID or a group crosses split boundaries")
        ids.add(row["id"])
        pairs[row["pair_id"]].append((group, row["label"]))
        splits[split].append(i)
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test splits")
    for members in pairs.values():
        if (len(members) != 2 or len({row[0] for row in members}) != 1
                or {row[1] for row in members} != {0, 1}):
            raise ValueError("Each pair must contain a supported and an unsupported row in one group")
    return {split: np.array(indices) for split, indices in splits.items()}


def pair_accuracy(rows, scores):
    by_pair = defaultdict(dict)
    for row, risk in zip(rows, scores, strict=True):
        by_pair[row["pair_id"]][row["label"]] = risk
    credit = [float(values[1] > values[0]) + 0.5 * float(values[1] == values[0])
              for values in by_pair.values()]
    return float(np.mean(credit))


def grouped_intervals(rows, scores, threshold, replicates=1000):
    """Percentile bootstrap across counterbalanced groups, conditional on the held-out layout."""
    by_group = defaultdict(list)
    for i, row in enumerate(rows):
        by_group[row["group_id"]].append(i)
    groups = list(by_group.values())
    y = np.array([row["label"] for row in rows])
    rng = np.random.default_rng(1729)
    samples = defaultdict(list)
    for _ in range(replicates):
        selection = rng.integers(len(groups), size=len(groups))
        indices = np.concatenate([groups[i] for i in selection])
        metrics = classification_metrics(y[indices], scores[indices], threshold)
        for name in ("auroc", "f1", "false_positive_rate"):
            samples[name].append(metrics[name])
    return {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in samples.items()}


def run(args, folder):
    torch.set_num_threads(args.threads)
    torch.manual_seed(1729)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu --dtype float32.")
    if args.device == "cpu" and args.dtype == "float16":
        raise ValueError("Use --dtype float32 for CPU inference")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    verify_model_source(args.model, args.revision, args.model_path)
    rows = list(build_dataset())
    splits = split_indices(rows)
    labels = np.array([row["label"] for row in rows])
    write_jsonl(folder / "dataset.jsonl", rows)
    model = ResidualModel(args.model, args.revision, args.layer, args.device, args.dtype,
                          args.model_path)
    print(f"Extracting {len(rows)} claim-end activations at block {args.layer}", flush=True)
    x = model.encode_rows(rows, args.batch_size)
    np.savez_compressed(folder / "activations.npz", row_ids=np.array([r["id"] for r in rows]),
                        activations=x)
    train, valid, test = (splits[s] for s in ("train", "validation", "test"))
    fitted = fit_probe(x[train], labels[train], args.l2)
    scores = score(x, fitted)
    threshold = choose_threshold(labels[valid], scores[valid])
    probe = {
        "schema_version": "linear-probe.v1", "model_id": args.model, "model_revision": args.revision,
        "layer": args.layer, "hidden_size": model.hidden_size, "positive_label": "unsupported",
        "activation": "post-block residual at consumed candidate token, before final normalization",
        "training_position": "last token overlapping the target span in a teacher-forced completion",
        "formula": "sigmoid(dot(activation, weights) + bias)", "threshold": threshold,
        "threshold_rule": "score >= threshold; maximize validation F1, then minimize FP, then largest threshold",
        **fitted,
    }
    write_json(folder / "probe.json", probe)
    probe_id = "sha256:" + digest(folder / "probe.json")
    # A simple whole-note lexical check: a prior-visit mention also matches.
    lexical = np.array([float(row["target_text"].casefold() not in row["prompt"].casefold())
                        for row in rows])
    predictions = []
    for i, row in enumerate(rows):
        prediction = int(scores[i] >= threshold)
        predictions.append({
            "id": row["id"], "pair_id": row["pair_id"], "group_id": row["group_id"],
            "split": row["split"], "label": int(labels[i]), "risk_score": float(scores[i]),
            "predicted_label": prediction, "lexical_risk": float(lexical[i]),
        })
    write_jsonl(folder / "predictions.jsonl", predictions)

    test_rows = [rows[i] for i in test]
    subgroup_metrics = {}
    for location in sorted({row["alternative_location"] for row in test_rows}):
        indices = [i for i in test if rows[i]["alternative_location"] == location]
        subgroup_metrics[location] = {
            "probe": classification_metrics(labels[indices], scores[indices], threshold),
            "lexical": classification_metrics(labels[indices], lexical[indices], 0.5),
        }
    metrics = {
        "schema_version": "probe-evaluation.v1", "probe_id": probe_id,
        "unit": "one teacher-forced target per candidate completion",
        "positive_label": "unsupported",
        "splits": {s: {"rows": len(idx), "pairs": len({rows[i]["pair_id"] for i in idx}),
                       "groups": len({rows[i]["group_id"] for i in idx})} for s, idx in splits.items()},
        "threshold": threshold,
        "validation": classification_metrics(labels[valid], scores[valid], threshold),
        "test": classification_metrics(labels[test], scores[test], threshold),
        "test_pair_ranking_accuracy": pair_accuracy(test_rows, scores[test]),
        "test_group_bootstrap_95pct": grouped_intervals(test_rows, scores[test], threshold),
        "bootstrap": {"groups_sampled_with_replacement": len({r["group_id"] for r in test_rows}),
                      "replicates": 1000, "seed": 1729, "conditional_on": "one fixed test layout"},
        "test_lexical_baseline": classification_metrics(labels[test], lexical[test], 0.5),
        "test_by_alternative_location": subgroup_metrics,
        "lexical_rule": "unsupported if case-folded target span is absent from the entire prompt",
        "error_examples": {},
    }
    for name, label in (("false_alarm", 0), ("missed_fabrication", 1)):
        errors = [p for p in predictions if p["split"] == "test" and p["label"] == label
                  and p["label"] != p["predicted_label"]]
        if errors:
            # Select deterministically by largest error margin, retaining the row ID for auditing.
            metrics["error_examples"][name] = max(errors, key=lambda p: abs(p["risk_score"] - label))
    write_json(folder / "metrics.json", metrics)

    if args.trace_row is None:
        trace_source = next(row for row in test_rows if row["label"] == 0)
    else:
        trace_source = next((row for row in test_rows if row["id"] == args.trace_row), None)
    if trace_source is None:
        raise ValueError("--trace-row must identify a row from the test split")
    tokens = list(model.generate_tokens(trace_source["prompt"], probe, args.trace_tokens))
    trace = {
        "schema_version": "research-trace.v1",
        "model_id": args.model, "model_revision": args.revision, "layer": args.layer,
        "probe_id": probe_id, "prompt": trace_source["prompt"], "tokens": tokens,
        "source_row": trace_source["id"], "generation": "greedy; raw prompt; no chat template",
    }
    trace["trace_id"] = trace_identifier(trace)
    write_json(folder / "trace.json", trace)
    export_wire(folder / "trace.json", folder / "probe.json", folder / "trace.wire.jsonl")
    manifest = {
        "schema_version": "baseline-run.v1", "model_id": args.model, "model_revision": args.revision,
        "layer": args.layer, "device": args.device, "dtype": args.dtype, "batch_size": args.batch_size,
        "l2": args.l2, "threads": args.threads, "max_new_tokens": args.trace_tokens,
        "trace_source": trace_source["id"], "attention": "eager", "seed": 1729,
        "format": "raw text; joint tokenization; no added special tokens; no truncation before target",
        "python": platform.python_version(), "torch": str(torch.__version__),
        "transformers": transformers.__version__, "numpy": np.__version__,
        "tokenizers": version("tokenizers"), "safetensors": version("safetensors"),
        "huggingface_hub": version("huggingface-hub"),
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
        "sha256": {p.name: digest(p) for p in sorted(folder.iterdir())},
    }
    write_json(folder / "manifest.json", manifest)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="immutable 40-character Hub commit")
    parser.add_argument("--model-path", type=Path, help="offline snapshot of that model and revision")
    parser.add_argument("--layer", type=int, default=12, help="zero-based, non-final transformer block")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--l2", type=float, default=0.1)
    parser.add_argument("--trace-tokens", type=int, default=12)
    parser.add_argument("--trace-row", help="test prompt ID; defaults to the first supported test example")
    parser.add_argument("--output", type=Path, required=True, help="new directory for completed artifacts")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("--revision must be a full model commit")
    if min(args.threads, args.batch_size, args.trace_tokens) < 1:
        parser.error("thread count, batch size and maximum generated tokens must be positive")
    if args.output.exists():
        parser.error(f"Output path already exists: {args.output}. Select a new run directory.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Publish all artifacts together. An interrupted run leaves a previous result intact.
    with tempfile.TemporaryDirectory(prefix=".baseline-", dir=args.output.parent) as temp:
        stage = Path(temp) / "artifacts"
        stage.mkdir()
        metrics = run(args, stage)
        stage.rename(args.output)
    print(f"Wrote {args.output}; held-out AUROC={metrics['test']['auroc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
