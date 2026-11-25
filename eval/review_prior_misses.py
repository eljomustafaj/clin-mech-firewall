"""Re-score the original chart tests that motivated the prior-visit follow-up."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from eval.evaluate_source import cue_score
from eval.run import digest, write_json
from eval.validate_trace import parse_json
from mech_interp.model import ResidualModel
from mech_interp.probe import score
from mech_interp.snapshot import verify_model_source
from mech_interp.source_probe import candidate_prefix, score_prefixes
from mech_interp.temporal_probe import BASELINE_SHA256


ORIGINAL_MANIFEST = "c789cfc0e9cdea3492ef433cfa9f3d0c78e27f5afc67721f7b5b18f76f739dc5"


def checked_json(path, expected_sha):
    if digest(path) != expected_sha:
        raise ValueError(f"Unexpected SHA-256 for {path.name}")
    return parse_json(path.read_bytes())


def count_decisions(rows, methods):
    categories = {
        "prior_caught": [r for r in rows if r["label"] == 1 and r["condition"] == "prior"],
        "absent_caught": [r for r in rows if r["label"] == 1 and r["condition"] == "absent"],
        "supported_flagged": [r for r in rows if r["label"] == 0],
    }
    return {name: {key: {"flagged": sum(r["flagged"][name] for r in cases), "total": len(cases)}
                   for key, cases in categories.items()} for name in methods}


def review(args):
    baseline, development = args.baseline, args.development
    selection = checked_json(development / "selection.json", args.selection_sha256)
    source_probe = checked_json(args.source_probe, args.source_sha256)
    manifest = checked_json(baseline / "manifest.json", ORIGINAL_MANIFEST)
    for name in ("dataset.jsonl", "activations.npz", "predictions.jsonl", "probe.json"):
        if digest(baseline / name) != manifest["sha256"][name]:
            raise ValueError(f"Original baseline digest mismatch: {name}")
    probes = {}
    for name in ("frozen-v1", selection["selected"]):
        entry = selection["models"][name]
        probes[name] = checked_json(development / entry["file"], entry["sha256"])
    if selection["models"]["frozen-v1"]["sha256"] != BASELINE_SHA256:
        raise ValueError("Original comparison probe was replaced")

    rows = [parse_json(line) for line in (baseline / "dataset.jsonl").read_bytes().splitlines()]
    predictions = [parse_json(line) for line in (baseline / "predictions.jsonl").read_bytes().splitlines()]
    ids = [r["id"] for r in rows]
    take = [i for i, row in enumerate(rows) if row["split"] == "test"]
    with np.load(baseline / "activations.npz", allow_pickle=False) as archive:
        if archive["row_ids"].tolist() != ids or [r["id"] for r in predictions] != ids:
            raise ValueError("Baseline rows are misaligned")
        x = archive["activations"][take]
    rows = [rows[i] for i in take]
    risks = {name: score(x, p) for name, p in probes.items()}
    if not np.allclose(risks["frozen-v1"], [predictions[i]["risk_score"] for i in take],
                       rtol=0, atol=1e-10):
        raise ValueError("Fallback scores differ from the originally published predictions")

    verify_model_source(manifest["model_id"], manifest["model_revision"], args.model_path)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(manifest["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ResidualModel(manifest["model_id"], manifest["model_revision"], manifest["layer"],
                          args.device, args.dtype, args.model_path)
    prefixes = [candidate_prefix(model, row) for row in rows]
    evidence = score_prefixes(model, [r["prompt"] for r in rows], prefixes, risks["frozen-v1"],
                              source_probe, manifest["batch_size"])
    risks["source-role"] = [info["risk_score"] for info in evidence]
    cues = [cue_score(row["prompt"], info["match"]["start"] if info["match"] else None, fallback)
            for row, info, fallback in zip(rows, evidence, risks["frozen-v1"], strict=True)]
    risks["chronology-words"] = [cue[0] for cue in cues]
    thresholds = {name: probe["threshold"] for name, probe in probes.items()}
    thresholds["source-role"] = source_probe["threshold"]
    thresholds["chronology-words"] = probes["frozen-v1"]["threshold"]
    cases = []
    for i, row in enumerate(rows):
        match = evidence[i]["match"]
        cases.append({
            "id": row["id"], "label": row["label"], "condition": row["alternative_location"],
            "current": row["current_value"], "earlier": row["prior_value"],
            "target": row["target_text"], "candidate_prefix": prefixes[i], "match": match, "cue": cues[i][1],
            "matched_text": row["prompt"][match["start"]:match["end"]] if match else None,
            "scores": {key: float(values[i]) for key, values in risks.items()},
            "flagged": {key: bool(values[i] >= thresholds[key]) for key, values in risks.items()},
        })
    missed = [r for r in cases if r["label"] and r["condition"] == "prior" and not r["flagged"]["frozen-v1"]]
    return {
        "scope": "Known original holdout; failures motivated this study.",
        "original_manifest_sha256": ORIGINAL_MANIFEST, "selection_sha256": args.selection_sha256,
        "source_probe_sha256": args.source_sha256, "device": args.device, "dtype": args.dtype,
        "thresholds": thresholds, "counts": count_decisions(cases, thresholds),
        "source_matched": sum(r["match"] is not None for r in cases),
        "original_prior_misses": len(missed),
        "original_prior_misses_recovered": sum(r["flagged"]["source-role"] for r in missed),
        "word_comparator_original_prior_misses_recovered": sum(r["flagged"]["chronology-words"] for r in missed),
        "original_test_rows": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--source-probe", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    result = review(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(result["counts"])


if __name__ == "__main__":
    main()
