"""Fit the chart-source probe from an SHA-pinned, training/validation-only development bundle."""

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from eval.evaluate_temporal import load_development, rows_from, verified_activations
from eval.run import digest, write_json, write_jsonl
from mech_interp.model import ResidualModel
from mech_interp.probe import fit_probe, score
from mech_interp.snapshot import verify_model_source
from mech_interp.source_probe import MATCH_VERSION, candidate_prefix, extract_sources, score_cached
from mech_interp.temporal_probe import BASELINE_SHA256, THRESHOLD_RULE, threshold_for_validation, validation_counts


def train(development, selection_hash, model_path, stage):
    selection, candidates, rows = load_development(development, selection_hash)
    original = candidates["frozen-v1"]
    model_id, revision, layer = (original[k] for k in ("model_id", "model_revision", "layer"))
    verify_model_source(model_id, revision, model_path)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(selection["threads"])
    torch.manual_seed(selection["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ResidualModel(model_id, revision, layer, selection["device"], selection["dtype"], model_path)
    if model.hidden_size != original["hidden_size"]:
        raise ValueError("Published fallback and model widths differ")
    baseline_matrix = verified_activations(development / "activations.npz", rows, model.hidden_size)
    fallback = score(baseline_matrix, original)
    val_rows = [row for row in rows if row["split"] == "validation"]
    val_indices = np.array([i for i, row in enumerate(rows) if row["split"] == "validation"])
    logged_val = rows_from(development / "validation-predictions.jsonl")
    logged_risks = [row["scores"]["frozen-v1"] for row in logged_val]
    if not np.allclose(fallback[val_indices], logged_risks, rtol=0, atol=1e-10):
        raise ValueError("Recomputed fallback differs from published validation risks")
    fallback[val_indices] = logged_risks

    prefixes = [candidate_prefix(model, row) for row in rows]
    matches, indices, vectors = extract_sources(
        model, [row["prompt"] for row in rows], prefixes, selection["batch_size"])
    train_mask = np.array([rows[i]["split"] == "train" for i in indices], dtype=bool)
    matched_train = indices[train_mask]
    fitted = fit_probe(vectors[train_mask], [rows[i]["label"] for i in matched_train], l2=0.1)
    probe = {
        "schema_version": "source-role-probe.v1", "match_version": MATCH_VERSION,
        "model_id": model_id, "model_revision": revision, "layer": layer, "hidden_size": model.hidden_size,
        "positive_label": "unsupported", "fallback_probe_sha256": BASELINE_SHA256,
        "training_position": "post-block residual at the end of a unique matching chart suffix",
        "score_rule": "sigmoid(dot(source_residual, weights) + bias) on matches; original v1 risk otherwise",
        "threshold_rule": THRESHOLD_RULE, "training_rows_matched": len(matched_train),
        **fitted,
    }
    risks = score_cached(vectors, indices, fallback, probe)
    val_risks = risks[val_indices]
    probe["threshold"] = threshold_for_validation(val_rows, val_risks)
    validation = validation_counts(val_rows, val_risks, probe["threshold"])
    write_json(stage / "source-probe.json", probe)
    np.savez_compressed(stage / "source-activations.npz", row_ids=np.array([row["id"] for row in rows]),
                        matched_indices=indices, activations=vectors)
    predictions = []
    for i in val_indices:
        row = rows[i]
        predictions.append({
            **{k: row[k] for k in ("id", "label", "layout_id", "alternative_location")},
            "candidate_prefix": prefixes[i], "match": asdict(matches[i]) if matches[i] else None,
            "fallback_risk": float(fallback[i]), "risk_score": float(risks[i]),
        })
    write_jsonl(stage / "validation-predictions.jsonl", predictions)
    summary = {
        "schema_version": "source-role-development.v1", "selection_sha256": selection_hash,
        "probe_sha256": digest(stage / "source-probe.json"), "fallback_probe_sha256": BASELINE_SHA256,
        "threshold": probe["threshold"], "validation": validation,
        "coverage": {
            "train": {"matched": len(matched_train), "rows": len(rows) - len(val_rows)},
            "validation": {"matched": len(indices) - len(matched_train), "rows": len(val_rows)},
        },
        "device": selection["device"], "dtype": selection["dtype"], "batch_size": selection["batch_size"],
        "model_id": model_id, "model_revision": revision, "layer": layer,
        "sha256": {p.name: digest(p) for p in sorted(stage.iterdir())},
    }
    write_json(stage / "source-development.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".source-dev-", dir=args.output.parent) as tmp:
        stage = Path(tmp) / "completed"
        stage.mkdir()
        summary = train(args.development, args.selection_sha256, args.model_path, stage)
        stage.rename(args.output)
    print(f"Wrote {args.output}, probe SHA-256 {summary['probe_sha256']}, threshold {summary['threshold']}")


if __name__ == "__main__":
    main()
