"""Audit the separately registered source-role probe against the frozen temporal benchmark."""

import argparse
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from eval.evaluate_temporal import (
    V1_SHA256, audit, audited_threshold, load_development, paired_transitions, require, rows_from,
    sha, summarize,
)
from eval.validate_trace import parse_json


PRIOR_PATTERN = re.compile(r"\b(?:prior|previous|previously|earlier|older|past|last|historical|archived|closed|preceding|former)\b", re.I)
CURRENT_PATTERN = re.compile(r"\b(?:current|today|active|present|latest|now|ongoing|confirmed|this encounter|this visit|most recent|in progress)\b", re.I)
SOURCE = "source-role"
TEXT = "chronology-words"


def cue_score(prompt, start, frozen_risk):
    """Use only the prefix of the matching chart line and the frozen fallback risk."""
    if start is None:
        return frozen_risk, "unmatched"
    require(type(start) is int and 0 <= start < len(prompt), "match cursor outside prompt")
    line_prefix = prompt[prompt.rfind("\n", 0, start) + 1:start]
    prior = PRIOR_PATTERN.search(line_prefix) is not None
    current = CURRENT_PATTERN.search(line_prefix) is not None
    if prior and not current:
        return 1.0, "prior-cue"
    if current and not prior:
        return 0.0, "current-cue"
    return frozen_risk, "conflicting-cues" if prior else "no-cues"


def match_record(match):
    return {"start": match.start, "end": match.end, "words": match.words} if match else None


def source_predictions(rows, raw_preds, prefixes, indices, vectors, fallback, source_probe, raw_probes, primary):
    from mech_interp.source_probe import locate_source, score_cached

    require(len({len(rows), len(raw_preds), len(prefixes), len(fallback)}) == 1, "unequal prediction input lengths")
    require(np.array_equal(fallback, [r["scores"]["frozen-v1"] for r in raw_preds]), "incorrect v1 fallback")
    matches = [locate_source(row["prompt"], prefix) for row, prefix in zip(rows, prefixes, strict=True)]
    found = [i for i, match in enumerate(matches) if match is not None]
    require(np.array_equal(indices, found), "cached indices differ from independently located matches")
    require(vectors.shape == (len(found), source_probe["hidden_size"]) and np.isfinite(vectors).all(),
            "invalid source activation cache")
    risks = score_cached(vectors, indices, fallback, source_probe)
    require(len(risks) == len(rows), "wrong count of hybrid risks")
    # Re-evaluate raw logits and all fallback decisions without using the scorer's implementation.
    independent = fallback.copy()
    for i, vector in zip(indices, vectors, strict=True):
        z = math.fsum(float(x) * w for x, w in zip(vector, source_probe["weights"], strict=True))
        independent[i] = .5 + .5 * math.tanh((z + source_probe["bias"]) / 2)
    require(np.allclose(independent, risks, rtol=0, atol=1e-10), "source scores disagree with raw activation formula")
    predictions = []
    for i, (row, raw, match) in enumerate(zip(rows, raw_preds, matches, strict=True)):
        require(row["id"] == raw["id"], "raw baseline row order differs")
        text_score, reason = cue_score(row["prompt"], match.start if match else None, fallback[i])
        scores = {m: raw["scores"][m] for m in ("frozen-v1", primary)}
        scores.update({SOURCE: float(risks[i]), TEXT: float(text_score)})
        flags = {m: raw["flags"][m] for m in ("frozen-v1", primary)}
        flags.update({SOURCE: int(risks[i] >= source_probe["threshold"]),
                      TEXT: int(text_score >= raw_probes["frozen-v1"]["threshold"])})
        predictions.append({**{k: raw[k] for k in ("id", "pair_id", "group_id", "layout_id",
                                                     "field_id", "chart_order", "label", "alternative_location")},
                            "candidate_prefix": prefixes[i], "match": match_record(match),
                            "cue": reason, "scores": scores, "flags": flags})
    return predictions


def decision_counts(records, thresholds):
    # Matched/fallback strata can contain only one class. Aggregate denominators stay explicit.
    if {r["label"] for r in records} == {0, 1}:
        return summarize(records, thresholds)
    result = {}
    for method in thresholds:
        groups = {"supported_false_alarm": [r for r in records if not r["label"]],
                  "absent_catch": [r for r in records if r["label"] and r["alternative_location"] == "absent"],
                  "prior_catch": [r for r in records if r["label"] and r["alternative_location"] == "prior"]}
        result[method] = {"n": len(records)}
        for name, members in groups.items():
            hits = sum(r["flags"][method] for r in members)
            result[method][name] = {"flagged": hits, "total": len(members),
                                    "rate": hits / len(members) if members else None}
    return result


def source_metrics(rows, thresholds, primary):
    by_layout, by_order, joint, coverage = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        by_layout[r["layout_id"]].append(r)
        by_order[r["chart_order"]].append(r)
        joint[r["layout_id"], r["chart_order"]].append(r)
        coverage["unmatched" if r["match"] is None else "matched"].append(r)
    error_ids = {method: {"supported_false_alarm": [], "prior_missed": [], "absent_missed": []}
                 for method in (SOURCE, TEXT)}
    for r in rows:
        for method, errors in error_ids.items():
            if r["flags"][method] != r["label"]:
                category = f"{r['alternative_location']}_missed" if r["label"] else "supported_false_alarm"
                errors[category].append(r["id"])
    return {
        "schema_version": "source-role-evaluation.v1", "source_method": SOURCE, "raw_primary": primary,
        "rows": len(rows), "thresholds": thresholds,
        "overall": decision_counts(rows, thresholds), "coverage": {k: len(v) for k, v in coverage.items()},
        "cue_counts": dict(Counter(r["cue"] for r in rows)),
        "by_coverage": {k: decision_counts(v, thresholds) for k, v in coverage.items()},
        "by_layout": {k: decision_counts(v, thresholds) for k, v in by_layout.items()},
        "by_chart_order": {k: decision_counts(v, thresholds) for k, v in by_order.items()},
        "by_layout_and_order": [{"layout_id": k[0], "chart_order": k[1], "methods": decision_counts(v, thresholds),
                                 "cue_counts": dict(Counter(r["cue"] for r in v))}
                                for k, v in sorted(joint.items())],
        "paired": {ref: {method: paired_transitions(rows, ref, method) for method in (SOURCE, TEXT) if method != ref}
                   for ref in ("frozen-v1", primary, TEXT)},
        "paired_by_layout_and_order": [{"layout_id": k[0], "chart_order": k[1],
                                        "methods": {m: paired_transitions(v, "frozen-v1", m, False)
                                                    for m in (SOURCE, TEXT)}} for k, v in sorted(joint.items())],
        "error_row_ids": error_ids,
    }


def load_source_development(folder, probe_sha, raw_dev, selection_sha, selection, probes, development):
    """Recompute the operating point using frozen validation scores and source states."""
    from mech_interp.source_probe import MATCH_VERSION, locate_source, score_cached

    path = folder / "source-probe.json"
    require(sha(path) == probe_sha, "source probe differs from the registered freeze")
    probe = parse_json(path.read_bytes())
    summary = parse_json((folder / "source-development.json").read_bytes())
    require(summary["probe_sha256"] == probe_sha and summary["selection_sha256"] == selection_sha,
            "source development refers to a different freeze")
    for name in ("source-probe.json", "validation-predictions.jsonl", "source-activations.npz"):
        require(sha(folder / name) == summary["sha256"][name], f"source development {name} SHA mismatch")
    base = probes["frozen-v1"]
    require(all(probe[k] == base[k] for k in ("model_id", "model_revision", "hidden_size", "layer")),
            "source role and fallback model metadata differ")
    require(probe["schema_version"] == "source-role-probe.v1" and probe["match_version"] == MATCH_VERSION
            and probe["fallback_probe_sha256"] == V1_SHA256, "unexpected source scorer")
    val_rows = [r for r in development if r["split"] == "validation"]
    val_positions = [i for i, r in enumerate(development) if r["split"] == "validation"]
    logged = rows_from(folder / "validation-predictions.jsonl")
    raw = rows_from(raw_dev / "validation-predictions.jsonl")
    require([r["id"] for r in logged] == [r["id"] for r in val_rows] == [r["id"] for r in raw],
            "validation IDs differ")
    fallback = np.array([r["scores"]["frozen-v1"] for r in raw])
    require(np.allclose([r["fallback_risk"] for r in logged], fallback, atol=1e-10, rtol=0),
            "source validation fallback differs from v1")
    with np.load(folder / "source-activations.npz", allow_pickle=False) as cache:
        require(cache["row_ids"].tolist() == [r["id"] for r in development], "development cache row IDs differ")
        keys, states = cache["matched_indices"], cache["activations"]
    require(keys.dtype.kind in "iu" and np.all(np.diff(keys) > 0) and
            np.all((0 <= keys) & (keys < len(development))) and states.shape == (len(keys), base["hidden_size"]),
            "malformed source development cache")
    by_index = {index: j for j, index in enumerate(keys)}
    matched, vectors = [], []
    for i, (row, saved, global_index) in enumerate(zip(val_rows, logged, val_positions, strict=True)):
        hit = locate_source(row["prompt"], saved["candidate_prefix"])
        require(saved["match"] == match_record(hit) and (hit is not None) == (global_index in by_index),
                "validation evidence lookup differs")
        if hit is not None:
            matched.append(i)
            vectors.append(by_index[global_index])
    risks = score_cached(states[vectors], np.array(matched, dtype=int), fallback, probe)
    stored = np.array([r["risk_score"] for r in logged])
    require(np.allclose(risks, stored, rtol=0, atol=1e-10), "source validation risk mismatch")
    _, threshold = audited_threshold(val_rows, stored)
    require(probe["threshold"] == threshold, "source threshold differs from independent validation choice")
    return probe


def write_evaluation(folder, test, model_path, selection, probes, source, probe_sha, selection_sha):
    import torch
    from eval.run import write_json, write_jsonl
    from mech_interp.model import ResidualModel
    from mech_interp.snapshot import verify_model_source
    from mech_interp.source_probe import candidate_prefix, extract_sources

    base = probes["frozen-v1"]
    verify_model_source(base["model_id"], base["model_revision"], model_path)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(selection["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ResidualModel(base["model_id"], base["model_revision"], base["layer"],
                          selection["device"], selection["dtype"], model_path)
    rows, raw = rows_from(test / "dataset.jsonl"), rows_from(test / "predictions.jsonl")
    prefixes = [candidate_prefix(model, row) for row in rows]
    _, indices, vectors = extract_sources(model, [r["prompt"] for r in rows], prefixes, selection["batch_size"])
    fallback = np.array([p["scores"]["frozen-v1"] for p in raw])
    preds = source_predictions(rows, raw, prefixes, indices, vectors, fallback, source, probes, selection["selected"])
    np.savez_compressed(folder / "source-activations.npz", row_ids=np.array([r["id"] for r in rows]),
                        matched_indices=indices, activations=vectors)
    write_jsonl(folder / "predictions.jsonl", preds)
    thresholds = {k: probes[k]["threshold"] for k in ("frozen-v1", selection["selected"])}
    thresholds.update({SOURCE: source["threshold"], TEXT: base["threshold"]})
    write_json(folder / "metrics.json", source_metrics(preds, thresholds, selection["selected"]))
    write_json(folder / "manifest.json", {
        "schema_version": "source-role-test.v1", "selection_sha256": selection_sha, "source_probe_sha256": probe_sha,
        "raw_test_sha256": {k: sha(test / k) for k in ("dataset.jsonl", "predictions.jsonl", "manifest.json")},
        "model_id": base["model_id"], "model_revision": base["model_revision"], "layer": base["layer"],
        "device": selection["device"], "dtype": selection["dtype"], "batch_size": selection["batch_size"],
        "sha256": {p.name: sha(p) for p in sorted(folder.iterdir())},
    })


def audit_source(folder, test, selection, probes, source, probe_sha, selection_sha):
    manifest = parse_json((folder / "manifest.json").read_bytes())
    require(manifest["source_probe_sha256"] == probe_sha and manifest["selection_sha256"] == selection_sha,
            "source test freeze mismatch")
    for name in ("source-activations.npz", "predictions.jsonl", "metrics.json"):
        require(sha(folder / name) == manifest["sha256"][name], f"source test {name} SHA mismatch")
    for name in ("dataset.jsonl", "predictions.jsonl", "manifest.json"):
        require(sha(test / name) == manifest["raw_test_sha256"][name], "raw test archive changed")
    rows, raw, logged = rows_from(test / "dataset.jsonl"), rows_from(test / "predictions.jsonl"), rows_from(folder / "predictions.jsonl")
    with np.load(folder / "source-activations.npz", allow_pickle=False) as cache:
        require(cache["row_ids"].tolist() == [r["id"] for r in rows], "test cache row IDs differ")
        indices, states = cache["matched_indices"], cache["activations"]
    prefixes = [p["candidate_prefix"] for p in logged]
    for row, prefix in zip(rows, prefixes, strict=True):
        require(row["completion"].startswith(prefix) and len(prefix) >= row["target_end"],
                "cached prefix does not cover scored candidate")
    expected = source_predictions(rows, raw, prefixes, indices, states,
                                  np.array([p["scores"]["frozen-v1"] for p in raw]), source, probes, selection["selected"])
    for old, new in zip(logged, expected, strict=True):
        require(old.keys() == new.keys() and all(old[k] == new[k] for k in old if k != "scores"),
                "source row metadata, evidence position or decisions differ")
        require(old["scores"].keys() == new["scores"].keys() and
                all(abs(old["scores"][k] - new["scores"][k]) < 1e-10 for k in old["scores"]),
                "cached source scores do not reproduce")
    thresholds = {k: probes[k]["threshold"] for k in ("frozen-v1", selection["selected"])}
    thresholds.update({SOURCE: source["threshold"], TEXT: probes["frozen-v1"]["threshold"]})
    require(source_metrics(logged, thresholds, selection["selected"]) == parse_json((folder / "metrics.json").read_bytes()),
            "reported source metrics differ from audited predictions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--source-development", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True, help="registered source-probe.json SHA-256")
    parser.add_argument("--test", required=True, type=Path, help="completed frozen raw evaluation")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    selection, probes, dev = load_development(args.development, args.selection_sha256)
    source = load_source_development(args.source_development, args.source_sha256, args.development,
                                     args.selection_sha256, selection, probes, dev)
    audit(args.test, args.development, args.selection_sha256, selection, probes, dev)
    params = (args.test, selection, probes, source, args.source_sha256, args.selection_sha256)
    if args.audit_only:
        audit_source(args.output, *params)
    else:
        if args.output.exists():
            parser.error("Choose a new output directory or --audit-only")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
            stage = Path(temp) / "source-evaluation"
            stage.mkdir()
            write_evaluation(stage, args.test, args.model_path, *params[1:])
            audit_source(stage, *params)
            stage.rename(args.output)
    print(f"Verified source-role evaluation at {args.output}")


if __name__ == "__main__":
    main()
