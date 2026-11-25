# Mechanistic Interpretability & Hallucination Firewall for Clinical AI

`clin-mech-firewall` is a research project investigating whether an open-weight model’s internal activations can flag fabricated clinical details as it generates them. The first milestone is a single clinical prompt whose real token-by-token risk scores can be replayed through a firewall and inspected in a browser. Clinical accuracy, intervention effects, and latency are open questions.

## Task 1 — Research baseline (owner: eljomustafaj)

**Completed.** The baseline uses a linear probe trained on block 12 of Qwen2.5-0.5B-Instruct using 160 candidate answers to fictional chart questions. A separate 40 answers set the decision threshold. The final 40 answers came from 10 matched groups on a held-out chart layout. Each group pairs a supported detail with a substitution and reverses the values to balance answer wording. [Dataset and split design](data/README.md)

| Held-out scorer | AUROC | F1 | Unsupported answers caught | Supported answers flagged |
| --- | ---: | ---: | ---: | ---: |
| Residual-stream linear probe | 0.990 | 0.667 | 10/20 | 0/20 |
| Flag a term absent from the whole prompt | 0.750 | 0.667 | 10/20 | 0/20 |

The probe ranked unsupported answers above their matched supported answers in 19 of 20 chart pairs. At the validation-selected threshold of `0.9552176915681485`, it caught every substitution absent from the note and missed every substitution copied from a prior visit. For example, one chart recorded a blood sample today and a urine sample previously. The unsupported answer “a urine sample” scored `0.4964` and passed. These results measure association on templated chart questions. Clinical generalization, latency and causal steering require separate experiments. [Metrics](artifacts/baseline/metrics.json) · [Per-answer predictions](artifacts/baseline/predictions.jsonl)

After installing the [research dependencies](docs/RESEARCH.md), this command reproduces the evaluation, refits the weights, and generates the real trace from the pinned model. To inspect the published results immediately, the second command needs only Python.

```sh
python -m eval.run --device cuda --dtype float16 --batch-size 8 --output runs/reproduction
python -m eval.report artifacts/baseline
```

## Task 2 — Clad owns the streaming firewall and visualizer

**Completed.** Clad verified the measured trace in Chrome and through the separate HTTP response. The firewall withheld token 0, `" by"`, and released zero bytes. [Browser capture](artifacts/replay-v1/browser.png) · [Decision telemetry](artifacts/replay-v1/telemetry.jsonl) · [Replay guide and tests](docs/REPLAY.md)

Start by agreeing on a versioned event schema with eljomustafaj, before either owner builds against it. Define how a trace identifies the model and probe, orders tokens, records a risk score, and links a policy decision to a token. Allow sparse autoencoder feature IDs to be absent while the linear probe is the only scorer.

Build the first vertical slice through a Python inference worker, a Rust streaming proxy, and a TypeScript replay view. The proxy should apply a configurable `ALLOW` or `HALT` decision before releasing each token and stream the decision to the view. Record the triggering token in telemetry even when it is withheld from the response.

**First handoff.** Replay the trace from Task 1 through the proxy. Use a threshold that exercises `HALT`, verify that the triggering token never reaches the response, and show its score and decision at the same token position in the browser. Save the replay command and integration test so both owners can reproduce the result.

## Research and replay handoff

**Source-role scorer frozen for fresh evaluation.** At its validation-selected threshold of `0.9412578522854479`, it caught 6/96 prior substitutions, caught 95/96 absent substitutions, and flagged 4/192 supported answers. A rule using explicit chronology words on the same matched text caught 96/96 prior substitutions and flagged 0/192 supported answers. Both go into the fresh evaluation alongside the original and recalibrated probe. The source probe SHA-256 is `8d0f4e8d359affa07be74cb3a2e8c564296c21f654861068530833f9d436358c`. [Frozen source probe](artifacts/temporal-v2/source-dev/source-probe.json) · [Development record](artifacts/temporal-v2/source-dev/source-development.json)

**eljomustafaj → Clad · Prior-visit follow-up.** I read your measured replay update on `clad/measured-handoff`. I am comparing the original scorer with recalibration and two new linear probes. The primary test uses unseen chart layouts and fields, and a separate generation check will measure early halts on supported answers. I will post the winning development configuration before scoring the fresh test cases. [Frozen experiment rules](docs/PRIOR_VISIT_PROTOCOL.md)

**eljomustafaj → Clad · Answer-token probe frozen.** Validation selected the original weights at a higher threshold of `0.9740611507233753`. That choice cannot increase recall over v1. I am separately testing a source-role probe that reads the model state at the candidate’s matching occurrence in the chart. Both methods will be frozen before any fresh scoring. The answer-token selection digest is `6b7dc3b52e77721705bd5fe4a9d7e2c6321783c2ce450f84605b9f74ec01b6ae`; the sealed test source digest is `176d849ee2504d425d604bba6037d433be2ad0e5295aa3eda75c037d6ecaaa20`. [Selection](artifacts/temporal-v2/dev/selection.json) · [Data card](data/temporal_README.md)

**eljomustafaj → Clad · Measured trace delivered and replayed.** The chart said the patient arrived by bus today and by car at the previous visit. The model generated ` by car.` The first token ` by` scored `0.965618595746429`, so the proxy halted at index 0 and released zero bytes. The token ` car` scored `0.9019257519510192`, below the same threshold. This trace shows why the token that triggers a halt can differ from the factual error. The browser displayed the same token, score and decision. [Wire input](artifacts/baseline/trace.wire.jsonl) · [Prompt and token IDs](artifacts/baseline/trace.json)

The model revision is `7ae557604adf67be50417f59c2c2f167def9a775`. The probe revision is `sha256:3c7bd6c3ab4547618a013f4aae07e1ad69708ca151064ba3b8da8b230d945350`. A second GPU run reproduced all eight output files byte for byte. A separate model hook reproduced the 12 generated tokens and their scores. [Probe weights](artifacts/baseline/probe.json) · [Run manifest](artifacts/baseline/manifest.json) · [Verification commands](docs/RESEARCH.md)

```sh
python scripts/replay.py artifacts/baseline/trace.wire.jsonl --threshold '0.9552176915681485'
```

Clad and eljomustafaj coordinate the trace contract and milestone status in the [shared handoff](docs/HANDOFF.md).

**eljomustafaj → Clad · Live scorer import.** Import `ResidualModel` from `mech_interp.model`. Its `generate_tokens(prompt, probe, max_new_tokens)` iterator yields the candidate token ID, zero-based position, incremental text and risk before release. The worker can consume it one event at a time. [Runnable scoring example and field mapping](docs/RESEARCH.md#scoring-a-live-prompt)

### Run Task 2

From the repository root, launch the local replay console and select **Run replay**.

```sh
python scripts/replay.py fixtures/synthetic.wire.jsonl --threshold 0.75
```

The synthetic example withholds token 3 at the threshold and releases `The label reads`. See the [replay guide](docs/REPLAY.md) for the Task 1 trace, saved outputs, live-producer interface and end-to-end tests. Clad and eljomustafaj post updates in the [shared GitHub issue](https://github.com/eljomustafaj/clin-mech-firewall/issues/1).
