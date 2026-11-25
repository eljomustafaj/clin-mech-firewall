# Mechanistic Interpretability & Hallucination Firewall for Clinical AI

`clin-mech-firewall` is a research project investigating whether an open-weight model’s internal activations can flag fabricated clinical details as it generates them. The first milestone is a single clinical prompt whose real token-by-token risk scores can be replayed through a firewall and inspected in a browser. Clinical accuracy, intervention effects, and latency are open questions.

## Prior-visit details on fresh charts

A rule that checks **where the candidate detail appears in the chart** caught **123/160** answers copied from an earlier visit, up from **39/160** for the original probe. It recovered all ten misses from the original study. On the fresh charts it retained all 39 original detections, added 84, and flagged **35/320** supported answers, compared with **39/320** before. Both methods caught **154/160** substitutions absent from the chart. The recalibrated answer-token probe and the learned source-role probe caught 27 and 24 prior substitutions respectively.

The rule succeeded when the matching line contained an explicit cue such as “preceding attendance.” It caught just **3/40** when a separate header identified which encounter was historical. In free generation, it halted by the end of both directly matched prior-value openings and one of six correct openings. The other 24 openings were unclassified. [Full results and reproduction commands](docs/PRIOR_VISIT_RESULTS.md) · [Live scorer](mech_interp/chronology.py)

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

**eljomustafaj → Clad · Prior-visit results ready.** The fixed chronology rule outperformed the learned alternatives on the new candidate test. The 32 recorded generations include every candidate token, raw score, cue and released prefix. The live iterator yields those decisions before release. [Results and failure cases](docs/PRIOR_VISIT_RESULTS.md) · [Recorded generations](artifacts/temporal-v2/generation/generations.jsonl) · [Selection protocol](docs/PRIOR_VISIT_PROTOCOL.md)

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
