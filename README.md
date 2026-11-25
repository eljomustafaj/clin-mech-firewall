# Mechanistic Interpretability & Hallucination Firewall for Clinical AI

`clin-mech-firewall` is a research project investigating whether an open-weight model’s internal activations can flag fabricated clinical details as it generates them. The first milestone is a single clinical prompt whose real token-by-token risk scores can be replayed through a firewall and inspected in a browser. Clinical accuracy, intervention effects, and latency are open questions.

## Task 1 — Boli owns the research baseline

Choose an open-weight model and residual-stream layer that fit the available hardware. Build a small, source-traceable set of paired clinical prompts, including cases where a patient detail is supported by the context and cases where it is invented. Keep both members of each pair in the same data split. Capture activations, train a linear probe, and measure its performance on held-out cases.

**First handoff.** Provide the versioned probe weights, model and layer identifiers, and a command that reproduces one real token trace and the held-out evaluation. Document a false alarm or missed fabrication if one occurs. A probe score measures association. Claims about steering will require a later controlled intervention, such as activation patching or ablation.

## Task 2 — Clad owns the streaming firewall and visualizer

Start by agreeing on a versioned event schema with Boli, before either owner builds against it. Define how a trace identifies the model and probe, orders tokens, records a risk score, and links a policy decision to a token. Allow sparse autoencoder feature IDs to be absent while the linear probe is the only scorer.

Build the first vertical slice through a Python inference worker, a Rust streaming proxy, and a TypeScript replay view. The proxy should apply a configurable `ALLOW` or `HALT` decision before releasing each token and stream the decision to the view. Record the triggering token in telemetry even when it is withheld from the response.

**First handoff.** Replay the trace from Task 1 through the proxy. Use a threshold that exercises `HALT`, verify that the triggering token never reaches the response, and show its score and decision at the same token position in the browser. Save the replay command and integration test so both owners can reproduce the result.

## Boli and Clad handoff

**Boli → Clad · Wire v1 accepted.** I will publish the measured trace as `artifacts/baseline/trace.wire.jsonl` in Clad's start/token/end format, plus a research archive with the original token IDs and prompt. The planned scorer is Qwen2.5-0.5B-Instruct at zero-based post-block layer 12. I will publish the model revision, exact probe digest, replay command and an example threshold with the measured trace.

Clad and Boli coordinate the trace contract and milestone status in the [shared handoff](docs/HANDOFF.md).

### Run Task 2

From the repository root, launch the local replay console and select **Run replay**.

```sh
python scripts/replay.py fixtures/synthetic.wire.jsonl --threshold 0.75
```

The synthetic example withholds token 3 at the threshold and releases `The label reads`. See the [replay guide](docs/REPLAY.md) for the Task 1 trace, saved outputs, live-producer interface and end-to-end tests. Clad and Boli post updates in the [shared GitHub issue](https://github.com/eljomustafaj/clin-mech-firewall/issues/1).
