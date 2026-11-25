# Task 2 handoff · Clad and Boli

## Clad → Boli · proposed trace v1

Clad owns Task 2. Boli owns Task 1. Boli, please acknowledge the following wire format here or in a GitHub issue, or post your existing format and its path so we can reconcile them. Please identify the real trace, model revision, probe revision and reproduction command when ready.

The worker emits UTF-8 newline-delimited JSON. One stream contains one trace. The first record identifies the trace and scorer, followed by contiguous zero-based token records in generation order, then an explicit end record. Token text is the exact incremental decoded text, including whitespace. Scores are finite numbers from zero through one, with larger scores indicating more risk. The layer is a zero-based transformer block index. The producer can omit SAE feature IDs.

```jsonl
{"type":"start","schema_version":1,"trace_id":"example","model_id":"fixture","model_revision":"v1","layer":0,"probe_id":"fixture","probe_revision":"v1"}
{"type":"token","index":0,"text":"Hello","risk":0.1}
{"type":"token","index":1,"text":" world","risk":0.8,"sae_feature_ids":[3]}
{"type":"end","token_count":2}
```

The proxy adds the configured threshold to the start record. It emits one decision record per processed token, including trace_id, index, text, risk, decision and optional sae_feature_ids. If risk >= threshold, the token receives HALT, is recorded in telemetry and is withheld from the response. HALT is terminal. The telemetry end record includes trace_id, status (halted or complete), and token_count (processed decisions, including a triggering token). Any stream or validation failure is a failed replay. The response contains exactly the UTF-8 text of the ALLOW prefix.

Example values above are synthetic. An example is suitable for testing the wire format. The first research handoff requires Task 1's actual trace.

## Boli → Clad · wire v1 accepted

I accept the JSONL contract above. I will write the real replay input to `artifacts/baseline/trace.wire.jsonl`. Each token maps to one archive position. The start record will include a stable probe family ID and the SHA-256 revision of the exact probe bytes. The source archive at `artifacts/baseline/trace.json` will also keep the prompt and model token IDs for verification. I will post measured results and reproducible commands in the README when ready.

## Clad → Boli · Task 2 implementation

The [wire v1 schema](../schema/wire-v1.schema.json) describes individual input and telemetry records. The worker and proxy also validate record order and counts. The [replay guide](REPLAY.md) contains the commands and endpoint behavior. Use [issue #1](https://github.com/eljomustafaj/clin-mech-firewall/issues/1) for acknowledgements, questions, trace commit IDs and integration results.

## Boli → Clad · real trace verified

The [measured wire input](../artifacts/baseline/trace.wire.jsonl) and [fitted probe](../artifacts/baseline/probe.json) are ready. The [README handoff](../README.md#boli-and-clad-handoff) gives the immutable revisions and replay command. The trace contains 12 tokens from Qwen block 12. With the published validation threshold the Rust proxy records `HALT` at index 0 and releases an empty response. At threshold `1` it releases all 12 tokens. Both the offline files and the HTTP response were checked with Clad’s worker and Windows proxy. The browser independently showed the halt at the same index. Clad can use these files in the measured-trace integration and browser tests.

## Clad → Boli · measured replay accepted

I replayed the measured input from Boli's [baseline commit](https://github.com/eljomustafaj/clin-mech-firewall/commit/4560636f65506bffba4eecb834b584b49f08a6c3) at the published threshold of 0.9552176915681485. Chrome displayed HALT at index 0 for `" by"`, risk 0.965618595746429. The response endpoint and saved response both contained zero bytes. The integration test regenerated the wire input from the research archive and verified the exact probe digest before comparing the response and telemetry. [Saved decisions](../artifacts/replay-v1/telemetry.jsonl) · [Response file](../artifacts/replay-v1/response.txt) · [Browser capture](../artifacts/replay-v1/browser.png) · [Reproduction commands](REPLAY.md#run-a-measured-trace)
