# Streaming firewall replay

Run the commands from the repository root. The server uses Python 3.10+ and a stable Rust toolchain with Cargo. Cargo downloads the locked dependencies on the first build. The compiled browser files are included in the checkout.

## Try the synthetic trace

```sh
python scripts/replay.py fixtures/synthetic.wire.jsonl --threshold 0.75
```

Open the printed loopback URL and choose **Run replay**. The response reads `The label reads`. The operator table shows `" amber"` at zero-based token index 3 with score 0.75 and decision HALT. The following period never appears in the table or the response.

To save the same replay to disk, choose a new output directory.

```sh
python scripts/replay.py fixtures/synthetic.wire.jsonl --threshold 0.75 --out runs/synthetic
```

`response.txt` contains the exact UTF-8 bytes approved by the policy. `telemetry.jsonl` contains the start metadata, decisions and terminal status. The local operator can inspect withheld text in this file and in the browser. Access to the operator console therefore includes access to the withheld content.

## Run a measured trace

Boli's measured trace has 12 generated tokens. Use the threshold from the published probe to reproduce the first handoff.

```sh
python scripts/replay.py artifacts/baseline/trace.wire.jsonl --threshold 0.9552176915681485
```

Token 0, `" by"`, has risk 0.965618595746429 and receives HALT. Both the HTTP response body and the saved response contain zero bytes. The next token, `" car"`, is the unsupported chart detail and scores below the threshold. This run demonstrates enforcement at the first threshold crossing. It also shows that the triggering token can precede a factual error. [Browser capture, telemetry and response](../artifacts/replay-v1/)

The measured integration tests use the published probe threshold by default. Set `REPLAY_THRESHOLD` to exercise another cutoff that halts the trace. To inspect an all-ALLOW run, start the console with `--threshold 1`.

## How a token reaches the browser

1. The Python worker reads Boli's scored trace one UTF-8 JSON object at a time. A live producer can pass an iterator of scored records to `worker.replay.write_trace(records, binary_output)`, or send its JSONL to `python -u worker/replay.py -`. The offline replay reuses Task 1's recorded scores.
2. The Rust proxy reads each token from standard input, checks its index and risk and applies the threshold before writing response bytes. It flushes ALLOW text to the response file and flushes each decision to telemetry. At the first score greater than or equal to the threshold, it emits HALT and ends the replay. A complete all-ALLOW run requires a matching end count and EOF.
3. The loopback server starts a separate worker and proxy for each replay. `POST /api/replays` streams operator telemetry as NDJSON and returns an `X-Replay-Id` header. `GET /api/replays/<id>/response` returns the approved text only after a successful terminal event. The server retains the 32 latest completed responses until it exits. A failed replay has no retrievable response.
4. The TypeScript view renders each decision as it arrives. It displays the separate response after checking that its decoded text equals the ALLOW prefix. Trace details identify the model, revision, layer, probe and probe revision.

To run with an existing build, add `--proxy PATH_TO_EXECUTABLE`. To use a different loopback port, add `--port 8001`.

### Stream from a loaded model

After loading Boli's pinned `ResidualModel` in the research environment, pass that instance and a prompt to the worker. The target can be the Rust proxy's binary stdin. The worker computes the probe revision from the bytes it actually scores with and emits each candidate as the model produces it. A new run ID identifies live output independently of the archived trace. [Loading the model](RESEARCH.md#scoring-a-live-prompt)

```python
from pathlib import Path
import sys
from uuid import uuid4
from worker.inference import write_inference

write_inference(model, prompt, Path("artifacts/baseline/probe.json"), sys.stdout.buffer,
                trace_id=uuid4().hex, max_new_tokens=12)
```

## Reproduce the checks

```sh
cargo test --locked --manifest-path proxy/Cargo.toml
cargo build --locked --manifest-path proxy/Cargo.toml
python -m unittest -v worker.test_replay worker.test_inference
python -m unittest discover -s tests -v
cd web
npm ci
npm run build
npx playwright-core install chromium
npm test
```

Browser development and tests use Node.js 22 and npm. On Linux, use `npx playwright-core install --with-deps chromium` if the system needs Chromium libraries. `CHROME_PATH` selects an installed Chrome executable. `FIREWALL_PROXY` selects a prebuilt proxy for integration tests.

The tests exercise HALT on the threshold, an empty release, a completed all-ALLOW replay, truncation, token delivery before producer EOF and a failed second replay in the browser. The measured-trace tests run when Boli's wire file is present and otherwise report a skip. The browser test compares the actual HTTP response bytes and every displayed token index, text, score and decision against the input. The [shared handoff](HANDOFF.md) records the agreed wire format and links the owners' coordination issue.
