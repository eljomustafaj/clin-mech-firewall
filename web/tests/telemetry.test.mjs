import assert from "node:assert/strict";
import { test } from "node:test";
import { consume, Telemetry } from "../dist/telemetry.js";

const start = {
  type: "start", schema_version: 1, trace_id: "sample", model_id: "m", model_revision: "v",
  layer: 0, probe_id: "p", probe_revision: "v", threshold: 0.5,
};
const token = { type: "decision", trace_id: "sample", index: 0, text: " café\n", risk: 0.3, decision: "ALLOW" };
const end = { type: "end", trace_id: "sample", status: "complete", token_count: 1 };
const noObserver = { start() {}, decision() {} };

test("delivers a complete token before EOF across split multibyte UTF-8 chunks", async () => {
  const bytes = new TextEncoder().encode([start, token].map((event) => JSON.stringify(event)).join("\r\n") + "\r\n");
  let controller;
  const stream = new ReadableStream({ start(c) { controller = c; } });
  let observed;
  const seen = new Promise((res) => { observed = res; });
  const state = new Telemetry({ start() {}, decision: observed });
  const done = consume(stream, state);
  for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
  assert.equal((await seen).text, token.text);
  controller.enqueue(new TextEncoder().encode(JSON.stringify(end)));
  controller.close();
  await done;
  assert.equal(state.expectedResponse(), token.text);
});

test("rejects unversioned, inconsistent, interrupted, and post-HALT telemetry", () => {
  const halted = { ...token, decision: "HALT", risk: 0.5 };
  for (const events of [
    [{ ...start, schema_version: 2 }], [start], [start, token], [start, { ...token, index: 1 }],
    [start, { ...token, risk: NaN }], [start, { ...token, risk: 0.5 }],
    [start, { ...token, trace_id: "other" }], [start, { ...token, sae_feature_ids: [-1] }],
    [start, halted, end], [start, halted, { ...token, index: 1 }],
    [start, token, end, end], [start, { type: "error", message: "Worker failed." }],
  ]) {
    assert.throws(() => {
      const state = new Telemetry(noObserver);
      for (const event of events) state.accept(event);
      state.expectedResponse();
    });
  }
});
