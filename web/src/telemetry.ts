import type { TokenDecision, TraceInfo } from "./view.js";

export function object(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Expected a JSON object.");
  }
  return value as Record<string, unknown>;
}

export function identifier(value: unknown): string {
  if (typeof value !== "string" || !/[^\p{White_Space}]/u.test(value)) {
    throw new Error("Missing trace metadata.");
  }
  return value;
}

export function score(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error("Risk and threshold must be finite numbers between 0 and 1.");
  }
  return value;
}

function position(value: unknown): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new Error("Invalid position in telemetry.");
  }
  return value;
}

interface Observer {
  start(info: TraceInfo, threshold: number): void;
  decision(token: TokenDecision): void;
}

export class Telemetry {
  private trace?: string;
  private threshold = 0;
  private ended = false;
  private prefix: string[] = [];
  count = 0;
  halted = false;

  constructor(private observer: Observer) {}

  accept(value: unknown): void {
    const record = object(value);
    if (this.ended) throw new Error("Unexpected telemetry after the end record.");
    if (record.type === "error") {
      throw new Error(typeof record.message === "string" ? record.message : "Replay failed on the server.");
    }
    if (this.trace === undefined) {
      if (record.type !== "start" || record.schema_version !== 1) {
        throw new Error("Expected a version 1 start record.");
      }
      const info: TraceInfo = {
        trace: identifier(record.trace_id),
        model: identifier(record.model_id),
        modelRevision: identifier(record.model_revision),
        layer: position(record.layer),
        probe: identifier(record.probe_id),
        probeRevision: identifier(record.probe_revision),
      };
      this.trace = info.trace;
      this.threshold = score(record.threshold);
      this.observer.start(info, this.threshold);
      return;
    }
    if (record.trace_id !== this.trace) throw new Error("Trace ID changed during replay.");
    if (record.type === "end") {
      const expected = this.halted ? "halted" : "complete";
      if (record.token_count !== this.count || record.status !== expected) {
        throw new Error("End record disagrees with the token decisions.");
      }
      this.ended = true;
      return;
    }
    if (this.halted || record.type !== "decision" || record.index !== this.count) {
      throw new Error("Unexpected token or token position in telemetry.");
    }
    const risk = score(record.risk);
    const decision = risk >= this.threshold ? "HALT" : "ALLOW";
    if (record.decision !== decision || typeof record.text !== "string") {
      throw new Error("Invalid decision or token text in telemetry.");
    }
    const ids = record.sae_feature_ids;
    if (ids !== undefined && !Array.isArray(ids)) throw new Error("Invalid SAE feature IDs.");
    const features = ids?.map(position);
    const token = { index: this.count, text: record.text, risk, decision, features } as const;
    this.count++;
    this.halted = decision === "HALT";
    if (!this.halted) this.prefix.push(token.text);
    this.observer.decision(token);
  }

  expectedResponse(): string {
    if (!this.ended) throw new Error("Telemetry ended before the end record.");
    return this.prefix.join("");
  }
}

export async function consume(body: ReadableStream<Uint8Array>, telemetry: Telemetry): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let pending = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      let start = 0;
      for (let end = pending.indexOf("\n"); end !== -1; end = pending.indexOf("\n", start)) {
        telemetry.accept(JSON.parse(pending.slice(start, end)));
        start = end + 1;
      }
      pending = pending.slice(start);
      if (done) break;
    }
    if (pending.length) telemetry.accept(JSON.parse(pending));
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally {
    reader.releaseLock();
  }
}
