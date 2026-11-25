function element<T extends keyof HTMLElementTagNameMap>(id: string, tag: T): HTMLElementTagNameMap[T] {
  const node = document.getElementById(id);
  if (!node || node.localName !== tag) throw new Error(`Missing ${tag}#${id}`);
  return node as HTMLElementTagNameMap[T];
}

export const runButton = element("run", "button");
const status = element("status", "dd");
const count = element("count", "span");
const empty = element("empty", "p");
const body = element("decisions", "tbody");
const response = element("response", "pre");
const responseNote = element("response-note", "p");
const details = element("details", "dl");
const detailNote = element("detail-note", "p");
const error = element("error", "p");

export interface TraceInfo {
  trace: string;
  model: string;
  modelRevision: string;
  layer: number;
  probe: string;
  probeRevision: string;
}

export interface TokenDecision {
  index: number;
  text: string;
  risk: number;
  decision: "ALLOW" | "HALT";
  features?: number[];
}

function setStatus(phase: string, label: string): void {
  status.dataset.phase = phase;
  status.textContent = label;
}

export function configure(name: string, threshold: number): void {
  error.hidden = true;
  element("trace-source", "dd").textContent = name;
  element("threshold", "dd").textContent = `≥ ${threshold}`;
  runButton.disabled = false;
  runButton.textContent = "Run replay";
  setStatus("ready", "Ready");
}

export function begin(): void {
  runButton.disabled = true;
  runButton.textContent = "Replaying…";
  error.hidden = true;
  body.replaceChildren();
  count.textContent = "0 tokens";
  empty.textContent = "Waiting for token decisions.";
  empty.hidden = false;
  response.textContent = "";
  response.hidden = true;
  responseNote.textContent = "Waiting for the released response.";
  responseNote.hidden = false;
  details.replaceChildren();
  details.hidden = true;
  detailNote.hidden = false;
  setStatus("active", "Replaying");
}

export function showTrace(info: TraceInfo, threshold: number): void {
  const pairs: [string, string][] = [
    ["Trace ID", info.trace], ["Model", info.model], ["Model revision", info.modelRevision],
    ["Layer", String(info.layer)], ["Probe", info.probe], ["Probe revision", info.probeRevision],
  ];
  for (const [name, value] of pairs) {
    const group = document.createElement("div");
    const label = document.createElement("dt");
    const content = document.createElement("dd");
    label.textContent = name;
    content.textContent = value;
    group.append(label, content);
    details.append(group);
  }
  details.hidden = false;
  detailNote.hidden = true;
  element("threshold", "dd").textContent = `≥ ${threshold}`;
}

export function appendDecision(token: TokenDecision): void {
  empty.hidden = true;
  const row = body.insertRow();
  row.dataset.decision = token.decision;
  row.insertCell().textContent = String(token.index);
  const tokenCell = row.insertCell();
  const text = document.createElement("span");
  text.className = "token";
  text.textContent = token.text;
  text.title = JSON.stringify(token.text);
  tokenCell.append(text);
  if (token.features?.length) {
    const features = document.createElement("span");
    features.className = "features";
    features.textContent = `SAE features · ${token.features.join(", ")}`;
    tokenCell.append(features);
  }
  const risk = row.insertCell();
  risk.className = "risk";
  risk.textContent = String(token.risk);
  const decision = row.insertCell();
  decision.className = "decision";
  decision.textContent = token.decision;
  const total = token.index + 1;
  count.textContent = `${total} ${total === 1 ? "token" : "tokens"}`;
}

export function finish(halted: boolean, index: number, released: string): void {
  response.textContent = released;
  response.hidden = released === "";
  responseNote.textContent = "No text was released.";
  responseNote.hidden = released !== "";
  empty.textContent = "The trace contained no tokens.";
  setStatus(halted ? "halted" : "complete", halted ? `Halted at #${index}` : "Complete");
}

export function fail(message: string): void {
  error.textContent = message;
  error.hidden = false;
  empty.textContent = "No token decisions received.";
  responseNote.textContent = "Released response unavailable.";
  responseNote.hidden = false;
  response.textContent = "";
  response.hidden = true;
  setStatus("failed", "Failed");
}

export function enableReplay(): void {
  runButton.textContent = "Run again";
  runButton.disabled = false;
}
