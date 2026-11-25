import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join, relative, resolve } from "node:path";
import { createInterface } from "node:readline";
import { after, before, test } from "node:test";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const root = fileURLToPath(new URL("../../", import.meta.url));
const fixture = join(root, "fixtures/synthetic.wire.jsonl");
const baseline = join(root, "artifacts/baseline/trace.wire.jsonl");
const archive = process.env.REAL_TRACE ? resolve(process.env.REAL_TRACE) : baseline;
const executable = process.platform === "win32" ? "firewall-proxy.exe" : "firewall-proxy";
const proxy = process.env.FIREWALL_PROXY || join(root, "proxy/target/debug", executable);
let browser;

before(async () => {
  browser = await chromium.launch({ executablePath: process.env.CHROME_PATH });
});
after(async () => { await browser?.close(); });

async function startServer(t, trace, threshold) {
  const child = spawn(process.env.PYTHON || "python", [
    join(root, "scripts/replay.py"), trace, "--threshold", String(threshold),
    "--proxy", resolve(proxy), "--port", "0",
  ], { cwd: root, stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
  const stdout = createInterface({ input: child.stdout });
  let diagnostics = "";
  child.stderr.on("data", (chunk) => { diagnostics += chunk; });
  t.after(async () => {
    stdout.close();
    if (child.pid && child.exitCode === null && child.signalCode === null) {
      const exited = once(child, "close");
      child.kill();
      await exited;
    }
  });
  return new Promise((res, reject) => {
    child.once("error", reject);
    child.once("exit", (code) => reject(new Error(`Replay runner exited (${code}): ${diagnostics}`)));
    stdout.on("line", (line) => {
      const match = /^Replay console: (http:\/\/127\.0\.0\.1:\d+\/)$/u.exec(line);
      if (match) res(match[1]);
    });
  });
}

async function verify(t, trace, supplied, { requireHalt = true } = {}) {
  const records = (await readFile(trace, "utf8")).trimEnd().split("\n").map((line) => JSON.parse(line));
  const tokens = records.filter((r) => r.type === "token");
  const threshold = supplied ?? tokens.reduce((max, token) => Math.max(max, token.risk), 0);
  const halt = tokens.findIndex((r) => r.risk >= threshold);
  if (requireHalt) assert.ok(halt >= 0, "The requested threshold must exercise HALT on this trace.");
  const allowed = halt < 0 ? tokens : tokens.slice(0, halt);
  const expected = allowed.map((r) => r.text).join("");
  const processed = halt < 0 ? tokens : tokens.slice(0, halt + 1);
  const url = await startServer(t, trace, threshold);
  const page = await browser.newPage();
  t.after(() => page.close());
  await page.goto(url);
  const resultReceived = page.waitForResponse((r) => r.url().endsWith("/response"));
  // Promise.all surfaces either the failed click or the missing response.
  const [result] = await Promise.all([
    resultReceived,
    page.getByRole("button", { name: "Run replay", exact: true }).click(),
  ]);
  assert.equal(result.status(), 200);
  assert.deepEqual(await result.body(), Buffer.from(expected, "utf8"));
  await page.getByRole("button", { name: "Run again" }).waitFor();
  assert.equal(await page.locator("#error").isHidden(), true, await page.locator("#error").textContent());
  assert.equal(await page.locator("#threshold").textContent(), `≥ ${threshold}`);
  assert.equal(await page.locator("#status").textContent(), halt < 0 ? "Complete" : `Halted at #${halt}`);
  assert.equal(await page.locator("#response").textContent(), expected);
  assert.deepEqual(await page.locator("#decisions tr").evaluateAll((rows) => rows.map((row) => ({
    index: Number(row.cells[0].textContent), text: row.querySelector(".token").textContent,
    risk: Number(row.cells[2].textContent), decision: row.cells[3].textContent,
  }))), processed.map((r) => ({
    index: r.index, text: r.text, risk: r.risk, decision: r.index === halt ? "HALT" : "ALLOW",
  })));
  const start = records[0];
  assert.deepEqual(await page.locator("#details dd").allTextContents(), [
    start.trace_id, start.model_id, start.model_revision, String(start.layer), start.probe_id, start.probe_revision,
  ]);
  return page;
}

test("browser shows the proxy's threshold trigger and its exact released bytes", async (t) => {
  const page = await verify(t, fixture, 0.75);
  assert.match(await page.locator('tr[data-decision="HALT"] .features').textContent(), /7/u);
  await page.setViewportSize({ width: 360, height: 740 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));

  // An interrupted second run must clear the earlier success and response.
  await page.route("**/api/replays", (route) => route.fulfill({
    contentType: "application/x-ndjson", headers: { "X-Replay-Id": "interrupted" }, body: "",
  }));
  await page.getByRole("button", { name: "Run again" }).click();
  await page.getByRole("alert").waitFor();
  assert.equal(await page.locator("#status").textContent(), "Failed");
  assert.equal(await page.locator("#response").textContent(), "");
  assert.equal(await page.locator("#decisions tr").count(), 0);
});

test("browser renders a completed all-ALLOW response", async (t) => {
  await verify(t, fixture, 1, { requireHalt: false });
});

test("browser preserves Unicode and markup as text, and detects a mismatched response", async (t) => {
  const page = await browser.newPage();
  t.after(() => page.close());
  const metadata = {
    type: "start", schema_version: 1, trace_id: "unicode", model_id: "sample", model_revision: "v1",
    layer: 0, probe_id: "sample", probe_revision: "v1", threshold: 0.5,
  };
  const allowed = "\uFEFF Café\n";
  const held = "<strong>withheld</strong>";
  const stream = [
    metadata,
    { type: "decision", trace_id: "unicode", index: 0, risk: 0.11, text: allowed, decision: "ALLOW" },
    { type: "decision", trace_id: "unicode", index: 1, risk: 0.5, text: held, decision: "HALT" },
    { type: "end", trace_id: "unicode", status: "halted", token_count: 2 },
  ].map((record) => JSON.stringify(record)).join("\n");
  let response = allowed;
  const assets = { "/": "index.html", "/main.js": "main.js", "/telemetry.js": "telemetry.js", "/view.js": "view.js", "/style.css": "style.css" };
  await page.route("http://127.0.0.1/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/config") return route.fulfill({ json: { trace_name: "unicode", threshold: 0.5 } });
    if (path === "/api/replays") {
      return route.fulfill({ contentType: "application/x-ndjson", headers: { "X-Replay-Id": "u" }, body: stream });
    }
    if (path === "/api/replays/u/response") return route.fulfill({ contentType: "text/plain", body: Buffer.from(response) });
    const asset = assets[path];
    if (!asset) return route.fulfill({ status: 404 });
    const type = asset.endsWith(".js") ? "text/javascript" : asset.endsWith(".css") ? "text/css" : "text/html; charset=utf-8";
    return route.fulfill({ contentType: type, path: join(root, "web/dist", asset) });
  });
  await page.goto("http://127.0.0.1/");
  await page.getByRole("button", { name: "Run replay" }).click();
  await page.getByRole("button", { name: "Run again" }).waitFor();
  assert.equal(await page.locator("#status").textContent(), "Halted at #1");
  assert.equal(await page.locator("#response").textContent(), allowed);
  assert.equal(await page.locator('tr[data-decision="HALT"] .token').textContent(), held);
  assert.equal(await page.locator("#decisions strong").count(), 0);
  response += held;
  await page.getByRole("button", { name: "Run again" }).click();
  await page.getByRole("alert").waitFor();
  assert.equal(await page.locator("#status").textContent(), "Failed");
  assert.equal(await page.locator("#response").textContent(), "");
});

test("browser matches Task 1 archive through the real worker and proxy", {
  skip: !process.env.REAL_TRACE && !existsSync(archive) ? "Task 1 wire trace is not present" : false,
}, async (t) => {
  let threshold;
  if (process.env.REPLAY_THRESHOLD !== undefined) {
    threshold = Number(process.env.REPLAY_THRESHOLD);
  } else if (relative(baseline, archive) === "") {
    const probe = JSON.parse(await readFile(join(root, "artifacts/baseline/probe.json"), "utf8"));
    assert.ok(Number.isFinite(probe.threshold) && probe.threshold >= 0 && probe.threshold <= 1,
      "The published probe must include its validation threshold.");
    threshold = probe.threshold;
  }
  await verify(t, archive, threshold);
});
