import { consume, identifier, object, score, Telemetry } from "./telemetry.js";
import * as view from "./view.js";
async function get(path, method = "GET") {
    const result = await fetch(path, { method, cache: "no-store", redirect: "error" });
    if (!result.ok)
        throw new Error(`Replay service returned HTTP ${result.status}.`);
    return result;
}
async function replay() {
    view.begin();
    try {
        const stream = await get("/api/replays", "POST");
        const id = stream.headers.get("X-Replay-Id");
        if (!id || !/^[a-zA-Z0-9_-]+$/.test(id) || !stream.body) {
            throw new Error("The replay service did not return a valid stream and replay ID.");
        }
        const state = new Telemetry({ start: view.showTrace, decision: view.appendDecision });
        await consume(stream.body, state);
        const expected = state.expectedResponse();
        const result = await get(`/api/replays/${id}/response`);
        const text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(await result.arrayBuffer());
        if (text !== expected)
            throw new Error("Released response disagrees with the allowed token prefix.");
        view.finish(state.halted, state.count - 1, text);
    }
    catch (error) {
        view.fail(error instanceof Error ? error.message : "Replay failed.");
    }
    finally {
        view.enableReplay();
    }
}
async function initialize() {
    view.runButton.disabled = true;
    try {
        const response = await get("/api/config");
        const config = object(await response.json());
        view.configure(identifier(config.trace_name), score(config.threshold));
        view.runButton.onclick = replay;
    }
    catch (error) {
        view.fail(error instanceof Error ? error.message : "Could not load the replay configuration.");
        view.runButton.textContent = "Retry connection";
        view.runButton.disabled = false;
        view.runButton.onclick = initialize;
    }
}
void initialize();
