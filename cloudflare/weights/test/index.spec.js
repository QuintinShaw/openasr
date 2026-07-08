import { createExecutionContext, fetchMock, waitOnExecutionContext } from "cloudflare:test";
import { afterEach, beforeAll, describe, expect, it } from "vitest";

import worker, { RESOLVE_PATH } from "../src/index.js";

const ORIGIN = "https://weights.openasr.org";

async function call(method, path) {
  const ctx = createExecutionContext();
  const response = await worker.fetch(
    new Request(`${ORIGIN}${path}`, { method }),
    { BUILD_COMMIT: "test-sha" },
    ctx,
  );
  await waitOnExecutionContext(ctx);
  return response;
}

beforeAll(() => {
  fetchMock.activate();
  fetchMock.disableNetConnect();
});

afterEach(() => {
  fetchMock.assertNoPendingInterceptors();
});

describe("weights resolve proxy gating", () => {
  it("answers the version endpoint without touching the network", async () => {
    const response = await call("GET", "/_version");
    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body).toEqual({ service: "openasr-weights", build_commit: "test-sha" });
  });

  it("rejects non-GET/HEAD methods with an Allow header", async () => {
    const response = await call("POST", "/OpenASR/qwen3-asr-1.7b/resolve/main/model.oasr");
    expect(response.status).toBe(405);
    expect(response.headers.get("Allow")).toBe("GET, HEAD");
  });

  it("rejects paths outside the OpenASR org", async () => {
    const response = await call("GET", "/someone-else/model/resolve/main/model.oasr");
    expect(response.status).toBe(404);
  });

  it("rejects OpenASR paths that are not a resolve request", async () => {
    const response = await call("GET", "/OpenASR/qwen3-asr-1.7b/blob/main/model.oasr");
    expect(response.status).toBe(404);
  });

  it("rejects path traversal inside the file segment", async () => {
    const response = await call("GET", "/OpenASR/qwen3-asr-1.7b/resolve/main/../../secrets");
    expect(response.status).toBe(404);
  });

  it("accepts only well-formed OpenASR resolve paths", () => {
    expect(RESOLVE_PATH.test("/OpenASR/qwen3-asr-1.7b/resolve/main/qwen3-asr-1.7b-q4_k.oasr")).toBe(true);
    expect(RESOLVE_PATH.test("/OpenASR/qwen3-asr-1.7b/blob/main/model.oasr")).toBe(false);
    expect(RESOLVE_PATH.test("/other-org/model/resolve/main/model.oasr")).toBe(false);
    expect(RESOLVE_PATH.test("/OpenASR//resolve/main/model.oasr")).toBe(false);
  });

  it("passes a legit resolve redirect through verbatim (does not follow the blob)", async () => {
    const blobUrl = "https://cas-bridge.xethub.hf.co/xet-blob/deadbeef";
    fetchMock
      .get("https://huggingface.co")
      .intercept({
        path: "/OpenASR/qwen3-asr-1.7b/resolve/main/model.oasr",
        method: "GET",
      })
      .reply(302, "", { headers: { location: blobUrl } });

    const response = await call("GET", "/OpenASR/qwen3-asr-1.7b/resolve/main/model.oasr");
    expect(response.status).toBe(302);
    expect(response.headers.get("Location")).toBe(blobUrl);
    expect(response.headers.get("Cache-Control")).toBe("no-store");
    // The worker must not have consumed/streamed a body from the blob host --
    // there is nothing to read here since redirects carry no body.
    expect(await response.text()).toBe("");
  });

  it("fails closed (does not hang) when upstream returns 5xx", async () => {
    fetchMock
      .get("https://huggingface.co")
      .intercept({
        path: "/OpenASR/qwen3-asr-1.7b/resolve/main/model.oasr",
        method: "GET",
      })
      .reply(503, "service unavailable");

    const response = await call("GET", "/OpenASR/qwen3-asr-1.7b/resolve/main/model.oasr");
    expect(response.status).toBe(503);
  });

  it("fails closed on an upstream network error", async () => {
    fetchMock
      .get("https://huggingface.co")
      .intercept({
        path: "/OpenASR/qwen3-asr-1.7b/resolve/main/model.oasr",
        method: "GET",
      })
      .replyWithError(new Error("simulated network failure"));

    const response = await call("GET", "/OpenASR/qwen3-asr-1.7b/resolve/main/model.oasr");
    expect(response.status).toBe(502);
  });
});
