import { createExecutionContext, waitOnExecutionContext } from "cloudflare:test";
import { describe, expect, it } from "vitest";

import worker, { CATALOG_PATH } from "../src/index.js";

const ORIGIN = "https://catalog.openasr.org";

async function call(method, path) {
  const ctx = createExecutionContext();
  const response = await worker.fetch(new Request(`${ORIGIN}${path}`, { method }), {}, ctx);
  await waitOnExecutionContext(ctx);
  return response;
}

// These exercise the request-gating logic, which needs no asset binding (it
// returns before touching env.ASSETS). The asset-serving path is covered by the
// downstream Rust signature/sha256 verification and the manual smoke check in
// README.md.
describe("catalog host gating", () => {
  it("answers the CORS preflight", async () => {
    const response = await call("OPTIONS", "/v1/catalog.json");
    expect(response.status).toBe(204);
    expect(response.headers.get("Access-Control-Allow-Origin")).toBe("*");
    expect(response.headers.get("Access-Control-Allow-Methods")).toContain("GET");
  });

  it("rejects non-GET methods with an Allow header", async () => {
    const response = await call("POST", "/v1/catalog.json");
    expect(response.status).toBe(405);
    expect(response.headers.get("Allow")).toBe("GET, OPTIONS");
  });

  it("never serves model weights or other repos", async () => {
    const response = await call("GET", "/OpenASR/whisper-large/resolve/main/model.oasr");
    expect(response.status).toBe(403);
  });

  it("rejects catalog-repo paths that are not the catalog objects", async () => {
    const response = await call("GET", "/v1/README.md");
    expect(response.status).toBe(403);
  });

  it("accepts only the v1 catalog objects", () => {
    expect(CATALOG_PATH.test("/v1/catalog.json")).toBe(true);
    expect(CATALOG_PATH.test("/v1/catalog.signature.json")).toBe(true);
    // ...but not weights, other repos, traversal, or sub-paths.
    expect(CATALOG_PATH.test("/v1/README.md")).toBe(false);
    expect(CATALOG_PATH.test("/OpenASR/whisper/resolve/main/model.oasr")).toBe(false);
    expect(CATALOG_PATH.test("/v1/../secrets")).toBe(false);
  });
});
