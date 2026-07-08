// OpenASR weights resolve proxy (Cloudflare Worker).
//
// PROXIES only the Hugging Face `resolve` endpoint for the `OpenASR/*` org, so a
// client whose direct route to `huggingface.co` is blocked (the resolve hop, not
// the CDN blob hop) can still reach it via `weights.openasr.org`. For normal
// `.oasr` packs Hugging Face's resolve response is a 3xx redirect to a Xet/CDN
// blob host (e.g. `cas-bridge.xethub.hf.co`) that is directly reachable; this
// worker returns that redirect verbatim and never fetches the blob itself, so
// the (large) weight bytes never transit Cloudflare.
//
// NOT a trust anchor: the client verifies sha256 + the ed25519-signed catalog
// (see cloudflare/catalog) against whatever bytes it ends up downloading,
// regardless of which host served the redirect. This worker's only job is
// reachability; its own integrity story is "open source + publicly deployed by
// CI from this commit", checkable via GET /_version.
//
// Hard scope limits (deliberate, do not relax without re-reviewing abuse risk):
//   - GET/HEAD only.
//   - Path must match `/OpenASR/<repo>/resolve/<rev>/<file...>` exactly. This is
//     NOT a general Hugging Face proxy: any other org, repo action (blob/tree/
//     api/...), or path shape is rejected before any outbound fetch happens.
//   - No caching (`Cache-Control: no-store` on every response).
//   - No logging of request paths, UA, or client IP (only the standard Workers
//     platform request log applies; this code adds none of its own).

const ALLOWED_METHODS = "GET, HEAD";

// /OpenASR/<repo>/resolve/<rev>/<file...>
// - <repo> and <rev> are restricted to the character set Hugging Face itself
//   uses for repo names and git refs (no "/", no "..").
// - <file...> may contain "/" (nested paths inside a repo) but individual
//   segments may not be "." or ".." (no traversal), and may not be empty.
export const RESOLVE_PATH =
  /^\/OpenASR\/([A-Za-z0-9][A-Za-z0-9._-]*)\/resolve\/([A-Za-z0-9][A-Za-z0-9._-]*)\/(.+)$/;

const UPSTREAM_TIMEOUT_MS = 15_000;

function hasPathTraversal(filePath) {
  return filePath.split("/").some((segment) => segment === "" || segment === "." || segment === "..");
}

function denied(status, message, extraHeaders = {}) {
  return new Response(`${message}\n`, {
    status,
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-store",
      ...extraHeaders,
    },
  });
}

function versionResponse(env) {
  const body = {
    service: "openasr-weights",
    build_commit: env.BUILD_COMMIT || "unknown",
  };
  return new Response(JSON.stringify(body, null, 2) + "\n", {
    status: 200,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/" || url.pathname === "/_version") {
      if (request.method !== "GET" && request.method !== "HEAD") {
        return denied(405, "Method Not Allowed", { Allow: ALLOWED_METHODS });
      }
      return versionResponse(env);
    }

    if (request.method !== "GET" && request.method !== "HEAD") {
      return denied(405, "Method Not Allowed: the weights proxy serves GET/HEAD only", {
        Allow: ALLOWED_METHODS,
      });
    }

    const match = RESOLVE_PATH.exec(url.pathname);
    if (!match || hasPathTraversal(match[3])) {
      return denied(
        404,
        "Not Found: only huggingface.co/OpenASR/*/resolve/*/* is proxied here",
      );
    }

    // Rebuild the exact upstream URL (path + query verbatim) against
    // huggingface.co; this worker never talks to any other host.
    const upstreamUrl = new URL(url.pathname + url.search, "https://huggingface.co");

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

    // Forward a minimal Accept/Range so HF can serve conditional/partial
    // requests for the rare inline (200) case; drop everything else (cookies,
    // UA, etc.) the client sent -- none of it is needed to resolve a public
    // repo, and we do not want to fingerprint-forward it upstream.
    const upstreamHeaders = new Headers({ Accept: request.headers.get("Accept") || "*/*" });
    const range = request.headers.get("Range");
    if (range) upstreamHeaders.set("Range", range);

    let upstream;
    try {
      upstream = await fetch(upstreamUrl.toString(), {
        method: request.method,
        redirect: "manual",
        signal: controller.signal,
        headers: upstreamHeaders,
      });
    } catch (err) {
      // Network error, DNS failure, or the abort from the timeout above.
      // Fail closed rather than hang the client.
      const timedOut = err && err.name === "AbortError";
      return denied(timedOut ? 504 : 502, `Upstream fetch failed: ${timedOut ? "timeout" : "error"}`);
    } finally {
      clearTimeout(timeout);
    }

    if (upstream.status >= 300 && upstream.status < 400) {
      // The expected path for real .oasr weights: HF resolves to a redirect
      // (Xet/CDN blob host). Pass the redirect through untouched -- the
      // client follows it directly and this worker never sees the blob.
      const location = upstream.headers.get("Location");
      if (!location) {
        return denied(502, "Upstream redirect missing Location");
      }
      return new Response(null, {
        status: upstream.status,
        headers: {
          Location: location,
          "Cache-Control": "no-store",
        },
      });
    }

    if (upstream.status >= 200 && upstream.status < 300) {
      // Rare: small files HF chooses to inline instead of redirecting.
      // Stream the body through as-is; still no caching.
      const headers = new Headers({ "Cache-Control": "no-store" });
      const contentType = upstream.headers.get("Content-Type");
      const contentLength = upstream.headers.get("Content-Length");
      if (contentType) headers.set("Content-Type", contentType);
      if (contentLength) headers.set("Content-Length", contentLength);
      return new Response(upstream.body, { status: upstream.status, headers });
    }

    // Upstream 4xx/5xx: fail closed, forward the status without leaking
    // upstream response details beyond the status code.
    return denied(upstream.status, `Upstream returned ${upstream.status}`);
  },
};
