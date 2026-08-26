// OpenASR signed-catalog host (Cloudflare Worker + Static Assets).
//
// HOSTS the signed model catalog on Cloudflare: `catalog.json` and
// `catalog.signature.json` are served from CF static assets, NEVER fetched from
// Hugging Face. Hugging Face hosts model *weights* only. The bytes are served
// verbatim; the OpenASR client verifies the ed25519 signature, the sha256, and
// the monotonic epoch, so this host is a distribution layer, not a trust anchor.
//
// Clients request `/v1/catalog(.signature).json`; the current published snapshot
// (in `public/`, refreshed at deploy from `model-registry/`) is returned and
// verified client-side.

export const CATALOG_PATH = /^\/v1\/(catalog\.json|catalog\.signature\.json)$/;

const ALLOWED_METHODS = "GET, OPTIONS";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": ALLOWED_METHODS,
  "Access-Control-Max-Age": "86400",
};

function denied(status, message, extraHeaders = {}) {
  return new Response(`${message}\n`, {
    status,
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      ...CORS_HEADERS,
      ...extraHeaders,
    },
  });
}

/// Shared by the Cloudflare Worker and the Aliyun ESA twin (`index.esa.js`).
/// `getAsset(filename)` must return the committed catalog / signature bytes
/// verbatim — no JSON parse/stringify, which would break the signed sha256.
export async function handleCatalogRequest(request, getAsset) {
  const url = new URL(request.url);

  if (request.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }
  if (request.method !== "GET") {
    return denied(405, "Method Not Allowed: the catalog host serves GET only", {
      Allow: ALLOWED_METHODS,
    });
  }
  const match = CATALOG_PATH.exec(url.pathname);
  if (!match) {
    return denied(
      403,
      "Forbidden: only the OpenASR catalog and its signature manifest are served here",
    );
  }

  const asset = await getAsset(match[1]);
  if (!asset || !asset.ok) {
    const status = asset?.status === 404 ? 404 : 502;
    return denied(status, `Catalog asset unavailable (${asset?.status ?? "missing"})`);
  }

  const headers = new Headers(CORS_HEADERS);
  headers.set("Content-Type", "application/json; charset=utf-8");
  headers.set("Cache-Control", "public, max-age=300");
  return new Response(asset.body, { status: 200, headers });
}

export default {
  async fetch(request, env) {
    const origin = new URL(request.url).origin;
    return handleCatalogRequest(request, (name) =>
      env.ASSETS.fetch(new Request(new URL(`/${name}`, origin))),
    );
  },
};
