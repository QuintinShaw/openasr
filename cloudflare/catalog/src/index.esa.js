// Aliyun ESA twin of catalog.openasr.org.
//
// Same allowlist and CORS as the Cloudflare Worker; bytes are baked into the
// bundle at `esa-cli` build time (see scripts/build-esa.mjs) so ESA never
// origin-fetches Cloudflare. Identity stays
// `https://catalog.openasr.org/v1/catalog.json`.
//
// ESA runtime facts (from bugim-cap / bugim-edge, verified on edgeworker2):
//   - `export default { fetch }`; second arg is ctx with waitUntil, no bindings.
//   - Static `assets.directory` would auto-serve files and skip this gate, so
//     catalog JSON is inlined rather than placed at /v1/*.json.

import { handleCatalogRequest } from "./index.js";

const CATALOG_JSON = globalThis.__ESA_CATALOG_JSON__;
const CATALOG_SIGNATURE_JSON = globalThis.__ESA_CATALOG_SIGNATURE_JSON__;

const BAKED = {
  "catalog.json": CATALOG_JSON,
  "catalog.signature.json": CATALOG_SIGNATURE_JSON,
};

export default {
  fetch(request) {
    return handleCatalogRequest(request, async (name) => {
      const body = BAKED[name];
      if (typeof body !== "string" || body.length === 0) {
        return new Response("Catalog asset not baked into this ESA build\n", {
          status: 404,
        });
      }
      return new Response(body, { status: 200 });
    });
  },
};
