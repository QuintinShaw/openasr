// ESA build for catalog.bug.im. Mirrors bugim-cap/scripts/build-esa.mjs:
// esbuild bundles the ESA entry and inlines the two catalog files as strings
// via `define`. ESA has no Worker bindings; baking is how we push bytes
// instead of origin-fetching Cloudflare.
import { readFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = join(here, "..");
const catalogPath = join(pkg, "public", "catalog.json");
const signaturePath = join(pkg, "public", "catalog.signature.json");

function mustRead(path) {
  try {
    return readFileSync(path, "utf8");
  } catch (error) {
    throw new Error(
      `ESA catalog build needs ${path} (copy the candidate pair into public/ first): ${error.message}`,
    );
  }
}

const catalogJson = mustRead(catalogPath);
const signatureJson = mustRead(signaturePath);
if (!catalogJson.startsWith("{") || !signatureJson.startsWith("{")) {
  throw new Error("baked catalog files must be raw JSON objects, not empty/transformed");
}

mkdirSync(join(pkg, "dist", "esa"), { recursive: true });

await build({
  entryPoints: [join(pkg, "src", "index.esa.js")],
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  outfile: join(pkg, "dist", "esa", "index.js"),
  define: {
    "globalThis.__ESA_CATALOG_JSON__": JSON.stringify(catalogJson),
    "globalThis.__ESA_CATALOG_SIGNATURE_JSON__": JSON.stringify(signatureJson),
  },
});

console.log(
  `[build:esa] baked catalog.json (${catalogJson.length} chars) + catalog.signature.json (${signatureJson.length} chars)`,
);
