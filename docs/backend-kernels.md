# Desktop inference kernel manifest (`backends-manifest.json`)

This document is the contract for the **desktop app's Windows "inference
kernel" switch** (Vulkan / CUDA / HIP): a signed manifest that tells the
desktop app which prebuilt `openasr-cli` release archive to download for each
backend, and how to verify it before ever launching it as the sidecar
process.

This is a **different mechanism** from the `GGML_BACKEND_DL` dynamic
plugin-loading path described in [`GPU_PLUGIN_BUILD.md`](GPU_PLUGIN_BUILD.md)
and implemented by `install_backend_pack`/`ensure_backends_loaded`. That path
loads a GPU backend as a runtime plugin DLL *into one running process*. The
mechanism here instead swaps the **entire sidecar binary**: the Windows
release channel ships three separately, statically-linked `openasr.exe`
builds (one per backend), each a complete, independent CLI. Desktop picks
which whole binary to launch, not which plugin to load into one. Do not
conflate the two -- a future consumer wanting the DL-plugin path should extend
that mechanism, not this manifest.

## Files

Published by release CI (see the sibling `feat/sign-and-publish-backend-kernels`
work) alongside the release's binary assets, both at:

- `https://dl.openasr.org/core/v<version>/backends-manifest.json` (primary)
- `https://github.com/QuintinShaw/openasr/releases/download/v<version>/backends-manifest.json` (fallback)

and its detached signature at the same location with a `.signature.json`
suffix swapped in for `.json` (`backends-manifest.signature.json`).

## Schema (`schema_version: 1`)

```jsonc
{
  "schema_version": 1,
  "core_version": "0.1.14",              // openasr-cli semver these archives were built from
  "source_commit": "<full 40-char sha>", // git commit the archives were built from
  "platforms": {
    "windows-x86_64": {
      "backends": {
        "vulkan": {
          "asset": "openasr-0.1.14-windows-x86_64-vulkan.zip",
          "size_bytes": 123456,
          "sha256": "<64 lowercase hex chars>",
          "urls": [
            "https://dl.openasr.org/core/v0.1.14/openasr-0.1.14-windows-x86_64-vulkan.zip",
            "https://github.com/QuintinShaw/openasr/releases/download/v0.1.14/openasr-0.1.14-windows-x86_64-vulkan.zip"
          ],
          "pe_import_markers": ["vulkan-1.dll"]
        },
        "cuda": { "...": "...", "pe_import_markers": ["cublas64_"] },
        "hip":  { "...": "...", "pe_import_markers": ["amdhip64_"] }
      }
    }
  }
}
```

Only `windows-x86_64` exists today (macOS uses Metal and has no kernel
switch; the "Inference kernel" section of Advanced Settings is Windows-only
in the UI for the same reason). A future platform key would need matching
support added to the downloader, not just the manifest.

Unrecognized top-level fields are rejected (`#[serde(deny_unknown_fields)]`)
so a manifest shape drift fails loudly instead of silently dropping data a
future field depends on.

### `urls`

Try-in-order. `dl.openasr.org` is listed first (CDN-fronted, expected common
case); the GitHub Releases download URL is the fallback if the first fails
(network error, non-2xx, or a body that fails the sha256/PE checks below).
Every URL in the list must point at bit-identical bytes -- there is exactly
one `sha256` per backend entry, not one per URL.

### `pe_import_markers`

After extracting the archive, the downloader inspects the resulting
`openasr.exe`'s PE **import table** (the list of DLLs it declares it will
load at process start, e.g. via `DllImport`/`LoadLibrary`-at-startup-style
static imports -- not a runtime `dlopen`/`LoadLibraryW` call, which would not
appear here). The check passes if **at least one** imported DLL name
case-insensitively starts with **at least one** marker in the list, e.g.
`cublas64_` matches an import named `cublas64_12.dll`.

This is a linkage sanity check, orthogonal to the `sha256` check:

- `sha256` catches a corrupted/tampered/wrong-bytes download (transport
  integrity).
- `pe_import_markers` catches "the bytes are intact but this is not actually
  a CUDA-linked build" -- e.g. a build-pipeline mixup that uploaded the
  vulkan archive's contents under the cuda asset name with a correctly
  computed (for those wrong bytes) sha256. Both checks are mandatory; neither
  substitutes for the other.

## Signing

Signature verification lives in `crates/openasr-core/src/backends_manifest_security.rs`
(module `backends_manifest_security`, re-exported at the crate root as
`verify_backends_manifest_signature` / `render_backends_manifest_signature`),
**not** in `backend_manifest.rs`. It reuses the model catalog's production
signing key and trust root (`catalog_security::OPENASR_CATALOG_TRUST_ROOTS`,
key id `openasr-catalog-v1`) -- one signing seed, one trust root, one place a
maintainer manages key custody -- but signs under its own
domain-separation label, `openasr.backends_manifest.v1` (vs. the catalog's
`openasr.catalog_manifest.v1`), so a catalog signature can never be replayed
as a backends-manifest signature or vice versa even though both verify under
the same public key. The signature envelope is its own shape (`schema_version`,
`manifest_url`, `manifest_sha256`, `signature{algorithm,key_id,value}`) --
notably no `catalog_epoch`: unlike the catalog, this manifest has no shared
mutable endpoint a stale signature could roll back (it is generated fresh per
immutable, version-namespaced release URL), so it carries no anti-rollback
epoch.

`backend_manifest.rs` (this manifest's JSON *schema* -- `BackendsManifest`,
`PlatformBackends`, `BackendEntry`, the `core_version` match rule, and the
per-backend sha256 helper) calls into `backends_manifest_security` for the
signature check and never re-implements Ed25519 verification itself.
Verification entry point: `openasr_core::backend_manifest::verify_and_parse`
(or `verify_and_parse_for_core_version`, which additionally enforces the
version-match rule below). Both are fail-closed: a missing signature file, a
tampered manifest, a signature that does not verify, or an unsupported
`schema_version` (anything other than `1`) all return `Err` with no
partial-trust fallback. See `crates/openasr-core/src/backend_manifest.rs`'s
module doc comment and unit tests for the exact failure modes covered.

Signing itself stays a local, maintainer-run operation via the hidden CLI
subcommand `__openasr-sign-backends-manifest` (the signing seed never enters
CI) -- see `backends_manifest_security.rs`'s module doc comment.

## Version-matching rule

The desktop app determines "my own `core_version`" by running its
**currently active** sidecar binary with `--version` (clap's
`#[command(version)]` prints `openasr <CARGO_PKG_VERSION>`, e.g.
`openasr 0.1.14`) and parsing out the version. It then only ever accepts a
`backends-manifest.json` whose `core_version` field matches that string
exactly. A manifest for any other core version is rejected before any
download starts -- this is what stops a desktop build from switching to a
kernel binary built against an incompatible `openasr-server`/API surface.

The same probe (`<exe> --version`) is run again, this time against the
**downloaded** archive's extracted `openasr.exe`, as the last step of the
verification chain (after sha256 and the PE import-table check): its printed
version must also equal `core_version`. This closes the gap where a
manifest's declared `core_version` is honest but the actual asset bytes
(perhaps re-served from a stale CDN cache, or hand-edited urls array) are
not what the manifest describes.

## Desktop sidecar CLI contract

The manifest only decides which archive to fetch. Once extracted and
verified, desktop launches the resulting `openasr.exe` exactly the way it
already launches its bundled sidecar: `serve --backend native --parent-pid
<pid> ...` (see `apps/desktop/src-tauri/src/sidecar.rs`). The hidden
`--backend`/`--parent-pid` `serve` flags are a separate concept from the
kernel choice here -- they select the *transcription backend* (`mock` vs.
`native`) and the parent-process watchdog pid, and are identical across all
three kernel builds. `crates/openasr-cli/src/main.rs`'s
`serve_accepts_desktop_sidecar_contract_flags` test locks their presence and
shape so a future "clean up hidden flags" pass cannot silently break every
desktop build's ability to launch its sidecar.
