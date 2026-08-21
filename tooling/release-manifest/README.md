# Legacy release backends manifest

> Historical compatibility only. Core 0.1.33 is the final release allowed to
> use this manifest and whole-engine Windows GPU sidecars. Core 0.1.34 and
> later publish a neutral BACKEND_DL host plus target-scoped CUDA/HIP backend
> packs through the production-signed catalog. Do not invoke this manifest
> generation/signing flow for new releases.

Generates `backends-manifest.json`: the per-release index the desktop app
reads to download a switchable Windows GPU-kernel sidecar (vulkan / cuda /
hip) at runtime, instead of shipping every backend in the base install.

Since core `0.1.20`, this is `schema_version: 2`: `cuda`/`hip` ship as a small
per-release "sidecar" archive (`openasr.exe` + docs, named
`openasr-<version>-windows-x86_64-{cuda,rocm}-sidecar.zip`) that references a
separate, large, content-addressed `vendor_layers` archive (NVIDIA's/AMD's GPU
runtime DLLs, named `openasr-vendor-{cuda,rocm}-runtime-<sha12>.zip`) shared
across every core release that pins a compatible toolchain. `vulkan` stays
self-contained (no vendor layer -- same shape as a v1 manifest's `vulkan`
entry). See `crates/openasr-core/src/backend_manifest.rs`'s module doc and
`docs/backend-kernels.md` for the full schema and the disk-layout/install-order
this unlocks on the desktop side.

- `backends_manifest.py` -- assembles the UNSIGNED manifest from already-built
  release archives and vendor-layer archives (reads bytes, computes sha256 +
  size; locates each vendor archive by glob since its filename embeds its own
  content hash, then sanity-checks that embedded short hash against the freshly
  computed full sha256). No secret involved;
  `.github/workflows/release-binaries.yml`'s `checksums` job runs this in CI
  once the Windows GPU sidecar + vendor archives are staged, and uploads the
  result as both a workflow artifact and (on a real tag release) a
  `backends-manifest.json` release asset.
- `deterministic_zip.py` -- packages a directory into a byte-deterministic zip
  (fixed entry order + fixed `date_time`/permission bits), used ONLY for the
  vendor-layer archive (PowerShell's `Compress-Archive`, used for every other
  archive, is not byte-deterministic, which would break vendor-layer dedup by
  its own sha256 -- see that script's module doc).
- `backends_manifest_test.py` / `deterministic_zip_test.py` -- fixture-driven
  unit tests (temp dir + fake archive bytes, no network). Run with:

  ```bash
  python3 -m unittest discover -s tooling/release-manifest -p '*_test.py'
  ```

## Signing (LOCAL ONLY -- never in CI)

Exactly like the model catalog
(`tooling/publish-model/scripts/publish_catalog.sh`), the Ed25519 signing
seed for `backends-manifest.signature.json` never enters CI. It reuses the
SAME production key/trust root as the catalog
(`openasr-catalog-v1` -- see `crates/openasr-core/src/backends_manifest_security.rs`
for why one keypair covers both).

**Run the one atomic script** (see `RELEASING.md`'s "Backends-manifest
signing" section for the full contract) instead of the individual commands
below by hand:

```bash
OPENASR_CATALOG_SIGNING_KEY_SEED_HEX=<real production seed> \
  scripts/sign-and-verify-backends-manifest.sh v<version>
```

It downloads the unsigned manifest from the release, signs it, uploads the
signature, then re-downloads and re-verifies the published pair against the
production trust root -- aborting loudly (`SIGNING/VERIFY FAILED`) on the
first failure instead of leaving the release half-signed. This replaces the
three manual commands that used to live here (download / sign / upload),
which is exactly the multi-step-that-gets-forgotten shape that shipped core
0.1.16-0.1.19 with a never-signed manifest.

Under the hood, step 1 is still `__openasr-sign-backends-manifest`, run
equivalently to:

```bash
gh release download v<version> -p backends-manifest.json -O backends-manifest.json

OPENASR_CATALOG_SIGNING_KEY_SEED_HEX=<real production seed> \
  cargo run --quiet -p openasr-cli -- __openasr-sign-backends-manifest \
    backends-manifest.json --out backends-manifest.signature.json \
    --manifest-url https://dl.openasr.org/core/v<version>/backends-manifest.json

gh release upload v<version> backends-manifest.signature.json --clobber
```

`__openasr-sign-backends-manifest` self-verifies the signature it just
produced against the production trust root before writing it out, so signing
with anything other than the real production seed fails loudly instead of
silently writing an unverifiable signature. Only fall back to running these
individual commands by hand to debug a failure of the script itself.

**`--manifest-url` must be the CANONICAL URL** -- exactly what
`openasr_core::backend_manifest::canonical_manifest_url(core_version)` returns
(`https://dl.openasr.org/core/v<version>/backends-manifest.json`), never
whichever mirror/base URL a maintainer happens to be testing against. This
function is the single source of truth on BOTH sides of the signature: the
signing step here binds the signature to it, and every desktop fetch path
must pass the same canonical string as `expected_manifest_url` regardless of
which mirror (`dl.openasr.org` direct, the China-accel proxy, or the GitHub
Releases fallback) it actually downloaded the bytes from -- using the real
per-mirror fetch URL there instead is the bug this fixes (#145: every mirror
except the primary CDN failed signature verification, since the signed
payload only ever names the canonical host).

## Legacy dl.openasr.org sync (0.1.33 and earlier only)

For core 0.1.34 and later, do not use the whole-engine example below. Run
`scripts/sync-windows-backend-cdn.sh vX.Y.Z`; it derives the exact
hardware-approved target-scoped files from release metadata, uploads them, and
HEAD-verifies the canonical CDN objects before catalog signing is allowed.

`b2_sync.py` -- uploads files to `core/v<version>/<filename>` in the SAME
Backblaze B2 bucket / Cloudflare Worker the desktop installers
already publish to (`https://dl.openasr.org/desktop/...` there,
`https://dl.openasr.org/core/...` here). It is a Python port of that desktop
product's SigV4 signer (same env var names,
same virtual-hosted-style request shape, same ETag-based immutability gate)
plus its upload-with-immutability-check logic --
cross-validated against AWS's published SigV4 worked example in
`b2_sync_test.py`, no network required to test.

The B2 write credentials are kept **only** in `~/.openasr/b2-release.env`
(chmod 600, outside the repo) -- never in `~/.zshrc` / `~/.bashrc` or any
long-lived shell profile (a release-bucket write key must not be inherited by
every terminal process), and never as a CI secret / repository variable /
workflow literal for this script (core's sync is a local, human-run step).
`source` that file in the shell that runs the sync, then run the command:

```bash
source ~/.openasr/b2-release.env   # provides B2_S3_ENDPOINT / B2_APPLICATION_KEY_ID / B2_APPLICATION_KEY (and optional B2_BUCKET / B2_S3_REGION); never logged

python3 tooling/release-manifest/b2_sync.py sync --version 0.1.20 \
  dist/openasr-0.1.20-windows-x86_64-vulkan.zip \
  dist/openasr-0.1.20-windows-x86_64-cuda-sidecar.zip \
  dist/openasr-0.1.20-windows-x86_64-rocm-sidecar.zip \
  dist/backends-manifest.json \
  dist/backends-manifest.signature.json
```

**Vendor-layer archives are OPTIONAL and separate** (`sync-vendor`, not
`sync` -- they have no per-version key, since they are version-independent
and content-addressed):

```bash
python3 tooling/release-manifest/b2_sync.py sync-vendor \
  dist/openasr-vendor-cuda-runtime-<sha12>.zip \
  dist/openasr-vendor-rocm-runtime-<sha12>.zip
```

**Decided policy**: these vendor archives are large (several hundred MB each)
and GitHub Releases is the primary distribution point for them --
`release-binaries.yml` already uploads them there as a normal release asset,
so a release is fully usable without ever running `sync-vendor`. Syncing them
to B2/dl.openasr.org too is a purely optional CDN-fronting step a maintainer
can run later if `dl.openasr.org`'s speed/reliability is specifically wanted
for this layer; it is not part of the release-blocking checklist below.

This is deliberately **NOT wired into any GitHub Actions workflow**. Publish
is always a local, human-run step:

- **Decided**: core's release assets share the SAME `openasr-releases` B2
  bucket the desktop installers already publish to, under a `core/v<version>/`
  key prefix (`B2_BUCKET` defaults to `openasr-releases`; override only if the
  bucket is ever split). Credentials (`B2_APPLICATION_KEY_ID` /
  `B2_APPLICATION_KEY`) stay out of CI -- this sync always runs from a
  maintainer's machine, with the B2 vars provided by `source
  ~/.openasr/b2-release.env` in that shell (the same credential file desktop
  releases use), not a repo secret and not a long-lived shell-profile export.
- The desktop product's own release workflow only runs this kind of publish
  from a `workflow_dispatch` with an explicit `publish: true` input, gated by
  repo secrets on that product repo -- i.e. even there, publishing to
  dl.openasr.org is a deliberate, per-run opt-in, not a side effect of every
  green build. Core follows the same posture, taken one step further: not
  even a gated dispatch, just a local script run after the maintainer has
  reviewed the release.
- Publishing to a public, production distribution endpoint is a
  release/deploy decision this script does not make on its own. If core
  releases ever need CI-driven publish, that is a separate, explicit
  go-ahead (which secrets, which trigger) -- not a default.

## Current release checklist

For core 0.1.34 and later, the manifest workflow in this document is retired.
Follow the authoritative checklist in `RELEASING.md`: hardware evidence, B2/CDN
payload sync, catalog preparation/signing, catalog deployment (with a repeated
CDN gate), then draft finalization (with the third CDN gate). Metadata must
never become live before every URL it signs.
