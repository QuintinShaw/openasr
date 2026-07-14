# Release backends manifest

Generates `backends-manifest.json`: the per-release index the desktop app
reads to download a switchable Windows GPU-kernel sidecar (vulkan / cuda /
hip) at runtime, instead of shipping every backend in the base install.

- `backends_manifest.py` -- assembles the UNSIGNED manifest from already-built
  release archives (reads bytes, computes sha256 + size). No secret involved;
  `.github/workflows/release-binaries.yml`'s `checksums` job runs this in CI
  once all three Windows GPU archives (`-vulkan`, `-cuda`, `-rocm`) are staged,
  and uploads the result as both a workflow artifact and (on a real tag
  release) a `backends-manifest.json` release asset.
- `backends_manifest_test.py` -- fixture-driven unit tests (temp dir + fake
  archive bytes, no network). Run with:

  ```bash
  python3 -m unittest discover -s tooling/release-manifest -p '*_test.py'
  ```

## Signing (LOCAL ONLY -- never in CI)

Exactly like the model catalog
(`tooling/publish-model/scripts/publish_catalog.sh`), the Ed25519 signing
seed for `backends-manifest.signature.json` never enters CI. It reuses the
SAME production key/trust root as the catalog
(`openasr-catalog-v1` -- see `crates/openasr-core/src/backends_manifest_security.rs`
for why one keypair covers both). After a release finishes:

```bash
# Download the CI-generated unsigned manifest from the release, or from the
# `backends-manifest` workflow artifact.
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
silently writing an unverifiable signature.

## dl.openasr.org sync

`b2_sync.py` -- uploads files to `core/v<version>/<filename>` in the SAME
Backblaze B2 bucket / Cloudflare Worker `openasr-app`'s desktop installers
already publish to (`https://dl.openasr.org/desktop/...` there,
`https://dl.openasr.org/core/...` here). It is a Python port of that repo's
`apps/desktop/scripts/b2-s3-client.mjs` SigV4 signer (same env var names,
same virtual-hosted-style request shape, same ETag-based immutability gate)
plus `release-publish.mjs`'s upload-with-immutability-check logic --
cross-validated against AWS's published SigV4 worked example in
`b2_sync_test.py`, no network required to test.

```bash
export B2_S3_ENDPOINT=https://s3.us-east-005.backblazeb2.com   # confirm the real value with whoever owns the B2 account
export B2_APPLICATION_KEY_ID=...
export B2_APPLICATION_KEY=...                                   # never logged

python3 tooling/release-manifest/b2_sync.py sync --version 0.1.10 \
  dist/openasr-0.1.10-windows-x86_64-vulkan.zip \
  dist/openasr-0.1.10-windows-x86_64-cuda.zip \
  dist/openasr-0.1.10-windows-x86_64-rocm.zip \
  dist/backends-manifest.json \
  dist/backends-manifest.signature.json
```

This is deliberately **NOT wired into any GitHub Actions workflow yet**. That
is a credential/scope decision, not a technical blocker:

- It is unconfirmed whether core's release assets should share the
  `openasr-releases` B2 bucket (and its credentials) with the desktop
  installers, or use a separate bucket/prefix-scoped key. Whoever owns the B2
  account needs to decide and, if sharing, mint a key scoped to `core/*` (B2
  application keys support per-prefix scoping) rather than reusing the
  desktop key wholesale.
- `openasr-app`'s own `release-desktop.yml` only runs this kind of publish
  from a `workflow_dispatch` with an explicit `publish: true` input, gated by
  repo secrets on the app repo -- i.e. even there, publishing to
  dl.openasr.org is a deliberate, per-run opt-in, not a side effect of every
  green build. The same posture should carry over here once wired in: an
  explicit, opt-in step or dispatch input, not an automatic push-tag action.
- Publishing to a public, production distribution endpoint is a
  release/deploy decision this script does not make on its own -- wiring it
  into CI (which secrets, which trigger, which bucket/prefix) needs an
  explicit go-ahead before it runs unattended.
