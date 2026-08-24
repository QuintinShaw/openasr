# Windows provider release and qualification authority

## Exact-cell backend qualification

`qualification_manifest.py` compiles one inert manifest for every exact Windows
CUDA/HIP target plus one generic Vulkan provider artifact. The Vulkan manifest
does not claim a physical target at release time; the isolated runner derives
and binds the exact `vk_caps_*` identity from the real device. The compiler does
not infer a provider or target from an asset filename. Instead it joins and
cross-checks:

- the final Authenticode-signed neutral ZIP, including the exact `openasr.exe`,
  complete unpacked tree, and host ABI;
- one `backend-pack-*.json` candidate and the exact plugin/vendor bytes it
  declares, including the signed Vulkan plugin and loader archive;
- the successful `actions/attest-build-provenance` output bundle and its subject
  digests.

ZIPs are streamed rather than extracted. The compiler rejects traversal,
Windows-unsafe names, case collisions, encrypted/non-regular entries, byte/tree
drift, host/plugin ABI drift, and an attestation missing any referenced release
subject. It computes both the ordinary backend archive tree with its `vendor/`
install prefix and the qualification namespace's empty-root tree; those are
different identities and must not be substituted for one another.

`.github/workflows/release-binaries.yml` copies the `bundle-path` from the
successful attestation attempt to the stable release asset
`openasr-<version>-build-provenance.bundle.json`, then emits collision-free
`openasr-<version>-qualification-<exact-cell>.json` files. These files and the
bundle are publication metadata only: they are not appended to the already
attested subject set, contain no activation mode, and never enter the ordinary
backend catalog.

The workflow never receives the production Ed25519 seed. While the GitHub
release is still a draft, a maintainer checks out its exact tag and runs the one
local atomic command:

```bash
OPENASR_CATALOG_SIGNING_KEY_SEED_HEX=<real production seed> \
  scripts/sign-and-verify-qualification-manifests.sh v<version>
```

The script requires a clean checkout at the exact tag, verifies every release
byte against its manifest and the offline Sigstore bundle, confirms the local
annotated tag object and peeled commit still equal GitHub's current tag, asks `gh attestation
verify` to enforce repository/workflow/source/predicate/non-self-hosted-runner
constraints, signs every exact cell required by the tagged release matrix
(currently 21) in the qualification-specific domain,
uploads only detached signatures, and re-downloads every pair for production-root
verification. A failure at any point leaves qualification incomplete; it never
publishes a catalog entry or activates a provider.

The signer, the formal Actions upload path, and the final `draft -> published`
transition acquire the same atomic draft-release asset lock before reading,
replacing, or publishing manifests/signatures. Lock ownership is nonce-bound,
and release completeness rejects a stale lock, so a local signing run cannot
race release-asset replacement or cross publication.
The finalizer rebuilds the exact manifest/signature set from the release's
backend packs and re-verifies artifact hashes, the qualification signature, and
Sigstore provenance while holding that lock. Manual workflow dispatch remains
diagnostic-only and cannot recover, replace, sign, or publish release assets.

Run generator and workflow contract tests with the shared release-manifest gate:

```bash
python3 -m unittest discover -s tooling/release-manifest -p '*_test.py'
```

This directory owns the open-core release metadata and the two evidence gates
for the terminal Windows topology: one CPU-neutral `GGML_BACKEND_DL` host plus
optional signed Vulkan, CUDA, and HIP provider packs. It does not own Desktop
UX and it does not provide an alternate whole-engine sidecar switch.

## One catalog, two evidence gates

`backend_catalog.py` constructs, merges, and verifies provider entries and CDN
payloads. `backend_target_identity.py` is the shared target vocabulary:

- CUDA: one exact `sm_XX` or `sm_XXX` target;
- HIP: one exact `gfxXXXX` target (including a permitted trailing letter); and
- Vulkan: one `vk_caps_<vendor-id>_<device-id>_<pipeline-cache-uuid>` capability
  class, with 8, 8, and 32 lowercase hexadecimal digits respectively. The
  exact driver version remains receipt-bound.

Qualification has exactly two class-separated authorities:

- `backend_hardware_evidence.py` verifies release subjects, `SHA256SUMS`, build
  and qualification provenance, the exact provider/target/backend id and
  artifact tree, fresh-process nonces, FullDevice placement, and the absence of
  CPU fallback. It does not prove model token correctness.
- `gpu_correctness_gate.py` projects the matrix from the architecture inventory,
  model catalog, and backend catalog, then validates cold/reuse CPU-oracle and
  GPU receipts/traces for an exact `(provider, device_target, backend_id)` cell.
  It binds the release, executable, plugin, pack, fixture, catalogs, and trace
  bytes. It does not replace placement/resource evidence.

Neither gate can broaden evidence across targets or providers. In particular,
an `sm_89` receipt cannot qualify `sm_75`, and CPU, Metal, HIP, or Vulkan
evidence cannot close a CUDA cell.

## State machine

```text
PublishedInert
  -> Qualified
  -> Activated
  -> Revoked
```

`PublishedInert` bytes are signed and public but unavailable to ordinary Auto
or explicit runtime selection. `Qualified` binds exact hardware and release
provenance but has no token-correctness authority. `Activated` additionally
binds the complete correctness matrix and is the only selectable state.
`Revoked` is fail-safe and one-way; it preserves prior bindings for audit while
remaining unselectable.

The activation preparation script validates a distinct Qualified intermediate
projection before deriving Activated. Both transitions may be signed into one
reviewed post-publication catalog epoch; a separately deployed Qualified epoch
is not required. A revoked backend cannot be requalified or reactivated.

## Supported entrypoints

Do not hand-edit provider entries or activation bindings.

```bash
# Before publishing a core release: make every exact provider byte public but inert.
scripts/sync-windows-backend-cdn.sh vX.Y.Z
scripts/prepare-windows-backend-catalog-release.sh vX.Y.Z
OPENASR_DEPLOY_CATALOG_RUN_ID=<release-core-run-id> \
  scripts/finalize-core-release.sh vX.Y.Z

# After exact-tag hardware qualification: prepare one reviewed activation epoch.
scripts/activate-windows-backend-catalog-release.sh \
  vX.Y.Z BACKEND_ID QUALIFICATION_RUN_ID [RUN_ID ...]

# Prepare a one-way exact-backend fail-safe revocation.
scripts/revoke-windows-backend-catalog-release.sh vX.Y.Z BACKEND_ID
```

The CDN sync writes the immutable provider payloads to B2 but does not change a
catalog or GitHub release. Catalog preparation writes only the five local
catalog/epoch files; it does not commit, push, deploy, publish, or activate
anything. The finalizer only publishes a draft whose already-deployed catalog
exposes the release provider entries as PublishedInert.

Real-hardware evidence must come from a tag-scoped dispatch of
`.github/workflows/qualify-windows-backend.yml` with exact `release_tag`,
`provider`, `runner_label`, `device_target`, `backend_id`, `model_id`, and
`quant`. The workflow produces attested evidence but never mutates production.

After reviewing and committing the catalog epoch produced by the activation or
revocation script, deployment still requires a separate manual dispatch:

- `.github/workflows/activate-backend-catalog.yml` requires
  `activate:<backend_id>` and the complete qualification run-id set.
- `.github/workflows/revoke-backend-catalog.yml` requires
  `revoke:<backend_id>`.

Only reusable `.github/workflows/deploy-catalog.yml` writes the public catalog.
For PublishedInert release publication it records `deploy-catalog-binding.json`;
the finalizer verifies its tag, orchestrator/deploy run, source commit, catalog
SHA, and signature SHA, then independently compares the live bytes and CDN.
Activation and revocation calls replay their exact transition before the same
deployment seam. Qualification success alone never deploys anything.

## Local contract checks

```bash
python3 -m unittest discover -s tooling/release-manifest -p '*_test.py'
python3 tooling/release-manifest/backend_catalog.py --help
python3 tooling/release-manifest/backend_hardware_evidence.py --help
python3 tooling/release-manifest/gpu_correctness_gate.py --help
```

`backends-manifest.json`, its signing script, whole-engine Windows GPU
sidecars, and the Desktop legacy kernel store/loader are retired authority.
They must not be restored as a fallback release or activation path.
