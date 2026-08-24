# Windows provider release and qualification authority

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
