# Model Catalog, Registry, and Distribution

This note defines the current model-distribution catalog ownership chain, the
`openasr pull` install mechanics, and the local registry cards. For current
product behavior, see [Roadmap](ROADMAP.md) (Implemented-baseline section).

## Invariants

- OpenASR ships zero model weights in the app or CLI distribution. The desktop
  bundle carries the sidecar binary and registry/catalog metadata only.
- `openasr-core::pull` is the only download/install engine. The CLI, daemon,
  and desktop Models page call into that path; the webview never downloads
  artifacts directly.
- No silent downloads. `serve`, API transcription, `doctor`, the shared
  resolve path, and default tests never download models. The CLI `transcribe` /
  `live` handlers install a missing model only through a visible consent prompt,
  and fail closed when non-interactive or `--offline`; tests stay offline by
  passing `--backend mock`.
- A downloaded pack must be verified before execution: HTTPS catalog pack URL,
  pinned pack URL validation, size/sha256 match, Rust GGUF preflight, runtime
  source validation, then same-directory atomic rename into the installed pack.
- Public catalog entries require a public-listing gate. License metadata labels
  user-visible behavior; gated/vendor models require pull-time link-out UX
  rather than silent re-hosting.

## Ownership Chain

The catalog has three tiers:

1. Human-edited publishing catalog:
   `tooling/publish-model/models-core.toml`,
   `tooling/publish-model/models-publish.toml`, and the shared series taxonomy
   `crates/openasr-core/catalog-series.toml`.
2. Generated artifacts:
   `model-registry/catalog.json`, per-model cards under
   `model-registry/models/*.toml`, and the published Hugging Face model cards.
3. Consumers:
   core registry/catalog parsing, `openasr pull`, daemon catalog/local/pull
   endpoints, desktop Models install, and website catalog rendering.

The human-edited file is the source of truth for model identity, upstream
source, import subcommand, destination HF repo, model size token, registry id,
license fields, quantization set, and recommended quant. Series aliases,
member sizes, and default sizes live in
`crates/openasr-core/catalog-series.toml`; core resolution and the publish
catalog reader both consume that same taxonomy. Generated files must not become
independent truth. If a generated catalog or registry card drifts from the
publishing catalog, regenerate it from the publishing pipeline instead of
patching the generated file by hand.

## Generated Catalog

`model-registry/catalog.json` is the machine-readable pull catalog. It keeps
`schema_version = 1` and a flat `models[]` array. Each entry carries an explicit
`kind`: `asr-model` for transcription models, `translation-model` for standalone
local translation packs, or `capability-pack` for auxiliary packs. Translation
models carry explicit `source_langs` and `target_langs` metadata and are not
modeled as capability packs: they have independent licenses, revisions,
quantization choices, storage/memory budgets, and release gates. Capability packs
also carry
`capability = { feature = "speaker-diarization", role = "speaker-embedder" |
"speaker-segmenter" }`. All entries still carry ids/aliases, license metadata,
public visibility, recommended quant, and per-quant pack entries with pull
tokens, filenames, HTTPS URLs, sha256, size, and performance metadata.

`public` means published/downloadable/importable. It is not the model-market
predicate. The Rust market-list helper is `CatalogModel::is_market_listed()`,
defined as `public && kind in {asr-model, translation-model}`; capability packs
may be `public:true` so they can be pulled/imported while staying out of ASR
model listings. UI consumers should still partition the market by `kind` so
translation models are visible installable items without appearing in the default
ASR model selector.

The catalog is consumed by `openasr pull <id>:<quant>` and by bare
`openasr pull <id>`, which resolves to the recommended quant. ASR models,
translation models, and public capability packs are pullable by digest-verified
catalog entries. Pulling a translation model installs a reusable text-to-text
runtime pack; it does not change the default ASR model.

Local registry cards under `model-registry/models/*.toml` remain the local model
metadata surface for list/config/API-id validation and native pack selection.
They are related to the catalog but do not authorize an implicit runtime fetch.

A model can be staged in `tooling/publish-model/models-core.toml` before any
public artifact exists. While a source entry is not `release_public`, it must not
enter the signed public projection: a real `.oasr` pack has to be built, its
sha256/size sidecars generated, the Hugging Face revision recorded, and
the public-listing gate has to pass first. The pack must embed the upstream
license file and the OpenASR `NOTICE.openasr.txt` modification notice declared by
the publish metadata. `regenerate_all.sh --check` supports a source-only staged
state by warning that no generated `catalog.json` entry exists yet. To promote a
staged model to a full-catalog entry, build its pack under
`tmp/publish/<id>/packs/`, run
`python3 tooling/publish-model/scripts/materialize_result_sidecars.py <id> --quant <quant>`,
record the Hugging Face revision in `tmp/publish/<id>/hf_revision.txt`, then run
`tooling/publish-model/scripts/regenerate_all.sh <id>`. Do not pass `--public` or
add `release_public = true` until the public-listing gate passes.

## Local registry cards

The local registry is the TOML card set under `model-registry/models/*.toml`. The
cards are local metadata only; they do not install artifacts and do not authorize
any implicit runtime fetch. They back `openasr list`, config / default-model
validation, API model-id validation, `openasr pull` catalog
validation/resolution, and native model-id / family / variant selection for local
`.oasr` packs. `variant.*` is local pack-selection metadata (`model[:tag]`), not
remote artifact routing. The committed card set (one or more per bundled family
plus capability/translation packs) is the source of truth — read
`model-registry/models/` rather than maintaining a duplicate list here.

## Pull and install mechanics

`openasr pull` is the explicit, user-initiated install path for published packs.
The same core pull engine backs three surfaces:

- CLI: `openasr pull <id>:<quant>` (or a bare `<id>` for the recommended quant).
- Daemon: `POST /v1/models/{id}/pull`, `GET /v1/models/pull/{job_id}`, and the
  pull SSE endpoint. `GET /v1/models/pulls` (operator-only) lists all
  currently non-terminal jobs -- read-only, so a client that lost its
  in-memory job list (e.g. the desktop shell after a daemon restart) can
  rediscover in-flight downloads without re-triggering them.
- Desktop: the Models page installs through the local daemon, never from the
  webview.

Pulling a `capability-pack` (e.g. `wespeaker-voxceleb-resnet34-lm:f32`) or a
`translation-model` does not change the default ASR model. `openasr transcribe
--diarize` and `live --diarize` are explicit consent for the
CLI to install a missing required `speaker-diarization` capability pack before the
fail-closed capability check; `serve` / `session.start` never download. The
default CLI `transcribe` / `live` flow installs a missing ASR model only with a
visible consent prompt (or fails closed when non-interactive / `--offline`);
`serve` and the shared resolve path never execute downloads. The pull path is
fail-closed: HTTPS-only catalog pack URLs, size/sha256 checks, GGUF preflight,
runtime-source validation, and a same-directory atomic rename are required before
a pack counts as installed, and untrusted catalog pack filenames must be
relative basename-only `.oasr` targets.

The public anonymous distribution path is exercised by
`tooling/public-hf-e2e/run.sh` and the manual/scheduled `public-hf-e2e` workflow,
which pull a real public pack into an isolated `OPENASR_HOME` and transcribe with
the native backend (kept outside push/PR CI because it downloads and runs real
models). Local development / benchmark workflows may stage artifacts under
`./tmp/` with provenance recorded (source identity, revision/path, SHA256, size,
mirror endpoint if used); do not commit downloaded artifacts.

## Hosting: Cloudflare Catalog Endpoint

The signed **public** catalog projection is hosted on OpenASR's own host,
`catalog.openasr.org` (a Cloudflare Worker + Static Assets under
`cloudflare/catalog/`), not on Hugging Face — Hugging Face hosts model **weights**
only, and the HF catalog repo is no longer required to serve clients. Only the
`public:true` projection (`catalog.public.json`) is hosted; staged `public:false`
entries are never exposed. Public capability packs remain in that projection
because `public` is the download/import gate, not the ASR market-list gate. The
catalog's URLs stay pinned to `huggingface.co` as
the signed, canonical *identity*; the client rewrites only the transport *host*
via `http::apply_catalog_endpoint`, controlled by `OPENASR_CATALOG_ENDPOINT`
(default `https://catalog.openasr.org`; override only for a self-host/mirror).
Because the signed `catalog_url` is unchanged, the host is an availability layer,
not a trust anchor — moving it needs no re-sign or signing seed. This is
independent of `HF_ENDPOINT`, which routes weight downloads only. Deploys are
automated by the `deploy-catalog` workflow on push; signing stays local (the seed
never enters CI).

## Cache And Rollback Boundary

The catalog cache is a signed fetch-on-error fallback. On a successful HTTPS
fetch, OpenASR fetches the adjacent `catalog.signature.json`, verifies the
Ed25519 signature against the built-in OpenASR catalog key, rejects catalog epoch
rollback, validates the catalog, and writes the exact validated contents to
`$OPENASR_HOME/catalog.json`. It also caches
`$OPENASR_HOME/catalog.signature.json` and records the highest accepted epoch in
`$OPENASR_HOME/catalog.epoch`.

If the next HTTPS fetch fails, OpenASR attempts to load only that signed local
cache. If the signed local cache is also unavailable, it falls back to a catalog
snapshot **embedded in the binary** at build time (`include_str!` of the committed
PUBLIC projection `catalog.public.json` + `catalog.public.signature.json` — never
the full catalog, so no staged `public:false` entries ship), verified through the
same Ed25519 signature and anti-rollback epoch checks and scoped to the default
catalog (an explicit `OPENASR_CATALOG_URL` override is honoured, not replaced).
This guarantees a fresh, fully offline install still shows the verified model
list; because every installer ships the sidecar binary, the offline catalog is
bundled transitively with no per-installer packaging. The embedded snapshot is
kept current by the catalog drift and bundled-signature CI gates. Current trust
comes from HTTPS, signature verification, anti-rollback epoch checks, schema
validation, pinned immutable pack URLs, sha256/size verification, Rust GGUF
preflight, runtime-source validation, and atomic install.

A LOCAL (`file://` or bare filesystem path) `catalog_url` override -- CLI
`--catalog-url`/`OPENASR_CATALOG_URL`, the server's equivalent, or the CLI's
repo-checkout auto-discovery of `model-registry/catalog.json` with no override
set -- goes through the same signature/schema/anti-rollback pipeline as an
HTTPS catalog: there is no unsigned local path. Trust roots are chosen from
the *identity a signature is checked against* (`catalog_security::classify_catalog_identity`),
not merely from how the bytes were read:

- A production (`https://`) identity -- including the repo-checkout
  auto-discovery of `model-registry/catalog.json`, which is verified against
  the canonical `DEFAULT_CATALOG_URL` identity, not its incidental local path
  -- accepts **only the production key**. A widely-known dev key must never
  be able to stand in for the canonical production catalog just because a
  malicious CWD happens to contain a `model-registry/catalog.json` +
  `catalog.signature.json` pair.
- Any other (local) identity -- i.e. an explicit `file://<path>` override via
  `--catalog-url`/`OPENASR_CATALOG_URL` -- additionally trusts a public,
  non-secret **local-dev signing key** (`openasr-catalog-local-dev-v1` /
  `LOCAL_CATALOG_DEV_SIGNING_KEY_SEED_HEX` in
  `crates/openasr-core/src/catalog_security.rs`). That key carries no
  confidentiality (whoever supplies a local catalog file already controls its
  contents); it only forces every local catalog through real
  signature/sha256/catalog_url verification instead of a bypass.

A signature is bound to the exact catalog_url identity it was issued for (an
HTTPS URL for a production catalog, or the literal `file://<path>` for an
explicit override) -- copying a signed local catalog to a different path/URL
does not carry its signature with it.

A local-dev-key-verified catalog also never touches the shared, cross-source
anti-rollback epoch floor in `$OPENASR_HOME/catalog.epoch` (neither reading it
as a floor nor writing to it): that floor exists to protect genuine production
distribution channels (HTTPS, the on-disk signed cache, the embedded offline
snapshot) from a stale re-serve, and the dev key's self-signed preview content
has no such channel to protect -- letting it participate would let one
locally-signed catalog with an inflated epoch permanently brick every
subsequent production catalog load until `catalog.epoch` was deleted by hand.

To preview local/staged catalog edits (e.g. after `regenerate_all.sh`) without
the real production signing seed, run
`tooling/publish-model/scripts/sign_local_catalog.sh` to sign a dev copy bound
to an explicit `file://<path>` identity, then load it with
`OPENASR_CATALOG_URL=file://<path>` (the repo-checkout auto-discovery path no
longer accepts the dev key, since it asserts the production identity). Never
commit a dev-signed manifest over the committed, production-signed
`catalog.signature.json`.

## Forward Compatibility

Each catalog model carries a `min_cli_version`. A model that requires a newer
OpenASR than the running build does **not** fail catalog loading and is **not**
hidden: the whole catalog still loads, and the model is surfaced via
`CatalogModel::availability()` as `RequiresUpdate` so the model market can list it
with an "update to use" badge. Actually pulling such a model is refused at resolve
time with a clear "requires OpenASR >= X" error rather than downloading a pack the
build cannot run. Only a malformed `min_cli_version` (not a merely too-new one) is
a catalog validation error.

## Consumer Rules

- Consumers resolve models through catalog/registry APIs; they do not hand-edit
  catalog truth.
- Download surfaces are explicit only: `openasr pull`, daemon pull API, and the
  desktop Models install path.
- Runtime surfaces accept local `.oasr` paths and fail closed on remote URLs,
  directories, invalid extensions, missing files, invalid runtime metadata, or
  invalid tensor/layout preflight.
- `hf-mirror` may be a transport fallback during publishing workflows, but it is
  never a trust anchor for client-side execution.
