# Catalog Forward Compatibility and Client Resilience

This is the normative contract for how a running OpenASR build must handle a
model catalog it did not ship with -- new taxonomy values, new language codes,
a corrupted/incompatible cached copy, or a locally stale anti-rollback floor.
It complements [Model Catalog, Registry, and Distribution](MODEL_CATALOG_ARCHITECTURE.md)
(ownership chain, signing, `openasr pull`) with the specific "what happens when
this build sees data from a *different* catalog epoch" question, motivated by
a real incident (see "Incident background" below).

## The contract

### Must stay fail-closed (security/integrity boundary -- unchanged by this doc)

- **Signature verification failure.** Every catalog source (network, on-disk
  cache, embedded snapshot, explicit local override) requires a valid
  `catalog.signature.json` sidecar under the correct trust root for its
  identity. No source ever skips verification.
- **Epoch rollback on a REMOTE-sourced trust tier.** A catalog fetched over
  the network, or the on-disk cache mirroring what a network fetch previously
  verified, must never be accepted at an epoch below this machine's recorded
  floor (`$OPENASR_HOME/catalog.epoch`). This is the actual attack surface: a
  compromised/malicious catalog endpoint (or MITM) replaying a stale catalog.
  See "Epoch floor at boot" below for the one *narrower* exception this PR
  adds, which is NOT a relaxation of this rule.
- **Structurally broken JSON.** Malformed JSON fails the parse; there is no
  partial/best-effort JSON repair.
- **`schema_version` major above what this build supports.** A structural
  version bump this build cannot interpret at all fails closed.
- **Missing required fields.** A field with no `#[serde(default)]` that is
  absent from the wire document fails the parse for the whole document (not
  scoped to one model) -- this is a "the document doesn't match this build's
  understanding of the format" signal, not a forward-compat case.

None of the above changed in this PR. `model-registry/`'s catalog *data*,
the signing keys, and the epoch/signature verification math are untouched.

### Must tolerate degradation (data evolution -- what this PR fixes)

- **Unknown language/dialect code.** `languages` (and `source_langs` /
  `target_langs`) is a plain `Vec<String>`; any code -- including one a future
  catalog epoch invents -- is preserved verbatim. It never gates model
  visibility and never fails the catalog. Display falls back to the raw code
  when no curated label exists (`crate::models::language::language_display_label`
  returns `None`); this is a display concern, not a catalog-loading one. The
  Rust-side `REGISTERED_DIALECT_CODES` / `validate_language_code` guard in
  `crates/openasr-core/src/models/language.rs` (mirrored in
  `tooling/publish-model/scripts/_catalog.py`) is an **authoring-time typo
  guard only** -- it runs when a maintainer adds a new dialect code to the
  catalog, so `zh-sichaun` fails loudly *before* a signed catalog ships. It
  must never be wired into the runtime catalog-loading path; doing so would
  reintroduce exactly the failure class this document fixes.
- **Unknown model `kind`, `license_class`, or capability `role`.** Each of
  these wire enums (`CatalogModelKind`, `LicenseClass`, `CatalogCapabilityRole`,
  and, for symmetry, `CatalogBackendVendor` / `CatalogBackendFileRole` /
  `CatalogLanguageMode`) carries a `#[serde(other)]` catch-all `Unknown`
  variant, so an unrecognized wire string never fails `serde_json::from_str`.
  `registry::filter_forward_compatible_catalog` then drops (hides) exactly the
  affected model or backend pack -- not the whole catalog -- with a one-line
  `eprintln!` diagnostic (`registry::parse_model_catalog` calls the filter
  right after deserializing, before the existing structural `validate_model_catalog`
  checks run). `license_class` and capability `role` can gate what a client is
  allowed to show/download/stage, so "hide" is the only safe degrade for those
  two; `kind` similarly determines dispatch (market listing vs. capability
  pack vs. translation model), so an unrecognized value must not silently
  masquerade as `asr-model`.
- **Unknown JSON object keys.** Neither `ModelCatalog` nor `CatalogModel`
  declare `#[serde(deny_unknown_fields)]`, so a future field this build
  doesn't know about is already ignored by serde's default behavior. (The
  security-critical signature envelope in `catalog_security.rs` --
  `CatalogSignatureManifest` / `CatalogSignature` -- deliberately keeps
  `deny_unknown_fields`: that is trust-boundary data, not catalog business
  data, and stays strict.)

### Epoch floor at boot (new in this PR, narrowly scoped)

The anti-rollback floor's job is to stop a **remote** source from replaying a
stale catalog. It was never meant to gate whether a **boot-local candidate**
-- the binary's own embedded snapshot, or an explicit local/bundled catalog
file (`OPENASR_CATALOG_FILE`/`OPENASR_CATALOG_IDENTITY`, the desktop's actual
`openasr serve` startup path) -- is allowed to boot the daemon at all. A
boot-local candidate can legitimately sit below a floor this same machine
previously recorded with no attack involved: an older release reinstalled
over a newer one (its embedded epoch predates what the newer build fetched),
or a dev/test tool populating `$OPENASR_HOME/catalog.epoch` from an unrelated,
newer catalog snapshot (the actual root cause below).

So, **only** for `load_embedded_signed_catalog` and
`load_local_catalog_file_with_identity` (`registry.rs`), a below-floor epoch
that is otherwise fully verified (signature, structure) degrades instead of
failing closed:

- `catalog_security::enforce_boot_catalog_epoch_for_verified` returns
  `BootEpochOutcome::BelowFloor { floor }` rather than
  `Err(EpochRollback)`.
- The caller logs a warning, records `CatalogDegradedStatus { tier, reason }`
  (see "Status surface" below), and serves the catalog anyway.
- The recorded floor is **never** pulled backward: a degraded load must not
  call `record_catalog_epoch_for_verified` for its own (lower) epoch. A
  later, genuinely fresher network catalog is still held to the real, unmoved
  floor.

The network/cache trust tier (`load_model_catalog`'s primary source,
`load_signed_catalog_from_cache`) keeps the strict, unchanged
`enforce_catalog_epoch_for_verified` -- a rollback there still fails that
*source*, though the overall `load_model_catalog` call still degrades to a
good cache/embedded catalog via the fallback chain below (that part is not
new; a below-floor primary source has always fallen through to
`load_cached_signed_catalog`).

Test scenarios (`crates/openasr-core/src/registry/tests/catalog.rs` and
`catalog_security.rs`'s own test module):

- **Scenario A** -- `embedded_catalog_degrades_instead_of_bricking_on_epoch_rollback`:
  the embedded snapshot, as the last-resort tier, degrades rather than
  erroring when the recorded floor is ahead of it.
- **Scenario B** -- `bundled_local_catalog_degrades_instead_of_bricking_on_epoch_rollback`:
  the same for `load_local_catalog_file_with_identity` (the desktop bundled-catalog
  path), using the real committed `model-registry/catalog.json` + signature
  with an artificially inflated recorded floor (no forged signature needed).
- **Scenario C** -- `boot_epoch_degrades_below_floor_while_strict_enforcement_stays_fail_closed`
  (in `catalog_security.rs`): for the identical (production-key, below-floor)
  verified signature, `enforce_catalog_epoch_for_verified` (remote/cache tier)
  still fails closed while `enforce_boot_catalog_epoch_for_verified`
  (boot-local tier) degrades -- pinning the exact behavioral split this
  section describes.

## Client resilience (load -> cache -> serve)

1. **Verify-then-persist.** A catalog is only ever written to
   `$OPENASR_HOME/catalog.json` (+ signature/epoch sidecars) *after* it fully
   verifies, parses, and passes forward-compat filtering + structural
   validation (`registry::persist_catalog_cache`, called only from the
   `Ok(catalog)` branch of every loader). A bad candidate never overwrites a
   good cache -- see `catalog_loader_does_not_cache_invalid_source` and
   `catalog_loader_falls_back_to_last_good_cache_when_local_source_is_tampered_without_resigning`.
   A disk-write failure while persisting a verified catalog does not fail the
   in-memory result either (`persist_catalog_cache` logs and continues) --
   the catalog this call already verified is safe to serve for this request
   regardless of whether the cache write succeeds.
2. **Fallback chain.** `load_model_catalog`'s primary source
   (`Ok(verified) => ...`) now falls through to
   `load_cached_signed_catalog` (on-disk cache, then the embedded snapshot) on
   ANY post-signature failure -- parse error, structural validation error, or
   the staged-entries anomaly below -- not just a transport/signature
   failure. Before this PR, `let catalog = parse_model_catalog(...)?;`
   propagated the error directly out of `load_model_catalog`, skipping the
   fallback chain entirely: a signature-valid but structurally-wrong primary
   payload bricked the load with a perfectly good on-disk cache sitting right
   there unused. `load_local_catalog_file_with_identity` (desktop bundled
   catalog) got the same fix, reusing the identical fallback chain.
3. **No same-cause crash loop.** Because (a) a bad primary source now always
   falls back instead of erroring, and (b) a below-floor boot-local candidate
   degrades instead of erroring, there is no longer a `load_*` path that
   fails identically on every restart with a perfectly good fallback
   available. `catalog_security::record_catalog_degraded` /
   `clear_catalog_degraded` persist which state the daemon is in
   (`$OPENASR_HOME/catalog.degraded.json`) so this is visible, not silent.

### Status surface

`openasr_core::read_catalog_degraded_status(home) -> Option<CatalogDegradedStatus>`
(`{ tier: "cache" | "embedded" | "local", reason: String }`) is best-effort
(never errors) and is surfaced two ways:

- **`openasr doctor`**: `Model registry: degraded, serving the <tier> catalog
  (<n> models) -- <reason>` instead of `Model registry: ok (<n> models)`.
- **Server `GET /health`**: additive `catalog_degraded: string | null` field
  (`null` = primary source, or no catalog load recorded yet). See
  `crates/openasr-server/generated/http-wire/HealthResponse.ts`.

## The "staged entries under the production identity" anomaly

The production `catalog.openasr.org` endpoint -- and therefore the on-disk
cache mirroring what it served, and the binary's embedded snapshot -- only
ever serves the **public projection** (`catalog.public.json`: `public: true`
entries only). The full, internal `model-registry/catalog.json` intentionally
carries staged (`public: false`) pre-release entries so a contributor can
preview an unreleased model locally (see "Root cause" below); it is
independently signed under the same production key and the same
`catalog_url` identity as the public projection (both are legitimate,
production-signed artifacts -- signing them separately is intentional, not a
bug).

If a payload verified under the **production** signing key (a real network
fetch, or the on-disk cache of one -- see
`catalog_security::participates_in_epoch_floor`) carries ANY `public: false`
entry, that is a data anomaly for that trust tier, not a security violation
(the signature is still valid): `registry::parse_and_check_production_catalog`
refuses it (`CatalogError::UnexpectedStagedEntries`) and the caller's fallback
chain takes over exactly as it would for any other post-signature failure.

This check is deliberately **not** applied to
`load_local_catalog_file_with_identity` / `preview_local_catalog_file_with_identity`:
the desktop's `OPENASR_CATALOG_FILE` bundled-catalog path and the CLI's
repo-checkout dev-preview path are both legitimate reasons to load the full
catalog under the production identity (dev preview needs the staged entries
on purpose).

Test: `cache_polluted_with_full_non_public_catalog_degrades_to_embedded_instead_of_bricking`
reproduces this exactly (copies the real, production-signed full catalog into
the cache position, then loads through `load_cached_signed_catalog` offline)
and asserts the daemon still starts, serving the embedded (public-only)
catalog, with `catalog_degraded.tier == "embedded"`.

## Incident background and root cause

**2026-07-16, corrected narrative.** The initial report described a "line
catalog with new dialect codes bricked v0.1.15". Real-binary testing showed
this was wrong: v0.1.15 loads the online (dialect-carrying) catalog fine.
Forensics on a contributor machine instead found:

- **Cache pollution.** `$OPENASR_HOME/catalog.json` (+ signature + epoch) held
  the repo's **full, non-public** catalog projection instead of the public
  one -- see "The staged entries anomaly" above for why that is meaningful.
- **The actual write point:** `catalog_cli.rs`'s repo-checkout dev-preview
  auto-discovery. `CONTRIBUTING.md` documents `cargo run -p openasr-cli --
  doctor` (and `list`, `transcribe`, ...) as the normal contributor dev loop,
  run with no `OPENASR_HOME` override -- which resolves to the real
  `$HOME/.openasr`. Before this PR, that auto-discovery path loaded
  `model-registry/catalog.json` (the full, staged-entries-including file)
  under the production identity via
  `load_local_catalog_file_with_identity`, which **also cached it** into
  `$OPENASR_HOME/catalog.json` -- the exact same path the real, installed
  release binary reads as its offline fallback. A single ordinary `cargo run
  -p openasr-cli -- doctor` from a checkout was enough to contaminate the
  real daemon's cache.
- **Why it then bricked (rather than just leaking staged models):** the
  contaminated cache's `catalog.epoch` (from a newer catalog epoch than the
  installed v0.1.15 build's own embedded snapshot) became the recorded
  anti-rollback floor. When the installed build later needed to fall back to
  its embedded snapshot (e.g. offline), that snapshot's own (older, real)
  epoch was now BELOW the locally-inflated floor, so
  `enforce_catalog_epoch_for_verified` rejected it with `EpochRollback` --
  and, before this PR, there was no further fallback: the daemon had nothing
  left to serve.

**Fixes landed here, mapped to the mechanism:**

1. `catalog_cli.rs`'s auto-discovery now calls the new
   `preview_local_catalog_file_with_identity` (read-only: same
   verify/parse/degrade behavior, but never writes
   `$OPENASR_HOME/catalog.json`) instead of the caching
   `load_local_catalog_file_with_identity` -- closes the actual write point.
   `load_local_catalog_file_with_identity` remains available (and caching)
   for its one real caller, the desktop's `OPENASR_CATALOG_FILE` bundled
   catalog.
2. "Epoch floor at boot": even if a floor mismatch like this recurs (a
   dev-tool artifact, or a legitimate app downgrade/reinstall), the affected
   boot-local candidate now degrades instead of bricking.
3. The fallback chain fix (`load_model_catalog`'s primary tier now falls
   through to cache/embedded on any post-signature failure, not just a
   transport/signature failure) is defense in depth for the same class of
   failure regardless of its trigger.
4. The "staged entries under production identity" guard specifically detects
   *this* contamination shape (full catalog where only the public projection
   belongs) even if it reappears through some other path in the future.

No signing key, catalog data, or epoch/signature verification math changed;
`model-registry/` is untouched by this fix.
