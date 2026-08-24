# Model-family lifecycle and architecture contract

Status: normative for new model-family work and migrations.

This document is the v2 contract for bringing an ASR or auxiliary model into
OpenASR. It complements [Model Onboarding](../MODEL_ONBOARDING.md), which is the
implementation walkthrough, and the [Model Onboarding Contract](model-onboarding-contract.md),
which is the reviewer checklist. The live Rust inventory and tests remain the
implementation authority; this document defines the boundaries a change must
preserve.

## The three invariants

The lifecycle is organized around three independently testable invariants:

1. **Pack contract:** a pack that can be published or installed has already
   passed the same fail-closed contract the runtime uses. A writer cannot omit
   required public metadata and a malformed artifact cannot be admitted by a
   later path.
2. **Completeness:** a family cannot silently omit a shared capability or a
   newly required optimization. Structural obligations fail compilation; result
   obligations fail conformance or benchmark gates.
3. **Hot-path shape:** family differences are resolved while a service root is
   prepared. Tensor, graph, backend, and token loops use shared, monomorphic
   paths after preparation; dynamic lookup is not added to those loops.

These are architecture invariants, not suggestions for individual model PRs.

## One inventory row, required facets

The source of truth is the existing
`OpenAsrArchitectureDescriptor` inventory. Do not create a second
`FamilyDefinition` registry or a parallel vocabulary. A family row is a product
of required facets:

```text
OpenAsrArchitectureDescriptor {
    identity,
    pack_contract,
    execution_contract,
    topology_contract,
    optimization_contract,
    quantization_contract,
    conformance_contract,
}
```

Each facet owns one kind of fact:

- **Identity:** family, architecture, adapter, catalog identity, and language
  hint.
- **Pack:** frontend, tokenizer, runtime tensor contract, importer surface, and
  the family runtime validator.
- **Execution:** executor factory, offline/streaming cadence, speaker source,
  invocation span, product capabilities, and typed runtime policies.
- **Topology:** decode policy, decoder-state topology, decode-driver strategy,
  and block-stack strategy.
- **Optimization:** encoder attention span, placement policy, and other
  family-varying typed policies. Universal ownership, content-id eviction, graph
  reuse, cancellation, and admission live in shared modules and are not repeated
  as self-certified family fields.
- **Quantization:** semantic tensor-role classification and its quantization
  policy. Tensor spelling is an importer detail, not the final eligibility
  decision.
- **Conformance:** profile id, reference source, negative fixtures, and the
  gates a family must run.

All fields are explicit. Do not add `Default`, `..base` struct updates,
wildcard matches, or a runtime `Deferred` value to make a new row compile.
`NotApplicable { reason }` is the only valid statement that a required policy
does not apply. A new required field intentionally makes every existing row
fail to compile until it is reviewed.

### Typed execution policies

The execution facet records capability policy, not a family-id lookup table.
These policies are deliberately typed so a new row must make an explicit
choice:

- **Phrase bias:** `Unsupported`, `Always`, or `RequiresTensor { tensor_name }`.
  The `RequiresTensor` case is admitted only when the named tensor was found by
  pack preflight; runtime code must not rediscover it from a family id.
- **LoRA binding:** `Unsupported` or a concrete executable binding strategy. The
  shared LoRA path owns the lifecycle and dispatch cross-checks the row against
  the materialized executor, so a boolean cannot self-certify support.
- **Word timestamps:** `DecodeInvariant` or `DecodeSensitive`, selecting the
  shared timestamp policy without a Whisper-style family branch.
- **Prepared runtime:** `FamilyOwned` or a named shared reusable component.
  A family reusing an existing component changes only its inventory row; it
  does not add a central family match. A genuinely new reusable component is
  added once to the typed component registry and then referenced by rows.

These choices are projected to adapters, inventory exports, and conformance
audits. They replace runtime family-id whitelists and branches while keeping
the hot path on the prepared concrete component.

### Decode and graph strategy

The topology facet must name one of the shared decode drivers or an explicitly
reasoned dedicated driver. A dedicated topology is not an escape hatch: its
mathematical reason is inventory data and it still receives shared admission,
cancellation fences, ownership, and conformance gates.

Likewise, block assembly is either a shared `OpenAsrBlockStackDescriptor` or an
`ArchitectureGraph { reason }` for a topology the current composer cannot
express. The reason belongs beside the choice; an unlabeled absence is invalid.

An existing shared mathematical structure means tensor binding, parameters,
and assembly only. A genuinely new topology may add a narrow architecture
adapter. A reusable primitive belongs in `nn/`, ggml, or the shared backend
layer, with all platform conformance completed before a family can use it.

## Pack proof chain

The only public lifecycle across the trust boundary is:

```text
PackCandidate -> PackVerifier -> VerifiedPack
                  (owns exact GgufRuntimeSourcePreflight)
        -> ContentStore -> AdmittedPack
        -> NativeExecutionServices
```

`VerifiedPack` is constructed only by `PackVerifier` after the exact source has
passed metadata, tensor, package, route, and family-validator checks. Its fields
are private so callers cannot manufacture the proof. `AdmittedPack` carries the
same verified fact together with content-store admission and the ownership
lease. A filesystem or process boundary may begin with an untrusted path, but
the first in-process ingress converts it to a proof value. Downstream install,
writer, and runtime code consumes the proof and cannot fall back to reopening a
bare path as an alternative. Public converter results carry the writer-returned
proof beside their diagnostic path; core execute requests carry `VerifiedPack`.
FFI open performs the complete verification once and retains that proof.

### Writer rule

Every ASR and auxiliary importer uses the shared `PackEnvelope` and transactional
`OasrPackWriter` seam:

1. Build the envelope with protected public keys (`package.version`, family,
   architecture, frontend, decode policy, tokenizer contract, and build
   provenance).
2. Allow a family to add only family-specific metadata. Protected keys cannot
   be replaced or made optional.
3. Write to staging, verify the exact bytes with `PackVerifier`, then expose the
   final artifact atomically and return `VerifiedPack`.

The same verifier and content identity apply to `PackRoute::Asr` and
`PackRoute::Aux`; the route determines the family contract, not a second pack
lifecycle. Raw GGUF writing is a crate-internal fixture/tooling primitive, not a
production family import API.

`.oadp` is a distinct, local unsigned adapter sidecar trust domain, not an ASR or
Aux route. Its writer follows the same transactional discipline (private staging,
production-reader verification, no-clobber exposure) and returns the parsed
adapter pack; runtime additionally validates exact base model id and base-pack
content hash before materialization.

### Publish staging and receipt

Release tooling does not copy a pack and then run a second, approximate Python
scanner. The Rust CLI owns one transactional staging operation:

```text
openasr model-pack preflight <source.oasr> \
  --stage <release-stage.oasr> --json
```

It creates a new destination, copies and syncs it, runs the production
`PackVerifier` over the staged bytes, seals the accepted stage read-only, and
emits `openasr.model-pack-preflight.v1`. The receipt binds the exact content id
and size to `PackRoute`, the canonical catalog family id, architecture, and
`openasr.build.commit`. Publishing rejects a route/family mismatch, an absent or
unpinned 40-hex build commit, or a digest/size mismatch against the conversion
result. A rejected staged file is removed.

The JSON receipt is evidence for tooling, not an execution capability and not
a substitute for `VerifiedPack`. Complete SLSA provenance remains a later
release concern; the v1 receipt closes artifact/client-contract and
artifact/catalog-family drift without creating another verifier.

Content-store admission similarly hashes each byte while copying it once. The
same private held descriptor is then mapped and preflighted; it is not rehashed
through a reopened path or admitted through a legacy zero-copy writer.
For a catalog-resolved install, the signed catalog's canonical family id is
carried in `ResolvedCatalogPull` and compared with the family projected from
the verified pack route inside this staging closure. A mismatch prevents the
object from being exposed in the content store, not merely from receiving a
logical ref. Explicit legacy-store migration has no catalog claim to invent;
it trusts the canonical route proven from its own admitted bytes.

## Optimization contract

Optimization obligations are split so a family cannot claim an optimization by
merely writing a boolean in its row:

### A. Shared invariants

These are provided by narrow shared interfaces and cannot be bypassed by a
family: one-open proof-carrying preflight, public cancellation fences, content-identity
admission, prepared-runtime ownership, poisoned-state rebuild, and fail-closed
dispatch. A family does not repeat these in its inventory; it uses the shared
seams.

### B. Required family policies

These are explicit typed fields in `optimization_contract`,
`execution_contract`, and `topology_contract`, including streaming granularity,
decode-driver selection, encoder attention span, backend auto policy,
phrase-bias policy, concrete LoRA binding, word-timestamp policy, and
prepared-runtime strategy. Universal ownership, content-id eviction, and graph
reuse stay in category A and do not appear as per-family declarations. No
`Default` or wildcard can hide a missing choice. Adding a varying policy adds a
field and deliberately turns every family row red until it is classified.

### C. Measured outcomes

GPU placement, cold/warm latency, resident RSS, quantization quality, streaming
cadence, and similar outcomes are not proven by a descriptor bit. A missing
typed conformance profile or gate declaration is a compile/weight-free-CI
failure. The real-weight backend smoke and benchmark receipt are release/manual
gates unless a dedicated artifact-backed CI job runs them; ordinary weight-free
CI does not count as a measurement. Static code shape is evidence of structure
only; do not claim a performance or quality improvement without the applicable
receipt.

## Generated projections

The inventory is the only registration input. Code generation or inventory
projection owns:

- offline and streaming dispatch;
- executor materialization and ownership scope;
- content-id eviction and idle unload coverage;
- package-import force-linking;
- install/runtime validator dispatch;
- capability projections for phrase bias, LoRA binding, word timestamps, and
  prepared-runtime/component strategy;
- audit and conformance enumeration; and
- the Rust-to-machine-readable inventory export consumed by publishing tools
  and Python catalog views.

Migration may use a temporary compatibility reader, but it must have a deletion
date and a failing gate. Once a generated projection owns a behavior, remove the
old hand-written family-id match, list, mirror, branch, and its obsolete
tests/docs. A generated table plus a hand-maintained table is two sources of
truth and is not an acceptable steady state.

## Public compute-layer boundary

The family adapter owns only irreducible semantics:

- frontend parameters and input shaping;
- one-time tensor-name-to-`TensorRole` binding;
- mathematical topology and graph assembly; and
- the narrow step executor required by that topology.

The shared layers own decode drivers, cancellation, progress/history, cache and
memory admission, backend-neutral graph context, device placement, and reusable
`nn`/ggml primitives. A family must not add a platform branch, a second token
loop, a second cancellation callback, or a parallel cache lifecycle/keying
scheme. If a new
primitive is reusable, it enters the shared layer first and gains backend
conformance before family code references it.

The performance boundary is explicit: dynamic dispatch may exist at service
preparation or the request seam, but tensor/op/token loops remain on the
prepared concrete path. Correctness and performance claims must be validated by
the relevant tests and benchmark receipts, not inferred from names.

## Onboarding sequence and reference families

Use this order for the v2 migration and for future examples:

1. **FunASR-Nano:** exercise public metadata, transactional writing, validator
   dispatch, and fail-closed rejection for the pack-contract incident class.
2. **Parakeet-CTC:** exercise a non-LLM CTC topology, shared CTC greedy driver,
   explicit block-stack strategy, and common runtime ownership.
3. **Qwen3-ASR:** exercise an autoregressive decoder, KV/cache reuse, serve-batch
   policy, native quantized bindings, and cancellation under a higher-load path.
4. **Parakeet-TDT:** freeze the dedicated-driver conformance boundary so a
   dedicated topology cannot become an unreviewed escape from shared lifecycle
   rules.

The `cargo xtask family new <module_slug> [--profile-id <profile-id>]` command
creates an intentionally incomplete skeleton under the Rust module directory:
`mod.rs`, `architecture.rs`, `package_import.rs`, `runtime_contract.rs`, and a
README. `mod.rs` declares the three implementation seams and contains a
fail-closed `compile_error!`; `architecture.rs` exposes all seven facet fields
as explicit `todo!` sites; the pack-import and runtime files contain only
compile-checked seam references and a fixed validator error, never a fabricated
contract. It does not create a machine-readable sidecar or claim to generate a
runnable implementation.

`module_slug` is the snake_case Rust directory name. `profile_id` is a separate
lower-kebab conformance identifier; when omitted, the scaffold performs a
one-time `_` to `-` spelling conversion and writes the resulting literal into
the author-facing checklist. Runtime code never derives one identifier from
the other. The author then implements the narrow adapter, adds positive and
negative fixtures, runs the weight-free structural command
`cargo xtask family conformance [--profile-id <profile-id>]`, and does not edit
central dispatch matches. That command runs global inventory, Rust, Python,
regeneration, and static GPU-placement gates. Real weights, backend smoke, and
benchmark receipts are release/manual C-class obligations and are deliberately
outside this command.

Lifecycle integration and distribution are separate milestones. Every change
states whether it is **core-only**, a **staged release candidate**, or
**public-ready**. The first needs no fabricated catalog state; the latter two
follow [Model Onboarding, Step 5](../MODEL_ONBOARDING.md#step-5--choose-the-integration-scope-and-close-the-release-handoff)
and the catalog ownership chain. Human-edited publishing inputs may be staged,
but generated registry/catalog files never become a second source of truth.
Public visibility, signing, uploading, and deployment remain separately
authorized release actions.

## Acceptance and cleanup

A migration is complete only when:

- the writer cannot omit public envelope keys;
- a bare path cannot reach publish, install, or runtime;
- missing required policies fail compile or CI;
- ASR and auxiliary packs use the same verifier/admission lifecycle;
- generated projections replace the old manual wiring;
- FunASR, Parakeet-CTC, Qwen, and TDT reference paths pass their real gates; and
- obsolete paths, tests, names, and documentation are deleted after the new
  path owns the behavior.

A staged/public-ready model additionally satisfies the Step 5 catalog,
family-audit, real-weight receipt, and family-regression obligations appropriate
to that scope. Scheduled and release-event family-regression jobs are CPU-only
post-publication monitoring, but the reusable `family-regression.yml`
pre-publication contract is a release-candidate blocker: it verifies and runs
the exact draft CLI against the committed staging catalog. GPU provider
correctness is a separate post-publication activation obligation. Its exact
public release bytes must pass the generated target/backend-scoped hardware and
family/token matrices before a new signed catalog epoch can make that provider
selectable; missing evidence leaves the provider `PublishedInert`. Passing this
architecture contract alone never implies that a model is published or
performance-qualified.

Do not leave a compatibility alias merely to make a stale example compile. The
source tree and the docs must teach the current path so a later contributor
cannot copy the retired design.
