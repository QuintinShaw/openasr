# Model onboarding contract: shared facilities every new family MUST reuse

Status: normative v2 for new-architecture PRs and migrations. Complements the
[model-family lifecycle](model-family-lifecycle.md) and the "how do I add a
family" walkthrough in [Model Onboarding](../MODEL_ONBOARDING.md); this doc is
the narrower, checklist-shaped contract a reviewer holds a PR against.

## Why this exists

The FireRedASR-AED long-audio repetition bug (issue #60) traced back to one
root cause: a family hand-wrote its own decode step loop instead of going
through the shared greedy-decode driver, so it silently missed the
degenerate-loop guard and drifted argmax/suppression/stop-token semantics from
every other family. The fix was structural (route FireRed through
`run_seq2seq_greedy_decode_loop_v0` like everyone else), and
[`AGENTS.md`](../../AGENTS.md) now carries a **"One greedy decode driver"**
invariant so it cannot regress.

That invariant covers decode. This doc generalizes the same discipline to the
descriptor facets, pack proof chain, generated projections, quantization roles,
shared compute layer, and progress/cancel plumbing. The pattern that produced
the FireRed bug -- "each family builds its own version instead of reusing the
shared one" -- can recur in any seam. New-model PRs check every item below; any
dedicated topology needs a **structural** reason in the inventory and a matching
conformance fixture, not convenience.

## Shared facilities (reuse, do not re-implement)

### 1. Descriptor facets and generated dispatch

Register the family as one complete
`OpenAsrArchitectureDescriptor` row in
`crates/openasr-core/src/arch/mod.rs::BUILTIN_ARCHITECTURE_DESCRIPTORS`.
Every facet is required: `identity`, `pack_contract`, `execution_contract`,
`topology_contract`, `optimization_contract`, `quantization_contract`, and
`conformance_contract`. Each component id resolves through its narrow typed
component registry only when materialization is needed; there is no giant
shadow component table mirroring every architecture row.

The executor is materialized through the descriptor's typed runtime factory
(`materialize_builtin_executor::<E>`) into the service-owned executor scope.
Offline/streaming dispatch, force-linking, validator routing, and eviction
coverage are projections of this inventory. Do not add a family-specific
central match or a second registry.

The descriptor also requires an `encoder_attention_span`
(`OpenAsrEncoderAttentionSpan`, issue #68) declaring how the new
architecture's encoder scales with chunk length -- this is a mandatory field,
so a new architecture cannot compile without it:

- Full self-attention over the whole encoder input (the common case for a
  Conformer/Transformer/E-Branchformer/RoPE encoder) is `GlobalQuadratic`.
  Use `arch::DEFAULT_ENCODER_SAFE_CHUNK_SECONDS` (30s) for
  `max_safe_chunk_seconds`. This is the **memory** ceiling -- the longest
  chunk this architecture may be handed before its attention activations are
  a risk on commodity RAM -- and it is deliberately a different constant from
  `arch::DEFAULT_ENCODER_CHUNK_SECONDS`, the default chunk length long-form
  slicing aims for. They hold the same number today, but for unrelated
  reasons: the chunk length is where every major encoder family this repo has
  surveyed converges (Whisper's fixed window, Moonshine's own "<30s"
  guidance, NeMo/Parakeet's 20-30s guidance, FunASR's 30000ms default,
  Dolphin's 30s training/eval padding, Cohere's 30s reference sliding
  window), which is a transcription-quality argument and cannot certify a
  memory bound. Do not collapse the two; see
  `DEFAULT_ENCODER_SAFE_CHUNK_SECONDS`'s doc comment. **Only** override
  `max_safe_chunk_seconds` when the upstream model card states an explicit,
  different recommended chunk length, and cite that source in a comment next
  to the override.
- An architecture-fixed attention window (like Whisper's 30s log-mel frame)
  is `FixedWindow`.
- A local/chunked streaming encoder with a bounded per-chunk cache (like
  Zipformer2's multi-scale cache) is `LocalChunked`.

**Do not** add a parallel hand-written family dispatch branch in a backend
ingress or adapter list outside the descriptor-driven path. Public compatibility
helpers must delegate to the same descriptor and pack-proof path; they must not
become a second validator or runtime admission universe. Family selection keeps
the registry's fail-closed unknown/ambiguous behavior.

The authoring helper is deliberately a skeleton generator, not an
implementation generator. Run
`cargo xtask family new <module_slug> [--profile-id <profile-id>]` to create
`mod.rs`, `architecture.rs`, `package_import.rs`, `runtime_contract.rs`, and a
README. The module is wired to a compile-time fail-closed sentinel; every facet
and every contract must be implemented by the migration. `module_slug` is the
snake_case Rust directory name, while `profile_id` is an independent
lower-kebab conformance id (the scaffold only writes a literal default when
the flag is omitted). No `contract.toml`, fake quantization value, or runnable
placeholder is generated.

`cargo xtask family conformance [--profile-id <profile-id>]` is the
weight-free structural gate. It validates the current builtin profile (when
selected), the generated inventory, the full openasr-core library test suite,
the publishing-tool Python tests, regeneration drift, and the static GPU
weight-placement gate. It intentionally does not run model weights, real
backend smoke, or benchmarks; those C-class obligations require release/manual
receipts and remain part of the reviewer checklist below.

The real-weight streaming recipe list and catalog documentation-keyword map are
artifact/prose evidence rosters, not runtime family registries. Add an entry
only when providing that corresponding smoke asset or public catalog family;
the entry must resolve its profile/family through the generated Rust inventory
instead of restating runtime capabilities.

### 2. Pack proof and admission

Every ASR and auxiliary importer uses `PackEnvelope` and the transactional
`OasrPackWriter`, then verifies the exact staged bytes through `PackVerifier`:

```text
PackCandidate -> PackVerifier -> VerifiedPack
                  (owns exact GgufRuntimeSourcePreflight)
        -> ContentStore -> AdmittedPack
        -> NativeExecutionServices
```

`VerifiedPack` and `AdmittedPack` are proof values with non-forgeable
construction. Publish, install, core execute, and runtime entry points consume
them rather than accepting a bare path or preflight as package proof. Public
converter results carry the writer-returned proof; FFI open verifies once and
retains it. `PackRoute::Asr` and `PackRoute::Aux` share the
verifier and content-identity lifecycle; only their route-specific contract
differs. A family may add metadata but cannot override envelope keys or bypass
the verifier. Once this path owns a behavior, remove any old writer, scanner,
or duplicate preflight and its tests/docs.

A signed-catalog install also carries its canonical catalog family id through
`ResolvedCatalogPull`. Content admission compares that target with the family
projected from `VerifiedPack` before exposing the object. Equal digest/size or a
matching model id cannot authorize a pack whose proven route belongs to a
different family.

Publishing stages through the Rust CLI, not a Python copy plus metadata mirror:

```text
openasr model-pack preflight <source.oasr> --stage <dest.oasr> --json
```

The `openasr.model-pack-preflight.v1` receipt must bind the staged content
id/size, route, canonical catalog family id, architecture, and pinned
`openasr.build.commit`. Release tooling compares those facts with the conversion
result and catalog entry, removes a rejected stage, and never treats the JSON
receipt as an in-process execution capability.

### 3. Decode driver

- Seq2seq / AED / autoregressive families implement
  `Seq2SeqGreedyDecodeStepExecutor` and run through the shared
  `run_seq2seq_greedy_decode_loop_v0` (invoked via
  `run_builtin_seq2seq_decode_policy` in
  `crates/openasr-core/src/models/seq2seq_greedy_decode.rs`).
- CTC / non-autoregressive families use
  `crates/openasr-core/src/models/ctc_greedy_decode.rs`'s `ctc_greedy_decode`.
- Eligible batched serving goes through the shared serve-batch policy and owner
  path (`crates/openasr-core/src/models/serve_batch_env.rs` and
  `seq2seq_serve_batch.rs`), not a per-family batch loop. Eligibility and the
  effective width are server-owned: the native session admission limit is the
  only operator input, while CPU, scheduler-backed, adapter, and non-enabled
  families explicitly remain serial.

**Do not** hand-write a `for`/`while` + argmax step loop that bypasses these.
A hand-rolled loop is exactly what caused issue #60: it misses the shared
degenerate-loop guard and drifts stop-token/suppression semantics from every
other family.

### 4. Decode policy

Stop tokens, suppression, and text post-processing (including longform carry)
are carried by the `BuiltinDecodePolicyComponentDescriptor` embedded in the
row's typed decode-driver strategy. Reusable policy constants live in
`crates/openasr-core/src/models/decode_policy_component_registry.rs`; there is
no second family-to-policy registry. Reuse an existing constant when behavior
matches; do not write a new if/else post-processing branch elsewhere.

### 5. Package import

Reuse the shared import primitives: `local_source_import` (the per-family
module calls the shared helper and does not reimplement path/zip handling),
`PackEnvelope`/`OasrPackWriter` for transactional emission, and
`PackVerifier` for the exact staged bytes. The envelope owns the protected
`openasr.*` keys documented in [`.oasr` Package Contract v1`](../format/OASR_PACKAGE_CONTRACT_V1.md);
family metadata is additive only. The writer returns `VerifiedPack` before a
successful importer may expose its result, and every public conversion result
carries that exact proof beside its diagnostic path. Do not add a raw production
writer, a family-local metadata validator, a proof-dropping result, or a second
preflight.

Quantization uses the shared semantic `TensorRole` classifier. Importers may
map source names to roles once, but eligibility and axis/orientation policy are
decided by the shared quantization contract, not by a new family enum or a
string-name match duplicated in the audit.

### 6. Optimization contract

Optimization obligations have three forms:

- **Shared invariants (A):** the shared interfaces provide single-pass
  preflight, cancellation fences, content-identity admission, prepared-runtime
  ownership, content-id eviction, graph reuse, poisoned-state rebuild, and
  fail-closed dispatch. These universal facts are not per-family fields; a
  family cannot opt out by writing a parallel callback or cache.
- **Typed family policy (B):** the descriptor must fill family-varying choices:
  streaming granularity, decode-driver strategy, encoder attention span,
  placement/auto-backend policy, phrase bias, concrete LoRA binding, word
  timestamps, and prepared-runtime strategy. No `Default`, wildcard, or runtime
  `Deferred` value is accepted.
- **Measured result (C):** GPU placement, cold/warm latency, RSS, RTF, quant
  quality, and streaming cadence require conformance plus a real backend smoke
  or benchmark receipt. A descriptor bit or static code shape is not a result.

When a new shared structural optimization is introduced, add the shared seam or
required typed field first so every family fails compile/CI until it is covered.
When a new measured obligation is introduced, add the conformance/benchmark
gate; do not mark it complete by assertion.

### 7. Tokenizer

- BPE families use the shared `gpt2_bpe` tokenizer path (see
  `crates/openasr-core/src/models/whisper/tokenizer.rs` and
  `crates/openasr-core/src/models/qwen/tokenizer.rs` for the calling
  convention).
- SentencePiece / metaspace families: a shared `SpmDecoder` is planned
  (tracked separately, not yet landed). Until it lands, do not hand-roll a
  one-off `▁` / `<0x..>` byte-fallback / id-to-token table inside a new
  family module if an equivalent already exists elsewhere in the tree --
  factor it out to a shared location instead of adding a third copy.

### 8. Neural network layers

Encoder/decoder stacks compose from the shared blocks in
`crates/openasr-core/src/nn/` (`attn.rs`, `ffn.rs`, `norm.rs`, `conv.rs`,
plus `encoder.rs` / `decoder.rs` helpers). Bypassing `nn/` for a new attention
or normalization variant needs a structural reason in the PR description (for
example X-ASR's Zipformer2 multi-scale streaming cache, which does not fit the
existing block shapes) -- add the new primitive to `nn/` rather than growing it
inline in the family module when the pattern is reusable.

### 9. Capabilities

`supports_phrase_bias`, `emits_punctuation`, and streaming registration are
declared **once**, on the descriptor's execution contract. The executor
component registry audits the concrete executor against that row, and the
streaming-executor completeness gate validates the generated projection. The
model catalog (`model-registry/catalog.json`) and any client/TS capability
surface must be generated or read from the inventory export, not maintained as
a second constant. **Do not** declare the same capability as separate literals
in a catalog card, client table, and executor -- those mirrors are how
capabilities and decode logic drift.

Speaker routing is the related architecture-level contract. Every ASR
descriptor must declare exactly one `speaker_segmentation` source:

- `InDecoder` only when the family itself emits usable recording-local speaker
  turns (currently MOSS).
- `External` for every other family. That choice automatically opts the family
  into the shared FireRed VAD + selected segmenter + ReDimNet2-B6 + clustering
  pipeline for local file Voice ID.

The signed catalog's `speaker_source` is generated from this value and checked
against the Rust descriptor. Do not add a model-id allowlist, a family-local
diarizer, or a second identity matcher. Both speaker sources converge on
`diarize::voice_id`; even an `InDecoder` family still needs ReDimNet2-B6 for
cross-scope reconciliation and enrolled-person matching.

### 10. Generated projections and cleanup

The one inventory row projects offline/streaming dispatch, executor
force-linking, validator routing, content-id eviction, audit enumeration, and
the machine-readable inventory used by publishing tooling. A migration may
temporarily read old data only behind a deletion gate. Once the projection owns
the behavior, delete the old hand-written match/list/mirror and its obsolete
tests and documentation. A generated table plus a manually maintained table is
not an accepted steady state.

### 11. GPU weight placement

A new encoder/decoder's persistent 2D matmul weights MUST bind through
`load_gguf_weight_context_from_preflight` (zero-copy, native quantized type)
and its 1D norm/bias tensors through `GgmlStaticTensorArena` -- both land in a
`GGML_BACKEND_BUFFER_USAGE_WEIGHTS` buffer, which is the only thing the ggml
scheduler will offload to a GPU backend. The bare-source
`load_gguf_weight_context` helper is test-only; production family code must
consume the existing preflight proof and may not reopen a path.
`runner.start_graph()` + an upload
call (`uploads.push` / `pending_uploads.push` / `.upload(...)`) is for genuine
per-request input (features, token ids, step state) only -- **never** for
persistent model weights; using it for weights pins the whole subgraph to CPU
regardless of the configured backend, and byte-identical golden/parity output
gives zero signal that this happened (short fixtures produce the same numbers
on CPU or GPU). See [GPU weight placement](gpu-weight-placement.md) for the
full defect writeup (this is exactly what Dolphin's and X-ASR/Zipformer's
encoders got wrong, #131/#115) and the two-part gate: the static
`scripts/gpu-weight-placement-gate.sh` plus a one-shot
`GGML_SCHED_DEBUG=2` real forward pass proving the encoder's splits actually
land on the GPU backend.

### 12. Progress, history, cancel

Long-running transcription progress, history reporting, and cancel/pause
semantics run through the shared driver plumbing a new family's executor and
streaming registration plug into. Do not add a second progress/cancel path
that only exists for "batch mode" or "file mode" or a specific family --
single-request vs batch and file vs realtime must stay expressed as
parameters/paths through the one shared mechanism, not a forked
implementation.

All single-job ggml graph execution must use the shared compute-scoped
cancellation contract described in [graph-cancellation.md](graph-cancellation.md).
Do not add a model-family-specific backend callback or retain a job's callback
pointer in a cached runtime. Algorithmic multi-graph boundaries (for example a
prefill chunk loop) may add earlier typed checks, but do not replace L2 graph
cancellation. A graph that can write resident KV or other session tensors must
participate in the shared poison/rebuild contract: any incomplete compute makes
that state ineligible for reuse. Do not treat a cached backend/device handle as
model state -- keep the handle and immutable weights, but drop/rebuild the
poisoned graph and its mutable session tensors. A serve-batch graph contains multiple jobs and therefore keeps
per-member cancellation at batch/chunk boundaries; aborting that shared graph
from one member's flag would incorrectly cancel healthy siblings.

## Reviewer checklist

Copy this into the PR description and check off each line (or replace the box
with a one-line structural justification for going another way):

- [ ] New architecture is one complete `OpenAsrArchitectureDescriptor` entry
      in `arch/mod.rs` with explicit `identity`, `pack_contract`,
      `execution_contract`, `topology_contract`, `optimization_contract`,
      `quantization_contract`, and `conformance_contract` facets. No `Default`,
      `..base`, wildcard, parallel registry, or runtime `Deferred` escape.
- [ ] Descriptor selection is covered by a test that fails closed on unknown
      and ambiguous families. The descriptor's `encoder_attention_span` is
      explicit: a `GlobalQuadratic` encoder uses
      `arch::DEFAULT_ENCODER_SAFE_CHUNK_SECONDS` unless a cited upstream source
      justifies another bound; the production attention-span tests cover it.
- [ ] The AST-backed family source gate proves a shared seq2seq/CTC strategy
      calls its declared driver in production code; the family implements
      `Seq2SeqGreedyDecodeStepExecutor` or calls `ctc_greedy_decode`. Manual grep
      may supplement this gate but is not the architectural proof.
- [ ] Import uses `local_source_import` plus `PackEnvelope`/`OasrPackWriter`;
      exact staged bytes pass `PackVerifier` and yield `VerifiedPack` before
      exposure, and the public result carries that proof. Untrusted paths are
      converted at the first ingress; no bare-path fallback, raw production
      writer, duplicate preflight, metadata validator, or zip parser is added
      downstream.
- [ ] ASR and auxiliary packs use the same verifier/content-admission lifecycle
      with their explicit route; auxiliary packs do not masquerade as ASR
      descriptor rows.
- [ ] Quant handling maps source names once to shared semantic `TensorRole`s;
      no per-family quant enum or string-name eligibility table is added.
- [ ] Family production code does not parse raw backend/provider names. It
      consumes typed backend kind/capability values resolved by shared runtime
      code; any typed backend branch has a documented correctness reason.
- [ ] Tokenizer reuses `gpt2_bpe` (BPE) or the shared SPM path once it lands;
      no new hand-rolled `▁`/byte-fallback table duplicating an existing one.
- [ ] Capabilities (`supports_phrase_bias`, `emits_punctuation`, streaming) are
      declared once on the descriptor execution facet and exported from the
      inventory; no second literal in the catalog card or client table.
- [ ] The architecture descriptor declares `speaker_segmentation` as
      `InDecoder` or `External`; the generated catalog mirrors it through
      `speaker_source`, and no family-local Voice ID/diarization pipeline or
      model-id allowlist was added.
- [ ] Optimization A invariants use the shared admission, cancellation,
      ownership/eviction/reuse, and poisoned-state seams rather than family
      declaration fields; varying Optimization B policies are explicit typed
      fields; Optimization C claims have conformance, real backend smoke, or
      benchmark receipts. No static descriptor bit is presented as a result.
- [ ] Generated offline/streaming dispatch, executor force-linking, validator
      routing, eviction, and audit enumeration cover the row. Any migration
      deletion gate removes the old hand-written projection and its obsolete
      tests/docs; no dual source of truth remains.
- [ ] Encoder/decoder stack composes over `nn::{attn, ffn, norm, conv}`; any
      bypass has a structural reason stated in the PR description.
- [ ] GPU weight placement (see [GPU weight placement](gpu-weight-placement.md)):
      `scripts/gpu-weight-placement-gate.sh` shows no new finding for this
      family, and a one-shot `GGML_SCHED_DEBUG=2 GGML_DEBUG=1
      OPENASR_GGML_BACKEND=<gpu backend>` real forward pass (pasted into the PR)
      shows the encoder's/decoder's matmul splits on the GPU backend, not
      `CPU`. Byte-identical golden/parity output on a short fixture is **not**
      evidence of this -- it is identical whether the subgraph ran on CPU or
      GPU (this is exactly how Dolphin's and X-ASR/Zipformer's encoders
      shipped CPU-pinned despite passing review, #131/#115).
- [ ] Progress/cancel/history reuse the shared driver plumbing; no new
      single-vs-batch or file-vs-realtime second path.
- [ ] If extending or refactoring an existing family: byte-identity is proven
      (golden-diff / stash-diff per [Model Onboarding](../MODEL_ONBOARDING.md#step-4--gate-the-pack-runtime-and-output)).
      A brand-new family adds a bench-suite entry and freezes its first
      transcript as the reference instead.
- [ ] The integration scope is explicit: `core-only`, `staged release
      candidate`, or `public-ready`. Staged/public-ready work follows
      [Model Onboarding, Step 5](../MODEL_ONBOARDING.md#step-5--choose-the-integration-scope-and-close-the-release-handoff):
      publishing inputs are human-edited, registry/catalog files are generated,
      and no URL, digest, revision, metric, or public status is fabricated.
- [ ] Before a first public release, the family audit is complete and the
      smallest public checkpoint/quant has a committed real-model regression
      golden plus workflow entry. Weight-free conformance is not presented as
      WER, RTF, RSS, accelerator-utilization, or quantization-quality evidence.

## Relationship to Model Onboarding

[`MODEL_ONBOARDING.md`](../MODEL_ONBOARDING.md) is the "how do I write the
per-family code" walkthrough (descriptor facets, pack proof, shared compute
seams, quantized-weights runtime contract, and the honest gap list).
[`model-family-lifecycle.md`](model-family-lifecycle.md) states the v2 invariants
and cleanup rule. This document is the narrower anti-fragmentation contract:
as more families land, shared facilities stay singular instead of accumulating
one bespoke variant per family. When a mechanical detail disagrees, the live
code and the lifecycle contract are authoritative; reconcile this checklist
before copying it into a new PR.
