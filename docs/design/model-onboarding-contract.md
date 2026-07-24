# Model onboarding contract: shared facilities every new family MUST reuse

Status: normative for new-architecture PRs. Complements the "how do I add a
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

That invariant covers decode. This doc generalizes the same discipline to
every shared facility a new model family touches: registration, decode,
packaging, tokenization, tensor layers, capability declaration, and
progress/cancel plumbing. The pattern that produced the FireRed bug --
"each family builds its own version instead of reusing the shared one" -- can
recur in any of these seams, not just decode. New-model PRs check every item
below; any item marked "self-built" needs a **structural** reason in the PR
description (a genuinely new shape/algorithm), not convenience.

## Shared facilities (reuse, do not re-implement)

### 1. Registration and dispatch

Register the family as data in
`crates/openasr-core/src/arch/mod.rs::BUILTIN_ARCHITECTURE_DESCRIPTORS`
(component ids, `execution_capability`, hparam schema, `block_stack`) plus the
matching `BUILTIN_COMPONENT_DESCRIPTORS`, and wire the executor through
`materialize_builtin_executor_component` in
`crates/openasr-core/src/models/executor_component_registry.rs`.

The descriptor also requires an `encoder_attention_span`
(`OpenAsrEncoderAttentionSpan`, issue #68) declaring how the new
architecture's encoder scales with chunk length -- this is a mandatory field,
so a new architecture cannot compile without it:

- Full self-attention over the whole encoder input (the common case for a
  Conformer/Transformer/E-Branchformer/RoPE encoder) is `GlobalQuadratic`.
  Use `arch::DEFAULT_ENCODER_SAFE_CHUNK_SECONDS` (30s) for
  `max_safe_chunk_seconds` -- the value every major encoder family this repo
  has surveyed converges on (Whisper's fixed window, Moonshine's own "<30s"
  guidance, NeMo/Parakeet's 20-30s guidance, FunASR's 30000ms default,
  Dolphin's 30s training/eval padding, Cohere's 30s reference sliding
  window). **Only** override it with a different `max_safe_chunk_seconds`
  when the upstream model card states an explicit, different recommended
  chunk length, and cite that source in a comment next to the override (see
  `DEFAULT_ENCODER_SAFE_CHUNK_SECONDS`'s own doc comment for the citation
  format).
- An architecture-fixed attention window (like Whisper's 30s log-mel frame)
  is `FixedWindow`.
- A local/chunked streaming encoder with a bounded per-chunk cache (like
  Zipformer2's multi-scale cache) is `LocalChunked`.

**Do not** add a parallel hand-written family dispatch branch in
`crates/openasr-core/src/api/backend/native.rs` (`validate_local_native_model_pack_path`,
`validate_native_runtime_model_pack_contract`) or in
`crates/openasr-core/src/models/ggml_family_registry.rs`'s adapter list outside
the descriptor-driven path. Family selection goes through the registry so it
inherits the fail-closed unknown/ambiguous behavior for free (see reviewer
checklist).

### 2. Decode driver

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

### 3. Decode policy

Stop tokens, suppression, and text post-processing (including longform carry)
are data rows in
`crates/openasr-core/src/models/decode_policy_component_registry.rs`
(`BuiltinDecodePolicyComponentDescriptor`), keyed by the family's
`decode_policy_id`. Add a descriptor there; do not write a new
if/else post-processing branch elsewhere to get the same effect.

### 4. Package import

Reuse the shared import primitives: `local_source_import` (per-family module
under `crates/openasr-core/src/models/<family>/package_import.rs` calls the
shared helper, it does not reimplement path/zip handling),
`crates/openasr-core/src/ggml_runtime/gguf_write.rs` for GGUF emission, and the
shared metadata builder for the `openasr.*` GGUF KV keys documented in
[`.oasr` Package Contract v1`](../format/OASR_PACKAGE_CONTRACT_V1.md).

**Known interim state, not a template to copy:** quantization mode is
currently a **per-family** enum (`WhisperRuntimeQuantizationMode`,
`FireRedAedQuantizationMode`, `DolphinQuantizationMode`, ...) re-exported from
`crates/openasr-core/src/lib.rs`. A shared `PackQuant` /
`classify_quant_tensor` unification is planned (tracked separately). Until it
lands, matching the existing per-family enum shape is acceptable; once it
lands, **no new family may add another per-family quant enum** -- use the
shared one.

### 5. Tokenizer

- BPE families use the shared `gpt2_bpe` tokenizer path (see
  `crates/openasr-core/src/models/whisper/tokenizer.rs` and
  `crates/openasr-core/src/models/qwen/tokenizer.rs` for the calling
  convention).
- SentencePiece / metaspace families: a shared `SpmDecoder` is planned
  (tracked separately, not yet landed). Until it lands, do not hand-roll a
  one-off `▁` / `<0x..>` byte-fallback / id-to-token table inside a new
  family module if an equivalent already exists elsewhere in the tree --
  factor it out to a shared location instead of adding a third copy.

### 6. Neural network layers

Encoder/decoder stacks compose from the shared blocks in
`crates/openasr-core/src/nn/` (`attn.rs`, `ffn.rs`, `norm.rs`, `conv.rs`,
plus `encoder.rs` / `decoder.rs` helpers). Bypassing `nn/` for a new attention
or normalization variant needs a structural reason in the PR description (for
example X-ASR's Zipformer2 multi-scale streaming cache, which does not fit the
existing block shapes) -- add the new primitive to `nn/` rather than growing it
inline in the family module when the pattern is reusable.

### 7. Capabilities

`supports_phrase_bias`, diarization support, `emits_punctuation`, and
streaming registration are declared **once**, on the family's executor
(`capabilities()`), and read everywhere else through
`crates/openasr-core/src/models/executor_component_registry.rs`
(`builtin_executor_supports_phrase_bias_for_model_architecture` and its
siblings) or the streaming-executor completeness gate in
`build_builtin_ggml_streaming_execution_dispatch`. The model catalog
(`model-registry/catalog.json`) and any client/TS-side capability surface must
be generated or read from this single source, not hand-maintained as a second
constant. **Do not** declare the same capability as a separate literal in the
catalog card, a client-side table, and the executor -- three places drift the
way capabilities and decode logic drifted before.

### 8. GPU weight placement

A new encoder/decoder's persistent 2D matmul weights MUST bind through
`load_gguf_weight_context` (zero-copy, native quantized type) and its 1D
norm/bias tensors through `GgmlStaticTensorArena` -- both land in a
`GGML_BACKEND_BUFFER_USAGE_WEIGHTS` buffer, which is the only thing the ggml
scheduler will offload to a GPU backend. `runner.start_graph()` + an upload
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

### 9. Progress, history, cancel

Long-running transcription progress, history reporting, and cancel/pause
semantics run through the shared driver plumbing a new family's executor and
streaming registration plug into. Do not add a second progress/cancel path
that only exists for "batch mode" or "file mode" or a specific family --
single-request vs batch and file vs realtime must stay expressed as
parameters/paths through the one shared mechanism, not a forked
implementation.

## Reviewer checklist

Copy this into the PR description and check off each line (or replace the box
with a one-line structural justification for going another way):

- [ ] New architecture is a `BUILTIN_ARCHITECTURE_DESCRIPTORS` entry in
      `arch/mod.rs`; `ggml_family_registry` selection is covered by a test that
      fails closed on unknown and ambiguous family (see
      `dispatch_reports_unknown_family` / `returns_ambiguous_when_multiple_descriptors_match`
      in `crates/openasr-core/src/models/ggml_family_registry.rs` for the
      pattern to extend).
- [ ] Descriptor declares `encoder_attention_span` (issue #68). A
      `GlobalQuadratic` encoder uses `arch::DEFAULT_ENCODER_SAFE_CHUNK_SECONDS`
      unless the upstream model card states an explicit, different
      recommended chunk length -- if it does, the override cites that source
      in a comment. `builtin_architectures_declare_encoder_attention_span`
      (`arch/mod.rs`) and `encoder_attention_span_caps_every_builtin_architecture_on_the_production_path`
      (`api/backend/native_transcribe.rs`) must cover the new architecture.
- [ ] No hand-written decode step loop: `grep -rn 'for .*argmax\|while .*argmax'`
      (or an equivalent manual scan of the new executor) turns up nothing; the
      family implements `Seq2SeqGreedyDecodeStepExecutor` or calls
      `ctc_greedy_decode`.
- [ ] No parallel `validate_*` family-dispatch branch added to
      `api/backend/native.rs` outside the descriptor-driven path.
- [ ] `package_import` reuses `local_source_import` + `gguf_write`; no new
      ad hoc GGUF-writing or zip-parsing code. Quant handling matches the
      current per-family-enum convention (or the shared `PackQuant` once it
      lands) -- not a third scheme.
- [ ] Tokenizer reuses `gpt2_bpe` (BPE) or the shared SPM path once it lands;
      no new hand-rolled `▁`/byte-fallback table duplicating an existing one.
- [ ] Capabilities (`supports_phrase_bias`, diarization, `emits_punctuation`,
      streaming) are declared once on the executor and read via
      `executor_component_registry.rs`; no second literal in the catalog card
      or a client-side table.
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
      (golden-diff / stash-diff per [Model Onboarding](../MODEL_ONBOARDING.md#step-4--gate-it-byte-identically)).
      A brand-new family adds a bench-suite entry and freezes its first
      transcript as the reference instead.

## Relationship to Model Onboarding

[`MODEL_ONBOARDING.md`](../MODEL_ONBOARDING.md) is the "how do I write the
per-family code" walkthrough (steps 1-4, the quantized-weights runtime
contract, the honest gap list). This document is the narrower anti-fragmentation
contract: it exists so that as more families land, the shared facilities stay
singular instead of accumulating one bespoke variant per family. When the two
disagree on a mechanical detail, `MODEL_ONBOARDING.md` and the live code are
authoritative; file an issue to reconcile this doc.
