# Model onboarding: adding or migrating an ASR architecture

This is the contributor checklist for adding or migrating an ASR architecture
to OpenASR. Read the normative [Model-family lifecycle](design/model-family-lifecycle.md)
first; it defines the required descriptor facets, pack proof chain, optimization
contract, generated projections, and cleanup rule. For getting OpenASR running,
see [QUICKSTART](QUICKSTART.md). For the reviewer-facing anti-fragmentation
checklist, see the [Model Onboarding Contract](design/model-onboarding-contract.md).

The architectural model is **one inventory row plus a narrow family adapter**:
shared `nn/` blocks, decode/cancel drivers, runtime ownership, and backend
placement stay in common layers; irreducible tensor binding and mathematical
topology stay under the family. A new family must not create a second registry,
hand-written central match, platform branch, or runtime lifecycle. A dedicated
graph is allowed only when its topology has a structural reason recorded in the
descriptor and covered by conformance.

Families are onboarded across several orchestration shapes:

- `Seq2SeqEncoderDecoder` — Whisper (hand-written reference, the bit-identity
  regression gate), Cohere Transcribe (data-driven composer), Moonshine
  (dedicated executor), FireRedASR-AED (Conformer encoder + Transformer
  decoder, no CTC branch, dedicated executor).
- `LlmDecoder` — Qwen3-ASR (data-driven composer).
- `Ctc` (non-autoregressive encoder + CTC head) — Parakeet-CTC and wav2vec2-CTC
  (`+data2vec`), dedicated executors; SenseVoice (SAN-M/DFSMN encoder with a
  20-block tp stage and a 4-token prompt splice), dedicated executor.
- Joint CTC + attention (E-Branchformer encoder + CTC head + Transformer
  decoder rescoring) — Dolphin, a dedicated executor over the WeNet recipe.
- Transducer (Zipformer2 encoder + RNN-T decoder/joiner) — X-ASR, a dedicated
  executor with its own multi-scale streaming cache topology; and Parakeet-TDT
  (FastConformer encoder + LSTM prediction network + Token-and-Duration joint,
  duration-driven frame skipping), a dedicated executor that reuses the shared
  conformer block for its encoder.

`whisper`, `moonshine`, `dolphin`, `x-asr`, `parakeet-tdt`, and `firered-aed` use
dedicated graph strategies because their mathematical topology is not expressed
by the current shared composer. Each row records the reason explicitly through
`OpenAsrBlockStackStrategy::ArchitectureGraph { reason }` and still uses the
shared admission, cancellation, ownership, and conformance seams. Composer
families call `validate_stage_against_descriptor` at construction so shape,
block kind, tensor scope, and layer count fail closed.

## What you get for free (shared, data-driven)

A new model inherits these without writing them. The exact symbol names are
implementation details; the lifecycle and descriptor contracts are normative:

- Top-level dispatch by `model_architecture` (`models/ggml_composed_executor.rs`).
- Pack admission through `PackEnvelope` -> `PackVerifier` -> `VerifiedPack` ->
  `AdmittedPack`; public converters return the writer's proof beside a diagnostic
  output path, execute requests carry the proof, and the direct-path ingress
  converts an untrusted candidate exactly once at the seam.
- The shared greedy decode loop, driven by your one step-executor impl
  (`models/seq2seq_greedy_decode.rs`).
- The decode policy (stop tokens, suppression, text post-processing) embedded in
  the row's typed decode-driver strategy. Reusable policy constants live in
  `models/decode_policy_component_registry.rs`; there is no separate family map.
- Layer-stack assembly over the shared `nn/` blocks plus the `compose_*` walkers
  and `validate_stage_against_descriptor`, which fails closed unless the stack
  matches the descriptor's shape / kind / scope / count (`arch/`).
- Registries for the audio frontend, tokenizer, prepared-runtime cache, runtime
  tensor contract, and other reusable typed runtime components, keyed by the
  component ids on your descriptor.
- Typed execution policies for phrase bias, LoRA binding, word timestamps, and
  prepared-runtime strategy. Reusing an existing component changes only the
  inventory row; it does not require a central family-id match. Add a new
  reusable component to its typed component registry once, then reference it
  from inventory rows.
- Generated offline/streaming dispatch, executor force-linking, validator
  dispatch, shared ownership/eviction coverage, and audit enumeration from the one
  descriptor inventory. Do not add a central hand-written match.
- One-open GGUF provenance through
  [`GgufRuntimeSourcePreflight`](design/runtime-source-preflight.md): contract
  validation, quoting, tensor readers, graph actors, and native weight contexts
  all consume the same preflighted file generation.
- Universal local file Voice ID routing. The architecture descriptor's
  `speaker_segmentation` selects either the family's in-decoder turns or the
  shared external FireRed/segmenter/ReDimNet/clustering path; both feed one
  identity and transcript-attribution stage. A new family does not implement a
  bespoke diarizer or person matcher.

**Punctuation fidelity is a product promise.** Whatever the model decodes is what
the user sees, in every mode (batch, streaming, dictation, server API). Text
production goes through the shared paths above -- a family may strip its own
control/tag tokens (special token tables are family-specific), but must never
add, drop, or rewrite punctuation in the transcript body. Do not introduce
family-local text munging; if the raw decode carries no punctuation, that is the
model's honest output.

## Step 1 — DATA: fill the descriptor facets

Run `cargo xtask family new <module_slug> [--profile-id <profile-id>]`, then fill
the one `OpenAsrArchitectureDescriptor` row in `arch/mod.rs`. Every required
facet must be explicit: `identity`, `pack_contract`, `execution_contract`,
`topology_contract`, `optimization_contract`, `quantization_contract`, and
`conformance_contract`. The row supplies component ids, an hparam schema in
`arch/hparams.rs`, the runtime validator, streaming cadence, encoder attention
span, semantic tensor classification, and named decode/block strategies.
Ownership, content-id eviction, graph reuse, cancellation, and admission are
shared-module invariants and are deliberately not self-declared by each family.

The execution facet also requires explicit typed capability choices:

- `phrase_bias`: `Unsupported`, `Always`, or `RequiresTensor { tensor_name }`;
  the last case is valid only when pack preflight proves that tensor exists.
- `adapter_binding`: `Unsupported` or the concrete executable binding strategy
  implemented by the family executor. Dispatch cross-checks both sides; a bool
  cannot self-certify support.
- `word_timestamps`: `DecodeInvariant` or `DecodeSensitive`.
- `prepared_runtime`: `FamilyOwned` or an existing shared reusable component.

These fields replace runtime family-id whitelists and branches. A new family
that reuses an existing component must not edit a central family match; only a
genuinely new reusable component extends the typed component registry.

Use `OpenAsrBlockStackStrategy::Shared(...)` when the topology is expressed by
the existing composer. Use `ArchitectureGraph { reason }` only for a genuinely
different mathematical topology, and record the structural reason in the row.
There is no unlabeled absence or `Default` escape. Startup validation and the
family conformance audit fail closed on dangling ids, duplicate keys,
shape/block mismatches, missing policies, or a missing dedicated reason.

The migration reference order is FunASR-Nano (pack contract), Parakeet-CTC
(shared CTC path), Qwen3-ASR (autoregressive/KV path), and Parakeet-TDT
(dedicated-topology boundary). Follow that order when choosing examples for a
new contributor guide or migration.

## Step 2 — DATA: select the decode policy

Set `topology_contract.decode_driver` to the shared seq2seq/CTC strategy carrying
the exact `BuiltinDecodePolicyComponentDescriptor`, or to a dedicated driver
with a structural reason. Reuse an existing policy constant when behavior is
identical; add one reusable constant only for genuinely new policy behavior.
Do not add a family-to-policy match table. The shared loop and cancellation fence
remain shared.

## Step 3 — CODE: the per-architecture pieces (the irreducible part)

These are the genuinely model-specific seams permitted by the architecture
contract:

- **Frontend** loader/params (log-mel vs fbank vs raw waveform; sample rate,
  n_mels, hop, ...).
- **Weights loader** that reads the GGUF/`.oasr` tensors by your
  `tensor_name_scope` and builds the resident layer handles. Bind matmul weights
  at their **native quantized type** — see [Runtime contract: keep quantized
  weights quantized](#runtime-contract-keep-quantized-weights-quantized).
  Dequantizing everything to f32 here silently throws away the whole q8/q4 win.
  Production constructors accept the shared runtime preflight, never a path or
  bare runtime source; see the
  [runtime-source provenance contract](design/runtime-source-preflight.md).
- **Audio encoder** glue — assemble its stage via the shared `compose_*` walker
  over the appropriate `nn/` block; add a new block under `nn/` only if your
  attention variant or head does not exist yet.
- **Logits / CTC head** (RMSNorm vs affine LN, tied vs untied embeddings; or a
  CTC greedy head for the `Ctc` shape).
- **One step-executor impl** that owns its per-step state (KV / cross-KV caches,
  position counters) and returns step logits; keep it small by reusing the `nn/`
  blocks and leaf helpers.
- Expose one `runtime_factory` in the descriptor row. The inventory projects it
  into offline and streaming dispatch; do not edit a central match or registry.

The adapter must not parse backend/provider names, create a second runtime cache,
or install a second cancellation callback. Reusable math belongs in `nn/`, ggml,
or a shared backend-neutral layer; the family consumes typed backend kinds and
capabilities. A typed backend-kind branch may express a real mathematical or
correctness policy, but raw provider spelling and platform discovery stay shared.
If a family can use an existing prepared-runtime or weight component, select it
in the execution facet; do not add a family-id branch. Only a genuinely new
reusable component may extend its typed component registry.

Composer-shape families must call `validate_stage_against_descriptor` once per
stage at construction so a data/code drift fails closed; a family that declares a
`block_stack` but skips the call leaves the descriptor informational.

### Realtime cadence is automatic — register a streaming executor

Live captions / dictation cadence is **descriptor-driven**, not something you
tune per family and not something the `.oasr` pack declares. There is no pack
metadata streaming flag and there is no third "buffered file-per-utterance"
realtime mode to wire up — that old path was removed. A realtime session can only
land on one of the shared mechanisms declared by `StreamingPartialGranularity`:

- **Revisable snapshot** (the default for encoder-decoder / CTC families): the
  shared driver re-decodes a growing/windowed buffer on an adaptive cadence, so
  partials appear *while the user is still speaking* and the FINAL is
  byte-identical to offline `execute()`. Incomplete windows are expected to
  produce displayable text. Implement `GgmlAsrStreamingExecutor` for your
  executor — reuse `build_seq2seq_streaming_session` (offline re-decode; works
  for CTC/attention and seq2seq alike) or `build_ctc_streaming_driver` (when you
  have a cheap CTC-greedy partial surface) — and declare
  `StreamingPartialGranularity::RevisableSnapshot` in the execution facet. The
  shared inventory projection wires it into the dispatch.
- **Utterance-complete snapshot** (ChatML utterance LLMs such as FunASR-Nano):
  incomplete windows may legally decode empty. Declare
  `StreamingPartialGranularity::UtteranceComplete` in the same facet. The shared
  driver may pad a short silence tail onto PARTIAL windows only; FINAL still
  uses the real unpadded audio.
- **Frame-sync append** (append-only, never revises emitted text): only for
  genuinely frame-synchronous architectures like X-ASR. Declare
  `StreamingPartialGranularity::FrameSyncAppend` in the same facet.

If the descriptor factory returns an executor without a valid streaming path, the startup
completeness gate in `build_builtin_ggml_streaming_execution_dispatch` **fails
loudly** rather than silently degrading your family to a stuttering
final-only cadence. Do not go looking for a metadata key or a per-family cadence
switch — there isn't one.

## Step 4 — gate the pack, runtime, and output

Package import is part of onboarding, not a separate publishing concern. The
family importer must write through `PackEnvelope`/`OasrPackWriter`; the writer
runs `PackVerifier` on the exact staged bytes and returns `VerifiedPack` before
the importer can expose its result. Every public converter result carries that
same proof; its `output_path` is diagnostic, not an execution capability. The
install and runtime path creates and carries the corresponding proof into
`AdmittedPack`; the core execute request consumes `VerifiedPack` and cannot fall
back to a bare path or a second family-local preflight. The public direct-path
ingress is the only candidate seam and verifies once before dispatch. FFI open
does the same full verification once and retains the proof for later calls.
Catalog installs additionally bind the signed catalog family to the family
projected from the verified route inside staging, before any content object is
exposed. Auxiliary packs use the same verifier and content admission with an
auxiliary route.

Before release, stage the finished artifact through the same Rust gate:

```text
openasr model-pack preflight <source.oasr> --stage <dest.oasr> --json
```

The publisher consumes the `openasr.model-pack-preflight.v1` receipt and must
match its content id, size, route, canonical catalog family id, architecture,
and pinned build commit to the conversion result/catalog row. Do not add a
Python metadata scanner or copy path; the CLI owns copy, sync, verification,
read-only sealing, and cleanup of invalid stages. The receipt is data-only
release evidence, not a replacement for the in-process `VerifiedPack` proof.

Conformance obligations have two distinct gates. Omitting a typed conformance
profile or gate declaration is a compile-time/weight-free-CI failure. Real
weights, backend smoke, and benchmark receipts remain release/manual gates
unless a dedicated artifact-backed CI job runs them; passing ordinary
weight-free CI is not evidence of measured performance or quality.

Artifact-backed recipes are deliberately not runtime registries. When local
weights exist, add the family-specific source path/import arguments to
`tooling/native-streaming-smoke/streaming_smoke.py`; when a catalog family is
published, add its documentation evidence keywords to
`tooling/publish-model/scripts/check_catalog_drift.py`. These rosters own test
assets and prose evidence, not dispatch facts, and must resolve their runtime
identity through the generated inventory.

After the pack gate, gate the execution result byte-identically:

If you extend or refactor an existing working family, you MUST prove byte-identity
(see [Performance](../perf/PERFORMANCE.md) and the bit-identity discipline): qwen
golden-diff, cohere stash-diff. A brand-new family has no prior golden, so add it
to the bench-suite (`perf/suite.toml`) and freeze its first transcript as the
reference. Then run the [keep-quantized self-check](#self-check-after-publishing)
on the rendered card.

**Byte-identity is necessary but not sufficient.** A golden/parity fixture
proves the *numbers* are right; it says nothing about which backend actually
computed them, because ggml produces the same output on CPU or GPU. Run the
[GPU weight placement gate](design/gpu-weight-placement.md) as well --
`scripts/gpu-weight-placement-gate.sh` plus a one-shot `GGML_SCHED_DEBUG=2`
real forward pass -- so a new encoder/decoder that quietly uploads its weights
per-request (and is therefore pinned to CPU under any GPU backend) doesn't
pass review on golden-diff output alone. This is exactly how Dolphin's and
X-ASR/Zipformer's encoders shipped GPU-invisible despite passing golden/parity
(#131/#115).

## Step 5 — choose the integration scope and close the release handoff

Runtime integration and public model distribution are separate scopes. Record
one of these scopes in the change and do not silently drift from one to another:

- **Core-only:** the family, importer, runtime, fixtures, and weight-free gates
  are complete, but there is no release candidate. Do not invent a publishing
  row, registry card, URL, digest, metrics, or public catalog entry merely to make
  the integration look complete.
- **Staged release candidate:** add the human-edited model source and publishing
  inputs under `tooling/publish-model/` with `release_public = false`, including
  `models-core.toml`, the corresponding `models-publish.toml` entry, and
  `cards/<model-id>.toml`. Copy `docs/model-audits/TEMPLATE.md` to
  `docs/model-audits/<family>.md` and begin recording real evidence. Generated
  `model-registry/models/*.toml` and catalog files are outputs; never edit them
  as independent truth. A source-only staged row is valid while real artifacts
  do not yet exist.
- **Public-ready:** in addition to the staged inputs, every shipped quant has a
  verified `.oasr` result sidecar, immutable upstream/Hugging Face revision,
  measured C-class receipt, completed family audit, and a real regression case
  plus committed golden under `tooling/family-regression/`. The release tooling,
  not the model-family code, decides when those facts may project to
  `public:true`.

The catalog ownership and staging rules are normative in
[Model Catalog, Registry, and Distribution](MODEL_CATALOG_ARCHITECTURE.md).
Use the read-only readiness report to identify the next safe release action:

```bash
python3 tooling/publish-model/scripts/onboarding_readiness.py --model <model-id>
```

For generated registry/catalog output, follow that document and use
`tooling/publish-model/scripts/regenerate_all.sh`; `--check` is the CI-safe drift
gate. Before the first public release, complete the audit requirements in
[Model release audits](model-audits/README.md) and register the smallest public
checkpoint/quant with a committed golden as described by
[`tooling/family-regression/README.md`](../tooling/family-regression/README.md).
The real-model workflow is deliberately outside PR CI because it downloads
weights and runs native inference.

No integration task authorizes upload, `release_public = true`, catalog signing,
publication, or deployment. Those are separately authorized release actions.
Until that authorization and the public-listing gate both exist, keep the model
core-only or staged and state the remaining evidence explicitly.

## Runtime contract: keep quantized weights quantized

**Hard requirement.** A quantized `.oasr` pack MUST feed its weights to ggml
`mul_mat` in their **native quantized type** (`Q8_0`, `Q4_K`, ...). **Never
dequantize every weight to f32 at load time.** A load-time dequant still produces
a smaller-on-disk q8/q4 file, but the graph then holds f32-resident weights and
computes in f32 — so you lose **both** wins the quant existed for: no RAM
reduction (peak RSS goes flat across quants) and no compute change (RTF goes flat
across quants). The point of a quant build is that the quantized blocks live in
the backend buffer and the matmul runs the int8 vec-dot path.

**The seam** (carry the raw blocks from pack to graph; never turn them into
`Vec<f32>`):

- Read the tensor as a native payload with
  `GgufTensorDataReader::weight_tensor_payload_by_name` (or the `owned_` variant)
  — it hands back `{ ggml_type, dims, bytes }`, not a dequantized copy.
- Allocate + upload at the native type via `new_tensor_from_weight_payload` /
  `new_matmul_weight_2d_typed` + `set_matmul_weight_bytes`, then pass the tensor
  straight into `graph.mul_mat`.
- **Reference family: `qwen`** (`models/qwen/llm_transformer.rs`,
  `models/qwen/logits_head.rs`) — every hot projection and the output head bind
  native; `dolphin` and `cohere` follow the same pattern.

**Orientation rule.** ggml `mul_mat(weight, input)` wants the weight operand as
`[ne0 = in, ne1 = out]`. Store and validate quant blocks in that **`[in, out]`**
orientation (qwen asserts `payload.dims == [input_width, output_width]`) so they
bind with **no repack**. A transpose at load defeats native binding.

**What stays f32/f16** — these are NOT `mul_mat` weights, so do not quantize them;
route them through the f32 vector loader (`host_tensor_f32_copy_dequantized_by_name`,
as qwen does for its 1-D tensors):

- 1-D norm weights and biases (RMSNorm/LayerNorm gamma/beta, projection biases).
- Convolution kernels (conv frontends, depthwise conv1d, convnext stems).
- Anything consumed by `get_rows` — token / decoder **embeddings** — ggml needs
  f32/f16 rows there.
- Positional / rotary tables and attention masks.
- Activation-times-activation matmuls (attention scores) are runtime tensors, not
  weights, and stay f32/f16 regardless of the pack's quant.

### Self-check after publishing

Open the model card's **Available builds** table (the publisher renders RAM peak +
RTF per quant) and read down the columns:

- **RAM peak must order `q4 < q8 < fp16`.** This is the load-bearing signal. If
  peak RSS is *flat* across quants, the family is almost certainly dequantizing
  every weight to f32 at load — the pitfall above. Fix the loader before shipping.
- **RTF should trend `q4 <= fp16`** on M1 CPU for encoder-heavy families. Two
  documented exceptions where flat/inverted RTF is expected and is NOT a defect:
  1. **Very small models** (whisper tiny/base tier): fixed non-matmul overhead
     dominates, so `q4 ~= fp16` RTF even when binding natively.
  2. **Autoregressive-decoder-dominated families at batch=1** (Phase-3 finding on
     `cohere-transcribe`): native `q8_0` can be *slightly slower* than fp16 on M1
     CPU because ggml's per-call quantize-activation + int8 vec-dot overhead is
     not amortized at M=1, while M1's native f16 FMA is very fast; `q4_k`'s larger
     bandwidth saving only tips it marginally faster. Both still win on **RAM** —
     which is exactly why RAM, not RTF, is the reliable keep-quantized check.

When RAM is flat *and* RTF is flat across quants, treat it as a keep-quantized
regression and audit the weights loader, not the pack.

### CI gate (K1) — enforced at integration, not after publishing

The RAM-ordering self-check above is a *post-publish* human read of the model
card. It is now backstopped by a **hard CI gate** that fails at integration time,
before any pack is built:

- `models::resident_runtime_audit::k1_host_f32_loader_sites_match_inventory` locks
  the set of source files that materialize a tensor to a host `Vec<f32>` (the
  reader's `host_tensor_f32_copy*` helpers) against the committed inventory
  `docs/model-audits/host_f32_loader_sites.txt`. **Adding a new host-f32 loader
  site turns CI red** until the file is listed there.
- Listing a file is a reviewed certification that it loads **only** the sanctioned
  f32/f16 tensors above (norms, biases, conv kernels, `get_rows` embeddings,
  positional/rotary tables, CMVN vectors). **Loading a rank-2 `mul_mat` weight to
  host f32 is forbidden** — bind it natively through the seam
  (`weight_tensor_payload_by_name` + `new_matmul_weight_2d_typed`; see
  `dolphin::executor::insert_pool_tensor`, which classifies each tensor). So the
  load-time-dequant pitfall now shows up as a red gate a reviewer must clear, not
  as a flat model-card column someone has to notice.

This complements the pack-header quant floor (`models::pack_quant_audit`, which
fails closed on a sub-Q8 encoder), which checks the *pack*; K1 checks the *loader*.

## Runtime contract: keep the prepared runtime resident

**Hard requirement.** A family's executor MUST NOT rebuild its heavy graph
runtime (encoder / decoder / adapter graphs, device-uploaded weights, embedding
and logits tables) **on every request**. Build it once per `(pack content id,
backend)` and keep it **resident**, reusing it across requests; only the
per-request, audio-dependent work (frontend, encode, prompt, decode) runs each
call. A per-request `Runtime::new()` on the hot path re-pays the full
weight-upload + graph-construction cost (seconds for an 8B-class decoder) to
re-derive state that never changes between requests against the same pack — this
is what mimo-asr did before its `MimoAsrPreparedRuntime` cache, and what granite
did before it was caught.

**The seam** is the explicitly injected `Arc<NativeExecutionServices>`:

- Send-safe parsed state belongs in its content/representation/lane-keyed
  admitted host cache. Mutable or thread-confined ggml state belongs in an
  admitted pinned-runtime actor or checkout pool. Cache identity includes the
  verified pack content id and exact execution lane; a bare path is never a
  runtime identity.
- Every owner carries the memory lease that admitted its allocation. Idle
  unload and pack replacement evict through the service root, then synchronously
  destroy owner-thread runtimes before refunding bytes. Do not add family-global
  maps, thread-local model owners, generation clocks, or no-op compatibility
  cleanup hooks.
- **Reference families:** `firered_llm`, `mimo_asr`, and `dolphin` for ASR
  runtimes; ReDimNet and DiariZen for auxiliary parsed-host plus pinned-actor
  ownership. New families must reuse these shared ownership primitives.

### CI gate (K2) — every ggml-executor family is checked

- `models::resident_runtime_audit::k2_every_ggml_executor_family_is_registered`
  derives the expected `models/<module_slug>/{executor,ggml_executor}.rs` set
  from the canonical architecture descriptor facets and compares it with the
  on-disk tree. **A new family's executor added without a descriptor row turns
  CI red**; there is no parallel classification table or exemption escape hatch.
- `k2_registered_families_reference_a_resident_cache` then requires every
  descriptor-derived family to reference a resident-cache primitive in its own
  module. The audit parses production Rust syntax (excluding comments, imports,
  and test-only code); shared `NativeExecutionServices` ownership, content-id
  eviction, and prepared-runtime reuse are module invariants, not per-family
  declaration fields.
- Per-family byte-identity of a **cache hit vs a fresh build** is proved by that
  family's own dev-pack e2e test (`resident_*_cache_reuse_across_consecutive_calls_stays_byte_identical`),
  which the static K2 gate backstops.

## Honest gap list — what still blocks true zero-code onboarding

1. **Step-executor construction is irreducibly per-family** and is not closeable
   by a generic executor: a factory returning `Box<dyn StepExecutor + 'assets>`
   that borrows prepared assets is a self-referential owner+borrow, not
   expressible in safe Rust. Accept it as the per-arch unit and minimize it with
   templates + leaf-helper reuse.
2. **Asset loaders are per-family by necessity** (shapes, norms, positional
   schemes, tokenizers). No descriptor erases these.
3. **`block_stack` validates but does not yet *compose* at execution time** — the
   descriptor gates shape / kind / scope / count; it does not drive the graph
   build. Data-driven *assembly* re-hits the `&mut` builder GAT lifetime wall and
   stays blocked until either Rust lifetime ergonomics change or an
   arena/slotmap-of-handles indirection is proven byte-identical.
4. **A genuinely new shape still needs a new `orchestration_shape` variant**
   (e.g. streaming, rotating/sparse KV, transducer). Existing shapes
   (`LlmDecoder`, `Seq2SeqEncoderDecoder`, `Ctc`) onboard as data + executor;
   anything outside them is new shared code regardless.
