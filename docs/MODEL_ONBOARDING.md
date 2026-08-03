# Model onboarding: adding a new ASR architecture

This is the contributor checklist for adding a new ASR architecture to OpenASR.
(For getting OpenASR running, see [QUICKSTART](QUICKSTART.md).) For the
reviewer-facing anti-fragmentation checklist -- which shared facilities a new
family PR must reuse instead of re-implementing -- see the
[Model Onboarding Contract](design/model-onboarding-contract.md).

The architectural model is exactly llama.cpp's: **shared `nn/` blocks + a thin
per-family executor that hand-writes its `compose_*` layer loop, gated by a
load-bearing block-stack descriptor.** There is intentionally **no generic
executor** — the per-family step executor is the per-architecture unit of code,
just as llama.cpp keeps a per-architecture `build_graph`. So "new model = data +
a few new blocks + one step executor" is the target, not "new model = data, zero
code". What is genuinely **data** (no new code) vs the irreducible
**per-architecture code** is spelled out below.

Eleven families are onboarded today across several orchestration shapes:

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

`whisper`, `moonshine`, `dolphin`, `x-asr`, `parakeet-tdt`, and `firered-aed` are intentional `block_stack: None`
dedicated executors (the "a few new blocks" a genuinely-new architecture is
permitted); all composer-shape families call `validate_stage_against_descriptor`
at construction so a descriptor's shape / block-kind / tensor-scope / layer-count
is enforced fail-closed.

## What you get for free (shared, data-driven)

A new model inherits these without writing them. Pointers are module-level on
purpose — the exact symbol names drift, so read the code for current names:

- Top-level dispatch by `model_architecture` (`models/ggml_composed_executor.rs`).
- The shared greedy decode loop, driven by your one step-executor impl
  (`models/seq2seq_greedy_decode.rs`).
- The decode policy (stop tokens, suppression, text post-processing), keyed by
  `decode_policy_id` (`models/decode_policy_component_registry.rs`).
- Layer-stack assembly over the shared `nn/` blocks plus the `compose_*` walkers
  and `validate_stage_against_descriptor`, which fails closed unless the stack
  matches the descriptor's shape / kind / scope / count (`arch/`).
- Registries for the audio frontend, tokenizer, prepared-runtime cache, and
  runtime tensor contract, keyed by the component ids on your descriptor.
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

## Step 1 — DATA: register the architecture (no graph code)

In `arch/mod.rs`, add the component id consts, then a
`BUILTIN_ARCHITECTURE_DESCRIPTORS` entry (family / architecture / component ids,
`execution_capability`, `speaker_segmentation`, an hparam schema in
`arch/hparams.rs`, and a `block_stack`
in `arch/block_stack.rs`) plus the matching `BUILTIN_COMPONENT_DESCRIPTORS`. If
you pick an existing `orchestration_shape` (`LlmDecoder`, `Seq2SeqEncoderDecoder`,
`Ctc`, or transducer) with existing block kinds, the whole step is pure data — an
acceptance test proves a new such model validates with no new shape / kind /
executor. Set `block_stack: None` only for a hand-written, never-composed family
(the regression-gate role whisper plays). Startup validation fails closed on any
dangling id, duplicate hparam key, or block-kind/shape mismatch.

## Step 2 — DATA: register the decode policy

Add a descriptor to the decode-policy registry keyed by your `decode_policy_id`:
execution kind (greedy seq2seq or CTC), stop-token kind, suppression, and
text-normalization rules. This is data; the loop itself is shared.

## Step 3 — CODE: the per-architecture pieces (the irreducible part)

These are genuinely model-specific and are the "a few new blocks" the model
permits:

- **Frontend** loader/params (log-mel vs fbank vs raw waveform; sample rate,
  n_mels, hop, ...).
- **Weights loader** that reads the GGUF/`.oasr` tensors by your
  `tensor_name_scope` and builds the resident layer handles. Bind matmul weights
  at their **native quantized type** — see [Runtime contract: keep quantized
  weights quantized](#runtime-contract-keep-quantized-weights-quantized).
  Dequantizing everything to f32 here silently throws away the whole q8/q4 win.
- **Audio encoder** glue — assemble its stage via the shared `compose_*` walker
  over the appropriate `nn/` block; add a new block under `nn/` only if your
  attention variant or head does not exist yet.
- **Logits / CTC head** (RMSNorm vs affine LN, tied vs untied embeddings; or a
  CTC greedy head for the `Ctc` shape).
- **One step-executor impl** that owns its per-step state (KV / cross-KV caches,
  position counters) and returns step logits; keep it small by reusing the `nn/`
  blocks and leaf helpers.
- **Register** it on the composed executor.

Composer-shape families must call `validate_stage_against_descriptor` once per
stage at construction so a data/code drift fails closed; a family that declares a
`block_stack` but skips the call leaves the descriptor informational.

### Realtime cadence is automatic — register a streaming executor

Live captions / dictation cadence is **descriptor-driven**, not something you
tune per family and not something the `.oasr` pack declares. There is no pack
metadata streaming flag and there is no third "buffered file-per-utterance"
realtime mode to wire up — that old path was removed. A realtime session can only
land on one of two shared mechanisms:

- **Incremental re-decode** (the default for every non-frame-sync family): the
  shared driver re-decodes a growing/windowed buffer on an adaptive cadence, so
  partials appear *while the user is still speaking* and the FINAL is
  byte-identical to offline `execute()`. Wire it by implementing
  `GgmlAsrStreamingExecutor` for your executor — reuse
  `build_seq2seq_streaming_session` (offline re-decode; works for CTC/attention
  and seq2seq alike) or `build_ctc_streaming_driver` (when you have a cheap
  CTC-greedy partial surface) — then register it in
  `build_builtin_ggml_streaming_execution_dispatch` with
  `StreamingPartialGranularity::Buffered`.
- **Frame-sync** (append-only, never revises emitted text): only for genuinely
  frame-synchronous architectures like X-ASR. Register with
  `StreamingPartialGranularity::FrameSync`.

If you register an offline executor but forget the streaming one, the startup
completeness gate in `build_builtin_ggml_streaming_execution_dispatch` **fails
loudly** rather than silently degrading your family to a stuttering
final-only cadence. Do not go looking for a metadata key or a per-family cadence
switch — there isn't one.

## Step 4 — gate it byte-identically

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

- `models::resident_runtime_audit::k2_every_ggml_executor_family_is_classified`
  locks the set of `models/<family>/{executor,ggml_executor}.rs` files against the
  `GGML_EXECUTOR_FAMILY_GATES` table. **A new family's executor added without a
  table entry turns CI red**, so a new AED family (e.g. a future granite / funasr
  executor) cannot land unclassified.
- `k2_resident_families_reference_a_resident_cache` then requires every family
  classified `Resident` to reference a resident-cache primitive in its own module.
  A one-shot family that genuinely should rebuild per request must be explicitly
  `Exempt("<reason>")` — reviewed, not silent.
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
