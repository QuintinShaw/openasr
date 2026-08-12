# GPU weight placement: the acceptance gate every new encoder/decoder must pass

Status: normative for new-architecture PRs and for any PR that adds or rewrites
a ggml compute subgraph (encoder, decoder, adapter, head). Companion to
[Model Onboarding](../MODEL_ONBOARDING.md) and the
[Model Onboarding Contract](model-onboarding-contract.md); this doc is the
narrow write-up of one specific defect class and the two-part gate that
catches it.

## The defect: golden-diff-correct, GPU-invisible

An encoder can pass golden-diff / parity review -- the transcripts are
byte-identical to the reference -- while still running **100% on CPU** even
when the process is configured for a GPU backend (Metal/HIP/Vulkan). This
happens when its weights are never placed in a buffer the selected ggml
backend can treat as resident accelerator weights.

**Why golden-diff didn't catch it:** golden/parity fixtures are short audio
clips run to prove numerical correctness, and ggml produces numerically
identical output whether an op runs on CPU or GPU. A short fixture "passes"
identically regardless of which backend actually executed the encoder, so it
provides *zero* signal about backend placement. The only way to see this
defect is to look at which backend actually ran the subgraph, which no
existing test did before this gate existed. This is why the acceptance gate
below has a mandatory dynamic half, not just review of the code.

## Why this happens: three weight-placement paths, only two are GPU-eligible

The ggml backend can keep a `MUL_MAT`/`MUL_MAT_ID` weight operand resident on
the selected GPU only
when its weight operand's buffer usage is
`GGML_BACKEND_BUFFER_USAGE_WEIGHTS` (`ggml-backend.cpp:908-928` in the vendored
ggml). OpenASR has exactly three code paths that can place a weight tensor,
and only two of them produce a WEIGHTS-usage buffer:

| Path | Entry point | Buffer usage | GPU-offload eligible? |
|---|---|---|---|
| **A. Static arena** | `GgmlStaticTensorArena` (`ggml_runtime/cpu_graph.rs`, `ensure_backend_buffer()` -> `allocate_with_usage(..., USAGE_WEIGHTS)`) | WEIGHTS | Yes |
| **B. Zero-copy bind** | `load_gguf_weight_context_from_preflight` / `bind_loaded` (`ggml_runtime/cpu_graph.rs`, `maybe_allocate_weight_buffer_from_host_ptr` -> `from_raw(..., USAGE_WEIGHTS)`) | WEIGHTS | Yes |
| **C. Per-request upload** | `runner.start_graph()` + `uploads.push(...)` / `pending_uploads.push(...)` / `<binding>.upload(...)` | the graph's transient **compute** buffer | **Not a resident-weight path** -- using it for a persistent weight can prevent the affected op or subgraph from being offloaded |

Path C exists for a reason: it is the *correct* way to feed genuine per-request
input (mel/fbank features, token ids, hidden states carried between steps).
The defect is specifically **using path C to carry the model's persistent
matmul weights** -- something that should be loaded once and reused across
every request, not re-uploaded (and thereby made ineligible for the intended
resident-weight placement) on every call. Genuine request inputs can share a
graph with resident weights without forcing the whole graph to CPU.

**Correct pattern for a new encoder/decoder:** bind 2D matmul weights via path
B (`load_gguf_weight_context_from_preflight`, keeps the native quantized type,
zero-copy) and
1D norm/bias tensors via path A (`GgmlStaticTensorArena`). Never carry
persistent weights through `start_graph()` + an upload call -- that call
shape is reserved for real per-request input. `whisper`, `qwen3-asr` (matmul
path), `cohere`, `moonshine`, `parakeet-tdt`/`parakeet-ctc` (via the shared
`fastconformer` core), `sensevoice`, `firered-aed`, and `wav2vec2-ctc` all
follow this pattern today. Every family must use these resident WEIGHTS paths;
there are no exceptions for persistent rank-2 `mul_mat` weights. This does not
ban the deliberately host-materialized classes in the audited inventory:
`get_rows` embedding tables, 1D norm/bias/CMVN values, convolution kernels,
positional/rotary tables, or construction-time derived constants such as a
folded BatchNorm affine. Those values still need bounded ownership and must be
uploaded or bound to a resident device tensor before accelerated graph
execution when they participate in device compute.

Geometry-stable model constants must not silently regress into per-request
work. Whisper's encoder prelude caches its convolution tensors and sliced
positional embedding in the owner-thread actor's static WEIGHTS arena; after
the first build, each request uploads only mel input. XASR's compact relative
position table is generated and uploaded during reusable full-encoder graph
construction, guarded by that graph's `static_uploaded` state; the temporary
host vector is released after upload and is not rebuilt for later chunks or
requests using the same graph geometry. These are construction-time constants,
not path-C persistent weights and not per-request neural computation.

## Resolved backend is a request contract

Execution policy resolves `Auto`, explicit CPU, generic accelerated, and
provider-constrained accelerated requests before a family constructs any graph.
Family graph configuration consumes that resolved backend. It may choose a
scheduler, thread count, graph capacity, precision, or a backend-native operator
lowering, but it must not replace a resolved GPU backend with CPU.

`FullDevice` is both the neural placement contract and the boundary that
selects direct-backend graph execution. ggml's multi-backend scheduler requires
a CPU fallback participant, so it is incompatible with a request that forbids
CPU neural compute. The shared graph-config helper and the runner constructor
therefore both disable the scheduler for a GPU `FullDevice` candidate. Runtime
telemetry still verifies that every compute node used the selected provider;
any CPU/BLAS or different-provider node is a placement violation and
invalidates the candidate.

This separation is what makes an explicit accelerated request auditable: every
neural graph for that candidate remains on the selected accelerator, and a
placement mismatch fails closed. Family-specific CPU preferences are legal only
while resolving `Auto`; they must be attached to the original Auto intent before
executor dispatch, never exposed as stage environment kill switches or inferred
again from a resolved GPU backend. Cohere's measured multi-chunk Metal decoder
preference is one such Auto-only hint. Explicit and provider-constrained
accelerated Cohere requests ignore it and keep both encoder and decoder on the
resolved accelerator.

## The acceptance gate

### Static half: `scripts/gpu-weight-placement-gate.sh`

Pure grep over committed source, no build, no inference -- cheap enough to run
on every PR (wired into CI's `lint` job). For each family directory under
`crates/openasr-core/src/models/`, it scans the files named `*encoder*.rs` /
`*executor*.rs` in that directory (family-scope, not single-file -- see the
script's own header comment for why: some families, like whisper, legitimately
split "the per-request graph" from "the resident weight arena" across two
files) and flags the family when that scope shows upload-fed graph
construction (`start_graph()` + an upload call) but **no** evidence anywhere in
scope of ever binding a WEIGHTS-usage buffer (`load_gguf_weight_context_from_preflight` /
`GgmlStaticTensorArena` / `bind_loaded`).

Run it locally:

```bash
scripts/gpu-weight-placement-gate.sh          # gate mode: exit 1 on any finding
scripts/gpu-weight-placement-gate.sh --list   # report mode: always exit 0, just print findings
```

The gate has zero exemptions: every finding fails immediately. A family that
hits this check has a real weight-placement defect to fix, not an entry to add
to a suppression list.

This is a heuristic over source text, not a proof -- see the script header for
the false-positive analysis that was done by hand across all eleven onboarded
families before this gate was written. It is not a substitute for the dynamic
half below; it is the cheap, always-on check that a hand-rolled encoder graph
at least *attempted* to use a WEIGHTS-usage path.

### Dynamic half: one real forward pass with placement telemetry

The static gate can be fooled by a family that technically calls
`load_gguf_weight_context_from_preflight` somewhere in scope but doesn't actually route the
encoder's hot matmuls through the bound tensors (or binds only a token
embedding while the real transformer stack still uploads per-request). The
static gate narrows the search space; this step proves placement empirically.

Run a native short-audio receipt against a real pack and the explicit device:

```bash
openasr bench-receipt short-audio \
  --model <new-family>:<quant> \
  --model-pack /path/to/model.oasr \
  --audio fixtures/jfk.wav \
  --backend native --device metal \
  --runs 3 --warmup-runs 2 \
  --out /tmp/<new-family>-metal.json
```

Use `--device cuda`, `hip`, or `vulkan` on platforms that ship those providers.
The receipt binds the pack and audio SHA-256 and records the actual compute
backend keys reported by the runtime telemetry collector. For example:

```text
"observed_placement": {
  "compute_nodes_by_backend": { "MTL0": 1702 }
}
```

**Pass condition:** the receipt succeeds and every observed compute key belongs
to the requested provider (`MTL*` for Metal, with no `CPU`/`BLAS`). **Fail
condition:** the receipt fails with a placement violation, reports no real
compute nodes, or contains a CPU/different-provider key. The first case usually
means the weights are not reaching a WEIGHTS-usage buffer; the latter cases can
also expose a dispatch or telemetry wiring defect.

This step is why a **short golden/parity fixture is not sufficient evidence of
correct GPU placement** -- it proves numerical correctness, never backend
residency. Any PR introducing or materially changing an encoder/decoder graph
must run this dynamic check once against a real (not `mock`) backend and
attach the validated receipt (or its pack/audio/placement fields) to the PR
evidence. A small recurrent network is not by itself a reason for host neural
compute: Parakeet-TDT's predictor/joint run as persistent device graphs on
accelerated routes.

## Model-onboarding checklist addition

Add to the [Model Onboarding Contract](model-onboarding-contract.md) reviewer
checklist and to the [Model Onboarding](../MODEL_ONBOARDING.md) walkthrough:

- [ ] `scripts/gpu-weight-placement-gate.sh` passes for the new family (no
      finding).
- [ ] A real `bench-receipt short-audio --device <gpu provider>` run succeeds,
      binds the expected pack/audio hashes, and reports only the requested
      provider's compute keys -- attached to the PR evidence.
- [ ] 2D matmul weights bind via `load_gguf_weight_context_from_preflight`
      (native quantized type, zero-copy); 1D norm/bias tensors bind via
      `GgmlStaticTensorArena`. The bare-source loader is test-only; production
      code carries the existing preflight proof.
      `runner.start_graph()` + an upload call (`uploads.push` /
      `pending_uploads.push` / `.upload(...)`) is reserved for genuine
      per-request input (features, token ids, step state) -- never for
      persistent model weights.

## Related

- [Model Onboarding](../MODEL_ONBOARDING.md#runtime-contract-keep-quantized-weights-quantized) --
  the adjacent "keep quantized weights quantized" contract (native `Q8_0`/`Q4_K`
  binding vs. dequantizing to f32 at load time). That contract is about
  *numeric type*; this doc is about *buffer placement*. A family can get one
  right and the other wrong independently -- check both.
- [Model Onboarding Contract](model-onboarding-contract.md) -- the general
  anti-fragmentation reviewer checklist this gate is now one item of.
