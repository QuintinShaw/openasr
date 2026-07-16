# GPU weight placement: the acceptance gate every new encoder/decoder must pass

Status: normative for new-architecture PRs and for any PR that adds or rewrites
a ggml compute subgraph (encoder, decoder, adapter, head). Companion to
[Model Onboarding](../MODEL_ONBOARDING.md) and the
[Model Onboarding Contract](model-onboarding-contract.md); this doc is the
narrow write-up of one specific defect class and the two-part gate that
catches it.

## The defect: golden-diff-correct, GPU-invisible

Dolphin's E-Branchformer encoder and (independently) X-ASR/Zipformer's
encoder both passed golden-diff / parity review -- the transcripts were
byte-identical to the reference. Both still shipped with their encoder
running **100% on CPU** even when the process was configured for a GPU
backend (Metal/HIP/Vulkan), because their encoder weights were never placed
in a buffer the ggml scheduler is allowed to offload.

**Why golden-diff didn't catch it:** golden/parity fixtures are short audio
clips run to prove numerical correctness, and ggml produces numerically
identical output whether an op runs on CPU or GPU. A short fixture "passes"
identically regardless of which backend actually executed the encoder, so it
provides *zero* signal about backend placement. The only way to see this
defect is to look at which backend actually ran the subgraph, which no
existing test did before this gate existed. This is why the acceptance gate
below has a mandatory dynamic half, not just review of the code.

## Why this happens: three weight-placement paths, only two are GPU-eligible

The ggml scheduler offloads a `MUL_MAT`/`MUL_MAT_ID` op to a GPU backend only
when its weight operand's buffer usage is
`GGML_BACKEND_BUFFER_USAGE_WEIGHTS` (`ggml-backend.cpp:908-928` in the vendored
ggml). OpenASR has exactly three code paths that can place a weight tensor,
and only two of them produce a WEIGHTS-usage buffer:

| Path | Entry point | Buffer usage | GPU-offload eligible? |
|---|---|---|---|
| **A. Static arena** | `GgmlStaticTensorArena` (`ggml_runtime/cpu_graph.rs`, `ensure_backend_buffer()` -> `allocate_with_usage(..., USAGE_WEIGHTS)`) | WEIGHTS | Yes |
| **B. Zero-copy bind** | `load_gguf_weight_context` / `bind_loaded` (`ggml_runtime/cpu_graph.rs`, `maybe_allocate_weight_buffer_from_host_ptr` -> `from_raw(..., USAGE_WEIGHTS)`) | WEIGHTS | Yes |
| **C. Per-request upload** | `runner.start_graph()` + `uploads.push(...)` / `pending_uploads.push(...)` / `<binding>.upload(...)` | the graph's transient **compute** buffer | **No** -- the scheduler puts the whole subgraph on CPU |

Path C exists for a reason: it is the *correct* way to feed genuine per-request
input (mel/fbank features, token ids, hidden states carried between steps).
The defect is specifically **using path C to carry the model's persistent
matmul weights** -- something that should be loaded once and reused across
every request, not re-uploaded (and thereby CPU-pinned) on every call.

**Correct pattern for a new encoder/decoder:** bind 2D matmul weights via path
B (`load_gguf_weight_context`, keeps the native quantized type, zero-copy) and
1D norm/bias tensors via path A (`GgmlStaticTensorArena`). Never carry
persistent weights through `start_graph()` + an upload call -- that call
shape is reserved for real per-request input. `whisper`, `qwen3-asr` (matmul
path), `cohere`, `moonshine`, `parakeet-tdt`/`parakeet-ctc` (via the shared
`fastconformer` core), `sensevoice`, `firered-aed`, and `wav2vec2-ctc` all
follow this pattern today; `dolphin`'s and `xasr_zipformer`'s encoders do not
(tracked: #131, #115).

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
scope of ever binding a WEIGHTS-usage buffer (`load_gguf_weight_context` /
`GgmlStaticTensorArena` / `bind_loaded`).

Run it locally:

```bash
scripts/gpu-weight-placement-gate.sh          # gate mode: exit 1 on a new, un-allowlisted finding
scripts/gpu-weight-placement-gate.sh --list   # report mode: always exit 0, just print findings
```

Two known, already-tracked violations (`dolphin`, `xasr_zipformer`) are
pre-declared in an `ALLOWLIST` at the top of the script so the gate does not
immediately fail every unrelated PR. **A new family must never be added to
`ALLOWLIST`** -- the allowlist exists only to grandfather the two families
found by the initial audit until they are fixed; a new architecture that hits
this gate has a real bug to fix, not a list entry to add. When a family's
encoder is fixed, remove its entry from `ALLOWLIST` in the same PR (the script
prints a "stale allowlist entry" note if it detects the family no longer
reproduces, as a reminder).

This is a heuristic over source text, not a proof -- see the script header for
the false-positive analysis that was done by hand across all eleven onboarded
families before this gate was written. It is not a substitute for the dynamic
half below; it is the cheap, always-on check that a hand-rolled encoder graph
at least *attempted* to use a WEIGHTS-usage path.

### Dynamic half: one real forward pass with the scheduler's split dump

The static gate can be fooled by a family that technically calls
`load_gguf_weight_context` somewhere in scope but doesn't actually route the
encoder's hot matmuls through the bound tensors (or binds only a token
embedding while the real transformer stack still uploads per-request). The
static gate narrows the search space; this step proves placement empirically.

Run a single real forward pass with the ggml scheduler's own instrumentation:

```bash
GGML_SCHED_DEBUG=2 GGML_DEBUG=1 OPENASR_GGML_BACKEND=metal \
  cargo run -p openasr-cli -- transcribe fixtures/jfk.wav --model <new-family> --format json
```

(`OPENASR_GGML_BACKEND` also accepts `hip`/`vulkan` where those backends are
built; use whichever GPU backend the target platform ships.) `GGML_SCHED_DEBUG`
and `GGML_DEBUG` are upstream ggml environment variables read directly by the
vendored scheduler/backend code (not OpenASR-specific), and print a per-split
dump like:

```text
## SPLIT #12: Metal # ... (encoder self-attention / FFN matmuls)
## SPLIT #13: CPU   # ... (this is the failure signature: the encoder landed here)
```

**Pass condition:** the encoder's/decoder's matmul-heavy splits show the
target GPU backend, not `CPU`. **Fail condition:** the encoder's op range is
entirely (or mostly) under a `## SPLIT #N: CPU` block while a GPU backend was
requested -- this is exactly the Dolphin/X-ASR signature, and it means the
weights loaded via whatever mechanism the code uses are not actually reaching
a WEIGHTS-usage buffer for the hot matmuls.

This step is why a **short golden/parity fixture is not sufficient evidence of
correct GPU placement** -- it proves numerical correctness, never backend
residency. Any PR introducing or materially changing an encoder/decoder graph
must run this dynamic check once against a real (not `mock`) backend and
either paste the relevant split-dump lines into the PR description, or state a
structural reason the subgraph is intentionally host-side (e.g.
`parakeet_tdt`'s prediction network, which is a genuinely host-side small RNN,
not a ggml graph at all).

## Model-onboarding checklist addition

Add to the [Model Onboarding Contract](model-onboarding-contract.md) reviewer
checklist and to the [Model Onboarding](../MODEL_ONBOARDING.md) walkthrough:

- [ ] `scripts/gpu-weight-placement-gate.sh` passes for the new family (no new,
      un-allowlisted finding).
- [ ] A one-shot `GGML_SCHED_DEBUG=2 GGML_DEBUG=1 OPENASR_GGML_BACKEND=<gpu
      backend>` real forward pass shows the encoder's/decoder's matmul splits
      on the GPU backend, not `CPU` -- pasted into the PR description (or a
      structural reason cited for why a subgraph is intentionally host-side).
- [ ] 2D matmul weights bind via `load_gguf_weight_context` (native quantized
      type, zero-copy); 1D norm/bias tensors bind via `GgmlStaticTensorArena`.
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
