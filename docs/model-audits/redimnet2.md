# Model release audit: redimnet2

> **Policy.** Should-support items MUST be `Supported`; anything else requires a
> detailed justification and an explicit unlock condition. This form ships with
> the model release. A family without a completed form does not enter the
> release flow: `tooling/publish-model/scripts/_manifest.py --public` fails
> closed on a missing or half-filled form.

| Field | Value |
| --- | --- |
| Family (`models-core.toml` `family`) | redimnet2 |
| Models covered | redimnet2-b6-cn (ReDimNet2-B6 CN-enhanced speaker embedder; fp16-only capability pack) |
| Auditor / date | Quintin / 2026-08-11 |
| Core version + commit audited | 0.1.31 macOS Metal comprehensive runtime: persistent ReDimNet2 graph, request-scoped execution policy, actual-placement telemetry, and bounded resident actor ownership |
| Bench hardware | Apple M1, 16GB, macOS (reference host). Embedder is a short-utterance support pack; ASR RTF/WER cells are Not applicable |

**How to fill.** Status is exactly one of:

- `Supported` -- implemented and verified for this family in this repo. Cite
  the evidence (test name, bench run, code path).
- `Not applicable` -- architecturally impossible or meaningless for this
  family. Say why, so nobody re-derives it.
- `Deferred` -- applicable but intentionally not done yet. Give the detailed
  justification AND the unlock condition (what measurement, upstream change,
  or milestone flips it to Supported).

Replace every `TODO:fill` HTML-comment marker; the release gate rejects any
leftover marker. Do not delete or rename the ten numbered section headings; the
gate checks all ten. Keep entries terse -- one form should take an afternoon,
not a week. The goal is that every release ships in its best known state, with
every consciously skipped optimization on the record.

## 1. Graph & scheduling

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Graph reuse / persistent session (no per-request graph rebuild) | Supported | `RedimNetResidentRuntime` owns one resident weight arena and a `GgmlPersistentGraphSession`; identical frame geometry reuses the graph, while a shape change or poisoned session rebuilds it fail-closed (`diarize/embed/redimnet/backbone.rs`). |
| Op fusion opportunities reviewed (norm+matmul, QKV, rope, ...) | Supported | Backbone ops reviewed against upstream ReDimNet2; stage blocks use the shared ggml path (`diarize/embed/redimnet/{backbone,ops}.rs`). No ASR-style rope/QKV fusion surface. |
| Batching / serve-batch path | Supported | `embed_batch` prepares bounded clip batches and executes them through the admitted actor checkout pool; Metal is intentionally limited to one owner-thread worker. This is finite speaker-window batching, not autoregressive ASR serve-batch. |
| Encode-decode pipelining | Not applicable | Single forward embedding; no autoregressive decode stage. |
| Arena / gallocr reuse across steps (no per-step allocator churn) | Supported | The resident static arena and same-shape persistent graph survive repeated embeddings on the actor owner thread; frame-geometry changes replace only the graph session, not parsed weights or the resident arena. |

## 2. Precision & quantization

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| KV cache quantization | Not applicable | No KV cache; embedding is a single feed-forward pass. |
| Activation precision policy chosen deliberately (f32 vs f16) | Supported | Graph runs f32 activations; pack stores projection weights at the requested quant while norms/biases/ASTP/BN stay f32 (`tooling/redimnet2/convert_redimnet2.py` `is_force_f32`). |
| Keep-quantized matmul (native Q blocks bound, no load-time dequant; RAM orders q4 < q8 < fp16) | Supported | Converter and runtime still support quantized GGUF matmul paths; the shipped public tier is fp16-only (q8_0/f32 packs withdrawn from the catalog and HF repo). |
| Quant tiers complete (q4_k / q8_0 / fp16) | Deferred | Public ship is fp16 only. q8_0/f32 were withdrawn (q8 slower and no meaningful size win on this 12.5M net; f32 is a parity/dev pack). q4_k intentionally omitted where cosine drift matters more than another size cut. Unlock: re-evaluate q8_0/q4_k only if same/other-speaker separation and size/speed justify a second public tier. |

## 3. Memory & data movement

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| mmap weight loading | Supported | GGUF `.oasr` load via `Weights::from_oasr` / ggml tensor reader (same path as other native packs). |
| Resident pool reuse across requests (weights stay resident) | Supported | Parsed host weights live in the service-root admitted host-neutral LRU; executable runtimes live in the bounded backend/lane-keyed pinned actor checkout pool. Target eviction and service unload release idle owners; there is no process-wide `shared_embedder` singleton (`diarize/embed/policy_runtime.rs`). |
| View contiguity tradeoffs audited (`cont`/copy nodes justified) | Supported | Backbone bring-up fixed a real gallocr view-of-output corruption and a `to1d` vs plain-reshape pre-pool flatten mismatch; parity harness pins the correct shapes (`redimnet/backbone.rs`). |
| Peak RSS/VRAM per shipped quant measured (quiet host) and reconciled against the weights+KV+activations budget; unexplained excess blocks release; catalog RAM requirement matches the measured peak | Supported | `redimnet_backend_benchmark` records current/peak RSS and Darwin physical footprint; `redimnet_fifteen_minute_bounded_batch_endurance` records bounded 15-minute execution. Shared ggml metadata contexts and native buffers are admitted before allocation rather than hidden in a family-specific allowance. Capability packs still have no separate public RAM column. |

## 4. Decode algorithms

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Greedy logits shortcuts (argmax path skips needless softmax/sort work) | Not applicable | No token decode. |
| Speculative decode: per-family verdict recorded (do it, defer it, or dead) | Not applicable | No token decode. |
| CTC blank-skip fast path (CTC families; otherwise Not applicable) | Not applicable | Not CTC. |
| Decode guards are zero-cost on the hot path (degenerate-loop guard etc.) | Not applicable | No token decode loop. |

## 5. Frontend & IO

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Mel/fbank frontend SIMD + parallelized | Supported | `RedimNetFrontend` / TFMelBanks port with parity vs `frontend_dump/` goldens (`redimnet/frontend.rs` + ignored parity tests). |
| Zero-copy audio path (no avoidable resample/copy hops) | Supported | Embedder takes mono f32 samples at 16 kHz; diarization feeds already-sliced segments without an extra resample hop when the session is 16 kHz. |
| VAD cost measured and accounted | Not applicable | VAD is a separate FireRedVAD stage; this pack only embeds speech regions it is given. |

## 6. Platform-specific

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Metal command batching + wired memory budget respected | Supported | Explicit Metal resolves to a direct full-device ggml backbone on one pinned owner thread. Official-reference parity observes all compute nodes on `MTL0`; the benchmark/endurance harness records RSS and physical footprint. Auto remains on CPU under the current measured Pareto policy. |
| CPU thread pool sized for P/E cores | Supported | Uses the shared ggml CPU runner defaults already tuned for the host pool. |
| Accelerate/BLAS used where it wins | Supported | ggml CPU backend picks host GEMM kernels; no family-specific BLAS bypass. |

## 7. Backend coverage matrix

Every cell must be answered. An unsupported backend is acceptable ONLY with a
justification and an unlock plan -- "nobody tried" is not a justification.
Golden-verified means byte/parity fixtures pass ON that backend;
utilization-measured means the GPU weight placement gate (or an equivalent
profile) proved the compute actually runs there (golden output alone cannot,
see `docs/design/gpu-weight-placement.md`).

| Backend | Supported? | Golden-verified? | Utilization measured? | Justification + unlock plan if unsupported |
| --- | --- | --- | --- | --- |
| CPU | yes | yes (official-reference cosine >= 0.9999) | yes (`redimnet_matches_official_reference_on_aux_audio` observes CPU/BLAS compute nodes) | |
| Metal | yes (explicit; Auto stays CPU) | yes (official-reference cosine >= 0.9999 and CPU/Metal semantic parity) | yes (the official gate observes all compute nodes on `MTL0`) | Host mel preparation remains preprocessing; the ggml backbone is the full-device candidate. |
| CUDA | no | no | no | The descriptor currently declares only CPU and Metal for this capability pack. Unlock: declare CUDA only after the shared aux route, official-reference parity, actual-placement telemetry, and bounded-memory gates pass on CUDA hardware. |
| Vulkan | no | no | no | Not in scope for this support pack until a shared Vulkan ggml path exists for aux packs. |
| HIP | no | no | no | Not in scope for this support pack until a shared HIP ggml path exists for aux packs. |

## 8. Correctness & quality

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| WER vs fp16 measured for every shipped quant tier | Not applicable | Not an ASR model; quality metric is embedding cosine / same-other speaker separation, not WER. |
| Model ref alias forms resolve identically everywhere (bare family / `family:canonical` / every `quant_tag_cases.json` alias accepted by CLI and server match logic; covered by the catalog-wide alias matrix test) | Supported | Catalog id `redimnet2-b6-cn` with the shipped fp16 suffix; capability-pack pull path shares the catalog-wide alias matrix. |
| Golden coverage includes long audio AND a cross-backend parity fixture | Supported | `redimnet_matches_official_reference_on_aux_audio` runs independently on CPU and Metal; `redimnet_cpu_and_metal_embeddings_stay_semantically_close` is the cross-backend gate; the bounded 15-minute diarization-window endurance test covers long-run ordering, cancellation, and memory stability. |
| Official decode parameters honored (suppression, stop tokens, upstream reference settings) | Not applicable | No decode parameters; frontend constants match the upstream TFMelBanks spec (`B6_FRONTEND_SPEC.md`). |
| Long-audio degradation checked (repetition, drift, truncation) | Not applicable | Stateless per-segment embed; no growing transcript/KV state inside the pack. |

## 9. Resource limits & fail-closed

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Max audio length / context budget derived and over-limit behavior fails closed | Supported | Backbone rejects inputs shorter than `TIME_STRIDE` frames with a structured error; diarization callers feed finite VAD segments rather than unbounded streams (`RedimNet2Model::forward`). |
| Streaming first-token latency floor documented (chunk accumulation math; streaming families, otherwise Not applicable) | Not applicable | Not a streaming ASR family; realtime diarization invokes the embedder on finite change-detector windows. |
| KV growth rate per audio second known | Not applicable | No KV cache. |
| Metal wired-memory profile captured | Supported | The backend benchmark and 15-minute endurance harness capture process RSS plus Darwin physical footprint while the Metal runtime is warm; native allocation admission is shared with the other ggml families. |
| Multi-session scaling behavior known (server concurrency) | Supported | Parsed weights are shared through the bounded host cache; executable runtimes use a bounded checkout pool with per-key instance limits, Metal owner-thread confinement, exact failed-candidate eviction, and service-root unload. |
| Energy footprint noted (battery-relevant platforms) | Deferred | No dedicated energy capture for the embedder. Unlock: optional Instruments energy sample on a multi-speaker diarize workload if battery becomes a product gate. |

## 10. Engineering completeness

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| `warm_up` is a real implementation, not a stub | Not applicable | Aux embedder packs do not expose the ASR executor `warm_up` hook. Candidate materialization nevertheless builds the admitted runtime and may run an explicit warm embedding before publishing the actor. |
| Reference dumper exists for this family | Supported | Stage-1 spike dumpers under `tmp/redimnet2-spike/` (frontend + backbone stage tensors + final embeddings) plus `tooling/redimnet2/convert_redimnet2.py`. |
| Registry / catalog / docs wired (MODEL_ONBOARDING checklist done) | Supported | `redimnet2-b6-cn` capability-pack entry, diarize card, ACKNOWLEDGMENTS credit, aux_pack_registry architecture id, pull-time validation. |
| Peer benchmark recorded (table below, all fields) | Deferred | Quality gate for this pack is cosine parity vs the upstream Python reference, not an ASR RTF peer table. Unlock: optional same-host embed RTF vs another published embedder on a fixed segment set if product needs a speed claim. |

### Peer benchmark record

Record enough that anyone can re-run this comparison later. "Faster than X" is
not auditable without the exact peer version, model build, audio, and machine.

| Field | Value |
| --- | --- |
| Peer project (name + commit or version) | Upstream PalabraAI/redimnet2 Python reference (tag v1.0.0 / commit 2a8d15f) — correctness peer, not an ASR RTF peer |
| Peer model + quant build | `b6-vb2+vox2+cnc2_v0-lm.pt` (fp32 checkpoint) |
| Peer program version | spike env recorded under `tmp/redimnet2-spike/` |
| Test audio (file, duration, language) | `fixtures/jfk.wav` + zh_sample + en_zh_mixed |
| Machine (chip, RAM, OS) | Apple M1, 16GB, macOS |
| Peer numbers (RTF / peak memory / utilization) | Not an RTF race; reference embeddings used for cosine parity |
| OpenASR numbers (RTF / peak memory / utilization) | CPU and explicit Metal both satisfy official-reference cosine >= 0.9999; Metal official runs report compute exclusively on `MTL0`. Reproducible RTF/RSS/physical-footprint values are emitted by `redimnet_backend_benchmark` and the bounded 15-minute endurance gate; public ship remains fp16. |

## Known dead ends (do not re-litigate)

Verdicts that apply to this family, so future work does not re-run dead
investigations. Repo-wide precedents to inherit where relevant: F16 activation
on Apple M1 (encoder-only gave zero win, cast economics lock the trunk;
verdict 2026-07-14); qwen speculative decode (acceptance alpha ~= 0.05, judged
dead). Add family-specific verdicts with the measurement behind each; write
"None yet" if the family has none.

| Dead end | Verdict / evidence | Date |
| --- | --- | --- |
| Pure-Rust hand-written ReDimNet2 forward (legacy WeSpeaker-style) | Rejected: family is ggml-graph by design (ggml-only invariant); converter emits ggml `ne` order packs | 2026-07 |
| Keeping WeSpeaker as a permanent public embedder fallback beside ReDimNet2 | Rejected after B6 public ship: diarization fails closed on ReDimNet2-B6 only; WeSpeaker was fully removed in a separate cutover | 2026-07 |
