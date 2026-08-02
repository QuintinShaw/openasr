# Model release audit: granite-speech

> **Policy.** Should-support items MUST be `Supported`; anything else requires a
> detailed justification and an explicit unlock condition. This form ships with
> the model release. A family without a completed form does not enter the
> release flow: `tooling/publish-model/scripts/_manifest.py --public` fails
> closed on a missing or half-filled form.

| Field | Value |
| --- | --- |
| Family (`models-core.toml` `family`) | granite-speech |
| Models covered | granite-speech-4.1-2b (IBM Granite Speech 4.1 2B; Conformer encoder + Q-Former projector + 2B decoder); tiers fp16 / q8_0 / q4_k |
| Auditor / date | main-loop / 2026-08-02 |
| Core version + commit audited | origin/main @ `a614a0c7` (post #264 family + #268 publish-prep; packs rebuilt 2026-08-02 with ggml `[in,out]` rank-2 dims) |
| Bench hardware | Apple M1, 16 GB unified memory, macOS (single reference host; quiet-window numbers disclosed) |

**How to fill.** Status is exactly one of:

- `Supported` -- implemented and verified for this family in this repo. Cite
  the evidence (test name, bench run, code path).
- `Not applicable` -- architecturally impossible or meaningless for this
  family. Say why, so nobody re-derives it.
- `Deferred` -- applicable but intentionally not done yet. Give the detailed
  justification AND the unlock condition (what measurement, upstream change,
  or milestone flips it to Supported).

## 1. Graph & scheduling

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Graph reuse / persistent session (no per-request graph rebuild) | Supported | `GraniteSpeechPreparedRuntime` is thread-local cached by `(PackContentKey, backend)` with idle-unload generation tagging (`executor.rs`). Decoder Metal path: device-resident KV arena + reusable single-token decode graph + prefill batch seed (`decode_session.rs`, `f26ae51f` / `8af561b9`). Encoder/projector are request-invariant runtimes bound once from mmap. Test: `metal_resident_reusable_graph_matches_reference_cold_and_warm` cold+warm transcript parity. |
| Op fusion opportunities reviewed (norm+matmul, QKV, rope, ...) | Supported | Decoder uses Metal `flash_attn_ext` where geometry allows; prefill last-position lm-head; Granite scalars (attention/residual/embedding/logits multipliers) preserved on the resident path. Encoder is Conformer (macaron FF + Shaw rel-pos + depthwise conv) with host-folded BatchNorm affine -- no missing fused-QKV class gap on the rank-2 weights (they bind keep-quantized). Rejected as primary wins after A/B: full-vocab D2H (~0.05 ms/tok), F16 KV, packed gate/up without gain (see Known dead ends). |
| Batching / serve-batch path | Deferred | Shared serve-batch engine is opt-in / niche for local; granite reaches decode via the shared greedy driver but has no family serve-batch runtime. Unlock: only if cloud multi-tenant demand arises (same product judgment as funasr-nano / firered2-llm). |
| Encode-decode pipelining | Supported | File path is single-shot encoder->projector->prefill->decode; long audio uses the generic longform / P1 slice pipeline upstream. Family is carry-disabled seq2seq, so concurrent slice overlap is admission-eligible when the shared pipeline is on. |
| Arena / gallocr reuse across steps (no per-step allocator churn) | Supported | Metal decode: fixed resident K/V arena + persistent reuse graph (scheduler off on GPU). CPU path rebuilds per token by design of the scheduler-on fallback. Shared Metal cancellation no longer segments every 32 nodes (`863a1201`), removing a measured multi-hundred-ms stall class on long graphs. |

## 2. Precision & quantization

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| KV cache quantization | Deferred | Production path is device-resident f32 KV (Metal) / host f32 history (CPU). No q8_0 KV policy wired for this fork (Granite scalars + custom pre/post attention). Unlock: port shared phase-1 q8 KV only after a quiet-host A/B shows peak-RSS win without WER regression on the frozen en set. |
| Activation precision policy chosen deliberately (f32 vs f16) | Supported | Deliberate f32 activations. Repo-wide M1 verdict (2026-07-14): F16 activation encoder-only gave zero win; cast economics lock the trunk. Family-specific F16 KV A/B also rejected (see Known dead ends). |
| Keep-quantized matmul (native Q blocks bound, no load-time dequant; RAM orders q4 < q8 < fp16) | Supported | Decoder projections/norms/lm_head: zero-copy keep-quantized via `GraniteSpeechDecodeSession::new_keep_quantized` + `GgmlLoadedWeightContext` (`7539491f`). Encoder: all rank-2 / conv weights bound from loaded context; host-f32 ONLY for per-layer BatchNorm 1-D fold inputs (`encoder_graph.rs` `GraniteSpeechEncoderRuntime`). Projector: matrices mmap-bound; host-f32 ONLY tiny `projector.query` (`qformer.rs`). Production still dequants the get_rows token-embedding table once into the prepared runtime (sanctioned embedding class; `runtime_provider.rs` prefix load). K1 inventory lists the four sites. Historical catastrophe closed: peak RSS ~16.1 GB all-f32 -> ~3.6 GB class keep-quantized. |
| Quant tiers complete (q4_k / q8_0 / fp16) | Supported | Three packs built: `granite-speech-4.1-2b.{fp16,q8_0,q4_k}.oasr` (~4.3 / 2.3 / 1.4 GiB). en WER on frozen-eval 40 utts: fp16 = q8_0 = q4_k = 0.54%. JFK transcript parity q8/q4 Metal cold/warm. |

## 3. Memory & data movement

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| mmap weight loading | Supported | Pack tensors bound through `load_gguf_weight_context` / mmap reader; shared loaded-weight context cache prevents multi-context OOM on Apple Silicon (`642be734`). |
| Resident pool reuse across requests (weights stay resident) | Supported | K2: `("granite_speech", Resident)` in `GGML_EXECUTOR_FAMILY_GATES`. Prepared runtime (encoder + projector + decode session + embed table) cached cross-request; `release_session_scoped_buffers` resets session-scoped state before re-store. |
| View contiguity tradeoffs audited (`cont`/copy nodes justified) | Supported | Resident decode path audited during KV/set_rows + reusable-graph work; cont/copy limited to mask/seed and embedding row assembly. No unexplained full-weight cont on the hot path. |
| Peak RSS/VRAM per shipped quant measured (quiet host) and reconciled against the weights+KV+activations budget; unexplained excess blocks release; catalog RAM requirement matches the measured peak | Supported | Quiet 3-run `/usr/bin/time -l` median max RSS on Apple M1 (isolated `OPENASR_HOME`, JFK 11s, native Metal, packs rebuilt 2026-08-02): fp16 **5909938176** (~5636 MiB) / q8_0 **3648585728** (~3479 MiB) / q4_k **2627207168** (~2505 MiB). Ordering q4 < q8 < fp16 holds; was ~16.1 GB pre keep-quantized. Catalog `peak_rss_bytes` written from these medians on the public flip (`tmp/publish/granite-speech-4.1-2b/metrics.json`). |

## 4. Decode algorithms

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Greedy logits shortcuts (argmax path skips needless softmax/sort work) | Supported | Shared driver `run_builtin_seq2seq_decode_policy` / `run_seq2seq_greedy_decode_loop_v0`; single-pass argmax, no top-k sort on the full vocab. Prefill emits last-position logits only. |
| Speculative decode: per-family verdict recorded (do it, defer it, or dead) | Deferred (dead) | 2B-class acceptance is expected near the qwen 0.6B dead verdict (alpha ~= 0.05). No draft model in tree. Unlock: only revisit if a measured acceptance rate on this checkpoint exceeds ~0.3 on real ASR prompts. |
| CTC blank-skip fast path (CTC families; otherwise Not applicable) | Not applicable | Production path is encoder->Q-Former->LLM greedy decode. Encoder CTC heads exist as graph tensors for parity with the checkpoint but are not the product decode path. |
| Decode guards are zero-cost on the hot path (degenerate-loop guard etc.) | Supported | Shared driver degenerate n-gram guard (issue #60 class); registry-routed policy id `granite-speech.greedy.seq2seq.v0`. No per-step full-vocab host work added by the family. |

## 5. Frontend & IO

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Mel/fbank frontend SIMD + parallelized | Supported | Family mel frontend (`GraniteSpeechMelFrontend`, 80-mel / 16 kHz) on the shared audio prep path; cost is negligible vs 2B decode on jfk-class audio. |
| Zero-copy audio path (no avoidable resample/copy hops) | Supported | Uses the shared prepared-audio path (in-memory f32 when non-WAV; WAV passthrough). No family-specific extra temp-WAV hop. |
| VAD cost measured and accounted | Not applicable | No VAD stage in the granite-speech offline pipeline. |

## 6. Platform-specific

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Metal command batching + wired memory budget respected | Supported | Metal is the product path: resident KV + reusable graph + flash-attn; shared cancel segmentation fixed (`863a1201`, q4 A/B ~3390->2592 ms). Apple7 residency sets enabled in shared ggml pin. Cold/warm Metal transcript test green. |
| CPU thread pool sized for P/E cores | Supported | Inherits shared `GgmlCpuGraphConfig` adaptive thread policy by workload class; no family override. |
| Accelerate/BLAS used where it wins | Supported | Inherits vendored ggml CPU GEMM / Accelerate selection; no family-specific BLAS gap. |

## 7. Backend coverage matrix

Every cell must be answered. An unsupported backend is acceptable ONLY with a
justification and an unlock plan -- "nobody tried" is not a justification.
Golden-verified means byte/parity fixtures pass ON that backend;
utilization-measured means the GPU weight placement gate (or an equivalent
profile) proved the compute actually runs there (golden output alone cannot,
see `docs/design/gpu-weight-placement.md`).

| Backend | Supported? | Golden-verified? | Utilization measured? | Justification + unlock plan if unsupported |
| --- | --- | --- | --- | --- |
| CPU | Yes | Yes | N/A (CPU) | Decoder/encoder unit + e2e paths run on CPU; incremental-KV bit-exact vs recompute on CPU. |
| Metal | Yes | Yes | Yes | `metal_resident_reusable_graph_matches_reference_cold_and_warm`; ggml_metal_init logs; warm E2E peer gate on M1; cancel/residency A/B on Metal. |
| CUDA | Deferred | No | No | No CUDA host in the audit window. Family is ggml-op based (mul_mat, flash_attn_ext, set_rows, norms, conv). Unlock: CI/community CUDA golden + one utilization profile. |
| Vulkan | Deferred | No | No | Same as CUDA. Unlock: AMD/Intel Vulkan host golden + utilization. |
| HIP | Deferred | No | No | Same as CUDA. Unlock: HIP host golden + utilization. |

## 8. Correctness & quality

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| WER vs fp16 measured for every shipped quant tier | Supported | Frozen en eval (40 utts): fp16 / q8_0 / q4_k all **0.54%** WER; q8 matches fp16 at the transcript level on the measured set; q4_k lossless on that set. zh not in upstream language claim -- not measured. |
| Model ref alias forms resolve identically everywhere (bare family / `family:canonical` / every `quant_tag_cases.json` alias accepted by CLI and server match logic; covered by the catalog-wide alias matrix test) | Supported | Architecture + registries wired (`GRANITE_SPEECH_*` in `arch/mod.rs`). Public catalog row ships canonical quant tags `fp16` / `q8_0` / `q4_k` with aliases `granite-speech` / `granite`; catalog-wide alias matrix walks every public row (including this family after the public flip). Shared `quant_tag_cases.json` covers the quant-tag surface. |
| Golden coverage includes long audio AND a cross-backend parity fixture | Supported | Short JFK unit goldens + Metal cold/warm resident-graph parity (`metal_resident_reusable_graph_matches_reference_cold_and_warm`). Long-audio: weight-free SharedWindow planner/policy gates and stop-reason truncation mapping in `native_transcribe.rs` / `executor.rs`; ignore-form multi-slice longform e2e + CPU/Accelerated parity skeletons on `fixtures/longform_en_zh.wav` (pack-gated). Cross-backend: CPU incremental-KV bit-exact vs recompute + Metal cold/warm transcript parity. |
| Official decode parameters honored (suppression, stop tokens, upstream reference settings) | Supported | Greedy path, EOT `100257`, fixed transcription question prompt, audio-token embedding splice, Granite attention/residual/embedding/logits multipliers on the resident graph. Shared degenerate-loop guard for repetition. |
| Long-audio degradation checked (repetition, drift, truncation) | Supported | Single-buffer ceiling is decoder-context-derived `GRANITE_SPEECH_MAX_INPUT_SECONDS = 382` with typed `AudioTooLong` (`capacity.rs` / `ensure_audio_within_capacity`); generation backstop `GRANITE_SPEECH_MAX_GENERATED_TOKENS = 256` is not a longform substitute (`generation_budget_backstop_is_finite_and_not_a_longform_substitute`). Ordinary multi-minute audio routes through the shared SharedWindow longform slicer; stop_reason maps to visible decode truncation. Ignore-form pack e2e skeletons assert no pathological token runs on `fixtures/longform_en_zh.wav`. |

## 9. Resource limits & fail-closed

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Max audio length / context budget derived and over-limit behavior fails closed | Supported | Derived from decoder max positions 4096 - fixed prompt 19 - generation budget 256 at 10 audio tokens/s => **382.0 s** (`capacity::derive_max_input_seconds` / `GRANITE_SPEECH_MAX_INPUT_SECONDS`). Over-limit single buffers fail closed with typed `GraniteSpeechGgmlExecutorError::AudioTooLong` (`audio_past_derived_capacity_fails_closed_with_typed_error`); longer audio is the shared longform slicer's job. Arch capacity basis = `Derived(DecoderContext)`. |
| Streaming first-token latency floor documented (chunk accumulation math; streaming families, otherwise Not applicable) | Not applicable | Offline file-transcribe family. Snapshot streaming is registered only to satisfy the workspace streaming-completeness gate (`build_seq2seq_streaming_session`); not a real-time path and not product-advertised. |
| KV growth rate per audio second known | Supported | Device-resident KV grows with prompt+generated tokens (not raw audio seconds). Encoder frames -> fixed Q-Former query count dominates prompt audio tokens; generation adds O(layers * kv_heads * head_dim * bytes) per token. Peak stays in the ~3-4 GB class on jfk after keep-quantized. |
| Metal wired-memory profile captured | Supported | M1 unified memory; residency-set / shared-buffer path; peak class measured on quiet host (section 3). IOAccelerator wired is not the dominant term vs mmap weights + resident KV. |
| Multi-session scaling behavior known (server concurrency) | Deferred | One resident 2B decoder per (pack, backend) per worker thread; concurrent sessions can multiply that footprint. Unlock: admission / concurrency cap tied to measured peak before multi-tenant server default-on. |
| Energy footprint noted (battery-relevant platforms) | Deferred | Not measured (needs powermetrics window). Unlock: one quiet transcription with energy sampling on battery hardware. |

## 10. Engineering completeness

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| `warm_up` is a real implementation, not a stub | Supported | Snapshot streaming session is built via `build_seq2seq_streaming_session`, whose driver `warm_up` runs `decode_warm_up_silence` (real silent decode). |
| Reference dumper exists for this family | Supported | `tooling/granite-speech-reference-dumper/` (`dump_golden.py` full greedy generate; `dump_intermediate.py` stage taps for mel/encoder/projector/prefill/decode matching `parity.rs` harness names). Pinned to the upstream HF `transformers` Granite Speech path; no third-party code or weights vendored. |
| Registry / catalog / docs wired (MODEL_ONBOARDING checklist done) | Supported | Code registries done (architecture, executor, decode policy, runtime tensor contract, K1/K2). Public catalog row + registry card + model card prose (`tooling/publish-model/cards/granite-speech-4.1-2b.toml`) + HF `OpenASR/granite-speech-4.1-2b` packs (fp16/q8_0/q4_k) + signed catalog epoch land on the public flip. |
| Peer benchmark recorded (table below, all fields) | Supported | Valid warm product E2E vs handy-computer/transcribe.cpp on the same host/audio; see table. Note: peer CLI "realtime" partial timers (mel+encode+prefill only) are **invalid** as E2E and must not be re-cited. |

### Peer benchmark record

Record enough that anyone can re-run this comparison later. "Faster than X" is
not auditable without the exact peer version, model build, audio, and machine.

| Field | Value |
| --- | --- |
| Peer project (name + commit or version) | handy-computer/transcribe.cpp @ `b6a6acad` |
| Peer model + quant build | Granite Speech 4.1 2B; peer Q8 pack sha `8c0b2dce…`; peer Q4_K_M pack sha `3171de08…`. OpenASR packs: q8 `ae36ede2…`, q4_k `7ecfd528…` (q4 recipe differs from peer Q4_K_M -- q4 is informational only, not a gate). |
| Peer program version | transcribe.cpp @ `b6a6acad` (built on the audit host) |
| Test audio (file, duration, language) | `fixtures/jfk.wav`, 11.00 s, English |
| Machine (chip, RAM, OS) | Apple M1, 16 GB, macOS (quiet window; no concurrent inference) |
| Peer numbers (RTF / peak memory / utilization) | Warm E2E wall median **1821.577 ms** (samples 1812.162 / 1824.168 / 1821.577); peak memory footprint ~3.46 GiB (q8). q4_k_m warm median ~1629 ms / ~2.12 GiB (recipe-mismatch caveat). |
| OpenASR numbers (RTF / peak memory / utilization) | Warm E2E wall median **1788.167 ms** (samples 1768.801 / 1788.167 / 1792.475) => **~1.83% faster than peer Q8** (PASS). q4 warm median ~1497.649 ms (not judged vs peer). Keep-quantized peak class ~3.2-4.2 GB on this host. |

## Known dead ends (do not re-litigate)

Verdicts that apply to this family, so future work does not re-run dead
investigations. Repo-wide precedents to inherit where relevant: F16 activation
on Apple M1 (encoder-only gave zero win, cast economics lock the trunk;
verdict 2026-07-14); qwen speculative decode (acceptance alpha ~= 0.05, judged
dead). Add family-specific verdicts with the measurement behind each; write
"None yet" if the family has none.

| Dead end | Verdict / evidence | Date |
| --- | --- | --- |
| Peer CLI "realtime" as full E2E | INVALID. Peer "realtime" is mel+encode_compute+prefill_compute only; rebuilt warm product E2E is the gate. Old 3.0-3.4x claims discarded. | 2026-08-01 |
| Host-f32 whole-model dequant loader | Catastrophe path (RSS ~16 GB, flattened quants). Replaced by keep-quantized mmap decoder + resident encoder/projector. Do not restore `load_tensors_from_oasr_pack` for rank-2 matmul weights. | 2026-08-01 |
| F16 KV as primary win | A/B on q8 showed no product-level win worth the complexity/quality risk on M1; stay f32 resident KV. | 2026-08-01 |
| Full-vocab logits D2H as primary win | ~0.05 ms/tok -- noise vs decode body. | 2026-08-01 |
| Packed gate/up without measured gain | Explored; no keep as default without a quiet A/B win. | 2026-08-01 |
| Fixed+256 attention span alone | Insufficient as the main lever; device-resident KV + reusable graph + shared cancel fix were the real closes. | 2026-08-01 |
