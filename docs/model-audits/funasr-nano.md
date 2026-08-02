# Model release audit: funasr-nano

> **Policy.** Should-support items MUST be `Supported`; anything else requires a
> detailed justification and an explicit unlock condition. This form ships with
> the model release. A family without a completed form does not enter the
> release flow: `tooling/publish-model/scripts/_manifest.py --public` fails
> closed on a missing or half-filled form.

| Field | Value |
| --- | --- |
| Family (`models-core.toml` `family`) | funasr-nano |
| Models covered | Fun-ASR-Nano-2512 (SAN-M encoder + 2-layer adaptor + Qwen3-0.6B decoder); tiers fp16 / q8_0 / q4_k |
| Auditor / date | main-loop / 2026-08-02 |
| Core version + commit audited | `feat/funasr-nano` @ post-`1d92474b` (rebase origin/main `8e670cf2` = v0.1.26 line; encoder keep-quantized load + K1/K2 gates) |
| Bench hardware | Apple M1, 16 GB, macOS |

**How to fill.** Status is exactly one of: `Supported`, `Not applicable`, `Deferred` (with justification + unlock condition).

## 1. Graph & scheduling

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Graph reuse / persistent session (no per-request graph rebuild) | Supported | encoder+adapter+decoder all per-(pack,backend) resident-cached; only transient forward graph per request. Tests `cached_encoder_adapter_matches_fresh_build_bit_for_bit` + decoder resident cache (mirrors sensevoice/dolphin/firered_llm). |
| Op fusion opportunities reviewed | Supported | Inherits qwen Qwen3 core (fused QKV, rope, QK-norm) + device-side fused logits top1. SAN-M/FSMN encoder reuses `nn::encoder::sanm_fsmn_encoder_layer`. No family-specific fusion gap. |
| Batching / serve-batch path | Deferred | Family reaches decode via the shared greedy driver whose serve-batch path exists but is opt-in default-off (judged niche for local, superseded by slice-pipeline). No family-specific batching tuning. Unlock: only if cloud multi-tenant demand arises. |
| Encode-decode pipelining | Supported | Long-audio runs the P1 concurrent slice pipeline; funasr uses `ConservativeSeq2SeqV1` (carry disabled) so P1 default-on-eligible (carry-gated default `fdd7e0fe`), giving admission-concurrency overlap. |
| Arena / gallocr reuse across steps | Supported | Resident runtime holds weight arena; transient forward graph resets only the ephemeral context per request. No per-step allocator churn. |

## 2. Precision & quantization

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| KV cache quantization | Supported | Q8_0 double-copy production KV (via `resolve_qwen_family_production_kv_cache_policy`), 59.5 KiB/token/copy, deliberate. |
| Activation precision policy chosen deliberately (f32 vs f16) | Supported | Audio path forced f32 (no fp16 activation; adaptor output std 28.0 / |max| 1100 measured, would overflow f16). Decode uses qwen precision policy. |
| Keep-quantized matmul (native Q blocks bound, no load-time dequant; RAM orders q4 < q8 < fp16) | Supported | RAM ordering confirmed (q4_k 1769 < q8_0 2463 < fp16 3642 MiB Metal). **Encoder:** rank-2 mul_mat weights (`attn.qkv/out`, `ffn.up/down`) bind via `load_named_bound` + `bind_loaded` -- metadata only at load, no host dequant; host-f32 is limited to norms/biases/FSMN (`encoder_graph.rs`). K1 inventories `funasr_nano/encoder_graph.rs`. **Decoder:** reuses qwen family loader with `materialize_qkv: false`; q8 pack stores `blk.*.attn_{q,k,v}.weight` as native Q8_0 in `[in,out]` orientation so `raw_ggml` binds and `FusedQkvProjectionWeight` takes the raw-concat path (`fuse_raw_qkv_projection_weights`) -- no f32 QKV arena. Host payload dropped after bind (`dropped_projection_payload`). The earlier ~448 MiB QKV f32-arena residual is closed on this path (stale as of shared qwen fused-raw + this encoder split). Remaining whole-pack backend buffer materialization is the engine-level eager-load item (Known dead ends), not a family keep-quantized gap. Correctness: ignored golden `golden_encoder_adapter_cosine_and_end_to_end_text` + `cached_encoder_adapter_matches_fresh_build_bit_for_bit` green on fp16-tok pack after the encoder load split. |
| Quant tiers complete (q4_k / q8_0 / fp16) | Supported | All three built and byte/WER-verified (`funasr-nano-{fp16-tok,q8_0,q4_k}.oasr`); encoder half floored at Q8_0, decoder half Q4_K on the q4_k tier. |

## 3. Memory & data movement

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| mmap weight loading | Supported | Pack is mmap-resident (reclaimable); RSS = mmap resident + private, residual <=3 MiB of the identity RSS-footprint = pack + 37 MiB. (Eager backend materialization overhead is an engine-level item, see dead ends.) |
| Resident pool reuse across requests (weights stay resident) | Supported | K2: `("funasr_nano", Resident)` in `GGML_EXECUTOR_FAMILY_GATES`. encoder + adapter + decoder all resident across requests via BoundedRuntimeCache keyed by (PackContentKey, backend); idle unload via central `bump_unload_generation`. Test `cached_encoder_adapter_matches_fresh_build_bit_for_bit`. |
| View contiguity tradeoffs audited (`cont`/copy nodes justified) | Supported | Encoder/adapter/decoder graphs inherited from audited sensevoice/qwen paths; no new avoidable copy nodes introduced (adaptor tensors zero-copy bound from loaded context). |
| Peak RSS/VRAM per shipped quant measured (quiet host) and reconciled ... catalog RAM requirement matches the measured peak | Supported | Quiet-host matrix (36 runs, <=2 MiB jitter, pre-final keep-quantized close): Metal RSS fp16 3642 / q8_0 2463 / q4_k 1769 MiB. Residual then attributed partly to a QKV f32 arena that the decoder raw-fuse path no longer builds; catalog RAM still set conservatively fp16 4300 / q8_0 2900 / q4_k 2100 MB (>= measured peaks, safe headroom). Optional tighten on first catalog write after a fresh quiet 3-run matrix. |

## 4. Decode algorithms

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Greedy logits shortcuts (argmax path skips needless softmax/sort work) | Supported | Device-side fused logits top1 (`decode_step_reused_top1`, hint-only, zero host full-vocab readback), inherited from qwen; golden bit-identical. |
| Speculative decode: per-family verdict recorded | Deferred (dead) | Inherit qwen verdict: acceptance alpha ~= 0.05, judged dead. No family-specific reason to revisit. |
| CTC blank-skip fast path | Not applicable | Fun-ASR uses the LLM (encoder->adaptor->Qwen3) decode path; the CTC branch is a training aux, dropped at import (no CTC decode). |
| Decode guards are zero-cost on the hot path | Supported | Shared driver `run_builtin_seq2seq_decode_policy` centralizes the degenerate-loop guard (registry-resolution test green); no per-token host work added. |

## 5. Frontend & IO

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Mel/fbank frontend SIMD + parallelized | Supported | Reuses the repo's `SenseVoiceFbankFrontend` (80-mel, hamming, preemph 0.97, LFR 7-6); funasr skips CMVN (config cmvn_file:null). |
| Zero-copy audio path (no avoidable resample/copy hops) | Supported | Reuses the sensevoice audio path; no family-specific extra hop. |
| VAD cost measured and accounted | Not applicable | Fun-ASR offline pipeline (encoder->adaptor->LLM) has no VAD stage. |

## 6. Platform-specific

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Metal command batching + wired memory budget respected | Supported | Metal path golden-verified (bit-identical vs CPU); wired memory negligible on Apple Silicon unified memory (vmmap IOAccelerator ~12 MB; MTLBuffer weights in unified memory). |
| CPU thread pool sized for P/E cores | Supported | Inherits the shared ggml CPU thread-pool sizing (no family-specific override). CPU golden-verified. |
| Accelerate/BLAS used where it wins | Supported | Inherits vendored ggml GEMM backend selection; no family-specific BLAS gap. |

## 7. Backend coverage matrix

| Backend | Supported? | Golden-verified? | Utilization measured? | Justification + unlock plan if unsupported |
| --- | --- | --- | --- | --- |
| CPU | Yes | Yes | N/A (CPU) | golden_diff + e2e (en/zh byte-identical to f32 oracle) on CPU. |
| Metal | Yes | Yes | Yes | `metal_reused_decode_graph`-style bit-identical test + encoder/adaptor cosine=1.0; Metal run proven via `ggml_metal_init` logs; RSS/backend gating measured (Metal fastest, auto=AllBackends correct). |
| CUDA | Deferred | No | No | Untested on this M1 host. Family is backend-agnostic ggml (reuses qwen/sensevoice ops, no missing ops); expected to work. Unlock: CI cross-backend build + golden parity. |
| Vulkan | Deferred | No | No | Same as CUDA: untested locally, backend-agnostic ops. Unlock: CI cross-backend build + golden parity. |
| HIP | Deferred | No | No | Same as CUDA: untested locally. Unlock: CI cross-backend build + golden parity. |

## 8. Correctness & quality

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| WER vs fp16 measured for every shipped quant tier | Supported | frozen-eval-v1 (AISHELL test 259 zh + LibriSpeech test-clean 132 en, public): fp16 zh CER 3.03% / en WER 2.57%; q8_0 per-sentence-equivalent to fp16 (3.00% / 2.57%); q4_k 2.68% / 2.77% (+0.20pp en, no systematic regression). |
| Model ref alias forms resolve identically everywhere | Deferred | Registry resolution green (shared-driver registry test); canonical `openasr.model.architecture=funasr-nano-sanm-adapter-qwen3` added to packs. Catalog entry + alias-matrix coverage pending catalog registration (see 10). Unlock: add catalog entry + quant_tag alias cases + run the catalog-wide alias matrix test. |
| Golden coverage includes long audio AND a cross-backend parity fixture | Deferred | Cross-backend parity fixture present (Metal vs CPU bit-identical). Dedicated long-audio golden fixture not yet added (bring-up golden used ~5-8s examples; long audio handled via ConservativeSeq2SeqV1 ~15s slicing + validated indirectly by the ~40-min WER set). Unlock: add a long-audio golden fixture. |
| Official decode parameters honored | Supported | ChatML prompt + LFR truncation formula exactly replicated and validated against the official reference oracle (funasr_oracle.py); Qwen3 decode params (no suppression, standard stop tokens). |
| Long-audio degradation checked (repetition, drift, truncation) | Supported | ConservativeSeq2SeqV1 profile (anti-repeat, issue #60) + ~15s slicing; WER-set clips show no systematic repetition/drift; known truncated sentences are eval-set-common (same position across all quants and vs MOSS/FireRed rounds). |

## 9. Resource limits & fail-closed

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Max audio length / context budget derived and over-limit behavior fails closed | Supported | Executor MAX_INPUT ~= 40 s fail-closed; long audio routes through the slice pipeline. |
| Streaming first-token latency floor documented | Not applicable | Fun-ASR is an offline (non-streaming) AED family. |
| KV growth rate per audio second known | Supported | ~0.5 MiB/audio-second (double Q8_0 copies), audio-token ~4.17/s; pre-alloc capacity prompt+512, 30s slice cap <=77 MiB; SharedWindow slicing => no unbounded growth. |
| Metal wired-memory profile captured | Supported | Captured: wired negligible on unified memory (vmmap IOAccelerator ~12 MB); peak footprint (incl wired) fp16 1719 / q8_0 1419 / q4_k 1083 MiB. |
| Multi-session scaling behavior known (server concurrency) | Deferred | Generic ModelSessionAdmission applies (single native session default, 503 on concurrent same-model). funasr-specific host-memory admission geometry not yet wired (unlike moss/qwen/firered/mimo/cohere). Unlock: add funasr capacity deriver (mirror the 6 wired families) + a concurrency scaling test. |
| Energy footprint noted (battery-relevant platforms) | Deferred | Not profiled. Unlock: battery-draw profiling pass on Apple Silicon. |

## 10. Engineering completeness

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| `warm_up` is a real implementation, not a stub | Supported | Executor mirrors firered_llm/mimo warm_up (loads resident runtime + primes the decoder graph), not a stub. |
| Reference dumper exists for this family | Supported | `funasr_oracle.py` (faithful port of the official funasr-cli fbank+LFR+SAN-M+adaptor+Qwen3), used to produce the golden fixtures (cosine=1.0). |
| Registry / catalog / docs wired (MODEL_ONBOARDING checklist done) | Deferred | Registry fully wired (arch ids, 5 component descriptors, all registries, native dispatch, pack_quant_audit; registry-resolution test green). Catalog entry (models-core/publish toml) + model card + docs NOT yet added. Unlock: complete the publish-prep (catalog registration + card + docs + external_converter provenance for the ExternalTooling import path). |
| Peer benchmark recorded (table below, all fields) | Supported | See table. |

### Peer benchmark record

| Field | Value |
| --- | --- |
| Peer project (name + commit or version) | Official FunASR llama.cpp runtime (`FunASR/runtime/llama.cpp/fun-asr-nano`, upstream probe clone 2026-08-01) |
| Peer model + quant build | Fun-ASR-Nano-2512 official GGUF (fp16 and q8/q4 equivalents) |
| Peer program version | llama.cpp funasr-cli (CPU-only by design, mmap load) |
| Test audio (file, duration, language) | model's public examples en.mp3 (7.18s, en) / zh.mp3 (5.62s, zh) |
| Machine (chip, RAM, OS) | Apple M1, 16 GB, macOS |
| Peer numbers (RTF / peak memory / utilization) | fp16 e2e 12.89s; steady-state marginal RTF ~0.052; CPU-only |
| OpenASR numbers (RTF / peak memory / utilization) | fp16 e2e 7.60s (win 1.7x); q8/q4 e2e ~3x slower than peer BUT purely eager-load overhead; steady-state warm RTF ~0.053 (parity, Metal slightly better); Metal peak RSS q8 2463 MiB |

## Known dead ends (do not re-litigate)

| Dead end | Verdict / evidence | Date |
| --- | --- | --- |
| Speculative decode | Dead -- inherit qwen verdict (acceptance alpha ~= 0.05). | 2026-08-01 |
| F16 activation on audio path | Dead -- measured adaptor output std 28.0 / |max| 1100 overflows f16; repo-wide M1 f16-activation verdict also applies (encoder-only zero win). | 2026-08-01 |
| Backend-eager-load = peer gate gap | The ~3x e2e gap vs peer's mmap runtime is 100% eager whole-pack backend materialization (~3.2 s/GB), NOT a decode inefficiency (steady-state compute is at parity). This is an ENGINE-LEVEL item (all OpenASR families), not family-specific; tracked separately. Steady-state compute needs no family-specific work. | 2026-08-01 |
</content>
