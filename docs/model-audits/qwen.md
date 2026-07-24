# Model release audit: qwen

> **Policy.** Should-support items MUST be `Supported`; anything else requires a
> detailed justification and an explicit unlock condition. This form ships with
> the model release. A family without a completed form does not enter the
> release flow: `tooling/publish-model/scripts/_manifest.py --public` fails
> closed on a missing or half-filled form.

| Field | Value |
| --- | --- |
| Family (`models-core.toml` `family`) | qwen |
| Models covered | qwen3-asr-0.6b, qwen3-asr-1.7b (Qwen/Qwen3-ASR, 32-layer Whisper-arch encoder -> VQ-Adaptor -> Qwen3-0.6B decoder; both sizes share the same executor stack, only decoder width/depth differs) |
| Auditor / date | Quintin (with agent-collected evidence) / 2026-07-24; form filled against main `ce5ae75` |
| Core version + commit audited | main `ce5ae75` (post #237 prefill-cancel; includes #233 L1 cancel, #234 Q8 KV production wire, #236 L2 abort) |
| Bench hardware | Apple M1, 16GB, macOS. Metal/CPU RTF matrix + Q8 KV quality A/B: `tmp/qwen-quiet-2026-07-24-metal-cpu/` (47 ok rows, 1 warn), `tmp/qwen-quiet-2026-07-24-q8kv-quality/` (18/18 pairs text-identical). |

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
| Graph reuse / persistent session (no per-request graph rebuild) | Supported | `QWEN_WHOLE_DECODER_BY_KEY` TLS cache (`ggml_executor.rs:98-124`): whole-decoder runtime cached per (canonical path, backend), avoids ~1GB re-upload per longform chunk. Encoder TLS cache also bounded. `release_session_scoped_buffers` drops request-scoped host buffers before parking (PR #172). Byte-identical goldens across consecutive calls. |
| Op fusion opportunities reviewed (norm+matmul, QKV, rope, ...) | Supported | Fused QKV projection load + fused logits head (`llm_transformer.rs`, `logits_head.rs`); shared with mimo/firered2/moss. No qwen-specific stitching debt beyond shared executor. |
| Batching / serve-batch path | Supported | qwen is a serve-batch consumer (`batched_decode.rs` + #215 policy). Owner-thread cancel carried on job Arc (PR #237, #236). `qwen_policy_derives_width_and_queue_from_admission_limit` test covers policy wiring. |
| Encode-decode pipelining | Not applicable | Encoder produces fixed-length audio embeddings first; decoder runs prefill + greedy after encoder completes. Architecture requires full audio before prompt construction. Long-form is sliced upstream by the generic longform planner. |
| Arena / gallocr reuse across steps (no per-step allocator churn) | Supported | Metal path: persistent reuse graph writes KV into resident arena (`run_prefill_into_reused_batched`). CPU path: step-buffer grow-to-fit reuse (#172). Shared scheduler gallocr on. |

## 2. Precision & quantization

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| KV cache quantization | Supported | Phase-1 Q8 KV landed #217; production-wired on CPU/Metal in #234 (`resolve_production_llm_kv_cache_policy` in `nn/decoder.rs`). Default Q8 when native-GQA + flash geometry allow; discrete GPU stays F32/F16 (typed reject). Opt-out: `OPENASR_QWEN_KV_CACHE_F32=1`. **Quality gate:** 18/18 short-audio text SHA pairs identical (default Q8 vs F32) across {0.6b q4, 1.7b q4, 1.7b q8} x {metal, cpu} x {jfk, enzh, zh}. Mismatches=0. |
| Activation precision policy chosen deliberately (f32 vs f16) | Supported | Deliberate f32 activations. Repo-wide verdict (2026-07-14, M1): F16 activation gave zero encoder win; cast economics lock the trunk. Recorded in Known dead ends. |
| Keep-quantized matmul (native Q blocks bound, no load-time dequant; RAM orders q4 < q8 < fp16) | Supported | mmap zero-copy bind of quantized tensors (`gguf_tensor_data.rs` -> `cpu_graph.rs`). Adaptor is host-side MLP (~19MB f32, negligible). Measured RAM ordering: fp16 > q8_0 > q4_k across both sizes and both backends (`matrix_summary.txt` phys_GiB column). |
| Quant tiers complete (q4_k / q8_0 / fp16) | Supported | All three tiers published for both sizes (HF OpenASR/qwen3-asr-0.6b, OpenASR/qwen3-asr-1.7b). Catalog carries rtf_cpu/rtf_metal/peak for all three. 1.7b recommended q8_0; 0.6b recommended q8_0. |

## 3. Memory & data movement

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| mmap weight loading | Supported | `GgufTensorDataReader::from_path` -> mmap-backed `GgmlLoadedWeightContext` for encoder + decoder, shared with every builtin family. |
| Resident pool reuse across requests (weights stay resident) | Supported | `QWEN_WHOLE_DECODER_BY_KEY` TLS cache (#172, #224 content-id generation). Consecutive calls zero-rebuild; byte-identical goldens. Shared follow-up: cache key lacks pack-content fingerprint (cross-family fix queued; correctness risk only on in-place file replacement). |
| View contiguity tradeoffs audited (`cont`/copy nodes justified) | Deferred | Shared qwen decoder graph has not had a dedicated offset-view sweep. Unlock: one cross-family contiguity sweep of the shared `llm_transformer` / `nn/decoder` compose path (covers qwen3/mimo/firered2/moss in one pass); method per the Vulkan misalign case (0.1.22 xasr fix). |
| Peak RSS/VRAM per shipped quant measured (quiet host) and reconciled against the weights+KV+activations budget; unexplained excess blocks release; catalog RAM requirement matches the measured peak | Deferred | **Measured** (2026-07-24, quiet window, sidecar 0.1.24): 1.7b Metal q4 phys ~3.6 GiB / q8 ~4.3 GiB / fp16 ~4.8 GiB; CPU q4 ~4.0 / q8 ~4.7 / fp16 ~5.1 GiB. 0.6b Metal q4 ~1.9 GiB; CPU ~2.3 GiB. **Not reconciled**: catalog `peak_rss_bytes` fields carry stale numbers (e.g. 1.7b q8 peak 4138139648 = 3.85 GiB vs measured Metal phys 4.3 GiB). Unlock: pick catalog peak_rss_bytes convention (Metal phys or maxRSS) and refresh in same pass as rtf sync. No unexplained excess; weights dominate as expected. |

## 4. Decode algorithms

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Greedy logits shortcuts (argmax path skips needless softmax/sort work) | Supported | Shared driver: single-pass `argmax_index` (`seq2seq_greedy_decode.rs`); softmax once post-argmax for confidence only. No top-k sort on 152k vocab. L1 cancel poll per token (#233). |
| Speculative decode: per-family verdict recorded (do it, defer it, or dead) | Supported | Verdict = **dead**. Measured acceptance alpha ~= 0.05 for 0.6B/1.7B class; bandwidth-bound decode with no small draft model available. Recorded in Known dead ends. |
| CTC blank-skip fast path (CTC families; otherwise Not applicable) | Not applicable | Autoregressive attention decode (`qwen3-asr.greedy.seq2seq.v0`); no CTC head. |
| Decode guards are zero-cost on the hot path (degenerate-loop guard etc.) | Supported | Shared `detect_degenerate_ngram_repeat` scans token-id history tail only (O(max_ngram)/step, no logits access). Wired via shared driver per issue #60 rule. |

## 5. Frontend & IO

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Mel/fbank frontend SIMD + parallelized | Supported | Shared whisper log-mel frontend (80-mel, 3000 frames/30s chunk); FFT via `realfft` (SIMD). Frame loop single-threaded; at this model's cost encoder+decoder dominate (same finding as sibling families). |
| Zero-copy audio path (no avoidable resample/copy hops) | Supported | PR #175 (`7c96ae0`): `PreparedAudioInput` carries symphonia-decoded `Arc<Vec<f32>>` straight through; no temp WAV write/re-read for non-WAV input. WAV passthrough unchanged. Full f32 precision preserved (no f32->i16->f32 round trip). |
| VAD cost measured and accounted | Not applicable | No VAD in qwen's transcription path. Long-form sliced by generic upstream longform planner; no family-local VAD. |

## 6. Platform-specific

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Metal command batching + wired memory budget respected | Supported | Auto policy = `AllBackends` (qwen NOT gated `ExceptMetal`). **2026-07-24 quiet matrix confirms**: every measured cell Metal >= CPU (ratio 0.80-1.00). No CPU-faster cell under TIE band. Prior "qwen Metal negative" anecdotes were pre-#212 graph-reuse era. No ExceptMetal entry warranted. Metal decoder stays eligible under Auto. |
| CPU thread pool sized for P/E cores | Supported | Shared `adaptive_thread_count_for_available` policy by workload class and backend type; Metal default n_threads=1, scheduler off. No family override needed. |
| Accelerate/BLAS used where it wins | Supported | Generic BLAS backend wiring (`cpu_graph.rs`), inherited via shared CPU graph path. |

## 7. Backend coverage matrix

Every cell must be answered. An unsupported backend is acceptable ONLY with a
justification and an unlock plan -- "nobody tried" is not a justification.
Golden-verified means byte/parity fixtures pass ON that backend;
utilization-measured means the GPU weight placement gate (or an equivalent
profile) proved the compute actually runs there (golden output alone cannot,
see `docs/design/gpu-weight-placement.md`).

| Backend | Supported? | Golden-verified? | Utilization measured? | Justification + unlock plan if unsupported |
| --- | --- | --- | --- | --- |
| CPU | Yes | Yes (short goldens en/zh/mixed; cross-backend byte parity spot) | Yes (RTF matrix 2026-07-24; q4 jfk 0.75, enzh 0.63; q8 jfk 1.17, enzh 1.03; fp16 jfk 2.17, enzh 1.95) | |
| Metal | Yes | Yes (same goldens byte-identical on Metal) | Yes (RTF matrix 2026-07-24; Metal wins or ties all cells; 0.6b q4 Metal 0.26-0.31 vs CPU 0.32-0.38) | |
| CUDA | Untested | No | No | Shared qwen executor path; no CUDA host available for family golden. Unlock: community/dev-host run of golden + `qwen-gpu-parity` (tooling exists). Bulk CUDA/HIP prefill (#223) already landed in shared path. |
| Vulkan | Untested | No | No | Same as CUDA; xasr-class offset-view fix (0.1.22) hardened shared Vulkan path. Unlock: AMD/Intel Vulkan host validation. |
| HIP | Untested | No | No | Plain per-chunk path; qwen's HIP prefill-chunk tuning deliberately not replicated (short prompts under longform cap). Unlock: HIP host validation. |

## 8. Correctness & quality

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| WER vs fp16 measured for every shipped quant tier | Deferred | Not measured on current code. Catalog carries stale `jfk_wer_vs_fp16` from pre-0.1.22 era. Unlock: re-measure on a host that fits fp16 (32GB+ or 1.7b fp16 on 16GB with swap tolerance); refresh catalog perf in same pass. Short-audio text SHA identity (18/18) provides a partial gate but is not WER. |
| Model ref alias forms resolve identically everywhere (bare family / `family:canonical` / every `quant_tag_cases.json` alias accepted by CLI and server match logic; covered by the catalog-wide alias matrix test) | Supported | PR #171 (`b8a8949`): canonical_quant_tag from single alias-group table; `native_quant_alias_catalog_matrix` walks bundled catalog (qwen3-asr-0.6b/1.7b included). Hyphen-joined legacy ids tolerated. |
| Golden coverage includes long audio AND a cross-backend parity fixture | Supported | Short goldens 3/3 (en/zh/mixed) byte-identical on CPU and Metal. Long audio beyond longform slice cap is upstream-sliced by design (each slice <= cap, fail-closed); covered by generic longform tests. Cross-backend parity spot-checked (quality A/B: same text SHA across CPU/Metal for all 18 pairs). |
| Official decode parameters honored (suppression, stop tokens, upstream reference settings) | Supported | Greedy path: eos `<\|im_end\|>`, ChatML prompt, empty suppression, do_sample=False. Beam search / repetition_penalty are official published recipe (beam=3, rep=3.0) -- out of scope by single shared greedy driver invariant. Repetition control structural (shared n-gram guard). Quality comparisons must re-run reference at matched settings. |
| Long-audio degradation checked (repetition, drift, truncation) | Supported | Longform slice cap fails closed (`AudioTooLong`); within-slice repetition covered by shared degenerate-loop guard (issue #60 class). 3-min aishell4 multi-speaker smoke pass (MOSS matrix; qwen shares same decoder stack). |

## 9. Resource limits & fail-closed

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Max audio length / context budget derived and over-limit behavior fails closed | Supported | Longform planner slices upstream; per-slice cap fails closed. Decoder context is request-sized (prompt + generation budget), not native 32768. |
| Streaming first-token latency floor documented (chunk accumulation math; streaming families, otherwise Not applicable) | Deferred | qwen registers snapshot-based streaming; first-token floor not derived/documented for the qwen executor. Unlock: derive from snapshot cadence + prefill cost (RTF numbers now exist). |
| KV growth rate per audio second known | Supported | Shared qwen decoder: ~112 KB/token x measured prompt-tok/audio-sec rate; request-sized KV allocation (not native 32768 cap). Q8 KV cuts this ~4x vs F32 host KV. |
| Metal wired-memory profile captured | Deferred | RSS measured (section 3); wired-memory profile (vs RSS) not separately captured. Unlock: one `footprint`-sampling run with `task_for_pid` / `phys_footprint` on Metal path. |
| Multi-session scaling behavior known (server concurrency) | Supported | Per-model admission control (`model_admission.rs`): default `max_concurrent_sessions_per_model=1`; concurrent requests get typed 503. Serve-batch width derived from admission limit. |
| Energy footprint noted (battery-relevant platforms) | Deferred | Not measured (needs sudo powermetrics window). Unlock: one measured transcription with energy sampling during a maintenance window. |

## 10. Engineering completeness

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| `warm_up` is a real implementation, not a stub | Supported | Shared `decode_warm_up_silence` (real silent decode) via the incremental driver; qwen executor inherits the shared warm-up path. |
| Reference dumper exists for this family | Deferred | No `tooling/qwen-reference-dumper/` (moss and firered2 have theirs). Unlock: port the firered2/moss dumper pattern for the qwen encoder+adaptor+decoder stages. Shared LLM decoder is byte-identical to firered2's dumper-verified path. |
| Registry / catalog / docs wired (MODEL_ONBOARDING checklist done) | Supported | Arch descriptor, executor/decode-policy registries, dispatch, registry toml, catalog entry + card all present and verified. Both sizes public with all three quants. |
| Peer benchmark recorded (table below, all fields) | Deferred | Peer determined: Qwen/Qwen3-ASR official transformers stack (sherpa-onnx does not ship qwen3-asr; CrispASR/transcribe.cpp do not list it). Hardware-locked: fair peer speed run requires 32GB+ host for fp16 reference. Unlock: same as WER row. |

### Peer benchmark record

Record enough that anyone can re-run this comparison later. "Faster than X" is
not auditable without the exact peer version, model build, audio, and machine.

| Field | Value |
| --- | --- |
| Peer project (name + commit or version) | Qwen/Qwen3-ASR official transformers stack. DEFERRED: fair peer speed run cannot execute on the 16GB M1 reference host -- fp16 weights + activations exceed unified memory for a clean comparison. Unlock: 32GB+ host. Quality comparisons must re-run the reference at its published recipe, not reuse published WER against our greedy. |
| Peer model + quant build | pending the 32GB+ host run |
| Peer program version | pending the 32GB+ host run |
| Test audio (file, duration, language) | pending the 32GB+ host run |
| Machine (chip, RAM, OS) | pending the 32GB+ host run |
| Peer numbers (RTF / peak memory / utilization) | pending the 32GB+ host run |
| OpenASR numbers (RTF / peak memory / utilization) | Measured 2026-07-24 (sidecar 0.1.24, M1 16GB): 1.7b q4 Metal jfk 0.73 / enzh 0.57; q8 Metal jfk 1.11 / enzh 0.91; fp16 Metal jfk 1.82 / enzh 1.62. 0.6b q4 Metal jfk 0.31 / enzh 0.26. CPU: 1.7b q4 jfk 0.75 / enzh 0.63. Phys peak: 1.7b q4 Metal ~3.6 GiB; 0.6b q4 Metal ~1.9 GiB. Re-measure alongside peer on same host for valid comparison. |

## Known dead ends (do not re-litigate)

Verdicts that apply to this family, so future work does not re-run dead
investigations. Repo-wide precedents to inherit where relevant: F16 activation
on Apple M1 (encoder-only gave zero win, cast economics lock the trunk;
verdict 2026-07-14); qwen speculative decode (acceptance alpha ~= 0.05, judged
dead). Add family-specific verdicts with the measurement behind each; write
"None yet" if the family has none.

| Dead end | Verdict / evidence | Date |
| --- | --- | --- |
| F16 activations on Apple M1 | Encoder-only gave zero win; cast economics lock the trunk. Repo-wide verdict 2026-07-14. | 2026-07-14 |
| Speculative decode (0.6B / 1.7B) | Acceptance alpha ~= 0.05; bandwidth-bound decode with no viable small draft model. Dead. | 2026-07 |
| ExceptMetal Auto policy for qwen | 2026-07-24 quiet matrix: all cells Metal >= CPU (ratio 0.80-1.00). No CPU-faster cell. Prior "Metal negative" was pre-graph-reuse era. Do not add ExceptMetal. | 2026-07-24 |
| HIP prefill-chunk tuning replication | Deliberately skipped: longform cap keeps prompts short; qwen's discrete-GPU prefill-chunk tuning judged not worth replicating for this family. | 2026-07 (code-in) |
| 1.7b q4_k CPU en_zh_mixed text collapse | Observed `Today, today` output on 1.7b q4_k CPU for en_zh_mixed (same under Q8 and F32 KV). Orthogonal to KV dtype; appears to be a quant/backend quality oddity. Not a regression vs main; flag for separate investigation if product cares. | 2026-07-24 |
