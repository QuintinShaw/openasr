# Model release audit: firered2-llm

> **Policy.** Should-support items MUST be `Supported`; anything else requires a
> detailed justification and an explicit unlock condition. This form ships with
> the model release. A family without a completed form does not enter the
> release flow: `tooling/publish-model/scripts/_manifest.py --public` fails
> closed on a missing or half-filled form.

| Field | Value |
| --- | --- |
| Family (`models-core.toml` `family`) | firered2-llm |
| Models covered | firered2-llm (FireRedASR2-LLM, 8B Encoder-Adapter-LLM, q4_k) |
| Auditor / date | Quintin (with agent-collected evidence) / 2026-07-22, IN PROGRESS |
| Core version + commit audited | 0.1.22 line; baseline measurements at `ace19ee`, adapter rework merged as PR #156 (`00a3bca`) |
| Bench hardware | Apple M1, 16GB, macOS 15 (single reference host; noise disclosed per number) |

**How to fill.** Status is exactly one of:

- `Supported` -- implemented and verified for this family in this repo. Cite
  the evidence (test name, bench run, code path).
- `Not applicable` -- architecturally impossible or meaningless for this
  family. Say why, so nobody re-derives it.
- `Deferred` -- applicable but intentionally not done yet. Give the detailed
  justification AND the unlock condition (what measurement, upstream change,
  or milestone flips it to Supported).

Replace every `<!-- TODO:fill -->` marker; the release gate rejects any
leftover marker. Do not delete or rename the ten numbered section headings; the
gate checks all ten. Keep entries terse -- one form should take an afternoon,
not a week. The goal is that every release ships in its best known state, with
every consciously skipped optimization on the record.

## 1. Graph & scheduling

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Graph reuse / persistent session (no per-request graph rebuild) | Supported | `run_step_auto`/`run_prefill_auto_last_hidden` on the shared qwen executor (`llm_transformer.rs:272-306`, merge `4d7d8f8`). Clean medians (quiet host): Metal RTF 0.760->0.737 (jfk), 0.772->0.732 (zh). Transcripts byte-identical to golden. |
| Op fusion opportunities reviewed (norm+matmul, QKV, rope, ...) | Supported | Shared qwen graph: fused QKV projection load (`llm_transformer.rs:67-108`), fused logits head (`:129-134`). No firered-specific op stitching added. |
| Batching / serve-batch path | Deferred | Not wired: no `firered_llm` consumer of `seq2seq_serve_batch.rs` (only cohere/qwen have one). Unlock: implement `FireRedLlmServeBatchDecoderRuntime` after the shared serve-batch engine is promoted out of its env gate (pillar-2 program). |
| Encode-decode pipelining | Not applicable | Single-shot architecture with a 40s hard input cap (`executor.rs:71-76`); frontend->encoder->adapter->prefill->decode is one pass per request, no chunk stream to overlap. Long audio is sliced upstream by the generic longform planner. |
| Arena / gallocr reuse across steps (no per-step allocator churn) | Supported | GPU path: persistent reuse graph writes KV into the resident arena (`run_prefill_into_reused_batched`). CPU path rebuilds per token by design of the shared executor (same as qwen; measured decode cost is bandwidth-bound, see section 9 numbers). |

## 2. Precision & quantization

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| KV cache quantization | Deferred | KV is host f32 via shared `Qwen3AsrLayerKvCacheState` (~112 KB/token, ~117 MB at the 40s cap -- small relative to weights). Unlock: shared qwen-family KV-quant project (mobile-driven); per-family change is not possible without touching shared infra. |
| Activation precision policy chosen deliberately (f32 vs f16) | Supported | Deliberate f32 activations; repo-wide verdict (2026-07-14, M1): F16 activation gave zero encoder win, cast economics lock the trunk. Recorded in Known dead ends. |
| Keep-quantized matmul (native Q blocks bound, no load-time dequant; RAM orders q4 < q8 < fp16) | Supported | Decoder: zero-copy mmap bind of quantized tensors (`arena_weight_pipeline.rs`). Adapter: previously an 88MB per-call f32 dequant exception; removed by PR #156 -- adapter now runs as a ggml graph feeding quantized weights to `mul_mat` directly (stage 2868ms -> 136-163ms, parity max_abs_diff 1.9e-6). |
| Quant tiers complete (q4_k / q8_0 / fp16) | Deferred | Only q4_k ships (HF `OpenASR/firered2-llm` has a single .oasr; `models-core.toml:638`). Justification: on the 16GB M1 reference host q8_0/fp16 exceed the unified-memory working set (measured: q8_0 CPU decode thrashes swap, unusable). Unlock: re-evaluate on a 32GB+ or discrete-GPU host; if viable there, ship the tiers rather than exporting the M1 constraint to all platforms. |

## 3. Memory & data movement

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| mmap weight loading | Supported | `GgufTensorDataReader::from_path` -> `Mmap::map` -> host-ptr-backed ggml buffers (`gguf_tensor_data.rs:197-226`, `cpu_graph.rs:1084-1132`). |
| Resident pool reuse across requests (weights stay resident) | Deferred | `FireRedLlmGgmlExecutor` builds the decoder runtime per `execute()` (no `unload_idle_state` override); decoder graph/runtime construction measured at ~1.8-2.0s per request (amortized only by OS page cache). Unlock: resident executor cache keyed like the streaming worker pool; needs an idle/memory eviction policy for an 8B resident. |
| View contiguity tradeoffs audited (`cont`/copy nodes justified) | Deferred | Adapter is now a ggml graph with plain contiguous matmuls (PR #156); the shared qwen decoder graph has not had a dedicated offset-view sweep. Unlock: one cross-family contiguity sweep of `qwen` shared executor (covers qwen3/mimo/firered2 in one pass), method per the Vulkan misalign case. |

## 4. Decode algorithms

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Greedy logits shortcuts (argmax path skips needless softmax/sort work) | Supported | Shared driver: single-pass `argmax_index` (`seq2seq_greedy_decode.rs:501-509`); softmax computed once post-argmax for confidence only (`:403-412`). No top-k sort on the 152k vocab. |
| Speculative decode: per-family verdict recorded (do it, defer it, or dead) | Deferred | Not evaluated for the 8B tier. The qwen 0.6B verdict (alpha ~= 0.05, dead) does NOT transfer: bandwidth-bound 8B decode is the profile where a small draft pays most (upper bound = decode share, 21-32% of wall on jfk-class input; higher on output-dense audio). Unlock: draft-model selection + acceptance-rate measurement on current code. |
| CTC blank-skip fast path (CTC families; otherwise Not applicable) | Not applicable | AED/autoregressive LLM decode (`firered-llm.greedy.seq2seq.v0`); no CTC head. |
| Decode guards are zero-cost on the hot path (degenerate-loop guard etc.) | Supported | `detect_degenerate_ngram_repeat` scans token-id history tail only (O(max_ngram)/step, no logits access) (`seq2seq_greedy_decode.rs:467-499`); wired via the shared driver per the issue #60 rule. |

## 5. Frontend & IO

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Mel/fbank frontend SIMD + parallelized | Supported | Shared kaldi fbank engine; FFT via `realfft` (SIMD). Frame loop is single-threaded, and that is the right call here: measured share is 0.3% of execute (2.1ms Metal / 4.8ms CPU on jfk). Parallelizing it buys nothing for this family. |
| Zero-copy audio path (no avoidable resample/copy hops) | Deferred | Copy-hop count from wav decode to fbank not yet audited for this family. Unlock: one pass over `load_wav_16khz_mono_f32_v0` -> frontend chain; expected small (frontend total is 0.3% of wall). |
| VAD cost measured and accounted | Not applicable | No VAD in this family's path (grep: zero references); 40s single-shot with upstream longform slicing instead. |

## 6. Platform-specific

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Metal command batching + wired memory budget respected | Supported | Deliberate budget gate: `resolve_decoder_backend_override` keeps the decoder off Metal when `pack_bytes * 2 > total_RAM` (`executor.rs:421-486`). Decoder-stage placement under Auto is under active A/B (whole-decoder CPU vs Metal; prefill favors Metal 42 vs 59 ms/prompt-tok, decode favors CPU 88 vs 129 ms/tok) -- verdict pending, current shipped behavior unchanged. |
| CPU thread pool sized for P/E cores | Supported | Shared `adaptive_thread_count_for_available` policy (`cpu_graph.rs:257-289`) by workload class and backend type; no family override needed. |
| Accelerate/BLAS used where it wins | Supported | Generic BLAS backend wiring (`cpu_graph.rs:4844-4852`), inherited via the shared CPU graph path. |

## 7. Backend coverage matrix

Every cell must be answered. An unsupported backend is acceptable ONLY with a
justification and an unlock plan -- "nobody tried" is not a justification.
Golden-verified means byte/parity fixtures pass ON that backend;
utilization-measured means the GPU weight placement gate (or an equivalent
profile) proved the compute actually runs there (golden output alone cannot,
see `docs/design/gpu-weight-placement.md`).

| Backend | Supported? | Golden-verified? | Utilization measured? | Justification + unlock plan if unsupported |
| --- | --- | --- | --- | --- |
| CPU | Yes | Yes (3/3 byte-identical: jfk/zh/en_zh, dev q8_0 pack; q4_k spot) | Yes (stage breakdown + bandwidth-ceiling profile, `tmp/fr2-measure`, commit `ace19ee`) | |
| Metal | Yes | Yes (q4_k spot-checks byte-identical on jfk + zh, post graph-reuse) | Yes (same profile run; decode 129.1 ms/tok = 2.21x bandwidth floor) | |
| CUDA | Untested | No | No | No family-level exclusion (shared qwen executor path); no CUDA host available. Unlock: community/dev-host run of the family golden + `firered_llm_perf_ab`; Windows CUDA leg of the release matrix builds it. |
| Vulkan | Untested | No | No | Same as CUDA; additionally the xasr-class offset-view fix (0.1.22) hardened the shared Vulkan path. Unlock: run on AMD/Intel Vulkan host (planned: user's Windows/AMD machine). |
| HIP | Untested | No | No | Runs the plain per-chunk path; qwen's HIP prefill-chunk tuning deliberately not replicated (see Known dead ends). Unlock: HIP host validation. |

## 8. Correctness & quality

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| WER vs fp16 measured for every shipped quant tier | Deferred | Single shipped tier (q4_k). The 0.1.17-era `jfk_wer_vs_fp16: 0.0` in catalog perf is STALE (measured before the 0.1.22 rework) and must not be cited. Unlock: re-measure current code against the internal fp16 pack on a host that fits it; refresh catalog perf fields in the same pass. |
| Golden coverage includes long audio AND a cross-backend parity fixture | Supported | Short goldens 3/3 (en/zh/mixed, `executor.rs:678-729`); Metal-vs-CPU byte parity spot-checked. "Long audio" beyond 40s is upstream longform slicing by design (each slice <= 40s, fail-closed at the cap), covered by the generic longform tests. |
| Official decode parameters honored (suppression, stop tokens, upstream reference settings) | Deferred | Prompt text verified verbatim against upstream `fireredasr2/asr.py`; greedy decode matches upstream's simple greedy path. Remaining: line-item cross-check of sampling/penalty defaults against `FireRedTeam/FireRedASR2S@4e7d9aa`. Unlock: that one comparison pass. |
| Long-audio degradation checked (repetition, drift, truncation) | Supported | 40s hard cap fails closed (`AudioTooLong`, no silent truncation); within-cap repetition covered by the shared degenerate-loop guard (issue #60 class). |

## 9. Resource limits & fail-closed

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Max audio length / context budget derived and over-limit behavior fails closed | Supported | `FIRERED_LLM_MAX_INPUT_SECONDS = 40`; over-limit -> typed `AudioTooLong` error (`executor.rs:99-100,232-237`). KV capacity is request-sized (prompt + generation budget), not the decoder's native 32768. |
| Streaming first-token latency floor documented (chunk accumulation math; streaming families, otherwise Not applicable) | Deferred | Family registers snapshot-based streaming; the first-token floor for the 8B snapshot path is not yet derived/documented. Unlock: derive from snapshot cadence + prefill cost (numbers now exist in `tmp/fr2-measure`). |
| KV growth rate per audio second known | Supported | 112 KB/token x 12.45 prompt-tok/audio-sec ~= 1.43 MB/audio-sec; ~117 MB at the 40s cap (measured frame rate, `tmp/fr2-measure`). |
| Metal wired-memory profile captured | Supported | Peak RSS 5410 MB (Metal q4_k, jfk, commit `ace19ee`) plus the `pack_bytes*2` budget rule; RSS-vs-wired caveat noted (footprint sampling used). |
| Multi-session scaling behavior known (server concurrency) | Deferred | No per-model admission control: concurrent requests each construct a ~5GB decoder runtime; concurrent OOM risk on small hosts is real and undocumented to users. Unlock: per-model concurrency cap or serialization in the server (flagged as a pre-GA concern). |
| Energy footprint noted (battery-relevant platforms) | Deferred | Not measured (needs sudo powermetrics window). Unlock: one measured transcription with energy sampling during a maintenance window. |

## 10. Engineering completeness

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| `warm_up` is a real implementation, not a stub | Supported | Streaming path uses shared `decode_warm_up_silence` (real silent decode) via the incremental driver (`incremental_streaming_driver.rs:786-789`). |
| Reference dumper exists for this family | Deferred | Only the one-shot T5 `parity_tests` scratchpad dump exists (`llm_transformer.rs:397-655`); no standing `tools/reference_backends/`-style dumper. Unlock: extract the parity dump into a standard per-family dumper (program-wide tooling item). |
| Registry / catalog / docs wired (MODEL_ONBOARDING checklist done) | Supported | Arch descriptor, executor/decode-policy registries, dispatch, registry toml, catalog entry + card all present (verified per-file in the static audit). |
| Peer benchmark recorded (table below, all fields) | Deferred | Peer determined: official PyTorch stack `FireRedTeam/FireRedASR2S@4e7d9aa` (sherpa-onnx ships only the CTC/AED variants; CrispASR/transcribe.cpp do not list the LLM variant). Run pending -- exclusive measurement window required. |

### Peer benchmark record

Record enough that anyone can re-run this comparison later. "Faster than X" is
not auditable without the exact peer version, model build, audio, and machine.

| Field | Value |
| --- | --- |
| Peer project (name + commit or version) | FireRedTeam/FireRedASR2S @ 4e7d9aaf (official PyTorch stack) -- run pending <!-- TODO:fill --> |
| Peer model + quant build | <!-- TODO:fill --> |
| Peer program version | <!-- TODO:fill --> |
| Test audio (file, duration, language) | <!-- TODO:fill --> |
| Machine (chip, RAM, OS) | <!-- TODO:fill --> |
| Peer numbers (RTF / peak memory / utilization) | <!-- TODO:fill --> |
| OpenASR numbers (RTF / peak memory / utilization) | <!-- TODO:fill --> |

## Known dead ends (do not re-litigate)

Verdicts that apply to this family, so future work does not re-run dead
investigations. Repo-wide precedents to inherit where relevant: F16 activation
on Apple M1 (encoder-only gave zero win, cast economics lock the trunk;
verdict 2026-07-14); qwen speculative decode (acceptance alpha ~= 0.05, judged
dead). Add family-specific verdicts with the measurement behind each; write
"None yet" if the family has none.

| Dead end | Verdict / evidence | Date |
| --- | --- | --- |
| HIP prefill-chunk tuning replication | Deliberately skipped: 40s cap keeps prompts short; qwen's discrete-GPU prefill-chunk tuning judged not worth replicating (`llm_transformer.rs:13-19`) | 2026-07 (code-in) |
| Per-stage backend split inside one executor instance (prefill Metal + decode CPU) | Blocked without double-loading 4.7GB weights (shared executor binds weights to one backend at construction); double-load proven unsafe on 16GB (swap abort). Revisit only with a shared-executor multi-backend rework. | 2026-07-22 |
| Noisy-host benchmarking | "31% win" and "mimo regression" both proved to be shared-host load artifacts; every number in this form must carry commit + idle-state disclosure | 2026-07-22 |
