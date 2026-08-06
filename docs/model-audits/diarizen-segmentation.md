# Model release audit: diarizen-segmentation

> **Policy.** Should-support items MUST be `Supported`; anything else requires a
> detailed justification and an explicit unlock condition. This form ships with
> the model release. A family without a completed form does not enter the
> release flow: `tooling/publish-model/scripts/_manifest.py --public` fails
> closed on a missing or half-filled form.

| Field | Value |
| --- | --- |
| Family (`models-core.toml` `family`) | diarizen-segmentation |
| Models covered | diarizen-large-s80-v2 (BUT-FIT Large-s80-md-v2; fp16 capability pack) |
| Auditor / date | Quintin / 2026-08-06 |
| Core version + commit audited | OpenASR 0.1.30 / `5b199b29` plus this public-catalog change; runtime landed before 0.1.30 |
| Bench hardware | Apple M1, 16 GB, macOS; locked quality corpus is six 10-minute AISHELL-4/AliMeeting excerpts |

## 1. Graph & scheduling

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Graph reuse / persistent session (no per-request graph rebuild) | Supported | `DiariZenRuntime` owns a prepared `GgmlPersistentGraphSession`; the content-keyed pinned actor cache reuses it and rebuilds only a poisoned graph (`diarizen/runtime.rs`, `policy_runtime.rs`). |
| Op fusion opportunities reviewed (norm+matmul, QKV, rope, ...) | Supported | WavLM/Conformer uses shared ggml attention and norm helpers; Metal reports fusion enabled in the locked native benchmark. There is no RoPE in this family. |
| Batching / serve-batch path | Deferred | Working-set policy deliberately limits this large auxiliary graph to one 16 s window at a time. Unlock: benchmark multi-window batching and raise `max_parallel_windows` only if throughput improves without exceeding admission quotes. |
| Encode-decode pipelining | Not applicable | EEND segmentation is one feed-forward encoder/classifier pass with no autoregressive decode stage. |
| Arena / gallocr reuse across steps (no per-step allocator churn) | Supported | The graph and static tensor arena are built once per admitted actor and reused for all overlapping windows. |

## 2. Precision & quantization

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| KV cache quantization | Not applicable | There is no decode KV cache. |
| Activation precision policy chosen deliberately (f32 vs f16) | Supported | Dense weights are F16 while 411 normalization/affine/bias tensors remain F32; graph activations/output are F32. The locked Python adapter measured 8.1232% DER for this pack versus 8.1274% for the F32 reference. |
| Keep-quantized matmul (native Q blocks bound, no load-time dequant; RAM orders q4 < q8 < fp16) | Supported | Dense F16 tensors stay in the loaded ggml weight context; only small static affine tensors are explicitly materialized. No Q4/Q8 tier is shipped, so tier-ordering is not claimed. |
| Quant tiers complete (q4_k / q8_0 / fp16) | Deferred | The public model is fp16-only because diarization quality was validated only for the mixed F16/F32 pack. Unlock: add another tier only after the same six-file DER, golden argmax, memory and speed gates pass. |

## 3. Memory & data movement

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| mmap weight loading | Supported | `.oasr` is preflighted once and bound through the shared GGUF weight-context loader and immutable content snapshot. |
| Resident pool reuse across requests (weights stay resident) | Supported | `NativeExecutionServices::diarizen_segmenter_actors` caches the admitted runtime by architecture, content ID, representation and backend. |
| View contiguity tradeoffs audited (`cont`/copy nodes justified) | Supported | Explicit `cont` nodes appear only at convolution/attention layout boundaries required by ggml; complete input windows share the source buffer and only the padded tail allocates (`local_activity_owns_complete_windows_without_copying_the_source`). |
| Peak RSS/VRAM per shipped quant measured (quiet host) and reconciled against the weights+KV+activations budget; unexplained excess blocks release; catalog RAM requirement matches the measured peak | Deferred | The capability catalog has no RAM requirement field and the qualification captured throughput but not `/usr/bin/time` peak RSS. Runtime construction is still fail-closed behind derived system-memory admission. Unlock: add a quiet-host 3-run peak-RSS/Metal allocation capture when auxiliary packs gain a catalog RAM field. |

## 4. Decode algorithms

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Greedy logits shortcuts (argmax path skips needless softmax/sort work) | Supported | Post-processing takes a direct powerset argmax and median filter; it does not compute softmax or sort logits (`postprocess_logits`). |
| Speculative decode: per-family verdict recorded (do it, defer it, or dead) | Not applicable | No autoregressive decoder. |
| CTC blank-skip fast path (CTC families; otherwise Not applicable) | Not applicable | Not a CTC family. |
| Decode guards are zero-cost on the hot path (degenerate-loop guard etc.) | Not applicable | No token decode loop. Window length and cancellation are checked at the segmenter boundary. |

## 5. Frontend & IO

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Mel/fbank frontend SIMD + parallelized | Not applicable | The model consumes raw 16 kHz waveform through WavLM convolution, not mel/fbank features. |
| Zero-copy audio path (no avoidable resample/copy hops) | Supported | Complete 16 s windows are `PcmSlice` views over the normalized 16 kHz recording; only the final short window is padded and copied. |
| VAD cost measured and accounted | Not applicable | FireRed Stream-VAD is a separate shared stage. The recorded 60 s segmenter RTF covers only DiariZen's 29 overlapping windows and is labeled accordingly. |

## 6. Platform-specific

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Metal command batching + wired memory budget respected | Supported | Native Apple M1 qualification ran through the Metal scheduler with graph optimization, concurrency and residency sets enabled; system-memory construction remains transactionally admitted. |
| CPU thread pool sized for P/E cores | Supported | Candidate resolution uses shared `GgmlCpuGraphConfig::resolve_runtime_thread_count_for(...EncoderPrelude)` rather than a family-local thread count. |
| Accelerate/BLAS used where it wins | Supported | The shared ggml CPU/Metal runner selects host kernels; the family adds no scalar matrix implementation or BLAS bypass. |

## 7. Backend coverage matrix

Every cell must be answered. An unsupported backend is acceptable ONLY with a
justification and an unlock plan -- "nobody tried" is not a justification.

| Backend | Supported? | Golden-verified? | Utilization measured? | Justification + unlock plan if unsupported |
| --- | --- | --- | --- | --- |
| CPU | yes | yes (stage tensors, logits and powerset argmax vs pinned PyTorch dump) | yes (explicit CPU test backend) | Production fallback and parity reference. |
| Metal | yes | yes (locked native six-file DER and exact final segment parity) | yes (benchmark log identifies Apple M1 Metal scheduler) | 16 s median 1.8334 s; 60 s / 29-window effective RTF 0.5128. |
| CUDA | no | no | no | Auxiliary ggml CUDA qualification is not shipped. Unlock: run the same external golden, six-file DER and memory gate on a supported CUDA runner. |
| Vulkan | no | no | no | Auxiliary ggml Vulkan qualification is not shipped. Unlock: same parity/DER/resource suite on the Vulkan runner. |
| HIP | no | no | no | Auxiliary ggml HIP qualification is not shipped. Unlock: same parity/DER/resource suite on the HIP runner. |

## 8. Correctness & quality

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| WER vs fp16 measured for every shipped quant tier | Not applicable | This model emits speaker activity, not words. The relevant metric is DER; the sole shipped tier measured 7.9491% native DER. |
| Model ref alias forms resolve identically everywhere (bare family / `family:canonical` / every `quant_tag_cases.json` alias accepted by CLI and server match logic; covered by the catalog-wide alias matrix test) | Supported | One recommended fp16 variant is generated through the common capability-pack catalog and pull resolver. |
| Golden coverage includes long audio AND a cross-backend parity fixture | Supported | One-second stage golden validates CPU tensors; 60 s throughput and six 10-minute recordings validate long input; locked native Metal results match the qualified final-segment output. |
| Official decode parameters honored (suppression, stop tokens, upstream reference settings) | Supported | Runtime validates pinned 16 s window/1.6 s step, 20 ms frame stride, four local speakers, 16 powerset classes and 11-frame median filter from pack metadata. |
| Long-audio degradation checked (repetition, drift, truncation) | Supported | Six 10-minute recordings were scored with overlap included; native DER was 7.9491%. AISHELL speaker-count underestimation (4/6, 4/6, 3/6) remains explicitly documented. |

## 9. Resource limits & fail-closed

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Max audio length / context budget derived and over-limit behavior fails closed | Supported | The graph accepts exactly the pinned 16 s window; long recordings are deterministically windowed and the tail padded. Invalid geometry returns typed errors. |
| Streaming first-token latency floor documented (chunk accumulation math; streaming families, otherwise Not applicable) | Not applicable | Universal Voice ID is file-only; this provider is not advertised for realtime streaming. |
| KV growth rate per audio second known | Not applicable | No KV cache; per-window graph dimensions are fixed. |
| Metal wired-memory profile captured | Deferred | Scheduler residency behavior was logged, but a dedicated wired-memory trace was not retained. Unlock: capture the Metal resource profile together with the peak-RSS gate in section 3. |
| Multi-session scaling behavior known (server concurrency) | Supported | Content/backend-keyed actor ownership serializes mutable persistent-graph inference and reuses one admitted model; the segmenter working-set policy caps parallel windows at one. |
| Energy footprint noted (battery-relevant platforms) | Deferred | No Instruments energy sample is retained. Unlock: capture a 10-minute file job before enabling this optional provider on battery-sensitive realtime surfaces. |

## 10. Engineering completeness

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| `warm_up` is a real implementation, not a stub | Supported | Actor construction runs one zero-filled 16 s `infer` before publishing the runtime to the cache. |
| Reference dumper exists for this family | Supported | `tooling/diarizen-reference-dumper/dump_golden.py` emits the pinned PyTorch stages and final outputs consumed by the ignored native parity test. |
| Registry / catalog / docs wired (MODEL_ONBOARDING checklist done) | Supported | Capability metadata, strict pack validator, converter, runtime provider, non-commercial pull gate, model card and public catalog generation are all present. |
| Peer benchmark recorded (table below, all fields) | Supported | Same-host locked comparison records MOSS and segmentation-3.0 under one DER protocol; throughput is reported only for OpenASR native DiariZen. |

### Peer benchmark record

| Field | Value |
| --- | --- |
| Peer project (name + commit or version) | OpenASR MOSS in-decoder baseline and OpenASR segmentation-3.0 research adapter, locked 2026-08-02 bakeoff |
| Peer model + quant build | `moss-transcribe-diarize:q8`; pyannote segmentation-3.0 f32 + ReDimNet2-B6 fp16 |
| Peer program version | Locked bakeoff scripts and core runtime provenance summarized in `docs/DIARIZATION_PACK_PUBLISHING.md` |
| Test audio (file, duration, language) | Six 10-minute AISHELL-4/AliMeeting Mandarin meeting excerpts; fixed UEM; 0.25 s collar; overlap scored |
| Machine (chip, RAM, OS) | Apple M1, 16 GB, macOS |
| Peer numbers (RTF / peak memory / utilization) | MOSS DER 18.6787%; segmentation-3.0 research adapter DER 12.4466%; peer RTF/memory were not used for the quality claim |
| OpenASR numbers (RTF / peak memory / utilization) | Native DiariZen fp16 DER 7.9491%; 16 s Metal median 1.8334 s (RTF 0.1146); 60 s / 29 windows 30.7679 s (effective RTF 0.5128); peak memory deferred in section 3 |

## Known dead ends (do not re-litigate)

| Dead end | Verdict / evidence | Date |
| --- | --- | --- |
| DiariZen Base-s80 as the product option | Replaced by Large-s80-md-v2: locked F32 Base DER 9.0481% versus native Large v2 fp16 7.9491%; Base packs were removed. | 2026-08-03 |
| Q8 as a second public tier without a quality gate | Rejected for this release: only the mixed fp16/F32 pack has native golden and six-file DER evidence. | 2026-08-06 |
