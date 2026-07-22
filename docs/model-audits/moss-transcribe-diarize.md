# Model release audit: moss-transcribe-diarize

> **Policy.** Should-support items MUST be `Supported`; anything else requires a
> detailed justification and an explicit unlock condition. This form ships with
> the model release. A family without a completed form does not enter the
> release flow: `tooling/publish-model/scripts/_manifest.py --public` fails
> closed on a missing or half-filled form.

| Field | Value |
| --- | --- |
| Family (`models-core.toml` `family`) | moss-transcribe-diarize |
| Models covered | moss-transcribe-diarize (OpenMOSS/MOSS-Transcribe-Diarize, 0.9B: Whisper-Medium-arch audio encoder [80-mel, d_model 1024, 24 layers, 16 heads, max_source_positions 1500] -> VQAdaptor [4x time-merge + 3-layer MLP+LayerNorm, ~9.4M params, NO VQ codebook] -> Qwen3-0.6B decoder [28 layers, QK-norm, no attn bias, GQA, tied embeddings]). Joint transcription + self-diarization + inline timestamps (`[S01]`/`[2.32]` are ordinary BPE tokens the decoder emits). |
| Auditor / date | Quintin (with agent-collected evidence) / 2026-07-23 static pass. **Pre-release gate**: this family is NOT shipped -- no `catalog.json` entry, no `models-core.toml` entry, no published `.oasr` pack (only a private dev fp16 pack exists). This form is the release-blocking completion certificate, not a backfill. |
| Core version + commit audited | Static pass: main `8f033ec`. Family landed as PR #157 (`e04b7df`, single squashed commit spanning importer + full whisper-qwen3 execution graph + KV cap + Metal pin-off + resident-f16/flash encoder + shared-driver decode). No measurement pass has been run for this family yet (see sections 3/8/9 sentinels). |
| Bench hardware | None captured yet. The family is unreleased and no quiet-window measurement matrix exists; every measurement row below carries a sentinel + measurement plan and asserts NO numbers. Reference host when measured: Apple M1, 16GB, macOS (CPU only -- Metal is disabled for this family, section 6). |

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
| Graph reuse / persistent session (no per-request graph rebuild) | Supported | Intra-request: the `MossEncoderRuntime` is built ONCE and reused across every 30s audio chunk, binding the six 2D projection weights per layer zero-copy from the mmap'd pack on each `encode()` (`encoder_graph.rs:22-33` module doc, `executor.rs:259-294`). The Qwen3 decode graph is reused across decode steps on the Metal/single-GPU lane via `run_prefill_auto_last_hidden`/`run_step_auto` (`llm_decoder.rs:199-207,332-341`); CPU rebuilds the forward graph per step by design of the shared qwen executor (same as firered2-llm/qwen). NOTE: there is NO cross-request thread-local runtime cache yet (unlike firered-aed's `BoundedRuntimeCache`) -- a second transcription reloads both weight contexts. Tracked as the section-3 "resident pool" Deferred row, deliberately left to a follow-up by the correctness-first landing. |
| Op fusion opportunities reviewed (norm+matmul, QKV, rope, ...) | Supported | Encoder rides the shared `nn::encoder::transformer_layer` primitive (`encoder_graph.rs:382-392`), inheriting whatever fusion that composer carries; no moss-specific op stitching. Decoder reuses qwen's `Qwen3AsrLlmWholeDecoderGraphExecutor::new_with_rms_norm_epsilon_and_fused_logits_head` (`llm_decoder.rs:118-124`) -- fused QKV projection load and fused logits head, byte-for-byte the same graph qwen3/firered2-llm use. Adaptor is a plain host-side MLP (not a graph, `adaptor_graph.rs:7-10`), too small to fuse (~9.4M params). |
| Batching / serve-batch path | Deferred | No `moss` consumer of `seq2seq_serve_batch.rs` (only cohere/qwen have one); each `execute()` runs exactly one utterance (`executor.rs:181-434`). Unlock: implement a moss serve-batch decoder runtime after the shared serve-batch engine is promoted out of its env gate (same pillar-2 dependency firered-aed/firered2-llm cite). |
| Encode-decode pipelining | Not applicable | Single-shot per utterance: the executor loops the fixed 30s encoder over independent chunks, concatenates, runs the adaptor over the full sequence, builds ONE ChatML+audio-span prompt, then decodes (`executor.rs:267-434`). The architecture needs the FULL audio up front to place its numeric time-anchor markers ahead of the prompt (`executor.rs:16-18`, `decode_prompt.rs:20-24`), so there is no chunk stream to overlap. |
| Arena / gallocr reuse across steps (no per-step allocator churn) | Supported | Encoder: the small tensors (conv stem, every 1D norm/bias, the fixed positional embedding, final LayerNorm) live in a WEIGHTS-usage static-tensor arena uploaded ONCE per `encode()`; only per-chunk mel + the all-zero mask are real graph inputs, and the scheduler gallocr is on (`encoder_graph.rs:28-33`, `graph_config.rs:18-19` `default_use_scheduler_when_unset: true`). Decoder KV lives in the preallocated per-utterance `Qwen3AsrLayerKvCacheState` arena (`llm_decoder.rs:163-180`); CPU rebuilds per token by design of the shared executor. |

## 2. Precision & quantization

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| KV cache quantization | Deferred | Decoder KV is host f32 via the shared `Qwen3AsrLayerKvCacheState` (`llm_decoder.rs:163-180`), bounded by the 8192-position cap (~1.9 GB reservation, section 9). Unlock: the shared qwen-family KV-quant project (mobile-driven); a per-family change is impossible without touching shared infra, same constraint as firered2-llm's row. |
| Activation precision policy chosen deliberately (f32 vs f16) | Supported | Deliberate: decode activations run f32 (shared qwen executor), while the encoder binds native f16 projection weights and runs flash attention (`encoder_graph.rs:22-33,392`). Repo-wide verdict (2026-07-14, M1): F16 activation gave zero encoder win, cast economics lock the trunk -- inherited (same Whisper/Qwen-audio encoder compute shape). Recorded in Known dead ends. |
| Keep-quantized matmul (native Q blocks bound, no load-time dequant; RAM orders q4 < q8 < fp16) | Supported | Encoder proj weights bind zero-copy as native f16 from the mmap'd pack, never touching host memory (`encoder_graph.rs:22-33`, `load_moss_encoder_weights_from_reader`). Decoder reuses qwen's mmap keep-quant path (`llm_decoder.rs:115-150`). The importer emits native quant blocks for linear weights (`package_import.rs:416-465`, `quantize_f32_to_ggml_tensor_data` per `classify_quant_tensor`), so q4_k/q8_0 packs will bind quantized directly into `mul_mat`. Only host dequant is the adaptor's ~19MB f32 MLP (`adaptor_graph.rs:7-10`), negligible. Caveat: RAM ordering is unverified in practice because only an fp16 dev pack has been built (see next row). |
| Quant tiers complete (q4_k / q8_0 / fp16) | Deferred | The importer supports all three via the shared `PackQuant` (`package_import.rs:162,416-465`), but NO pack has been built or published -- only a private dev fp16 `.oasr` exists (`executor.rs:512-516`), and there is no `catalog.json`/`models-core.toml` entry at all. Unlock (release-blocking): build fp16/q8_0/q4_k packs via the importer, WER-verify each (section 8), add the catalog + registry entries, publish. This is the core pre-release gate for the family. |

## 3. Memory & data movement

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| mmap weight loading | Supported | `GgufTensorDataReader::from_path` -> mmap-backed `GgmlLoadedWeightContext` for both encoder (`encoder_graph.rs:27,39-42`) and decoder (`llm_decoder.rs:115`), shared with every builtin family. |
| Resident pool reuse across requests (weights stay resident) | Deferred | Encoder + decoder runtimes are constructed fresh per `execute()` (`executor.rs:259,342-343`) -- there is NO thread-local `BoundedRuntimeCache` like firered-aed (`executor.rs:66-129`) or cohere. Within a request the encoder runtime is reused across 30s chunks, but a second transcription reloads both weight contexts (measured per-request rebuild cost ~1.8-2.0s for the sibling 8B family; smaller here but non-zero). Unlock: add the qwen-family thread-local runtime cache keyed on (canonical pack path, resolved backend), mirroring firered-aed's design; the correctness-first landing deliberately skipped it. |
| View contiguity tradeoffs audited (`cont`/copy nodes justified) | Deferred | No dedicated offset-view / fusible-copy sweep of the moss graphs. The encoder rides the shared `transformer_layer` and the decoder rides the shared qwen graph, both of which carry their own `cont`/permute nodes not audited from moss's side. Unlock: one cross-family contiguity sweep of the shared `transformer_layer` + qwen decoder (covers moss/qwen3/mimo/firered2 in one pass), method per the xasr/Vulkan misalign case. |
| Peak RSS/VRAM per shipped quant measured (quiet host) and reconciled against the weights+KV+activations budget; unexplained excess blocks release; catalog RAM requirement matches the measured peak | Deferred | **SENTINEL -- NOT MEASURED.** No quiet-window peak-RSS matrix has been captured for this family (it is unreleased and no measurement pass has run). Static budget (derived, not measured): encoder f16-resident proj weights + Qwen3-0.6B decoder weights (pack size, fp16 ~1.8 GB) + KV reservation ~1.9 GB across 28 layers at the 8192-position cap (`runtime_contract.rs:45-59`) + per-chunk encoder activations. The arch descriptor cites a dev-run ~2.8 GB CPU peak (`arch/mod.rs:1643`), but that is an informal dev observation, NOT a formal median-of-3 matrix -- do not cite it as a verified number. Measurement plan (release-blocking): once quant packs exist, run `/usr/bin/time -l` `maximum resident set size`, median of 3, isolated `OPENASR_HOME`, quiet-host-gated, CPU backend, {fp16,q8_0,q4_k} x {jfk 11s, 3-5min clip}; reconcile measured-minus-weights-minus-KV against the budget; write the catalog `peak_rss_bytes` from the CPU numbers. Metal peak is deferred behind the two Metal defects (section 6). No catalog RAM field to match yet (no catalog entry). |

## 4. Decode algorithms

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Greedy logits shortcuts (argmax path skips needless softmax/sort work) | Supported | Routes the ONE shared greedy driver `run_builtin_seq2seq_decode_policy` -> `run_seq2seq_greedy_decode_loop_v0` (`executor.rs:394-413`); single-pass `argmax_index`, softmax computed once post-argmax for confidence only, no top-k sort on the ~152k Qwen3 vocab. moss supplies `greedy_token_hint: None` (`executor.rs:151-152,173-176`), so the shared host argmax owns it. |
| Speculative decode: per-family verdict recorded (do it, defer it, or dead) | Deferred | Not evaluated, but the economics point at dead: the decoder IS Qwen3-0.6B, the exact model class the qwen speculative-decode "dead" verdict (acceptance alpha ~= 0.05) was measured on -- that verdict transfers strongly here (unlike the 8B firered2-llm case where it explicitly does not). Unlock: only if a decode-share profile (once section-3 measurements exist) shows decode dominating wall time on output-dense audio; otherwise inherit the qwen verdict and mark dead. |
| CTC blank-skip fast path (CTC families; otherwise Not applicable) | Not applicable | Pure autoregressive attention decode (`moss-transcribe-diarize.greedy.seq2seq.v0`, `decode_policy_component_registry.rs:266,289` `ctc_blank_token_id: None`). No CTC head in the architecture. |
| Decode guards are zero-cost on the hot path (degenerate-loop guard etc.) | Supported | The shared driver's degenerate n-gram-repeat guard is inherited for free by routing `run_builtin_seq2seq_decode_policy` (`executor.rs:394`, AGENTS.md "One greedy decode driver" invariant). The guard scans token-id history tail only (no logits access). moss declares no phrase bias (`supports_phrase_bias` false, `executor.rs:450-452`), no suppression, no extra stop tokens (`decode_policy_component_registry.rs:274-277`, all `None`); eot is ChatML `<|im_end|>` supplied per-request (`executor.rs:390-391`). |

## 5. Frontend & IO

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Mel/fbank frontend SIMD + parallelized | Supported | Whisper log-mel frontend `whisper_log_mel_spectrogram_16khz_mono_v0` (80-mel, 3000 frames per 30s chunk, `executor.rs:270-277`), FFT via SIMD `realfft` shared with the whisper family. Frame loop is single-threaded; at this model's cost that is the right call (encoder+decoder dominate), the same finding the sibling families reach. NOTE (cosmetic debt, not a correctness issue): the arch descriptor's `audio_frontend_id` is named `...fbank80.16khz.mono.v0` (`arch/mod.rs:205`) but the executor actually runs whisper log-mel, not kaldi fbank -- the id string is misleading and should be renamed to `...whisper-logmel80...` in a follow-up. Unlock to revisit the single-thread choice only if a stage profile shows the frontend as a real share. |
| Zero-copy audio path (no avoidable resample/copy hops) | Deferred | Inherits the shared `PreparedAudioInput` path (`executor.rs:219` reads `prepared_audio.samples_f32`). Direct 16kHz-mono-WAV is clean; the known cross-family gap is that any NON-WAV input decodes+resamples to in-memory f32 then writes a temp WAV that is re-read and re-parsed (the double full-buffer disk round trip firered2-llm's audit flagged, `audio/prepare.rs`). Not moss-local. Unlock: the shared in-memory-samples variant on `PreparedAudioInput` (same fix dispatched for the firered2-llm audit). |
| VAD cost measured and accounted | Not applicable | No VAD in this family's path (grep: zero VAD references in the moss module). The decode policy forces `LongFormMode::Off` via `SelfChunkingExecutorV1` (`decode_policy_component_registry.rs:278-290`) so the native slicer never runs; the executor does its own fixed 30s encoder chunking with globally-continuous time anchors instead. |

## 6. Platform-specific

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Metal command batching + wired memory budget respected | Deferred | **Metal is currently OFF for this family** -- `auto_gpu_policy: AutoGpuPolicy::ExceptMetal` (`arch/mod.rs:1650`) pins Auto to CPU pending two measured, open Metal defects (`arch/mod.rs:1634-1650`): (1) **encoder numeric divergence** -- the Whisper encoder's deep layers decorrelate on Metal (final-layer cosine ~0.20 vs the fp32 CPU reference, which matches), collapsing decode to an empty `[..][S01][..]` shell; (2) **per-step decode wired-memory blow-up** -- the per-token decode rebuilds the whole 28-layer ggml graph each step and on Metal the wired working set exhausts a 16GB machine (CPU peaks far lower). The KV-over-allocation HALF of defect 2 is already fixed via the 8192-position cap (`runtime_contract.rs:59`), and the `run_step_auto`/`run_prefill_auto` reuse path is wired as groundwork (`llm_decoder.rs:199-207,332-341`), but PR #157's own note confirms that does NOT by itself fix the blow-up -- the remaining per-step graph/wired behavior is Metal-only and still open. `ExceptMetal` only steers Auto; an explicit `execution_target=accelerated` still reaches Metal (and both defects), by design. Unlock: fix both defects, then flip to `AllBackends` (a one-line change per the descriptor note). |
| CPU thread pool sized for P/E cores | Supported | Shared `configure_model_runtime_graph_config_from_env` backend/threading resolution (`graph_config.rs:15-23`); CPU thread count via the shared adaptive policy, no family override. |
| Accelerate/BLAS used where it wins | Supported | Inherited via the shared CPU graph path (generic BLAS backend wiring in `cpu_graph.rs`); no moss-specific opt-out. |

## 7. Backend coverage matrix

Every cell must be answered. An unsupported backend is acceptable ONLY with a
justification and an unlock plan -- "nobody tried" is not a justification.
Golden-verified means byte/parity fixtures pass ON that backend;
utilization-measured means the GPU weight placement gate (or an equivalent
profile) proved the compute actually runs there (golden output alone cannot,
see `docs/design/gpu-weight-placement.md`).

| Backend | Supported? | Golden-verified? | Utilization measured? | Justification + unlock plan if unsupported |
| --- | --- | --- | --- | --- |
| CPU | Yes | Dev-only (not CI-gated) | No | The reference decode is CPU (Metal disabled). Two dev-gated e2e goldens (`executor.rs:597-627`, `#[ignore]`, private fp16 pack): jfk and en_zh_mixed transcripts are text-identical to the HF fp32 reference including `[S01]`/`[S02]` speaker labels and time anchors within 0.05s (`executor.rs:528-547`); a 3-min aishell multispeaker clip is also pinned. But all are `#[ignore]` on a private pack, so NOTHING moss-specific runs in `cargo nextest run --workspace`. Unlock (release-blocking): commit a small-fixture CI golden. Utilization: CPU is the compute by construction; a stage-cost profile is the same quiet-window unlock as section 3. |
| Metal | No (disabled) | No | No | Pinned off via `ExceptMetal` (`arch/mod.rs:1650`) for the two open defects in section 6. Byte-parity cannot even be attempted until defect 1 (encoder numeric divergence) is fixed. Unlock: fix both Metal defects, then run the weight-placement gate on a Metal host to prove encoder+decoder compute lands on Metal, plus a CPU-vs-Metal byte-parity golden. |
| CUDA | Untested | No | No | No family-level exclusion (shared graph path, no family-specific kernels). No CUDA host available. Unlock: community / Windows-CUDA-release-leg run of a moss golden + a utilization profile. |
| Vulkan | Untested | No | No | Same as CUDA; the shared Vulkan path carries the xasr-class offset-view hardening. Unlock: run on an AMD/Intel Vulkan host (user's Windows/AMD machine). |
| HIP | Untested | No | No | Shared per-chunk path; no HIP host. Unlock: HIP host validation. |

## 8. Correctness & quality

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| WER vs fp16 measured for every shipped quant tier | Deferred | **SENTINEL -- NOT MEASURED.** Only a dev fp16 pack exists; q8_0/q4_k packs are not built, and there is no dataset WER (only single-clip text goldens). Unlock (release-blocking): build the quant packs, measure WER of q8_0 and q4_k vs the fp16 pack on a real Mandarin+English dataset (this family's target), and -- because the model emits speaker labels + timestamps -- also track diarization label accuracy and time-anchor drift, not just token WER. |
| Model ref alias forms resolve identically everywhere (bare family / `family:canonical` / every `quant_tag_cases.json` alias accepted by CLI and server match logic; covered by the catalog-wide alias matrix test) | Deferred | The family has NO `catalog.json`/`models-core.toml` entry yet, so it is not in the bundled catalog and the catalog-wide `native_quant_alias_catalog_matrix` test (PR #171) does not walk it. Unlock (release-blocking): add the catalog + registry entries with canonical ids (`moss-transcribe-diarize`, `:q4`/`:q8`/`:fp16`), after which the existing alias matrix covers it automatically. |
| Golden coverage includes long audio AND a cross-backend parity fixture | Deferred | Dev-only goldens exist (jfk 11s, en_zh_mixed code-switch, aishell 3-min multispeaker -- `executor.rs:534-547`, `tmp/moss-td/golden/*.json`) but are ALL `#[ignore]` on the private fp16 pack, so none run in CI, and there is NO cross-backend parity fixture (impossible while Metal is disabled). Unlock: commit a small-fixture CI golden (a short clip AND a multi-slice long clip); cross-backend parity is blocked on the Metal defect fixes (section 6). |
| Official decode parameters honored (suppression, stop tokens, upstream reference settings) | Supported | The ChatML prompt is ported one-for-one from upstream `processing_moss_transcribe_diarize.py` (`MossTranscribeDiarizeProcessor.__call__` / `expand_audio_token` / `_audio_span_ids`) and verified token-for-token against the real golden `prompt_input_ids` (`decode_prompt.rs:1-30`), including the sparse time-anchor splicing and the verified-absent `enable_thinking` scaffold. Greedy decode to `<|im_end|>` with empty suppression and no extra stop tokens (`decode_policy_component_registry.rs:266-277`, all `None`), matching the upstream greedy path. Repetition control is structural (shared degenerate n-gram guard), not logit scaling. |
| Long-audio degradation checked (repetition, drift, truncation) | Supported | Three structural guards: (1) the shared degenerate n-gram-repeat guard via the shared driver (issue #60 class); (2) the whole audio is folded into ONE prompt/decode with globally-continuous time anchors (`SelfChunkingExecutorV1` forces `LongFormMode::Off`, `decode_policy_component_registry.rs:278-290`) so timestamps never reset per window; (3) a hard `AudioExceedsContext` fail-closed when the audio prompt would exceed the 8192-position KV cap (~7-10 min), computed up front before any decode (`executor.rs:321-339`). No silent truncation. The 3-min aishell golden exercises the long path (dev-only). |

## 9. Resource limits & fail-closed

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| Max audio length / context budget derived and over-limit behavior fails closed | Supported | The encoder is fixed 30s windows and never over-attends (`arch/mod.rs:1666-1672`, `FixedWindow`, same class as whisper). The decoder fails closed up front: when the audio prompt length reaches the 8192-position KV cap (`moss_td_kv_cache_positions`, `runtime_contract.rs:59-67`), the executor returns a typed `AudioExceedsContext` error with an approximate-minutes message BEFORE building the decoder (`executor.rs:321-339`), instead of a cryptic mid-decode KV-bounds error or silent truncation. |
| Streaming first-token latency floor documented (chunk accumulation math; streaming families, otherwise Not applicable) | Not applicable | This family has no real streaming path: its architecture needs the full audio up front to place time-anchor markers, so the registered snapshot-streaming session is an explicit completeness placeholder that degrades to "one final result at end of audio" (`executor.rs:467-496` doc). There is no partial cadence and therefore no first-token floor to derive. (The placeholder exists only because the builtin dispatch's completeness gate requires SOME streaming executor, same precedent as firered-llm.) |
| KV growth rate per audio second known | Supported | Derived from the Qwen3-0.6B decoder metadata (`runtime_contract.rs:50`: 28 layers, n_kv_heads 8, head_dim 128, f32 K+V): 8 x 128 x 2(k+v) x 28 layers x 4 B ~= 229 KB per decoder position. The audio-pad prompt grows at ~12.5 audio tokens/sec (`executor.rs:79` `AUDIO_TOKENS_PER_SECOND_FOR_LIMIT`), so ~2.8 MB of KV per audio-second, plus the fixed ChatML template and generated tokens. The full 8192-position cap reserves ~1.9 GB (matches the `runtime_contract.rs:57-58` note), the hard ceiling the section-8 fail-closed enforces. |
| Metal wired-memory profile captured | Deferred | **SENTINEL -- NOT MEASURED.** Metal is disabled for this family (section 6), so no wired-memory profile exists. Unlock: after the two Metal defects are fixed, capture a wired-memory profile (ggml-side Metal-buffer accounting, not process RSS -- per the firered-aed measurement-blindness finding) and fold it into the section-3 matrix. |
| Multi-session scaling behavior known (server concurrency) | Deferred | No per-model admission control: each concurrent request constructs its own encoder + decoder runtime (pack weights + ~1.9 GB KV each), so concurrent OOM on a small host is a real, undocumented risk (same shape as the firered families). Unlock: per-model concurrency cap or serialization in the server (shared pre-GA concern). |
| Energy footprint noted (battery-relevant platforms) | Deferred | Not measured (needs a `powermetrics` window, and Metal is off so only a CPU figure is possible today). Unlock: one measured transcription with energy sampling during a maintenance window, once packs exist. |

## 10. Engineering completeness

| Item | Status | Justification / evidence (+ unlock condition if not Supported) |
| --- | --- | --- |
| `warm_up` is a real implementation, not a stub | Supported | The snapshot-streaming path wires the shared `decode_warm_up_silence` (real silent decode) via `build_seq2seq_streaming_session` + `STREAMING_PARTIAL_TUNING_HEAVY_SNAPSHOT` (`executor.rs:34-36,477-496`), the same real warm-up the sibling seq2seq families use -- not a stub. |
| Reference dumper exists for this family | Deferred | No committed in-tree dumper (`find tooling -iname '*moss*'` = none). The golden constants (`tmp/moss-td/golden/*.json`, used by the `executor.rs` e2e tests) came from an HF fp32 reference run on a dev machine and are pinned as test goldens, but the producing script is NOT committed -- same caveat firered-aed carries. Unlock (nice-to-have, aids reproducibility): commit an equivalent in-tree dumper like `tooling/firered2-reference-dumper/`. |
| Registry / catalog / docs wired (MODEL_ONBOARDING checklist done) | Deferred | The CODE registries are all wired and cross-checked: arch descriptor (`arch/mod.rs:1614-1673`), decode-policy descriptor (`decode_policy_component_registry.rs:262-291`), executor component (`executor_component_registry.rs:97`), builtin dispatch (`builtin_execution_dispatch.rs:210`), hparam schema (`hparams.rs:248-269`), family registry (`ggml_family_registry.rs:302`), runtime-tensor contract (`runtime_tensor_contract_registry.rs:262`), capability declarations (`self_diarizes: true`, `emits_punctuation: None` pinned in the exhaustive test `arch/mod.rs:1711`). But the RELEASE surface is NOT wired: no `models-core.toml` entry, no `catalog.json` entry, no model card, no published `.oasr` pack. ADDITIONAL DEBT: the module doc (`mod.rs:11-17`) is STALE -- it still claims "only the importer exists so far ... the ggml execution graph ... has not been implemented ... not yet runnable", which directly contradicts the wired executor landed in the same PR; fix it. Unlock (release-blocking): build packs -> add `models-core.toml` + `catalog.json` + card -> fix the stale `mod.rs` doc -> complete the MODEL_ONBOARDING checklist. |
| Peer benchmark recorded (table below, all fields) | Deferred | **SENTINEL -- NOT RUN.** No third-party ggml port of MOSS-Transcribe-Diarize exists (transcribe.cpp, CrispASR, and sherpa-onnx do not carry it; it is not in `skills/openasr/references/`), so the only peer is the official PyTorch stack `OpenMOSS/MOSS-Transcribe-Diarize`. Hardware feasibility: the 0.9B model fits the 16GB M1 reference host in PyTorch fp16 (~1.8 GB weights), so a same-host peer run IS feasible (unlike the 8B firered2-llm). Unlock: once quant packs + a measurement pass exist, run the official stack and OpenASR on the same host + exclusive quiet window and fill the table below. |

### Peer benchmark record

Record enough that anyone can re-run this comparison later. "Faster than X" is
not auditable without the exact peer version, model build, audio, and machine.

| Field | Value |
| --- | --- |
| Peer project (name + commit or version) | Official PyTorch stack `OpenMOSS/MOSS-Transcribe-Diarize` (no third-party ggml port exists: transcribe.cpp / CrispASR / sherpa-onnx do not carry this family, and it is absent from `skills/openasr/references/`). PENDING an actual run. |
| Peer model + quant build | pending (official HF checkpoint, PyTorch fp16) |
| Peer program version | pending |
| Test audio (file, duration, language) | pending (planned: `fixtures/jfk.wav` English + a Mandarin clip + a multispeaker clip, since this family self-diarizes) |
| Machine (chip, RAM, OS) | pending (Apple M1, 16GB, macOS -- CPU only; feasible for both sides since 0.9B fits) |
| Peer numbers (RTF / peak memory / utilization) | pending |
| OpenASR numbers (RTF / peak memory / utilization) | pending -- no measurement pass has run for this family; every measurement row above carries a sentinel, not a number |

## Known dead ends (do not re-litigate)

Verdicts that apply to this family, so future work does not re-run dead
investigations. Repo-wide precedents to inherit where relevant: F16 activation
on Apple M1 (encoder-only gave zero win, cast economics lock the trunk;
verdict 2026-07-14); qwen speculative decode (acceptance alpha ~= 0.05, judged
dead). Add family-specific verdicts with the measurement behind each; write
"None yet" if the family has none.

| Dead end | Verdict / evidence | Date |
| --- | --- | --- |
| Conflating `max_position_embeddings` with the KV-cache working-set size | Dead: the checkpoint's `text_config.max_position_embeddings` is 131072 (the RoPE context ceiling), but `Qwen3AsrLayerKvCacheState` eagerly allocates the full span, so feeding it straight through reserves ~30 GB (Metal physically wires it -> exhausts a 16GB machine). Capped at a pragmatic 8192 ASR context (mirrors qwen3-asr's audio-encoder value), dropping the reservation to ~1.9 GB (`runtime_contract.rs:45-59`). Lesson recorded so it does not recur: an attention/positional ceiling is not a working-set size. | 2026-07 (code-in, #157) |
| Hand-rolled moss argmax decode loop | Dead: a hand-rolled loop misses the shared degenerate n-gram guard (issue #60 class) and drifts argmax/suppression/stop-token semantics; moss must route the shared `run_builtin_seq2seq_decode_policy` driver (`executor.rs:394`, AGENTS.md "One greedy decode driver" invariant). | 2026-07 (code-in, #157) |
| Treating the "VQAdaptor" as a vector-quantization codebook | Dead: despite the name there is NO VQ codebook in this checkpoint -- it is a plain 3-layer MLP+LayerNorm (`Linear 4096->1024 -> SiLU -> Linear 1024->1024 -> LayerNorm`), run as a small host-side computation, not a ggml graph (`adaptor_graph.rs:1-10`). Do not implement codebook lookup. | 2026-07 (code-in, #157) |
| F16 activation on Apple M1 | Inherited repo-wide verdict: encoder-only gave zero win, cast economics lock the trunk (same Whisper/Qwen-audio encoder shape as this family). Encoder still binds native f16 weights + flash; activations stay f32. | 2026-07-14 |
| Speculative decode for the Qwen3-0.6B decoder | Provisionally dead (inherited, not yet re-measured on moss): the decoder IS the Qwen3-0.6B class the qwen speculative-decode verdict measured (acceptance alpha ~= 0.05). Flip only if a decode-share profile later shows decode dominating. | 2026-07-14 (qwen), applied here |

> NOTE (open blocker, NOT a dead end): the two Metal defects in section 6/7
> (encoder numeric divergence + per-step wired-memory blow-up) are open follow-ups
> with a concrete unlock, not settled verdicts -- do not move them here.
