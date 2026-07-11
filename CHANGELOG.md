# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Server: `/health` now reports `model_resident`, whether the bound model's native runtime is currently loaded in memory versus idle-unloaded (or not yet loaded this boot) -- lets clients (e.g. the desktop status indicator) distinguish "ready, instant transcription" from "bound but will pay a cold rebuild on the next request" without guessing from the `idle_unload` timer. Additive; `model_installed` is unchanged.
- Server: `GET /v1/devices` enumerates the daemon's own ggml compute devices (Auto/CPU/accelerated), so a UI can read the backends inference actually runs on instead of enumerating its own runtime -- a shell built in a different backend shape than the sidecar (e.g. a CPU-only desktop supervising a Vulkan sidecar on Windows) previously hid the GPU. Device shaping (`compute_devices_from_runtime`, `default_execution_target`, `ComputeDevice`) lives in `openasr-core` as the single source of truth, reused by the endpoint and by a shell offline fallback.

### Changed

- Core: the offline and realtime-streaming dispatch stacks now share one process-wide executor instance for qwen, cohere, whisper, and moonshine (the families that host-materialize a prepared runtime instead of relying purely on the pack's own mmap), instead of each stack holding its own independent instance and cache. A model warmed on both stacks no longer pays for its resident weights twice: measured on `qwen3-asr-0.6b` q4_k, warming both stacks on the same pack now costs ~1x instead of ~2x (2965 MiB -> 2197 MiB physical footprint in a same-process repro). AED/CTC/transducer families keep zero-copy mmap-shared weights already and are unaffected.

### Fixed

- Server: a failed realtime native-streaming attach (the reused-worker send racing a dead decode thread) no longer leaks the process-wide native activity count. The leaked count made `idle_unload`'s reaper read permanently non-idle for the rest of the daemon's life -- silently disabling the resident-model eviction feature with no log or error, and pinning `/health`'s `model_resident` to `true`. The activity accounting is now an RAII guard carried through the attach message itself, so every exit path (successful attach, failed send, or a mid-session worker panic) retires it exactly once.
- `ggml`: statically-linked builds (macOS, Linux, and Windows GPU-feature builds where `GGML_BACKEND_DL` is off) no longer unconditionally `dlopen()` every `ggml-*.dll` next to the exe on startup -- harmless on its own, but actively dangerous when the exe directory also carries CPU BACKEND_DL plugin DLLs (e.g. a desktop bundle shipping them for other components): loading a second copy of ggml core collided with the statically-linked copy's global state and fastfailed the process. The backend directory scan is now gated on `OPENASR_GGML_BACKEND_DL_ENABLED` (same pattern as `OPENASR_GGML_NATIVE_ENABLED`); genuine `GGML_BACKEND_DL` builds are unaffected. Also: `catalog.public.json`/`catalog.public.signature.json` were missing from the `eol=lf` rules covering the private catalog trio, so Windows checkouts with `core.autocrlf=true` rewrote them to CRLF and broke the bundled-catalog sha256 signature check (fail-closed, blocking desktop bundling on Windows).

## [0.1.12] - 2026-07-11

### Added

- OpenAI API compatibility: `verbose_json` now carries `duration`, segment `id`s, and a top-level `words` array; error envelope includes `param`/`code`; the OpenAI-SDK `stream` form field is rejected with an actionable error instead of silently returning a non-streaming body
- Agent Skill: split into `SKILL.md` + `references/http-api.md` (progressive disclosure) with a verified OpenAI parameter compatibility matrix

- CLI redesign (round 1 + 2): newcomer-friendly subcommand surface, language capability framework, improved help output
- Sentence segmentation for long-form transcription, long-form progress endpoint, and Hugging Face token authentication for gated model pulls
- Full Whisper family (tiny/base/small/medium/large-v3) and remaining ASR models published to catalog; catalog signing pipeline
- Speaker diarization: full pipeline (engine + CLI), CampPlus speaker embedder, per-word speaker labels, diarization export
- Per-word confidence scores across all model families (seq2seq, CTC, X-ASR)
- Word-level timestamps across all families (acoustic cross-attention alignment for Whisper, frame spans for the CTC/transducer families, token-position estimates elsewhere)
- X-ASR (Zipformer) model family: catalog integration, frame-sync streaming, alias resolution
- Realtime translation pipeline: Hy-MT2 pack, streaming translation, HIP GPU support
- GPU backend plugin system (GGML_BACKEND_DL) for Windows; all-AMD HIP arch list
- Realtime translation: livelock fix for mixed-language (CJK + Latin) input
- Windows: UTF-8 console output, PATH executable detection, host RAM/disk probing, mmap'd model re-pull error
- Docker smoke test in CI; serve-batch real-pack parity lane
- Server: the daemon's bound native model pack is now warmed up in the background right after boot (bind), instead of on the first realtime WS attach, so the first dictation session no longer pays the cold model-pack-load cost (observed 1.7-2.1s) before its first partial; `/health` remains unaffected (bind-then-serve, never gated on warm-up)
- Server: `idle_unload` now actually releases the cached native model runtime (mmap/materialized tensors/Metal or CPU graph context) once idle past the configured threshold, freeing the RAM a bound pack otherwise held for the daemon's whole lifetime; a later request just rebuilds it through the normal load-and-warm-up path
- Server: stage timing and timestamps in daemon logs -- server boot, model-pack load, and realtime warm-up are now timestamped (wall-clock + monotonic), and `OPENASR_TIMING=1` adds a finer per-request tier (model resolution, longform slice decode); local-only (stderr), no telemetry
- `pull`: model-pack downloads now split into concurrent 64 MiB range-request segments (`OPENASR_PULL_CONNECTIONS`, default 4) for a 2-4x wall-clock improvement on large packs, with an ETag-guarded probe and automatic fallback to the existing single-stream path when the server does not support Range requests
- CI: release archives (the core fast-path build and the full binaries matrix, including the xcframework) now carry a verifiable SLSA build-provenance attestation (`gh attestation verify`) tying each shipped asset back to the CI run and source tag that built it

### Changed

- Default model changed to `qwen3-asr-0.6b`; Qwen3-ASR family promoted to primary recommendation
- `idle_unload` preference default changed from `never` to `10m`: a bound native model pack is now released from RAM after 10 minutes with no active request/realtime session, instead of staying resident until the daemon exits; an explicitly configured `idle_unload` (including `never`) is unaffected -- only the default changed
- Catalog: dropped ModelScope mirror; unified quant tag scheme (`canonical_quant_tag`)
- Server: extracted config, history, and translation routes into dedicated modules
- Pre-open-source cleanup: removed private docs/artifacts, aligned license metadata, dep hygiene
- Qwen3-ASR: audio encoder self-attention now runs through flash attention by default (opt-out via `OPENASR_QWEN_GGML_DISABLE_AUDIO_ENCODER_FLASH_ATTN`), sharing a Metal head-dim compatibility guard with Whisper's existing flash-attention path
- Dolphin: n-best rescoring now builds the decoder graph once per utterance and reuses it across hypotheses instead of rebuilding and re-uploading ~200 decoder weight tensors per hypothesis; measured on M1 with a 2.38s clip, RTF improves from 0.89/0.72 to 0.34/0.29 (CPU cold/warm) and peak RSS drops from ~3.7 GB to ~1.88 GB

### Fixed

- `serve --model <id>` no longer rejects a catalog-resolved quant-pinned ref against a bare pack runtime id (previously failed with a self-contradictory "requires --model to match local source id 'X', got 'X'" error); the startup gate now uses the same tolerant bare-id matcher as transcribe and the server request path
- `idle_unload` now actually reaches every native model family: the composed dispatch executor (used by Qwen3-ASR) previously inherited a no-op default for the unload hook, so a bound qwen pack's cached runtime never released on idle despite the server reporting the unload; qwen's thread-local decoder cache and the streaming warm-up gate are now keyed on a process-wide unload generation so a decode-worker thread that survives past an eviction re-warms and rebuilds instead of silently serving stale, pre-unload state
- X-ASR (Zipformer) streaming: fixed a use-after-free that could abort the whole daemon (`GGML_ASSERT(device) failed`) when a pooled streaming runtime migrated to a new decode-worker thread after the previous thread's Metal/GPU backend was torn down on its 60s idle release; the encoder graph now rebinds its cached ggml runners to the current thread, and a fail-closed guard turns any remaining stale-backend case into a typed per-session error instead of a process abort
- `pull`: model-pack downloads no longer silently fail after 30s on large packs -- a stall-detection duration was being applied as reqwest's total request timeout (which defaults to 30s if unset), killing any download whose wall-clock time exceeded that regardless of active progress; GPU backend-library pack downloads also gained the same retry/resume/stall-guard machinery the model-pack path already had, so a single network hiccup no longer fails the whole ~150 MB pack permanently
- Safetensors package-import parser (shared across 14+ model families): hardened against a crafted header driving an unbounded allocation, duplicate JSON keys, out-of-range/overlapping tensor offsets, and shape x dtype size mismatches, bringing it to parity with the already-hardened Whisper local-source parser; Whisper's package importer now reads through this shared hardened parser instead of a duplicated private copy
- Realtime streaming: terminal punctuation is no longer emitted at soft (mid-utterance) streaming boundaries, which previously left a stray sentence-ending mark mid-utterance in live captions
- `openasr live` now resolves catalog-only model aliases (e.g. `qwen:q8`) the same way `transcribe` does, instead of reporting an already-installed pack as "not installed"
- f32-to-f16 weight quantization: converged 12 divergent implementations (one of which did not round at all) onto a single round-to-nearest-even routine, fixing inconsistent bit-level quantization of the same source weight depending on which model family imported it (`wav2vec2_ctc` is the one family whose output bit pattern changes as a result)
