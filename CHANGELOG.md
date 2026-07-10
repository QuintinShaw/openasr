# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Changed

- Default model changed to `qwen3-asr-0.6b`; Qwen3-ASR family promoted to primary recommendation
- `idle_unload` preference default changed from `never` to `10m`: a bound native model pack is now released from RAM after 10 minutes with no active request/realtime session, instead of staying resident until the daemon exits; an explicitly configured `idle_unload` (including `never`) is unaffected -- only the default changed
- Catalog: dropped ModelScope mirror; unified quant tag scheme (`canonical_quant_tag`)
- Server: extracted config, history, and translation routes into dedicated modules
- Pre-open-source cleanup: removed private docs/artifacts, aligned license metadata, dep hygiene

### Fixed

- `serve --model <id>` no longer rejects a catalog-resolved quant-pinned ref against a bare pack runtime id (previously failed with a self-contradictory "requires --model to match local source id 'X', got 'X'" error); the startup gate now uses the same tolerant bare-id matcher as transcribe and the server request path
