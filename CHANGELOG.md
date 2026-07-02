# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

### Changed

- Default model changed to `qwen3-asr-0.6b`; Qwen3-ASR family promoted to primary recommendation
- Catalog: dropped ModelScope mirror; unified quant tag scheme (`canonical_quant_tag`)
- Server: extracted config, history, and translation routes into dedicated modules
- Pre-open-source cleanup: removed private docs/artifacts, aligned license metadata, dep hygiene
