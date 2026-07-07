# Roadmap

This file is the source of truth for OpenASR sequencing. Implemented status is
the [Implemented baseline](#implemented-baseline) section below.

OpenASR is licensed under Apache-2.0 and maintained as the public open core.
Public binary-release readiness (signed/checksummed release artifacts and
package-manager channels) is still deferred; build from source meanwhile.

## Implemented baseline

These were prior roadmap goals and are now shipped on the native runtime path:

- Ten native model families: Whisper, Cohere Transcribe, Qwen3-ASR,
  Parakeet-CTC, Parakeet-TDT (25 European languages), wav2vec2-CTC (incl.
  data2vec), Moonshine, Dolphin (Chinese dialects), SenseVoice
  (zh/yue/en/ja/ko), and X-ASR (Zipformer).
- Data-driven architecture registry (`arch/`): composer families (Cohere, Qwen)
  materialize from descriptors; dedicated executors (Whisper, Moonshine,
  Parakeet-CTC/TDT, wav2vec2, Dolphin, SenseVoice, X-ASR) own their loaders.
- `.oasr` packs are GGUF-backed and portable, with zero-copy mmap weight binding
  and graph buffer-reuse to bound peak RSS.
- Quantization profiles `fp16`, `q8_0`, `q4_k` (Qwen also `q3_k`).
- Performance harness with regression gates (`gate_peak_rss`, `gate_vs_cpp`) run
  from `perf/suite.toml` against committed baselines, plus competitive
  comparison vs whisper.cpp wired as gated entries.
- Active backends are `mock` and guarded `native`; wrapper-era backends and
  legacy command surfaces are removed.
- Native realtime has a declared-pack true-streaming path for Qwen3-ASR,
  Whisper, Cohere Transcribe, Moonshine, Parakeet-CTC, Parakeet-TDT, and
  wav2vec2-CTC through family-specific streaming executors. Local temporary packs pass the ignored
  real-runtime smoke; published release packs and product claims remain gated.
- Desktop/server requests carry a generic native execution target
  (`auto`/`cpu`/`accelerated`) through preferences, file transcription, remote
  file transcription, realtime, and dictation. Per-device GPU/provider pinning is
  intentionally not part of the current surface.
- Desktop remote compute has secure HTTPS/WSS client/server plumbing with
  approved pairing, TOFU fingerprint pinning, keychain device credentials, file
  transcription routing, realtime routing, revocation, and server-history
  isolation for paired device-token compute requests. The remaining release
  gate is end-to-end multi-device Desktop UI validation evidence.

## Active priorities

### P1: Architecture cohesion for scale

- Keep module boundaries high-cohesion/low-coupling.
- Keep public APIs minimal; keep model-family specifics under their executors /
  arch descriptors.
- Reduce per-family special-casing so new families onboard with minimal new code.

### P2: Quantization and performance

- Keep `.oasr` packs canonical and portable (no platform-specific embedded
  compute-cache payloads).
- Hold and tighten the committed performance gates; keep claims tied to executed
  baselines rather than asserting any final win over open-source runtimes.

### P3: Native quality and coverage

- Broaden longform/timestamp validation beyond internal smoke lanes toward a
  defensible WER/quality statement.
- Add further model families on the same `.oasr` path as needed.

## Deferred

- Public binary-release readiness (signed/checksummed release artifacts, package channels).
- Production distribution channels.
- Public true-streaming release guarantees, official streaming-enabled pack
  publication, and broad multilingual feature claims.

## Validation gate

For roadmap-impacting changes, run targeted validation first, then broader
regressions, and keep claims tied to executed checks.
