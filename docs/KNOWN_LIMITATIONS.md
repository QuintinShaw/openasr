# Known Limitations

This page lists current user-visible limits. For implementation truth and
sequencing, see [Roadmap](ROADMAP.md) (Implemented-baseline section).

## Current limitations

- OpenASR is Apache-2.0 open core, but there is no public binary release yet: no
  published binaries/installers/checksums, and no package-manager channels. Build
  from source meanwhile. Public model-pack distribution is limited to catalog
  entries explicitly marked `public:true`.
- The only executable backends are the default `native` and the opt-in `mock`
  stub. Native transcription runs offline from `.oasr` runtime packs -- a pinned
  `--model-pack`, an installed pack, or one the CLI installs on first use through
  a visible consent prompt -- and stays fail-closed by stage.
- The consent-gated CLI pull, the no-silent-download boundary, and pull/install
  mechanics are centralized in [Model Catalog, Registry, and Distribution](MODEL_CATALOG_ARCHITECTURE.md);
  the HTTP server never pulls.
- True-streaming native ASR exists only for runtime packs that explicitly declare
  `openasr.features.streaming=ggml-true-streaming-v1` and whose family has a
  registered streaming executor. Local temporary packs for Qwen3-ASR, Whisper,
  Cohere Transcribe, Moonshine, Parakeet-CTC, and wav2vec2-CTC have passed the
  ignored smoke, but official published packs with that metadata and public
  product guarantees are still pending. Dolphin has no streaming executor at
  all yet, so it always runs final-per-utterance.
- Speaker diarization is opt-in (`--diarize` / the API `diarize` flag). It uses
  pure-Rust WeSpeaker speaker-embedding and pyannote segmentation capability packs
  (pulled/installed on demand) to attribute anonymous `SPEAKER_NN` labels onto any
  model's transcript; Cohere packs that declare
  `openasr.features.diarization=cohere-token-stream-v1` can additionally emit
  inline speaker tokens. Without the required capability pack a diarize request
  fails closed rather than fabricating speaker labels. Labels are session-relative
  and are not a stable cross-file speaker identity; see [SECURITY.md](../SECURITY.md)
  for the diarization privacy model and the remote-mode trust contract.
- Phrase bias / hotword boosting is implemented for the native runtime decode
  path. Requests still fail closed when the selected model tokenizer cannot
  encode a requested phrase, and the mock backend still rejects non-empty
  phrase-bias requests.
- Word-level timestamp requests are accepted and exported in JSON/VTT. Whisper
  uses native decoder cross-attention frame probabilities and the CTC families
  (parakeet-ctc, wav2vec2-ctc) use decoder frame spans; Qwen, Cohere, and
  Moonshine fall back to decoder token-position estimates because those runtimes
  do not expose acoustic attention, so their word timings are approximate.
  Dolphin does not emit word-level timestamps at all -- its CTC/attention joint
  decode only returns a single segment-level span, so `--word-timestamps`
  requests against a Dolphin pack yield an empty word list rather than an error.
- Hardware execution target selection is generic: Desktop/server requests support
  `auto`, `cpu`, and `accelerated` when the native runtime reports an accelerated
  device. There is no per-provider/per-device pinning surface such as `gpu0`,
  and unavailable accelerated targets still fail closed or collapse back to
  supported choices as appropriate.
- No public reproducible real-backend benchmark or long-audio stability evidence
  is published. The performance harness, regression gates, and competitive
  comparisons are internal (see [Performance](../perf/PERFORMANCE.md)); no claim of
  having finally beaten open-source baselines is made — only that the harness and
  gates are in place.
- No public quality/WER guarantee is claimed for longform timestamps/exports;
  these are validated on internal smoke lanes only.
- Cohere longform carries a model-specific safety policy on top of the shared
  planner contract (chunk cap, no overlap, prompt carry disabled, Metal multichunk
  prefers CPU decoder). It matches current correctness/perf evidence and is not yet
  generalized into a model-agnostic runtime tuning layer.
- System-audio capture now has native smoke backends on macOS, Windows, and
  Linux, but Windows remains all-system WASAPI loopback rather than per-process
  capture, Linux depends on `pactl`/`parec` monitor-source capture, and
  Windows/Linux real playback smoke still needs to be executed on those OS
  sessions.
- `serve` is single-model: it runs the one pack resolved at launch
  (`--model-pack` / an installed `--model`). There is no per-request lazy model
  loading or an `openasr ps`-style multi-model runner yet -- restart `serve` to
  switch models.
- `openasr pull` always fetches and re-installs; it has no incremental update
  (no revision/digest diff, `up to date` check, or `--force`). Re-pulling a model
  re-downloads it.
- Source-language control is per-model and capability-gated (see
  `openasr show <pack>` / `/v1/capabilities`). Multilingual Whisper auto-detects an
  unset language and accepts an explicit `--language`; Cohere and the English-only
  families resolve to their fixed/default language. Qwen3-ASR auto-detects
  internally but does **not** expose the detected language, so its reported
  `language` is null and an explicit `--language` is rejected rather than silently
  ignored -- use a multilingual Whisper pack when you need to force or read back the
  language. (Wiring Qwen's text-prompt language conditioning is tracked, but needs a
  real-pack parity check against the reference inference before it can be claimed.)
  Dolphin is specify-only: it does not auto-detect, so an explicit `--language`
  selects one of its 14 recognition codes (`zh` plus 13 Chinese regional-dialect
  codes such as `zh-sichuan`, `zh-shanghai`, `zh-hebei`) via a decode-prompt
  token, defaulting to `zh` when unset; an unsupported code is rejected rather
  than silently falling back.

## What works now

See [Roadmap](ROADMAP.md) (Implemented-baseline section) for the current
working behavior matrix.

## Related docs

- [Model Catalog, Registry, and Distribution](MODEL_CATALOG_ARCHITECTURE.md)
- [Roadmap](ROADMAP.md)
- [FAQ](FAQ.md)
- [Docs Index](DOCS_INDEX.md)
