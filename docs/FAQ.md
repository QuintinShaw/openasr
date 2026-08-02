# OpenASR FAQ

This FAQ answers current behavior questions. Source-of-truth status is
[Roadmap](ROADMAP.md) (see its Implemented-baseline section).

## What is OpenASR?

OpenASR is a local-first ASR runner with a Rust CLI, a local server, a model
metadata registry, and a transcription-focused OpenAI-compatible API subset. It
runs native ggml-backed model packs offline.

## What native model families run today?

Sixteen, dispatched by a data-driven architecture registry (`arch/`): Whisper,
Cohere Transcribe, Qwen3-ASR, Parakeet-CTC, Parakeet-TDT (25 European
languages), wav2vec2-CTC (incl. data2vec), Moonshine, Dolphin (Chinese
dialects), SenseVoice (zh/yue/en/ja/ko), MiMo-V2.5-ASR (zh/en/yue; RVQ
tokenizer + Qwen2 decoder), FireRedASR-AED (Mandarin + English bilingual),
FireRedASR2-LLM (Mandarin + English bilingual; Encoder-Adapter-LLM),
Fun-ASR-Nano (zh/en; SAN-M encoder + Qwen3 decoder), Granite Speech 4.1
(en/fr/de/es/pt/ja; Conformer + Q-Former + 2B decoder),
MOSS-Transcribe-Diarize (zh/en meeting transcription with in-decoder speaker
segmentation), and X-ASR (Zipformer). They run local offline transcription on CPU and Metal
lanes. See [Known Limitations](KNOWN_LIMITATIONS.md) for per-family
streaming and word-timestamp support, and the per-model cards under
[`model-registry/models/`](../model-registry/models/) for quant tiers.

## What backends are active right now?

- `native` (default): local `.oasr` pack execution with staged fail-closed
  boundaries.
- `mock`: deterministic local mock transcription, opt-in via `--backend mock`
  (hidden in `--help`) for plumbing and CI.

## Does OpenASR download models automatically?

Never silently. `openasr pull` is the explicit command for published packs. In
addition, `transcribe`/`live` will install a missing model for you, but **only
through a visible consent prompt** (showing model, quant, size, host, and
license); `--offline` or any non-interactive run fails closed before touching the
network. The shared resolve path and the HTTP server never pull -- the server
runs only an explicit local pack.

## Can I download models for local experiments?

Yes. For published packs, use `openasr pull <id>:<quant>` or a bare `<id>` for
the recommended quant. For unpublished local development/benchmark workflows,
stage artifacts under `./tmp/` with provenance recorded (source identity,
revision/path, SHA256, size, mirror endpoint if used). Do not commit downloaded
artifacts.

## What pack format does the runtime accept?

Only `.oasr` (GGUF-backed internally). The bare `.gguf` extension is no longer
accepted as CLI run input or importer output. See
[Format Contract](format/OASR_PACKAGE_CONTRACT_V1.md).

## How do I build a pack?

Use the per-family importer on a local HF-style source directory:
`import whisper`, `import qwen`, `import cohere`,
`import parakeet-ctc`, `import parakeet-tdt`, `import wav2vec2-ctc`,
`import moonshine`, `import dolphin`, `import sensevoice`. Each accepts
`--quantization fp16|q8_0|q4_k`
(Qwen adds `q3_k`). `openasr pull` installs already-published packs; it does not
replace local importer workflows. There is no `quantize` command.

## Can I validate or inspect a local pack?

Yes. `openasr verify <path.oasr>` and `openasr show <path.oasr>` probe a
caller-provided local `.oasr` file via ggml. They reject remote URLs, missing
paths, and directory paths, and
do not run inference.

## Can I transcribe with a local native pack?

Yes. `openasr transcribe <audio>` uses the native backend by default and runs an
installed model (offering to install the default model on first use). To pin an
exact local pack instead, pass `--model-pack <local.oasr>`: the path must be a
local regular `.oasr` file; remote URLs/downloads are rejected. Current scope is
offline/final-only and fail-closed by stage.

## What is the staged fail-closed contract on native?

Execution is split into separate stages so failures are unambiguous rather than
a single opaque "native failed": runtime-pack path → metadata → tensor index →
encoder tensor binding/materialization → encoder graph → tokenizer load →
decoder tensor binding/materialization → decoder graph → greedy decode →
decode text.

## Is realtime true streaming ASR available?

For runtime packs that declare the streaming feature and whose family has a
registered streaming executor (e.g. X-ASR/Zipformer, Qwen3-ASR, Whisper),
native frame-synchronous streaming emits incremental partials. Packs without that
metadata fall back to final-per-utterance output. Official published streaming
packs and public product guarantees are still pending — see
[Known Limitations](KNOWN_LIMITATIONS.md).

## Is diarization available?

Yes, opt-in via `--diarize` (and the API `diarize` flag). It uses the
ReDimNet2-B6 speaker-embedding pack (ggml-graph `.oasr`) and the optional
pyannote segmentation capability pack (pulled or
installed on demand) to attribute anonymous `SPEAKER_NN` labels onto any model's
transcript; without the required ReDimNet2-B6 pack the request fails closed rather
than fabricating speakers.

Cross-file speaker identity -- naming the same person consistently across
separate recordings, rather than a fresh `SPEAKER_NN` number each time -- is a
separate, opt-in system called Voice ID, exposed through the operator-only
`/v1/voice-id/*` API (`crates/openasr-server/src/routes/voice_id.rs`): person
and sample CRUD, consent revoke, and metadata export. Enrolling a person
requires at least 10 seconds of clean speech per registered sample
(`MIN_SAMPLE_SPEECH_SECONDS`,
`crates/openasr-core/src/diarize/voice_id/quality.rs`); once a person is
enrolled, automatically naming an anonymous `SPEAKER_NN` label to that person
during diarization additionally needs at least 8 continuous seconds of that
person speaking uninterrupted in one turn to clear naming's evidence gate
(`MIN_CONTINUOUS_SPEECH_SECONDS_FOR_NAMING`,
`crates/openasr-core/src/diarize/voice_id/identity.rs`). A speaker who is not
enrolled, or whose evidence falls short of that gate, simply keeps a
session-relative `SPEAKER_NN` number -- Voice ID never fabricates an identity
match.

OpenASR does not perform source/audio separation (isolating vocals from
background music or noise, the way Demucs or similar stem-splitting tools do).
Diarization and Voice ID only label, and optionally name, existing speech
segments within a single mixed audio stream; there is no implementation of, or
plan for, stem separation.

## Does the server keep the model resident in RAM forever?

No. By default the daemon releases the bound model's resident runtime (mmap,
tensors, Metal state) after 10 minutes with no active transcription request or
realtime session; the next request just reloads and re-warms it, paying the
normal cold-build cost again. Adjust or disable this via the `idle_unload`
preference (`never`, `now`, `2m`, `10m` (default), `1h`).

## Is OpenASR public/open source now?

Yes. OpenASR is licensed under Apache-2.0 and maintained as the public open core.
Model packs are distributed separately under their own upstream licenses. See
[Roadmap](ROADMAP.md).

## Are official installers/releases available?

Not yet. Public binary-release readiness (signed, checksummed release artifacts
and package-manager channels) remains deferred. Build from source meanwhile — see
the README and CONTRIBUTING.

## Is ffmpeg required?

Not for current mock and native paths. OpenASR does not bundle/install/manage
ffmpeg.

## Is the API fully OpenAI-compatible?

No. OpenASR implements a focused local transcription subset plus local realtime
routes. See [Known Limitations](KNOWN_LIMITATIONS.md).

## How are performance checks run?

Via the committed `bench-suite` (`perf/suite.toml`), which records RTF and peak
RSS per entry under subprocess isolation and gates against committed baselines
(`gate_peak_rss`, `gate_vs_cpp`), including competitive comparison vs
whisper.cpp. See [Performance](../perf/PERFORMANCE.md).

## Related docs

- [Roadmap](ROADMAP.md)
- [Known Limitations](KNOWN_LIMITATIONS.md)
- [Docs Index](DOCS_INDEX.md)
- [Format Contract](format/OASR_PACKAGE_CONTRACT_V1.md)
