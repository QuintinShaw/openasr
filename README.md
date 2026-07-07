# OpenASR

[![CI](https://github.com/QuintinShaw/openasr/actions/workflows/ci.yml/badge.svg)](https://github.com/QuintinShaw/openasr/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)

**The App Store for local ASR models -- signed, offline, private.**

OpenASR is the Apache-2.0 **open core** of a local-first speech-to-text platform:
a single `openasr` Rust CLI, a local OpenAI-compatible HTTP API, and a signed
model catalog, all running native [ggml](https://github.com/ggml-org/ggml)-backed
inference across ten model families on CPU and Apple Metal. No cloud, no
telemetry, fail-closed by design.

[Website](https://openasr.org) - [Documentation](docs/DOCS_INDEX.md) - [Acknowledgments](ACKNOWLEDGMENTS.md) - [License](LICENSE)

> **Early stage (pre-v1).** OpenASR is under active development. CLI flags, the
> HTTP API, and the `.oasr` pack format can change between `0.y` releases without
> a compatibility promise. Not yet recommended for production. Website and docs:
> **<https://openasr.org>**.

<!-- demo GIF: placeholder, uploading 2026-07-07 -->

```bash
git clone --recurse-submodules https://github.com/QuintinShaw/openasr.git && cd openasr
cargo build --release -p openasr-cli          # binary at target/release/openasr
target/release/openasr transcribe audio.wav   # first run offers the default model, then all-local
```

The first run offers the default model (`qwen3-asr-0.6b`) with a visible
confirmation, then everything runs offline on your hardware.

## Why OpenASR

whisper.cpp and faster-whisper are excellent Whisper runners. OpenASR is a
broader local-first STT *platform* built around three things they do not bundle.

### Signed model distribution, not a pile of GGUFs

Models come from a catalog whose manifest is cryptographically signed; every pull
is checked against a committed public key and a SHA-256 hash before it runs.
`openasr pull qwen3-asr-0.6b:q8` fetches one of **50 signed quant variants across
17 catalog packs** -- no hand-managed GGUF files and no silent downloads. An
install only happens behind a **visible consent prompt** showing the model, quant,
size, host, and license.

### Depth, not just Whisper

**Ten model families** run through one binary and one ggml runtime: Whisper,
Cohere Transcribe, Qwen3-ASR, Parakeet-CTC, Parakeet-TDT (25 European
languages), wav2vec2-CTC, Moonshine, Dolphin (Chinese dialects), SenseVoice
(zh/yue/en/ja/ko), and X-ASR (Zipformer). You get
frame-synchronous streaming partials, opt-in speaker diarization, word-level
timestamps with **per-word confidence**, and phrase-bias hotwords -- pick the
model that fits the task and keep one toolchain.

### Local and private, fail-closed

No telemetry, no phone-home, no background uploads; audio never leaves the machine.
The native runtime either produces a real transcript or returns a typed error --
it never fabricates output and never reaches for the network silently. `--offline`
and non-interactive runs fail closed before any download.

## What you can do

| You want to... | Command |
| --- | --- |
| Transcribe a file or a whole folder | `openasr transcribe audio.wav` |
| Live captions from mic or system audio | `openasr live` |
| Generate subtitles (SRT / VTT) | `openasr transcribe talk.wav -f srt -f vtt` |
| Word-level timestamps + per-word confidence | `openasr transcribe audio.wav -f json --word-timestamps` |
| Separate speakers (diarization) | `openasr transcribe meeting.wav --diarize` |
| Bias toward domain terms (hotwords) | `openasr transcribe call.wav --hotword "OpenASR"` |
| Serve an OpenAI-compatible API locally | `openasr serve` |

See `openasr --help` for the full command set and [QUICKSTART](docs/QUICKSTART.md)
for a guided first transcript.

## Models

Browse and install packs straight from the signed catalog; the `<id>:<quant>`
syntax selects a specific quantization tier:

```bash
openasr search                        # browse the catalog
openasr pull whisper-small            # install the recommended quant
openasr pull qwen3-asr-0.6b:q8        # or pin a specific quant tier
openasr list                          # what's installed
```

The catalog ships **17 ready-to-pull ASR packs** spanning seven of the ten
native families (50 signed quant downloads), plus diarization capability
packs; the remaining three (Parakeet-CTC, Parakeet-TDT, wav2vec2-CTC) run via
`import` of your own checkpoints. Ten native families run offline on CPU and Apple Metal,
dispatched by the data-driven architecture registry. All families support opt-in diarization; most also export word-level
timestamps -- the columns below show where they differ.

| Family | Streaming | Word timestamps | Quant tiers |
| --- | --- | --- | --- |
| Whisper (multilingual + English-only) | declared-pack | acoustic | fp16 / q8_0 / q4_k |
| Cohere Transcribe | declared-pack | approximate | fp16 / q8_0 / q4_k |
| Qwen3-ASR (default) | declared-pack | approximate | fp16 / q8_0 / q4_k / q3_k |
| Parakeet-CTC | declared-pack | acoustic | fp16 / q8_0 / q4_k |
| Parakeet-TDT (25 European languages) | declared-pack | acoustic | fp16 / q8_0 / q4_k |
| wav2vec2-CTC (incl. data2vec) | declared-pack | acoustic | fp16 / q8_0 / q4_k |
| Moonshine | declared-pack | approximate | fp16 / q8_0 / q4_k |
| Dolphin (Chinese dialects) | none | none | fp16 / q8_0 / q4_k |
| SenseVoice (zh/yue/en/ja/ko) | declared-pack | none | fp16 / q8_0 / q4_k |
| X-ASR (Zipformer, RNN-T) | declared-pack | acoustic | fp16 / q8_0 / q4_k |

- **Streaming** -- native frame-synchronous streaming emits incremental partials
  for packs that declare the streaming feature; other packs fall back to
  final-per-utterance output. Dolphin has no streaming executor yet, so it
  always runs final-per-utterance.
- **Word timestamps** -- *acoustic* means real acoustic frame alignment
  (Whisper decoder cross-attention; frame spans for the CTC/transducer families);
  *approximate* means decoder token-position estimates. Both export to JSON/VTT.
  Dolphin does not emit word-level timing at all (segment-level only).
- **Diarization** -- `--diarize` attributes anonymous `SPEAKER_NN` labels onto
  any family's transcript via pure-Rust WeSpeaker + pyannote capability packs;
  Cohere packs can additionally emit inline speaker tokens.

Multilingual coverage is per pack (a multilingual Whisper pack spans ~100
languages; Qwen3-ASR ~29; Parakeet-TDT 25 European languages; others are
English-only or bilingual). See the
per-model cards under [`model-registry/models/`](model-registry/models/) and
[Known Limitations](docs/KNOWN_LIMITATIONS.md) for the exact scope of each
capability.

## Benchmarks

Real numbers from the committed performance baseline
([`perf/baselines/macos-aarch64.json`](perf/baselines/macos-aarch64.json)) on a
macOS aarch64 CPU lane over a fixed 6.13 s LibriSpeech clip. RTF is
compute-time / audio-time (lower is faster; 0.10 = ~10x faster than real time);
WER is on the fixed harness clip.

| Model pack | Family | Quant | RTF | WER |
| --- | --- | --- | --- | --- |
| `whisper-tiny.en` | Whisper | q8_0 | 0.04 | 5.9% |
| `whisper-small.en` | Whisper | q8_0 | 0.13 | 0.0% |
| `whisper-large-v3-turbo` | Whisper | q8_0 | 0.39 | 0.0% |
| `cohere-transcribe` | Cohere | q8_0 | 0.11 | 0.0% |
| `qwen3-asr-0.6b` | Qwen3-ASR | q8_0 | 0.41 | 0.0% |
| `parakeet-ctc-0.6b`\* | Parakeet-CTC | q8_0 | 0.06 | 0.0% |
| `wav2vec2-base-960h`\* | wav2vec2-CTC | q4_k | 0.05 | 0.0% |
| `moonshine-base`\* | Moonshine | q4_k | 0.06 | 5.9% |

\* Import-only: not in the signed catalog. These packs are built locally with
`openasr model-pack import` from source weights; the unstarred rows install with
`openasr pull`.

The suite (`cargo run --release -p openasr-cli -- bench-suite`) drives the real
`transcribe` call path and gates against the committed baseline. It reads
host-local packs from the paths in [`perf/suite.toml`](perf/suite.toml)
(gitignored `tmp/`, every entry optional), so on a fresh clone it skips entries
until you install or build the packs -- see
[Performance](perf/PERFORMANCE.md) for setup, gates, and caveats. WER on a
17-word clip is coarse (one word is ~5.9%).

## Building from source

Build once (the ggml backend compiles from source, so clone recursively and have
`cmake`, a C/C++ toolchain, and on Linux `libasound2-dev`; Rust 1.95.0 is pinned
by `rust-toolchain.toml`; expect the first build to take several minutes while
ggml compiles):

```bash
git clone --recurse-submodules https://github.com/QuintinShaw/openasr.git && cd openasr
cargo build --release -p openasr-cli      # binary at target/release/openasr
```

Then, with `openasr` on your PATH:

```bash
# Transcribe a file. The first run offers to download the default model
# (qwen3-asr-0.6b) with a visible confirmation, then runs offline.
openasr transcribe audio.wav

# Choose a model, a format, write to a file (-m/-f/-o; `t` aliases transcribe).
openasr t audio.wav -m whisper-small -f srt -o audio.srt

# A whole folder (one transcript per file), or several formats at once.
openasr transcribe ./recordings -o ./transcripts
openasr transcribe audio.wav -f srt -f vtt -f json
```

During development run without installing via
`cargo run -p openasr-cli -- <args>`; `--backend mock` gives deterministic,
network-free output for CI.

## Execution posture

- **`native` is the default backend.** It runs local ggml-backed `.oasr` model
  packs and is fail-closed by staged runtime boundaries.
- **No silent downloads.** For `transcribe`/`live`, the CLI installs a missing
  model only through a **visible consent prompt** showing the model, quant, size,
  host, and license; `--offline` (or any non-interactive run) fails closed before
  touching the network. The **HTTP server never downloads to satisfy a request**
  -- transcription runs only an explicit local pack, and the only server-side
  install path is the operator-authenticated pull API.
- **`mock` is an opt-in stub** (`--backend mock`) that emits deterministic
  placeholder text for plumbing and CI. It downloads nothing and needs no weights.
- **`.oasr` is the only user-facing pack format** (GGUF-backed internally); bare
  `.gguf` is not accepted as run input or importer output.
- **No telemetry.** OpenASR does not phone home.

## CLI overview

| Command | Purpose |
| --- | --- |
| `transcribe <inputs>...` | Transcribe files or directories. `--benchmark` prints run timing instead of the transcript; aliased as `t`. |
| `live` | Microphone / system-audio capture; emits frame-synchronous streaming partials for packs that declare streaming, else final-per-utterance. |
| `serve` | Local OpenAI-compatible HTTP API; secured remote serving via TLS + pairing. |
| `search [query]` / `pull <id>` | Browse the model catalog / download and install a pack. |
| `list` / `show <id-or-pack>` / `rm <id>` | List installed packs / show catalog or pack details / remove a pack. |
| `verify <pack.oasr>` | Probe a local pack's ggml integrity (no inference, no download). |
| `speaker enroll/clear` | Manage local voice-match profiles for diarization display names (embeddings only; not authentication). |
| `model-pack import <family>` | Build a local `.oasr` pack from source weights (maintainer tool). |
| `config` / `doctor` / `bench-suite` | Edit config / print diagnostics / run the performance suite. |

Useful flags on `transcribe`: `-m/--model` (also `OPENASR_MODEL`), `-f/--format`
(`text`, `json`, `srt`, `vtt`, `verbose_json`, `markdown`), `-o/--output`,
`-l/--language` (`auto` or a hint like `en`), `--diarize`, `--word-timestamps`,
`--continue-on-error` (multi-input), and `--hotword <PHRASE>` / `--hotword-boost`
for phrase bias.

## Local HTTP API

```bash
cargo run -p openasr-cli -- serve --addr 127.0.0.1:8080 --backend native --model-pack /path/to/model.oasr --model your-runtime-model-id

curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=your-runtime-model-id \
  -F response_format=verbose_json
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/audio/transcriptions`.
Response formats: `json`, `text`, `srt`, `vtt`, `verbose_json`, `markdown`.

Key constraints:

- `serve --backend native` runs a local `.oasr`: pass `--model-pack <local.oasr>`,
  or omit it to use an already-installed pack resolved by `--model` id (a missing
  pack fails closed -- transcription never triggers a download). A supplied
  `--model` must match the pack's runtime model id.
- `stream=true` is SSE and rejects `response_format=srt|vtt`; use the
  non-streaming endpoint for subtitle files.
- Longform fields (`segment_mode`, `chunk_seconds`, `segment_overlap_seconds`,
  `vad_*`, `min_segment_seconds`, `suppress_silent_slices`) are native-only;
  default is `segment_mode=auto`.
- Phrase-bias / hotword fields are request-validated; unsupported backends return
  an explicit error rather than ignoring them.
- `serve` runs a single model (the launched pack); restart to switch. Transcription
  history is local-only and governed by the `history_retention` preference
  (default keeps the last 5 entries; `off` disables it). See
  [SECURITY](SECURITY.md) for what's stored.

Non-loopback binds are rejected unless launched with HTTPS/WSS and pairing auth:

```bash
export OPENASR_PAIRING_ADMIN_TOKEN="$(openssl rand -hex 32)"
cargo run -p openasr-cli -- serve --addr 0.0.0.0:8443 --tls-self-signed \
  --pairing-admin-token-env OPENASR_PAIRING_ADMIN_TOKEN \
  --backend native --model-pack /path/to/model.oasr --model your-runtime-model-id
```

## Building model packs

Native packs are built from a local HF-style source directory with one per-family
importer. `openasr pull` installs already-published catalog packs; local importing
stays the path for caller-provided source weights and vendor-gated sources that
must not be silently re-hosted.

```bash
cargo run -p openasr-cli -- model-pack import whisper      <source_dir> <out.oasr> --package-id whisper-small --source-revision <rev>
cargo run -p openasr-cli -- model-pack import qwen         <source_dir> <out.oasr> --package-id qwen3-asr-0.6b --source-revision <rev>
cargo run -p openasr-cli -- model-pack import moonshine    <source_dir> <out.oasr> --package-id moonshine-tiny --source-revision <rev>
```

Other families: `cohere`, `parakeet-ctc`, `parakeet-tdt`, `wav2vec2-ctc`, `dolphin`,
`xasr-zipformer`, `hymt2-gguf`, `wespeaker`, `pyannote`. Each accepts
`--quantization fp16|q8_0|q4_k` (Qwen also exposes `q3_k`).

## Validation

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo nextest run --workspace        # or: cargo test
cargo test --workspace --doc
```

Tests default to the CPU ggml backend and pass `--backend mock` where they need
deterministic, weight-free output. On x86/x86_64 host builds, OpenASR enables
ggml's `GGML_NATIVE` CPU tuning by default; set `OPENASR_GGML_NATIVE=0` for
portable distribution builds.

## Documentation

- [Docs Index](docs/DOCS_INDEX.md) - map of all documentation
- [Architecture](ARCHITECTURE.md) - crate map and the transcription pipeline
- [Quickstart](docs/QUICKSTART.md) - three commands to a real transcript
- [Roadmap](docs/ROADMAP.md) - what runs today and what is deferred
- [Known Limitations](docs/KNOWN_LIMITATIONS.md)
- [Model Catalog, Registry, and Distribution](docs/MODEL_CATALOG_ARCHITECTURE.md)
- [Format Contract](docs/format/OASR_PACKAGE_CONTRACT_V1.md)
- [Releasing](RELEASING.md) - versioning and release process
- [Performance](perf/PERFORMANCE.md)
- [Contributing](CONTRIBUTING.md) - build setup, branch naming, PR checklist, DCO

## License and acknowledgments

OpenASR -- the `openasr-core`, `openasr-cli`, `openasr-client`, `openasr-server`,
and `openasr-system-audio` crates -- is licensed under the
[Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for attribution.

The inference backend `crates/openasr-core/third_party/openasr-ggml` is a fork of
[ggml](https://github.com/ggml-org/ggml) under the MIT License. OpenASR is built
directly on it; we gratefully acknowledge Georgi Gerganov and the ggml /
llama.cpp / whisper.cpp communities. See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md)
for the projects and model authors OpenASR builds on.

Model packs are distributed separately under their own upstream licenses (all
permissive, MIT / Apache-2.0). A free, openly licensed pack -- e.g.
`whisper-small` (Apache-2.0) or `moonshine-tiny` (MIT) -- runs the engine
end-to-end.
</content>
</invoke>
