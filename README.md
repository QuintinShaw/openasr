[English](README.md) | [简体中文](README.zh-CN.md)

<div align="center">

# OpenASR

**Turn speech into text, entirely on your device.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![CI](https://github.com/QuintinShaw/openasr/actions/workflows/ci.yml/badge.svg)](https://github.com/QuintinShaw/openasr/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/QuintinShaw/openasr)](https://github.com/QuintinShaw/openasr/releases)
[![Downloads](https://img.shields.io/github/downloads/QuintinShaw/openasr/total)](https://github.com/QuintinShaw/openasr/releases)

[Website](https://openasr.org) · [Documentation](docs/DOCS_INDEX.md) · [License](LICENSE)

<img src="https://openasr.org/assets/openasr-desktop-preview-en.gif" alt="OpenASR Desktop App" width="720" />

<sub>Pre-v1 — under active development. CLI flags, API surface, and pack format may change between 0.x releases.</sub>

</div>

---

<div align="center">
<h3><a href="https://openasr.org/download/">Download the Desktop App</a></h3>
<p><strong>macOS</strong> (Apple Silicon) · <strong>Windows</strong> (x64, Windows 10+) · Linux desktop coming soon</p>
</div>

No terminal needed. Install the app, drop in an audio file, and get your transcript — everything runs on your machine.

> This repository is the Apache-2.0 open core behind the desktop app: a Rust CLI, a local OpenAI-compatible HTTP API, and the ggml inference engine. The desktop app wraps the same engine in a native GUI — no hidden network calls.

---

## What it does

- **Transcribe audio files** — single files or entire folders, output as plain text, SRT/VTT subtitles, or JSON with word-level timestamps
- **Live captions** — real-time transcription from your microphone with streaming partial results
- **System audio capture** — caption meetings, lectures, and podcasts by recording what your computer plays
- **Speaker separation** — automatically label who said what
- **Translation** — transcribe and translate to English in one step
- **Local API** — OpenAI-compatible `/v1/audio/transcriptions` endpoint, works with existing SDKs

## Why OpenASR

**Private.** In the default local mode, audio stays on your machine. Remote compute is available only when you explicitly pair and enable it; see [SECURITY.md](SECURITY.md#local-first-security-notes). No telemetry, no silent uploads, and no silent network fallback. The engine either produces a real transcript or tells you why it can't.

**Broad.** 28 models across 13 families — Whisper, Qwen3-ASR, Parakeet, SenseVoice, FireRed, Dolphin, Moonshine, and more. Pick the one that fits your language and workload. All run through one binary on CPU and Apple Metal.

**Open.** The engine is Apache-2.0. Each model pack ships under its own upstream license as recorded in the registry and pack metadata. Every model download is verified against a signed catalog before it runs.

---

## For developers

### CLI quickstart

```bash
# Option A: Homebrew (macOS / Linux)
brew install quintinshaw/tap/openasr

# Option B: one-line installer (macOS / Linux)
curl -fsSL https://dl.openasr.org/install.sh | sh

# Option C: grab a prebuilt binary from Releases
# https://github.com/QuintinShaw/openasr/releases

# Transcribe a file (first run offers to download a model — you confirm first)
openasr transcribe recording.wav

# Live mic captions
openasr live

# SRT subtitles with speaker labels
openasr transcribe meeting.wav -f srt --diarize
```

See [Quickstart](docs/QUICKSTART.md) for a guided walkthrough, or run `openasr --help`.

### Local API

```bash
openasr serve

curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@audio.wav -F model=qwen3-asr-0.6b
```

Drop-in compatible with OpenAI SDKs (`base_url="http://127.0.0.1:8080/v1"`). See [Agent Integration](docs/AGENT_INTEGRATION.md) for API key setup and agent workflows.

### Building from source

```bash
git clone --recurse-submodules https://github.com/QuintinShaw/openasr.git
cd openasr
cargo build --release -p openasr-cli
```

Requires Rust (pinned via `rust-toolchain.toml`), CMake, and a C/C++ toolchain. Full build setup and development workflow in [CONTRIBUTING.md](CONTRIBUTING.md).

## Models

28 models across 13 families, from tiny English-only models that run faster than real-time to large multilingual models covering 100+ languages. Browse them at [openasr.org/models](https://openasr.org/models/) or from the CLI:

```bash
openasr search            # browse available models
openasr pull whisper-small  # install one
```

Benchmarks from the committed performance baseline are in [Performance](perf/PERFORMANCE.md).

## Documentation

| | |
|---|---|
| [Docs Index](docs/DOCS_INDEX.md) | Full documentation map |
| [Quickstart](docs/QUICKSTART.md) | First transcript in three commands |
| [FAQ](docs/FAQ.md) | Common questions answered |
| [Known Limitations](docs/KNOWN_LIMITATIONS.md) | What works and what does not yet |
| [Roadmap](docs/ROADMAP.md) | What is planned next |
| [Architecture](ARCHITECTURE.md) | Crate map and transcription pipeline |

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for build setup, branch naming, the PR checklist, and DCO sign-off.

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

The ggml inference backend is MIT-licensed. Each model pack's license is defined by its registry entry and pack metadata; packs may use Apache-2.0, MIT, CC-BY, FunASR, or other upstream terms. This is not an exhaustive license guarantee. See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for the projects and model authors OpenASR builds on.
