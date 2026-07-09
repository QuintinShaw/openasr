---
name: openasr
description: Transcribes audio and video, captures live dictation, and manages local speech-to-text models with the OpenASR CLI (`openasr`) and its local OpenAI-compatible HTTP API. Use when the user asks to transcribe an audio/video file or recording, generate subtitles (SRT/VTT), capture microphone or system-audio dictation, list/install/remove ASR models, or point an OpenAI SDK client at a local transcription endpoint. Local-first -- no cloud calls, no telemetry, no silent downloads.
license: Apache-2.0
allowed-tools: Bash(openasr *)
---

# OpenASR

OpenASR is a local-first speech-to-text CLI (`openasr`) and local HTTP server.
Everything runs on-device: no network calls unless you explicitly `pull` a
model, no telemetry, no cloud fallback.

Prerequisite: the `openasr` binary must be on `PATH`. Check with
`openasr --version`; if missing, tell the user to install it (`cargo install
--path crates/openasr-cli` from a source checkout, or a released binary)
rather than guessing a path.

## Quick decision guide

- One-off file/batch transcription -> `openasr transcribe`.
- Live microphone or system-audio capture -> `openasr live`.
- "What models do I have / can I get" -> `openasr list` / `openasr search`.
- Another tool needs to POST audio over HTTP (or an OpenAI SDK client should
  hit a local endpoint) -> `openasr serve`, see "Local HTTP API" below.

## Transcribing files

```bash
openasr transcribe audio.wav --model whisper-small --format json
```

- Multiple inputs or a directory: pass several paths or a directory; use
  `-o/--output <dir>` to write one file per input instead of stdout.
- `--format` (short `-f`) accepts `text`, `json`, `srt`, `vtt`,
  `verbose_json`, `markdown`; repeat `-f` to write several formats at once as
  sidecar files.
- `--model <id>` selects a model id from the registry (see `openasr search`);
  omit it to use the configured default. If the model is not installed, the
  CLI prompts to download it interactively -- pass `-y/--yes` to accept
  non-interactively, or `--offline` to fail closed instead (use `--offline`
  in any non-interactive/CI context).
- `--diarize` labels speakers (`SPEAKER_00`, ...); `--word-timestamps` asks
  for per-word timing where the model supports it.
- `--benchmark` prints timing (elapsed, audio duration, real-time factor)
  instead of the transcript, for a single input.
- Non-WAV input (mp3, mp4, ...) needs `ffmpeg` on `PATH`, or pass
  `--ffmpeg-bin <path>`.

Common failure modes:

- "model not installed" in a non-interactive shell: add `-y` (install) or
  `--offline` (fail closed) depending on the user's intent -- do not assume.
- Unsupported audio container without `ffmpeg`: install `ffmpeg` or convert
  first (`ffmpeg -i in.mp4 -ac 1 -ar 16000 -c:a pcm_s16le out.wav`).
- A partially-supported model/format combination returns an explicit error
  naming the unsupported field rather than silently ignoring it -- surface
  that error text to the user, do not retry blindly.

## Live dictation / capture

```bash
openasr live --model whisper-small
```

Streams from the default input device and prints frame-synchronous partial
results for packs that declare streaming support, otherwise final text per
utterance. Run `openasr live --help` for device selection and output flags;
model resolution and consent-pull rules mirror `transcribe`.

## Managing models

```bash
openasr search              # browse the full catalog
openasr search whisper      # filter by name/family
openasr list                # installed packs only
openasr show <id>           # catalog card or a local .oasr pack's details
openasr pull <id>[:<quant>] # download + install (e.g. `pull moonshine-tiny:q8`)
openasr rm <id>             # remove an installed pack
```

Models are distributed as `.oasr` packs (GGUF-backed internally); bare
`.gguf` files are not accepted directly as run input.

## Local HTTP API

`openasr serve` exposes a local OpenAI-compatible HTTP API subset. Use it
when a tool wants a long-lived endpoint instead of a process per file.

```bash
openasr serve   # binds 127.0.0.1:8080, serves an installed local .oasr pack

curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=<installed-model-id> \
  -F response_format=verbose_json
```

- `--addr` defaults to `127.0.0.1:8080` (fixed, not random) so the base URL
  can be hardcoded. Loopback callers are trusted by default (no auth header).
- The server never downloads models: a request for a model other than the
  loaded pack fails closed with an explicit error. Install models with
  `openasr pull` first, restart `serve` to switch models.
- OpenAI SDK clients work out of the box for non-streaming calls
  (`base_url="http://127.0.0.1:8080/v1"`, any placeholder `api_key`). SDK
  `stream=True` is rejected with an explicit error -- SSE streaming uses an
  OpenASR-specific protocol (`?stream=true` query parameter), not OpenAI
  `transcript.text.*` events.

For the full endpoint list, the OpenAI parameter compatibility matrix, SDK
examples, API keys, and streaming details, read
[references/http-api.md](references/http-api.md).

## Guardrails

- Never suggest or attempt to send audio to a cloud service; OpenASR is
  local-only by design.
- Do not fabricate a transcript or model id -- if a command fails, surface
  the actual error, and for missing-model/consent cases ask the user how to
  proceed rather than guessing `-y` vs `--offline`.
- `.oasr` is the only accepted user-facing pack format.
