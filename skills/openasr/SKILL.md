---
name: openasr
description: Transcribe audio, dictate, and manage local speech-to-text models with the OpenASR CLI (`openasr`) -- local-first, no cloud calls, no telemetry. Use this whenever the user asks to transcribe an audio/video file, capture a microphone/system-audio dictation, list or install ASR models, or run a local OpenAI-compatible transcription API.
license: Apache-2.0
---

# OpenASR

OpenASR is a local-first speech-to-text CLI (`openasr`) and local HTTP server.
Everything below runs on-device: no network calls unless you explicitly
`pull` a model, no telemetry, no cloud fallback.

Prerequisite: the `openasr` binary must be on `PATH`. Check with `openasr
--version`; if missing, tell the user to install it (`cargo install --path
crates/openasr-cli` from a source checkout, or a released binary) rather than
guessing a path.

## Quick decision guide

- One-off file/batch transcription -> `openasr transcribe`.
- Live microphone or system-audio capture -> `openasr live`.
- "What models do I have / can I get" -> `openasr list` / `openasr search`.
- Need a long-lived local HTTP endpoint (e.g. another tool wants to POST
  audio) -> `openasr serve`, see "Local HTTP API" below.

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
  omit it to use the configured default. If the model is not installed yet,
  the CLI prompts to download it interactively -- pass `-y/--yes` to accept
  non-interactively, or `--offline` to fail closed instead of prompting/
  downloading (use `--offline` in any non-interactive/CI context).
- `--diarize` labels speakers (`SPEAKER_00`, ...); `--word-timestamps` asks
  for per-word timing where the model supports it.
- `--benchmark` prints timing (elapsed, audio duration, real-time factor)
  instead of the transcript, for a single input.
- Non-WAV input (mp3, mp4, ...) needs `ffmpeg` on `PATH`, or pass
  `--ffmpeg-bin <path>`.

Common failure modes:

- "model not installed" + a non-interactive shell: add `-y` (install) or
  `--offline` (explicitly skip and fail) depending on intent -- do not assume.
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
behavior mirrors `transcribe`'s model-resolution and consent-pull rules.

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

`openasr serve` exposes a local OpenAI-compatible HTTP API -- use this when a
tool needs to POST audio over HTTP instead of shelling out per file.

```bash
openasr serve --addr 127.0.0.1:8080 --backend native \
  --model-pack /path/to/model.oasr --model your-runtime-model-id
```

`--addr` defaults to `127.0.0.1:8080` (fixed, not random) so you can hardcode
the base URL. Loopback (`127.0.0.1`) requests are trusted by default -- no
`Authorization` header needed:

```bash
curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=your-runtime-model-id \
  -F response_format=verbose_json
```

Key endpoints: `GET /health` (liveness, unauthenticated), `GET /v1/models`
(installed packs), `POST /v1/audio/transcriptions` (the transcription call
above). `response_format` accepts `json`, `text`, `srt`, `vtt`,
`verbose_json`, `markdown`.

### Optional: requiring an API key even on loopback

If the deployment wants an explicit credential even for local callers, create
one first:

```bash
openasr apikey create --name "my-agent"
# prints the plaintext key exactly once -- save it, it cannot be re-shown
```

Once any key exists, every `serve` request (except `/health`) must carry
`Authorization: Bearer <key>`, including from `127.0.0.1`:

```bash
curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer oasr_sk_<...>" \
  -F file=@audio.wav -F model=your-runtime-model-id
```

Manage keys with `openasr apikey list` / `openasr apikey revoke <id>`.
Revoking the last key returns loopback `serve` to its key-free default.

An API key is a **loopback-only** convenience -- binding `serve` to a
non-loopback address for real remote/network access always requires
HTTPS/WSS plus device pairing (`--tls-self-signed
--pairing-admin-token-env ...`), regardless of any configured key.

Full reference (longform/segment fields, phrase-bias/hotword fields,
streaming, pairing flow): see `docs/AGENT_INTEGRATION.md` and the "Local HTTP
API" section of the repo README at
https://github.com/QuintinShaw/openasr.

## Guardrails

- Never suggest or attempt to send audio to a cloud service; OpenASR is
  local-only by design.
- Do not fabricate a transcript or model id -- if a command fails, surface
  the actual error and, if it is a missing-model/consent case, ask the user
  how to proceed rather than guessing `-y` vs `--offline`.
- `.oasr` is the only accepted user-facing pack format.
