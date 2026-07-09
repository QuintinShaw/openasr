# OpenASR local HTTP API reference

Verified against `openasr serve` (core 0.1.11). Everything here is local:
the server never downloads models, never phones home, and fails closed on
anything it cannot execute.

## Contents

- Starting the server
- Endpoints
- OpenAI parameter compatibility matrix
- Response shapes
- Using the OpenAI SDK
- Errors
- Streaming (OpenASR SSE, not OpenAI events)
- API keys and remote binds
- OpenASR extension fields

## Starting the server

```bash
openasr serve                          # 127.0.0.1:8080, installed default pack
openasr serve --model <installed-id>   # pick one installed pack by id
openasr serve --model-pack /path/to/model.oasr   # explicit local pack file
```

One model per server process; restart to switch. A fresh install with no
models still starts (health works) but transcription requests fail closed
until a pack is installed (`openasr pull <id>`). `--model` accepts bare or
quant-pinned ids (`whisper-tiny`, `whisper-tiny:q8`); it must name the same
pack that is being served.

## Endpoints

- `GET /health` -- liveness + server identity; never requires auth.
- `GET /v1/models` -- OpenAI-style list of the served pack
  (`{"object":"list","data":[{"id","object":"model","owned_by":"openasr"}]}`;
  no `created` field).
- `POST /v1/audio/transcriptions` -- OpenAI-compatible transcription
  (multipart form).
- `POST /v1/audio/translations` -- OpenAI-compatible X->English speech
  translation (non-streaming; model families without a translate task reject
  it explicitly).
- `POST /v1/audio/transcriptions/{id}/pause|resume|cancel` -- OpenASR
  extension: control an in-flight request that supplied a `transcription_id`
  form field.

## OpenAI parameter compatibility matrix

Behavior of each OpenAI `audio/transcriptions` request parameter:

| OpenAI parameter | OpenASR behavior |
| --- | --- |
| `file` | Supported (multipart). Non-WAV containers need `ffmpeg` on the server's `PATH`. |
| `model` | Required. Must name the loaded pack (quant-tag tolerant, e.g. `whisper-tiny:q8` matches a bare `whisper-tiny` pack). Anything else is a 400 -- the server never downloads. |
| `language` | Supported as a language hint where the model family supports one. |
| `prompt` | Forwarded to the model; families without prompt support reject it with an explicit error instead of ignoring it. |
| `response_format` | `json` (default), `text`, `srt`, `vtt`, `verbose_json` supported; `markdown` is an OpenASR extension. `diarized_json` is not supported (use the `diarize=true` extension field and read per-segment `speaker`). |
| `timestamp_granularities[]` | `segment` (always present) and `word` supported; `word_aligned` is an OpenASR extension that requires the forced-aligner capability pack and fails closed when it is missing. Unlike OpenAI, `verbose_json` is not strictly required. |
| `stream` (form field) | Rejected with a 400 and an actionable message. OpenAI SDK `stream=True` cannot work because the SSE protocol is OpenASR's, not `transcript.text.*` events -- see "Streaming" below. |
| `temperature` | Accepted and ignored (decoding is deterministic greedy). |
| `include[]` (`logprobs`) | Accepted and ignored; no logprobs in responses. |
| `chunking_strategy` | Accepted and ignored; long-form segmentation is automatic and tunable via the OpenASR `segment_mode`/`vad_*` extension fields. |
| `known_speaker_names[]` / `known_speaker_references[]` | Accepted and ignored; use `diarize=true` (+ optional `speakers=N`) instead. |

## Response shapes

- `json`: `{"text", "segments":[{"start","end","text",...}]}` -- OpenAI's
  `json` plus a `segments` extension.
- `verbose_json`: `{"language" (English name, when reported), "duration"
  (last segment end, seconds), "text", "segments":[{"id","start","end",
  "text", "words":[...] when word timing was requested, "speaker"/
  "speaker_label"/"speaker_profile_id" when diarizing}], "words":[flattened
  per-word timing] when word timing was requested}`.
  Not produced: `task`, `usage`, and per-segment decoder internals
  (`seek`, `tokens`, `avg_logprob`, `compression_ratio`, `no_speech_prob`).
- `text`, `srt`, `vtt`: plain bodies, same as OpenAI. `markdown` (extension)
  renders a `# Transcript` document.

## Using the OpenAI SDK

Non-streaming calls work out of the box:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="<installed-model-id>",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )
print(result.text)
```

`api_key` can be any placeholder unless an API key was created (below). Do
not pass `stream=True` -- the server rejects it with a 400 explaining why.

## Errors

OpenAI-style envelope with every key clients expect:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": null}}
```

Statuses: 400 (invalid/unsupported/fail-closed refusals, including
model-not-loaded), 401 (missing/bad API key when keys exist), 404, 409
(canceled), 413 (upload too large), 429/503 (busy), 507 (disk full).
Messages are self-describing; surface them verbatim.

## Streaming (OpenASR SSE, not OpenAI events)

`POST /v1/audio/transcriptions?stream=true` (query parameter, not form
field) returns `text/event-stream` with OpenASR realtime events (observed
one-shot sequence): `session.created`, `session.configured`,
`audio.input.started`, `segment_start`, `transcript.final`, `final`,
`segment_end`, `audio.input.stopped`, `session.closed`, `done` (or an
`error` event on failure). It does not
emit OpenAI `transcript.text.delta`/`transcript.text.done`, so OpenAI SDK
streaming clients cannot parse it; consume it with an SSE library instead.
`response_format=srt|vtt` is rejected for streaming. Live microphone/system
audio uses a WebSocket session instead of this one-shot SSE path.

## API keys and remote binds

Loopback needs no credential by default. To require one anyway:

```bash
openasr apikey create --name "my-agent"   # plaintext shown exactly once
```

Once any key exists, every request except `GET /health` needs
`Authorization: Bearer oasr_sk_...` (the OpenAI SDK sends this automatically
when the key is passed as `api_key`). `openasr apikey list` / `revoke`
manage keys; revoking the last key returns to the key-free default.

API keys are a loopback-only convenience: binding to a non-loopback address
always requires TLS + device pairing (`--tls-self-signed
--pairing-admin-token-env ...`); see the repo README's "Local HTTP API"
section.

## OpenASR extension fields

Multipart form fields beyond the OpenAI set (all optional, all validated --
unsupported combinations fail closed with explicit errors):

- `task` (`transcribe`|`translate`), `diarize`, `speakers`, `punctuate`
- `hotword`/`phrase_bias` (repeatable) + `hotword_boost`/`phrase_bias_boost`
- Long-form segmentation: `segment_mode` (`off|auto|fixed|energy|vad`),
  `chunk_seconds`, `segment_overlap_seconds`, `vad_threshold_db`,
  `vad_min_silence_ms`, `vad_padding_ms`, `min_segment_seconds`,
  `suppress_silent_slices`
- Runtime: `inference_threads`, `execution_target` (`auto|cpu|accelerated`)
- Control: `transcription_id` (enables pause/resume/cancel endpoints)
