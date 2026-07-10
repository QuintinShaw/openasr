# Agent Integration

How to point a coding agent (Claude Code, Codex, or any other agent that can
run a shell / call a local HTTP API) at OpenASR. Two independent paths:

1. **Skill + CLI** -- the agent shells out to the `openasr` binary directly.
   This is the primary, recommended path: no server process, no network
   surface, works offline.
2. **Local OpenAI-compatible HTTP API** -- for agents/tools that only speak
   HTTP (or want a long-lived local daemon instead of spawning a process per
   request).

Both paths run entirely on-device: no telemetry, no cloud fallback, no
implicit model download (see [SECURITY](../SECURITY.md) and the repo-root
"Product principles").

## 1. Skill (CLI) path

The published Skill lives at [`skills/openasr/SKILL.md`](../skills/openasr/SKILL.md)
and follows the [`vercel-labs/skills`](https://github.com/vercel-labs/skills)
convention: any file host on GitHub is a valid registry entry, no separate
publishing step. Install it into a project with:

```bash
npx skills add QuintinShaw/openasr --skill openasr --agent claude-code
# or interactively: npx skills add QuintinShaw/openasr
```

This copies the whole skill directory -- `SKILL.md` plus
`references/http-api.md` -- into the agent's local skills directory (for
example `.claude/skills/openasr/` for Claude Code). Verified locally with
`npx skills add <path-to-checkout> --skill openasr --agent claude-code`,
which installs both files under `./.claude/skills/openasr/`. The Skill
teaches the agent the `openasr` CLI surface: `transcribe`, `live`,
`search`/`pull`/`list`, `serve`, and `apikey`, including expected output
shapes and common failure modes (missing model, `--offline` fail-closed,
non-WAV without `ffmpeg`); the reference file carries the full HTTP API
parameter matrix so it only enters the agent's context when needed
(progressive disclosure).

Prerequisite: the `openasr` binary must be on `PATH` (`cargo install
--path crates/openasr-cli` from a source checkout, or a released binary).
The Skill itself does not install OpenASR.

## 2. Local HTTP API path

```bash
openasr serve --addr 127.0.0.1:8080 --backend native \
  --model-pack /path/to/model.oasr --model your-runtime-model-id
```

`--addr` defaults to `127.0.0.1:8080` -- a fixed, predictable port on purpose,
so an agent (or a script) can hardcode the base URL instead of discovering it.
This is independent of the desktop app's sidecar, which negotiates its own
port via a local handshake file and is unaffected by this default.

### Loopback trust and API keys

Loopback (`127.0.0.1`) callers are trusted by default: no `Authorization`
header is required. That is deliberate for the common case (an agent running
on the same machine). If you want an explicit credential anyway -- for
example a shared dev box, or just to be strict about who can call the daemon
-- create a key:

```bash
openasr apikey create --name "claude-code-laptop"
# Created API key key_1a2b3c4d5e6f7a8b (claude-code-laptop).
# This is the ONLY time the full key is shown -- store it now:
#
#   oasr_sk_<64 hex chars>
#
# Once any key exists, `openasr serve` requires it (Authorization: Bearer <key>)
# even on 127.0.0.1. Run `openasr apikey revoke key_1a2b3c4d5e6f7a8b` to remove it.
```

Only a SHA-256 hash of the key is ever written to disk
(`~/.openasr/apikeys.json`, permissions `0600`); the plaintext is shown
exactly once, at creation, and cannot be recovered afterward. From that point
on, every loopback request to `serve` -- including `/v1/audio/transcriptions`
-- must carry `Authorization: Bearer <key>`, or it gets `401`. `GET /health`
stays unauthenticated (used for liveness probes).

```bash
curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer oasr_sk_<...>" \
  -F file=@audio.wav \
  -F model=your-runtime-model-id \
  -F response_format=verbose_json
```

Manage keys with:

```bash
openasr apikey list      # id, name, created-at, key preview (never the full key)
openasr apikey revoke key_1a2b3c4d5e6f7a8b
```

Revoking the last remaining key returns loopback `serve` to its key-free
default.

API keys apply to **manually-launched** `openasr serve` only. The desktop
app's supervisor-managed daemon (identified by the instance-token environment
variable the supervisor sets when spawning it) is exempt: its UI talks to it
over loopback without bearer headers, and its remote access is secured by
TLS + pairing instead -- so creating a key never locks the desktop app out
of its own daemon.

### What API keys do not change

A locally-created API key is a **loopback-only** convenience. Binding `serve`
to a non-loopback address (remote/network access) still requires HTTPS/WSS
and device pairing exactly as before (`--tls-self-signed
--pairing-admin-token-env ...`); an API key never substitutes for that, and
non-loopback binds ignore any configured keys entirely. See the "Non-loopback
binds" section of the root [README](../README.md#local-http-api) for the
remote-serving flow.

### Endpoints an agent typically needs

- `GET /health` -- liveness + server identity, unauthenticated. Includes
  `model_installed` (a pack is bound) and `model_resident` (0.1.13+: that
  pack's runtime is currently loaded in memory, vs. idle-unloaded or not yet
  loaded this boot).
- `GET /v1/models` -- installed packs.
- `POST /v1/audio/transcriptions` -- OpenAI-compatible transcription
  (multipart `file` + `model`; `response_format` of `json`, `text`, `srt`,
  `vtt`, `verbose_json`, or `markdown`).
- `POST /v1/audio/translations` -- OpenAI-compatible X->English translation
  (non-streaming; families without a translate task reject it explicitly).

### OpenAI SDK compatibility

Official OpenAI SDKs work against `serve` out of the box for non-streaming
calls (`base_url="http://127.0.0.1:8080/v1"`, any placeholder `api_key`,
or a real key once one exists). Errors use the OpenAI envelope
(`error.{message,type,param,code}`). `verbose_json` carries `language`,
`duration`, segment `id`s, and a top-level `words` array when word
timestamps are requested; it does not fabricate `task`, `usage`, or decoder
internals (`avg_logprob`, `no_speech_prob`, ...). OpenAI parameters with no
local equivalent (`temperature`, `include[]`, `chunking_strategy`,
`known_speaker_*`) are accepted and ignored. SDK `stream=True` (the `stream`
form field) is rejected with an actionable 400: SSE streaming is the
OpenASR realtime protocol behind the `?stream=true` query parameter, not
OpenAI `transcript.text.*` events.

The full parameter-by-parameter matrix lives in the Skill's
[`references/http-api.md`](../skills/openasr/references/http-api.md); the
root [README](../README.md#local-http-api) documents the request-field
reference (longform/segment options, phrase-bias/hotword fields,
streaming).
