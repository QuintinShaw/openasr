# Security Policy

## Supported Versions and Current Status

OpenASR is pre-v1. There are no long-term supported release branches yet; security fixes land on `main`. Please report suspected vulnerabilities privately, before public disclosure (see below).

## Reporting a Vulnerability

Please report suspected vulnerabilities privately, before any public disclosure, through
GitHub's private vulnerability reporting — the **Report a vulnerability** button on the
repository's **Security** tab ([how-to](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)).
We aim to acknowledge reports within a few days.

Do not include private audio files, credentials, model files, secrets, or sensitive local paths in public reports.

## What to Include

Please include:

- OpenASR version or commit SHA.
- Operating system and architecture.
- Rust version if built from source.
- Exact command or API request shape.
- Whether `OPENASR_HOME` was customized.
- Whether `mock` or `native` backend was selected.
- Expected behavior and actual behavior.
- Relevant logs with secrets removed.

## Security Expectations

- Do not commit model weights, runtime binaries, large generated artifacts, private audio, credentials, API keys, or secrets.
- Do not add fake or unverified model/runtime/release URL/hash metadata.
- Keep examples and demos local-first by default.
- Keep default tests network-free and independent of private artifacts.

## Local-First Security Notes

OpenASR is local-first by default.

Current active behavior:

- `native` is the default backend with a fail-closed planning/runtime contract;
  `mock` is an opt-in deterministic stub for plumbing/CI;
- no silent downloads: a missing model is installed only by an explicit
  `openasr pull`, or by an interactive `transcribe`/`live` consent prompt (showing
  model, quant, size, host, and license) that fails closed when non-interactive or
  `--offline`/`--no-pull`; the HTTP server never downloads;
- no telemetry by default;
- no cloud transcription by default.

Bundled model cards are metadata-only planning/provenance records and are non-downloadable in active behavior.

### Catalog metadata transport

When a flow does fetch the model catalog (an explicit `pull`, or a
consent-approved auto-install during `transcribe`/`live` — never silently), the catalog is served from
OpenASR-operated infrastructure at `https://catalog.openasr.org` (a Cloudflare
static-asset host) rather than Hugging Face. The signed `catalog_url` is always
`https://catalog.openasr.org/v1/catalog.json`; `catalog.bug.im` is an allowlisted
fetch replica, not a second identity. The ed25519 signature, sha256, and
monotonic-epoch checks are unchanged. Model **weights stay identified by the
signed Hugging Face pack URL**; China transport may fetch the same bytes from
ModelScope. The catalog host never sees or serves weights. Only
the **public** projection is served/embedded — staged `public:false` entries are
never exposed. It is not usage telemetry — only the catalog index is requested —
but the catalog fetch's network metadata (e.g. client IP) is observable by the
project's host. Override to a self-hosted endpoint with `OPENASR_CATALOG_ENDPOINT`
(Hugging Face no longer serves the catalog, so it is not a fallback host). Offline
devices fall back to the on-disk cache and finally the signed snapshot embedded in
the binary, so no network is required to view the model list.

### Speaker diarization privacy

Diarization is privacy-by-default: it answers "who spoke when", not "who is this
named person".

- Labels are anonymous and session-relative (`SPEAKER_00/01`, ...) until a
  speaker has enough evidence to match an explicitly enrolled local person.
  Temporary recording-level embeddings are not persisted as identities.
- Identity enrollment is opt-in, and the person library stays on the device.
  Voice ID stores enrolled voiceprints for named people so a person's label can
  remain consistent across turns and later local file transcriptions. Enrolled voiceprints and their
  samples are persisted only in a local SQLite database under
  `$OPENASR_HOME/diarize/` -- never committed, uploaded, or attached to
  transcription output. All Voice ID reads and writes go through the
  operator-only HTTP surface (`/v1/voice-id/*`, gated by `is_operator_only_path`),
  so a paired remote-compute client can never read or write the voiceprint store.
- Surface boundary: operator-only routing prevents non-operator clients from
  reading or mutating the local person library, but that CRUD boundary alone is
  not proof that a transcription/realtime response is anonymous. Any surface
  advertised as anonymous must separately disable identity matching and strip
  `speaker_person_id`/named labels at its response boundary; the universal
  Voice ID contract documented here is qualified for local file transcription.
- The default local pipeline uses FireRedVAD Stream-VAD (Apache-2.0),
  pyannote-segmentation-3.0 (MIT), and ReDimNet2-B6 (MIT), with attribution.
  DiariZen Large-s80-md-v2 weights are CC BY-NC 4.0: they are not bundled and no
  DiariZen artifact is currently exposed in either signed catalog. Any future
  distribution must be an explicit, license-linked non-commercial consent flow;
  staged source metadata alone never authorizes a download.

### Transcription history

The local server can record a transcription history (the `/v1/history` endpoint
backs the desktop history page). Recording is governed by the
`history_retention` preference (the desktop's "saved history" setting, or
`preferences.history_retention` in `~/.openasr/config.json`): the default keeps
only the five most recent entries, and `off` disables recording entirely.
Entries -- model, source name, duration, and transcript text -- are written
under `~/.openasr` (never transmitted anywhere) and pruned per the retention
scope; authorized remote-compute provider runs are excluded. The `auto_save`
preference only controls transcript-file exports, not history.

## Out of Scope for Security Reports

Please use normal GitHub issues for:

- feature requests;
- documentation corrections;
- model-family roadmap requests;
- expected unsupported/fail-closed behavior for non-implemented native inference.
