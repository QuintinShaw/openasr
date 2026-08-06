# Short-audio receipt (`openasr.short-audio-receipt.v0`)

Machine-readable evidence for the short-audio audit gate that precedes full
WER/CER. A receipt binds the exact core commit, pack bytes, audio fixture,
backend/device/OS, command, warmup/cache state, transcript, and optional RTF
samples so later quality or performance claims stay comparable.

This document is the schema contract for tooling. It is **not** an execution
capability and does **not** replace pack install sealing
(`openasr.model-pack-preflight.v1`).

## Schema id

```text
openasr.short-audio-receipt.v0
```

## Field summary

| Field | Required | Notes |
| --- | --- | --- |
| `schema` | yes | Must equal the schema id above. |
| `core_commit` | yes | 40-hex git sha of the openasr core that produced the run. |
| `pack.model_id` | yes | Model ref as run (`id` or `id:quant`). |
| `pack.content_sha256` | yes | Lowercase hex sha256 of the exact pack bytes (no `sha256:` prefix). |
| `pack.size_bytes` | yes | Pack byte length. |
| `pack.quant` | yes | Quant id (for example `q4_k`). |
| `audio.path_or_label` | yes | Path or stable label of the short clip. |
| `audio.sha256` | yes | Lowercase hex sha256 of the audio file bytes. |
| `audio.duration_s` | no | Duration in seconds when known. |
| `run.backend` | yes | `native` or `mock`. |
| `run.device` | yes | Requested device label (`cpu`, `metal`, `cuda`, `auto`, ...). |
| `run.os` | yes | `darwin`, `linux`, or `windows`. |
| `run.command` | yes | Effective argv vector for the receipt command. |
| `run.env_allowlist` | no | Small allowlisted env snapshot (`OPENASR_HOME`, ...). Never a full env dump. |
| `run.warmup` | yes | `cold` or `warm`. |
| `run.cache_state` | yes | `empty` or `populated`. |
| `metrics.rtf_samples` | no | Wall-clock RTF samples; may be empty. |
| `metrics.rtf_median` | no | Median of `rtf_samples` when samples exist. |
| `metrics.measurement_method` | no | v0 uses `wall_clock_process_elapsed`. |
| `metrics.wer_or_cer` / `ttft_s` / `peak_*` | no | Optional; leave null/absent when not measured. |
| `transcript.text` | yes | Final transcript text (UTF-8). |
| `transcript.text_sha256` | yes | Lowercase hex sha256 of the UTF-8 transcript bytes. |
| `placement` | yes | v0 records the requested device label (runtime weight placement is not introspected). |
| `scope` | yes | Default `short-audio-gate`. |
| `notes` | no | Free-form annotations. |

## Emitter

```bash
openasr bench-receipt short-audio \
  --model <id[:quant]> \
  --audio <path> \
  --backend native \
  --device cpu \
  --out receipt.json
```

Optional flags:

- `--model-pack <path.oasr>` - bind an explicit pack file
- `--runs N` - timed passes that contribute RTF samples (default 1)
- `--warmup-runs N` - untimed passes before sampling (marks warm/populated)
- `--core-commit <40-hex>` - otherwise `OPENASR_BUILD_COMMIT` or `git rev-parse HEAD`
- `--scope <label>` - default `short-audio-gate`
- `--backend mock` - plumbing only; not a quality/perf claim

The command is an **explicit tooling surface**. It does not change the default
`transcribe` path and does not add public catalog fields.

## Validation rules

- Fail closed on schema mismatch, empty required strings, non-40-hex
  `core_commit`, or non-64-lowercase-hex digests.
- `transcript.text_sha256` must match `sha256(text UTF-8 bytes)`.
- `rtf_median` may be absent when `rtf_samples` is empty; when present it must
  match the median of the samples.
- Mock backend receipts may use an all-zero pack digest only as a plumbing
  placeholder; native receipts must bind real pack bytes.

## Relationship to pack preflight

| Receipt | Purpose |
| --- | --- |
| `openasr.model-pack-preflight.v1` | Install-time pack seal (structure + runtime contract). |
| `openasr.short-audio-receipt.v0` | Short-audio gate evidence after a real decode. |

Publish tooling should keep consuming pack preflight for staging. Short-audio
receipts feed family audit / release review, not the install path.

## Non-goals (v0)

- Full WER/CER corpora
- Fabricated accelerator numbers
- Public catalog schema changes
- Silent changes to default CLI transcription UX
