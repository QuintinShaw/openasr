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

`openasr.short-audio-receipt.v0` remains the only top-level receipt schema. A
receipt may carry an optional, versioned `evidence` object with schema
`openasr.short-audio-receipt.evidence.v1`. Old v0 JSON remains readable, but a
receipt without this object is explicitly not GPU token-correctness evidence.
The extension is privacy-safe and bounded: provider, opaque stable device
label, and placement are short labels; artifact bindings are labels, SHA-256
hashes, and optional sizes. New writers never retain raw audio/model/output
paths or `OPENASR_HOME`. Historical v0 documents containing a path remain
readable, but those legacy values are not valid new qualification output.

The evidence object has one disjoint `evidence_class`:

| Class | Proves | Cannot prove |
| --- | --- | --- |
| `build_packaging` | Candidate binary/pack/fixture artifact identity and packaging result | Runtime placement or model token correctness |
| `placement_resource` | Selected provider/device, resolved placement, and observed resource/compute result | Token or transcript correctness |
| `token_transcript` | Family host-oracle parity for a declared output plan and fresh/reuse mode | Packaging completeness unless the artifact bindings are separately checked |

`token_transcript` additionally requires `family`, `model_id`, `quant`, concrete
runtime topology, provider/device/placement, capture and scheduler mode,
`execution.mode` (`cold` or `reuse`), a resolved output plan, a typed family
oracle/tie policy, SHA-256 token trace artifact, optional logits artifact, and
bounded top-k/margin summaries. All evidence classes carry the strict
`openasr.gpu-correctness-artifact.v1` contract, matrix digest, candidate release
subject/core commit, binary/plugin/pack/fixture identity, and all three
inventory/model/backend catalog digests. The validator hashes the supplied
trace files; a digest written into JSON without matching content is rejected.

## Correctness extension

The release matrix projects every public family/provider lane from the
architecture inventory, public model catalog, and staged backend catalog. Each
cell requires separate build/packaging, placement/resource, and token/transcript
receipts for both cold and same-process reuse where applicable. The matrix is a
staging requirement document; it must not claim a result for a provider that has
not produced a bound receipt. The finalizer fails closed on missing, stale,
wrong-artifact, placement-only, CPU-only, or partial matrices.


| Field | Required | Notes |
| --- | --- | --- |
| `schema` | yes | Must equal the schema id above. |
| `core_commit` | yes | 40-hex git sha of the openasr core that produced the run. |
| `pack.model_id` | yes | Model ref as run (`id` or `id:quant`). |
| `pack.content_sha256` | yes | Lowercase hex sha256 of the exact pack bytes (no `sha256:` prefix). |
| `pack.size_bytes` | yes | Pack byte length. |
| `pack.quant` | yes | Quant id (for example `q4_k`). |
| `audio.path_or_label` | yes | New writers emit `audio-sha256:<digest>` or another stable label, never a local path. Raw paths are legacy-read-only. |
| `audio.sha256` | yes | Lowercase hex sha256 of the audio file bytes. |
| `audio.duration_s` | no | Duration in seconds when known. |
| `run.backend` | yes | `native` or `mock`. |
| `run.device` | yes | Requested device label (`cpu`, `metal`, `cuda`, `auto`, ...). |
| `run.os` | yes | `darwin`, `linux`, or `windows`. |
| `run.command` | yes | Privacy-safe semantic argv projection. Ingress/output/model-pack paths are replaced by byte-bound or typed labels; it is not a shell replay script. |
| `run.env_allowlist` | no | Small non-path allowlisted env snapshot. `OPENASR_HOME` is forbidden. Never a full env dump. |
| `run.warmup` | yes | `cold` or `warm`. |
| `run.cache_state` | yes | `empty` or `populated`. |
| `metrics.rtf_samples` | no | Wall-clock RTF samples; may be empty. |
| `metrics.rtf_median` | no | Median of `rtf_samples` when samples exist. |
| `metrics.measurement_method` | no | v0 uses `wall_clock_process_elapsed`. |
| `metrics.wer_or_cer` / `metrics.ttft_s` | no | Optional quality/latency values; leave null/absent when not measured. |
| `metrics.peak_rss_before_model_bytes` / `metrics.peak_rss_bytes` | no | Process RSS high-water before model execution and after all runs. Their difference isolates model-created high-water from CLI/audio setup. |
| `metrics.rss_before_model_bytes` / `metrics.rss_after_model_bytes` | no | Current process RSS immediately before the first model run and after the last run while runtime caches remain warm. |
| `metrics.phys_footprint_before_model_bytes` / `metrics.phys_footprint_after_model_bytes` | no | Darwin current physical footprint at the same lifecycle boundaries; absent on unsupported platforms. |
| `metrics.peak_phys_footprint_before_model_bytes` / `metrics.peak_phys_footprint_bytes` | no | Darwin lifetime maximum physical footprint before model execution and after all runs. |
| `metrics.peak_vram_bytes` | no | Optional backend/device high-water when a trustworthy probe is available. |
| `transcript.text` | yes | Final transcript text (UTF-8). |
| `transcript.text_sha256` | yes | Lowercase hex sha256 of the UTF-8 transcript bytes. |
| `placement` | yes | Legacy/requested placement label retained for v0 compatibility. It is not proof of where graph compute ran. |
| `observed_placement` | no | Actual graph-node placement observed during compute: total/compute-node counts by backend, graph compute count, output bytes, and bounded fallback samples. Native Metal acceptance requires selected-device compute and rejects disallowed CPU/alternate-accelerator compute according to the execution placement. |
| `evidence` | no | Versioned, class-separated build/placement/token evidence nested in this same v0 receipt. Omission is never GPU token approval. This is not a third top-level JSON or policy channel. |
| `execution` | no | Additive projection of the existing request/runtime receipts: request and candidate attempts, safe exact lanes/domains, live lease reconciliation, independent live/event completeness, four-phase timings, and typed terminal. No new journal/schema authority is created. |
| `decode_diagnostics` | yes | Fail-closed projection of the runtime `GgmlDecodeOutputPlan` and reuse mode, including the unique `full_logits` fallback when compact selection is unproven. Dual-output agreement here is not compact-path authorization. |
| `scope` | yes | Default `short-audio-gate`. Hardware runners may append exactly one `/<32-lower-hex nonce>` segment; absolute, UNC, drive and traversal paths are invalid. |
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
- `--trace-out <path>` - native-only strict token trace. It is create-new (never replaces an existing path), rejects any final-path alias with `--out` including dangling symlink chains and symlinked parent directories, and fails closed if the trace directory metadata cannot be synced. Both targets pin canonical parent directories before trace publication, so later caller-path symlink swaps cannot redirect the receipt write. A trace artifact retained after that sync failure is not release-valid.
- `--backend mock` - plumbing only; not a quality/perf claim

The command is an **explicit tooling surface**. It does not change the default
`transcribe` path and does not add public catalog fields.

## Validation rules

- Fail closed on schema mismatch, empty required strings, non-40-hex
  `core_commit`, or non-64-lowercase-hex digests.
- Core structural loading remains legacy-compatible, but new construction,
  `to_pretty_json`, and qualification validation enforce the privacy-safe
  audio/command/environment/scope projection. Loading an old path-bearing v0
  document never authorizes rewriting it as new evidence.
- `transcript.text_sha256` must match `sha256(text UTF-8 bytes)`.
- `rtf_median` may be absent when `rtf_samples` is empty; when present it must
  match the median of the samples.
- Mock backend receipts may use an all-zero pack digest only as a plumbing
  placeholder; native receipts must bind real pack bytes.
- `evidence.schema` must equal `openasr.short-audio-receipt.evidence.v1`; its
  class is one of `build_packaging`, `placement_resource`, or
  `token_transcript`, and the strict artifact contract must be present.
- `matrix_sha256`, candidate release subject/core commit, catalog digests,
  binary/plugin/pack/fixture identities, topology, capture/scheduler mode, and
  model/quant bindings must match the canonical staging matrix.
- `token_transcript` evidence must carry a family oracle, resolved typed output
  plan, cold/reuse mode, non-empty token trace artifact, and bounded logits/top-k/
  margin summary. `placement_resource` requires observed placement. Classes are
  never interchangeable. The generic bench command refuses native output until
  the runtime trace producer is connected; it cannot emit a fixed `evidence:null`
  correctness claim.
- When a native accelerated run executes a ggml graph, `observed_placement` is
  populated from runtime telemetry and the emitter fails closed if observed
  compute violates the resolved FullDevice/Hybrid placement. Older v0 receipts
  remain readable because the evidence field is optional.
- `execution.live_state_complete` and `execution.event_history_complete` are
  independent. `event-capacity-exceeded` does not negate a `matched` live
  owner/broker reconciliation, but it makes the receipt ineligible for every
  evidence class that requires complete event history. A qualification gate
  must call the strict qualification-eligibility validator and never
  reinterpret dropped history as success.
- The request attempt is distinct from the pause/cancel `transcription_id` and
  candidate/cache attempt. Warmup and measured passes receive fresh request
  attempts; they are not reused merely because the process or runtime cache is
  warm.

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
