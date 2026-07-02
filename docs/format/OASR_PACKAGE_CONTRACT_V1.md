# `.oasr` Package Contract v1 (GGUF)

Status: normative contract for the ggml-only migration track.

## Scope

- `.oasr` is the OpenASR distribution extension.
- v1 payload is **standard GGUF bytes**; OpenASR does not define a custom outer container for v1.

## Two distinct boundaries

OpenASR deliberately separates a low-level *container probe* from the
user-facing *extension contract*; they are not the same check.

### Low-level container magic probe (extension-agnostic)

`ggml_runtime::package_probe::probe_ggml_package_path` and
`validate_ggml_runtime_source_path` identify a package by its first four magic
bytes, never by extension:

1. `GGUF` magic is accepted (the `.oasr` v1 payload is standard GGUF bytes).
2. `OASR` magic is reserved for a future real container and is rejected today.
3. Any other/unknown magic fails closed.

This reader is intentionally extension-agnostic so internal GGUF test fixtures
remain loadable regardless of filename.

### User-facing `.oasr` extension contract

Public producers and consumers gate on the filename extension via
`has_openasr_runtime_pack_extension` (`OPENASR_RUNTIME_PACK_EXTENSION = "oasr"`):

1. `.oasr` is the sole supported user-facing runtime-pack extension.
2. The CLI run input (`--model-pack`) and the `import-*-local` converter output
   accept/produce only `.oasr`. The legacy `.gguf` extension is no longer
   accepted at any public boundary.

## OpenASR Metadata Keys (GGUF KV)

OpenASR `.oasr` v1 packages carry these GGUF metadata keys (defined in
`crates/openasr-core/src/models/oasr_metadata.rs`):

- `openasr.package.version`
- `openasr.model.family`
- `openasr.model.architecture`
- `openasr.runtime.min_version`
- `openasr.audio.frontend`
- `openasr.decode.policy`

Required value contract:

- `openasr.package.version` must be string `"1"` (`OASR_PACKAGE_VERSION_V1`).
- Other keys are string-typed OpenASR descriptors consumed by the loader and
  runtime selection layers.

Optional feature metadata:

- `openasr.features.diarization = "cohere-token-stream-v1"` declares that a
  Cohere runtime pack can expose speaker diarization when the tokenizer also
  carries the required diarization/timestamp speaker tokens. OpenASR still
  fails closed for other families or incomplete tokenizers.
- `openasr.features.streaming = "ggml-true-streaming-v1"` declares that the
  runtime pack is intended for a product true-streaming GGML adapter. OpenASR
  exposes this as a realtime capability only when a matching built-in streaming
  executor is also registered; metadata alone must never make offline execution
  look like true streaming.

## Runtime selection

Backend and runtime execution are selected from the typed metadata keys above
(family/architecture/frontend/decode-policy/feature flags), never by parsing
free-form strings. Package format tier names (`fp16`, `q8_0`, `q4_k`, ...)
describe portable payload facts only and do not name hardware-specific kernels;
backend/kernel identifiers stay inside the ggml runtime layer that consumes this
metadata.

## Forward Compatibility Boundary

- If OpenASR introduces a true container later, it must use distinct `OASR`
  magic and a new format revision document.
- v1 readers must fail closed on unknown/unsupported magic or malformed
  required metadata.
- Runtime-source validation rejecting the reserved `OASR` magic is a
  fail-closed guardrail; it does not imply support for a non-GGUF `.oasr`
  runtime format.
