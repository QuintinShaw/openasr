# Runtime Source Preflight and Provenance

This document defines the construction boundary for every native GGUF runtime.
It applies to ASR families, auxiliary models, offline execution, and streaming
session construction.

## Invariant

One execution ingress admits one immutable file generation and performs one
bounded metadata-and-tensor-index preflight. The result is a
`GgufRuntimeSourcePreflight` containing three inseparable views:

- the already-open `GgmlRuntimeSource` and its mmap;
- the parsed GGUF metadata;
- the parsed tensor index whose offsets were validated against that mmap.

Contract validation, memory quoting, tokenizer/frontend preparation, graph
construction, and weight materialization must consume that preflight. A
runtime constructor must not accept a path, reopen a path, or reconstruct a
tensor reader from a bare `GgmlRuntimeSource` after the preflight exists.

This is both a performance and a correctness contract. Re-parsing adds bounded
but avoidable header allocations to the same phase that allocates weights and
workspace. Reopening can combine the index from one file generation with the
payload from another after an atomic pack replacement.

## Construction flow

```text
path ingress
  -> validate/open/map exact generation
  -> bounded metadata + tensor-index preflight
  -> contract validation and physical quote
  -> execution admission
  -> family runtime materialization
       -> reader from preflight parts
       -> native weight context from preflight
```

The family boundary is typed: production executor, actor-pool, prepared-runtime,
and streaming-session constructors carry `&GgufRuntimeSourcePreflight` (or an
owned clone when a queued job must outlive the caller) until every
source-dependent object has been materialized. A later queued job may instead
carry source-independent prepared weights plus the content/build identity; it
must not carry a bare path/source that could restart parsing. Code may derive a
reference to `preflight.runtime_source` for content identity and diagnostics,
but must not use that derived source to restart parsing.

`GgufTensorDataReader::from_path` and
`GgufTensorDataReader::from_runtime_source` remain unpreflighted boundary tools
for explicit import/conversion code and isolated tests. They are not runtime
materialization APIs. Likewise, a production graph loads native weights with
`load_gguf_weight_context_from_preflight`; the bare-source loader is test-only.

The native weight context still performs one bounded C-side structural parse
to create ggml tensor declarations. That parse is distinct from extracting
Rust metadata and the tensor index: it uses the admitted generation and the
pack-wide loaded-weight cache shares its result across graph stages. In
sandbox mode the parser child necessarily opens its own mapping; the parent
accepts the result only after strong file identity and metadata-prefix content
checks prove it parsed the admitted generation. Parser-budget tests must
account for this structural parse explicitly rather than misclassifying it as
a provenance bypass.

## Cache identity

Reusable runtime and capability objects are keyed by content identity, never by
path alone. A path is only an ingress locator (and may additionally participate
when path semantics such as the model-id file-stem fallback affect the result).
Replacing bytes at the same path must produce a cache miss and a new preflight;
a missing or invalid path is not negative-cached, because it may later become a
valid installed pack. Concurrent cold misses for one content identity are
single-flight; invalid adapter builds are removed after current waiters observe
the failure.

Cached objects may keep metadata-derived small values, prepared host tensors,
or admitted native owners according to their own lifetime policy. A capability
cache does not keep a model mmap alive merely to avoid parsing. On a cache miss,
the adapter is derived from the same preflight metadata and index used for that
content generation.

## Family onboarding gate

A new native family is complete only when:

1. its offline and streaming ingresses accept the shared preflight;
2. every queued job or actor checkout preserves that preflight until source
   materialization completes, then carries only source-independent prepared
   state and explicit content/build identity;
3. every tensor reader and native weight context is built from the preflight;
4. import-only unpreflighted readers stay outside production runtime modules;
5. same-path replacement and bounded-parse regression gates pass.

No family-specific cache, graph wrapper, or convenience constructor may weaken
these requirements.
