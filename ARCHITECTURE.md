# Architecture

A code map for contributors who want to understand OpenASR in about 30 minutes.
It sketches the crates, the transcription pipeline, and the file-layout
convention every model family follows. For the load-bearing invariants a change
must not regress, read [AGENTS.md](AGENTS.md); for contributor setup, read
[CONTRIBUTING.md](CONTRIBUTING.md).

## Crates

OpenASR is a Rust workspace of five crates centered on the engine crate,
`openasr-core`, which links a vendored [ggml](https://github.com/ggml-org/ggml)
fork compiled from source.

```text
                       openasr-cli            (the `openasr` binary)
                      /     |      \
                     v      v       v
      openasr-system-audio  |   openasr-server (local OpenAI-compatible HTTP API,
       (mic / loopback)     |    |             pairing + remote compute)
                            v    v
                        openasr-core  <---- openasr-client
                     (engine: families,      (client trust primitives:
                      ggml runtime, .oasr      TOFU pinning, pairing
                      packs, catalog,          safety codes)
                      trust boundaries)
                            |
                            v
        crates/openasr-core/third_party/openasr-ggml   (ggml fork; C/C++/Metal,
                                                         built via build.rs + CMake)
```

- **`openasr-core`** -- the engine. Model families, the ggml runtime, `.oasr`
  pack parsing, the signed catalog, and every trust-boundary decision (audio
  flow, signature verification, fail-closed dispatch, local-path validation,
  zip parsing) live here. It depends on no other workspace crate.
- **`openasr-cli`** -- the `openasr` binary. Wires the engine to the terminal
  and, for `serve`, to `openasr-server`.
- **`openasr-server`** -- the local HTTP API plus pairing and remote-compute
  serving. Transcription serves only an explicit local pack and never triggers
  a download; the only server-side install path is the operator-authenticated
  pull API.
- **`openasr-client`** -- reusable client-side trust primitives (fingerprint
  pinning, pairing safety codes) for callers that talk to a remote server.
- **`openasr-system-audio`** -- platform system-audio / loopback capture
  backends (macOS / Windows / Linux).

## Transcription pipeline

Every family runs the same shape:

```text
audio bytes -> frontend -> ggml graph -> decode -> transcript
```

1. **Audio.** Input is decoded to 16 kHz mono PCM.
2. **Frontend.** A family-specific feature extractor (e.g. log-mel or fbank)
   turns PCM into the encoder input tensor. See each family's `frontend.rs`.
3. **ggml graph.** Weights are `mmap`ed zero-copy from the `.oasr` pack and bound
   into an encoder (and, for seq2seq, a decoder) compute graph. Graph buffers are
   reused to bound peak RSS.
4. **Decode.** Greedy seq2seq, CTC, or RNN-T decoding produces tokens; the
   tokenizer detokenizes them.
5. **Transcript.** Tokens become the transcript, with optional word timestamps,
   diarization labels, and the requested output format (`text`/`json`/`srt`/...).

The whole path is **fail-closed by stage**: each step (pack path, metadata,
tensor index, encoder binding, encoder graph, tokenizer, decoder binding,
decoder graph, decode, text) returns a typed error rather than fabricating
output, and never reaches for the network.

The performance harness drives this exact call path (`transcribe_with_backend`
-> `NativeBackend`), not a re-implementation; see [perf/PERFORMANCE.md](perf/PERFORMANCE.md).

## Model-family layout (`arch/` + `models/`)

Adding or reading a family means understanding two directories under
`crates/openasr-core/src/`:

- **`arch/`** -- the data-driven **architecture registry**. Descriptors declare
  each family's block stack, hyperparameter schema, and component ids
  (frontend / tokenizer / decode policy / executor). Composer families (Cohere,
  Qwen) are materialized from these descriptors; dedicated-executor families
  (Whisper, Moonshine, Parakeet-CTC, Parakeet-TDT, wav2vec2-CTC, Dolphin,
  SenseVoice, X-ASR)
  are routed by
  them. This is the "what does this pack need" layer and stays model-agnostic.
- **`models/`** -- the executors and shared building blocks. Reusable neural
  blocks live in the sibling top-level `nn/` module (attention, conv, ffn,
  norm, encoder, decoder). Each family has its own subdirectory (`models/whisper/`, `models/qwen/`,
  `models/cohere/`, `models/parakeet_ctc/`, `models/parakeet_tdt/`,
  `models/wav2vec2_ctc/`,
  `models/moonshine/`, `models/dolphin/`, `models/sensevoice/`,
`models/xasr_zipformer/`, plus the
  diarization capability packs `models/wespeaker/` and `models/pyannote/`) that
  assembles those blocks into a graph and owns family-specific tensor binding,
  frontend, decode, and local-source import.

A typical family subdirectory (e.g. `models/whisper/`) contains `frontend.rs`,
`ggml_encoder_graph.rs` / `ggml_decoder_graph.rs`, `*_weights.rs` /
`tensor_binding.rs`, `tokenizer.rs`, `greedy_decode.rs`, `package_import.rs`,
and a `runtime_contract.rs`. New families onboard by adding an `arch/` descriptor
plus a thin `models/<family>/` executor over the shared `nn/` blocks -- see
[docs/MODEL_ONBOARDING.md](docs/MODEL_ONBOARDING.md). The rule of thumb from
[AGENTS.md](AGENTS.md): **generic capability sinks into the base layers;
family-specific tensor logic stays under `models/` and `arch/`.**

## Where to look next

| You want to understand... | Start at |
| --- | --- |
| The rules a change must not break | [AGENTS.md](AGENTS.md) |
| How to build and run from source | [CONTRIBUTING.md](CONTRIBUTING.md), [docs/QUICKSTART.md](docs/QUICKSTART.md) |
| Adding a new model architecture | [docs/MODEL_ONBOARDING.md](docs/MODEL_ONBOARDING.md) |
| The `.oasr` pack format | [docs/format/OASR_PACKAGE_CONTRACT_V1.md](docs/format/OASR_PACKAGE_CONTRACT_V1.md) |
| Catalog, registry, and pull mechanics | [docs/MODEL_CATALOG_ARCHITECTURE.md](docs/MODEL_CATALOG_ARCHITECTURE.md) |
| The performance harness and gates | [perf/PERFORMANCE.md](perf/PERFORMANCE.md) |
| What runs today vs. what is deferred | [docs/ROADMAP.md](docs/ROADMAP.md) |
