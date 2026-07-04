# Docs Index

Source-of-truth map for active OpenASR documentation. Implementation truth and
sequencing live in [Roadmap](ROADMAP.md) (see its Implemented-baseline section).

The repo-root [Architecture](../ARCHITECTURE.md) is the fast code map for new
contributors -- crate relationships, the audio-to-transcript pipeline, and the
`arch/` + `models/` per-family layout convention. The tables below map `docs/`.

## Top-level docs (`docs/`)

| Doc | What it covers |
| --- | --- |
| [Roadmap](ROADMAP.md) | Implementation truth, sequencing, and active priorities; the Implemented-baseline section records what runs today (active `mock`/`native` backends, the seven native model families, the `arch/` registry, the `.oasr`-only pack contract) and what is deferred. OpenASR is Apache-2.0 open core. |
| [Quickstart](QUICKSTART.md) | Three commands to a real transcript: build, transcribe (native by default, consent-pull on first run), and pick a model. |
| [Model Onboarding](MODEL_ONBOARDING.md) | Contributor checklist for adding a new ASR architecture: shared `nn/` blocks plus a thin per-family step executor gated by a load-bearing block-stack descriptor (the llama.cpp model). |
| [Model Catalog, Registry, and Distribution](MODEL_CATALOG_ARCHITECTURE.md) | Catalog ownership chain (human-edited publishing catalog -> generated `model-registry/catalog.json`), `openasr pull` install mechanics, the local `model-registry/models/*.toml` cards, signed catalog hosting/cache, and the no-implicit-download boundary. |
| [Known Limitations](KNOWN_LIMITATIONS.md) | Current user-visible limits: no public binary release yet, `.oasr`-only native packs, gated streaming/diarization (declared/capability packs only), generic accelerator selection, and internal-only benchmarks. |
| [FAQ](FAQ.md) | Current-behavior questions: what OpenASR is, which families run, which backends are active, and the conservative offline transcription lane. |
| [Releasing](../RELEASING.md) | The commit-driven release process: the single workspace version, `scripts/bump-version.sh`, and the version-triggered `Release core` workflow. |

## Format contracts (`docs/format/`)

| Doc | What it covers |
| --- | --- |
| [OASR Package Contract v1](format/OASR_PACKAGE_CONTRACT_V1.md) | Normative `.oasr` distribution contract: v1 payload is standard GGUF bytes; separates the extension-agnostic container probe from the user-facing extension check; runtime/backend selection is metadata-driven, not free-form string parsing. |

## Speaker diarization

| Doc | What it covers |
| --- | --- |
| [Diarization Pack Publishing](DIARIZATION_PACK_PUBLISHING.md) | How the WeSpeaker speaker-embedding and pyannote segmentation capability packs are built and published for the `--diarize` path. |
| [WeSpeaker Embedder](WESPEAKER_EMBEDDER.md) | The pure-Rust WeSpeaker ResNet34 speaker-embedding forward pass used for diarization and speaker-change detection. |
| [VBx PLDA Resegmentation](VBX_PLDA_RESEGMENTATION.md) | The PLDA-mixture / HMM VBx resegmentation refinement for diarization. |

The diarization privacy model and remote-mode trust contract (anonymous labels,
no persistent voiceprint, identity-stays-on-client) live in
[`../SECURITY.md`](../SECURITY.md).

## Build & platform

| Doc | What it covers |
| --- | --- |
| [GPU Plugin Build](GPU_PLUGIN_BUILD.md) | Building the optional GPU backend plugin packs (Vulkan / HIP / CUDA). |
| [Android Build](ANDROID_BUILD.md) | Android (aarch64) cross-compilation. |

## Notes

- The user-facing quantization path is import-time tier selection (`fp16` /
  `q8_0` / `q4_k`, plus `q3_k` for Qwen). The earlier offline mixed-quant
  research lane (OMIX / quant-policy / quant-tier docs + `scripts/quant_*`) was
  removed; rewrite from scratch if revived.
- Performance harness, regression gates, and competitive comparisons are
  documented in [`../perf/PERFORMANCE.md`](../perf/PERFORMANCE.md); helper scripts
  are described in [`../scripts/README.md`](../scripts/README.md).
