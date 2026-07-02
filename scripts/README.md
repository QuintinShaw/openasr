# Scripts Guide

Local engineering scripts for validation and iteration.

> Performance benchmarking is the Rust `bench-suite` CLI (see `perf/suite.toml`
> and `perf/baselines/`), not Python scripts. The earlier Python benchmark and
> quantization-research scripts (`benchmark_*`, `quant_*`, `omix_*`,
> `download_*_gguf`) were removed once `bench-suite` plus the import-time
> quantization tiers (`fp16` / `q8_0` / `q4_k`) superseded them. Rewrite from
> scratch if that research lane is ever revived.

## Current scripts

- `generate_longform_pause_probe.py`
  - Generate a deterministic longform pause probe from a local speech WAV — a
    test-data generator for longform planner validation.
When adding new scripts, prefer ggml-compatible `.oasr` package assumptions and
magic-led loader behavior (`GGUF` container for `.oasr` v1).
