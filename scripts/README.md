# Scripts Guide

Local engineering scripts for validation and iteration.

> Performance benchmarking is the Rust `bench-suite` CLI (see `perf/suite.toml`
> and `perf/baselines/`), not Python scripts. The earlier Python benchmark and
> quantization-research scripts (`benchmark_*`, `quant_*`, `omix_*`,
> `download_*_gguf`) were removed once `bench-suite` plus the import-time
> quantization tiers (`fp16` / `q8_0` / `q4_k`) superseded them. Rewrite from
> scratch if that research lane is ever revived.

## Current scripts

- `bump-version.sh`
  - Bump the workspace release version, commit, and tag it (`--notes`
    required; see `RELEASING.md`).
- `render-install-verify.sh`
  - Render a release's "Install & Verify" section from its actual asset list
    (used by `.github/workflows/release-core.yml`, both for the initial stub
    and the final finalize-notes pass).
- `splice-install-verify.py`
  - Replace the `<!-- install-verify:start/end -->` section of a release
    body in place (used by `.github/workflows/release-core.yml`'s
    `finalize-notes` job).
- `update-homebrew-formula.py`
  - Bump `Formula/openasr.rb`'s version and per-target sha256 in place (used
    by `.github/workflows/release-core.yml`'s `update-homebrew-tap` job
    against a checkout of `QuintinShaw/homebrew-tap`). See `RELEASING.md`'s
    "Homebrew tap" section.
- `generate_longform_pause_probe.py`
  - Generate a deterministic longform pause probe from a local speech WAV — a
    test-data generator for longform planner validation.
- `generate-ffi-header.sh`
  - Regenerate `crates/openasr-ffi/include/openasr.h` from the `openasr-ffi`
    Rust source via `cbindgen` (`--check` verifies the committed header is
    current; used by the `xcframework` CI job). See `docs/SDK_IOS_MACOS.md`.
- `build-xcframework.sh`
  - Build `OpenASR.xcframework` (ios-arm64 / ios-arm64-simulator /
    macos-arm64 slices) from `crates/openasr-ffi`, degrading gracefully to
    whichever slices the host's Xcode install can actually produce. See
    `docs/SDK_IOS_MACOS.md`.

When adding new scripts, prefer ggml-compatible `.oasr` package assumptions and
magic-led loader behavior (`GGUF` container for `.oasr` v1).
