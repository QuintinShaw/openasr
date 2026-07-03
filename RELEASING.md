# Releasing

OpenASR uses a single workspace version and a commit-driven release flow: a
version bump pushed to `main` IS the release.

Feature, fix, and any other content changes go through pull requests as usual.
The release bump itself is the exception: a maintainer pushes it directly to
`main` as a single `chore(release)` commit. Routing the bump through a PR adds
nothing (the release fires on the merge commit anyway) and CI runs on `main`
push regardless.

## Versioning

The version lives in one logical place, bumped in lockstep:

- `[workspace.package] version` in the root `Cargo.toml` (all crates inherit it)
- the three inter-crate pins under `[workspace.dependencies]`
  (`openasr-core`, `openasr-server`, `openasr-system-audio`)

Two lockfiles pin the workspace crates and must be regenerated in the same
commit, or CI's `--locked` builds fail:

- the root `Cargo.lock`: `cargo update -w --offline`
- `tooling/system-audio-check/Cargo.lock` (standalone CI-gate workspace):
  `cargo update --offline -p openasr-system-audio` inside that directory

## Cutting a release

1. On `main`, bump the version in the four spots above (one edit block in
   `Cargo.toml`), regenerate both lockfiles, and commit everything together,
   e.g. `chore(release): v0.2.0`.
2. Push to `main`. The `Release core` workflow
   (`.github/workflows/release-core.yml`) triggers on `Cargo.toml` changes:
   - reads the workspace version;
   - exits cleanly if the tag `vX.Y.Z` already exists (so unrelated
     `Cargo.toml` edits and re-runs are no-ops);
   - otherwise builds release binaries for macOS arm64 and Linux x86_64,
     creates the `vX.Y.Z` tag, and publishes a GitHub Release with
     `openasr-<version>-<target>.tar.gz` artifacts plus `SHA256SUMS`.

No pre-release channels: the core releases plain `X.Y.Z` versions only.

## Manual runs

`workflow_dispatch` on the `Release core` workflow performs the same
resolve-and-release for the version currently on `main` (useful for retries;
it is idempotent thanks to the existing-tag check).
