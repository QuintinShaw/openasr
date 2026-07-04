# Releasing

OpenASR uses a single workspace version and a commit-driven release flow: a
version bump pushed to `main` IS the release.

Feature, fix, and any other content changes go through pull requests as usual.
The release bump itself is the exception: a maintainer pushes it directly to
`main` as a single `chore(release)` commit. Routing the bump through a PR adds
nothing (the release fires on the merge commit anyway) and CI runs on `main`
push regardless.

## Versioning

The version lives in exactly one place: `[workspace.package] version` in the
root `Cargo.toml`. Every member crate inherits it via `version.workspace =
true`, and the `openasr-core` / `openasr-server` / `openasr-system-audio`
entries under `[workspace.dependencies]` are plain path dependencies with no
version pin to keep in sync.

Two lockfiles pin the workspace crates and must be regenerated alongside the
bump, or CI's `--locked` builds fail:

- the root `Cargo.lock`
- `tooling/system-audio-check/Cargo.lock` (standalone CI-gate workspace,
  depends on `openasr-system-audio` by path)

## Cutting a release

1. On `main`, run:

   ```bash
   scripts/bump-version.sh X.Y.Z
   ```

   This bumps the version, regenerates both lockfiles, and self-checks the
   result with `cargo metadata --locked`. It is idempotent -- rerunning it
   with the version already at `X.Y.Z` is a no-op diff.
2. Commit and push to `main`:

   ```bash
   git add Cargo.toml Cargo.lock tooling/system-audio-check/Cargo.lock
   git commit -m "chore(release): vX.Y.Z"
   git push
   ```
3. The `Release core` workflow (`.github/workflows/release-core.yml`)
   triggers on `Cargo.toml` changes:
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
