# Releasing OpenASR (open core)

OpenASR is pre-1.0 and versioned with **SemVer `0.y.z`**, **workspace lockstep**:
every crate in this workspace ships the same version, bumped together.

Pre-1.0 semantics:

- `0.y.0` (minor) -- new features and/or breaking changes. Until `1.0.0`, breaking
  changes ride a minor bump; there is no compatibility promise across `0.y`.
- `0.y.z` (patch) -- backwards-compatible fixes within a `0.y` line.

Versioning is **manual**. There is no automated release tooling; the steps below
are run by a maintainer.

## Where the version lives

The single source of truth is the root [`Cargo.toml`](../Cargo.toml):

- `[workspace.package] version` -- every crate inherits this via
  `version.workspace = true`.
- `[workspace.dependencies]` -- the three inter-crate pins
  (`openasr-core`, `openasr-server`, `openasr-system-audio`) carry a matching
  `version = "..."` so the graph stays internally consistent.

A version bump edits **both** in the same file. Nothing else in the tree hard-codes
a crate version.

```bash
# bump to e.g. 0.2.0: edit the four version strings in root Cargo.toml
#   [workspace.package]    version = "0.2.0"
#   [workspace.dependencies] openasr-core = { ..., version = "0.2.0" }
#                            openasr-server = { ..., version = "0.2.0" }
#                            openasr-system-audio = { ..., version = "0.2.0" }
cargo update --workspace        # refresh Cargo.lock to the new versions
```

## Pre-release checklist

Run the workspace gates (see [AGENTS.md](../AGENTS.md) for the full list); raise the
bar when the release touches a trust boundary, the catalog, runtime dispatch, the
`.oasr` format, server auth/pairing, system audio, or model pull:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo nextest run --workspace
cargo test --workspace --doc
tooling/publish-model/scripts/regenerate_all.sh --check
```

Then confirm release hygiene:

- No forbidden artifacts in tree (model weights, runtime binaries, secrets/seeds,
  private/customer audio, unverified URL/hash metadata). Run a secret scan.
- `CHANGELOG`/release notes reflect user-visible changes since the last tag.
- The bundled catalog and model cards regenerate clean.

## Tag and publish

Tags follow `vX.Y.Z` (e.g. `v0.2.0`).

```bash
git tag -a v0.2.0 -m "openasr 0.2.0"
```

Do not `git push` or push tags without explicit, per-action approval from the
maintainer. Release artifacts for OpenASR are source, the published Hugging Face
model packs, and the desktop app -- not a crates.io publish; the inter-crate
version pins exist to keep the graph consistent if that changes.
