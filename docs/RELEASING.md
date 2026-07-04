# Releasing OpenASR (open core)

Moved: the release process now lives in the root [`RELEASING.md`](../RELEASING.md).

That is the single source of truth for versioning, the `scripts/bump-version.sh`
bump flow, and how the `Release core` workflow (`.github/workflows/release-core.yml`)
publishes a release on a version bump pushed to `main`. Update [README.md](../README.md)
and [DOCS_INDEX.md](DOCS_INDEX.md) links to point at the root file directly when
convenient.
