#!/usr/bin/env bash
# Bump the workspace release version in one command.
#
# Single source of truth: [workspace.package] version in the root Cargo.toml
# (every member crate inherits it via `version.workspace = true`; the
# workspace.dependencies path pins carry no version to keep in sync). This
# script updates that one field, then regenerates both lockfiles that pin the
# resolved workspace version, and self-checks the result with
# `cargo metadata --locked` so a stale lockfile fails loud instead of at CI.
#
# Usage:
#   scripts/bump-version.sh X.Y.Z
#
# Idempotent: running it again with the same version is a no-op (no diff).
#
# After it succeeds:
#   git add -A
#   git commit -m "chore(release): vX.Y.Z"
#   git push
# Pushing to main triggers `.github/workflows/release-core.yml`, which tags
# and publishes the release (see RELEASING.md).

set -euo pipefail

usage() {
  echo "usage: $(basename "$0") X.Y.Z" >&2
  exit 1
}

[ $# -eq 1 ] || usage

version="$1"

# SemVer 0.y.z (and future X.Y.Z once past 1.0); no pre-release/build suffix,
# matching what release-core.yml expects to tag as vX.Y.Z.
if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "error: version must look like X.Y.Z (got: ${version})" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

CARGO_TOML="Cargo.toml"

if [ ! -f "$CARGO_TOML" ]; then
  echo "error: expected to find ${CARGO_TOML} at repo root (${REPO_ROOT})" >&2
  exit 1
fi

echo "==> bumping workspace version to ${version}"

# Only the [workspace.package] version line is anchored at column 0
# (`^version = "..."`); the workspace.dependencies entries below it are path
# dependencies with no version field, so this single substitution is the only
# hand-edit a release needs.
if command -v perl >/dev/null 2>&1; then
  perl -0pi -e "s/^version = \"[^\"]+\"/version = \"${version}\"/m" "$CARGO_TOML"
else
  # Portable fallback without perl: sed -E with a one-shot flag so only the
  # first (anchored) match is touched.
  sed -i.bak -E "0,/^version = \"[^\"]+\"/s//version = \"${version}\"/" "$CARGO_TOML"
  rm -f "${CARGO_TOML}.bak"
fi

new_version="$(grep -m1 -E '^version = "' "$CARGO_TOML" | sed -E 's/^version = "([^"]+)".*/\1/')"
if [ "$new_version" != "$version" ]; then
  echo "error: failed to update ${CARGO_TOML} (got version=${new_version})" >&2
  exit 1
fi

echo "==> regenerating root Cargo.lock"
cargo update --workspace --offline

echo "==> regenerating tooling/system-audio-check/Cargo.lock"
(cd tooling/system-audio-check && cargo update --offline -p openasr-system-audio)

echo "==> self-check: cargo metadata --locked (root workspace)"
cargo metadata --locked --format-version 1 >/dev/null

echo "==> self-check: cargo metadata --locked (tooling/system-audio-check)"
(cd tooling/system-audio-check && cargo metadata --locked --format-version 1 >/dev/null)

echo "==> done. changed files:"
git status --porcelain -- Cargo.toml Cargo.lock tooling/system-audio-check/Cargo.lock

cat <<EOF

Next steps:
  git add Cargo.toml Cargo.lock tooling/system-audio-check/Cargo.lock
  git commit -m "chore(release): v${version}"
  git push   # triggers .github/workflows/release-core.yml on main
EOF
