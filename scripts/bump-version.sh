#!/usr/bin/env bash
# Bump the workspace release version, commit it, and tag the release --
# in one command.
#
# Single source of truth: [workspace.package] version in the root Cargo.toml
# (every member crate inherits it via `version.workspace = true`; the
# workspace.dependencies path pins carry no version to keep in sync). This
# script updates that one field, regenerates both lockfiles that pin the
# resolved workspace version, self-checks the result with
# `cargo metadata --locked` so a stale lockfile fails loud instead of at CI,
# commits the result, and creates an *annotated* `vX.Y.Z` tag whose message
# is the release notes.
#
# Usage:
#   scripts/bump-version.sh X.Y.Z --notes "Release highlights go here."
#
# --notes is required and fail-closed: without it there is nothing to put in
# the tag annotation, and `release-core.yml` reads that annotation verbatim
# as the release's "Highlights" section, so a release cut without notes would
# silently ship with an empty/stale Highlights block. Multi-line notes work
# (quote the whole argument); each line becomes a line of the tag message.
#
# Idempotent: rerunning with the same version and no working-tree changes
# skips the commit; if the `vX.Y.Z` tag already exists locally, tag creation
# is skipped too (with a warning) rather than failing -- to change the notes
# of an already-tagged version, delete the local tag first
# (`git tag -d vX.Y.Z`) and rerun.
#
# After it succeeds:
#   git push --follow-tags
# Pushing the commit + annotated tag together to main triggers
# `.github/workflows/release-core.yml`, which reads the tag annotation for
# Highlights and publishes the release (see RELEASING.md).

set -euo pipefail

usage() {
  echo "usage: $(basename "$0") X.Y.Z --notes \"release highlights\"" >&2
  exit 1
}

version=""
notes=""

while [ $# -gt 0 ]; do
  case "$1" in
    --notes=*)
      notes="${1#--notes=}"
      shift
      ;;
    --notes)
      [ $# -ge 2 ] || usage
      notes="$2"
      shift 2
      ;;
    -*)
      usage
      ;;
    *)
      if [ -n "$version" ]; then
        usage
      fi
      version="$1"
      shift
      ;;
  esac
done

[ -n "$version" ] || usage

# Fail-closed: no notes, no release. This is the one thing the tag
# annotation exists to carry, so an empty/whitespace-only value is rejected
# the same as a missing flag.
if [ -z "${notes// /}" ]; then
  echo "error: --notes is required (it becomes the release's Highlights section)" >&2
  usage
fi

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
tag="v${version}"

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

echo "==> changed files:"
git status --porcelain -- Cargo.toml Cargo.lock tooling/system-audio-check/Cargo.lock

if [ -n "$(git status --porcelain -- Cargo.toml Cargo.lock tooling/system-audio-check/Cargo.lock)" ]; then
  echo "==> committing chore(release): ${tag}"
  git add Cargo.toml Cargo.lock tooling/system-audio-check/Cargo.lock
  git commit -m "chore(release): ${tag}"
else
  echo "==> no file changes to commit (already at ${version})"
fi

if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
  echo "==> tag ${tag} already exists locally -- skipping (delete it with \`git tag -d ${tag}\` to redo the notes)"
else
  echo "==> creating annotated tag ${tag}"
  git tag -a "$tag" -m "$notes"
fi

cat <<EOF

Next step:
  git push --follow-tags   # ships the commit + ${tag}; triggers
                           # .github/workflows/release-core.yml on main
EOF
