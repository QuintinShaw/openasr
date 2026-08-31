#!/usr/bin/env bash
# Verify a GitHub Release (draft or public) has the required subject set.
# Draft releases are invisible to contents:read tokens; callers must supply
# a token that can see the release.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

[ "$#" -ge 1 ] && [ "$#" -le 2 ] || {
  echo "usage: $(basename "$0") vX.Y.Z [owner/repo]" >&2
  exit 1
}
tag="$1"
[[ "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  echo "tag must be vX.Y.Z" >&2
  exit 1
}
version="${tag#v}"
repository="${2:-${GITHUB_REPOSITORY:-QuintinShaw/openasr}}"
command -v gh >/dev/null 2>&1 || { echo "gh is required" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }

pack_names="$(mktemp)"
pack_dir="$(mktemp -d)"
actual_file="$(mktemp)"
cleanup() { rm -rf -- "$pack_names" "$pack_dir" "$actual_file"; }
trap cleanup EXIT

python3 tooling/release-manifest/release_completeness.py pack-names > "$pack_names"
while IFS= read -r pack_name; do
  [ -n "$pack_name" ] || continue
  gh release download "$tag" --repo "$repository" -p "$pack_name" -D "$pack_dir"
done < "$pack_names"
gh release view "$tag" --repo "$repository" --json assets \
  --jq '.assets[].name' | sort > "$actual_file"
python3 tooling/release-manifest/release_completeness.py compare \
  --version "$version" \
  --pack-dir "$pack_dir" \
  --actual-file "$actual_file"
