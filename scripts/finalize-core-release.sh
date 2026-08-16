#!/usr/bin/env bash
# Publish a draft core release only after both signed runtime manifests are live.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

fail() { printf '\nRELEASE FINALIZATION FAILED for %s\n%s\n' "${tag:-<unknown>}" "$1" >&2; exit 1; }
trap 'fail "aborted at line $LINENO"' ERR

[ "$#" -eq 1 ] || fail "usage: $(basename "$0") vX.Y.Z"
version="${1#v}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "version must be X.Y.Z or vX.Y.Z"
tag="v${version}"
command -v gh >/dev/null 2>&1 || fail "gh is required"
command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v cargo >/dev/null 2>&1 || fail "cargo is required"
gh auth status >/dev/null 2>&1 || fail "gh is not authenticated"

is_draft="$(gh release view "$tag" --json isDraft --jq .isDraft 2>/dev/null)" \
  || fail "GitHub release ${tag} does not exist"
[ "$is_draft" = "true" ] || fail "release ${tag} is already public or is not a draft"

workdir="$(mktemp -d "${TMPDIR:-/tmp}/openasr-release-finalize.XXXXXX")"
trap 'rm -rf "$workdir"' EXIT
gh release download "$tag" \
  -p backend-pack-cuda.json -p backend-pack-hip.json \
  -p backends-manifest.json -p backends-manifest.signature.json \
  -D "$workdir" --clobber

cargo run --quiet -p openasr-cli -- __openasr-verify-backends-manifest \
  "$workdir/backends-manifest.json" \
  --signature "$workdir/backends-manifest.signature.json" \
  --manifest-url "https://dl.openasr.org/core/v${version}/backends-manifest.json"

cache_bust="$(date +%s)"
curl -fsSL "https://catalog.openasr.org/v1/catalog.json?release=${tag}-${cache_bust}" \
  -o "$workdir/catalog.json"
curl -fsSL "https://catalog.openasr.org/v1/catalog.signature.json?release=${tag}-${cache_bust}" \
  -o "$workdir/catalog.signature.json"
OPENASR_HOME="$workdir/home" \
OPENASR_CATALOG_FILE="$workdir/catalog.json" \
OPENASR_CATALOG_IDENTITY="https://catalog.openasr.org/v1/catalog.json" \
  cargo run --quiet -p openasr-cli -- doctor >/dev/null
python3 tooling/release-manifest/backend_catalog.py verify-catalog \
  --catalog "$workdir/catalog.json" \
  --entry "$workdir/backend-pack-cuda.json" \
  --entry "$workdir/backend-pack-hip.json"

echo "==> all signed distribution metadata is live; publishing ${tag}"
gh release edit "$tag" --draft=false --latest
echo "RELEASE-PUBLISHED ${tag}"
