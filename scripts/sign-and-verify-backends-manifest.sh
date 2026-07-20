#!/usr/bin/env bash
# Sign + publish + self-verify a release's backends-manifest.json in one
# atomic, fail-loud step. This is the PRIMARY gate for the LOCAL-ONLY
# backends-manifest signing step -- not an optional/"remember to run this"
# note. If this script does not print SIGNED-AND-VERIFIED and exit 0, the
# release is NOT signed and must not be announced.
#
# core 0.1.16-0.1.19 shipped with a never-signed backends-manifest.json
# because the old process was three separate manual `gh`/`cargo run`
# commands a maintainer had to remember, in order, after CI finished (see
# tooling/release-manifest/README.md's former "Signing" walkthrough). This
# script replaces that walkthrough with one command that cannot silently
# stop halfway: sign -> upload -> re-download-and-verify against the actual
# published release asset, each step checked, any failure aborting loudly
# with the release's version named in the error.
#
# Usage:
#   scripts/sign-and-verify-backends-manifest.sh vX.Y.Z
#   scripts/sign-and-verify-backends-manifest.sh X.Y.Z
#
# Required environment:
#   OPENASR_CATALOG_SIGNING_KEY_SEED_HEX  the REAL production Ed25519 seed.
#     LOCAL ONLY. Never set this in CI, never commit it, never put it in a
#     repo secret. See tooling/publish-model/scripts/publish_catalog.sh for
#     the sibling process that uses the same seed for the model catalog.
#
# Optional environment (best-effort dl.openasr.org mirror sync -- see
# tooling/release-manifest/README.md's "dl.openasr.org sync" section; this is
# a documented OPTIONAL CDN-fronting step, not release-blocking, so its
# absence does not fail this script):
#   B2_S3_ENDPOINT, B2_APPLICATION_KEY_ID, B2_APPLICATION_KEY
#
# Requires: gh (authenticated), cargo, python3 (only if B2 env vars are set).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  echo "usage: $(basename "$0") vX.Y.Z" >&2
  exit 1
}

fail() {
  echo "" >&2
  echo "############################################################" >&2
  echo "# SIGNING/VERIFY FAILED for ${tag:-<unknown version>}" >&2
  echo "# Release is NOT signed. Do NOT ship or announce this release." >&2
  echo "# $1" >&2
  echo "############################################################" >&2
  exit 1
}

trap 'fail "aborted at line $LINENO"' ERR

[ $# -eq 1 ] || usage
raw_arg="$1"

case "$raw_arg" in
  v*) version="${raw_arg#v}" ;;
  *) version="$raw_arg" ;;
esac

if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  fail "version must look like X.Y.Z or vX.Y.Z (got: ${raw_arg})"
fi

tag="v${version}"
manifest_url="https://dl.openasr.org/core/v${version}/backends-manifest.json"

echo "==> sign-and-verify-backends-manifest: ${tag}"

# --- Preflight: fail early, never halfway. ------------------------------

# The signing seed is LOCAL ONLY (see AGENTS.md / RELEASING.md); refuse to
# even attempt this in a CI environment so a misconfigured workflow cannot
# smuggle the seed into a runner by invoking this script.
if [ "${CI:-}" = "true" ] || [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  fail "refusing to run inside CI (CI/GITHUB_ACTIONS is set) -- this script uses the production signing seed and must only run on a maintainer's local machine."
fi

if [ -z "${OPENASR_CATALOG_SIGNING_KEY_SEED_HEX:-}" ]; then
  fail "OPENASR_CATALOG_SIGNING_KEY_SEED_HEX is not set -- export the real production seed before running this script (see tooling/publish-model/scripts/publish_catalog.sh for where this seed lives)."
fi

if ! command -v gh >/dev/null 2>&1; then
  fail "gh (GitHub CLI) is not installed."
fi

if ! gh auth status >/dev/null 2>&1; then
  fail "gh is not authenticated -- run 'gh auth login' first."
fi

if ! command -v cargo >/dev/null 2>&1; then
  fail "cargo is not installed."
fi

if ! gh release view "$tag" >/dev/null 2>&1; then
  fail "GitHub release ${tag} does not exist yet -- wait for release-core.yml / release-binaries.yml to finish publishing it first."
fi

workdir="$(mktemp -d "${TMPDIR:-/tmp}/openasr-backends-manifest-sign.XXXXXX")"
cleanup() { rm -rf "$workdir"; }
trap cleanup EXIT

# --- Step 1: sign -------------------------------------------------------

echo "==> [1/3] downloading unsigned backends-manifest.json from release ${tag}"
if ! gh release download "$tag" -p backends-manifest.json -D "$workdir" --clobber; then
  fail "could not download backends-manifest.json from release ${tag} -- did release-binaries.yml's 'checksums' job finish and attach it?"
fi
if [ ! -s "${workdir}/backends-manifest.json" ]; then
  fail "downloaded backends-manifest.json from release ${tag} is missing or empty."
fi

echo "==> [1/3] signing backends-manifest.json (manifest-url: ${manifest_url})"
if ! cargo run --quiet -p openasr-cli -- __openasr-sign-backends-manifest \
    "${workdir}/backends-manifest.json" \
    --out "${workdir}/backends-manifest.signature.json" \
    --manifest-url "$manifest_url"; then
  fail "__openasr-sign-backends-manifest failed -- signature was not produced."
fi
if [ ! -s "${workdir}/backends-manifest.signature.json" ]; then
  fail "backends-manifest.signature.json was not written (or is empty) after signing."
fi

# --- Step 2: publish -----------------------------------------------------

echo "==> [2/3] uploading backends-manifest.signature.json to release ${tag}"
if ! gh release upload "$tag" "${workdir}/backends-manifest.signature.json" --clobber; then
  fail "gh release upload of backends-manifest.signature.json failed."
fi

if [ -n "${B2_S3_ENDPOINT:-}" ] && [ -n "${B2_APPLICATION_KEY_ID:-}" ] && [ -n "${B2_APPLICATION_KEY:-}" ]; then
  echo "==> [2/3] B2/dl.openasr.org env vars present -- syncing manifest + signature"
  if ! python3 tooling/release-manifest/b2_sync.py sync --version "$version" \
      "${workdir}/backends-manifest.json" \
      "${workdir}/backends-manifest.signature.json"; then
    fail "b2_sync.py sync failed for backends-manifest.json/.signature.json -- release asset upload already succeeded (GitHub Releases still serves it), but dl.openasr.org is out of sync. Re-run 'python3 tooling/release-manifest/b2_sync.py sync --version ${version} <files>' once fixed."
  fi
else
  echo "==> [2/3] B2_S3_ENDPOINT/B2_APPLICATION_KEY_ID/B2_APPLICATION_KEY not all set -- skipping dl.openasr.org sync (documented as OPTIONAL, not release-blocking; see tooling/release-manifest/README.md)."
fi

# --- Step 3: verify against what is ACTUALLY published -------------------
# Re-download fresh (do not reuse the local files we just signed/uploaded)
# so this step really proves the round trip through GitHub Releases worked,
# not just that signing succeeded locally.

verify_dir="$(mktemp -d "${TMPDIR:-/tmp}/openasr-backends-manifest-verify.XXXXXX")"
echo "==> [3/3] re-downloading published manifest + signature from release ${tag} for self-verification"
if ! gh release download "$tag" \
    -p backends-manifest.json -p backends-manifest.signature.json \
    -D "$verify_dir" --clobber; then
  rm -rf "$verify_dir"
  fail "could not re-download backends-manifest.json/.signature.json from release ${tag} after upload."
fi
if [ ! -s "${verify_dir}/backends-manifest.json" ] || [ ! -s "${verify_dir}/backends-manifest.signature.json" ]; then
  rm -rf "$verify_dir"
  fail "re-downloaded manifest and/or signature from release ${tag} is missing or empty."
fi

echo "==> [3/3] verifying signature against the production trust root"
if ! cargo run --quiet -p openasr-cli -- __openasr-verify-backends-manifest \
    "${verify_dir}/backends-manifest.json" \
    --signature "${verify_dir}/backends-manifest.signature.json" \
    --manifest-url "$manifest_url"; then
  rm -rf "$verify_dir"
  fail "__openasr-verify-backends-manifest rejected the PUBLISHED release asset -- the release is unsigned/mis-signed from the reader's point of view even though this script uploaded something. Do not ship."
fi
rm -rf "$verify_dir"

echo ""
echo "============================================================"
echo "SIGNED-AND-VERIFIED backends-manifest for ${tag}"
echo "  manifest-url: ${manifest_url}"
echo "============================================================"
