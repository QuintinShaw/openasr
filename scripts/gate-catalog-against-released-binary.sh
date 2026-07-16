#!/usr/bin/env bash
# Fail-closed pre-deploy gate: proves that the latest published *stable* CLI
# release binary can still load and initialize a candidate catalog before
# that catalog ships to catalog.openasr.org.
#
# Background: catalog data (model-registry/catalog.public.json) and CLI
# binaries (release-binaries.yml) ship on independent cadences. A catalog
# edit that is fine for the binary being cut *right now* can still break an
# already-shipped, already-in-the-wild binary if it depends on a shape or
# constraint that binary doesn't know about -- and there is nothing else in
# this repo's CI that ever runs a *real released binary* against a *candidate*
# catalog to check for that. This script is that check.
#
# Method: download the latest stable CLI release's linux-x86_64 tarball,
# point it at the candidate catalog.json + catalog.signature.json pair via
# OPENASR_CATALOG_FILE + OPENASR_CATALOG_IDENTITY (the same mechanism the
# desktop client uses to load a bundled, production-signed catalog file under
# its real https:// identity -- see load_local_catalog_file_with_identity in
# crates/openasr-core/src/registry.rs), and run `openasr doctor` against a
# throwaway OPENASR_HOME. `doctor` loads the full model registry from that
# catalog and resolves the default model, so it fails closed (non-zero exit)
# on anything the running binary's catalog-loading/registry pipeline rejects.
#
# Fail-closed by design: any failure to obtain and run a *real* released
# binary (network error, missing asset, resolution ambiguity) blocks the
# catalog rather than silently skipping verification. A green gate proves the
# candidate catalog was actually exercised by a real binary; it never proves
# that vacuously.
#
# Usage:
#   scripts/gate-catalog-against-released-binary.sh <catalog.json> <catalog.signature.json>
#
# Env overrides (for local iteration / testing):
#   OPENASR_RELEASE_REPO     GitHub repo to pull the release from
#                            (default: QuintinShaw/openasr)
#   OPENASR_GATE_BINARY      Path to an already-extracted `openasr` binary to
#                            use instead of downloading one (skips the
#                            release-resolution/download steps entirely).
#   OPENASR_GATE_IDENTITY    Catalog identity to verify the candidate against
#                            (default: the production catalog_url,
#                            https://catalog.openasr.org/v1/catalog.json).
#
# Requires: `gh` authenticated with access to OPENASR_RELEASE_REPO (public
# repo, so the default GITHUB_TOKEN in Actions is sufficient), `tar`.

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <catalog.json> <catalog.signature.json>" >&2
  exit 2
fi

candidate_catalog="$1"
candidate_signature="$2"
repo="${OPENASR_RELEASE_REPO:-QuintinShaw/openasr}"
identity="${OPENASR_GATE_IDENTITY:-https://catalog.openasr.org/v1/catalog.json}"

for f in "$candidate_catalog" "$candidate_signature"; do
  if [ ! -f "$f" ]; then
    echo "::error::gate input '$f' does not exist" >&2
    exit 1
  fi
done

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

resolve_binary() {
  if [ -n "${OPENASR_GATE_BINARY:-}" ]; then
    echo "==> using OPENASR_GATE_BINARY override: ${OPENASR_GATE_BINARY}" >&2
    printf '%s\n' "$OPENASR_GATE_BINARY"
    return 0
  fi

  echo "==> resolving latest stable release for ${repo}" >&2
  local tag
  if ! tag="$(gh api "repos/${repo}/releases/latest" --jq '.tag_name' 2>&1)"; then
    echo "::error::could not resolve the latest release for ${repo} (network or gh-auth failure). Fail-closed: refusing to bless a catalog without exercising a real released binary against it. gh output: ${tag}" >&2
    exit 1
  fi

  case "$tag" in
    v[0-9]*.[0-9]*.[0-9]*) ;;
    *)
      echo "::error::resolved 'latest' release tag '${tag}' for ${repo} does not look like a CLI release tag (expected vMAJOR.MINOR.PATCH -- e.g. it may have resolved to a non-CLI release such as a desktop-vX.Y.Z tag). Refusing to guess which asset to test against; fail-closed." >&2
      exit 1
      ;;
  esac
  echo "==> latest stable CLI release: ${tag}" >&2

  local asset_dir="$work/release-asset"
  mkdir -p "$asset_dir"
  # Pattern match (not a hardcoded filename) so this survives a version bump
  # without edits. Anchored to the plain (non-CUDA/Vulkan/ROCm/musl) Linux
  # x86_64 CPU build: the runner is native linux-x86_64, so this is the one
  # build we can execute directly with no extra toolchain.
  if ! gh release download "$tag" --repo "$repo" \
        --pattern 'openasr-*-linux-x86_64.tar.gz' \
        --dir "$asset_dir" --clobber; then
    echo "::error::failed to download the ${tag} linux-x86_64 CLI release asset from ${repo}. Fail-closed: refusing to bless a catalog without exercising a real released binary against it." >&2
    exit 1
  fi

  local archive
  archive="$(find "$asset_dir" -maxdepth 1 -type f -name '*.tar.gz' | head -n1)"
  if [ -z "$archive" ]; then
    echo "::error::no linux-x86_64 tar.gz asset found on release ${tag} of ${repo}" >&2
    exit 1
  fi

  local extract_dir="$work/extracted"
  mkdir -p "$extract_dir"
  tar xzf "$archive" -C "$extract_dir"

  local binary
  binary="$(find "$extract_dir" -type f -name openasr | head -n1)"
  if [ -z "$binary" ]; then
    echo "::error::could not find an 'openasr' executable anywhere inside ${archive}" >&2
    exit 1
  fi
  chmod +x "$binary"
  printf '%s\n' "$binary"
}

binary="$(resolve_binary)"
echo "==> gating candidate catalog against released binary: ${binary}" >&2
"$binary" --version >&2 || true

stage="$work/candidate"
home="$work/home"
mkdir -p "$stage" "$home"
# The sidecar signature must sit next to the catalog file under the exact
# name catalog_security::CATALOG_SIGNATURE_FILE_NAME ("catalog.signature.json"):
# load_local_catalog_file_with_identity derives it via path.with_file_name(...).
cp "$candidate_catalog" "$stage/catalog.json"
cp "$candidate_signature" "$stage/catalog.signature.json"

export OPENASR_HOME="$home"
export OPENASR_CATALOG_FILE="$stage/catalog.json"
export OPENASR_CATALOG_IDENTITY="$identity"

echo "==> running: openasr doctor (OPENASR_CATALOG_FILE=${OPENASR_CATALOG_FILE}, OPENASR_CATALOG_IDENTITY=${identity})" >&2
# NOTE: deliberately not `if "$binary" doctor; then ... fi` -- when the
# condition of an `if` with no `else` is false, `$?` immediately after `fi`
# is the *if statement's* exit status (0), not the condition command's; that
# would silently turn a real failure into a fabricated "exited 0" success
# report. `cmd || status=$?` is also `-e`-safe: as the left side of `||`,
# cmd's failure does not trigger errexit.
status=0
"$binary" doctor || status=$?
if [ "$status" -eq 0 ]; then
  echo "==> OK: ${binary} loaded and initialized the candidate catalog cleanly." >&2
  exit 0
fi
echo "::error::the latest released CLI binary (${binary}) failed to load/initialize the candidate catalog (doctor exited ${status}). A currently-shipping client would fail closed against this catalog. Blocking the deploy -- fix the catalog (or cut a compatible release) before retrying." >&2
exit "$status"
