#!/usr/bin/env bash
# Render the "Install & Verify" section of a GitHub Release body from the
# release's *actual* asset list, so it never drifts from what really shipped
# (the old hardcoded "macOS arm64 and Linux x86_64" sentence was exactly this
# kind of drift once release-binaries.yml started shipping 11 targets).
#
# Usage:
#   scripts/render-install-verify.sh <version> <repo-slug> <tag> < asset-names.txt
#
# asset-names.txt: one release asset filename per line (as returned by
# `gh release view <tag> --json assets --jq '.assets[].name'`), e.g.:
#   openasr-0.1.10-macos-arm64.tar.gz
#   openasr-0.1.10-windows-x86_64.zip
#   SHA256SUMS
#
# Prints a self-delimited markdown section (including its start/end HTML
# comment markers) on stdout, so a caller can splice it into an existing
# release body by replacing everything between the markers.

set -euo pipefail

if [ $# -ne 3 ]; then
  echo "usage: $(basename "$0") <version> <repo-slug> <tag> < asset-names.txt" >&2
  exit 1
fi

version="$1"
repo="$2"
tag="$3"

# Human label for each known friendly asset segment (the `asset:` field in
# release-binaries.yml, e.g. "linux-x86_64", "windows-x86_64-cuda"). Order
# matters: a GPU/libc-suffixed name must be matched before its base, since a
# glob like windows-x86_64* would otherwise catch windows-x86_64-cuda first.
label_for_target() {
  case "$1" in
    windows-x86_64-vulkan) echo "Windows x86_64 (Vulkan)" ;;
    windows-x86_64-cuda) echo "Windows x86_64 (CUDA)" ;;
    windows-x86_64-rocm) echo "Windows x86_64 (AMD ROCm)" ;;
    windows-x86_64) echo "Windows x86_64" ;;
    windows-arm64) echo "Windows arm64" ;;
    linux-x86_64-vulkan) echo "Linux x86_64 (Vulkan)" ;;
    linux-x86_64-cuda) echo "Linux x86_64 (CUDA)" ;;
    linux-x86_64-rocm) echo "Linux x86_64 (AMD ROCm)" ;;
    linux-x86_64-musl) echo "Linux x86_64 (musl)" ;;
    linux-x86_64) echo "Linux x86_64" ;;
    linux-arm64-musl) echo "Linux arm64 (musl)" ;;
    linux-arm64) echo "Linux arm64" ;;
    macos-x86_64) echo "macOS x86_64 (Intel)" ;;
    macos-arm64) echo "macOS arm64 (Apple Silicon)" ;;
    *) echo "$1" ;;
  esac
}

echo "<!-- install-verify:start -->"
echo "> [!NOTE]"
echo "> **This release is the OpenASR engine and command-line tool.** Looking for"
echo "> the desktop app (dictation, live captions, GUI)? Download it from"
echo "> [openasr.org/download](https://openasr.org/download) or the"
echo "> [\`desktop-vX.Y.Z\` releases](https://github.com/${repo}/releases?q=desktop&expanded=true)."
echo
echo "## Install & Verify"
echo
echo "Download the archive for your platform from the assets below, extract it,"
echo "and run the \`openasr\` binary directly (no installer)."
echo

any_asset=0
while IFS= read -r name; do
  [ -n "$name" ] || continue
  [ "$name" != "SHA256SUMS" ] || continue
  case "$name" in
    "openasr-${version}-"*.tar.gz) target="${name#openasr-"${version}"-}"; target="${target%.tar.gz}" ;;
    "openasr-${version}-"*.zip) target="${name#openasr-"${version}"-}"; target="${target%.zip}" ;;
    *) continue ;;
  esac
  label="$(label_for_target "$target")"
  echo "- **${label}**: [\`${name}\`](https://github.com/${repo}/releases/download/${tag}/${name})"
  any_asset=1
done

if [ "$any_asset" -eq 0 ]; then
  echo "_(no platform archives found for this release yet)_"
fi

cat <<EOF

### Verify

Every archive is listed in \`SHA256SUMS\`, published alongside them on this
release:

\`\`\`bash
curl -LO https://github.com/${repo}/releases/download/${tag}/SHA256SUMS
curl -LO https://github.com/${repo}/releases/download/${tag}/<the archive you downloaded>
sha256sum -c SHA256SUMS --ignore-missing
\`\`\`
EOF
echo "<!-- install-verify:end -->"
