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
#   openasr-0.1.8-aarch64-apple-darwin.tar.gz
#   openasr-0.1.8-x86_64-pc-windows-msvc.zip
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

# Human label for each known target substring. Order matters: more specific
# substrings (with a GPU-feature suffix) must be matched before their base
# target, since e.g. "x86_64-pc-windows-msvc-vulkan" also contains
# "x86_64-pc-windows-msvc".
label_for_target() {
  case "$1" in
    x86_64-pc-windows-msvc-vulkan) echo "Windows x86_64 (Vulkan)" ;;
    x86_64-pc-windows-msvc-cuda) echo "Windows x86_64 (CUDA)" ;;
    x86_64-pc-windows-msvc-hip) echo "Windows x86_64 (AMD HIP/ROCm)" ;;
    x86_64-pc-windows-msvc) echo "Windows x86_64" ;;
    x86_64-unknown-linux-gnu-vulkan) echo "Linux x86_64 (Vulkan)" ;;
    x86_64-unknown-linux-gnu-cuda) echo "Linux x86_64 (CUDA)" ;;
    x86_64-unknown-linux-gnu-rocm) echo "Linux x86_64 (AMD HIP/ROCm)" ;;
    x86_64-unknown-linux-gnu) echo "Linux x86_64" ;;
    aarch64-unknown-linux-gnu) echo "Linux arm64" ;;
    x86_64-apple-darwin) echo "macOS x86_64 (Intel)" ;;
    aarch64-apple-darwin) echo "macOS arm64 (Apple Silicon)" ;;
    *) echo "$1" ;;
  esac
}

echo "<!-- install-verify:start -->"
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
