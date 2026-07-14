#!/bin/sh
# OpenASR CLI installer.
#
#   curl -fsSL https://dl.openasr.org/install.sh | sh
#
# POSIX sh on purpose (invoked as `... | sh`, not `| bash`): this must run
# unmodified under dash (Debian/Ubuntu's /bin/sh), busybox ash, and bash.
#
# What it does:
#   1. Detects OS + arch and maps them to a released asset name.
#   2. Resolves the latest GitHub release tag via the `releases/latest`
#      redirect (no GitHub API call, so no unauthenticated rate limit).
#   3. Downloads the release tarball and the release's SHA256SUMS file, and
#      verifies the tarball's checksum before touching anything else. This
#      is fail-closed by design, matching OpenASR's "no cloud, no unverified
#      binaries" posture: a checksum mismatch (or missing checksum tool)
#      deletes the download and exits non-zero rather than installing
#      something unverified.
#   4. Installs the `openasr` binary into --prefix (default ~/.local/bin,
#      created if missing), never using sudo.
#
# Usage:
#   install.sh [--prefix DIR] [--version vX.Y.Z]
#
# Env overrides (equivalent to the flags above): OPENASR_INSTALL_PREFIX,
# OPENASR_INSTALL_VERSION.

set -eu

REPO="QuintinShaw/openasr"
GITHUB="https://github.com"

prefix="${OPENASR_INSTALL_PREFIX:-$HOME/.local/bin}"
version="${OPENASR_INSTALL_VERSION:-}"

err() {
  echo "error: $*" >&2
  exit 1
}

info() {
  echo "==> $*"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix)
      [ $# -ge 2 ] || err "--prefix requires a directory argument"
      prefix="$2"
      shift 2
      ;;
    --prefix=*)
      prefix="${1#--prefix=}"
      shift
      ;;
    --version)
      [ $# -ge 2 ] || err "--version requires a tag argument (e.g. v0.1.14)"
      version="$2"
      shift 2
      ;;
    --version=*)
      version="${1#--version=}"
      shift
      ;;
    -h | --help)
      sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      err "unknown argument: $1 (see --help)"
      ;;
  esac
done

# -- OS / arch detection -----------------------------------------------------

os_raw="$(uname -s)"
arch_raw="$(uname -m)"

case "$os_raw" in
  Darwin) os="macos" ;;
  Linux) os="linux" ;;
  *) err "unsupported OS: $os_raw (OpenASR ships macOS and Linux builds; see $GITHUB/$REPO/releases for other platforms, e.g. Windows)" ;;
esac

case "$arch_raw" in
  arm64 | aarch64) arch="arm64" ;;
  x86_64 | amd64) arch="x86_64" ;;
  *) err "unsupported architecture: $arch_raw" ;;
esac

# Linux uses the statically linked musl build by default: it runs unmodified
# across glibc and musl distros alike (no dynamic-loader version to match),
# which matters far more for a curl|sh installer hitting arbitrary hosts than
# for the Homebrew tap (Homebrew-on-Linux already requires glibc, so it uses
# the dynamic build instead). macOS has only one libc, so no musl variant
# exists there.
if [ "$os" = "linux" ]; then
  target="linux-${arch}-musl"
else
  target="${os}-${arch}"
fi

# -- Resolve version ----------------------------------------------------------

if [ -z "$version" ]; then
  info "resolving latest release"
  # `releases/latest` 302-redirects to `.../releases/tag/vX.Y.Z`; read the
  # final URL rather than following it, so this needs one request and no
  # GitHub API call (which is rate-limited per-IP for unauthenticated use).
  latest_url="$(curl -fsSL -o /dev/null -w '%{url_effective}' "$GITHUB/$REPO/releases/latest")" \
    || err "could not resolve the latest release from $GITHUB/$REPO/releases/latest"
  version="${latest_url##*/}"
  [ -n "$version" ] || err "could not parse a release tag from $latest_url"
fi

case "$version" in
  v*) version_num="${version#v}" ;;
  *) version_num="$version" ;;
esac

info "installing openasr $version ($target) to $prefix"

# -- Download + verify --------------------------------------------------------

asset="openasr-${version_num}-${target}.tar.gz"
base_url="$GITHUB/$REPO/releases/download/$version"

workdir="$(mktemp -d)"
cleanup() {
  rm -rf "$workdir"
}
trap cleanup EXIT INT TERM

archive="$workdir/$asset"
sums="$workdir/SHA256SUMS"

info "downloading $asset"
curl -fsSL -o "$archive" "$base_url/$asset" \
  || err "download failed: $base_url/$asset (check that $version shipped a $target build)"
curl -fsSL -o "$sums" "$base_url/SHA256SUMS" \
  || err "download failed: $base_url/SHA256SUMS"

expected_line="$(grep -F " $asset" "$sums" || true)"
[ -n "$expected_line" ] || err "SHA256SUMS has no entry for $asset -- refusing to install an unverifiable binary"
expected_sha="${expected_line%% *}"

if command -v sha256sum >/dev/null 2>&1; then
  actual_sha="$(sha256sum "$archive" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  actual_sha="$(shasum -a 256 "$archive" | awk '{print $1}')"
else
  err "no sha256sum/shasum found -- refusing to install without checksum verification"
fi

if [ "$actual_sha" != "$expected_sha" ]; then
  err "checksum mismatch for $asset (expected $expected_sha, got $actual_sha) -- download deleted, not installing"
fi
info "checksum verified"

# -- Extract + install ---------------------------------------------------------

extract_dir="$workdir/extract"
mkdir -p "$extract_dir"
tar -xzf "$archive" -C "$extract_dir"

binary="$(find "$extract_dir" -type f -name openasr -maxdepth 2 | head -n 1)"
[ -n "$binary" ] || err "extracted archive did not contain an 'openasr' binary"
chmod +x "$binary"

mkdir -p "$prefix"
dest="$prefix/openasr"

if [ -x "$dest" ]; then
  old_version="$("$dest" --version 2>/dev/null || echo unknown)"
  info "found existing install ($old_version) at $dest -- upgrading to $version"
fi

mv "$binary" "$dest"
info "installed $("$dest" --version) to $dest"

case ":$PATH:" in
  *":$prefix:"*) ;;
  *)
    echo
    echo "$prefix is not on your PATH. Add it, e.g.:"
    echo "  export PATH=\"$prefix:\$PATH\""
    echo "(append that line to your shell's rc file to make it permanent)"
    ;;
esac
