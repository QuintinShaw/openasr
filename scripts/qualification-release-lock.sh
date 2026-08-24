#!/usr/bin/env bash
# Cross-process mutex for draft qualification-manifest/signature mutations.
# GitHub Release asset creation is the atomic compare-and-create operation;
# callers retain a random ownership token and may delete only their own lock.

set -euo pipefail
umask 077

fail() {
  printf 'qualification release lock: %s\n' "$1" >&2
  exit 1
}

[ "$#" -eq 3 ] || fail "usage: $(basename "$0") <acquire|release> vX.Y.Z <token-file>"
action="$1"
tag="$2"
token_file="$3"
version="${tag#v}"
[[ "$tag" = "v${version}" && "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
  || fail "tag must be canonical vX.Y.Z"
repository="QuintinShaw/openasr"
asset_name="openasr-${version}-qualification-mutation.lock"
command -v gh >/dev/null 2>&1 || fail "gh is required"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"

temporary="$(mktemp -d "${TMPDIR:-/tmp}/openasr-qualification-lock.XXXXXX")"
cleanup() {
  rm -rf -- "$temporary"
}
trap cleanup EXIT

case "$action" in
  acquire)
    [ ! -e "$token_file" ] && [ ! -L "$token_file" ] \
      || fail "token file already exists: $token_file"
    token="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
    [[ "$token" =~ ^[0-9a-f]{64}$ ]] || fail "could not generate an ownership token"
    printf '%s\n' "$token" > "$token_file"
    cp -- "$token_file" "$temporary/$asset_name"
    if ! gh release upload "$tag" "$temporary/$asset_name" --repo "$repository"; then
      rm -f -- "$token_file"
      fail "another signer/uploader owns $asset_name, or the lock could not be created"
    fi
    ;;
  release)
    [ -s "$token_file" ] || fail "ownership token is missing: $token_file"
    gh release download "$tag" --repo "$repository" -p "$asset_name" \
      -D "$temporary" --clobber >/dev/null
    cmp -s -- "$token_file" "$temporary/$asset_name" \
      || fail "refusing to delete a qualification lock owned by another process"
    gh release delete-asset "$tag" "$asset_name" --repo "$repository" --yes
    rm -f -- "$token_file"
    ;;
  *) fail "action must be acquire or release" ;;
esac
