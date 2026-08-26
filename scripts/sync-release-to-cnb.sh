#!/usr/bin/env bash
# Idempotent GitHub Release → CNB (cnb.cool) asset sync.
#
# GitHub + B2 remain the only build/sign sources. CNB is a read-only China
# mirror. Missing CNB_TOKEN skips (exit 0) so overseas publish stays green.
# A CNB failure after auth is a notice unless OPENASR_CNB_STRICT=1.
#
# Usage:
#   scripts/sync-release-to-cnb.sh <tag> [file ...]
#
# If no files are given, downloads every asset of GitHub release <tag> into a
# temp dir and uploads those. Desktop callers still pass extra local files
# (macOS .app.tar.gz plus its .sig) so the minisign signature, which is not
# a GitHub asset, lands on CNB.
#
# Env:
#   CNB_TOKEN            required for upload (Bearer)
#   OPENASR_CNB_REPO     default openasr/openasr
#   OPENASR_CNB_API      default https://api.cnb.cool
#   OPENASR_CNB_STRICT=1 fail the caller on CNB errors
#   OPENASR_CNB_GITHUB_REPO  git+release source (default QuintinShaw/openasr).
#                            Never inherit GITHUB_REPOSITORY: Actions sets that
#                            to the host repo, which is not always this public
#                            open core. CNB tags must come from here.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

notice() { printf 'CNB sync: %s\n' "$1"; }
fail() {
  printf 'CNB sync failed: %s\n' "$1" >&2
  if [ "${OPENASR_CNB_STRICT:-}" = "1" ]; then
    exit 1
  fi
  exit 0
}

if [ "${1:-}" = "" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "usage: $(basename "$0") <tag> [file ...]" >&2
  exit 2
fi

tag="$1"
shift

if [ -z "${CNB_TOKEN:-}" ]; then
  notice "skipped (no CNB_TOKEN)"
  exit 0
fi

CNB_REPO="${OPENASR_CNB_REPO:-openasr/openasr}"
CNB_API="${OPENASR_CNB_API:-https://api.cnb.cool}"
GITHUB_REPO="${OPENASR_CNB_GITHUB_REPO:-QuintinShaw/openasr}"
ACCEPT="application/vnd.cnb.api+json"

cnb_curl() {
  curl -sS --connect-timeout 15 --max-time 120 \
    -H "Accept: ${ACCEPT}" \
    -H "Authorization: Bearer ${CNB_TOKEN}" \
    "$@"
}

# Empty CNB repos have no `main`; a Release still needs a git tag object.
# Fetch the GitHub tag into a throwaway repo and push it (username is always `cnb`).
# Shallow fetches are rejected (`shallow update not allowed`).
cnb_push_tag() {
  local askpass tmpgit
  tmpgit="$(mktemp -d "${TMPDIR:-/tmp}/openasr-cnb-git.XXXXXX")"
  askpass="$(mktemp "${TMPDIR:-/tmp}/openasr-cnb-askpass.XXXXXX")"
  printf '%s\n' '#!/bin/sh' 'case "$1" in *[Uu]sername*) echo cnb ;; *) printf "%s\n" "$CNB_TOKEN" ;; esac' > "$askpass"
  chmod 700 "$askpass"
  git init --bare "$tmpgit" >/dev/null
  if ! git --git-dir="$tmpgit" fetch --no-tags \
      "https://github.com/${GITHUB_REPO}.git" "refs/tags/${tag}:refs/tags/${tag}"; then
    rm -rf "$tmpgit" "$askpass"
    fail "could not fetch GitHub tag ${tag}"
  fi
  target_commitish="$(git --git-dir="$tmpgit" rev-parse "refs/tags/${tag}^{commit}")"
  if ! GIT_TERMINAL_PROMPT=0 GIT_ASKPASS="$askpass" \
      git --git-dir="$tmpgit" push "https://cnb.cool/${CNB_REPO}" \
      "refs/tags/${tag}:refs/tags/${tag}"; then
    rm -rf "$tmpgit" "$askpass"
    fail "could not push tag ${tag} to CNB"
  fi
  rm -rf "$tmpgit" "$askpass"
  notice "pushed git tag ${tag} to CNB (${target_commitish})"
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

file_size() {
  python3 -c 'import os,sys; print(os.path.getsize(sys.argv[1]))' "$1"
}

# `gh release download` has no retry and has truncated 60MB+ assets with
# "unexpected EOF". Pull the public GitHub asset URL with curl, matching
# the CNB PUT retry policy, then refuse a size mismatch.
download_github_asset() {
  local name="$1"
  local size="$2"
  local dest="$3"
  local url="https://github.com/${GITHUB_REPO}/releases/download/${tag}/${name}"
  local attempt got
  for attempt in 1 2 3; do
    rm -f "$dest"
    if curl --fail --show-error --location --progress-bar \
        --connect-timeout 30 --max-time 7200 \
        --retry 3 --retry-delay 5 --retry-all-errors \
        --output "$dest" "$url"; then
      got="$(file_size "$dest")"
      if [ "$got" = "$size" ]; then
        return 0
      fi
      echo "GitHub download size mismatch for ${name}: ${got} != ${size}" >&2
    else
      echo "GitHub download failed for ${name} (attempt ${attempt}/3)" >&2
    fi
    sleep $((attempt * 5))
  done
  rm -f "$dest"
  return 1
}

json_field() {
  python3 -c '
import json,sys
raw=sys.stdin.read().strip()
if not raw:
    raise SystemExit(0)
try:
    data=json.loads(raw)
except json.JSONDecodeError:
    raise SystemExit(0)
print(data.get(sys.argv[1]) or "")
' "$1"
}

# Swagger: GET /{repo}/-/releases/tags/{tag} ; POST /{repo}/-/releases
# (query-string ?tag_name= is not an OpenAPI path).
tag_path="$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$tag")"
api_releases="${CNB_API}/${CNB_REPO}/-/releases"

lookup_response="$(
  curl -sS --connect-timeout 15 --max-time 120 -w "\n%{http_code}" \
    "${api_releases}/tags/${tag_path}" \
    -H "Accept: ${ACCEPT}" \
    -H "Authorization: Bearer ${CNB_TOKEN}" || true
)"
lookup_status="$(printf '%s' "$lookup_response" | tail -n1)"
lookup_body="$(printf '%s' "$lookup_response" | sed '$d')"
release_id=""

if [ "$lookup_status" = "200" ]; then
  release_id="$(printf '%s' "$lookup_body" | json_field id)"
  notice "found existing CNB release ${tag}"
elif [ "$lookup_status" = "404" ]; then
  # make_latest stays false: core and desktop tags share one CNB repo; "latest"
  # is not the download identity (install.sh / accel JSON pin explicit tags).
  target_commitish=""
  cnb_push_tag
  create_body="$(python3 -c 'import json,sys; print(json.dumps({"tag_name":sys.argv[1],"name":sys.argv[1],"body":"OpenASR China mirror of GitHub "+sys.argv[1],"draft":False,"prerelease":False,"make_latest":"false","target_commitish":sys.argv[2]}))' "$tag" "$target_commitish")"
  create_response="$(
    curl -sS --connect-timeout 15 --max-time 120 -w "\n%{http_code}" \
      -X POST "${api_releases}" \
      -H "Accept: ${ACCEPT}" \
      -H "Authorization: Bearer ${CNB_TOKEN}" \
      -H "Content-Type: application/json" \
      -d "$create_body" || true
  )"
  create_status="$(printf '%s' "$create_response" | tail -n1)"
  create_body_json="$(printf '%s' "$create_response" | sed '$d')"
  if [ "$create_status" != "201" ]; then
    fail "create CNB release ${tag} HTTP ${create_status}: ${create_body_json:0:240}"
  fi
  release_id="$(printf '%s' "$create_body_json" | json_field id)"
  notice "created CNB release ${tag}"
else
  fail "lookup CNB release ${tag} HTTP ${lookup_status}: ${lookup_body:0:240}"
fi

if [ -z "$release_id" ]; then
  fail "could not create or find CNB release ${tag} on ${CNB_REPO}"
fi

release_detail="$(cnb_curl "${api_releases}/${release_id}" || true)"

workdir="$(mktemp -d "${TMPDIR:-/tmp}/openasr-cnb-sync.XXXXXX")"
trap 'rm -rf "$workdir"' EXIT

already_on_cnb() {
  local name="$1"
  local size="$2"
  printf '%s' "$release_detail" | python3 -c '
import json,sys
raw=sys.stdin.read().strip()
name, size = sys.argv[1], int(sys.argv[2])
try:
    data=json.loads(raw) if raw else {}
except json.JSONDecodeError:
    raise SystemExit(1)
for asset in data.get("assets") or []:
    if asset.get("name")==name and int(asset.get("size") or -1)==size:
        raise SystemExit(0)
raise SystemExit(1)
' "$name" "$size"
}

upload_one() {
  local path="$1"
  local name size upload_response upload_status upload_body upload_url verify_url verify_status
  name="$(basename "$path")"
  size="$(wc -c < "$path" | tr -d ' ')"
  if already_on_cnb "$name" "$size"; then
    notice "skip ${name} (already on CNB, size match)"
    return 0
  fi

  upload_response="$(
    curl -sS --connect-timeout 15 --max-time 120 -w "\n%{http_code}" \
      -X POST "${CNB_API}/${CNB_REPO}/-/releases/${release_id}/asset-upload-url" \
      -H "Accept: ${ACCEPT}" \
      -H "Authorization: Bearer ${CNB_TOKEN}" \
      -H "Content-Type: application/json" \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"asset_name":sys.argv[1],"size":int(sys.argv[2]),"overwrite":True,"ttl":0}))' "$name" "$size")"
  )"
  upload_status="$(printf '%s' "$upload_response" | tail -n1)"
  upload_body="$(printf '%s' "$upload_response" | sed '$d')"
  if [ "$upload_status" != "201" ] && [ "$upload_status" != "200" ]; then
    echo "CNB asset-upload-url failed for ${name} (HTTP ${upload_status}): ${upload_body}" >&2
    return 1
  fi
  upload_url="$(printf '%s' "$upload_body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["upload_url"])')"
  verify_url="$(printf '%s' "$upload_body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["verify_url"])')"
  curl --fail-with-body --show-error --location --progress-bar \
    --connect-timeout 30 --max-time 7200 \
    --retry 3 --retry-delay 5 --retry-all-errors \
    -X PUT --upload-file "$path" "$upload_url"
  verify_status="$(
    curl -sS --connect-timeout 15 --max-time 120 -o /dev/null -w "%{http_code}" \
      -X POST "$verify_url" \
      -H "Accept: ${ACCEPT}" \
      -H "Authorization: Bearer ${CNB_TOKEN}"
  )"
  case "$verify_status" in
    200|201|204) ;;
    *)
      echo "CNB did not confirm ${name} (HTTP ${verify_status})" >&2
      return 1
      ;;
  esac
  notice "uploaded ${name} sha256=$(sha256_of "$path")"
}

errors=0
synced=0
if [ "$#" -eq 0 ]; then
  if ! command -v gh >/dev/null 2>&1; then
    fail "gh is required to list GitHub release assets when no files are given"
  fi
  # Stream one GitHub asset at a time so a 3GB+ core release does not have to
  # land on disk in full before the first CNB PUT.
  while IFS=$'\t' read -r name size; do
    [ -n "$name" ] || continue
    if already_on_cnb "$name" "$size"; then
      notice "skip ${name} (already on CNB, size match)"
      synced=$((synced + 1))
      continue
    fi
    notice "downloading ${name} from GitHub (${size} bytes)"
    path="$workdir/$name"
    if ! download_github_asset "$name" "$size" "$path"; then
      echo "GitHub download failed for ${name}" >&2
      errors=$((errors + 1))
      continue
    fi
    if upload_one "$path"; then
      synced=$((synced + 1))
    else
      errors=$((errors + 1))
    fi
    rm -f "$path"
  done < <(gh release view "$tag" --repo "$GITHUB_REPO" --json assets --jq '.assets[] | select(.state=="uploaded") | [.name, (.size|tostring)] | @tsv')
else
  for path in "$@"; do
    [ -f "$path" ] || fail "not a file: $path"
    if upload_one "$path"; then
      synced=$((synced + 1))
    else
      errors=$((errors + 1))
    fi
  done
fi

if [ "$errors" -ne 0 ]; then
  fail "${errors} asset(s) failed for ${tag}"
fi
if [ "$synced" -eq 0 ]; then
  notice "no files to upload for ${tag}"
  exit 0
fi
notice "synced ${synced} file(s) to ${CNB_REPO} @ ${tag}"
