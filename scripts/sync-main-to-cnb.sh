#!/usr/bin/env bash
# Fast-forward GitHub `main` onto CNB `main`. Never mirror, prune, or force.
#
# CNB also holds desktop-v* tags that do not exist on GitHub. A mirror push
# would delete those refs. This script only updates refs/heads/main.
#
# Usage:
#   scripts/sync-main-to-cnb.sh [commit-ish]
#
# Default source: origin/main locally, HEAD in GitHub Actions (must be
# refs/heads/main). Feature-branch HEAD is never the implicit source.
#
# Env:
#   CNB_TOKEN            required (git username is always `cnb`)
#   OPENASR_CNB_REPO     default openasr/openasr
#   OPENASR_CNB_STRICT=1 fail even when this is a notice-style helper
#                        (default: fail; this is not a release sidecar)

set -euo pipefail

notice() { printf 'CNB main sync: %s\n' "$1"; }
fail() {
  printf 'CNB main sync failed: %s\n' "$1" >&2
  exit 1
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "usage: $(basename "$0") [commit-ish]" >&2
  exit 2
fi

if [ -z "${CNB_TOKEN:-}" ]; then
  fail "CNB_TOKEN is not set"
fi

if [ "${GITHUB_ACTIONS:-}" = "true" ] && [ "${GITHUB_REF:-}" != "refs/heads/main" ]; then
  fail "refusing to update CNB main from ${GITHUB_REF:-unknown-ref}"
fi

CNB_REPO="${OPENASR_CNB_REPO:-openasr/openasr}"
CNB_GIT="https://cnb.cool/${CNB_REPO}"
source_ref="${1:-}"
if [ -z "$source_ref" ]; then
  if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    source_ref="HEAD"
  elif git rev-parse --verify --quiet origin/main >/dev/null; then
    source_ref="origin/main"
  else
    fail "no origin/main; pass an explicit commit-ish"
  fi
fi

commit="$(git rev-parse "${source_ref}^{commit}")"
notice "source ${source_ref} = ${commit}"

askpass="$(mktemp "${TMPDIR:-/tmp}/openasr-cnb-askpass.XXXXXX")"
cleanup() {
  rm -f "$askpass"
  git update-ref -d refs/cnb-sync/main >/dev/null 2>&1 || true
}
trap cleanup EXIT
printf '%s\n' '#!/bin/sh' 'case "$1" in *[Uu]sername*) echo cnb ;; *) printf "%s\n" "$CNB_TOKEN" ;; esac' > "$askpass"
chmod 700 "$askpass"

export GIT_TERMINAL_PROMPT=0
export GIT_ASKPASS="$askpass"

cnb_main="$(git ls-remote --heads "$CNB_GIT" refs/heads/main | awk '{print $1}')"
if [ -n "$cnb_main" ]; then
  notice "CNB main is ${cnb_main}"
  if [ "$cnb_main" = "$commit" ]; then
    notice "already in sync"
    exit 0
  fi
  git fetch --no-tags "$CNB_GIT" "+refs/heads/main:refs/cnb-sync/main"
  if ! git merge-base --is-ancestor refs/cnb-sync/main "$commit"; then
    fail "CNB main ${cnb_main} is not an ancestor of ${commit}; refusing a non-ff update"
  fi
else
  notice "CNB has no main yet; creating refs/heads/main"
fi

if ! git push --no-follow-tags "$CNB_GIT" "${commit}:refs/heads/main"; then
  fail "git push of main was rejected"
fi

notice "updated CNB refs/heads/main -> ${commit}"
