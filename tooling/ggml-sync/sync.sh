#!/usr/bin/env bash
# Re-vendor the ggml submodule onto a newer upstream ggml-org/ggml commit,
# replaying the OpenASR-local patches that sit on top of the current pin.
#
# The vendored ggml lives at crates/openasr-core/third_party/openasr-ggml and is
# a git submodule of the QuintinShaw/openasr-ggml fork. Each pin is a clean
# upstream base commit plus a small stack of OpenASR-local patches (Metal
# pipeline-cache persistence + isolation, residency-set default, quiet debug
# logs, GGUF tensor-dim accessors, the Windows backend-dl loader fix, ...). This
# script derives that patch stack from the current pin, checks out the requested
# upstream target, and cherry-picks the stack back on top so the only diff vs
# upstream stays the reviewed OpenASR delta.
#
# It never pushes and never touches upstream or the fork remote: it only writes
# a local branch in the submodule. Bumping the superproject gitlink and any push
# are left to a human.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel)"
sub_path="crates/openasr-core/third_party/openasr-ggml"
sub_dir="${repo_root}/${sub_path}"
upstream_url="https://github.com/ggml-org/ggml.git"

target="upstream/master"
dry_run=0
branch=""

usage() {
  cat <<EOF
Re-vendor the ggml submodule onto a newer upstream commit, replaying the
OpenASR-local patch stack that sits on top of the current pin.

Usage:
  sync.sh [--target <ref>] [--branch <name>] [--dry-run]

Options:
  --target <ref>   Upstream commit/ref to rebase onto (default: upstream/master,
                   i.e. latest after the fetch below). Pass a pinned SHA for a
                   reproducible sync.
  --branch <name>  Name of the local branch to create in the submodule
                   (default: oasr/pin-<short-target>).
  --dry-run        Fetch, resolve the patch stack, and print the plan without
                   creating a branch or cherry-picking.
  -h, --help       Show this help.

After a successful run the submodule sits on the new branch. Review it, then
from the superproject: build + full regression, and only then
  git add ${sub_path}
to record the new gitlink. This script does not commit, bump the gitlink, or push.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="$2"; shift 2 ;;
    --branch) branch="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -d "${sub_dir}/.git" || -f "${sub_dir}/.git" ]] || {
  echo "ggml submodule not initialised at ${sub_path}" >&2
  echo "run: git submodule update --init ${sub_path}" >&2
  exit 2
}

git_sub() { git -C "${sub_dir}" "$@"; }

# Refuse to run over a dirty submodule so a failed cherry-pick can never mix with
# unrelated local edits.
if [[ -n "$(git_sub status --porcelain)" ]]; then
  echo "submodule working tree is dirty; commit/stash before syncing" >&2
  exit 1
fi

pin="$(git_sub rev-parse HEAD)"

if ! git_sub remote get-url upstream >/dev/null 2>&1; then
  echo "== adding upstream remote ${upstream_url}"
  git_sub remote add upstream "${upstream_url}"
fi

echo "== fetching upstream"
git_sub fetch --no-tags upstream master

target_sha="$(git_sub rev-parse --verify "${target}^{commit}")"
base="$(git_sub merge-base "${target_sha}" "${pin}")"

# The OpenASR-local stack = non-merge commits reachable from the current pin but
# not from its clean upstream base. On a clean pin this is exactly the reviewed
# patch stack, oldest first.
patches=()
while IFS= read -r sha; do
  patches+=("${sha}")
done < <(git_sub log --reverse --no-merges --format='%H' "${base}..${pin}")

echo
echo "current pin:   ${pin}"
echo "upstream base: ${base}"
echo "target:        ${target}  (${target_sha})"
echo "patch stack (${#patches[@]} commits, oldest first):"
for sha in "${patches[@]}"; do
  echo "  - $(git_sub log -1 --format='%h %s' "${sha}")"
done

if [[ "${base}" == "${target_sha}" ]]; then
  echo
  echo "target equals the current upstream base; nothing to sync."
  exit 0
fi

if [[ ${#patches[@]} -eq 0 ]]; then
  echo
  echo "no OpenASR-local patches found above the base; the pin is plain upstream." >&2
  echo "refusing to guess -- inspect the pin history manually." >&2
  exit 1
fi

if [[ ${dry_run} -eq 1 ]]; then
  echo
  echo "dry run: not creating a branch or cherry-picking."
  exit 0
fi

branch="${branch:-oasr/pin-$(git_sub rev-parse --short "${target_sha}")}"
echo
echo "== creating ${branch} at ${target_sha} and replaying the stack"
git_sub checkout -b "${branch}" "${target_sha}"

for sha in "${patches[@]}"; do
  short="$(git_sub log -1 --format='%h %s' "${sha}")"
  echo "== cherry-pick ${short}"
  if ! git_sub cherry-pick "${sha}"; then
    echo >&2
    echo "conflict replaying ${short}" >&2
    echo "resolve in ${sub_dir}, then 'git cherry-pick --continue', or --abort to bail." >&2
    echo "if upstream already subsumes this patch, 'git cherry-pick --skip'." >&2
    exit 1
  fi
done

echo
echo "done. submodule ${sub_path} is on ${branch} ($(git_sub rev-parse --short HEAD))."
echo "next: build + full family regression from the superproject, then"
echo "  git add ${sub_path}   # record the new gitlink (not done here)"
