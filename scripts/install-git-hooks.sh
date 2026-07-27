#!/usr/bin/env bash
# Point the checkout's git hooks at the repo-tracked .githooks/ directory, so
# the catalog consistency pre-commit guard (and anything added there later)
# runs for this clone. Opt-in: git does not auto-activate tracked hooks.
#
#   scripts/install-git-hooks.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

git -C "$REPO_ROOT" config core.hooksPath .githooks
echo "git hooks path set to .githooks for $REPO_ROOT"
