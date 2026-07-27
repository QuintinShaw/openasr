#!/usr/bin/env bash
# Shared helpers for the OpenASR publishing harness. Source, don't execute.
#
#   source "$(dirname "$0")/lib.sh"
#
# Conventions: every stage works under tmp/publish/<model>/ (gitignored,
# host-local) and is individually re-runnable. Downloads use the hf-mirror
# endpoint; uploads use the real Hugging Face API.

set -euo pipefail

PUB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The harness lives INSIDE the OpenASR repo, so the repo root resolves from
# this script's own location -- stages work from any cwd (the old
# caller-cwd assumption broke every run not launched from the checkout root).
# OPENASR_REPO_ROOT remains the explicit override.
REPO_ROOT="${OPENASR_REPO_ROOT:-$(git -C "$PUB_DIR" rev-parse --show-toplevel)}"
CATALOG_PY="$PUB_DIR/_catalog.py"

# hf-mirror for downloads only; uploads must hit the real hub.
HF_MIRROR_ENDPOINT="https://hf-mirror.com"

# Per-model working tree under the gitignored tmp/.
work_root()   { echo "$REPO_ROOT/tmp/publish/$1"; }
src_dir()     { echo "$(work_root "$1")/src"; }
packs_dir()   { echo "$(work_root "$1")/packs"; }
repo_dir()    { echo "$(work_root "$1")/repo"; }      # staged HF upload folder
metrics_json(){ echo "$(work_root "$1")/metrics.json"; }

# Catalog accessors.
cat_field()  { python3 "$CATALOG_PY" field "$1" "$2"; }
# Optional catalog fields: empty output when the key is absent (never dies).
cat_field_opt() { python3 "$CATALOG_PY" field "$1" "$2" 2>/dev/null || true; }
# Optional LIST-valued catalog fields, one item per line (items may contain
# spaces -- prep scripts / command templates -- where `field`'s space-join
# would corrupt them). Usage: mapfile -t items < <(cat_lines <model> <key>)
cat_lines() { python3 "$CATALOG_PY" field-lines "$1" "$2"; }
cat_quants() { python3 "$CATALOG_PY" quants "$1"; }
quant_token(){ python3 "$CATALOG_PY" token "$1"; }    # q8_0 -> q8-0
quant_suffix(){ python3 "$CATALOG_PY" suffix "$1"; }  # q8_0 -> q8

# Fail-closed completeness guard: every glob pattern must match at least one
# existing file under the given dir. This is the wall between "a stage
# returned rc=0" and "the files the next stage needs actually exist" -- the
# class of bug where an exclude rule strips the only checkpoint (or a
# tokenless fetch lands a 29-byte error page) while the exit code says fine.
require_files() {
  local dir="$1"; shift
  [[ $# -gt 0 ]] || return 0
  python3 - "$dir" "$@" <<'PY'
import glob
import os
import sys

root, patterns = sys.argv[1], sys.argv[2:]
missing = [pattern for pattern in patterns if not glob.glob(os.path.join(root, pattern))]
if missing:
    sys.stderr.write(
        "required source file(s) missing under %s: %s\n" % (root, ", ".join(missing))
    )
    sys.exit(1)
PY
}

# retry <attempts> <cmd...>: run a command until it succeeds with exponential
# backoff. hf-mirror is flaky; a transient failure must not abort a
# multi-hour release run -- and exhausting the attempts must fail loudly.
retry() {
  local attempts="$1"; shift
  local delay=10 attempt=1
  while true; do
    if "$@"; then return 0; fi
    (( attempt >= attempts )) || { log "attempt $attempt/$attempts failed; retrying in ${delay}s"; }
    (( attempt >= attempts )) && return 1
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 3))
  done
}

# Expand {key} placeholders in a recipe template. Args after the template are
# KEY=VALUE pairs; {KEY} is replaced with VALUE. Emits the expanded string.
expand_template() {
  local template="$1"; shift
  local out="$template" pair key
  for pair in "$@"; do
    key="{${pair%%=*}}"
    out="${out//"$key"/${pair#*=}}"
  done
  printf '%s\n' "$out"
}

# The pack filename for one (model, quant): qwen3-asr-1.7b-q8_0.oasr
pack_file() { echo "$(packs_dir "$1")/$1-$2.oasr"; }

# The release CLI binary (built once; bench-suite + imports use it).
openasr_bin() {
  local bin="$REPO_ROOT/target/release/openasr"
  if [[ ! -x "$bin" ]]; then
    echo "release binary missing; building..." >&2
    (cd "$REPO_ROOT" && cargo build --release -p openasr-cli >&2)
  fi
  echo "$bin"
}

# Resolve the modern Hugging Face `hf` CLI. Prefer the explicit CLI installed by
# https://hf.co/cli/install.sh; do not fall back to the legacy
# `huggingface-cli` command for publish/download side effects.
hf_cli() {
  if [[ -n "${HF_CLI_BIN:-}" && -x "$HF_CLI_BIN" ]]; then echo "$HF_CLI_BIN"; return; fi
  if [[ -x "$HOME/.local/bin/hf" ]]; then echo "$HOME/.local/bin/hf"; return; fi
  if command -v hf >/dev/null 2>&1; then command -v hf; return; fi
  echo "ERR_NO_HF_CLI"
}

# A Python interpreter with huggingface_hub importable (for helper scripts like
# _hf_ns.py). Prefer the system python if the package is already installed; else
# run via uvx so no global install is required.
hf_py() {
  if [[ -x "$HOME/.hf-cli/venv/bin/python" ]] \
    && "$HOME/.hf-cli/venv/bin/python" -c "import huggingface_hub, socksio" >/dev/null 2>&1; then
    echo "$HOME/.hf-cli/venv/bin/python"; return
  fi
  # Require socksio too: a SOCKS proxy (HTTPS_PROXY/ALL_PROXY) makes httpx need it,
  # and the system python may have huggingface_hub but not socksio.
  if python3 -c "import huggingface_hub, socksio" >/dev/null 2>&1; then echo "python3"; return; fi
  if command -v uvx >/dev/null 2>&1; then
    echo "uvx --from huggingface_hub --with socksio python"; return
  fi
  echo "python3"
}

log()  { printf '\033[1;36m[publish]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[publish:err]\033[0m %s\n' "$*" >&2; exit 1; }
