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
cat_quants() { python3 "$CATALOG_PY" quants "$1"; }
quant_token(){ python3 "$CATALOG_PY" token "$1"; }    # q8_0 -> q8-0
quant_suffix(){ python3 "$CATALOG_PY" suffix "$1"; }  # q8_0 -> q8

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
