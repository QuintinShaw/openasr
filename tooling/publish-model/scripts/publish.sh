#!/usr/bin/env bash
# Resumable public release driver for OpenASR model packs.
#
# Default lane: qwen3-asr-0.6b, fp16/q8_0/q4_k, Hugging Face only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

log() { printf '\033[1;36m[publish]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[publish:err]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  cat >&2 <<'EOF'
usage: publish.sh [--model <id>] [--quant <quant>] [--target hf] [--targets hf]
                  [--public] [--dry-run] [--force] [--reset-checkpoints]
                  [--no-publish-catalog]

Runs the resumable OpenASR model publishing flow:
  materialize result sidecars -> publish each target -> registry -> manifest -> signed public catalog

Defaults:
  --model qwen3-asr-0.6b
  --quant fp16 --quant q8_0 --quant q4_k
  --target hf

Environment:
  HF_TOKEN is required for real Hugging Face publishing.
  OPENASR_CATALOG_SIGNING_KEY_SEED_HEX is required when signing the public catalog.
EOF
}

MODEL="qwen3-asr-0.6b"
QUANTS=()
TARGETS=()
PUBLIC=0
DRY_RUN=0
FORCE=0
RESET_CHECKPOINTS=0
PUBLISH_CATALOG=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="${2:?--model requires <id>}"
      shift 2
      ;;
    --quant)
      QUANTS+=("${2:?--quant requires <quant>}")
      shift 2
      ;;
    --target)
      TARGETS+=("${2:?--target requires hf}")
      shift 2
      ;;
    --targets)
      IFS=',' read -r -a parsed_targets <<< "${2:?--targets requires comma-separated targets}"
      TARGETS+=("${parsed_targets[@]}")
      shift 2
      ;;
    --public)
      PUBLIC=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --reset-checkpoints)
      RESET_CHECKPOINTS=1
      shift
      ;;
    --no-publish-catalog)
      PUBLISH_CATALOG=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "unknown option: $1"
      ;;
  esac
done

if [[ "${#QUANTS[@]}" -eq 0 ]]; then
  QUANTS=(fp16 q8_0 q4_k)
fi
if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  TARGETS=(hf)
fi
for target in "${TARGETS[@]}"; do
  case "$target" in
    hf) ;;
    *) die "unsupported publish target: $target" ;;
  esac
done

WORK_ROOT="${OPENASR_PUBLISH_WORK_ROOT:-$REPO_ROOT/tmp/publish/$MODEL}"
CHECKPOINT_DIR="$WORK_ROOT/checkpoints"

if [[ "$RESET_CHECKPOINTS" == "1" ]]; then
  rm -rf "$CHECKPOINT_DIR"
fi
mkdir -p "$CHECKPOINT_DIR"

hash_args() {
  python3 - "$@" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys

payload = json.dumps(sys.argv[1:], ensure_ascii=False, separators=(",", ":")).encode()
print(hashlib.sha256(payload).hexdigest())
PY
}

checkpoint_matches() {
  local file="$1"
  local input_sha="$2"
  [[ -f "$file" ]] || return 1
  python3 - "$file" "$input_sha" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = sys.argv[2]
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if data.get("input_sha256") == expected else 1)
PY
}

write_checkpoint() {
  local file="$1"
  local step="$2"
  local input_sha="$3"
  shift 3
  python3 - "$file" "$step" "$input_sha" "$@" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
data = {
    "schema_version": 1,
    "step": sys.argv[2],
    "input_sha256": sys.argv[3],
    "command": sys.argv[4:],
    "completed_at": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_step() {
  local step="$1"
  local override_var="$2"
  shift 2
  local -a command=("$@")
  local override="${!override_var:-}"
  if [[ -n "$override" ]]; then
    command=("$override" "${command[@]}")
  fi
  local input_sha
  input_sha="$(hash_args "$step" "${command[@]}")"
  local checkpoint="$CHECKPOINT_DIR/$step.done.json"
  if [[ "$FORCE" != "1" ]] && checkpoint_matches "$checkpoint" "$input_sha"; then
    log "skip $step (checkpoint)"
    return 0
  fi
  log "run $step"
  "${command[@]}"
  write_checkpoint "$checkpoint" "$step" "$input_sha" "${command[@]}"
}

quant_args=()
for quant in "${QUANTS[@]}"; do
  quant_args+=(--quant "$quant")
done

run_step \
  materialize_results \
  OPENASR_PUBLISH_MATERIALIZE_CMD \
  python3 "$SCRIPT_DIR/materialize_result_sidecars.py" "$MODEL" "${quant_args[@]}"

for target in "${TARGETS[@]}"; do
  target_args=(--model "$MODEL" "${quant_args[@]}" --target "$target")
  if [[ "$DRY_RUN" == "1" ]]; then
    target_args+=(--dry-run)
  fi
  run_step \
    "publish_$target" \
    OPENASR_PUBLISH_TARGET_CMD \
    python3 "$SCRIPT_DIR/publish_model_targets.py" "${target_args[@]}"
done

if [[ "$DRY_RUN" == "1" ]]; then
  log "dry run complete; registry, manifest, and catalog signing were not changed"
  exit 0
fi

run_step \
  registry \
  OPENASR_PUBLISH_REGISTRY_CMD \
  python3 "$SCRIPT_DIR/_registry.py" "$MODEL"

manifest_args=("$MODEL")
if [[ "$PUBLIC" == "1" ]]; then
  manifest_args+=(--public)
fi
run_step \
  manifest \
  OPENASR_PUBLISH_MANIFEST_CMD \
  python3 "$SCRIPT_DIR/_manifest.py" "${manifest_args[@]}"

if [[ "$PUBLIC" == "1" && "$PUBLISH_CATALOG" == "1" ]]; then
  run_step \
    public_catalog \
    OPENASR_PUBLISH_CATALOG_CMD \
    "$SCRIPT_DIR/publish_catalog.sh"
fi

log "publish flow complete for $MODEL"
