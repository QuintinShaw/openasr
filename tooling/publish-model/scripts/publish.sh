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
# Where the registry card / catalog / signed manifests live. Real runs never
# override this; the seam exists so the flow's checkpoint binding can be tested
# hermetically (mirrors OPENASR_PUBLISH_WORK_ROOT).
REGISTRY_ROOT="${OPENASR_PUBLISH_REGISTRY_ROOT:-$REPO_ROOT/model-registry}"
REGISTRY_ID="$(python3 "$SCRIPT_DIR/_catalog.py" field "$MODEL" registry_id)" \
  || die "model '$MODEL' has no registry_id in the publish catalog"

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

# Content-addressed checkpoints bind a step three ways: its command (step +
# argv), the bytes of files it CONSUMES (declared --inputs), and the bytes of
# files it PRODUCES (declared --outputs). Drift in any of the three re-runs
# the step; an undeclared dimension is not part of the binding.
#
# Binding consumed bytes -- not just argv -- is what makes invalidation
# transitive down the flow: rebuild one pack and its bytes change, so
# materialize re-runs and rewrites the sidecar; the publish/registry steps
# consume pack+sidecar bytes, so they re-run; registry rewrites the card;
# manifest consumes card+sidecar+metrics bytes, so it re-runs and rewrites
# catalog.json; public_catalog consumes catalog.json bytes, so it re-runs.
# The old argv-only checkpoint broke this chain: after a pack rebuild,
# publish/registry/manifest still "matched" and skipped, so the fixed pack
# was never uploaded and the catalog kept the stale sha -- the release-phase
# residue of the exact "rc=0 but wrong bytes shipped" incident class this
# branch removes. Binding produced bytes additionally catches a step's
# artifact being rebuilt in place between runs.
#
# Legacy checkpoints missing a dimension the step now declares (schema v1 has
# no outputs_sha256; early v2 has no inputs_sha256) count as a miss and
# re-run once; the rewritten checkpoint then records both.
fingerprint_files() {
  python3 - "$1" <<'PY'
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

digests = {}
for pattern in sys.argv[1].split():
    for path in glob.glob(pattern):
        if os.path.isfile(path):
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            digests[path] = digest.hexdigest()
print(json.dumps(digests, sort_keys=True))
PY
}

# fingerprint_arg <glob-list>: "null" when the step declares no such
# dimension (skip the check), else the current fingerprint JSON.
fingerprint_arg() {
  if [[ -n "$1" ]]; then
    fingerprint_files "$1"
  else
    printf 'null'
  fi
}

checkpoint_matches() {
  local file="$1"
  local input_sha="$2"
  local inputs_glob="$3"
  local outputs_glob="$4"
  [[ -f "$file" ]] || return 1
  local current_inputs current_outputs
  current_inputs="$(fingerprint_arg "$inputs_glob")"
  current_outputs="$(fingerprint_arg "$outputs_glob")"
  python3 - "$file" "$input_sha" "$current_inputs" "$current_outputs" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_argv = sys.argv[2]
current_inputs = sys.argv[3]
current_outputs = sys.argv[4]
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if data.get("input_sha256") != expected_argv:
    raise SystemExit(1)


def dimension_matches(recorded_key: str, current_text: str) -> bool:
    # "null" = dimension undeclared for this step: not part of the binding.
    # Declared but unrecorded (legacy checkpoint): a miss, re-run once.
    if current_text == "null":
        return True
    return data.get(recorded_key) == json.loads(current_text)


ok = dimension_matches("inputs_sha256", current_inputs) and dimension_matches(
    "outputs_sha256", current_outputs
)
raise SystemExit(0 if ok else 1)
PY
}

write_checkpoint() {
  local file="$1"
  local step="$2"
  local input_sha="$3"
  local inputs_glob="$4"
  local outputs_glob="$5"
  shift 5
  local inputs_fingerprint outputs_fingerprint
  inputs_fingerprint="$(fingerprint_files "$inputs_glob")"
  outputs_fingerprint="$(fingerprint_files "$outputs_glob")"
  python3 - "$file" "$step" "$input_sha" "$inputs_fingerprint" "$outputs_fingerprint" "$@" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
data = {
    "schema_version": 2,
    "step": sys.argv[2],
    "input_sha256": sys.argv[3],
    "inputs_sha256": json.loads(sys.argv[4]),
    "outputs_sha256": json.loads(sys.argv[5]),
    "command": sys.argv[6:],
    "completed_at": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

# run_step <name> <override-var> [--inputs "<glob>..."] [--outputs "<glob>..."] -- <cmd...>
run_step() {
  local step="$1"
  local override_var="$2"
  shift 2
  local inputs_glob=""
  local outputs_glob=""
  while [[ "${1:-}" == "--inputs" || "${1:-}" == "--outputs" ]]; do
    case "$1" in
      --inputs)
        inputs_glob="${2:?--inputs requires a glob list}"
        shift 2
        ;;
      --outputs)
        outputs_glob="${2:?--outputs requires a glob list}"
        shift 2
        ;;
    esac
  done
  [[ "${1:-}" == "--" ]] && shift
  local -a command=("$@")
  local override="${!override_var:-}"
  if [[ -n "$override" ]]; then
    command=("$override" "${command[@]}")
  fi
  local input_sha
  input_sha="$(hash_args "$step" "${command[@]}")"
  local checkpoint="$CHECKPOINT_DIR/$step.done.json"
  if [[ "$FORCE" != "1" ]] &&
    checkpoint_matches "$checkpoint" "$input_sha" "$inputs_glob" "$outputs_glob"; then
    log "skip $step (checkpoint)"
    return 0
  fi
  log "run $step"
  "${command[@]}"
  write_checkpoint "$checkpoint" "$step" "$input_sha" "$inputs_glob" "$outputs_glob" "${command[@]}"
}

quant_args=()
for quant in "${QUANTS[@]}"; do
  quant_args+=(--quant "$quant")
done

# File sets the checkpoints bind, named after the flow's data dependencies:
PACK_GLOB="$WORK_ROOT/packs/$MODEL-*.oasr"
SIDECAR_GLOB="$WORK_ROOT/packs/$MODEL.*.result.json"
TOOLING_ROOT="$SCRIPT_DIR/.."

# Consumes the packs, produces the sidecars. Rebuild a pack (e.g. after a
# quantization-policy fix) and its bytes change, so the checkpoint no longer
# matches and materialize re-runs instead of silently keeping the stale
# sidecar -- the "--reset-checkpoints required by hand" workaround the q4_k
# incident forced.
run_step \
  materialize_results \
  OPENASR_PUBLISH_MATERIALIZE_CMD \
  --inputs "$PACK_GLOB" \
  --outputs "$SIDECAR_GLOB" \
  -- \
  python3 "$SCRIPT_DIR/materialize_result_sidecars.py" "$MODEL" "${quant_args[@]}"

for target in "${TARGETS[@]}"; do
  target_args=(--model "$MODEL" "${quant_args[@]}" --target "$target")
  if [[ "$DRY_RUN" == "1" ]]; then
    target_args+=(--dry-run)
  fi
  # Uploads the packs described by the sidecars; records the resolved repo
  # and immutable revision the manifest later binds into catalog.json.
  run_step \
    "publish_$target" \
    OPENASR_PUBLISH_TARGET_CMD \
    --inputs "$PACK_GLOB $SIDECAR_GLOB" \
    --outputs "$WORK_ROOT/hf_repo.txt $WORK_ROOT/hf_revision.txt" \
    -- \
    python3 "$SCRIPT_DIR/publish_model_targets.py" "${target_args[@]}"
done

if [[ "$DRY_RUN" == "1" ]]; then
  log "dry run complete; registry, manifest, and catalog signing were not changed"
  exit 0
fi

# _registry.py reads the publish-catalog sources plus the resolved HF repo
# and writes the registry card.
run_step \
  registry \
  OPENASR_PUBLISH_REGISTRY_CMD \
  --inputs "$WORK_ROOT/hf_repo.txt $TOOLING_ROOT/models-core.toml $TOOLING_ROOT/models-publish.toml" \
  --outputs "$REGISTRY_ROOT/models/$REGISTRY_ID.toml" \
  -- \
  python3 "$SCRIPT_DIR/_registry.py" "$MODEL"

manifest_args=("$MODEL")
if [[ "$PUBLIC" == "1" ]]; then
  manifest_args+=(--public)
fi
# _manifest.py bakes the sidecar sha256/size + metrics + resolved revision +
# prose card into catalog.json (upsert over the current catalog). Consuming
# the sidecar bytes here is the pack-sha link into the signed catalog.
run_step \
  manifest \
  OPENASR_PUBLISH_MANIFEST_CMD \
  --inputs "$SIDECAR_GLOB $WORK_ROOT/metrics.json $WORK_ROOT/hf_repo.txt $WORK_ROOT/hf_revision.txt $TOOLING_ROOT/cards/$MODEL.toml $REGISTRY_ROOT/catalog.json" \
  --outputs "$REGISTRY_ROOT/catalog.json" \
  -- \
  python3 "$SCRIPT_DIR/_manifest.py" "${manifest_args[@]}"

if [[ "$PUBLIC" == "1" && "$PUBLISH_CATALOG" == "1" ]]; then
  # publish_catalog.sh signs the committed catalog (full + public projection).
  run_step \
    public_catalog \
    OPENASR_PUBLISH_CATALOG_CMD \
    --inputs "$REGISTRY_ROOT/catalog.json $REGISTRY_ROOT/catalog.epoch" \
    --outputs "$REGISTRY_ROOT/catalog.signature.json $REGISTRY_ROOT/catalog.public.json $REGISTRY_ROOT/catalog.public.signature.json" \
    -- \
    "$SCRIPT_DIR/publish_catalog.sh"
fi

log "publish flow complete for $MODEL"
