#!/usr/bin/env bash
# Regenerate/check registry cards and catalog entries from repo-owned tooling
# model metadata.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
CATALOG_PY="$SCRIPT_DIR/_catalog.py"

usage() {
  cat >&2 <<'EOF'
usage: regenerate_all.sh [--check] [--public] [model-id ...]

Regenerates model-registry/models/<id>.toml and model-registry/catalog.json from
tooling/publish-model/models-core.toml plus models-publish.toml. With --check,
runs a CI-safe drift gate that does not require tmp/publish evidence.
EOF
}

log() { printf '\033[1;36m[publish]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[publish:err]\033[0m %s\n' "$*" >&2; exit 1; }
cat_field() { python3 "$CATALOG_PY" field "$1" "$2"; }

CHECK=0
PUBLIC=0
MODELS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      CHECK=1
      shift
      ;;
    --public)
      PUBLIC=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      usage
      die "unknown option: $1"
      ;;
    *)
      MODELS+=("$1")
      shift
      ;;
  esac
done

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  while IFS= read -r model; do
    MODELS+=("$model")
  done < <(python3 "$CATALOG_PY" models)
fi

if [[ "$CHECK" -eq 1 ]]; then
  python3 "$SCRIPT_DIR/check_catalog_drift.py" "${MODELS[@]}"
  exit 0
fi

catalog_entry_is_public() {
  local registry_id="$1"
  python3 - "$REPO_ROOT/model-registry/catalog.json" "$registry_id" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
registry_id = sys.argv[2]
if not path.exists():
    sys.exit(1)
data = json.loads(path.read_text())
for model in data.get("models", []):
    if isinstance(model, dict) and model.get("id") == registry_id:
        sys.exit(0 if model.get("public") is True else 1)
sys.exit(1)
PY
}

for model in "${MODELS[@]}"; do
  registry_id="$(cat_field "$model" registry_id)"
  if [[ "$PUBLIC" -ne 1 ]] && catalog_entry_is_public "$registry_id"; then
    die "$model is already public:true in catalog; rerun with --public to preserve the public gate"
  fi
  log "regenerating registry card for $model"
  python3 "$SCRIPT_DIR/_registry.py" "$model"
  log "regenerating catalog entry for $model"
  manifest_args=("$SCRIPT_DIR/_manifest.py" "$model")
  if [[ "$PUBLIC" -eq 1 ]]; then
    manifest_args+=(--public)
  fi
  python3 "${manifest_args[@]}"
done

# Refresh the catalog's top-level language/dialect label map (generated data,
# independent of any single model) so a full regenerate keeps it in lockstep
# with _catalog.LANGUAGE_DISPLAY_LABELS. Idempotent; leaves models[] untouched.
log "refreshing catalog language-label map"
python3 "$CATALOG_PY" write-language-labels "$REPO_ROOT/model-registry/catalog.json"
