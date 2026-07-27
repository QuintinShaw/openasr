#!/usr/bin/env bash
# Stage 5 — publish to Hugging Face. Stages the upload folder (README + packs),
# creates the destination repo PRIVATE, and uploads. Irreversible-ish (creates a
# remote repo) — invoked from the human-in-loop skill, never from a subagent.
#
#   tooling/publish-model/scripts/hf_push.sh <model-id> [--public] [--no-packs]
#
# --public    create/keep the repo public (default: private).
# --no-packs  upload only the card/metadata (README etc.), NOT the .oasr weights
#             — for validating the publish flow without pushing GB-scale files.
#
# Uploads go to the REAL Hugging Face hub (never hf-mirror, which is download
# only). Auth uses $HF_TOKEN from the environment.
source "$(dirname "$0")/lib.sh"

MODEL="${1:?usage: hf_push.sh <model-id> [--public] [--no-packs]}"
shift || true
VISIBILITY="--private"
WITH_PACKS=1
STRICT_NS=0
for arg in "$@"; do
  case "$arg" in
    --public)   VISIBILITY=""; STRICT_NS=1 ;;
    --no-packs) WITH_PACKS=0 ;;
    *) die "unknown flag: $arg" ;;
  esac
done

: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
CLI="$(hf_cli)"
[[ "$CLI" == "ERR_NO_HF_CLI" ]] && die "no hf CLI and no uvx available"
CLI_PY="$(hf_py)"
CATALOG_REPO="$(cat_field "$MODEL" hf_repo)"   # e.g. openasr/qwen3-asr-1.7b
REPO="$(repo_dir "$MODEL")"
mkdir -p "$REPO"

# Resolve the namespace the token actually owns (catalog brands "openasr/...",
# but the authed account may be "OpenASR" or an org). hf upload --private
# auto-creates the repo on first push, so no separate `repo create` is needed.
NS_ARGS=("${CATALOG_REPO%%/*}")
[[ "$STRICT_NS" == "1" ]] && NS_ARGS+=(--strict)
NS="$(HF_TOKEN="$HF_TOKEN" $CLI_PY "$PUB_DIR/_hf_ns.py" "${NS_ARGS[@]}")"
HF_REPO="$NS/${CATALOG_REPO##*/}"

# Stage README (rendered by render_card.py) and, unless --no-packs, the packs.
[[ -f "$REPO/README.md" ]] || die "no README.md in $REPO; run render_card.py $MODEL first"
declare -a EXCLUDE=()
COMMIT_MSG="Publish $MODEL OpenASR packs (fp16/q8_0/q4_k)"
if [[ "$WITH_PACKS" == "1" ]]; then
  for q in $(cat_quants "$MODEL"); do
    pack="$(pack_file "$MODEL" "$q")"
    [[ -f "$pack" ]] || die "pack missing: $pack"
    cp -f "$pack" "$REPO/$(basename "$pack")"
  done
else
  # Card-only: never upload weights, and drop any previously-staged packs.
  rm -f "$REPO"/*.oasr
  EXCLUDE=(--exclude "*.oasr")
  COMMIT_MSG="Publish $MODEL model card + metadata (no weights)"
  log "card-only mode: excluding *.oasr weights from upload"
fi

log "uploading $REPO -> $HF_REPO ($([[ -n $VISIBILITY ]] && echo private || echo public))"
UPLOAD_ARGS=("$HF_REPO" "$REPO" . --type model)
[[ -n "$VISIBILITY" ]] && UPLOAD_ARGS+=("$VISIBILITY")
((${#EXCLUDE[@]})) && UPLOAD_ARGS+=("${EXCLUDE[@]}")
UPLOAD_ARGS+=(--commit-message "$COMMIT_MSG")

HF_TOKEN="$HF_TOKEN" $CLI upload "${UPLOAD_ARGS[@]}"

printf '%s\n' "$HF_REPO" >"$(work_root "$MODEL")/hf_repo.txt"
HF_REVISION="$(HF_TOKEN="$HF_TOKEN" $CLI_PY "$PUB_DIR/_hf_revision.py" "$HF_REPO")"
printf '%s\n' "$HF_REVISION" >"$(work_root "$MODEL")/hf_revision.txt"
log "published: https://huggingface.co/$HF_REPO"
log "revision: $HF_REVISION"
