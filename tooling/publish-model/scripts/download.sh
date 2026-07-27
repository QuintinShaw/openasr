#!/usr/bin/env bash
# Stage 1 — download an upstream model from hf-mirror into the per-model staging
# dir. Idempotent: re-running resumes/refreshes the local snapshot.
#
#   tooling/publish-model/scripts/download.sh <model-id>
#
# Honors the standing rule: model downloads go through hf-mirror.com, never the
# default endpoint.
source "$(dirname "$0")/lib.sh"
source "$PUB_DIR/portable.sh"

MODEL="${1:?usage: download.sh <model-id>}"
UPSTREAM="$(cat_field "$MODEL" upstream_repo)"
REV="$(cat_field "$MODEL" source_revision 2>/dev/null || echo main)"
DEST="$(src_dir "$MODEL")"
CLI="$(hf_cli)"
[[ "$CLI" == "ERR_NO_HF_CLI" ]] && die "no hf CLI and no uvx; install huggingface_hub or uv"

mkdir -p "$DEST"
log "downloading $UPSTREAM@$REV from hf-mirror -> $DEST"
env -u ALL_PROXY -u all_proxy \
  HF_ENDPOINT="$HF_MIRROR_ENDPOINT" HF_XET_HIGH_PERFORMANCE=1 \
  $CLI download "$UPSTREAM" --revision "$REV" --local-dir "$DEST" \
  --exclude "*.pt" --exclude "*.onnx" --exclude "original/*" \
  --exclude "*.msgpack" --exclude "*.h5" --exclude "pytorch_model.bin"

if ! portable_has_non_cache_file "$DEST"; then
  die "download completed but staged source has no non-cache files: $DEST"
fi

log "downloaded; staged source at $DEST"
ls -la "$DEST" >&2 || true
