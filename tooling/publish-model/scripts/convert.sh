#!/usr/bin/env bash
# Stage 2 (single quant) — import the staged upstream source into one `.oasr`
# pack at one quantization. Runnable standalone and also invoked, three times in
# parallel by the publish-model skill.
#
#   tooling/publish-model/scripts/convert.sh <model-id> <quant-id>
#   # quant-id: fp16|q8_0|q4_k|q3_k
#
# Writes the pack plus a sidecar <model>.<quant>.result.json (path + byte size +
# sha256) so the orchestrator can verify the artifact on disk instead of
# trusting subagent prose.
source "$(dirname "$0")/lib.sh"
source "$PUB_DIR/portable.sh"

MODEL="${1:?usage: convert.sh <model-id> <quant-id>}"
QUANT="${2:?usage: convert.sh <model-id> <quant-id>}"

SRC="$(src_dir "$MODEL")"
[[ -d "$SRC" ]] || die "source not staged; run download.sh $MODEL first ($SRC missing)"
mkdir -p "$(packs_dir "$MODEL")"
OUT="$(pack_file "$MODEL" "$QUANT")"
TOKEN="$(quant_token "$QUANT")"
SUBCMD="$(cat_field "$MODEL" import_subcommand)"
REGISTRY_ID="$(cat_field "$MODEL" registry_id)"
BIN="$(openasr_bin)"

# Build provenance, fail-closed: the pipeline always runs from a repo checkout,
# so record the exact commit whose quantization policy built this pack into the
# pack's own GGUF metadata (openasr.build.commit). The writer rejects anything
# that is not a 40-hex sha; a checkout without a resolvable HEAD cannot build.
BUILD_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)" \
  || die "cannot resolve git HEAD for build provenance; the publish lane requires a committed checkout"
export OPENASR_BUILD_COMMIT="$BUILD_COMMIT"

# Two CLI arg shapes: the CTC families (parakeet/wav2vec2) take only
# --package-id + --quantization; the rest also accept source/license metadata.
args=("$SRC" "$OUT" --package-id "$REGISTRY_ID" --quantization "$TOKEN")
if [[ "$(cat_field "$MODEL" needs_license_flags)" == "true" ]]; then
  args+=(
    --source-name    "$(cat_field "$MODEL" upstream_repo)"
    --source-revision "$(cat_field "$MODEL" source_revision)"
    --license-name   "$(cat_field "$MODEL" license_name)"
    --license-source "$(cat_field "$MODEL" license_source)"
  )
fi

log "import $MODEL @ $QUANT ($TOKEN) -> $OUT"
read -r -a subcmd_parts <<< "$SUBCMD"
"$BIN" model-pack "${subcmd_parts[@]}" "${args[@]}"
[[ -f "$OUT" ]] || die "import produced no pack at $OUT"

# Validate the pack reads back, then emit the verifiable sidecar.
"$BIN" verify "$OUT" >&2
SIZE="$(portable_file_size "$OUT")"
SHA="$(portable_sha256 "$OUT")"
RESULT="$(packs_dir "$MODEL")/$MODEL.$QUANT.result.json"
cat >"$RESULT" <<JSON
{"model":"$MODEL","quant":"$QUANT","cli_token":"$TOKEN","pack":"$OUT","size_bytes":$SIZE,"sha256":"$SHA"}
JSON
log "done $QUANT: $(portable_human_bytes "$SIZE") -> $RESULT"
