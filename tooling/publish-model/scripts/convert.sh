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
#
# Import dispatch is data-driven from models-publish.toml:
#   required_source_files  globs that must exist under the staged source (after
#                          prep.sh) before import -- fail closed, not "import
#                          dies later on a missing file".
#   external_converter     command template for families converted outside
#                          `openasr model-pack import` (mimo, redimnet2); must
#                          produce the pack at {out}.
#   import_command         `model-pack ...` invocation template for families
#                          whose CLI shape differs from the generic default
#                          (dolphin's language scheme, firered's multi-input,
#                          hymt2/pyannote file-shaped sources).
# With neither override, falls back to the generic `import <import_subcommand>`
# shapes; a model with no usable recipe fails closed instead of guessing.
source "$(dirname "$0")/lib.sh"
source "$PUB_DIR/portable.sh"

MODEL="${1:?usage: convert.sh <model-id> <quant-id>}"
QUANT="${2:?usage: convert.sh <model-id> <quant-id>}"

SRC="$(src_dir "$MODEL")"
WORK="$(work_root "$MODEL")"
PACKS="$(packs_dir "$MODEL")"
[[ -d "$SRC" ]] || die "source not staged; run download.sh $MODEL first ($SRC missing)"
mkdir -p "$PACKS"
OUT="$(pack_file "$MODEL" "$QUANT")"
TOKEN="$(quant_token "$QUANT")"
REGISTRY_ID="$(cat_field "$MODEL" registry_id)"
BIN="$(openasr_bin)"

# Build provenance, fail-closed: the pipeline always runs from a repo checkout,
# so record the exact commit whose quantization policy built this pack into the
# pack's own GGUF metadata (openasr.build.commit). The writer rejects anything
# that is not a 40-hex sha; a checkout without a resolvable HEAD cannot build.
BUILD_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)" \
  || die "cannot resolve git HEAD for build provenance; the publish lane requires a committed checkout"
export OPENASR_BUILD_COMMIT="$BUILD_COMMIT"

# The import can only run on a complete source tree. When the declared files
# are missing (prep never ran, or the download was silently stripped) this
# fails here with the gap named -- not as an opaque importer error.
required=()
while IFS= read -r pattern; do
  [[ -n "$pattern" ]] || continue
  required+=("$pattern")
done < <(cat_lines "$MODEL" required_source_files)
if (( ${#required[@]} )); then
  require_files "$SRC" "${required[@]}" \
    || die "source for $MODEL is incomplete; run download.sh + prep.sh first"
fi

template_vars=(
  "src=$SRC"
  "work=$WORK"
  "out=$OUT"
  "packs_dir=$PACKS"
  "registry_id=$REGISTRY_ID"
  "quant=$TOKEN"
  "model=$MODEL"
)

EXTERNAL_CONVERTER="$(cat_field_opt "$MODEL" external_converter)"
IMPORT_COMMAND="$(cat_field_opt "$MODEL" import_command)"
REQUANT_SOURCE_QUANT="$(cat_field_opt "$MODEL" requant_source_quant)"

# Requantized external-converter sources are deliberately staged next to the
# model's other build artifacts, not in the system temp directory.  This keeps
# multi-gigabyte intermediate packs on the publish volume and lets the EXIT
# trap remove them if either half of the two-step conversion fails.
REQUANT_STAGING=""
REQUANT_STAGING_DIR=""
cleanup_requant_staging() {
  if [[ -n "$REQUANT_STAGING" ]]; then
    rm -f "$REQUANT_STAGING"
  fi
  if [[ -n "$REQUANT_STAGING_DIR" ]]; then
    rmdir "$REQUANT_STAGING_DIR" 2>/dev/null || true
  fi
}

# External recipes are trusted configuration, but their substituted paths are
# not shell syntax.  Quote every data value before the recipe is evaluated so
# a checkout/model path containing whitespace or shell metacharacters remains
# one argument.  The legacy direct external-converter path intentionally keeps
# its historical expansion below; this hardened expansion is only for the new
# requant staging branch.
expand_quoted_external_template() {
  local quant_value="$1" out_value="$2"
  local quoted_vars=(
    "src=$(printf '%q' "$SRC")"
    "work=$(printf '%q' "$WORK")"
    "out=$(printf '%q' "$out_value")"
    "packs_dir=$(printf '%q' "$PACKS")"
    "registry_id=$(printf '%q' "$REGISTRY_ID")"
    "quant=$(printf '%q' "$quant_value")"
    "model=$(printf '%q' "$MODEL")"
  )
  expand_template "$EXTERNAL_CONVERTER" "${quoted_vars[@]}"
}

# Re-runnable by contract: the staged source is the truth, the pack a derived
# artifact. Drop any previous build so the import never dies on OutputExists.
rm -f "$OUT"

if [[ -n "$EXTERNAL_CONVERTER" ]]; then
  if [[ "$QUANT" == "q4_k" && -n "$REQUANT_SOURCE_QUANT" ]]; then
    [[ "$REQUANT_SOURCE_QUANT" != "$QUANT" ]] \
      || die "model $MODEL requant_source_quant must differ from target $QUANT"
    SOURCE_TOKEN="$(quant_token "$REQUANT_SOURCE_QUANT")" \
      || die "model $MODEL declares unsupported requant_source_quant '$REQUANT_SOURCE_QUANT'"
    REQUANT_STAGING_DIR="$(mktemp -d "$WORK/.requant-source.XXXXXX")" \
      || die "cannot allocate requant staging pack under $WORK"
    # Keep a unique directory and give the converter one explicit `.oasr`
    # destination inside it.  The directory template is portable across BSD
    # and GNU mktemp, while the converter owns creation of the output and may
    # reject an already-existing destination.
    REQUANT_STAGING="$REQUANT_STAGING_DIR/source.oasr"
    trap cleanup_requant_staging EXIT

    cmd="$(expand_quoted_external_template "$SOURCE_TOKEN" "$REQUANT_STAGING")"
    log "external converter $MODEL @ $REQUANT_SOURCE_QUANT -> $REQUANT_STAGING"
    (
      cd "$REPO_ROOT"
      # gguf-py uses tempfile internally even when its final output lives on
      # another volume. Keep those multi-gigabyte scratch tensors beside the
      # staging pack instead of silently exhausting the system disk.
      export TMPDIR="$REQUANT_STAGING_DIR"
      eval "$cmd"
    ) \
      || die "external converter failed for $MODEL @ $REQUANT_SOURCE_QUANT"
    [[ -f "$REQUANT_STAGING" ]] \
      || die "external converter produced no staging pack at $REQUANT_STAGING"

    log "requant $MODEL @ $QUANT ($TOKEN) -> $OUT"
    "$BIN" model-pack requant "$REQUANT_STAGING" "$OUT" --quant q4-k \
      || die "requant failed for $MODEL @ $QUANT"
    cleanup_requant_staging
    REQUANT_STAGING=""
    REQUANT_STAGING_DIR=""
    trap - EXIT
  else
    cmd="$(expand_template "$EXTERNAL_CONVERTER" "${template_vars[@]}")"
    log "external converter $MODEL @ $QUANT -> $OUT"
    (cd "$REPO_ROOT" && eval "$cmd")
  fi
elif [[ -n "$IMPORT_COMMAND" ]]; then
  cmd="$(expand_template "$IMPORT_COMMAND" "${template_vars[@]}")"
  log "import $MODEL @ $QUANT ($TOKEN) -> $OUT"
  read -r -a import_parts <<< "$cmd"
  "$BIN" "${import_parts[@]}"
else
  SUBCMD="$(cat_field "$MODEL" import_subcommand)"
  case "$SUBCMD" in
    "import "*) ;;
    *)
      die "model $MODEL has no import recipe: declare import_command or external_converter in models-publish.toml (current import_subcommand is descriptive prose: $SUBCMD)"
      ;;
  esac
  # Two generic CLI arg shapes: the CTC families (parakeet/wav2vec2) take only
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
fi

[[ -f "$OUT" ]] || die "import produced no pack at $OUT"

# Validate the pack reads back AND passes the quantization-strategy audit
# (audio-encoder Q8_0 floor + the tier the pack filename declares) -- a pack
# that violates the current policy cannot enter the pipeline.
"$BIN" verify "$OUT" >&2
SIZE="$(portable_file_size "$OUT")"
SHA="$(portable_sha256 "$OUT")"
RESULT="$PACKS/$MODEL.$QUANT.result.json"
cat >"$RESULT" <<JSON
{"model":"$MODEL","quant":"$QUANT","cli_token":"$TOKEN","pack":"$OUT","size_bytes":$SIZE,"sha256":"$SHA"}
JSON
log "done $QUANT: $(portable_human_bytes "$SIZE") -> $RESULT"
