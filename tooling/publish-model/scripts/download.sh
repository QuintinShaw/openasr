#!/usr/bin/env bash
# Stage 1 — download an upstream model from hf-mirror into the per-model staging
# dir. Idempotent: re-running resumes/refreshes the local snapshot.
#
#   tooling/publish-model/scripts/download.sh <model-id>
#
# Honors the standing rule: model downloads go through hf-mirror.com, never the
# default endpoint.
#
# Data-driven from models-publish.toml (all optional):
#   download_includes        fetch ONLY these patterns (hf --include); when set,
#                            no excludes apply -- use for repos that ship
#                            several redundant weight formats (whisper-large-v3
#                            ships the same model four times over).
#   download_excludes        extra patterns on top of the harness defaults
#                            (redundant formats no importer consumes).
#   extra_sources            "repo@rev:subdir" lines: auxiliary upstream repos
#                            fetched under the staging dir (e.g. MiMo's audio
#                            tokenizer encoder).
#   required_download_files  globs that MUST match after the download, or the
#                            stage fails closed. This is the fix for the class
#                            of bug where an exclude rule strips the model's
#                            only checkpoint (or a tokenless fetch of a private
#                            repo lands a 29-byte error page) while the exit
#                            code still says success.
source "$(dirname "$0")/lib.sh"
source "$PUB_DIR/portable.sh"

MODEL="${1:?usage: download.sh <model-id>}"
UPSTREAM="$(cat_field "$MODEL" upstream_repo)"
REV="$(cat_field_opt "$MODEL" source_revision)"
REV="${REV:-main}"
DEST="$(src_dir "$MODEL")"
CLI="$(hf_cli)"
[[ "$CLI" == "ERR_NO_HF_CLI" ]] && die "no hf CLI and no uvx; install huggingface_hub or uv"

# Redundant weight formats no importer consumes. Family checkpoints keep their
# own names (model.safetensors, <size>.pt, model.pth.tar, model.onnx for
# pyannote), so these defaults never strip a needed file -- and anything that
# IS needed is enforced by required_download_files below.
DEFAULT_DOWNLOAD_EXCLUDES=(flax_model.msgpack tf_model.h5 "*.msgpack" "*.h5" "original/*")

fetch() {
  # fetch <repo> <rev> <dest> [-- <extra hf args>...] with retries.
  local repo="$1" rev="$2" dest="$3"
  shift 3
  [[ "${1:-}" == "--" ]] && shift
  mkdir -p "$dest"
  log "downloading $repo@$rev from hf-mirror -> $dest"
  retry 3 env -u ALL_PROXY -u all_proxy \
    HF_ENDPOINT="$HF_MIRROR_ENDPOINT" HF_XET_HIGH_PERFORMANCE=1 \
    $CLI download "$repo" --revision "$rev" --local-dir "$dest" "$@"
}

mkdir -p "$DEST"

args=()
have_includes=0
while IFS= read -r pattern; do
  [[ -n "$pattern" ]] || continue
  args+=(--include "$pattern")
  have_includes=1
done < <(cat_lines "$MODEL" download_includes)
if [[ "$have_includes" == "0" ]]; then
  for pattern in "${DEFAULT_DOWNLOAD_EXCLUDES[@]}"; do
    args+=(--exclude "$pattern")
  done
  while IFS= read -r pattern; do
    [[ -n "$pattern" ]] || continue
    args+=(--exclude "$pattern")
  done < <(cat_lines "$MODEL" download_excludes)
fi

fetch "$UPSTREAM" "$REV" "$DEST" -- "${args[@]}"

# Auxiliary upstream repos, staged under the model's src dir.
while IFS= read -r spec; do
  [[ -n "$spec" ]] || continue
  repo_rev="${spec%%:*}"
  subdir="${spec#*:}"
  erepo="${repo_rev%%@*}"
  erev="${repo_rev#*@}"
  [[ -n "$erepo" && -n "$subdir" && "$erev" != "$repo_rev" ]] \
    || die "malformed extra_sources entry (want repo@rev:subdir): $spec"
  fetch "$erepo" "$erev" "$DEST/$subdir"
done < <(cat_lines "$MODEL" extra_sources)

# Fail-closed completeness: what the rest of the pipeline needs must be on
# disk, or the download is a failure -- not "rc=0 with only config files".
required=()
while IFS= read -r pattern; do
  [[ -n "$pattern" ]] || continue
  required+=("$pattern")
done < <(cat_lines "$MODEL" required_download_files)
if (( ${#required[@]} )); then
  require_files "$DEST" "${required[@]}" \
    || die "download of $MODEL completed but staged source is incomplete (see above)"
fi

if ! portable_has_non_cache_file "$DEST"; then
  die "download completed but staged source has no non-cache files: $DEST"
fi

log "downloaded; staged source at $DEST"
ls -la "$DEST" >&2 || true
