#!/usr/bin/env bash
# Stage 1.5 — run a model's declared prep recipe over the staged source.
#
#   tooling/publish-model/scripts/prep.sh <model-id>
#
# Some families cannot go straight from the upstream download to
# `openasr model-pack import`: their checkpoints need normalization first
# (PyTorch .pt/.pth.tar -> safetensors, Kaldi cmvn.ark -> cmvn.txt, LoRA
# merge, ESPnet CMVN fold, ONNX extract). Those steps used to be ad-hoc
# one-offs; they are now declared per model in models-publish.toml's
# `prep_scripts` and run here as a first-class pipeline stage, once per
# model (convert.sh's per-quant fan-out only validates the result via
# required_source_files, never reruns prep).
#
# Each entry is a command template; placeholders {src} {work} {model}
# {packs_dir} {registry_id} expand from the pipeline's own variables, so
# recipes never hardcode staging paths. Commands run from the repo root.
source "$(dirname "$0")/lib.sh"

MODEL="${1:?usage: prep.sh <model-id>}"
SRC="$(src_dir "$MODEL")"
WORK="$(work_root "$MODEL")"
[[ -d "$SRC" ]] || die "source not staged; run download.sh $MODEL first ($SRC missing)"

ran=0
while IFS= read -r template; do
  [[ -n "$template" ]] || continue
  cmd="$(expand_template "$template" \
    "src=$SRC" \
    "work=$WORK" \
    "model=$MODEL" \
    "packs_dir=$(packs_dir "$MODEL")" \
    "registry_id=$(cat_field "$MODEL" registry_id)")"
  log "prep [$MODEL]: $cmd"
  (cd "$REPO_ROOT" && eval "$cmd")
  ran=$((ran + 1))
done < <(cat_lines "$MODEL" prep_scripts)

if (( ran == 0 )); then
  log "no prep recipe declared for $MODEL; staged source goes straight to convert.sh"
else
  log "prep complete for $MODEL ($ran step(s))"
fi
