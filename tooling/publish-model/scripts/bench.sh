#!/usr/bin/env bash
# Stage 3 — measure RTF + peak RSS per pack, SEQUENTIALLY, on M1 CPU and Metal.
#
#   tooling/publish-model/scripts/bench.sh <model-id> [audio.wav]
#
# Why sequential: bench_suite_cli forks one subprocess per entry precisely to
# capture an uncontaminated peak-RSS high-water mark. Running packs concurrently
# on one machine would cross-contaminate RTF and RAM-peak, so this stage must
# never be parallelized (the quant-conversion fan-out is the only parallel part).
#
# Reuses the committed bench-suite as the measurement "ruler": writes a throwaway
# suite config pointing at the freshly built packs, runs it once forcing the CPU
# backend and once forcing Metal, then merges the two baseline JSONs into
# tmp/publish/<model>/metrics.json (size + RAM-peak + RTF-cpu + RTF-metal +
# JFK ΔWER vs fp16).
# The default benchmark input is the repo-tracked 11s JFK sample from whisper.cpp.
source "$(dirname "$0")/lib.sh"
source "$PUB_DIR/portable.sh"

MODEL="${1:?usage: bench.sh <model-id> [audio.wav]}"
AUDIO="${2:-$REPO_ROOT/fixtures/jfk.wav}"
[[ -f "$AUDIO" ]] || die "audio clip not found: $AUDIO"
BIN="$(openasr_bin)"
FAMILY="$(cat_field "$MODEL" family)"
WORK="$(work_root "$MODEL")"
SUITE="$WORK/bench.suite.toml"
mkdir -p "$WORK"

# Build a throwaway single-model suite over the three packs.
{
  echo "schema_version = 1"
  echo "[default_tolerances]"
  echo "rtf_rel = 1.0"; echo "peak_rss_rel = 1.0"; echo "wer_abs = 1.0"
  echo "gate_peak_rss = false"; echo "gate_vs_cpp = false"; echo "cpp_slack = 1.0"
  for q in $(cat_quants "$MODEL"); do
    pack="$(pack_file "$MODEL" "$q")"
    [[ -f "$pack" ]] || { log "skip $q (pack missing: $pack)"; continue; }
    echo "[[entries]]"
    echo "id = \"$MODEL-$q\""
    echo "family = \"$FAMILY\""
    echo "quant = \"$q\""
    echo "pack_path = \"$pack\""
    echo "audio_path = \"$AUDIO\""
    echo "optional = true"
  done
} >"$SUITE"

run_backend() {
  local backend="$1" out="$2"
  log "benchmarking $MODEL on $backend (sequential, best-of-3)"
  OPENASR_BENCH_SUITE_CAPTURE_TRANSCRIPT=1 OPENASR_GGML_BACKEND="$backend" "$BIN" bench-suite \
    --config "$SUITE" --write-baseline "$out" --runs 3 --format json >/dev/null
}

CPU_JSON="$WORK/bench.cpu.json"
METAL_JSON="$WORK/bench.metal.json"
run_backend cpu   "$CPU_JSON"
run_backend metal "$METAL_JSON" || log "metal run failed (non-Metal host?); RTF-metal will be n/a"

python3 "$PUB_DIR/_merge_metrics.py" "$MODEL" "$CPU_JSON" "$METAL_JSON" >"$(metrics_json "$MODEL")"
log "metrics -> $(metrics_json "$MODEL")"
cat "$(metrics_json "$MODEL")" >&2
