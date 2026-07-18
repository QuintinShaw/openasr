#!/usr/bin/env bash
# GPU weight placement gate: a static lint that catches a specific model-onboarding
# defect class before it ships -- an ASR family's encoder (or other reactivated
# ggml subgraph) feeding its 2D matmul weights through a per-request upload path
# instead of a resident WEIGHTS-usage backend buffer.
#
# Background (see docs/design/gpu-weight-placement.md for the full writeup):
# ggml's scheduler only offloads a MUL_MAT/MUL_MAT_ID op to a GPU backend when
# its weight operand's buffer usage is GGML_BACKEND_BUFFER_USAGE_WEIGHTS
# (ggml-backend.cpp:908-928). OpenASR has exactly two code paths that land a
# tensor in a WEIGHTS-usage buffer:
#
#   A. GgmlStaticTensorArena (persistent arena, allocate_with_usage(..., USAGE_WEIGHTS))
#   B. load_gguf_weight_context / bind_loaded (zero-copy mmap bind, USAGE_WEIGHTS)
#
# A subgraph that instead builds its weights via runner.start_graph() +
# uploads.push(...) / pending_uploads.push(...) / <binding>.upload(...) puts
# those tensors in the graph's transient *compute* buffer, not a WEIGHTS
# buffer -- so the scheduler can never offload their matmuls, no matter how
# much GPU memory is free. This is exactly the defect found in Dolphin's
# E-Branchformer encoder and (independently) X-ASR/Zipformer's encoder: both
# passed golden/parity review because the *numbers* were right, but their
# encoders silently ran 100% on CPU under a GPU backend.
#
# This script is the static half of the two-part acceptance gate (see
# docs/design/gpu-weight-placement.md for the dynamic half, a one-shot
# GGML_SCHED_DEBUG=2 run). It is pure grep over committed source: no build, no
# inference, no weights on disk. Safe to run on every PR.
#
# Method: for each model family directory under crates/openasr-core/src/models/,
# scan the files whose name contains "encoder" or "executor" (case-insensitive,
# excluding test files) -- this is deliberately whole-family-scope rather than
# single-file, because some families (whisper) legitimately split "the
# per-request graph" (ggml_encoder_graph.rs) from "the resident weight arena"
# (ggml_executor.rs) across two files in the same directory. A family is
# flagged when, across that scanned file set:
#   - at least one file shows the risk pattern: it calls `runner.start_graph(`
#     *and* pushes tensors via an upload-style call
#     (`uploads.push(` / `pending_uploads.push(` / `.upload(`), AND
#   - *no* file in the set shows a WEIGHTS-usage path
#     (`load_gguf_weight_context` / `GgmlStaticTensorArena` / `bind_loaded`).
#
# This is a heuristic, not a proof -- it cannot see *which* tensors an upload
# call feeds (weights vs. per-request input), only whether the family's
# encoder/executor scope contains zero evidence of ever binding a WEIGHTS
# buffer at all. Families that legitimately only ever feed real per-request
# input via uploads.push (mel features, hidden states) always also show a
# safe-path call somewhere in scope, because ggml requires *some* WEIGHTS-usage
# buffer to hold their matmul weights in the first place. A family with zero
# safe-path evidence anywhere in its encoder/executor files is a real finding,
# not a false positive from this ambiguity -- confirmed by manual review for
# every family in this tree as of 2026-07 (see ALLOWLIST below and
# docs/design/gpu-weight-placement.md).
#
# Known findings are pre-declared in ALLOWLIST below so the gate does not
# immediately fail every unrelated PR on families we already know about and
# haven't fixed yet. Fixing a family's weight placement means removing it from
# ALLOWLIST in the same PR -- the script will tell you to if it detects the
# family no longer violates the check.
#
# Usage:
#   scripts/gpu-weight-placement-gate.sh            # run the gate (CI mode)
#   scripts/gpu-weight-placement-gate.sh --list      # just print scanned families + verdicts, exit 0
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_dir="$repo_root/crates/openasr-core/src/models"

if [[ ! -d "$models_dir" ]]; then
  echo "error: $models_dir not found (run from an openasr checkout)" >&2
  exit 1
fi

# --- Known findings -----------------------------------------------------
# family|tracking note. Remove an entry once its encoder's matmul weights are
# re-bound through GgmlStaticTensorArena or load_gguf_weight_context and the
# gate stops flagging it -- do not remove an entry just because it becomes
# inconvenient; verify with GGML_SCHED_DEBUG=2 first
# (docs/design/gpu-weight-placement.md). Deliberately a plain indexed array,
# not an associative one (`declare -A`) -- macOS ships bash 3.2, which lacks
# associative arrays, and this script must run unmodified there too.
ALLOWLIST=(
)

allowlist_note_for() {
  local family="$1" entry
  # macOS ships bash 3.2, where "${ALLOWLIST[@]}" on an empty array is an
  # unbound-variable error under `set -u`; short-circuit when the allowlist is
  # empty (the intended steady state once every family is fixed).
  [[ ${#ALLOWLIST[@]} -eq 0 ]] && return 1
  for entry in "${ALLOWLIST[@]}"; do
    if [[ "${entry%%|*}" == "$family" ]]; then
      echo "${entry#*|}"
      return 0
    fi
  done
  return 1
}

mode="${1:-}"
list_only=0
if [[ "$mode" == "--list" ]]; then
  list_only=1
elif [[ -n "$mode" ]]; then
  echo "error: unknown argument '$mode' (expected --list or nothing)" >&2
  exit 2
fi

risk_pattern='uploads\.push\(|pending_uploads\.push\(|\.upload\('
safe_pattern='load_gguf_weight_context|GgmlStaticTensorArena|bind_loaded'

new_violations=()
stale_allowlist=()
findings_report=()

for family_dir in "$models_dir"/*/; do
  family="$(basename "$family_dir")"

  # Collect encoder/executor-named source files for this family, top-level
  # only (skip nested test-support dirs like whisper/ggml_executor/tests.rs),
  # excluding obvious test files. Built with a portable read loop (not
  # `mapfile`, which needs bash 4+) so this runs unmodified under macOS's
  # stock bash 3.2 as well as CI's modern bash.
  candidate_files=()
  while IFS= read -r -d '' f; do
    candidate_files+=("$f")
  done < <(
    find "$family_dir" -maxdepth 1 -type f \( -iname '*encoder*.rs' -o -iname '*executor*.rs' \) \
      ! -iname '*test*' -print0 2>/dev/null
  )

  if [[ ${#candidate_files[@]} -eq 0 ]]; then
    continue
  fi

  has_safe_signal=0
  risk_hits=()
  for f in "${candidate_files[@]}"; do
    if rg -q "$safe_pattern" "$f"; then
      has_safe_signal=1
    fi
    if rg -q 'start_graph\s*\(' "$f" && rg -q "$risk_pattern" "$f"; then
      risk_hits+=("$f")
    fi
  done

  if [[ ${#risk_hits[@]} -gt 0 && $has_safe_signal -eq 0 ]]; then
    # Flagged: risk pattern present somewhere in scope, safe pattern absent everywhere.
    detail="$family: no WEIGHTS-usage path ($safe_pattern) found in scope, but upload-fed graph construction found in:"
    for f in "${risk_hits[@]}"; do
      line="$(rg -n -m1 "$risk_pattern" "$f" | head -1)"
      detail="$detail
    ${f#"$repo_root"/} (first hit: ${line})"
    done
    findings_report+=("$detail")
    if allowlist_note_for "$family" >/dev/null; then
      : # known finding, allowlisted -- do not fail the build
    else
      new_violations+=("$family")
    fi
  else
    if allowlist_note_for "$family" >/dev/null; then
      stale_allowlist+=("$family")
    fi
  fi
done

echo "GPU weight placement gate -- scanned $(find "$models_dir" -maxdepth 1 -type d | tail -n +2 | wc -l | tr -d ' ') model family dirs"
echo

if [[ ${#findings_report[@]} -gt 0 ]]; then
  echo "Findings (encoder/executor scope with upload-fed graph construction and no WEIGHTS-usage bind):"
  for entry in "${findings_report[@]}"; do
    echo "  - $entry"
  done
  echo
fi

if [[ ${#findings_report[@]} -eq 0 ]]; then
  echo "No findings. Every family's encoder/executor scope shows a WEIGHTS-usage bind"
  echo "(GgmlStaticTensorArena / load_gguf_weight_context / bind_loaded)."
fi

if [[ ${#stale_allowlist[@]} -gt 0 ]]; then
  echo
  echo "note: ALLOWLIST entries no longer reproduce and should be removed from"
  echo "scripts/gpu-weight-placement-gate.sh (confirm with GGML_SCHED_DEBUG=2 first, then remove):"
  for family in "${stale_allowlist[@]}"; do
    echo "  - $family ($(allowlist_note_for "$family"))"
  done
fi

if [[ $list_only -eq 1 ]]; then
  exit 0
fi

if [[ ${#new_violations[@]} -gt 0 ]]; then
  echo
  echo "FAIL: new GPU weight placement violation(s) not in ALLOWLIST:"
  for family in "${new_violations[@]}"; do
    echo "  - $family"
  done
  echo
  echo "Fix: bind the encoder/executor's 2D matmul weights via load_gguf_weight_context"
  echo "(zero-copy) or GgmlStaticTensorArena (norm/bias), not runner.start_graph() +"
  echo "uploads.push()/.upload(). See docs/design/gpu-weight-placement.md."
  echo "If this is a known, not-yet-fixed family, add it to ALLOWLIST in this script"
  echo "with a tracking issue reference -- do not silently widen the risk pattern to"
  echo "dodge the finding."
  exit 1
fi

echo
echo "PASS (allowlisted findings, if any, are pre-existing and tracked above)."
