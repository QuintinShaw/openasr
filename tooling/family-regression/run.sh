#!/usr/bin/env bash
# Family regression: pull a real public pack and run the real user chain
# (transcribe on the CPU backend, or a ggml integrity probe for packs without
# a batch transcribe path), then compare against committed goldens.
set -euo pipefail

usage() {
  cat <<EOF
Run one model-family regression case end to end.

Usage:
  run.sh --case <name> --ref <id:quant> [options]

Options:
  --case <name>          Golden dir name under tooling/family-regression/goldens/
  --ref <id:quant>       Catalog pull reference, e.g. whisper-tiny:q4
  --mode <mode>          transcribe (default) | verify | diarize
  --strategy <s>         exact (default) | normalized | wer:<t>
  --audio <path>         Audio fixture (default fixtures/jfk.wav)
  --extra-pull <ref>     Additional pack to pull first (repeatable; diarize deps)
  --bin <path>           openasr binary (default target/release/openasr)
  --transcript-out <p>   Where to write the raw transcript (default tmp/family-regression/<case>.txt)
  --update-golden        Write the transcript to the golden dir instead of comparing
  -h, --help             Show this help

Environment:
  OPENASR_FAMILY_REGRESSION_SOURCE   pull --source override (auto|hf|hf-mirror)
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

case_name=""
ref=""
mode="transcribe"
strategy="exact"
audio="${repo_root}/fixtures/jfk.wav"
extra_pulls=()
bin="${repo_root}/target/release/openasr"
transcript_out=""
update_golden=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case) case_name="$2"; shift 2 ;;
    --ref) ref="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --strategy) strategy="$2"; shift 2 ;;
    --audio) audio="$2"; shift 2 ;;
    --extra-pull) extra_pulls+=("$2"); shift 2 ;;
    --bin) bin="$2"; shift 2 ;;
    --transcript-out) transcript_out="$2"; shift 2 ;;
    --update-golden) update_golden=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${case_name}" && -n "${ref}" ]] || { usage >&2; exit 2; }
[[ -x "${bin}" ]] || { echo "openasr binary not found: ${bin}" >&2; exit 2; }

golden_dir="${script_dir}/goldens/${case_name}"
transcript_out="${transcript_out:-${repo_root}/tmp/family-regression/${case_name}.txt}"
mkdir -p "$(dirname "${transcript_out}")"

pull_source_args=()
if [[ -n "${OPENASR_FAMILY_REGRESSION_SOURCE:-}" ]]; then
  pull_source_args=(--source "${OPENASR_FAMILY_REGRESSION_SOURCE}")
fi

for pull_ref in "${extra_pulls[@]+"${extra_pulls[@]}"}" "${ref}"; do
  echo "== pull ${pull_ref}"
  "${bin}" pull "${pull_ref}" --accept-license "${pull_source_args[@]+"${pull_source_args[@]}"}"
done
# `openasr list` rows are TSV: <ref>\t<size>\t<sha256>\t<installed path>
pack_path="$("${bin}" list | awk -F'\t' -v ref="${ref}" '$1 == ref { print $4 }' | head -1)"
[[ -f "${pack_path}" ]] || { echo "installed pack path not found for ${ref}" >&2; exit 1; }

case "${mode}" in
  verify)
    echo "== verify ${pack_path}"
    OPENASR_GGML_BACKEND=cpu "${bin}" verify "${pack_path}"
    echo "PASS verify ${ref}"
    exit 0
    ;;
  transcribe|diarize)
    diarize_args=()
    [[ "${mode}" == "diarize" ]] && diarize_args=(--diarize)
    echo "== transcribe ${audio} with ${ref} (cpu)"
    OPENASR_GGML_BACKEND=cpu "${bin}" transcribe "${audio}" \
      --model "${ref}" --format text \
      --output "${transcript_out}" \
      "${diarize_args[@]+"${diarize_args[@]}"}"
    ;;
  *)
    echo "unknown mode: ${mode}" >&2; exit 2 ;;
esac

if [[ "${update_golden}" == "1" ]]; then
  mkdir -p "${golden_dir}"
  cp "${transcript_out}" "${golden_dir}/golden.txt"
  echo "wrote ${golden_dir}/golden.txt"
  exit 0
fi

python3 "${script_dir}/compare.py" \
  --transcript "${transcript_out}" \
  --golden-dir "${golden_dir}" \
  --strategy "${strategy}"
echo "PASS ${case_name} (${strategy})"
