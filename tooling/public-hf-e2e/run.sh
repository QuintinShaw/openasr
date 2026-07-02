#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Run the public Hugging Face pull-to-transcribe E2E smoke.

Defaults:
  model       moonshine-tiny:q8
  catalog     ${canonical_catalog_url}
  audio       fixtures/jfk.wav

Environment overrides:
  OPENASR_PUBLIC_HF_E2E_MODEL
  OPENASR_PUBLIC_HF_E2E_CATALOG_URL
  OPENASR_PUBLIC_HF_E2E_AUDIO
  OPENASR_PUBLIC_HF_E2E_BIN
  OPENASR_PUBLIC_HF_E2E_WORKDIR
  OPENASR_PUBLIC_HF_E2E_MIN_CHARS
  OPENASR_PUBLIC_HF_E2E_EXPECT_REGEX
  OPENASR_PUBLIC_HF_E2E_GGML_BACKEND
  OPENASR_PUBLIC_HF_E2E_KEEP=1
  OPENASR_PUBLIC_HF_E2E_SUMMARY_JSON
  OPENASR_PUBLIC_HF_E2E_SUMMARY_MD

Options:
  --model <id:quant>
  --catalog-url <url-or-path>
  --audio <path>
  --bin <path>
  --workdir <path>
  --min-chars <n>
  --keep
  --dry-run
  --summary-json <path>
  --summary-md <path>
  --strict-evidence
  -h, --help
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
canonical_catalog_url="$(
  PYTHONPATH="${repo_root}/tooling/publish-model/scripts${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 - <<'PY'
from _catalog import CATALOG_URL

print(CATALOG_URL)
PY
)"

model="${OPENASR_PUBLIC_HF_E2E_MODEL:-moonshine-tiny:q8}"
catalog_url="${OPENASR_PUBLIC_HF_E2E_CATALOG_URL:-$canonical_catalog_url}"
audio="${OPENASR_PUBLIC_HF_E2E_AUDIO:-fixtures/jfk.wav}"
openasr_bin="${OPENASR_PUBLIC_HF_E2E_BIN:-}"
workdir="${OPENASR_PUBLIC_HF_E2E_WORKDIR:-}"
min_chars="${OPENASR_PUBLIC_HF_E2E_MIN_CHARS:-12}"
ggml_backend="${OPENASR_PUBLIC_HF_E2E_GGML_BACKEND:-cpu}"
keep="${OPENASR_PUBLIC_HF_E2E_KEEP:-0}"
summary_json="${OPENASR_PUBLIC_HF_E2E_SUMMARY_JSON:-}"
summary_md="${OPENASR_PUBLIC_HF_E2E_SUMMARY_MD:-}"
dry_run=0
strict_evidence=0

die() {
  echo "$*" >&2
  exit 2
}

preflight_output_path() {
  local path="$1"
  local flag="$2"
  [[ -n "${path}" ]] || return 0
  if [[ -d "${path}" ]]; then
    die "${flag} path is a directory: ${path}"
  fi
  local parent
  parent="$(dirname "${path}")"
  mkdir -p "${parent}"
  if [[ -e "${path}" && ! -f "${path}" ]]; then
    die "${flag} path is not a regular file: ${path}"
  fi
  local probe="${path}.tmp.$$"
  if ! : > "${probe}"; then
    die "cannot write ${flag} path: ${path}"
  fi
  rm -f "${probe}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      model="$2"
      shift 2
      ;;
    --catalog-url)
      catalog_url="$2"
      shift 2
      ;;
    --audio)
      audio="$2"
      shift 2
      ;;
    --bin)
      openasr_bin="$2"
      shift 2
      ;;
    --workdir)
      workdir="$2"
      shift 2
      ;;
    --min-chars)
      min_chars="$2"
      shift 2
      ;;
    --keep)
      keep=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --summary-json)
      summary_json="$2"
      shift 2
      ;;
    --summary-md)
      summary_md="$2"
      shift 2
      ;;
    --strict-evidence)
      strict_evidence=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${repo_root}"

if [[ "${audio}" != /* ]]; then
  audio="${repo_root}/${audio}"
fi

if [[ ! -f "${audio}" ]]; then
  echo "audio fixture not found: ${audio}" >&2
  exit 1
fi

if ! [[ "${min_chars}" =~ ^[0-9]+$ ]] || [[ "${min_chars}" -lt 1 ]]; then
  echo "--min-chars must be a positive integer, got: ${min_chars}" >&2
  exit 2
fi

if [[ "${strict_evidence}" -eq 1 ]]; then
  [[ "${dry_run}" -eq 0 ]] || die "--strict-evidence cannot be used with --dry-run"
  [[ -n "${summary_json}" ]] || die "--summary-json is required with --strict-evidence"
  [[ "${catalog_url}" == "${canonical_catalog_url}" ]] || die "--strict-evidence requires the canonical public catalog URL: ${canonical_catalog_url}"
fi

preflight_output_path "${summary_json}" "--summary-json"
preflight_output_path "${summary_md}" "--summary-md"

created_workdir=0
if [[ -z "${workdir}" ]]; then
  workdir="$(mktemp -d "${TMPDIR:-/tmp}/openasr-public-hf-e2e.XXXXXX")"
  created_workdir=1
fi
mkdir -p "${workdir}"
workdir="$(cd "${workdir}" && pwd)"

cleanup() {
  status=$?
  if [[ "${created_workdir}" -eq 1 && "${keep}" != "1" && "${status}" -eq 0 ]]; then
    rm -rf "${workdir}"
  else
    echo "Public-HF E2E workdir retained: ${workdir}" >&2
  fi
}
trap cleanup EXIT

openasr_home="${workdir}/openasr-home"
transcript="${workdir}/transcript.txt"
pull_stdout="${workdir}/pull.stdout"
pull_stderr="${workdir}/pull.stderr"
transcribe_stderr="${workdir}/transcribe.stderr"
mkdir -p "${openasr_home}"

write_summary() {
  local installed_pack_file="${1:-}"
  local installed_pack_sha256="${2:-}"
  local installed_pack_size_bytes="${3:-}"
  local transcript_metrics_file="${4:-}"
  [[ -n "${summary_json}${summary_md}" ]] || return 0
  python3 - \
    "${summary_json}" \
    "${summary_md}" \
    "${model}" \
    "${catalog_url}" \
    "${canonical_catalog_url}" \
    "${audio}" \
    "${ggml_backend}" \
    "${min_chars}" \
    "${dry_run}" \
    "${strict_evidence}" \
    "${openasr_bin:-}" \
    "${installed_pack_file}" \
    "${installed_pack_sha256}" \
    "${installed_pack_size_bytes}" \
    "${transcript_metrics_file}" \
    <<'PY'
import json
import os
import sys
from pathlib import Path

(
    summary_json,
    summary_md,
    model,
    catalog_url,
    canonical_catalog_url,
    audio,
    ggml_backend,
    min_chars,
    dry_run,
    strict_evidence,
    openasr_bin,
    installed_pack_file,
    installed_pack_sha256,
    installed_pack_size_bytes,
    transcript_metrics_file,
) = sys.argv[1:]


def bool_arg(value: str) -> bool:
    return value == "1"


def public_catalog_name(value: str) -> str:
    if value.startswith("https://"):
        return value
    return Path(value).name or "local-catalog"


def load_transcript_metrics(path: str) -> dict[str, object]:
    if not path:
        return {}
    metrics_path = Path(path)
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


metrics = load_transcript_metrics(transcript_metrics_file)
summary: dict[str, object] = {
    "schema_version": 1,
    "tool": "public-hf-e2e",
    "model": model,
    "catalog": public_catalog_name(catalog_url),
    "canonical_catalog": canonical_catalog_url,
    "catalog_is_canonical_public_hf": catalog_url == canonical_catalog_url,
    "audio_file": Path(audio).name,
    "ggml_backend": ggml_backend,
    "min_chars": int(min_chars),
    "expect_regex_set": bool(os.environ.get("OPENASR_PUBLIC_HF_E2E_EXPECT_REGEX")),
    "dry_run": bool_arg(dry_run),
    "strict_evidence": bool_arg(strict_evidence),
    "executed": not bool_arg(dry_run),
}

if not bool_arg(dry_run):
    pack_size = int(installed_pack_size_bytes) if installed_pack_size_bytes else 0
    summary.update(
        {
            "openasr_bin_file": Path(openasr_bin).name,
            "installed_pack_file": installed_pack_file,
            "installed_pack_sha256": installed_pack_sha256,
            "installed_pack_size_bytes": pack_size,
        }
    )
    summary.update(metrics)

rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

if summary_json:
    path = Path(summary_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")

if summary_md:
    path = Path(summary_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    transcript_preview = str(summary.get("transcript_preview", "")).replace("`", "'")
    lines = [
        "### Public-HF E2E evidence",
        "",
        f"- model: `{summary['model']}`",
        f"- catalog: `{summary['catalog']}`",
        f"- canonical public catalog: `{'yes' if summary['catalog_is_canonical_public_hf'] else 'no'}`",
        f"- audio fixture: `{summary['audio_file']}`",
        f"- dry run: `{summary['dry_run']}`",
        f"- strict evidence mode: `{summary['strict_evidence']}`",
        f"- executed: `{summary['executed']}`",
    ]
    if summary["executed"]:
        lines.extend(
            [
                f"- openasr binary: `{summary['openasr_bin_file']}`",
                f"- installed pack: `{summary['installed_pack_file']}`",
                f"- installed pack sha256: `{summary['installed_pack_sha256']}`",
                f"- installed pack size bytes: `{summary['installed_pack_size_bytes']}`",
                f"- transcript chars: `{summary.get('transcript_chars', 0)}`",
                f"- transcript letters: `{summary.get('transcript_letters', 0)}`",
                f"- transcript preview: `{transcript_preview}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

echo "Public-HF E2E"
echo "  model:       ${model}"
echo "  catalog:     ${catalog_url}"
echo "  audio:       ${audio}"
echo "  openasrHome: ${openasr_home}"

if [[ "${dry_run}" -eq 1 ]]; then
  write_summary "" "" "" ""
  echo "dry run only; no build, download, or transcription was executed"
  exit 0
fi

if [[ -z "${openasr_bin}" ]]; then
  cargo build -p openasr-cli --release
  openasr_bin="${repo_root}/target/release/openasr"
fi

if [[ "${openasr_bin}" != /* ]]; then
  openasr_bin="${repo_root}/${openasr_bin}"
fi
if [[ ! -x "${openasr_bin}" ]]; then
  echo "openasr binary is not executable: ${openasr_bin}" >&2
  exit 1
fi

run_openasr() {
  env \
    -u HF_TOKEN \
    -u HUGGINGFACE_TOKEN \
    -u HUGGINGFACE_HUB_TOKEN \
    -u HUGGING_FACE_HUB_TOKEN \
    OPENASR_HOME="${openasr_home}" \
    OPENASR_GGML_BACKEND="${ggml_backend}" \
    "${openasr_bin}" "$@"
}

echo "Pulling public pack anonymously..."
pull_output="$(run_openasr pull "${model}" --catalog-url "${catalog_url}" 2> >(tee "${pull_stderr}" >&2))"
printf '%s\n' "${pull_output}" | tee "${pull_stdout}"

pack_path="$(awk -F '\t' 'NF >= 4 { path = $4 } END { print path }' "${pull_stdout}")"
if [[ -z "${pack_path}" || ! -f "${pack_path}" ]]; then
  echo "could not locate installed model pack from pull output" >&2
  exit 1
fi
pack_file="$(basename "${pack_path}")"
pack_sha256="$(shasum -a 256 "${pack_path}" | awk '{ print $1 }')"
pack_size_bytes="$(wc -c < "${pack_path}" | tr -d '[:space:]')"

python3 - "${openasr_home}" "${pack_path}" <<'PY'
import sys
from pathlib import Path

home = Path(sys.argv[1]).resolve()
pack = Path(sys.argv[2]).resolve()
try:
    pack.relative_to(home)
except ValueError:
    raise SystemExit(f"installed pack is outside OPENASR_HOME: {pack}")
if pack.suffix != ".oasr":
    raise SystemExit(f"installed pack is not a .oasr file: {pack}")
PY

echo "Transcribing with installed pack..."
run_openasr transcribe "${audio}" \
  --backend native \
  --model-pack "${pack_path}" \
  --format text \
  --output "${transcript}" \
  2> >(tee "${transcribe_stderr}" >&2)

transcript_metrics="${workdir}/transcript.metrics.json"
python3 - "${transcript}" "${min_chars}" "${transcript_metrics}" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
min_chars = int(sys.argv[2])
metrics_path = Path(sys.argv[3])
text = path.read_text(encoding="utf-8").strip()
letters = sum(1 for char in text if char.isalpha())
if len(text) < min_chars or letters < min(4, min_chars):
    raise SystemExit(
        f"transcript is too short or non-textual: {len(text)} chars, {letters} letters"
    )
if "mock transcription" in text.lower():
    raise SystemExit("transcript appears to come from the mock backend")
pattern = os.environ.get("OPENASR_PUBLIC_HF_E2E_EXPECT_REGEX")
if pattern and not re.search(pattern, text, flags=re.IGNORECASE):
    raise SystemExit(f"transcript does not match OPENASR_PUBLIC_HF_E2E_EXPECT_REGEX={pattern!r}")
metrics_path.write_text(
    json.dumps(
        {
            "transcript_chars": len(text),
            "transcript_letters": letters,
            "transcript_preview": text[:160],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print("Transcript preview:")
print(text[:500])
PY

write_summary "${pack_file}" "${pack_sha256}" "${pack_size_bytes}" "${transcript_metrics}"
echo "Public-HF E2E passed"
