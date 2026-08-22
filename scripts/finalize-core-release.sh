#!/usr/bin/env bash
# Publish a draft core release only after the signed runtime catalog is live.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

fail() { printf '\nRELEASE FINALIZATION FAILED for %s\n%s\n' "${tag:-<unknown>}" "$1" >&2; exit 1; }
trap 'fail "aborted at line $LINENO"' ERR

[ "$#" -eq 1 ] || fail "usage: $(basename "$0") vX.Y.Z"
version="${1#v}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "version must be X.Y.Z or vX.Y.Z"
tag="v${version}"
command -v gh >/dev/null 2>&1 || fail "gh is required"
command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v cargo >/dev/null 2>&1 || fail "cargo is required"
gh auth status >/dev/null 2>&1 || fail "gh is not authenticated"
if [ -z "${SSL_CERT_FILE:-}" ]; then
  certifi="$(python3 -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
  if [ -n "$certifi" ]; then
    export SSL_CERT_FILE="$certifi"
  fi
fi

is_draft="$(gh release view "$tag" --json isDraft --jq .isDraft 2>/dev/null)" \
  || fail "GitHub release ${tag} does not exist"
[ "$is_draft" = "true" ] || fail "release ${tag} is already public or is not a draft"

workdir="$(mktemp -d "${TMPDIR:-/tmp}/openasr-release-finalize.XXXXXX")"
trap 'rm -rf "$workdir"' EXIT
gh release download "$tag" \
  -p 'backend-pack-*.json' \
  -p 'backend-hardware-evidence-*.json' \
  -p 'gpu-correctness-matrix.v1.json' \
  -p 'gpu-correctness-receipt-*.json' \
  -p 'gpu-correctness-source-inventory.json' \
  -p 'gpu-correctness-source-model-catalog.json' \
  -p 'gpu-correctness-source-backend-catalog.json' \
  -p 'gpu-correctness-trace-*.jsonl' \
  -D "$workdir" --clobber

shopt -s nullglob
backend_entries=("$workdir"/backend-pack-*.json)
hardware_evidence=("$workdir"/backend-hardware-evidence-*.json)
cuda_entries=("$workdir"/backend-pack-cuda-sm_*.json)
hip_entries=("$workdir"/backend-pack-hip-gfx*.json)
if [ "${#cuda_entries[@]}" -ne 6 ] || [ "${#hip_entries[@]}" -ne 14 ] || [ "${#backend_entries[@]}" -ne 20 ]; then
  fail "release ${tag} must contain exactly 6 CUDA SM and 14 HIP gfx backend-pack metadata files"
fi
all_backend_entry_args=()
for entry in "${backend_entries[@]}"; do
  all_backend_entry_args+=(--entry "$entry")
done
[ "${#hardware_evidence[@]}" -gt 0 ] \
  || fail "release ${tag} has no real-hardware backend evidence"
hardware_evidence_args=()
for evidence in "${hardware_evidence[@]}"; do
  hardware_evidence_args+=(--evidence "$evidence")
done
python3 tooling/release-manifest/backend_hardware_evidence.py \
  "${all_backend_entry_args[@]}" "${hardware_evidence_args[@]}" \
  > "$workdir/hardware-approved-entries.txt"
# Native Windows Python writes CRLF even when invoked from Git Bash. Strip
# the record terminator before using each emitted path as an argv value.
# Portable read loop (not mapfile) so macOS stock bash 3.2 can sign.
approved_entries=()
while IFS= read -r line || [ -n "$line" ]; do
  [ -n "$line" ] || continue
  approved_entries+=("$line")
done < <(tr -d '\r' < "$workdir/hardware-approved-entries.txt")
[ "${#approved_entries[@]}" -gt 0 ] \
  || fail "release ${tag} has no backend entry approved by hardware evidence"

correctness_matrix="$workdir/gpu-correctness-matrix.v1.json"
[ -f "$correctness_matrix" ] \
  || fail "release ${tag} has no staged GPU correctness matrix"
correctness_receipts=("$workdir"/gpu-correctness-receipt-*.json)
correctness_traces=("$workdir"/gpu-correctness-trace-*.jsonl)
[ "${#correctness_receipts[@]}" -gt 0 ] \
  || fail "release ${tag} has no staged GPU correctness receipts"
[ "${#correctness_traces[@]}" -gt 0 ] \
  || fail "release ${tag} has no staged GPU trace artifacts"
for source in \
  "$workdir/gpu-correctness-source-inventory.json" \
  "$workdir/gpu-correctness-source-model-catalog.json" \
  "$workdir/gpu-correctness-source-backend-catalog.json"; do
  [ -f "$source" ] || fail "release ${tag} is missing a canonical correctness source snapshot: $source"
done
correctness_receipt_args=()
for receipt in "${correctness_receipts[@]}"; do
  correctness_receipt_args+=(--receipt "$receipt")
done
correctness_trace_args=()
for trace in "${correctness_traces[@]}"; do
  correctness_trace_args+=(--trace "$trace")
done
python3 tooling/release-manifest/gpu_correctness_gate.py validate \
  --manifest "$correctness_matrix" \
  --inventory "$workdir/gpu-correctness-source-inventory.json" \
  --catalog "$workdir/gpu-correctness-source-model-catalog.json" \
  --backend-catalog "$workdir/gpu-correctness-source-backend-catalog.json" \
  "${correctness_receipt_args[@]}" "${correctness_trace_args[@]}"

# The reusable deploy workflow is the only catalog activation path. Requiring
# its successful run here makes publishing the GitHub release strictly follow
# correctness validation and public-catalog deployment.
deploy_run_id="${OPENASR_DEPLOY_CATALOG_RUN_ID:-}"
[ -n "$deploy_run_id" ] || fail "set OPENASR_DEPLOY_CATALOG_RUN_ID to the successful reusable catalog-deploy run"
deploy_conclusion="$(gh run view "$deploy_run_id" --json conclusion --jq .conclusion)"
[ "$deploy_conclusion" = "success" ] \
  || fail "catalog deploy run $deploy_run_id did not succeed (conclusion=$deploy_conclusion)"

backend_entry_args=()
for entry in "${approved_entries[@]}"; do
  backend_entry_args+=(--entry "$entry")
done

cache_bust="$(date +%s)"
curl -fsSL "https://catalog.openasr.org/v1/catalog.json?release=${tag}-${cache_bust}" \
  -o "$workdir/catalog.json"
curl -fsSL "https://catalog.openasr.org/v1/catalog.signature.json?release=${tag}-${cache_bust}" \
  -o "$workdir/catalog.signature.json"
OPENASR_HOME="$workdir/home" \
OPENASR_CATALOG_FILE="$workdir/catalog.json" \
OPENASR_CATALOG_IDENTITY="https://catalog.openasr.org/v1/catalog.json" \
  cargo run --quiet -p openasr-cli -- doctor >/dev/null
python3 tooling/release-manifest/backend_catalog.py verify-catalog \
  --catalog "$workdir/catalog.json" \
  "${backend_entry_args[@]}"
python3 tooling/release-manifest/backend_hardware_evidence.py \
  "${all_backend_entry_args[@]}" "${hardware_evidence_args[@]}" \
  --catalog "$workdir/catalog.json" --version "$version" >/dev/null
python3 tooling/release-manifest/backend_catalog.py verify-cdn \
  --catalog "$workdir/catalog.json" \
  --version "$version"

echo "==> signed backend catalog is live; publishing ${tag}"
gh release edit "$tag" --draft=false --latest
echo "RELEASE-PUBLISHED ${tag}"
