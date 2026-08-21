#!/usr/bin/env bash
# Copy hardware-approved Windows CUDA/HIP release bytes to
# https://dl.openasr.org/core/vX.Y.Z/.
#
# The signed catalog's files[].url values point only at this prefix. GitHub
# release mirrors are provenance, not a runtime download fallback. This step
# is local: B2 write credentials never enter CI.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

fail() {
  printf '\nBACKEND CDN SYNC FAILED for %s\n%s\n' "${tag:-<unknown>}" "$1" >&2
  exit 1
}

trap 'fail "aborted at line $LINENO"' ERR

[ "$#" -eq 1 ] || fail "usage: $(basename "$0") vX.Y.Z"
version="${1#v}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "version must be X.Y.Z or vX.Y.Z"
tag="v${version}"

if [ "${CI:-}" = "true" ] || [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  fail "refusing to use B2 write credentials in CI"
fi
[ -n "${B2_S3_ENDPOINT:-}" ] && [ -n "${B2_APPLICATION_KEY_ID:-}" ] && [ -n "${B2_APPLICATION_KEY:-}" ] \
  || fail "source ~/.openasr/b2-release.env before running (B2_S3_ENDPOINT / B2_APPLICATION_KEY_ID / B2_APPLICATION_KEY)"
command -v gh >/dev/null 2>&1 || fail "gh is required"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"
gh auth status >/dev/null 2>&1 || fail "gh is not authenticated"
gh release view "$tag" --json tagName >/dev/null 2>&1 \
  || fail "GitHub release ${tag} does not exist"

if [ -z "${SSL_CERT_FILE:-}" ]; then
  certifi="$(python3 -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
  if [ -n "$certifi" ]; then
    export SSL_CERT_FILE="$certifi"
  fi
fi

workdir="$(mktemp -d "${TMPDIR:-/tmp}/openasr-backend-cdn.XXXXXX")"
trap 'rm -rf "$workdir"' EXIT

echo "==> downloading backend entries for ${tag}"
gh release download "$tag" \
  -p 'backend-pack-*.json' \
  -p 'backend-hardware-evidence-*.json' \
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
approved_entries=()
while IFS= read -r line || [ -n "$line" ]; do
  [ -n "$line" ] || continue
  approved_entries+=("$line")
done < <(tr -d '\r' < "$workdir/hardware-approved-entries.txt")
[ "${#approved_entries[@]}" -gt 0 ] \
  || fail "release ${tag} has no backend entry approved by hardware evidence"

echo "==> downloading approved release bytes"
python3 - "$workdir" "${approved_entries[@]}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
downloaded: set[str] = set()
for entry_path in sys.argv[2:]:
    entry = json.loads(Path(entry_path).read_text(encoding="utf-8"))
    for file in entry.get("files", []):
        name = file.get("filename")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise SystemExit(f"unsafe backend release filename: {name!r}")
        dest = root / name
        if name not in downloaded:
            subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    f"v{entry['version']}",
                    "-p",
                    name,
                    "-D",
                    str(root),
                    "--clobber",
                ],
                check=True,
            )
            downloaded.add(name)
        digest = hashlib.sha256()
        size = 0
        with dest.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if digest.hexdigest() != file["sha256"] or size != int(file["size_bytes"]):
            raise SystemExit(f"hash mismatch: {name}")
print(f"verified {len(downloaded)} unique files", file=sys.stderr)
PY

sync_files=()
python3 - "$workdir" "${approved_entries[@]}" <<'PY' > "$workdir/sync-files.txt"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seen = set()
for entry_path in sys.argv[2:]:
    entry = json.loads(Path(entry_path).read_text(encoding="utf-8"))
    for file in entry.get("files", []):
        name = file["filename"]
        if name in seen:
            continue
        seen.add(name)
        print(root / name)
PY
while IFS= read -r line || [ -n "$line" ]; do
  [ -n "$line" ] || continue
  sync_files+=("$line")
done < "$workdir/sync-files.txt"
[ "${#sync_files[@]}" -gt 0 ] || fail "no approved files to upload"

echo "==> uploading ${#sync_files[@]} objects to core/v${version}/"
python3 tooling/release-manifest/b2_sync.py sync --version "$version" "${sync_files[@]}"

python3 - "$workdir" "$version" "${approved_entries[@]}" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "tooling/release-manifest")
import backend_catalog

root = Path(sys.argv[1])
version = sys.argv[2]
backends = [json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[3:]]
synthetic = root / "cdn-catalog.json"
backend_catalog._write_utf8_lf(synthetic, {"backends": backends})
print(json.dumps(backend_catalog.verify_catalog_cdn(synthetic, version), sort_keys=True))
PY

echo
echo "BACKEND-CDN-SYNCED for ${tag}"
echo "  uploaded ${#sync_files[@]} hardware-approved objects to https://dl.openasr.org/core/v${version}/"
echo "  next: load the production catalog signing seed and run:"
echo "    scripts/prepare-windows-backend-catalog-release.sh ${tag}"
