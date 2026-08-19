#!/usr/bin/env bash
# Prepare the exact signed model/backend catalog update for a draft core release.
#
# This is deliberately LOCAL ONLY: it consumes the production catalog signing
# seed, but it does not push, deploy, publish, or undraft anything.  A release
# remains incomplete until the resulting catalog commit is reviewed, pushed,
# deployed by deploy-catalog.yml, and finalize-core-release.sh verifies the
# live bytes before publishing the draft.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

fail() {
  printf '\nCATALOG PREPARATION FAILED for %s\n%s\n' "${tag:-<unknown>}" "$1" >&2
  exit 1
}

trap 'fail "aborted at line $LINENO"' ERR

[ "$#" -eq 1 ] || fail "usage: $(basename "$0") vX.Y.Z"
version="${1#v}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "version must be X.Y.Z or vX.Y.Z"
tag="v${version}"

if [ "${CI:-}" = "true" ] || [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  fail "refusing to use the production signing seed in CI"
fi
[[ "${OPENASR_CATALOG_SIGNING_KEY_SEED_HEX:-}" =~ ^[0-9a-fA-F]{64}$ ]] \
  || fail "OPENASR_CATALOG_SIGNING_KEY_SEED_HEX must contain the production 64-hex seed"
command -v gh >/dev/null 2>&1 || fail "gh is required"
command -v cargo >/dev/null 2>&1 || fail "cargo is required"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"
gh auth status >/dev/null 2>&1 || fail "gh is not authenticated"

dirty="$(git status --porcelain --untracked-files=normal)"
[ -z "$dirty" ] || fail "the open-core worktree must be clean before catalog preparation"
is_draft="$(gh release view "$tag" --json isDraft --jq .isDraft 2>/dev/null)" \
  || fail "GitHub release ${tag} does not exist"
[ "$is_draft" = "true" ] || fail "release ${tag} is not a draft; backend catalog must be live before publication"

workdir="$(mktemp -d "${TMPDIR:-/tmp}/openasr-backend-catalog.XXXXXX")"
restore=1
cleanup() {
  if [ "$restore" = "1" ]; then
    for name in catalog.json catalog.signature.json catalog.public.json catalog.public.signature.json catalog.epoch; do
      if [ -f "$workdir/original-$name" ]; then
        cp "$workdir/original-$name" "model-registry/$name"
      fi
    done
  fi
  rm -rf "$workdir"
}
trap cleanup EXIT

for name in catalog.json catalog.signature.json catalog.public.json catalog.public.signature.json catalog.epoch; do
  cp "model-registry/$name" "$workdir/original-$name"
done

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
  || fail "release ${tag} has no real-hardware backend evidence; build artifacts alone are not publishable catalog claims"
hardware_evidence_args=()
for evidence in "${hardware_evidence[@]}"; do
  hardware_evidence_args+=(--evidence "$evidence")
done
python3 tooling/release-manifest/backend_hardware_evidence.py \
  "${all_backend_entry_args[@]}" "${hardware_evidence_args[@]}" \
  > "$workdir/hardware-approved-entries.txt"
mapfile -t approved_entries < "$workdir/hardware-approved-entries.txt"
[ "${#approved_entries[@]}" -gt 0 ] \
  || fail "release ${tag} has no backend entry approved by hardware evidence"
backend_entry_args=()
for entry in "${approved_entries[@]}"; do
  backend_entry_args+=(--entry "$entry")
done

python3 - "$workdir" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
downloaded = set()
for entry_path in sorted(root.glob("backend-pack-*.json")):
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    for file in entry.get("files", []):
        name = file.get("filename")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise SystemExit(f"unsafe backend release filename: {name!r}")
        if name in downloaded:
            continue
        subprocess.run(
            ["gh", "release", "download", f"v{entry['version']}", "-p", name, "-D", str(root), "--clobber"],
            check=True,
        )
        downloaded.add(name)
PY

echo "==> verifying every release byte against both backend entries"
python3 tooling/release-manifest/backend_catalog.py verify-assets \
  "${all_backend_entry_args[@]}" \
  --asset-directory "$workdir" \
  --version "$version"

python3 tooling/release-manifest/backend_catalog.py merge \
  --catalog model-registry/catalog.json \
  "${backend_entry_args[@]}" \
  --out "$workdir/catalog.merged.json"
python3 - "$workdir/catalog.merged.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
catalog = json.loads(path.read_text(encoding="utf-8"))
catalog["generated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

old_epoch="$(tr -d '[:space:]' < model-registry/catalog.epoch)"
[[ "$old_epoch" =~ ^[0-9]+$ ]] || fail "model-registry/catalog.epoch is invalid"
new_epoch="$((old_epoch + 1))"
cp "$workdir/catalog.merged.json" model-registry/catalog.json
printf '%s\n' "$new_epoch" > model-registry/catalog.epoch

echo "==> signing full + public catalogs at epoch ${new_epoch}"
OPENASR_CATALOG_EPOCH="$new_epoch" \
  tooling/publish-model/scripts/publish_catalog.sh

python3 tooling/release-manifest/backend_catalog.py verify-catalog \
  --catalog model-registry/catalog.json \
  "${backend_entry_args[@]}"
python3 tooling/release-manifest/backend_catalog.py verify-catalog \
  --catalog model-registry/catalog.public.json \
  "${backend_entry_args[@]}"
python3 tooling/publish-model/scripts/check_catalog_consistency.py

restore=0
echo
echo "CATALOG-PREPARED for ${tag}"
echo "  hardware-approved backend entries: ${#approved_entries[@]} of ${#backend_entries[@]} built"
echo "  epoch: ${old_epoch} -> ${new_epoch}"
echo "  next: review and commit model-registry/catalog{,.public}{,.signature}.json + catalog.epoch"
echo "  then push the catalog commit, wait for deploy-catalog.yml, and run:"
echo "    scripts/finalize-core-release.sh ${tag}"
