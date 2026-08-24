#!/usr/bin/env bash
# Prepare (without deploying) one post-publication PublishedInert -> Qualified
# -> Activated catalog transition from attested Windows qualification runs.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

fail() { printf '\nBACKEND ACTIVATION PREPARATION FAILED\n%s\n' "$1" >&2; exit 1; }
trap 'fail "aborted at line $LINENO"' ERR

resolve_tag_commit() {
  local repository="$1" tag="$2" object_type object_sha next_type next_sha
  object_type="$(gh api "repos/${repository}/git/ref/tags/${tag}" --jq .object.type 2>/dev/null)" || return 1
  object_sha="$(gh api "repos/${repository}/git/ref/tags/${tag}" --jq .object.sha 2>/dev/null)" || return 1
  while [ "$object_type" = "tag" ]; do
    next_type="$(gh api "repos/${repository}/git/tags/${object_sha}" --jq .object.type 2>/dev/null)" || return 1
    next_sha="$(gh api "repos/${repository}/git/tags/${object_sha}" --jq .object.sha 2>/dev/null)" || return 1
    object_type="$next_type"
    object_sha="$next_sha"
  done
  [ "$object_type" = "commit" ] || return 1
  printf '%s\n' "$object_sha"
}

[ "$#" -ge 3 ] || fail "usage: $(basename "$0") vX.Y.Z BACKEND_ID QUALIFICATION_RUN_ID [RUN_ID ...]"
version="${1#v}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "version must be X.Y.Z or vX.Y.Z"
tag="v${version}"
backend_id="$2"
shift 2
run_ids=("$@")
[[ "$backend_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$ ]] || fail "backend id is invalid"
for run_id in "${run_ids[@]}"; do
  [[ "$run_id" =~ ^[1-9][0-9]*$ ]] || fail "qualification run ids must be positive integers"
done

if [ "${CI:-}" = "true" ] || [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  fail "refusing to use the production catalog signing seed in CI"
fi
[[ "${OPENASR_CATALOG_SIGNING_KEY_SEED_HEX:-}" =~ ^[0-9a-fA-F]{64}$ ]] \
  || fail "OPENASR_CATALOG_SIGNING_KEY_SEED_HEX must contain the production 64-hex seed"
for command in gh cargo curl python3; do
  command -v "$command" >/dev/null 2>&1 || fail "$command is required"
done
gh auth status >/dev/null 2>&1 || fail "gh is not authenticated"
[ -z "$(git status --porcelain --untracked-files=normal)" ] \
  || fail "the open-core worktree must be clean before activation preparation"

repository="${GITHUB_REPOSITORY:-QuintinShaw/openasr}"
tag_commit="$(resolve_tag_commit "$repository" "$tag" || true)"
[[ "$tag_commit" =~ ^[0-9a-f]{40}$ ]] || fail "cannot peel ${tag} to one commit"
git merge-base --is-ancestor "$tag_commit" HEAD \
  || fail "current catalog branch does not descend from ${tag}"
release_state="$(gh release view "$tag" --repo "$repository" \
  --json isDraft,isPrerelease,tagName,publishedAt 2>/dev/null)" \
  || fail "release ${tag} does not exist"
python3 - "$release_state" "$tag" <<'PY' \
  || fail "qualification may consume only already-public stable PublishedInert bytes"
import json, sys
value = json.loads(sys.argv[1])
if (value.get("isDraft") or value.get("isPrerelease")
        or value.get("tagName") != sys.argv[2]
        or not value.get("publishedAt")):
    raise SystemExit("release is not public and stable")
PY

workdir="$(mktemp -d "${TMPDIR:-/tmp}/openasr-backend-activation.XXXXXX")"
restore=1
cleanup() {
  if [ "$restore" = "1" ]; then
    for name in catalog.json catalog.signature.json catalog.public.json catalog.public.signature.json catalog.epoch; do
      [ -f "$workdir/original-$name" ] && cp "$workdir/original-$name" "model-registry/$name"
    done
  fi
  rm -rf "$workdir"
}
trap cleanup EXIT
for name in catalog.json catalog.signature.json catalog.public.json catalog.public.signature.json catalog.epoch; do
  cp "model-registry/$name" "$workdir/original-$name"
done
python3 tooling/publish-model/scripts/check_catalog_consistency.py \
  || fail "current committed catalog/signature pair is not production-valid"
curl -fsSL "https://catalog.openasr.org/v1/catalog.json?activation-preflight=${tag_commit}" \
  -o "$workdir/live-catalog.json"
curl -fsSL "https://catalog.openasr.org/v1/catalog.signature.json?activation-preflight=${tag_commit}" \
  -o "$workdir/live-catalog.signature.json"
python3 tooling/publish-model/scripts/check_catalog_consistency.py \
  --catalog-pair "$workdir/live-catalog.json" "$workdir/live-catalog.signature.json" \
  || fail "live catalog/signature pair is not production-valid"
cmp -s model-registry/catalog.public.json "$workdir/live-catalog.json" \
  || fail "local activation base catalog differs from the current live catalog"
cmp -s model-registry/catalog.public.signature.json "$workdir/live-catalog.signature.json" \
  || fail "local activation base signature differs from the current live signature"

mkdir -p "$workdir/runs" "$workdir/release"
shopt -s nullglob
for run_id in "${run_ids[@]}"; do
  run_meta="$workdir/run-${run_id}.json"
  gh run view "$run_id" --repo "$repository" \
    --json workflowName,conclusion,headSha,databaseId > "$run_meta"
  python3 - "$run_meta" "$tag_commit" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("workflowName") != "Qualify Windows backend":
    raise SystemExit("qualification evidence came from another workflow")
if value.get("conclusion") != "success" or value.get("headSha") != sys.argv[2]:
    raise SystemExit("qualification run did not succeed on the exact release commit")
PY
  gh run download "$run_id" --repo "$repository" --dir "$workdir/runs/$run_id"
done

first_run_root="$workdir/runs/${run_ids[0]}"
find_one() {
  local root="$1" pattern="$2" label="$3"
  local paths=()
  while IFS= read -r path || [ -n "$path" ]; do
    [ -n "$path" ] && paths+=("$path")
  done < <(find "$root" -type f -name "$pattern" -print | sort)
  [ "${#paths[@]}" -eq 1 ] || fail "expected one $label under $root, found ${#paths[@]}"
  printf '%s\n' "${paths[0]}"
}

matrix="$(find_one "$first_run_root" 'gpu-correctness-matrix.v1.json' matrix)"
inventory="$(find_one "$first_run_root" 'gpu-correctness-source-inventory.json' inventory)"
model_catalog="$(find_one "$first_run_root" 'gpu-correctness-source-model-catalog.json' 'model catalog')"
backend_catalog="$(find_one "$first_run_root" 'gpu-correctness-source-backend-catalog.json' 'backend catalog')"
hardware_evidence="$(find_one "$first_run_root" 'backend-hardware-evidence-*.json' 'hardware evidence')"
hardware_raw="$(find_one "$first_run_root" 'backend-hardware-audit-*.json' 'hardware audit')"

correctness_receipts=()
correctness_traces=()
for run_id in "${run_ids[@]}"; do
  run_root="$workdir/runs/$run_id"
  for canonical in \
    'gpu-correctness-matrix.v1.json' \
    'gpu-correctness-source-inventory.json' \
    'gpu-correctness-source-model-catalog.json' \
    'gpu-correctness-source-backend-catalog.json'; do
    candidate="$(find_one "$run_root" "$canonical" "$canonical")"
    reference="$(find_one "$first_run_root" "$canonical" "$canonical")"
    cmp -s "$candidate" "$reference" || fail "qualification runs used different $canonical bytes"
  done
  while IFS= read -r path || [ -n "$path" ]; do
    [ -n "$path" ] && correctness_receipts+=("$path")
  done < <(find "$run_root" -type f -name 'gpu-correctness-receipt-*.json' -print | sort)
  while IFS= read -r path || [ -n "$path" ]; do
    [ -n "$path" ] && correctness_traces+=("$path")
  done < <(find "$run_root" -type f -name 'gpu-correctness-trace-*.jsonl' -print | sort)
done
[ "${#correctness_receipts[@]}" -ge 4 ] || fail "qualification runs contain no correctness receipt cell"
[ "${#correctness_traces[@]}" -ge 2 ] || fail "qualification runs contain no correctness traces"

gh release download "$tag" --repo "$repository" \
  -p 'backend-pack-*.json' -p 'SHA256SUMS' \
  -D "$workdir/release" --clobber
backend_entries=("$workdir"/release/backend-pack-*.json)
[ "${#backend_entries[@]}" -eq 21 ] || fail "release does not contain exactly 21 backend entries"

while IFS= read -r filename || [ -n "$filename" ]; do
  [ -n "$filename" ] || continue
  gh release download "$tag" --repo "$repository" \
    -p "$filename" -D "$workdir/release" --clobber
done < <(python3 - "$hardware_raw" <<'PY'
import json, pathlib, sys
raw = json.load(open(sys.argv[1], encoding="utf-8"))
names = set()
for subject in raw.get("attested_release_subjects", []):
    name = subject.get("filename") if isinstance(subject, dict) else None
    if not isinstance(name, str) or pathlib.Path(name).name != name:
        raise SystemExit("hardware audit has an unsafe release subject")
    names.add(name)
print("\n".join(sorted(names)))
PY
)

entry_args=()
for entry in "${backend_entries[@]}"; do entry_args+=(--entry "$entry"); done
release_subject_args=()
while IFS= read -r filename || [ -n "$filename" ]; do
  [ -n "$filename" ] || continue
  release_subject_args+=(--release-subject "$workdir/release/$filename")
done < <(python3 - "$hardware_raw" <<'PY'
import json, sys
print("\n".join(sorted(item["filename"] for item in json.load(open(sys.argv[1]))["attested_release_subjects"])))
PY
)

release_signer="${repository}/.github/workflows/release-binaries.yml"
qualification_signer="${repository}/.github/workflows/qualify-windows-backend.yml"
python3 tooling/release-manifest/backend_hardware_evidence.py \
  "${entry_args[@]}" --evidence "$hardware_evidence" --raw-audit "$hardware_raw" \
  "${release_subject_args[@]}" --checksums "$workdir/release/SHA256SUMS" \
  --repo "$repository" --signer-workflow "$release_signer" \
  --qualification-signer-workflow "$qualification_signer" \
  --source-digest "$tag_commit" > "$workdir/hardware-approved.txt"

receipt_args=()
for receipt in "${correctness_receipts[@]}"; do receipt_args+=(--receipt "$receipt"); done
trace_args=()
for trace in "${correctness_traces[@]}"; do trace_args+=(--trace "$trace"); done
hardware_args=(--hardware-evidence "$hardware_evidence" --hardware-raw-audit "$hardware_raw")
evidence_attestation_args=(
  --evidence-repo "$repository"
  --evidence-signer-workflow "$qualification_signer"
  --evidence-source-digest "$tag_commit"
)

python3 tooling/release-manifest/gpu_correctness_gate.py qualify-catalog \
  --manifest "$matrix" --inventory "$inventory" --catalog "$model_catalog" \
  --backend-catalog "$backend_catalog" \
  --current-activation-catalog model-registry/catalog.public.json \
  --backend-id "$backend_id" "${entry_args[@]}" "${hardware_args[@]}" \
  --out "$workdir/catalog.qualified.json"

python3 tooling/release-manifest/gpu_correctness_gate.py activate-catalog \
  --manifest "$matrix" --inventory "$inventory" --catalog "$model_catalog" \
  --backend-catalog "$backend_catalog" \
  --current-activation-catalog "$workdir/catalog.qualified.json" \
  --backend-id "$backend_id" "${entry_args[@]}" "${hardware_args[@]}" \
  "${receipt_args[@]}" "${trace_args[@]}" "${evidence_attestation_args[@]}" \
  --out "$workdir/catalog.activated.json"

python3 tooling/release-manifest/gpu_correctness_gate.py verify-catalog-transition \
  --manifest "$matrix" --inventory "$inventory" --catalog "$model_catalog" \
  --backend-catalog "$backend_catalog" \
  --current-activation-catalog model-registry/catalog.public.json \
  --candidate-activation-catalog "$workdir/catalog.activated.json" \
  --backend-id "$backend_id" "${entry_args[@]}" "${hardware_args[@]}" \
  "${receipt_args[@]}" "${trace_args[@]}" "${evidence_attestation_args[@]}"

python3 - model-registry/catalog.json "$workdir/catalog.activated.json" "$backend_id" <<'PY'
import json, pathlib, sys
full_path, activated_path = map(pathlib.Path, sys.argv[1:3])
backend_id = sys.argv[3]
full = json.loads(full_path.read_text(encoding="utf-8"))
activated = json.loads(activated_path.read_text(encoding="utf-8"))
matches = [item for item in activated.get("backends", []) if item.get("id") == backend_id]
if len(matches) != 1:
    raise SystemExit("activated projection does not contain one requested backend")
found = 0
for index, item in enumerate(full.get("backends", [])):
    if item.get("id") == backend_id:
        full["backends"][index] = matches[0]
        found += 1
if found != 1:
    raise SystemExit("full catalog does not contain one requested backend")
full_path.write_bytes((json.dumps(full, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
PY

python3 - model-registry/catalog.json <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
catalog = json.loads(path.read_text(encoding="utf-8"))
catalog["generated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
path.write_bytes((json.dumps(catalog, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
PY
old_epoch="$(tr -d '[:space:]' < model-registry/catalog.epoch)"
[[ "$old_epoch" =~ ^[0-9]+$ ]] || fail "catalog.epoch is invalid"
new_epoch="$((old_epoch + 1))"
printf '%s\n' "$new_epoch" > model-registry/catalog.epoch
OPENASR_CATALOG_EPOCH="$new_epoch" tooling/publish-model/scripts/publish_catalog.sh

python3 - model-registry/catalog.public.json "$workdir/catalog.activated.json" <<'PY'
import json, sys
actual = json.load(open(sys.argv[1], encoding="utf-8"))
expected = json.load(open(sys.argv[2], encoding="utf-8"))
if actual.get("backends") != expected.get("backends"):
    raise SystemExit("signed public catalog changed the verified backend transition")
PY
python3 tooling/publish-model/scripts/check_catalog_consistency.py

restore=0
echo
echo "BACKEND-ACTIVATION-PREPARED"
echo "  release: $tag"
echo "  backend: $backend_id"
echo "  qualification runs: ${run_ids[*]}"
echo "  epoch: ${old_epoch} -> ${new_epoch}"
echo "  next: review and commit the five model-registry catalog/epoch files"
echo "  then dispatch activate-backend-catalog.yml and separately authorize deployment"
