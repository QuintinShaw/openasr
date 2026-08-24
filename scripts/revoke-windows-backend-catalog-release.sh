#!/usr/bin/env bash
# Prepare (without deploying) one signed catalog epoch that revokes an exact
# Windows backend while preserving its former qualification bindings for audit.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

fail() { printf '\nBACKEND REVOCATION PREPARATION FAILED\n%s\n' "$1" >&2; exit 1; }
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

[ "$#" -eq 2 ] || fail "usage: $(basename "$0") vX.Y.Z BACKEND_ID"
version="${1#v}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "version must be X.Y.Z or vX.Y.Z"
tag="v${version}"
backend_id="$2"
[[ "$backend_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$ ]] || fail "backend id is invalid"

if [ "${CI:-}" = "true" ] || [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  fail "refusing to use the production catalog signing seed in CI"
fi
[[ "${OPENASR_CATALOG_SIGNING_KEY_SEED_HEX:-}" =~ ^[0-9a-fA-F]{64}$ ]] \
  || fail "OPENASR_CATALOG_SIGNING_KEY_SEED_HEX must contain the production 64-hex seed"
for command in gh curl python3; do
  command -v "$command" >/dev/null 2>&1 || fail "$command is required"
done
gh auth status >/dev/null 2>&1 || fail "gh is not authenticated"
[ -z "$(git status --porcelain --untracked-files=normal)" ] \
  || fail "the open-core worktree must be clean before revocation preparation"

repository="${GITHUB_REPOSITORY:-QuintinShaw/openasr}"
tag_commit="$(resolve_tag_commit "$repository" "$tag" || true)"
[[ "$tag_commit" =~ ^[0-9a-f]{40}$ ]] || fail "cannot peel ${tag} to one commit"
git merge-base --is-ancestor "$tag_commit" HEAD \
  || fail "current catalog branch does not descend from ${tag}"
release_state="$(gh release view "$tag" --repo "$repository" \
  --json isDraft,isPrerelease,tagName,publishedAt 2>/dev/null)" \
  || fail "release ${tag} does not exist"
python3 - "$release_state" "$tag" <<'PY' \
  || fail "revocation requires an already-public stable release"
import json, sys
value = json.loads(sys.argv[1])
if (value.get("isDraft") or value.get("isPrerelease")
        or value.get("tagName") != sys.argv[2]
        or not value.get("publishedAt")):
    raise SystemExit("release is not public and stable")
PY

workdir="$(mktemp -d "${TMPDIR:-/tmp}/openasr-backend-revocation.XXXXXX")"
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
curl -fsSL "https://catalog.openasr.org/v1/catalog.json?revocation-preflight=${tag_commit}" \
  -o "$workdir/live-catalog.json"
curl -fsSL "https://catalog.openasr.org/v1/catalog.signature.json?revocation-preflight=${tag_commit}" \
  -o "$workdir/live-catalog.signature.json"
python3 tooling/publish-model/scripts/check_catalog_consistency.py \
  --catalog-pair "$workdir/live-catalog.json" "$workdir/live-catalog.signature.json" \
  || fail "live catalog/signature pair is not production-valid"
cmp -s model-registry/catalog.public.json "$workdir/live-catalog.json" \
  || fail "local revocation base catalog differs from the current live catalog"
cmp -s model-registry/catalog.public.signature.json "$workdir/live-catalog.signature.json" \
  || fail "local revocation base signature differs from the current live signature"

python3 - model-registry/catalog.public.json "$backend_id" "$version" <<'PY' \
  || fail "backend is absent, already revoked, or belongs to another release"
import json, sys
catalog = json.load(open(sys.argv[1], encoding="utf-8"))
matches = [item for item in catalog.get("backends", []) if item.get("id") == sys.argv[2]]
if len(matches) != 1 or str(matches[0].get("version")) != sys.argv[3]:
    raise SystemExit("backend identity/version mismatch")
if matches[0].get("activation", {}).get("state", "published-inert") == "revoked":
    raise SystemExit("backend is already revoked")
PY

python3 tooling/release-manifest/gpu_correctness_gate.py revoke-catalog \
  --current-activation-catalog model-registry/catalog.public.json \
  --backend-id "$backend_id" --out "$workdir/catalog.revoked.json"
python3 tooling/release-manifest/gpu_correctness_gate.py verify-revocation-transition \
  --current-activation-catalog model-registry/catalog.public.json \
  --candidate-activation-catalog "$workdir/catalog.revoked.json" \
  --backend-id "$backend_id"

python3 - model-registry/catalog.json "$workdir/catalog.revoked.json" "$backend_id" <<'PY'
import json, pathlib, sys
full_path, revoked_path = map(pathlib.Path, sys.argv[1:3])
backend_id = sys.argv[3]
full = json.loads(full_path.read_text(encoding="utf-8"))
revoked = json.loads(revoked_path.read_text(encoding="utf-8"))
matches = [item for item in revoked.get("backends", []) if item.get("id") == backend_id]
if len(matches) != 1:
    raise SystemExit("revocation projection does not contain one requested backend")
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
python3 tooling/release-manifest/gpu_correctness_gate.py verify-revocation-transition \
  --current-activation-catalog "$workdir/original-catalog.public.json" \
  --candidate-activation-catalog model-registry/catalog.public.json \
  --backend-id "$backend_id"
python3 tooling/publish-model/scripts/check_catalog_consistency.py

restore=0
echo
echo "BACKEND-REVOCATION-PREPARED"
echo "  release: $tag"
echo "  backend: $backend_id"
echo "  epoch: ${old_epoch} -> ${new_epoch}"
echo "  next: review and commit the five model-registry catalog/epoch files"
echo "  then dispatch revoke-backend-catalog.yml and separately authorize deployment"
