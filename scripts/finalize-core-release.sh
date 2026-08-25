#!/usr/bin/env bash
# Publish a draft core release only after its signed public catalog exposes the
# new Windows GPU provider bytes as PublishedInert. Real-hardware qualification
# happens after publication and can only activate a later signed catalog epoch.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

fail() { printf '\nRELEASE FINALIZATION FAILED for %s\n%s\n' "${tag:-<unknown>}" "$1" >&2; exit 1; }
trap 'fail "aborted at line $LINENO"' ERR

resolve_tag_commit() {
  local repository="$1"
  local tag="$2"
  local object_type object_sha next_type next_sha
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

[ "$#" -eq 1 ] || fail "usage: $(basename "$0") vX.Y.Z"
version="${1#v}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "version must be X.Y.Z or vX.Y.Z"
tag="v${version}"
repository="QuintinShaw/openasr"
command -v gh >/dev/null 2>&1 || fail "gh is required"
command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v cargo >/dev/null 2>&1 || fail "cargo is required"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"
gh auth status >/dev/null 2>&1 || fail "gh is not authenticated"
if [ -z "${SSL_CERT_FILE:-}" ]; then
  certifi="$(python3 -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
  if [ -n "$certifi" ]; then
    export SSL_CERT_FILE="$certifi"
  fi
fi

repository="${GITHUB_REPOSITORY:-QuintinShaw/openasr}"
is_draft="$(gh release view "$tag" --repo "$repository" --json isDraft --jq .isDraft 2>/dev/null)" \
  || fail "GitHub release ${tag} does not exist"
[ "$is_draft" = "true" ] || fail "release ${tag} is already public or is not a draft"

tag_commit="$(resolve_tag_commit "$repository" "$tag" || true)"
[[ "$tag_commit" =~ ^[0-9a-f]{40}$ ]] || fail "cannot peel ${tag} to one commit"
current_commit="$(git rev-parse HEAD)"
[[ "$current_commit" =~ ^[0-9a-f]{40}$ ]] || fail "current worktree HEAD is invalid"
git merge-base --is-ancestor "$tag_commit" "$current_commit" \
  || fail "current catalog commit does not descend from ${tag}"
[ -z "$(git status --porcelain --untracked-files=normal)" ] \
  || fail "the open-core worktree must be clean before finalization"
release_signer="${repository}/.github/workflows/release-binaries.yml"

workdir="$(mktemp -d "${TMPDIR:-/tmp}/openasr-release-finalize.XXXXXX")"
lock_token="$workdir/qualification-release-lock.token"
lock_acquired=false
cleanup() {
  if [ "$lock_acquired" = "true" ]; then
    scripts/qualification-release-lock.sh release "$tag" "$lock_token" \
      || printf 'warning: qualification release lock requires manual cleanup\n' >&2
  fi
  rm -rf -- "$workdir"
}
trap cleanup EXIT
scripts/qualification-release-lock.sh acquire "$tag" "$lock_token"
lock_acquired=true
[ "$(gh release view "$tag" --repo "$repository" --json isDraft --jq .isDraft)" = "true" ] \
  || fail "release ${tag} stopped being a draft while acquiring the finalization lock"
gh release download "$tag" --repo "$repository" \
  -p 'backend-pack-*.json' \
  -p 'backend-plugin-hints.json' \
  -p 'catalog.backends.candidate.json' \
  -p 'openasr-*' \
  -p 'SHA256SUMS' \
  -D "$workdir" --clobber

shopt -s nullglob
backend_entries=("$workdir"/backend-pack-*.json)
cuda_entries=("$workdir"/backend-pack-cuda-sm_*.json)
hip_entries=("$workdir"/backend-pack-hip-gfx*.json)
vulkan_entries=("$workdir"/backend-pack-vulkan-generic.json)
if [ "${#cuda_entries[@]}" -ne 6 ] || [ "${#hip_entries[@]}" -ne 14 ] || [ "${#vulkan_entries[@]}" -ne 1 ] || [ "${#backend_entries[@]}" -ne 21 ]; then
  fail "release ${tag} must contain 1 Vulkan, 6 CUDA SM, and 14 HIP gfx backend-pack metadata files"
fi
checksums="$workdir/SHA256SUMS"
[ -f "$checksums" ] || fail "release ${tag} has no SHA256SUMS"

# SHA256SUMS is not trusted by itself. Every subject named by it must both match
# and carry GitHub provenance from the exact peeled release commit. Qualification
# manifests, detached signatures, and the copied provenance bundle are verified
# separately below because they are intentionally not self-listed as subjects.
verified_subjects=0
while read -r _checksum asset_name || [ -n "${asset_name:-}" ]; do
  asset_name="${asset_name#\*}"
  [ -n "$asset_name" ] || continue
  case "$asset_name" in
    */*|*\\*) fail "SHA256SUMS contains an unsafe release basename: $asset_name" ;;
  esac
  subject="$workdir/$asset_name"
  [ -f "$subject" ] || fail "SHA256SUMS subject is missing from the release: $asset_name"
  python3 tooling/release-manifest/release_asset_verifier.py \
    --asset "$subject" --checksums "$checksums" >/dev/null
  gh attestation verify "$subject" \
    --repo "$repository" --signer-workflow "$release_signer" \
    --source-digest "$tag_commit" --format=json >/dev/null \
    || fail "release subject attestation failed: $(basename "$subject")"
  verified_subjects=$((verified_subjects + 1))
done < <(tr -d '\r' < "$checksums")
[ "$verified_subjects" -gt 21 ] || fail "release ${tag} did not expose a complete attested subject set"

all_backend_entry_args=()
for entry in "${backend_entries[@]}"; do
  all_backend_entry_args+=(--entry "$entry")
done

python3 tooling/release-manifest/backend_catalog.py verify-assets \
  "${all_backend_entry_args[@]}" \
  --asset-directory "$workdir" --version "$version"

candidate="$workdir/catalog.backends.candidate.json"
[ -f "$candidate" ] || fail "release ${tag} has no backend catalog candidate"
python3 tooling/release-manifest/backend_catalog.py verify-catalog \
  --catalog "$candidate" "${all_backend_entry_args[@]}"
python3 tooling/release-manifest/backend_hardware_evidence.py \
  "${all_backend_entry_args[@]}" --catalog "$candidate" --version "$version" >/dev/null

# The reusable deploy workflow is the only catalog publication path. Initial
# release publication accepts only its already-verified PublishedInert epoch.
deploy_run_id="${OPENASR_DEPLOY_CATALOG_RUN_ID:-}"
[ -n "$deploy_run_id" ] || fail "set OPENASR_DEPLOY_CATALOG_RUN_ID to the successful reusable catalog-deploy run"
deploy_conclusion="$(gh run view "$deploy_run_id" --repo "$repository" --json conclusion --jq .conclusion)"
[ "$deploy_conclusion" = "success" ] \
  || fail "catalog deploy run $deploy_run_id did not succeed (conclusion=$deploy_conclusion)"

# Re-read the annotated remote tag while holding the qualification mutation
# lock. Its peeled commit must still equal the release identity checked before
# the lock was acquired.
IFS=$'\t' read -r remote_tag_type remote_tag_object < <(
  gh api "repos/${repository}/git/ref/tags/${tag}" --jq '[.object.type,.object.sha] | @tsv'
)
[ "$remote_tag_type" = "tag" ] || fail "${tag} is not an annotated GitHub release tag"
IFS=$'\t' read -r remote_target_type locked_tag_commit < <(
  gh api "repos/${repository}/git/tags/${remote_tag_object}" \
    --jq '[.object.type,.object.sha] | @tsv'
)
[ "$remote_target_type" = "commit" ] && [[ "$locked_tag_commit" =~ ^[0-9a-f]{40}$ ]] \
  || fail "cannot peel annotated tag ${tag} to one source commit"
[ "$locked_tag_commit" = "$tag_commit" ] \
  || fail "release tag ${tag} changed while acquiring the finalization lock"

# Qualification metadata is intentionally generated after SHA256SUMS and its
# provenance statement, so it is authorized by detached production signatures
# rather than pretending to be one of that statement's subjects. Rebuild the
# exact set from the release's backend packs, re-hash every referenced release
# byte, and verify both signature domains before publication.
python3 - "$workdir" "$version" "$tag_commit" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
version = sys.argv[2]
tag_commit = sys.argv[3]
sys.path.insert(0, "tooling/release-manifest")
import qualification_manifest as compiler

cells = set()
for entry_path in sorted(root.glob("backend-pack-*.json")):
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    try:
        cell = compiler.artifact_cell(entry)
    except compiler.QualificationManifestError as error:
        raise SystemExit(f"invalid backend qualification entry {entry_path.name}: {error}") from error
    if cell in cells:
        raise SystemExit(f"duplicate qualification cell: {cell}")
    cells.add(cell)

expected_manifests = {
    compiler.manifest_asset_name(version, provider, target)
    for provider, target in cells
}
expected_signatures = {
    f"{Path(name).stem}.signature.json" for name in expected_manifests
}
qualification_paths = set(root.glob(f"openasr-{version}-qualification-*.json"))
actual_manifests = {
    path.name for path in qualification_paths if not path.name.endswith(".signature.json")
}
actual_signatures = {
    path.name for path in qualification_paths if path.name.endswith(".signature.json")
}
if actual_manifests != expected_manifests or actual_signatures != expected_signatures:
    raise SystemExit(
        "qualification manifest/signature set differs from backend-pack cells: "
        f"manifest_missing={sorted(expected_manifests-actual_manifests)} "
        f"manifest_extra={sorted(actual_manifests-expected_manifests)} "
        f"signature_missing={sorted(expected_signatures-actual_signatures)} "
        f"signature_extra={sorted(actual_signatures-expected_signatures)}"
    )

bundle_name = f"openasr-{version}-build-provenance.bundle.json"
bundle_path = root / bundle_name
predicate, subjects = compiler._attestation_subjects(bundle_path)
if predicate != compiler.ATTESTATION_PREDICATE_TYPE:
    raise SystemExit("qualification provenance predicate changed")
referenced = set()
rows = []
for manifest_name in sorted(expected_manifests):
    value = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    if value.get("schema_version") != compiler.SCHEMA_VERSION or value.get("release_subject") != f"v{version}":
        raise SystemExit(f"qualification manifest identity changed: {manifest_name}")
    provider_target = value.get("provider_target")
    if not isinstance(provider_target, dict):
        raise SystemExit(f"qualification manifest has no exact cell: {manifest_name}")
    cell = (provider_target.get("provider"), provider_target.get("target"))
    if cell not in cells or compiler.manifest_asset_name(version, *cell) != manifest_name:
        raise SystemExit(f"qualification manifest cell/name mismatch: {manifest_name}")
    attestation = value.get("attestation")
    artifacts = value.get("artifacts")
    if not isinstance(attestation, dict) or not isinstance(artifacts, dict):
        raise SystemExit(f"qualification manifest is incomplete: {manifest_name}")
    if attestation.get("source_digest") != tag_commit:
        raise SystemExit(f"qualification manifest source differs from tag: {manifest_name}")
    bundle = attestation.get("bundle")
    if not isinstance(bundle, dict) or bundle.get("file_name") != bundle_name:
        raise SystemExit(f"qualification manifest bundle differs: {manifest_name}")
    binary = artifacts.get("binary")
    if not isinstance(binary, dict):
        raise SystemExit(f"qualification manifest binary is missing: {manifest_name}")
    values = [binary.get("bundle"), artifacts.get("plugin"), *artifacts.get("vendor", []), bundle]
    for artifact in values:
        if artifact is None:
            continue
        if not isinstance(artifact, dict):
            raise SystemExit(f"qualification artifact is malformed: {manifest_name}")
        file_name = compiler._safe_basename(artifact.get("file_name"), "release artifact")
        path = root / file_name
        digest, size = compiler._sha256_size(path)
        if digest != artifact.get("sha256") or size != artifact.get("size_bytes"):
            raise SystemExit(f"qualification release bytes changed: {file_name}")
        if file_name != bundle_name:
            compiler._require_attested(subjects, path)
            referenced.add(file_name)
    signature_name = f"{Path(manifest_name).stem}.signature.json"
    rows.append(
        (manifest_name, f"https://dl.openasr.org/core/v{version}/{manifest_name}", signature_name)
    )

(root / "qualification-index.tsv").write_text(
    "".join("\t".join(row) + "\n" for row in rows), encoding="utf-8"
)
(root / "qualification-subjects.txt").write_text(
    "".join(name + "\n" for name in sorted(referenced)), encoding="utf-8"
)
(root / "qualification-bundle-name.txt").write_text(bundle_name + "\n", encoding="utf-8")
PY

while IFS=$'\t' read -r manifest_name manifest_url signature_name; do
  OPENASR_HOME="$workdir/home-qualification" \
    cargo run --quiet -p openasr-cli -- __openasr-verify-qualification-manifest \
    "$workdir/$manifest_name" --signature "$workdir/$signature_name" \
    --manifest-url "$manifest_url" >/dev/null
done < "$workdir/qualification-index.tsv"
qualification_bundle="$(tr -d '\r\n' < "$workdir/qualification-bundle-name.txt")"
while IFS= read -r asset_name || [ -n "$asset_name" ]; do
  [ -n "$asset_name" ] || continue
  gh attestation verify "$workdir/$asset_name" \
    --repo "$repository" \
    --signer-workflow "${repository}/.github/workflows/release-binaries.yml" \
    --source-digest "$tag_commit" \
    --predicate-type "https://slsa.dev/provenance/v1" \
    --deny-self-hosted-runners \
    --bundle "$workdir/$qualification_bundle" \
    --format json >/dev/null
done < "$workdir/qualification-subjects.txt"

deploy_metadata="$workdir/deploy-run.json"
gh run view "$deploy_run_id" --repo "$repository" \
  --json workflowName,conclusion,headSha,event,jobs,url > "$deploy_metadata" \
  || fail "cannot inspect catalog deploy run $deploy_run_id"
python3 - "$deploy_metadata" "$current_commit" <<'PY' \
  || fail "catalog deploy run is not bound to this committed PublishedInert catalog"
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("workflowName") != "Release core":
    raise SystemExit("deploy binding came from another workflow")
if value.get("conclusion") != "success" or value.get("event") != "workflow_dispatch":
    raise SystemExit("release finalization run did not complete successfully by explicit dispatch")
if value.get("headSha") != sys.argv[2]:
    raise SystemExit("release finalization run used another catalog commit")
jobs = value.get("jobs")
if not isinstance(jobs, list) or not any(
    isinstance(job, dict)
    and "Deploy PublishedInert candidate catalog" in str(job.get("name", ""))
    and job.get("conclusion") == "success"
    for job in jobs
):
    raise SystemExit("run has no successful PublishedInert catalog deploy job")
PY

deploy_binding_dir="$workdir/deploy-binding"
mkdir -p "$deploy_binding_dir"
gh run download "$deploy_run_id" --repo "$repository" \
  --name "deploy-catalog-binding-${deploy_run_id}" --dir "$deploy_binding_dir" \
  || fail "catalog deploy run has no immutable release binding artifact"
deploy_binding="$deploy_binding_dir/deploy-catalog-binding.json"
[ -f "$deploy_binding" ] \
  || fail "catalog deploy binding artifact has no deploy-catalog-binding.json"
python3 - "$deploy_binding" "$tag" "$deploy_run_id" "$current_commit" \
  model-registry/catalog.public.json model-registry/catalog.public.signature.json <<'PY' \
  || fail "catalog deploy binding does not match this tag and committed catalog"
import hashlib, json, pathlib, sys

binding_path, tag, run_id, commit, catalog_path, signature_path = sys.argv[1:]
value = json.load(open(binding_path, encoding="utf-8"))
expected_keys = {
    "schema_version", "release_tag", "activation_transition", "backend_id",
    "orchestrator_run_id", "deploy_run_id", "source_commit", "catalog_sha256",
    "catalog_signature_sha256",
}
if not isinstance(value, dict) or set(value) != expected_keys:
    raise SystemExit("deploy binding has an unexpected schema")
def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
expected = {
    "schema_version": 1,
    "release_tag": tag,
    "activation_transition": "published-inert",
    "backend_id": "",
    "orchestrator_run_id": run_id,
    "deploy_run_id": run_id,
    "source_commit": commit,
    "catalog_sha256": digest(catalog_path),
    "catalog_signature_sha256": digest(signature_path),
}
if value != expected:
    raise SystemExit("deploy binding values do not match the requested release")
PY

python3 tooling/publish-model/scripts/check_catalog_consistency.py \
  || fail "committed catalog/signature pair does not verify under production trust roots"

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
  --catalog "$workdir/catalog.json" "${all_backend_entry_args[@]}"
python3 tooling/release-manifest/backend_hardware_evidence.py \
  "${all_backend_entry_args[@]}" \
  --catalog "$workdir/catalog.json" --version "$version" >/dev/null
python3 tooling/release-manifest/backend_catalog.py verify-cdn \
  --catalog "$workdir/catalog.json" --version "$version"
cmp -s model-registry/catalog.public.json "$workdir/catalog.json" \
  || fail "live catalog bytes differ from the deploy run's committed catalog"
cmp -s model-registry/catalog.public.signature.json "$workdir/catalog.signature.json" \
  || fail "live catalog signature differs from the deploy run's committed signature"

echo "==> signed catalog exposes ${tag} provider bytes as PublishedInert; publishing release"
[ "$(gh release view "$tag" --repo "$repository" --json isDraft --jq .isDraft)" = "true" ] \
  || fail "release ${tag} stopped being a draft before publication"
gh release edit "$tag" --repo "$repository" --draft=false --latest
scripts/qualification-release-lock.sh release "$tag" "$lock_token"
lock_acquired=false
echo "RELEASE-PUBLISHED-INERT ${tag}"
