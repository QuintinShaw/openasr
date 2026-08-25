#!/usr/bin/env bash
# Sign every inert exact-cell qualification manifest on a draft release, upload
# only the detached signatures, then re-download and verify the published pairs.
# The production seed is local-only and must never enter GitHub Actions.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

fail() {
  printf '\nQUALIFICATION SIGNING FAILED for %s\n%s\n' "${tag:-<unknown>}" "$1" >&2
  exit 1
}

trap 'fail "aborted at line $LINENO"' ERR

[ "$#" -eq 1 ] || {
  [ "$#" -eq 3 ] && [ "$2" = "--promote-cuda-targets" ] && [ -n "$3" ] \
    || fail "usage: $(basename "$0") vX.Y.Z [--promote-cuda-targets <csv>]"
}
version="${1#v}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
  || fail "version must be X.Y.Z or vX.Y.Z"
tag="v${version}"
repository="QuintinShaw/openasr"
promote_cuda_targets="${3:-}"

if [ "${CI:-}" = "true" ] || [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  fail "refusing to use the production catalog seed in CI"
fi
[[ "${OPENASR_CATALOG_SIGNING_KEY_SEED_HEX:-}" =~ ^[0-9a-fA-F]{64}$ ]] \
  || fail "OPENASR_CATALOG_SIGNING_KEY_SEED_HEX must contain the production 64-hex seed"
signing_key_seed="$OPENASR_CATALOG_SIGNING_KEY_SEED_HEX"
unset OPENASR_CATALOG_SIGNING_KEY_SEED_HEX
command -v gh >/dev/null 2>&1 || fail "gh is required"
command -v cargo >/dev/null 2>&1 || fail "cargo is required"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"
gh auth status >/dev/null 2>&1 || fail "gh is not authenticated"

[ -z "$(git status --porcelain --untracked-files=normal)" ] \
  || fail "the open-core worktree must be clean before qualification signing"
tag_commit="$(git rev-parse "${tag}^{commit}" 2>/dev/null)" \
  || fail "local tag ${tag} is missing"
[ "$(git cat-file -t "$tag")" = "tag" ] \
  || fail "${tag} must be an annotated release tag"
local_tag_object="$(git rev-parse "$tag")"
head_commit="$(git rev-parse HEAD^{commit})"
[ "$head_commit" = "$tag_commit" ] \
  || fail "HEAD ${head_commit} does not equal ${tag} commit ${tag_commit}"
IFS=$'\t' read -r remote_tag_type remote_tag_object < <(
  gh api "repos/${repository}/git/ref/tags/${tag}" --jq '[.object.type,.object.sha] | @tsv'
)
[ "$remote_tag_type" = "tag" ] && [ "$remote_tag_object" = "$local_tag_object" ] \
  || fail "local annotated tag ${tag} differs from the current GitHub tag object"
IFS=$'\t' read -r remote_target_type remote_tag_commit < <(
  gh api "repos/${repository}/git/tags/${remote_tag_object}" \
    --jq '[.object.type,.object.sha] | @tsv'
)
[ "$remote_target_type" = "commit" ] && [ "$remote_tag_commit" = "$tag_commit" ] \
  || fail "GitHub tag ${tag} does not peel to local commit ${tag_commit}"
is_draft="$(gh release view "$tag" --repo "$repository" --json isDraft --jq .isDraft 2>/dev/null)" \
  || fail "GitHub release ${tag} does not exist"
[ "$is_draft" = "true" ] \
  || fail "qualification manifests must be signed while ${tag} is still a draft"

workdir="$(mktemp -d "${TMPDIR:-/tmp}/openasr-qualification-sign.XXXXXX")"
verify_dir="$(mktemp -d "${TMPDIR:-/tmp}/openasr-qualification-verify.XXXXXX")"
lock_token="$workdir/release-lock-token"
lock_acquired=false
cleanup() {
  if [ "$lock_acquired" = "true" ]; then
    scripts/qualification-release-lock.sh release "$tag" "$lock_token" \
      || printf 'warning: qualification release lock requires manual cleanup\n' >&2
  fi
  rm -rf -- "$workdir" "$verify_dir"
}
trap cleanup EXIT

scripts/qualification-release-lock.sh acquire "$tag" "$lock_token"
lock_acquired=true
[ "$(gh release view "$tag" --repo "$repository" --json isDraft --jq .isDraft)" = "true" ] \
  || fail "${tag} stopped being a draft while acquiring the qualification lock"

echo "==> downloading inert qualification assets for ${tag}"
gh release download "$tag" --repo "$repository" \
  -p "openasr-${version}-qualification-*.json" \
  -p "openasr-${version}-build-provenance.bundle.json" \
  -p "openasr-${version}-windows-x86_64-neutral.zip" \
  -D "$workdir" --clobber

python3 - "$workdir" "$version" "$tag_commit" "$promote_cuda_targets" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
version = sys.argv[2]
tag_commit = sys.argv[3]
promote_cuda_targets = sys.argv[4]
sys.path.insert(0, "tooling/release-manifest")
import qualification_manifest as compiler

cells = set()
sources = set()
bundles = set()
referenced = set()
rows = []
for path in sorted(root.glob(f"openasr-{version}-qualification-*.json")):
    if path.name.endswith(".signature.json"):
        continue
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != compiler.SCHEMA_VERSION or value.get("release_subject") != f"v{version}":
        raise SystemExit(f"invalid qualification manifest identity: {path.name}")
    provider_target = value.get("provider_target")
    if not isinstance(provider_target, dict):
        raise SystemExit(f"missing provider_target: {path.name}")
    provider = provider_target.get("provider")
    target = provider_target.get("target")
    if provider not in {"cuda", "hip", "vulkan"} or not isinstance(target, str):
        raise SystemExit(f"invalid exact cell: {path.name}")
    expected = compiler.manifest_asset_name(version, provider, target)
    if path.name != expected or (provider, target) in cells:
        raise SystemExit(f"qualification asset name/cell mismatch: {path.name}")
    cells.add((provider, target))
    attestation = value.get("attestation")
    artifacts = value.get("artifacts")
    if not isinstance(attestation, dict) or not isinstance(artifacts, dict):
        raise SystemExit(f"incomplete qualification manifest: {path.name}")
    source_digest = attestation.get("source_digest")
    if source_digest != tag_commit:
        raise SystemExit(
            f"qualification source {source_digest!r} differs from tag commit {tag_commit}: {path.name}"
        )
    sources.add(source_digest)
    if (
        attestation.get("predicate_type") != compiler.ATTESTATION_PREDICATE_TYPE
        or attestation.get("repository") != compiler.ATTESTATION_REPOSITORY
        or attestation.get("signer_workflow") != compiler.ATTESTATION_SIGNER_WORKFLOW
        or attestation.get("deny_self_hosted_runners") is not True
    ):
        raise SystemExit(f"qualification attestation authority drifted: {path.name}")
    bundle = attestation.get("bundle")
    if not isinstance(bundle, dict) or not isinstance(bundle.get("file_name"), str):
        raise SystemExit(f"missing attestation bundle: {path.name}")
    bundle_name = compiler._safe_basename(bundle["file_name"], "attestation bundle")
    expected_bundle = f"openasr-{version}-build-provenance.bundle.json"
    if bundle_name != expected_bundle:
        raise SystemExit(f"unexpected attestation bundle name: {bundle_name}")
    bundles.add(bundle_name)
    binary = artifacts.get("binary")
    if not isinstance(binary, dict):
        raise SystemExit(f"missing binary artifact: {path.name}")
    for artifact in [binary.get("bundle"), artifacts.get("plugin")]:
        if artifact is not None:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("file_name"), str):
                raise SystemExit(f"malformed release artifact: {path.name}")
            referenced.add(compiler._safe_basename(artifact["file_name"], "release artifact"))
    vendor = artifacts.get("vendor", [])
    if not isinstance(vendor, list):
        raise SystemExit(f"malformed vendor artifacts: {path.name}")
    for artifact in vendor:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("file_name"), str):
            raise SystemExit(f"malformed vendor artifact: {path.name}")
        referenced.add(compiler._safe_basename(artifact["file_name"], "vendor artifact"))
    signature = f"{path.stem}.signature.json"
    manifest_url = f"https://dl.openasr.org/core/v{version}/{path.name}"
    rows.append((path.name, manifest_url, signature))
matrix = json.loads(
    Path("tooling/release-manifest/release_binaries_matrix.json").read_text(encoding="utf-8")
)
promoted = {
    token.strip().removeprefix("sm_")
    for token in promote_cuda_targets.replace(",", " ").split()
    if token.strip()
}
known_cuda = {
    str(row["cuda_gpu_target"])
    for row in matrix
    if row.get("provider") == "cuda"
}
if unknown := promoted - known_cuda:
    raise SystemExit(f"unknown promoted CUDA target(s): {sorted(unknown)}")
expected_cells = compiler.expected_artifact_cells(matrix, promoted_cuda_targets=promoted)
if cells != expected_cells:
    raise SystemExit(
        f"qualification cells differ from the tag matrix: missing={sorted(expected_cells-cells)} extra={sorted(cells-expected_cells)}"
    )
if len(sources) != 1 or len(bundles) != 1:
    raise SystemExit("qualification manifests do not share one provenance bundle/source digest")
(root / "manifest-index.tsv").write_text(
    "".join("\t".join(row) + "\n" for row in rows), encoding="utf-8"
)
(root / "release-subjects.txt").write_text(
    "".join(name + "\n" for name in sorted(referenced)), encoding="utf-8"
)
(root / "source-digest.txt").write_text(next(iter(sources)) + "\n", encoding="utf-8")
(root / "bundle-name.txt").write_text(next(iter(bundles)) + "\n", encoding="utf-8")
(root / "expected-signature-count.txt").write_text(
    str(len(expected_cells)) + "\n", encoding="utf-8"
)
PY

while IFS= read -r asset_name || [ -n "$asset_name" ]; do
  [ -n "$asset_name" ] || continue
  [ -f "$workdir/$asset_name" ] || gh release download "$tag" --repo "$repository" \
    -p "$asset_name" -D "$workdir" --clobber
done < "$workdir/release-subjects.txt"

echo "==> verifying release bytes and Sigstore provenance before signing"
python3 - "$workdir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, "tooling/release-manifest")
import qualification_manifest as compiler

bundle_name = (root / "bundle-name.txt").read_text(encoding="utf-8").strip()
bundle_path = root / bundle_name
predicate, subjects = compiler._attestation_subjects(bundle_path)
if predicate != compiler.ATTESTATION_PREDICATE_TYPE:
    raise SystemExit("qualification bundle predicate changed")
for manifest_name, _url, _signature in (
    line.rstrip("\n").split("\t")
    for line in (root / "manifest-index.tsv").read_text(encoding="utf-8").splitlines()
):
    manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    values = [artifacts["binary"]["bundle"]]
    if "plugin" in artifacts:
        values.append(artifacts["plugin"])
    values.extend(artifacts.get("vendor", []))
    values.append(manifest["attestation"]["bundle"])
    for artifact in values:
        path = root / artifact["file_name"]
        digest, size = compiler._sha256_size(path)
        if digest != artifact["sha256"] or size != artifact["size_bytes"]:
            raise SystemExit(f"release bytes differ from {manifest_name}: {path.name}")
    for artifact in values[:-1]:
        compiler._require_attested(subjects, root / artifact["file_name"])
PY

bundle_name="$(tr -d '\r\n' < "$workdir/bundle-name.txt")"
source_digest="$(tr -d '\r\n' < "$workdir/source-digest.txt")"
[ "$source_digest" = "$tag_commit" ] \
  || fail "qualification provenance source ${source_digest} differs from ${tag} commit ${tag_commit}"
expected_signature_count="$(tr -d '\r\n' < "$workdir/expected-signature-count.txt")"
[[ "$expected_signature_count" =~ ^[1-9][0-9]*$ ]] \
  || fail "invalid exact-cell signature count: ${expected_signature_count}"
while IFS= read -r asset_name || [ -n "$asset_name" ]; do
  [ -n "$asset_name" ] || continue
  gh attestation verify "$workdir/$asset_name" \
    --repo "$repository" \
    --signer-workflow "${repository}/.github/workflows/release-binaries.yml" \
    --source-digest "$source_digest" \
    --predicate-type "https://slsa.dev/provenance/v1" \
    --deny-self-hosted-runners \
    --bundle "$workdir/$bundle_name" \
    --format json >/dev/null
done < "$workdir/release-subjects.txt"

echo "==> signing ${expected_signature_count} exact-cell manifests with the qualification domain"
cargo_events="$workdir/cargo-build.jsonl"
cargo build --quiet -p openasr-cli --bin openasr --message-format=json > "$cargo_events"
signer_bin="$(python3 - "$cargo_events" <<'PY'
import json
import sys
from pathlib import Path

executables = set()
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    event = json.loads(line)
    target = event.get("target", {})
    executable = event.get("executable")
    if (
        event.get("reason") == "compiler-artifact"
        and target.get("name") == "openasr"
        and "bin" in target.get("kind", [])
        and isinstance(executable, str)
    ):
        executables.add(executable)
if len(executables) != 1:
    raise SystemExit(f"could not resolve one freshly built signer binary: {sorted(executables)}")
print(executables.pop())
PY
)"
[ -x "$signer_bin" ] || fail "freshly built signer is not executable: $signer_bin"
signature_paths=()
while IFS=$'\t' read -r manifest_name manifest_url signature_name; do
  OPENASR_HOME="$workdir/home" \
    OPENASR_CATALOG_SIGNING_KEY_SEED_HEX="$signing_key_seed" \
    "$signer_bin" __openasr-sign-qualification-manifest \
    "$workdir/$manifest_name" \
    --out "$workdir/$signature_name" \
    --manifest-url "$manifest_url"
  [ -s "$workdir/$signature_name" ] || fail "signature was not written: $signature_name"
  signature_paths+=("$workdir/$signature_name")
done < "$workdir/manifest-index.tsv"
[ "${#signature_paths[@]}" -eq "$expected_signature_count" ] \
  || fail "did not produce all ${expected_signature_count} signatures"
unset signing_key_seed

echo "==> uploading detached signatures to draft ${tag}"
[ "$(gh release view "$tag" --repo "$repository" --json isDraft --jq .isDraft)" = "true" ] \
  || fail "${tag} stopped being a draft before signature upload"
gh release upload "$tag" --repo "$repository" "${signature_paths[@]}" --clobber

echo "==> re-downloading and verifying published manifest/signature pairs"
gh release download "$tag" --repo "$repository" \
  -p "openasr-${version}-qualification-*.json" \
  -D "$verify_dir" --clobber
while IFS=$'\t' read -r manifest_name manifest_url signature_name; do
  [ -s "$verify_dir/$manifest_name" ] || fail "published manifest is missing: $manifest_name"
  [ -s "$verify_dir/$signature_name" ] || fail "published signature is missing: $signature_name"
  OPENASR_HOME="$verify_dir/home" \
    "$signer_bin" __openasr-verify-qualification-manifest \
    "$verify_dir/$manifest_name" \
    --signature "$verify_dir/$signature_name" \
    --manifest-url "$manifest_url" >/dev/null
done < "$workdir/manifest-index.tsv"

scripts/qualification-release-lock.sh release "$tag" "$lock_token"
lock_acquired=false

echo
echo "QUALIFICATION-MANIFESTS-SIGNED-AND-VERIFIED for ${tag}"
echo "  exact cells: ${expected_signature_count}"
echo "  next: sync manifests, signatures, and ${bundle_name} to core/v${version}/"
