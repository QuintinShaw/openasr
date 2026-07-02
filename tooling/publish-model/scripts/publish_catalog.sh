#!/usr/bin/env bash
# Publish the public OpenASR model catalog projection to Cloudflare.
#
# This hosts only entries that have passed the public-listing gate
# (`public: true`) on catalog.openasr.org. Staged/private entries remain committed
# locally but are not exposed through the public catalog URL consumed by released
# clients. Model WEIGHTS are unaffected — they stay on Hugging Face.
#
#   publish_catalog.sh [--dry-run] [--summary-json path] [--summary-md path] [--strict-evidence]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

log() { printf '\033[1;36m[publish]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[publish:err]\033[0m %s\n' "$*" >&2; exit 1; }

# The Cloudflare host the public catalog is deployed to (transport identity only;
# the signed catalog_url stays HF-canonical).
CATALOG_TARGET="catalog.openasr.org"
DRY_RUN=0
SUMMARY_JSON=""
SUMMARY_MD=""
STRICT_EVIDENCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --summary-json)
      SUMMARY_JSON="${2:?--summary-json requires <path>}"
      shift 2
      ;;
    --summary-md)
      SUMMARY_MD="${2:?--summary-md requires <path>}"
      shift 2
      ;;
    --strict-evidence)
      STRICT_EVIDENCE=1
      shift
      ;;
    *)
      die "unknown flag: $1"
      ;;
  esac
done

if [[ "$STRICT_EVIDENCE" == "1" ]]; then
  [[ "$DRY_RUN" == "0" ]] || die "--strict-evidence cannot be used with --dry-run"
  [[ -n "$SUMMARY_JSON" ]] || die "--summary-json is required with --strict-evidence"
fi

preflight_summary_destinations() {
  local destination probe
  for destination in "$SUMMARY_JSON" "$SUMMARY_MD"; do
    [[ -n "$destination" ]] || continue
    [[ ! -d "$destination" ]] || die "summary output points to a directory: $destination"
    mkdir -p "$(dirname "$destination")"
    probe="$(dirname "$destination")/.openasr-summary-write-test.$$"
    : > "$probe" || die "summary output is not writable: $destination"
    rm -f "$probe"
  done
}

CATALOG_SRC="${OPENASR_CATALOG_SRC:-$REPO_ROOT/model-registry/catalog.json}"
CATALOG_EPOCH_SRC="${OPENASR_CATALOG_EPOCH_SRC:-$REPO_ROOT/model-registry/catalog.epoch}"
PUBLIC_DIR="${OPENASR_PUBLIC_DIR:-$REPO_ROOT/tmp/publish/catalog}"
PUBLIC_CATALOG="$PUBLIC_DIR/catalog.json"
PUBLIC_MANIFEST="$PUBLIC_DIR/catalog.signature.json"
SIGNING_KEY_SEED="${OPENASR_CATALOG_SIGNING_KEY_SEED_HEX:-}"

[[ "$SIGNING_KEY_SEED" =~ ^[0-9a-fA-F]{64}$ ]] || die "OPENASR_CATALOG_SIGNING_KEY_SEED_HEX must be set to a 64-hex Ed25519 seed before writing the public catalog projection"

mkdir -p "$PUBLIC_DIR"
if [[ "$STRICT_EVIDENCE" == "1" ]]; then
  mkdir -p "$(dirname "$SUMMARY_JSON")"
  PUBLIC_DIR_ABS="$(cd "$PUBLIC_DIR" && pwd -P)"
  SUMMARY_DIR_ABS="$(cd "$(dirname "$SUMMARY_JSON")" && pwd -P)"
  [[ "$SUMMARY_DIR_ABS" == "$PUBLIC_DIR_ABS" ]] || die "--strict-evidence requires --summary-json next to catalog.json and catalog.signature.json in $PUBLIC_DIR"
fi
preflight_summary_destinations

log "validating committed catalog"
(cd "$REPO_ROOT" && cargo test -p openasr-core bundled_catalog_json_parses_and_matches_registry_cards)

PUBLIC_COUNT="$(python3 - "$CATALOG_SRC" "$PUBLIC_CATALOG" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
catalog = json.loads(source.read_text())
public_models = [model for model in catalog.get("models", []) if model.get("public") is True]
if not public_models:
    raise SystemExit("catalog has no public:true models; refusing to publish an empty public catalog")

projection = {
    "schema_version": catalog["schema_version"],
    "generated_at": catalog["generated_at"],
    "catalog_url": catalog["catalog_url"],
    "models": public_models,
}
target.write_text(json.dumps(projection, indent=2, sort_keys=False) + "\n")
print(len(public_models))
PY
)"

if [[ -n "${OPENASR_CATALOG_EPOCH:-}" ]]; then
  CATALOG_EPOCH="$OPENASR_CATALOG_EPOCH"
else
  [[ -f "$CATALOG_EPOCH_SRC" ]] || die "catalog epoch file is missing: $CATALOG_EPOCH_SRC"
  CATALOG_EPOCH="$(tr -d '[:space:]' < "$CATALOG_EPOCH_SRC")"
fi
[[ "$CATALOG_EPOCH" =~ ^[0-9]+$ && "$CATALOG_EPOCH" != "0" ]] || die "catalog epoch must be a positive integer"

log "signing public catalog projection at epoch $CATALOG_EPOCH"
(cd "$REPO_ROOT" && cargo run --quiet -p openasr-cli -- __openasr-sign-catalog-manifest "$PUBLIC_CATALOG" --out "$PUBLIC_MANIFEST" --epoch "$CATALOG_EPOCH")

PUBLIC_CATALOG_SHA256="$(shasum -a 256 "$PUBLIC_CATALOG" | awk '{print $1}')"
PUBLIC_MANIFEST_SHA256="$(shasum -a 256 "$PUBLIC_MANIFEST" | awk '{print $1}')"

write_summary() {
  local signed="$1"
  [[ -n "$SUMMARY_JSON$SUMMARY_MD" ]] || return 0
  OPENASR_SUMMARY_JSON="$SUMMARY_JSON" \
  OPENASR_SUMMARY_MD="$SUMMARY_MD" \
  OPENASR_SUMMARY_TARGET="$CATALOG_TARGET" \
  OPENASR_SUMMARY_DRY_RUN="$DRY_RUN" \
  OPENASR_SUMMARY_STRICT="$STRICT_EVIDENCE" \
  OPENASR_SUMMARY_SIGNED="$signed" \
  OPENASR_SUMMARY_PUBLIC_COUNT="$PUBLIC_COUNT" \
  OPENASR_SUMMARY_PUBLIC_CATALOG="$PUBLIC_CATALOG" \
  OPENASR_SUMMARY_CATALOG_EPOCH="$CATALOG_EPOCH" \
  OPENASR_SUMMARY_CATALOG_FILE="$(basename "$PUBLIC_CATALOG")" \
  OPENASR_SUMMARY_MANIFEST_FILE="$(basename "$PUBLIC_MANIFEST")" \
  OPENASR_SUMMARY_CATALOG_SHA256="$PUBLIC_CATALOG_SHA256" \
  OPENASR_SUMMARY_MANIFEST_SHA256="$PUBLIC_MANIFEST_SHA256" \
  python3 - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path


def env_bool(name: str) -> bool:
    return os.environ.get(name, "") in {"1", "true", "True"}


def inline(value: object) -> str:
    return str(value).replace("`", "'").replace("\r", " ").replace("\n", " ")


summary = {
    "schema_version": 1,
    "probe": "catalog_publish",
    "target": os.environ["OPENASR_SUMMARY_TARGET"],
    "dry_run": env_bool("OPENASR_SUMMARY_DRY_RUN"),
    "strict_evidence": env_bool("OPENASR_SUMMARY_STRICT"),
    "signed": env_bool("OPENASR_SUMMARY_SIGNED"),
    "public_model_count": int(os.environ["OPENASR_SUMMARY_PUBLIC_COUNT"]),
    "catalog_epoch": int(os.environ["OPENASR_SUMMARY_CATALOG_EPOCH"]),
    "catalog_file": os.environ["OPENASR_SUMMARY_CATALOG_FILE"],
    "manifest_file": os.environ["OPENASR_SUMMARY_MANIFEST_FILE"],
    "catalog_sha256": os.environ["OPENASR_SUMMARY_CATALOG_SHA256"],
    "manifest_sha256": os.environ["OPENASR_SUMMARY_MANIFEST_SHA256"],
}

public_catalog = json.loads(
    Path(os.environ["OPENASR_SUMMARY_PUBLIC_CATALOG"]).read_text(encoding="utf-8")
)
public_model_ids = sorted(
    str(model["id"]) for model in public_catalog.get("models", []) if model.get("public")
)
if len(public_model_ids) != summary["public_model_count"]:
    raise SystemExit("public model id list does not match public model count")
summary["public_model_ids"] = public_model_ids

summary_json = os.environ.get("OPENASR_SUMMARY_JSON", "")
if summary_json:
    path = Path(summary_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote redacted catalog publish summary JSON: {path}")

summary_md = os.environ.get("OPENASR_SUMMARY_MD", "")
if summary_md:
    path = Path(summary_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "### YYYY-MM-DD — Signed public catalog publish evidence\n\n"
        "Scope:\n\n"
        "- Generated and signed the public OpenASR catalog projection.\n"
        "- Recorded only artifact filenames, public model ids/count, epoch, hashes, and deploy state.\n\n"
        "Validation:\n\n"
        f"- target: `{inline(summary['target'])}`\n"
        f"- dry run: `{inline(summary['dry_run'])}`\n"
        f"- strict evidence mode: `{inline(summary['strict_evidence'])}`\n"
        f"- signed: `{inline(summary['signed'])}`\n"
        f"- public model count: `{inline(summary['public_model_count'])}`\n"
        f"- public models: `{inline(', '.join(summary['public_model_ids']))}`\n"
        f"- catalog epoch: `{inline(summary['catalog_epoch'])}`\n"
        f"- catalog artifact: `{inline(summary['catalog_file'])}` sha256 `{inline(summary['catalog_sha256'])}`\n"
        f"- signature artifact: `{inline(summary['manifest_file'])}` sha256 `{inline(summary['manifest_sha256'])}`\n"
        "\n"
        "Notes:\n\n"
        "- Replace `YYYY-MM-DD` with the actual run date.\n"
        "- Do not paste `CLOUDFLARE_API_TOKEN`, `OPENASR_CATALOG_SIGNING_KEY_SEED_HEX`, or local absolute artifact paths.\n",
        encoding="utf-8",
    )
    print(f"Wrote redacted catalog publish summary Markdown: {path}")
PY
}

if [[ "$DRY_RUN" == "1" ]]; then
  write_summary "0"
  log "dry run: signed public catalog projection written to $PUBLIC_CATALOG and $PUBLIC_MANIFEST ($PUBLIC_COUNT public model(s)); committed artifacts not touched"
  exit 0
fi

# Sign + refresh the committed catalog artifacts. Signing stays LOCAL (the seed
# never leaves this machine); deployment is handled by the deploy-catalog CI
# workflow when the committed catalog.public.* is pushed (CLOUDFLARE_API_TOKEN).
#
# Sign the FULL catalog too so the bundled-signature gate stays green after a
# regenerate; the public projection is what the binary embeds and Cloudflare serves.
log "signing the full committed catalog at epoch $CATALOG_EPOCH"
(cd "$REPO_ROOT" && cargo run --quiet -p openasr-cli -- __openasr-sign-catalog-manifest "$CATALOG_SRC" --out "$REPO_ROOT/model-registry/catalog.signature.json" --epoch "$CATALOG_EPOCH")

COMMITTED_PUBLIC_CATALOG="$REPO_ROOT/model-registry/catalog.public.json"
COMMITTED_PUBLIC_MANIFEST="$REPO_ROOT/model-registry/catalog.public.signature.json"
log "refreshing committed public projection: $COMMITTED_PUBLIC_CATALOG (+ signature)"
cp "$PUBLIC_CATALOG" "$COMMITTED_PUBLIC_CATALOG"
cp "$PUBLIC_MANIFEST" "$COMMITTED_PUBLIC_MANIFEST"

write_summary "1"

log "signed full + public catalog at epoch $CATALOG_EPOCH ($PUBLIC_COUNT public model(s))"
log "next: commit model-registry/{catalog.json,catalog.signature.json,catalog.public.json,catalog.public.signature.json} and push"
log "-> the deploy-catalog workflow then deploys the public catalog to https://$CATALOG_TARGET"
