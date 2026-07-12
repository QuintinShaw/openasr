#!/usr/bin/env bash
# Sign a LOCAL model catalog for dev/preview, using the public, non-secret
# local-dev catalog signing key -- so a contributor iterating on
# model-registry/catalog.json (e.g. after regenerate_all.sh) can preview
# staged/unpublished entries through `openasr` without the real production
# signing seed.
#
# A local ("file://" or bare filesystem path) catalog_url now requires a
# signed catalog.signature.json sidecar exactly like the production HTTPS
# catalog does (see docs/MODEL_CATALOG_ARCHITECTURE.md). This script fills
# that requirement for local dev catalogs.
#
#   sign_local_catalog.sh [<catalog.json>] [--out <path>] [--epoch <n>] [--catalog-url <url>]
#
# Defaults:
#   <catalog.json>   model-registry/catalog.json (repo-relative)
#   --out            the catalog's own directory, i.e. <catalog.json's dir>/catalog.signature.json
#   --catalog-url    `file://<absolute path to <catalog.json>>`. The dev key
#                     is ONLY accepted for a non-production (local) identity
#                     (see catalog_security::classify_catalog_identity /
#                     docs/MODEL_CATALOG_ARCHITECTURE.md): the CLI's
#                     repo-checkout auto-discovery of model-registry/catalog.json
#                     verifies against the canonical production
#                     `https://.../catalog.json` identity and requires the
#                     real production signature, so a dev-signed manifest
#                     bound to that identity would be rejected everywhere.
#                     Load the dev-signed output of this script via an
#                     explicit `OPENASR_CATALOG_URL=file://<path>` (or
#                     `--catalog-url file://<path>`) override instead of
#                     auto-discovery. Pass a different `--catalog-url` only if
#                     you are intentionally signing for some other explicit
#                     local override path.
#   --epoch          the epoch already recorded in model-registry/catalog.epoch,
#                     or 1 if that file does not exist yet. A dev-key-signed
#                     manifest never advances (or is blocked by) the shared
#                     $OPENASR_HOME/catalog.epoch anti-rollback floor, so any
#                     positive value is safe to reuse across runs.
#
# Environment:
#   OPENASR_LOCAL_CATALOG_SIGNING_KEY_SEED_HEX overrides the (public,
#   non-secret) local-dev signing seed; almost never needed.
#
# WARNING: model-registry/catalog.signature.json is git-tracked and normally
# holds the REAL PRODUCTION signature (see publish_catalog.sh). Running this
# script overwrites it locally with a dev-signed manifest so you can preview
# catalog edits via `OPENASR_CATALOG_URL=file://<path>` -- never commit that
# dev-signed file. Restore the real one with
# `git checkout -- model-registry/catalog.signature.json` (or by rerunning
# publish_catalog.sh with the real signing seed) before committing anything else.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

log() { printf '\033[1;36m[sign-local-catalog]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[sign-local-catalog:err]\033[0m %s\n' "$*" >&2; exit 1; }

# The public, deterministic local-dev catalog signing key seed --
# sha256("openasr.catalog_manifest.v1.local-dev-signing-key-seed"). NOT a
# secret (see the doc comment on `CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID` in
# crates/openasr-core/src/catalog_security.rs, the single source of truth).
#
# Deliberately read from ITS OWN env var (OPENASR_LOCAL_CATALOG_SIGNING_KEY_SEED_HEX),
# never falling through to OPENASR_CATALOG_SIGNING_KEY_SEED_HEX: a maintainer's
# shell may already export the latter (the REAL production seed) for
# publish_catalog.sh, and silently reusing it here would render a "local dev"
# manifest with production key material under the dev key id -- which then
# fails self-verification (safe, but confusing) instead of doing what this
# script is for. Set OPENASR_LOCAL_CATALOG_SIGNING_KEY_SEED_HEX yourself only
# if you have set up a different local trust root.
LOCAL_DEV_SEED="${OPENASR_LOCAL_CATALOG_SIGNING_KEY_SEED_HEX:-7181d685f3c226e1c111574368512b603d67964c057165ad004683b84998960e}"
LOCAL_DEV_KEY_ID="openasr-catalog-local-dev-v1"

CATALOG=""
OUT=""
EPOCH=""
CATALOG_URL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT="${2:?--out requires <path>}"
      shift 2
      ;;
    --epoch)
      EPOCH="${2:?--epoch requires <n>}"
      shift 2
      ;;
    --catalog-url)
      CATALOG_URL="${2:?--catalog-url requires <url>}"
      shift 2
      ;;
    -h|--help)
      sed -n '2,33p' "$0"
      exit 0
      ;;
    -*)
      die "unknown flag: $1"
      ;;
    *)
      [[ -z "$CATALOG" ]] || die "unexpected extra argument: $1"
      CATALOG="$1"
      shift
      ;;
  esac
done

CATALOG="${CATALOG:-$REPO_ROOT/model-registry/catalog.json}"
[[ -f "$CATALOG" ]] || die "catalog file not found: $CATALOG"
OUT="${OUT:-$(dirname "$CATALOG")/catalog.signature.json}"

if [[ -z "$EPOCH" ]]; then
  EPOCH_FILE="$(dirname "$CATALOG")/catalog.epoch"
  if [[ -f "$EPOCH_FILE" ]]; then
    EPOCH="$(tr -d '[:space:]' < "$EPOCH_FILE")"
  else
    EPOCH=1
  fi
fi
[[ "$EPOCH" =~ ^[0-9]+$ && "$EPOCH" != "0" ]] || die "epoch must be a positive integer, got: $EPOCH"

if [[ -z "$CATALOG_URL" ]]; then
  # The dev key only verifies against a non-production (local) identity (see
  # the --catalog-url doc comment above), so default to the literal
  # `file://<absolute path>` of the catalog being signed -- NOT the catalog
  # JSON's own `catalog_url` field, which is the production https identity
  # and would produce a manifest that verifies nowhere.
  CATALOG_DIR="$(cd "$(dirname "$CATALOG")" && pwd)"
  CATALOG_URL="file://$CATALOG_DIR/$(basename "$CATALOG")"
fi
CATALOG_URL_ARGS=(--catalog-url "$CATALOG_URL")

log "signing '$CATALOG' with the public local-dev key ($LOCAL_DEV_KEY_ID) at epoch $EPOCH"
log "catalog_url identity: $CATALOG_URL"
# The explicit assignment overrides whatever OPENASR_CATALOG_SIGNING_KEY_SEED_HEX
# may already be set to in this shell.
OPENASR_CATALOG_SIGNING_KEY_SEED_HEX="$LOCAL_DEV_SEED" \
  cargo run --quiet -p openasr-cli -- __openasr-sign-catalog-manifest "$CATALOG" \
    --out "$OUT" --epoch "$EPOCH" --key-id "$LOCAL_DEV_KEY_ID" \
    "${CATALOG_URL_ARGS[@]}"

log "wrote dev-signed manifest: $OUT"
log "reminder: this is a LOCAL preview signature -- never commit it; restore the"
log "committed production manifest with 'git checkout -- $OUT' before committing anything else"
log "load it with: OPENASR_CATALOG_URL='$CATALOG_URL' cargo run -p openasr-cli -- <command>"
