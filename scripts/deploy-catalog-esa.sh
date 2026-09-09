#!/usr/bin/env bash
# Push the signed catalog pair to Aliyun ESA (catalog.bug.im).
#
# Same topology as bugim-cap / bugim-edge: `esa-cli deploy` of a bundled
# EdgeRoutine. Bytes are baked into the bundle so ESA never origin-fetches
# Cloudflare (domestic networks that wall CF would otherwise make the replica
# useless).
#
# Skip (exit 0) when esa-cli is missing or not logged in — overseas catalog
# deploy must stay green. After a successful ESA deploy, reachable
# catalog.bug.im bytes must match the candidate files (exit 1 on mismatch).
# Skip only when live HTTPS is still unreachable (DNS/TLS not ready).
#
# Usage:
#   scripts/deploy-catalog-esa.sh [catalog.json] [catalog.signature.json]
#
# Defaults: model-registry/catalog.public.json + .public.signature.json
# Auth: `esa-cli login` (AK/SK), same as bugim. Optional env:
#   ESA_CLI                 override esa-cli binary
#   OPENASR_ESA_CATALOG_SKIP=1  force skip

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CATALOG_PKG="${REPO_ROOT}/cloudflare/catalog"

catalog="${1:-${REPO_ROOT}/model-registry/catalog.public.json}"
signature="${2:-${REPO_ROOT}/model-registry/catalog.public.signature.json}"

skip() {
  printf 'ESA catalog deploy skipped: %s\n' "$1"
  exit 0
}

if [ "${OPENASR_ESA_CATALOG_SKIP:-}" = "1" ]; then
  skip "OPENASR_ESA_CATALOG_SKIP=1"
fi

command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 \
  || skip "no sha256sum/shasum"
[ -f "$catalog" ] && [ -f "$signature" ] || skip "catalog files missing"

ESA_CLI="${ESA_CLI:-esa-cli}"
if ! command -v "$ESA_CLI" >/dev/null 2>&1; then
  skip "esa-cli not installed (npm i -g esa-cli && esa-cli login, same as bugim-cap)"
fi

# esa-cli has no documented --dry-run auth probe; `esa-cli --help` succeeding
# after login is not enough. Try a non-mutating call and skip on auth failure.
if ! "$ESA_CLI" --help >/dev/null 2>&1; then
  skip "esa-cli not runnable"
fi

# Local: `esa-cli login`. CI: ALIBABA_CLOUD_ACCESS_KEY_ID/SECRET.
# Probe before deploy so a missing login cannot fail overseas catalog CI.
if ! "$ESA_CLI" site list >/dev/null 2>&1; then
  skip "esa-cli not authenticated (esa-cli login, or ALIBABA_CLOUD_ACCESS_KEY_ID/SECRET)"
fi

file_sha() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

want_catalog_sha="$(file_sha "$catalog")"
want_sig_sha="$(file_sha "$signature")"

mkdir -p "${CATALOG_PKG}/public"
cp "$catalog" "${CATALOG_PKG}/public/catalog.json"
cp "$signature" "${CATALOG_PKG}/public/catalog.signature.json"

(
  cd "$CATALOG_PKG"
  if [ ! -d node_modules/esbuild ]; then
    npm install --omit=optional
  fi
  npm run build:esa
)

echo "==> esa-cli deploy (catalog.bug.im)"
(
  cd "$CATALOG_PKG"
  # Already bundled by build:esa (catalog JSON baked). Skip TTY env/version prompts.
  if ! "$ESA_CLI" deploy --no-bundle -e production -d "OpenASR China catalog replica"; then
    echo "ESA catalog deploy failed; overseas Cloudflare catalog is unchanged." >&2
    exit 1
  fi
  "$ESA_CLI" domain add catalog.bug.im 2>&1 || true
  "$ESA_CLI" route add -r "catalog.bug.im/*" -s bug.im -a catalog-bug-im 2>&1 || true
)

echo "==> verifying live catalog.bug.im matches candidate sha256"
base="https://catalog.bug.im/v1"
ok=0
reachable=0
for attempt in $(seq 1 12); do
  bust="cb=$(date +%s)-${attempt}"
  if curl -fsSL "${base}/catalog.json?${bust}" -o /tmp/esa-catalog.json \
    && curl -fsSL "${base}/catalog.signature.json?${bust}" -o /tmp/esa-catalog.sig.json; then
    reachable=1
    live_catalog_sha="$(file_sha /tmp/esa-catalog.json)"
    live_sig_sha="$(file_sha /tmp/esa-catalog.sig.json)"
    if [ "$live_catalog_sha" = "$want_catalog_sha" ] && [ "$live_sig_sha" = "$want_sig_sha" ]; then
      ok=1
      break
    fi
    echo "catalog.bug.im sha256 mismatch (attempt ${attempt}); retrying in 10s"
  else
    echo "catalog.bug.im HTTPS not reachable (attempt ${attempt}); retrying in 10s"
  fi
  sleep 10
done

if [ "$ok" = "1" ]; then
  echo "catalog.bug.im matches the candidate catalog pair"
  exit 0
fi

if [ "$reachable" = "1" ]; then
  echo "ESA catalog live sha256 does not match the candidate after 12 attempts." >&2
  echo "  candidate catalog ${want_catalog_sha}" >&2
  echo "  candidate signature ${want_sig_sha}" >&2
  echo "  live catalog ${live_catalog_sha:-unavailable}" >&2
  echo "  live signature ${live_sig_sha:-unavailable}" >&2
  exit 1
fi

echo "ESA catalog EdgeRoutine is deployed and catalog.bug.im is bound." >&2
echo "Live HTTPS is not ready yet (DNS/TLS). Candidate sha256:" >&2
echo "  catalog ${want_catalog_sha}" >&2
echo "  signature ${want_sig_sha}" >&2
echo "Add a DNS-only CNAME: catalog.bug.im -> catalog.bug.im.a1.initac.com (same pattern as cap.bug.im)." >&2
exit 0
