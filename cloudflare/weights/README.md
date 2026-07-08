# OpenASR weights resolve proxy (Cloudflare Worker)

**Proxies** the Hugging Face `resolve` endpoint for the `OpenASR/*` org, so a
client whose direct route to `huggingface.co` is blocked can still resolve and
download model weights. It is the server side of `weights.openasr.org`, an
alternative to pointing straight at `huggingface.co` (some networks block the
`huggingface.co` resolve hop specifically, while the CDN blob host it redirects
to is directly reachable).

## Why this is safe (and why it is not a trust anchor)

- **It only forwards a redirect.** For a normal `.oasr` pack, Hugging Face's
  `resolve` response is a `3xx` pointing at a Xet/CDN blob host (e.g.
  `cas-bridge.xethub.hf.co`). This Worker returns that `Location` header
  **verbatim** and never fetches the blob itself -- the (large) weight bytes
  never transit Cloudflare, only the small resolve request/redirect does.
- **It verifies nothing, and clients don't trust it to.** The client
  independently checks the downloaded bytes' sha256 against the signed catalog
  (see `cloudflare/catalog`) regardless of which host produced the redirect.
  If this Worker (or DNS to it) were ever compromised, the worst case is a
  wrong/failed download, not a supply-chain substitution -- the sha256 check
  still fails closed.
- **Its own integrity story is "open source + publicly deployed by CI from a
  known commit."** `GET /_version` reports the `git` commit the live Worker
  was built from, so anyone can diff that commit against this repository.

## Strict scope (deliberate -- do not relax without re-reviewing abuse risk)

This is **not** a general Hugging Face proxy. It rejects everything except:

| Method | Path | Action |
| --- | --- | --- |
| `GET`/`HEAD` | `/OpenASR/<repo>/resolve/<rev>/<file...>` | forward to `huggingface.co`, pass the response through |
| `GET`/`HEAD` | `/` or `/_version` | build-commit / health JSON |
| anything else | any | `404`/`405` |

- Only the `OpenASR` org, only the `resolve` action (no `blob`, `tree`, `api`,
  other orgs, etc.) -- checked with a strict path regex before any outbound
  fetch, so unrelated/abusive traffic never reaches Hugging Face through this
  Worker's quota.
- `..`/empty path segments are rejected (no traversal).
- Upstream errors (`4xx`/`5xx`/timeout/network failure) **fail closed**: the
  client gets an explicit error status, never a hang or a silently wrong
  response. The upstream fetch has a bounded timeout.
- **No caching**: every response sets `Cache-Control: no-store`.
- **No logging of user activity**: this code does not log request paths, User-
  Agent, or client IP (no-telemetry product promise). Only Cloudflare's
  standard platform observability applies, same as `cloudflare/catalog`.

## Deploy

`openasr.org` must already be a zone on the deploying Cloudflare account (the
Custom Domain provisions TLS for `weights.openasr.org` automatically).

```sh
npm install
npm test              # gating unit tests (vitest-pool-workers; no real network)
npx wrangler login
npx wrangler deploy --var BUILD_COMMIT:"$(git rev-parse HEAD)"
```

In CI, `.github/workflows/deploy-weights.yml` runs this on every push to `main`
that touches `cloudflare/weights/**`, injecting `github.sha` as `BUILD_COMMIT`
so `GET https://weights.openasr.org/_version` always names the exact deployed
commit. Secret configuration (`CLOUDFLARE_API_TOKEN`) and the first deploy are
done by a maintainer with account access -- see that workflow's comments.

## Smoke-testing after deploy

```sh
curl -sSD - -o /dev/null \
  "https://weights.openasr.org/OpenASR/<repo>/resolve/<rev>/<file>.oasr"
# Expect: HTTP/2 302 with a Location header pointing at a Hugging Face CDN/Xet
# blob host, not a 200 body.

curl -s "https://weights.openasr.org/_version"
# Expect: {"service":"openasr-weights","build_commit":"<the commit you deployed>"}
```
