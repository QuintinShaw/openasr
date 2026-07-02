# OpenASR catalog host (Cloudflare Worker)

**Hosts** the signed model catalog — `catalog.json` and `catalog.signature.json` —
on OpenASR's own domain, served from **Cloudflare Static Assets**. The catalog is
**not fetched from Hugging Face**; Hugging Face hosts model **weights** only.

It is the server side of the default catalog URL:
`https://catalog.openasr.org/v1/catalog.json`.

## Why host on Cloudflare (and why it's safe)

- **HF hosts weights, not the catalog.** The catalog index lives on
  `catalog.openasr.org`, fully decoupled from Hugging Face — the HF catalog repo is
  no longer required to serve clients.
- **It is not a trust anchor.** Bytes are served verbatim; the client verifies the
  ed25519 signature, the sha256, and the monotonic epoch. The signed `catalog_url`
  stays HF-canonical purely as a verification identity, so hosting on CF needs **no
  re-sign / signing seed** beyond the normal catalog publish step.
- **Offline still works.** If `catalog.openasr.org` is unreachable, the client falls
  back to its on-disk cache and finally the catalog snapshot embedded in the binary.

## What it serves

| Method | Path | Action |
| --- | --- | --- |
| `GET` | `/v1/catalog.json` | serve hosted asset verbatim |
| `GET` | `/v1/catalog.signature.json` | serve hosted asset verbatim |
| `OPTIONS` | any | CORS preflight (`204`) |
| anything else | any | `403` / `405` |

The current published snapshot is returned and verified client-side.

## Assets

`catalog.json` + `catalog.signature.json` are the single source of truth in
`model-registry/`. The `build:assets` script copies them into `public/` (gitignored)
at deploy time; Cloudflare serves them byte-for-byte. **Re-deploy on every catalog
publish** so the hosted snapshot tracks `model-registry/` (fold `npm run deploy`
into the publish recipe).

## Deploy

`openasr.org` must already be a zone on your Cloudflare account (the Custom Domain
provisions TLS for `catalog.openasr.org` automatically).

```sh
npm install
npm test            # gating unit tests (vitest-pool-workers; no network)
npx wrangler login
npm run deploy       # build:assets (copy from model-registry) -> wrangler deploy
```

Clients default to this host; to point elsewhere (self-host, local testing, or
straight to Hugging Face) set `OPENASR_CATALOG_ENDPOINT`.

## Caching

The Worker sets a short response TTL (`max-age=300`). Because the client re-verifies
signature + epoch on every load, a brief post-publish skew is self-healing (it may
surface as a transient epoch-rollback rejection that falls back to cache/embedded,
not a silent stale read). Re-deploying on publish replaces the hosted bytes
immediately; purge the edge cache if you also rely on `cf` edge caching.
