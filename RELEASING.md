# Releasing

OpenASR uses a single workspace version. A version bump and annotated
`vX.Y.Z` tag pin the commit; they do **not** publish. Publishing is a
maintainer-triggered `workflow_dispatch` of `.github/workflows/release-core.yml`,
the same posture as the desktop product's `release-desktop.yml`.

Feature, fix, and other content changes go through pull requests as usual.
The release bump may be pushed directly to `main` as a single
`chore(release)` commit plus its annotated tag; ordinary CI still runs, but
the release workflows do not. Routine CI is PR-only to avoid rebuilding every
merge; its narrow `main` push gate runs only for this direct `chore(release)`
commit.

## Versioning

The version lives in exactly one place: `[workspace.package] version` in the
root `Cargo.toml`. Every member crate inherits it via `version.workspace =
true`, and the `openasr-core` / `openasr-server` / `openasr-system-audio`
entries under `[workspace.dependencies]` are plain path dependencies with no
version pin to keep in sync.

Two lockfiles pin the workspace crates and must be regenerated alongside the
bump, or CI's `--locked` builds fail:

- the root `Cargo.lock`
- `tooling/system-audio-check/Cargo.lock` (standalone CI-gate workspace,
  depends on `openasr-system-audio` by path)

## Cutting a release

1. On `main`, run:

   ```bash
   scripts/bump-version.sh X.Y.Z --notes "Release highlights go here."
   ```

   `--notes` is **required** (the script fails closed without it, or with a
   blank/whitespace-only value): it becomes the message of an *annotated*
   `vX.Y.Z` git tag, which `release-core.yml` reads verbatim as the
   release's **Highlights** section. Write it like the top of a changelog
   entry -- one or a few lines of plain markdown, no need to restate the
   version number.

   The script bumps the version, regenerates both lockfiles, self-checks the
   result with `cargo metadata --locked`, commits `chore(release): vX.Y.Z`,
   and creates the annotated `vX.Y.Z` tag on that commit. It is idempotent:
   rerunning with the same version and no pending file changes skips the
   commit, and if the tag already exists locally it is left alone (delete it
   first with `git tag -d vX.Y.Z` to redo the notes).

2. Push the commit **and** the tag together (this is only the pin; it must
   not start a release):

   ```bash
   git push --follow-tags
   ```

   Pushing just the commit without the tag (plain `git push`) leaves nothing
   for `release-core.yml` to check out. The dispatch fails closed until the
   annotated tag exists on origin.

3. Confirm `main` is green at that bump commit, then dispatch `Release core`
   (`release-core.yml`) from **protected `main`**. The `version` input must
   equal `Cargo.toml` -- it confirms the committed pin and cannot override
   it. For a new draft the annotated tag must peel to this same commit.
   Agents do not dispatch this workflow.

   ```bash
   gh workflow run release-core.yml -f version=X.Y.Z
   ```

   - If no GitHub Release exists yet, the workflow creates the **draft**, then
     calls `release-binaries.yml` with `formal_release: true` (Linux
     x86_64/arm64, macOS x86_64/arm64, Windows, plus Vulkan/CUDA/HIP
     variants), then waits for `core-release` Environment approval and syncs
     the Windows GPU backend packs to B2. There is no bootstrap macOS/Linux
     rebuild and no `push: tags` matrix racing the orchestrator. Completeness
     fails the run if a required archive is missing.
   - If the matching release is still a draft, a second dispatch runs the
     pre-publication family gate, deploys the committed PublishedInert
     catalog, and waits for `core-release` Environment approval to publish
     the draft (see below).
   - A published (non-draft) release is a no-op.

### Release notes structure

Every release body has three sections:

- **Highlights** -- the `--notes` text from the annotated tag, verbatim.
- **What's Changed** -- GitHub's auto-generated PR list between this tag and
  the previous one, plus a "Full Changelog" compare link.
- **Install & Verify** -- one bullet per shipped platform archive (label +
  direct download link) plus a `sha256sum -c` snippet, generated from the
  release's actual asset list. Never hand-written, so it can't drift the way
  a fixed "macOS arm64 and Linux x86_64" sentence would once more platforms
  ship.

No pre-release channels: the core releases plain `X.Y.Z` versions only.

## Manual runs

`Release core` is `workflow_dispatch` only. The first dispatch (no GitHub
Release yet) creates the draft, builds the matrix, and syncs backend CDN
bytes after `core-release` Environment approval. After the generated
PublishedInert catalog has been reviewed, signed, committed, and pushed,
dispatch it again with the same `version` to run the pre-publication family
gate, catalog deployment, and `publish-release` (same Environment approval)
for that draft. Failed build or aggregation jobs are recovered only with
GitHub's **Re-run failed jobs** on the original release run, which preserves
its exact tag/source/artifact lineage.

`workflow_dispatch` on `Release binaries` (`.github/workflows/release-binaries.yml`)
is diagnostic-only: `only_target` is mandatory and the run may publish only an
Actions artifact for that one matrix row. It cannot aggregate or attest a
release subject set and cannot upload, replace, or delete either draft or
public GitHub Release assets. Full release assembly is available only through
the `formal_release` capability declared on `workflow_call` and supplied by
`Release core`.

The core GitHub Release is created as a **draft**. This is load-bearing for the
Windows plugin topology: provider payload hashes exist only after the release
matrix has built them, while the neutral host resolves those hashes from the
production-signed catalog. Core 0.1.34 and later publish no legacy whole-engine
Windows sidecars and no per-release `backends-manifest.json`.

Artifact publication and runtime capability activation are separate. A signed
provider pack may be public as `PublishedInert`, but ordinary Auto and explicit
selection reject it. Real-hardware evidence can therefore test the exact public
bytes without exposing an unqualified provider to users.

### Publish the inert release bytes

The only remaining local maintainer step is catalog signing. Everything else
runs in Actions after Environment approval.

1. Dispatch `Release core`. Stage 1 creates the draft and lets its single
   formal `release-binaries.yml` call build and checksum the complete release
   matrix, sign the applicable Windows binaries, and attest the full subject
   set. Do not use the diagnostic one-target dispatch to assemble or mutate a
   release.
2. Approve the `sync-backend-cdn` job in the Actions UI. It runs under the
   `core-release` GitHub Environment (required reviewers) and executes
   `scripts/sync-windows-backend-cdn.sh vX.Y.Z --allow-ci`. The job verifies
   and copies every file declared by the 1 Vulkan, 6 CUDA, and 14 HIP release
   entries to `https://dl.openasr.org/core/vX.Y.Z/`. Uploads are idempotent
   and never overwrite a different object at the same key. GitHub is the
   provenance mirror, not a runtime download fallback.
3. Run `scripts/prepare-windows-backend-catalog-release.sh vX.Y.Z` locally
   with the production catalog signing seed. It verifies all 21 exact entries
   and their already-live CDN bytes, merges every entry as `PublishedInert`,
   bumps the epoch, and signs the full/public catalogs. It consumes no
   hardware or token-correctness receipt. Review, commit, and push the five
   catalog/epoch files. This is the only local release step that still needs
   the signing seed.
4. Dispatch `Release core` again. Stage 2 re-verifies the draft's complete
   attested subject set (the same completeness gate as the original matrix, so
   a later CI-only fix can still see the draft), then runs the reusable
   pre-publication family contract against the immutable draft CLI and the
   real-pack CPU family gate. The orchestrator then calls the sole deployment
   entrypoint, `.github/workflows/deploy-catalog.yml`, with
   `activation_transition: published-inert`. The deploy job checks the signed
   catalog, CDN payloads, and released-binary compatibility before writing it.
   `finalize-notes` then rewrites Install & Verify from the real asset list.
5. Approve the `publish-release` job in the Actions UI (same `core-release`
   Environment). It runs `scripts/finalize-core-release.sh` with
   `OPENASR_DEPLOY_CATALOG_RUN_ID` set to this orchestrator run. The
   reusable deploy workflow shares that run id, so the finalizer looks up
   the "Deploy PublishedInert candidate catalog" job on this run, binds the
   exact tag, source commit, catalog SHA, signature SHA, release subjects,
   and live CDN bytes (retrying the no-credential catalog/CDN plane for up
   to 10 minutes), then removes the GitHub draft flag.

Any failure leaves the GitHub release as a draft.

### GitHub Environment `core-release`

Create this Environment on `QuintinShaw/openasr` before the first
Actions-driven publish. Required reviewers must be enabled so the B2 write
key and the undraft step are not available until a human approves the job
in the Actions UI.

Environment secrets (not repository secrets):

- `B2_S3_ENDPOINT`
- `B2_APPLICATION_KEY_ID`
- `B2_APPLICATION_KEY`

Key policy: a dedicated B2 application key restricted to the release bucket
and the `core/` prefix with `listFiles` / `readFiles` / `writeFiles` only
(no `deleteFiles`), separate from the desktop installer key.

Publishing the release triggers two independent GitHub Actions workflows:

- `publish-core-channels.yml` moves Docker/Homebrew only after the canonical
  catalog/CDN plane is complete.
- `sync-release-to-cnb.yml` mirrors the published GitHub assets to
  [cnb.cool/openasr/openasr](https://cnb.cool/openasr/openasr). GitHub stays
  the signed source; CI downloads each public asset and uploads it. Local
  `finalize-core-release.sh` must not pull the matrix onto a maintainer
  laptop for this. Missing `CNB_TOKEN` skips; a token present with a failed
  upload fails that workflow only and does not roll back GitHub.

### Qualify and activate one exact backend after publication

1. Dispatch `.github/workflows/qualify-windows-backend.yml` on the exact release
   tag (`--ref vX.Y.Z`) for one `(provider, device_target, backend_id, model_id,
   quant)` cell. The workflow accepts only an already-public stable release and
   re-hashes every executed release subject against `SHA256SUMS` and build
   provenance.
2. The qualification workflow emits separately attested placement/resource
   evidence (`backend-hardware-evidence-*.json` plus its raw audit) and
   token/transcript evidence (the projected matrix, receipts, and traces). It
   proves the exact provider, target, backend id, artifact tree, pack, fixture,
   driver, FullDevice placement, cold/reuse behavior, and fresh-process nonces.
   It never edits a release or catalog and never grants runtime authority.
3. Once the run ids cover every required matrix cell for that exact backend,
   run `scripts/activate-windows-backend-catalog-release.sh vX.Y.Z BACKEND_ID
   RUN_ID...` locally with the production signing seed. The script independently
   authenticates and replays `PublishedInert -> Qualified -> Activated`, verifies
   the intermediate Qualified projection, and signs one new reviewed catalog
   epoch. A separate public Qualified epoch is not required.
4. Review, commit, and push the catalog/epoch files, then dispatch
   `.github/workflows/activate-backend-catalog.yml` with those run ids and the
   exact authorization text `activate:<backend_id>`. The reusable deploy workflow
   replays both gates before deployment; qualification success alone cannot
   activate anything.
5. To fail safe, prepare a one-way exact-backend revocation with
   `scripts/revoke-windows-backend-catalog-release.sh`, then dispatch
   `.github/workflows/revoke-backend-catalog.yml` with
   `revoke:<backend_id>`. Revocation preserves old bindings for audit and cannot
   activate another backend as a side effect.

Hardware qualification is exact-target and exact-backend scoped: an `sm_89`
receipt cannot approve `sm_75`, a HIP/Vulkan/CPU result cannot approve CUDA, and
placement evidence cannot replace token correctness. A failed post-publication
qualification leaves the release public but the provider inert and unselectable.
See `tooling/release-manifest/README.md` for the two gate authorities and local
validation commands.

## Legacy backends manifests

`backends-manifest.json`, its signature, and the whole-engine Windows GPU
sidecars are historical compatibility tooling for core 0.1.33 and earlier.
Do not generate, sign, attach, or CDN-sync them for core 0.1.34 or later.
Current releases distribute one neutral Windows host plus target-scoped
CUDA/HIP backend packs; the production-signed model/backend catalog is their
only activation trust plane.

## Homebrew tap

`publish-core-channels.yml` bumps
`Formula/openasr.rb` in [`QuintinShaw/homebrew-tap`](https://github.com/QuintinShaw/homebrew-tap)
(version + per-target sha256 for `macos-arm64`, `linux-x86_64`, `linux-arm64`,
read from the just-published release's `SHA256SUMS`) and pushes straight to
that repo's `main`. It uses `scripts/update-homebrew-formula.py`, which fails
closed if the formula's shape does not match what it expects (e.g. a target's
`url` line has no corresponding `--sha256` given), rather than risk writing a
formula with a stale hash paired with the new version.

This needs a `HOMEBREW_TAP_TOKEN` repository secret: a **fine-grained GitHub
PAT** scoped to the `QuintinShaw/homebrew-tap` repository only, with
**Contents: Read and write** permission (nothing else). If the secret is not
set, the job prints a `::notice::` and skips -- the release itself still
succeeds and stays green; the tap formula just does not get bumped for that
release (bump it manually by re-running the `update-homebrew-tap` job, or by
hand, once the secret exists).

## Docker Hub images

`publish-core-channels.yml` runs only after the GitHub draft is published by
the catalog/CDN finalizer. It builds runtime images from the published Linux
release assets (no second cargo build inside Docker):

- CPU multi-arch (`linux/amd64` + `linux/arm64`) from
  `openasr-<version>-linux-x86_64.tar.gz` / `linux-arm64.tar.gz` via
  `Dockerfile.release`
- CUDA (`linux/amd64` only) from `openasr-<version>-linux-x86_64-cuda.tar.gz`
  via `Dockerfile.cuda.release`

Images push to Docker Hub under the `quintinshaw` namespace:

```text
quintinshaw/openasr:<version>
quintinshaw/openasr:latest
quintinshaw/openasr:sha-<short>
quintinshaw/openasr:cuda-<version>
quintinshaw/openasr:cuda-latest
quintinshaw/openasr:cuda-sha-<short>
```

`latest` / `cuda-latest` move only on a successful formal release
(`mark_latest: true`). Images ship the CLI binary plus bundled
`model-registry` metadata only -- never model weights. Data lives under
`OPENASR_HOME=/data`.

This needs a `DOCKER_PAT` repository secret: a Docker Hub access token for
user `quintinshaw` with permission to push `quintinshaw/openasr`. If the
secret is not set, the job prints a `::notice::` and builds without pushing
-- the GitHub Release itself still succeeds; only the Hub publish is skipped.
A red Docker job fails that leg of the workflow only and does not delete or
roll back the already-published Release.

The distribution gate and Homebrew formula updater check out helper scripts
from the repository default branch, not the release tag, so a later CI-only
fix still applies when repairing an already-public tag. Docker images still
build from the tag's Dockerfiles and the published Linux tarballs.

Manual dry-run / re-publish against an existing tag:

```bash
gh workflow run publish-core-channels.yml -f tag=vX.Y.Z
gh workflow run docker-release.yml \
  -f version=X.Y.Z -f tag=vX.Y.Z -f push=false -f mark_latest=false -f variants=all
```

Local source-build Dockerfiles (`Dockerfile`, `Dockerfile.cuda`) remain for
development and `docker-smoke.yml`; they are not the release path.

## China mirror (CNB)

`sync-release-to-cnb.yml` runs on `release: published` for a stable `vX.Y.Z`
core tag (not `desktop-v*`). It does not require the tag to be GitHub
`latest`, so a later repair of an older tag is still valid. The job streams
one GitHub asset at a time through `scripts/sync-release-to-cnb.sh` and is
idempotent: already-mirrored names with a matching size are skipped.

Repair or finish a partial mirror:

```bash
gh workflow run sync-release-to-cnb.yml -f tag=vX.Y.Z
```

`sync-main-to-cnb.yml` only fast-forwards git `main`. It never copies release
assets.
