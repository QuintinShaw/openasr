# Releasing

OpenASR uses a single workspace version and a commit-driven release flow: a
version bump pushed to `main` IS the release.

Feature, fix, and any other content changes go through pull requests as usual.
The release bump itself is the exception: a maintainer pushes it directly to
`main` as a single `chore(release)` commit plus its annotated `vX.Y.Z` tag.
Routing the bump through a PR adds nothing (the release fires on the merge
commit anyway). Routine CI is PR-only to avoid rebuilding every merge; its
narrow `main` push gate runs only for this direct `chore(release)` commit.

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

2. Push the commit **and** the tag together:

   ```bash
   git push --follow-tags
   ```

   Pushing just the commit without the tag (plain `git push`) is a mistake
   `release-core.yml` catches and fails loudly on -- it needs the tag's
   annotation for Highlights and refuses to guess.

3. The `Release core` workflow (`.github/workflows/release-core.yml`)
   triggers on `Cargo.toml` changes:
   - reads the workspace version and confirms the `vX.Y.Z` tag exists on
     origin (failing loudly if it's missing -- see step 2);
   - exits cleanly if a GitHub Release for `vX.Y.Z` already exists (so
     unrelated `Cargo.toml` edits and re-runs are no-ops);
   - otherwise reads the tag annotation for Highlights and immediately creates
     an empty draft GitHub Release with a three-part body (see below);
   - then calls `.github/workflows/release-binaries.yml` directly (as a
     `workflow_call`, the only formal matrix entrypoint) to build the full
     release matrix (Linux x86_64/arm64, macOS
     x86_64/arm64, Windows, plus Vulkan/CUDA/HIP feature variants) and
     upload every archive to the draft. There is no bootstrap macOS/Linux
     rebuild and no second `push: tags` matrix racing the orchestrator;
   - `release-binaries.yml`'s own completeness gate then asserts the release
     ends up with every expected platform archive, failing the run if one is
     missing instead of silently shipping a partial release;
   - finally, `release-core.yml` rewrites the release's Install & Verify
     section from the now-complete, real asset list.

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

`workflow_dispatch` on the `Release core` workflow performs the same
resolve-and-release for the version currently on `main`. It can create a
missing draft. After the generated PublishedInert catalog has been reviewed,
signed, committed, and pushed, dispatching it again runs the pre-publication
family gate and catalog deployment for that same draft. Failed build or
aggregation jobs are recovered only with GitHub's **Re-run failed jobs** on the
original release run, which preserves its exact tag/source/artifact lineage.

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

1. Let `Release core` create the draft and let its single formal
   `release-binaries.yml` call build and checksum the complete release matrix,
   sign the applicable Windows binaries, and attest the full subject set. Do not
   use the diagnostic one-target dispatch to assemble or mutate a release.
2. Run `scripts/sync-windows-backend-cdn.sh vX.Y.Z` locally with the B2 release
   credentials. It verifies and copies every file declared by the 1 Vulkan, 6
   CUDA, and 14 HIP release entries to
   `https://dl.openasr.org/core/vX.Y.Z/`. GitHub is the provenance mirror, not a
   runtime download fallback.
3. Run `scripts/prepare-windows-backend-catalog-release.sh vX.Y.Z` locally with
   the production catalog signing seed. It verifies all 21 exact entries and
   their already-live CDN bytes, merges every entry as `PublishedInert`, bumps
   the epoch, and signs the full/public catalogs. It consumes no hardware or
   token-correctness receipt. Review, commit, and push the five catalog/epoch
   files.
4. Dispatch `Release core` again. Its reusable pre-publication family contract
   verifies the immutable draft CLI and runs the real-pack CPU family gate. The
   orchestrator then calls the sole deployment entrypoint,
   `.github/workflows/deploy-catalog.yml`, with
   `activation_transition: published-inert`. The deploy job checks the signed
   catalog, CDN payloads, and released-binary compatibility before writing it.
5. Record the successful orchestrator run id as
   `OPENASR_DEPLOY_CATALOG_RUN_ID`, wait until the no-credential catalog/CDN
   checks see the committed bytes, and run
   `scripts/finalize-core-release.sh vX.Y.Z`. The finalizer binds the exact tag,
   source commit, deploy run, catalog SHA, signature SHA, release subjects, and
   live CDN bytes before removing the GitHub draft flag.

Publishing the release triggers `publish-core-channels.yml`, which moves
Docker/Homebrew only after the canonical catalog/CDN plane is complete. Any
failure in this pre-publication sequence leaves the GitHub release as a draft.

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

Manual dry-run / re-publish against an existing tag:

```bash
gh workflow run docker-release.yml \
  -f version=X.Y.Z -f tag=vX.Y.Z -f push=false -f mark_latest=false -f variants=all
```

Local source-build Dockerfiles (`Dockerfile`, `Dockerfile.cuda`) remain for
development and `docker-smoke.yml`; they are not the release path.
