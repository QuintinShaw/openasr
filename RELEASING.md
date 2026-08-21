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
missing draft, but an already-existing draft is an intentional no-op; recover
failed matrix legs through the `Release binaries` workflow below rather than
starting a competing second orchestrator.

`workflow_dispatch` on `Release binaries` (`.github/workflows/release-binaries.yml`)
independently rebuilds/re-uploads the full matrix for an existing tag: pass
`ref: vX.Y.Z` to target it, or `dry_run: true` to exercise the tag-resolution,
upload, and completeness-gate logic without mutating the release's assets
(the completeness check still runs and will fail loudly if that release is
genuinely incomplete -- that failure is expected and informative, not a bug
in the dry run).

The core GitHub Release is created as a **draft**. This is load-bearing for the
Windows plugin topology: CUDA/HIP payload hashes exist only after the release
matrix has built them, while the neutral host resolves those hashes from the
production-signed catalog. Core 0.1.34 and later publish no legacy whole-engine
Windows sidecars and no per-release `backends-manifest.json`. A draft is not
made public until the signed catalog distribution plane is complete:

1. Attach `backend-hardware-evidence-*.json` receipts for every provider that is
   intended to become runtime-selectable. Schema v1 approves only the exact
   tested target; schema v2 may approve an explicit provider matrix. For 0.1.36,
   produce schema v2 with
   `tooling/release-manifest/generate_backend_hardware_evidence.py` and attach
   both its `backend-hardware-evidence-*.json` summary and separately named
   `backend-hardware-audit-*.json` raw audit. The runner verifies every release
   subject against `SHA256SUMS` and GitHub build provenance, proves the executed
   binary and its complete companion-file tree match the neutral release ZIP,
   restricts the local preview catalog to exact file-URL substitutions from the
   attested candidate, cryptographically preflights its local-dev signature in
   a fresh empty home, checks the evidence-home cache is the same signed pair,
   and checks the model pack against that candidate. It owns at least five fresh
   child processes, checks the exact activation before
   and after each child, and binds each raw receipt to a unique nonce. Model and
   audio inputs are content-hash bound but are not release subjects. The
   summary's `evidence_sha256` is the
   canonical raw-audit digest. The v0.1.36 tag gate validates the schema-v2
   summary and provider matrix; it does not parse the raw audit, which must be
   downloaded and independently checked before catalog publication. A future
   release may make that binding part of the tag-integrated schema. A
   provider-matrix receipt is compatibility policy, not a claim that every GPU
   ran. Building a target, copying one receipt five times, scheduler/hybrid
   execution, or CPU/other-provider compute is rejected.
2. Run `scripts/sync-windows-backend-cdn.sh vX.Y.Z` locally with the B2
   release credentials. It copies the hardware-approved plugin and vendor
   files to `https://dl.openasr.org/core/vX.Y.Z/`. The signed catalog's
   `files[].url` values point only at this prefix; GitHub release mirrors are
   not a runtime download fallback.
3. Run `scripts/prepare-windows-backend-catalog-release.sh vX.Y.Z` locally with
   the production catalog signing seed. It downloads and hashes all 6 CUDA and
   14 HIP build artifacts, but merges only the target entries approved by those
   receipts. Before touching the catalog it verifies that every selected CDN
   payload is already live, then bumps the epoch and signs the full/public
   catalogs. Review, commit, and push those catalog files.
4. Wait for `deploy-catalog.yml` to repeat the no-credential CDN gate and prove
   the signed public bytes are live. Metadata is never deployed ahead of its
   immutable payloads.
5. Run `scripts/finalize-core-release.sh vX.Y.Z`. It requires the live signed
   catalog target set to equal the hardware-approved subset exactly, and it
   HEADs every signed CDN URL for that version; only then does it publish the
   draft and mark it latest.

None of these scripts publishes code or a catalog implicitly. A failure leaves
the release draft and therefore unavailable to users. Publishing the GitHub
release triggers `publish-core-channels.yml`, which moves Docker/Homebrew only
after the canonical catalog/CDN plane is complete.

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
