# Releasing

OpenASR uses a single workspace version and a commit-driven release flow: a
version bump pushed to `main` IS the release.

Feature, fix, and any other content changes go through pull requests as usual.
The release bump itself is the exception: a maintainer pushes it directly to
`main` as a single `chore(release)` commit plus its annotated `vX.Y.Z` tag.
Routing the bump through a PR adds nothing (the release fires on the merge
commit anyway) and CI runs on `main` push regardless.

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
   - otherwise builds its own macOS-arm64 + Linux-x86_64 binaries, reads the
     tag annotation for Highlights, and creates the GitHub Release with a
     three-part body (see below);
   - then calls `.github/workflows/release-binaries.yml` directly (as a
     `workflow_call`, not a webhook -- tags created through the
     `GITHUB_TOKEN` API do not cascade into further Actions triggers, so a
     real `push: tags` event alone would never fire for a tag this workflow
     created) to build the full release matrix (Linux x86_64/arm64, macOS
     x86_64/arm64, Windows, plus Vulkan/CUDA/HIP feature variants) and
     upload every archive to the same release;
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
resolve-and-release for the version currently on `main` (useful for retries;
it is idempotent thanks to the existing-release check).

`workflow_dispatch` on `Release binaries` (`.github/workflows/release-binaries.yml`)
independently rebuilds/re-uploads the full matrix for an existing tag: pass
`ref: vX.Y.Z` to target it, or `dry_run: true` to exercise the tag-resolution,
upload, and completeness-gate logic without mutating the release's assets
(the completeness check still runs and will fail loudly if that release is
genuinely incomplete -- that failure is expected and informative, not a bug
in the dry run).

## Backends-manifest signing (REQUIRED, LOCAL ONLY -- not optional)

Every release that ships a Windows GPU-kernel `backends-manifest.json`
(schema v2 -- see `tooling/release-manifest/README.md`) is **not complete**
until that manifest is signed and the signature is verified against the
actual published release asset. This is the exact gap that shipped core
0.1.16-0.1.19 with a never-signed manifest: the signing seed
(`OPENASR_CATALOG_SIGNING_KEY_SEED_HEX`) is LOCAL ONLY and must never enter
CI, so it cannot be automated away -- but "a maintainer must remember to run
three commands afterwards" is exactly the kind of step that gets forgotten.

**The primary gate is a single atomic script, not a checklist:**

```bash
OPENASR_CATALOG_SIGNING_KEY_SEED_HEX=<real production seed> \
  scripts/sign-and-verify-backends-manifest.sh vX.Y.Z
```

Run this once `release-binaries.yml` has finished and its `checksums` job has
attached the unsigned `backends-manifest.json` to the release. The script:

1. downloads the unsigned manifest from the release and signs it with
   `__openasr-sign-backends-manifest`;
2. uploads `backends-manifest.signature.json` to the release (and, only if
   `B2_S3_ENDPOINT`/`B2_APPLICATION_KEY_ID`/`B2_APPLICATION_KEY` are already
   set in the environment, best-effort syncs both files to
   dl.openasr.org -- that sync is documented as optional in
   `tooling/release-manifest/README.md` and never blocks this script);
3. **re-downloads** the manifest + signature it just published (not the
   local copies) and self-verifies them with
   `__openasr-verify-backends-manifest` against the production trust root.

Any of the three steps failing aborts immediately with a `SIGNING/VERIFY
FAILED for vX.Y.Z` banner and a non-zero exit. Treat that exactly like a
failed test: **the release is not signed and must not be announced or
shipped until the script exits 0 with `SIGNED-AND-VERIFIED`.** Do not fall
back to running the old individual `gh`/`cargo run` commands by hand except
to debug why the script itself failed.

This step cannot be folded into `scripts/bump-version.sh` (that script runs
*before* the tag is pushed and before CI has built the release archives the
manifest is generated from -- the unsigned `backends-manifest.json` does not
exist yet at that point). It is, by construction, the last step of a
release.

CI also runs a **secondary safety net** -- `release-binaries.yml`'s
`verify-backends-manifest-signature` job (`workflow_dispatch` only, since the
signature cannot exist until after this local script has run). It performs
the exact same re-download-and-verify check `sign-and-verify-backends-manifest.sh`
already does. Treat a red run there as confirmation the local step above was
skipped, not as the primary way this gets caught -- the local script above
is the primary gate; CI's job is a fallback in case a release ever gets
announced without it having been run.

## Homebrew tap

`release-core.yml`'s final job, `update-homebrew-tap`, bumps
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
