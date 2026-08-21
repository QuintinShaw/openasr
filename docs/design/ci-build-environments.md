# CI build environments

OpenASR's public GitHub Actions stay on free hosted runners. The expensive
work is compiling ggml plus the CUDA/ROCm toolkits, so the release matrix
pins three Linux environments and leaves Windows on the hosted VS image.

## Three Linux environments

### CPU (`openasr-ci-linux`)

- Image: `ghcr.io/quintinshaw/openasr-ci-linux`, digest in
  `.github/ci/linux-build-env.lock`.
- Source: `.github/ci/linux-build-env/`.
- Publish: `.github/workflows/ci-linux-build-env.yml`, only when that tree
  changes. Tag pushes do not publish.
- Consumers: `ci.yml`, `family-regression.yml`, `public-hf-e2e.yml`,
  `serve-batch-parity.yml`, and the
  `x86_64-unknown-linux-gnu` release-binaries leg.
- Contract: `.github/ci/check-linux-build-env.py` requires every listed
  consumer to pin the lock digest. Fully migrated consumers must not
  `apt-get`.

The image is Ubuntu 24.04 with C/C++, CMake, Ninja, ALSA, and Python. It
does not ship a Vulkan SDK. The Linux Vulkan release leg therefore stays on
`ubuntu-22.04` and LunarG's jammy apt recipe so the live SDK source does not
change.

### CUDA 13.2.0

- Consumer pin: `.github/ci/linux-cuda.lock`
  (`ghcr.io/quintinshaw/openasr-ci-linux-cuda@sha256:7d3a80aa…`).
- Wrapper source: `.github/ci/linux-cuda/`, published by
  `.github/workflows/ci-linux-cuda-env.yml` on its own path changes only.
- Base: official
  `nvidia/cuda:13.2.0-devel-ubuntu22.04@sha256:c7732db6b0128a468fab3d4c45d7063e075e7001c96e0b5303bb406cd59eb8c3`
  (linux/amd64). The wrapper adds git, cmake, ninja, ALSA, and a non-root user
  so `actions/checkout` and bash steps work.

### ROCm 7.2.1

- Consumer pin: `.github/ci/linux-rocm.lock`
  (`ghcr.io/quintinshaw/openasr-ci-linux-rocm@sha256:98ed2646…`).
- Wrapper source: `.github/ci/linux-rocm/`, published by
  `.github/workflows/ci-linux-rocm-env.yml` on its own path changes only.
- Base: official
  `rocm/dev-ubuntu-22.04:7.2.1@sha256:42851dac319afce41cf993e25f95005b7f2cd0a0f6abd32ad8f25cd876ec56df`.

Do not change those CUDA/ROCm versions to match a newer toolkit. Live
release pins win.

## Why Windows is not containerized

Windows CUDA 12.6.3 / 12.8.1 and HIP SDK 26.Q1 are installed by the existing
composite actions onto `windows-2022`. There is no official, digest-pinned
Windows container on GitHub-hosted runners that already contains those
toolkits, and pulling a Windows container would not remove the VS / CUDA /
HIP install cost that dominates those legs. Plugin legs now compile
`openasr-core` instead of `openasr-cli` so CMake still emits
`openasr-backend-packs/<provider>/ggml-<provider>.dll` without linking the
CLI.

## Cache keys

| Cache | Key | Failure mode |
| --- | --- | --- |
| `./.github/actions/rust-cache` | `release-binaries-${{ matrix.target }}` (and `release-binaries-xcframework`) | `continue-on-error: true` plus an explicit `::warning::`. A cache miss or action outage must not fail the release. |
| cbindgen | `cbindgen-${{ runner.os }}-${{ runner.arch }}-${{ hashFiles('rust-toolchain.toml', 'Cargo.lock') }}` | Miss installs with `cargo install cbindgen --locked`. |
| docker-smoke buildx | `type=gha` | Layer cache only; the smoke assertions are unchanged. |
| Windows HIP SDK | `hip-sdk-${{ env.HIPSDK_INSTALLER_VERSION }}-${{ runner.os }}` | Unchanged. |
| Vulkan loader zip | `vulkan-loader-${{ env.VULKAN_LOADER_VERSION }}` | Unchanged; hash is rechecked on every run. |

There is no remote sccache. Do not add one: forks cannot write a shared
cache safely, and a poisoned cache would ship into release artifacts.

## Rollback

Revert the workflow and matrix commits. Consumers that still pin a digest
keep working; the previous `apt-get` / `Jimver/cuda-toolkit` / ROCm apt
paths remain in git history. Wrapper image workflows are additive: deleting
or ignoring them does not affect a consumer that still uses the official
NVIDIA/ROCm digest.

A failed rust-cache step is not a rollback trigger. Re-run the job.

## Forks and `packages:write`

Image publish workflows need `packages: write` to push to
`ghcr.io/quintinshaw/...`. Forks do not receive that permission on the
canonical package namespace, so a fork PR that only touches
`.github/ci/linux-*` will fail the publish job. That is expected. Routine
PR CI (`ci.yml`, `docker-smoke.yml`, `serve-batch-parity.yml`) only *pulls*
the already-published CPU image with `packages: read` and does not publish.

Do not point a consumer at a mutable tag such as `:latest` or `:${{ github.sha }}`
from an unpublished fork build.
