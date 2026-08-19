# Linux CUDA CI build environment
#
# Wrapper around the official NVIDIA CUDA 13.2.0 devel image. Release consumers
# in this change still pin that official digest directly; switch them to the
# published GHCR digest only after this workflow has produced one.
#
# Base digest verified 2026-08-20 with `docker buildx imagetools inspect`
# nvidia/cuda:13.2.0-devel-ubuntu22.04:
#   index  sha256:46a7128f65491ae6ed6e3a4ef31fc140a4ca77d5dddb68a4350fea9d53a95c5f
#   amd64  sha256:c7732db6b0128a468fab3d4c45d7063e075e7001c96e0b5303bb406cd59eb8c3

FROM nvidia/cuda:13.2.0-devel-ubuntu22.04@sha256:c7732db6b0128a468fab3d4c45d7063e075e7001c96e0b5303bb406cd59eb8c3

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# cmake/ninja/alsa/git are not part of the official CUDA devel contract.
# rustup is installed so later consumers can skip a cold toolchain fetch; jobs
# still honor rust-toolchain.toml via rustup override.
RUN success=0; \
    for attempt in 1 2 3; do \
      if apt-get update -o Acquire::Retries=5 \
        && apt-get install -y --no-install-recommends \
          build-essential \
          ca-certificates \
          cmake \
          curl \
          git \
          libasound2-dev \
          ninja-build \
          pkg-config \
          python3 \
          python3-pip \
          unzip \
          xz-utils \
          zstd; then \
        success=1; \
        break; \
      fi; \
      echo "apt attempt ${attempt}/3 failed" >&2; \
      sleep $((attempt * 10)); \
    done; \
    test "${success}" -eq 1; \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1001 openasr-ci \
    && useradd --uid 1001 --gid 1001 --home-dir /github/home \
      --create-home --shell /bin/bash openasr-ci

ENV CUDA_PATH=/usr/local/cuda \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:/github/home/.cargo/bin:${PATH}"

COPY --chmod=0755 verify.sh /usr/local/bin/openasr-ci-verify

USER 1001:1001

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain 1.95.0 --profile minimal \
    && /github/home/.cargo/bin/rustup default 1.95.0

RUN openasr-ci-verify

LABEL org.opencontainers.image.source="https://github.com/QuintinShaw/openasr" \
      org.opencontainers.image.description="Pinned Linux CUDA 13.2.0 build environment for OpenASR CI" \
      org.opencontainers.image.licenses="Apache-2.0"
