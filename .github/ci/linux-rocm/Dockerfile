# Linux ROCm CI build environment
#
# Wrapper around the official AMD ROCm 7.2.1 devel image plus hipBLAS/rocBLAS
# (the official devel image has hipcc but not the BLAS cmake packages).
#
# Base digest verified 2026-08-20 with `docker buildx imagetools inspect`
# rocm/dev-ubuntu-22.04:7.2.1:
#   sha256:42851dac319afce41cf993e25f95005b7f2cd0a0f6abd32ad8f25cd876ec56df

FROM rocm/dev-ubuntu-22.04:7.2.1@sha256:42851dac319afce41cf993e25f95005b7f2cd0a0f6abd32ad8f25cd876ec56df

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

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
          zstd \
          rocm-hip-sdk; then \
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

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH="/opt/rocm/bin:/github/home/.cargo/bin:${PATH}"

COPY --chmod=0755 verify.sh /usr/local/bin/openasr-ci-verify

USER 1001:1001

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain 1.95.0 --profile minimal \
    && /github/home/.cargo/bin/rustup default 1.95.0

RUN openasr-ci-verify

LABEL org.opencontainers.image.source="https://github.com/QuintinShaw/openasr" \
      org.opencontainers.image.description="Pinned Linux ROCm 7.2.1 build environment for OpenASR CI" \
      org.opencontainers.image.licenses="Apache-2.0"
