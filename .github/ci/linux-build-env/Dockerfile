FROM ubuntu:24.04@sha256:d78ab76437b1afc5f01e223d6bf0172763f404bb166441328845adbef44518cb

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Keep the routine Linux CI jobs independent of Ubuntu mirror availability.
# This networked install runs only when the image contract changes; consumers
# pin the resulting immutable GHCR digest and perform no apt work themselves.
RUN success=0; \
    for attempt in 1 2 3; do \
      if apt-get update -o Acquire::Retries=5 \
        && apt-get install -y --no-install-recommends \
          build-essential \
          ca-certificates \
          cmake \
          curl \
          file \
          git \
          jq \
          libasound2-dev \
          ninja-build \
          pkg-config \
          python3 \
          python3-numpy \
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

COPY --chmod=0755 verify.sh /usr/local/bin/openasr-ci-verify

USER 1001:1001

RUN openasr-ci-verify

LABEL org.opencontainers.image.source="https://github.com/QuintinShaw/openasr" \
      org.opencontainers.image.description="Pinned Linux build environment for OpenASR CI" \
      org.opencontainers.image.licenses="Apache-2.0"
