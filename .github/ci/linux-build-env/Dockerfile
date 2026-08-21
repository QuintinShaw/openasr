FROM ubuntu:24.04@sha256:d78ab76437b1afc5f01e223d6bf0172763f404bb166441328845adbef44518cb

ARG DEBIAN_FRONTEND=noninteractive
ARG GH_VERSION=2.98.0
ARG GH_LINUX_AMD64_SHA256=3b8ac6b30336802fc1a858d7c084e11cdf24ac1a761ca90b68022d7d729208de

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

RUN curl --fail --location --retry 3 \
      --output /tmp/gh.tar.gz \
      "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz" \
    && echo "${GH_LINUX_AMD64_SHA256}  /tmp/gh.tar.gz" | sha256sum --check --strict \
    && tar --extract --gzip --file /tmp/gh.tar.gz \
      --directory /usr/local/bin --strip-components=2 \
      "gh_${GH_VERSION}_linux_amd64/bin/gh" \
    && rm /tmp/gh.tar.gz

RUN groupadd --gid 1001 openasr-ci \
    && useradd --uid 1001 --gid 1001 --home-dir /github/home \
      --create-home --shell /bin/bash openasr-ci

COPY --chmod=0755 verify.sh /usr/local/bin/openasr-ci-verify

USER 1001:1001

RUN openasr-ci-verify

LABEL org.opencontainers.image.source="https://github.com/QuintinShaw/openasr" \
      org.opencontainers.image.description="Pinned Linux build environment for OpenASR CI" \
      org.opencontainers.image.licenses="Apache-2.0"
