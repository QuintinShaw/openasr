FROM rust:1.95.0-trixie AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends libasound2-dev cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN cargo build --release -p openasr-cli

FROM debian:trixie-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libasound2t64 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 openasr \
    && mkdir -p /app /data \
    && chown -R openasr:openasr /app /data

WORKDIR /app
COPY --from=builder /app/target/release/openasr /usr/local/bin/openasr
COPY --from=builder /app/model-registry ./model-registry

ENV OPENASR_HOME=/data
# Binding 0.0.0.0 inside the container is the standard Docker pattern (exposure is
# controlled by the operator's port-publish / orchestration). The server is
# fail-closed against non-loopback plaintext by default; this image opts in
# explicitly. Front it with TLS (or a TLS-terminating proxy) for untrusted networks.
ENV OPENASR_ALLOW_INSECURE_NON_LOOPBACK=1

EXPOSE 8080

USER openasr
ENTRYPOINT ["openasr"]
CMD ["serve", "--addr", "0.0.0.0:8080"]
