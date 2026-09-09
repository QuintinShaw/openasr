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
# Default command binds 0.0.0.0 with HTTPS (self-signed) and device pairing.
# On first start, if OPENASR_PAIRING_ADMIN_TOKEN is unset, `serve` generates a
# random token, writes it owner-only to /data/pairing-admin-token, and prints
# it to stdout. Behind a TLS-terminating reverse proxy you may drop
# --tls-self-signed and set OPENASR_ALLOW_INSECURE_NON_LOOPBACK=1; that env
# only waives TLS, never pairing.

EXPOSE 8080

USER openasr
ENTRYPOINT ["openasr"]
CMD ["serve", "--addr", "0.0.0.0:8080", "--tls-self-signed", "--pairing-admin-token-file", "/data/pairing-admin-token"]
