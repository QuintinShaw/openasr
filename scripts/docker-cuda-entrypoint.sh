#!/usr/bin/env bash
# Fail-closed GPU gate for the CUDA Docker image (Dockerfile.cuda).
#
# Why this exists: openasr's per-request execution target (`auto` / `cpu` /
# `accelerated`) is a request-level API field, not a server-wide CLI flag --
# there is no single "did the server pick CUDA" switch to check at startup.
# What we *can* check unconditionally, offline, and without a loaded model is
# `openasr doctor`'s device enumeration: it calls into ggml's CUDA backend
# init and lists every device it actually found, tagged with a `kind` (see
# crates/openasr-core/src/ggml_runtime/backend.rs's `GgmlBackendKind` and
# crates/openasr-cli/src/doctor_cli.rs's `device_kind_label`, which prints
# GPU devices as "(gpu, ...)"). This image only ever compiles the `cuda`
# ggml backend in, so any device doctor reports with kind "gpu" is a CUDA
# device -- there is no other GPU backend in this binary to confuse it with.
#
# If no such device shows up, the most likely causes are: the container
# was not started with `--gpus ...` / a `deploy.resources.reservations.devices`
# GPU reservation, the NVIDIA Container Toolkit is not installed on the host,
# or the host driver does not support this image's CUDA arch list. In every
# one of those cases we refuse to silently serve on CPU -- that would turn a
# broken GPU passthrough into a silent, much-slower-than-expected transcription
# service instead of a loud, immediately-diagnosable startup failure.
set -euo pipefail

fail() {
    echo "docker-cuda-entrypoint: $*" >&2
    exit 1
}

run_gpu_check() {
    local doctor_output
    if ! doctor_output="$(openasr doctor 2>&1)"; then
        echo "${doctor_output}" >&2
        fail "'openasr doctor' exited non-zero; cannot verify GPU availability."
    fi

    if ! printf '%s\n' "${doctor_output}" | grep -Eq '\(gpu,|\(integrated-gpu,'; then
        echo "${doctor_output}" >&2
        fail "no GPU device reported by 'openasr doctor'. This image only" \
             "runs GPU-accelerated (see Dockerfile.cuda); it refuses to fall" \
             "back to CPU silently. Start the container with GPU access" \
             "(e.g. 'docker run --gpus all ...', or the compose 'gpu' profile" \
             "in compose.yaml) and confirm the NVIDIA Container Toolkit is" \
             "installed and the host driver is compatible."
    fi
}

# The HEALTHCHECK directive invokes this script with --gpu-check-only so it
# can re-verify GPU visibility on a running container without touching the
# main process; the normal container startup path (no args, or the compose
# `command:` override) runs the same check once, fail-closed, before execing
# the real openasr command.
if [ "${1:-}" = "--gpu-check-only" ]; then
    run_gpu_check
    exit 0
fi

run_gpu_check
exec openasr "$@"
