#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "OpenASR routine CI must run as a non-root user" >&2
  exit 1
fi

cmake --version
cc --version
hipcc --version
pkg-config --exists alsa
python3 --version
test -e /opt/rocm/lib/cmake/hipblas/hipblas-config.cmake
test -e /opt/rocm/lib/cmake/rocblas/rocblas-config.cmake
