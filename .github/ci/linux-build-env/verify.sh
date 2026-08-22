#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "OpenASR routine CI must run as a non-root user" >&2
  exit 1
fi

cmake --version
cc --version
gh --version
gh attestation verify --help >/dev/null
pkg-config --exists alsa
python3 -c 'import numpy; print(numpy.__version__)'
