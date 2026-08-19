#!/usr/bin/env bash
set -euo pipefail

cmake --version
cc --version
pkg-config --exists alsa
python3 -c 'import numpy; print(numpy.__version__)'
