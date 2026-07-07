#!/usr/bin/env bash
# Regenerates crates/openasr-ffi/include/openasr.h from the openasr-ffi Rust
# source via cbindgen. Run after changing any `#[no_mangle] pub extern "C"`
# item, `#[repr(C)]` type, or doc comment on them in crates/openasr-ffi/src/lib.rs.
#
# Usage:
#   scripts/generate-ffi-header.sh          # regenerate in place
#   scripts/generate-ffi-header.sh --check  # verify committed header is current (CI gate)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
crate_dir="$repo_root/crates/openasr-ffi"
header_path="$crate_dir/include/openasr.h"

if ! command -v cbindgen >/dev/null 2>&1; then
  echo "error: cbindgen not found on PATH. Install with: cargo install cbindgen --locked" >&2
  exit 1
fi

mkdir -p "$crate_dir/include"

if [[ "${1:-}" == "--check" ]]; then
  tmp_header="$(mktemp)"
  trap 'rm -f "$tmp_header"' EXIT
  cbindgen --config "$crate_dir/cbindgen.toml" --crate openasr-ffi --output "$tmp_header" "$crate_dir"
  if ! diff -u "$header_path" "$tmp_header"; then
    echo "error: crates/openasr-ffi/include/openasr.h is stale." >&2
    echo "Run scripts/generate-ffi-header.sh to regenerate and commit the result." >&2
    exit 1
  fi
  echo "openasr.h is up to date."
else
  cbindgen --config "$crate_dir/cbindgen.toml" --crate openasr-ffi --output "$header_path" "$crate_dir"
  echo "Wrote $header_path"
fi
