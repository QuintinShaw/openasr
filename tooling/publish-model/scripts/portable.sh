#!/usr/bin/env bash
# Portable shell helpers for publish-model stages. Source after lib.sh.

portable_file_size() {
  local path="${1:?portable_file_size <path>}" size os
  [[ -f "$path" ]] || die "file missing for size probe: $path"
  os="$(uname -s 2>/dev/null || echo unknown)"
  case "$os" in
    Darwin|FreeBSD|OpenBSD|NetBSD)
      size="$(stat -f '%z' "$path" 2>/dev/null)" || die "stat -f failed for $path"
      ;;
    Linux)
      size="$(stat -c '%s' "$path" 2>/dev/null)" || die "stat -c failed for $path"
      ;;
    *)
      size="$(python3 - "$path" <<'PY'
import os
import sys

print(os.path.getsize(sys.argv[1]))
PY
)" || die "python size probe failed for $path"
      ;;
  esac
  [[ "$size" =~ ^[0-9]+$ ]] || die "non-numeric size for $path: $size"
  printf '%s\n' "$size"
}

portable_sha256() {
  local path="${1:?portable_sha256 <path>}" output sha
  [[ -f "$path" ]] || die "file missing for sha256 probe: $path"
  command -v openssl >/dev/null 2>&1 || die "openssl is required for portable sha256"
  output="$(openssl dgst -sha256 "$path")" || die "openssl sha256 failed for $path"
  sha="${output##* }"
  sha="$(printf '%s' "$sha" | tr 'A-F' 'a-f')"
  [[ "$sha" =~ ^[0-9a-f]{64}$ ]] || die "invalid sha256 from openssl for $path: $output"
  printf '%s\n' "$sha"
}

portable_human_bytes() {
  local size="${1:?portable_human_bytes <size>}"
  python3 - "$size" <<'PY'
import sys

size = int(sys.argv[1])
units = ["B", "KiB", "MiB", "GiB", "TiB"]
value = float(size)
for unit in units:
    if value < 1024 or unit == units[-1]:
        if unit == "B":
            print(f"{int(value)} B")
        else:
            print(f"{value:.1f} {unit}")
        break
    value /= 1024
PY
}

portable_has_non_cache_file() {
  local root="${1:?portable_has_non_cache_file <dir>}"
  python3 - "$root" <<'PY'
import os
import sys

root = os.path.abspath(sys.argv[1])
cache = os.path.join(root, ".cache")
for dirpath, dirnames, filenames in os.walk(root):
    current = os.path.abspath(dirpath)
    if current == cache or current.startswith(cache + os.sep):
        dirnames[:] = []
        continue
    if filenames:
        sys.exit(0)
sys.exit(1)
PY
}
