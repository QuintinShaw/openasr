#!/usr/bin/env bash
# Builds OpenASR.xcframework from crates/openasr-ffi: a static-library
# xcframework bundling three architecture slices --
#   - ios-arm64            (device,        aarch64-apple-ios)
#   - ios-arm64-simulator  (simulator,     aarch64-apple-ios-sim)
#   - macos-arm64          (host,          aarch64-apple-darwin)
#
# Building a device or simulator slice requires a full Xcode install (the
# iphoneos/iphonesimulator SDKs; Command Line Tools alone do not ship them --
# see docs/SDK_IOS_MACOS.md and the ios-compile CI job's comment in
# .github/workflows/ci.yml). This script probes for those SDKs and builds
# whichever slices the host can actually produce; missing slices are skipped
# with a clear warning rather than failing the whole build, so CPU-only local
# iteration on a Command-Line-Tools-only Mac still produces a usable
# macOS-only xcframework, while CI (macos-latest, full Xcode) produces all
# three.
#
# Usage:
#   scripts/build-xcframework.sh [--output-dir DIR] [--configuration release|debug]
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$repo_root/target/xcframework"
configuration="release"
# Keep every iOS object (Rust, C/C++, Metal, and linked SDK inputs) at the
# deployment target required by the embedding app. This is the sole iOS target
# contract; build_slice exports it through every build-system environment.
readonly ios_deployment_target="26.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --configuration)
      configuration="$2"
      shift 2
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

case "$configuration" in
  release|debug) ;;
  *)
    echo "error: --configuration must be 'release' or 'debug', got '$configuration'" >&2
    exit 1
    ;;
esac

lib_name="libopenasr_ffi.a"
cargo_profile_dir="$configuration"
if [[ "$configuration" == "release" ]]; then
  cargo_flag="--release"
else
  cargo_flag=""
fi

mkdir -p "$output_dir"
work_dir="$output_dir/slices"
rm -rf "$work_dir"
mkdir -p "$work_dir"

# --- SDK detection -----------------------------------------------------
# `xcrun --sdk <name> --show-sdk-path` fails closed (nonzero exit, no output)
# when the active developer directory has no such SDK, e.g. Command Line
# Tools only. We rely on that instead of parsing `xcodebuild -showsdks`,
# which itself hard-errors under Command Line Tools.
have_sdk() {
  xcrun --sdk "$1" --show-sdk-path >/dev/null 2>&1
}

have_iphoneos=0
have_iphonesimulator=0
have_sdk iphoneos && have_iphoneos=1
have_sdk iphonesimulator && have_iphonesimulator=1

if [[ "$have_iphoneos" -eq 0 || "$have_iphonesimulator" -eq 0 ]]; then
  cat >&2 <<'EOF'
warning: full Xcode (iphoneos/iphonesimulator SDKs) not found -- this host
  only has Command Line Tools (or an incomplete Xcode install). Skipping the
  iOS device and/or simulator slice(s); only the macOS slice will be built
  here. CI (.github/workflows/ci.yml, workflow_dispatch xcframework job) runs
  on macos-latest with full Xcode and produces all three slices. Install
  Xcode from the App Store and run `sudo xcode-select -s
  /Applications/Xcode.app/Contents/Developer` to build the missing slice(s)
  locally.
EOF
fi

# --- helpers -------------------------------------------------------------

# Rust ships a small set of precompiled compiler_builtins objects with the
# legacy LC_VERSION_MIN_IPHONEOS command. Its fixed-size command can be updated
# in place, whereas vtool cannot grow it without load-command padding. Normalize
# that legacy command inside the archive so every explicitly platform-tagged
# object observes the same iOS contract; platform-neutral Rust objects are left
# unchanged.
normalize_ios_archive() {
  local archive="$1"
  local platform="$2"
  python3 - "$archive" "$platform" "$ios_deployment_target" <<'PY'
import struct
import sys
from pathlib import Path

archive = Path(sys.argv[1])
expected_platform = int(sys.argv[2])
target_major, target_minor = (int(part) for part in sys.argv[3].split('.', 1))
target_version = (target_major << 16) | (target_minor << 8)
data = bytearray(archive.read_bytes())
if data[:8] != b'!<arch>\n':
    raise SystemExit(f"{archive}: not a BSD archive")

offset = 8
while offset < len(data):
    header = data[offset:offset + 60]
    if len(header) != 60 or header[58:60] != b'`\n':
        raise SystemExit(f"{archive}: malformed archive header at offset {offset}")
    size = int(header[48:58].decode('ascii').strip())
    member_start = offset + 60
    member_end = member_start + size
    if member_end > len(data):
        raise SystemExit(f"{archive}: truncated member at offset {offset}")
    raw_name = header[:16].decode('ascii', errors='replace').rstrip()
    payload_start = member_start
    if raw_name.startswith('#1/'):
        name_size = int(raw_name[3:])
        payload_start += name_size
    payload_size = member_end - payload_start
    if payload_size >= 32 and data[payload_start:payload_start + 4] == b'\xcf\xfa\xed\xfe':
        ncmds = struct.unpack_from('<I', data, payload_start + 16)[0]
        command_offset = payload_start + 32
        payload_limit = payload_start + payload_size
        for _ in range(ncmds):
            if command_offset + 8 > payload_limit:
                raise SystemExit(f"{archive}: truncated Mach-O load command")
            command, command_size = struct.unpack_from('<II', data, command_offset)
            if command_size < 8 or command_offset + command_size > payload_limit:
                raise SystemExit(f"{archive}: malformed Mach-O load command")
            if command == (0x25 if expected_platform == 2 else 0x26):  # LC_VERSION_MIN_IPHONEOS(_SIMULATOR)
                version = struct.unpack_from('<I', data, command_offset + 8)[0]
                if version < target_version:
                    struct.pack_into('<I', data, command_offset + 8, target_version)
            elif command == 0x32:  # LC_BUILD_VERSION
                platform_id, version = struct.unpack_from('<II', data, command_offset + 8)
                if platform_id != expected_platform:
                    raise SystemExit(
                        f"{archive}: object has platform {platform_id}, expected {expected_platform}"
                    )
                if version < target_version:
                    struct.pack_into('<I', data, command_offset + 12, target_version)
            command_offset += command_size
    offset = member_end + (size & 1)

archive.write_bytes(data)
PY
}

build_slice() {
  local rust_target="$1"
  local slice_dir="$2"

  echo "==> building openasr-ffi for $rust_target ($configuration)"
  if ! rustup target list --installed | grep -qx "$rust_target"; then
    echo "    (adding missing rustup target $rust_target)"
    rustup target add "$rust_target"
  fi

  # All iOS producers must receive the same deployment target. In particular,
  # do not let the active Xcode SDK version (for example 26.5) become the object
  # minimum when the embedding app contract is 26.0.
  local rustflags=""
  local -a build_env=()
  case "$rust_target" in
    aarch64-apple-ios)
      rustflags="-C link-arg=-target -C link-arg=arm64-apple-ios${ios_deployment_target}"
      build_env=(
        "OPENASR_IOS_DEPLOYMENT_TARGET=$ios_deployment_target"
        "IPHONEOS_DEPLOYMENT_TARGET=$ios_deployment_target"
        "CMAKE_OSX_DEPLOYMENT_TARGET=$ios_deployment_target"
      )
      ;;
    aarch64-apple-ios-sim)
      rustflags="-C link-arg=-target -C link-arg=arm64-apple-ios${ios_deployment_target}-simulator"
      build_env=(
        "OPENASR_IOS_DEPLOYMENT_TARGET=$ios_deployment_target"
        "IPHONEOS_DEPLOYMENT_TARGET=$ios_deployment_target"
        "CMAKE_OSX_DEPLOYMENT_TARGET=$ios_deployment_target"
      )
      ;;
  esac

  if [[ -n "$rustflags" ]]; then
    (cd "$repo_root" && env "${build_env[@]}" RUSTFLAGS="$rustflags" cargo build -p openasr-ffi $cargo_flag --target "$rust_target")
  else
    (cd "$repo_root" && cargo build -p openasr-ffi $cargo_flag --target "$rust_target")
  fi

  case "$rust_target" in
    aarch64-apple-ios)
      normalize_ios_archive "$repo_root/target/$rust_target/$cargo_profile_dir/$lib_name" 2
      ;;
    aarch64-apple-ios-sim)
      normalize_ios_archive "$repo_root/target/$rust_target/$cargo_profile_dir/$lib_name" 7
      ;;
  esac

  mkdir -p "$slice_dir/lib" "$slice_dir/include"
  cp "$repo_root/target/$rust_target/$cargo_profile_dir/$lib_name" "$slice_dir/lib/$lib_name"
  cp "$repo_root/crates/openasr-ffi/include/openasr.h" "$slice_dir/include/openasr.h"
}

xcframework_args=()

# macOS (host) slice -- always buildable on macOS, CLT or full Xcode.
macos_dir="$work_dir/macos-arm64"
build_slice "aarch64-apple-darwin" "$macos_dir"
xcframework_args+=(-library "$macos_dir/lib/$lib_name" -headers "$macos_dir/include")

# iOS device slice.
if [[ "$have_iphoneos" -eq 1 ]]; then
  ios_dir="$work_dir/ios-arm64"
  build_slice "aarch64-apple-ios" "$ios_dir"
  xcframework_args+=(-library "$ios_dir/lib/$lib_name" -headers "$ios_dir/include")
else
  echo "==> skipping ios-arm64 (device) slice: no iphoneos SDK on this host"
fi

# iOS simulator slice.
if [[ "$have_iphonesimulator" -eq 1 ]]; then
  ios_sim_dir="$work_dir/ios-arm64-simulator"
  build_slice "aarch64-apple-ios-sim" "$ios_sim_dir"
  xcframework_args+=(-library "$ios_sim_dir/lib/$lib_name" -headers "$ios_sim_dir/include")
else
  echo "==> skipping ios-arm64-simulator slice: no iphonesimulator SDK on this host"
fi

# --- assemble the xcframework --------------------------------------------
xcframework_path="$output_dir/OpenASR.xcframework"
rm -rf "$xcframework_path"

functional_xcodebuild=1
if ! command -v xcodebuild >/dev/null 2>&1; then
  functional_xcodebuild=0
elif ! xcodebuild -version >/dev/null 2>&1; then
  # Command Line Tools ship a `xcodebuild` *binary* that refuses to run
  # (rather than being absent) -- `-create-xcframework` needs full Xcode
  # regardless of how many slices got built above.
  functional_xcodebuild=0
fi

if [[ "$functional_xcodebuild" -eq 0 ]]; then
  cat >&2 <<EOF
warning: xcodebuild requires a full Xcode install (this host has only
  Command Line Tools) -- cannot run 'xcodebuild -create-xcframework'.
  $lib_name and openasr.h were still built and staged per-slice under:
    $work_dir
  Use those directly for a manual smoke test (link against the macos-arm64
  slice's .a locally), or install Xcode and rerun this script to produce the
  real OpenASR.xcframework. CI (workflow_dispatch xcframework job,
  macos-latest) always has full Xcode and produces the complete xcframework.
EOF
  exit 0
fi

echo "==> creating $xcframework_path"
xcodebuild -create-xcframework "${xcframework_args[@]}" -output "$xcframework_path"

echo "==> done: $xcframework_path"
find "$xcframework_path" -maxdepth 2 -print
