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

# Builds `openasr-ffi` for one Rust target and stages `<slice_dir>/lib/` +
# `<slice_dir>/include/` for `xcodebuild -create-xcframework`.
build_slice() {
  local rust_target="$1"
  local slice_dir="$2"

  echo "==> building openasr-ffi for $rust_target ($configuration)"
  if ! rustup target list --installed | grep -qx "$rust_target"; then
    echo "    (adding missing rustup target $rust_target)"
    rustup target add "$rust_target"
  fi

  # openasr-core's build.rs floors IPHONEOS_DEPLOYMENT_TARGET at 15.0 for the
  # CMake-built ggml C/C++ objects (see ios_deployment_target_from in
  # crates/openasr-core/build.rs), and ring's build script picks up the
  # host's SDK version (26.5 on this runner) for its precompiled asm objects.
  # But rustc's own final link of openasr-ffi's cdylib/staticlib hardcodes
  # `-target arm64-apple-ios10.0.0` regardless of IPHONEOS_DEPLOYMENT_TARGET
  # (that env var is not honored for the plain, non-simulator aarch64-apple-ios
  # target as of rustc 1.95.0). Linking against that old a minimum makes the
  # linker treat `___chkstk_darwin` (a stack-probe helper gated to iOS 11+ in
  # the SDK's libSystem.tbd) as unavailable, so ggml's/ring's newer-floor
  # object code fails with an undefined-symbol error. Force the real minimum
  # via an explicit `-target` link-arg pair, which clang resolves last-wins
  # over the one rustc already inserted, so the whole slice (C++ and Rust)
  # links against one consistent, high-enough minimum.
  local rustflags=""
  case "$rust_target" in
    aarch64-apple-ios) rustflags="-C link-arg=-target -C link-arg=arm64-apple-ios15.0" ;;
    aarch64-apple-ios-sim) rustflags="-C link-arg=-target -C link-arg=arm64-apple-ios15.0-simulator" ;;
  esac

  if [[ -n "$rustflags" ]]; then
    (cd "$repo_root" && RUSTFLAGS="$rustflags" cargo build -p openasr-ffi $cargo_flag --target "$rust_target")
  else
    (cd "$repo_root" && cargo build -p openasr-ffi $cargo_flag --target "$rust_target")
  fi

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
