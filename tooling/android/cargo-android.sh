#!/usr/bin/env bash
# Cross-compile OpenASR for Android (aarch64) with the NDK.
#
# Resolves the NDK linker/compilers from ANDROID_NDK_HOME (or the Homebrew
# `android-ndk` cask default) and exports the cargo/cc cross-compile env so the
# committed `.cargo/config.toml` stays portable (no machine-specific absolute
# paths). build.rs picks up the same NDK for the ggml cmake toolchain file.
#
# Usage:
#   tooling/android/cargo-android.sh build -p openasr-core
#   tooling/android/cargo-android.sh build -p openasr-core --features vulkan
#   tooling/android/cargo-android.sh build -p openasr-cli --release
#
# Env overrides:
#   ANDROID_NDK_HOME      NDK root (default: Homebrew android-ndk)
#   OPENASR_ANDROID_API   min API level (default: 24). Vulkan builds are bumped to
#                         >=28 automatically (ggml-vulkan links the Vulkan 1.1 core
#                         symbol vkGetPhysicalDeviceFeatures2, exported by the NDK
#                         libvulkan.so only from API 28) so the rustc final link
#                         resolves against an API-28+ sysroot loader.
set -euo pipefail

NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-${NDK_HOME:-/opt/homebrew/share/android-ndk}}}"
if [ ! -f "$NDK/build/cmake/android.toolchain.cmake" ]; then
  echo "error: Android NDK not found at '$NDK'" >&2
  echo "       (no build/cmake/android.toolchain.cmake). Set ANDROID_NDK_HOME." >&2
  exit 1
fi
export ANDROID_NDK_HOME="$NDK"

API="${OPENASR_ANDROID_API:-24}"
case "$API" in
  '' | *[!0-9]*)
    echo "error: OPENASR_ANDROID_API must be a positive integer (got '$API')" >&2
    exit 1
    ;;
esac
# Vulkan needs an API-28+ sysroot loader (see header). Bump the link API floor when
# the cargo invocation requests the vulkan feature, so it matches build.rs.
case " $* " in
  *vulkan*) if [ "$API" -lt 28 ]; then API=28; fi ;;
esac
PREBUILT="$NDK/toolchains/llvm/prebuilt"
HOSTTAG="$(ls "$PREBUILT" 2>/dev/null | head -1)"
if [ -z "${HOSTTAG:-}" ]; then
  echo "error: no prebuilt toolchain under $PREBUILT" >&2
  exit 1
fi
BIN="$PREBUILT/$HOSTTAG/bin"
TARGET=aarch64-linux-android
CLANG="$BIN/${TARGET}${API}-clang"

# rustc final-link uses the cargo target linker; cc-built C deps (zstd-sys, ring,
# …) use the per-target CC/CXX/AR. NDK uses underscore-form env names.
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CLANG"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_AR="$BIN/llvm-ar"
export CC_aarch64_linux_android="$CLANG"
export CXX_aarch64_linux_android="${CLANG}++"
export AR_aarch64_linux_android="$BIN/llvm-ar"
export RANLIB_aarch64_linux_android="$BIN/llvm-ranlib"

exec cargo "$@" --target "$TARGET"
