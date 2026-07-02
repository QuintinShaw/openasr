# Android (aarch64) cross-compilation

`openasr-core` (the inference engine, including the vendored ggml) cross-compiles
to `aarch64-linux-android` so it can be embedded in an on-device Android app via
FFI. This covers the **CPU** backend (works on any Android SoC) and the
**Vulkan** GPU backend (Adreno / Mali / Immortalis).

> NPU backends (Qualcomm Hexagon) are intentionally **not** wired here — they
> require the proprietary Hexagon SDK and clash with this repo's
> `GGML_BACKEND_DL=OFF` static-link design. Use CPU or Vulkan on Android.

## Prerequisites

| Tool | Notes |
|------|-------|
| Android NDK (r27+) | `brew install --cask android-ndk`. The build auto-detects the Homebrew location (`/opt/homebrew/share/android-ndk`); otherwise set `ANDROID_NDK_HOME`. |
| Rust target | `rustup target add aarch64-linux-android` |
| Vulkan host deps (Vulkan only) | `brew install shaderc vulkan-headers spirv-headers`. All host-side / arch-neutral: `glslc` cross-compiles ggml's Vulkan shaders, `vulkan-headers` provides `vulkan.hpp` (the NDK sysroot ships only the old C `vulkan.h`), and `spirv-headers` provides the cmake config ggml-vulkan requires. The `libvulkan.so` **loader** comes from the NDK sysroot. Override discovery with `VULKAN_SDK` or `OPENASR_GLSLC`. |

The NDK host toolchain is `darwin-x86_64`; on Apple Silicon it runs under
Rosetta 2 (`softwareupdate --install-rosetta`).

## Build

Use the wrapper — it resolves the NDK linker/compilers and exports the cargo/cc
cross-compile env so the committed cargo config stays machine-independent:

```bash
# CPU (default)
tooling/android/cargo-android.sh build -p openasr-core --release

# Vulkan GPU
tooling/android/cargo-android.sh build -p openasr-core --release --features vulkan
```

The wrapper appends `--target aarch64-linux-android`. Override the min API level
with `OPENASR_ANDROID_API=<n>` (default 24) and the ABI with
`OPENASR_ANDROID_ABI` (default `arm64-v8a`). **Vulkan builds are bumped to API ≥28
automatically** — ggml-vulkan links the Vulkan 1.1 core symbol
`vkGetPhysicalDeviceFeatures2`, which the NDK `libvulkan.so` only exports from
API 28 (Android 9, the practical Vulkan 1.1 floor anyway).

### What the build does for Android

`build.rs` adds an `is_android` path that:

- passes the NDK CMake toolchain file (`-DCMAKE_TOOLCHAIN_FILE=…`,
  `-DANDROID_ABI`, `-DANDROID_PLATFORM`) so ggml cross-compiles for arm64;
- links the NDK C++ runtime (`c++_shared`) instead of GNU `libstdc++`, and emits
  no `libgomp` (the android triple also contains `"linux"`, so this branch is
  ordered before the desktop-Linux one);
- forces `GGML_OPENMP=OFF` (bionic ships no `libgomp` / `pthread_setaffinity`);
- leaves `GGML_NATIVE` off (cross build → portable codegen) but sets
  `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16` so the CPU kernels keep the int8/fp16
  matmul accelerators (dotprod/fp16) that the portable baseline would otherwise
  drop — a large speedup for quantized ASR. Covers Cortex-A55/A75+ (≈all Android
  devices since 2018). Override via `OPENASR_ANDROID_ARM_ARCH` (e.g. add `+i8mm`
  for armv8.6 flagships, or `armv8-a` for the broadest hardware floor).

Desktop builds (macOS/Linux/Windows) are byte-for-byte unchanged.

## Verifying the artifact

A library build does not exercise the final link; build a binary or test
executable to validate it:

```bash
tooling/android/cargo-android.sh test -p openasr-core --no-run
file target/aarch64-linux-android/debug/deps/openasr_core-*    # ELF 64-bit aarch64
```

A correctly linked binary lists `libc++_shared.so`, `libdl/libm/libc` as
`NEEDED` (and **not** `libstdc++` / `libgomp`):

```bash
"$ANDROID_NDK_HOME"/toolchains/llvm/prebuilt/*/bin/llvm-readelf -d <binary> | grep NEEDED
```

## On-device validation (your phone over USB)

CPU output must match the host CPU reference (the WER-0 oracle). Any Android
phone works for the CPU path (it is SoC-agnostic); a phone with a Vulkan driver
validates the GPU path. Enable USB debugging, authorize `adb`, then `adb push`
the binary together with `libc++_shared.so` (from
`$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so`)
and a model pack, and run with `LD_LIBRARY_PATH` pointing at the pushed `.so`.
The engine is consumed on-device as a library by the Android app (FFI); the
mobile app is the canonical end-to-end on-device vehicle.
