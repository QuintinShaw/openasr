# Building the GPU backend plugins (Vulkan / HIP / CUDA)

OpenASR's Windows engine is a `GGML_BACKEND_DL` build: a CPU core
(`ggml-base.dll` + `ggml.dll` + `ggml-cpu-*.dll`) that loads GPU backends as
runtime plugin DLLs from `OPENASR_HOME/backends/<vendor>/<version>/`, with
automatic CPU fallback. The GPU plugins are built **separately** from the same
vendored ggml commit as the core (so the `ggml-base` C ABI matches across the
DLL boundary) and distributed as downloads.

This document records the **verified** standalone build recipes and the
local validation results on an RX 9060 XT (gfx1200, RDNA4) device.

> The SDK paths below are examples; adjust for your install.
> Use a **short build directory** (`E:\vk`, `E:\hip`) — the deep
> `vulkan-shaders-gen` / HIP template-instance paths trip `MAX_PATH` (C1083)
> under a normal nested target dir.

## Vulkan plugin (`ggml-vulkan.dll`)

Single self-contained DLL (~55 MB); its only runtime dep is the driver's
`vulkan-1.dll`. Cross-vendor fallback (AMD / NVIDIA / Intel). Built with the
**VS generator** (MSVC is fine for Vulkan):

```
cmake -G "Visual Studio 17 2022" -A x64 ^
  -S <repo>\crates\openasr-core\third_party\openasr-ggml -B E:\vk ^
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_VULKAN=ON ^
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF ^
  -DVulkan_INCLUDE_DIR="C:\VulkanSDK\1.4.350.0\Include" ^
  -DVulkan_LIBRARY="C:\VulkanSDK\1.4.350.0\Lib\vulkan-1.lib" ^
  -DVulkan_GLSLC_EXECUTABLE="C:\VulkanSDK\1.4.350.0\Bin\glslc.exe"
cmake --build E:\vk --config Release --target ggml-vulkan
```

Artifact: `E:\vk\bin\Release\ggml-vulkan.dll`. coopmat1 / integer-dot / bf16 are
auto-detected at build time by glslc feature-test shaders — there are no perf
`-D` flags.

## HIP plugin (`ggml-hip.dll`, all-AMD)

Must use **Ninja with ROCm clang as the C/CXX compiler** — the VS generator
cannot work because `ggml-hip/CMakeLists.txt` forces `CXX_IS_HIPCC=TRUE` on
Windows (the `.cu` files compile as CXX in `-x hip` mode), and the VS generator
binds CXX to MSVC `cl.exe`. `vcvars64` is still required: ROCm clang delegates
the final link to MSVC `link.exe`.

On ROCm 7.1 **no SDK shim is needed** — the import libs (`lib\amdhip64.lib`,
`rocblas.lib`, `libhipblas.dll.a`) and cmake config packages
(`lib\cmake\{hip,hipblas,rocblas}\`) ship with the SDK, so
`find_package(hip/hipblas/rocblas)` resolves natively. `ROCM_PATH` must be set
(only `HIP_PATH` is set by the installer).

```
call "...\VC\Auxiliary\Build\vcvars64.bat"
set "ROCM_PATH=C:\Program Files\AMD\ROCm\7.1"
set "HIP_PATH=C:\Program Files\AMD\ROCm\7.1"
:: ninja on PATH

cmake -G Ninja ^
  -S <repo>\crates\openasr-core\third_party\openasr-ggml -B E:\hip ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER="C:/Program Files/AMD/ROCm/7.1/bin/clang.exe" ^
  -DCMAKE_CXX_COMPILER="C:/Program Files/AMD/ROCm/7.1/bin/clang++.exe" ^
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF ^
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF ^
  -DGGML_HIP=ON -DCMAKE_HIP_PLATFORM=amd ^
  -DGPU_TARGETS=gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201 ^
  -DROCM_PATH="C:/Program Files/AMD/ROCm/7.1" -DCMAKE_PREFIX_PATH="C:/Program Files/AMD/ROCm/7.1" ^
  -DGGML_HIP_GRAPHS=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF ^
  -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_HIP_MMQ_MFMA=ON -DGGML_CUDA_FORCE_MMQ=OFF ^
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_EXPORT_METRICS=OFF
cmake --build E:\hip --target ggml-hip -j
```

(Use `-DGPU_TARGETS=gfx1200` alone for a fast local-only build.) Artifact:
`E:\hip\bin\ggml-hip.dll` (~73 MB). `GPU_TARGETS` is the right knob (not
`AMDGPU_TARGETS`, which is deprecated; not `CMAKE_HIP_ARCHITECTURES`, which is
only read on the Linux `enable_language(HIP)` path).

**`GGML_HIP_ROCWMMA_FATTN=OFF` is load-bearing.** OFF keeps the native MMA-F16
flash-attn path that the vendored naive-masked-attention workaround (pinned to
ggml commit `643b5659`) targets; ON would divert to the slower `fattn-wmma-f16`
kernel (needs rocwmma 2.0+) and re-expose the wide-GQA correctness bug. Do not
bump the vendored ggml without re-validating the workaround.

### Satellite DLLs (clean-machine distribution)

`ggml-hip.dll` imports `amdhip64_7.dll`, `libhipblas.dll`, `rocblas.dll`. On a
dev box with ROCm on `PATH` these resolve automatically. The shipped pack must
bundle them next to `ggml-hip.dll` (the `_7` suffix is tied to ROCm 7.1) plus
the whole `bin\rocblas\library\` Tensile directory (~150 MB), and load via
`LoadLibraryEx(LOAD_WITH_ALTERED_SEARCH_PATH)` after stripping the
`Zone.Identifier` MOTW. That packaging step is tracked for the distribution
phase.

## Local validation (RX 9060 XT / gfx1200)

Both plugins were staged into `OPENASR_HOME/backends/<vendor>/<ver>/` and loaded
by the engine (`openasr doctor`): the Vulkan and ROCm devices enumerate and
`init_best` ranks the GPU above CPU. HIP transcription output is **byte-identical
to CPU** (deterministic correctness gate — validates the RDNA4 flash-attn path).

`openasr transcribe --benchmark` (moonshine-tiny, 59.4 s of audio, identical 1608-char
output across all three):

| Backend | elapsed | real-time factor | vs CPU |
| --- | --- | --- | --- |
| CPU | 95.2 s | 1.60 | 1.0× |
| HIP (ROCm0) | 33.8 s | 0.57 | 2.82× |
| Vulkan0 | 35.2 s | 0.59 | 2.71× |

On this small encoder-decoder model HIP ≈ Vulkan; HIP's larger advantage shows
on the LLM matmuls (translation/large-ASR), where prefill throughput diverges.
