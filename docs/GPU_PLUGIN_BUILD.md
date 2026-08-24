# Building the GPU backend plugins (Vulkan / HIP / CUDA)

OpenASR's terminal Windows topology is one CPU-neutral `GGML_BACKEND_DL` host.
`ggml-base.dll`, `ggml.dll`, and the CPU variants are installer-owned; Vulkan,
CUDA, and HIP are all optional signed open-core provider packs. The host loads
only its verified CPU modules from the application directory. It may load one
provider from its verified, content-addressed directory under
`OPENASR_HOME/backends/` only after the signed catalog marks that exact pack
`activated`. It never chooses a first enumerated GPU, scans every installed
pack, or accepts a module merely because its filename looks compatible.

Every module is built from the same vendored ggml revision and generated host
ABI contract. Activation binds provider, SM/gfx targets, minimum driver,
plugin bytes, and vendor-tree bytes. A matching ABI string is necessary but is
not sufficient: installation rehashes every declared file and activation
probes the real exported backend API before committing the selected provider.
Prepared or published-inert bytes are not executable providers: ordinary Auto
and explicit runtime selection are Activated-only.

Historical whole, statically linked GPU sidecars are a legacy rollback format,
not an alternative build or runtime switch. Current neutral hosts reject them,
and legacy-static hosts must never load an optional provider pack.

This document records the standalone build recipes and historical local
validation results on an RX 9060 XT (gfx1200, RDNA4) device. Those measurements
do not qualify the current release bytes.

> The SDK paths below are examples; adjust for your install.
> Use a **short build directory** (`E:\vk`, `E:\hip`) — the deep
> `vulkan-shaders-gen` / HIP template-instance paths trip `MAX_PATH` (C1083)
> under a normal nested target dir.

## Vulkan provider (`ggml-vulkan.dll`)

This cross-vendor optional provider (AMD / NVIDIA / Intel) is not a bundled
rescue path. Its signed pack carries the required Vulkan loader runtime and
may enter a CPU-neutral host only after catalog activation. Build it with the
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

Release plugin legs do not invoke this cmake recipe directly. They run
`cargo build -p openasr-core --release --features hip` so `build.rs` keeps the
same cmake flag contract and stages
`target\release\openasr-backend-packs\hip\ggml-hip.dll`.

**`GGML_HIP_ROCWMMA_FATTN=OFF` is load-bearing.** OFF keeps the native MMA-F16
flash-attn path that the vendored naive-masked-attention workaround (pinned to
ggml commit `643b5659`) targets; ON would divert to the slower `fattn-wmma-f16`
kernel (needs rocwmma 2.0+) and re-expose the wide-GQA correctness bug. Do not
bump the vendored ggml without re-validating the workaround.

### Satellite DLLs (clean-machine distribution)

`ggml-hip.dll` imports the versioned HIP runtime, hipBLAS, and rocBLAS DLLs.
The shipped pack declares those files plus the complete rocBLAS Tensile
library tree in the signed catalog. The installer stores the vendor tree by
content hash and the loader uses an absolute plugin path with restricted DLL
search rooted at that verified tree. It does not depend on the process `PATH`.

Application downloads normally do not create a `Zone.Identifier` stream.
Installation may detect and report one; it may remove it only from an
individually verified file when a Windows loading test proves it is necessary.
Recursive, unconditional MOTW removal is not part of the trust model.

## CUDA plugin (`ggml-cuda.dll`, NVIDIA)

On Windows, the `cuda` Cargo feature stages `ggml-cuda.dll` as a separate
optional provider pack for the CPU-neutral host. The
`legacy-windows-static-sidecar` feature names the historical rollback format;
it is rejected from normal neutral-host release and activation paths. The
release workflow extracts the DLL, stages the CUDA runtime/cuBLAS vendor tree
separately, compiles one signed catalog entry, and publishes both byte
identities. Other platforms retain their platform-specific distribution
topology.

Release plugin legs compile `openasr-core` only so CMake still stages
`target\release\openasr-backend-packs\cuda\ggml-cuda.dll` with the same flags
the CLI build would have used. They do not compile `openasr-cli`.

For a local Windows host+plugin build, use the CLI crate and select the GPU
targets explicitly when validating one machine:

```text
set OPENASR_CUDA_GPU_TARGETS=86
cargo build -p openasr-cli --release --features cuda
```

The matching release plugin command is:

```text
set OPENASR_CUDA_GPU_TARGETS=86
cargo build -p openasr-core --release --features cuda
```

The resulting optional module is staged under
`target\release\openasr-backend-packs\cuda\ggml-cuda.dll`; the application
directory contains only the CPU-neutral host and its CPU modules.

The default arch list is `75;80;86;89;90` -- sm_75 (Turing: RTX 20xx, GTX 16xx,
T4, 2080 Ti) through sm_90 (Hopper). That is also this build's **hardware
floor**: CUDA 13 removed device-code generation for Volta/Pascal/Maxwell
(sm_70 and below) outright, so a default `cuda`-feature binary does not target
those cards. Supporting them needs a separate CUDA 12 toolchain build leg,
which is tracked but not shipped yet. Override the list with
`OPENASR_CUDA_GPU_TARGETS` for a narrower, wider, or newer set (e.g. add `120`
for Blackwell on a CUDA 12.8+ toolchain).

Users on pre-Turing NVIDIA hardware (Pascal/Volta and older) may use an
Activated Vulkan provider when a compatible signed pack and driver are
available. This is not a bundled fallback and has not been separately verified
against that specific old hardware.

## Validation boundary

The formal release publishes all 21 signed provider entries only as
`PublishedInert`. After publication, qualification runs from the exact release
tag and a separate backend-scoped authorization may activate one new signed
catalog epoch. See [RELEASING.md](../RELEASING.md) and
[`tooling/release-manifest/README.md`](../tooling/release-manifest/README.md).

The current Windows release gate builds and packages all three optional
providers, but hardware claims remain provider- and exact-target-specific. This
integration checkpoint has no release-bound CUDA, physical Vulkan, or HIP
receipt, so every such entry remains `PublishedInert`. A machine without the
matching hardware may prove only build/package/catalog/loader structure; it
must not report inference, placement, or token correctness as passed. Historical
host measurements below preserve tuning context only and cannot activate a
current release.

### Historical RX 9060 XT / gfx1200 measurement

The following measurement predates the current release identity. It guards the
HIP tuning choices, but it does not replace a current release-SHA hardware run.

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
