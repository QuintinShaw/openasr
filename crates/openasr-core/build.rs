use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let source_dir = manifest_dir.join("third_party/openasr-ggml");
    if !source_dir.join("CMakeLists.txt").is_file() {
        panic!(
            "openasr-ggml submodule is missing at {}; run `git submodule update --init --recursive`",
            source_dir.display()
        );
    }

    let target = env::var("TARGET").unwrap_or_default();
    let is_macos = target.contains("apple-darwin");
    let feat_cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    let feat_vulkan = env::var("CARGO_FEATURE_VULKAN").is_ok();
    let feat_hip = env::var("CARGO_FEATURE_HIP").is_ok();
    let feat_sycl = env::var("CARGO_FEATURE_SYCL").is_ok();
    let feat_openmp = env::var("CARGO_FEATURE_OPENMP").is_ok();
    let feat_native = env::var("CARGO_FEATURE_NATIVE").is_ok();
    let is_windows = target.contains("windows");
    // Windows arm64, always cross-compiled from an x86_64 host runner (there is
    // no arm64 GitHub-hosted Windows runner today). See the Ninja-generator
    // block below for why this needs the same generator override as the GPU
    // Windows legs.
    let is_windows_arm64 = is_windows && target.starts_with("aarch64");
    // The android triple (e.g. aarch64-linux-android) also contains "linux", so
    // it must be detected explicitly and BEFORE any `contains("linux")` check.
    let is_android = target.contains("android");
    // iOS device or simulator target (aarch64-apple-ios /
    // aarch64-apple-ios-sim). Deliberately distinct from `is_macos` (which only
    // matches "apple-darwin"): iOS builds have no bundled libomp (same as
    // macOS) and, in this phase, no Metal/Accelerate/BLAS -- those already
    // stay off because their gates key off `is_macos`.
    let is_ios = target.contains("apple-ios");
    // Rust's simulator targets are suffixed `-sim` (e.g. aarch64-apple-ios-sim);
    // the device target (aarch64-apple-ios) has no such suffix. Only this
    // distinguishes them -- both contain "apple-ios" -- and CMake needs to
    // point at the right SDK (iphonesimulator vs iphoneos) below, or the
    // resulting objects have the wrong Apple platform tag and fail to link
    // against the other slices' objects ("building for 'iOS-simulator', but
    // linking in object file ... built for 'iOS'").
    let is_ios_simulator = is_ios && target.ends_with("-sim");
    let host = env::var("HOST").unwrap_or_default();
    // Backend-DL plugin build for the CPU-only (no GPU feature) WINDOWS base:
    // ship ggml-base.dll + ggml.dll + ggml-cpu-<variant>.dll loaded via the ggml
    // registry, so GPU acceleration can be dropped in later as a downloadable
    // plugin DLL with automatic CPU fallback. This is the v1 product surface — a
    // Windows NSIS installer whose GPU packs are pulled on demand.
    //
    // Scoped to Windows on purpose. macOS ships a self-contained static binary
    // with Metal/Accelerate (no plugin story). Linux is the CI/CLI platform: a
    // static single binary is simpler to distribute and, crucially, avoids the
    // unverified Linux runtime plugin-discovery path (ggml dlopen of the
    // ggml-cpu-<variant>.so set, which `copy_runtime_dlls` does not stage). The
    // host-side registry refactor (init_by_type / ensure_backends_loaded) still
    // runs on the macOS+Linux static builds — init_by_type(CPU) resolves the
    // statically-registered backend and load_all is a harmless no-op there — so
    // the DL and static paths share one code path. GPU-feature builds keep the
    // static link regardless of OS (migrated to plugins in a later phase).
    // GGML_CPU_ALL_VARIANTS requires GGML_NATIVE=OFF (CMake FATAL_ERROR otherwise),
    // and a portable base must not bake the build host's ISA anyway.
    let use_backend_dl = is_windows && !feat_cuda && !feat_vulkan && !feat_hip;
    // GGML_CPU_ALL_VARIANTS compiles the multi-ISA CPU dispatch set (sse42/avx/
    // avx2/... on x86). ggml has NO Windows ARM entry in that variant table
    // (src/CMakeLists.txt only wires ARM ALL_VARIANTS for Linux/Android/Apple),
    // and on the windows-arm64 cross the host-arch fallback would otherwise emit
    // x86 variants whose x86-only GEMM/repack kernels have no ARM implementation
    // and fail the link with unresolved externals (ggml_gemm_q6_K_8x4_q8_K, ...).
    // So the arm64 cross builds a single ARM64 CPU backend instead (still a
    // GGML_BACKEND_DL plugin DLL, just not the multi-variant set).
    let ggml_cpu_all_variants = use_backend_dl && !is_windows_arm64;
    let ggml_native = resolve_ggml_native_enabled(
        feat_native,
        &target,
        &host,
        env::var("OPENASR_GGML_NATIVE").ok().as_deref(),
    ) && !use_backend_dl;
    let cuda_tuning = CudaTuning::from_env();
    let hip_tuning = HipTuning::from_env();

    // OpenMP CPU threading is on by default (~2x CPU). It links cleanly for the
    // CPU/CUDA/Vulkan builds (ggml-cpu is compiled by MSVC, whose `/openmp`
    // resolves against the system `vcomp`), but it is unsupported on three targets:
    //  - Windows HIP: HIP compiles the whole project with ROCm's clang, whose
    //    `-fopenmp` emits LLVM `__kmpc_*` calls, and ROCm for Windows ships no
    //    `libomp`, so `hip + openmp` fails to link (LNK2019 __kmpc_*). HIP runs
    //    decode on the GPU, so CPU OpenMP is not a meaningful loss.
    //  - macOS: Apple clang has no bundled `libomp` and the Mac path uses
    //    Metal/Accelerate; leave its build behavior unchanged.
    //  - android: bionic ships no `libgomp` and lacks `pthread_setaffinity` (the
    //    NDK's OpenMP is opt-in `libomp`, not the GOMP runtime ggml-cpu links); CPU
    //    threading on android comes from ggml's own thread pool instead.
    // We neutralize OpenMP for those rather than forcing the whole feature
    // opt-in. `OPENASR_GGML_OPENMP=0` force-disables everywhere.
    let openmp_requested = feat_openmp
        && !matches!(
            env::var("OPENASR_GGML_OPENMP").ok().as_deref(),
            Some("0" | "off" | "OFF" | "false" | "FALSE")
        );
    let openmp_unsupported_target = is_macos || is_ios || is_android || (feat_hip && is_windows);
    let effective_openmp = openmp_requested && !openmp_unsupported_target;
    if openmp_requested && !effective_openmp && feat_hip && is_windows {
        println!(
            "cargo:warning=OpenMP disabled for this build: AMD ROCm on Windows ships no libomp to \
             resolve clang's __kmpc_* symbols, so hip+openmp cannot link. The HIP binary runs \
             decode on the GPU (CPU OpenMP is not a meaningful loss); build the CPU/CUDA/Vulkan \
             provider for the OpenMP speedup."
        );
    }
    println!("cargo:rerun-if-env-changed=OPENASR_GGML_OPENMP");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let build_dir = out_dir.join("openasr-ggml-build");
    let lib_dir = build_dir.join("lib");
    fs::create_dir_all(&build_dir).expect("create openasr-ggml build dir");
    fs::create_dir_all(&lib_dir).expect("create openasr-ggml lib dir");

    // On Windows, cmake's Ninja generator picks the first C compiler on PATH. The
    // AMD ROCm SDK puts its clang.exe ahead of MSVC, so a non-HIP build would
    // accidentally use ROCm clang — the wrong ABI provider for the msvc Rust
    // target, and the reason OpenMP cannot link (clang emits LLVM __kmpc_* that
    // ROCm-Windows has no libomp for). Pin MSVC `cl` (+ its vcvars INCLUDE/LIB/PATH
    // env) for non-HIP Windows builds so ggml CPU/CUDA/Vulkan is MSVC-compiled and
    // OpenMP resolves against the system vcomp. HIP keeps ROCm clang (set below).
    let msvc_tool = (is_windows && !feat_hip)
        .then(|| cc::windows_registry::find_tool(&target, "cl.exe"))
        .flatten();
    if msvc_tool.is_some() {
        // A cmake cache configured with a different compiler (e.g. ROCm clang that
        // Ninja picked off PATH) cannot be reconfigured to cl; wipe it so the
        // pinned MSVC compiler takes effect on the next configure.
        let cache = build_dir.join("CMakeCache.txt");
        if let Ok(text) = fs::read_to_string(&cache) {
            let uses_cl = text.lines().any(|line| {
                line.starts_with("CMAKE_C_COMPILER:") && line.to_lowercase().contains("cl.exe")
            });
            if !uses_cl {
                let _ = fs::remove_dir_all(&build_dir);
                fs::create_dir_all(&lib_dir).expect("recreate openasr-ggml lib dir");
            }
        }
    }

    let hip_path = feat_hip.then(hip_toolkit_path).flatten();
    let cuda_path = feat_cuda.then(cuda_toolkit_path).flatten();
    let vulkan_sdk = feat_vulkan.then(vulkan_sdk_path).flatten();
    let windows_hip_shim = if feat_hip && is_windows {
        Some(prepare_windows_hip_sdk_shim(
            &target,
            hip_path
                .as_deref()
                .expect("HIP_PATH, ROCM_PATH, or ROCM_HOME must point to AMD HIP SDK"),
            &out_dir,
        ))
    } else {
        None
    };
    let mut cmake_prefix_paths = Vec::new();

    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(&source_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        // ggml's static archives are linked into a Rust binary that is PIE by
        // default on Linux. Host gcc/clang compile PIC anyway, but the ROCm
        // and CUDA device-host compilers do not (amdclang++ emits non-PIC
        // .eh_frame relocations that fail the final rust-lld link with
        // "relocation R_X86_64_32 cannot be used against local symbol").
        // Forcing PIC on is correct everywhere and required there.
        .arg("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")
        .arg(cmake_flag("BUILD_SHARED_LIBS", use_backend_dl))
        .arg(cmake_flag("GGML_BACKEND_DL", use_backend_dl))
        .arg(cmake_flag("GGML_CPU_ALL_VARIANTS", ggml_cpu_all_variants))
        .arg("-DGGML_BUILD_TESTS=OFF")
        .arg("-DGGML_BUILD_EXAMPLES=OFF")
        .arg(cmake_flag("GGML_NATIVE", ggml_native))
        .arg(cmake_flag("GGML_OPENMP", effective_openmp))
        .arg(cmake_flag("GGML_CUDA", feat_cuda))
        .arg(cmake_flag("GGML_VULKAN", feat_vulkan))
        .arg(cmake_flag("GGML_HIP", feat_hip))
        .arg(cmake_flag("GGML_SYCL", feat_sycl))
        .arg(cmake_flag(
            "GGML_ACCELERATE",
            is_macos && !feat_cuda && !feat_vulkan,
        ))
        .arg(cmake_flag(
            "GGML_BLAS",
            is_macos && !feat_cuda && !feat_vulkan,
        ))
        .arg(format!(
            "-DGGML_BLAS_VENDOR={}",
            if is_macos { "Apple" } else { "Generic" }
        ))
        .arg(cmake_flag(
            "GGML_METAL",
            is_macos && !feat_cuda && !feat_vulkan,
        ))
        .arg(cmake_flag(
            "GGML_METAL_EMBED_LIBRARY",
            is_macos && !feat_cuda && !feat_vulkan,
        ))
        .arg(format!(
            "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}",
            cmake_path(&lib_dir)
        ))
        .arg(format!(
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}",
            cmake_path(&lib_dir)
        ))
        .arg(format!(
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={}",
            cmake_path(&build_dir.join("bin"))
        ));
    if feat_vulkan
        && !is_android
        && let Some(path) = vulkan_sdk.as_deref()
    {
        cmake_prefix_paths.push(path.to_path_buf());
        if path.join("Include").is_dir() {
            configure.arg(format!(
                "-DVulkan_INCLUDE_DIR={}",
                cmake_path(&path.join("Include"))
            ));
        }
        let vulkan_lib = if is_windows {
            vec![path.join("Lib/vulkan-1.lib")]
        } else {
            vec![
                path.join("lib/libvulkan.so"),
                path.join("lib/libvulkan.dylib"),
            ]
        }
        .into_iter()
        .find(|candidate| candidate.is_file());
        if let Some(vulkan_lib) = vulkan_lib {
            configure.arg(format!("-DVulkan_LIBRARY={}", cmake_path(&vulkan_lib)));
        }
        let glslc = [
            path.join("Bin/glslc.exe"),
            path.join("bin/glslc"),
            path.join("bin/glslc.exe"),
        ]
        .into_iter()
        .find(|candidate| candidate.is_file());
        if let Some(glslc) = glslc {
            configure.arg(format!("-DVulkan_GLSLC_EXECUTABLE={}", cmake_path(&glslc)));
        }
        let spirv_headers_dir = [
            path.join("Lib/cmake/SPIRV-Headers"),
            path.join("lib/cmake/SPIRV-Headers"),
        ]
        .into_iter()
        .find(|candidate| candidate.is_dir());
        if let Some(spirv_headers_dir) = spirv_headers_dir {
            configure.arg(format!(
                "-DSPIRV-Headers_DIR={}",
                cmake_path(&spirv_headers_dir)
            ));
        }
    }
    // CUDA joins HIP/Vulkan on the Ninja generator because the default Visual
    // Studio generator resolves enable_language(CUDA) through NVIDIA's VS
    // build customizations, which trail new VS majors (VS 2026 has no CUDA
    // toolset yet -> "No CUDA toolset found"). Ninja drives nvcc directly
    // with the MSVC host compiler pinned below instead.
    //
    // The windows arm64 cross build (host x86_64, target aarch64-pc-windows-msvc)
    // joins them so the target ISA is driven purely by the compiler + explicit
    // CMAKE_SYSTEM_PROCESSOR (set in the is_windows_arm64 block below) rather
    // than the default Visual Studio generator's multi-arch project shape, which
    // configures for the HOST platform (x64) and would fight the arm64 cross
    // toolchain. See that block for the ARM-arch and clang-cl requirements.
    if (feat_hip || feat_vulkan || feat_cuda || is_windows_arm64) && is_windows {
        configure.arg("-G").arg("Ninja");
    }
    if is_windows_arm64 {
        // Cross-compile ggml for ARM64 Windows from an x86_64 host.
        //
        // CMAKE_SYSTEM_NAME=Windows puts CMake into explicit cross-compile mode;
        // CMAKE_SYSTEM_PROCESSOR=ARM64 makes ggml's ggml_get_system_arch()
        // resolve GGML_SYSTEM_ARCH=ARM. Under the Ninja generator there is no
        // CMAKE_GENERATOR_PLATFORM signal, so ggml would otherwise fall back to
        // the host processor (AMD64) and wrongly select the x86 CPU backend --
        // the source of the "unresolved external ggml_gemm_q6_K_8x4_q8_K" link
        // failures (x86-only kernels compiled for an ARM target).
        //
        // ggml's ARM CPU backend refuses MSVC cl outright
        // (src/ggml-cpu/CMakeLists.txt: "MSVC is not supported for ARM, use
        // clang"), so the arm64 cross must be compiled with clang-cl. clang-cl
        // still consumes the MSVC ARM64 headers/libs (the INCLUDE/LIB/PATH env
        // that msvc_tool.env() exports below); CMAKE_<LANG>_COMPILER_TARGET makes
        // CMake pass `--target=arm64-pc-windows-msvc` so clang-cl emits ARM64
        // objects. GGML_NATIVE is already OFF for a cross build, and there are no
        // check_cxx_source_runs() probes on this path (the ARM feature detection
        // uses compile-only checks), so no arm64 test binary is ever executed on
        // the x64 host.
        configure
            .arg("-DCMAKE_SYSTEM_NAME=Windows")
            .arg("-DCMAKE_SYSTEM_PROCESSOR=ARM64")
            .arg("-DCMAKE_C_COMPILER=clang-cl")
            .arg("-DCMAKE_CXX_COMPILER=clang-cl")
            .arg("-DCMAKE_C_COMPILER_TARGET=arm64-pc-windows-msvc")
            .arg("-DCMAKE_CXX_COMPILER_TARGET=arm64-pc-windows-msvc");
    }
    if is_macos {
        configure.arg(format!(
            "-DCMAKE_OSX_DEPLOYMENT_TARGET={}",
            macos_deployment_target()
        ));
    }
    if is_ios {
        // Cross-compile ggml for the iOS device or simulator ABI. Unlike
        // Android this needs no separate CMake toolchain file: CMake's
        // built-in Apple platform support drives Clang cross-compilation
        // straight from CMAKE_OSX_SYSROOT (the SDK Clang targets) +
        // CMAKE_OSX_ARCHITECTURES (only arm64 is wired up -- no armv7/i386
        // device, no x86_64 simulator). Phase 1 is a CPU-only compile gate:
        // Metal/Accelerate/BLAS already stay off here because their
        // cmake_flag(...) calls above key off `is_macos`, which is `false` for
        // the "apple-ios" target triple.
        let sysroot = if is_ios_simulator {
            "iphonesimulator"
        } else {
            "iphoneos"
        };
        configure
            .arg("-DCMAKE_SYSTEM_NAME=iOS")
            .arg(format!("-DCMAKE_OSX_SYSROOT={sysroot}"))
            .arg("-DCMAKE_OSX_ARCHITECTURES=arm64")
            .arg(format!(
                "-DCMAKE_OSX_DEPLOYMENT_TARGET={}",
                ios_deployment_target()
            ));
    }
    if is_android {
        // Cross-compile ggml with the NDK's CMake toolchain file. build.rs shells
        // out to cmake directly (no compiler inheritance), so without this cmake
        // would configure for the host and the objects would fail the rustc link.
        // The toolchain file sets CMAKE_SYSTEM_NAME=Android + the NDK clang/sysroot
        // from ANDROID_ABI/ANDROID_PLATFORM. GGML_NATIVE already resolves OFF for a
        // cross build (host != target), giving portable arm64 codegen.
        let ndk = android_ndk_path().unwrap_or_else(|| {
            panic!(
                "aarch64-linux-android build requires the Android NDK: set ANDROID_NDK_HOME \
                 (or ANDROID_NDK_ROOT / NDK_HOME) to the NDK root — the directory containing \
                 build/cmake/android.toolchain.cmake"
            )
        });
        // Vulkan needs a min API of 28: ggml-vulkan directly links the Vulkan 1.1
        // core symbol vkGetPhysicalDeviceFeatures2, which the NDK libvulkan.so only
        // exports from API 28+. CPU keeps the lower default for broader device reach.
        let android_api = android_api_level(feat_vulkan);
        let abi = android_abi();
        // Only arm64-v8a is wired end-to-end: the cargo wrapper's rustc target, the
        // C++/loader link lines, and the sysroot Vulkan loader path are all aarch64.
        // Reject any other OPENASR_ANDROID_ABI loudly here rather than configuring ggml
        // for an arch the rustc link step (always aarch64-linux-android) won't match.
        assert!(
            abi == "arm64-v8a",
            "OPENASR_ANDROID_ABI={abi} is not supported — only arm64-v8a is wired \
             end-to-end for the android cross build"
        );
        configure
            .arg(format!(
                "-DCMAKE_TOOLCHAIN_FILE={}",
                cmake_path(&ndk.join("build/cmake/android.toolchain.cmake"))
            ))
            .arg(format!("-DANDROID_ABI={abi}"))
            .arg(format!("-DANDROID_PLATFORM=android-{android_api}"));
        // A cross build defaults GGML_NATIVE off → a portable armv8-a baseline that
        // disables dotprod/fp16, the key int8/fp16 matmul accelerators for quantized
        // ASR on mobile. Target armv8.2-a+dotprod+fp16 (Cortex-A55/A75+, ~all Android
        // devices since 2018) for a large speedup; override via OPENASR_ANDROID_ARM_ARCH
        // (e.g. add +i8mm for armv8.6 flagships, or "armv8-a" for the broadest floor).
        configure.arg(format!("-DGGML_CPU_ARM_ARCH={}", android_arm_arch()));
        if feat_vulkan {
            // ggml-vulkan needs HOST Vulkan-Headers (incl. vulkan.hpp), SPIRV-Headers,
            // and glslc at build time (all arch-neutral); the NDK sysroot supplies the
            // libvulkan.so LOADER for the final link. The NDK sysroot only ships an old
            // vulkan.h without the C++ bindings, so point cmake at the host headers and
            // the sysroot loader explicitly. ggml-vulkan builds the vulkan-shaders-gen
            // tool for the HOST itself (its CMake detects the host compiler under a cross
            // build), so only the loader is target-specific here.
            let inc = android_vulkan_include_dir().unwrap_or_else(|| {
                panic!(
                    "android --features vulkan needs Vulkan-Headers with vulkan.hpp on the \
                     host (arch-neutral): install them (`brew install vulkan-headers`) or set \
                     VULKAN_SDK"
                )
            });
            let glslc = host_glslc().unwrap_or_else(|| {
                panic!(
                    "android --features vulkan needs a host glslc: install it \
                     (`brew install shaderc`) or set VULKAN_SDK with bin/glslc"
                )
            });
            configure
                .arg(format!("-DVulkan_INCLUDE_DIR={}", cmake_path(&inc)))
                .arg(format!("-DVulkan_GLSLC_EXECUTABLE={}", cmake_path(&glslc)));
            if let Some(loader) = android_sysroot_vulkan_lib(&ndk, android_api) {
                configure.arg(format!("-DVulkan_LIBRARY={}", cmake_path(&loader)));
            }
            if let Some(spirv_dir) = spirv_headers_config_dir() {
                configure.arg(format!("-DSPIRV-Headers_DIR={}", cmake_path(&spirv_dir)));
            }
        }
    }
    if feat_hip {
        configure.arg("-DCMAKE_HIP_PLATFORM=amd");
        if let Some(path) = hip_path.as_deref() {
            cmake_prefix_paths.push(path.to_path_buf());
            if is_windows && let Some(clang) = hip_sdk_clang_path(path) {
                let clang = cmake_path(&clang);
                configure
                    .arg(format!("-DCMAKE_C_COMPILER={clang}"))
                    .arg(format!("-DCMAKE_CXX_COMPILER={clang}"));
            }
        }
        if let Some(path) = windows_hip_shim.as_deref() {
            cmake_prefix_paths.push(path.to_path_buf());
        }
        let targets = hip_gpu_targets();
        configure
            .arg(format!("-DGPU_TARGETS={targets}"))
            .arg(format!("-DAMDGPU_TARGETS={targets}"))
            .arg(cmake_flag("GGML_HIP_GRAPHS", hip_tuning.graphs))
            .arg(cmake_flag("GGML_CUDA_FA", hip_tuning.flash_attention))
            .arg(cmake_flag(
                "GGML_CUDA_FA_ALL_QUANTS",
                hip_tuning.flash_attention_all_quants,
            ))
            .arg(cmake_flag(
                "GGML_HIP_ROCWMMA_FATTN",
                hip_tuning.rocwmma_flash_attention,
            ))
            .arg(cmake_flag("GGML_HIP_MMQ_MFMA", hip_tuning.mmq_mfma))
            .arg(cmake_flag("GGML_CUDA_FORCE_MMQ", hip_tuning.force_mmq))
            .arg(cmake_flag("GGML_HIP_NO_VMM", hip_tuning.no_vmm))
            .arg(cmake_flag(
                "GGML_HIP_EXPORT_METRICS",
                hip_tuning.export_metrics,
            ));
    }
    if feat_cuda {
        if let Some(path) = cuda_path.as_deref() {
            let cuda_root = cmake_path(path);
            let nvcc = path.join("bin/nvcc.exe");
            configure
                .env("CUDA_PATH", path)
                .env("CUDA_HOME", path)
                .env("CudaToolkitDir", path)
                .arg(format!("-DCUDAToolkit_ROOT={cuda_root}"))
                .arg(format!("-DCudaToolkitDir={cuda_root}"));
            if nvcc.is_file() {
                configure.arg(format!("-DCMAKE_CUDA_COMPILER={}", cmake_path(&nvcc)));
            }
        }
        // Under Ninja nothing tells nvcc which host compiler to use (the VS
        // generator used to imply it), so pin it to the same MSVC cl the rest
        // of the build is compiled with; otherwise nvcc takes whatever cl.exe
        // (or clang) PATH happens to expose first.
        if is_windows && let Some(tool) = msvc_tool.as_ref() {
            configure.arg(format!(
                "-DCMAKE_CUDA_HOST_COMPILER={}",
                cmake_path(tool.path())
            ));
        }
        configure
            .arg(format!("-DCMAKE_CUDA_ARCHITECTURES={}", cuda_gpu_targets()))
            .arg(cmake_flag("GGML_CUDA_FA", cuda_tuning.flash_attention))
            .arg(cmake_flag(
                "GGML_CUDA_FA_ALL_QUANTS",
                cuda_tuning.flash_attention_all_quants,
            ))
            .arg(cmake_flag("GGML_CUDA_FORCE_MMQ", cuda_tuning.force_mmq))
            // Single-GPU ASR inference does not use NVIDIA's multi-GPU collective
            // comm. ggml defaults GGML_CUDA_NCCL=ON and, when NCCL is present
            // (e.g. CUDA images that ship libnccl), compiles a comm path that
            // references ncclAllReduce/ncclCommInitAll/… into the static
            // ggml-cuda lib; that PRIVATE link does not propagate to the final
            // Rust link, so the binary fails with undefined NCCL symbols. Disable
            // it: no multi-GPU dependency, smaller binary, faster build.
            .arg(cmake_flag("GGML_CUDA_NCCL", false));
    }
    if !cmake_prefix_paths.is_empty() {
        configure.arg(format!(
            "-DCMAKE_PREFIX_PATH={}",
            cmake_list_path(&cmake_prefix_paths)
        ));
    }
    if feat_sycl {
        configure
            .arg("-DCMAKE_C_COMPILER=icx")
            .arg("-DCMAKE_CXX_COMPILER=icpx");
    }
    if let Some(tool) = msvc_tool.as_ref() {
        // The windows-arm64 cross pins clang-cl (+ its --target) above because
        // ggml's ARM CPU backend rejects MSVC cl; here it only needs the arm64
        // MSVC INCLUDE/LIB/PATH env that find_tool resolved. Every other Windows
        // leg compiles ggml with cl directly.
        if !is_windows_arm64 {
            let cl = cmake_path(tool.path());
            configure
                .arg(format!("-DCMAKE_C_COMPILER={cl}"))
                .arg(format!("-DCMAKE_CXX_COMPILER={cl}"));
        }
        for (key, val) in tool.env() {
            configure.env(key, val);
        }
    }
    run(&mut configure);

    let mut build = Command::new("cmake");
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--config")
        .arg("Release")
        .arg("--target")
        .arg("ggml")
        .arg("-j")
        .arg(cmake_build_jobs());
    if feat_cuda && let Some(path) = cuda_path.as_deref() {
        build
            .env("CUDA_PATH", path)
            .env("CUDA_HOME", path)
            .env("CudaToolkitDir", path);
    }
    if let Some(tool) = msvc_tool.as_ref() {
        for (key, val) in tool.env() {
            build.env(key, val);
        }
    }
    run(&mut build);

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    if is_windows {
        println!(
            "cargo:rustc-link-search=native={}",
            lib_dir.join("Release").display()
        );
    }
    if use_backend_dl {
        // Backend-DL: the core is two shared libs (ggml-base = runtime, ggml =
        // registry/loader). The CPU compute backend and every GPU backend are
        // runtime-loaded plugin DLLs, never linked here. The host calls
        // ggml_init/ggml_mul_mat (ggml-base) and the registry APIs incl.
        // ggml_backend_load_all_from_path / init_by_type (ggml), so both import
        // libs are required. The GPU/macOS/static-C++ blocks below are no-ops
        // here (their conditions are false under use_backend_dl); flow continues
        // to the rerun-if-changed directives.
        println!("cargo:rustc-link-lib=dylib=ggml-base");
        println!("cargo:rustc-link-lib=dylib=ggml");
        copy_runtime_dlls(&build_dir.join("bin"), &out_dir);
    } else {
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-cpu");
        println!("cargo:rustc-link-lib=static=ggml-base");
    }

    if is_macos && !feat_cuda && !feat_vulkan {
        println!("cargo:rustc-link-lib=static=ggml-metal");
        println!("cargo:rustc-link-lib=static=ggml-blas");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
    }

    if feat_cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        if let Some(cuda_path) = cuda_path.as_deref() {
            println!(
                "cargo:rustc-link-search=native={}",
                cuda_path.join("lib64").display()
            );
            println!(
                "cargo:rustc-link-search=native={}",
                cuda_path.join("lib").display()
            );
            println!(
                "cargo:rustc-link-search=native={}",
                cuda_path.join("lib/x64").display()
            );
            // libcuda.so is the DRIVER library: a real one only exists on a
            // machine with an NVIDIA driver. Toolkit installs provide a link
            // stub under lib64/stubs (cuda-driver-dev on Linux) precisely so
            // driver-linking binaries can be built on driver-less hosts (CI).
            // Listed last: a real driver library earlier on the search path
            // still wins.
            println!(
                "cargo:rustc-link-search=native={}",
                cuda_path.join("lib64/stubs").display()
            );
            println!(
                "cargo:rustc-link-search=native={}",
                cuda_path.join("lib/stubs").display()
            );
        }
    }

    if feat_vulkan {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
        if is_windows {
            println!("cargo:rustc-link-lib=dylib=vulkan-1");
        } else {
            println!("cargo:rustc-link-lib=dylib=vulkan");
        }
        // On android the libvulkan.so loader is in the NDK sysroot, already on the
        // NDK linker's search path; the desktop VULKAN_SDK lib dirs do not apply.
        if !is_android && let Some(path) = vulkan_sdk.as_deref() {
            println!(
                "cargo:rustc-link-search=native={}",
                path.join("Lib").display()
            );
            println!(
                "cargo:rustc-link-search=native={}",
                path.join("lib").display()
            );
        }
    }

    if feat_hip {
        println!("cargo:rustc-link-lib=static=ggml-hip");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        if is_windows {
            println!("cargo:rustc-link-lib=dylib=libhipblas");
        } else {
            println!("cargo:rustc-link-lib=dylib=hipblas");
        }
        println!("cargo:rustc-link-lib=dylib=rocblas");
        if let Some(path) = windows_hip_shim.as_deref() {
            println!(
                "cargo:rustc-link-search=native={}",
                path.join("lib").display()
            );
        }
        if let Some(path) = hip_path.as_deref() {
            println!(
                "cargo:rustc-link-search=native={}",
                path.join("lib").display()
            );
            println!(
                "cargo:rustc-link-search=native={}",
                path.join("lib64").display()
            );
        }
    }

    if feat_sycl {
        println!("cargo:rustc-link-lib=static=ggml-sycl");
    }

    if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if is_android {
        // The Android NDK ships LLVM libc++ (linked as `c++_shared`), not GNU
        // libstdc++, and has no libgomp. ggml is built with the NDK toolchain
        // (ANDROID_STL defaults to c++_shared) and OpenMP is disabled for android,
        // so link the shared libc++ runtime and emit no gomp. This MUST be checked
        // before the `linux` arm because aarch64-linux-android also contains "linux".
        println!("cargo:rustc-link-lib=dylib=c++_shared");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        // The static ggml-cpu.a references OpenMP runtime symbols (GOMP_*/omp_*)
        // on Linux in this ggml revision even when configured GGML_OPENMP=OFF, so
        // link libgomp (ships with gcc; `--as-needed` drops it if unreferenced).
        println!("cargo:rustc-link-lib=dylib=gomp");
    }

    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("build.rs").display()
    );
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    println!("cargo:rerun-if-env-changed=OPENASR_GGML_BUILD_JOBS");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("src/CMakeLists.txt").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("include/ggml.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("include/ggml-backend.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("include/ggml-cpu.h").display()
    );
    // The GGML_BACKEND_DL loader shim is OpenASR-authored and decides how plugin
    // DLLs (and their co-located satellite runtime DLLs) are opened on each OS, so
    // edits to it must trigger a ggml rebuild. It was previously untracked, so
    // changing the loader silently kept the stale compiled shim.
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("src/ggml-backend-dl.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("src/ggml-backend-dl.h").display()
    );
    // Watch the linked GPU backend's source tree so editing a backend kernel
    // actually triggers a rebuild. HIP is built from the ggml-cuda sources
    // (hipified), so it watches both. Gated by feature to avoid spurious
    // rebuilds on the CPU/Metal-only build.
    if feat_hip || feat_cuda {
        println!(
            "cargo:rerun-if-changed={}",
            source_dir.join("src/ggml-cuda").display()
        );
    }
    if feat_hip {
        println!(
            "cargo:rerun-if-changed={}",
            source_dir.join("src/ggml-hip").display()
        );
    }
    if feat_vulkan {
        println!(
            "cargo:rerun-if-changed={}",
            source_dir.join("src/ggml-vulkan").display()
        );
    }
    if feat_sycl {
        println!(
            "cargo:rerun-if-changed={}",
            source_dir.join("src/ggml-sycl").display()
        );
    }
    println!("cargo:rerun-if-env-changed=OPENASR_HIP_GPU_TARGETS");
    println!("cargo:rerun-if-env-changed=OPENASR_HIP_GRAPHS");
    println!("cargo:rerun-if-env-changed=OPENASR_HIP_FLASH_ATTENTION");
    println!("cargo:rerun-if-env-changed=OPENASR_HIP_FA_ALL_QUANTS");
    println!("cargo:rerun-if-env-changed=OPENASR_HIP_ROCWMMA_FATTN");
    println!("cargo:rerun-if-env-changed=OPENASR_HIP_MMQ_MFMA");
    println!("cargo:rerun-if-env-changed=OPENASR_HIP_FORCE_MMQ");
    println!("cargo:rerun-if-env-changed=OPENASR_HIP_NO_VMM");
    println!("cargo:rerun-if-env-changed=OPENASR_HIP_EXPORT_METRICS");
    println!("cargo:rerun-if-env-changed=OPENASR_CUDA_GPU_TARGETS");
    println!("cargo:rerun-if-env-changed=OPENASR_CUDA_FLASH_ATTENTION");
    println!("cargo:rerun-if-env-changed=OPENASR_CUDA_FA_ALL_QUANTS");
    println!("cargo:rerun-if-env-changed=OPENASR_CUDA_FORCE_MMQ");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=OPENASR_GGML_NATIVE");
    println!("cargo:rerun-if-env-changed=OPENASR_GGML_BUILD_JOBS");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=VK_SDK_PATH");
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_HOME");
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_ROOT");
    println!("cargo:rerun-if-env-changed=NDK_HOME");
    println!("cargo:rerun-if-env-changed=OPENASR_ANDROID_ABI");
    println!("cargo:rerun-if-env-changed=OPENASR_ANDROID_API");
    println!("cargo:rerun-if-env-changed=OPENASR_ANDROID_ARM_ARCH");
    println!("cargo:rerun-if-env-changed=OPENASR_GLSLC");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_HOME");
    println!(
        "cargo:rustc-env=OPENASR_GGML_NATIVE_ENABLED={}",
        if ggml_native { "1" } else { "0" }
    );
    println!(
        "cargo:rustc-env=OPENASR_HIP_TUNING={}",
        if feat_hip {
            hip_tuning.summary()
        } else {
            "disabled".to_string()
        }
    );
    println!(
        "cargo:rustc-env=OPENASR_CUDA_TUNING={}",
        if feat_cuda {
            cuda_tuning.summary()
        } else {
            "disabled".to_string()
        }
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("include/gguf.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir.join("src/gguf.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir
            .join("src/ggml-metal/ggml-metal-device.cpp")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir
            .join("src/ggml-metal/ggml-metal-device.m")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir
            .join("src/ggml-metal/ggml-metal-device.h")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir
            .join("src/ggml-metal/ggml-metal-context.m")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir
            .join("src/ggml-metal/ggml-metal-ops.cpp")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        source_dir
            .join("src/ggml-metal/ggml-metal-impl.h")
            .display()
    );
    if is_macos {
        println!(
            "cargo:rerun-if-changed={}",
            source_dir.join("include/ggml-metal.h").display()
        );
    }
}

fn cmake_flag(name: &str, enabled: bool) -> String {
    format!("-D{}={}", name, if enabled { "ON" } else { "OFF" })
}

/// Copy the backend-DL runtime DLLs (ggml-base / ggml / ggml-cpu-<variant>) next
/// to where cargo runs binaries and tests, so they resolve at load time without
/// a PATH dance. Windows searches the executable's own directory first; cargo
/// emits final bins to `target/<profile>/` and test bins to
/// `target/<profile>/deps/`, so seed both. Windows-only by construction: under
/// BUILD_SHARED_LIBS the Windows runtime DLLs land in the cmake RUNTIME dir
/// (`bin`), while ELF/Mach-O shared objects go to the LIBRARY dir and resolve via
/// rpath instead — Linux DL packaging is handled separately.
fn copy_runtime_dlls(bin_dir: &Path, out_dir: &Path) {
    // out_dir = target/<profile>/build/<pkg>-<hash>/out -> nth(3) = target/<profile>
    let Some(profile_dir) = out_dir.ancestors().nth(3) else {
        return;
    };
    // Single-config (Ninja) emits the runtime DLLs into bin/; a multi-config
    // generator nests them under bin/Release/. Gather from both.
    let mut dlls: Vec<PathBuf> = Vec::new();
    for dir in [bin_dir.to_path_buf(), bin_dir.join("Release")] {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        dlls.extend(entries.flatten().map(|entry| entry.path()).filter(|path| {
            path.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("dll"))
        }));
    }
    for dest in [profile_dir.to_path_buf(), profile_dir.join("deps")] {
        let _ = fs::create_dir_all(&dest);
        for dll in &dlls {
            if let Some(name) = dll.file_name() {
                let _ = fs::copy(dll, dest.join(name));
            }
        }
    }
}

/// Decide whether to pass `-DGGML_NATIVE=ON` (`-march=native`-style host CPU
/// tuning) to the ggml cmake build.
///
/// Precedence: explicit `--features native` wins, then the `OPENASR_GGML_NATIVE`
/// env override, then an implicit default. The implicit default auto-enables
/// native tuning only for a host==target x86 build — i.e. building to run on the
/// same machine, where tuning is a free win and the binary is not shipped.
///
/// IMPORTANT for distribution: a host==target x86 build is also what release CI
/// does, so any pipeline that BUILDS-TO-DISTRIBUTE x86 binaries must set
/// `OPENASR_GGML_NATIVE=0` (see `.github/workflows/release-binaries.yml`).
/// Native-tuned binaries can SIGILL on older end-user CPUs.
fn resolve_ggml_native_enabled(
    feature_native: bool,
    target: &str,
    host: &str,
    env_value: Option<&str>,
) -> bool {
    if feature_native {
        return true;
    }
    if let Some(enabled) = parse_bool_env(env_value) {
        return enabled;
    }
    host == target && target_is_x86(target)
}

fn target_is_x86(target: &str) -> bool {
    target.starts_with("x86_64-") || target.starts_with("i686-") || target.starts_with("i586-")
}

fn parse_bool_env(raw: Option<&str>) -> Option<bool> {
    let value = raw?.trim();
    if value.is_empty() {
        return None;
    }
    if ["1", "true", "yes", "on"]
        .iter()
        .any(|enabled| value.eq_ignore_ascii_case(enabled))
    {
        return Some(true);
    }
    if ["0", "false", "no", "off"]
        .iter()
        .any(|disabled| value.eq_ignore_ascii_case(disabled))
    {
        return Some(false);
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CudaTuning {
    flash_attention: bool,
    flash_attention_all_quants: bool,
    force_mmq: bool,
}

impl CudaTuning {
    fn from_env() -> Self {
        Self {
            flash_attention: env_bool_or("OPENASR_CUDA_FLASH_ATTENTION", true),
            flash_attention_all_quants: env_bool_or("OPENASR_CUDA_FA_ALL_QUANTS", false),
            force_mmq: env_bool_or("OPENASR_CUDA_FORCE_MMQ", false),
        }
    }

    #[cfg(test)]
    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Self {
        Self {
            flash_attention: bool_lookup_or(&mut lookup, "OPENASR_CUDA_FLASH_ATTENTION", true),
            flash_attention_all_quants: bool_lookup_or(
                &mut lookup,
                "OPENASR_CUDA_FA_ALL_QUANTS",
                false,
            ),
            force_mmq: bool_lookup_or(&mut lookup, "OPENASR_CUDA_FORCE_MMQ", false),
        }
    }

    fn summary(self) -> String {
        format!(
            "fa={},fa_all_quants={},force_mmq={}",
            on_off(self.flash_attention),
            on_off(self.flash_attention_all_quants),
            on_off(self.force_mmq),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HipTuning {
    graphs: bool,
    flash_attention: bool,
    flash_attention_all_quants: bool,
    rocwmma_flash_attention: bool,
    mmq_mfma: bool,
    force_mmq: bool,
    no_vmm: bool,
    export_metrics: bool,
}

impl HipTuning {
    fn from_env() -> Self {
        Self {
            graphs: env_bool_or("OPENASR_HIP_GRAPHS", true),
            flash_attention: env_bool_or("OPENASR_HIP_FLASH_ATTENTION", true),
            flash_attention_all_quants: env_bool_or("OPENASR_HIP_FA_ALL_QUANTS", false),
            rocwmma_flash_attention: env_bool_or("OPENASR_HIP_ROCWMMA_FATTN", false),
            mmq_mfma: env_bool_or("OPENASR_HIP_MMQ_MFMA", true),
            force_mmq: env_bool_or("OPENASR_HIP_FORCE_MMQ", false),
            no_vmm: env_bool_or("OPENASR_HIP_NO_VMM", true),
            export_metrics: env_bool_or("OPENASR_HIP_EXPORT_METRICS", false),
        }
    }

    #[cfg(test)]
    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Self {
        Self {
            graphs: bool_lookup_or(&mut lookup, "OPENASR_HIP_GRAPHS", true),
            flash_attention: bool_lookup_or(&mut lookup, "OPENASR_HIP_FLASH_ATTENTION", true),
            flash_attention_all_quants: bool_lookup_or(
                &mut lookup,
                "OPENASR_HIP_FA_ALL_QUANTS",
                false,
            ),
            rocwmma_flash_attention: bool_lookup_or(
                &mut lookup,
                "OPENASR_HIP_ROCWMMA_FATTN",
                false,
            ),
            mmq_mfma: bool_lookup_or(&mut lookup, "OPENASR_HIP_MMQ_MFMA", true),
            force_mmq: bool_lookup_or(&mut lookup, "OPENASR_HIP_FORCE_MMQ", false),
            no_vmm: bool_lookup_or(&mut lookup, "OPENASR_HIP_NO_VMM", true),
            export_metrics: bool_lookup_or(&mut lookup, "OPENASR_HIP_EXPORT_METRICS", false),
        }
    }

    fn summary(self) -> String {
        format!(
            "graphs={},fa={},fa_all_quants={},rocwmma_fattn={},mmq_mfma={},force_mmq={},no_vmm={},export_metrics={}",
            on_off(self.graphs),
            on_off(self.flash_attention),
            on_off(self.flash_attention_all_quants),
            on_off(self.rocwmma_flash_attention),
            on_off(self.mmq_mfma),
            on_off(self.force_mmq),
            on_off(self.no_vmm),
            on_off(self.export_metrics),
        )
    }
}

fn env_bool_or(key: &str, default: bool) -> bool {
    parse_bool_env(env::var(key).ok().as_deref()).unwrap_or(default)
}

#[cfg(test)]
fn bool_lookup_or(
    lookup: &mut impl FnMut(&str) -> Option<String>,
    key: &str,
    default: bool,
) -> bool {
    parse_bool_env(lookup(key).as_deref()).unwrap_or(default)
}

fn on_off(enabled: bool) -> &'static str {
    if enabled { "on" } else { "off" }
}

fn run(command: &mut Command) {
    let program = command.get_program().to_string_lossy().into_owned();
    let args = command
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(" ");
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to run {program} {args}: {error}"));
    if !status.success() {
        panic!("{program} {args} failed with status {status}");
    }
}

fn cmake_path(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
}

fn cmake_list_path(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| cmake_path(path))
        .collect::<Vec<_>>()
        .join(";")
}

fn vulkan_sdk_path() -> Option<PathBuf> {
    env::var_os("VULKAN_SDK")
        .or_else(|| env::var_os("VK_SDK_PATH"))
        .map(PathBuf::from)
}

/// Resolve the Android NDK root for a cross build. Honors the standard NDK env
/// vars (validating each points at a real NDK), then falls back to the Homebrew
/// `android-ndk` cask location so a Mac dev build works with zero env setup.
fn android_ndk_path() -> Option<PathBuf> {
    env::var_os("ANDROID_NDK_HOME")
        .or_else(|| env::var_os("ANDROID_NDK_ROOT"))
        .or_else(|| env::var_os("NDK_HOME"))
        .map(PathBuf::from)
        .filter(|path| is_android_ndk_root(path))
        .or_else(default_homebrew_android_ndk_path)
}

fn default_homebrew_android_ndk_path() -> Option<PathBuf> {
    let path = PathBuf::from("/opt/homebrew/share/android-ndk");
    is_android_ndk_root(&path).then_some(path)
}

fn is_android_ndk_root(path: &Path) -> bool {
    path.join("build/cmake/android.toolchain.cmake").is_file()
}

/// Target ABI for the android cross build (default arm64; override via env).
fn android_abi() -> String {
    non_empty_env("OPENASR_ANDROID_ABI").unwrap_or_else(|| "arm64-v8a".to_string())
}

/// ARM `-march` passed as GGML_CPU_ARM_ARCH for the android CPU kernels. Default
/// armv8.2-a+dotprod+fp16 (Cortex-A55/A75+, ~all Android since 2018) enables the
/// int8/fp16 matmul accelerators a portable cross build would otherwise disable.
/// Override via OPENASR_ANDROID_ARM_ARCH (add +i8mm for armv8.6, or "armv8-a" floor).
fn android_arm_arch() -> String {
    non_empty_env("OPENASR_ANDROID_ARM_ARCH")
        .unwrap_or_else(|| "armv8.2-a+dotprod+fp16".to_string())
}

/// Min android API level for the cross build. Defaults to 24 (Android 7) for broad
/// device reach; Vulkan is bumped to >=28 (Android 9) because ggml-vulkan directly
/// links the Vulkan 1.1 core symbol vkGetPhysicalDeviceFeatures2, which the NDK
/// libvulkan.so only exports from API 28. Overridable via OPENASR_ANDROID_API.
fn android_api_level(vulkan: bool) -> u32 {
    let requested = non_empty_env("OPENASR_ANDROID_API")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(24);
    if vulkan { requested.max(28) } else { requested }
}

fn non_empty_env(key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// The NDK sysroot `libvulkan.so` loader for the target ABI/API. The Vulkan
/// loader is target-specific (unlike the headers/glslc/SPIRV-Headers, which are
/// host/arch-neutral), so point cmake's FindVulkan at the sysroot copy.
fn android_sysroot_vulkan_lib(ndk: &Path, api: u32) -> Option<PathBuf> {
    let prebuilt = ndk.join("toolchains/llvm/prebuilt");
    let host_tag = fs::read_dir(&prebuilt)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.is_dir())?;
    let candidate = host_tag.join(format!(
        "sysroot/usr/lib/aarch64-linux-android/{api}/libvulkan.so"
    ));
    candidate.is_file().then_some(candidate)
}

/// Host Vulkan headers dir (must contain `vulkan/vulkan.hpp`) for an android
/// cross build. Headers are arch-neutral; prefer VULKAN_SDK, then Homebrew/system.
fn android_vulkan_include_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(sdk) = vulkan_sdk_path() {
        candidates.push(sdk.join("Include"));
        candidates.push(sdk.join("include"));
    }
    candidates.push(PathBuf::from("/opt/homebrew/include"));
    candidates.push(PathBuf::from("/usr/local/include"));
    candidates.push(PathBuf::from("/usr/include"));
    candidates
        .into_iter()
        .find(|dir| dir.join("vulkan/vulkan.hpp").is_file())
}

/// Directory containing `SPIRV-HeadersConfig.cmake`, passed explicitly as
/// `SPIRV-Headers_DIR` so `find_package(SPIRV-Headers CONFIG)` (required by
/// ggml-vulkan) resolves under the android toolchain — which sets
/// `CMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY` and so will NOT search the host
/// CMAKE_PREFIX_PATH. SPIRV-Headers is header-only / arch-neutral. Prefer
/// VULKAN_SDK, then Homebrew/system prefixes.
fn spirv_headers_config_dir() -> Option<PathBuf> {
    let mut prefixes = Vec::new();
    if let Some(sdk) = vulkan_sdk_path() {
        prefixes.push(sdk);
    }
    prefixes.push(PathBuf::from("/opt/homebrew"));
    prefixes.push(PathBuf::from("/usr/local"));
    prefixes.push(PathBuf::from("/usr"));
    prefixes
        .into_iter()
        .flat_map(|prefix| {
            ["share/cmake/SPIRV-Headers", "lib/cmake/SPIRV-Headers"]
                .into_iter()
                .map(move |rel| prefix.join(rel))
        })
        .find(|dir| dir.join("SPIRV-HeadersConfig.cmake").is_file())
}

/// Locate a host `glslc` (SPIR-V shader compiler): explicit OPENASR_GLSLC, then
/// VULKAN_SDK bin, then PATH. ggml-vulkan compiles its shaders with it at build
/// time; for a cross build this must be the HOST compiler.
fn host_glslc() -> Option<PathBuf> {
    if let Some(path) = non_empty_env("OPENASR_GLSLC").map(PathBuf::from)
        && path.is_file()
    {
        return Some(path);
    }
    if let Some(sdk) = vulkan_sdk_path() {
        let candidate = [
            sdk.join("bin/glslc"),
            sdk.join("Bin/glslc.exe"),
            sdk.join("bin/glslc.exe"),
        ]
        .into_iter()
        .find(|candidate| candidate.is_file());
        if candidate.is_some() {
            return candidate;
        }
    }
    which_on_path("glslc")
}

fn which_on_path(program: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|dir| dir.join(program))
            .find(|candidate| candidate.is_file())
    })
}

fn hip_toolkit_path() -> Option<PathBuf> {
    env::var_os("HIP_PATH")
        .or_else(|| env::var_os("ROCM_PATH"))
        .or_else(|| env::var_os("ROCM_HOME"))
        .map(PathBuf::from)
}

fn cuda_toolkit_path() -> Option<PathBuf> {
    env::var_os("CUDA_PATH")
        .or_else(|| env::var_os("CUDA_HOME"))
        .map(PathBuf::from)
        .or_else(default_windows_cuda_toolkit_path)
}

fn default_windows_cuda_toolkit_path() -> Option<PathBuf> {
    if !cfg!(windows) {
        return None;
    }
    let root = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
    let mut versions = fs::read_dir(root)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.join("bin/nvcc.exe").is_file())
        .collect::<Vec<_>>();
    versions.sort();
    versions.pop()
}

fn hip_sdk_clang_path(hip_path: &Path) -> Option<PathBuf> {
    [hip_path.join("bin/clang.exe"), hip_path.join("bin/clang")]
        .into_iter()
        .find(|path| path.is_file())
}

fn hip_gpu_targets() -> String {
    // A consumer RDNA2/3/3.5/4 arch list: one fat code object covers every
    // supported AMD card and the HIP runtime selects the ISA at load. Union
    // of llama.cpp's current Windows HIP release list (gfx1030/31/32,
    // gfx1100/01/02, gfx1150/51, gfx1200/01) and gfx1035 from a competing
    // ASR product's HIP build, biased toward RDNA2/3/4 gaming/consumer cards.
    // Deliberately excludes CDNA/datacenter compute cards (gfx906/908/90a):
    // those are compute accelerators, not something an end user's desktop/
    // laptop ships, and would meaningfully lengthen every HIP build for a
    // target this product does not support. Override with
    // OPENASR_HIP_GPU_TARGETS for a narrower/wider set.
    env::var("OPENASR_HIP_GPU_TARGETS")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            "gfx1030;gfx1031;gfx1032;gfx1035;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"
                .to_string()
        })
}

fn cuda_gpu_targets() -> String {
    cuda_gpu_targets_from_raw(env::var("OPENASR_CUDA_GPU_TARGETS").ok().as_deref())
}

fn cuda_gpu_targets_from_raw(raw: Option<&str>) -> String {
    raw.map(str::trim)
        .filter(|value| !value.is_empty())
        .map(normalize_cuda_gpu_targets)
        .unwrap_or_else(|| "80;86;89;90".to_string())
}

fn normalize_cuda_gpu_targets(raw: &str) -> String {
    raw.split([',', ';', ' '])
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .strip_prefix("sm_")
                .or_else(|| value.strip_prefix("SM_"))
                .unwrap_or(value)
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn prepare_windows_hip_sdk_shim(target: &str, hip_path: &Path, out_dir: &Path) -> PathBuf {
    let shim_dir = out_dir.join("openasr-windows-hip-sdk-shim");
    let import_lib_dir = shim_dir.join("lib");
    fs::create_dir_all(&import_lib_dir).expect("create Windows HIP import lib dir");

    let bin_dir = hip_path.join("bin");
    let sdk_include_dir = hip_path.join("include");
    prepare_windows_import_lib(
        target,
        &bin_dir.join("libhipblas.dll"),
        &import_lib_dir.join("libhipblas.lib"),
    );
    prepare_windows_import_lib(
        target,
        &bin_dir.join("rocblas.dll"),
        &import_lib_dir.join("rocblas.lib"),
    );
    write_windows_hip_cmake_package(
        &shim_dir,
        "hipblas",
        "roc::hipblas",
        &bin_dir.join("libhipblas.dll"),
        &import_lib_dir.join("libhipblas.lib"),
        &sdk_include_dir,
    );
    write_windows_hip_cmake_package(
        &shim_dir,
        "rocblas",
        "roc::rocblas",
        &bin_dir.join("rocblas.dll"),
        &import_lib_dir.join("rocblas.lib"),
        &sdk_include_dir,
    );
    shim_dir
}

/// Build a Command for an MSVC binutils-style tool (dumpbin.exe, lib.exe).
///
/// These live next to cl.exe in the VC tools bin directory, which is NOT on
/// PATH outside a Developer Command Prompt (CI runners invoke cargo from a
/// plain shell). cc's windows_registry finds cl.exe through the VS installer
/// metadata, so derive the sibling tool from there and inherit the tool env
/// (PATH additions for the DLLs the tool itself needs). Falls back to plain
/// PATH lookup for developer prompts / exotic setups.
fn msvc_bin_tool(target: &str, tool: &str) -> Command {
    if let Some(cl) = cc::windows_registry::find_tool(target, "cl.exe") {
        let path = cl.path().with_file_name(tool);
        if path.is_file() {
            let mut command = Command::new(path);
            for (key, value) in cl.env() {
                command.env(key, value);
            }
            return command;
        }
    }
    Command::new(tool)
}

fn prepare_windows_import_lib(target: &str, dll_path: &Path, import_lib_path: &Path) {
    if import_lib_path.is_file() {
        return;
    }
    if !dll_path.is_file() {
        panic!(
            "required Windows HIP SDK DLL is missing: {}",
            dll_path.display()
        );
    }

    let output = msvc_bin_tool(target, "dumpbin.exe")
        .arg("/exports")
        .arg(dll_path)
        .output()
        .unwrap_or_else(|error| {
            panic!("failed to run dumpbin for {}: {error}", dll_path.display())
        });
    if !output.status.success() {
        panic!(
            "dumpbin /exports {} failed with status {}: {}",
            dll_path.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let exports = parse_dumpbin_exports(&String::from_utf8_lossy(&output.stdout));
    if exports.is_empty() {
        panic!("dumpbin found no exports in {}", dll_path.display());
    }

    let def_path = import_lib_path.with_extension("def");
    let library_name = dll_path
        .file_name()
        .expect("HIP DLL path must have a file name")
        .to_string_lossy();
    let def = format!("LIBRARY {library_name}\nEXPORTS\n{}\n", exports.join("\n"));
    fs::write(&def_path, def).expect("write Windows HIP import library definition");

    let status = msvc_bin_tool(target, "lib.exe")
        .arg(format!("/def:{}", def_path.display()))
        .arg("/machine:x64")
        .arg(format!("/out:{}", import_lib_path.display()))
        .status()
        .unwrap_or_else(|error| {
            panic!("failed to run lib.exe for {}: {error}", dll_path.display())
        });
    if !status.success() {
        panic!(
            "lib.exe could not create import library {} from {} (status {status})",
            import_lib_path.display(),
            dll_path.display()
        );
    }
}

fn parse_dumpbin_exports(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            parts.next()?.parse::<u32>().ok()?;
            let hint = parts.next()?;
            let rva = parts.next()?;
            if !is_hex_token(hint) || !is_hex_token(rva) {
                return None;
            }
            parts.next().map(str::to_string)
        })
        .collect()
}

fn macos_deployment_target() -> String {
    let configured = env::var("MACOSX_DEPLOYMENT_TARGET").ok();
    macos_deployment_target_from(configured.as_deref())
}

fn ios_deployment_target() -> String {
    let configured = env::var("IPHONEOS_DEPLOYMENT_TARGET").ok();
    ios_deployment_target_from(configured.as_deref())
}

fn ios_deployment_target_from(configured: Option<&str>) -> String {
    // arm64-only device hardware (see is_ios above) already implies iOS 11+, but
    // pin a newer floor to match the Rust std minimum for aarch64-apple-ios.
    const MINIMUM: &str = "15.0";
    match configured {
        Some(value) if version_at_least(value, MINIMUM) => value.trim().to_string(),
        _ => MINIMUM.to_string(),
    }
}

fn macos_deployment_target_from(configured: Option<&str>) -> String {
    const MINIMUM: &str = "13.3";
    match configured {
        // Emit the normalized (trimmed) version, not the raw env string, so a
        // value like " 14.0\n" that still parses cannot reach CMake verbatim.
        Some(value) if version_at_least(value, MINIMUM) => value.trim().to_string(),
        _ => MINIMUM.to_string(),
    }
}

fn version_at_least(value: &str, minimum: &str) -> bool {
    let Some(current) = parse_version(value) else {
        return false;
    };
    let Some(required) = parse_version(minimum) else {
        return false;
    };
    let width = current.len().max(required.len());
    for index in 0..width {
        let left = current.get(index).copied().unwrap_or(0);
        let right = required.get(index).copied().unwrap_or(0);
        if left != right {
            return left > right;
        }
    }
    true
}

fn parse_version(value: &str) -> Option<Vec<u32>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    trimmed
        .split('.')
        .map(|part| {
            if part.is_empty() {
                return None;
            }
            part.parse::<u32>().ok()
        })
        .collect()
}

fn is_hex_token(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn write_windows_hip_cmake_package(
    shim_dir: &Path,
    package_name: &str,
    target_name: &str,
    dll_path: &Path,
    import_lib_path: &Path,
    include_dir: &Path,
) {
    let package_dir = shim_dir.join("lib/cmake").join(package_name);
    fs::create_dir_all(&package_dir).expect("create Windows HIP CMake package dir");
    let config = format!(
        "if(NOT TARGET {target_name})\n\
         add_library({target_name} SHARED IMPORTED)\n\
         set_target_properties({target_name} PROPERTIES\n\
         IMPORTED_LOCATION \"{}\"\n\
         IMPORTED_IMPLIB \"{}\"\n\
         INTERFACE_INCLUDE_DIRECTORIES \"{}\")\n\
         endif()\n\
         set({package_name}_FOUND TRUE)\n",
        cmake_path(dll_path),
        cmake_path(import_lib_path),
        cmake_path(include_dir),
    );
    fs::write(
        package_dir.join(format!("{package_name}-config.cmake")),
        config,
    )
    .expect("write Windows HIP CMake package config");
}

fn cmake_build_jobs() -> String {
    env::var("OPENASR_GGML_BUILD_JOBS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|jobs| *jobs > 0)
        .or_else(|| {
            std::thread::available_parallelism()
                .ok()
                .map(|jobs| jobs.get())
        })
        .unwrap_or(1)
        .to_string()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{
        CudaTuning, HipTuning, cuda_gpu_targets_from_raw, ios_deployment_target_from,
        macos_deployment_target_from, parse_bool_env, resolve_ggml_native_enabled, target_is_x86,
        version_at_least,
    };

    #[test]
    fn native_feature_forces_ggml_native_on() {
        assert!(resolve_ggml_native_enabled(
            true,
            "aarch64-apple-darwin",
            "x86_64-pc-windows-msvc",
            Some("0")
        ));
    }

    #[test]
    fn env_value_overrides_default_native_policy() {
        assert!(!resolve_ggml_native_enabled(
            false,
            "x86_64-pc-windows-msvc",
            "x86_64-pc-windows-msvc",
            Some("off")
        ));
        assert!(resolve_ggml_native_enabled(
            false,
            "aarch64-apple-darwin",
            "aarch64-apple-darwin",
            Some("1")
        ));
    }

    #[test]
    fn x86_host_build_defaults_to_native() {
        assert!(resolve_ggml_native_enabled(
            false,
            "x86_64-pc-windows-msvc",
            "x86_64-pc-windows-msvc",
            None
        ));
        assert!(resolve_ggml_native_enabled(
            false,
            "i686-pc-windows-msvc",
            "i686-pc-windows-msvc",
            None
        ));
    }

    #[test]
    fn cross_or_non_x86_build_defaults_to_portable() {
        assert!(!resolve_ggml_native_enabled(
            false,
            "x86_64-unknown-linux-gnu",
            "x86_64-pc-windows-msvc",
            None
        ));
        assert!(!resolve_ggml_native_enabled(
            false,
            "aarch64-apple-darwin",
            "aarch64-apple-darwin",
            None
        ));
    }

    #[test]
    fn parses_bool_env_values() {
        assert_eq!(parse_bool_env(Some(" yes ")), Some(true));
        assert_eq!(parse_bool_env(Some("OFF")), Some(false));
        assert_eq!(parse_bool_env(Some("native")), None);
        assert_eq!(parse_bool_env(None), None);
    }

    #[test]
    fn detects_x86_target_triples() {
        assert!(target_is_x86("x86_64-pc-windows-msvc"));
        assert!(target_is_x86("i686-unknown-linux-gnu"));
        assert!(target_is_x86("i586-pc-windows-msvc"));
        assert!(!target_is_x86("aarch64-apple-darwin"));
    }

    #[test]
    fn cuda_targets_default_to_common_cloud_arches() {
        assert_eq!(cuda_gpu_targets_from_raw(None), "80;86;89;90");
        assert_eq!(cuda_gpu_targets_from_raw(Some("   ")), "80;86;89;90");
    }

    #[test]
    fn cuda_targets_accept_common_arch_spellings() {
        assert_eq!(
            cuda_gpu_targets_from_raw(Some("sm_80,86; SM_89 90")),
            "80;86;89;90"
        );
    }

    #[test]
    fn cuda_tuning_defaults_match_fast_compile_safe_defaults() {
        assert_eq!(
            CudaTuning::from_lookup(|_| None),
            CudaTuning {
                flash_attention: true,
                flash_attention_all_quants: false,
                force_mmq: false,
            }
        );
    }

    #[test]
    fn cuda_tuning_env_overrides_each_build_flag() {
        let env = HashMap::from([
            ("OPENASR_CUDA_FLASH_ATTENTION", "off"),
            ("OPENASR_CUDA_FA_ALL_QUANTS", "yes"),
            ("OPENASR_CUDA_FORCE_MMQ", "true"),
        ]);
        assert_eq!(
            CudaTuning::from_lookup(|key| env.get(key).map(ToString::to_string)),
            CudaTuning {
                flash_attention: false,
                flash_attention_all_quants: true,
                force_mmq: true,
            }
        );
    }

    #[test]
    fn cuda_tuning_summary_is_stable_for_doctor_output() {
        let summary = CudaTuning::from_lookup(|_| None).summary();
        assert_eq!(summary, "fa=on,fa_all_quants=off,force_mmq=off");
    }

    #[test]
    fn hip_tuning_defaults_match_upstream_safe_performance_defaults() {
        assert_eq!(
            HipTuning::from_lookup(|_| None),
            HipTuning {
                graphs: true,
                flash_attention: true,
                flash_attention_all_quants: false,
                rocwmma_flash_attention: false,
                mmq_mfma: true,
                force_mmq: false,
                no_vmm: true,
                export_metrics: false,
            }
        );
    }

    #[test]
    fn hip_tuning_env_overrides_each_build_flag() {
        let env = HashMap::from([
            ("OPENASR_HIP_GRAPHS", "0"),
            ("OPENASR_HIP_FLASH_ATTENTION", "off"),
            ("OPENASR_HIP_FA_ALL_QUANTS", "yes"),
            ("OPENASR_HIP_ROCWMMA_FATTN", "1"),
            ("OPENASR_HIP_MMQ_MFMA", "false"),
            ("OPENASR_HIP_FORCE_MMQ", "true"),
            ("OPENASR_HIP_NO_VMM", "no"),
            ("OPENASR_HIP_EXPORT_METRICS", "on"),
        ]);
        assert_eq!(
            HipTuning::from_lookup(|key| env.get(key).map(ToString::to_string)),
            HipTuning {
                graphs: false,
                flash_attention: false,
                flash_attention_all_quants: true,
                rocwmma_flash_attention: true,
                mmq_mfma: false,
                force_mmq: true,
                no_vmm: false,
                export_metrics: true,
            }
        );
    }

    #[test]
    fn hip_tuning_summary_is_stable_for_doctor_output() {
        let summary = HipTuning::from_lookup(|_| None).summary();
        assert_eq!(
            summary,
            "graphs=on,fa=on,fa_all_quants=off,rocwmma_fattn=off,mmq_mfma=on,force_mmq=off,no_vmm=on,export_metrics=off"
        );
    }

    #[test]
    fn deployment_target_clamps_below_minimum_or_malformed_values() {
        assert_eq!(macos_deployment_target_from(None), "13.3");
        assert_eq!(macos_deployment_target_from(Some("11.0")), "13.3");
        assert_eq!(macos_deployment_target_from(Some("13.2.9")), "13.3");
        assert_eq!(macos_deployment_target_from(Some("14.x")), "13.3");
        assert_eq!(macos_deployment_target_from(Some("13.a.4")), "13.3");
        assert_eq!(macos_deployment_target_from(Some("")), "13.3");
    }

    #[test]
    fn deployment_target_keeps_valid_minimum_or_newer_values() {
        assert_eq!(macos_deployment_target_from(Some("13.3")), "13.3");
        assert_eq!(macos_deployment_target_from(Some("13.3.0")), "13.3.0");
        assert_eq!(macos_deployment_target_from(Some("14.0")), "14.0");
    }

    #[test]
    fn ios_deployment_target_clamps_below_minimum_or_malformed_values() {
        assert_eq!(ios_deployment_target_from(None), "15.0");
        assert_eq!(ios_deployment_target_from(Some("12.0")), "15.0");
        assert_eq!(ios_deployment_target_from(Some("14.x")), "15.0");
        assert_eq!(ios_deployment_target_from(Some("")), "15.0");
    }

    #[test]
    fn ios_deployment_target_keeps_valid_minimum_or_newer_values() {
        assert_eq!(ios_deployment_target_from(Some("15.0")), "15.0");
        assert_eq!(ios_deployment_target_from(Some("17.2")), "17.2");
    }

    #[test]
    fn version_compare_requires_strict_numeric_segments() {
        assert!(version_at_least("13.3", "13.3"));
        assert!(version_at_least("13.3.1", "13.3"));
        assert!(!version_at_least("13.2.9", "13.3"));
        assert!(!version_at_least("14.x", "13.3"));
    }
}
