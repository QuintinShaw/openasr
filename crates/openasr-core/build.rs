use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use sha2::{Digest, Sha256};

#[path = "src/pe_image_identity.rs"]
mod pe_image_identity;
#[path = "src/windows_cmake_cache.rs"]
mod windows_cmake_cache;

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
    // musl triples (e.g. x86_64-unknown-linux-musl) also contain "linux" and must
    // be detected explicitly before any `contains("linux")` check; see the
    // is_musl link-lib arm below for why they need a different C++ runtime.
    let is_musl = target.contains("musl");
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
    // Backend-DL plugin build for the neutral Windows host:
    // ship ggml-base.dll + ggml.dll + ggml-cpu-<variant>.dll loaded via the ggml
    // registry. The installer owns this neutral host plus CPU/Vulkan rescue;
    // signed CUDA/HIP packs are installed and selected independently.
    //
    // Scoped to Windows on purpose. macOS ships a self-contained static binary
    // with Metal/Accelerate (no plugin story). Linux is the CI/CLI platform: a
    // static single binary is simpler to distribute and, crucially, avoids the
    // unverified Linux runtime plugin-discovery path (ggml dlopen of the
    // ggml-cpu-<variant>.so set, which `copy_runtime_dlls` does not stage). The
    // host-side registry refactor (init_by_type / ensure_backends_loaded) still
    // runs on the macOS+Linux+GPU-feature static builds, but
    // `ensure_backends_loaded` (ggml_runtime/backend.rs) skips the directory
    // scan there and relies on the statically registered backend instead. That
    // scan is NOT a harmless no-op
    // when it finds `ggml-*.dll` plugins sitting next to the exe (e.g. a desktop
    // bundle that ships the CPU BACKEND_DL variant alongside a statically-linked
    // GPU exe) — it dlopens a second copy of ggml core into the process and
    // `ggml.cpp:22 GGML_ASSERT(prev != ggml_uncaught_exception)` fastfails
    // (0xc0000409). Non-Windows GPU-feature builds remain platform-static;
    // Windows GPU features stage optional modules unless the explicit legacy
    // sidecar feature is selected.
    // GGML_CPU_ALL_VARIANTS requires GGML_NATIVE=OFF (CMake FATAL_ERROR otherwise),
    // and a portable base must not bake the build host's ISA anyway.
    // Windows defaults to one neutral ggml host. During the single published
    // migration cycle only, a dedicated release leg may compile the old
    // whole-sidecar topology through an auditable Cargo feature. An ambient
    // environment variable can never change a production host's topology.
    let legacy_static_windows =
        is_windows && env::var_os("CARGO_FEATURE_LEGACY_WINDOWS_STATIC_SIDECAR").is_some();
    let use_backend_dl = is_windows && !legacy_static_windows;
    // The neutral Windows host always carries a Vulkan rescue module. CUDA/HIP
    // are optional packs; CPU+Vulkan are the installer-owned LKG and must not
    // depend on a consumer remembering to enable a feature.
    let build_vulkan = feat_vulkan || use_backend_dl;
    println!(
        "cargo:rustc-env=OPENASR_WINDOWS_GGML_TOPOLOGY={}",
        if legacy_static_windows {
            "legacy-static-sidecar"
        } else if is_windows {
            "neutral-backend-dl"
        } else {
            "platform-static"
        }
    );
    let (backend_host_abi_fingerprint, backend_host_abi_json) =
        emit_backend_host_abi(&manifest_dir, &source_dir, &target, use_backend_dl);
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
    //  - musl: built with zig cc/c++ (clang), which does not bundle libomp for
    //    musl targets, so `-fopenmp` would fail to link the same way it does on
    //    android; CPU threading falls back to ggml's own thread pool there too.
    // We neutralize OpenMP for those rather than forcing the whole feature
    // opt-in. `OPENASR_GGML_OPENMP=0` force-disables everywhere.
    let openmp_requested = feat_openmp
        && !matches!(
            env::var("OPENASR_GGML_OPENMP").ok().as_deref(),
            Some("0" | "off" | "OFF" | "false" | "FALSE")
        );
    let openmp_unsupported_target =
        is_macos || is_ios || is_android || is_musl || (feat_hip && is_windows);
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
    let source_fingerprint = windows_cmake_cache::build_relevant_fingerprint(&source_dir)
        .unwrap_or_else(|error| {
            panic!(
                "fingerprint build-relevant openasr-ggml inputs under {}: {error}",
                source_dir.display()
            )
        });
    let source_fingerprint_stamp = build_dir.join(windows_cmake_cache::SOURCE_FINGERPRINT_STAMP);
    let stored_source_fingerprint = fs::read_to_string(&source_fingerprint_stamp).ok();
    // A restored Cargo target directory can contain CMake objects newer than a
    // freshly checked-out native source tree. In that state CMake's timestamp-
    // based incremental build can accept old objects even though Cargo reran this
    // build script. Treat source contents as part of the cache contract and force
    // a fresh private CMake tree whenever that identity changes or is absent.
    let mut reset_native_build_dir = windows_cmake_cache::source_fingerprint_requires_reset(
        stored_source_fingerprint.as_deref(),
        &source_fingerprint,
    );

    let hip_path = feat_hip.then(hip_toolkit_path).flatten();
    let cuda_path = feat_cuda.then(cuda_toolkit_path).flatten();
    let vulkan_sdk = build_vulkan.then(vulkan_sdk_path).flatten();

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
    if is_windows {
        // CMake deletes and internally re-runs its cache when a configured
        // compiler changes. That internal re-run loses this invocation's `-D`
        // contract, which can leave a Debug shared-library build behind while
        // rustc expects fresh Release static archives. Validate every
        // topology-driving cache entry up front and rebuild only this crate's
        // private CMake directory when the contract has drifted.
        let expected_compiler = if is_windows_arm64 {
            Some("clang-cl".to_owned())
        } else if feat_hip {
            hip_path
                .as_deref()
                .and_then(hip_sdk_clang_path)
                .map(|path| cmake_path(&path))
        } else {
            msvc_tool.as_ref().map(|tool| cmake_path(tool.path()))
        };
        // CMake embeds the absolute source directory in its cache and refuses to
        // reuse that tree from a new immutable staging path, even when the native
        // contents have the same fingerprint.
        let mut tool_expectations = vec![("CMAKE_HOME_DIRECTORY", cmake_path(&source_dir))];
        if let Some(compiler) = expected_compiler {
            tool_expectations.push(("CMAKE_C_COMPILER", compiler.clone()));
            tool_expectations.push(("CMAKE_CXX_COMPILER", compiler));
        }
        if feat_cuda
            && let Some(path) = cuda_path.as_deref()
            && path.join("bin/nvcc.exe").is_file()
        {
            tool_expectations.push((
                "CMAKE_CUDA_COMPILER",
                cmake_path(&path.join("bin/nvcc.exe")),
            ));
        }
        if feat_cuda && let Some(tool) = msvc_tool.as_ref() {
            tool_expectations.push(("CMAKE_CUDA_HOST_COMPILER", cmake_path(tool.path())));
        }
        let on_off = |enabled| if enabled { "ON" } else { "OFF" }.to_owned();
        let scalar_expectations = vec![
            ("CMAKE_BUILD_TYPE", "Release".to_owned()),
            ("BUILD_SHARED_LIBS", on_off(use_backend_dl)),
            ("GGML_BACKEND_DL", on_off(use_backend_dl)),
            (
                "OPENASR_VERIFIED_BACKEND_LOADING_ONLY",
                on_off(use_backend_dl),
            ),
            ("GGML_CPU_ALL_VARIANTS", on_off(ggml_cpu_all_variants)),
            ("GGML_BUILD_TESTS", "OFF".to_owned()),
            ("GGML_BUILD_EXAMPLES", "OFF".to_owned()),
            ("GGML_NATIVE", on_off(ggml_native)),
            ("GGML_OPENMP", on_off(effective_openmp)),
            ("GGML_CUDA", on_off(feat_cuda)),
            ("GGML_VULKAN", on_off(build_vulkan)),
            ("GGML_HIP", on_off(feat_hip)),
            ("GGML_SYCL", on_off(feat_sycl)),
            (
                "OPENASR_BACKEND_ABI_V1",
                backend_host_abi_fingerprint.clone(),
            ),
        ];
        if !reset_native_build_dir {
            let cache = build_dir.join("CMakeCache.txt");
            reset_native_build_dir = fs::read_to_string(&cache).map_or(true, |text| {
                !windows_cmake_cache::cache_matches_contract(
                    &text,
                    &tool_expectations,
                    &scalar_expectations,
                )
            });
        }
    }
    if reset_native_build_dir && build_dir.exists() {
        fs::remove_dir_all(&build_dir).expect("reset incompatible openasr-ggml build dir");
    }
    fs::create_dir_all(&lib_dir).expect("create openasr-ggml lib dir");
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
        .arg(cmake_flag(
            "OPENASR_VERIFIED_BACKEND_LOADING_ONLY",
            use_backend_dl,
        ))
        .arg(cmake_flag("GGML_CPU_ALL_VARIANTS", ggml_cpu_all_variants))
        .arg("-DGGML_BUILD_TESTS=OFF")
        .arg("-DGGML_BUILD_EXAMPLES=OFF")
        .arg(cmake_flag("GGML_NATIVE", ggml_native))
        .arg(cmake_flag("GGML_OPENMP", effective_openmp))
        .arg(cmake_flag("GGML_CUDA", feat_cuda))
        // Keep CUDA Graph capture fail-closed and deterministic. A host-local
        // X-ASR experiment showed that cached graph executables can outlive
        // the CUDA primary context during Windows TLS teardown; explicitly
        // pinning this OFF also prevents a stale CMake cache from silently
        // carrying that rejected experiment into later production builds.
        .arg("-DGGML_CUDA_GRAPHS=OFF")
        .arg(cmake_flag("GGML_VULKAN", build_vulkan))
        .arg(cmake_flag("GGML_HIP", feat_hip))
        .arg(cmake_flag("GGML_SYCL", feat_sycl))
        .arg(format!(
            "-DOPENASR_BACKEND_ABI_V1={backend_host_abi_fingerprint}"
        ))
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
    if build_vulkan
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
    if (feat_hip || build_vulkan || feat_cuda || is_windows_arm64) && is_windows {
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
    if is_musl {
        // Every musl leg is a cross build (there is no musl GitHub-hosted host
        // runner): without an explicit toolchain file cmake configures for the
        // HOST compiler/arch, silently producing host-arch objects that rustc's
        // linker skips as format-incompatible -- which surfaces downstream as a
        // confusing wall of "undefined symbol: ggml_*" at the final binary link,
        // not a clear cross-compile error here.
        //
        // `cargo zigbuild` (the required build entry point for musl targets; see
        // CI) sets `CMAKE_TOOLCHAIN_FILE_<target>` to a toolchain file it
        // generates, pinning CC/CXX/AR/RANLIB to its zig cc/zig c++ wrappers plus
        // the CMAKE_FIND_ROOT_PATH_MODE_* settings cross-compiling CMake needs;
        // prefer that. Fall back to a minimal, self-authored toolchain file built
        // from CC_<target>/CXX_<target> (or plain CC/CXX) so a build that
        // exports those manually (without `cargo zigbuild`) still cross-compiles
        // correctly instead of silently falling back to the host toolchain.
        let env_target = target.replace('-', "_");
        let toolchain_file = env::var(format!("CMAKE_TOOLCHAIN_FILE_{env_target}"))
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| write_musl_cmake_toolchain_file(&target, &out_dir));
        configure.arg(format!(
            "-DCMAKE_TOOLCHAIN_FILE={}",
            cmake_path(&toolchain_file)
        ));
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
    // CUDA is still compiled when the Windows host uses BACKEND_DL; only the
    // final Rust link step is suppressed in that topology. CMake needs the
    // toolkit, nvcc host compiler, target SM list and tuning flags in both the
    // static and module builds or the optional ggml-cuda DLL is either
    // unbuildable or silently built with a different contract.
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
    fs::write(&source_fingerprint_stamp, format!("{source_fingerprint}\n")).unwrap_or_else(
        |error| {
            panic!(
                "write openasr-ggml source fingerprint stamp {}: {error}",
                source_fingerprint_stamp.display()
            )
        },
    );

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
        stage_windows_backend_dl_artifacts(
            &build_dir.join("bin"),
            &out_dir,
            &backend_host_abi_fingerprint,
            &backend_host_abi_json,
            &target,
            build_vulkan,
            feat_cuda,
            feat_hip,
            feat_sycl,
            ggml_cpu_all_variants,
            ggml_native,
            effective_openmp,
            &cuda_tuning.summary(),
            &hip_tuning.summary(),
        );
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

    if feat_cuda && !use_backend_dl {
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

    if feat_vulkan && !use_backend_dl {
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

    if feat_hip && !use_backend_dl {
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

    if feat_sycl && !use_backend_dl {
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
    } else if is_musl {
        // musl targets are built with zig cc/c++ (clang), not GCC, and crt-static
        // defaults ON: there is no dynamic libstdc++/libgomp to link against in a
        // musl sysroot, and a portable static musl binary must not require them at
        // runtime anyway. Statically link LLVM's libc++ (the runtime clang/zig
        // pairs ggml's C++ objects with) instead of GNU libstdc++. No gomp: clang
        // silently ignores `#pragma omp` without `-fopenmp` (unlike GCC, which
        // partially recognizes it even when GGML_OPENMP=OFF), so a clang-built
        // ggml-cpu.a has no GOMP_*/omp_* undefined symbols to satisfy. This MUST be
        // checked before the `linux` arm below because musl triples also contain
        // "linux" (e.g. x86_64-unknown-linux-musl).
        //
        // rustc validates a `kind=static` native library's existence (via the
        // emitted `-L` search paths) while compiling THIS crate, not later at the
        // final binary's link step -- so the .a files must already exist in
        // `lib_dir` by the time this build script exits. zig only builds its
        // bundled libc++/libc++abi/libunwind from source (into its own opaque,
        // content-hashed global cache) on demand, the first time something asks
        // to link them; materialize them into `lib_dir` here so both the
        // existence check and the real link succeed.
        materialize_musl_libcxx_archives(&target, &lib_dir);
        println!("cargo:rustc-link-lib=static=c++");
        println!("cargo:rustc-link-lib=static=c++abi");
        println!("cargo:rustc-link-lib=static=unwind");
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
    // Cargo recursively watches directory inputs. Use the same complete roots
    // as the source fingerprint above instead of a hand-maintained list of
    // selected native files; adding an operation to a previously unlisted .c,
    // header, shader, or CMake module must always rerun this build script.
    for relative in windows_cmake_cache::BUILD_RELEVANT_DIRECTORIES
        .iter()
        .chain(windows_cmake_cache::BUILD_RELEVANT_FILES)
    {
        println!(
            "cargo:rerun-if-changed={}",
            source_dir.join(relative).display()
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
        "cargo:rustc-env=OPENASR_GGML_BACKEND_DL_ENABLED={}",
        if use_backend_dl { "1" } else { "0" }
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
}

fn cmake_flag(name: &str, enabled: bool) -> String {
    format!("-D{}={}", name, if enabled { "ON" } else { "OFF" })
}

/// Fallback CMake toolchain file for a musl cross build when `cargo zigbuild`'s
/// own `CMAKE_TOOLCHAIN_FILE_<target>` env var isn't set (e.g. a manual `cargo
/// build` with CC/CXX exported by hand). Mirrors the shape cargo-zigbuild itself
/// generates: explicit CMAKE_SYSTEM_NAME/PROCESSOR (cross-compiling cmake does
/// NOT infer these from the target triple on its own -- left alone it detects
/// the BUILD host, which is wrong for a cross build) and the FIND_ROOT_PATH
/// modes that keep `find_package`/`find_library` (e.g. ggml's `pkg-config`-based
/// probes) scoped to the target sysroot rather than falling back to the host.
fn write_musl_cmake_toolchain_file(target: &str, out_dir: &Path) -> PathBuf {
    let env_target = target.replace('-', "_");
    let cc = env::var(format!("CC_{env_target}"))
        .or_else(|_| env::var("CC"))
        .unwrap_or_else(|_| {
            panic!(
                "no C compiler found for musl target {target}: set CC_{env_target} (cargo \
                 zigbuild sets this automatically -- build via `cargo zigbuild`, not plain \
                 `cargo build`) or CC to a zig-cc-for-musl wrapper"
            )
        });
    let cxx = env::var(format!("CXX_{env_target}"))
        .or_else(|_| env::var("CXX"))
        .unwrap_or_else(|_| {
            panic!(
                "no C++ compiler found for musl target {target}: set CXX_{env_target} (cargo \
                 zigbuild sets this automatically -- build via `cargo zigbuild`, not plain \
                 `cargo build`) or CXX to a zig-c++-for-musl wrapper"
            )
        });
    let processor = if target.starts_with("aarch64") {
        "aarch64"
    } else if target.starts_with("x86_64") {
        "x86_64"
    } else {
        panic!("unsupported musl cross arch in target {target}: only x86_64/aarch64 are wired up")
    };
    let content = format!(
        "set(CMAKE_SYSTEM_NAME Linux)\n\
         set(CMAKE_SYSTEM_PROCESSOR {processor})\n\
         set(CMAKE_C_COMPILER {cc})\n\
         set(CMAKE_CXX_COMPILER {cxx})\n\
         set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)\n\
         set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)\n\
         set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)\n\
         set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)\n"
    );
    let path = out_dir.join("musl-toolchain.cmake");
    fs::write(&path, content).expect("write fallback musl cmake toolchain file");
    path
}

/// Ensure `libc++.a` / `libc++abi.a` / `libunwind.a` for a musl cross target exist
/// in `lib_dir` (already on the crate's `-L` search path), building them via zig's
/// C++ frontend if needed. A no-op if they are already there (keeps incremental
/// rebuilds fast).
///
/// Why this exists: musl targets are cross-compiled with zig cc/c++ (see
/// `docs`/CI: `cargo zigbuild`, which sets `CC_<target>`/`CXX_<target>` env vars to
/// its generated wrapper scripts). Zig bundles LLVM's libc++/libc++abi/libunwind
/// source for musl targets and compiles+archives them on first use into its own
/// global cache (`zig env`'s `global_cache_dir`, content-hashed, not a stable
/// path) -- there is no prebuilt musl libc++ package to fetch instead (unlike
/// e.g. Alpine's alsa-lib, which ships a normal shared build). rustc's own
/// `#[link(kind = "static")]` handling requires the archive to already exist in a
/// `-L` dir at the time THIS crate compiles (unlike `dylib` links, which are only
/// resolved at the final binary's link step) -- so it must be produced here in
/// build.rs, not left for the final `openasr` binary link to trigger lazily.
///
/// Approach: compile+link a trivial `<iostream>`-using stub with `-v` (verbose),
/// which makes zig print the exact `ld.lld` invocation it runs -- including the
/// fully resolved, absolute paths to the three archives it just built into its
/// cache -- then copy those three files into `lib_dir` under their conventional
/// names so the plain `cargo:rustc-link-lib=static=c++` (+ c++abi, unwind)
/// directives resolve normally.
fn materialize_musl_libcxx_archives(target: &str, lib_dir: &Path) {
    let archives = ["libc++.a", "libc++abi.a", "libunwind.a"];
    if archives.iter().all(|name| lib_dir.join(name).is_file()) {
        return;
    }

    let env_target = target.replace('-', "_");
    let cxx = env::var(format!("CXX_{env_target}"))
        .or_else(|_| env::var("CXX"))
        .unwrap_or_else(|_| {
            panic!(
                "no C++ compiler found for musl target {target}: set CXX_{env_target} (cargo \
                 zigbuild sets this automatically -- build via `cargo zigbuild`, not plain \
                 `cargo build`) or CXX to a zig-c++-for-musl wrapper (`zig c++ -target \
                 {target_arch}-linux-musl`)",
                target_arch = target.split('-').next().unwrap_or("x86_64"),
            )
        });

    let stub_source = lib_dir.join("musl_libcxx_probe.cpp");
    fs::write(
        &stub_source,
        "#include <iostream>\nint main() { std::cout << \"probe\"; return 0; }\n",
    )
    .expect("write musl libc++ probe source");
    let stub_binary = lib_dir.join("musl_libcxx_probe");

    // Split on whitespace: CXX_<target>/CXX may be a "compiler plus flags" string
    // (as cargo/cc-rs conventionally allow), not just a bare executable path.
    let mut parts = cxx.split_whitespace();
    let cxx_program = parts
        .next()
        .unwrap_or_else(|| panic!("CXX_{env_target}/CXX is empty"));
    let mut probe = Command::new(cxx_program);
    probe
        .args(parts)
        .arg("-v")
        .arg("-static")
        .arg("-o")
        .arg(&stub_binary)
        .arg(&stub_source);
    let output = probe
        .output()
        .unwrap_or_else(|e| panic!("failed to run {cxx} to build the musl libc++ probe: {e}"));
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() {
        panic!(
            "musl libc++ probe build failed (exit {:?}) using `{cxx}`:\n{combined}",
            output.status.code()
        );
    }

    for name in archives {
        let found = combined
            .split_whitespace()
            .find(|tok| tok.ends_with(name) && Path::new(tok).is_file())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!(
                    "could not find an absolute path to {name} in the musl libc++ probe's \
                     verbose link output; zig's `-v` output format may have changed. Full \
                     output:\n{combined}"
                )
            });
        fs::copy(&found, lib_dir.join(name)).unwrap_or_else(|e| {
            panic!(
                "failed to copy {} to {}: {e}",
                found.display(),
                lib_dir.display()
            )
        });
    }

    let _ = fs::remove_file(&stub_source);
    let _ = fs::remove_file(&stub_binary);
}

/// Copy the backend-DL host and every feature-selected backend module next to
/// where cargo runs binaries and tests, so they resolve at load time without a
/// PATH dance. Windows searches the executable's own directory first; cargo
/// emits final bins to `target/<profile>/` and test bins to
/// `target/<profile>/deps/`, so seed both. Windows-only by construction: under
/// BUILD_SHARED_LIBS the Windows runtime DLLs land in the cmake RUNTIME dir
/// (`bin`), while ELF/Mach-O shared objects go to the LIBRARY dir and resolve via
/// rpath instead — Linux DL packaging is handled separately.
fn emit_backend_host_abi(
    manifest_dir: &Path,
    source_dir: &Path,
    target: &str,
    backend_dl: bool,
) -> (String, serde_json::Value) {
    let backend_impl = source_dir.join("src/ggml-backend-impl.h");
    let header_paths = [
        source_dir.join("include/ggml.h"),
        source_dir.join("include/ggml-backend.h"),
        source_dir.join("include/ggml-alloc.h"),
        backend_impl.clone(),
    ];
    let headers_sha256 = hash_named_files(&header_paths);
    let openasr_ffi = manifest_dir.join("src/ggml_runtime/ffi.rs");
    let openasr_ffi_sha256 = hash_named_files(std::slice::from_ref(&openasr_ffi));
    let openasr_extension_sha256 = hash_named_files(&[
        openasr_ffi.clone(),
        source_dir.join("include/ggml.h"),
        backend_impl.clone(),
    ]);
    let backend_api_version = parse_backend_api_version(&backend_impl);
    // Source snapshots used by the Desktop packager intentionally contain no
    // `.git` metadata. Let that packager carry the revision from the clean
    // source checkout so a host built from an immutable snapshot has the same
    // ABI identity as optional modules built from the checkout itself. The
    // content-bearing header/FFI/extension hashes below remain authoritative;
    // this token cannot make different source bytes compatible.
    let ggml_revision = env::var("OPENASR_GGML_REVISION_OVERRIDE")
        .ok()
        .inspect(|revision| {
            assert!(
                revision.len() == 40
                    && revision
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f')),
                "OPENASR_GGML_REVISION_OVERRIDE must be a lowercase 40-hex git object id"
            );
        })
        .or_else(|| read_git_revision(source_dir))
        .unwrap_or_else(|| format!("source-{headers_sha256}"));
    let crt = if target.ends_with("windows-msvc") {
        "msvc-md"
    } else if target.ends_with("windows-gnu") {
        "gnu"
    } else {
        "platform-default"
    };
    let default_toolchain = if target.ends_with("windows-msvc") {
        "msvc-v143"
    } else if target.ends_with("windows-gnu") {
        "gnu-w64"
    } else {
        "platform-default"
    };
    let toolchain = env::var("OPENASR_BACKEND_TOOLCHAIN_CONTRACT")
        .unwrap_or_else(|_| default_toolchain.to_string());
    assert!(
        !toolchain.is_empty()
            && toolchain.len() <= 128
            && toolchain
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')),
        "OPENASR_BACKEND_TOOLCHAIN_CONTRACT must be a short ASCII contract token"
    );
    let compile_flags_contract = format!(
        "build_shared_libs={}\nggml_backend_dl={}\nverified_backend_loading_only={}\nposition_independent_code=1\nopenasr_backend_abi_export=1\n",
        u8::from(backend_dl),
        u8::from(backend_dl),
        u8::from(backend_dl),
    );
    let compile_flags_sha256 = sha256_hex(compile_flags_contract.as_bytes());
    let build_contract = format!(
        "schema=2\ntarget={target}\ncrt={crt}\ntoolchain={toolchain}\ncompile_flags_sha256={compile_flags_sha256}\nbackend_dl={}\nshared={}\nbackend_api_version={backend_api_version}\nggml_revision={ggml_revision}\nggml_headers_sha256={headers_sha256}\nopenasr_ffi_sha256={openasr_ffi_sha256}\nopenasr_extension_sha256={openasr_extension_sha256}\n",
        u8::from(backend_dl),
        u8::from(backend_dl),
    );
    let fingerprint = sha256_hex(build_contract.as_bytes());

    println!("cargo:rustc-env=OPENASR_BACKEND_ABI_SCHEMA_VERSION=2");
    println!("cargo:rustc-env=OPENASR_BACKEND_HOST_ABI_FINGERPRINT={fingerprint}");
    println!("cargo:rustc-env=OPENASR_BACKEND_TARGET={target}");
    println!("cargo:rustc-env=OPENASR_BACKEND_CRT={crt}");
    println!("cargo:rustc-env=OPENASR_BACKEND_TOOLCHAIN={toolchain}");
    println!("cargo:rustc-env=OPENASR_BACKEND_COMPILE_FLAGS_SHA256={compile_flags_sha256}");
    println!("cargo:rustc-env=OPENASR_GGML_BACKEND_API_VERSION={backend_api_version}");
    println!("cargo:rustc-env=OPENASR_GGML_REVISION={ggml_revision}");
    println!("cargo:rustc-env=OPENASR_GGML_HEADERS_SHA256={headers_sha256}");
    println!("cargo:rustc-env=OPENASR_GGML_FFI_SHA256={openasr_ffi_sha256}");
    println!("cargo:rustc-env=OPENASR_GGML_EXTENSION_SHA256={openasr_extension_sha256}");
    println!("cargo:rerun-if-env-changed=OPENASR_BACKEND_TOOLCHAIN_CONTRACT");
    println!("cargo:rerun-if-env-changed=OPENASR_GGML_REVISION_OVERRIDE");
    println!("cargo:rerun-if-changed={}", openasr_ffi.display());
    let json = serde_json::json!({
        "schema_version": 2,
        "fingerprint": fingerprint,
        "target": target,
        "crt": crt,
        "toolchain": toolchain,
        "compile_flags_sha256": compile_flags_sha256,
        "ggml_backend_api_version": backend_api_version,
        "ggml_revision": ggml_revision,
        "ggml_headers_sha256": headers_sha256,
        "openasr_ffi_sha256": openasr_ffi_sha256,
        "openasr_extension_sha256": openasr_extension_sha256,
    });
    (
        json["fingerprint"]
            .as_str()
            .expect("host ABI fingerprint is a string")
            .to_string(),
        json,
    )
}

fn hash_named_files(paths: &[PathBuf]) -> String {
    let mut hasher = Sha256::new();
    for path in paths {
        let name = path
            .file_name()
            .expect("ABI input must have a file name")
            .to_string_lossy();
        let bytes = fs::read(path)
            .unwrap_or_else(|error| panic!("failed to read ABI input {}: {error}", path.display()));
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    format!("{:x}", hasher.finalize())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn parse_backend_api_version(path: &Path) -> u32 {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    text.lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("#define GGML_BACKEND_API_VERSION ")
                .and_then(|value| value.trim().parse().ok())
        })
        .unwrap_or_else(|| {
            panic!(
                "GGML_BACKEND_API_VERSION is missing from {}",
                path.display()
            )
        })
}

fn read_git_revision(source_dir: &Path) -> Option<String> {
    let dot_git = source_dir.join(".git");
    let git_dir = if dot_git.is_dir() {
        dot_git
    } else {
        let text = fs::read_to_string(&dot_git).ok()?;
        let relative = text.trim().strip_prefix("gitdir:")?.trim();
        source_dir.join(relative)
    };
    let head = fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let head = head.trim();
    if is_git_object_id(head) {
        return Some(head.to_ascii_lowercase());
    }
    let reference = head.strip_prefix("ref:")?.trim();
    let direct = git_dir.join(reference);
    if let Ok(value) = fs::read_to_string(direct) {
        let value = value.trim();
        if is_git_object_id(value) {
            return Some(value.to_ascii_lowercase());
        }
    }
    let common = fs::read_to_string(git_dir.join("commondir")).ok()?;
    let common_dir = git_dir.join(common.trim());
    let value = fs::read_to_string(common_dir.join(reference)).ok()?;
    let value = value.trim();
    is_git_object_id(value).then(|| value.to_ascii_lowercase())
}

fn is_git_object_id(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn stage_windows_backend_dl_artifacts(
    bin_dir: &Path,
    out_dir: &Path,
    backend_host_abi_fingerprint: &str,
    backend_host_abi_json: &serde_json::Value,
    target: &str,
    feat_vulkan: bool,
    feat_cuda: bool,
    feat_hip: bool,
    feat_sycl: bool,
    ggml_cpu_all_variants: bool,
    ggml_native: bool,
    effective_openmp: bool,
    cuda_tuning: &str,
    hip_tuning: &str,
) {
    // out_dir = target/<profile>/build/<pkg>-<hash>/out -> nth(3) = target/<profile>
    let Some(profile_dir) = out_dir.ancestors().nth(3) else {
        return;
    };
    let cmake_contract = windows_cmake_build_contract(
        &bin_dir
            .parent()
            .expect("CMake runtime directory has a build parent")
            .join("CMakeCache.txt"),
        feat_cuda,
    );
    let cmake_contract_sha256 = sha256_hex(
        &serde_json::to_vec(&cmake_contract).expect("serialize Windows CMake build contract"),
    );
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
    let by_name = |wanted: &str| {
        dlls.iter().find(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().eq_ignore_ascii_case(wanted))
        })
    };
    let mut bundled = dlls
        .iter()
        .filter(|path| {
            path.file_name().is_some_and(|name| {
                let name = name.to_string_lossy().to_ascii_lowercase();
                name == "ggml.dll"
                    || name == "ggml-base.dll"
                    || name.starts_with("ggml-cpu")
                    || (feat_vulkan && name == "ggml-vulkan.dll")
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    bundled.sort();

    let manifest_files = bundled
        .iter()
        .map(|path| {
            let filename = path
                .file_name()
                .expect("bundled backend DLL has a filename")
                .to_string_lossy()
                .to_string();
            let lower = filename.to_ascii_lowercase();
            let provider = if lower.starts_with("ggml-cpu") {
                "cpu"
            } else if lower == "ggml-vulkan.dll" {
                "vulkan"
            } else {
                "host"
            };
            let bytes = fs::read(path)
                .unwrap_or_else(|error| panic!("hash bundled backend {}: {error}", path.display()));
            let image = pe_image_identity::pe_image_identity(&bytes).unwrap_or_else(|error| {
                panic!("derive stable PE identity for {}: {error}", path.display())
            });
            (
                serde_json::json!({
                    "filename": filename.clone(),
                    "provider": provider,
                    "sha256": sha256_hex(&bytes),
                    "size_bytes": bytes.len(),
                    "image_sha256": image.sha256.clone(),
                    "image_size_bytes": image.size_bytes,
                }),
                pe_image_identity::BackendBundleContractEntry {
                    filename,
                    provider: provider.to_string(),
                    image_sha256: image.sha256,
                    image_size_bytes: image.size_bytes,
                },
            )
        })
        .collect::<Vec<_>>();
    let bundled_contract_sha256 = pe_image_identity::backend_bundle_contract_sha256(
        backend_host_abi_fingerprint,
        &manifest_files
            .iter()
            .map(|(_, contract)| contract.clone())
            .collect::<Vec<_>>(),
    );
    let provider_contract = |provider: &str| {
        pe_image_identity::backend_bundle_contract_sha256(
            backend_host_abi_fingerprint,
            &manifest_files
                .iter()
                .map(|(_, contract)| contract)
                .filter(|contract| contract.provider == "host" || contract.provider == provider)
                .cloned()
                .collect::<Vec<_>>(),
        )
    };
    let bundled_cpu_contract_sha256 = provider_contract("cpu");
    let bundled_vulkan_contract_sha256 = provider_contract("vulkan");
    println!("cargo:rustc-env=OPENASR_BUNDLED_BACKEND_CONTRACT_SHA256={bundled_contract_sha256}");
    println!("cargo:rustc-env=OPENASR_BUNDLED_CPU_CONTRACT_SHA256={bundled_cpu_contract_sha256}");
    println!(
        "cargo:rustc-env=OPENASR_BUNDLED_VULKAN_CONTRACT_SHA256={bundled_vulkan_contract_sha256}"
    );
    println!("cargo:rerun-if-env-changed=OPENASR_BUNDLED_VULKAN_LOADER_SHA256");
    if let Ok(loader_sha256) = env::var("OPENASR_BUNDLED_VULKAN_LOADER_SHA256") {
        assert!(
            loader_sha256.len() == 64
                && loader_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
            "OPENASR_BUNDLED_VULKAN_LOADER_SHA256 must be lowercase 64-hex"
        );
        println!("cargo:rustc-env=OPENASR_BUNDLED_VULKAN_LOADER_SHA256={loader_sha256}");
    }
    let mut bundled_manifest = serde_json::to_vec(&serde_json::json!({
        "schema_version": 3,
        "host_abi_fingerprint": backend_host_abi_fingerprint,
        "bundle_contract_sha256": bundled_contract_sha256,
        "cpu_contract_sha256": bundled_cpu_contract_sha256,
        "vulkan_contract_sha256": bundled_vulkan_contract_sha256,
        "files": manifest_files
            .iter()
            .map(|(file, _)| file)
            .collect::<Vec<_>>(),
    }))
    .expect("serialize bundled backend manifest");
    bundled_manifest.push(b'\n');
    let cuda_targets = feat_cuda.then(cuda_gpu_targets).unwrap_or_default();
    let hip_targets = feat_hip.then(hip_gpu_targets).unwrap_or_default();
    let build_manifest = serde_json::json!({
        "schema_version": 1,
        "host_abi": backend_host_abi_json,
        "target": target,
        "topology": "neutral-backend-dl",
        "providers": {
            "cpu": true,
            "vulkan": feat_vulkan,
            "cuda": feat_cuda,
            "hip": feat_hip,
            "sycl": feat_sycl,
        },
        "backend_targets": {
            "cuda": cuda_targets
                .split(';')
                .filter(|value| !value.is_empty())
                .map(|value| format!("sm_{value}"))
                .collect::<Vec<_>>(),
            "hip": hip_targets
                .split(';')
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>(),
        },
        "build_flags": {
            "backend_dl": true,
            "shared": true,
            "verified_backend_loading_only": true,
            "cpu_all_variants": ggml_cpu_all_variants,
            "native": ggml_native,
            "openmp": effective_openmp,
            "cuda_tuning": cuda_tuning,
            "hip_tuning": hip_tuning,
        },
        "cmake_contract": cmake_contract,
        "cmake_contract_sha256": cmake_contract_sha256,
        "bundled_backend_contract_sha256": bundled_contract_sha256,
    });
    let mut host_abi_bytes =
        serde_json::to_vec_pretty(backend_host_abi_json).expect("serialize host ABI manifest");
    host_abi_bytes.push(b'\n');
    let mut build_manifest_bytes =
        serde_json::to_vec_pretty(&build_manifest).expect("serialize backend build manifest");
    build_manifest_bytes.push(b'\n');

    for required in ["ggml.dll", "ggml-base.dll"] {
        assert!(
            by_name(required).is_some(),
            "missing required BACKEND_DL host DLL {required}"
        );
    }
    assert!(
        bundled.iter().any(|path| {
            path.file_name().is_some_and(|name| {
                name.to_string_lossy()
                    .to_ascii_lowercase()
                    .starts_with("ggml-cpu")
            })
        }),
        "missing required BACKEND_DL CPU plugin"
    );
    if feat_vulkan {
        assert!(
            by_name("ggml-vulkan.dll").is_some(),
            "missing bundled Vulkan rescue plugin"
        );
    }

    let flat_destinations = [profile_dir.to_path_buf(), profile_dir.join("deps")];
    let destinations = flat_destinations
        .iter()
        .flat_map(|root| {
            [
                root.clone(),
                root.join("openasr-backend-bundles")
                    .join(backend_host_abi_fingerprint),
            ]
        })
        .collect::<Vec<_>>();
    for dest in destinations {
        fs::create_dir_all(&dest).expect("create BACKEND_DL runtime directory");
        // Optional accelerators are never application-directory plugins. Clear
        // stale copies left by a previous build topology before staging the
        // neutral host and bundled rescue set.
        for optional in ["ggml-cuda.dll", "ggml-hip.dll"] {
            match fs::remove_file(dest.join(optional)) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => panic!("remove stale optional backend {optional}: {error}"),
            }
        }
        for dll in &bundled {
            let name = dll.file_name().expect("runtime DLL has a filename");
            fs::copy(dll, dest.join(name)).unwrap_or_else(|error| {
                panic!("stage bundled backend DLL {}: {error}", dll.display())
            });
        }
        fs::write(
            dest.join("openasr-backend-bundle-v1.json"),
            &bundled_manifest,
        )
        .expect("write bundled backend manifest");
        fs::write(
            dest.join("openasr-backend-host-abi-v1.json"),
            &host_abi_bytes,
        )
        .expect("write backend host ABI manifest");
        fs::write(
            dest.join("openasr-backend-build-v1.json"),
            &build_manifest_bytes,
        )
        .expect("write backend build manifest");
    }

    for (enabled, provider, filename) in [
        (feat_cuda, "cuda", "ggml-cuda.dll"),
        (feat_hip, "hip", "ggml-hip.dll"),
    ] {
        if !enabled {
            continue;
        }
        let source = by_name(filename)
            .unwrap_or_else(|| panic!("missing optional backend module {filename}"));
        let dest = profile_dir.join("openasr-backend-packs").join(provider);
        fs::create_dir_all(&dest).expect("create optional backend pack staging directory");
        fs::copy(source, dest.join(filename)).unwrap_or_else(|error| {
            panic!(
                "stage optional backend module {}: {error}",
                source.display()
            )
        });
        fs::write(
            dest.join("openasr-backend-host-abi-v1.json"),
            &host_abi_bytes,
        )
        .expect("write optional backend host ABI manifest");
        fs::write(
            dest.join("openasr-backend-build-v1.json"),
            &build_manifest_bytes,
        )
        .expect("write optional backend build manifest");
    }
}

fn windows_cmake_build_contract(
    cache_path: &Path,
    require_cuda_compiler: bool,
) -> serde_json::Value {
    let cache = fs::read_to_string(cache_path).unwrap_or_else(|error| {
        panic!(
            "read configured Windows CMake cache {}: {error}",
            cache_path.display()
        )
    });
    let mut entries = BTreeMap::new();
    for name in [
        "CMAKE_BUILD_TYPE",
        "CMAKE_GENERATOR",
        "CMAKE_GENERATOR_PLATFORM",
        "CMAKE_GENERATOR_TOOLSET",
        "CMAKE_MSVC_RUNTIME_LIBRARY",
        "BUILD_SHARED_LIBS",
        "GGML_BACKEND_DL",
        "OPENASR_VERIFIED_BACKEND_LOADING_ONLY",
        "GGML_CPU_ALL_VARIANTS",
        "GGML_NATIVE",
        "GGML_OPENMP",
        "GGML_CUDA",
        "GGML_CUDA_GRAPHS",
        "GGML_VULKAN",
        "GGML_HIP",
        "GGML_SYCL",
        "CMAKE_CUDA_ARCHITECTURES",
        "AMDGPU_TARGETS",
    ] {
        if let Some(value) = windows_cmake_cache::cache_value(&cache, name) {
            entries.insert(name.to_string(), value.to_string());
        }
    }
    for required in [
        "CMAKE_BUILD_TYPE",
        "CMAKE_GENERATOR",
        "BUILD_SHARED_LIBS",
        "GGML_BACKEND_DL",
        "OPENASR_VERIFIED_BACKEND_LOADING_ONLY",
        "GGML_CPU_ALL_VARIANTS",
        "GGML_NATIVE",
        "GGML_OPENMP",
        "GGML_CUDA",
        "GGML_CUDA_GRAPHS",
        "GGML_VULKAN",
        "GGML_HIP",
        "GGML_SYCL",
    ] {
        assert!(
            entries.contains_key(required),
            "configured Windows CMake cache is missing release contract field {required}"
        );
    }

    let mut compilers = BTreeMap::new();
    for (role, key, required) in [
        ("c", "CMAKE_C_COMPILER", true),
        ("cxx", "CMAKE_CXX_COMPILER", true),
        ("cuda", "CMAKE_CUDA_COMPILER", require_cuda_compiler),
    ] {
        match windows_cmake_cache::cache_value(&cache, key) {
            Some(path) => {
                compilers.insert(role.to_string(), windows_tool_identity(Path::new(path)));
            }
            None if required => panic!("configured Windows CMake cache is missing {key}"),
            None => {}
        }
    }
    let cmake_version = Command::new("cmake")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|text| text.lines().next().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string());

    serde_json::json!({
        "schema_version": 1,
        "cmake_version": cmake_version,
        "entries": entries,
        "compilers": compilers,
    })
}

fn windows_tool_identity(path: &Path) -> serde_json::Value {
    let bytes = fs::read(path)
        .unwrap_or_else(|error| panic!("read configured compiler {}: {error}", path.display()));
    let filename = path
        .file_name()
        .expect("configured compiler has a filename")
        .to_string_lossy()
        .to_ascii_lowercase();
    serde_json::json!({
        "filename": filename,
        "sha256": sha256_hex(&bytes),
        "size_bytes": bytes.len(),
    })
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
        // CMake accepts forward slashes on Windows. Backslashes passed through
        // `-D` are serialized into generated CMake compiler files and parsed a
        // second time, where paths such as `D:\workspace` contain invalid
        // escapes (`\o`). Normalizing once keeps command-line and generated-file
        // parsing identical.
        .replace('\\', "/")
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

// Pure CUDA-arch-list parsing/defaults (`cuda_gpu_targets_from_raw`,
// `normalize_cuda_gpu_targets`, `DEFAULT_CUDA_GPU_TARGETS`) live in
// `src/cuda_targets.rs`, not here: that file is `include!`d verbatim (this
// build script cannot depend on the crate it configures) and is ALSO
// compiled as an ordinary `#[cfg(test)] mod` of this crate (see `lib.rs`),
// so its regression canary -- guarding the class of bug fixed in #255/#196
// where OpenASR's own CUDA arch-list default silently narrowed below
// vendored ggml's sm_75 floor -- actually runs under `cargo nextest`. A
// `#[cfg(test)] mod tests` living only in a build script is never collected
// by any test runner, so before this the canary had no teeth.
include!("src/cuda_targets.rs");

fn cuda_gpu_targets() -> String {
    cuda_gpu_targets_from_raw(env::var("OPENASR_CUDA_GPU_TARGETS").ok().as_deref())
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
        CudaTuning, HipTuning, ios_deployment_target_from, macos_deployment_target_from,
        parse_bool_env, resolve_ggml_native_enabled, target_is_x86, version_at_least,
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

    // The CUDA-arch-target parsing/default tests (including the sm_75-floor
    // regression canary for #255/#196) moved to `src/cuda_targets.rs`, which
    // this build script `include!`s -- see that file's doc comment for why:
    // a `#[cfg(test)] mod` defined only in a build script is never collected
    // by `cargo test`/`cargo nextest`, so it ran nowhere.

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
