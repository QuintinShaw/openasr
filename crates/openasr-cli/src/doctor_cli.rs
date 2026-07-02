use super::*;

pub(super) fn print_native_doctor() {
    println!("- native: local GGUF runtime source only; unsupported paths fail closed");
}

pub(super) fn print_runtime_doctor() {
    println!("- mock: built-in (not-required)");
    println!("- native: ggml runtime path (local source + strict fail-closed policy)");
    print_ggml_runtime_doctor();
}

fn print_ggml_runtime_doctor() {
    let info = openasr_core::ggml_runtime_info();
    let best_backend = info.best_backend_name.as_deref().unwrap_or("unavailable");
    println!(
        "- ggml: best backend {best_backend}; CPU backend {}; native CPU tune {}",
        info.cpu_backend_name,
        on_off(openasr_core::ggml_native_build_enabled())
    );
    println!(
        "  - CPU features: {}",
        format_cpu_features(&info.cpu_features)
    );
    if let Some(summary) = openasr_core::ggml_hip_tuning_summary() {
        println!("  - HIP tuning: {summary}");
    }
    if info.devices.is_empty() {
        println!("  - devices: none reported");
        return;
    }
    println!("  - devices:");
    for device in info.devices {
        let memory = device.memory.as_ref().map_or_else(
            || "memory unknown".to_string(),
            |memory| {
                format!(
                    "{} MiB free / {} MiB total",
                    bytes_to_mib(memory.free_bytes),
                    bytes_to_mib(memory.total_bytes)
                )
            },
        );
        println!(
            "    - {} ({}, {memory})",
            device.name,
            device_kind_label(device.kind)
        );
        let supported = device
            .supported_matmul_weight_types()
            .into_iter()
            .filter_map(|(name, ok)| ok.then_some(name))
            .collect::<Vec<_>>()
            .join(" ");
        let supported = if supported.is_empty() {
            "none".to_string()
        } else {
            supported
        };
        println!("      matmul weight types: {supported}");
    }
}

fn device_kind_label(kind: openasr_core::GgmlBackendKind) -> &'static str {
    match kind {
        openasr_core::GgmlBackendKind::Cpu => "cpu",
        openasr_core::GgmlBackendKind::Gpu => "gpu",
        openasr_core::GgmlBackendKind::IntegratedGpu => "integrated-gpu",
        openasr_core::GgmlBackendKind::Accelerator => "accelerator",
        openasr_core::GgmlBackendKind::Meta => "meta",
        openasr_core::GgmlBackendKind::Unknown(_) => "unknown",
    }
}

fn bytes_to_mib(bytes: usize) -> usize {
    bytes / (1024 * 1024)
}

fn on_off(enabled: bool) -> &'static str {
    if enabled { "on" } else { "off" }
}

fn format_cpu_features(features: &openasr_core::GgmlCpuFeatures) -> String {
    let mut enabled = Vec::new();
    push_feature(&mut enabled, features.sse3, "sse3");
    push_feature(&mut enabled, features.ssse3, "ssse3");
    push_feature(&mut enabled, features.avx, "avx");
    push_feature(&mut enabled, features.avx_vnni, "avx-vnni");
    push_feature(&mut enabled, features.avx2, "avx2");
    push_feature(&mut enabled, features.fma, "fma");
    push_feature(&mut enabled, features.f16c, "f16c");
    push_feature(&mut enabled, features.bmi2, "bmi2");
    push_feature(&mut enabled, features.avx512, "avx512");
    push_feature(&mut enabled, features.avx512_vbmi, "avx512-vbmi");
    push_feature(&mut enabled, features.avx512_vnni, "avx512-vnni");
    push_feature(&mut enabled, features.avx512_bf16, "avx512-bf16");
    push_feature(&mut enabled, features.amx_int8, "amx-int8");
    push_feature(&mut enabled, features.neon, "neon");
    push_feature(&mut enabled, features.arm_fma, "arm-fma");
    push_feature(&mut enabled, features.fp16_va, "fp16-va");
    push_feature(&mut enabled, features.dotprod, "dotprod");
    push_feature(&mut enabled, features.matmul_int8, "matmul-int8");
    push_feature(&mut enabled, features.sve, "sve");
    push_feature(&mut enabled, features.sme, "sme");
    push_feature(&mut enabled, features.riscv_v, "riscv-v");
    push_feature(&mut enabled, features.vsx, "vsx");
    push_feature(&mut enabled, features.vxe, "vxe");
    push_feature(&mut enabled, features.wasm_simd, "wasm-simd");
    push_feature(&mut enabled, features.llamafile, "llamafile");

    if enabled.is_empty() {
        "portable".to_string()
    } else {
        enabled.join(",")
    }
}

fn push_feature(features: &mut Vec<&'static str>, enabled: bool, name: &'static str) {
    if enabled {
        features.push(name);
    }
}

pub(super) fn binary_status(path: Option<&Path>) -> &'static str {
    let Some(path) = path else {
        return "missing";
    };
    let Ok(metadata) = fs::metadata(path) else {
        return "missing";
    };
    if !metadata.is_file() {
        return "missing";
    }
    if is_executable(&metadata) {
        "ok"
    } else {
        "not executable"
    }
}

pub(super) fn print_optional_audio_tool(tool: &str) {
    match find_in_path(tool) {
        Some(path) => println!(
            "- {tool}: optional, not required (found at {})",
            path.display()
        ),
        None => println!("- {tool}: optional, not required (missing)"),
    }
}

pub(super) fn print_ffmpeg_doctor(config: &OpenAsrConfig) {
    let configured = env_path(OPENASR_FFMPEG_BIN)
        .or_else(|| config.media.ffmpeg_bin.as_ref().map(PathBuf::from));
    if let Some(path) = configured {
        println!(
            "- ffmpeg: optional for WAV/mock; used to prepare recognized non-WAV inputs for the native backend (configured at {}; {})",
            path.display(),
            binary_status(Some(&path))
        );
        return;
    }

    match find_in_path("ffmpeg") {
        Some(path) => println!(
            "- ffmpeg: optional for WAV/mock; used to prepare recognized non-WAV inputs for the native backend (found at {})",
            path.display()
        ),
        None => println!(
            "- ffmpeg: optional for WAV/mock; required only when the native backend must prepare recognized non-WAV inputs (missing)"
        ),
    }
}

pub(super) fn find_in_path(tool: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    for dir in env::split_paths(&path) {
        for candidate_name in executable_candidate_names(tool) {
            let candidate = dir.join(candidate_name);
            if is_executable_file(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

pub(super) fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    metadata.is_file() && is_executable(&metadata)
}

#[cfg(windows)]
pub(super) fn executable_candidate_names(tool: &str) -> Vec<String> {
    let pathext = env::var_os("PATHEXT")
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| ".COM;.EXE;.BAT;.CMD".to_string());
    let mut names = vec![tool.to_string()];
    names.extend(
        pathext
            .split(';')
            .filter(|extension| !extension.is_empty())
            .map(|extension| format!("{tool}{extension}")),
    );
    names
}

#[cfg(not(windows))]
pub(super) fn executable_candidate_names(tool: &str) -> Vec<String> {
    vec![tool.to_string()]
}

#[cfg(unix)]
pub(super) fn is_executable(metadata: &fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;

    metadata.permissions().mode() & 0o111 != 0
}

#[cfg(windows)]
pub(super) fn is_executable(_metadata: &fs::Metadata) -> bool {
    // Windows has no executable permission bit; runnability is decided by the
    // file extension (PATHEXT). `find_in_path` only ever probes candidates whose
    // names carry a PATHEXT extension (see `executable_candidate_names`), and an
    // explicitly configured tool path (e.g. ffmpeg.exe) carries one too. The
    // caller (`is_executable_file` / `binary_status`) has already confirmed the
    // target is a regular file, so treat any such file as executable. Returning
    // `false` here — the previous `not(unix)` behavior — made `find_in_path`
    // always yield `None`, so a Windows user with ffmpeg on PATH could not
    // transcribe non-WAV input and `doctor` reported it missing.
    true
}

#[cfg(not(any(unix, windows)))]
pub(super) fn is_executable(_metadata: &fs::Metadata) -> bool {
    false
}

// Windows-only: gate the whole module so Linux/macOS builds (where every test
// here is #[cfg(windows)]) don't see an otherwise-unused `use super::*`, which
// would trip `unused_imports` under the workspace's `-D warnings` CI.
#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn windows_regular_file_is_executable_so_find_in_path_resolves_ffmpeg() {
        let temp = tempfile::tempdir().unwrap();
        let exe = temp.path().join("ffmpeg.exe");
        std::fs::write(&exe, b"stub").unwrap();

        // Regression: is_executable() was hardcoded false on Windows, so
        // is_executable_file() and thus find_in_path("ffmpeg") always failed —
        // a Windows user with ffmpeg on PATH could not transcribe non-WAV input.
        assert!(
            is_executable_file(&exe),
            "a regular .exe file must count as executable on Windows"
        );
        // A directory and a missing path are never executable files.
        assert!(!is_executable_file(temp.path()));
        assert!(!is_executable_file(&temp.path().join("absent.exe")));
    }

    #[test]
    fn windows_candidate_names_include_pathext_extensions() {
        let names = executable_candidate_names("ffmpeg");
        assert!(names.contains(&"ffmpeg".to_string()));
        assert!(
            names.iter().any(|n| n.eq_ignore_ascii_case("ffmpeg.exe")),
            "PATHEXT candidates must include ffmpeg.exe, got {names:?}"
        );
    }
}
