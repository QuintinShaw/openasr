use std::env;

use super::{
    ApplePlatformHints, CpuArchitectureFamily, CpuCapabilities, ProviderAvailabilityState,
};

pub(super) fn normalized_target_os(os: &str) -> &str {
    match os {
        "macos" => "darwin",
        other => other,
    }
}

pub(super) fn detect_cpu_capabilities() -> CpuCapabilities {
    CpuCapabilities {
        architecture: env::consts::ARCH.to_string(),
        family: cpu_family(),
        features: detect_cpu_features(),
    }
}

fn cpu_family() -> CpuArchitectureFamily {
    match env::consts::ARCH {
        "x86_64" => CpuArchitectureFamily::X86_64,
        "aarch64" => CpuArchitectureFamily::Aarch64,
        _ => CpuArchitectureFamily::Other,
    }
}

fn detect_cpu_features() -> Vec<String> {
    let mut features = Vec::new();
    detect_x86_features(&mut features);
    detect_aarch64_features(&mut features);
    features.sort();
    features.dedup();
    features
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_x86_features(features: &mut Vec<String>) {
    if std::is_x86_feature_detected!("sse2") {
        features.push("sse2".to_string());
    }
    if std::is_x86_feature_detected!("sse3") {
        features.push("sse3".to_string());
    }
    if std::is_x86_feature_detected!("ssse3") {
        features.push("ssse3".to_string());
    }
    if std::is_x86_feature_detected!("sse4.1") {
        features.push("sse4.1".to_string());
    }
    if std::is_x86_feature_detected!("sse4.2") {
        features.push("sse4.2".to_string());
    }
    if std::is_x86_feature_detected!("avx") {
        features.push("avx".to_string());
    }
    if std::is_x86_feature_detected!("avx2") {
        features.push("avx2".to_string());
    }
    if std::is_x86_feature_detected!("fma") {
        features.push("fma".to_string());
    }
    if std::is_x86_feature_detected!("f16c") {
        features.push("f16c".to_string());
    }
    if std::is_x86_feature_detected!("bmi1") {
        features.push("bmi1".to_string());
    }
    if std::is_x86_feature_detected!("bmi2") {
        features.push("bmi2".to_string());
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_x86_features(_features: &mut [String]) {}

#[cfg(target_arch = "aarch64")]
fn detect_aarch64_features(features: &mut Vec<String>) {
    if std::arch::is_aarch64_feature_detected!("neon") {
        features.push("neon".to_string());
    }
    if std::arch::is_aarch64_feature_detected!("crc") {
        features.push("crc".to_string());
    }
    if std::arch::is_aarch64_feature_detected!("aes") {
        features.push("aes".to_string());
    }
    if std::arch::is_aarch64_feature_detected!("sha2") {
        features.push("sha2".to_string());
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn detect_aarch64_features(_features: &mut [String]) {}

pub(super) fn detect_apple_platform_hints() -> ApplePlatformHints {
    apple_platform_hints_for(env::consts::OS, env::consts::ARCH)
}

fn apple_platform_hints_for(os: &str, arch: &str) -> ApplePlatformHints {
    let is_apple_platform = matches!(os, "macos" | "ios");
    let is_apple_silicon = os == "macos" && matches!(arch, "aarch64" | "arm64ec");
    let platform_state = if is_apple_platform {
        ProviderAvailabilityState::Available
    } else {
        ProviderAvailabilityState::Unavailable
    };
    let apple_silicon_state = if is_apple_silicon {
        ProviderAvailabilityState::Available
    } else if os == "macos" {
        ProviderAvailabilityState::Unknown
    } else {
        ProviderAvailabilityState::Unavailable
    };

    let mut notes = vec![
        "Apple platform hints are compile-target hints only; they do not prove Metal or Core ML execution."
            .to_string(),
    ];
    if is_apple_silicon {
        notes.push("Compilation target appears to be Apple Silicon.".to_string());
    } else if os == "macos" {
        notes.push("macOS target is not a direct Apple Silicon target; Rosetta or Intel host state is not probed by M58A.".to_string());
    }

    ApplePlatformHints {
        platform_state,
        apple_silicon_state,
        notes,
    }
}
