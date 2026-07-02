use std::{env, fmt};

use serde::{Deserialize, Serialize};

#[path = "capabilities_cpu_facts.rs"]
mod capabilities_cpu_facts;
#[path = "capabilities_diagnostics.rs"]
mod capabilities_diagnostics;
#[path = "capabilities_fallback_trace.rs"]
mod capabilities_fallback_trace;
#[path = "capabilities_provider_availability.rs"]
mod capabilities_provider_availability;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    pub target_os: String,
    pub target_architecture: String,
    pub cpu: CpuCapabilities,
    pub apple_platform: ApplePlatformHints,
    pub providers: Vec<ProviderAvailability>,
    pub fallback_policy: HardwareFallbackPolicy,
}

impl HardwareCapabilities {
    pub fn detect() -> Self {
        detect_hardware_capabilities()
    }

    pub fn provider(&self, provider: HardwareProvider) -> Option<&ProviderAvailability> {
        self.providers
            .iter()
            .find(|availability| availability.provider == provider)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuCapabilities {
    pub architecture: String,
    pub family: CpuArchitectureFamily,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuArchitectureFamily {
    X86_64,
    Aarch64,
    Other,
}

impl CpuArchitectureFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::X86_64 => "x86_64",
            Self::Aarch64 => "aarch64",
            Self::Other => "other",
        }
    }
}

impl fmt::Display for CpuArchitectureFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplePlatformHints {
    pub platform_state: ProviderAvailabilityState,
    pub apple_silicon_state: ProviderAvailabilityState,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderAvailability {
    pub provider: HardwareProvider,
    pub availability: ProviderAvailabilityState,
    pub native_core_integration: ProviderAvailabilityState,
    pub checked_hints: Vec<String>,
    pub detected_hints: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareProvider {
    Cpu,
    #[serde(rename = "apple-silicon")]
    AppleSilicon,
    Metal,
    #[serde(rename = "coreml")]
    CoreMl,
    Cuda,
    Rocm,
    Vulkan,
    #[serde(rename = "openvino")]
    OpenVino,
    #[serde(rename = "directml")]
    DirectMl,
    Sycl,
}

impl HardwareProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::AppleSilicon => "apple-silicon",
            Self::Metal => "metal",
            Self::CoreMl => "coreml",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Vulkan => "vulkan",
            Self::OpenVino => "openvino",
            Self::DirectMl => "directml",
            Self::Sycl => "sycl",
        }
    }
}

impl fmt::Display for HardwareProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderAvailabilityState {
    Available,
    Unavailable,
    Unknown,
    NotBuilt,
}

impl ProviderAvailabilityState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Available => "available",
            Self::Unavailable => "unavailable",
            Self::Unknown => "unknown",
            Self::NotBuilt => "not_built",
        }
    }
}

impl fmt::Display for ProviderAvailabilityState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareFallbackPolicy {
    pub default_execution_target: HardwareProvider,
    pub accelerated_provider_integration: ProviderAvailabilityState,
    pub automatic_provider_selection: bool,
    pub external_probe_commands: bool,
    pub benchmark_gate_required: bool,
    pub notes: Vec<String>,
}

impl HardwareFallbackPolicy {
    pub fn conservative_cpu_default() -> Self {
        capabilities_fallback_trace::conservative_cpu_default()
    }
}

pub fn detect_hardware_capabilities() -> HardwareCapabilities {
    let target_os = capabilities_cpu_facts::normalized_target_os(env::consts::OS).to_string();
    let target_architecture = env::consts::ARCH.to_string();
    let cpu = capabilities_cpu_facts::detect_cpu_capabilities();
    let apple_platform = capabilities_cpu_facts::detect_apple_platform_hints();
    let providers =
        capabilities_provider_availability::detect_provider_availability(&apple_platform);

    HardwareCapabilities {
        target_os,
        target_architecture,
        cpu,
        apple_platform,
        providers,
        fallback_policy: HardwareFallbackPolicy::conservative_cpu_default(),
    }
}

#[cfg(test)]
mod tests {
    use std::{env, ffi::OsString};

    use super::*;

    #[test]
    fn hardware_detection_is_non_panicking_and_reports_target_basics() {
        let capabilities = detect_hardware_capabilities();

        assert!(!capabilities.target_os.trim().is_empty());
        assert!(!capabilities.target_architecture.trim().is_empty());
        assert_eq!(
            capabilities.target_os,
            capabilities_cpu_facts::normalized_target_os(env::consts::OS)
        );
        assert_eq!(capabilities.cpu.architecture, env::consts::ARCH);
        assert!(capabilities.provider(HardwareProvider::Cpu).is_some());
        assert!(capabilities.provider(HardwareProvider::Cuda).is_some());
    }

    #[test]
    fn unsupported_provider_placeholders_do_not_claim_native_core_integration() {
        let capabilities = detect_hardware_capabilities();

        for provider in [
            HardwareProvider::Cpu,
            HardwareProvider::AppleSilicon,
            HardwareProvider::Metal,
            HardwareProvider::CoreMl,
            HardwareProvider::Cuda,
            HardwareProvider::Rocm,
            HardwareProvider::Vulkan,
            HardwareProvider::OpenVino,
            HardwareProvider::DirectMl,
            HardwareProvider::Sycl,
        ] {
            let availability = capabilities.provider(provider).expect("provider present");
            if provider != HardwareProvider::Cpu && provider != HardwareProvider::AppleSilicon {
                if availability.availability == ProviderAvailabilityState::Available {
                    assert!(
                        !availability.detected_hints.is_empty(),
                        "available provider placeholders must be backed by a structured hint"
                    );
                } else {
                    assert!(matches!(
                        availability.availability,
                        ProviderAvailabilityState::Unavailable | ProviderAvailabilityState::Unknown
                    ));
                }
            }
            assert_eq!(
                availability.native_core_integration,
                expected_native_core_integration_for_provider(provider)
            );
        }
    }

    #[test]
    fn fallback_policy_reports_truthful_acceleration_state() {
        let policy = HardwareFallbackPolicy::conservative_cpu_default();

        assert_eq!(policy.default_execution_target, HardwareProvider::Cpu);
        // Acceleration state must mirror the actual device probe: GGML Metal
        // is built in and auto-selected where a GPU exists; without one the
        // policy reports Unavailable (never the old blanket NotBuilt).
        let accelerated_available = crate::ggml_available_devices()
            .iter()
            .any(|device| device.kind.is_gpu());
        if accelerated_available {
            assert_eq!(
                policy.accelerated_provider_integration,
                ProviderAvailabilityState::Available
            );
            assert!(policy.automatic_provider_selection);
        } else {
            assert_eq!(
                policy.accelerated_provider_integration,
                ProviderAvailabilityState::Unavailable
            );
            assert!(!policy.automatic_provider_selection);
        }
        assert!(!policy.external_probe_commands);
        assert!(policy.benchmark_gate_required);
    }

    #[test]
    fn serialized_and_debug_output_avoid_misleading_acceleration_claims() {
        let capabilities = detect_hardware_capabilities();
        let debug_output = format!("{capabilities:?}").to_lowercase();
        let json_output = serde_json::to_string(&capabilities)
            .expect("hardware capabilities serialize")
            .to_lowercase();

        for output in [debug_output, json_output] {
            assert!(!output.contains("hardware acceleration supported"));
            assert!(!output.contains("gpu supported"));
            assert!(!output.contains("cuda supported"));
            assert!(!output.contains("rocm supported"));
            assert!(!output.contains("openvino supported"));
        }
    }

    #[test]
    fn serialization_pins_public_state_and_provider_spellings() {
        let state =
            serde_json::to_value(ProviderAvailabilityState::NotBuilt).expect("state serializes");
        assert_eq!(state, serde_json::json!("not_built"));

        for provider in [
            HardwareProvider::Cpu,
            HardwareProvider::AppleSilicon,
            HardwareProvider::Metal,
            HardwareProvider::CoreMl,
            HardwareProvider::Cuda,
            HardwareProvider::Rocm,
            HardwareProvider::Vulkan,
            HardwareProvider::OpenVino,
            HardwareProvider::DirectMl,
            HardwareProvider::Sycl,
        ] {
            let value = serde_json::to_value(provider).expect("provider serializes");
            assert_eq!(value, serde_json::json!(provider.as_str()));
        }

        let policy = HardwareFallbackPolicy::conservative_cpu_default();
        let policy_json = serde_json::to_value(&policy).expect("policy serializes");
        assert_eq!(policy_json["default_execution_target"], "cpu");
        // Spelling pin only: the value is device-dependent (see the truthful
        // acceleration-state test); here we pin that it serializes to one of
        // the public snake_case spellings.
        assert!(matches!(
            policy_json["accelerated_provider_integration"].as_str(),
            Some("available" | "unavailable")
        ));
        assert_eq!(policy_json["external_probe_commands"], false);
        assert_eq!(policy_json["benchmark_gate_required"], true);
    }

    #[test]
    fn empty_environment_hints_are_ignored() {
        let provider = capabilities_provider_availability::env_hint_provider_with_lookup(
            HardwareProvider::Cuda,
            &["OPENASR_TEST_EMPTY_HINT", "OPENASR_TEST_PRESENT_HINT"],
            |name| match name {
                "OPENASR_TEST_EMPTY_HINT" => Some(OsString::new()),
                "OPENASR_TEST_PRESENT_HINT" => Some(OsString::from("1")),
                _ => None,
            },
        );

        assert_eq!(provider.availability, ProviderAvailabilityState::Available);
        assert_eq!(
            provider.native_core_integration,
            ProviderAvailabilityState::NotBuilt
        );
        assert_eq!(
            provider.detected_hints,
            vec!["OPENASR_TEST_PRESENT_HINT".to_string()]
        );
        assert!(
            provider
                .notes
                .iter()
                .any(|note| note.contains("OPENASR_TEST_PRESENT_HINT"))
        );
        assert!(
            !provider
                .notes
                .iter()
                .any(|note| note.contains("OPENASR_TEST_EMPTY_HINT"))
        );
    }

    #[test]
    fn rocm_native_integration_reflects_hip_link_state() {
        let capabilities = detect_hardware_capabilities();
        let rocm = capabilities
            .provider(HardwareProvider::Rocm)
            .expect("rocm provider present");

        if crate::ggml_hip_tuning_summary().is_some() {
            assert_eq!(
                rocm.native_core_integration,
                ProviderAvailabilityState::Available
            );
            assert!(rocm.notes.iter().any(|note| {
                note.contains("ROCm/HIP ggml backend is linked") && note.contains("HIP tuning")
            }));
        } else {
            assert_eq!(
                rocm.native_core_integration,
                ProviderAvailabilityState::NotBuilt
            );
        }
    }

    fn expected_native_core_integration_for_provider(
        provider: HardwareProvider,
    ) -> ProviderAvailabilityState {
        if provider == HardwareProvider::Rocm && crate::ggml_hip_tuning_summary().is_some() {
            ProviderAvailabilityState::Available
        } else {
            ProviderAvailabilityState::NotBuilt
        }
    }
}
