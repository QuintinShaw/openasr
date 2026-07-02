use std::{env, ffi::OsString};

use crate::{
    GgmlBackendDevice, GgmlBackendKind, ggml_available_devices, ggml_hip_tuning_summary,
    ggml_runtime::GgmlDeviceMemory,
};

use super::{
    ApplePlatformHints, HardwareProvider, ProviderAvailability, ProviderAvailabilityState,
    capabilities_diagnostics::{
        backend_not_linked_note, directml_placeholder_note, env_absent_note, env_present_note,
        rocm_backend_linked_no_device_note, rocm_backend_linked_with_devices_note,
    },
};

pub(super) fn detect_provider_availability(
    apple: &ApplePlatformHints,
) -> Vec<ProviderAvailability> {
    let mut providers = vec![cpu_provider(), apple_provider(apple)];
    providers.extend(
        [HardwareProvider::Metal, HardwareProvider::CoreMl]
            .into_iter()
            .map(|provider| mac_scoped_provider(provider, apple.platform_state)),
    );
    providers.extend(
        [
            (HardwareProvider::Cuda, &["CUDA_PATH", "CUDA_HOME"][..]),
            (HardwareProvider::Vulkan, &["VULKAN_SDK"][..]),
            (
                HardwareProvider::OpenVino,
                &["OPENVINO_DIR", "INTEL_OPENVINO_DIR"][..],
            ),
            (HardwareProvider::Sycl, &["ONEAPI_ROOT", "DPCPP_HOME"][..]),
        ]
        .into_iter()
        .map(|(provider, env_vars)| env_hint_provider(provider, env_vars)),
    );
    providers.push(rocm_provider());
    providers.push(directml_provider(env::consts::OS));
    providers
}

fn provider_availability(
    provider: HardwareProvider,
    availability: ProviderAvailabilityState,
    checked_hints: Vec<String>,
    detected_hints: Vec<String>,
    notes: Vec<String>,
) -> ProviderAvailability {
    ProviderAvailability {
        provider,
        availability,
        native_core_integration: ProviderAvailabilityState::NotBuilt,
        checked_hints,
        detected_hints,
        notes,
    }
}

fn cpu_provider() -> ProviderAvailability {
    simple_provider_v0(
        HardwareProvider::Cpu,
        ProviderAvailabilityState::Available,
        vec![
            "CPU is the conservative baseline target for planning and fallback.".to_string(),
            "Existing CPU-capable backends remain unchanged; M58A does not add a Native ASR Core CPU provider or provider selection path.".to_string(),
        ],
    )
}

fn apple_provider(apple: &ApplePlatformHints) -> ProviderAvailability {
    simple_provider_v0(
        HardwareProvider::AppleSilicon,
        apple.apple_silicon_state,
        vec![
            "Apple Silicon detection is a target hint, not an accelerated OpenASR runtime."
                .to_string(),
        ],
    )
}

fn mac_scoped_provider(
    provider: HardwareProvider,
    platform_state: ProviderAvailabilityState,
) -> ProviderAvailability {
    let availability = if platform_state == ProviderAvailabilityState::Available {
        ProviderAvailabilityState::Unknown
    } else {
        ProviderAvailabilityState::Unavailable
    };
    simple_provider_v0(provider, availability, backend_not_linked_note())
}

fn directml_provider(os: &str) -> ProviderAvailability {
    let availability = if os == "windows" {
        ProviderAvailabilityState::Unknown
    } else {
        ProviderAvailabilityState::Unavailable
    };
    simple_provider_v0(
        HardwareProvider::DirectMl,
        availability,
        directml_placeholder_note(),
    )
}

fn env_hint_provider(provider: HardwareProvider, env_vars: &[&str]) -> ProviderAvailability {
    env_hint_provider_with_lookup(provider, env_vars, |name| env::var_os(name))
}

fn rocm_provider() -> ProviderAvailability {
    let env_provider = env_hint_provider(
        HardwareProvider::Rocm,
        &["ROCM_PATH", "ROCM_HOME", "HIP_PATH"],
    );
    let Some(tuning) = ggml_hip_tuning_summary() else {
        return env_provider;
    };

    let devices = rocm_devices_from_ggml_devices(ggml_available_devices());
    rocm_provider_from_linked_devices(env_provider, tuning, devices)
}

fn rocm_provider_from_linked_devices(
    env_provider: ProviderAvailability,
    tuning: &str,
    devices: Vec<RocmDeviceHint>,
) -> ProviderAvailability {
    let availability = if devices.is_empty() {
        env_provider.availability
    } else {
        ProviderAvailabilityState::Available
    };
    let detected_hints = if devices.is_empty() {
        env_provider.detected_hints
    } else {
        devices.iter().map(RocmDeviceHint::detected_hint).collect()
    };
    let notes = if devices.is_empty() {
        rocm_backend_linked_no_device_note(tuning)
    } else {
        rocm_backend_linked_with_devices_note(tuning, &device_summaries(&devices))
    };

    ProviderAvailability {
        provider: HardwareProvider::Rocm,
        availability,
        native_core_integration: ProviderAvailabilityState::Available,
        checked_hints: env_provider.checked_hints,
        detected_hints,
        notes,
    }
}

pub(super) fn env_hint_provider_with_lookup(
    provider: HardwareProvider,
    env_vars: &[&str],
    lookup: impl Fn(&str) -> Option<OsString>,
) -> ProviderAvailability {
    let present: Vec<String> = env_vars
        .iter()
        .copied()
        .filter(|name| env_value_present(lookup(name)))
        .map(str::to_string)
        .collect();

    let availability = if present.is_empty() {
        ProviderAvailabilityState::Unknown
    } else {
        ProviderAvailabilityState::Available
    };
    let notes = if present.is_empty() {
        env_absent_note()
    } else {
        env_present_note(&present)
    };

    provider_availability(
        provider,
        availability,
        env_vars.iter().map(|name| (*name).to_string()).collect(),
        present,
        notes,
    )
}

fn env_value_present(value: Option<OsString>) -> bool {
    value.is_some_and(|value| !value.is_empty())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RocmDeviceHint {
    name: String,
    description: String,
    kind: GgmlBackendKind,
    memory: Option<GgmlDeviceMemory>,
}

impl RocmDeviceHint {
    fn from_device(device: GgmlBackendDevice) -> Option<Self> {
        is_rocm_ggml_device(&device).then_some(Self {
            name: device.name,
            description: device.description,
            kind: device.kind,
            memory: device.memory,
        })
    }

    fn detected_hint(&self) -> String {
        format!("ggml:{}", self.name)
    }
}

fn rocm_devices_from_ggml_devices(devices: Vec<GgmlBackendDevice>) -> Vec<RocmDeviceHint> {
    devices
        .into_iter()
        .filter_map(RocmDeviceHint::from_device)
        .collect()
}

fn is_rocm_ggml_device(device: &GgmlBackendDevice) -> bool {
    if !device.kind.is_gpu() {
        return false;
    }
    let name = device.name.to_ascii_lowercase();
    let description = device.description.to_ascii_lowercase();
    name.contains("rocm")
        || name.contains("hip")
        || name.contains("amd")
        || description.contains("rocm")
        || description.contains("hip")
        || description.contains("amd")
}

fn device_summaries(devices: &[RocmDeviceHint]) -> Vec<String> {
    devices.iter().map(device_summary).collect()
}

fn device_summary(device: &RocmDeviceHint) -> String {
    let memory = device
        .memory
        .map(|memory| {
            format!(
                "{} MiB free / {} MiB total",
                bytes_to_mib(memory.free_bytes),
                bytes_to_mib(memory.total_bytes)
            )
        })
        .unwrap_or_else(|| "memory unknown".to_string());
    format!(
        "{} ({:?}, {memory})",
        non_empty_or(&device.description, &device.name),
        device.kind
    )
}

fn bytes_to_mib(bytes: usize) -> usize {
    bytes / (1024 * 1024)
}

fn non_empty_or<'a>(value: &'a str, fallback: &'a str) -> &'a str {
    if value.trim().is_empty() {
        fallback
    } else {
        value
    }
}

fn simple_provider_v0(
    provider: HardwareProvider,
    availability: ProviderAvailabilityState,
    notes: Vec<String>,
) -> ProviderAvailability {
    provider_availability(provider, availability, Vec::new(), Vec::new(), notes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_linked_provider_reports_native_integration_and_devices() {
        let env_provider = provider_availability(
            HardwareProvider::Rocm,
            ProviderAvailabilityState::Unknown,
            vec!["HIP_PATH".to_string()],
            Vec::new(),
            env_absent_note(),
        );
        let provider = rocm_provider_from_linked_devices(
            env_provider,
            "graphs=on",
            vec![RocmDeviceHint {
                name: "ROCm0".to_string(),
                description: "AMD Radeon RX 9060 XT".to_string(),
                kind: GgmlBackendKind::Gpu,
                memory: Some(GgmlDeviceMemory {
                    free_bytes: 15 * 1024 * 1024,
                    total_bytes: 16 * 1024 * 1024,
                }),
            }],
        );

        assert_eq!(provider.provider, HardwareProvider::Rocm);
        assert_eq!(provider.availability, ProviderAvailabilityState::Available);
        assert_eq!(
            provider.native_core_integration,
            ProviderAvailabilityState::Available
        );
        assert_eq!(provider.detected_hints, vec!["ggml:ROCm0".to_string()]);
        assert!(provider.notes.iter().any(|note| {
            note.contains("ROCm/HIP ggml backend is linked")
                && note.contains("AMD Radeon RX 9060 XT")
                && note.contains("graphs=on")
        }));
    }

    #[test]
    fn rocm_linked_provider_without_device_keeps_env_availability_but_marks_linked() {
        let env_provider = provider_availability(
            HardwareProvider::Rocm,
            ProviderAvailabilityState::Unknown,
            vec!["HIP_PATH".to_string()],
            Vec::new(),
            env_absent_note(),
        );
        let provider = rocm_provider_from_linked_devices(env_provider, "graphs=on", Vec::new());

        assert_eq!(provider.availability, ProviderAvailabilityState::Unknown);
        assert_eq!(
            provider.native_core_integration,
            ProviderAvailabilityState::Available
        );
        assert!(provider.detected_hints.is_empty());
        assert!(provider.notes.iter().any(|note| {
            note.contains("ROCm/HIP ggml backend is linked")
                && note.contains("did not report a ROCm/HIP GPU device")
        }));
    }

    #[test]
    fn rocm_env_hint_without_linked_backend_remains_not_built() {
        let provider = env_hint_provider_with_lookup(
            HardwareProvider::Rocm,
            &["ROCM_PATH", "HIP_PATH"],
            |name| match name {
                "HIP_PATH" => Some(OsString::from("C:\\Program Files\\AMD\\ROCm\\7.1")),
                _ => None,
            },
        );

        assert_eq!(provider.availability, ProviderAvailabilityState::Available);
        assert_eq!(
            provider.native_core_integration,
            ProviderAvailabilityState::NotBuilt
        );
        assert_eq!(provider.detected_hints, vec!["HIP_PATH".to_string()]);
    }
}
