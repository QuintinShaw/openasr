//! Canonical compute-device enumeration for the UI execution-target picker.
//!
//! The device list a desktop/mobile shell shows in its settings ("Auto", "CPU",
//! and any accelerated GPU backend) must reflect the ggml runtime of the
//! **process that actually runs inference** -- the daemon/sidecar -- not whoever
//! happened to ask. On platforms where the shell and the inference process are
//! built in different backend shapes (e.g. a CPU-only desktop shell supervising
//! a Vulkan sidecar on Windows), asking the shell's own [`GgmlRuntimeInfo`]
//! enumerates the wrong process and hides the GPU. Keeping the shaping here in
//! open core lets the server expose it over its local HTTP API (authoritative,
//! runs in the inference process) while a shell can still call the same function
//! for an offline fallback -- one vocabulary, no drift.

use serde::Serialize;

use crate::ggml_runtime::{GgmlBackendKind, GgmlRuntimeInfo};

const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// One selectable execution target for the UI picker, derived from the ggml
/// runtime. `id`/`kind`/`target` use the stable wire vocabulary
/// (`auto`/`cpu`/`accelerated`) the desktop `ExecutionTarget` mirrors;
/// `effective_target` is what `auto` actually resolves to on this machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ComputeDevice {
    pub id: String,
    pub name: String,
    pub meta: String,
    pub kind: String,
    pub target: String,
    pub effective_target: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<String>,
}

/// Build the canonical `Auto` + `CPU` (+ optional `Accelerated`) device list
/// from a ggml runtime snapshot. `Auto` resolves to the accelerated backend
/// when one is present, otherwise CPU. The accelerated entry is emitted only
/// when the runtime reports a GPU device, so a CPU-only runtime yields exactly
/// `Auto` + `CPU`.
pub fn compute_devices_from_runtime(runtime: &GgmlRuntimeInfo) -> Vec<ComputeDevice> {
    let cpu_name = cpu_device_name(runtime);
    let accelerated = runtime
        .devices
        .iter()
        .find(|device| device.kind.is_gpu())
        .map(|device| {
            let name = non_empty_device_label(&device.description, &device.name, "Accelerated");
            ComputeDevice {
                id: "accelerated".to_string(),
                name,
                meta: format!("{} backend", backend_kind_label(device.kind)),
                kind: "accelerated".to_string(),
                target: "accelerated".to_string(),
                effective_target: "accelerated".to_string(),
                memory: device.memory.map(|memory| format_gib(memory.total_bytes)),
            }
        });

    let auto_effective_target = accelerated
        .as_ref()
        .map(|_| "accelerated")
        .unwrap_or("cpu")
        .to_string();
    let auto_name = accelerated
        .as_ref()
        .map(|device| device.name.clone())
        .unwrap_or_else(|| cpu_name.clone());

    let mut devices = vec![
        ComputeDevice {
            id: "auto".to_string(),
            name: auto_name,
            meta: "best available backend".to_string(),
            kind: "auto".to_string(),
            target: "auto".to_string(),
            effective_target: auto_effective_target,
            memory: None,
        },
        ComputeDevice {
            id: "cpu".to_string(),
            name: cpu_name,
            meta: "local CPU backend".to_string(),
            kind: "cpu".to_string(),
            target: "cpu".to_string(),
            effective_target: "cpu".to_string(),
            memory: None,
        },
    ];

    if let Some(accelerated) = accelerated {
        devices.push(accelerated);
    }

    devices
}

/// The effective target the `Auto` entry resolves to (`accelerated` when a GPU
/// is present, else `cpu`). Falls back to `cpu` on an empty list.
pub fn default_execution_target(devices: &[ComputeDevice]) -> String {
    devices
        .iter()
        .find(|device| device.target == "auto")
        .map(|device| device.effective_target.clone())
        .unwrap_or_else(|| "cpu".to_string())
}

fn cpu_device_name(runtime: &GgmlRuntimeInfo) -> String {
    runtime
        .devices
        .iter()
        .find(|device| device.kind == GgmlBackendKind::Cpu)
        .map(|device| device.description.trim())
        .filter(|description| !description.is_empty())
        .or_else(|| {
            (!runtime.cpu_backend_name.trim().is_empty()
                && runtime.cpu_backend_name != "unavailable")
                .then_some(runtime.cpu_backend_name.trim())
        })
        .unwrap_or("CPU")
        .to_string()
}

fn non_empty_device_label(description: &str, name: &str, fallback: &str) -> String {
    let description = description.trim();
    if !description.is_empty() {
        return description.to_string();
    }
    let name = name.trim();
    if !name.is_empty() {
        return name.to_string();
    }
    fallback.to_string()
}

fn backend_kind_label(kind: GgmlBackendKind) -> &'static str {
    match kind {
        GgmlBackendKind::Cpu => "CPU",
        GgmlBackendKind::Gpu => "GPU",
        GgmlBackendKind::IntegratedGpu => "integrated GPU",
        GgmlBackendKind::Accelerator => "accelerator",
        GgmlBackendKind::Meta => "metadata",
        GgmlBackendKind::Unknown(_) => "unknown",
    }
}

fn format_gib(bytes: usize) -> String {
    format!("{:.0} GB", bytes as f64 / BYTES_PER_GIB)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgmlBackendDevice, GgmlCpuFeatures, GgmlDeviceMemory};

    fn runtime_with(devices: Vec<GgmlBackendDevice>, cpu_backend_name: &str) -> GgmlRuntimeInfo {
        GgmlRuntimeInfo {
            cpu_backend_name: cpu_backend_name.to_string(),
            best_backend_name: None,
            metal_backend_name: None,
            devices,
            cpu_features: GgmlCpuFeatures::default(),
        }
    }

    #[test]
    fn cpu_only_runtime_yields_auto_and_cpu_resolving_to_cpu() {
        let runtime = runtime_with(
            vec![GgmlBackendDevice::for_test(
                "CPU",
                "AMD Ryzen 9",
                GgmlBackendKind::Cpu,
                None,
            )],
            "CPU",
        );
        let devices = compute_devices_from_runtime(&runtime);
        let ids: Vec<_> = devices.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, ["auto", "cpu"], "no GPU -> exactly auto + cpu");
        assert_eq!(default_execution_target(&devices), "cpu");
        let auto = &devices[0];
        assert_eq!(auto.effective_target, "cpu");
        assert_eq!(auto.name, "AMD Ryzen 9");
        assert_eq!(devices[1].name, "AMD Ryzen 9");
    }

    #[test]
    fn gpu_runtime_adds_accelerated_and_auto_resolves_to_it() {
        // A Windows Vulkan sidecar reports both a CPU and a GPU device: the
        // picker must surface the accelerated entry and make Auto resolve to it,
        // which is exactly what the CPU-only desktop shell could not see.
        let runtime = runtime_with(
            vec![
                GgmlBackendDevice::for_test("CPU", "Intel Core", GgmlBackendKind::Cpu, None),
                GgmlBackendDevice::for_test(
                    "Vulkan0",
                    "NVIDIA GeForce RTX 4070",
                    GgmlBackendKind::Gpu,
                    Some(GgmlDeviceMemory {
                        free_bytes: 8 * 1024 * 1024 * 1024,
                        total_bytes: 12 * 1024 * 1024 * 1024,
                    }),
                ),
            ],
            "CPU",
        );
        let devices = compute_devices_from_runtime(&runtime);
        let ids: Vec<_> = devices.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, ["auto", "cpu", "accelerated"]);
        assert_eq!(default_execution_target(&devices), "accelerated");
        let accelerated = devices.iter().find(|d| d.id == "accelerated").unwrap();
        assert_eq!(accelerated.name, "NVIDIA GeForce RTX 4070");
        assert_eq!(accelerated.meta, "GPU backend");
        assert_eq!(accelerated.memory.as_deref(), Some("12 GB"));
        // Auto mirrors the accelerated device's label so the picker's default
        // reads as the GPU, not a bare "CPU".
        assert_eq!(devices[0].name, "NVIDIA GeForce RTX 4070");
    }

    #[test]
    fn cpu_name_falls_back_to_backend_name_then_placeholder() {
        // No CPU device row, unusable backend name -> literal "CPU".
        let runtime = runtime_with(vec![], "unavailable");
        let devices = compute_devices_from_runtime(&runtime);
        assert_eq!(devices[1].name, "CPU");
        // Backend name is used when the device description is missing.
        let runtime = runtime_with(
            vec![GgmlBackendDevice::for_test(
                "CPU",
                "",
                GgmlBackendKind::Cpu,
                None,
            )],
            "AVX2 CPU",
        );
        let devices = compute_devices_from_runtime(&runtime);
        assert_eq!(devices[1].name, "AVX2 CPU");
    }

    #[test]
    fn default_execution_target_falls_back_to_cpu_on_empty() {
        assert_eq!(default_execution_target(&[]), "cpu");
    }
}
