//! Canonical compute-device enumeration for the UI execution-target picker.
//!
//! The device list a desktop/mobile shell shows in its settings ("Auto", "CPU",
//! and any accelerated GPU backend) must reflect the ggml runtime of the
//! **process that actually runs inference** -- the daemon/sidecar -- not whoever
//! happened to ask. A Windows shell that inspects itself instead of the
//! CPU-neutral inference host after it activates a signed optional provider
//! enumerates the wrong process and hides that GPU. Keeping the shaping here in
//! open core lets the server expose it over its local HTTP API (authoritative,
//! runs in the inference process) while a shell can still call the same function
//! for an offline fallback -- one vocabulary, no drift.

use serde::Serialize;

use crate::device::execution_route::physical_gpu_identities_from_ggml;
use crate::ggml_runtime::{GgmlBackendKind, GgmlRuntimeInfo, preferred_accelerated_device};

const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// One selectable execution target for the UI picker, derived from the ggml
/// runtime. Coarse `id`/`kind`/`target` stay `auto`/`cpu`/`accelerated` for
/// desktop compatibility; physical GPU rows use `kind: "gpu"` and a stable
/// id. `effective_target` is what `auto` actually resolves to on this machine.
// TS export for the `/v1/devices` HTTP wire contract (openasr-server's
// `DevicesResponse` pulls this type in): gated to `cfg(test)` so ts-rs is a
// dev-only dependency, never part of the shipped rlib. See
// crates/openasr-server/src/http_wire_bindings_test.rs for the golden
// "regenerate == committed" guard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[cfg_attr(any(test, feature = "ts-export"), derive(ts_rs::TS))]
#[cfg_attr(
    any(test, feature = "ts-export"),
    ts(export_to = "generated/http-wire/")
)]
pub struct ComputeDevice {
    pub id: String,
    pub name: String,
    pub meta: String,
    pub kind: String,
    pub target: String,
    pub effective_target: String,
    /// Typed provider identity of the concrete device behind this row. This
    /// is independent from the coarse `accelerated` target and lets a shell
    /// attest that the Activated-only optional provider actually loaded instead
    /// of mistaking an unactivated GPU device for activation success.
    pub provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<String>,
    /// Present on physical GPU rows. Omitted on coarse auto/cpu/accelerated rows.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_total_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_free_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selectable: Option<bool>,
}

/// Build the canonical `Auto` + `CPU` (+ optional `Accelerated`) device list
/// from a ggml runtime snapshot, then one `kind=gpu` row per physical GPU.
/// The accelerated entry is emitted only when the runtime reports a GPU
/// device, so a CPU-only runtime yields exactly `Auto` + `CPU`.
pub fn compute_devices_from_runtime(runtime: &GgmlRuntimeInfo) -> Vec<ComputeDevice> {
    let cpu_name = cpu_device_name(runtime);
    // On a hybrid-graphics host (Optimus-style: an integrated + a discrete
    // GPU both surfaced by the same Vulkan backend), prefer the discrete GPU
    // -- see `preferred_accelerated_device`'s doc comment for the ranking.
    let accelerated =
        preferred_accelerated_device(&runtime.devices, GgmlBackendKind::is_gpu).map(|device| {
            let name = non_empty_device_label(&device.description, &device.name, "Accelerated");
            ComputeDevice {
                id: "accelerated".to_string(),
                name,
                meta: format!("{} backend", backend_kind_label(device.kind)),
                kind: "accelerated".to_string(),
                target: "accelerated".to_string(),
                effective_target: "accelerated".to_string(),
                provider: crate::ExecutionProvider::from_backend_name(&device.name)
                    .as_str()
                    .to_string(),
                memory: device.memory.map(|memory| format_gib(memory.total_bytes)),
                memory_total_bytes: None,
                memory_free_bytes: None,
                selectable: None,
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
            provider: accelerated
                .as_ref()
                .map(|device| device.provider.clone())
                .unwrap_or_else(|| "cpu".to_string()),
            memory: None,
            memory_total_bytes: None,
            memory_free_bytes: None,
            selectable: None,
        },
        ComputeDevice {
            id: "cpu".to_string(),
            name: cpu_name,
            meta: "local CPU backend".to_string(),
            kind: "cpu".to_string(),
            target: "cpu".to_string(),
            effective_target: "cpu".to_string(),
            provider: "cpu".to_string(),
            memory: None,
            memory_total_bytes: None,
            memory_free_bytes: None,
            selectable: None,
        },
    ];

    if let Some(accelerated) = accelerated {
        devices.push(accelerated);
    }

    devices.extend(physical_gpu_rows(runtime));
    devices
}

fn physical_gpu_rows(runtime: &GgmlRuntimeInfo) -> Vec<ComputeDevice> {
    physical_gpu_identities_from_ggml(&runtime.devices)
        .into_iter()
        .map(|gpu| {
            let meta = gpu.id_limitation.unwrap_or("physical GPU").to_string();
            ComputeDevice {
                id: gpu.public_id.clone(),
                name: gpu.name,
                meta,
                kind: "gpu".to_string(),
                target: gpu.public_id.clone(),
                effective_target: gpu.public_id.clone(),
                provider: gpu.provider.as_str().to_string(),
                memory: gpu.memory.map(|memory| format_gib(memory.total_bytes)),
                memory_total_bytes: gpu.memory.map(|memory| memory.total_bytes as u64),
                memory_free_bytes: gpu.memory.map(|memory| memory.free_bytes as u64),
                selectable: Some(true),
            }
        })
        .collect()
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
        // An Activated Vulkan provider in a neutral Windows host reports both
        // a CPU and a GPU device. The picker must surface the accelerated entry
        // and make Auto resolve to it.
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
        assert_eq!(
            ids,
            [
                "auto",
                "cpu",
                "accelerated",
                "vulkan:nvidia-geforce-rtx-4070"
            ]
        );
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
    fn hybrid_graphics_runtime_surfaces_discrete_gpu_not_integrated() {
        // Optimus-style laptop: Intel UHD (integrated) enumerates before the
        // NVIDIA discrete GPU. The picker's "accelerated" entry (and Auto,
        // which mirrors it) must still be the discrete GPU, not whichever
        // device the registry happened to list first.
        let runtime = runtime_with(
            vec![
                GgmlBackendDevice::for_test("CPU", "Intel Core i7", GgmlBackendKind::Cpu, None),
                GgmlBackendDevice::for_test(
                    "Vulkan0",
                    "Intel(R) UHD Graphics 630",
                    GgmlBackendKind::IntegratedGpu,
                    None,
                ),
                GgmlBackendDevice::for_test(
                    "Vulkan1",
                    "NVIDIA GeForce RTX 4070",
                    GgmlBackendKind::Gpu,
                    Some(GgmlDeviceMemory {
                        free_bytes: 8 * 1024 * 1024 * 1024,
                        total_bytes: 12 * 1024 * 1024 * 1024,
                    }),
                ),
            ],
            "Intel Core i7",
        );
        let devices = compute_devices_from_runtime(&runtime);
        let accelerated = devices.iter().find(|d| d.id == "accelerated").unwrap();
        assert_eq!(accelerated.name, "NVIDIA GeForce RTX 4070");
        assert_eq!(accelerated.meta, "GPU backend");
        assert_eq!(
            devices[0].name, "NVIDIA GeForce RTX 4070",
            "Auto mirrors it"
        );
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

    fn gib(n: usize) -> usize {
        n * 1024 * 1024 * 1024
    }

    fn three_gpu_runtime() -> GgmlRuntimeInfo {
        runtime_with(
            vec![
                GgmlBackendDevice::for_test("CPU", "AMD Ryzen 9", GgmlBackendKind::Cpu, None),
                GgmlBackendDevice::for_test(
                    "Vulkan0",
                    "NVIDIA GeForce RTX 2070 SUPER",
                    GgmlBackendKind::Gpu,
                    Some(GgmlDeviceMemory {
                        free_bytes: gib(6),
                        total_bytes: gib(8),
                    }),
                ),
                GgmlBackendDevice::for_test(
                    "Vulkan1",
                    "AMD Radeon RX 7900 XTX",
                    GgmlBackendKind::Gpu,
                    Some(GgmlDeviceMemory {
                        free_bytes: gib(20),
                        total_bytes: gib(24),
                    }),
                ),
                GgmlBackendDevice::for_test(
                    "Vulkan2",
                    "AMD Radeon Graphics",
                    GgmlBackendKind::IntegratedGpu,
                    None,
                ),
            ],
            "CPU",
        )
    }

    fn coarse_rows(devices: &[ComputeDevice]) -> Vec<&ComputeDevice> {
        devices
            .iter()
            .filter(|device| device.kind != "gpu")
            .collect()
    }

    #[test]
    fn three_vulkan_gpus_list_distinct_stable_ids_and_keep_coarse_rows() {
        let runtime = three_gpu_runtime();
        let first = compute_devices_from_runtime(&runtime);
        let second = compute_devices_from_runtime(&runtime);
        assert_eq!(first, second, "same input must yield the same ids");
        let gpu_rows: Vec<_> = first.iter().filter(|device| device.kind == "gpu").collect();
        assert_eq!(gpu_rows.len(), 3);
        let gpu_ids: Vec<_> = gpu_rows.iter().map(|device| device.id.as_str()).collect();
        assert_eq!(
            gpu_ids,
            [
                "vulkan:nvidia-geforce-rtx-2070-super",
                "vulkan:amd-radeon-rx-7900-xtx",
                "vulkan:amd-radeon-graphics",
            ]
        );
        assert_eq!(gpu_rows[0].provider, "vulkan");
        assert_eq!(gpu_rows[0].selectable, Some(true));
        assert_eq!(gpu_rows[0].memory_total_bytes, Some(gib(8) as u64));
        assert_eq!(gpu_rows[0].memory_free_bytes, Some(gib(6) as u64));
        assert_eq!(gpu_rows[1].memory_total_bytes, Some(gib(24) as u64));
        let ids: Vec<_> = first.iter().map(|device| device.id.as_str()).collect();
        assert_eq!(
            &ids[..3],
            ["auto", "cpu", "accelerated"],
            "coarse rows stay first and unchanged in identity"
        );
        let accelerated = first
            .iter()
            .find(|device| device.id == "accelerated")
            .unwrap();
        assert_eq!(accelerated.name, "NVIDIA GeForce RTX 2070 SUPER");
        assert_eq!(accelerated.kind, "accelerated");
        assert_eq!(default_execution_target(&first), "accelerated");
    }

    #[test]
    fn duplicate_gpu_names_do_not_share_an_id() {
        let runtime = runtime_with(
            vec![
                GgmlBackendDevice::for_test_with_device_id(
                    "Vulkan0",
                    "NVIDIA GeForce RTX 2070 SUPER",
                    GgmlBackendKind::Gpu,
                    None,
                    Some("0000:01:00.0"),
                ),
                GgmlBackendDevice::for_test_with_device_id(
                    "Vulkan1",
                    "NVIDIA GeForce RTX 2070 SUPER",
                    GgmlBackendKind::Gpu,
                    None,
                    Some("0000:02:00.0"),
                ),
            ],
            "CPU",
        );
        let devices = compute_devices_from_runtime(&runtime);
        let gpu_ids: Vec<_> = devices
            .iter()
            .filter(|device| device.kind == "gpu")
            .map(|device| device.id.as_str())
            .collect();
        assert_eq!(gpu_ids.len(), 2);
        assert_ne!(gpu_ids[0], gpu_ids[1]);
        assert!(
            devices
                .iter()
                .filter(|d| d.kind == "gpu")
                .all(|d| d.meta.contains("duplicate"))
        );
    }

    #[test]
    fn cpu_only_device_list_json_matches_pre_gpu_row_contract() {
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
        assert_eq!(
            devices.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
            ["auto", "cpu"]
        );
        let json = serde_json::to_string(&devices).expect("serialize");
        assert!(
            !json.contains("memory_total_bytes")
                && !json.contains("selectable")
                && !json.contains("\"kind\":\"gpu\""),
            "{json}"
        );
    }

    #[test]
    fn single_gpu_keeps_coarse_row_json_and_adds_one_gpu_row() {
        let runtime = runtime_with(
            vec![
                GgmlBackendDevice::for_test("CPU", "Intel Core", GgmlBackendKind::Cpu, None),
                GgmlBackendDevice::for_test(
                    "Vulkan0",
                    "NVIDIA GeForce RTX 4070",
                    GgmlBackendKind::Gpu,
                    Some(GgmlDeviceMemory {
                        free_bytes: gib(8),
                        total_bytes: gib(12),
                    }),
                ),
            ],
            "CPU",
        );
        let devices = compute_devices_from_runtime(&runtime);
        let coarse = coarse_rows(&devices);
        assert_eq!(
            coarse.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
            ["auto", "cpu", "accelerated"]
        );
        for row in &coarse {
            assert!(row.memory_total_bytes.is_none());
            assert!(row.selectable.is_none());
            let json = serde_json::to_string(row).expect("serialize");
            assert!(
                !json.contains("memory_total_bytes") && !json.contains("selectable"),
                "{json}"
            );
        }
        let gpu_rows: Vec<_> = devices.iter().filter(|d| d.kind == "gpu").collect();
        assert_eq!(gpu_rows.len(), 1);
        assert_eq!(gpu_rows[0].id, "vulkan:nvidia-geforce-rtx-4070");
        assert_eq!(gpu_rows[0].selectable, Some(true));
    }
}
