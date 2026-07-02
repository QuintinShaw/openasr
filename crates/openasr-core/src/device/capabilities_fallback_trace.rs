use super::{HardwareFallbackPolicy, HardwareProvider, ProviderAvailabilityState};

pub(super) fn conservative_cpu_default() -> HardwareFallbackPolicy {
    // Truthful report of what the GGML runtime actually does: Metal is built
    // in and auto-selected on macOS when a GPU device is present (per-family
    // graph configs may still prefer CPU where it measures faster, e.g. the
    // xasr chunked encoder); execution_target=cpu|accelerated forces the
    // backend per request and accelerated hard-errors without a GPU.
    let accelerated_available = crate::ggml_available_devices()
        .iter()
        .any(|device| device.kind.is_gpu());
    HardwareFallbackPolicy {
        default_execution_target: HardwareProvider::Cpu,
        accelerated_provider_integration: if accelerated_available {
            ProviderAvailabilityState::Available
        } else {
            ProviderAvailabilityState::Unavailable
        },
        automatic_provider_selection: accelerated_available,
        external_probe_commands: false,
        benchmark_gate_required: true,
        notes: vec![
            "GGML accelerated execution (Metal on macOS) is built in; Auto selects it per family unless the family's measured default prefers CPU."
                .to_string(),
            "execution_target=cpu and =accelerated force the backend per request; accelerated fails closed when no GPU device exists."
                .to_string(),
        ],
    }
}
