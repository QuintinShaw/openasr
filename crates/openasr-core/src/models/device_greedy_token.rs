//! Shared policy and first-max index helpers for device-only greedy decode.
//!
//! A family may return only a token id when its decode policy does not need
//! logits, probabilities, phrase bias, or timestamps. The route and reuse
//! decision is resolved once by [`ResolvedFamilyRuntimeInput`]; this module
//! only translates that immutable output plan into the graph-facing mode.

use crate::ggml_runtime::{
    GgmlCpuGraphError, GgmlDecodeLogitsConsumers, GgmlDecodeOutputContract, GgmlDecodeOutputPlan,
    ResolvedFamilyRuntimeInput,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DeviceGreedyStepOutputMode {
    FullLogits,
    DeviceTop1,
}

pub(crate) fn device_greedy_step_output_mode_for_resolved_runtime(
    resolved_runtime: ResolvedFamilyRuntimeInput,
) -> DeviceGreedyStepOutputMode {
    match resolved_runtime.output_plan() {
        GgmlDecodeOutputPlan::NativeFirstMaxToken => DeviceGreedyStepOutputMode::DeviceTop1,
        GgmlDecodeOutputPlan::FullLogits | GgmlDecodeOutputPlan::CompleteScores => {
            DeviceGreedyStepOutputMode::FullLogits
        }
    }
}

/// Family host-oracle contracts that must not enter native first-max compact
/// selection. XASR keeps last-max host selection; SenseVoice keeps complete
/// frame logits. Other token families request the native-first-max fallback.
pub(crate) fn decode_output_contract_for_adapter(adapter_id: &str) -> GgmlDecodeOutputContract {
    if adapter_id == crate::arch::XASR_ZIPFORMER_GGML_ADAPTER_ID
        || adapter_id == crate::arch::SENSEVOICE_GGML_ADAPTER_ID
    {
        GgmlDecodeOutputContract::FullLogits
    } else {
        GgmlDecodeOutputContract::NativeFirstMaxTokenOrFullLogits
    }
}

pub(crate) fn decode_logits_consumers_for_request(
    adapter_id: &str,
    phrase_bias_active: bool,
    word_timestamps: bool,
) -> GgmlDecodeLogitsConsumers {
    let debug_logits = adapter_id == crate::arch::COHERE_TRANSCRIBE_GGML_ADAPTER_ID
        && std::env::var_os("OPENASR_COHERE_DEBUG_TOKENS").is_some();
    GgmlDecodeLogitsConsumers::new(phrase_bias_active, word_timestamps, false, debug_logits)
}

pub(crate) fn device_top1_token_id(
    token_id: i32,
    vocab_size: usize,
) -> Result<u32, GgmlCpuGraphError> {
    let token = u32::try_from(token_id).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
        reason: "device top-1 token id is negative",
    })?;
    if usize::try_from(token)
        .ok()
        .is_none_or(|id| id >= vocab_size)
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "device top-1 token id is outside vocab size",
        });
    }
    Ok(token)
}

#[cfg(test)]
mod tests {
    use crate::device::execution_route::{
        DeviceAddressability, ResolvedExecutionRoute, RouteDeviceKind,
    };
    use crate::ggml_runtime::{AutoGpuPolicy, GgmlCpuGraphBackend, RequestBackendPreference};

    use super::*;

    fn exact_preference(
        provider: crate::device::execution_route::ExecutionProvider,
    ) -> RequestBackendPreference {
        RequestBackendPreference::Exact(ResolvedExecutionRoute {
            provider,
            stable_id: format!("{}0", provider.as_str()),
            registry_ordinal: 0,
            kind: RouteDeviceKind::Accelerated,
            addressability: DeviceAddressability::NotExactlyAddressable {
                reason: "device-greedy-token route-policy fixture",
            },
        })
    }

    #[test]
    fn exact_cuda_and_vulkan_without_selected_device_evidence_stay_complete() {
        for provider in [
            crate::device::execution_route::ExecutionProvider::Cuda,
            crate::device::execution_route::ExecutionProvider::Vulkan,
        ] {
            let resolved = ResolvedFamilyRuntimeInput::resolve_with_output_contract(
                Some(exact_preference(provider)),
                AutoGpuPolicy::AllBackends,
                crate::ggml_runtime::GgmlDecodeOutputContract::NativeFirstMaxTokenOrFullLogits,
            );
            assert_eq!(resolved.backend(), GgmlCpuGraphBackend::Gpu);
            assert_eq!(resolved.output_plan(), GgmlDecodeOutputPlan::FullLogits);
            assert_eq!(
                resolved.reuse_mode(),
                crate::ggml_runtime::GgmlDecodeReuseMode::FreshGraph
            );
            assert_eq!(
                device_greedy_step_output_mode_for_resolved_runtime(resolved),
                DeviceGreedyStepOutputMode::FullLogits
            );
        }
    }

    #[test]
    fn cpu_lane_authorizes_native_first_max_without_reuse() {
        let resolved = ResolvedFamilyRuntimeInput::resolve(
            Some(RequestBackendPreference::CpuOnly),
            AutoGpuPolicy::AllBackends,
        );
        assert_eq!(resolved.backend(), GgmlCpuGraphBackend::Cpu);
        assert_eq!(
            resolved.output_plan(),
            GgmlDecodeOutputPlan::NativeFirstMaxToken
        );
        assert_eq!(
            resolved.reuse_mode(),
            crate::ggml_runtime::GgmlDecodeReuseMode::FreshGraph
        );
        assert_eq!(
            device_greedy_step_output_mode_for_resolved_runtime(resolved),
            DeviceGreedyStepOutputMode::DeviceTop1
        );
    }

    #[test]
    fn device_top1_token_id_rejects_out_of_range_values() {
        assert_eq!(device_top1_token_id(2, 4).expect("in-range token"), 2);
        assert!(device_top1_token_id(-1, 4).is_err());
        assert!(device_top1_token_id(4, 4).is_err());
    }

    #[test]
    fn unproven_gpu_lanes_keep_full_device_and_complete_outputs() {
        use crate::ggml_runtime::GgmlDecodeReuseMode;

        for provider in [
            crate::device::execution_route::ExecutionProvider::Cuda,
            crate::device::execution_route::ExecutionProvider::Vulkan,
            crate::device::execution_route::ExecutionProvider::Hip,
            crate::device::execution_route::ExecutionProvider::Metal,
        ] {
            let resolved = ResolvedFamilyRuntimeInput::resolve_with_output_contract(
                Some(exact_preference(provider)),
                AutoGpuPolicy::AllBackends,
                GgmlDecodeOutputContract::NativeFirstMaxTokenOrFullLogits,
            );
            assert!(
                resolved.backend().is_gpu_class(),
                "unproven {provider:?} must keep the selected GPU lane, not fall back to CPU"
            );
            assert_eq!(resolved.output_plan(), GgmlDecodeOutputPlan::FullLogits);
            assert_eq!(resolved.reuse_mode(), GgmlDecodeReuseMode::FreshGraph);
            assert_eq!(
                device_greedy_step_output_mode_for_resolved_runtime(resolved),
                DeviceGreedyStepOutputMode::FullLogits
            );

            let scores = ResolvedFamilyRuntimeInput::resolve_with_output_contract(
                Some(exact_preference(provider)),
                AutoGpuPolicy::AllBackends,
                GgmlDecodeOutputContract::CompleteScores,
            );
            assert!(scores.backend().is_gpu_class());
            assert_eq!(scores.output_plan(), GgmlDecodeOutputPlan::CompleteScores);
        }
    }

    #[test]
    fn logits_consumers_force_full_logits_even_on_proven_cpu() {
        let cpu = ResolvedFamilyRuntimeInput::resolve(
            Some(RequestBackendPreference::CpuOnly),
            AutoGpuPolicy::AllBackends,
        );
        assert_eq!(cpu.output_plan(), GgmlDecodeOutputPlan::NativeFirstMaxToken);

        for consumers in [
            GgmlDecodeLogitsConsumers::none().with_phrase_bias(true),
            GgmlDecodeLogitsConsumers::none().with_timestamps(true),
            GgmlDecodeLogitsConsumers::none().with_suppression(true),
            GgmlDecodeLogitsConsumers::none().with_debug_logits(true),
        ] {
            let resolved = ResolvedFamilyRuntimeInput::resolve_with_output_contract_and_consumers(
                Some(RequestBackendPreference::CpuOnly),
                AutoGpuPolicy::AllBackends,
                GgmlDecodeOutputContract::NativeFirstMaxTokenOrFullLogits,
                consumers,
            );
            assert_eq!(resolved.backend(), GgmlCpuGraphBackend::Cpu);
            assert_eq!(resolved.output_plan(), GgmlDecodeOutputPlan::FullLogits);
            assert_eq!(
                device_greedy_step_output_mode_for_resolved_runtime(resolved),
                DeviceGreedyStepOutputMode::FullLogits
            );
        }
    }

    #[test]
    fn xasr_and_sensevoice_host_oracles_do_not_enter_native_first_max() {
        for adapter in [
            crate::arch::XASR_ZIPFORMER_GGML_ADAPTER_ID,
            crate::arch::SENSEVOICE_GGML_ADAPTER_ID,
        ] {
            assert_eq!(
                decode_output_contract_for_adapter(adapter),
                GgmlDecodeOutputContract::FullLogits
            );
            let resolved = ResolvedFamilyRuntimeInput::resolve_with_output_contract(
                Some(RequestBackendPreference::CpuOnly),
                AutoGpuPolicy::AllBackends,
                decode_output_contract_for_adapter(adapter),
            );
            assert_eq!(resolved.backend(), GgmlCpuGraphBackend::Cpu);
            assert_eq!(resolved.output_plan(), GgmlDecodeOutputPlan::FullLogits);
            assert_eq!(
                device_greedy_step_output_mode_for_resolved_runtime(resolved),
                DeviceGreedyStepOutputMode::FullLogits
            );
        }
    }

    #[test]
    fn untested_discrete_gpu_cannot_activate_compact_without_hardware() {
        use crate::device::execution_route::enumerate_compute_devices_from_ggml;
        use crate::ggml_runtime::ggml_available_devices;

        let inventory = enumerate_compute_devices_from_ggml(&ggml_available_devices());
        for provider in [
            crate::device::execution_route::ExecutionProvider::Cuda,
            crate::device::execution_route::ExecutionProvider::Vulkan,
            crate::device::execution_route::ExecutionProvider::Hip,
        ] {
            let present = inventory.iter().any(|device| device.provider == provider);
            let resolved = ResolvedFamilyRuntimeInput::resolve_with_output_contract(
                Some(exact_preference(provider)),
                AutoGpuPolicy::AllBackends,
                GgmlDecodeOutputContract::NativeFirstMaxTokenOrFullLogits,
            );
            assert_eq!(
                resolved.output_plan(),
                GgmlDecodeOutputPlan::FullLogits,
                "{provider:?} compact stays unactivatable (hardware present={present})"
            );
            assert_ne!(
                resolved.output_plan(),
                GgmlDecodeOutputPlan::NativeFirstMaxToken
            );
        }
    }
}
