//! Shared policy and first-max index helpers for device-only greedy decode.
//!
//! A family may return only a token id when its decode policy does not need
//! logits, probabilities, phrase bias, or timestamps. The route and reuse
//! decision is resolved once by [`ResolvedFamilyRuntimeInput`]; this module
//! only translates that immutable output plan into the graph-facing mode.

use crate::device::execution_policy::{ExecutionCandidate, ExecutionPlacement};
use crate::ggml_runtime::{
    AutoGpuPolicy, GgmlCpuGraphError, GgmlDecodeLogitsConsumers, GgmlDecodeOutputContract,
    GgmlDecodeOutputPlan, RequestBackendPreference, ResolvedFamilyRuntimeInput,
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
    adapter_active: bool,
) -> GgmlDecodeLogitsConsumers {
    let debug_logits = adapter_id == crate::arch::COHERE_TRANSCRIBE_GGML_ADAPTER_ID
        && std::env::var_os("OPENASR_COHERE_DEBUG_TOKENS").is_some();
    GgmlDecodeLogitsConsumers::new(
        phrase_bias_active,
        word_timestamps,
        suppression_active_for_adapter(adapter_id),
        debug_logits,
    )
    .with_host_visible(adapter_active)
}

/// Convert one immutable policy candidate into the exact request preference
/// consumed by the shared runtime planner. Offline, streaming, warm-up, and
/// activation must not grow independent provider/placement tables.
pub(crate) fn request_backend_preference_for_candidate(
    candidate: &ExecutionCandidate,
) -> Option<RequestBackendPreference> {
    match candidate.placement {
        ExecutionPlacement::CpuOnly => Some(RequestBackendPreference::CpuOnly),
        ExecutionPlacement::FullDevice | ExecutionPlacement::Hybrid => Some(
            RequestBackendPreference::Exact(candidate.device.route.clone()),
        ),
    }
}

/// Resolve one immutable family runtime input from a concrete candidate.
/// This is the shared output-plan/reuse combiner for every execution surface.
pub(crate) fn resolved_runtime_for_family_candidate(
    candidate: &ExecutionCandidate,
    auto_gpu_policy: AutoGpuPolicy,
    adapter_id: &str,
    logits_consumers: GgmlDecodeLogitsConsumers,
) -> ResolvedFamilyRuntimeInput {
    ResolvedFamilyRuntimeInput::resolve_with_output_contract_and_consumers(
        request_backend_preference_for_candidate(candidate),
        auto_gpu_policy,
        decode_output_contract_for_adapter(adapter_id),
        logits_consumers,
    )
}

fn suppression_active_for_adapter(adapter_id: &str) -> bool {
    crate::arch::OpenAsrArchitectureRegistry::with_builtins()
        .find_by_adapter_id(adapter_id)
        .and_then(|descriptor| descriptor.topology_contract.decode_driver.shared_policy())
        .is_some_and(|policy| {
            !matches!(
                policy.seq2seq_suppression_kind,
                crate::models::decode_policy_component_registry::BuiltinDecodePolicySeq2SeqSuppressionKind::None
            )
        })
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
            GgmlDecodeLogitsConsumers::none().with_host_visible(true),
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

    fn resolve_consumers(consumers: GgmlDecodeLogitsConsumers) -> ResolvedFamilyRuntimeInput {
        ResolvedFamilyRuntimeInput::resolve_with_output_contract_and_consumers(
            Some(RequestBackendPreference::CpuOnly),
            AutoGpuPolicy::AllBackends,
            GgmlDecodeOutputContract::NativeFirstMaxTokenOrFullLogits,
            consumers,
        )
    }

    #[test]
    fn shipped_combiner_wires_whisper_suppression_and_forces_full_logits() {
        let whisper = decode_logits_consumers_for_request(
            crate::arch::WHISPER_GGML_ADAPTER_ID,
            false,
            false,
            false,
        );
        assert!(whisper.requires_complete_logits());
        assert_eq!(
            resolve_consumers(whisper).output_plan(),
            GgmlDecodeOutputPlan::FullLogits
        );

        let qwen = decode_logits_consumers_for_request(
            crate::arch::QWEN3_ASR_GGML_ADAPTER_ID,
            false,
            false,
            false,
        );
        assert!(!qwen.requires_complete_logits());
        assert_eq!(
            resolve_consumers(qwen).output_plan(),
            GgmlDecodeOutputPlan::NativeFirstMaxToken
        );
    }

    #[test]
    fn shipped_combiner_forces_full_logits_for_each_request_consumer() {
        let adapter = crate::arch::QWEN3_ASR_GGML_ADAPTER_ID;
        for consumers in [
            decode_logits_consumers_for_request(adapter, true, false, false),
            decode_logits_consumers_for_request(adapter, false, true, false),
            decode_logits_consumers_for_request(adapter, false, false, true),
            decode_logits_consumers_for_request(
                crate::arch::WHISPER_GGML_ADAPTER_ID,
                false,
                false,
                false,
            ),
        ] {
            assert!(consumers.requires_complete_logits());
            assert_eq!(
                resolve_consumers(consumers).output_plan(),
                GgmlDecodeOutputPlan::FullLogits
            );
        }
    }
}
