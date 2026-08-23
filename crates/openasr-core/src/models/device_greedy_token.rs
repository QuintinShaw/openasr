//! Shared policy and first-max index helpers for device-only greedy decode.
//!
//! A family may return only a token id when its decode policy does not need
//! logits, probabilities, phrase bias, or timestamps. The route and reuse
//! decision is resolved once by [`ResolvedFamilyRuntimeInput`]; this module
//! only translates that immutable output plan into the graph-facing mode.

use crate::device::execution_policy::ExecutionPlacement;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphError, GgmlDecodeOutputPlan, RequestBackendPreference,
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

/// Compatibility shim for families not yet migrated to the request-scoped
/// planner. Production always fails closed; the test-only branch preserves the
/// pre-migration graph fixtures until those families consume the planner.
#[allow(dead_code)]
pub(crate) fn device_greedy_step_output_mode(
    backend: GgmlCpuGraphBackend,
    use_scheduler: bool,
    backend_preference: Option<&RequestBackendPreference>,
    placement: Option<ExecutionPlacement>,
) -> DeviceGreedyStepOutputMode {
    #[cfg(test)]
    if backend == GgmlCpuGraphBackend::Gpu
        && !use_scheduler
        && placement == Some(ExecutionPlacement::FullDevice)
        && matches!(backend_preference, Some(RequestBackendPreference::Exact(_)))
    {
        return DeviceGreedyStepOutputMode::DeviceTop1;
    }
    let _ = (backend, use_scheduler, backend_preference, placement);
    DeviceGreedyStepOutputMode::FullLogits
}

pub(crate) fn first_max_argmax_reverse_indices(
    vocab_size: usize,
) -> Result<Vec<i32>, GgmlCpuGraphError> {
    (0..vocab_size)
        .rev()
        .map(|index| {
            i32::try_from(index).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "first-max argmax vocab index exceeds ggml int boundary",
            })
        })
        .collect()
}

pub(crate) fn first_max_token_id_from_reversed_argmax(
    reversed_token_id: i32,
    vocab_size: usize,
) -> Result<i32, GgmlCpuGraphError> {
    let reversed_index =
        usize::try_from(reversed_token_id).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
            reason: "first-max argmax reversed token id is negative",
        })?;
    if reversed_index >= vocab_size {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "first-max argmax reversed token id is outside vocab size",
        });
    }
    let original_index = vocab_size - 1 - reversed_index;
    i32::try_from(original_index).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
        reason: "first-max argmax token id exceeds ggml int boundary",
    })
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
    fn complete_output_plan_is_the_only_production_default() {
        let resolved = ResolvedFamilyRuntimeInput::resolve(None, AutoGpuPolicy::AllBackends);
        assert_eq!(
            device_greedy_step_output_mode_for_resolved_runtime(resolved),
            DeviceGreedyStepOutputMode::FullLogits
        );
    }

    #[test]
    fn cpu_graph_first_max_oracle_mapping_remains_testable_without_production_reachability() {
        assert_eq!(
            first_max_token_id_from_reversed_argmax(2, 4).expect("mapped token"),
            1
        );
    }
}
