//! Shared policy and first-max index helpers for device-only greedy decode.
//!
//! A family may return only a token id when its decode policy does not need
//! logits, probabilities, phrase bias, or timestamps. The route gate here is
//! intentionally narrow: the first rollout is limited to an explicitly pinned
//! CUDA/Vulkan FullDevice candidate on a direct (non-scheduler) GPU runner.

use crate::device::execution_policy::ExecutionPlacement;
use crate::device::execution_route::ExecutionProvider;
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphError, RequestBackendPreference};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DeviceGreedyStepOutputMode {
    FullLogits,
    DeviceTop1,
}

pub(crate) fn device_greedy_step_output_mode(
    backend: GgmlCpuGraphBackend,
    use_scheduler: bool,
    backend_preference: Option<&RequestBackendPreference>,
    placement: Option<ExecutionPlacement>,
) -> DeviceGreedyStepOutputMode {
    let exact_provider = match backend_preference {
        Some(RequestBackendPreference::Exact(route)) => route.provider,
        Some(RequestBackendPreference::CpuOnly | RequestBackendPreference::Accelerated) | None => {
            return DeviceGreedyStepOutputMode::FullLogits;
        }
    };
    if backend == GgmlCpuGraphBackend::Gpu
        && !use_scheduler
        && placement == Some(ExecutionPlacement::FullDevice)
        && matches!(
            exact_provider,
            ExecutionProvider::Cuda | ExecutionProvider::Vulkan
        )
    {
        DeviceGreedyStepOutputMode::DeviceTop1
    } else {
        DeviceGreedyStepOutputMode::FullLogits
    }
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
    use crate::ggml_runtime::{GgmlCpuGraphConfig, GgmlCpuGraphRunner};

    use super::*;

    fn exact_preference(provider: ExecutionProvider) -> RequestBackendPreference {
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
    fn route_policy_only_enables_direct_exact_cuda_and_vulkan_full_device() {
        for provider in [ExecutionProvider::Cuda, ExecutionProvider::Vulkan] {
            let preference = exact_preference(provider);
            assert_eq!(
                device_greedy_step_output_mode(
                    GgmlCpuGraphBackend::Gpu,
                    false,
                    Some(&preference),
                    Some(ExecutionPlacement::FullDevice),
                ),
                DeviceGreedyStepOutputMode::DeviceTop1
            );
        }

        for provider in [
            ExecutionProvider::Cpu,
            ExecutionProvider::Metal,
            ExecutionProvider::Hip,
            ExecutionProvider::Accelerator,
            ExecutionProvider::Unknown,
        ] {
            let preference = exact_preference(provider);
            assert_eq!(
                device_greedy_step_output_mode(
                    GgmlCpuGraphBackend::Gpu,
                    false,
                    Some(&preference),
                    Some(ExecutionPlacement::FullDevice),
                ),
                DeviceGreedyStepOutputMode::FullLogits,
                "provider={provider:?}"
            );
        }
    }

    #[test]
    fn route_policy_rejects_scheduler_hybrid_non_gpu_and_non_exact_paths() {
        let cuda = exact_preference(ExecutionProvider::Cuda);
        for (backend, scheduler, preference, placement) in [
            (
                GgmlCpuGraphBackend::Gpu,
                true,
                Some(&cuda),
                Some(ExecutionPlacement::FullDevice),
            ),
            (
                GgmlCpuGraphBackend::Gpu,
                false,
                Some(&cuda),
                Some(ExecutionPlacement::Hybrid),
            ),
            (
                GgmlCpuGraphBackend::Metal,
                false,
                Some(&cuda),
                Some(ExecutionPlacement::FullDevice),
            ),
            (
                GgmlCpuGraphBackend::Cpu,
                false,
                Some(&cuda),
                Some(ExecutionPlacement::CpuOnly),
            ),
            (
                GgmlCpuGraphBackend::Gpu,
                false,
                None,
                Some(ExecutionPlacement::FullDevice),
            ),
        ] {
            assert_eq!(
                device_greedy_step_output_mode(backend, scheduler, preference, placement),
                DeviceGreedyStepOutputMode::FullLogits
            );
        }
        assert_eq!(
            device_greedy_step_output_mode(
                GgmlCpuGraphBackend::Gpu,
                false,
                Some(&RequestBackendPreference::Accelerated),
                Some(ExecutionPlacement::FullDevice),
            ),
            DeviceGreedyStepOutputMode::FullLogits
        );
    }

    #[test]
    fn cpu_graph_first_max_oracle_matches_shared_host_mapping() {
        let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut arena = runner
            .start_static_tensor_arena(GgmlCpuGraphConfig::metadata_context_bytes(1))
            .expect("static arena should initialize");
        let reverse_indices = arena
            .new_tensor_1d_i32(4, "device_greedy_reverse_indices")
            .expect("reverse-index allocation should succeed");
        arena
            .set_i32_slice(
                reverse_indices,
                &first_max_argmax_reverse_indices(4).expect("reverse indices"),
                "device_greedy_reverse_indices",
            )
            .expect("reverse indices upload should succeed");

        let mut graph = runner.start_graph();
        let logits = graph
            .new_tensor_2d_f32(4, 1, "device_greedy_logits")
            .expect("logits allocation should succeed");
        graph.set_input(logits).expect("logits input");
        let top1 = graph
            .top1_argmax_first_max_reversed(logits, arena.graph_tensor(reverse_indices))
            .expect("first-max graph should build");
        graph.set_output(top1).expect("top1 output");
        graph
            .set_f32_slice(logits, &[1.0, 5.0, 5.0, 2.0], "device_greedy_logits")
            .expect("logits upload");
        let reversed = graph.compute_output_i32(top1, 1).expect("top1 compute")[0];
        assert_eq!(
            first_max_token_id_from_reversed_argmax(reversed, 4).expect("mapped token"),
            1
        );
    }
}
