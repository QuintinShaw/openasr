use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
};

const OPENASR_PARAKEET_TDT_RESIDENT_PREDICTOR_STATE: &str =
    "OPENASR_PARAKEET_TDT_RESIDENT_PREDICTOR_STATE";

pub(crate) fn parakeet_tdt_encoder_graph_config(
    backend: GgmlCpuGraphBackend,
) -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: None,
            default_n_threads_when_unset: None,
        },
    )
}

pub(crate) fn parakeet_tdt_resident_predictor_state(config: GgmlCpuGraphConfig) -> bool {
    let raw = std::env::var(OPENASR_PARAKEET_TDT_RESIDENT_PREDICTOR_STATE).ok();
    let preference = crate::ggml_runtime::request_backend_override();
    parakeet_tdt_resident_predictor_state_with_inputs(
        config,
        preference.as_ref(),
        crate::models::native_execution_services::current_execution_placement(),
        raw.as_deref(),
    )
}

fn parakeet_tdt_resident_predictor_state_with_inputs(
    config: GgmlCpuGraphConfig,
    preference: Option<&crate::ggml_runtime::RequestBackendPreference>,
    placement: Option<crate::device::execution_policy::ExecutionPlacement>,
    raw: Option<&str>,
) -> bool {
    // CUDA's small host transfers are cheaper than the extra recurrent-state
    // CPY nodes on RTX 3060 (five-run q8 median regressed by 0.69%). Vulkan's
    // corresponding median improved by 3.72%, so keep the shared mechanism but
    // qualify only the provider with a measured Pareto win.
    let validated_discrete_gpu = matches!(
        (config.backend, config.use_scheduler, preference, placement),
        (
            GgmlCpuGraphBackend::Gpu,
            false,
            Some(crate::ggml_runtime::RequestBackendPreference::Exact(route)),
            Some(crate::device::execution_policy::ExecutionPlacement::FullDevice),
        ) if route.provider == crate::device::execution_route::ExecutionProvider::Vulkan
    );
    validated_discrete_gpu && crate::ggml_runtime::env_toggle_with_raw(None, raw, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::execution_policy::ExecutionPlacement;
    use crate::device::execution_route::{
        DeviceAddressability, ExecutionProvider, ResolvedExecutionRoute, RouteDeviceKind,
    };
    use crate::ggml_runtime::RequestBackendPreference;

    fn exact_preference(provider: ExecutionProvider) -> RequestBackendPreference {
        RequestBackendPreference::Exact(ResolvedExecutionRoute {
            provider,
            stable_id: format!("{}0", provider.as_str()),
            registry_ordinal: 0,
            kind: RouteDeviceKind::Accelerated,
            addressability: DeviceAddressability::NotExactlyAddressable {
                reason: "parakeet-tdt resident predictor policy fixture",
            },
        })
    }

    #[test]
    fn encoder_preserves_the_resolved_metal_backend() {
        assert_eq!(
            parakeet_tdt_encoder_graph_config(GgmlCpuGraphBackend::Metal).backend,
            GgmlCpuGraphBackend::Metal
        );
    }

    #[test]
    fn resident_predictor_state_is_exact_vulkan_direct_only() {
        let mut direct_gpu =
            GgmlCpuGraphConfig::runtime_default_for_resolved_backend(GgmlCpuGraphBackend::Gpu);
        direct_gpu.use_scheduler = false;
        let vulkan = exact_preference(ExecutionProvider::Vulkan);
        assert!(parakeet_tdt_resident_predictor_state_with_inputs(
            direct_gpu,
            Some(&vulkan),
            Some(ExecutionPlacement::FullDevice),
            None,
        ));
        assert!(!parakeet_tdt_resident_predictor_state_with_inputs(
            direct_gpu,
            Some(&vulkan),
            Some(ExecutionPlacement::FullDevice),
            Some("0"),
        ));

        for provider in [ExecutionProvider::Cuda, ExecutionProvider::Hip] {
            let preference = exact_preference(provider);
            assert!(!parakeet_tdt_resident_predictor_state_with_inputs(
                direct_gpu,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
                None,
            ));
        }
        let cuda = exact_preference(ExecutionProvider::Cuda);
        let mut scheduled = direct_gpu;
        scheduled.use_scheduler = true;
        assert!(!parakeet_tdt_resident_predictor_state_with_inputs(
            scheduled,
            Some(&cuda),
            Some(ExecutionPlacement::FullDevice),
            None,
        ));
        assert!(!parakeet_tdt_resident_predictor_state_with_inputs(
            direct_gpu,
            Some(&cuda),
            Some(ExecutionPlacement::Hybrid),
            None,
        ));
        assert!(!parakeet_tdt_resident_predictor_state_with_inputs(
            GgmlCpuGraphConfig::runtime_default_for_resolved_backend(GgmlCpuGraphBackend::Metal,),
            Some(&exact_preference(ExecutionProvider::Metal)),
            Some(ExecutionPlacement::FullDevice),
            None,
        ));
    }
}
