use crate::device::execution_policy::ExecutionPlacement;
use crate::device::execution_route::ExecutionProvider;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, RequestBackendPreference, request_backend_override,
};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
};
use crate::models::native_execution_services::current_execution_placement;

pub(crate) fn sensevoice_encoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: None,
            default_n_threads_when_unset: None,
        },
    )
}

pub(crate) fn sensevoice_sanm_flash_attention_enabled(
    config: &GgmlCpuGraphConfig,
    backend_preference: Option<&RequestBackendPreference>,
    placement: Option<ExecutionPlacement>,
) -> bool {
    config.backend == GgmlCpuGraphBackend::Gpu
        && !config.use_scheduler
        && placement == Some(ExecutionPlacement::FullDevice)
        && matches!(
            backend_preference,
            Some(RequestBackendPreference::Exact(route))
                if route.addressability.is_exactly_addressable()
                    && matches!(route.provider, ExecutionProvider::Cuda | ExecutionProvider::Vulkan)
        )
}

pub(crate) fn sensevoice_sanm_flash_attention_for_current_request(
    config: &GgmlCpuGraphConfig,
) -> bool {
    sensevoice_sanm_flash_attention_enabled(
        config,
        request_backend_override().as_ref(),
        current_execution_placement(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exactly_addressable_preference(provider: ExecutionProvider) -> RequestBackendPreference {
        RequestBackendPreference::Exact(crate::device::execution_route::ResolvedExecutionRoute {
            provider,
            stable_id: format!("{}0", provider.as_str()),
            registry_ordinal: 0,
            kind: crate::device::execution_route::RouteDeviceKind::Accelerated,
            addressability:
                crate::device::execution_route::DeviceAddressability::ExactlyAddressable {
                    physical_key: crate::device::execution_route::PhysicalResourceKey::new(
                        "0000:01:00.0",
                    )
                    .expect("physical key"),
                },
        })
    }

    #[test]
    fn encoder_preserves_the_resolved_metal_backend() {
        assert_eq!(
            sensevoice_encoder_graph_config(GgmlCpuGraphBackend::Metal).backend,
            GgmlCpuGraphBackend::Metal
        );
    }

    #[test]
    fn sanm_flash_is_limited_to_exact_direct_cuda_and_vulkan() {
        let direct_gpu = GgmlCpuGraphConfig {
            backend: GgmlCpuGraphBackend::Gpu,
            use_scheduler: false,
            ..GgmlCpuGraphConfig::conservative_default()
        };
        for provider in [ExecutionProvider::Cuda, ExecutionProvider::Vulkan] {
            let preference = exactly_addressable_preference(provider);
            assert!(sensevoice_sanm_flash_attention_enabled(
                &direct_gpu,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
            ));
            assert!(!sensevoice_sanm_flash_attention_enabled(
                &direct_gpu,
                Some(&preference),
                Some(ExecutionPlacement::Hybrid),
            ));
        }
        for provider in [
            ExecutionProvider::Cpu,
            ExecutionProvider::Metal,
            ExecutionProvider::Hip,
            ExecutionProvider::Accelerator,
            ExecutionProvider::Unknown,
        ] {
            let preference = exactly_addressable_preference(provider);
            assert!(!sensevoice_sanm_flash_attention_enabled(
                &direct_gpu,
                Some(&preference),
                Some(ExecutionPlacement::FullDevice),
            ));
        }
        let scheduled_gpu = GgmlCpuGraphConfig {
            use_scheduler: true,
            ..direct_gpu
        };
        let cuda = exactly_addressable_preference(ExecutionProvider::Cuda);
        assert!(!sensevoice_sanm_flash_attention_enabled(
            &scheduled_gpu,
            Some(&cuda),
            Some(ExecutionPlacement::FullDevice),
        ));
    }
}
