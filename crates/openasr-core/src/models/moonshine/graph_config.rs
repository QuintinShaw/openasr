use crate::device::execution_policy::ExecutionPlacement;
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphThreadingWorkload};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    has_explicit_thread_override,
};

const OPENASR_MOONSHINE_ENABLE_DECODER_GPU: &str = "OPENASR_MOONSHINE_ENABLE_DECODER_GPU";

/// Shared base for both stages: everything except the scheduler default,
/// which the encoder and decoder now set independently (see
/// [`moonshine_encoder_graph_config`] / [`moonshine_decoder_graph_config`]).
fn moonshine_runtime_graph_config_with_scheduler_default(
    backend: GgmlCpuGraphBackend,
    default_use_scheduler_when_unset: Option<bool>,
) -> GgmlCpuGraphConfig {
    let mut config = configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset,
            default_n_threads_when_unset: Some(1),
        },
    );
    // The request-resolved backend is authoritative for this family. An
    // ambient placement override may not silently turn an already-selected
    // accelerator into a CPU graph; the decoder's explicit Vulkan hybrid
    // policy below is the only intentional stage-local downgrade.
    if backend.is_gpu_class() && config.backend == GgmlCpuGraphBackend::Cpu {
        config.backend = backend;
        config.use_scheduler = false;
    }
    config
}

/// Moonshine's waveform preparation and token handling stay on the host, while
/// the neural encoder and decoder are complete device graphs. Bind those graphs
/// to the exact accelerator lane selected by policy; FullDevice also removes
/// ggml's mandatory CPU scheduler fallback.
fn apply_moonshine_neural_graph_placement(config: GgmlCpuGraphConfig) -> GgmlCpuGraphConfig {
    if config.backend.is_gpu_class() {
        crate::models::graph_runtime_config::apply_execution_placement(
            config,
            ExecutionPlacement::FullDevice,
        )
    } else {
        config
    }
}

pub(crate) fn moonshine_encoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    // Resolve operator and thread defaults before narrowing the neural graph to
    // its device-complete placement below.
    let mut config = moonshine_runtime_graph_config_with_scheduler_default(backend, Some(true));
    config.graph_size = config.graph_size.max(16_384);
    if !has_explicit_thread_override() {
        config.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            config.backend,
            GgmlCpuGraphThreadingWorkload::EncoderPrelude,
        );
    }
    apply_moonshine_neural_graph_placement(config)
}

/// Keep decoder placement separate from encoder placement. Vulkan is the
/// validated hybrid route: the encoder remains accelerated while the decoder
/// defaults to CPU, unless the diagnostic stage override explicitly enables
/// GPU decode. The request output/reuse contract is resolved at the dispatch
/// boundary and consumed by the executor; this graph config only determines
/// the stage backend and scheduler settings.
pub(crate) fn moonshine_decoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    let mut config = moonshine_runtime_graph_config_with_scheduler_default(backend, None);
    if config.backend.is_gpu_class() && !decoder_gpu_enabled(config.backend) {
        config.backend = GgmlCpuGraphBackend::Cpu;
        config.use_scheduler = false;
    }
    if !has_explicit_thread_override() {
        config.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            config.backend,
            GgmlCpuGraphThreadingWorkload::Decoder,
        );
    }
    apply_moonshine_neural_graph_placement(config)
}

fn decoder_gpu_enabled(backend: GgmlCpuGraphBackend) -> bool {
    let gpu_raw = std::env::var(OPENASR_MOONSHINE_ENABLE_DECODER_GPU).ok();
    let backend_preference = crate::ggml_runtime::request_backend_override();
    decoder_gpu_enabled_with_inputs(backend, gpu_raw.as_deref(), backend_preference.as_ref())
}

fn decoder_gpu_enabled_with_inputs(
    backend: GgmlCpuGraphBackend,
    gpu_raw: Option<&str>,
    backend_preference: Option<&crate::ggml_runtime::RequestBackendPreference>,
) -> bool {
    let exact_vulkan = matches!(
        (backend, backend_preference),
        (
            GgmlCpuGraphBackend::Gpu,
            Some(crate::ggml_runtime::RequestBackendPreference::Exact(route))
        ) if route.provider == crate::device::execution_route::ExecutionProvider::Vulkan
    );
    if exact_vulkan {
        crate::ggml_runtime::env_toggle_with_raw(None, gpu_raw, false)
    } else {
        // FullDevice providers must not be silently rewritten into a CPU
        // decoder by a stage-local setting.
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_route(
        provider: crate::device::execution_route::ExecutionProvider,
    ) -> crate::ggml_runtime::RequestBackendPreference {
        crate::ggml_runtime::RequestBackendPreference::Exact(
            crate::device::execution_route::ResolvedExecutionRoute {
                provider,
                stable_id: format!("{provider:?}0"),
                registry_ordinal: 0,
                kind: crate::device::execution_route::RouteDeviceKind::Accelerated,
                addressability:
                    crate::device::execution_route::DeviceAddressability::ExactlyAddressable {
                        physical_key: crate::device::execution_route::PhysicalResourceKey::new(
                            "0000:02:00.0",
                        )
                        .expect("synthetic PCI key is valid"),
                    },
            },
        )
    }

    fn with_decoder_env<T>(gpu: Option<&str>, run: impl FnOnce() -> T) -> T {
        crate::test_process_env::with_test_process_env(
            [(
                OPENASR_MOONSHINE_ENABLE_DECODER_GPU,
                gpu.map(std::ffi::OsString::from),
            )],
            run,
        )
    }

    #[test]
    fn full_device_keeps_moonshine_neural_graphs_on_device() {
        let mut config =
            GgmlCpuGraphConfig::runtime_default_for_resolved_backend(GgmlCpuGraphBackend::Metal);
        config.use_scheduler = true;
        assert!(!apply_moonshine_neural_graph_placement(config).use_scheduler);

        let mut cpu =
            GgmlCpuGraphConfig::runtime_default_for_resolved_backend(GgmlCpuGraphBackend::Cpu);
        cpu.use_scheduler = true;
        assert!(apply_moonshine_neural_graph_placement(cpu).use_scheduler);
    }

    #[test]
    fn encoder_and_decoder_preserve_the_resolved_metal_backend() {
        assert_eq!(
            moonshine_encoder_graph_config(GgmlCpuGraphBackend::Metal).backend,
            GgmlCpuGraphBackend::Metal
        );
        assert_eq!(
            moonshine_decoder_graph_config(GgmlCpuGraphBackend::Metal).backend,
            GgmlCpuGraphBackend::Metal
        );
    }

    #[test]
    fn exact_vulkan_graph_config_defaults_decoder_to_cpu() {
        let preference = exact_route(crate::device::execution_route::ExecutionProvider::Vulkan);
        with_decoder_env(None, || {
            let _guard = crate::ggml_runtime::install_request_backend_override(Some(preference));
            let config = moonshine_decoder_graph_config(GgmlCpuGraphBackend::Gpu);
            assert_eq!(config.backend, GgmlCpuGraphBackend::Cpu);
            assert!(!config.use_scheduler);
        });
    }

    #[test]
    fn exact_cuda_and_hip_graph_configs_keep_decoder_on_gpu() {
        with_decoder_env(None, || {
            for provider in [
                crate::device::execution_route::ExecutionProvider::Cuda,
                crate::device::execution_route::ExecutionProvider::Hip,
            ] {
                let _guard = crate::ggml_runtime::install_request_backend_override(Some(
                    exact_route(provider),
                ));
                let config = moonshine_decoder_graph_config(GgmlCpuGraphBackend::Gpu);
                assert_eq!(config.backend, GgmlCpuGraphBackend::Gpu, "{provider:?}");
                assert!(!config.use_scheduler, "{provider:?} must remain FullDevice");
            }
        });
    }

    #[test]
    fn explicit_gpu_stage_override_can_force_vulkan_decoder() {
        let preference = exact_route(crate::device::execution_route::ExecutionProvider::Vulkan);
        with_decoder_env(Some("1"), || {
            let _guard = crate::ggml_runtime::install_request_backend_override(Some(preference));
            let config = moonshine_decoder_graph_config(GgmlCpuGraphBackend::Gpu);
            assert_eq!(config.backend, GgmlCpuGraphBackend::Gpu);
            assert!(!config.use_scheduler);
        });
    }

    #[test]
    fn non_vulkan_full_device_backends_ignore_cpu_stage_override() {
        with_decoder_env(Some("0"), || {
            let preference = exact_route(crate::device::execution_route::ExecutionProvider::Cuda);
            let _guard = crate::ggml_runtime::install_request_backend_override(Some(preference));
            assert_eq!(
                moonshine_decoder_graph_config(GgmlCpuGraphBackend::Gpu).backend,
                GgmlCpuGraphBackend::Gpu
            );
            assert!(decoder_gpu_enabled_with_inputs(
                GgmlCpuGraphBackend::Metal,
                Some("0"),
                None,
            ));
        });
    }
}
