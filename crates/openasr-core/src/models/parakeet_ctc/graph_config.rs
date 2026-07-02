use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    gpu_stage_enabled_for_backend,
};

const OPENASR_PARAKEET_CTC_ENABLE_ENCODER_GPU: &str = "OPENASR_PARAKEET_CTC_ENABLE_ENCODER_GPU";

pub(crate) fn parakeet_ctc_encoder_graph_config() -> GgmlCpuGraphConfig {
    let mut config = configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::default(),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: None,
            default_n_threads_when_unset: None,
        },
    );
    if config.backend.is_gpu_class() && !encoder_gpu_enabled(config.backend) {
        config.backend = GgmlCpuGraphBackend::Cpu;
        config.use_scheduler = false;
    }
    config
}

fn encoder_gpu_enabled(backend: GgmlCpuGraphBackend) -> bool {
    gpu_stage_enabled_for_backend(
        backend,
        OPENASR_PARAKEET_CTC_ENABLE_ENCODER_GPU,
        true,
        None,
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::graph_runtime_config::gpu_stage_enabled_for_backend_raw;

    #[test]
    fn defaults_gpu_encoder_to_direct_single_backend() {
        let mut config = GgmlCpuGraphConfig::conservative_default();
        config.backend = GgmlCpuGraphBackend::Gpu;
        config.use_scheduler = true;
        let config = crate::models::graph_runtime_config::configure_model_runtime_graph_config(
            config,
            false,
            false,
            ModelMetalRuntimeOverrides {
                default_use_scheduler_when_unset: None,
                default_n_threads_when_unset: None,
            },
        );
        assert!(!config.use_scheduler);
    }

    #[test]
    fn gpu_encoder_can_fallback_to_cpu() {
        assert!(!gpu_stage_enabled_for_backend_raw(
            GgmlCpuGraphBackend::Gpu,
            Some("0"),
            true,
            None,
            true,
        ));
    }
}
