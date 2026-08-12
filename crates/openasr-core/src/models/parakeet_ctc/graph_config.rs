use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
};

pub(crate) fn parakeet_ctc_encoder_graph_config(
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn encoder_preserves_the_resolved_metal_backend() {
        assert_eq!(
            parakeet_ctc_encoder_graph_config(GgmlCpuGraphBackend::Metal).backend,
            GgmlCpuGraphBackend::Metal
        );
    }
}
