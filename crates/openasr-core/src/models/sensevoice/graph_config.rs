use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
};

pub(crate) fn sensevoice_encoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
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
    fn encoder_preserves_the_resolved_metal_backend() {
        assert_eq!(
            sensevoice_encoder_graph_config(GgmlCpuGraphBackend::Metal).backend,
            GgmlCpuGraphBackend::Metal
        );
    }
}
