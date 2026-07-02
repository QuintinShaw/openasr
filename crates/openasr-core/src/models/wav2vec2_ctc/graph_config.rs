use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    gpu_stage_enabled_for_backend,
};

const OPENASR_WAV2VEC2_CTC_ENABLE_ENCODER_GPU: &str = "OPENASR_WAV2VEC2_CTC_ENABLE_ENCODER_GPU";

pub(crate) fn wav2vec2_ctc_encoder_graph_config() -> GgmlCpuGraphConfig {
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
        OPENASR_WAV2VEC2_CTC_ENABLE_ENCODER_GPU,
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
