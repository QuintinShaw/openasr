// Consumed by the executor wired in the follow-up stage; tested meanwhile.
#![allow(dead_code)]

use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    gpu_stage_enabled_for_backend,
};

const OPENASR_PARAKEET_TDT_ENABLE_ENCODER_GPU: &str = "OPENASR_PARAKEET_TDT_ENABLE_ENCODER_GPU";

pub(crate) fn parakeet_tdt_encoder_graph_config() -> GgmlCpuGraphConfig {
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
        OPENASR_PARAKEET_TDT_ENABLE_ENCODER_GPU,
        true,
        None,
        true,
    )
}
