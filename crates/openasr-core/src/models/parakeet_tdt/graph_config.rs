use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig, env_var_truthy};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    gpu_stage_enabled_for_backend,
};

const OPENASR_PARAKEET_TDT_ENABLE_ENCODER_GPU: &str = "OPENASR_PARAKEET_TDT_ENABLE_ENCODER_GPU";

/// Experiment (default off): cast selected mul_mat-adjacent conformer-block
/// activations to F16 to trade precision for Metal memory bandwidth. See
/// `nn::encoder::ConformerBlockConfig::f16_activations`.
const OPENASR_PARAKEET_TDT_ENCODER_F16_ACT: &str = "OPENASR_PARAKEET_TDT_ENCODER_F16_ACT";

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

/// Whether the encoder's experimental F16-activation conformer-block path is
/// enabled for `backend`. Metal-only and opt-in (default false): the CPU
/// backend has no measured bandwidth benefit and this has not been verified
/// there, so it always reports disabled regardless of the env var.
pub(crate) fn parakeet_tdt_encoder_f16_activations_enabled(backend: GgmlCpuGraphBackend) -> bool {
    matches!(backend, GgmlCpuGraphBackend::Metal)
        && env_var_truthy(OPENASR_PARAKEET_TDT_ENCODER_F16_ACT)
}
