use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphThreadingWorkload};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    gpu_stage_enabled_for_backend, has_explicit_thread_override,
};

const OPENASR_MOONSHINE_ENABLE_ENCODER_METAL: &str = "OPENASR_MOONSHINE_ENABLE_ENCODER_METAL";
const OPENASR_MOONSHINE_ENABLE_DECODER_METAL: &str = "OPENASR_MOONSHINE_ENABLE_DECODER_METAL";
const OPENASR_MOONSHINE_ENABLE_ENCODER_GPU: &str = "OPENASR_MOONSHINE_ENABLE_ENCODER_GPU";
const OPENASR_MOONSHINE_ENABLE_DECODER_GPU: &str = "OPENASR_MOONSHINE_ENABLE_DECODER_GPU";

pub(crate) fn moonshine_runtime_graph_config() -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::default(),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: Some(true),
            default_n_threads_when_unset: Some(1),
        },
    )
}

pub(crate) fn moonshine_encoder_graph_config() -> GgmlCpuGraphConfig {
    let mut config = moonshine_runtime_graph_config();
    config.graph_size = config.graph_size.max(16_384);
    if config.backend.is_gpu_class() && !encoder_gpu_enabled(config.backend) {
        config.backend = GgmlCpuGraphBackend::Cpu;
        config.use_scheduler = false;
    }
    if !has_explicit_thread_override() {
        config.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            config.backend,
            GgmlCpuGraphThreadingWorkload::EncoderPrelude,
        );
    }
    config
}

pub(crate) fn moonshine_decoder_graph_config(prefer_cpu_backend: bool) -> GgmlCpuGraphConfig {
    let mut config = moonshine_runtime_graph_config();
    if config.backend.is_gpu_class() && (!decoder_gpu_enabled(config.backend) || prefer_cpu_backend)
    {
        config.backend = GgmlCpuGraphBackend::Cpu;
        config.use_scheduler = false;
    }
    if !has_explicit_thread_override() {
        config.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            config.backend,
            GgmlCpuGraphThreadingWorkload::Decoder,
        );
    }
    config
}

fn encoder_gpu_enabled(backend: GgmlCpuGraphBackend) -> bool {
    gpu_stage_enabled_for_backend(
        backend,
        OPENASR_MOONSHINE_ENABLE_ENCODER_GPU,
        true,
        Some(OPENASR_MOONSHINE_ENABLE_ENCODER_METAL),
        true,
    )
}

fn decoder_gpu_enabled(backend: GgmlCpuGraphBackend) -> bool {
    gpu_stage_enabled_for_backend(
        backend,
        OPENASR_MOONSHINE_ENABLE_DECODER_GPU,
        true,
        Some(OPENASR_MOONSHINE_ENABLE_DECODER_METAL),
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_gpu_defaults_to_unified_gpu_lane() {
        assert!(decoder_gpu_enabled(GgmlCpuGraphBackend::Gpu));
    }

    #[test]
    fn decoder_gpu_keeps_cpu_and_metal_defaults() {
        assert!(decoder_gpu_enabled(GgmlCpuGraphBackend::Cpu));
        assert!(decoder_gpu_enabled(GgmlCpuGraphBackend::Metal));
    }
}
