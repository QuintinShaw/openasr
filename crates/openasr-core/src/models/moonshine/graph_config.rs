use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphThreadingWorkload};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    gpu_stage_enabled_for_backend, has_explicit_thread_override,
};

const OPENASR_MOONSHINE_ENABLE_ENCODER_METAL: &str = "OPENASR_MOONSHINE_ENABLE_ENCODER_METAL";
const OPENASR_MOONSHINE_ENABLE_DECODER_METAL: &str = "OPENASR_MOONSHINE_ENABLE_DECODER_METAL";
const OPENASR_MOONSHINE_ENABLE_ENCODER_GPU: &str = "OPENASR_MOONSHINE_ENABLE_ENCODER_GPU";
const OPENASR_MOONSHINE_ENABLE_DECODER_GPU: &str = "OPENASR_MOONSHINE_ENABLE_DECODER_GPU";

/// Shared base for both stages: everything except the scheduler default,
/// which the encoder and decoder now set independently (see
/// [`moonshine_encoder_graph_config`] / [`moonshine_decoder_graph_config`]).
fn moonshine_runtime_graph_config_with_scheduler_default(
    default_use_scheduler_when_unset: Option<bool>,
) -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::default(),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset,
            default_n_threads_when_unset: Some(1),
        },
    )
}

pub(crate) fn moonshine_encoder_graph_config() -> GgmlCpuGraphConfig {
    // The encoder keeps the scheduler on for Metal: multi-backend liveness
    // allocation (`prepare_outputs_for_upload`'s gallocr) is how the encoder
    // forward graph gets built today, and it has not been re-verified to run
    // correctly with the scheduler off. Only the decoder's `use_scheduler`
    // default changed (see `moonshine_decoder_graph_config`).
    let mut config = moonshine_runtime_graph_config_with_scheduler_default(Some(true));
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

/// Decode-graph reuse (`nn::decoder::reusable_decode_graph_supported`) only
/// activates when the backend is GPU-class *and* the scheduler is off (a
/// multi-backend scheduler's `sched_alloc_graph` drops the per-token inputs
/// a reused, in-place-KV graph depends on). The decoder previously inherited
/// the encoder's `default_use_scheduler_when_unset: Some(true)`, which meant
/// Metal decode never got the persistent incremental-step graph
/// (`compute_incremental_step_logits`) and always fell back to rebuilding a
/// full-prefix graph every token (`compute_full_prefix_step_logits`) -- an
/// O(n^2) cost with no large encoder to amortize it against, measured 1.67x
/// slower than CPU. Leaving this `None` keeps the base (scheduler-off on
/// GPU-class backends, see `configure_model_graph_config`) so Metal decode
/// now gets the same persistent reused graph qwen's decoder already uses.
/// This is a pure backend/scheduling choice: output must stay byte-identical
/// (verified via the moonshine golden test), since it does not change which
/// arithmetic runs, only whether the graph is rebuilt per token.
pub(crate) fn moonshine_decoder_graph_config(prefer_cpu_backend: bool) -> GgmlCpuGraphConfig {
    let mut config = moonshine_runtime_graph_config_with_scheduler_default(None);
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
