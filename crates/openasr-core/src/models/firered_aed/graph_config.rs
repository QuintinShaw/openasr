//! firered-aed ggml graph backend/threading policy.
//!
//! Stage 2/3 landed CPU-only by design (correctness-first, GPU staged as an
//! explicit follow-up once decoder/executor parity was established -- see the
//! prior module docs on [`super::encoder_graph`] and [`super::decoder_graph`]
//! for the CPU-only-era history). That parity is now verified end to end
//! (CPU vs Metal transcripts match byte-for-byte on real packs), so this
//! mirrors the cohere/moonshine template -- dynamic backend resolution via
//! [`configure_model_runtime_graph_config_from_env`] (Metal auto-selected
//! through `GgmlCpuGraphConfig::resolve_runtime_backend()`), with an explicit
//! per-stage opt-out that falls back to CPU. Unlike cohere, firered-aed never
//! does multi-chunk longform batching (the executor is single-segment plain
//! transcription -- see `executor.rs` module docs), so there is no
//! `prefer_cpu_backend` request-level override to thread through here.

use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphThreadingWorkload};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    gpu_stage_enabled_for_backend, has_explicit_thread_override,
};

const FIRERED_ENCODER_GRAPH_SIZE: usize = 32_768;
const FIRERED_DECODER_GRAPH_SIZE: usize = 8192;

const OPENASR_FIRERED_ENABLE_ENCODER_METAL: &str = "OPENASR_FIRERED_ENABLE_ENCODER_METAL";
const OPENASR_FIRERED_ENABLE_DECODER_METAL: &str = "OPENASR_FIRERED_ENABLE_DECODER_METAL";
const OPENASR_FIRERED_ENABLE_ENCODER_GPU: &str = "OPENASR_FIRERED_ENABLE_ENCODER_GPU";
const OPENASR_FIRERED_ENABLE_DECODER_GPU: &str = "OPENASR_FIRERED_ENABLE_DECODER_GPU";

pub(crate) fn firered_runtime_graph_config() -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::default(),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: Some(true),
            default_n_threads_when_unset: Some(1),
        },
    )
}

pub(crate) fn firered_encoder_graph_config() -> GgmlCpuGraphConfig {
    // `no_alloc` metadata context sized from the actual node count (see
    // `GgmlCpuGraphConfig::metadata_context_bytes`); previously a flat
    // hardcoded 512 MiB per cached encoder runtime (see the thread-local
    // cache in `executor.rs`).
    let mut config = firered_runtime_graph_config();
    config.graph_size = config.graph_size.max(FIRERED_ENCODER_GRAPH_SIZE);
    config.context_bytes = config
        .context_bytes
        .max(GgmlCpuGraphConfig::metadata_context_bytes(
            config.graph_size,
        ));
    if config.backend.is_gpu_class() && !firered_encoder_gpu_enabled(config.backend) {
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

pub(crate) fn firered_decoder_graph_config() -> GgmlCpuGraphConfig {
    // See the matching comment in `firered_encoder_graph_config`: this is a
    // `no_alloc` metadata pool sized from the actual node count, not the real
    // tensor bytes (those live in the arena's own backend buffer).
    let mut config = firered_runtime_graph_config();
    config.graph_size = config.graph_size.max(FIRERED_DECODER_GRAPH_SIZE);
    config.context_bytes = config
        .context_bytes
        .max(GgmlCpuGraphConfig::metadata_context_bytes(
            config.graph_size,
        ));
    if config.backend.is_gpu_class() && !firered_decoder_gpu_enabled(config.backend) {
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

fn firered_encoder_gpu_enabled(backend: GgmlCpuGraphBackend) -> bool {
    gpu_stage_enabled_for_backend(
        backend,
        OPENASR_FIRERED_ENABLE_ENCODER_GPU,
        true,
        Some(OPENASR_FIRERED_ENABLE_ENCODER_METAL),
        true,
    )
}

fn firered_decoder_gpu_enabled(backend: GgmlCpuGraphBackend) -> bool {
    gpu_stage_enabled_for_backend(
        backend,
        OPENASR_FIRERED_ENABLE_DECODER_GPU,
        true,
        Some(OPENASR_FIRERED_ENABLE_DECODER_METAL),
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_gpu_defaults_to_unified_gpu_lane() {
        assert!(firered_encoder_gpu_enabled(GgmlCpuGraphBackend::Gpu));
    }

    #[test]
    fn decoder_gpu_defaults_to_unified_gpu_lane() {
        assert!(firered_decoder_gpu_enabled(GgmlCpuGraphBackend::Gpu));
    }

    #[test]
    fn encoder_and_decoder_gpu_keep_cpu_and_metal_defaults() {
        assert!(firered_encoder_gpu_enabled(GgmlCpuGraphBackend::Cpu));
        assert!(firered_encoder_gpu_enabled(GgmlCpuGraphBackend::Metal));
        assert!(firered_decoder_gpu_enabled(GgmlCpuGraphBackend::Cpu));
        assert!(firered_decoder_gpu_enabled(GgmlCpuGraphBackend::Metal));
    }

    #[test]
    fn encoder_graph_size_floor_is_preserved() {
        assert!(firered_encoder_graph_config().graph_size >= FIRERED_ENCODER_GRAPH_SIZE);
    }

    #[test]
    fn decoder_graph_size_floor_is_preserved() {
        assert!(firered_decoder_graph_config().graph_size >= FIRERED_DECODER_GRAPH_SIZE);
    }
}
