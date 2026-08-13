use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphThreadingWorkload};
#[cfg(test)]
use crate::models::graph_runtime_config::configure_model_runtime_graph_config;
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    has_explicit_thread_override,
};

/// qwen is NOT gated (`AutoGpuPolicy::AllBackends`, see `arch::mod`), and
/// that is the *measured-correct* default: a 2026-07 re-measurement across
/// the full `size x quant` matrix (0.6b/1.7b x q4_k/q8_0/fp16, M1, warm
/// compute medians) has Metal *faster* than CPU everywhere -- 1.7x-2.4x,
/// e.g. 1.7b @ q8_0 RTF 0.327 (CPU) vs 0.180 (Metal) on an 11s clip and
/// 0.387 vs 0.159 on a 69s clip. An early 2026-07-04 measurement of the same
/// 1.7b @ q8_0 config had Metal 1.71x *slower*; that slowdown was an
/// artifact of the pre-rework decode path and no longer reproduces after the
/// decode changes (persistent reused decode graph, resident KV arena,
/// batched resident prefill seeding). Do not re-gate Metal off that stale
/// number. Should a future platform regression appear, the `AutoGpuPolicy`
/// gate machinery already exists (see
/// `xasr_zipformer::graph_config::encoder_gpu_enabled` for the pattern).
pub(crate) fn qwen_runtime_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: None,
            default_n_threads_when_unset: Some(1),
        },
    )
}

/// Graph config for the audio-encoder runner. The encoder runs the whole
/// utterance's conformer stack as one graph call with wide per-layer
/// parallelism (many independent frames / attention heads) -- the same shape
/// as firered-aed's encoder, which is why it takes the same `EncoderPrelude`
/// tier (see `firered_aed::graph_config::firered_encoder_graph_config` for
/// the precedent this mirrors). Unlike the decode path below, there is no
/// per-token reuse here: one call per chunk, so maximizing threads for that
/// one call is a clear win, not a per-step overhead trade-off.
pub(crate) fn qwen_encoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    let mut config = qwen_runtime_graph_config(backend);
    if !has_explicit_thread_override() {
        config.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            config.backend,
            GgmlCpuGraphThreadingWorkload::EncoderPrelude,
        );
    }
    config
}

/// Graph config for the LLM decode-path runners (the whole-decoder executor
/// and the logits head that feeds off it). Both are resident graphs reused
/// across the whole pack lifetime, dominated by thousands of single-token
/// autoregressive decode-step calls versus a handful of larger prefill
/// chunks; per-token graphs have little row-level parallelism to hand out
/// regardless of thread count. This takes the `Decoder` tier, mirroring
/// firered-aed's decoder graph
/// (`firered_aed::graph_config::firered_decoder_graph_config`) and
/// deliberately requesting fewer threads than `Default` to cut thread-pool
/// wake/join overhead on the dominant small-graph call instead of
/// over-provisioning threads the per-token op mix cannot use. mimo-asr,
/// firered2-llm and moss-transcribe-diarize construct their
/// whole-decoder executor through the shared prepared-plan compile seam, so
/// they inherit this tier automatically without a family-local constructor.
pub(crate) fn qwen_decoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    let mut config = qwen_runtime_graph_config(backend);
    if !has_explicit_thread_override() {
        config.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            config.backend,
            GgmlCpuGraphThreadingWorkload::Decoder,
        );
    }
    config
}

#[cfg(test)]
fn qwen_runtime_graph_config_with_overrides(
    base: GgmlCpuGraphConfig,
    has_explicit_scheduler_override: bool,
    has_explicit_thread_override: bool,
) -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config(
        base,
        has_explicit_scheduler_override,
        has_explicit_thread_override,
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: None,
            default_n_threads_when_unset: Some(1),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_qwen_metal_threads_to_one_without_explicit_override() {
        let config = qwen_runtime_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                n_threads: Some(4),
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
            false,
        );

        assert_eq!(config.n_threads, Some(1));
        assert!(!config.use_scheduler);
    }

    #[test]
    fn keeps_qwen_explicit_thread_override_on_metal() {
        let config = qwen_runtime_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                n_threads: Some(6),
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
            true,
        );

        assert_eq!(config.n_threads, Some(6));
    }

    #[test]
    fn does_not_force_single_thread_for_cpu_backend() {
        let config = qwen_runtime_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Cpu,
                n_threads: Some(7),
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
            false,
        );

        assert_eq!(config.n_threads, Some(7));
        assert!(config.use_scheduler);
    }
}
