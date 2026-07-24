use crate::ggml_runtime::{GgmlCpuGraphConfig, GgmlCpuGraphThreadingWorkload};
#[cfg(test)]
use crate::models::graph_runtime_config::configure_model_runtime_graph_config;
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    has_explicit_thread_override,
};

/// qwen is NOT gated (`AutoGpuPolicy::AllBackends`, see `arch::mod`) as of
/// this audit. The measured 1.71x Metal slowdown at qwen's recommended 1.7B
/// @ q8_0 config looks like a fixed `size x quant` platform trade-off rather
/// than a qwen-specific code bug -- mimo/firered-llm share this exact decode
/// driver (fused logits head, on-device argmax, persistent reused decode
/// graph, scheduler-off) and measure clearly *faster* on Metal at their 8B @
/// q4_k config -- but that read is not yet confirmed by a dedicated
/// follow-up investigation, so it is deliberately left un-gated rather than
/// baking an unconfirmed platform verdict into the default. If a follow-up
/// study confirms the trade-off (or finds an actual fix), flipping qwen to
/// `AutoGpuPolicy::ExceptMetal` here and in its `arch::mod` descriptor is a
/// one-line change (the `AutoGpuPolicy` gate machinery already exists, see
/// `xasr_zipformer::graph_config::encoder_gpu_enabled` for the pattern).
pub(crate) fn qwen_runtime_graph_config() -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::default(),
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
pub(crate) fn qwen_encoder_graph_config() -> GgmlCpuGraphConfig {
    let mut config = qwen_runtime_graph_config();
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
/// firered2-llm, moss-transcribe-diarize, and hymt2 all construct their
/// whole-decoder executor through
/// `Qwen3AsrLlmWholeDecoderGraphExecutor::new_with_rms_norm_epsilon_and_fused_logits_head`,
/// so they inherit this tier automatically.
pub(crate) fn qwen_decoder_graph_config() -> GgmlCpuGraphConfig {
    let mut config = qwen_runtime_graph_config();
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
    use crate::ggml_runtime::GgmlCpuGraphBackend;

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
