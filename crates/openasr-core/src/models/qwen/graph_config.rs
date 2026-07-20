use crate::ggml_runtime::GgmlCpuGraphConfig;
#[cfg(test)]
use crate::models::graph_runtime_config::configure_model_runtime_graph_config;
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
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
