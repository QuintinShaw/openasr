use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphThreadingWorkload};
#[cfg(test)]
use crate::models::graph_runtime_config::configure_model_runtime_graph_config;
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    has_explicit_thread_override,
};

pub(crate) fn cohere_runtime_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: Some(true),
            default_n_threads_when_unset: Some(1),
        },
    )
}

pub(crate) fn cohere_decoder_graph_config(
    backend: GgmlCpuGraphBackend,
    prefer_cpu_backend: bool,
) -> GgmlCpuGraphConfig {
    let mut config = cohere_runtime_graph_config(backend);
    if config.backend.is_gpu_class() && prefer_cpu_backend {
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

pub(crate) fn cohere_encoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    cohere_encoder_graph_config_with_overrides(
        cohere_runtime_graph_config(backend),
        has_explicit_thread_override(),
    )
}

fn cohere_encoder_graph_config_with_overrides(
    mut base: GgmlCpuGraphConfig,
    has_explicit_thread_override: bool,
) -> GgmlCpuGraphConfig {
    base.graph_size = base.graph_size.max(16_384);
    if !has_explicit_thread_override {
        base.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            base.backend,
            GgmlCpuGraphThreadingWorkload::EncoderPrelude,
        );
    }
    base
}

#[cfg(test)]
fn cohere_runtime_graph_config_with_overrides(
    base: GgmlCpuGraphConfig,
    has_explicit_scheduler_override: bool,
    has_explicit_thread_override: bool,
) -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config(
        base,
        has_explicit_scheduler_override,
        has_explicit_thread_override,
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: Some(true),
            default_n_threads_when_unset: Some(1),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_decoder_metal_threads_to_decoder_profile() {
        let mut config = cohere_runtime_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                n_threads: Some(8),
                use_scheduler: false,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
            false,
        );
        config.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            config.backend,
            GgmlCpuGraphThreadingWorkload::Decoder,
        );

        assert_eq!(
            config.n_threads,
            GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
                GgmlCpuGraphBackend::Metal,
                GgmlCpuGraphThreadingWorkload::Decoder,
            )
        );
        assert!(config.use_scheduler);
    }

    #[test]
    fn keeps_explicit_scheduler_override_on_cohere_metal() {
        let config = cohere_runtime_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                use_scheduler: false,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            true,
            false,
        );

        assert!(!config.use_scheduler);
        assert_eq!(config.n_threads, Some(1));
    }

    #[test]
    fn encoder_defaults_metal_runtime_to_metal_backend() {
        let config = cohere_encoder_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
        );

        assert!(matches!(config.backend, GgmlCpuGraphBackend::Metal));
    }

    #[test]
    fn encoder_preserves_the_resolved_generic_gpu_backend() {
        let config = cohere_encoder_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Gpu,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
        );

        assert!(matches!(config.backend, GgmlCpuGraphBackend::Gpu));
    }

    #[test]
    fn decoder_can_prefer_cpu_backend_for_longform_policy() {
        let generic_backend = GgmlCpuGraphConfig::runtime_default().backend;
        let config = cohere_decoder_graph_config(generic_backend, true);
        if matches!(generic_backend, GgmlCpuGraphBackend::Metal) {
            assert!(matches!(config.backend, GgmlCpuGraphBackend::Cpu));
            assert!(!config.use_scheduler);
        }
    }
}
