use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::ggml_runtime::GgmlCpuGraphConfig;
use crate::ggml_runtime::GgmlCpuGraphThreadingWorkload;
#[cfg(test)]
use crate::models::graph_runtime_config::configure_model_runtime_graph_config;
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    gpu_stage_enabled_for_backend, has_explicit_thread_override,
};

const OPENASR_WHISPER_ENABLE_ENCODER_PRELUDE_GPU: &str =
    "OPENASR_WHISPER_ENABLE_ENCODER_PRELUDE_GPU";
const OPENASR_WHISPER_ENABLE_ENCODER_PRELUDE_METAL: &str =
    "OPENASR_WHISPER_GGML_ENABLE_ENCODER_PRELUDE_METAL";

pub(crate) fn whisper_runtime_graph_config() -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::default(),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: Some(true),
            default_n_threads_when_unset: Some(1),
        },
    )
}

pub(crate) fn whisper_encoder_prelude_graph_config() -> GgmlCpuGraphConfig {
    whisper_encoder_prelude_graph_config_with_overrides(
        whisper_runtime_graph_config(),
        whisper_encoder_prelude_gpu_enabled,
        has_explicit_thread_override(),
    )
}

pub(crate) fn whisper_decoder_graph_config() -> GgmlCpuGraphConfig {
    let mut config = whisper_runtime_graph_config();
    if !has_explicit_thread_override() {
        config.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            config.backend,
            GgmlCpuGraphThreadingWorkload::Decoder,
        );
    }
    config
}

fn whisper_encoder_prelude_gpu_enabled(backend: GgmlCpuGraphBackend) -> bool {
    gpu_stage_enabled_for_backend(
        backend,
        OPENASR_WHISPER_ENABLE_ENCODER_PRELUDE_GPU,
        true,
        Some(OPENASR_WHISPER_ENABLE_ENCODER_PRELUDE_METAL),
        false,
    )
}

fn whisper_encoder_prelude_graph_config_with_overrides(
    mut base: GgmlCpuGraphConfig,
    prelude_gpu_enabled: impl FnOnce(GgmlCpuGraphBackend) -> bool,
    has_explicit_thread_override: bool,
) -> GgmlCpuGraphConfig {
    if base.backend.is_gpu_class() && !prelude_gpu_enabled(base.backend) {
        base.backend = GgmlCpuGraphBackend::Cpu;
        base.use_scheduler = false;
    }
    if !has_explicit_thread_override {
        base.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            base.backend,
            crate::ggml_runtime::GgmlCpuGraphThreadingWorkload::EncoderPrelude,
        );
    }
    base
}

#[cfg(test)]
fn whisper_runtime_graph_config_with_overrides(
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
    fn defaults_whisper_metal_graphs_to_scheduler_when_not_overridden() {
        let config = whisper_runtime_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                use_scheduler: false,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
            false,
        );

        assert!(config.use_scheduler);
        assert_eq!(config.n_threads, Some(1));
    }

    #[test]
    fn keeps_explicit_scheduler_override_on_whisper_metal() {
        let config = whisper_runtime_graph_config_with_overrides(
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
    fn keeps_explicit_thread_override_on_whisper_metal() {
        let config = whisper_runtime_graph_config_with_overrides(
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
    fn keeps_cpu_scheduler_setting_when_not_overridden() {
        let config = whisper_runtime_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Cpu,
                use_scheduler: true,
                n_threads: Some(7),
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
            false,
        );

        assert!(config.use_scheduler);
        assert_eq!(config.n_threads, Some(7));
    }

    #[test]
    fn prelude_defaults_metal_runtime_to_cpu_backend() {
        let config = whisper_encoder_prelude_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            |_| false,
            false,
        );

        assert!(matches!(config.backend, GgmlCpuGraphBackend::Cpu));
        assert!(!config.use_scheduler);
    }

    #[test]
    fn prelude_can_explicitly_keep_metal_backend() {
        let config = whisper_encoder_prelude_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            |_| true,
            false,
        );

        assert!(matches!(config.backend, GgmlCpuGraphBackend::Metal));
    }

    #[test]
    fn prelude_defaults_gpu_runtime_to_cpu_backend() {
        let config = whisper_encoder_prelude_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Gpu,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            |_| false,
            false,
        );

        assert!(matches!(config.backend, GgmlCpuGraphBackend::Cpu));
        assert!(!config.use_scheduler);
    }

    #[test]
    fn prelude_can_explicitly_keep_gpu_backend() {
        let config = whisper_encoder_prelude_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Gpu,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            |_| true,
            false,
        );

        assert!(matches!(config.backend, GgmlCpuGraphBackend::Gpu));
    }
}
