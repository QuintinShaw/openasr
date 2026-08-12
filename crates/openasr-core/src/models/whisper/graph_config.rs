use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::ggml_runtime::GgmlCpuGraphConfig;
use crate::ggml_runtime::GgmlCpuGraphThreadingWorkload;
#[cfg(test)]
use crate::models::graph_runtime_config::configure_model_runtime_graph_config;
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    has_explicit_thread_override,
};

pub(crate) fn whisper_runtime_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: Some(true),
            default_n_threads_when_unset: Some(1),
        },
    )
}

pub(crate) fn whisper_encoder_prelude_graph_config(
    backend: GgmlCpuGraphBackend,
) -> GgmlCpuGraphConfig {
    whisper_encoder_prelude_graph_config_with_overrides(
        whisper_runtime_graph_config(backend),
        has_explicit_thread_override(),
    )
}

pub(crate) fn whisper_decoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    let mut config = whisper_runtime_graph_config(backend);
    if !has_explicit_thread_override() {
        config.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            config.backend,
            GgmlCpuGraphThreadingWorkload::Decoder,
        );
    }
    config
}

fn whisper_encoder_prelude_graph_config_with_overrides(
    mut base: GgmlCpuGraphConfig,
    has_explicit_thread_override: bool,
) -> GgmlCpuGraphConfig {
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
    fn prelude_preserves_the_resolved_metal_backend() {
        let config = whisper_encoder_prelude_graph_config_with_overrides(
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
    fn prelude_preserves_the_resolved_generic_gpu_backend() {
        let config = whisper_encoder_prelude_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Gpu,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
        );

        assert!(matches!(config.backend, GgmlCpuGraphBackend::Gpu));
    }

    /// A family's own graph-config path must return the backend the caller
    /// explicitly resolved and passed in -- never silently prefer a
    /// thread-local override installed behind its back. Building the base
    /// config from `GgmlCpuGraphConfig::default()` would read that TLS
    /// internally, so a family could observe a *different* backend than the
    /// one the shared dispatch resolved into the request. Installing a
    /// `CpuOnly` override and then passing `Metal` explicitly pins that this
    /// can never happen: the explicit parameter must win, full stop.
    #[test]
    fn family_graph_config_ignores_a_stale_tls_override_and_uses_the_explicit_backend() {
        let _guard = crate::ggml_runtime::install_request_backend_override(Some(
            crate::ggml_runtime::RequestBackendPreference::CpuOnly,
        ));

        let decoder_config = whisper_decoder_graph_config(GgmlCpuGraphBackend::Metal);
        assert_eq!(
            decoder_config.backend,
            GgmlCpuGraphBackend::Metal,
            "whisper_decoder_graph_config must return the explicit backend passed in, \
             not the CpuOnly value installed in the (unrelated, stale) TLS override"
        );

        let runtime_config = whisper_runtime_graph_config(GgmlCpuGraphBackend::Metal);
        assert_eq!(
            runtime_config.backend,
            GgmlCpuGraphBackend::Metal,
            "whisper_runtime_graph_config must return the explicit backend passed in, \
             not the CpuOnly value installed in the (unrelated, stale) TLS override"
        );
    }
}
