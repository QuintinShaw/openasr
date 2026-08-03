use std::cell::Cell;

use crate::device::execution_policy::ExecutionPlacement;
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig};

use super::ggml_graph_config::configure_model_graph_config;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ModelMetalRuntimeOverrides {
    pub default_use_scheduler_when_unset: Option<bool>,
    pub default_n_threads_when_unset: Option<usize>,
}

thread_local! {
    static REQUEST_INFERENCE_THREADS: Cell<Option<usize>> = const { Cell::new(None) };
}

pub(crate) struct RequestInferenceThreadsOverrideGuard {
    previous: Option<usize>,
}

impl Drop for RequestInferenceThreadsOverrideGuard {
    fn drop(&mut self) {
        REQUEST_INFERENCE_THREADS.with(|threads| threads.set(self.previous));
    }
}

pub(crate) fn install_request_inference_threads_override(
    inference_threads: Option<usize>,
) -> RequestInferenceThreadsOverrideGuard {
    let previous = REQUEST_INFERENCE_THREADS.with(|threads| {
        let previous = threads.get();
        threads.set(inference_threads);
        previous
    });
    RequestInferenceThreadsOverrideGuard { previous }
}

pub(crate) fn request_inference_threads_override() -> Option<usize> {
    REQUEST_INFERENCE_THREADS.with(Cell::get)
}

pub(crate) fn has_explicit_thread_override() -> bool {
    std::env::var_os(GgmlCpuGraphConfig::THREADS_ENV).is_some()
        || request_inference_threads_override().is_some()
}

pub(crate) fn apply_request_inference_threads_override(
    mut config: GgmlCpuGraphConfig,
) -> GgmlCpuGraphConfig {
    if let Some(inference_threads) = request_inference_threads_override() {
        config.n_threads = Some(inference_threads);
    }
    config
}

/// Make the policy candidate's placement an executable request contract.
/// Family defaults and operator scheduler knobs are resolved first; an active
/// candidate then wins because fallback must attempt a materially distinct
/// placement rather than relabel the same graph configuration.
pub(crate) fn apply_request_execution_placement(config: GgmlCpuGraphConfig) -> GgmlCpuGraphConfig {
    let Some(placement) = crate::models::native_execution_services::current_execution_placement()
    else {
        return config;
    };
    apply_execution_placement(config, placement)
}

fn apply_execution_placement(
    mut config: GgmlCpuGraphConfig,
    placement: ExecutionPlacement,
) -> GgmlCpuGraphConfig {
    match placement {
        ExecutionPlacement::CpuOnly => {
            config.backend = GgmlCpuGraphBackend::Cpu;
            // A scheduler is not synonymous with a CPU/device hybrid. With a
            // CPU backend the scheduler's optional helpers are CPU-class
            // accelerators (for example BLAS), so preserving the family's
            // validated scheduler choice still satisfies CpuOnly. Disabling
            // it here silently selected different kernels and changed model
            // numerics for the same explicit CPU request.
        }
        ExecutionPlacement::FullDevice => {
            // A policy plan never emits FullDevice for a CPU route.
            if config.backend.is_gpu_class() {
                config.use_scheduler = false;
            }
        }
        ExecutionPlacement::Hybrid => {
            // The ggml multi-backend scheduler is the implemented CPU/device
            // split path. CPU-only stages remain direct CPU stages.
            if config.backend.is_gpu_class() {
                config.use_scheduler = true;
            }
        }
    }
    config
}

pub(crate) fn configure_model_runtime_graph_config(
    base: GgmlCpuGraphConfig,
    has_explicit_scheduler_override: bool,
    has_explicit_thread_override: bool,
    metal_overrides: ModelMetalRuntimeOverrides,
) -> GgmlCpuGraphConfig {
    let mut config = configure_model_graph_config(base, has_explicit_scheduler_override);
    config = apply_request_inference_threads_override(config);
    if matches!(config.backend, GgmlCpuGraphBackend::Metal) {
        if !has_explicit_scheduler_override
            && let Some(default_use_scheduler) = metal_overrides.default_use_scheduler_when_unset
        {
            config.use_scheduler = default_use_scheduler;
        }
        if !has_explicit_thread_override
            && let Some(default_n_threads) = metal_overrides.default_n_threads_when_unset
        {
            config.n_threads = Some(default_n_threads);
        }
    }
    apply_request_execution_placement(config)
}

pub(crate) fn configure_model_runtime_graph_config_from_env(
    base: GgmlCpuGraphConfig,
    metal_overrides: ModelMetalRuntimeOverrides,
) -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config(
        base,
        std::env::var_os(GgmlCpuGraphConfig::USE_SCHEDULER_ENV).is_some(),
        has_explicit_thread_override(),
        metal_overrides,
    )
}

pub(crate) fn gpu_stage_enabled_for_backend(
    backend: GgmlCpuGraphBackend,
    gpu_env: &str,
    default_gpu_enabled: bool,
    legacy_metal_env: Option<&str>,
    default_metal_enabled: bool,
) -> bool {
    let gpu_raw = std::env::var(gpu_env).ok();
    let legacy_metal_raw = legacy_metal_env.and_then(|name| std::env::var(name).ok());
    // An explicit `execution_target=accelerated` request always wins over a
    // stage's tuned Auto-mode default -- these per-stage knobs (encoder
    // prelude, decoder, ...) exist to keep Auto from picking a GPU path that
    // measured worse for that specific op mix, not to second-guess a user who
    // explicitly asked for acceleration. An explicit env var still wins over
    // this (an operator-set kill switch is a deployment decision, not the
    // engine choosing on the user's behalf), so only the *default* shifts.
    let explicit_accelerated = match crate::ggml_runtime::request_backend_override() {
        Some(crate::ggml_runtime::RequestBackendPreference::Accelerated) => true,
        Some(crate::ggml_runtime::RequestBackendPreference::Exact(route)) => {
            route.provider != crate::device::execution_route::ExecutionProvider::Cpu
        }
        Some(crate::ggml_runtime::RequestBackendPreference::CpuOnly) | None => false,
    };
    gpu_stage_enabled_for_backend_raw(
        backend,
        gpu_raw.as_deref(),
        default_gpu_enabled || explicit_accelerated,
        legacy_metal_raw.as_deref(),
        default_metal_enabled || explicit_accelerated,
    )
}

pub(crate) fn gpu_stage_enabled_for_backend_raw(
    backend: GgmlCpuGraphBackend,
    gpu_raw: Option<&str>,
    default_gpu_enabled: bool,
    legacy_metal_raw: Option<&str>,
    default_metal_enabled: bool,
) -> bool {
    match backend {
        GgmlCpuGraphBackend::Cpu => true,
        GgmlCpuGraphBackend::Metal => env_toggle_with_optional_legacy(
            gpu_raw,
            default_gpu_enabled,
            legacy_metal_raw,
            default_metal_enabled,
        ),
        GgmlCpuGraphBackend::Gpu => {
            crate::ggml_runtime::env_toggle_with_raw(None, gpu_raw, default_gpu_enabled)
        }
    }
}

fn env_toggle_with_optional_legacy(
    gpu_raw: Option<&str>,
    default_gpu_enabled: bool,
    legacy_metal_raw: Option<&str>,
    default_metal_enabled: bool,
) -> bool {
    if legacy_metal_raw.is_some() {
        return crate::ggml_runtime::env_toggle_with_raw(
            None,
            legacy_metal_raw,
            default_metal_enabled,
        );
    }
    crate::ggml_runtime::env_toggle_with_raw(None, gpu_raw, default_gpu_enabled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_metal_scheduler_override_when_scheduler_env_is_unset() {
        let config = configure_model_runtime_graph_config(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                use_scheduler: false,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
            false,
            ModelMetalRuntimeOverrides {
                default_use_scheduler_when_unset: Some(true),
                default_n_threads_when_unset: None,
            },
        );
        assert!(config.use_scheduler);
    }

    #[test]
    fn keeps_explicit_scheduler_override_on_metal() {
        let config = configure_model_runtime_graph_config(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                use_scheduler: false,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            true,
            false,
            ModelMetalRuntimeOverrides {
                default_use_scheduler_when_unset: Some(true),
                default_n_threads_when_unset: None,
            },
        );
        assert!(!config.use_scheduler);
    }

    #[test]
    fn applies_metal_thread_override_when_thread_env_is_unset() {
        let config = configure_model_runtime_graph_config(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                n_threads: Some(8),
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
            false,
            ModelMetalRuntimeOverrides {
                default_use_scheduler_when_unset: None,
                default_n_threads_when_unset: Some(1),
            },
        );
        assert_eq!(config.n_threads, Some(1));
    }

    #[test]
    fn keeps_explicit_thread_override_on_metal() {
        let config = configure_model_runtime_graph_config(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                n_threads: Some(8),
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
            true,
            ModelMetalRuntimeOverrides {
                default_use_scheduler_when_unset: None,
                default_n_threads_when_unset: Some(1),
            },
        );
        assert_eq!(config.n_threads, Some(8));
    }

    #[test]
    fn does_not_apply_metal_overrides_to_cpu_backend() {
        let config = configure_model_runtime_graph_config(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Cpu,
                n_threads: Some(8),
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
            false,
            ModelMetalRuntimeOverrides {
                default_use_scheduler_when_unset: Some(false),
                default_n_threads_when_unset: Some(1),
            },
        );
        assert_eq!(config.n_threads, Some(8));
        assert!(config.use_scheduler);
    }

    #[test]
    fn cpu_only_preserves_cpu_scheduler_and_cpu_class_accelerators() {
        let config = apply_execution_placement(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            ExecutionPlacement::CpuOnly,
        );

        assert_eq!(config.backend, GgmlCpuGraphBackend::Cpu);
        assert!(config.use_scheduler);
    }

    #[test]
    fn request_thread_override_beats_metal_default() {
        let _guard = install_request_inference_threads_override(Some(3));
        let config = configure_model_runtime_graph_config_from_env(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                n_threads: Some(8),
                ..GgmlCpuGraphConfig::conservative_default()
            },
            ModelMetalRuntimeOverrides {
                default_use_scheduler_when_unset: None,
                default_n_threads_when_unset: Some(1),
            },
        );

        assert_eq!(config.n_threads, Some(3));
    }

    #[test]
    fn gpu_stage_knob_applies_to_generic_gpu() {
        assert!(!gpu_stage_enabled_for_backend_raw(
            GgmlCpuGraphBackend::Gpu,
            Some("0"),
            true,
            None,
            true,
        ));
    }

    #[test]
    fn metal_stage_knob_prefers_legacy_env_when_set() {
        assert!(!gpu_stage_enabled_for_backend_raw(
            GgmlCpuGraphBackend::Metal,
            Some("1"),
            true,
            Some("0"),
            true,
        ));
    }

    /// Regression for a stage gate whose tuned Auto default disables a
    /// backend (`default_gpu_enabled = false`, e.g. a hypothetical future
    /// per-stage knob tuned off by default on some host class): with no env
    /// var set, an explicit `execution_target=accelerated` request must
    /// still enable the stage, not silently inherit the Auto-tuned default.
    /// None of today's builtin stage gates actually reach this path in the
    /// common (env-unset) case -- they all default enabled -- but every
    /// gate goes through this one function, so this pins the override
    /// priority for the whole class rather than per family.
    #[test]
    fn explicit_accelerated_overrides_a_stage_gate_disabled_by_default() {
        use crate::ggml_runtime::{RequestBackendPreference, install_request_backend_override};

        assert!(!gpu_stage_enabled_for_backend(
            GgmlCpuGraphBackend::Metal,
            "OPENASR_TEST_STAGE_ENABLE_GPU_NEVER_SET",
            false,
            Some("OPENASR_TEST_STAGE_ENABLE_METAL_NEVER_SET"),
            false,
        ));

        let _guard = install_request_backend_override(Some(RequestBackendPreference::Accelerated));
        assert!(gpu_stage_enabled_for_backend(
            GgmlCpuGraphBackend::Metal,
            "OPENASR_TEST_STAGE_ENABLE_GPU_NEVER_SET",
            false,
            Some("OPENASR_TEST_STAGE_ENABLE_METAL_NEVER_SET"),
            false,
        ));
    }

    #[test]
    fn request_backend_override_forces_resolution() {
        use crate::ggml_runtime::{
            GgmlCpuGraphConfig, RequestBackendPreference, install_request_backend_override,
        };

        {
            let _guard = install_request_backend_override(Some(RequestBackendPreference::CpuOnly));
            let config = configure_model_runtime_graph_config_from_env(
                GgmlCpuGraphConfig::runtime_default(),
                ModelMetalRuntimeOverrides::default(),
            );
            assert_eq!(config.backend, GgmlCpuGraphBackend::Cpu);
        }

        #[cfg(target_os = "macos")]
        {
            let _guard =
                install_request_backend_override(Some(RequestBackendPreference::Accelerated));
            let config = configure_model_runtime_graph_config_from_env(
                GgmlCpuGraphConfig::runtime_default(),
                ModelMetalRuntimeOverrides::default(),
            );
            assert_eq!(config.backend, GgmlCpuGraphBackend::Metal);
        }
    }

    #[test]
    fn request_backend_override_guard_restores_previous() {
        use crate::ggml_runtime::{
            RequestBackendPreference, install_request_backend_override, request_backend_override,
        };

        let outer = install_request_backend_override(Some(RequestBackendPreference::CpuOnly));
        {
            let _inner =
                install_request_backend_override(Some(RequestBackendPreference::Accelerated));
            assert_eq!(
                request_backend_override(),
                Some(RequestBackendPreference::Accelerated)
            );
        }
        assert_eq!(
            request_backend_override(),
            Some(RequestBackendPreference::CpuOnly)
        );
        drop(outer);
        assert_eq!(request_backend_override(), None);
    }
}
