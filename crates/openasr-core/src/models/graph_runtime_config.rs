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

pub(crate) fn apply_execution_placement(
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
            // ggml's multi-backend scheduler requires a CPU backend as its
            // final fallback participant. That is incompatible with the
            // FullDevice contract even when every currently known op happens
            // to offload: a later unsupported op could legally execute on the
            // CPU before the post-compute telemetry gate observes it. Use the
            // selected accelerator directly instead.
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
    fn full_device_disables_the_cross_backend_scheduler() {
        let config = apply_execution_placement(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            ExecutionPlacement::FullDevice,
        );

        assert_eq!(config.backend, GgmlCpuGraphBackend::Metal);
        assert!(!config.use_scheduler);
    }

    #[test]
    fn hybrid_enables_the_cross_backend_scheduler() {
        let config = apply_execution_placement(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                use_scheduler: false,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            ExecutionPlacement::Hybrid,
        );

        assert_eq!(config.backend, GgmlCpuGraphBackend::Metal);
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
