use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::ggml_runtime::GgmlCpuGraphConfig;
use crate::ggml_runtime::GgmlCpuGraphThreadingWorkload;
#[cfg(test)]
use crate::models::graph_runtime_config::configure_model_runtime_graph_config;
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    has_explicit_thread_override,
};

const OPENASR_WHISPER_ENABLE_ENCODER_PRELUDE_GPU: &str =
    "OPENASR_WHISPER_ENABLE_ENCODER_PRELUDE_GPU";
const OPENASR_WHISPER_ENABLE_ENCODER_PRELUDE_METAL: &str =
    "OPENASR_WHISPER_GGML_ENABLE_ENCODER_PRELUDE_METAL";
const OPENASR_WHISPER_ENABLE_DECODER_GPU: &str = "OPENASR_WHISPER_ENABLE_DECODER_GPU";
const OPENASR_WHISPER_USE_DECODER_SCHEDULER: &str = "OPENASR_WHISPER_GGML_USE_DECODER_SCHEDULER";

/// Typed request-route snapshot that selects the CUDA/Vulkan Exact-mode default.
///
/// The constructor reads only the already-resolved request route. This avoids
/// model IDs and device-name parsing in family code while keeping the policy
/// identical across every published Whisper geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WhisperDecoderPlacementPolicy {
    exact_cuda_or_vulkan: bool,
}

impl WhisperDecoderPlacementPolicy {
    pub(crate) fn resolve() -> Self {
        let exact_cuda_or_vulkan = matches!(
            crate::ggml_runtime::request_backend_override(),
            Some(crate::ggml_runtime::RequestBackendPreference::Exact(route))
                if matches!(
                    route.provider,
                    crate::device::execution_route::ExecutionProvider::Cuda
                        | crate::device::execution_route::ExecutionProvider::Vulkan
                )
        );
        Self {
            exact_cuda_or_vulkan,
        }
    }

    #[cfg(test)]
    fn for_test(exact_cuda_or_vulkan: bool) -> Self {
        Self {
            exact_cuda_or_vulkan,
        }
    }

    fn uses_exact_cuda_or_vulkan_default(self, backend: GgmlCpuGraphBackend) -> bool {
        self.exact_cuda_or_vulkan && matches!(backend, GgmlCpuGraphBackend::Gpu)
    }
}

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
        whisper_encoder_prelude_gpu_enabled,
        has_explicit_thread_override(),
    )
}

pub(crate) fn whisper_decoder_graph_config(
    backend: GgmlCpuGraphBackend,
    placement_policy: WhisperDecoderPlacementPolicy,
) -> GgmlCpuGraphConfig {
    let decoder_gpu_raw = std::env::var(OPENASR_WHISPER_ENABLE_DECODER_GPU).ok();
    let decoder_scheduler_raw = std::env::var(OPENASR_WHISPER_USE_DECODER_SCHEDULER).ok();
    whisper_decoder_graph_config_with_overrides(
        whisper_runtime_graph_config(backend),
        placement_policy,
        decoder_gpu_raw.as_deref(),
        decoder_scheduler_raw.as_deref(),
        has_explicit_thread_override(),
    )
}

fn whisper_decoder_gpu_enabled(
    backend: GgmlCpuGraphBackend,
    placement_policy: WhisperDecoderPlacementPolicy,
    decoder_gpu_raw: Option<&str>,
) -> bool {
    if placement_policy.uses_exact_cuda_or_vulkan_default(backend) {
        // Current CUDA/Vulkan measurements favor the direct reusable decoder
        // across published Whisper geometries. The stage knob remains an
        // explicit diagnostic opt-out; production FullDevice attestation will
        // reject that CPU override instead of silently weakening placement.
        return crate::ggml_runtime::env_toggle_with_raw(None, decoder_gpu_raw, true);
    }

    // CPU, Metal, HIP, and all non-Exact requests retain their existing
    // resolution behavior exactly.
    backend == GgmlCpuGraphBackend::Cpu
        || crate::ggml_runtime::env_toggle_with_raw(None, decoder_gpu_raw, true)
}

fn whisper_decoder_graph_config_with_overrides(
    mut config: GgmlCpuGraphConfig,
    placement_policy: WhisperDecoderPlacementPolicy,
    decoder_gpu_raw: Option<&str>,
    decoder_scheduler_raw: Option<&str>,
    has_explicit_thread_override: bool,
) -> GgmlCpuGraphConfig {
    if config.backend.is_gpu_class()
        && !whisper_decoder_gpu_enabled(config.backend, placement_policy, decoder_gpu_raw)
    {
        // A production FullDevice candidate rejects this CPU graph at the
        // runner boundary. This remains useful for direct benchmarking and
        // for future descriptors that explicitly declare a split topology;
        // it cannot silently weaken the current production candidate.
        config.backend = GgmlCpuGraphBackend::Cpu;
        config.use_scheduler = false;
    } else if config.backend.is_gpu_class() {
        // CUDA/Vulkan Exact decoder execution defaults to the reusable direct
        // graph. The scheduler knob is considered only after final placement
        // remains GPU, so an explicit scheduler value can never revive a
        // scheduler after the diagnostic CPU opt-out.
        let scheduler_default =
            if placement_policy.uses_exact_cuda_or_vulkan_default(config.backend) {
                false
            } else {
                config.use_scheduler
            };
        config.use_scheduler = crate::ggml_runtime::env_toggle_with_raw(
            None,
            decoder_scheduler_raw,
            scheduler_default,
        );
    }
    if !has_explicit_thread_override {
        config.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            config.backend,
            GgmlCpuGraphThreadingWorkload::Decoder,
        );
    }
    config
}

fn whisper_encoder_prelude_gpu_enabled(backend: GgmlCpuGraphBackend) -> bool {
    if backend == GgmlCpuGraphBackend::Cpu {
        return true;
    }
    let gpu_raw = std::env::var(OPENASR_WHISPER_ENABLE_ENCODER_PRELUDE_GPU).ok();
    if backend == GgmlCpuGraphBackend::Metal {
        let legacy = std::env::var(OPENASR_WHISPER_ENABLE_ENCODER_PRELUDE_METAL).ok();
        if legacy.is_some() {
            return crate::ggml_runtime::env_toggle_with_raw(None, legacy.as_deref(), false);
        }
    }
    crate::ggml_runtime::env_toggle_with_raw(None, gpu_raw.as_deref(), true)
}

fn whisper_encoder_prelude_graph_config_with_overrides(
    mut base: GgmlCpuGraphConfig,
    prelude_gpu_enabled: impl FnOnce(GgmlCpuGraphBackend) -> bool,
    has_explicit_thread_override: bool,
) -> GgmlCpuGraphConfig {
    if base.backend.is_gpu_class() && !prelude_gpu_enabled(base.backend) {
        // As with decoder tuning, a production FullDevice candidate rejects
        // this direct-diagnostic CPU graph rather than silently relabeling it.
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

    #[test]
    fn decoder_can_disable_scheduler_without_changing_gpu_backend() {
        let config = whisper_decoder_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Gpu,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            WhisperDecoderPlacementPolicy::for_test(true),
            None,
            Some("0"),
            false,
        );

        assert_eq!(config.backend, GgmlCpuGraphBackend::Gpu);
        assert!(!config.use_scheduler);
    }

    #[test]
    fn decoder_explicit_opt_out_can_move_to_cpu_without_changing_other_stages() {
        let config = whisper_decoder_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Gpu,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            WhisperDecoderPlacementPolicy::for_test(true),
            Some("0"),
            None,
            false,
        );

        assert_eq!(config.backend, GgmlCpuGraphBackend::Cpu);
        assert!(!config.use_scheduler);
    }

    #[test]
    fn decoder_preserves_resolved_scheduler_for_non_exact_requests_when_stage_knob_is_unset() {
        for use_scheduler in [false, true] {
            let config = whisper_decoder_graph_config_with_overrides(
                GgmlCpuGraphConfig {
                    backend: GgmlCpuGraphBackend::Gpu,
                    use_scheduler,
                    ..GgmlCpuGraphConfig::conservative_default()
                },
                WhisperDecoderPlacementPolicy::for_test(false),
                None,
                None,
                false,
            );

            assert_eq!(config.backend, GgmlCpuGraphBackend::Gpu);
            assert_eq!(config.use_scheduler, use_scheduler);
        }
    }

    #[test]
    fn exact_cuda_vulkan_direct_gpu_default_starts_without_scheduler() {
        let config = whisper_decoder_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Gpu,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            WhisperDecoderPlacementPolicy::for_test(true),
            None,
            None,
            false,
        );

        assert_eq!(config.backend, GgmlCpuGraphBackend::Gpu);
        assert!(!config.use_scheduler);
    }

    #[test]
    fn exact_cuda_vulkan_env_decoder_enable_keeps_gpu_and_can_enable_scheduler() {
        let config = whisper_decoder_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Gpu,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            WhisperDecoderPlacementPolicy::for_test(true),
            Some("1"),
            Some("1"),
            false,
        );

        assert_eq!(config.backend, GgmlCpuGraphBackend::Gpu);
        assert!(config.use_scheduler);
    }

    #[test]
    fn exact_cuda_vulkan_env_decoder_disable_overrides_direct_gpu_default() {
        let config = whisper_decoder_graph_config_with_overrides(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Gpu,
                use_scheduler: false,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            WhisperDecoderPlacementPolicy::for_test(true),
            Some("0"),
            Some("1"),
            false,
        );

        assert_eq!(config.backend, GgmlCpuGraphBackend::Cpu);
        assert!(
            !config.use_scheduler,
            "scheduler env must not apply after an explicit CPU decoder override"
        );
    }

    #[test]
    fn exact_cuda_vulkan_policy_never_applies_to_cpu_or_metal_graphs() {
        let policy = WhisperDecoderPlacementPolicy::for_test(true);
        assert!(!policy.uses_exact_cuda_or_vulkan_default(GgmlCpuGraphBackend::Cpu));
        assert!(!policy.uses_exact_cuda_or_vulkan_default(GgmlCpuGraphBackend::Metal));
    }

    #[test]
    fn placement_policy_uses_only_typed_exact_cuda_or_vulkan_routes() {
        let policy_for = |provider| {
            let _guard = crate::ggml_runtime::install_request_backend_override(Some(
                crate::ggml_runtime::RequestBackendPreference::Exact(
                    crate::device::execution_route::ResolvedExecutionRoute {
                        provider,
                        stable_id: "test-device".to_string(),
                        registry_ordinal: 0,
                        kind: crate::device::execution_route::RouteDeviceKind::Accelerated,
                        addressability:
                            crate::device::execution_route::DeviceAddressability::NotExactlyAddressable {
                                reason: "test fixture",
                            },
                    },
                ),
            ));
            WhisperDecoderPlacementPolicy::resolve()
        };

        for provider in [
            crate::device::execution_route::ExecutionProvider::Cuda,
            crate::device::execution_route::ExecutionProvider::Vulkan,
        ] {
            assert!(
                policy_for(provider).uses_exact_cuda_or_vulkan_default(GgmlCpuGraphBackend::Gpu)
            );
        }
        for provider in [
            crate::device::execution_route::ExecutionProvider::Hip,
            crate::device::execution_route::ExecutionProvider::Metal,
            crate::device::execution_route::ExecutionProvider::Cpu,
        ] {
            assert!(
                !policy_for(provider).uses_exact_cuda_or_vulkan_default(GgmlCpuGraphBackend::Gpu)
            );
        }
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

        let decoder_config = whisper_decoder_graph_config(
            GgmlCpuGraphBackend::Metal,
            WhisperDecoderPlacementPolicy::for_test(true),
        );
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
