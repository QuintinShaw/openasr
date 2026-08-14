//! moss-transcribe-diarize ggml graph backend/threading policy.
//!
//! The family descriptor's `auto_gpu_policy` (see `arch/mod.rs`
//! `MOSS_TD_GGML_ARCHITECTURE_ID`) is the SSOT for Auto backend selection.
//! Whoever builds the request resolves it once (via
//! `ResolvedFamilyRuntimeInput::resolve`, using this descriptor's policy),
//! and the executor passes that value in here explicitly -- the graph config
//! below never re-derives or re-checks it.
//!
//! Post-#212 quiet-window A/B: true accelerated Metal beats CPU on short and
//! 3-min clips, so the descriptor is `AllBackends` (Auto may pick Metal on
//! Apple Silicon).
use crate::device::execution_policy::ExecutionPlacement;
use crate::device::execution_route::ExecutionProvider;
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig};
use crate::ggml_runtime::{RequestBackendPreference, request_backend_override};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
};
use crate::models::qwen::qwen_decoder_graph_config;

const MOSS_TD_ENCODER_GRAPH_SIZE: usize = 16_384;
const MOSS_TD_ENABLE_DECODER_GPU: &str = "OPENASR_MOSS_TD_ENABLE_DECODER_GPU";

fn moss_td_base_runtime_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: Some(true),
            default_n_threads_when_unset: Some(1),
        },
    )
}

fn current_exact_provider() -> Option<ExecutionProvider> {
    match request_backend_override() {
        Some(RequestBackendPreference::Exact(route)) => Some(route.provider),
        _ => None,
    }
}

fn scheduler_override_from_env() -> Option<bool> {
    std::env::var_os(GgmlCpuGraphConfig::USE_SCHEDULER_ENV)
        .is_some()
        .then(GgmlCpuGraphConfig::resolve_runtime_scheduler_usage)
}

pub(crate) fn moss_td_runtime_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    // Preserve the shared Qwen decoder's threading and operator defaults, then
    // freeze only MOSS's family-owned Hybrid stage placement below.
    let config = qwen_decoder_graph_config(backend);
    apply_moss_td_hybrid_decoder_policy(
        config,
        crate::models::native_execution_services::current_execution_placement(),
        current_exact_provider(),
        std::env::var(MOSS_TD_ENABLE_DECODER_GPU).ok().as_deref(),
        scheduler_override_from_env(),
    )
}

/// MOSS remains a Hybrid pipeline because its adaptor and prompt assembly are
/// host stages. The ggml encoder/decoder/logits stages themselves are complete
/// CUDA/Vulkan graphs, though, and running those stages through the hybrid
/// scheduler disables resident KV and reusable decoding. Keep the request's
/// Hybrid placement while selecting direct GPU execution for those graph
/// stages on the two providers qualified by the Windows true-pack matrix.
/// Metal and HIP retain their existing scheduler policy.
fn apply_moss_td_hybrid_encoder_policy(
    mut config: GgmlCpuGraphConfig,
    placement: Option<ExecutionPlacement>,
    provider: Option<ExecutionProvider>,
    scheduler_override: Option<bool>,
) -> GgmlCpuGraphConfig {
    if config.backend.is_gpu_class()
        && placement == Some(ExecutionPlacement::Hybrid)
        && matches!(
            provider,
            Some(ExecutionProvider::Cuda | ExecutionProvider::Vulkan)
        )
    {
        config.use_scheduler = scheduler_override.unwrap_or(false);
    }
    config
}

/// The recurrent Qwen decoder has a different Pareto point from the encoder on
/// the qualified RTX 3060 Vulkan route. Keep the encoder on Vulkan, but default
/// decoder/logits graphs to direct CPU; the request remains Hybrid and retains
/// its exact Vulkan encoder observation. CUDA keeps the fully direct GPU path.
/// The MOSS-specific env knob is an operator diagnostic override and is read
/// only while constructing the request-scoped decoder runtime.
fn apply_moss_td_hybrid_decoder_policy(
    mut config: GgmlCpuGraphConfig,
    placement: Option<ExecutionPlacement>,
    provider: Option<ExecutionProvider>,
    decoder_gpu_raw: Option<&str>,
    scheduler_override: Option<bool>,
) -> GgmlCpuGraphConfig {
    if !config.backend.is_gpu_class() || placement != Some(ExecutionPlacement::Hybrid) {
        return config;
    }
    let default_gpu = match provider {
        Some(ExecutionProvider::Cuda) => true,
        Some(ExecutionProvider::Vulkan) => false,
        _ => return config,
    };
    if !crate::ggml_runtime::env_toggle_with_raw(None, decoder_gpu_raw, default_gpu) {
        config.backend = GgmlCpuGraphBackend::Cpu;
        config.use_scheduler = false;
        return config;
    }
    config.use_scheduler = scheduler_override.unwrap_or(false);
    config
}

pub(crate) fn moss_td_encoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    let mut config = apply_moss_td_hybrid_encoder_policy(
        moss_td_base_runtime_graph_config(backend),
        crate::models::native_execution_services::current_execution_placement(),
        current_exact_provider(),
        scheduler_override_from_env(),
    );
    config.graph_size = config.graph_size.max(MOSS_TD_ENCODER_GRAPH_SIZE);
    config.context_bytes = config
        .context_bytes
        .max(GgmlCpuGraphConfig::metadata_context_bytes(
            config.graph_size,
        ));
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{
        MOSS_TD_GGML_ARCHITECTURE_ID, family_auto_gpu_policy_for_model_architecture,
    };
    use crate::ggml_runtime::{
        AutoGpuPolicy, RequestBackendPreference, ResolvedFamilyRuntimeInput,
    };

    #[test]
    fn family_auto_policy_is_all_backends() {
        assert_eq!(
            family_auto_gpu_policy_for_model_architecture(MOSS_TD_GGML_ARCHITECTURE_ID),
            AutoGpuPolicy::AllBackends
        );
    }

    #[test]
    fn encoder_graph_config_follows_family_auto_policy_under_auto() {
        // With AllBackends, Auto may resolve to Metal on Apple Silicon. The
        // encoder config must match the descriptor's resolved backend and must
        // not reintroduce a CPU-only gate.
        let policy = family_auto_gpu_policy_for_model_architecture(MOSS_TD_GGML_ARCHITECTURE_ID);
        let backend = ResolvedFamilyRuntimeInput::resolve(None, policy).backend();
        assert_eq!(moss_td_encoder_graph_config(backend).backend, backend);
    }

    #[test]
    fn encoder_graph_config_honors_explicit_accelerated_request() {
        let policy = family_auto_gpu_policy_for_model_architecture(MOSS_TD_GGML_ARCHITECTURE_ID);
        let resolved_without_override = ResolvedFamilyRuntimeInput::resolve(None, policy).backend();
        if !resolved_without_override.is_gpu_class() {
            return;
        }
        let accelerated = ResolvedFamilyRuntimeInput::resolve(
            Some(RequestBackendPreference::Accelerated),
            policy,
        )
        .backend();
        assert_eq!(
            moss_td_encoder_graph_config(accelerated).backend,
            resolved_without_override
        );
    }

    #[test]
    fn encoder_graph_size_floor_is_preserved() {
        let policy = family_auto_gpu_policy_for_model_architecture(MOSS_TD_GGML_ARCHITECTURE_ID);
        let backend = ResolvedFamilyRuntimeInput::resolve(None, policy).backend();
        assert!(moss_td_encoder_graph_config(backend).graph_size >= MOSS_TD_ENCODER_GRAPH_SIZE);
    }

    fn scheduled_gpu_config() -> GgmlCpuGraphConfig {
        GgmlCpuGraphConfig {
            backend: GgmlCpuGraphBackend::Gpu,
            use_scheduler: true,
            ..GgmlCpuGraphConfig::conservative_default()
        }
    }

    #[test]
    fn hybrid_cuda_and_vulkan_encoder_stages_default_to_direct_gpu() {
        for provider in [ExecutionProvider::Cuda, ExecutionProvider::Vulkan] {
            let config = apply_moss_td_hybrid_encoder_policy(
                scheduled_gpu_config(),
                Some(ExecutionPlacement::Hybrid),
                Some(provider),
                None,
            );
            assert_eq!(config.backend, GgmlCpuGraphBackend::Gpu);
            assert!(!config.use_scheduler);
        }
    }

    #[test]
    fn explicit_scheduler_override_wins_for_qualified_hybrid_routes() {
        for requested in [false, true] {
            let config = apply_moss_td_hybrid_encoder_policy(
                scheduled_gpu_config(),
                Some(ExecutionPlacement::Hybrid),
                Some(ExecutionProvider::Vulkan),
                Some(requested),
            );
            assert_eq!(config.use_scheduler, requested);
        }
    }

    #[test]
    fn metal_hip_and_non_hybrid_routes_keep_the_resolved_scheduler_policy() {
        for provider in [ExecutionProvider::Metal, ExecutionProvider::Hip] {
            let config = apply_moss_td_hybrid_encoder_policy(
                scheduled_gpu_config(),
                Some(ExecutionPlacement::Hybrid),
                Some(provider),
                None,
            );
            assert!(config.use_scheduler);
        }
        for placement in [ExecutionPlacement::CpuOnly, ExecutionPlacement::FullDevice] {
            let config = apply_moss_td_hybrid_encoder_policy(
                scheduled_gpu_config(),
                Some(placement),
                Some(ExecutionProvider::Vulkan),
                None,
            );
            assert!(config.use_scheduler);
        }
    }

    #[test]
    fn hybrid_decoder_defaults_to_cuda_gpu_and_vulkan_cpu() {
        let cuda = apply_moss_td_hybrid_decoder_policy(
            scheduled_gpu_config(),
            Some(ExecutionPlacement::Hybrid),
            Some(ExecutionProvider::Cuda),
            None,
            None,
        );
        assert_eq!(cuda.backend, GgmlCpuGraphBackend::Gpu);
        assert!(!cuda.use_scheduler);

        let vulkan = apply_moss_td_hybrid_decoder_policy(
            scheduled_gpu_config(),
            Some(ExecutionPlacement::Hybrid),
            Some(ExecutionProvider::Vulkan),
            None,
            None,
        );
        assert_eq!(vulkan.backend, GgmlCpuGraphBackend::Cpu);
        assert!(!vulkan.use_scheduler);
    }

    #[test]
    fn decoder_gpu_and_scheduler_diagnostics_override_provider_defaults() {
        let vulkan_gpu = apply_moss_td_hybrid_decoder_policy(
            scheduled_gpu_config(),
            Some(ExecutionPlacement::Hybrid),
            Some(ExecutionProvider::Vulkan),
            Some("1"),
            Some(true),
        );
        assert_eq!(vulkan_gpu.backend, GgmlCpuGraphBackend::Gpu);
        assert!(vulkan_gpu.use_scheduler);

        let cuda_cpu = apply_moss_td_hybrid_decoder_policy(
            scheduled_gpu_config(),
            Some(ExecutionPlacement::Hybrid),
            Some(ExecutionProvider::Cuda),
            Some("0"),
            Some(true),
        );
        assert_eq!(cuda_cpu.backend, GgmlCpuGraphBackend::Cpu);
        assert!(!cuda_cpu.use_scheduler);
    }
}
