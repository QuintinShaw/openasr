//! moss-transcribe-diarize ggml graph backend/threading policy.
//!
//! The family descriptor's `auto_gpu_policy` (see `arch/mod.rs`
//! `MOSS_TD_GGML_ARCHITECTURE_ID`) is the SSOT for Auto backend selection.
//! `configure_model_runtime_graph_config_from_env` alone resolves the *generic*
//! backend and does not consult per-family policy, so the encoder-graph builder
//! re-checks `resolve_family_runtime_backend` with the descriptor policy and
//! downgrades only when that gate disagrees. Explicit
//! `execution_target=accelerated` still wins via the request-level override.
//!
//! Post-#212 quiet-window A/B: true accelerated Metal beats CPU on short and
//! 3-min clips, so the descriptor is `AllBackends` (Auto may pick Metal on
//! Apple Silicon). Keep this file wired to the descriptor policy so a future
//! pin cannot silently drift from the encoder path.
use crate::arch::{MOSS_TD_GGML_ARCHITECTURE_ID, family_auto_gpu_policy_for_model_architecture};
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
};

const MOSS_TD_ENCODER_GRAPH_SIZE: usize = 16_384;

pub(crate) fn moss_td_runtime_graph_config() -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::default(),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: Some(true),
            default_n_threads_when_unset: Some(1),
        },
    )
}

/// Whether Auto should keep an already-resolved GPU-class backend for the
/// encoder, per this family's descriptor `auto_gpu_policy`.
fn encoder_gpu_enabled() -> bool {
    let policy = family_auto_gpu_policy_for_model_architecture(MOSS_TD_GGML_ARCHITECTURE_ID);
    GgmlCpuGraphConfig::resolve_family_runtime_backend(policy).is_gpu_class()
}

pub(crate) fn moss_td_encoder_graph_config() -> GgmlCpuGraphConfig {
    let mut config = moss_td_runtime_graph_config();
    config.graph_size = config.graph_size.max(MOSS_TD_ENCODER_GRAPH_SIZE);
    config.context_bytes = config
        .context_bytes
        .max(GgmlCpuGraphConfig::metadata_context_bytes(
            config.graph_size,
        ));
    // Generic resolver above does not know the family policy. Align when the
    // family-aware gate disagrees -- explicit accelerated still keeps GPU.
    if config.backend.is_gpu_class() && !encoder_gpu_enabled() {
        config.backend = GgmlCpuGraphBackend::Cpu;
        config.use_scheduler = false;
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::family_auto_gpu_policy_for_model_architecture;
    use crate::ggml_runtime::{AutoGpuPolicy, RequestBackendPreference};

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
        // encoder config must match the family-aware resolver, not silently
        // force CPU (the old ExceptMetal trap).
        let family = GgmlCpuGraphConfig::resolve_family_runtime_backend(
            family_auto_gpu_policy_for_model_architecture(MOSS_TD_GGML_ARCHITECTURE_ID),
        );
        assert_eq!(moss_td_encoder_graph_config().backend, family);
    }

    #[test]
    fn encoder_graph_config_honors_explicit_accelerated_request() {
        let resolved_without_override = GgmlCpuGraphConfig::resolve_runtime_backend();
        if !resolved_without_override.is_gpu_class() {
            return;
        }
        let _guard = crate::ggml_runtime::install_request_backend_override(Some(
            RequestBackendPreference::Accelerated,
        ));
        assert_eq!(
            moss_td_encoder_graph_config().backend,
            resolved_without_override
        );
    }

    #[test]
    fn encoder_graph_size_floor_is_preserved() {
        assert!(moss_td_encoder_graph_config().graph_size >= MOSS_TD_ENCODER_GRAPH_SIZE);
    }
}
