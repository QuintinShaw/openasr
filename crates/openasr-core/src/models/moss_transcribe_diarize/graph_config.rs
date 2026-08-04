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
use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
};

const MOSS_TD_ENCODER_GRAPH_SIZE: usize = 16_384;

pub(crate) fn moss_td_runtime_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: Some(true),
            default_n_threads_when_unset: Some(1),
        },
    )
}

pub(crate) fn moss_td_encoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    let mut config = moss_td_runtime_graph_config(backend);
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
}
