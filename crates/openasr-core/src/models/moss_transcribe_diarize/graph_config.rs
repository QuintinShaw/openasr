//! moss-transcribe-diarize ggml graph backend/threading policy.
//!
//! The family descriptor's `auto_gpu_policy` (see `arch/mod.rs`
//! `MOSS_TD_GGML_ARCHITECTURE_ID`) is the SSOT for Auto backend selection.
//! The shared dispatch resolves it once per request (via
//! `install_resolved_family_runtime_input`, using this descriptor's policy)
//! before this executor runs, so the graph config below reads that resolved
//! value directly instead of re-deriving or re-checking it here.
//!
//! Post-#212 quiet-window A/B: true accelerated Metal beats CPU on short and
//! 3-min clips, so the descriptor is `AllBackends` (Auto may pick Metal on
//! Apple Silicon).
use crate::ggml_runtime::GgmlCpuGraphConfig;
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
};

const MOSS_TD_ENCODER_GRAPH_SIZE: usize = 16_384;

pub(crate) fn moss_td_runtime_graph_config() -> GgmlCpuGraphConfig {
    let backend = crate::ggml_runtime::resolved_family_runtime_input().backend();
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: Some(true),
            default_n_threads_when_unset: Some(1),
        },
    )
}

pub(crate) fn moss_td_encoder_graph_config() -> GgmlCpuGraphConfig {
    let mut config = moss_td_runtime_graph_config();
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
        // encoder config must match what the shared dispatch resolved for
        // this family's policy, not silently force CPU (the old ExceptMetal
        // trap).
        let policy = family_auto_gpu_policy_for_model_architecture(MOSS_TD_GGML_ARCHITECTURE_ID);
        let _resolved = crate::ggml_runtime::install_resolved_family_runtime_input(policy);
        let backend = crate::ggml_runtime::resolved_family_runtime_input().backend();
        assert_eq!(moss_td_encoder_graph_config().backend, backend);
    }

    #[test]
    fn encoder_graph_config_honors_explicit_accelerated_request() {
        let policy = family_auto_gpu_policy_for_model_architecture(MOSS_TD_GGML_ARCHITECTURE_ID);
        let resolved_without_override = {
            let _resolved = crate::ggml_runtime::install_resolved_family_runtime_input(policy);
            crate::ggml_runtime::resolved_family_runtime_input().backend()
        };
        if !resolved_without_override.is_gpu_class() {
            return;
        }
        let _guard = crate::ggml_runtime::install_request_backend_override(Some(
            RequestBackendPreference::Accelerated,
        ));
        let _resolved = crate::ggml_runtime::install_resolved_family_runtime_input(policy);
        assert_eq!(
            moss_td_encoder_graph_config().backend,
            resolved_without_override
        );
    }

    #[test]
    fn encoder_graph_size_floor_is_preserved() {
        let policy = family_auto_gpu_policy_for_model_architecture(MOSS_TD_GGML_ARCHITECTURE_ID);
        let _resolved = crate::ggml_runtime::install_resolved_family_runtime_input(policy);
        assert!(moss_td_encoder_graph_config().graph_size >= MOSS_TD_ENCODER_GRAPH_SIZE);
    }
}
