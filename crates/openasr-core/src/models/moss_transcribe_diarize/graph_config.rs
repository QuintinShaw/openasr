//! moss-transcribe-diarize ggml graph backend/threading policy.
//!
//! `arch/mod.rs`'s `MOSS_TD_GGML_ARCHITECTURE_ID` descriptor declares
//! `auto_gpu_policy: AutoGpuPolicy::ExceptMetal` (Auto should steer away from
//! Metal for this family: the Whisper-Medium-style encoder's deep layers were
//! measured to decorrelate on Metal). That gate is only real if something
//! downstream actually consults it -- `configure_model_runtime_graph_config_from_env`
//! alone does NOT: it resolves the *generic* `GgmlCpuGraphConfig::default()`
//! backend (`resolve_runtime_backend()`), which knows nothing about any
//! per-family policy and picks Metal on Apple Silicon whenever a GPU device
//! is present, `ExceptMetal` or not. Mirrors `xasr_zipformer::graph_config`'s
//! `encoder_gpu_enabled` / `firered_aed::graph_config`'s
//! `firered_encoder_gpu_enabled` pattern: the family's own encoder-graph
//! builder must explicitly re-check `resolve_family_runtime_backend` and
//! downgrade to CPU when the family-aware gate disagrees with what the
//! generic resolver picked. An explicit `execution_target=accelerated`
//! request still installs a `RequestBackendPreference::Accelerated`
//! thread-local override, which `resolve_family_runtime_backend` always
//! honors over the gate (see its own doc comment) -- so this only ever
//! changes what Auto picks, never overrides an explicit request.
use crate::ggml_runtime::{AutoGpuPolicy, GgmlCpuGraphBackend, GgmlCpuGraphConfig};
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

/// Whether Auto should actually keep an already-resolved GPU-class backend
/// for the encoder, per this family's `AutoGpuPolicy::ExceptMetal` gate. Pulled
/// into its own function (mirrors `xasr_zipformer::graph_config::encoder_gpu_enabled`)
/// so any future provenance/label code resolving through the same gate can
/// never drift from what the encoder graph config actually decided.
fn encoder_gpu_enabled() -> bool {
    GgmlCpuGraphConfig::resolve_family_runtime_backend(AutoGpuPolicy::ExceptMetal).is_gpu_class()
}

pub(crate) fn moss_td_encoder_graph_config() -> GgmlCpuGraphConfig {
    let mut config = moss_td_runtime_graph_config();
    config.graph_size = config.graph_size.max(MOSS_TD_ENCODER_GRAPH_SIZE);
    config.context_bytes = config
        .context_bytes
        .max(GgmlCpuGraphConfig::metadata_context_bytes(
            config.graph_size,
        ));
    // `moss_td_runtime_graph_config` resolved the *generic* backend above,
    // which does not know about this family's `ExceptMetal` gate. Downgrade
    // here if the family-aware gate disagrees -- an explicit accelerated
    // request still keeps Metal (`resolve_family_runtime_backend` always
    // honors an explicit `RequestBackendPreference` override).
    if config.backend.is_gpu_class() && !encoder_gpu_enabled() {
        config.backend = GgmlCpuGraphBackend::Cpu;
        config.use_scheduler = false;
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::RequestBackendPreference;

    #[test]
    fn encoder_graph_config_never_picks_metal_under_auto() {
        // Auto (no explicit request-level override installed) must never
        // resolve to Metal for the encoder, regardless of what's available on
        // the host running this test -- mirrors
        // `xasr_zipformer::graph_config`'s equivalent guard. This is the
        // regression test for the "ExceptMetal declared but unwired" gap: it
        // fails if `moss_td_encoder_graph_config` ever regresses back to the
        // bare `configure_model_runtime_graph_config_from_env` call with no
        // family-gate check.
        assert_ne!(
            moss_td_encoder_graph_config().backend,
            GgmlCpuGraphBackend::Metal
        );
    }

    #[test]
    fn encoder_graph_config_honors_explicit_accelerated_request() {
        // An explicit accelerated request always wins over the Auto gate --
        // the gate can only ever pin Auto, never override an explicit
        // per-request choice (same contract `resolve_family_runtime_backend`
        // documents and tests directly).
        let resolved_without_override = GgmlCpuGraphConfig::resolve_runtime_backend();
        if !resolved_without_override.is_gpu_class() {
            // No GPU-class backend available on this host at all (e.g. a
            // Linux CI runner with no CUDA/HIP/Vulkan device): an explicit
            // accelerated request has nothing to resolve to, so this branch
            // of the contract is untestable here. Every other builtin
            // family's equivalent test skips the same way (see
            // `resolve_family_runtime_backend_except_metal_gates_only_metal`).
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
