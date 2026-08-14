//! FireRedVAD **Stream-VAD** (`FireRedTeam/FireRedVAD`, Apache-2.0,
//! `Stream-VAD/model.pth.tar`): a causal (`N2 = 0`, no lookahead) DFSMN
//! voice-activity detector. Vendored as a ~2.3 MB `f32` safetensors blob
//! baked in via `include_bytes!` (no ggml/.oasr/catalog involvement), so it
//! is always available.
//!
//! This is the **sole VAD engine** in OpenASR: because it is strictly
//! causal, the same checkpoint backs both realtime endpointing
//! ([`crate::realtime`]'s `VadMode::ExternalProbability` path, via
//! [`FireRedStreamingVad`]) and long-form speech slicing (the
//! [`crate::longform::LongFormVadProvider`] seam, via
//! [`FireRedStreamVadProvider`]) and diarization's speech-region resolution.
//! There is no other neural engine and no runtime engine-selection
//! mechanism to opt out of it.

mod frontend;
mod ggml_runtime;
mod model;
mod provider;
mod realtime_runtime;
mod streaming;
mod weights;

#[cfg(test)]
mod tests;

use std::sync::OnceLock;

pub use model::FireRedStreamVadModel;
pub(crate) use provider::PolicyResolvedFireRedStreamVadProvider;
pub use provider::{FireRedStreamVadError, FireRedStreamVadProvider};
pub(crate) use realtime_runtime::{FireRedRealtimeVadRuntime, FireRedRealtimeVadSession};
pub use streaming::FireRedStreamingVad;

pub(crate) fn execution_capabilities() -> crate::device::execution_policy::ExecutionCapabilities {
    use crate::device::{
        execution_policy::{AcceleratedPlacementCapabilities, ExecutionCapabilities},
        execution_route::ExecutionProvider,
    };

    // Feature extraction remains host-side preprocessing on every backend;
    // the complete neural DFSMN graph executes on the selected device.
    ExecutionCapabilities::new(true)
        .with_provider(
            ExecutionProvider::Metal,
            AcceleratedPlacementCapabilities::FULL_DEVICE,
        )
        .with_provider(
            ExecutionProvider::Cuda,
            AcceleratedPlacementCapabilities::FULL_DEVICE,
        )
        .with_provider(
            ExecutionProvider::Vulkan,
            AcceleratedPlacementCapabilities::FULL_DEVICE,
        )
}

/// Realtime sessions keep automatic selection on CPU. Explicit accelerated
/// requests still use the unified stateful runtime and its replay contract.
pub(crate) const AUTO_GPU_POLICY: crate::ggml_runtime::AutoGpuPolicy =
    crate::ggml_runtime::AutoGpuPolicy::Never;

/// Offline slicing uses CUDA/Vulkan automatically while Metal remains an
/// explicit opt-in until its product-level latency evidence is promoted.
pub(crate) const OFFLINE_AUTO_GPU_POLICY: crate::ggml_runtime::AutoGpuPolicy =
    crate::ggml_runtime::AutoGpuPolicy::ExceptMetal;

static SHARED_MODEL: OnceLock<Option<FireRedStreamVadModel>> = OnceLock::new();

/// The process-wide Stream-VAD model, loaded once (~2.3 MB). Returns `None`
/// only if the vendored weights blob fails to parse (a build-integrity
/// problem, since the blob is a fixed, committed asset); callers should treat
/// that as an unexpected fail-closed condition, not a routine fallback.
pub fn shared_model() -> Option<&'static FireRedStreamVadModel> {
    SHARED_MODEL
        .get_or_init(|| FireRedStreamVadModel::embedded().ok())
        .as_ref()
}
