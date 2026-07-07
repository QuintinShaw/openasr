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
mod model;
mod provider;
mod streaming;
mod weights;

#[cfg(test)]
mod tests;

use std::sync::OnceLock;

pub use model::FireRedStreamVadModel;
pub use provider::{FireRedStreamVadError, FireRedStreamVadProvider};
pub use streaming::FireRedStreamingVad;

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
