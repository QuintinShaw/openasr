//! FireRedVAD (`FireRedTeam/FireRedVAD`, Apache-2.0): a pure-Rust forward
//! pass of the causal-FSMN `DetectModel`, dropping into the same
//! [`crate::longform::LongFormVadProvider`] seam as
//! [`super::SileroVadProvider`] so it can be selected as an alternative
//! long-form VAD engine (`OPENASR_VAD=firered`). Not wired into realtime
//! endpointing or diarization -- those stay on Silero/energy.
//!
//! Vendored the same way as Silero: the checkpoint is ~0.6 M parameters
//! (~2.3 MB as `f32` safetensors), so it is baked into the binary via
//! `include_bytes!` rather than pulled as a user-facing model pack.

// `pub(super)`: the Stream-VAD sibling engine (`super::firered_stream`)
// reuses this exact kaldi-fbank + global-CMVN frontend -- the two checkpoints
// share the same frontend/CMVN stats (verified byte-identical against the
// upstream `Stream-VAD/cmvn.ark`), so it stays the single source of truth
// rather than growing a second copy.
pub(super) mod frontend;
mod model;
mod provider;
mod weights;

#[cfg(test)]
mod tests;

use std::sync::OnceLock;

pub use model::FireRedVadModel;
pub use provider::{FireRedVadError, FireRedVadProvider};

static SHARED_MODEL: OnceLock<Option<FireRedVadModel>> = OnceLock::new();

/// The process-wide FireRedVAD model, loaded once (~2.3 MB). Returns `None`
/// if the vendored weights fail to parse, so callers fall back to
/// Silero/energy.
pub fn shared_model() -> Option<&'static FireRedVadModel> {
    SHARED_MODEL
        .get_or_init(|| FireRedVadModel::embedded().ok())
        .as_ref()
}
