//! FireRedVAD **Stream-VAD** (`FireRedTeam/FireRedVAD`, Apache-2.0,
//! `Stream-VAD/model.pth.tar`): the causal (`N2 = 0`, no lookahead) sibling
//! of [`crate::diarize::vad::firered`]'s non-streaming `VAD/model.pth.tar`.
//! Vendored the same way (a ~2.3 MB `f32` safetensors blob baked in via
//! `include_bytes!`, no ggml/.oasr/catalog involvement).
//!
//! Because it is strictly causal, Stream-VAD is the only FireRedVAD
//! checkpoint suitable for realtime endpointing: it wires into
//! [`crate::realtime`]'s `VadMode::ExternalProbability` path as a selectable
//! neural sub-engine alongside Silero (`OPENASR_VAD=firered-stream`), and
//! also drops into the [`crate::longform::LongFormVadProvider`] seam
//! (`OPENASR_VAD=firered-stream` for long-form too) so the same checkpoint
//! can be benchmarked as a long-form engine against the non-streaming
//! `VAD/model.pth.tar` -- the open question this module exists to let
//! callers measure, not to answer by picking a new default. **Silero stays
//! the default for both paths**; this is opt-in only.

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
/// if the vendored weights fail to parse, so callers fall back to
/// Silero/energy.
pub fn shared_model() -> Option<&'static FireRedStreamVadModel> {
    SHARED_MODEL
        .get_or_init(|| FireRedStreamVadModel::embedded().ok())
        .as_ref()
}
