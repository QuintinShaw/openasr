//! Speech and speaker analysis stages.
//!
//! VAD and speaker diarization live here as model-agnostic pre/post stages that
//! sit around any ASR executor and communicate with the rest of the engine only
//! through time-interval vocabulary. The ASR core (`GgmlAsrExecutor`) is never
//! touched, which keeps diarization additive across every model family.
//!
//! The stages are the neural VAD (FireRedVAD Stream-VAD, pure Rust) plus the
//! diarization pipeline (interval contract, speaker segmentation/embedding,
//! clustering, and attribution) under this module.

/// Whether the model-agnostic VAD + speaker-embedder diarization path can run:
/// the active speaker-embedder pack is installed (the Stream-VAD VAD is
/// vendored and always available). This is a presence-only probe for
/// capability reporting; a pack that fails to load still fails closed at
/// request time.
pub fn vad_diarization_available() -> bool {
    embed::embedder_pack_installed()
}

// Pull-time contract validation for diarization support packs (WeSpeaker
// speaker embedder, pyannote speaker segmenter) is dispatched through
// `crate::models::aux_pack_registry`, alongside the other auxiliary (non-ASR)
// families (translation, punctuation) -- one table instead of a per-family
// function called from an ad hoc chain in `api::backend::native`.

pub mod attribution;
#[doc(hidden)]
pub mod calibration;
pub mod clustering;
pub mod contract;
pub(crate) mod debug;
pub mod embed;
pub mod enrollment;
mod pack;
pub mod pipeline;
pub mod segment;
pub mod streaming;
pub mod vad;
pub(crate) mod vbx;
