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

/// Validate a diarization runtime pack by constructing its model from the pack
/// exactly the way the runtime loaders do — every required tensor must be
/// present with the right shape. Returns `None` when `metadata` does not
/// identify a diarization pack (the caller falls through to ASR runtime
/// validation).
pub fn validate_diarize_runtime_pack_contract(
    path: &std::path::Path,
    metadata: &crate::GgufMetadata,
) -> Option<Result<(), String>> {
    let architecture = metadata.get_string("general.architecture")?;
    match architecture.trim() {
        crate::models::wespeaker::WESPEAKER_GGML_ARCHITECTURE_ID => Some(
            embed::WeSpeakerEmbedder::from_oasr(path)
                .map(|_| ())
                .map_err(|error| error.to_string()),
        ),
        crate::models::pyannote::PYANNOTE_GGML_ARCHITECTURE_ID => Some(
            segment::PyannoteSegmenter::from_oasr(path)
                .map(|_| ())
                .map_err(|error| error.to_string()),
        ),
        _ => None,
    }
}

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
