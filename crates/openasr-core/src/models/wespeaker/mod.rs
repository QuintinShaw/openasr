//! WeSpeaker ResNet34 speaker-embedder `.oasr` packaging.
//!
//! This is an auxiliary diarization pack, not an ASR transcription architecture.
//! The source weights used by pyannote community-1 are
//! `pyannote/wespeaker-voxceleb-resnet34-LM` and are CC-BY-4.0; keep that
//! license/provenance in generated pack metadata.

pub mod package_import;

/// GGUF `general.architecture` / `openasr.model.architecture` id for WeSpeaker
/// ResNet34 speaker-embedder packs.
pub(crate) const WESPEAKER_GGML_ARCHITECTURE_ID: &str = "wespeaker-resnet34";
/// `openasr.model.family` id for WeSpeaker speaker-embedder packs.
pub(crate) const WESPEAKER_MODEL_FAMILY: &str = "wespeaker";
