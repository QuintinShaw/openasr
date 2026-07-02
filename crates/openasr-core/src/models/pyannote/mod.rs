//! pyannote segmentation-3.0 `.oasr` packaging (PyanNet, MIT).
//!
//! pyannote-seg is **not** an ASR transcription architecture — it emits per-frame
//! powerset speaker-activity probabilities — so it deliberately has no executor /
//! audio-frontend / decode-policy and is **not** registered in
//! `BUILTIN_ARCHITECTURE_DESCRIPTORS`. This module only converts the extracted
//! safetensors weights into a diarization `.oasr` (GGUF-v0) pack distributed via
//! the catalog + `openasr pull`; the pure-Rust forward pass and the runtime loader
//! live in [`crate::diarize::segment`].

pub mod package_import;

/// GGUF `general.architecture` / `openasr.model.architecture` id for pyannote packs.
pub(crate) const PYANNOTE_GGML_ARCHITECTURE_ID: &str = "pyannote-segmentation";
/// `openasr.model.family` id for pyannote packs.
pub(crate) const PYANNOTE_MODEL_FAMILY: &str = "pyannote-segmentation";
