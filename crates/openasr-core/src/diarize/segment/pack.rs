//! Content-addressed preflight for optional local-activity segmenters.
//!
//! This module selects a provider and pins only its immutable pack identity.
//! Runtime construction, memory admission, cache ownership, candidate retry,
//! and owner-thread destruction belong to `policy_runtime`.

use std::path::PathBuf;

use crate::config::VoiceIdSegmenterPreference;

use super::{DiariZenSegmenter, PyannoteSegmenter, SegmentError};

const PYANNOTE_PACK_ENV: &str = "OPENASR_PYANNOTE_PACK";
const PYANNOTE_MODEL_ID_HINT: &str = "pyannote-segmentation-3.0";
pub const SEGMENTER_PACK_ID: &str = PYANNOTE_MODEL_ID_HINT;
pub const DIARIZEN_PACK_ID: &str = super::diarizen::DIARIZEN_MODEL_ID;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SegmenterProvider {
    DiariZen,
    Segmentation3_0,
}

/// Lightweight request snapshot. It proves which provider and exact pack
/// content were selected without constructing any persistent runtime.
pub(crate) struct PreparedSelectedSegmenter {
    pub(crate) provider: SegmenterProvider,
    pub(crate) pack_path: PathBuf,
    pub(crate) content_id: String,
}

pub(crate) fn pyannote_pack_path() -> Option<PathBuf> {
    crate::diarize::pack::resolve_pack(PYANNOTE_PACK_ENV, PYANNOTE_MODEL_ID_HINT)
}

pub fn segmenter_pack_installed() -> bool {
    super::diarizen_pack_installed() || pyannote_pack_path().is_some()
}

/// `Auto` prefers an installed DiariZen pack. A present-but-invalid preferred
/// pack fails closed; it is never reinterpreted as absence. The explicit
/// baseline preference never probes DiariZen.
pub(crate) fn prepare_segmenter(
    preference: VoiceIdSegmenterPreference,
) -> Result<PreparedSelectedSegmenter, SegmentError> {
    if preference == VoiceIdSegmenterPreference::Auto
        && let Some(path) = super::diarizen::diarizen_pack_path()
    {
        let source = crate::validate_ggml_runtime_source_path(&path)
            .map_err(|error| SegmentError::LoadFailed(format!("{DIARIZEN_PACK_ID}: {error}")))?;
        DiariZenSegmenter::probe_runtime_source(&source)
            .map_err(|error| SegmentError::LoadFailed(format!("{DIARIZEN_PACK_ID}: {error}")))?;
        return Ok(PreparedSelectedSegmenter {
            provider: SegmenterProvider::DiariZen,
            pack_path: path,
            content_id: source.content_id().to_string(),
        });
    }

    let path = pyannote_pack_path().ok_or(SegmentError::MissingPack { preference })?;
    if !crate::diarize::pack::is_gguf(&path) {
        return Err(SegmentError::LoadFailed(format!(
            "{}: production segmenter packs must use GGUF .oasr storage",
            path.display()
        )));
    }
    let source = crate::validate_ggml_runtime_source_path(&path)
        .map_err(|error| SegmentError::LoadFailed(format!("{}: {error}", path.display())))?;
    let tensor_index = crate::read_gguf_tensor_index_from_runtime_source(&source)
        .map_err(|error| SegmentError::LoadFailed(format!("{}: {error}", path.display())))?;
    PyannoteSegmenter::quoted_persistent_host_commitment_bytes(&tensor_index)
        .map_err(|error| SegmentError::LoadFailed(format!("{}: {error}", path.display())))?;
    Ok(PreparedSelectedSegmenter {
        provider: SegmenterProvider::Segmentation3_0,
        pack_path: path,
        content_id: source.content_id().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_baseline_does_not_require_diarizen() {
        let home = tempfile::tempdir().expect("tempdir");
        let error = crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_HOME", Some(home.path().as_os_str().to_os_string())),
                ("OPENASR_DIARIZEN_PACK", None),
                (PYANNOTE_PACK_ENV, None),
            ],
            || prepare_segmenter(VoiceIdSegmenterPreference::Segmentation3_0).unwrap_err(),
        );
        assert!(matches!(
            error,
            SegmentError::MissingPack {
                preference: VoiceIdSegmenterPreference::Segmentation3_0
            }
        ));
    }
}
