//! Content-addressed preflight for optional local-activity segmenters.
//!
//! This module selects a provider and pins only its immutable pack identity.
//! Runtime construction, memory admission, cache ownership, candidate retry,
//! and owner-thread destruction belong to `policy_runtime`.

use std::path::PathBuf;

use crate::config::VoiceIdSegmenterPreference;
use crate::models::{
    aux_pack_registry::AuxPackKind,
    pack_verifier::{PackCandidate, PackRoute, PackVerifier},
};

use super::{PyannoteSegmenter, SegmentError};

const PYANNOTE_PACK_ENV: &str = "OPENASR_PYANNOTE_PACK";
const PYANNOTE_MODEL_ID_HINT: &str = "pyannote-segmentation-3.0";
const PYANNOTE_PREFERRED_QUANT: &str = "f32";
pub const SEGMENTER_PACK_ID: &str = PYANNOTE_MODEL_ID_HINT;
pub const DIARIZEN_PACK_ID: &str = super::diarizen::DIARIZEN_MODEL_ID;
pub(crate) const PYANNOTE_PACK_PREFERENCE: crate::capability_pack::CapabilityPackPreference =
    crate::capability_pack::CapabilityPackPreference::new(
        SEGMENTER_PACK_ID,
        PYANNOTE_MODEL_ID_HINT,
        PYANNOTE_PREFERRED_QUANT,
    );

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SegmenterProvider {
    DiariZen,
    Segmentation3_0,
}

/// Lightweight request snapshot. It proves which provider and exact pack
/// content were selected without constructing any persistent runtime.
pub(crate) struct PreparedSelectedSegmenter {
    pub(crate) provider: SegmenterProvider,
    pub(crate) source: PreparedSegmenterSource,
}

pub(crate) struct PreparedSegmenterSource {
    preflight: crate::ggml_runtime::GgufRuntimeSourcePreflight,
    content_id: String,
}

impl PreparedSegmenterSource {
    pub(crate) fn content_id(&self) -> &str {
        &self.content_id
    }

    pub(crate) fn preflight(&self) -> &crate::ggml_runtime::GgufRuntimeSourcePreflight {
        &self.preflight
    }

    pub(crate) fn into_parts(self) -> (crate::ggml_runtime::GgufRuntimeSourcePreflight, String) {
        (self.preflight, self.content_id)
    }
}

pub(crate) fn pyannote_pack_path() -> Option<PathBuf> {
    crate::diarize::pack::resolve_pack(PYANNOTE_PACK_ENV, PYANNOTE_PACK_PREFERENCE)
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
        let verified_pack = PackVerifier
            .verify_candidate(PackCandidate::new(&path))
            .map_err(|error| SegmentError::LoadFailed(format!("{DIARIZEN_PACK_ID}: {error}")))?;
        if !matches!(
            verified_pack.route(),
            PackRoute::Aux {
                kind: AuxPackKind::Diarization,
                ..
            }
        ) {
            return Err(SegmentError::LoadFailed(format!(
                "{DIARIZEN_PACK_ID}: pack route is not auxiliary diarization: {:?}",
                verified_pack.route()
            )));
        }
        let preflight = verified_pack.preflight().clone();
        let content_id = preflight.runtime_source.content_id().to_string();
        return Ok(PreparedSelectedSegmenter {
            provider: SegmenterProvider::DiariZen,
            source: PreparedSegmenterSource {
                preflight,
                content_id,
            },
        });
    }

    let path = pyannote_pack_path().ok_or(SegmentError::MissingPack { preference })?;
    let verified_pack = PackVerifier
        .verify_candidate(PackCandidate::new(&path))
        .map_err(|error| SegmentError::LoadFailed(format!("{}: {error}", path.display())))?;
    if !matches!(
        verified_pack.route(),
        PackRoute::Aux {
            kind: AuxPackKind::Diarization,
            ..
        }
    ) {
        return Err(SegmentError::LoadFailed(format!(
            "{}: pack route is not auxiliary diarization: {:?}",
            path.display(),
            verified_pack.route()
        )));
    }
    let preflight = verified_pack.preflight().clone();
    PyannoteSegmenter::quoted_persistent_host_commitment_bytes(&preflight.tensor_index)
        .map_err(|error| SegmentError::LoadFailed(format!("{}: {error}", path.display())))?;
    let content_id = preflight.runtime_source.content_id().to_string();
    Ok(PreparedSelectedSegmenter {
        provider: SegmenterProvider::Segmentation3_0,
        source: PreparedSegmenterSource {
            preflight,
            content_id,
        },
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
            || match prepare_segmenter(VoiceIdSegmenterPreference::Segmentation3_0) {
                Ok(_) => panic!("missing pack must fail closed"),
                Err(error) => error,
            },
        );
        assert!(matches!(
            error,
            SegmentError::MissingPack {
                preference: VoiceIdSegmenterPreference::Segmentation3_0
            }
        ));
    }
}
