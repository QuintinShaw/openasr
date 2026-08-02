//! One-shot selection and loading of local-activity segmenter packs.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::{LocalActivitySegmenter, PyannoteSegmenter, SegmentError};
use crate::config::VoiceIdSegmenterPreference;

static SEGMENTATION_3_0: OnceLock<PyannoteSegmenter> = OnceLock::new();

const PACK_ENV: &str = "OPENASR_PYANNOTE_PACK";
const INSTALLED_MODEL_ID_HINT: &str = "pyannote";
pub const SEGMENTER_PACK_ID: &str = "pyannote-segmentation-3.0";

/// The adapter selected during request preflight. Holding this value pins the
/// choice for the whole request: inference errors are returned directly and
/// are never interpreted as permission to try the next provider.
pub(crate) struct SelectedSegmenter {
    pub preference: VoiceIdSegmenterPreference,
    pub adapter: &'static dyn LocalActivitySegmenter,
}

fn segmentation_3_0_path() -> Option<PathBuf> {
    crate::diarize::pack::resolve_pack(PACK_ENV, INSTALLED_MODEL_ID_HINT)
}

pub fn segmenter_pack_installed() -> bool {
    segmentation_3_0_path().is_some()
}

/// Resolve the user's model-level preference once. `Auto` is intentionally a
/// provider registry rather than an alias for segmentation-3.0: a future
/// DiariZen adapter can be inserted ahead of the baseline without changing
/// the diarization module's interface. `Segmentation3_0` filters that registry
/// to the locked baseline and therefore disables any future preferred model.
pub(crate) fn resolve_segmenter(
    preference: VoiceIdSegmenterPreference,
) -> Result<SelectedSegmenter, SegmentError> {
    match preference {
        VoiceIdSegmenterPreference::Auto | VoiceIdSegmenterPreference::Segmentation3_0 => {
            let adapter = load_segmentation_3_0(preference)?;
            Ok(SelectedSegmenter {
                preference: VoiceIdSegmenterPreference::Segmentation3_0,
                adapter,
            })
        }
    }
}

fn load_segmentation_3_0(
    preference: VoiceIdSegmenterPreference,
) -> Result<&'static PyannoteSegmenter, SegmentError> {
    if let Some(segmenter) = SEGMENTATION_3_0.get() {
        return Ok(segmenter);
    }
    let path = segmentation_3_0_path().ok_or(SegmentError::MissingPack { preference })?;
    let segmenter = load_segmenter(&path)
        .map_err(|error| SegmentError::LoadFailed(format!("{}: {error}", path.display())))?;
    let _ = SEGMENTATION_3_0.set(segmenter);
    SEGMENTATION_3_0
        .get()
        .ok_or_else(|| SegmentError::LoadFailed("segmenter cache initialization failed".into()))
}

/// Compatibility probe for diagnostics. Production code uses
/// [`resolve_segmenter`] so selection errors retain their typed reason.
pub fn shared_segmenter() -> Option<&'static PyannoteSegmenter> {
    load_segmentation_3_0(VoiceIdSegmenterPreference::Segmentation3_0).ok()
}

fn load_segmenter(path: &Path) -> Result<PyannoteSegmenter, String> {
    if crate::diarize::pack::is_gguf(path) {
        PyannoteSegmenter::from_oasr(path).map_err(|error| error.to_string())
    } else {
        let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
        PyannoteSegmenter::from_safetensors(&bytes).map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_baseline_missing_pack_fails_closed_with_typed_error() {
        let home = tempfile::tempdir().unwrap();
        let error = crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_PYANNOTE_PACK", None),
                ("OPENASR_HOME", Some(home.path().as_os_str().to_os_string())),
            ],
            || {
                resolve_segmenter(VoiceIdSegmenterPreference::Segmentation3_0)
                    .err()
                    .expect("missing forced baseline must fail closed")
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
