//! One-shot selection and loading of local-activity segmenter packs.

use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use sha2::Digest;

use super::{LocalActivitySegmenter, PyannoteSegmenter, SegmentError};
use crate::config::VoiceIdSegmenterPreference;
use crate::models::thread_local_runtime_cache::PackContentKey;

static ACTIVE_SEGMENTATION_3_0: LazyLock<Mutex<Option<(PackContentKey, Arc<PyannoteSegmenter>)>>> =
    LazyLock::new(|| Mutex::new(None));

const PACK_ENV: &str = "OPENASR_PYANNOTE_PACK";
const INSTALLED_MODEL_ID_HINT: &str = "pyannote";
pub const SEGMENTER_PACK_ID: &str = "pyannote-segmentation-3.0";

/// The adapter selected during request preflight. Holding this value pins the
/// choice for the whole request: inference errors are returned directly and
/// are never interpreted as permission to try the next provider.
pub(crate) struct SelectedSegmenter {
    pub preference: VoiceIdSegmenterPreference,
    pub adapter: Arc<dyn LocalActivitySegmenter>,
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
) -> Result<Arc<PyannoteSegmenter>, SegmentError> {
    let Some(path) = segmentation_3_0_path() else {
        clear_active_segmentation_3_0();
        return Err(SegmentError::MissingPack { preference });
    };
    let (key, source) = snapshot_segmenter_source(&path).map_err(|error| {
        clear_active_segmentation_3_0();
        SegmentError::LoadFailed(format!("{}: {error}", path.display()))
    })?;
    if let Ok(cache) = ACTIVE_SEGMENTATION_3_0.lock()
        && let Some((cached_key, cached)) = cache.as_ref()
        && cached_key == &key
    {
        return Ok(Arc::clone(cached));
    }
    let built = Arc::new(source.load().map_err(|error| {
        clear_active_segmentation_3_0();
        SegmentError::LoadFailed(format!("{}: {error}", path.display()))
    })?);
    let Ok(mut cache) = ACTIVE_SEGMENTATION_3_0.lock() else {
        return Ok(built);
    };
    if let Some((cached_key, cached)) = cache.as_ref()
        && cached_key == &key
    {
        return Ok(Arc::clone(cached));
    }
    *cache = Some((key, Arc::clone(&built)));
    Ok(built)
}

/// Compatibility probe for diagnostics. Production code uses
/// [`resolve_segmenter`] so selection errors retain their typed reason.
pub fn shared_segmenter() -> Option<Arc<PyannoteSegmenter>> {
    load_segmentation_3_0(VoiceIdSegmenterPreference::Segmentation3_0).ok()
}

enum SegmenterSourceSnapshot {
    Gguf(crate::ggml_runtime::GgmlRuntimeSource),
    Safetensors(Vec<u8>),
}

impl SegmenterSourceSnapshot {
    fn load(self) -> Result<PyannoteSegmenter, String> {
        match self {
            Self::Gguf(source) => {
                PyannoteSegmenter::from_runtime_source(&source).map_err(|error| error.to_string())
            }
            Self::Safetensors(bytes) => {
                PyannoteSegmenter::from_safetensors(&bytes).map_err(|error| error.to_string())
            }
        }
    }
}

fn snapshot_segmenter_source(
    path: &Path,
) -> Result<(PackContentKey, SegmenterSourceSnapshot), String> {
    if crate::diarize::pack::is_gguf(path) {
        let source =
            crate::validate_ggml_runtime_source_path(path).map_err(|error| error.to_string())?;
        let key = PackContentKey::for_runtime_source(&source);
        Ok((key, SegmenterSourceSnapshot::Gguf(source)))
    } else {
        let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
        let key = PackContentKey::new(format!("sha256:{:x}", sha2::Sha256::digest(&bytes)));
        Ok((key, SegmenterSourceSnapshot::Safetensors(bytes)))
    }
}

fn clear_active_segmentation_3_0() {
    if let Ok(mut cache) = ACTIVE_SEGMENTATION_3_0.lock() {
        *cache = None;
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

    #[test]
    fn safetensors_snapshot_identity_tracks_same_path_replacement_and_deletion() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = dir.path().join("segmentation.safetensors");
        std::fs::write(&pack, b"segmentation-content-a").expect("write a");
        let (key_a, _) = snapshot_segmenter_source(&pack).expect("snapshot a");

        std::fs::write(&pack, b"segmentation-content-b").expect("replace b");
        let (key_b, _) = snapshot_segmenter_source(&pack).expect("snapshot b");
        assert_ne!(key_a, key_b, "same-path replacement must miss the cache");

        std::fs::remove_file(&pack).expect("delete pack");
        assert!(
            snapshot_segmenter_source(&pack).is_err(),
            "deleted pack must not resolve to the old content"
        );
    }
}
