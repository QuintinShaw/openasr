//! Content-addressed preflight for optional local-activity segmenters.
//!
//! This module selects a provider and pins only its immutable pack identity.
//! Runtime construction, memory admission, cache ownership, candidate retry,
//! and owner-thread destruction belong to `policy_runtime`.

use std::path::{Path, PathBuf};

use crate::config::VoiceIdSegmenterPreference;

use super::{DiariZenSegmenter, PyannoteSegmenter, SegmentError};

const PYANNOTE_PACK_ENV: &str = "OPENASR_PYANNOTE_PACK";
const PYANNOTE_MODEL_ID_HINT: &str = "pyannote-segmentation-3.0";
const PYANNOTE_RAW_SAFETENSORS_MAX_BYTES: u64 = 1024 * 1024 * 1024;
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
    pub(crate) source: PreparedSegmenterSource,
}

pub(crate) enum PreparedSegmenterSource {
    Gguf {
        preflight: crate::ggml_runtime::GgufRuntimeSourcePreflight,
        content_id: String,
    },
    Safetensors {
        path: PathBuf,
        content_id: String,
        source_bytes: u64,
        retained_quote: u64,
        parser_peak_quote: u64,
    },
}

impl PreparedSegmenterSource {
    pub(crate) fn content_id(&self) -> &str {
        match self {
            Self::Gguf { content_id, .. } | Self::Safetensors { content_id, .. } => content_id,
        }
    }
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
        let preflight =
            crate::ggml_runtime::load_runtime_source_metadata_and_tensor_index_from_source(&source)
                .map_err(|error| {
                    SegmentError::LoadFailed(format!("{DIARIZEN_PACK_ID}: {error}"))
                })?;
        DiariZenSegmenter::probe_preflight_parts(&preflight.metadata, &preflight.tensor_index)
            .map_err(|error| SegmentError::LoadFailed(format!("{DIARIZEN_PACK_ID}: {error}")))?;
        let content_id = preflight.runtime_source.content_id().to_string();
        return Ok(PreparedSelectedSegmenter {
            provider: SegmenterProvider::DiariZen,
            source: PreparedSegmenterSource::Gguf {
                preflight,
                content_id,
            },
        });
    }

    let path = pyannote_pack_path().ok_or(SegmentError::MissingPack { preference })?;
    if crate::diarize::pack::is_gguf(&path) {
        let source = crate::validate_ggml_runtime_source_path(&path)
            .map_err(|error| SegmentError::LoadFailed(format!("{}: {error}", path.display())))?;
        let preflight =
            crate::ggml_runtime::load_runtime_source_metadata_and_tensor_index_from_source(&source)
                .map_err(|error| {
                    SegmentError::LoadFailed(format!("{}: {error}", path.display()))
                })?;
        PyannoteSegmenter::quoted_persistent_host_commitment_bytes(&preflight.tensor_index)
            .map_err(|error| SegmentError::LoadFailed(format!("{}: {error}", path.display())))?;
        let content_id = preflight.runtime_source.content_id().to_string();
        return Ok(PreparedSelectedSegmenter {
            provider: SegmenterProvider::Segmentation3_0,
            source: PreparedSegmenterSource::Gguf {
                preflight,
                content_id,
            },
        });
    }

    let source = open_raw_safetensors(&path)?;
    let source_bytes = u64::try_from(source.bytes().len()).unwrap_or(u64::MAX);
    if source_bytes > PYANNOTE_RAW_SAFETENSORS_MAX_BYTES {
        return Err(SegmentError::LoadFailed(format!(
            "{}: raw safetensors source is {source_bytes} bytes, above the {PYANNOTE_RAW_SAFETENSORS_MAX_BYTES}-byte runtime limit",
            path.display()
        )));
    }
    let quote = PyannoteSegmenter::quoted_safetensors_materialization(source.bytes())
        .map_err(|error| SegmentError::LoadFailed(format!("{}: {error}", path.display())))?;
    let content_id = content_id(source.bytes());
    Ok(PreparedSelectedSegmenter {
        provider: SegmenterProvider::Segmentation3_0,
        source: PreparedSegmenterSource::Safetensors {
            path,
            content_id,
            source_bytes,
            retained_quote: quote.retained_bytes,
            parser_peak_quote: quote.parser_peak_bytes,
        },
    })
}

pub(crate) fn immutable_safetensors_snapshot(
    path: &Path,
    expected_content_id: &str,
) -> Result<memmap2::Mmap, SegmentError> {
    let source = open_raw_safetensors(path)?;
    if content_id(source.bytes()) != expected_content_id {
        return Err(content_changed(path, expected_content_id));
    }
    let mut snapshot = memmap2::MmapMut::map_anon(source.bytes().len()).map_err(|error| {
        crate::models::native_execution_services::record_current_execution_candidate_failure(
            crate::device::execution_policy::ExecutionCandidateFailure::capacity(
                "pyannote-safetensors-immutable-snapshot",
                error.to_string(),
            ),
        );
        SegmentError::LoadFailed(format!(
            "{}: could not allocate immutable safetensors snapshot: {error}",
            path.display()
        ))
    })?;
    snapshot.copy_from_slice(source.bytes());
    let snapshot = snapshot.make_read_only().map_err(|error| {
        SegmentError::LoadFailed(format!(
            "{}: could not seal immutable safetensors snapshot: {error}",
            path.display()
        ))
    })?;
    if content_id(&snapshot) != expected_content_id {
        return Err(content_changed(path, expected_content_id));
    }
    Ok(snapshot)
}

fn open_raw_safetensors(
    path: &Path,
) -> Result<crate::models::local_source_import::SafetensorsFile, SegmentError> {
    crate::models::local_source_import::SafetensorsFile::open(path)
        .map_err(|error| SegmentError::LoadFailed(format!("{}: {error}", path.display())))
}

fn content_id(bytes: &[u8]) -> String {
    use sha2::Digest;
    format!("sha256:{:x}", sha2::Sha256::digest(bytes))
}

fn content_changed(path: &Path, expected_content_id: &str) -> SegmentError {
    SegmentError::LoadFailed(format!(
        "{} changed after segmenter preflight (expected {expected_content_id})",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_raw_safetensors(path: &Path, value: f32) {
        let header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&value.to_le_bytes());
        std::fs::write(path, bytes).expect("write raw safetensors fixture");
    }

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

    #[test]
    fn raw_safetensors_dev_source_is_preflighted_and_kind_pinned() {
        let home = tempfile::tempdir().expect("tempdir");
        let pack = home.path().join("segmentation.safetensors");
        write_raw_safetensors(&pack, 1.0);
        let prepared = crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_HOME", Some(home.path().as_os_str().to_os_string())),
                (PYANNOTE_PACK_ENV, Some(pack.as_os_str().to_os_string())),
                ("OPENASR_DIARIZEN_PACK", None),
            ],
            || prepare_segmenter(VoiceIdSegmenterPreference::Segmentation3_0),
        )
        .expect("raw dev source must remain supported");
        assert!(matches!(
            &prepared.source,
            PreparedSegmenterSource::Safetensors { .. }
        ));
        assert!(prepared.source.content_id().starts_with("sha256:"));
    }

    #[test]
    fn raw_safetensors_snapshot_fails_closed_after_source_replacement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = dir.path().join("segmentation.safetensors");
        write_raw_safetensors(&pack, 1.0);
        let source = open_raw_safetensors(&pack).expect("open source a");
        let expected = content_id(source.bytes());
        drop(source);

        write_raw_safetensors(&pack, 2.0);
        let error = immutable_safetensors_snapshot(&pack, &expected)
            .expect_err("replacement bytes must not bind to the prepared identity");
        assert!(
            error
                .to_string()
                .contains("changed after segmenter preflight")
        );
    }
}
