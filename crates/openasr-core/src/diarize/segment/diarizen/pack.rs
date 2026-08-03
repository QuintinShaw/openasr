//! Runtime resolution and content-aware caching of the optional DiariZen pack.

use std::sync::{Arc, LazyLock, Mutex};

use super::{DiariZenSegmenter, DiariZenSegmenterError};
use crate::diarize::segment::{SegmenterExecutionKey, SegmenterRuntimeInput};
use crate::ggml_runtime::request_backend_override;
use crate::models::thread_local_runtime_cache::PackContentKey;

const PACK_ENV: &str = "OPENASR_DIARIZEN_PACK";
const INSTALLED_MODEL_ID_HINT: &str = super::DIARIZEN_MODEL_ID;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DiariZenRuntimeKey {
    pack: PackContentKey,
    execution: Vec<SegmenterExecutionKey>,
}

static ACTIVE_DIARIZEN: LazyLock<Mutex<Option<(DiariZenRuntimeKey, Arc<DiariZenSegmenter>, u64)>>> =
    LazyLock::new(|| Mutex::new(None));

pub(crate) struct PreparedDiariZenSegmenter {
    key: DiariZenRuntimeKey,
    source: crate::ggml_runtime::GgmlRuntimeSource,
    runtime_input: SegmenterRuntimeInput,
    pack_bytes: u64,
}

impl PreparedDiariZenSegmenter {
    pub(crate) const fn pack_bytes(&self) -> u64 {
        self.pack_bytes
    }

    pub(crate) fn minimum_vram_budget_bytes(&self) -> Option<u64> {
        self.runtime_input.minimum_vram_budget_bytes()
    }

    #[cfg(test)]
    pub(crate) fn content_id(&self) -> &str {
        &self.key.pack.pack_content_id
    }

    pub(crate) fn materialize(self) -> Result<Arc<DiariZenSegmenter>, DiariZenSegmenterError> {
        if let Ok(cache) = ACTIVE_DIARIZEN.lock()
            && let Some((cached_key, segmenter, _)) = cache.as_ref()
            && cached_key == &self.key
        {
            return Ok(Arc::clone(segmenter));
        }

        let immutable_source = self
            .source
            .immutable_snapshot_matching_content_id(&self.key.pack.pack_content_id)
            .map_err(|error| DiariZenSegmenterError::PackSource(error.to_string()))?;

        let built = Arc::new(DiariZenSegmenter::from_runtime_source(
            &immutable_source,
            self.runtime_input,
        )?);
        let Ok(mut cache) = ACTIVE_DIARIZEN.lock() else {
            return Ok(built);
        };
        if let Some((cached_key, segmenter, _)) = cache.as_ref()
            && cached_key == &self.key
        {
            return Ok(Arc::clone(segmenter));
        }
        *cache = Some((self.key, Arc::clone(&built), self.pack_bytes));
        Ok(built)
    }
}

/// Snapshot of the currently resolved DiariZen pack. Every call opens the
/// current path and derives an identity from that exact mapping. Equal content
/// reuses the resident graph, replacement atomically swaps it, and absence or
/// a broken pack clears the active slot. Returned `Arc`s pin in-flight work to
/// its original immutable graph snapshot.
pub fn shared_diarizen_segmenter() -> Option<Arc<DiariZenSegmenter>> {
    load_diarizen_segmenter().ok().flatten()
}

fn diarizen_pack_path() -> Option<std::path::PathBuf> {
    crate::diarize::pack::resolve_pack(PACK_ENV, INSTALLED_MODEL_ID_HINT)
}

/// Presence-only capability probe. It intentionally does not parse or load
/// the pack; callers that must distinguish an absent optional pack from a
/// broken installed pack use [`load_diarizen_segmenter`].
pub fn diarizen_pack_installed() -> bool {
    diarizen_pack_path().is_some()
}

/// Typed snapshot loader for strict `auto` selection.
///
/// `Ok(None)` means no optional pack resolves. A path that resolves but fails
/// admission, metadata/tensor validation, or graph construction returns
/// `Err`, clears the active cache, and must not be treated as an absent pack
/// by the selection layer.
pub fn load_diarizen_segmenter() -> Result<Option<Arc<DiariZenSegmenter>>, DiariZenSegmenterError> {
    let runtime_input = SegmenterRuntimeInput::resolve(request_backend_override())?;
    prepare_diarizen_segmenter_snapshot(runtime_input)?
        .map(PreparedDiariZenSegmenter::materialize)
        .transpose()
}

/// Lightweight, TOCTOU-safe pack snapshot for request admission. Metadata and
/// the tensor contract are checked now, but no weights, runner, or graph are
/// materialized until [`PreparedDiariZenSegmenter::materialize`] after audio
/// preparation and memory admission.
pub(crate) fn prepare_diarizen_segmenter_snapshot(
    runtime_input: SegmenterRuntimeInput,
) -> Result<Option<PreparedDiariZenSegmenter>, DiariZenSegmenterError> {
    let Some(path) = diarizen_pack_path() else {
        clear_active_segmenter();
        return Ok(None);
    };
    let source = match crate::validate_ggml_runtime_source_path(&path) {
        Ok(source) => source,
        Err(error) => {
            clear_active_segmenter();
            return Err(DiariZenSegmenterError::PackSource(error.to_string()));
        }
    };
    DiariZenSegmenter::probe_runtime_source(&source).inspect_err(|_| clear_active_segmenter())?;
    let key = DiariZenRuntimeKey {
        pack: PackContentKey::for_runtime_source(&source),
        execution: runtime_input.execution_keys(),
    };
    let pack_bytes = source.byte_len();
    Ok(Some(PreparedDiariZenSegmenter {
        key,
        source,
        runtime_input,
        pack_bytes,
    }))
}

fn clear_active_segmenter() {
    if let Ok(mut cache) = ACTIVE_DIARIZEN.lock() {
        *cache = None;
    }
}

pub(crate) fn unload_idle_diarizen_cache() {
    clear_active_segmenter();
    super::unload_idle_worker_runtimes();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installed_pack_identity_is_exact_and_stable() {
        assert_eq!(PACK_ENV, "OPENASR_DIARIZEN_PACK");
        assert_eq!(INSTALLED_MODEL_ID_HINT, "diarizen-large-s80-v2");
    }

    #[test]
    fn runtime_content_identity_tracks_same_path_replacement_and_deletion() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = dir.path().join("diarizen.oasr");
        std::fs::write(&pack, b"GGUFdiarizen-content-a").expect("write a");
        let source_a = crate::validate_ggml_runtime_source_path(&pack).expect("source a");
        let key_a = PackContentKey::for_runtime_source(&source_a);

        std::fs::write(&pack, b"GGUFdiarizen-content-b").expect("replace b");
        let source_b = crate::validate_ggml_runtime_source_path(&pack).expect("source b");
        let key_b = PackContentKey::for_runtime_source(&source_b);
        assert_ne!(key_a, key_b, "same-path replacement must miss the cache");

        std::fs::remove_file(&pack).expect("delete pack");
        assert!(
            crate::validate_ggml_runtime_source_path(&pack).is_err(),
            "deleted pack must not resolve to the old content"
        );
    }

    #[test]
    fn typed_loader_distinguishes_absent_from_present_broken_pack() {
        let dir = tempfile::tempdir().expect("tempdir");
        crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_HOME", Some(dir.path().as_os_str().to_os_string())),
                (PACK_ENV, None),
            ],
            || {
                assert!(!diarizen_pack_installed());
                assert!(
                    load_diarizen_segmenter()
                        .expect("an absent optional pack is not an error")
                        .is_none()
                );
            },
        );

        let broken = dir.path().join("diarizen-broken.oasr");
        std::fs::write(&broken, b"GGUFbroken-installed-diarizen-pack").expect("write broken pack");
        crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_HOME", Some(dir.path().as_os_str().to_os_string())),
                (PACK_ENV, Some(broken.as_os_str().to_os_string())),
            ],
            || {
                assert!(diarizen_pack_installed());
                assert!(
                    load_diarizen_segmenter().is_err(),
                    "a broken present pack must fail instead of looking absent"
                );
            },
        );
    }
}
