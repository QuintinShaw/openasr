//! Runtime resolution and content-aware caching of the optional DiariZen pack.

use std::sync::{Arc, LazyLock, Mutex};

use super::{DiariZenSegmenter, DiariZenSegmenterError};
use crate::models::thread_local_runtime_cache::PackContentKey;

const PACK_ENV: &str = "OPENASR_DIARIZEN_PACK";
const INSTALLED_MODEL_ID_HINT: &str = "diarizen";

static ACTIVE_DIARIZEN: LazyLock<Mutex<Option<(PackContentKey, Arc<DiariZenSegmenter>)>>> =
    LazyLock::new(|| Mutex::new(None));

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
    let key = PackContentKey::for_runtime_source(&source);
    if let Ok(cache) = ACTIVE_DIARIZEN.lock()
        && let Some((cached_key, segmenter)) = cache.as_ref()
        && cached_key == &key
    {
        return Ok(Some(Arc::clone(segmenter)));
    }

    // Build outside the cache mutex. Loading/dequantizing and graph creation
    // are expensive; inference must never wait while this lock is held.
    let built = match DiariZenSegmenter::from_runtime_source(&source) {
        Ok(segmenter) => Arc::new(segmenter),
        Err(error) => {
            clear_active_segmenter();
            return Err(error);
        }
    };
    let Ok(mut cache) = ACTIVE_DIARIZEN.lock() else {
        return Ok(Some(built));
    };
    if let Some((cached_key, segmenter)) = cache.as_ref()
        && cached_key == &key
    {
        return Ok(Some(Arc::clone(segmenter)));
    }
    *cache = Some((key, Arc::clone(&built)));
    Ok(Some(built))
}

fn clear_active_segmenter() {
    if let Ok(mut cache) = ACTIVE_DIARIZEN.lock() {
        *cache = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        unsafe {
            std::env::set_var("OPENASR_HOME", dir.path());
            std::env::remove_var(PACK_ENV);
        }
        assert!(!diarizen_pack_installed());
        assert!(
            load_diarizen_segmenter()
                .expect("an absent optional pack is not an error")
                .is_none()
        );

        let broken = dir.path().join("diarizen-broken.oasr");
        std::fs::write(&broken, b"GGUFbroken-installed-diarizen-pack").expect("write broken pack");
        unsafe { std::env::set_var(PACK_ENV, &broken) };
        assert!(diarizen_pack_installed());
        assert!(
            load_diarizen_segmenter().is_err(),
            "a broken present pack must fail instead of looking absent"
        );
    }
}
