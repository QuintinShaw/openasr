use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::stage_timing;

fn prepared_runtime_cache_key(runtime_path: &Path) -> PathBuf {
    std::fs::canonicalize(runtime_path).unwrap_or_else(|_| runtime_path.to_path_buf())
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedRuntimeCache<T> {
    cache_by_path: Arc<Mutex<HashMap<PathBuf, Arc<T>>>>,
}

impl<T> Default for PreparedRuntimeCache<T> {
    fn default() -> Self {
        Self {
            cache_by_path: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl<T> PreparedRuntimeCache<T> {
    pub(crate) fn get_or_try_insert_with<E, F, M>(
        &self,
        runtime_path: &Path,
        build: F,
        map_poisoned_lock: M,
    ) -> Result<Arc<T>, E>
    where
        F: FnOnce() -> Result<T, E>,
        M: Fn() -> E,
    {
        let cache_key = prepared_runtime_cache_key(runtime_path);
        if let Some(runtime) = self.get_by_key(&cache_key, &map_poisoned_lock)? {
            return Ok(runtime);
        }
        // Model pack loading (mmap + tensor materialization + context/graph
        // construction, up to inference-ready) happens exactly here, exactly
        // once per distinct runtime path per process (subsequent calls hit the
        // cache check above). This one call site covers every builtin model
        // family that goes through this cache, so it is the single place to
        // time "how long did loading this pack take" without instrumenting
        // each family's build function separately.
        let load_started = Instant::now();
        let prepared = Arc::new(build()?);
        stage_timing::log_event(
            "model_pack_load",
            format_args!(
                "path={} duration_ms={:.3}",
                runtime_path.display(),
                load_started.elapsed().as_secs_f64() * 1000.0
            ),
        );
        let mut cache = self.cache_by_path.lock().map_err(|_| map_poisoned_lock())?;
        let entry = cache
            .entry(cache_key)
            .or_insert_with(|| Arc::clone(&prepared));
        Ok(Arc::clone(entry))
    }

    fn get_by_key<E, M>(&self, cache_key: &Path, map_poisoned_lock: M) -> Result<Option<Arc<T>>, E>
    where
        M: Fn() -> E,
    {
        let cache = self.cache_by_path.lock().map_err(|_| map_poisoned_lock())?;
        Ok(cache.get(cache_key).cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct StubRuntime {
        value: usize,
    }

    #[test]
    fn reuses_cached_runtime_for_same_path() {
        let cache = PreparedRuntimeCache::<StubRuntime>::default();
        let path = Path::new("/tmp/test-runtime.gguf");

        let runtime_a = cache
            .get_or_try_insert_with(
                path,
                || Ok::<_, &'static str>(StubRuntime { value: 7 }),
                || "poisoned",
            )
            .expect("runtime a");
        let runtime_b = cache
            .get_or_try_insert_with(
                path,
                || Ok::<_, &'static str>(StubRuntime { value: 9 }),
                || "poisoned",
            )
            .expect("runtime b");

        assert!(Arc::ptr_eq(&runtime_a, &runtime_b));
        assert_eq!(runtime_b.value, 7);
    }

    #[test]
    fn reuses_cached_runtime_for_canonical_equivalent_paths() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("runtime.gguf");
        std::fs::write(&runtime_path, b"GGUF").expect("write runtime");
        let dotted_path = temp.path().join(".").join("runtime.gguf");
        let cache = PreparedRuntimeCache::<StubRuntime>::default();
        let build_count = Cell::new(0usize);

        let runtime_a = cache
            .get_or_try_insert_with(
                &dotted_path,
                || {
                    build_count.set(build_count.get() + 1);
                    Ok::<_, &'static str>(StubRuntime { value: 7 })
                },
                || "poisoned",
            )
            .expect("runtime a");
        let runtime_b = cache
            .get_or_try_insert_with(
                &runtime_path,
                || {
                    build_count.set(build_count.get() + 1);
                    Ok::<_, &'static str>(StubRuntime { value: 9 })
                },
                || "poisoned",
            )
            .expect("runtime b");

        assert_eq!(build_count.get(), 1);
        assert!(Arc::ptr_eq(&runtime_a, &runtime_b));
        assert_eq!(runtime_b.value, 7);
    }
}
