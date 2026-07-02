use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::thread::LocalKey;

pub(crate) fn canonical_runtime_cache_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

pub(crate) fn with_thread_local_cached_mut_by_key<K, T, E, R, F, U>(
    cache: &'static LocalKey<RefCell<HashMap<K, T>>>,
    key: K,
    build: F,
    use_cached: U,
) -> Result<R, E>
where
    K: Eq + Hash + Clone,
    F: FnOnce() -> Result<T, E>,
    U: FnOnce(&mut T) -> Result<R, E>,
{
    cache.with(|cache| -> Result<R, E> {
        let mut cache = cache.borrow_mut();
        if !cache.contains_key(&key) {
            let runtime = build()?;
            cache.insert(key.clone(), runtime);
        }
        let runtime = cache
            .get_mut(&key)
            .expect("thread-local runtime cache must contain inserted entry");
        use_cached(runtime)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    thread_local! {
        static STUB_MUT_CACHE: RefCell<HashMap<PathBuf, Vec<usize>>> = RefCell::new(HashMap::new());
        static STUB_MUT_CACHE_BY_KEY: RefCell<HashMap<(PathBuf, usize), Vec<usize>>> =
            RefCell::new(HashMap::new());
    }

    #[test]
    fn reuses_thread_local_mut_runtime() {
        let path = Path::new("/tmp/thread-local-mut.gguf");
        let first = with_thread_local_cached_mut_by_key(
            &STUB_MUT_CACHE,
            path.to_path_buf(),
            || Ok::<_, &'static str>(vec![1]),
            |cached| {
                cached.push(2);
                Ok::<_, &'static str>(cached.clone())
            },
        )
        .expect("first");
        let second = with_thread_local_cached_mut_by_key(
            &STUB_MUT_CACHE,
            path.to_path_buf(),
            || Ok::<_, &'static str>(vec![9]),
            |cached| {
                cached.push(3);
                Ok::<_, &'static str>(cached.clone())
            },
        )
        .expect("second");

        assert_eq!(first, vec![1, 2]);
        assert_eq!(second, vec![1, 2, 3]);
    }

    #[test]
    fn separates_thread_local_mut_runtime_by_custom_key() {
        let path = Path::new("/tmp/thread-local-mut-key.gguf");
        let cpu = with_thread_local_cached_mut_by_key(
            &STUB_MUT_CACHE_BY_KEY,
            (path.to_path_buf(), 0),
            || Ok::<_, &'static str>(vec![1]),
            |cached| {
                cached.push(2);
                Ok::<_, &'static str>(cached.clone())
            },
        )
        .expect("cpu");
        let gpu = with_thread_local_cached_mut_by_key(
            &STUB_MUT_CACHE_BY_KEY,
            (path.to_path_buf(), 1),
            || Ok::<_, &'static str>(vec![9]),
            |cached| {
                cached.push(3);
                Ok::<_, &'static str>(cached.clone())
            },
        )
        .expect("gpu");

        assert_eq!(cpu, vec![1, 2]);
        assert_eq!(gpu, vec![9, 3]);
    }

    #[test]
    fn canonical_runtime_cache_path_falls_back_to_original_path() {
        let path = Path::new("/tmp/openasr-missing-runtime-cache-path.gguf");
        assert_eq!(canonical_runtime_cache_path(path), path.to_path_buf());
    }
}
