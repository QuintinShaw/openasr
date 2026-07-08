use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::thread::LocalKey;

pub(crate) fn canonical_runtime_cache_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Default per-key entry cap for the model runtime caches keyed by
/// `with_thread_local_cached_mut_by_key`. Small on purpose: long-form
/// transcription of a single audio file can mint many distinct cache keys
/// (e.g. firered-aed keys on `(path, backend, encoder_frame_count)`, and a
/// 10s-chunked long clip commonly produces a dozen-plus distinct frame
/// counts), and each cached runtime can own hundreds of MB of ggml graph
/// context. Without a bound the per-thread cache grows without limit for the
/// life of the thread -- this is the root cause of the multi-GB memory
/// "roller coaster" during long-audio daemon transcription (issue tracked as
/// the daemon long-audio OOM). 4 keeps the common case (a handful of chunk
/// sizes, e.g. one steady-state frame count plus a shorter tail chunk) warm
/// while capping worst-case resident runtimes for a pathological key
/// explosion.
pub(crate) const DEFAULT_RUNTIME_CACHE_CAPACITY: usize = 4;

/// A small bounded LRU cache: at most `max_entries` `(key, value)` pairs are
/// resident at once. Insertion order is tracked in `order` (front = least
/// recently used, back = most recently used); eviction drops the least
/// recently used entry, which runs that entry's `Drop` and frees whatever
/// native resources (ggml graph context / static tensor arena / mmap) it
/// owns.
pub(crate) struct BoundedRuntimeCache<K, T> {
    entries: HashMap<K, T>,
    order: VecDeque<K>,
}

impl<K, T> BoundedRuntimeCache<K, T> {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

impl<K, T> Default for BoundedRuntimeCache<K, T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs `build`/`use_cached` against a bounded, thread-local LRU cache of
/// native runtimes keyed by `K`.
///
/// On a cache miss, a new entry is built via `build` and inserted; if the
/// cache is already at `max_entries`, the single least-recently-used entry is
/// evicted (and dropped, freeing its resources) first -- eviction only ever
/// happens here, before any `&mut` reference into the map is handed out, so
/// it never races with an in-flight borrow. On both a hit and a miss the
/// accessed key is promoted to most-recently-used before `use_cached` runs
/// against it.
///
/// `max_entries` must be at least 1 (a cache that can hold nothing is not a
/// cache); callers should use [`DEFAULT_RUNTIME_CACHE_CAPACITY`] unless they
/// have a specific reason to size differently.
pub(crate) fn with_thread_local_cached_mut_by_key<K, T, E, R, F, U>(
    cache: &'static LocalKey<RefCell<BoundedRuntimeCache<K, T>>>,
    key: K,
    max_entries: usize,
    build: F,
    use_cached: U,
) -> Result<R, E>
where
    K: Eq + Hash + Clone,
    F: FnOnce() -> Result<T, E>,
    U: FnOnce(&mut T) -> Result<R, E>,
{
    debug_assert!(max_entries > 0, "runtime cache capacity must be >= 1");
    let max_entries = max_entries.max(1);
    cache.with(|cache| -> Result<R, E> {
        let mut cache = cache.borrow_mut();
        if !cache.entries.contains_key(&key) {
            if cache.entries.len() >= max_entries {
                // Evict the single least-recently-used entry. This runs
                // before `build()` and before any `&mut` borrow into
                // `entries` is taken, so it never touches an entry that is
                // currently on loan to a caller.
                if let Some(lru_key) = cache.order.pop_front() {
                    cache.entries.remove(&lru_key);
                }
            }
            let runtime = build()?;
            cache.entries.insert(key.clone(), runtime);
        } else {
            // Promote: drop the key's current position so it can be
            // re-appended as most-recently-used below.
            cache.order.retain(|existing| existing != &key);
        }
        cache.order.push_back(key.clone());
        let runtime = cache
            .entries
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
        static STUB_MUT_CACHE: RefCell<BoundedRuntimeCache<PathBuf, Vec<usize>>> =
            RefCell::new(BoundedRuntimeCache::new());
        static STUB_MUT_CACHE_BY_KEY: RefCell<BoundedRuntimeCache<(PathBuf, usize), Vec<usize>>> =
            RefCell::new(BoundedRuntimeCache::new());
        static STUB_BOUNDED_CACHE: RefCell<BoundedRuntimeCache<usize, DropRecorder>> =
            RefCell::new(BoundedRuntimeCache::new());
    }

    #[test]
    fn reuses_thread_local_mut_runtime() {
        let path = Path::new("/tmp/thread-local-mut.gguf");
        let first = with_thread_local_cached_mut_by_key(
            &STUB_MUT_CACHE,
            path.to_path_buf(),
            DEFAULT_RUNTIME_CACHE_CAPACITY,
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
            DEFAULT_RUNTIME_CACHE_CAPACITY,
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
            DEFAULT_RUNTIME_CACHE_CAPACITY,
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
            DEFAULT_RUNTIME_CACHE_CAPACITY,
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

    /// Records how many live `DropRecorder` instances exist (via a shared
    /// counter) so eviction can be asserted by observing the count fall back
    /// down as entries are dropped.
    struct DropRecorder {
        counter: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl DropRecorder {
        fn new(counter: std::rc::Rc<std::cell::Cell<usize>>) -> Self {
            counter.set(counter.get() + 1);
            Self { counter }
        }
    }

    impl Drop for DropRecorder {
        fn drop(&mut self) {
            self.counter.set(self.counter.get() - 1);
        }
    }

    #[test]
    fn bounded_cache_evicts_least_recently_used_entry_and_drops_it() {
        let counter = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let cap = 2usize;

        for key in 0..2 {
            let counter = counter.clone();
            with_thread_local_cached_mut_by_key(
                &STUB_BOUNDED_CACHE,
                key,
                cap,
                move || Ok::<_, &'static str>(DropRecorder::new(counter)),
                |_recorder| Ok::<_, &'static str>(()),
            )
            .expect("insert within capacity");
        }
        assert_eq!(counter.get(), 2, "both entries within capacity stay live");

        // Insert a third distinct key: the cache is at capacity, so the
        // least-recently-used entry (key 0) must be evicted and dropped
        // before the new one is built.
        {
            let counter = counter.clone();
            with_thread_local_cached_mut_by_key(
                &STUB_BOUNDED_CACHE,
                2usize,
                cap,
                move || Ok::<_, &'static str>(DropRecorder::new(counter)),
                |_recorder| Ok::<_, &'static str>(()),
            )
            .expect("insert beyond capacity evicts lru");
        }

        assert_eq!(
            counter.get(),
            2,
            "cache never holds more than `max_entries` live entries"
        );
        STUB_BOUNDED_CACHE.with(|cache| {
            let cache = cache.borrow();
            assert_eq!(cache.len(), cap, "entry count stays at the configured cap");
            assert!(
                !cache.entries.contains_key(&0usize),
                "least-recently-used key 0 was evicted"
            );
            assert!(cache.entries.contains_key(&1usize));
            assert!(cache.entries.contains_key(&2usize));
        });
    }

    #[test]
    fn bounded_cache_promotes_recently_used_entry_ahead_of_eviction() {
        let counter = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let cap = 2usize;
        thread_local! {
            static PROMOTE_CACHE: RefCell<BoundedRuntimeCache<usize, DropRecorder>> =
                RefCell::new(BoundedRuntimeCache::new());
        }

        for key in 0..2 {
            let counter = counter.clone();
            with_thread_local_cached_mut_by_key(
                &PROMOTE_CACHE,
                key,
                cap,
                move || Ok::<_, &'static str>(DropRecorder::new(counter)),
                |_recorder| Ok::<_, &'static str>(()),
            )
            .expect("insert within capacity");
        }

        // Touch key 0 again so it becomes most-recently-used; key 1 is now
        // the least-recently-used one.
        with_thread_local_cached_mut_by_key(
            &PROMOTE_CACHE,
            0usize,
            cap,
            || -> Result<DropRecorder, &'static str> {
                panic!("key 0 must already be cached, build() must not run on a hit")
            },
            |_recorder| Ok::<_, &'static str>(()),
        )
        .expect("re-touch existing key");

        {
            let counter = counter.clone();
            with_thread_local_cached_mut_by_key(
                &PROMOTE_CACHE,
                2usize,
                cap,
                move || Ok::<_, &'static str>(DropRecorder::new(counter)),
                |_recorder| Ok::<_, &'static str>(()),
            )
            .expect("insert beyond capacity evicts lru");
        }

        PROMOTE_CACHE.with(|cache| {
            let cache = cache.borrow();
            assert!(
                cache.entries.contains_key(&0usize),
                "recently re-touched key 0 survives eviction"
            );
            assert!(
                !cache.entries.contains_key(&1usize),
                "untouched key 1 is the one evicted"
            );
            assert!(cache.entries.contains_key(&2usize));
        });
    }
}
