use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::LocalKey;

pub(crate) fn canonical_runtime_cache_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Process-wide generation of native idle-unload sweeps, bumped once per
/// successful `unload_idle_native_model_runtime_caches` call.
///
/// The thread-local runtime caches in this module (and the per-family caches
/// built on it) live in the TLS of whatever worker thread ran the model --
/// typically a reused `spawn_blocking` thread -- so the idle-unload reaper,
/// which runs on a different thread, cannot drop them directly. Instead each
/// cache records the generation it was last synced at and discards its
/// resident runtimes the next time its owning thread touches it after the
/// generation has moved on. Release is therefore *lazy*: a runtime pinned in
/// a thread that never runs that model family again stays resident until the
/// thread itself exits (for tokio's blocking pool, after its keep-alive
/// expiry) -- but it can never be handed out as a cache hit again.
static RUNTIME_CACHE_UNLOAD_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Current idle-unload generation. `Relaxed` is sufficient: this is a coarse
/// "has an unload happened since this cache was filled" signal, not a
/// synchronization primitive.
pub(crate) fn current_unload_generation() -> u64 {
    RUNTIME_CACHE_UNLOAD_GENERATION.load(Ordering::Relaxed)
}

/// Marks one idle-unload sweep. Called by
/// `unload_idle_native_model_runtime_caches` after the process-lifetime
/// dispatch caches have been dropped, so the thread-local caches follow suit
/// on their owning threads' next use.
pub(crate) fn bump_unload_generation() {
    RUNTIME_CACHE_UNLOAD_GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// Removes and returns the entry for `key` from a map of
/// `(unload generation, runtime)` pairs, discarding every entry (the
/// requested one included) whose recorded generation predates
/// `current_generation`. Stale runtimes are dropped in place -- on the thread
/// that owns them, which is the only thread that safely can -- including
/// stale entries under *other* keys, so a decoder cached for one pack does
/// not stay resident just because only a different pack is requested after an
/// idle unload.
pub(crate) fn take_generation_tagged<K: Eq + Hash, V>(
    entries: &mut HashMap<K, (u64, V)>,
    key: &K,
    current_generation: u64,
) -> Option<V> {
    entries.retain(|_, (generation, _)| *generation == current_generation);
    entries.remove(key).map(|(_, value)| value)
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
    /// The unload generation this cache was last synced at; see
    /// [`BoundedRuntimeCache::sync_to_unload_generation`].
    unload_generation: u64,
}

impl<K, T> BoundedRuntimeCache<K, T> {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            unload_generation: current_unload_generation(),
        }
    }

    /// Drops every cached runtime if an idle unload happened since the last
    /// access on this thread. The idle-unload reaper cannot reach another
    /// thread's TLS, so this lazy check on the owning thread is where the
    /// thread-local share of an `idle_unload` eviction actually happens.
    fn sync_to_unload_generation(&mut self, current_generation: u64) {
        if self.unload_generation != current_generation {
            self.entries.clear();
            self.order.clear();
            self.unload_generation = current_generation;
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

/// Wraps a thread-local cache payload (a map of resident runtimes/sessions)
/// with the unload generation it was last synced at. [`Self::synced`] resets
/// the payload to its default (dropping every resident entry, on the owning
/// thread) when an idle unload has happened since the previous access, then
/// hands out the payload -- callers do all their map operations through it so
/// no access path can see pre-unload entries.
pub(crate) struct UnloadGenerationGated<T> {
    unload_generation: u64,
    payload: T,
}

impl<T: Default> UnloadGenerationGated<T> {
    pub(crate) fn new() -> Self {
        Self {
            unload_generation: current_unload_generation(),
            payload: T::default(),
        }
    }

    pub(crate) fn synced(&mut self) -> &mut T {
        self.sync_to(current_unload_generation())
    }

    fn sync_to(&mut self, current_generation: u64) -> &mut T {
        if self.unload_generation != current_generation {
            self.payload = T::default();
            self.unload_generation = current_generation;
        }
        &mut self.payload
    }
}

impl<T: Default> Default for UnloadGenerationGated<T> {
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
        cache.sync_to_unload_generation(current_unload_generation());
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

/// Serializes tests that bump the process-wide unload generation against
/// tests that assert cache persistence across consecutive calls. Without it,
/// a bump landing between another test's two cache accesses would spuriously
/// clear that test's thread-local cache when the whole test binary runs in
/// one process (plain `cargo test`; `cargo nextest` isolates per process).
#[cfg(test)]
pub(crate) fn unload_generation_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
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
        let _generation_guard = unload_generation_test_lock();
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
        let _generation_guard = unload_generation_test_lock();
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
        let _generation_guard = unload_generation_test_lock();
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
        let _generation_guard = unload_generation_test_lock();
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

    #[test]
    fn take_generation_tagged_returns_entry_built_under_the_same_generation() {
        let mut entries: HashMap<&str, (u64, u32)> = HashMap::new();
        entries.insert("pack-a", (7, 41));

        assert_eq!(take_generation_tagged(&mut entries, &"pack-a", 7), Some(41));
        assert!(
            entries.is_empty(),
            "take removes the returned entry from the map"
        );
    }

    #[test]
    fn take_generation_tagged_discards_entries_built_before_the_current_generation() {
        let counter = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let mut entries: HashMap<&str, (u64, DropRecorder)> = HashMap::new();
        entries.insert("pack-a", (7, DropRecorder::new(counter.clone())));
        // A stale entry under a *different* key must be swept too: without
        // the sweep it would stay resident until its own pack is requested
        // again on this thread, which may never happen.
        entries.insert("pack-b", (7, DropRecorder::new(counter.clone())));
        assert_eq!(counter.get(), 2);

        assert!(
            take_generation_tagged(&mut entries, &"pack-a", 8).is_none(),
            "an entry built before the current unload generation must not be a hit"
        );
        assert_eq!(
            counter.get(),
            0,
            "both stale entries are dropped on the owning thread during the take"
        );
        assert!(entries.is_empty());
    }

    #[test]
    fn bounded_cache_sync_drops_all_entries_when_the_generation_moves_on() {
        let counter = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let mut cache: BoundedRuntimeCache<usize, DropRecorder> = BoundedRuntimeCache::new();
        cache.unload_generation = 3;
        cache.entries.insert(0, DropRecorder::new(counter.clone()));
        cache.entries.insert(1, DropRecorder::new(counter.clone()));
        cache.order.push_back(0);
        cache.order.push_back(1);
        assert_eq!(counter.get(), 2);

        cache.sync_to_unload_generation(3);
        assert_eq!(counter.get(), 2, "same generation keeps entries resident");

        cache.sync_to_unload_generation(4);
        assert_eq!(counter.get(), 0, "newer generation drops every entry");
        assert!(cache.entries.is_empty());
        assert!(cache.order.is_empty());
        assert_eq!(cache.unload_generation, 4);
    }

    #[test]
    fn unload_generation_gated_payload_resets_when_the_generation_moves_on() {
        let mut gated: UnloadGenerationGated<Vec<u32>> = UnloadGenerationGated::new();
        gated.unload_generation = 3;
        gated.sync_to(3).push(41);
        assert_eq!(gated.sync_to(3).as_slice(), &[41]);

        assert!(
            gated.sync_to(4).is_empty(),
            "newer generation resets the payload to its default"
        );
        assert_eq!(gated.unload_generation, 4);
    }

    #[test]
    fn shared_helper_rebuilds_after_an_unload_generation_bump() {
        let _generation_guard = unload_generation_test_lock();
        thread_local! {
            static BUMP_CACHE: RefCell<BoundedRuntimeCache<usize, DropRecorder>> =
                RefCell::new(BoundedRuntimeCache::new());
        }
        let counter = std::rc::Rc::new(std::cell::Cell::new(0usize));

        let build_calls = std::cell::Cell::new(0usize);
        let touch = |key: usize| {
            let counter = counter.clone();
            let build_calls = &build_calls;
            with_thread_local_cached_mut_by_key(
                &BUMP_CACHE,
                key,
                DEFAULT_RUNTIME_CACHE_CAPACITY,
                move || {
                    build_calls.set(build_calls.get() + 1);
                    Ok::<_, &'static str>(DropRecorder::new(counter))
                },
                |_recorder| Ok::<_, &'static str>(()),
            )
            .expect("cache access");
        };

        touch(0);
        touch(0);
        assert_eq!(build_calls.get(), 1, "same generation reuses the entry");
        assert_eq!(counter.get(), 1);

        bump_unload_generation();
        touch(0);
        assert_eq!(
            build_calls.get(),
            2,
            "a bump makes the next access rebuild instead of reusing"
        );
        assert_eq!(
            counter.get(),
            1,
            "the pre-bump entry was dropped, not retained alongside the rebuild"
        );
    }

    #[test]
    fn bump_unload_generation_advances_the_current_generation() {
        let _generation_guard = unload_generation_test_lock();
        let before = current_unload_generation();
        bump_unload_generation();
        assert!(
            current_unload_generation() > before,
            "bump must move the process-wide unload generation forward"
        );
    }
}
