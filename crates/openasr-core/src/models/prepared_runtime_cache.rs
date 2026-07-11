use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::stage_timing;

fn prepared_runtime_cache_key(runtime_path: &Path) -> PathBuf {
    std::fs::canonicalize(runtime_path).unwrap_or_else(|_| runtime_path.to_path_buf())
}

// One slot per canonicalized runtime path. `None` means "not built yet, or a
// previous build attempt failed and left nothing cached" (matching the
// original retry-on-failure contract: a failed build never poisons the path
// for future callers). The slot's own `Mutex` is held across `build()`, which
// gives single-flight semantics *scoped to this one path* -- see
// `get_or_try_insert_with` for why that scoping matters.
type PreparedRuntimeSlot<T> = Arc<Mutex<Option<Arc<T>>>>;

#[derive(Debug, Clone)]
pub(crate) struct PreparedRuntimeCache<T> {
    slots_by_path: Arc<Mutex<HashMap<PathBuf, PreparedRuntimeSlot<T>>>>,
}

impl<T> Default for PreparedRuntimeCache<T> {
    fn default() -> Self {
        Self {
            slots_by_path: Arc::new(Mutex::new(HashMap::new())),
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
        // Step 1: fetch (or create) this path's slot. The outer map lock is
        // only ever held for this cheap lookup/insert -- never across a build
        // -- so a slow cold load for one runtime path never blocks lookups or
        // builds for a different path sharing this cache (e.g. two families
        // that both route through `BuiltinPreparedRuntimeCache`, or two
        // distinct model packs of the same family).
        let slot = {
            let mut slots = self.slots_by_path.lock().map_err(|_| map_poisoned_lock())?;
            Arc::clone(
                slots
                    .entry(cache_key)
                    .or_insert_with(|| Arc::new(Mutex::new(None))),
            )
        };

        // Step 2: single-flight on this path's slot. Holding the slot's lock
        // across `build()` means a concurrent cold-miss for the *same* path
        // (e.g. the offline and streaming dispatch stacks racing to warm the
        // same shared executor instance's cache on first use) blocks on this
        // lock instead of independently materializing its own duplicate
        // prepared runtime; the loser just observes the winner's result once
        // it acquires the lock, so a distinct-path model pack load never gets
        // built twice, not even transiently.
        let mut slot_guard = slot.lock().map_err(|_| map_poisoned_lock())?;
        if let Some(runtime) = slot_guard.as_ref() {
            return Ok(Arc::clone(runtime));
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
        *slot_guard = Some(Arc::clone(&prepared));
        Ok(prepared)
    }

    /// Drops every cached prepared runtime, releasing the `Arc<T>` this cache
    /// holds. If nothing else is currently borrowing an entry (no in-flight
    /// request holding its own clone), this frees whatever native resources
    /// `T` owns -- mmap, materialized tensors, Metal/CPU graph context -- right
    /// away; otherwise the last outstanding clone's drop frees it once that
    /// request finishes. Used by the idle-unload reaper: a poisoned lock is
    /// swallowed (best-effort eviction, not a request-path operation) rather
    /// than propagated, since a subsequent `get_or_try_insert_with` will just
    /// rebuild on the next real request either way.
    ///
    /// This drops the per-path slots wholesale rather than resetting each
    /// slot's inner `Option` to `None`: any build that is still in flight for
    /// a slot at the moment `clear()` runs holds its own `Arc` clone of that
    /// slot (taken before `clear()` could remove it from the map), so it
    /// still completes and populates the slot it is holding -- that slot is
    /// just no longer reachable from the map, so the next `get_or_try_insert_with`
    /// call for that path creates a fresh slot and rebuilds, which is the same
    /// "pay the cold cost again" contract `clear()` has always documented.
    pub(crate) fn clear(&self) {
        if let Ok(mut slots) = self.slots_by_path.lock() {
            slots.clear();
        }
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

    #[test]
    fn clear_evicts_cached_entry_so_the_next_call_rebuilds() {
        let cache = PreparedRuntimeCache::<StubRuntime>::default();
        let path = Path::new("/tmp/test-runtime-clear.gguf");
        let build_count = Cell::new(0usize);

        let build = |value: usize| {
            build_count.set(build_count.get() + 1);
            Ok::<_, &'static str>(StubRuntime { value })
        };

        let runtime_a = cache
            .get_or_try_insert_with(path, || build(7), || "poisoned")
            .expect("runtime a");
        assert_eq!(build_count.get(), 1);

        cache.clear();

        let runtime_b = cache
            .get_or_try_insert_with(path, || build(9), || "poisoned")
            .expect("runtime b");

        assert_eq!(build_count.get(), 2, "clear must force a rebuild");
        assert!(!Arc::ptr_eq(&runtime_a, &runtime_b));
        assert_eq!(runtime_b.value, 9);
    }

    /// Proves the single-flight fix (see `get_or_try_insert_with`): two
    /// threads racing a cold miss on the *same* path must not both run
    /// `build()` -- the second thread has to block on the first thread's
    /// slot lock and observe its result, not materialize its own copy. This
    /// is exactly the "offline dispatch and streaming dispatch both warm the
    /// same shared executor's cache at once" race the shared-executor Phase 1
    /// change makes newly reachable (previously each dispatch had its own
    /// separate cache, so there was no shared slot to race on).
    #[test]
    fn concurrent_cold_miss_on_the_same_path_builds_exactly_once() {
        use std::sync::Barrier;
        use std::thread;

        let cache = Arc::new(PreparedRuntimeCache::<StubRuntime>::default());
        let path = Path::new("/tmp/test-runtime-concurrent.gguf");
        let build_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(2));

        let spawn_racer = |value: usize| {
            let cache = Arc::clone(&cache);
            let build_count = Arc::clone(&build_count);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                cache
                    .get_or_try_insert_with(
                        path,
                        || {
                            build_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            // Give the other thread a chance to reach the slot
                            // lock while this build is "in flight" -- if the
                            // fix regressed to build-outside-lock, it would
                            // race in here concurrently instead of blocking.
                            thread::sleep(std::time::Duration::from_millis(20));
                            Ok::<_, &'static str>(StubRuntime { value })
                        },
                        || "poisoned",
                    )
                    .expect("runtime")
            })
        };

        let racer_a = spawn_racer(1);
        let racer_b = spawn_racer(2);
        let runtime_a = racer_a.join().expect("racer a joined");
        let runtime_b = racer_b.join().expect("racer b joined");

        assert_eq!(
            build_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "single-flight must build exactly once for a concurrent same-path miss"
        );
        assert!(Arc::ptr_eq(&runtime_a, &runtime_b));
    }
}
