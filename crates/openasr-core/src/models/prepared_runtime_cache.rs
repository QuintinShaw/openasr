use std::any::Any;
use std::collections::HashMap;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::models::runtime_cache_coordinator::{
    RuntimeCacheCoordinator, is_cacheable_pack_content_id, pack_content_id_for_runtime_path,
};
use crate::stage_timing;

// One slot per pack content id. The slot also records the coordinator epoch at
// which the cached value was built; a generation mismatch is treated as a miss
// so idle unload / owner shutdown invalidate without relying solely on
// `clear()` emptying the map. Path alone is never a key -- same-path byte
// replacement resolves a different `pack_content_id` and must miss.
//
// `value = None` means "not built yet, a previous build attempt returned `Err`
// and left nothing cached, or a previous build attempt *panicked*" -- all three
// are retryable and leave the slot's `Mutex` unpoisoned (matching the original
// retry-on-failure contract). `get_or_try_insert_with` runs `build()` behind
// `catch_unwind` so a panic never unwinds through the held `MutexGuard`.
// Single-flight is scoped to one content id -- see `get_or_try_insert_with`.
//
// Unreadable packs skip the map entirely (one-shot uncached build) rather than
// inserting an `unreadable:*` token that would poison or falsely collide later.
struct PreparedRuntimeSlotInner<T> {
    generation: u64,
    value: Option<Arc<T>>,
}

impl<T> std::fmt::Debug for PreparedRuntimeSlotInner<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedRuntimeSlotInner")
            .field("generation", &self.generation)
            .field("value", &self.value.as_ref().map(|_| "<cached>"))
            .finish()
    }
}

type PreparedRuntimeSlot<T> = Arc<Mutex<PreparedRuntimeSlotInner<T>>>;

/// Best-effort human-readable panic message for logging. `std::panic` payloads
/// are `Box<dyn Any + Send>`; the standard library's own default panic hook
/// only special-cases `&str` and `String`, so that is what is worth matching
/// here too -- anything else (a custom payload type) just gets a placeholder,
/// which is fine since this is diagnostic-only and never part of the typed
/// error returned to callers.
fn describe_panic_payload(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedRuntimeCache<T> {
    slots_by_content_id: Arc<Mutex<HashMap<String, PreparedRuntimeSlot<T>>>>,
}

impl<T> Default for PreparedRuntimeCache<T> {
    fn default() -> Self {
        Self {
            slots_by_content_id: Arc::new(Mutex::new(HashMap::new())),
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
        let pack_content_id = pack_content_id_for_runtime_path(runtime_path);
        if !is_cacheable_pack_content_id(&pack_content_id) {
            // Fail closed on insert: unreadable / non-cacheable content ids never
            // enter the shared map. Still honor the caller's request with a
            // one-shot uncached build so a transient unreadable path does not
            // wedge the request path behind a permanent "unreadable" slot.
            return Self::build_once_uncached(runtime_path, build, map_poisoned_lock);
        }
        let epoch = RuntimeCacheCoordinator::global().epoch();

        // Step 1: fetch (or create) this content id's slot. The outer map lock is
        // only ever held for this cheap lookup/insert -- never across a build
        // -- so a slow cold load for one runtime identity never blocks lookups
        // or builds for a different identity sharing this cache (e.g. two
        // families that both route through `BuiltinPreparedRuntimeCache`, or two
        // distinct model packs of the same family).
        let slot = {
            let mut slots = self
                .slots_by_content_id
                .lock()
                .map_err(|_| map_poisoned_lock())?;
            Arc::clone(slots.entry(pack_content_id).or_insert_with(|| {
                Arc::new(Mutex::new(PreparedRuntimeSlotInner {
                    generation: epoch,
                    value: None,
                }))
            }))
        };

        // Step 2: single-flight on this content id's slot. Holding the slot's
        // lock across `build()` means a concurrent cold-miss for the *same*
        // content (e.g. the offline and streaming dispatch stacks racing to warm
        // the same shared executor instance's cache on first use) blocks on this
        // lock instead of independently materializing its own duplicate prepared
        // runtime; the loser just observes the winner's result once it acquires
        // the lock.
        let mut slot_guard = slot.lock().map_err(|_| map_poisoned_lock())?;
        // Re-read epoch under the slot lock so a concurrent invalidate between
        // the outer lookup and here cannot hand out a just-invalidated value.
        let epoch = RuntimeCacheCoordinator::global().epoch();
        if slot_guard.generation == epoch
            && let Some(runtime) = slot_guard.value.as_ref()
        {
            return Ok(Arc::clone(runtime));
        }
        // Stale generation: drop the old value before rebuilding so unload
        // without `clear()` still frees the previous Arc once this rebuild
        // finishes (and any external clones drop).
        if slot_guard.generation != epoch {
            slot_guard.value = None;
            slot_guard.generation = epoch;
        }

        // Model pack loading (mmap + tensor materialization + context/graph
        // construction, up to inference-ready) happens exactly here, exactly
        // once per distinct content identity per process epoch (subsequent calls
        // hit the cache check above). This one call site covers every builtin
        // model family that goes through this cache, so it is the single place
        // to time "how long did loading this pack take" without instrumenting
        // each family's build function separately.
        //
        // `build()` runs behind `catch_unwind` rather than being called
        // directly: this slot's `MutexGuard` (`slot_guard`) is held across the
        // call, and a `Mutex` is poisoned when a guard is dropped *while the
        // thread is unwinding from a panic*. Left uncaught, a single panicking
        // build would permanently wedge this one runtime identity -- every
        // future caller would get a poisoned-lock error instead of a clean
        // retry. `catch_unwind` fully absorbs the panic before this function
        // returns, so by the time `slot_guard` actually drops the thread is no
        // longer unwinding and the `Mutex` stays unpoisoned. `AssertUnwindSafe`
        // is sound here because `build()` is a pure host materialization
        // closure that never touches this cache's own state.
        let load_started = Instant::now();
        match panic::catch_unwind(AssertUnwindSafe(build)) {
            Ok(result) => {
                let prepared = Arc::new(result?);
                stage_timing::log_event(
                    "model_pack_load",
                    format_args!(
                        "path={} duration_ms={:.3}",
                        runtime_path.display(),
                        load_started.elapsed().as_secs_f64() * 1000.0
                    ),
                );
                slot_guard.generation = epoch;
                slot_guard.value = Some(Arc::clone(&prepared));
                Ok(prepared)
            }
            Err(panic_payload) => {
                // Deliberately do not write a value (slot stays empty for this
                // generation so the next caller retries a clean build) and do
                // not resume the unwind.
                stage_timing::log_event(
                    "model_pack_load_panicked",
                    format_args!(
                        "path={} duration_ms={:.3} message={}",
                        runtime_path.display(),
                        load_started.elapsed().as_secs_f64() * 1000.0,
                        describe_panic_payload(panic_payload.as_ref())
                    ),
                );
                Err(map_poisoned_lock())
            }
        }
    }

    fn build_once_uncached<E, F, M>(
        runtime_path: &Path,
        build: F,
        map_poisoned_lock: M,
    ) -> Result<Arc<T>, E>
    where
        F: FnOnce() -> Result<T, E>,
        M: Fn() -> E,
    {
        let load_started = Instant::now();
        match panic::catch_unwind(AssertUnwindSafe(build)) {
            Ok(result) => {
                let prepared = Arc::new(result?);
                stage_timing::log_event(
                    "model_pack_load",
                    format_args!(
                        "path={} duration_ms={:.3} cache=skip_uncacheable_content_id",
                        runtime_path.display(),
                        load_started.elapsed().as_secs_f64() * 1000.0
                    ),
                );
                Ok(prepared)
            }
            Err(panic_payload) => {
                stage_timing::log_event(
                    "model_pack_load_panicked",
                    format_args!(
                        "path={} duration_ms={:.3} message={} cache=skip_uncacheable_content_id",
                        runtime_path.display(),
                        load_started.elapsed().as_secs_f64() * 1000.0,
                        describe_panic_payload(panic_payload.as_ref())
                    ),
                );
                Err(map_poisoned_lock())
            }
        }
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
    /// This drops the per-content slots wholesale rather than resetting each
    /// slot's inner value to `None`: any build that is still in flight for a
    /// slot at the moment `clear()` runs holds its own `Arc` clone of that
    /// slot (taken before `clear()` could remove it from the map), so it still
    /// completes and populates the slot it is holding -- that slot is just no
    /// longer reachable from the map, so the next `get_or_try_insert_with` call
    /// for that content id creates a fresh slot and rebuilds, which is the same
    /// "pay the cold cost again" contract `clear()` has always documented.
    pub(crate) fn clear(&self) {
        if let Ok(mut slots) = self.slots_by_content_id.lock() {
            slots.clear();
        }
    }

    #[cfg(test)]
    fn len_for_test(&self) -> usize {
        self.slots_by_content_id
            .lock()
            .map(|guard| guard.len())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::runtime_cache_coordinator::bump_runtime_build_generation;
    use std::cell::Cell;
    use std::path::PathBuf;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct StubRuntime {
        value: usize,
    }

    fn write_pack(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).expect("write pack");
        path
    }

    #[test]
    fn reuses_cached_runtime_for_same_content() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_pack(&dir, "runtime.oasr", b"same-content");
        let cache = PreparedRuntimeCache::<StubRuntime>::default();

        let runtime_a = cache
            .get_or_try_insert_with(
                &path,
                || Ok::<_, &'static str>(StubRuntime { value: 7 }),
                || "poisoned",
            )
            .expect("runtime a");
        let runtime_b = cache
            .get_or_try_insert_with(
                &path,
                || Ok::<_, &'static str>(StubRuntime { value: 9 }),
                || "poisoned",
            )
            .expect("runtime b");

        assert!(Arc::ptr_eq(&runtime_a, &runtime_b));
        assert_eq!(runtime_b.value, 7);
        assert_eq!(cache.len_for_test(), 1);
    }

    #[test]
    fn reuses_cached_runtime_for_canonical_equivalent_paths() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("runtime.gguf");
        std::fs::write(&runtime_path, b"GGUF-bytes").expect("write runtime");
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
    fn same_path_byte_replacement_misses_cached_runtime() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_pack(&dir, "same-path.oasr", b"content-a-bytes");
        let cache = PreparedRuntimeCache::<StubRuntime>::default();
        let build_count = Cell::new(0usize);

        let runtime_a = cache
            .get_or_try_insert_with(
                &path,
                || {
                    build_count.set(build_count.get() + 1);
                    Ok::<_, &'static str>(StubRuntime { value: 1 })
                },
                || "poisoned",
            )
            .expect("runtime a");
        assert_eq!(build_count.get(), 1);
        assert_eq!(cache.len_for_test(), 1);

        std::fs::write(&path, b"content-b-bytes-different").expect("replace bytes");
        let runtime_b = cache
            .get_or_try_insert_with(
                &path,
                || {
                    build_count.set(build_count.get() + 1);
                    Ok::<_, &'static str>(StubRuntime { value: 2 })
                },
                || "poisoned",
            )
            .expect("runtime b");

        assert_eq!(
            build_count.get(),
            2,
            "same path with different pack bytes must rebuild"
        );
        assert!(!Arc::ptr_eq(&runtime_a, &runtime_b));
        assert_eq!(runtime_b.value, 2);
        // Both content identities may remain until clear; that is intentional --
        // content A is still valid if referenced elsewhere.
        assert!(cache.len_for_test() >= 1);
    }

    #[test]
    fn epoch_bump_misses_even_when_bytes_unchanged() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_pack(&dir, "stable.oasr", b"stable-bytes");
        let cache = PreparedRuntimeCache::<StubRuntime>::default();
        let build_count = Cell::new(0usize);

        let runtime_a = cache
            .get_or_try_insert_with(
                &path,
                || {
                    build_count.set(build_count.get() + 1);
                    Ok::<_, &'static str>(StubRuntime { value: 1 })
                },
                || "poisoned",
            )
            .expect("runtime a");
        let _ = bump_runtime_build_generation();
        let runtime_b = cache
            .get_or_try_insert_with(
                &path,
                || {
                    build_count.set(build_count.get() + 1);
                    Ok::<_, &'static str>(StubRuntime { value: 2 })
                },
                || "poisoned",
            )
            .expect("runtime b");

        assert_eq!(build_count.get(), 2);
        assert!(!Arc::ptr_eq(&runtime_a, &runtime_b));
        assert_eq!(runtime_b.value, 2);
    }

    #[test]
    fn unreadable_pack_is_not_inserted() {
        let cache = PreparedRuntimeCache::<StubRuntime>::default();
        let missing = Path::new("/tmp/openasr-definitely-missing-prepared-runtime.oasr");
        let build_count = Cell::new(0usize);

        let runtime_a = cache
            .get_or_try_insert_with(
                missing,
                || {
                    build_count.set(build_count.get() + 1);
                    Ok::<_, &'static str>(StubRuntime { value: 3 })
                },
                || "poisoned",
            )
            .expect("one-shot a");
        let runtime_b = cache
            .get_or_try_insert_with(
                missing,
                || {
                    build_count.set(build_count.get() + 1);
                    Ok::<_, &'static str>(StubRuntime { value: 4 })
                },
                || "poisoned",
            )
            .expect("one-shot b");

        assert_eq!(build_count.get(), 2, "unreadable must not cache");
        assert!(!Arc::ptr_eq(&runtime_a, &runtime_b));
        assert_eq!(cache.len_for_test(), 0);
        assert_eq!(runtime_a.value, 3);
        assert_eq!(runtime_b.value, 4);
    }

    #[test]
    fn clear_evicts_cached_entry_so_the_next_call_rebuilds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_pack(&dir, "clear.oasr", b"clear-bytes");
        let cache = PreparedRuntimeCache::<StubRuntime>::default();
        let build_count = Cell::new(0usize);

        let build = |value: usize| {
            build_count.set(build_count.get() + 1);
            Ok::<_, &'static str>(StubRuntime { value })
        };

        let runtime_a = cache
            .get_or_try_insert_with(&path, || build(7), || "poisoned")
            .expect("runtime a");
        assert_eq!(build_count.get(), 1);

        cache.clear();

        let runtime_b = cache
            .get_or_try_insert_with(&path, || build(9), || "poisoned")
            .expect("runtime b");

        assert_eq!(build_count.get(), 2, "clear must force a rebuild");
        assert!(!Arc::ptr_eq(&runtime_a, &runtime_b));
        assert_eq!(runtime_b.value, 9);
    }

    /// Proves the single-flight fix (see `get_or_try_insert_with`): two
    /// threads racing a cold miss on the *same* content identity must not both
    /// run `build()` while the coordinator epoch stays stable.
    #[test]
    fn concurrent_cold_miss_on_the_same_content_builds_exactly_once() {
        use std::sync::Barrier;
        use std::thread;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_pack(&dir, "concurrent.oasr", b"concurrent-bytes");

        // Retry when a parallel test bumps the process-global epoch mid-race:
        // that forces a legitimate rebuild and is not a single-flight regression.
        for attempt in 0..12 {
            let cache = Arc::new(PreparedRuntimeCache::<StubRuntime>::default());
            let build_epochs = Arc::new(Mutex::new(Vec::<u64>::new()));
            let barrier = Arc::new(Barrier::new(2));
            let epoch_at_start = RuntimeCacheCoordinator::global().epoch();

            let spawn_racer = |value: usize| {
                let cache = Arc::clone(&cache);
                let build_epochs = Arc::clone(&build_epochs);
                let barrier = Arc::clone(&barrier);
                let path = path.clone();
                thread::spawn(move || {
                    barrier.wait();
                    cache
                        .get_or_try_insert_with(
                            &path,
                            || {
                                let epoch = RuntimeCacheCoordinator::global().epoch();
                                build_epochs.lock().expect("build epoch lock").push(epoch);
                                thread::sleep(std::time::Duration::from_millis(30));
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
            let epochs = build_epochs.lock().expect("build epoch lock").clone();
            let epoch_at_end = RuntimeCacheCoordinator::global().epoch();

            if epochs.len() == 1 {
                assert!(
                    Arc::ptr_eq(&runtime_a, &runtime_b),
                    "attempt {attempt}: single build must be shared"
                );
                return;
            }

            if epoch_at_start != epoch_at_end || epochs.iter().any(|e| *e != epoch_at_start) {
                // Coordinator epoch moved during the race -- rebuild is correct.
                continue;
            }

            panic!(
                "attempt {attempt}: single-flight broke under stable epoch {epoch_at_start}; build_epochs={epochs:?}"
            );
        }

        // Extremely unlucky epoch churn from parallel tests; do not fail the suite.
        eprintln!(
            "concurrent single-flight test skipped after retries due to process-global epoch churn"
        );
    }

    /// Proves a `build()` panic does not poison the slot `Mutex` for the next
    /// caller on the same content id.
    #[test]
    fn build_panic_does_not_poison_the_slot_for_the_next_caller() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_pack(&dir, "panic.oasr", b"panic-bytes");
        let cache = PreparedRuntimeCache::<StubRuntime>::default();

        let previous_hook = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let first_result = cache.get_or_try_insert_with(
            &path,
            || -> Result<StubRuntime, &'static str> { panic!("simulated build panic") },
            || "poisoned",
        );
        panic::set_hook(previous_hook);

        assert_eq!(
            first_result,
            Err("poisoned"),
            "a build() panic must be caught and mapped through map_poisoned_lock, not left \
             to unwind out of get_or_try_insert_with"
        );

        let second_result = cache
            .get_or_try_insert_with(
                &path,
                || Ok::<_, &'static str>(StubRuntime { value: 42 }),
                || "poisoned",
            )
            .expect("build must succeed cleanly on retry after a prior build panic");
        assert_eq!(second_result.value, 42);
    }

    /// Proves `clear()` cannot be "undone" by a build that was already in
    /// flight when it ran: the in-flight winner still completes normally, but
    /// its result is orphaned once `clear()` removes the slot from the map.
    #[test]
    fn clear_during_in_flight_build_does_not_resurrect_the_evicted_slot() {
        use std::sync::Barrier;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_pack(&dir, "clear-in-flight.oasr", b"clear-in-flight-bytes");
        let cache = Arc::new(PreparedRuntimeCache::<StubRuntime>::default());
        let build_count = Arc::new(AtomicUsize::new(0));
        let builder_in_build = Arc::new(Barrier::new(2));

        let builder = {
            let cache = Arc::clone(&cache);
            let build_count = Arc::clone(&build_count);
            let barrier = Arc::clone(&builder_in_build);
            let path = path.clone();
            thread::spawn(move || {
                cache
                    .get_or_try_insert_with(
                        &path,
                        || {
                            build_count.fetch_add(1, Ordering::SeqCst);
                            barrier.wait();
                            thread::sleep(std::time::Duration::from_millis(50));
                            Ok::<_, &'static str>(StubRuntime { value: 1 })
                        },
                        || "poisoned",
                    )
                    .expect(
                        "in-flight build must still complete normally despite a concurrent clear()",
                    )
            })
        };

        builder_in_build.wait();
        cache.clear();

        let winner_runtime = builder.join().expect("builder thread joined");
        assert_eq!(build_count.load(Ordering::SeqCst), 1);

        let rebuilt_runtime = cache
            .get_or_try_insert_with(
                &path,
                || {
                    build_count.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, &'static str>(StubRuntime { value: 2 })
                },
                || "poisoned",
            )
            .expect("rebuild after clear must succeed");

        assert_eq!(
            build_count.load(Ordering::SeqCst),
            2,
            "clear() during an in-flight build must force the next caller to rebuild, not \
             reuse the orphaned slot"
        );
        assert!(
            !Arc::ptr_eq(&winner_runtime, &rebuilt_runtime),
            "the post-clear rebuild must be a distinct Arc from the orphaned in-flight build's result"
        );
        assert_eq!(rebuilt_runtime.value, 2);
    }
}
