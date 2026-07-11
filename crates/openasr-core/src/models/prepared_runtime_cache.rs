use std::any::Any;
use std::collections::HashMap;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::stage_timing;

fn prepared_runtime_cache_key(runtime_path: &Path) -> PathBuf {
    std::fs::canonicalize(runtime_path).unwrap_or_else(|_| runtime_path.to_path_buf())
}

// One slot per canonicalized runtime path. `None` means "not built yet, a
// previous build attempt returned `Err` and left nothing cached, or a
// previous build attempt *panicked*" -- all three are retryable and leave the
// slot's `Mutex` unpoisoned (matching the original retry-on-failure contract:
// no failed build, whether by typed error or panic, poisons the path for
// future callers). `get_or_try_insert_with` is what makes the panic case
// retryable too: it runs `build()` behind `catch_unwind` so a panic never
// unwinds through the held `MutexGuard` (which is what would poison the
// `Mutex`), and never writes anything into the slot. The slot's own `Mutex`
// is held across `build()`, which gives single-flight semantics *scoped to
// this one path* -- see `get_or_try_insert_with` for why that scoping matters.
type PreparedRuntimeSlot<T> = Arc<Mutex<Option<Arc<T>>>>;

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
        //
        // `build()` runs behind `catch_unwind` rather than being called
        // directly: this slot's `MutexGuard` (`slot_guard`) is held across the
        // call, and a `Mutex` is poisoned when a guard is dropped *while the
        // thread is unwinding from a panic*. Left uncaught, a single panicking
        // build would permanently wedge this one runtime path -- every future
        // caller would get a poisoned-lock error instead of a clean retry,
        // which is a strictly worse failure mode than the pre-single-flight
        // behavior (where `build()` ran outside any lock, so a panic there
        // never poisoned anything). `catch_unwind` fully absorbs the panic
        // before this function returns, so by the time `slot_guard` actually
        // drops the thread is no longer unwinding and the `Mutex` stays
        // unpoisoned -- restoring the original "a failed build attempt never
        // poisons the path for future callers" contract for panics, not just
        // typed `Err`s. `AssertUnwindSafe` is sound here because `build()` is
        // a pure host materialization closure that never touches this cache's
        // own state (no callback into `slots_by_path`, `slot`, or any other
        // shared mutable state reachable from here) -- if it panics partway,
        // whatever partial state existed lived entirely on `build()`'s own
        // stack and is simply dropped with it; nothing left reachable through
        // `self` or `slot_guard` can observe a half-built value.
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
                *slot_guard = Some(Arc::clone(&prepared));
                Ok(prepared)
            }
            Err(panic_payload) => {
                // Deliberately do not write to `*slot_guard` (it stays
                // `None`, so the next caller for this path retries a clean
                // build) and do not resume the unwind (that would just move
                // the "which thread crashes" problem around -- for the
                // offline path that thread is a `spawn_blocking` worker,
                // which tokio would report via a `JoinError` on the awaiting
                // task, not crash the process, but callers of this cache
                // expect a typed `Result`, not a propagated panic).
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

    /// Proves the SF-1 fix (see `get_or_try_insert_with`'s panic-handling
    /// comment above `catch_unwind`): before that fix, a `build()` panic
    /// unwound through the held slot `MutexGuard` and poisoned its `Mutex`,
    /// wedging this path so every future caller failed the same way until
    /// `clear()` ran (or forever, under `idle_unload=never`). Against the
    /// pre-fix code (`Arc::new(build()?)` with no `catch_unwind`) this panic
    /// propagates straight out of `get_or_try_insert_with` and aborts the
    /// test on the first call; against the fixed code both calls below
    /// succeed as documented.
    #[test]
    fn build_panic_does_not_poison_the_slot_for_the_next_caller() {
        let cache = PreparedRuntimeCache::<StubRuntime>::default();
        let path = Path::new("/tmp/test-runtime-panic.gguf");

        // The panic below is deliberate and is caught internally by
        // `get_or_try_insert_with`'s own `catch_unwind` (that catch is
        // exactly what this test proves happens); silence the default panic
        // hook's stderr noise for it so the test binary's output does not
        // read like a real crash, then restore the previous hook so other
        // tests keep normal panic reporting.
        let previous_hook = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let first_result = cache.get_or_try_insert_with(
            path,
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

        // The real assertion: the slot must not be poisoned by the panic
        // above, so the very next call for the SAME path builds cleanly
        // instead of inheriting a permanent poisoned-lock failure.
        let second_result = cache
            .get_or_try_insert_with(
                path,
                || Ok::<_, &'static str>(StubRuntime { value: 42 }),
                || "poisoned",
            )
            .expect("build must succeed cleanly on retry after a prior build panic");
        assert_eq!(second_result.value, 42);
    }

    /// Proves `clear()` cannot be "undone" by a build that was already in
    /// flight when it ran (see `clear()`'s doc comment on this exact
    /// scenario): the in-flight winner still completes normally -- a
    /// concurrent `clear()` must not corrupt or block it -- but its result is
    /// orphaned once `clear()` removes the slot from the map, so the next
    /// request for the same path rebuilds from scratch rather than somehow
    /// resurrecting the orphaned build. Production is never actually exposed
    /// to this race (the activity-gate reaper cannot call `clear()` while a
    /// request is still in flight, see the shared-executor review), but the
    /// cache itself should be correct independent of that caller-side
    /// guarantee.
    #[test]
    fn clear_during_in_flight_build_does_not_resurrect_the_evicted_slot() {
        use std::sync::Barrier;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let cache = Arc::new(PreparedRuntimeCache::<StubRuntime>::default());
        let path = Path::new("/tmp/test-runtime-clear-in-flight.gguf");
        let build_count = Arc::new(AtomicUsize::new(0));
        // Synchronizes "the builder thread is inside build(), holding the
        // slot lock" with "the main thread's clear() call", so the race this
        // test targets (clear() concurrent with an in-flight build for the
        // same path) is reliably exercised rather than left to chance thread
        // scheduling.
        let builder_in_build = Arc::new(Barrier::new(2));

        let builder = {
            let cache = Arc::clone(&cache);
            let build_count = Arc::clone(&build_count);
            let barrier = Arc::clone(&builder_in_build);
            thread::spawn(move || {
                cache
                    .get_or_try_insert_with(
                        path,
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
        // The builder is now inside build(), holding only the slot's own
        // lock (see get_or_try_insert_with's step 1/2 split -- the outer map
        // lock is never held across build()), so this clear() proceeds
        // immediately instead of blocking on the in-flight build.
        cache.clear();

        let winner_runtime = builder.join().expect("builder thread joined");
        assert_eq!(build_count.load(Ordering::SeqCst), 1);

        // clear() already dropped the map's only reference to the winner's
        // slot before the winner finished, so the cache has no path back to
        // that Arc anymore. A fresh request for the same path must rebuild
        // from scratch (new slot, new Arc, no double-write into a stale
        // slot) rather than somehow observing the orphaned in-flight build.
        let rebuilt_runtime = cache
            .get_or_try_insert_with(
                path,
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
