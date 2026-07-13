//! Process-wide native-request/session activity tracking, plus the
//! `idle_unload` background reaper.
//!
//! Two independent things must never race the resident model teardown:
//! an in-flight HTTP transcription/translation (or the realtime
//! per-utterance backend-job fallback -- both go through
//! `transcribe_with_runtime`), and an attached realtime native-streaming
//! session (`realtime::native_streaming_worker_for_key` /
//! `spawn_native_streaming_worker`'s acquire/release). Both call
//! [`native_activity_enter`]/[`native_activity_exit`] in lockstep with their
//! own request/session lifetime; the reaper only ever unloads the cached
//! native model runtime after the tracker has read zero active for at least
//! the configured threshold.

use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

struct NativeActivityTracker {
    active: AtomicUsize,
    idle_since: Mutex<Instant>,
}

impl NativeActivityTracker {
    fn new() -> Self {
        Self {
            active: AtomicUsize::new(0),
            idle_since: Mutex::new(Instant::now()),
        }
    }

    fn enter(&self) {
        self.active.fetch_add(1, Ordering::AcqRel);
    }

    fn exit(&self) {
        // Saturate at zero instead of a plain `fetch_sub`: a bare `fetch_sub`
        // on an already-zero counter wraps to `usize::MAX` (atomics use
        // wrapping arithmetic unconditionally, in debug and release alike),
        // which would pin `is_idle_for` to "never idle" -- silently and
        // permanently disabling `idle_unload` -- for the rest of the
        // process's life. `fetch_update` lets us clamp the new value while
        // still reading the pre-update count to detect the mismatch.
        let previous = self
            .active
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                Some(current.saturating_sub(1))
            })
            .expect("closure always returns Some, so fetch_update never returns Err");
        if previous == 0 {
            // An enter/exit accounting bug (a double-release, or an exit
            // whose matching enter never landed). Catch it loudly in debug
            // builds where a panic is safe and useful to a developer -- but
            // never panic a running release daemon over an internal
            // bookkeeping mismatch; a live server killed by this would be
            // far worse than idle_unload staying disabled (or, worst case,
            // firing while a session is still attached) until the next
            // balanced enter/exit. Log it (release and debug both) so an
            // operator/maintainer can still notice and file it.
            debug_assert!(
                false,
                "native activity exited more times than it was entered"
            );
            eprintln!(
                "openasr-server: native activity accounting underflow: exit() called with no \
                 matching enter() (idle_unload may misbehave until the next balanced enter/exit)"
            );
            return;
        }
        if previous == 1 {
            *self
                .idle_since
                .lock()
                .expect("native activity idle mutex poisoned") = Instant::now();
        }
    }

    fn is_idle_for(&self, now: Instant, idle_for: Duration) -> bool {
        if self.active.load(Ordering::Acquire) != 0 {
            return false;
        }
        let idle_since = *self
            .idle_since
            .lock()
            .expect("native activity idle mutex poisoned");
        now.checked_duration_since(idle_since).unwrap_or_default() >= idle_for
    }

    /// Current active native request/session count. Debug-observability
    /// getter (surfaced by `/health` as `native_active_count`); the reaper
    /// itself only ever needs [`Self::is_idle_for`]'s bool.
    fn active_count(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }

    /// Seconds elapsed since the active count last returned to zero, as of
    /// `now`. `0` while any activity is active -- there is no meaningful
    /// "idle duration" mid-request, and `0` (rather than some stale prior
    /// idle window) is the least surprising reading for a client polling
    /// `/health` moment to moment. Debug-observability getter (surfaced by
    /// `/health` as `idle_seconds`), expressed with the same `idle_since`
    /// field `is_idle_for` already reads, so the two can never disagree
    /// about what "idle" means.
    fn idle_seconds(&self, now: Instant) -> u64 {
        if self.active.load(Ordering::Acquire) != 0 {
            return 0;
        }
        let idle_since = *self
            .idle_since
            .lock()
            .expect("native activity idle mutex poisoned");
        now.checked_duration_since(idle_since)
            .unwrap_or_default()
            .as_secs()
    }
}

static NATIVE_ACTIVITY: OnceLock<NativeActivityTracker> = OnceLock::new();

fn native_activity() -> &'static NativeActivityTracker {
    NATIVE_ACTIVITY.get_or_init(NativeActivityTracker::new)
}

/// Process-wide count of successful `unload_idle_native_model_runtime_caches`
/// calls, bumped once per eviction by [`bump_native_unload_generation`]
/// (currently only [`spawn_idle_unload_reaper`]'s loop). A realtime-streaming
/// decode worker's OS thread survives an `idle_unload` eviction -- it only
/// tears down much later, at the separate hard-release threshold -- so its
/// thread-local "have I warmed this thread" state cannot be a bare bool: that
/// would keep reading "warmed" after the resident runtime it warmed has
/// already been evicted, skipping re-warm and pushing the cold rebuild onto
/// the first real decode of the next attach instead. The warm-up gate in
/// `native_worker.rs` instead records the generation it warmed at and
/// re-warms whenever the current generation has moved on.
static NATIVE_UNLOAD_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Current unload generation. `Relaxed` is sufficient: this is a coarse
/// "has an unload happened since I last checked" signal, not a coordination
/// primitive, and it is never combined with a specific unload's Ordering.
pub(crate) fn native_unload_generation() -> u64 {
    NATIVE_UNLOAD_GENERATION.load(Ordering::Relaxed)
}

/// Marks one `unload_idle_native_model_runtime_caches` eviction. Exposed
/// beyond [`spawn_idle_unload_reaper`]'s own use so tests can simulate an
/// idle-unload deterministically instead of waiting on the reaper's poll
/// interval.
pub(crate) fn bump_native_unload_generation() {
    NATIVE_UNLOAD_GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// Process-wide unload generation as of the most recent successful native
/// model warm state transition (an offline decode or a realtime streaming
/// warm-up that actually built or reused the resident runtime). Compared
/// against [`native_unload_generation`], this is the source of truth for
/// `/health`'s `model_resident` field: the two read equal only when no
/// `idle_unload` eviction has happened since that last successful load.
///
/// `u64::MAX` is the "never warmed yet" sentinel -- unreachable via the
/// generation counter's own increments in a running process -- so a fresh
/// boot reads as not-resident until the first successful load completes,
/// same as after a real eviction.
static LAST_WARM_GENERATION: AtomicU64 = AtomicU64::new(u64::MAX);

/// Records that the native model runtime is resident as of right now, at the
/// current unload generation. Call sites: [`crate::realtime::native_worker`]'s
/// streaming warm-up gate and the offline-decode path in
/// `routes::transcription`, both right after their respective build/decode
/// succeeds.
///
/// Safe against a racing `idle_unload` eviction: the reaper only unloads
/// while [`native_activity_is_idle_for`] reads true, i.e. while the active
/// count is zero, and every call site runs from inside an active
/// [`NativeActivityGuard`] (or an attached streaming session's equivalent
/// window) -- so the generation read here cannot be bumped out from under an
/// in-flight caller before its request finishes.
pub(crate) fn mark_native_model_warm() {
    LAST_WARM_GENERATION.store(native_unload_generation(), Ordering::Relaxed);
}

/// Whether the bound native model runtime is resident right now: warmed, and
/// not evicted by an `idle_unload` sweep since. Reads `false` before the
/// first successful load of the process's lifetime, and `false` again after
/// any eviction until the next successful load.
pub(crate) fn native_model_is_resident() -> bool {
    LAST_WARM_GENERATION.load(Ordering::Relaxed) == native_unload_generation()
}

/// Single shared lock serializing every test in this crate that mutates
/// `NATIVE_UNLOAD_GENERATION` (via [`bump_native_unload_generation`]) or
/// `LAST_WARM_GENERATION` (via [`mark_native_model_warm`], including
/// indirectly by exercising a real `warm_up_native_streaming_session_once`
/// success path with a fake session) against each other -- without it, a
/// bump or mark from one test could land between another test's own bump and
/// check under `cargo test`'s default same-process test-thread parallelism.
/// `cargo nextest` isolates each test in its own process and would not need
/// this at all.
///
/// `tokio::sync::Mutex` (not `std::sync::Mutex`) because some holders are
/// async tests that keep the guard alive across an `.await` (e.g. awaiting
/// `attach_and_run_boot_warmup`) -- a `std::sync::MutexGuard` is not `Send`,
/// so it cannot survive an await point on a multi-threaded runtime.
/// [`native_unload_generation_test_lock_blocking`] is the sync-test
/// counterpart, for callers (like this module's own `#[test]`s) that never
/// hold the guard across an await point.
#[cfg(test)]
fn native_unload_generation_test_lock_mutex() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

#[cfg(test)]
pub(crate) async fn native_unload_generation_test_lock() -> tokio::sync::MutexGuard<'static, ()> {
    native_unload_generation_test_lock_mutex().lock().await
}

/// Blocking counterpart of [`native_unload_generation_test_lock`] for plain
/// (non-`tokio::test`) `#[test]`s, which never run inside an async executor
/// -- `Mutex::blocking_lock` would panic if called from one.
#[cfg(test)]
pub(crate) fn native_unload_generation_test_lock_blocking() -> tokio::sync::MutexGuard<'static, ()>
{
    native_unload_generation_test_lock_mutex().blocking_lock()
}

/// Marks one native request/session as started. Must be paired with a later
/// [`native_activity_exit`] -- prefer [`NativeActivityGuard`] over calling
/// this directly when the activity's lifetime is a single lexical scope.
pub(crate) fn native_activity_enter() {
    native_activity().enter();
}

/// Marks one native request/session as finished.
pub(crate) fn native_activity_exit() {
    native_activity().exit();
}

/// Whether the process-wide tracker has read zero active native
/// requests/sessions for at least `idle_for`, as of `now`. Exposed (beyond
/// [`spawn_idle_unload_reaper`]'s own use) so integration tests can assert
/// the real attach/release call sites in `native_worker.rs` actually keep
/// this in lockstep with a real session's lifetime.
pub(crate) fn native_activity_is_idle_for(now: Instant, idle_for: Duration) -> bool {
    native_activity().is_idle_for(now, idle_for)
}

/// Process-wide active native request/session count right now. Debug-only
/// observability, surfaced by `/health` as `native_active_count` so an
/// operator/support session can tell "server looks idle but a session is
/// still counted active" apart from "idle_unload just has not swept yet"
/// without attaching a debugger.
pub(crate) fn native_activity_active_count() -> usize {
    native_activity().active_count()
}

/// Seconds since the process-wide tracker's active count last returned to
/// zero, as of `now` (`0` while any activity is active). Debug-only
/// observability, surfaced by `/health` as `idle_seconds`; expressed via the
/// same tracker state `native_activity_is_idle_for` reads, so the two can
/// never disagree about what "idle" means.
pub(crate) fn native_activity_idle_seconds(now: Instant) -> u64 {
    native_activity().idle_seconds(now)
}

/// RAII pairing of one `native_activity_enter`/`native_activity_exit` call.
/// Used at the offline/backend-job transcribe call site, where the request's
/// activity window is exactly one lexical scope, and at the realtime
/// native-streaming attach path in `native_worker.rs`, where the window spans
/// an async attach attempt and a handoff to a different OS thread: there the
/// guard is constructed once in `native_streaming_worker_for_key`, then
/// travels with the attach attempt as a value (through
/// `NativeStreamingWorkerHandle`, then the `Attach` message) until whichever
/// side ends up owning it when it drops -- the sender, if the attach never
/// reaches the worker thread, or the worker thread, once it finishes
/// processing that session. Moving the guard itself (rather than pairing bare
/// `enter`/`exit` calls by hand across those call sites) is what makes every
/// exit path -- including a failed `send` -- retire the count exactly once.
pub(crate) struct NativeActivityGuard(());

impl NativeActivityGuard {
    pub(crate) fn enter() -> Self {
        native_activity_enter();
        Self(())
    }
}

impl Drop for NativeActivityGuard {
    fn drop(&mut self) {
        native_activity_exit();
    }
}

/// A [`NativeActivityGuard`] shared between the native-streaming decode
/// worker OS thread that normally retires it and an external supervisor that
/// may need to force an early release if that thread never gets the chance
/// to retire it itself -- see `realtime::native_worker`'s decode watchdog
/// (`abandon_stuck_native_streaming_worker`, for a command that does not
/// return within its budget) and same-key preemption
/// (`native_streaming_worker_for_key`, for a new attach that finds the key's
/// worker still occupied by a session whose owning WS connection already
/// disconnected). Both of those triggers exist precisely because the worker
/// OS thread cannot be interrupted mid-decode (a stuck Metal
/// `waitUntilCompleted` cannot be aborted from outside its own thread), so
/// the guard itself must be releasable from outside that thread too.
///
/// [`Self::release`] is idempotent: whichever side -- the worker thread
/// finishing normally, or an external trigger firing first -- calls it
/// first actually drops the inner guard (retiring the process-wide native
/// activity count); every later call, from any clone, is a no-op. This is
/// enforced by taking the guard out of a `Mutex<Option<_>>` rather than
/// relying on `Drop` timing between independent owners racing each other.
/// If neither side ever calls `release()` explicitly (e.g. a worker-message
/// `send` failed before any attach reached the worker thread), the guard
/// still retires normally once every clone has dropped: dropping the last
/// `Arc` drops the `Mutex<Option<NativeActivityGuard>>` it wraps, which -- if
/// the inner guard is still `Some` -- drops that guard too.
#[derive(Clone)]
pub(crate) struct SharedNativeActivityGuard(std::sync::Arc<Mutex<Option<NativeActivityGuard>>>);

impl SharedNativeActivityGuard {
    pub(crate) fn new() -> Self {
        Self(std::sync::Arc::new(Mutex::new(Some(
            NativeActivityGuard::enter(),
        ))))
    }

    /// Idempotent early release -- see the type-level doc comment.
    pub(crate) fn release(&self) {
        let mut guard = self
            .0
            .lock()
            .expect("shared native activity guard mutex poisoned");
        guard.take();
    }
}

/// Spawns the background `idle_unload` reaper. Polls at a fraction of
/// `idle_for` so the actual unload lands within roughly one tick of crossing
/// the threshold, without spinning for a short threshold (the `now` policy's
/// 5s floor) or over-polling for the common 10m/1h thresholds. Callers only
/// spawn this when the resolved policy is not `never` (see
/// `IdleUnloadPolicy::idle_threshold`).
pub(crate) fn spawn_idle_unload_reaper(idle_for: Duration) {
    let poll_interval = (idle_for / 4).max(Duration::from_secs(1));
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(poll_interval).await;
            if native_activity_is_idle_for(Instant::now(), idle_for) {
                openasr_core::unload_idle_native_model_runtime_caches();
                bump_native_unload_generation();
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exercises the tracker logic against a private instance (not the process
    // singleton): the singleton is shared with every other test in this crate
    // that happens to transcribe something concurrently, so asserting on it
    // directly would be flaky by construction.
    #[test]
    fn idle_only_once_active_count_returns_to_zero() {
        let tracker = NativeActivityTracker::new();
        let threshold = Duration::from_secs(1);

        // Immediately after construction, real elapsed time is microseconds,
        // nowhere near the 1s threshold yet.
        assert!(!tracker.is_idle_for(Instant::now(), threshold));

        tracker.enter();
        tracker.enter();
        // While active, must never read idle no matter how far in the future
        // the supplied `now` claims to be.
        let far_future = Instant::now() + Duration::from_secs(10_000);
        assert!(
            !tracker.is_idle_for(far_future, threshold),
            "must never read idle while any activity is active"
        );

        tracker.exit();
        assert!(
            !tracker.is_idle_for(far_future, threshold),
            "one remaining active entry still blocks idle"
        );

        tracker.exit();
        assert!(
            tracker.is_idle_for(far_future, threshold),
            "idle_since resets to the moment the count returns to zero, so a \
             `now` far past that moment must read as idle"
        );
    }

    #[test]
    fn new_activity_after_going_idle_resets_the_idle_clock() {
        let tracker = NativeActivityTracker::new();
        tracker.enter();
        tracker.exit();
        assert!(
            tracker.is_idle_for(
                Instant::now() + Duration::from_secs(3600),
                Duration::from_secs(1)
            ),
            "far enough in the future, the first idle transition must already read as idle"
        );

        // A new request arrives and finishes again: the idle clock must
        // restart from this second transition, not stay pinned to the first.
        // Real elapsed time since this second `exit()` is microseconds, so
        // checking against `Instant::now()` (not a synthetic future instant)
        // with a large threshold must read as NOT idle -- it would only read
        // as idle if `idle_since` were still stuck at the first transition.
        tracker.enter();
        tracker.exit();
        assert!(
            !tracker.is_idle_for(Instant::now(), Duration::from_secs(3600)),
            "idle_since must have been bumped by the second enter/exit pair, not left at the first"
        );
    }

    #[test]
    fn active_count_and_idle_seconds_mirror_is_idle_for() {
        // The `/health` debug-observability getters (`active_count`,
        // `idle_seconds`) must never disagree with the boolean
        // `is_idle_for` the reaper actually acts on -- they read the exact
        // same `active`/`idle_since` state, just shaped differently.
        let tracker = NativeActivityTracker::new();
        assert_eq!(tracker.active_count(), 0);
        assert_eq!(
            tracker.idle_seconds(Instant::now() + Duration::from_secs(3600)),
            3600,
            "idle_seconds must read a real elapsed duration once idle, matching is_idle_for"
        );

        tracker.enter();
        assert_eq!(tracker.active_count(), 1);
        assert_eq!(
            tracker.idle_seconds(Instant::now() + Duration::from_secs(3600)),
            0,
            "idle_seconds must read zero while active, no matter how far `now` is"
        );

        tracker.enter();
        assert_eq!(tracker.active_count(), 2);

        tracker.exit();
        assert_eq!(
            tracker.active_count(),
            1,
            "one remaining active entry still counts as active"
        );
        assert_eq!(tracker.idle_seconds(Instant::now()), 0);

        tracker.exit();
        assert_eq!(tracker.active_count(), 0);
        assert_eq!(
            tracker.idle_seconds(Instant::now()),
            0,
            "real elapsed time just after the count returns to zero is microseconds, not a full second yet"
        );
        assert_eq!(
            tracker.idle_seconds(Instant::now() + Duration::from_secs(90)),
            90
        );
    }

    // Debug-only half of the contract: `debug_assert!` is compiled out under
    // `cargo test --release`, where the release-safe (saturate + log,
    // no panic) behavior applies instead -- see
    // `release_builds_do_not_panic_on_an_accounting_underflow` below.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "exited more times than it was entered")]
    fn exit_without_matching_enter_is_a_bug() {
        let tracker = NativeActivityTracker::new();
        tracker.exit();
    }

    #[test]
    fn extra_exits_saturate_instead_of_wrapping_and_do_not_corrupt_the_counter() {
        // A bare `fetch_sub` on an already-zero counter wraps to
        // `usize::MAX` (atomics use wrapping arithmetic unconditionally,
        // debug or release), which would pin `is_idle_for` to "never idle"
        // forever -- silently disabling `idle_unload` for the rest of the
        // process's life. The saturating fix in `exit` must land before the
        // debug_assert/log, so this invariant holds in EVERY build profile,
        // unlike the debug-only panic covered by
        // `exit_without_matching_enter_is_a_bug` above. In a debug build
        // (this one included, `cargo test`'s default) each extra `exit()`
        // still panics via that same debug_assert; catch it so this test can
        // assert the underlying counter state regardless.
        let tracker = NativeActivityTracker::new();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tracker.exit()));
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tracker.exit()));

        assert_eq!(
            tracker.active.load(Ordering::Acquire),
            0,
            "the counter must saturate at zero, never wrap past it"
        );
        assert!(
            tracker.is_idle_for(
                Instant::now() + Duration::from_secs(3600),
                Duration::from_secs(1)
            ),
            "a saturated (not wrapped) zero counter must still read as idle"
        );

        // The earlier accounting bug must not leave the tracker wedged: a
        // later, legitimate enter/exit pair still works normally.
        tracker.enter();
        tracker.exit();
        assert!(tracker.is_idle_for(
            Instant::now() + Duration::from_secs(3600),
            Duration::from_secs(1)
        ));
    }

    // Only compiled (and only meaningful) with `debug_assertions` off, i.e.
    // under `cargo test --release`: proves the release half of the
    // contract that `exit_without_matching_enter_is_a_bug` cannot -- an
    // accounting underflow must NOT panic a running release daemon. If this
    // panicked, the test itself would fail (no `catch_unwind` here, unlike
    // the profile-agnostic test above). Absent from the default
    // `cargo test`/`cargo nextest run` (debug) run, where `debug_assert`
    // deliberately still panics; that half of the contract is pinned by
    // `exit_without_matching_enter_is_a_bug` instead.
    #[cfg(not(debug_assertions))]
    #[test]
    fn release_builds_do_not_panic_on_an_accounting_underflow() {
        let tracker = NativeActivityTracker::new();
        tracker.exit();
        tracker.exit();
        assert_eq!(tracker.active.load(Ordering::Acquire), 0);
        assert!(tracker.is_idle_for(
            Instant::now() + Duration::from_secs(3600),
            Duration::from_secs(1)
        ));
    }

    #[test]
    fn guard_pairs_enter_and_exit_without_panicking() {
        // Exercises the guard against the real process-wide singleton (there
        // is no private-instance variant of the guard/free functions). Other
        // tests in this crate may concurrently touch the same singleton via
        // their own guards, so this only asserts that construction and drop
        // do not panic (the paired enter/exit debug_assert above is what
        // actually proves the accounting stays balanced) rather than any
        // absolute or relative count, which would be flaky under contention.
        let _guard = NativeActivityGuard::enter();
        drop(_guard);
    }

    #[test]
    fn shared_native_activity_guard_release_is_idempotent() {
        // Delta-based (not absolute) against the real process-wide singleton,
        // with no `.await`/sleep between snapshot and assertions, keeping the
        // window against other concurrently running tests as tight as
        // possible -- see `guard_pairs_enter_and_exit_without_panicking`'s
        // comment on why this file avoids absolute assertions on the
        // singleton.
        let before = native_activity_active_count();
        let guard = SharedNativeActivityGuard::new();
        assert_eq!(native_activity_active_count(), before + 1);

        let clone = guard.clone();
        // Either clone may be the one to observe/trigger the release first
        // (this mirrors the real race between a worker thread finishing
        // normally and an external watchdog/preemption trigger firing
        // first); whichever fires first must win, and the other must be a
        // no-op.
        clone.release();
        assert_eq!(
            before,
            native_activity_active_count(),
            "release from a clone must retire the shared guard"
        );

        guard.release();
        assert_eq!(
            before,
            native_activity_active_count(),
            "a second release from a different clone must be a no-op, not double-decrement"
        );

        drop(guard);
        drop(clone);
        assert_eq!(
            before,
            native_activity_active_count(),
            "dropping the already-released clones must not decrement again"
        );
    }

    #[test]
    fn shared_native_activity_guard_releases_on_last_drop_if_never_explicitly_released() {
        let before = native_activity_active_count();
        let guard = SharedNativeActivityGuard::new();
        let clone = guard.clone();
        assert_eq!(native_activity_active_count(), before + 1);

        drop(guard);
        assert_eq!(
            native_activity_active_count(),
            before + 1,
            "one remaining clone must still hold the guard"
        );

        drop(clone);
        assert_eq!(
            native_activity_active_count(),
            before,
            "dropping the last clone without an explicit release() must still retire the guard"
        );
    }

    #[test]
    fn native_model_is_not_resident_before_any_warm_mark() {
        // Regression guard for the `u64::MAX` sentinel: a generation counter
        // that starts at 0 must never accidentally equal it, or a fresh boot
        // (before the first successful load) would misreport resident.
        let _generation_guard = native_unload_generation_test_lock_blocking();
        assert_ne!(
            native_unload_generation(),
            u64::MAX,
            "a real process-wide generation must never coincide with the never-warmed sentinel"
        );
    }

    #[test]
    fn marking_warm_makes_native_model_read_resident() {
        let _generation_guard = native_unload_generation_test_lock_blocking();
        bump_native_unload_generation();
        assert!(
            !native_model_is_resident(),
            "a fresh bump with no matching mark must read as not resident"
        );

        mark_native_model_warm();
        assert!(
            native_model_is_resident(),
            "marking warm at the current generation must read as resident"
        );
    }

    #[test]
    fn idle_unload_eviction_flips_resident_back_to_false() {
        // Exercises the exact flip this field exists for: reload (mark warm)
        // makes it true, and a subsequent idle-unload eviction (bump the
        // generation, as `spawn_idle_unload_reaper` does) makes it false
        // again without any further mark -- the reader never has to poll a
        // second signal to notice the eviction.
        let _generation_guard = native_unload_generation_test_lock_blocking();
        bump_native_unload_generation();
        mark_native_model_warm();
        assert!(
            native_model_is_resident(),
            "just-marked warm at the current generation must read as resident"
        );

        bump_native_unload_generation();
        assert!(
            !native_model_is_resident(),
            "an idle-unload eviction must flip resident back to false until the next mark"
        );

        mark_native_model_warm();
        assert!(
            native_model_is_resident(),
            "a reload after the eviction (mark at the new generation) must flip resident back to true"
        );
    }
}
