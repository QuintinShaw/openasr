//! Process-owned exclusive checkout pool for admitted mutable runtimes.
//!
//! Immutable host materializations belong in `AdmittedHostObjectCache`. A
//! decoder/encoder runtime is different: inference needs exclusive `&mut`
//! access, while the expensive resident owner should return to a bounded pool
//! after the request. This module owns that lifecycle behind one interface.
//!
//! The pool enforces all cache invariants together:
//! - one builder per key at a time;
//! - an explicit active-plus-idle instance permit per key;
//! - entry-count and committed-requested-byte LRU limits for idle owners;
//! - an owner-bound SystemMemory lease throughout checkout;
//! - failed/panicking builds remain retryable;
//! - oversized owners execute once but never enter the idle cache; and
//! - clear/targeted eviction advances an epoch, so an older checkout cannot
//!   resurrect invalidated residency when it is dropped.
//!
//! Values must be `Send`: this is a process pool and a checkout may be returned
//! on a different worker. Thread-affine backend owners must use a thread-affine
//! adapter instead of asserting `Send` merely to fit this interface.

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::ops::{Deref, DerefMut};
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex};

use super::native_execution_services::stage_execution_cache_commit;
use super::system_memory_owner::SystemMemoryOwner;

/// Resident value understood by the exclusive pool's byte-accounting layer.
///
/// Most callers use [`SystemMemoryOwner<T>`] directly. Thread-pinned runtime
/// actors already retain that owner and its lease on their worker thread, so
/// they implement this trait themselves instead of being wrapped in a second,
/// zero-byte `SystemMemoryOwner` that would make the idle LRU undercount them.
pub(crate) trait AdmittedExclusivePoolOwner: Send + 'static {
    fn committed_requested_bytes(&self) -> u64;

    fn is_reusable(&self) -> bool {
        true
    }
}

impl<T> AdmittedExclusivePoolOwner for SystemMemoryOwner<T>
where
    T: Send + 'static,
{
    fn committed_requested_bytes(&self) -> u64 {
        SystemMemoryOwner::committed_requested_bytes(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdmittedExclusiveObjectPoolLimits {
    pub(crate) max_idle_entries: usize,
    pub(crate) max_idle_committed_requested_bytes: u64,
    pub(crate) max_instances_per_key: usize,
}

impl AdmittedExclusiveObjectPoolLimits {
    pub(crate) const fn new(
        max_idle_entries: usize,
        max_idle_committed_requested_bytes: u64,
        max_instances_per_key: usize,
    ) -> Self {
        Self {
            max_idle_entries,
            max_idle_committed_requested_bytes,
            max_instances_per_key,
        }
    }
}

#[derive(Debug)]
struct IdleOwner<T> {
    id: u64,
    owner: T,
}

#[derive(Debug)]
struct PoolEntry<T> {
    epoch: u64,
    building: bool,
    instances: usize,
    idle: VecDeque<IdleOwner<T>>,
}

impl<T> PoolEntry<T> {
    fn new(epoch: u64) -> Self {
        Self {
            epoch,
            building: false,
            instances: 0,
            idle: VecDeque::new(),
        }
    }
}

#[derive(Debug)]
struct PoolState<K, T> {
    entries: HashMap<K, PoolEntry<T>>,
    /// One record per idle owner; front is least recently returned/used.
    idle_lru: VecDeque<(K, u64)>,
    idle_entries: usize,
    idle_committed_requested_bytes: u64,
    next_owner_id: u64,
    global_epoch: u64,
}

impl<K, T> Default for PoolState<K, T> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            idle_lru: VecDeque::new(),
            idle_entries: 0,
            idle_committed_requested_bytes: 0,
            next_owner_id: 1,
            global_epoch: 1,
        }
    }
}

#[derive(Debug)]
struct PoolInner<K, T> {
    state: Mutex<PoolState<K, T>>,
    build_finished: Condvar,
    limits: AdmittedExclusiveObjectPoolLimits,
}

/// Process-shared pool for mutable resident owners.
#[derive(Debug)]
pub(crate) struct AdmittedExclusiveObjectPool<K, T> {
    inner: Arc<PoolInner<K, T>>,
}

impl<K, T> Clone for AdmittedExclusiveObjectPool<K, T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<K, T> AdmittedExclusiveObjectPool<K, T>
where
    K: Clone + Eq + Hash + Send + 'static,
    T: AdmittedExclusivePoolOwner,
{
    pub(crate) fn new(limits: AdmittedExclusiveObjectPoolLimits) -> Self {
        assert!(
            limits.max_instances_per_key > 0,
            "exclusive owner pool requires at least one instance permit per key"
        );
        Self {
            inner: Arc::new(PoolInner {
                state: Mutex::new(PoolState::default()),
                build_finished: Condvar::new(),
                limits,
            }),
        }
    }

    /// Checks out one owner exclusively, building only when no idle instance is
    /// available. `quote` and `build` run at most once and only after this key's
    /// build slot has been acquired.
    pub(crate) fn checkout_or_try_build<E, A, Q, F, M>(
        &self,
        key: K,
        quote: Q,
        build: F,
        map_internal_failure: M,
    ) -> Result<AdmittedExclusiveObjectCheckout<K, T>, E>
    where
        Q: FnOnce() -> Result<(u64, A), E>,
        F: FnOnce(A) -> Result<T, E>,
        M: Fn(String) -> E,
    {
        let mut quote = Some(quote);
        let mut build = Some(build);

        loop {
            let mut state =
                self.inner.state.lock().map_err(|_| {
                    map_internal_failure("exclusive owner pool lock poisoned".into())
                })?;
            let global_epoch = state.global_epoch;
            let epoch = state
                .entries
                .entry(key.clone())
                .or_insert_with(|| PoolEntry::new(global_epoch))
                .epoch;

            if let Some(idle) = state
                .entries
                .get_mut(&key)
                .and_then(|entry| entry.idle.pop_back())
            {
                state
                    .idle_lru
                    .retain(|candidate| candidate != &(key.clone(), idle.id));
                state.idle_entries = state.idle_entries.saturating_sub(1);
                state.idle_committed_requested_bytes = state
                    .idle_committed_requested_bytes
                    .saturating_sub(idle.owner.committed_requested_bytes());
                drop(state);
                return Ok(AdmittedExclusiveObjectCheckout {
                    inner: Arc::clone(&self.inner),
                    key,
                    epoch,
                    cacheable: true,
                    owner: Some(idle.owner),
                });
            }

            if state.entries.get(&key).is_some_and(|entry| entry.building) {
                state = self.inner.build_finished.wait(state).map_err(|_| {
                    map_internal_failure("exclusive owner pool build wait poisoned".into())
                })?;
                drop(state);
                continue;
            }

            let entry = state
                .entries
                .get_mut(&key)
                .expect("entry was inserted above");
            if entry.instances >= self.inner.limits.max_instances_per_key {
                state = self.inner.build_finished.wait(state).map_err(|_| {
                    map_internal_failure("exclusive owner pool permit wait poisoned".into())
                })?;
                drop(state);
                continue;
            }
            entry.instances += 1;
            entry.building = true;
            let build_epoch = entry.epoch;
            drop(state);

            let built = panic::catch_unwind(AssertUnwindSafe(|| {
                let (quoted_retained_bytes, allocation_quote) = (quote
                    .take()
                    .expect("quote closure is consumed by one acquired build slot"))(
                )?;
                self.make_room_for(quoted_retained_bytes, &map_internal_failure)?;
                let owner = (build
                    .take()
                    .expect("build closure is consumed by one acquired build slot"))(
                    allocation_quote,
                )?;
                Ok::<_, E>((quoted_retained_bytes, owner))
            }));

            let mut state =
                self.inner.state.lock().map_err(|_| {
                    map_internal_failure("exclusive owner pool lock poisoned".into())
                })?;
            if let Some(entry) = state.entries.get_mut(&key) {
                entry.building = false;
            }
            self.inner.build_finished.notify_all();

            match built {
                Ok(Ok((quoted_retained_bytes, owner))) => {
                    let attached = state
                        .entries
                        .get(&key)
                        .is_some_and(|entry| entry.epoch == build_epoch);
                    let cacheable = attached
                        && self.inner.limits.max_idle_entries > 0
                        && quoted_retained_bytes
                            <= self.inner.limits.max_idle_committed_requested_bytes
                        && owner.committed_requested_bytes()
                            <= self.inner.limits.max_idle_committed_requested_bytes;
                    drop(state);
                    return Ok(AdmittedExclusiveObjectCheckout {
                        inner: Arc::clone(&self.inner),
                        key,
                        epoch: build_epoch,
                        cacheable,
                        owner: Some(owner),
                    });
                }
                Ok(Err(error)) => {
                    release_failed_build(&mut state, &key);
                    drop(state);
                    return Err(error);
                }
                Err(payload) => {
                    release_failed_build(&mut state, &key);
                    drop(state);
                    return Err(map_internal_failure(format!(
                        "exclusive owner build panicked: {}",
                        describe_panic_payload(payload.as_ref())
                    )));
                }
            }
        }
    }

    /// Drops every idle owner and invalidates every active checkout. Active
    /// owners keep their leases until their users drop them, then cannot return.
    pub(crate) fn clear(&self) {
        let victims = {
            let Ok(mut state) = self.inner.state.lock() else {
                return;
            };
            state.global_epoch = state.global_epoch.wrapping_add(1).max(1);
            let epoch = state.global_epoch;
            let mut victims = Vec::new();
            for entry in state.entries.values_mut() {
                entry.epoch = epoch;
                entry.instances = entry.instances.saturating_sub(entry.idle.len());
                victims.extend(entry.idle.drain(..).map(|idle| idle.owner));
            }
            state
                .entries
                .retain(|_, entry| entry.instances > 0 || entry.building);
            state.idle_lru.clear();
            state.idle_entries = 0;
            state.idle_committed_requested_bytes = 0;
            victims
        };
        drop(victims);
        self.inner.build_finished.notify_all();
    }

    /// Drops idle owners whose keys match `predicate` and invalidates matching
    /// active/building checkouts without disturbing unrelated keys. An older
    /// checkout retains its owner until drop, but its captured epoch no longer
    /// matches and therefore cannot republish stale residency.
    pub(crate) fn evict_where(&self, mut predicate: impl FnMut(&K) -> bool) {
        let victims = {
            let Ok(mut state) = self.inner.state.lock() else {
                return;
            };
            let keys = state
                .entries
                .keys()
                .filter(|key| predicate(key))
                .cloned()
                .collect::<Vec<_>>();
            let mut victims = Vec::new();
            for key in &keys {
                let Some(entry) = state.entries.get_mut(key) else {
                    continue;
                };
                entry.epoch = entry.epoch.wrapping_add(1).max(1);
                entry.instances = entry
                    .instances
                    .checked_sub(entry.idle.len())
                    .expect("idle owners are included in the per-key instance ledger");
                victims.extend(entry.idle.drain(..).map(|idle| idle.owner));
            }
            state
                .entries
                .retain(|_, entry| entry.instances > 0 || entry.building);
            state.idle_lru.retain(|(key, _)| !keys.contains(key));
            state.idle_entries = state.entries.values().map(|entry| entry.idle.len()).sum();
            state.idle_committed_requested_bytes = state
                .entries
                .values()
                .flat_map(|entry| entry.idle.iter())
                .try_fold(0_u64, |total, idle| {
                    total.checked_add(idle.owner.committed_requested_bytes())
                })
                .expect("idle owner byte ledger cannot overflow its configured limit");
            victims
        };
        drop(victims);
        self.inner.build_finished.notify_all();
    }

    fn make_room_for<E, M>(&self, requested_bytes: u64, map_internal_failure: &M) -> Result<(), E>
    where
        M: Fn(String) -> E,
    {
        if self.inner.limits.max_idle_entries == 0
            || requested_bytes > self.inner.limits.max_idle_committed_requested_bytes
        {
            return Ok(());
        }
        let victims = {
            let mut state =
                self.inner.state.lock().map_err(|_| {
                    map_internal_failure("exclusive owner pool lock poisoned".into())
                })?;
            let mut victims = Vec::new();
            while state.idle_entries >= self.inner.limits.max_idle_entries
                || state
                    .idle_committed_requested_bytes
                    .checked_add(requested_bytes)
                    .is_none_or(|total| {
                        total > self.inner.limits.max_idle_committed_requested_bytes
                    })
            {
                let Some((victim_key, victim_id)) = state.idle_lru.pop_front() else {
                    break;
                };
                let victim = state.entries.get_mut(&victim_key).and_then(|entry| {
                    let index = entry.idle.iter().position(|idle| idle.id == victim_id)?;
                    entry.idle.remove(index)
                });
                let Some(victim) = victim else {
                    continue;
                };
                state.idle_entries = state.idle_entries.saturating_sub(1);
                state.idle_committed_requested_bytes = state
                    .idle_committed_requested_bytes
                    .saturating_sub(victim.owner.committed_requested_bytes());
                if let Some(entry) = state.entries.get_mut(&victim_key) {
                    entry.instances = entry.instances.saturating_sub(1);
                }
                victims.push(victim.owner);
            }
            state
                .entries
                .retain(|_, entry| entry.instances > 0 || entry.building);
            victims
        };
        drop(victims);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn usage_for_test(&self) -> (usize, u64) {
        self.inner
            .state
            .lock()
            .map(|state| (state.idle_entries, state.idle_committed_requested_bytes))
            .unwrap_or((0, 0))
    }
}

/// Exclusive mutable checkout. The admitted owner moves out of the pool, so
/// unique ownership - and therefore `&mut T` - is guaranteed by the type.
pub(crate) struct AdmittedExclusiveObjectCheckout<K, T>
where
    K: Clone + Eq + Hash + Send + 'static,
    T: AdmittedExclusivePoolOwner,
{
    inner: Arc<PoolInner<K, T>>,
    key: K,
    epoch: u64,
    cacheable: bool,
    owner: Option<T>,
}

impl<K, T> Deref for AdmittedExclusiveObjectCheckout<K, T>
where
    K: Clone + Eq + Hash + Send + 'static,
    T: AdmittedExclusivePoolOwner,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.owner
            .as_ref()
            .expect("exclusive owner checkout must contain an owner")
    }
}

impl<K, T> DerefMut for AdmittedExclusiveObjectCheckout<K, T>
where
    K: Clone + Eq + Hash + Send + 'static,
    T: AdmittedExclusivePoolOwner,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.owner
            .as_mut()
            .expect("exclusive owner checkout must contain an owner")
    }
}

impl<K, T> Drop for AdmittedExclusiveObjectCheckout<K, T>
where
    K: Clone + Eq + Hash + Send + 'static,
    T: AdmittedExclusivePoolOwner,
{
    fn drop(&mut self) {
        let Some(owner) = self.owner.take() else {
            return;
        };
        let pending = PendingReturn {
            inner: Arc::clone(&self.inner),
            key: self.key.clone(),
            epoch: self.epoch,
            cacheable: self.cacheable,
            owner: Some(owner),
            completed: false,
        };
        stage_execution_cache_commit(move || pending.commit());
    }
}

struct PendingReturn<K, T>
where
    K: Clone + Eq + Hash + Send + 'static,
    T: AdmittedExclusivePoolOwner,
{
    inner: Arc<PoolInner<K, T>>,
    key: K,
    epoch: u64,
    cacheable: bool,
    owner: Option<T>,
    completed: bool,
}

impl<K, T> PendingReturn<K, T>
where
    K: Clone + Eq + Hash + Send + 'static,
    T: AdmittedExclusivePoolOwner,
{
    fn commit(mut self) {
        let owner = self.owner.take().expect("pending return owns its runtime");
        let mut victims = Vec::new();
        if let Ok(mut state) = self.inner.state.lock() {
            let attached = state
                .entries
                .get(&self.key)
                .is_some_and(|entry| entry.epoch == self.epoch);
            let bytes = owner.committed_requested_bytes();
            let can_cache = attached
                && self.cacheable
                && owner.is_reusable()
                && self.inner.limits.max_idle_entries > 0
                && bytes <= self.inner.limits.max_idle_committed_requested_bytes;
            if can_cache {
                while state.idle_entries >= self.inner.limits.max_idle_entries
                    || state
                        .idle_committed_requested_bytes
                        .checked_add(bytes)
                        .is_none_or(|total| {
                            total > self.inner.limits.max_idle_committed_requested_bytes
                        })
                {
                    let Some((victim_key, victim_id)) = state.idle_lru.pop_front() else {
                        break;
                    };
                    let victim = state.entries.get_mut(&victim_key).and_then(|entry| {
                        let index = entry.idle.iter().position(|idle| idle.id == victim_id)?;
                        entry.idle.remove(index)
                    });
                    if let Some(victim) = victim {
                        state.idle_entries = state.idle_entries.saturating_sub(1);
                        state.idle_committed_requested_bytes = state
                            .idle_committed_requested_bytes
                            .saturating_sub(victim.owner.committed_requested_bytes());
                        if let Some(entry) = state.entries.get_mut(&victim_key) {
                            entry.instances = entry.instances.saturating_sub(1);
                        }
                        victims.push(victim.owner);
                    }
                }
                let id = state.next_owner_id;
                state.next_owner_id = state.next_owner_id.wrapping_add(1).max(1);
                if let Some(entry) = state.entries.get_mut(&self.key) {
                    entry.idle.push_back(IdleOwner { id, owner });
                }
                state.idle_lru.push_back((self.key.clone(), id));
                state.idle_entries += 1;
                state.idle_committed_requested_bytes = state
                    .idle_committed_requested_bytes
                    .checked_add(bytes)
                    .expect("idle byte limit proves sum fits");
            } else {
                if let Some(entry) = state.entries.get_mut(&self.key) {
                    entry.instances = entry.instances.saturating_sub(1);
                }
                victims.push(owner);
            }
            state
                .entries
                .retain(|_, entry| entry.instances > 0 || entry.building);
        } else {
            victims.push(owner);
        }
        self.completed = true;
        self.inner.build_finished.notify_all();
        drop(victims);
    }
}

impl<K, T> Drop for PendingReturn<K, T>
where
    K: Clone + Eq + Hash + Send + 'static,
    T: AdmittedExclusivePoolOwner,
{
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        let owner = self.owner.take();
        if let Ok(mut state) = self.inner.state.lock() {
            if let Some(entry) = state.entries.get_mut(&self.key) {
                entry.instances = entry.instances.saturating_sub(1);
            }
            state
                .entries
                .retain(|_, entry| entry.instances > 0 || entry.building);
        }
        self.inner.build_finished.notify_all();
        drop(owner);
    }
}

fn release_failed_build<K, T>(state: &mut PoolState<K, T>, key: &K)
where
    K: Eq + Hash,
{
    if let Some(entry) = state.entries.get_mut(key) {
        entry.instances = entry.instances.saturating_sub(1);
        entry.building = false;
    }
    state
        .entries
        .retain(|_, entry| entry.instances > 0 || entry.building);
}

fn describe_panic_payload(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::system_memory_owner::SystemMemoryOwner;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    fn owner(value: usize, bytes: u64) -> SystemMemoryOwner<usize> {
        SystemMemoryOwner::with_committed_requested_bytes_for_test(value, bytes)
    }

    fn checkout(
        pool: &AdmittedExclusiveObjectPool<&'static str, SystemMemoryOwner<usize>>,
        key: &'static str,
        bytes: u64,
        builds: &AtomicUsize,
    ) -> Result<AdmittedExclusiveObjectCheckout<&'static str, SystemMemoryOwner<usize>>, String>
    {
        pool.checkout_or_try_build(
            key,
            || Ok((bytes, ())),
            |()| {
                let value = builds.fetch_add(1, Ordering::SeqCst) + 1;
                Ok(owner(value, bytes))
            },
            |reason| reason,
        )
    }

    #[test]
    fn concurrent_cold_build_is_single_flight_per_key() {
        let pool =
            AdmittedExclusiveObjectPool::new(AdmittedExclusiveObjectPoolLimits::new(2, 1024, 2));
        let builds = Arc::new(AtomicUsize::new(0));
        let active_builders = Arc::new(AtomicUsize::new(0));
        let maximum_active_builders = Arc::new(AtomicUsize::new(0));
        let first_pool = pool.clone();
        let first_builds = Arc::clone(&builds);
        let first_active = Arc::clone(&active_builders);
        let first_maximum = Arc::clone(&maximum_active_builders);
        let (first_builder_entered_tx, first_builder_entered_rx) = mpsc::channel();
        let first = thread::spawn(move || {
            first_pool
                .checkout_or_try_build(
                    "same",
                    || Ok::<_, String>((32, ())),
                    |()| {
                        first_builds.fetch_add(1, Ordering::SeqCst);
                        let active = first_active.fetch_add(1, Ordering::SeqCst) + 1;
                        first_maximum.fetch_max(active, Ordering::SeqCst);
                        first_builder_entered_tx
                            .send(())
                            .expect("report first builder entry");
                        thread::sleep(Duration::from_millis(30));
                        first_active.fetch_sub(1, Ordering::SeqCst);
                        Ok(owner(1, 32))
                    },
                    |reason| reason,
                )
                .expect("first checkout")
        });
        first_builder_entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("first builder should enter before the second checkout starts");
        let second_pool = pool.clone();
        let second_builds = Arc::clone(&builds);
        let second_active = Arc::clone(&active_builders);
        let second_maximum = Arc::clone(&maximum_active_builders);
        let (second_builder_entered_tx, second_builder_entered_rx) = mpsc::channel();
        let second = thread::spawn(move || {
            let checkout = second_pool
                .checkout_or_try_build(
                    "same",
                    || Ok::<_, String>((32, ())),
                    |()| {
                        second_builds.fetch_add(1, Ordering::SeqCst);
                        let active = second_active.fetch_add(1, Ordering::SeqCst) + 1;
                        second_maximum.fetch_max(active, Ordering::SeqCst);
                        second_builder_entered_tx
                            .send(())
                            .expect("report second builder entry");
                        thread::sleep(Duration::from_millis(10));
                        second_active.fetch_sub(1, Ordering::SeqCst);
                        Ok(owner(2, 32))
                    },
                    |reason| reason,
                )
                .expect("second checkout");
            drop(checkout);
        });
        let first_checkout = first.join().expect("first thread");
        second_builder_entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("second builder should enter while the first checkout stays active");
        drop(first_checkout);
        second.join().expect("second thread");
        assert_eq!(maximum_active_builders.load(Ordering::SeqCst), 1);
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        assert_eq!(pool.usage_for_test().0, builds.load(Ordering::SeqCst));
    }

    #[test]
    fn checkout_waits_for_an_instance_permit_then_reuses_the_returned_owner() {
        let pool =
            AdmittedExclusiveObjectPool::new(AdmittedExclusiveObjectPoolLimits::new(2, 1024, 2));
        let builds = Arc::new(AtomicUsize::new(0));
        let first = checkout(&pool, "same", 32, &builds).expect("first");
        let second = checkout(&pool, "same", 32, &builds).expect("second");
        assert_eq!(builds.load(Ordering::SeqCst), 2);

        let waiting_pool = pool.clone();
        let waiting_builds = Arc::clone(&builds);
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let waiting = thread::spawn(move || {
            let checkout =
                checkout(&waiting_pool, "same", 32, &waiting_builds).expect("waiting checkout");
            acquired_tx.send(**checkout).expect("report acquisition");
            drop(checkout);
        });
        assert!(matches!(
            acquired_rx.recv_timeout(Duration::from_millis(30)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        drop(first);
        assert_eq!(
            acquired_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("returned permit must wake waiter"),
            1
        );
        drop(second);
        waiting.join().expect("waiting thread");
        assert_eq!(
            builds.load(Ordering::SeqCst),
            2,
            "the waiter must reuse a returned owner instead of building a third"
        );
    }

    #[test]
    fn failed_and_panicking_builds_release_the_permit_for_retry() {
        let pool =
            AdmittedExclusiveObjectPool::new(AdmittedExclusiveObjectPoolLimits::new(1, 64, 1));
        let failed = pool.checkout_or_try_build(
            "key",
            || Ok::<_, String>((32, ())),
            |()| Err::<SystemMemoryOwner<usize>, _>("failed".to_string()),
            |reason| reason,
        );
        assert!(matches!(failed, Err(ref reason) if reason == "failed"));
        let panicked = pool.checkout_or_try_build(
            "key",
            || Ok::<_, String>((32, ())),
            |()| -> Result<SystemMemoryOwner<usize>, String> { panic!("boom") },
            |reason| reason,
        );
        assert!(matches!(panicked, Err(ref reason) if reason.contains("boom")));
        let builds = AtomicUsize::new(0);
        drop(checkout(&pool, "key", 32, &builds).expect("retry"));
        assert_eq!(builds.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn byte_lru_evicts_idle_owner_but_checkout_survives_clear() {
        let pool =
            AdmittedExclusiveObjectPool::new(AdmittedExclusiveObjectPoolLimits::new(4, 100, 2));
        let builds = AtomicUsize::new(0);
        drop(checkout(&pool, "a", 60, &builds).unwrap());
        drop(checkout(&pool, "b", 50, &builds).unwrap());
        assert_eq!(pool.usage_for_test(), (1, 50));

        let in_flight = checkout(&pool, "b", 50, &builds).unwrap();
        pool.clear();
        assert_eq!(**in_flight, 2);
        assert_eq!(pool.usage_for_test(), (0, 0));
        drop(in_flight);
        assert_eq!(pool.usage_for_test(), (0, 0));
    }

    #[test]
    fn targeted_eviction_preserves_unrelated_idle_owner_and_blocks_resurrection() {
        let pool =
            AdmittedExclusiveObjectPool::new(AdmittedExclusiveObjectPoolLimits::new(4, 256, 2));
        let builds = AtomicUsize::new(0);
        drop(checkout(&pool, "a", 32, &builds).unwrap());
        drop(checkout(&pool, "b", 32, &builds).unwrap());
        let in_flight_a = checkout(&pool, "a", 32, &builds).unwrap();

        pool.evict_where(|key| *key == "a");
        assert_eq!(pool.usage_for_test(), (1, 32));
        drop(in_flight_a);
        assert_eq!(pool.usage_for_test(), (1, 32));

        drop(checkout(&pool, "b", 32, &builds).unwrap());
        assert_eq!(builds.load(Ordering::SeqCst), 2, "b must be reused");
        drop(checkout(&pool, "a", 32, &builds).unwrap());
        assert_eq!(builds.load(Ordering::SeqCst), 3, "a must rebuild");
    }

    #[test]
    fn oversized_owner_executes_but_is_never_cached() {
        let pool =
            AdmittedExclusiveObjectPool::new(AdmittedExclusiveObjectPoolLimits::new(4, 32, 1));
        let builds = AtomicUsize::new(0);
        drop(checkout(&pool, "large", 64, &builds).unwrap());
        assert_eq!(pool.usage_for_test(), (0, 0));
        drop(checkout(&pool, "large", 64, &builds).unwrap());
        assert_eq!(builds.load(Ordering::SeqCst), 2);
    }
}
