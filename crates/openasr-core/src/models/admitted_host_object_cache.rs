//! Bounded, single-flight ownership for admitted host objects.
//!
//! [`SingleFlightWeightedCache`] is the one reusable concurrency/LRU core. It
//! knows only keys, cloneable values and weights; typed wrappers decide which
//! values are legal. [`AdmittedHostObjectCache`] admits only
//! [`AdmittedHostObject`] values, while the heterogeneous auxiliary cache uses
//! the same core behind an erased-but-still-owner-bound facade.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, Condvar, Mutex};

use super::native_execution_services::ExecutionCacheAttemptId;
use super::system_memory_owner::AdmittedHostObject;

/// Small enough to bound multi-pack residency while still supporting the
/// common active/staged-model handoff without immediate churn.
pub(crate) const DEFAULT_ADMITTED_HOST_OBJECT_CACHE_MAX_ENTRIES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdmittedHostObjectCacheLimits {
    pub(crate) max_entries: usize,
    pub(crate) max_committed_requested_bytes: u64,
}

impl AdmittedHostObjectCacheLimits {
    pub(crate) const fn new(max_entries: usize, max_committed_requested_bytes: u64) -> Self {
        Self {
            max_entries,
            max_committed_requested_bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SingleFlightWeightedCacheError {
    Poisoned,
}

#[derive(Debug)]
enum WeightedSlotState<V> {
    Empty,
    Building,
    Staged {
        attempt_id: ExecutionCacheAttemptId,
        value: V,
    },
    Ready(V),
}

#[derive(Debug)]
struct WeightedSlot<V> {
    state: Mutex<WeightedSlotState<V>>,
    ready: Condvar,
}

impl<V> WeightedSlot<V> {
    fn empty() -> Self {
        Self {
            state: Mutex::new(WeightedSlotState::Empty),
            ready: Condvar::new(),
        }
    }
}

#[derive(Debug)]
struct WeightedEntry<V> {
    slot: Arc<WeightedSlot<V>>,
    weight: Option<u64>,
}

#[derive(Debug)]
struct WeightedState<K, V> {
    entries: HashMap<K, WeightedEntry<V>>,
    /// Ready keys only; front is least recently used.
    ready_lru: VecDeque<K>,
    ready_entries: usize,
    retained_weight: u64,
}

impl<K, V> Default for WeightedState<K, V> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            ready_lru: VecDeque::new(),
            ready_entries: 0,
            retained_weight: 0,
        }
    }
}

#[derive(Debug)]
struct SingleFlightWeightedCacheInner<K, V> {
    state: Mutex<WeightedState<K, V>>,
    limits: AdmittedHostObjectCacheLimits,
}

/// Cache-neutral single-flight, entry-count and weighted LRU state machine.
///
/// A build permit can be published immediately or retained in a transaction
/// journal. Dropping an unpublished permit rolls its slot back to `Empty` and
/// wakes waiters. Clear/evict detach entries atomically; every potentially
/// expensive value destructor runs only after the global map lock is released.
#[derive(Debug)]
pub(crate) struct SingleFlightWeightedCache<K, V> {
    inner: Arc<SingleFlightWeightedCacheInner<K, V>>,
}

impl<K, V> Clone for SingleFlightWeightedCache<K, V> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

pub(crate) enum SingleFlightWeightedLookup<K: Eq + Hash, V> {
    Ready(V),
    Build(SingleFlightWeightedBuildPermit<K, V>),
}

pub(crate) struct SingleFlightWeightedBuildPermit<K: Eq + Hash, V> {
    inner: Arc<SingleFlightWeightedCacheInner<K, V>>,
    key: K,
    slot: Arc<WeightedSlot<V>>,
    armed: bool,
}

/// A value visible only to the candidate transaction that built it. Commit
/// promotes it to `Ready`; dropping the journal callback rolls it back to an
/// empty retryable slot and wakes every waiter.
pub(crate) struct SingleFlightWeightedStagedPublication<K: Eq + Hash, V> {
    inner: Arc<SingleFlightWeightedCacheInner<K, V>>,
    key: K,
    slot: Arc<WeightedSlot<V>>,
    attempt_id: ExecutionCacheAttemptId,
    actual_weight: u64,
    retain: bool,
    armed: bool,
}

impl<K, V> SingleFlightWeightedCache<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    pub(crate) fn new(limits: AdmittedHostObjectCacheLimits) -> Self {
        Self {
            inner: Arc::new(SingleFlightWeightedCacheInner {
                state: Mutex::new(WeightedState::default()),
                limits,
            }),
        }
    }

    pub(crate) fn lookup_or_reserve(
        &self,
        key: K,
        visible_attempt: Option<ExecutionCacheAttemptId>,
    ) -> Result<SingleFlightWeightedLookup<K, V>, SingleFlightWeightedCacheError> {
        let slot = {
            let mut state = self
                .inner
                .state
                .lock()
                .map_err(|_| SingleFlightWeightedCacheError::Poisoned)?;
            Arc::clone(
                &state
                    .entries
                    .entry(key.clone())
                    .or_insert_with(|| WeightedEntry {
                        slot: Arc::new(WeightedSlot::empty()),
                        weight: None,
                    })
                    .slot,
            )
        };

        let mut slot_state = slot
            .state
            .lock()
            .map_err(|_| SingleFlightWeightedCacheError::Poisoned)?;
        loop {
            match &*slot_state {
                WeightedSlotState::Ready(value) => {
                    let value = value.clone();
                    drop(slot_state);
                    self.touch_if_attached(&key, &slot)?;
                    return Ok(SingleFlightWeightedLookup::Ready(value));
                }
                WeightedSlotState::Building => {
                    slot_state = slot
                        .ready
                        .wait(slot_state)
                        .map_err(|_| SingleFlightWeightedCacheError::Poisoned)?;
                }
                WeightedSlotState::Staged { attempt_id, value }
                    if Some(*attempt_id) == visible_attempt =>
                {
                    return Ok(SingleFlightWeightedLookup::Ready(value.clone()));
                }
                WeightedSlotState::Staged { .. } => {
                    slot_state = slot
                        .ready
                        .wait(slot_state)
                        .map_err(|_| SingleFlightWeightedCacheError::Poisoned)?;
                }
                WeightedSlotState::Empty => {
                    *slot_state = WeightedSlotState::Building;
                    drop(slot_state);
                    return Ok(SingleFlightWeightedLookup::Build(
                        SingleFlightWeightedBuildPermit {
                            inner: Arc::clone(&self.inner),
                            key,
                            slot,
                            armed: true,
                        },
                    ));
                }
            }
        }
    }

    /// Return a clone of an already-published value without waiting for an
    /// in-flight build and without reserving an empty slot. This is the cache
    /// probe for optional hot-path reuse: a miss must leave the single-flight
    /// state completely unchanged so the caller can use its allocation-free
    /// fallback.
    pub(crate) fn ready(&self, key: &K) -> Option<V> {
        let slot = self
            .inner
            .state
            .lock()
            .ok()
            .and_then(|state| state.entries.get(key).map(|entry| Arc::clone(&entry.slot)))?;
        let value = {
            let slot_state = slot.state.lock().ok()?;
            match &*slot_state {
                WeightedSlotState::Ready(value) => value.clone(),
                WeightedSlotState::Empty
                | WeightedSlotState::Building
                | WeightedSlotState::Staged { .. } => return None,
            }
        };
        // A concurrent eviction may detach the slot between the clone above
        // and this touch. `touch_if_attached` deliberately treats that as a
        // no-op; the returned owner clone remains valid on its own.
        self.touch_if_attached(key, &slot).ok()?;
        Some(value)
    }

    pub(crate) fn clear(&self) {
        let detached = if let Ok(mut state) = self.inner.state.lock() {
            let detached = std::mem::take(&mut state.entries);
            reset_accounting(&mut state);
            detached
        } else {
            HashMap::new()
        };
        drop(detached);
    }

    pub(crate) fn evict(&self, key: &K) {
        let detached = self
            .inner
            .state
            .lock()
            .ok()
            .and_then(|mut state| detach_entry(&mut state, key));
        drop(detached);
    }

    pub(crate) fn evict_where(&self, mut predicate: impl FnMut(&K) -> bool) {
        let detached = if let Ok(mut state) = self.inner.state.lock() {
            let keys = state
                .entries
                .keys()
                .filter(|key| predicate(key))
                .cloned()
                .collect::<Vec<_>>();
            keys.into_iter()
                .filter_map(|key| detach_entry(&mut state, &key))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        drop(detached);
    }

    fn touch_if_attached(
        &self,
        key: &K,
        slot: &Arc<WeightedSlot<V>>,
    ) -> Result<(), SingleFlightWeightedCacheError> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| SingleFlightWeightedCacheError::Poisoned)?;
        if state
            .entries
            .get(key)
            .is_some_and(|entry| Arc::ptr_eq(&entry.slot, slot))
        {
            state.ready_lru.retain(|candidate| candidate != key);
            state.ready_lru.push_back(key.clone());
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn usage_for_test(&self) -> (usize, u64) {
        self.inner
            .state
            .lock()
            .map(|state| (state.ready_entries, state.retained_weight))
            .unwrap_or((0, 0))
    }

    #[cfg(test)]
    pub(crate) fn len_for_test(&self) -> usize {
        self.inner
            .state
            .lock()
            .map(|state| state.entries.len())
            .unwrap_or(0)
    }
}

impl<K, V> SingleFlightWeightedBuildPermit<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    /// Detach enough idle LRU entries before a cold materialization asks the
    /// process broker for capacity. `false` means the value may still execute
    /// but must not be retained after publication.
    pub(crate) fn make_room_for(
        &self,
        incoming_weight: u64,
    ) -> Result<bool, SingleFlightWeightedCacheError> {
        if self.inner.limits.max_entries == 0
            || incoming_weight > self.inner.limits.max_committed_requested_bytes
        {
            return Ok(false);
        }
        let detached = {
            let mut state = self
                .inner
                .state
                .lock()
                .map_err(|_| SingleFlightWeightedCacheError::Poisoned)?;
            let Some(selected) =
                select_evictions(&state, &self.key, incoming_weight, self.inner.limits)
            else {
                return Ok(false);
            };
            selected
                .into_iter()
                .filter_map(|candidate| detach_entry(&mut state, &candidate))
                .collect::<Vec<_>>()
        };
        drop(detached);
        Ok(true)
    }

    /// Publish a completed value. This consumes/disarms the build permit.
    /// Publication after `clear` still wakes callers already waiting on the
    /// detached slot, but never resurrects the key into the cache map.
    pub(crate) fn publish(
        mut self,
        value: V,
        actual_weight: u64,
        retain: bool,
    ) -> Result<(), SingleFlightWeightedCacheError> {
        let detached = {
            let mut state = self
                .inner
                .state
                .lock()
                .map_err(|_| SingleFlightWeightedCacheError::Poisoned)?;
            let attached = state
                .entries
                .get(&self.key)
                .is_some_and(|entry| Arc::ptr_eq(&entry.slot, &self.slot));
            let selected = (attached
                && retain
                && self.inner.limits.max_entries > 0
                && actual_weight <= self.inner.limits.max_committed_requested_bytes)
                .then(|| select_evictions(&state, &self.key, actual_weight, self.inner.limits))
                .flatten();
            let detached = if let Some(selected) = selected {
                let detached = selected
                    .into_iter()
                    .filter_map(|candidate| detach_entry(&mut state, &candidate))
                    .collect::<Vec<_>>();
                let entry = state
                    .entries
                    .get_mut(&self.key)
                    .expect("attached build permit retains its cache entry");
                entry.weight = Some(actual_weight);
                state.ready_entries = state.ready_entries.saturating_add(1);
                state.retained_weight = state
                    .retained_weight
                    .checked_add(actual_weight)
                    .expect("weighted eviction selection proved sum fits");
                state.ready_lru.retain(|candidate| candidate != &self.key);
                state.ready_lru.push_back(self.key.clone());
                detached
            } else if attached {
                detach_entry(&mut state, &self.key)
                    .into_iter()
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };

            let mut slot_state = self
                .slot
                .state
                .lock()
                .map_err(|_| SingleFlightWeightedCacheError::Poisoned)?;
            *slot_state = WeightedSlotState::Ready(value);
            self.slot.ready.notify_all();
            self.armed = false;
            detached
        };
        drop(detached);
        Ok(())
    }

    /// Makes a completed value visible to the current candidate transaction
    /// without exposing it to other attempts. The returned publication guard
    /// must be stored in the execution cache journal.
    pub(crate) fn stage(
        mut self,
        value: V,
        actual_weight: u64,
        retain: bool,
        attempt_id: ExecutionCacheAttemptId,
    ) -> Result<SingleFlightWeightedStagedPublication<K, V>, SingleFlightWeightedCacheError> {
        let mut slot_state = self
            .slot
            .state
            .lock()
            .map_err(|_| SingleFlightWeightedCacheError::Poisoned)?;
        if !matches!(*slot_state, WeightedSlotState::Building) {
            return Err(SingleFlightWeightedCacheError::Poisoned);
        }
        *slot_state = WeightedSlotState::Staged { attempt_id, value };
        drop(slot_state);
        self.armed = false;
        Ok(SingleFlightWeightedStagedPublication {
            inner: Arc::clone(&self.inner),
            key: self.key.clone(),
            slot: Arc::clone(&self.slot),
            attempt_id,
            actual_weight,
            retain,
            armed: true,
        })
    }
}

impl<K, V> SingleFlightWeightedStagedPublication<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    pub(crate) fn commit(mut self) -> Result<(), SingleFlightWeightedCacheError> {
        let detached = {
            let mut state = self
                .inner
                .state
                .lock()
                .map_err(|_| SingleFlightWeightedCacheError::Poisoned)?;
            let attached = state
                .entries
                .get(&self.key)
                .is_some_and(|entry| Arc::ptr_eq(&entry.slot, &self.slot));
            let selected = (attached
                && self.retain
                && self.inner.limits.max_entries > 0
                && self.actual_weight <= self.inner.limits.max_committed_requested_bytes)
                .then(|| select_evictions(&state, &self.key, self.actual_weight, self.inner.limits))
                .flatten();
            let detached = if let Some(selected) = selected {
                let detached = selected
                    .into_iter()
                    .filter_map(|candidate| detach_entry(&mut state, &candidate))
                    .collect::<Vec<_>>();
                let entry = state
                    .entries
                    .get_mut(&self.key)
                    .expect("attached staged publication retains its entry");
                entry.weight = Some(self.actual_weight);
                state.ready_entries = state.ready_entries.saturating_add(1);
                state.retained_weight = state
                    .retained_weight
                    .checked_add(self.actual_weight)
                    .expect("weighted eviction selection proved sum fits");
                state.ready_lru.retain(|candidate| candidate != &self.key);
                state.ready_lru.push_back(self.key.clone());
                detached
            } else if attached {
                detach_entry(&mut state, &self.key)
                    .into_iter()
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };

            let mut slot_state = self
                .slot
                .state
                .lock()
                .map_err(|_| SingleFlightWeightedCacheError::Poisoned)?;
            let value = match std::mem::replace(&mut *slot_state, WeightedSlotState::Empty) {
                WeightedSlotState::Staged { attempt_id, value }
                    if attempt_id == self.attempt_id =>
                {
                    value
                }
                other => {
                    *slot_state = other;
                    return Err(SingleFlightWeightedCacheError::Poisoned);
                }
            };
            *slot_state = WeightedSlotState::Ready(value);
            self.slot.ready.notify_all();
            self.armed = false;
            detached
        };
        drop(detached);
        Ok(())
    }
}

impl<K, V> Drop for SingleFlightWeightedStagedPublication<K, V>
where
    K: Eq + Hash,
{
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let detached = if let Ok(mut state) = self.inner.state.lock() {
            let mut slot_state = match self.slot.state.lock() {
                Ok(state) => state,
                Err(_) => return,
            };
            if matches!(
                &*slot_state,
                WeightedSlotState::Staged { attempt_id, .. }
                    if *attempt_id == self.attempt_id
            ) {
                *slot_state = WeightedSlotState::Empty;
            }
            self.slot.ready.notify_all();
            let attached = state
                .entries
                .get(&self.key)
                .is_some_and(|entry| Arc::ptr_eq(&entry.slot, &self.slot));
            if attached && Arc::strong_count(&self.slot) == 2 {
                detach_entry(&mut state, &self.key)
            } else {
                None
            }
        } else {
            None
        };
        drop(detached);
    }
}

impl<K, V> Drop for SingleFlightWeightedBuildPermit<K, V>
where
    K: Eq + Hash,
{
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let detached = if let Ok(mut state) = self.inner.state.lock() {
            let mut slot_state = match self.slot.state.lock() {
                Ok(state) => state,
                Err(_) => return,
            };
            if matches!(*slot_state, WeightedSlotState::Building) {
                *slot_state = WeightedSlotState::Empty;
            }
            self.slot.ready.notify_all();
            let attached = state
                .entries
                .get(&self.key)
                .is_some_and(|entry| Arc::ptr_eq(&entry.slot, &self.slot));
            // Map + permit are the only handles when no lookup is waiting.
            if attached && Arc::strong_count(&self.slot) == 2 {
                detach_entry(&mut state, &self.key)
            } else {
                None
            }
        } else {
            None
        };
        drop(detached);
    }
}

fn select_evictions<K, V>(
    state: &WeightedState<K, V>,
    incoming_key: &K,
    incoming_weight: u64,
    limits: AdmittedHostObjectCacheLimits,
) -> Option<Vec<K>>
where
    K: Clone + Eq + Hash,
{
    let mut selected = Vec::new();
    let mut remaining_entries = state.ready_entries;
    let mut remaining_weight = state.retained_weight;
    for candidate in &state.ready_lru {
        let fits_entries = remaining_entries < limits.max_entries;
        let fits_weight = remaining_weight
            .checked_add(incoming_weight)
            .is_some_and(|total| total <= limits.max_committed_requested_bytes);
        if fits_entries && fits_weight {
            break;
        }
        if candidate == incoming_key {
            continue;
        }
        let Some(entry) = state.entries.get(candidate) else {
            continue;
        };
        // Do not detach a slot observed by a lookup/build. That would permit a
        // duplicate same-key materialization before the observer finishes.
        if Arc::strong_count(&entry.slot) != 1 {
            continue;
        }
        let Some(weight) = entry.weight else {
            continue;
        };
        remaining_entries = remaining_entries.saturating_sub(1);
        remaining_weight = remaining_weight.saturating_sub(weight);
        selected.push(candidate.clone());
    }
    let fits_entries = remaining_entries < limits.max_entries;
    let fits_weight = remaining_weight
        .checked_add(incoming_weight)
        .is_some_and(|total| total <= limits.max_committed_requested_bytes);
    (fits_entries && fits_weight).then_some(selected)
}

fn detach_entry<K, V>(state: &mut WeightedState<K, V>, key: &K) -> Option<WeightedEntry<V>>
where
    K: Eq + Hash,
{
    let entry = state.entries.remove(key)?;
    if let Some(weight) = entry.weight {
        state.ready_entries = state.ready_entries.saturating_sub(1);
        state.retained_weight = state.retained_weight.saturating_sub(weight);
    }
    state.ready_lru.retain(|candidate| candidate != key);
    Some(entry)
}

fn reset_accounting<K, V>(state: &mut WeightedState<K, V>) {
    state.ready_lru.clear();
    state.ready_entries = 0;
    state.retained_weight = 0;
}

/// Strongly typed admitted-host facade over the shared weighted cache core.
#[derive(Debug)]
pub(crate) struct AdmittedHostObjectCache<K, T> {
    core: SingleFlightWeightedCache<K, AdmittedHostObject<T>>,
}

impl<K, T> Clone for AdmittedHostObjectCache<K, T> {
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
        }
    }
}

impl<K, T> AdmittedHostObjectCache<K, T>
where
    K: Clone + Eq + Hash,
{
    pub(crate) fn new(limits: AdmittedHostObjectCacheLimits) -> Self {
        Self {
            core: SingleFlightWeightedCache::new(limits),
        }
    }

    pub(crate) fn get_or_try_insert_with<E, Q, F, M, A>(
        &self,
        key: K,
        quote: Q,
        build: F,
        map_poisoned_lock: M,
    ) -> Result<AdmittedHostObject<T>, E>
    where
        Q: FnOnce() -> Result<(u64, A), E>,
        F: FnOnce(A) -> Result<AdmittedHostObject<T>, E>,
        M: Fn() -> E,
    {
        match self
            .core
            .lookup_or_reserve(key, None)
            .map_err(|_| map_poisoned_lock())?
        {
            SingleFlightWeightedLookup::Ready(value) => Ok(value),
            SingleFlightWeightedLookup::Build(permit) => {
                let (quoted_weight, allocation_quote) = quote()?;
                let retain = permit
                    .make_room_for(quoted_weight)
                    .map_err(|_| map_poisoned_lock())?;
                let value = build(allocation_quote)?;
                let actual_weight = value.committed_requested_bytes();
                permit
                    .publish(Arc::clone(&value), actual_weight, retain)
                    .map_err(|_| map_poisoned_lock())?;
                Ok(value)
            }
        }
    }

    pub(crate) fn ready(&self, key: &K) -> Option<AdmittedHostObject<T>> {
        self.core.ready(key)
    }

    pub(crate) fn clear(&self) {
        self.core.clear();
    }

    pub(crate) fn evict(&self, key: &K) {
        self.core.evict(key);
    }

    pub(crate) fn evict_where(&self, predicate: impl FnMut(&K) -> bool) {
        self.core.evict_where(predicate);
    }

    #[cfg(test)]
    pub(crate) fn usage_for_test(&self) -> (usize, u64) {
        self.core.usage_for_test()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::system_memory_owner::SystemMemoryOwner;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    fn owner(value: usize, bytes: u64) -> AdmittedHostObject<usize> {
        Arc::new(SystemMemoryOwner::with_committed_requested_bytes_for_test(
            value, bytes,
        ))
    }

    fn get(
        cache: &AdmittedHostObjectCache<&'static str, usize>,
        key: &'static str,
        bytes: u64,
        builds: &AtomicUsize,
    ) -> AdmittedHostObject<usize> {
        cache
            .get_or_try_insert_with(
                key,
                || Ok::<_, ()>((bytes, ())),
                |()| {
                    let value = builds.fetch_add(1, Ordering::SeqCst) + 1;
                    Ok::<_, ()>(owner(value, bytes))
                },
                || (),
            )
            .expect("cache operation should succeed")
    }

    #[test]
    fn entry_budget_evicts_least_recently_used_ready_owner() {
        let cache = AdmittedHostObjectCache::new(AdmittedHostObjectCacheLimits::new(2, u64::MAX));
        let builds = AtomicUsize::new(0);
        drop(get(&cache, "a", 1, &builds));
        drop(get(&cache, "b", 1, &builds));
        drop(get(&cache, "a", 1, &builds));
        drop(get(&cache, "c", 1, &builds));
        assert_eq!(cache.usage_for_test(), (2, 2));
        drop(get(&cache, "a", 1, &builds));
        assert_eq!(builds.load(Ordering::SeqCst), 3, "MRU entry must hit");
        drop(get(&cache, "b", 1, &builds));
        assert_eq!(builds.load(Ordering::SeqCst), 4, "LRU entry must rebuild");
    }

    #[test]
    fn byte_budget_evicts_even_below_entry_limit() {
        let cache = AdmittedHostObjectCache::new(AdmittedHostObjectCacheLimits::new(8, 100));
        let builds = AtomicUsize::new(0);
        drop(get(&cache, "a", 60, &builds));
        drop(get(&cache, "b", 50, &builds));
        assert_eq!(cache.usage_for_test(), (1, 50));
        drop(get(&cache, "a", 60, &builds));
        assert_eq!(builds.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn oversized_owner_executes_but_is_not_cached() {
        let cache = AdmittedHostObjectCache::new(AdmittedHostObjectCacheLimits::new(4, 32));
        let builds = AtomicUsize::new(0);
        let first = get(&cache, "large", 64, &builds);
        assert_eq!(**first, 1);
        assert_eq!(cache.usage_for_test(), (0, 0));
        drop(first);
        drop(get(&cache, "large", 64, &builds));
        assert_eq!(builds.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn ready_probe_is_non_building_and_returns_only_published_owner() {
        let cache = AdmittedHostObjectCache::new(AdmittedHostObjectCacheLimits::new(2, 128));
        let builds = AtomicUsize::new(0);

        assert!(cache.ready(&"model").is_none());
        assert_eq!(
            cache.usage_for_test(),
            (0, 0),
            "probe must not reserve a slot"
        );

        let built = get(&cache, "model", 64, &builds);
        let ready = cache.ready(&"model").expect("published owner is visible");
        assert!(Arc::ptr_eq(&built, &ready));
        assert_eq!(builds.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn eviction_preserves_in_flight_owner_and_lease() {
        let cache = AdmittedHostObjectCache::new(AdmittedHostObjectCacheLimits::new(1, 64));
        let builds = AtomicUsize::new(0);
        let in_flight = get(&cache, "a", 64, &builds);
        drop(get(&cache, "b", 64, &builds));
        assert_eq!(**in_flight, 1);
        assert_eq!(cache.usage_for_test(), (1, 64));
        drop(in_flight);
    }

    struct DropProbe {
        core: Arc<SingleFlightWeightedCacheInner<&'static str, AdmittedHostObject<DropProbe>>>,
        saw_unlocked: Arc<AtomicBool>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.saw_unlocked
                .store(self.core.state.try_lock().is_ok(), Ordering::SeqCst);
        }
    }

    #[test]
    fn clear_destroys_owner_after_releasing_global_mutex() {
        let cache = AdmittedHostObjectCache::<&'static str, DropProbe>::new(
            AdmittedHostObjectCacheLimits::new(1, 64),
        );
        let core = Arc::clone(&cache.core.inner);
        let saw_unlocked = Arc::new(AtomicBool::new(false));
        drop(
            cache
                .get_or_try_insert_with(
                    "drop",
                    || Ok::<_, ()>((1, ())),
                    |()| {
                        Ok(Arc::new(
                            SystemMemoryOwner::with_committed_requested_bytes_for_test(
                                DropProbe {
                                    core: Arc::clone(&core),
                                    saw_unlocked: Arc::clone(&saw_unlocked),
                                },
                                1,
                            ),
                        ))
                    },
                    || (),
                )
                .expect("owner builds"),
        );
        cache.clear();
        assert!(saw_unlocked.load(Ordering::SeqCst));
    }

    #[test]
    fn abandoned_build_permit_restores_a_retryable_slot() {
        let cache = SingleFlightWeightedCache::<&'static str, usize>::new(
            AdmittedHostObjectCacheLimits::new(1, 1),
        );
        let permit = match cache
            .lookup_or_reserve("retry", None)
            .expect("lookup succeeds")
        {
            SingleFlightWeightedLookup::Build(permit) => permit,
            SingleFlightWeightedLookup::Ready(_) => panic!("first lookup cannot be ready"),
        };
        drop(permit);

        assert!(matches!(
            cache
                .lookup_or_reserve("retry", None)
                .expect("retry succeeds"),
            SingleFlightWeightedLookup::Build(_)
        ));
    }
}
