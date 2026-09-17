use std::{
    collections::{BTreeSet, HashMap},
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use openasr_core::NativeExecutionServices;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::realtime_execution::RealtimeExecutionGate;
use crate::remote_runtime_policy::RemoteRuntimePolicy;

/// Bounds concurrent native executions for one resolved runtime model identity.
///
/// Admission is deliberately non-blocking. Queuing a second heavyweight model
/// session would retain its request state while providing no useful progress;
/// callers receive a retryable overload error instead. Slots are removed after
/// their final permit is released so model switches cannot grow this registry
/// without bound.
#[derive(Clone, Debug)]
pub(crate) struct ModelSessionAdmission {
    state: Arc<Mutex<ModelSessionAdmissionState>>,
    execution_gate: Arc<RealtimeExecutionGate>,
}

#[derive(Debug)]
struct ModelSessionAdmissionState {
    limit: NonZeroUsize,
    slots: HashMap<String, ModelSessionSlot>,
}

#[derive(Debug)]
struct ModelSessionSlot {
    semaphore: Arc<Semaphore>,
    occupied: BTreeSet<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelSessionAdmissionError {
    pub(crate) model_identity: String,
    pub(crate) limit: NonZeroUsize,
}

/// RAII permit for a native model execution. It remains owned by the blocking
/// decode task or native streaming worker, so cancellation of the async caller
/// cannot release capacity while the model still executes.
#[derive(Debug)]
pub(crate) struct ModelSessionPermit {
    state: Arc<Mutex<ModelSessionAdmissionState>>,
    model_identity: String,
    slot_index: usize,
    permit: Option<OwnedSemaphorePermit>,
    execution_gate: Arc<RealtimeExecutionGate>,
}

impl ModelSessionPermit {
    pub(crate) fn execution_gate(&self) -> Arc<RealtimeExecutionGate> {
        Arc::clone(&self.execution_gate)
    }

    pub(crate) fn slot_index(&self) -> usize {
        self.slot_index
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeAdmissionKind {
    File,
    Realtime,
}

#[derive(Clone, Debug)]
pub struct NativeExecutionSupervisor {
    admission: ModelSessionAdmission,
    execution_services: Arc<NativeExecutionServices>,
    remote_policy: RemoteRuntimePolicy,
}

impl PartialEq for NativeExecutionSupervisor {
    fn eq(&self, other: &Self) -> bool {
        self.max_concurrent_sessions_per_model() == other.max_concurrent_sessions_per_model()
    }
}

impl Eq for NativeExecutionSupervisor {}

impl Default for NativeExecutionSupervisor {
    fn default() -> Self {
        Self::new(NonZeroUsize::new(1).expect("one is non-zero"))
    }
}

impl NativeExecutionSupervisor {
    pub fn new(max_concurrent_sessions_per_model: NonZeroUsize) -> Self {
        let execution_services = Arc::new(
            NativeExecutionServices::for_local_process()
                .expect("builtin native execution services must construct"),
        );
        Self::with_execution_services(max_concurrent_sessions_per_model, execution_services)
    }

    /// Constructs a supervisor around the process-owned native execution
    /// service root. Process hosts should use this constructor so offline,
    /// streaming, warm-up, eviction, and idle-unload paths share one scope.
    pub fn with_execution_services(
        max_concurrent_sessions_per_model: NonZeroUsize,
        execution_services: Arc<NativeExecutionServices>,
    ) -> Self {
        Self {
            admission: ModelSessionAdmission::new(max_concurrent_sessions_per_model),
            execution_services,
            remote_policy: RemoteRuntimePolicy::new(),
        }
    }

    pub fn execution_services(&self) -> &Arc<NativeExecutionServices> {
        &self.execution_services
    }

    pub(crate) fn remote_policy(&self) -> &RemoteRuntimePolicy {
        &self.remote_policy
    }

    pub(crate) fn realtime_execution_gate(&self) -> Arc<RealtimeExecutionGate> {
        Arc::clone(&self.admission.execution_gate)
    }

    pub(crate) fn try_acquire(
        &self,
        model_identity: impl Into<String>,
    ) -> Result<ModelSessionPermit, ModelSessionAdmissionError> {
        self.admission.try_acquire(model_identity)
    }

    pub(crate) fn has_active_sessions(&self) -> bool {
        self.admission.has_active_sessions()
    }

    pub fn max_concurrent_sessions_per_model(&self) -> NonZeroUsize {
        self.admission.limit()
    }
}

impl Default for ModelSessionAdmission {
    fn default() -> Self {
        Self::new(NonZeroUsize::new(1).expect("one is non-zero"))
    }
}

impl ModelSessionAdmission {
    pub(crate) fn new(limit: NonZeroUsize) -> Self {
        Self {
            state: Arc::new(Mutex::new(ModelSessionAdmissionState {
                limit,
                slots: HashMap::new(),
            })),
            execution_gate: Arc::new(RealtimeExecutionGate::default()),
        }
    }

    pub(crate) fn try_acquire(
        &self,
        model_identity: impl Into<String>,
    ) -> Result<ModelSessionPermit, ModelSessionAdmissionError> {
        let model_identity = model_identity.into();
        let mut state = self.lock_state();
        let limit = state.limit;
        let slot = state
            .slots
            .entry(model_identity.clone())
            .or_insert_with(|| ModelSessionSlot {
                semaphore: Arc::new(Semaphore::new(limit.get())),
                occupied: BTreeSet::new(),
            });
        let permit = match Arc::clone(&slot.semaphore).try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                return Err(ModelSessionAdmissionError {
                    model_identity,
                    limit,
                });
            }
        };
        // Pick the lowest free slot without allocating for the configured
        // upper bound. Stable bounded indices let streaming workers retain
        // their warm cache without serializing independent admitted sessions.
        let mut slot_index = 0;
        for &occupied in &slot.occupied {
            if occupied != slot_index {
                break;
            }
            slot_index += 1;
        }
        slot.occupied.insert(slot_index);
        drop(state);

        Ok(ModelSessionPermit {
            state: Arc::clone(&self.state),
            model_identity,
            slot_index,
            permit: Some(permit),
            execution_gate: Arc::clone(&self.execution_gate),
        })
    }

    pub(crate) fn has_active_sessions(&self) -> bool {
        self.lock_state()
            .slots
            .values()
            .any(|slot| !slot.occupied.is_empty())
    }

    #[cfg(test)]
    fn active_slot_count(&self) -> usize {
        self.lock_state().slots.len()
    }

    fn limit(&self) -> NonZeroUsize {
        self.lock_state().limit
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, ModelSessionAdmissionState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Drop for ModelSessionPermit {
    fn drop(&mut self) {
        let Some(permit) = self.permit.take() else {
            return;
        };
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let should_remove = match state.slots.get_mut(&self.model_identity) {
            Some(slot) => {
                slot.occupied.remove(&self.slot_index);
                slot.occupied.is_empty()
            }
            None => false,
        };
        if should_remove {
            state.slots.remove(&self.model_identity);
        }
        // Publish the free index and permit under the same registry lock.
        drop(permit);
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use tokio::sync::oneshot;

    use super::ModelSessionAdmission;

    fn admission(limit: usize) -> ModelSessionAdmission {
        ModelSessionAdmission::new(NonZeroUsize::new(limit).unwrap())
    }

    #[test]
    fn admitted_sessions_share_the_execution_gate_without_releasing_capacity() {
        let supervisor = super::NativeExecutionSupervisor::new(NonZeroUsize::new(2).unwrap());
        let first_session = supervisor.try_acquire("dual-track").unwrap();
        let second_session = supervisor.try_acquire("dual-track").unwrap();
        assert!(std::sync::Arc::ptr_eq(
            &first_session.execution_gate(),
            &second_session.execution_gate()
        ));
        assert!(std::sync::Arc::ptr_eq(
            &first_session.execution_gate(),
            &supervisor.realtime_execution_gate()
        ));
        drop(first_session.execution_gate().enter(|| false).unwrap());
        drop(second_session.execution_gate().enter(|| false).unwrap());
        assert!(
            supervisor.try_acquire("dual-track").is_err(),
            "finishing an operation must not release a session's admission"
        );
    }

    #[test]
    fn rejects_without_waiting_when_model_is_at_capacity() {
        let admission = admission(1);
        let _first = admission
            .try_acquire("native:whisper-small@pack-a")
            .unwrap();

        let error = admission
            .try_acquire("native:whisper-small@pack-a")
            .unwrap_err();

        assert_eq!(error.model_identity, "native:whisper-small@pack-a");
        assert_eq!(error.limit.get(), 1);
    }

    #[test]
    fn releases_capacity_and_prunes_idle_model_slot() {
        let admission = admission(1);
        let first = admission
            .try_acquire("native:whisper-small@pack-a")
            .unwrap();
        assert_eq!(admission.active_slot_count(), 1);

        drop(first);

        assert_eq!(admission.active_slot_count(), 0);
        assert!(admission.try_acquire("native:whisper-small@pack-a").is_ok());
    }

    #[test]
    fn has_active_sessions_tracks_live_permits() {
        let admission = admission(1);
        assert!(!admission.has_active_sessions());
        let first = admission
            .try_acquire("native:whisper-small@pack-a")
            .unwrap();
        assert!(admission.has_active_sessions());
        drop(first);
        assert!(!admission.has_active_sessions());
    }

    #[test]
    fn different_models_do_not_serialize_each_other() {
        let admission = admission(1);
        let _first = admission
            .try_acquire("native:whisper-small@pack-a")
            .unwrap();

        assert!(
            admission
                .try_acquire("native:qwen3-asr-0.6b@pack-b")
                .is_ok()
        );
    }

    #[test]
    fn configured_capacity_allows_that_many_sessions() {
        let admission = admission(2);
        let _first = admission
            .try_acquire("native:whisper-small@pack-a")
            .unwrap();
        let _second = admission
            .try_acquire("native:whisper-small@pack-a")
            .unwrap();

        assert!(
            admission
                .try_acquire("native:whisper-small@pack-a")
                .is_err()
        );
    }

    #[test]
    fn concurrent_slot_indices_are_distinct_bounded_and_reused() {
        let admission = admission(3);
        let first = admission.try_acquire("model").unwrap();
        let second = admission.try_acquire("model").unwrap();
        let third = admission.try_acquire("model").unwrap();
        assert_eq!(
            [first.slot_index(), second.slot_index(), third.slot_index()],
            [0, 1, 2]
        );
        assert!(admission.try_acquire("model").is_err());
        drop(second);
        let replacement = admission.try_acquire("model").unwrap();
        assert_eq!(replacement.slot_index(), 1);
        drop((first, third, replacement));
        assert_eq!(admission.active_slot_count(), 0);
        assert_eq!(admission.try_acquire("model").unwrap().slot_index(), 0);
    }

    #[test]
    fn panic_unwinds_and_releases_its_permit() {
        let admission = admission(1);
        let panic_admission = admission.clone();
        let result = std::thread::spawn(move || {
            let _permit = panic_admission
                .try_acquire("native:whisper-small@pack-a")
                .unwrap();
            panic!("test panic after model admission");
        })
        .join();

        assert!(result.is_err());
        assert!(admission.try_acquire("native:whisper-small@pack-a").is_ok());
    }

    #[tokio::test]
    async fn aborted_owner_releases_its_permit() {
        let admission = admission(1);
        let (acquired_tx, acquired_rx) = oneshot::channel();
        let (_hold_tx, hold_rx) = oneshot::channel::<()>();
        let task_admission = admission.clone();
        let task = tokio::spawn(async move {
            let _permit = task_admission
                .try_acquire("native:whisper-small@pack-a")
                .unwrap();
            let _ = acquired_tx.send(());
            let _ = hold_rx.await;
        });

        acquired_rx.await.unwrap();
        task.abort();
        let _ = task.await;

        assert!(admission.try_acquire("native:whisper-small@pack-a").is_ok());
    }
}
