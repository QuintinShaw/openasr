use std::{
    collections::HashMap,
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

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
}

#[derive(Debug)]
struct ModelSessionAdmissionState {
    limit: NonZeroUsize,
    slots: HashMap<String, ModelSessionSlot>,
}

#[derive(Debug)]
struct ModelSessionSlot {
    semaphore: Arc<Semaphore>,
    active: usize,
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
    permit: Option<OwnedSemaphorePermit>,
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
                active: 0,
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
        slot.active += 1;
        drop(state);

        Ok(ModelSessionPermit {
            state: Arc::clone(&self.state),
            model_identity,
            permit: Some(permit),
        })
    }

    #[cfg(test)]
    fn active_slot_count(&self) -> usize {
        self.lock_state().slots.len()
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
        drop(permit);

        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let should_remove = match state.slots.get_mut(&self.model_identity) {
            Some(slot) => {
                slot.active = slot.active.saturating_sub(1);
                slot.active == 0
            }
            None => false,
        };
        if should_remove {
            state.slots.remove(&self.model_identity);
        }
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
