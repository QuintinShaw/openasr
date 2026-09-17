//! Realtime stages share one process-owned provisional memory domain.
//!
//! Sessions retain independent bounded audio queues and worker-local state.
//! Only a native operation owns this gate, never an idle session lifetime.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

#[derive(Debug, Default)]
pub(crate) struct RealtimeExecutionGate {
    occupied: Mutex<bool>,
    available: Condvar,
    async_available: tokio::sync::Notify,
}

pub(crate) struct RealtimeExecutionGuard(Arc<RealtimeExecutionGate>);

impl RealtimeExecutionGate {
    pub(crate) async fn enter_async(
        self: &Arc<Self>,
        canceled: impl Fn() -> bool,
    ) -> Option<RealtimeExecutionGuard> {
        loop {
            let notified = self.async_available.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if canceled() {
                return None;
            }
            {
                let mut occupied = self
                    .occupied
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                if !*occupied {
                    *occupied = true;
                    return Some(RealtimeExecutionGuard(Arc::clone(self)));
                }
            }
            // Waiting in the async task leaves no detached blocking waiter if
            // its transport is dropped before preparation can begin.
            let _ = tokio::time::timeout(Duration::from_millis(100), notified).await;
        }
    }

    pub(crate) fn enter(
        self: &Arc<Self>,
        canceled: impl Fn() -> bool,
    ) -> Option<RealtimeExecutionGuard> {
        let mut occupied = self
            .occupied
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        loop {
            if canceled() {
                return None;
            }
            if !*occupied {
                *occupied = true;
                return Some(RealtimeExecutionGuard(Arc::clone(self)));
            }
            // Releases the mutex while asleep; bounded wakeups observe an
            // abandoned/canceled worker even if another native call is stuck.
            occupied = self
                .available
                .wait_timeout(occupied, Duration::from_millis(100))
                .unwrap_or_else(|error| error.into_inner())
                .0;
        }
    }
}

impl Drop for RealtimeExecutionGuard {
    fn drop(&mut self) {
        *self
            .0
            .occupied
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = false;
        self.0.available.notify_one();
        self.0.async_available.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    };

    #[tokio::test]
    async fn dropped_async_waiter_does_not_claim_later_capacity() {
        let gate = Arc::new(RealtimeExecutionGate::default());
        let owner = gate.enter(|| false).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(20), gate.enter_async(|| false))
                .await
                .is_err()
        );
        drop(owner);
        let next = tokio::time::timeout(Duration::from_secs(2), gate.enter_async(|| false))
            .await
            .unwrap()
            .unwrap();
        drop(next);
        assert!(gate.enter(|| false).is_some());
    }

    #[test]
    fn waiting_operations_resume_after_release_without_closing_sessions() {
        let gate = Arc::new(RealtimeExecutionGate::default());
        let first = gate.enter(|| false).unwrap();
        let (send, recv) = mpsc::channel();
        let worker_gate = Arc::clone(&gate);
        let worker = std::thread::spawn(move || {
            let _second = worker_gate.enter(|| false).unwrap();
            send.send(()).unwrap();
        });
        assert!(recv.recv_timeout(Duration::from_millis(20)).is_err());
        drop(first);
        recv.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.join().unwrap();
        assert!(gate.enter(|| false).is_some());
    }

    #[test]
    fn cancellation_does_not_wait_for_a_stuck_owner_or_release_its_gate() {
        let gate = Arc::new(RealtimeExecutionGate::default());
        let first = gate.enter(|| false).unwrap();
        let canceled = Arc::new(AtomicBool::new(false));
        let worker_gate = Arc::clone(&gate);
        let worker_cancel = Arc::clone(&canceled);
        let (send, recv) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            send.send(
                worker_gate
                    .enter(|| worker_cancel.load(Ordering::Acquire))
                    .is_none(),
            )
            .unwrap();
        });
        canceled.store(true, Ordering::Release);
        assert!(recv.recv_timeout(Duration::from_secs(2)).unwrap());
        worker.join().unwrap();
        assert!(*gate.occupied.lock().unwrap());
        drop(first);
        assert!(gate.enter(|| false).is_some());
    }
}
