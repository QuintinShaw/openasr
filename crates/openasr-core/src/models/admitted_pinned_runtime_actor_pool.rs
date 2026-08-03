//! Process-owned cache of runtimes pinned to dedicated owner threads.
//!
//! ggml backend contexts can contain `Rc` state and non-owning views into a
//! thread-local backend cache. Such a runtime is neither `Send` nor safely
//! destructible on an arbitrary request worker. This abstraction moves only a
//! build closure to a dedicated actor thread; the resulting
//! [`SystemMemoryOwner<R>`] is constructed, used, and deterministically dropped
//! on that same thread. Process callers cache and clone only a Send-safe command
//! handle.
//!
//! The cache delegates key single-flight, candidate-local staged visibility,
//! weighted LRU, and clear/evict no-resurrection semantics to the shared
//! [`SingleFlightWeightedCache`]. One actor serializes mutable calls for each
//! content/lane key, which is the deliberately finite per-key concurrency
//! bound. An evicted actor remains alive while an in-flight handle exists and
//! shuts down synchronously as soon as its last handle drops.

use std::any::Any;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};

use super::admitted_exclusive_object_pool::{
    AdmittedExclusiveObjectCheckout, AdmittedExclusiveObjectPool,
    AdmittedExclusiveObjectPoolLimits, AdmittedExclusivePoolOwner,
};
use super::admitted_host_object_cache::{
    AdmittedHostObjectCacheLimits, SingleFlightWeightedCache, SingleFlightWeightedLookup,
};
use super::native_execution_services::{
    current_execution_cache_attempt_id, current_native_execution_context,
    install_native_execution_context, stage_execution_cache_commit,
};
use super::system_memory_owner::SystemMemoryOwner;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PinnedRuntimeActorError {
    CachePoisoned,
    PoolFailure { reason: String },
    WorkerSpawnFailed { reason: String },
    BuildPanicked { message: String },
    OperationPanicked { message: String },
    ReentrantCall,
    WorkerTerminated,
}

impl fmt::Display for PinnedRuntimeActorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CachePoisoned => write!(f, "pinned runtime actor cache lock poisoned"),
            Self::PoolFailure { reason } => {
                write!(f, "pinned runtime actor checkout pool failed: {reason}")
            }
            Self::WorkerSpawnFailed { reason } => {
                write!(f, "pinned runtime actor worker spawn failed: {reason}")
            }
            Self::BuildPanicked { message } => {
                write!(f, "pinned runtime actor build panicked: {message}")
            }
            Self::OperationPanicked { message } => {
                write!(f, "pinned runtime actor operation panicked: {message}")
            }
            Self::ReentrantCall => write!(
                f,
                "pinned runtime actor cannot synchronously call itself from its owner thread"
            ),
            Self::WorkerTerminated => write!(f, "pinned runtime actor worker terminated"),
        }
    }
}

impl std::error::Error for PinnedRuntimeActorError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdmittedPinnedRuntimeActorPoolLimits {
    pub(crate) max_entries: usize,
    pub(crate) max_committed_requested_bytes: u64,
}

impl AdmittedPinnedRuntimeActorPoolLimits {
    pub(crate) const fn new(max_entries: usize, max_committed_requested_bytes: u64) -> Self {
        Self {
            max_entries,
            max_committed_requested_bytes,
        }
    }
}

type ActorJob<R> = Box<dyn FnOnce(&mut R) -> bool + Send + 'static>;

enum ActorCommand<R> {
    Run(ActorJob<R>),
    Shutdown,
}

struct PinnedRuntimeActorInner<R: 'static> {
    sender: mpsc::Sender<ActorCommand<R>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    alive: Arc<AtomicBool>,
    worker_thread_id: thread::ThreadId,
    committed_requested_bytes: u64,
    // Function-position phantom does not inherit R's Send/Sync auto traits.
    // The runtime itself never enters this process-side allocation.
    _runtime: PhantomData<fn() -> R>,
}

impl<R: 'static> fmt::Debug for PinnedRuntimeActorInner<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinnedRuntimeActorInner")
            .field("alive", &self.alive.load(Ordering::Acquire))
            .field("committed_requested_bytes", &self.committed_requested_bytes)
            .finish_non_exhaustive()
    }
}

impl<R: 'static> Drop for PinnedRuntimeActorInner<R> {
    fn drop(&mut self) {
        // Last-handle drop means no caller can enqueue another operation. The
        // shutdown message therefore follows every already-accepted job. Join
        // is load-bearing: it proves R and its memory lease were destroyed on
        // the owner thread before clear/evict returns.
        let _ = self.sender.send(ActorCommand::Shutdown);
        let worker = self
            .worker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(worker) = worker {
            if thread::current().id() == self.worker_thread_id {
                // A queued operation is allowed to own the final process-side
                // handle. Joining here would self-deadlock; dropping a
                // JoinHandle detaches, and the Shutdown queued above makes the
                // worker drop R immediately after the current job returns.
                drop(worker);
            } else {
                let _ = worker.join();
            }
        }
    }
}

/// Cloneable process-side command handle. `R` is intentionally allowed to be
/// `!Send` and `!Sync`; only operation closures and results cross the channel.
pub(crate) struct PinnedRuntimeActor<R: 'static> {
    inner: Arc<PinnedRuntimeActorInner<R>>,
}

impl<R: 'static> Clone for PinnedRuntimeActor<R> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<R: 'static> fmt::Debug for PinnedRuntimeActor<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PinnedRuntimeActor")
            .field(&self.inner)
            .finish()
    }
}

impl<R: 'static> PinnedRuntimeActor<R> {
    fn spawn<E, F>(worker_name: &str, build: F) -> Result<Self, ActorBuildError<E>>
    where
        E: Send + 'static,
        F: FnOnce() -> Result<SystemMemoryOwner<R>, E> + Send + 'static,
    {
        let (command_tx, command_rx) = mpsc::channel::<ActorCommand<R>>();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let alive = Arc::new(AtomicBool::new(true));
        let worker_alive = Arc::clone(&alive);
        let build_context = current_native_execution_context();
        let worker = thread::Builder::new()
            .name(worker_name.to_string())
            .spawn(move || {
                let built = panic::catch_unwind(AssertUnwindSafe(|| {
                    let _context = build_context.map(install_native_execution_context);
                    build()
                }));
                let mut owner = match built {
                    Ok(Ok(owner)) => owner,
                    Ok(Err(error)) => {
                        let _ = ready_tx.send(Err(ActorBuildError::Build(error)));
                        worker_alive.store(false, Ordering::Release);
                        return;
                    }
                    Err(payload) => {
                        let _ = ready_tx.send(Err(ActorBuildError::Actor(
                            PinnedRuntimeActorError::BuildPanicked {
                                message: describe_panic_payload(payload.as_ref()),
                            },
                        )));
                        worker_alive.store(false, Ordering::Release);
                        return;
                    }
                };
                if ready_tx
                    .send(Ok(owner.committed_requested_bytes()))
                    .is_err()
                {
                    worker_alive.store(false, Ordering::Release);
                    return;
                }

                while let Ok(command) = command_rx.recv() {
                    match command {
                        ActorCommand::Run(job) => {
                            if !job(&mut owner) {
                                break;
                            }
                        }
                        ActorCommand::Shutdown => break,
                    }
                }
                // `owner` (R first, lease second) is dropped right here, on
                // the same thread that constructed every backend context.
                drop(owner);
                worker_alive.store(false, Ordering::Release);
            })
            .map_err(|error| {
                ActorBuildError::Actor(PinnedRuntimeActorError::WorkerSpawnFailed {
                    reason: error.to_string(),
                })
            })?;

        let committed_requested_bytes = match ready_rx.recv() {
            Ok(Ok(bytes)) => bytes,
            Ok(Err(error)) => {
                drop(command_tx);
                let _ = worker.join();
                return Err(error);
            }
            Err(_) => {
                drop(command_tx);
                let _ = worker.join();
                return Err(ActorBuildError::Actor(
                    PinnedRuntimeActorError::WorkerTerminated,
                ));
            }
        };
        let worker_thread_id = worker.thread().id();
        Ok(Self {
            inner: Arc::new(PinnedRuntimeActorInner {
                sender: command_tx,
                worker: Mutex::new(Some(worker)),
                alive,
                worker_thread_id,
                committed_requested_bytes,
                _runtime: PhantomData,
            }),
        })
    }

    pub(crate) fn committed_requested_bytes(&self) -> u64 {
        self.inner.committed_requested_bytes
    }

    fn is_alive(&self) -> bool {
        self.inner.alive.load(Ordering::Acquire)
    }

    /// Runs one operation on the runtime's owner thread. A panicking operation
    /// terminates that actor rather than reusing potentially corrupted mutable
    /// state; callers receive a typed error and the cache can evict/rebuild it.
    pub(crate) fn call_mut<O, F>(&self, operation: F) -> Result<O, PinnedRuntimeActorError>
    where
        O: Send + 'static,
        F: FnOnce(&mut R) -> O + Send + 'static,
    {
        self.call_mut_async(operation)?.join()
    }

    /// Runs a fallible mutation and terminates the actor when the runtime
    /// reports an error. This is for backend owners whose failed operation may
    /// have poisoned native handles. The typed model error is preserved; the
    /// next cache lookup observes the dead actor and rebuilds it transactionally.
    pub(crate) fn call_mut_fallible<O, E, F>(
        &self,
        operation: F,
    ) -> Result<Result<O, E>, PinnedRuntimeActorError>
    where
        O: Send + 'static,
        E: Send + 'static,
        F: FnOnce(&mut R) -> Result<O, E> + Send + 'static,
    {
        self.call_mut_fallible_async(operation)?.join()
    }

    fn call_mut_fallible_async<O, E, F>(
        &self,
        operation: F,
    ) -> Result<PinnedRuntimeActorCall<Result<O, E>>, PinnedRuntimeActorError>
    where
        O: Send + 'static,
        E: Send + 'static,
        F: FnOnce(&mut R) -> Result<O, E> + Send + 'static,
    {
        if thread::current().id() == self.inner.worker_thread_id {
            return Err(PinnedRuntimeActorError::ReentrantCall);
        }
        if !self.inner.alive.load(Ordering::Acquire) {
            return Err(PinnedRuntimeActorError::WorkerTerminated);
        }
        let context = current_native_execution_context();
        let alive = Arc::clone(&self.inner.alive);
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        let job = Box::new(move |runtime: &mut R| {
            let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
                let _context = context.map(install_native_execution_context);
                operation(runtime)
            }));
            match outcome {
                Ok(result) => {
                    let keep_alive = result.is_ok();
                    if !keep_alive {
                        alive.store(false, Ordering::Release);
                    }
                    let _ = response_tx.send(ActorCallResponse::Completed(result));
                    keep_alive
                }
                Err(payload) => {
                    alive.store(false, Ordering::Release);
                    let _ = response_tx.send(ActorCallResponse::Panicked(describe_panic_payload(
                        payload.as_ref(),
                    )));
                    false
                }
            }
        });
        self.inner
            .sender
            .send(ActorCommand::Run(job))
            .map_err(|_| PinnedRuntimeActorError::WorkerTerminated)?;
        Ok(PinnedRuntimeActorCall {
            response: response_rx,
        })
    }

    /// Enqueues one owner-thread mutation and returns before it completes.
    /// This is intentionally narrower than a general task executor: FIFO actor
    /// ordering, panic poisoning, native execution-context propagation, and
    /// owner-thread destruction are identical to [`Self::call_mut`].
    fn call_mut_async<O, F>(
        &self,
        operation: F,
    ) -> Result<PinnedRuntimeActorCall<O>, PinnedRuntimeActorError>
    where
        O: Send + 'static,
        F: FnOnce(&mut R) -> O + Send + 'static,
    {
        // A synchronous channel round-trip to this same owner thread can never
        // make progress, and allowing self-enqueue would make completion depend
        // on the operation never joining its own queued child. Reject both forms
        // uniformly instead of admitting a latent owner-thread deadlock.
        if thread::current().id() == self.inner.worker_thread_id {
            return Err(PinnedRuntimeActorError::ReentrantCall);
        }
        if !self.inner.alive.load(Ordering::Acquire) {
            return Err(PinnedRuntimeActorError::WorkerTerminated);
        }
        let context = current_native_execution_context();
        let alive = Arc::clone(&self.inner.alive);
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        let job = Box::new(move |runtime: &mut R| {
            let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
                let _context = context.map(install_native_execution_context);
                operation(runtime)
            }));
            match outcome {
                Ok(output) => {
                    let _ = response_tx.send(ActorCallResponse::Completed(output));
                    true
                }
                Err(payload) => {
                    // Publish poisoning before waking the caller. The caller
                    // may immediately drop an exclusive checkout, whose pool
                    // return path consults `is_reusable`; responding first
                    // creates a race that can cache a poisoned actor.
                    alive.store(false, Ordering::Release);
                    let _ = response_tx.send(ActorCallResponse::Panicked(describe_panic_payload(
                        payload.as_ref(),
                    )));
                    false
                }
            }
        });
        self.inner
            .sender
            .send(ActorCommand::Run(job))
            .map_err(|_| PinnedRuntimeActorError::WorkerTerminated)?;
        Ok(PinnedRuntimeActorCall {
            response: response_rx,
        })
    }
}

impl<R: 'static> AdmittedExclusivePoolOwner for PinnedRuntimeActor<R> {
    fn committed_requested_bytes(&self) -> u64 {
        self.committed_requested_bytes()
    }

    fn is_reusable(&self) -> bool {
        self.inner.alive.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
enum ActorBuildError<E> {
    Build(E),
    Actor(PinnedRuntimeActorError),
}

/// Limits for families that need more than one mutable runtime for the same
/// content/lane key. `max_instances_per_key` bounds active plus idle actors;
/// callers beyond that bound wait for an exclusive checkout rather than
/// serializing all inference calls through one actor or starting an unbounded
/// build storm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdmittedPinnedRuntimeActorCheckoutPoolLimits {
    pub(crate) max_idle_entries: usize,
    pub(crate) max_idle_committed_requested_bytes: u64,
    pub(crate) max_instances_per_key: usize,
}

impl AdmittedPinnedRuntimeActorCheckoutPoolLimits {
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

pub(crate) type PinnedRuntimeActorCheckout<K, R> =
    AdmittedExclusiveObjectCheckout<K, PinnedRuntimeActor<R>>;

/// Pending operation that retains the exclusive checkout until completion, so
/// the pool cannot hand the same actor to another caller while work is queued.
pub(crate) struct CheckedOutPinnedRuntimeActorCall<K, R: 'static, O>
where
    K: Clone + Eq + Hash + Send + 'static,
{
    call: PinnedRuntimeActorCall<O>,
    checkout: PinnedRuntimeActorCheckout<K, R>,
}

impl<K, R, O> CheckedOutPinnedRuntimeActorCall<K, R, O>
where
    K: Clone + Eq + Hash + Send + 'static,
    R: 'static,
{
    pub(crate) fn join(self) -> Result<O, PinnedRuntimeActorError> {
        let Self { call, checkout } = self;
        let result = call.join();
        drop(checkout);
        result
    }
}

pub(crate) fn call_checked_out_actor_mut_async<K, R, O, F>(
    checkout: PinnedRuntimeActorCheckout<K, R>,
    operation: F,
) -> Result<CheckedOutPinnedRuntimeActorCall<K, R, O>, PinnedRuntimeActorError>
where
    K: Clone + Eq + Hash + Send + 'static,
    R: 'static,
    O: Send + 'static,
    F: FnOnce(&mut R) -> O + Send + 'static,
{
    let call = checkout.call_mut_async(operation)?;
    Ok(CheckedOutPinnedRuntimeActorCall { call, checkout })
}

/// Fallible counterpart to [`call_checked_out_actor_mut_async`]. A model
/// failure terminates the checked-out actor before the checkout returns to the
/// pool, so a potentially poisoned native runtime can never be reused. The
/// operation is still queued from the candidate-owning caller thread, which
/// preserves the exact native execution context while allowing several
/// independent actors to run concurrently.
pub(crate) fn call_checked_out_actor_mut_fallible_async<K, R, O, E, F>(
    checkout: PinnedRuntimeActorCheckout<K, R>,
    operation: F,
) -> Result<CheckedOutPinnedRuntimeActorCall<K, R, Result<O, E>>, PinnedRuntimeActorError>
where
    K: Clone + Eq + Hash + Send + 'static,
    R: 'static,
    O: Send + 'static,
    E: Send + 'static,
    F: FnOnce(&mut R) -> Result<O, E> + Send + 'static,
{
    let call = checkout.call_mut_fallible_async(operation)?;
    Ok(CheckedOutPinnedRuntimeActorCall { call, checkout })
}

/// Per-service-root pool of exclusively checked-out owner-thread runtimes.
///
/// This is the parallel-session counterpart to [`AdmittedPinnedRuntimeActorPool`].
/// The latter deliberately shares one actor per key and is appropriate for a
/// component whose own scheduling already guarantees one worker. This pool
/// retains a bounded idle LRU while allowing a configured finite number of
/// actors for the same key to execute concurrently.
#[derive(Debug)]
pub(crate) struct AdmittedPinnedRuntimeActorCheckoutPool<K, R: 'static> {
    pool: AdmittedExclusiveObjectPool<K, PinnedRuntimeActor<R>>,
    worker_name: &'static str,
}

impl<K, R> AdmittedPinnedRuntimeActorCheckoutPool<K, R>
where
    K: Clone + Eq + Hash + Send + 'static,
    R: 'static,
{
    pub(crate) fn new(
        worker_name: &'static str,
        limits: AdmittedPinnedRuntimeActorCheckoutPoolLimits,
    ) -> Self {
        Self {
            pool: AdmittedExclusiveObjectPool::new(AdmittedExclusiveObjectPoolLimits::new(
                limits.max_idle_entries,
                limits.max_idle_committed_requested_bytes,
                limits.max_instances_per_key,
            )),
            worker_name,
        }
    }

    pub(crate) fn checkout_or_try_build_with<E, A, Q, F, M>(
        &self,
        key: K,
        quote: Q,
        build: F,
        map_actor_error: M,
    ) -> Result<PinnedRuntimeActorCheckout<K, R>, E>
    where
        E: Send + 'static,
        A: Send + 'static,
        Q: FnOnce() -> Result<(u64, A), E>,
        F: FnOnce(A) -> Result<SystemMemoryOwner<R>, E> + Send + 'static,
        M: Fn(PinnedRuntimeActorError) -> E,
    {
        let worker_name = self.worker_name;
        let mapper = &map_actor_error;
        self.pool.checkout_or_try_build(
            key,
            quote,
            move |allocation_quote| {
                PinnedRuntimeActor::spawn(worker_name, move || build(allocation_quote)).map_err(
                    |error| match error {
                        ActorBuildError::Build(error) => error,
                        ActorBuildError::Actor(error) => mapper(error),
                    },
                )
            },
            |reason| mapper(PinnedRuntimeActorError::PoolFailure { reason }),
        )
    }

    pub(crate) fn clear(&self) {
        self.pool.clear();
    }

    pub(crate) fn evict_where(&self, predicate: impl FnMut(&K) -> bool) {
        self.pool.evict_where(predicate);
    }

    #[cfg(test)]
    pub(crate) fn usage_for_test(&self) -> (usize, u64) {
        self.pool.usage_for_test()
    }
}

enum ActorCallResponse<O> {
    Completed(O),
    Panicked(String),
}

/// A queued actor operation whose caller may perform independent work before
/// joining. Dropping the handle does not cancel an accepted mutation; the
/// owner thread still runs it in FIFO order and preserves runtime invariants.
struct PinnedRuntimeActorCall<O> {
    response: mpsc::Receiver<ActorCallResponse<O>>,
}

impl<O> PinnedRuntimeActorCall<O> {
    pub(crate) fn join(self) -> Result<O, PinnedRuntimeActorError> {
        match self.response.recv() {
            Ok(ActorCallResponse::Completed(output)) => Ok(output),
            Ok(ActorCallResponse::Panicked(message)) => {
                Err(PinnedRuntimeActorError::OperationPanicked { message })
            }
            Err(_) => Err(PinnedRuntimeActorError::WorkerTerminated),
        }
    }
}

/// Owned per-service-root cache. Scope identity is implicit in ownership; keys
/// need only contain pack content identity and the exact execution lane.
#[derive(Debug)]
pub(crate) struct AdmittedPinnedRuntimeActorPool<K, R: 'static> {
    cache: SingleFlightWeightedCache<K, PinnedRuntimeActor<R>>,
    worker_name: &'static str,
}

impl<K, R> AdmittedPinnedRuntimeActorPool<K, R>
where
    K: Clone + Eq + Hash + 'static,
    R: 'static,
{
    pub(crate) fn new(
        worker_name: &'static str,
        limits: AdmittedPinnedRuntimeActorPoolLimits,
    ) -> Self {
        Self {
            cache: SingleFlightWeightedCache::new(AdmittedHostObjectCacheLimits::new(
                limits.max_entries,
                limits.max_committed_requested_bytes,
            )),
            worker_name,
        }
    }

    pub(crate) fn get_or_try_insert_with<E, A, Q, F, M>(
        &self,
        key: K,
        quote: Q,
        build: F,
        map_actor_error: M,
    ) -> Result<PinnedRuntimeActor<R>, E>
    where
        E: Send + 'static,
        A: Send + 'static,
        Q: FnOnce() -> Result<(u64, A), E>,
        F: FnOnce(A) -> Result<SystemMemoryOwner<R>, E> + Send + 'static,
        M: Fn(PinnedRuntimeActorError) -> E,
    {
        let attempt_id = current_execution_cache_attempt_id();
        let mut quote = Some(quote);
        let mut build = Some(build);
        loop {
            match self
                .cache
                .lookup_or_reserve(key.clone(), attempt_id)
                .map_err(|_| map_actor_error(PinnedRuntimeActorError::CachePoisoned))?
            {
                SingleFlightWeightedLookup::Ready(actor) if actor.is_alive() => return Ok(actor),
                SingleFlightWeightedLookup::Ready(_) => {
                    // A panicking operation terminates the owner thread. Do
                    // not hand that poisoned handle to every later caller.
                    self.cache.evict(&key);
                }
                SingleFlightWeightedLookup::Build(permit) => {
                    let (quoted_weight, allocation_quote) = quote
                        .take()
                        .expect("actor quote is consumed by one acquired build slot")(
                    )?;
                    let retain = permit
                        .make_room_for(quoted_weight)
                        .map_err(|_| map_actor_error(PinnedRuntimeActorError::CachePoisoned))?;
                    let actor = PinnedRuntimeActor::spawn(self.worker_name, move || {
                        build
                            .take()
                            .expect("actor build is consumed by one acquired build slot")(
                            allocation_quote,
                        )
                    })
                    .map_err(|error| match error {
                        ActorBuildError::Build(error) => error,
                        ActorBuildError::Actor(error) => map_actor_error(error),
                    })?;
                    let actual_weight = actor.committed_requested_bytes();
                    if let Some(attempt_id) = attempt_id {
                        let publication = permit
                            .stage(actor.clone(), actual_weight, retain, attempt_id)
                            .map_err(|_| map_actor_error(PinnedRuntimeActorError::CachePoisoned))?;
                        stage_execution_cache_commit(move || {
                            let _ = publication.commit();
                        });
                    } else {
                        permit
                            .publish(actor.clone(), actual_weight, retain)
                            .map_err(|_| map_actor_error(PinnedRuntimeActorError::CachePoisoned))?;
                    }
                    return Ok(actor);
                }
            }
        }
    }

    pub(crate) fn clear(&self) {
        self.cache.clear();
    }

    pub(crate) fn evict_where(&self, predicate: impl FnMut(&K) -> bool) {
        self.cache.evict_where(predicate);
    }
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
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct ThreadPinnedRuntime {
        value: Rc<Cell<usize>>,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for ThreadPinnedRuntime {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn owner(
        value: usize,
        bytes: u64,
        drops: Arc<AtomicUsize>,
    ) -> SystemMemoryOwner<ThreadPinnedRuntime> {
        SystemMemoryOwner::with_committed_requested_bytes_for_test(
            ThreadPinnedRuntime {
                value: Rc::new(Cell::new(value)),
                drops,
            },
            bytes,
        )
    }

    fn checkout_actor(
        pool: &AdmittedPinnedRuntimeActorCheckoutPool<&'static str, ThreadPinnedRuntime>,
        key: &'static str,
        builds: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    ) -> Result<PinnedRuntimeActorCheckout<&'static str, ThreadPinnedRuntime>, String> {
        pool.checkout_or_try_build_with(
            key,
            || Ok((32, ())),
            move |()| {
                let value = builds.fetch_add(1, Ordering::SeqCst) + 1;
                Ok(owner(value, 32, drops))
            },
            |error| error.to_string(),
        )
    }

    #[test]
    fn exclusive_actor_pool_allows_finite_parallel_instances_and_reuses_them() {
        let pool = Arc::new(AdmittedPinnedRuntimeActorCheckoutPool::new(
            "pinned-checkout-parallel-test",
            AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(2, 64, 2),
        ));
        let builds = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let first = checkout_actor(&pool, "same", Arc::clone(&builds), Arc::clone(&drops))
            .expect("first checkout");
        let second = checkout_actor(&pool, "same", Arc::clone(&builds), Arc::clone(&drops))
            .expect("second checkout");
        assert_eq!(builds.load(Ordering::SeqCst), 2);

        let barrier = Arc::new(Barrier::new(3));
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let run = |checkout: PinnedRuntimeActorCheckout<_, _>| {
            let barrier = Arc::clone(&barrier);
            let active = Arc::clone(&active);
            let maximum = Arc::clone(&maximum);
            thread::spawn(move || {
                checkout
                    .call_mut(move |_| {
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        maximum.fetch_max(now, Ordering::SeqCst);
                        barrier.wait();
                        active.fetch_sub(1, Ordering::SeqCst);
                    })
                    .expect("parallel actor call");
                checkout
            })
        };
        let first_worker = run(first);
        let second_worker = run(second);
        barrier.wait();
        let first = first_worker.join().expect("first caller");
        let second = second_worker.join().expect("second caller");
        assert_eq!(maximum.load(Ordering::SeqCst), 2);

        let waiting_pool = Arc::clone(&pool);
        let waiting_builds = Arc::clone(&builds);
        let waiting_drops = Arc::clone(&drops);
        let (acquired_tx, acquired_rx) = mpsc::sync_channel(1);
        let waiter = thread::spawn(move || {
            let checkout = checkout_actor(&waiting_pool, "same", waiting_builds, waiting_drops)
                .expect("waiting checkout");
            acquired_tx.send(()).expect("report checkout");
            checkout
        });
        assert!(matches!(
            acquired_rx.recv_timeout(Duration::from_millis(30)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        drop(first);
        acquired_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("returned actor wakes waiter");
        let third = waiter.join().expect("waiting caller");
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        drop(second);
        drop(third);
        assert_eq!(pool.usage_for_test(), (2, 64));
        pool.clear();
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn panicked_checked_out_actor_is_destroyed_instead_of_returned_idle() {
        let pool = AdmittedPinnedRuntimeActorCheckoutPool::new(
            "pinned-checkout-panic-test",
            AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(1, 32, 1),
        );
        let builds = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let actor = checkout_actor(&pool, "same", Arc::clone(&builds), Arc::clone(&drops))
            .expect("first checkout");
        assert!(matches!(
            actor.call_mut::<(), _>(|_| panic!("poison")),
            Err(PinnedRuntimeActorError::OperationPanicked { .. })
        ));
        drop(actor);
        assert_eq!(pool.usage_for_test(), (0, 0));

        let rebuilt = checkout_actor(&pool, "same", Arc::clone(&builds), Arc::clone(&drops))
            .expect("poisoned actor must release its permit");
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        assert_eq!(rebuilt.call_mut(|runtime| runtime.value.get()).unwrap(), 2);
    }

    #[test]
    fn asynchronous_call_allows_independent_work_before_join() {
        let pool = AdmittedPinnedRuntimeActorCheckoutPool::new(
            "pinned-checkout-async-test",
            AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(1, 32, 1),
        );
        let actor = checkout_actor(
            &pool,
            "same",
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        )
        .expect("actor checkout");
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let pending = call_checked_out_actor_mut_async(actor, move |runtime| {
            started_tx.send(()).expect("report start");
            release_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("caller releases async operation");
            runtime.value.set(9);
            runtime.value.get()
        })
        .expect("enqueue async operation");
        started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("operation starts");
        release_tx.send(()).expect("release operation");
        assert_eq!(pending.join().expect("join async operation"), 9);
    }

    #[test]
    fn asynchronous_call_reports_panic_at_join_and_poisons_actor() {
        let pool = AdmittedPinnedRuntimeActorCheckoutPool::new(
            "pinned-checkout-async-panic-test",
            AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(1, 32, 1),
        );
        let builds = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let actor = checkout_actor(&pool, "same", Arc::clone(&builds), Arc::clone(&drops))
            .expect("actor checkout");
        let pending =
            call_checked_out_actor_mut_async::<_, _, (), _>(actor, |_| panic!("async poison"))
                .expect("enqueue panicking operation");
        assert!(matches!(
            pending.join(),
            Err(PinnedRuntimeActorError::OperationPanicked { .. })
        ));
        let rebuilt = checkout_actor(&pool, "same", Arc::clone(&builds), Arc::clone(&drops))
            .expect("poisoned async actor must be rebuilt");
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        assert_eq!(rebuilt.call_mut(|runtime| runtime.value.get()).unwrap(), 2);
    }

    #[test]
    fn asynchronous_fallible_call_preserves_error_and_rebuilds_actor() {
        let pool = AdmittedPinnedRuntimeActorCheckoutPool::new(
            "pinned-checkout-async-fallible-test",
            AdmittedPinnedRuntimeActorCheckoutPoolLimits::new(1, 32, 1),
        );
        let builds = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let actor = checkout_actor(&pool, "same", Arc::clone(&builds), Arc::clone(&drops))
            .expect("actor checkout");

        let pending =
            call_checked_out_actor_mut_fallible_async(actor, |_| Err::<usize, _>("device lost"))
                .expect("enqueue fallible operation");
        assert_eq!(pending.join().expect("actor transport"), Err("device lost"));

        let rebuilt = checkout_actor(&pool, "same", Arc::clone(&builds), Arc::clone(&drops))
            .expect("failed actor must be rebuilt");
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        assert_eq!(rebuilt.call_mut(|runtime| runtime.value.get()).unwrap(), 2);
    }

    #[test]
    fn same_key_build_is_single_flight_and_runtime_never_crosses_threads() {
        let pool = Arc::new(AdmittedPinnedRuntimeActorPool::new(
            "pinned-singleflight-test",
            AdmittedPinnedRuntimeActorPoolLimits::new(2, 128),
        ));
        let builds = Arc::new(AtomicUsize::new(0));
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::new();
        for _ in 0..2 {
            let pool = Arc::clone(&pool);
            let builds = Arc::clone(&builds);
            let active = Arc::clone(&active);
            let maximum = Arc::clone(&maximum);
            let drops = Arc::clone(&drops);
            workers.push(thread::spawn(move || {
                let actor = pool
                    .get_or_try_insert_with(
                        "same",
                        || Ok::<_, String>((32, ())),
                        move |()| {
                            builds.fetch_add(1, Ordering::SeqCst);
                            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                            maximum.fetch_max(now, Ordering::SeqCst);
                            thread::sleep(Duration::from_millis(20));
                            active.fetch_sub(1, Ordering::SeqCst);
                            Ok(owner(7, 32, drops))
                        },
                        |error| error.to_string(),
                    )
                    .unwrap();
                actor
                    .call_mut(|runtime| runtime.value.get())
                    .expect("actor call")
            }));
        }
        for worker in workers {
            assert_eq!(worker.join().unwrap(), 7);
        }
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert_eq!(maximum.load(Ordering::SeqCst), 1);
        pool.clear();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn weighted_lru_and_clear_drop_idle_runtime_before_returning() {
        let pool = AdmittedPinnedRuntimeActorPool::new(
            "pinned-eviction-test",
            AdmittedPinnedRuntimeActorPoolLimits::new(1, 64),
        );
        let drops = Arc::new(AtomicUsize::new(0));
        let first = pool
            .get_or_try_insert_with(
                "a",
                || Ok::<_, String>((64, ())),
                {
                    let drops = Arc::clone(&drops);
                    move |()| Ok(owner(1, 64, drops))
                },
                |error| error.to_string(),
            )
            .unwrap();
        drop(first);
        let second = pool
            .get_or_try_insert_with(
                "b",
                || Ok::<_, String>((64, ())),
                {
                    let drops = Arc::clone(&drops);
                    move |()| Ok(owner(2, 64, drops))
                },
                |error| error.to_string(),
            )
            .unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 1, "LRU drop is joined");
        drop(second);
        pool.clear();
        assert_eq!(drops.load(Ordering::SeqCst), 2, "clear drop is joined");
    }

    #[test]
    fn failed_and_panicking_builds_leave_key_retryable() {
        let pool = AdmittedPinnedRuntimeActorPool::new(
            "pinned-build-failure-test",
            AdmittedPinnedRuntimeActorPoolLimits::new(1, 64),
        );
        let failed = pool.get_or_try_insert_with(
            "retry",
            || Ok::<_, String>((16, ())),
            |()| Err::<SystemMemoryOwner<ThreadPinnedRuntime>, _>("failed".to_string()),
            |error| error.to_string(),
        );
        assert!(matches!(failed, Err(ref reason) if reason == "failed"));
        let panicked = pool.get_or_try_insert_with(
            "retry",
            || Ok::<_, String>((16, ())),
            |()| -> Result<SystemMemoryOwner<ThreadPinnedRuntime>, String> { panic!("boom") },
            |error| error.to_string(),
        );
        assert!(matches!(panicked, Err(ref reason) if reason.contains("boom")));

        let drops = Arc::new(AtomicUsize::new(0));
        let actor = pool
            .get_or_try_insert_with(
                "retry",
                || Ok::<_, String>((16, ())),
                {
                    let drops = Arc::clone(&drops);
                    move |()| Ok(owner(3, 16, drops))
                },
                |error| error.to_string(),
            )
            .expect("retry must rebuild");
        assert_eq!(actor.call_mut(|runtime| runtime.value.get()).unwrap(), 3);
    }

    #[test]
    fn operation_panic_is_typed_and_kills_corrupted_actor() {
        let pool = AdmittedPinnedRuntimeActorPool::new(
            "pinned-operation-panic-test",
            AdmittedPinnedRuntimeActorPoolLimits::new(1, 64),
        );
        let drops = Arc::new(AtomicUsize::new(0));
        let actor = pool
            .get_or_try_insert_with(
                "panic",
                || Ok::<_, String>((16, ())),
                {
                    let drops = Arc::clone(&drops);
                    move |()| Ok(owner(1, 16, drops))
                },
                |error| error.to_string(),
            )
            .unwrap();
        let error = actor
            .call_mut::<(), _>(|_| panic!("operation boom"))
            .expect_err("panic is converted to a typed actor failure");
        assert!(matches!(
            error,
            PinnedRuntimeActorError::OperationPanicked { ref message }
                if message.contains("operation boom")
        ));
        for _ in 0..100 {
            if drops.load(Ordering::SeqCst) == 1 {
                break;
            }
            thread::yield_now();
        }
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(matches!(
            actor.call_mut(|_| ()),
            Err(PinnedRuntimeActorError::WorkerTerminated)
        ));
        let rebuilt = pool
            .get_or_try_insert_with(
                "panic",
                || Ok::<_, String>((16, ())),
                {
                    let drops = Arc::clone(&drops);
                    move |()| Ok(owner(2, 16, drops))
                },
                |error| error.to_string(),
            )
            .expect("a terminated cached actor must be evicted and rebuilt");
        assert_eq!(rebuilt.call_mut(|runtime| runtime.value.get()).unwrap(), 2);
    }

    #[test]
    fn reentrant_call_is_rejected_instead_of_deadlocking_owner_thread() {
        let pool = AdmittedPinnedRuntimeActorPool::new(
            "pinned-reentrant-test",
            AdmittedPinnedRuntimeActorPoolLimits::new(1, 64),
        );
        let drops = Arc::new(AtomicUsize::new(0));
        let actor = pool
            .get_or_try_insert_with(
                "reentrant",
                || Ok::<_, String>((16, ())),
                {
                    let drops = Arc::clone(&drops);
                    move |()| Ok(owner(1, 16, drops))
                },
                |error| error.to_string(),
            )
            .expect("actor");
        let recursive = actor.clone();
        let result = actor
            .call_mut(move |_| recursive.call_mut(|runtime| runtime.value.get()))
            .expect("outer actor call must complete");
        assert_eq!(result, Err(PinnedRuntimeActorError::ReentrantCall));
        assert_eq!(actor.call_mut(|runtime| runtime.value.get()).unwrap(), 1);
    }

    #[test]
    fn clear_cannot_resurrect_an_in_flight_actor() {
        let pool = AdmittedPinnedRuntimeActorPool::new(
            "pinned-no-resurrection-test",
            AdmittedPinnedRuntimeActorPoolLimits::new(1, 64),
        );
        let drops = Arc::new(AtomicUsize::new(0));
        let builds = Arc::new(AtomicUsize::new(0));
        let build = |drops: Arc<AtomicUsize>, builds: Arc<AtomicUsize>| {
            move |()| {
                let value = builds.fetch_add(1, Ordering::SeqCst) + 1;
                Ok(owner(value, 16, drops))
            }
        };
        let first = pool
            .get_or_try_insert_with(
                "key",
                || Ok::<_, String>((16, ())),
                build(Arc::clone(&drops), Arc::clone(&builds)),
                |error| error.to_string(),
            )
            .unwrap();
        pool.clear();
        assert_eq!(first.call_mut(|runtime| runtime.value.get()).unwrap(), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(first);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let second = pool
            .get_or_try_insert_with(
                "key",
                || Ok::<_, String>((16, ())),
                build(Arc::clone(&drops), Arc::clone(&builds)),
                |error| error.to_string(),
            )
            .unwrap();
        assert_eq!(second.call_mut(|runtime| runtime.value.get()).unwrap(), 2);
        assert_eq!(builds.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn final_handle_dropped_by_owner_thread_detaches_instead_of_self_joining() {
        let drops = Arc::new(AtomicUsize::new(0));
        let actor = PinnedRuntimeActor::spawn("pinned-self-drop-test", {
            let drops = Arc::clone(&drops);
            move || Ok::<_, String>(owner(1, 16, drops))
        })
        .expect("actor builds");
        let captured_final_handle = actor.clone();
        let (completed_tx, completed_rx) = mpsc::sync_channel(1);
        actor
            .inner
            .sender
            .send(ActorCommand::Run(Box::new(move |_| {
                drop(captured_final_handle);
                let _ = completed_tx.send(());
                true
            })))
            .unwrap();
        drop(actor);
        completed_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("owner-thread final drop must not self-join");
        for _ in 0..100 {
            if drops.load(Ordering::SeqCst) == 1 {
                break;
            }
            thread::yield_now();
        }
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
