use std::sync::{Arc, Mutex};
use std::time::Duration;

use thiserror::Error;

use super::clause::ClauseId;
use super::queue::{
    LatestOnlyTranslationQueue, TranslationQueueError, TranslationQueueSubmit,
    TranslationWorkerOutput,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetLang {
    En,
}

impl TargetLang {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::En => "en",
        }
    }

    pub fn parse_mvp(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "en" => Some(Self::En),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalizedTranslationContext {
    pub source_text: String,
    pub target_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslationRequest {
    pub clause_id: ClauseId,
    pub replaces_clause_id: Option<ClauseId>,
    pub source_version: u64,
    pub source_text: String,
    pub finalized: bool,
    pub revised: bool,
    pub target_lang: TargetLang,
    pub finalized_context: Vec<FinalizedTranslationContext>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationOutput {
    pub clause_id: ClauseId,
    pub replaces_clause_id: Option<ClauseId>,
    pub source_version: u64,
    pub translation_version: u64,
    pub text: String,
    pub source_text: String,
    pub finalized: bool,
    pub revised: bool,
    pub target_lang: TargetLang,
    pub dropped_stale: bool,
    pub timings: TranslationTimings,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TranslationTimings {
    pub queue_wait: Duration,
    pub prefill: Duration,
    pub decode: Duration,
    pub total: Duration,
    pub prompt_tokens: usize,
    pub prefilled_tokens: usize,
    pub reused_prefix_tokens: usize,
    pub cache_backoff_tokens: usize,
    pub generated_tokens: usize,
}

#[derive(Debug, Error)]
pub enum TranslationSessionError {
    #[error("translation session context lock is poisoned")]
    ContextPoisoned,
    #[error("translation queue failed: {source}")]
    Queue {
        #[from]
        source: TranslationQueueError,
    },
}

pub struct TranslationSession {
    queue: LatestOnlyTranslationQueue,
    finalized_context: Arc<Mutex<Vec<FinalizedTranslationContext>>>,
}

impl TranslationSession {
    pub fn spawn(
        worker: impl FnMut(TranslationRequest) -> Result<TranslationWorkerOutput, TranslationQueueError>
        + Send
        + 'static,
    ) -> Self {
        Self {
            queue: LatestOnlyTranslationQueue::spawn(worker),
            finalized_context: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Spawns a session whose worker initialization (e.g. translation model
    /// cold load) happens asynchronously on the worker thread; the session is
    /// usable immediately and buffers requests until the worker is ready.
    /// Initialization failures surface through `try_recv`.
    pub fn spawn_thread_local<W>(
        init_worker: impl FnOnce() -> Result<W, TranslationQueueError> + Send + 'static,
    ) -> Self
    where
        W: FnMut(TranslationRequest) -> Result<TranslationWorkerOutput, TranslationQueueError>
            + 'static,
    {
        Self {
            queue: LatestOnlyTranslationQueue::spawn_thread_local(init_worker),
            finalized_context: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// True once the worker finished (asynchronous) initialization. Workers
    /// created with `spawn` are ready at birth.
    pub fn worker_ready(&self) -> bool {
        self.queue.worker_ready()
    }

    pub fn enqueue(
        &self,
        mut request: TranslationRequest,
    ) -> Result<TranslationQueueSubmit, TranslationSessionError> {
        if request.finalized_context.is_empty() {
            request.finalized_context = self.finalized_context_snapshot()?;
        }
        self.queue.enqueue(request).map_err(Into::into)
    }

    pub fn try_recv(&self) -> Result<Option<TranslationOutput>, TranslationSessionError> {
        Ok(self.queue.try_recv()?)
    }

    pub fn record_output_context(
        &self,
        output: &TranslationOutput,
    ) -> Result<(), TranslationSessionError> {
        if output.finalized && !output.dropped_stale {
            self.record_finalized_context(FinalizedTranslationContext {
                source_text: output.source_text.clone(),
                target_text: output.text.clone(),
            })?;
        }
        Ok(())
    }

    pub fn has_pending_or_running(&self) -> bool {
        self.queue.has_pending_or_running()
    }

    pub fn retire_clause_ids(
        &self,
        clause_ids: impl IntoIterator<Item = ClauseId>,
    ) -> Result<(), TranslationSessionError> {
        self.queue.retire_clause_ids(clause_ids).map_err(Into::into)
    }

    pub fn finalized_context_snapshot(
        &self,
    ) -> Result<Vec<FinalizedTranslationContext>, TranslationSessionError> {
        let context = self
            .finalized_context
            .lock()
            .map_err(|_| TranslationSessionError::ContextPoisoned)?;
        let mut recent = context.iter().rev().take(2).cloned().collect::<Vec<_>>();
        recent.reverse();
        Ok(recent)
    }

    pub fn record_finalized_context(
        &self,
        context: FinalizedTranslationContext,
    ) -> Result<(), TranslationSessionError> {
        let mut contexts = self
            .finalized_context
            .lock()
            .map_err(|_| TranslationSessionError::ContextPoisoned)?;
        contexts.push(context);
        if contexts.len() > 2 {
            let remove_count = contexts.len() - 2;
            contexts.drain(..remove_count);
        }
        Ok(())
    }
}
