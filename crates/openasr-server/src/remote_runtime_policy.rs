//! Server-side remote-compute task policy: file FIFO, idle rebind, run metadata.
//! Admission "may this task start" stays here, not in Desktop.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
};

use tokio::sync::Notify;

pub const REMOTE_RECONNECT_GRACE: Duration = Duration::from_secs(30);
pub const OPERATOR_RUN_LOG_TTL: Duration = Duration::from_secs(7 * 24 * 60 * 60);

pub const SERVER_BUSY_MESSAGE: &str = "The server is busy processing another task.";
pub const PENDING_IDLE_SWITCH_MESSAGE: &str =
    "The server is waiting to switch models after current tasks finish.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileAdmit {
    Running,
    Queued,
}

#[derive(Debug, Default)]
pub struct FileTaskFifo {
    running: Option<String>,
    queued: VecDeque<String>,
}

impl FileTaskFifo {
    #[allow(dead_code)]
    pub fn submit(&mut self, id: impl Into<String>) -> FileAdmit {
        self.submit_occupied(id, false)
    }

    pub fn submit_occupied(&mut self, id: impl Into<String>, occupied: bool) -> FileAdmit {
        let id = id.into();
        if !occupied && self.running.is_none() {
            self.running = Some(id);
            FileAdmit::Running
        } else {
            self.queued.push_back(id);
            FileAdmit::Queued
        }
    }

    pub fn cancel(&mut self, id: &str) -> bool {
        if self.running.as_deref() == Some(id) {
            self.running = self.queued.pop_front();
            return true;
        }
        let before = self.queued.len();
        self.queued.retain(|queued| queued != id);
        before != self.queued.len()
    }

    pub fn finish(&mut self, id: &str) -> Option<String> {
        if self.running.as_deref() != Some(id) {
            return None;
        }
        self.running = self.queued.pop_front();
        self.running.clone()
    }

    pub fn running(&self) -> Option<&str> {
        self.running.as_deref()
    }

    #[allow(dead_code)]
    pub fn queued(&self) -> impl Iterator<Item = &str> {
        self.queued.iter().map(String::as_str)
    }

    pub fn is_queued(&self, id: &str) -> bool {
        self.queued.iter().any(|queued| queued == id)
    }

    #[allow(dead_code)]
    pub fn contains(&self, id: &str) -> bool {
        self.running.as_deref() == Some(id) || self.is_queued(id)
    }

    pub fn promote_head_if_idle(&mut self) {
        if self.running.is_none() {
            self.running = self.queued.pop_front();
        }
    }
}

#[derive(Debug, Default)]
pub struct PendingIdleSwitch {
    pending: Option<String>,
}

impl PendingIdleSwitch {
    pub fn request(&mut self, model: impl Into<String>) {
        self.pending = Some(model.into());
    }

    pub fn cancel(&mut self) {
        self.pending = None;
    }

    pub fn pending(&self) -> Option<&str> {
        self.pending.as_deref()
    }

    pub fn admits_new_tasks(&self) -> bool {
        self.pending.is_none()
    }

    #[allow(dead_code)]
    pub fn apply_if_idle(&mut self, busy: bool) -> Option<String> {
        if busy {
            return None;
        }
        self.pending.take()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorRunRecord {
    pub device_name: String,
    pub started_at: SystemTime,
    pub duration: Duration,
    pub kind: String,
    pub success: bool,
    pub error_type: Option<String>,
}

impl OperatorRunRecord {
    pub fn expires_at(&self) -> SystemTime {
        self.started_at + OPERATOR_RUN_LOG_TTL
    }

    pub fn is_expired(&self, now: SystemTime) -> bool {
        now >= self.expires_at()
    }
}

#[derive(Debug, Default)]
pub struct OperatorRunLog {
    records: Vec<OperatorRunRecord>,
}

impl OperatorRunLog {
    pub fn record(&mut self, record: OperatorRunRecord) {
        self.records.push(record);
    }

    pub fn prune(&mut self, now: SystemTime) {
        self.records.retain(|record| !record.is_expired(now));
    }

    pub fn clear(&mut self) {
        self.records.clear();
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn records(&self) -> &[OperatorRunRecord] {
        &self.records
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct CapabilityIntent {
    pub device_id: String,
    pub features: Vec<String>,
}

#[derive(Debug, Default)]
pub struct CapabilityRequestQueue {
    pending: Vec<CapabilityIntent>,
}

impl CapabilityRequestQueue {
    pub fn submit(&mut self, intent: CapabilityIntent) {
        self.pending.push(intent);
    }

    pub fn approve_next(&mut self) -> Option<CapabilityIntent> {
        if self.pending.is_empty() {
            return None;
        }
        Some(self.pending.remove(0))
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn pending(&self) -> &[CapabilityIntent] {
        &self.pending
    }
}

pub fn reconnect_expired(disconnected_at: SystemTime, now: SystemTime) -> bool {
    reconnect_expired_after(disconnected_at, now, REMOTE_RECONNECT_GRACE)
}

pub fn reconnect_expired_after(
    disconnected_at: SystemTime,
    now: SystemTime,
    grace: Duration,
) -> bool {
    now.duration_since(disconnected_at)
        .map(|elapsed| elapsed >= grace)
        .unwrap_or(true)
}

pub fn recommended_catalog_id_for_feature(feature: &str) -> Option<&'static str> {
    match feature.trim().to_ascii_lowercase().as_str() {
        "speakers" | "speaker" | "diarize" | "diarization" => Some("diarizen-large-s80-v2"),
        "aligner" | "timeline" | "word_timestamps" => Some("qwen3-forced-aligner-0.6b"),
        _ => None,
    }
}

#[derive(Debug)]
struct HeldRealtimeSession {
    disconnected_at: SystemTime,
    control: Arc<openasr_core::TranscriptionControl>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteAdmitError {
    PendingIdleSwitch,
}

#[derive(Debug)]
struct RemoteRuntimePolicyInner {
    files: FileTaskFifo,
    idle: PendingIdleSwitch,
    runs: OperatorRunLog,
    capabilities: CapabilityRequestQueue,
    held_realtime: HashMap<String, HeldRealtimeSession>,
    reconnect_grace: Duration,
}

impl Default for RemoteRuntimePolicyInner {
    fn default() -> Self {
        Self {
            files: FileTaskFifo::default(),
            idle: PendingIdleSwitch::default(),
            runs: OperatorRunLog::default(),
            capabilities: CapabilityRequestQueue::default(),
            held_realtime: HashMap::new(),
            reconnect_grace: REMOTE_RECONNECT_GRACE,
        }
    }
}

#[derive(Debug)]
struct RemoteRuntimePolicyShared {
    state: Mutex<RemoteRuntimePolicyInner>,
    notify: Notify,
}

/// Process-wide remote admission controller shared by HTTP file jobs and
/// realtime/WS. Native semaphore capacity stays in `ModelSessionAdmission`;
/// this layer decides file FIFO vs immediate-busy and idle-switch gating.
#[derive(Clone, Debug)]
pub struct RemoteRuntimePolicy {
    inner: Arc<RemoteRuntimePolicyShared>,
}

impl Default for RemoteRuntimePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl RemoteRuntimePolicy {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RemoteRuntimePolicyShared {
                state: Mutex::new(RemoteRuntimePolicyInner::default()),
                notify: Notify::new(),
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, RemoteRuntimePolicyInner> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn notify(&self) {
        self.inner.notify.notify_waiters();
    }

    pub async fn notified(&self) {
        self.inner.notify.notified().await
    }

    pub fn admits_new_tasks(&self) -> bool {
        self.lock().idle.admits_new_tasks()
    }

    pub fn pending_idle_switch(&self) -> Option<String> {
        self.lock().idle.pending().map(str::to_string)
    }

    pub fn request_idle_switch(&self, model: impl Into<String>) {
        self.lock().idle.request(model);
        self.notify();
    }

    pub fn cancel_idle_switch(&self) {
        self.lock().idle.cancel();
        self.notify();
    }

    pub fn file_running(&self) -> Option<String> {
        self.lock().files.running().map(str::to_string)
    }

    pub fn is_file_queued(&self, id: &str) -> bool {
        self.lock().files.is_queued(id)
    }

    /// Admit a file job: run now, queue behind the occupant, or reject when an
    /// idle-after-busy ASR switch is pending. Idempotent for an already
    /// tracked id so HTTP waiters can re-poll.
    pub fn admit_file(&self, id: &str, occupied: bool) -> Result<FileAdmit, RemoteAdmitError> {
        let mut inner = self.lock();
        if inner.files.running() == Some(id) {
            return Ok(FileAdmit::Running);
        }
        if !occupied {
            inner.files.promote_head_if_idle();
        }
        if inner.files.running() == Some(id) {
            return Ok(FileAdmit::Running);
        }
        if inner.files.is_queued(id) {
            return Ok(FileAdmit::Queued);
        }
        if !inner.idle.admits_new_tasks() {
            return Err(RemoteAdmitError::PendingIdleSwitch);
        }
        Ok(inner.files.submit_occupied(id, occupied))
    }

    pub fn cancel_file(&self, id: &str) -> bool {
        let changed = self.lock().files.cancel(id);
        if changed {
            self.notify();
        }
        changed
    }

    pub fn finish_file(&self, id: &str) -> Option<String> {
        let next = self.lock().files.finish(id);
        self.notify();
        next
    }

    pub fn record_run(&self, record: OperatorRunRecord) {
        let mut inner = self.lock();
        inner.runs.record(record);
        inner.runs.prune(SystemTime::now());
    }

    pub fn prune_runs(&self, now: SystemTime) {
        self.lock().runs.prune(now);
    }

    pub fn clear_runs(&self) {
        self.lock().runs.clear();
    }

    pub fn run_records(&self) -> Vec<OperatorRunRecord> {
        self.lock().runs.records().to_vec()
    }

    pub fn submit_capability(&self, intent: CapabilityIntent) {
        self.lock().capabilities.submit(intent);
    }

    pub fn approve_next_capability(&self) -> Option<CapabilityIntent> {
        self.lock().capabilities.approve_next()
    }

    pub fn pending_capabilities(&self) -> Vec<CapabilityIntent> {
        self.lock().capabilities.pending().to_vec()
    }

    pub fn reconnect_grace(&self) -> Duration {
        self.lock().reconnect_grace
    }

    #[cfg(test)]
    pub fn set_reconnect_grace(&self, grace: Duration) {
        self.lock().reconnect_grace = grace;
    }

    pub fn hold_realtime(
        &self,
        session_id: impl Into<String>,
        control: Arc<openasr_core::TranscriptionControl>,
    ) {
        self.hold_realtime_at(session_id, control, SystemTime::now());
    }

    pub fn hold_realtime_at(
        &self,
        session_id: impl Into<String>,
        control: Arc<openasr_core::TranscriptionControl>,
        now: SystemTime,
    ) {
        let mut inner = self.lock();
        inner.held_realtime.insert(
            session_id.into(),
            HeldRealtimeSession {
                disconnected_at: now,
                control,
            },
        );
        self.inner.notify.notify_waiters();
    }

    pub fn resume_realtime(
        &self,
        session_id: &str,
        now: SystemTime,
    ) -> Option<Arc<openasr_core::TranscriptionControl>> {
        let mut inner = self.lock();
        let held = inner.held_realtime.get(session_id)?;
        if reconnect_expired_after(held.disconnected_at, now, inner.reconnect_grace) {
            return None;
        }
        inner.held_realtime.remove(session_id).map(|held| held.control)
    }

    pub fn has_held_realtime(&self) -> bool {
        !self.lock().held_realtime.is_empty()
    }

    pub fn expire_held_realtime(
        &self,
        now: SystemTime,
    ) -> Vec<(String, Arc<openasr_core::TranscriptionControl>)> {
        let mut inner = self.lock();
        let grace = inner.reconnect_grace;
        let expired: Vec<String> = inner
            .held_realtime
            .iter()
            .filter(|(_, held)| {
                if grace == REMOTE_RECONNECT_GRACE {
                    reconnect_expired(held.disconnected_at, now)
                } else {
                    reconnect_expired_after(held.disconnected_at, now, grace)
                }
            })
            .map(|(id, _)| id.clone())
            .collect();
        expired
            .into_iter()
            .filter_map(|id| {
                inner
                    .held_realtime
                    .remove(&id)
                    .map(|held| (id, held.control))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_fifo_runs_first_and_queues_the_rest() {
        let mut fifo = FileTaskFifo::default();
        assert_eq!(fifo.submit("a"), FileAdmit::Running);
        assert_eq!(fifo.submit("b"), FileAdmit::Queued);
        assert_eq!(fifo.submit("c"), FileAdmit::Queued);
        assert!(fifo.contains("c"));
        assert_eq!(fifo.queued().collect::<Vec<_>>(), vec!["b", "c"]);
        assert!(fifo.cancel("b"));
        assert_eq!(fifo.finish("a").as_deref(), Some("c"));
        assert_eq!(fifo.running(), Some("c"));
    }

    #[test]
    fn idle_switch_blocks_new_tasks_until_canceled_or_applied() {
        let mut idle = PendingIdleSwitch::default();
        idle.request("xasr-zh-en:fp16");
        assert!(!idle.admits_new_tasks());
        assert!(idle.apply_if_idle(true).is_none());
        idle.cancel();
        assert!(idle.admits_new_tasks());
        idle.request("moss-transcribe-diarize:q8");
        assert_eq!(idle.apply_if_idle(false).as_deref(), Some("moss-transcribe-diarize:q8"));
        assert!(idle.admits_new_tasks());
    }

    #[test]
    fn operator_run_log_expires_at_seven_days_and_has_no_content_fields() {
        let started = SystemTime::UNIX_EPOCH;
        let record = OperatorRunRecord {
            device_name: "Studio Mac".to_string(),
            started_at: started,
            duration: Duration::from_secs(12),
            kind: "file".to_string(),
            success: false,
            error_type: Some("busy".to_string()),
        };
        assert_eq!(record.expires_at(), started + OPERATOR_RUN_LOG_TTL);
        assert!(record.is_expired(started + OPERATOR_RUN_LOG_TTL));
        assert!(!record.is_expired(started + Duration::from_secs(6 * 24 * 60 * 60)));
        let mut log = OperatorRunLog::default();
        log.record(record);
        assert_eq!(log.records().len(), 1);
        log.prune(started + OPERATOR_RUN_LOG_TTL);
        assert_eq!(log.len(), 0);
        log.clear();
    }

    #[test]
    fn capability_requests_are_operator_confirmed_fifo() {
        let mut queue = CapabilityRequestQueue::default();
        queue.submit(CapabilityIntent {
            device_id: "phone".to_string(),
            features: vec!["speakers".to_string()],
        });
        queue.submit(CapabilityIntent {
            device_id: "laptop".to_string(),
            features: vec!["aligner".to_string()],
        });
        let first = queue.approve_next().unwrap();
        assert_eq!(first.device_id, "phone");
        assert_eq!(first.features, ["speakers"]);
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn reconnect_grace_is_thirty_seconds() {
        let start = SystemTime::UNIX_EPOCH;
        assert!(!reconnect_expired(start, start + Duration::from_secs(29)));
        assert!(reconnect_expired(start, start + Duration::from_secs(30)));
    }

    #[test]
    fn occupied_slot_queues_the_first_file() {
        let mut fifo = FileTaskFifo::default();
        assert_eq!(fifo.submit_occupied("a", true), FileAdmit::Queued);
        fifo.promote_head_if_idle();
        assert_eq!(fifo.running(), Some("a"));
    }

    #[test]
    fn admit_file_is_idempotent_and_blocks_new_work_during_idle_switch() {
        let policy = RemoteRuntimePolicy::new();
        assert_eq!(policy.admit_file("a", false).unwrap(), FileAdmit::Running);
        assert_eq!(policy.admit_file("a", true).unwrap(), FileAdmit::Running);
        assert_eq!(policy.admit_file("b", true).unwrap(), FileAdmit::Queued);
        policy.request_idle_switch("xasr-zh-en:fp16");
        assert_eq!(
            policy.admit_file("c", false),
            Err(RemoteAdmitError::PendingIdleSwitch)
        );
        assert_eq!(policy.admit_file("b", true).unwrap(), FileAdmit::Queued);
        policy.cancel_file("a");
        policy.cancel_idle_switch();
        assert_eq!(policy.admit_file("b", false).unwrap(), FileAdmit::Running);
    }

    #[test]
    fn held_realtime_resumes_inside_grace_and_cancels_after_timeout() {
        let policy = RemoteRuntimePolicy::new();
        let control = Arc::new(openasr_core::TranscriptionControl::new());
        let start = SystemTime::UNIX_EPOCH;
        policy.hold_realtime_at("rt_ws_1", Arc::clone(&control), start);
        assert!(policy.has_held_realtime());
        assert!(
            policy
                .resume_realtime("rt_ws_1", start + Duration::from_secs(29))
                .is_some()
        );
        assert!(!policy.has_held_realtime());
        assert!(!control.is_canceled());

        policy.hold_realtime_at("rt_ws_2", Arc::clone(&control), start);
        let expired = policy.expire_held_realtime(start + Duration::from_secs(30));
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].0, "rt_ws_2");
        assert!(
            policy
                .resume_realtime("rt_ws_2", start + Duration::from_secs(31))
                .is_none()
        );
    }

    #[test]
    fn capability_features_map_to_recommended_catalog_ids() {
        assert_eq!(
            recommended_catalog_id_for_feature("speakers"),
            Some("diarizen-large-s80-v2")
        );
        assert_eq!(
            recommended_catalog_id_for_feature("aligner"),
            Some("qwen3-forced-aligner-0.6b")
        );
        assert!(recommended_catalog_id_for_feature("unknown-pack").is_none());
    }
}
