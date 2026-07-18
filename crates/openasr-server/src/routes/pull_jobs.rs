//! Pull-job HTTP endpoints, job lifecycle/worker plumbing, and the
//! `PullJob*` snapshot/state types. Pure code-motion from `lib.rs`.

use serde::{Deserialize, Serialize};

use crate::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PullJobSnapshot {
    pub(crate) job_id: String,
    pub(crate) state: PullJobState,
    pub(crate) model_id: String,
    pub(crate) display_name: String,
    pub(crate) quant: String,
    pub(crate) pull: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) resolved: Option<PullJobResolvedSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source_path: Option<PathBuf>,
    pub(crate) bytes_done: u64,
    pub(crate) bytes_total: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) speed_bps: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) eta_s: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) installed_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) installed: Option<InstalledPack>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) control_requested: Option<PullControlRequest>,
    #[serde(skip)]
    pub(crate) last_progress_at_unix_millis: Option<u128>,
    #[serde(skip)]
    pub(crate) last_bytes_done: Option<u64>,
    /// Exponential-moving-average of `speed_bps` (bytes/sec), carried across
    /// `Downloading` progress ticks. The instantaneous delta-bytes/delta-time
    /// speed is jittery -- persistence happens on an irregular cadence (byte
    /// threshold OR time threshold, whichever fires first), so consecutive
    /// sample windows vary widely in length and a single noisy window can
    /// swing the raw speed by a large factor. Smoothing here (rather than
    /// only in the frontend) means every consumer of the snapshot -- HTTP
    /// poll, SSE stream, persisted-to-disk state -- sees the same stable
    /// number. Not serialized: it is reconstructible from history and is
    /// process-local smoothing state, not job identity.
    #[serde(skip)]
    pub(crate) smoothed_speed_bps: Option<u64>,
}

/// EMA weight applied to each new instantaneous speed sample. Lower = smoother
/// but slower to track real speed changes. Chosen below the frontend's
/// SPEED_EMA_ALPHA (0.35) since the server smooths first: stacking two EMAs at
/// the same weight would double-lag without adding stability, and the server
/// output is what SSE/poll clients and the frontend build on.
const PULL_PROGRESS_SPEED_EMA_ALPHA: f64 = 0.25;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PullJobResolvedSpec {
    pub(crate) requested: String,
    pub(crate) model_id: String,
    pub(crate) display_name: String,
    pub(crate) quant: String,
    pub(crate) suffix: String,
    pub(crate) pull: String,
    pub(crate) filename: String,
    pub(crate) url: String,
    #[serde(default)]
    pub(crate) mirrors: Vec<CatalogMirror>,
    pub(crate) hf_revision: String,
    pub(crate) sha256: String,
    pub(crate) size_bytes: u64,
    pub(crate) license: String,
    pub(crate) license_url: String,
    pub(crate) license_class: LicenseClass,
}

impl PullJobResolvedSpec {
    fn from_resolved(resolved: &ResolvedCatalogPull) -> Self {
        Self {
            requested: resolved.requested.clone(),
            model_id: resolved.model_id.clone(),
            display_name: resolved.display_name.clone(),
            quant: resolved.quant.clone(),
            suffix: resolved.suffix.clone(),
            pull: resolved.pull.clone(),
            filename: resolved.filename.clone(),
            url: resolved.url.clone(),
            mirrors: resolved.mirrors.clone(),
            hf_revision: resolved.hf_revision.clone(),
            sha256: resolved.sha256.clone(),
            size_bytes: resolved.size_bytes,
            license: resolved.license.clone(),
            license_url: resolved.license_url.clone(),
            license_class: resolved.license_class.clone(),
        }
    }
}

impl From<PullJobResolvedSpec> for ResolvedCatalogPull {
    fn from(spec: PullJobResolvedSpec) -> Self {
        Self {
            requested: spec.requested,
            model_id: spec.model_id,
            display_name: spec.display_name,
            quant: spec.quant,
            suffix: spec.suffix,
            pull: spec.pull,
            filename: spec.filename,
            url: spec.url,
            mirrors: spec.mirrors,
            hf_revision: spec.hf_revision,
            sha256: spec.sha256,
            size_bytes: spec.size_bytes,
            license: spec.license,
            license_url: spec.license_url,
            license_class: spec.license_class,
        }
    }
}

impl PullJobSnapshot {
    pub(crate) fn queued(
        job_id: String,
        resolved: &ResolvedCatalogPull,
        source_path: Option<PathBuf>,
    ) -> Self {
        Self {
            job_id,
            state: PullJobState::Queued,
            model_id: resolved.model_id.clone(),
            display_name: resolved.display_name.clone(),
            quant: resolved.quant.clone(),
            pull: resolved.pull.clone(),
            resolved: Some(PullJobResolvedSpec::from_resolved(resolved)),
            source_path,
            bytes_done: 0,
            bytes_total: resolved.size_bytes,
            speed_bps: None,
            eta_s: None,
            installed_path: None,
            installed: None,
            error: None,
            control_requested: None,
            last_progress_at_unix_millis: None,
            last_bytes_done: None,
            smoothed_speed_bps: None,
        }
    }

    fn already_installed(
        job_id: String,
        resolved: &ResolvedCatalogPull,
        pack: InstalledPack,
    ) -> Self {
        Self {
            job_id,
            state: PullJobState::AlreadyInstalled,
            model_id: resolved.model_id.clone(),
            display_name: resolved.display_name.clone(),
            quant: resolved.quant.clone(),
            pull: resolved.pull.clone(),
            resolved: Some(PullJobResolvedSpec::from_resolved(resolved)),
            source_path: None,
            bytes_done: resolved.size_bytes,
            bytes_total: resolved.size_bytes,
            speed_bps: None,
            eta_s: Some(0),
            installed_path: Some(pack.path.clone()),
            installed: Some(pack),
            error: None,
            control_requested: None,
            last_progress_at_unix_millis: None,
            last_bytes_done: None,
            smoothed_speed_bps: None,
        }
    }

    pub(crate) fn apply_progress(&mut self, progress: PullProgress) {
        match progress {
            PullProgress::UsingInstalled { path } => {
                self.state = PullJobState::AlreadyInstalled;
                self.control_requested = None;
                self.bytes_done = self.bytes_total;
                self.speed_bps = None;
                self.smoothed_speed_bps = None;
                self.eta_s = Some(0);
                self.installed_path = Some(path);
            }
            PullProgress::DownloadStarted {
                bytes_total,
                resume_from,
            } => {
                self.state = PullJobState::Downloading;
                self.control_requested = None;
                self.bytes_total = bytes_total;
                self.bytes_done = resume_from;
                self.speed_bps = None;
                self.smoothed_speed_bps = None;
                self.eta_s = None;
                self.last_progress_at_unix_millis = Some(unix_millis_now());
                self.last_bytes_done = Some(resume_from);
            }
            PullProgress::Downloading {
                bytes_done,
                bytes_total,
            } => {
                let now = unix_millis_now();
                if let (Some(last_at), Some(last_bytes)) =
                    (self.last_progress_at_unix_millis, self.last_bytes_done)
                {
                    let elapsed_ms = now.saturating_sub(last_at);
                    let delta_bytes = bytes_done.saturating_sub(last_bytes);
                    if elapsed_ms > 0 && delta_bytes > 0 {
                        let instant_speed = ((delta_bytes as u128) * 1000 / elapsed_ms) as u64;
                        let smoothed = match self.smoothed_speed_bps {
                            // First sample: no prior average to blend with, so
                            // seed the EMA directly rather than starting from
                            // zero (which would otherwise bias the first few
                            // displayed speeds low).
                            None => instant_speed,
                            Some(prev) => {
                                ema_blend(instant_speed, prev, PULL_PROGRESS_SPEED_EMA_ALPHA)
                            }
                        };
                        self.smoothed_speed_bps = Some(smoothed);
                        self.speed_bps = Some(smoothed);
                        self.eta_s = eta_seconds(bytes_done, bytes_total, smoothed);
                    }
                }
                self.state = PullJobState::Downloading;
                self.control_requested = None;
                self.bytes_total = bytes_total;
                self.bytes_done = bytes_done;
                self.last_progress_at_unix_millis = Some(now);
                self.last_bytes_done = Some(bytes_done);
            }
            PullProgress::Verifying { bytes_done } => {
                self.state = PullJobState::Verifying;
                self.control_requested = None;
                self.bytes_done = bytes_done.min(self.bytes_total);
                self.speed_bps = None;
                self.smoothed_speed_bps = None;
                self.eta_s = Some(0);
            }
            PullProgress::Installed { path } => {
                self.state = PullJobState::Completed;
                self.control_requested = None;
                self.bytes_done = self.bytes_total;
                self.speed_bps = None;
                self.smoothed_speed_bps = None;
                self.eta_s = Some(0);
                self.installed_path = Some(path);
            }
        }
    }
}

pub(crate) fn unix_millis_now() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

pub(crate) fn unix_seconds_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

/// Blend a new instantaneous sample into a running average:
/// `alpha * instant + (1 - alpha) * prev`. `alpha` in `[0, 1]`; higher tracks
/// the instantaneous value more closely, lower smooths harder. Kept as a
/// standalone `u64` helper (rather than inlined) so the blending arithmetic
/// has one place to reason about rounding/precision and is directly
/// unit-testable independent of `apply_progress`'s state threading.
pub(crate) fn ema_blend(instant: u64, prev: u64, alpha: f64) -> u64 {
    let blended = alpha * instant as f64 + (1.0 - alpha) * prev as f64;
    blended.round().max(0.0) as u64
}

pub(crate) fn eta_seconds(bytes_done: u64, bytes_total: u64, speed_bps: u64) -> Option<u64> {
    if bytes_done >= bytes_total {
        return Some(0);
    }
    if speed_bps == 0 {
        return None;
    }
    Some(bytes_total.saturating_sub(bytes_done).div_ceil(speed_bps))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PullJobState {
    Queued,
    Downloading,
    Verifying,
    Paused,
    Completed,
    AlreadyInstalled,
    Canceled,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PullControlRequest {
    Pause,
    Cancel,
}

impl PullJobState {
    pub(crate) fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::AlreadyInstalled | Self::Canceled | Self::Failed
        )
    }

    pub(crate) fn is_restart_resumable(self) -> bool {
        matches!(self, Self::Queued | Self::Downloading | Self::Verifying)
    }

    pub(crate) fn is_event_terminal(self) -> bool {
        self.is_terminal() || self == Self::Paused
    }
}

pub(crate) async fn start_pull_job(
    AxumPath(id): AxumPath<String>,
    Extension(distribution): Extension<DistributionContext>,
    Json(request): Json<StartPullRequest>,
) -> Result<Response, ApiError> {
    let home = distribution.openasr_home()?;
    let catalog = load_catalog_for_optional_source(distribution.catalog_source(), &home)
        .map_err(ApiError::Catalog)?;
    let resolved = resolve_catalog_pull(
        &catalog,
        &CatalogPullRequest {
            reference: id,
            quant: request.quant,
            size: request.size,
        },
    )
    .map_err(ApiError::Catalog)?;

    if resolved.license_class == LicenseClass::Gated && request.accept_license != Some(true) {
        return Err(ApiError::BadRequest(format!(
            "Model '{}' requires accepting the license before download: {}",
            resolved.model_id, resolved.license_url
        )));
    }

    distribution.ensure_restart_resumes_started();
    if let Some(snapshot) = distribution.nonterminal_snapshot_for_pull(&resolved) {
        return Ok((StatusCode::ACCEPTED, Json(snapshot)).into_response());
    }

    let job_id = distribution.next_job_id();
    if let Some(pack) = matching_installed_pack(&home, &resolved).map_err(ApiError::Pull)? {
        let snapshot = PullJobSnapshot::already_installed(job_id, &resolved, pack);
        distribution.insert_job(snapshot.clone())?;
        return Ok((StatusCode::OK, Json(snapshot)).into_response());
    }

    let source_path = request
        .from
        .map(resolve_local_pull_source_path)
        .transpose()?;
    let snapshot = PullJobSnapshot::queued(job_id.clone(), &resolved, source_path.clone());
    distribution.insert_job(snapshot.clone())?;
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let pause_flag = Arc::new(AtomicBool::new(false));
    spawn_pull_job(
        distribution,
        job_id,
        home,
        resolved,
        source_path,
        cancel_flag,
        pause_flag,
    );
    Ok((StatusCode::ACCEPTED, Json(snapshot)).into_response())
}

pub(crate) async fn pull_job(
    AxumPath(job_id): AxumPath<String>,
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<PullJobSnapshot>, ApiError> {
    distribution.ensure_restart_resumes_started();
    let snapshot = distribution
        .snapshot(&job_id)
        .ok_or_else(|| ApiError::NotFound(format!("Pull job not found: {job_id}")))?;
    Ok(Json(snapshot))
}

/// Lists currently non-terminal pull jobs so a client that lost its in-memory
/// job list (the desktop shell after a daemon restart kills and relaunches
/// the process) can rediscover in-flight downloads. Deliberately does **not**
/// call `ensure_restart_resumes_started`: this is a pure read of whatever
/// state `DistributionContext::new` already loaded from
/// `~/.openasr/pulls/*.json` at startup (restart-resumable jobs are
/// normalized to `Queued` synchronously at load time, before any request
/// arrives), so listing jobs can never itself start or resume a download --
/// the server-never-pulls-on-a-query invariant holds even for the very first
/// request after a restart.
pub(crate) async fn list_pull_jobs(
    Extension(distribution): Extension<DistributionContext>,
) -> Json<PullJobsListResponse> {
    Json(PullJobsListResponse {
        jobs: distribution.nonterminal_snapshots(),
    })
}

#[derive(Debug, Serialize)]
pub(crate) struct PullJobsListResponse {
    pub(crate) jobs: Vec<PullJobSnapshot>,
}

pub(crate) async fn pull_job_events(
    AxumPath(job_id): AxumPath<String>,
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    distribution.ensure_restart_resumes_started();
    let receiver = distribution
        .subscribe_job(&job_id)
        .ok_or_else(|| ApiError::NotFound(format!("Pull job not found: {job_id}")))?;
    let stream = stream::unfold(
        (receiver, true, false),
        |(mut receiver, first, done)| async move {
            if done {
                return None;
            }
            let snapshot = if first {
                receiver.borrow().clone()
            } else {
                receiver.changed().await.ok()?;
                receiver.borrow().clone()
            };
            let done = snapshot.state.is_event_terminal();
            let event = Event::default()
                .event("snapshot")
                .json_data(&snapshot)
                .expect("pull job snapshot should serialize");
            Some((Ok(event), (receiver, false, done)))
        },
    );
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

pub(crate) async fn cancel_pull_job(
    AxumPath(job_id): AxumPath<String>,
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Response, ApiError> {
    let snapshot = distribution
        .snapshot(&job_id)
        .ok_or_else(|| ApiError::NotFound(format!("Pull job not found: {job_id}")))?;
    if snapshot.state.is_terminal() {
        return Ok((StatusCode::OK, Json(snapshot)).into_response());
    }
    if !distribution.cancel_job(&job_id) {
        return Err(ApiError::BadRequest(format!(
            "Pull job '{job_id}' is not currently cancelable."
        )));
    }
    distribution.update_job(&job_id, |snapshot| {
        snapshot.control_requested = Some(PullControlRequest::Cancel);
        snapshot.error = Some("Cancellation requested.".to_string());
    })?;
    let snapshot = distribution
        .snapshot(&job_id)
        .ok_or_else(|| ApiError::NotFound(format!("Pull job not found: {job_id}")))?;
    Ok((StatusCode::ACCEPTED, Json(snapshot)).into_response())
}

pub(crate) async fn pause_pull_job(
    AxumPath(job_id): AxumPath<String>,
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Response, ApiError> {
    let snapshot = distribution
        .snapshot(&job_id)
        .ok_or_else(|| ApiError::NotFound(format!("Pull job not found: {job_id}")))?;
    if snapshot.state == PullJobState::Paused || snapshot.state.is_terminal() {
        return Ok((StatusCode::OK, Json(snapshot)).into_response());
    }
    if !distribution.pause_job(&job_id) {
        return Err(ApiError::BadRequest(format!(
            "Pull job '{job_id}' is not currently pausable."
        )));
    }
    distribution.update_job(&job_id, |snapshot| {
        snapshot.control_requested = Some(PullControlRequest::Pause);
        snapshot.error = Some("Pause requested.".to_string());
    })?;
    let snapshot = distribution
        .snapshot(&job_id)
        .ok_or_else(|| ApiError::NotFound(format!("Pull job not found: {job_id}")))?;
    Ok((StatusCode::ACCEPTED, Json(snapshot)).into_response())
}

pub(crate) async fn resume_pull_job(
    AxumPath(job_id): AxumPath<String>,
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Response, ApiError> {
    let snapshot = distribution
        .snapshot(&job_id)
        .ok_or_else(|| ApiError::NotFound(format!("Pull job not found: {job_id}")))?;
    if snapshot.state.is_terminal() {
        return Ok((StatusCode::OK, Json(snapshot)).into_response());
    }
    if snapshot.state != PullJobState::Paused {
        return Err(ApiError::BadRequest(format!(
            "Pull job '{job_id}' is not paused."
        )));
    }

    let home = distribution.openasr_home()?;
    let resolved = resolved_pull_from_snapshot(&snapshot)?;
    let source_path = snapshot.source_path.clone();
    distribution.update_job(&job_id, |stored| {
        stored.state = PullJobState::Queued;
        stored.error = None;
        stored.control_requested = None;
    })?;
    let snapshot = distribution
        .snapshot(&job_id)
        .ok_or_else(|| ApiError::NotFound(format!("Pull job not found: {job_id}")))?;
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let pause_flag = Arc::new(AtomicBool::new(false));
    spawn_pull_job(
        distribution,
        job_id,
        home,
        resolved,
        source_path,
        cancel_flag,
        pause_flag,
    );
    Ok((StatusCode::ACCEPTED, Json(snapshot)).into_response())
}

pub(crate) fn resolved_pull_from_snapshot(
    snapshot: &PullJobSnapshot,
) -> Result<ResolvedCatalogPull, ApiError> {
    snapshot
        .resolved
        .clone()
        .ok_or_else(|| {
            ApiError::BadRequest(format!(
                "Pull job '{}' does not contain an immutable resolved model pack spec; refusing to re-resolve the mutable catalog.",
                snapshot.job_id
            ))
        })
        .map(ResolvedCatalogPull::from)
}

pub(crate) fn resolve_local_pull_source_path(path: PathBuf) -> Result<PathBuf, ApiError> {
    if path.is_absolute() {
        return Ok(path);
    }
    let current_dir = env::current_dir().map_err(|error| {
        ApiError::BadRequest(format!(
            "Could not resolve local pull source path '{}': {error}",
            path.display()
        ))
    })?;
    Ok(current_dir.join(path))
}

pub(crate) fn fail_restart_resume<E: std::fmt::Display>(
    distribution: &DistributionContext,
    job_id: &str,
    error: E,
) {
    distribution.update_job_best_effort(job_id, |snapshot| {
        snapshot.state = PullJobState::Failed;
        snapshot.control_requested = None;
        snapshot.speed_bps = None;
        snapshot.eta_s = None;
        snapshot.error = Some(format!(
            "Could not resume pull job after OpenASR daemon restart: {error}"
        ));
    });
}

pub(crate) fn spawn_pull_job(
    distribution: DistributionContext,
    job_id: String,
    home: PathBuf,
    resolved: ResolvedCatalogPull,
    source_path: Option<PathBuf>,
    cancel_flag: Arc<AtomicBool>,
    pause_flag: Arc<AtomicBool>,
) {
    let limiter = pull_limiter_for_home(&home);
    let active_distribution = distribution.clone();
    let active_job_id = job_id.clone();
    let active_cancel_flag = cancel_flag.clone();
    let active_pause_flag = pause_flag.clone();
    active_distribution.register_active_job(&active_job_id, active_cancel_flag, active_pause_flag);
    task::spawn(async move {
        let Ok(permit) = limiter.acquire_owned().await else {
            fail_queued_pull_job(
                &distribution,
                &job_id,
                "Pull scheduler was closed before this job could start.",
            );
            distribution.clear_active_job(&job_id);
            return;
        };
        if cancel_flag.load(Ordering::SeqCst) {
            distribution.update_job_best_effort(&job_id, |snapshot| {
                snapshot.state = PullJobState::Canceled;
                snapshot.control_requested = None;
                snapshot.speed_bps = None;
                snapshot.eta_s = None;
                snapshot.error = Some("Pull job was canceled before download started.".to_string());
            });
            distribution.clear_active_job(&job_id);
            return;
        }
        let blocking_distribution = distribution.clone();
        let blocking_job_id = job_id.clone();
        let blocking = task::spawn_blocking(move || {
            let _permit = permit;
            run_pull_job(
                blocking_distribution,
                blocking_job_id,
                home,
                resolved,
                source_path,
                cancel_flag,
                pause_flag,
            );
        });
        if let Err(error) = blocking.await {
            fail_queued_pull_job(
                &distribution,
                &job_id,
                &format!("Pull worker task failed: {error}"),
            );
        }
        distribution.clear_active_job(&job_id);
    });
}

pub(crate) fn pull_limiter_for_home(home: &Path) -> Arc<Semaphore> {
    static LIMITERS: OnceLock<Mutex<HashMap<PathBuf, Arc<Semaphore>>>> = OnceLock::new();
    let key = home.canonicalize().unwrap_or_else(|_| home.to_path_buf());
    let mut limiters = LIMITERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("pull limiter registry mutex poisoned");
    limiters
        .entry(key)
        .or_insert_with(|| Arc::new(Semaphore::new(MAX_CONCURRENT_PULL_JOBS_PER_HOME)))
        .clone()
}

pub(crate) fn fail_queued_pull_job(
    distribution: &DistributionContext,
    job_id: &str,
    message: &str,
) {
    distribution.update_job_best_effort(job_id, |snapshot| {
        snapshot.state = PullJobState::Failed;
        snapshot.control_requested = None;
        snapshot.speed_bps = None;
        snapshot.eta_s = None;
        snapshot.error = Some(message.to_string());
    });
}

pub(crate) fn run_pull_job(
    distribution: DistributionContext,
    job_id: String,
    home: PathBuf,
    resolved: ResolvedCatalogPull,
    source_path: Option<PathBuf>,
    cancel_flag: Arc<AtomicBool>,
    pause_flag: Arc<AtomicBool>,
) {
    distribution.update_job_best_effort(&job_id, |snapshot| {
        snapshot.state = PullJobState::Queued;
        snapshot.control_requested = None;
        snapshot.speed_bps = None;
        snapshot.eta_s = None;
        snapshot.error = None;
    });

    let progress_distribution = distribution.clone();
    let progress_job_id = job_id.clone();
    let mut last_progress_persisted_at = Instant::now();
    let mut last_progress_persisted_bytes = 0_u64;
    let mut progress = move |progress| {
        let persist = should_persist_pull_progress(
            &progress,
            &mut last_progress_persisted_bytes,
            &mut last_progress_persisted_at,
        );
        if persist {
            progress_distribution.update_job_best_effort(&progress_job_id, |snapshot| {
                snapshot.apply_progress(progress);
            });
        } else {
            progress_distribution.update_job_in_memory(&progress_job_id, |snapshot| {
                snapshot.apply_progress(progress);
            });
        }
    };

    let result = if let Some(source_path) = source_path {
        install_model_pack_from_path(&resolved, source_path, &home, &mut progress)
    } else {
        PullModelPackRequest::new(&resolved, &home)
            .cancel(|| cancel_flag.load(Ordering::SeqCst))
            .pause(|| pause_flag.load(Ordering::SeqCst))
            .execute(&mut progress)
    };

    match result {
        Ok(pack) => {
            distribution.update_job_best_effort(&job_id, |snapshot| {
                if snapshot.state != PullJobState::AlreadyInstalled {
                    snapshot.state = PullJobState::Completed;
                }
                snapshot.control_requested = None;
                snapshot.bytes_done = snapshot.bytes_total;
                snapshot.speed_bps = None;
                snapshot.eta_s = Some(0);
                snapshot.installed_path = Some(pack.path.clone());
                snapshot.installed = Some(pack);
                snapshot.error = None;
            });
        }
        Err(PullError::Canceled { .. }) => {
            distribution.update_job_best_effort(&job_id, |snapshot| {
                snapshot.state = PullJobState::Canceled;
                snapshot.control_requested = None;
                snapshot.speed_bps = None;
                snapshot.eta_s = None;
                snapshot.error = Some("Pull job was canceled.".to_string());
            });
        }
        Err(PullError::Paused { .. }) => {
            distribution.update_job_best_effort(&job_id, |snapshot| {
                snapshot.state = PullJobState::Paused;
                snapshot.control_requested = None;
                snapshot.speed_bps = None;
                snapshot.eta_s = None;
                snapshot.error = Some("Pull job was paused.".to_string());
            });
        }
        Err(error) => {
            distribution.update_job_best_effort(&job_id, |snapshot| {
                snapshot.state = PullJobState::Failed;
                snapshot.control_requested = None;
                snapshot.speed_bps = None;
                snapshot.eta_s = None;
                snapshot.error = Some(error.to_string());
            });
        }
    }
}

pub(crate) fn should_persist_pull_progress(
    progress: &PullProgress,
    last_bytes: &mut u64,
    last_at: &mut Instant,
) -> bool {
    match progress {
        PullProgress::DownloadStarted { resume_from, .. } => {
            *last_bytes = *resume_from;
            *last_at = Instant::now();
            true
        }
        PullProgress::Downloading {
            bytes_done,
            bytes_total,
        } => {
            let now = Instant::now();
            let should_persist = bytes_done.saturating_sub(*last_bytes)
                >= PULL_JOB_PROGRESS_PERSIST_INTERVAL_BYTES
                || now.duration_since(*last_at) >= PULL_JOB_PROGRESS_PERSIST_INTERVAL
                || bytes_done >= bytes_total;
            if should_persist {
                *last_bytes = *bytes_done;
                *last_at = now;
            }
            should_persist
        }
        PullProgress::UsingInstalled { .. }
        | PullProgress::Verifying { .. }
        | PullProgress::Installed { .. } => true,
    }
}
