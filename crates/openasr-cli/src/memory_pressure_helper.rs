//! Qualification-only Windows host-memory pressure helper.
//!
//! The helper accepts a candidate rejection threshold rather than an arbitrary
//! allocation size. It commits and touches ordinary pageable memory only while
//! both an absolute and proportional safety floor remain intact. The process
//! owns a kill-on-close Job Object for its full lifetime, watches its parent,
//! and has a hard timeout. It never reads OpenASR state or participates in
//! runtime admission.

use std::io::Write as _;

use anyhow::{Context, Result, bail};
use serde::Serialize;

const PRESSURE_HELPER_SCHEMA: &str = "openasr.windows-memory-pressure-helper.v1";
#[cfg(any(windows, test))]
const MIB: u64 = 1024 * 1024;
#[cfg(any(windows, test))]
const GIB: u64 = 1024 * MIB;
#[cfg(any(windows, test))]
const MIN_ABSOLUTE_FLOOR_BYTES: u64 = 2 * GIB;
#[cfg(any(windows, test))]
const MIN_PROPORTIONAL_FLOOR_BASIS_POINTS: u16 = 2_000;
#[cfg(any(windows, test))]
const MAX_PROPORTIONAL_FLOOR_BASIS_POINTS: u16 = 9_000;
#[cfg(any(windows, test))]
const MIN_TIMEOUT_SECONDS: u64 = 5;
#[cfg(any(windows, test))]
const MAX_TIMEOUT_SECONDS: u64 = 120;
#[cfg(any(windows, test))]
const CROSSING_MARGIN_BYTES: u64 = 64 * MIB;
#[cfg(any(windows, test))]
const FLOOR_GUARD_BYTES: u64 = 256 * MIB;
#[cfg(any(windows, test))]
const MAX_ALLOCATION_FRACTION_DENOMINATOR: u64 = 2;

#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(any(windows, test)), allow(dead_code))]
pub(crate) struct PressureHelperOptions {
    pub(crate) parent_pid: u32,
    pub(crate) candidate_required_bytes: u64,
    pub(crate) absolute_floor_bytes: u64,
    pub(crate) proportional_floor_basis_points: u16,
    pub(crate) timeout_seconds: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(any(windows, test))]
struct PressurePlan {
    total_memory_bytes: u64,
    initial_available_bytes: u64,
    safety_floor_bytes: u64,
    target_available_bytes: u64,
    maximum_committed_bytes: u64,
}

#[cfg(any(windows, test))]
impl PressurePlan {
    fn build(
        options: PressureHelperOptions,
        total_memory_bytes: u64,
        initial_available_bytes: u64,
    ) -> Result<Self> {
        if options.parent_pid == 0 || options.parent_pid == std::process::id() {
            bail!("parent_pid must identify a distinct live parent process");
        }
        if options.absolute_floor_bytes < MIN_ABSOLUTE_FLOOR_BYTES {
            bail!("absolute_floor_bytes must be at least {MIN_ABSOLUTE_FLOOR_BYTES}");
        }
        if !(MIN_PROPORTIONAL_FLOOR_BASIS_POINTS..=MAX_PROPORTIONAL_FLOOR_BASIS_POINTS)
            .contains(&options.proportional_floor_basis_points)
        {
            bail!(
                "proportional_floor_basis_points must be between {MIN_PROPORTIONAL_FLOOR_BASIS_POINTS} and {MAX_PROPORTIONAL_FLOOR_BASIS_POINTS}"
            );
        }
        if !(MIN_TIMEOUT_SECONDS..=MAX_TIMEOUT_SECONDS).contains(&options.timeout_seconds) {
            bail!(
                "timeout_seconds must be between {MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS}"
            );
        }
        if total_memory_bytes == 0 || initial_available_bytes == 0 {
            bail!("native host memory observation is unavailable");
        }
        if initial_available_bytes <= options.candidate_required_bytes {
            bail!("baseline is already inadmissible for the candidate");
        }

        let proportional_floor = total_memory_bytes
            .saturating_mul(u64::from(options.proportional_floor_basis_points))
            / 10_000;
        let safety_floor_bytes = options.absolute_floor_bytes.max(proportional_floor);
        let guarded_floor = safety_floor_bytes
            .checked_add(FLOOR_GUARD_BYTES)
            .context("configured safety floor overflows")?;
        if initial_available_bytes <= guarded_floor {
            bail!("baseline available memory is already inside the guarded safety floor");
        }
        let target_available_bytes = options
            .candidate_required_bytes
            .checked_sub(CROSSING_MARGIN_BYTES)
            .context("candidate request is smaller than the required crossing margin")?;
        if target_available_bytes <= guarded_floor {
            bail!(
                "the candidate rejection threshold cannot be crossed without violating safety floors"
            );
        }

        let maximum_by_floor = initial_available_bytes - guarded_floor;
        let maximum_by_fraction = total_memory_bytes / MAX_ALLOCATION_FRACTION_DENOMINATOR;
        let maximum_committed_bytes = maximum_by_floor.min(maximum_by_fraction);
        let nominal_required = initial_available_bytes - target_available_bytes;
        if nominal_required > maximum_committed_bytes {
            bail!(
                "the candidate rejection threshold cannot be reached inside the bounded allocation budget"
            );
        }

        Ok(Self {
            total_memory_bytes,
            initial_available_bytes,
            safety_floor_bytes,
            target_available_bytes,
            maximum_committed_bytes,
        })
    }
}

#[derive(Debug, Serialize)]
struct PressureEvent<'a> {
    schema: &'static str,
    event: &'a str,
    result: &'a str,
    helper_pid: u32,
    parent_pid: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_memory_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    initial_available_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    observed_available_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lowest_available_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_floor_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate_required_bytes: Option<u64>,
    committed_bytes: u64,
    touched_bytes: u64,
    timeout_seconds: u64,
    job_kill_on_close: bool,
    parent_death_cleanup: bool,
    page_locking: bool,
}

fn emit(event: &PressureEvent<'_>) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string(event).context("could not serialize pressure-helper event")?
    );
    std::io::stdout()
        .flush()
        .context("could not flush pressure-helper event")
}

fn terminal_failure(options: PressureHelperOptions, reason: &str) {
    let _ = emit(&PressureEvent {
        schema: PRESSURE_HELPER_SCHEMA,
        event: "terminal",
        result: "fail",
        helper_pid: std::process::id(),
        parent_pid: options.parent_pid,
        reason: Some(reason),
        total_memory_bytes: None,
        initial_available_bytes: None,
        observed_available_bytes: None,
        lowest_available_bytes: None,
        safety_floor_bytes: None,
        candidate_required_bytes: Some(options.candidate_required_bytes),
        committed_bytes: 0,
        touched_bytes: 0,
        timeout_seconds: options.timeout_seconds,
        job_kill_on_close: false,
        parent_death_cleanup: false,
        page_locking: false,
    });
}

pub(crate) fn run(options: PressureHelperOptions) -> Result<()> {
    let result = platform::run(options);
    if let Err(error) = &result {
        terminal_failure(options, &error.to_string());
    }
    result
}

#[cfg(not(windows))]
mod platform {
    use super::*;

    pub(super) fn run(_options: PressureHelperOptions) -> Result<()> {
        bail!("the real host-memory pressure helper is available only on Windows")
    }
}

#[cfg(windows)]
mod platform {
    use std::{
        ffi::c_void,
        io::Read as _,
        ptr::{null, null_mut},
        sync::mpsc,
        time::{Duration, Instant},
    };

    use windows_sys::Win32::{
        Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0},
        System::{
            JobObjects::{
                AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
                SetInformationJobObject,
            },
            Memory::{
                MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
            },
            SystemInformation::{GetSystemInfo, SYSTEM_INFO},
            Threading::{GetCurrentProcess, OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject},
        },
    };

    use super::*;

    const ALLOCATION_CHUNK_BYTES: u64 = 64 * MIB;
    const OBSERVATION_SETTLE_MILLIS: u64 = 25;
    const MONITOR_INTERVAL_MILLIS: u64 = 100;
    const RECOVERY_TIMEOUT_SECONDS: u64 = 5;

    struct OwnedHandle(HANDLE);

    impl Drop for OwnedHandle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: the handle was returned by OpenProcess and is owned
                // exactly once by this wrapper.
                unsafe { CloseHandle(self.0) };
            }
        }
    }

    /// Intentionally has no Drop implementation. Closing the last handle of a
    /// kill-on-close Job Object would terminate this helper before its final
    /// receipt is flushed. Windows closes the process-lifetime handle during
    /// process teardown, which is exactly the containment boundary we need.
    struct ProcessLifetimeJob {
        _handle: HANDLE,
    }

    struct CommittedRegion {
        address: *mut c_void,
        size_bytes: usize,
    }

    impl Drop for CommittedRegion {
        fn drop(&mut self) {
            if !self.address.is_null() {
                // SAFETY: this is the original base address returned by
                // VirtualAlloc; dwSize must be zero with MEM_RELEASE.
                unsafe { VirtualFree(self.address, 0, MEM_RELEASE) };
            }
        }
    }

    fn last_os_error(context: &str) -> anyhow::Error {
        anyhow::anyhow!("{context}: {}", std::io::Error::last_os_error())
    }

    fn install_process_lifetime_job() -> Result<ProcessLifetimeJob> {
        // SAFETY: null security attributes/name request a private Job Object.
        let handle = unsafe { CreateJobObjectW(null(), null()) };
        if handle.is_null() {
            return Err(last_os_error("could not create pressure-helper Job Object"));
        }
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: `limits` has the exact structure required by the selected
        // information class and remains alive for the call.
        let configured = unsafe {
            SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                (&raw const limits).cast(),
                std::mem::size_of_val(&limits) as u32,
            )
        };
        if configured == 0 {
            // This process has not been assigned yet, so closing on setup
            // failure cannot terminate it.
            unsafe { CloseHandle(handle) };
            return Err(last_os_error(
                "could not configure Job Object kill-on-close",
            ));
        }
        // SAFETY: GetCurrentProcess is a valid pseudo-handle for assignment.
        let assigned = unsafe { AssignProcessToJobObject(handle, GetCurrentProcess()) };
        if assigned == 0 {
            unsafe { CloseHandle(handle) };
            return Err(last_os_error(
                "could not assign pressure helper to its Job Object",
            ));
        }
        Ok(ProcessLifetimeJob { _handle: handle })
    }

    fn open_parent(parent_pid: u32) -> Result<OwnedHandle> {
        // SAFETY: only SYNCHRONIZE access is requested; no process memory or
        // token access is needed by the helper.
        let handle = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, 0, parent_pid) };
        if handle.is_null() {
            return Err(last_os_error(
                "could not open qualification parent for liveness watch",
            ));
        }
        Ok(OwnedHandle(handle))
    }

    fn system_page_size() -> Result<usize> {
        let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
        // SAFETY: GetSystemInfo initializes the caller-owned structure.
        unsafe { GetSystemInfo(&mut info) };
        usize::try_from(info.dwPageSize)
            .ok()
            .filter(|size| *size > 0)
            .context("Windows returned an invalid system page size")
    }

    fn align_down(value: u64, alignment: usize) -> Result<usize> {
        let alignment = u64::try_from(alignment).context("page size does not fit u64")?;
        let aligned = value / alignment * alignment;
        usize::try_from(aligned)
            .context("pressure allocation does not fit this process address space")
    }

    fn commit_and_touch(size_bytes: usize, page_size: usize) -> Result<CommittedRegion> {
        if size_bytes == 0 {
            bail!("pressure allocation chunk rounded to zero");
        }
        // SAFETY: a null address asks Windows to choose a region. No executable
        // or locked pages are requested; MEM_COMMIT charges ordinary pageable
        // memory and writes below fault every page in.
        let address = unsafe {
            VirtualAlloc(
                null_mut(),
                size_bytes,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        if address.is_null() {
            return Err(last_os_error(
                "VirtualAlloc could not commit a bounded pressure chunk",
            ));
        }
        let region = CommittedRegion {
            address,
            size_bytes,
        };
        for offset in (0..size_bytes).step_by(page_size) {
            // SAFETY: every offset is inside the committed region. Volatile
            // writes prevent the compiler from deleting the page faults.
            unsafe { region.address.cast::<u8>().add(offset).write_volatile(0xA5) };
        }
        unsafe {
            region
                .address
                .cast::<u8>()
                .add(size_bytes - 1)
                .write_volatile(0x5A)
        };
        Ok(region)
    }

    fn available_memory() -> Result<u64> {
        openasr_core::host_available_memory_bytes()
            .context("Windows available-memory observation failed")
    }

    pub(super) fn run(options: PressureHelperOptions) -> Result<()> {
        let total_memory_bytes = openasr_core::host_total_memory_bytes()
            .context("Windows total-memory observation failed")?;
        let initial_available_bytes = available_memory()?;
        let plan = PressurePlan::build(options, total_memory_bytes, initial_available_bytes)?;
        let _job = install_process_lifetime_job()?;
        let parent = open_parent(options.parent_pid)?;
        let page_size = system_page_size()?;
        let mut regions = Vec::<CommittedRegion>::new();
        let mut committed_bytes = 0_u64;
        let mut touched_bytes = 0_u64;
        let mut observed_available_bytes = initial_available_bytes;
        let mut lowest_available_bytes = initial_available_bytes;

        while observed_available_bytes > plan.target_available_bytes
            && committed_bytes < plan.maximum_committed_bytes
        {
            let remaining_budget = plan.maximum_committed_bytes - committed_bytes;
            let desired = (observed_available_bytes - plan.target_available_bytes)
                .min(ALLOCATION_CHUNK_BYTES)
                .min(remaining_budget);
            let chunk = align_down(desired, page_size)?;
            if chunk == 0 {
                break;
            }
            let region = commit_and_touch(chunk, page_size)?;
            committed_bytes = committed_bytes.saturating_add(region.size_bytes as u64);
            touched_bytes = touched_bytes.saturating_add(region.size_bytes as u64);
            regions.push(region);
            std::thread::sleep(Duration::from_millis(OBSERVATION_SETTLE_MILLIS));
            observed_available_bytes = available_memory()?;
            lowest_available_bytes = lowest_available_bytes.min(observed_available_bytes);
            if observed_available_bytes < plan.safety_floor_bytes {
                bail!("native available memory crossed the configured safety floor");
            }
        }
        if observed_available_bytes >= options.candidate_required_bytes {
            bail!("bounded pressure did not cross the candidate rejection threshold");
        }

        emit(&PressureEvent {
            schema: PRESSURE_HELPER_SCHEMA,
            event: "ready",
            result: "holding",
            helper_pid: std::process::id(),
            parent_pid: options.parent_pid,
            reason: None,
            total_memory_bytes: Some(plan.total_memory_bytes),
            initial_available_bytes: Some(plan.initial_available_bytes),
            observed_available_bytes: Some(observed_available_bytes),
            lowest_available_bytes: Some(lowest_available_bytes),
            safety_floor_bytes: Some(plan.safety_floor_bytes),
            candidate_required_bytes: Some(options.candidate_required_bytes),
            committed_bytes,
            touched_bytes,
            timeout_seconds: options.timeout_seconds,
            job_kill_on_close: true,
            parent_death_cleanup: true,
            page_locking: false,
        })?;

        let (stop_tx, stop_rx) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("openasr-pressure-control".to_string())
            .spawn(move || {
                let mut byte = [0_u8; 1];
                let result = std::io::stdin()
                    .read(&mut byte)
                    .map(|count| (count == 1).then_some(byte[0]));
                let _ = stop_tx.send(result);
            })
            .context("could not start pressure-helper control reader")?;

        let deadline = Instant::now() + Duration::from_secs(options.timeout_seconds);
        loop {
            if let Ok(read_result) = stop_rx.try_recv() {
                match read_result.context("pressure-helper control channel failed")? {
                    Some(b'\n') => break,
                    Some(_) => bail!("pressure-helper received an invalid release command"),
                    None => bail!("pressure-helper control channel closed without release"),
                }
            }
            // SAFETY: parent.0 is a live SYNCHRONIZE handle owned by this scope.
            if unsafe { WaitForSingleObject(parent.0, 0) } == WAIT_OBJECT_0 {
                bail!("qualification parent exited while pressure was active");
            }
            if Instant::now() >= deadline {
                bail!("pressure-helper hard timeout expired");
            }
            observed_available_bytes = available_memory()?;
            lowest_available_bytes = lowest_available_bytes.min(observed_available_bytes);
            if observed_available_bytes < plan.safety_floor_bytes {
                bail!("native available memory crossed the configured safety floor");
            }
            std::thread::sleep(Duration::from_millis(MONITOR_INTERVAL_MILLIS));
        }

        drop(regions);
        let recovery_deadline = Instant::now() + Duration::from_secs(RECOVERY_TIMEOUT_SECONDS);
        observed_available_bytes = available_memory()?;
        while observed_available_bytes < options.candidate_required_bytes
            && Instant::now() < recovery_deadline
        {
            std::thread::sleep(Duration::from_millis(MONITOR_INTERVAL_MILLIS));
            observed_available_bytes = available_memory()?;
        }
        if observed_available_bytes < options.candidate_required_bytes {
            bail!("available memory did not recover after releasing pressure");
        }

        emit(&PressureEvent {
            schema: PRESSURE_HELPER_SCHEMA,
            event: "released",
            result: "pass",
            helper_pid: std::process::id(),
            parent_pid: options.parent_pid,
            reason: None,
            total_memory_bytes: Some(plan.total_memory_bytes),
            initial_available_bytes: Some(plan.initial_available_bytes),
            observed_available_bytes: Some(observed_available_bytes),
            lowest_available_bytes: Some(lowest_available_bytes),
            safety_floor_bytes: Some(plan.safety_floor_bytes),
            candidate_required_bytes: Some(options.candidate_required_bytes),
            committed_bytes,
            touched_bytes,
            timeout_seconds: options.timeout_seconds,
            job_kill_on_close: true,
            parent_death_cleanup: true,
            page_locking: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(candidate_required_bytes: u64) -> PressureHelperOptions {
        PressureHelperOptions {
            parent_pid: u32::MAX,
            candidate_required_bytes,
            absolute_floor_bytes: 2 * GIB,
            proportional_floor_basis_points: 2_000,
            timeout_seconds: 60,
        }
    }

    #[test]
    fn plan_requires_an_admissible_baseline_and_crossable_safe_threshold() {
        let plan = PressurePlan::build(options(5 * GIB), 16 * GIB, 7 * GIB).unwrap();
        assert_eq!(plan.safety_floor_bytes, 3 * GIB + GIB / 5);
        assert!(plan.target_available_bytes < 5 * GIB);
        assert!(plan.maximum_committed_bytes >= 2 * GIB);
    }

    #[test]
    fn plan_rejects_a_baseline_that_already_fails() {
        let error = PressurePlan::build(options(5 * GIB), 16 * GIB, 4 * GIB).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("baseline is already inadmissible")
        );
    }

    #[test]
    fn plan_rejects_crossing_that_would_enter_the_safety_floor() {
        let error = PressurePlan::build(options(3 * GIB), 16 * GIB, 7 * GIB).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot be crossed without violating")
        );
    }

    #[test]
    fn plan_rejects_weakened_or_unbounded_guardrails() {
        let mut invalid = options(5 * GIB);
        invalid.absolute_floor_bytes = GIB;
        assert!(PressurePlan::build(invalid, 16 * GIB, 7 * GIB).is_err());

        invalid = options(5 * GIB);
        invalid.proportional_floor_basis_points = 1_999;
        assert!(PressurePlan::build(invalid, 16 * GIB, 7 * GIB).is_err());

        invalid = options(5 * GIB);
        invalid.timeout_seconds = 121;
        assert!(PressurePlan::build(invalid, 16 * GIB, 7 * GIB).is_err());
    }
}
