//! Process resident-set-size probes for the performance harness.
//!
//! Peak RSS uses `getrusage`; `ru_maxrss` units differ by platform — **bytes on macOS,
//! kilobytes on Linux** — and are normalized to bytes here. On Windows it is
//! `K32GetProcessMemoryInfo` via the already-present `windows-sys` crate,
//! reading `PeakWorkingSetSize` (already in bytes).
//!
//! Current RSS is intentionally separate from the monotonic peak: it lets a
//! long-running model gate distinguish a transient driver/compiler spike from
//! memory that remains resident while the runtime is warm. Darwin uses Mach
//! task info, Linux reads `/proc/self/statm`, and Windows reads
//! `WorkingSetSize` from the same process counters. Darwin additionally exposes
//! `phys_footprint`, the kernel's preferred process-level physical-memory
//! accounting, including compressed memory and device-backed ledgers that
//! resident size alone can misrepresent.
//!
//! Caveat: this is a *process* high-water mark, not a per-call delta. A harness
//! that loads several multi-GB packs in one process will see later entries
//! inherit earlier peaks. Run entries sequentially and trust the largest-pack
//! entry; per-entry isolation would need a subprocess-per-entry mode.

/// Process-memory counters captured at one lifecycle boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProcessMemorySnapshot {
    pub current_rss_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    /// Darwin `phys_footprint`; unavailable on other platforms.
    pub current_phys_footprint_bytes: Option<u64>,
    /// Darwin lifetime maximum `phys_footprint`; unavailable on other platforms.
    pub peak_phys_footprint_bytes: Option<u64>,
}

/// Capture every process-memory counter supported by this platform.
pub fn process_memory_snapshot() -> ProcessMemorySnapshot {
    let (current_phys_footprint_bytes, peak_phys_footprint_bytes) = physical_footprint_bytes();
    ProcessMemorySnapshot {
        current_rss_bytes: current_rss_bytes(),
        peak_rss_bytes: peak_rss_bytes(),
        current_phys_footprint_bytes,
        peak_phys_footprint_bytes,
    }
}

/// Peak resident set size of the current process in bytes, or `None` if the
/// platform has no supported probe.
#[cfg(unix)]
pub fn peak_rss_bytes() -> Option<u64> {
    use std::os::raw::{c_int, c_long};

    // Minimal `struct rusage` layout. Only `ru_maxrss` is read; the leading
    // two `timeval`s and the trailing `c_long` counters are sized to match the
    // platform ABI so the offset of `ru_maxrss` is correct.
    #[repr(C)]
    struct Timeval {
        tv_sec: c_long,
        tv_usec: c_long,
    }

    #[repr(C)]
    struct Rusage {
        ru_utime: Timeval,
        ru_stime: Timeval,
        ru_maxrss: c_long,
        ru_ixrss: c_long,
        ru_idrss: c_long,
        ru_isrss: c_long,
        ru_minflt: c_long,
        ru_majflt: c_long,
        ru_nswap: c_long,
        ru_inblock: c_long,
        ru_oublock: c_long,
        ru_msgsnd: c_long,
        ru_msgrcv: c_long,
        ru_nsignals: c_long,
        ru_nvcsw: c_long,
        ru_nivcsw: c_long,
    }

    const RUSAGE_SELF: c_int = 0;

    unsafe extern "C" {
        fn getrusage(who: c_int, usage: *mut Rusage) -> c_int;
    }

    // SAFETY: `getrusage` fills a caller-owned `Rusage`; the struct matches the
    // platform ABI for the fields preceding and including `ru_maxrss`.
    let mut usage: Rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { getrusage(RUSAGE_SELF, &mut usage) };
    if rc != 0 || usage.ru_maxrss <= 0 {
        return None;
    }
    let max_rss = usage.ru_maxrss as u64;

    // `ru_maxrss` units follow the kernel, not just the OS name: Darwin (macOS
    // and iOS alike) reports bytes; Linux/BSD report kilobytes.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        Some(max_rss) // bytes
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        Some(max_rss.saturating_mul(1024)) // kilobytes -> bytes (Linux/BSD)
    }
}

/// Current resident set size of this process in bytes.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn current_rss_bytes() -> Option<u64> {
    let mut info: libc::mach_task_basic_info = unsafe { std::mem::zeroed() };
    let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
    #[allow(deprecated)]
    let task = unsafe { libc::mach_task_self() };
    let result = unsafe {
        libc::task_info(
            task,
            libc::MACH_TASK_BASIC_INFO,
            (&mut info as *mut libc::mach_task_basic_info).cast::<libc::integer_t>(),
            &mut count,
        )
    };
    if result != libc::KERN_SUCCESS || count < libc::MACH_TASK_BASIC_INFO_COUNT {
        return None;
    }
    let resident_size = info.resident_size;
    (resident_size > 0).then_some(resident_size)
}

/// Darwin process physical footprint and its lifetime maximum.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn physical_footprint_bytes() -> (Option<u64>, Option<u64>) {
    let mut usage: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
    let rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            (&mut usage as *mut libc::rusage_info_v4).cast(),
        )
    };
    if rc == 0 {
        return (
            (usage.ri_phys_footprint > 0).then_some(usage.ri_phys_footprint),
            (usage.ri_lifetime_max_phys_footprint > 0)
                .then_some(usage.ri_lifetime_max_phys_footprint),
        );
    }

    // V0 has carried the current footprint since macOS 10.9 / iOS 7. Fall
    // back to it if an older kernel rejects the lifetime-peak V4 flavor.
    let mut fallback: libc::rusage_info_v0 = unsafe { std::mem::zeroed() };
    let fallback_rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V0,
            (&mut fallback as *mut libc::rusage_info_v0).cast(),
        )
    };
    if fallback_rc != 0 {
        return (None, None);
    }
    (
        (fallback.ri_phys_footprint > 0).then_some(fallback.ri_phys_footprint),
        None,
    )
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn physical_footprint_bytes() -> (Option<u64>, Option<u64>) {
    (None, None)
}

/// Current resident set size on procfs Unix platforms.
#[cfg(all(unix, not(any(target_os = "macos", target_os = "ios"))))]
pub fn current_rss_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages = statm.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if resident_pages == 0 || page_size <= 0 {
        return None;
    }
    resident_pages.checked_mul(page_size as u64)
}

/// Windows: peak working set size (the process high-water resident memory) via
/// `K32GetProcessMemoryInfo`, already in bytes.
#[cfg(windows)]
pub fn peak_rss_bytes() -> Option<u64> {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: GetCurrentProcess returns a pseudo-handle (no close required);
    // K32GetProcessMemoryInfo fills the caller-owned `counters`, whose `cb` we
    // set to its size first as the API requires.
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if ok == 0 || counters.PeakWorkingSetSize == 0 {
        return None;
    }
    Some(counters.PeakWorkingSetSize as u64)
}

#[cfg(windows)]
pub fn current_rss_bytes() -> Option<u64> {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if ok == 0 || counters.WorkingSetSize == 0 {
        return None;
    }
    Some(counters.WorkingSetSize as u64)
}

/// Other unsupported platforms: no probe.
#[cfg(not(any(unix, windows)))]
pub fn peak_rss_bytes() -> Option<u64> {
    None
}

#[cfg(not(any(unix, windows)))]
pub fn current_rss_bytes() -> Option<u64> {
    None
}

#[cfg(all(test, any(unix, windows)))]
mod tests {
    use super::*;

    #[test]
    fn probe_reports_plausible_nonzero_peak() {
        // Allocate something measurable so the high-water mark is clearly set.
        let blob = vec![0u8; 8 * 1024 * 1024];
        std::hint::black_box(&blob);
        let current = current_rss_bytes().expect("platform exposes a current-RSS probe");
        assert!(
            current >= 1024 * 1024,
            "implausibly small current RSS: {current} bytes"
        );
        let peak = peak_rss_bytes().expect("unix/windows platforms expose a peak-RSS probe");
        assert!(peak >= 1024 * 1024, "implausibly small peak: {peak} bytes");
        // The two kernel APIs use different accounting snapshots (Mach task
        // residency versus getrusage high-water), so page-sized differences
        // must not be treated as a probe failure.
    }

    #[test]
    fn snapshot_preserves_supported_counters() {
        let snapshot = process_memory_snapshot();
        assert!(snapshot.current_rss_bytes.is_some());
        assert!(snapshot.peak_rss_bytes.is_some());
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            assert!(snapshot.current_phys_footprint_bytes.is_some());
            assert!(snapshot.peak_phys_footprint_bytes.is_some());
        }
    }
}
