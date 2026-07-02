//! Process-wide peak resident-set-size probe for the performance harness.
//!
//! On unix this is `getrusage` via a local `extern` block (avoiding a `libc`
//! dependency); `ru_maxrss` units differ by platform — **bytes on macOS,
//! kilobytes on Linux** — and are normalized to bytes here. On Windows it is
//! `K32GetProcessMemoryInfo` via the already-present `windows-sys` crate,
//! reading `PeakWorkingSetSize` (already in bytes).
//!
//! Caveat: this is a *process* high-water mark, not a per-call delta. A harness
//! that loads several multi-GB packs in one process will see later entries
//! inherit earlier peaks. Run entries sequentially and trust the largest-pack
//! entry; per-entry isolation would need a subprocess-per-entry mode.

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

    #[cfg(target_os = "macos")]
    {
        Some(max_rss) // bytes
    }
    #[cfg(not(target_os = "macos"))]
    {
        Some(max_rss.saturating_mul(1024)) // kilobytes -> bytes (Linux/BSD)
    }
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

/// Other unsupported platforms: no probe.
#[cfg(not(any(unix, windows)))]
pub fn peak_rss_bytes() -> Option<u64> {
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
        let peak = peak_rss_bytes().expect("unix/windows platforms expose a peak-RSS probe");
        // A running test process holds at least a few MB resident.
        assert!(peak >= 1024 * 1024, "implausibly small peak: {peak} bytes");
    }
}
