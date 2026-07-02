//! Host hardware probes (physical RAM) used to size pull-time quant
//! recommendations. These are host-capability queries, independent of catalog
//! parsing — the recommendation *type* they return lives with the catalog
//! schema in [`crate::registry`].

use crate::registry::CatalogQuantRecommendationProfile;

/// Best-effort total physical RAM of the host in bytes, used to pick a
/// device-recommended quant at pull time. Returns `None` on platforms without a
/// probe (callers then fall back to the catalog's static recommended quant).
pub fn host_total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let mut value: u64 = 0;
        let mut size = std::mem::size_of::<u64>();
        // SAFETY: hw.memsize writes a single u64; size is initialized to its width.
        let ret = unsafe {
            libc::sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&mut value as *mut u64).cast::<libc::c_void>(),
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        (ret == 0 && value > 0).then_some(value)
    }
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        meminfo.lines().find_map(|line| {
            line.strip_prefix("MemTotal:")?
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()
                .map(|kb| kb.saturating_mul(1024))
        })
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

        // SAFETY: GlobalMemoryStatusEx requires the caller to set dwLength to the
        // struct's byte size before the call; it then fills the remaining fields.
        // We zero the struct first so every field has a defined value, set
        // dwLength, and pass a valid out-pointer. A zero return means failure.
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        (ok != 0 && status.ullTotalPhys > 0).then_some(status.ullTotalPhys)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
    {
        None
    }
}

/// A quant-recommendation profile budgeting ~75% of host RAM (mirroring the
/// desktop install picker). Empty (no budget) when RAM cannot be probed, which
/// makes `recommend_catalog_quant` fall back to the catalog default.
pub fn host_quant_recommendation_profile() -> CatalogQuantRecommendationProfile {
    CatalogQuantRecommendationProfile {
        memory_budget_bytes: host_total_memory_bytes().map(|total| total / 4 * 3),
    }
}
