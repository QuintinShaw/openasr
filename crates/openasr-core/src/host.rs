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

/// Best-effort memory currently available to new allocations, in bytes (i.e.
/// free-or-reclaimable-without-swapping), or `None` on platforms without a
/// probe. Used only for local diagnostics -- the daemon-log system line and
/// the failure-context line's "how much headroom was left" field -- never for
/// admission control, so an occasional `None` (unsupported platform, a failed
/// syscall) degrading to an absent log field is acceptable.
pub fn host_available_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    // `libc::mach_host_self` is marked deprecated upstream in favor of the
    // separate `mach2` crate; pulling in another crate for one function this
    // narrow does not fit this workspace's dependency-trimming posture (see
    // this module's and `stage_timing`'s doc comments), so the deprecation is
    // acknowledged here rather than by adding a dependency.
    #[allow(deprecated)]
    {
        // Mach's `host_statistics64(HOST_VM_INFO64)` reports page *counts*, not
        // bytes; the host's own page size (16 KiB on Apple Silicon, 4 KiB on
        // Intel Macs -- never assume either) comes from `sysconf(_SC_PAGESIZE)`.
        // "Available" is approximated as free + inactive pages: inactive pages
        // are clean and reclaimable without swapping, which is the same rough
        // notion Activity Monitor and `vm_stat` use, and closer to "can a new
        // allocation land here" than `free_count` alone (which undercounts by
        // excluding easily-reclaimable file-backed pages).
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return None;
        }
        let mut info: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
        let mut count = libc::HOST_VM_INFO64_COUNT;
        // SAFETY: `mach_host_self` returns a borrowed (no-release-needed) host
        // port; `host_statistics64` fills the caller-owned `info`, sized via
        // `count` set to the flavor's expected word count as the API requires.
        let ret = unsafe {
            libc::host_statistics64(
                libc::mach_host_self(),
                libc::HOST_VM_INFO64,
                (&mut info as *mut libc::vm_statistics64).cast::<libc::integer_t>(),
                &mut count,
            )
        };
        if ret != 0 {
            return None;
        }
        let available_pages = u64::from(info.free_count) + u64::from(info.inactive_count);
        Some(available_pages.saturating_mul(page_size as u64))
    }
    #[cfg(target_os = "linux")]
    {
        // The kernel-computed `MemAvailable` estimate (present since Linux
        // 3.14) already accounts for reclaimable caches/slabs; prefer it over
        // hand-rolling `MemFree + Cached` from the same file.
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        meminfo.lines().find_map(|line| {
            line.strip_prefix("MemAvailable:")?
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

        // SAFETY: same contract as `host_total_memory_bytes`'s Windows arm.
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        (ok != 0).then_some(status.ullAvailPhys)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
    {
        None
    }
}

/// Best-effort human-readable CPU model string (e.g. `"Apple M1"` /
/// `"Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz"` / `"AMD Ryzen 9 7950X"`),
/// or `None` on platforms without a probe. Diagnostics only, never parsed
/// back by this codebase -- format is whatever the OS reports.
pub fn host_cpu_model() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        sysctl_string("machdep.cpu.brand_string")
    }
    #[cfg(target_os = "linux")]
    {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        // x86 reports "model name"; some ARM boards report "Hardware" or
        // "Model" instead (no "model name" key at all) -- try each in turn
        // rather than assuming x86's key exists on every kernel.
        for key in ["model name", "Model", "Hardware"] {
            if let Some(value) = cpuinfo.lines().find_map(|line| {
                let (line_key, value) = line.split_once(':')?;
                (line_key.trim() == key).then(|| value.trim().to_string())
            }) && !value.is_empty()
            {
                return Some(value);
            }
        }
        None
    }
    #[cfg(windows)]
    {
        registry_string_value(
            c"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            c"ProcessorNameString",
        )
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
    {
        None
    }
}

/// `(os_name, os_version)` best-effort pair, e.g. `("macOS", "15.5")` /
/// `("Linux", "Ubuntu 24.04.1 LTS")` / `("Windows", "Windows 11 Pro (build
/// 26100)")`. `os_name` is always a fixed literal per platform (never
/// `None`); `os_version` is `None` only when the platform-specific probe
/// fails.
pub fn host_os_name_and_version() -> (&'static str, Option<String>) {
    #[cfg(target_os = "macos")]
    {
        // `kern.osproductversion` is the user-facing marketing version (e.g.
        // "15.5"); it has existed since 10.13.4, which is far below this
        // workspace's realistic support floor. `kern.osrelease` (the Darwin
        // kernel version, e.g. "24.5.0") is appended for precise bug-report
        // correlation, matching Apple's own "macOS 15.5 (24F74)"-style
        // reporting convention without needing the private build-number API.
        let product_version = sysctl_string("kern.osproductversion");
        let darwin_release = sysctl_string("kern.osrelease");
        let version = match (product_version, darwin_release) {
            (Some(product), Some(darwin)) => Some(format!("{product} (Darwin {darwin})")),
            (Some(product), None) => Some(product),
            (None, Some(darwin)) => Some(format!("Darwin {darwin}")),
            (None, None) => None,
        };
        ("macOS", version)
    }
    #[cfg(target_os = "linux")]
    {
        // `/etc/os-release`'s `PRETTY_NAME` is the de-facto standard
        // distro-identification file (systemd spec, but honored far beyond
        // systemd distros); falls back to the raw kernel release
        // (`uname -r` equivalent) via `/proc/sys/kernel/osrelease` when the
        // distro file is missing (minimal/embedded images).
        let pretty_name = std::fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|contents| {
                contents.lines().find_map(|line| {
                    let value = line.strip_prefix("PRETTY_NAME=")?;
                    Some(value.trim_matches('"').to_string())
                })
            });
        let version = pretty_name.or_else(|| {
            std::fs::read_to_string("/proc/sys/kernel/osrelease")
                .ok()
                .map(|release| format!("kernel {}", release.trim()))
        });
        ("Linux", version)
    }
    #[cfg(windows)]
    {
        let product_name = registry_string_value(
            c"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
            c"ProductName",
        );
        let build_number = registry_string_value(
            c"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
            c"CurrentBuildNumber",
        );
        let version = match (product_name, build_number) {
            (Some(name), Some(build)) => Some(format!("{name} (build {build})")),
            (Some(name), None) => Some(name),
            (None, Some(build)) => Some(format!("Windows (build {build})")),
            (None, None) => None,
        };
        ("Windows", version)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
    {
        ("Unknown", None)
    }
}

/// Reads a string-valued `sysctlbyname` key (macOS only): a null-terminated C
/// string whose length is not known ahead of time, so this probes the
/// required buffer size with a `NULL` output pointer first (the documented
/// `sysctlbyname` idiom), allocates exactly that much, then reads the value.
#[cfg(target_os = "macos")]
fn sysctl_string(name: &str) -> Option<String> {
    let name = std::ffi::CString::new(name).ok()?;
    let mut size = 0usize;
    // SAFETY: a null oldp/oldlenp-only call is the documented way to query
    // the required buffer size without writing through a null pointer.
    let probe = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if probe != 0 || size == 0 {
        return None;
    }
    let mut buffer = vec![0u8; size];
    // SAFETY: `buffer` is exactly `size` bytes as just reported by the probe
    // call above; `size` is passed back in as the buffer's capacity.
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            buffer.as_mut_ptr().cast::<libc::c_void>(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret != 0 {
        return None;
    }
    // `size` may include the trailing NUL; strip any trailing NUL bytes
    // before lossily decoding (sysctl string values are not guaranteed UTF-8,
    // though in practice CPU brand strings and OS version strings are ASCII).
    while buffer.last() == Some(&0) {
        buffer.pop();
    }
    Some(String::from_utf8_lossy(&buffer).into_owned())
}

/// Reads a single `REG_SZ` value under `HKEY_LOCAL_MACHINE\{subkey}` (Windows
/// only). Returns `None` on any failure (key/value missing, wrong type,
/// permission denied) -- this is a best-effort diagnostics probe, never a
/// hard dependency.
#[cfg(windows)]
fn registry_string_value(subkey: &std::ffi::CStr, value_name: &std::ffi::CStr) -> Option<String> {
    use windows_sys::Win32::Foundation::ERROR_SUCCESS;
    use windows_sys::Win32::System::Registry::{
        HKEY, HKEY_LOCAL_MACHINE, KEY_READ, REG_SZ, RegCloseKey, RegOpenKeyExA, RegQueryValueExA,
    };

    // The wide (`*W`) registry APIs are the normal Rust-on-Windows idiom, but
    // every key/value name this module reads is ASCII, so the simpler `*A`
    // (ANSI) entry points avoid a UTF-16 round trip for no behavioral
    // difference here.
    let mut key: HKEY = std::ptr::null_mut();
    // SAFETY: `subkey` and `value_name` are caller-provided, NUL-terminated
    // `&CStr`s (guaranteed by the type); `key` is an out-pointer this call
    // fills on success.
    let open_status = unsafe {
        RegOpenKeyExA(
            HKEY_LOCAL_MACHINE,
            subkey.as_ptr().cast(),
            0,
            KEY_READ,
            &mut key,
        )
    };
    if open_status != ERROR_SUCCESS {
        return None;
    }

    let mut value_type: u32 = 0;
    let mut data_len: u32 = 0;
    // First call with a null data buffer to discover the required length
    // (the documented `RegQueryValueEx` idiom), matching `sysctl_string`'s
    // two-call shape above.
    // SAFETY: all out-pointers are valid locals; the data buffer pointer is
    // null, which `RegQueryValueExA` accepts to mean "size query only".
    let probe_status = unsafe {
        RegQueryValueExA(
            key,
            value_name.as_ptr().cast(),
            std::ptr::null_mut(),
            &mut value_type,
            std::ptr::null_mut(),
            &mut data_len,
        )
    };
    if probe_status != ERROR_SUCCESS || value_type != REG_SZ || data_len == 0 {
        // SAFETY: `key` was successfully opened above.
        unsafe { RegCloseKey(key) };
        return None;
    }

    let mut buffer = vec![0u8; data_len as usize];
    // SAFETY: `buffer` is exactly `data_len` bytes as just reported by the
    // probe call; `data_len` is passed back in as the buffer's capacity.
    let read_status = unsafe {
        RegQueryValueExA(
            key,
            value_name.as_ptr().cast(),
            std::ptr::null_mut(),
            &mut value_type,
            buffer.as_mut_ptr(),
            &mut data_len,
        )
    };
    // SAFETY: `key` was successfully opened above.
    unsafe { RegCloseKey(key) };
    if read_status != ERROR_SUCCESS {
        return None;
    }
    while buffer.last() == Some(&0) {
        buffer.pop();
    }
    Some(String::from_utf8_lossy(&buffer).into_owned())
}

/// One-line, structured `daemon.log` summary of host system facts: OS
/// name+version, CPU model, and total/available physical RAM. Logged once at
/// daemon boot right next to the existing `stage=ggml_backend` device
/// enumeration line, so a support bundle of `daemon.log` + `desktop.log`
/// alone (no separate "what's your OS/CPU/RAM" back-and-forth) is enough to
/// start triaging a report. Deliberately host-hardware-only: no user data, no
/// file paths, no network calls (matches this crate's no-telemetry contract).
pub fn host_system_boot_summary() -> String {
    let (os_name, os_version) = host_os_name_and_version();
    let os_version = os_version.as_deref().unwrap_or("unknown");
    let cpu_model = host_cpu_model();
    let cpu_model = cpu_model.as_deref().unwrap_or("unknown");
    let mem_total_mib = host_total_memory_bytes().map(bytes_to_mib);
    let mem_available_mib = host_available_memory_bytes().map(bytes_to_mib);
    format!(
        "os={os_name} os_version=\"{os_version}\" cpu=\"{cpu_model}\" mem_total_mib={} mem_available_mib={}",
        mem_total_mib.map_or_else(|| "unknown".to_string(), |mib| mib.to_string()),
        mem_available_mib.map_or_else(|| "unknown".to_string(), |mib| mib.to_string()),
    )
}

fn bytes_to_mib(bytes: u64) -> u64 {
    bytes / (1024 * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_available_memory_is_plausible_on_supported_platforms() {
        if cfg!(any(target_os = "macos", target_os = "linux", windows)) {
            let available = host_available_memory_bytes()
                .expect("supported platform should return an available-memory probe");
            assert!(available > 0, "implausible zero available memory");
            if let Some(total) = host_total_memory_bytes() {
                assert!(
                    available <= total,
                    "available ({available}) must not exceed total ({total})"
                );
            }
        }
    }

    #[test]
    fn host_cpu_model_is_nonempty_on_supported_platforms() {
        if cfg!(any(target_os = "macos", target_os = "linux", windows)) {
            let model = host_cpu_model();
            // Best-effort: still allow `None` (e.g. a locked-down /proc on an
            // exotic Linux sandbox), but if present it must not be blank.
            if let Some(model) = model {
                assert!(!model.trim().is_empty());
            }
        }
    }

    #[test]
    fn host_os_name_and_version_reports_a_fixed_nonempty_name() {
        let (name, _version) = host_os_name_and_version();
        assert!(!name.is_empty());
    }

    #[test]
    fn host_system_boot_summary_has_expected_keys() {
        let summary = host_system_boot_summary();
        for key in [
            "os=",
            "os_version=",
            "cpu=",
            "mem_total_mib=",
            "mem_available_mib=",
        ] {
            assert!(summary.contains(key), "missing {key} in {summary:?}");
        }
    }
}
