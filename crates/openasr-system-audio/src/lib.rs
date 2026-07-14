use std::sync::{Arc, atomic::AtomicBool};

use serde::Serialize;

#[cfg(any(target_os = "linux", target_os = "macos", windows, test))]
#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", windows)),
    allow(dead_code)
)]
mod pcm;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(test)]
mod smoke_test_support;
#[cfg(all(not(windows), not(target_os = "linux"), not(target_os = "macos")))]
mod stub;
#[cfg(windows)]
mod windows;

#[cfg(target_os = "linux")]
use linux as platform;
#[cfg(target_os = "macos")]
use macos as platform;
#[cfg(all(not(windows), not(target_os = "linux"), not(target_os = "macos")))]
use stub as platform;
#[cfg(windows)]
use windows as platform;

#[derive(Debug, Clone, Serialize)]
pub struct SystemAudioSupport {
    pub supported: bool,
    pub label: String,
    pub detail: String,
    pub platform: String,
}

#[derive(Debug)]
pub struct CaptureBackendError {
    pub code: &'static str,
    pub message: String,
    pub diagnostic: String,
}

/// Whether a target process should include or exclude its child-process tree
/// when the platform backend supports per-process loopback capture (see
/// `process_loopback_support`). Named after the Windows
/// `AUDIOCLIENT_PROCESS_LOOPBACK_MODE` semantics, the only backend that
/// implements this today; other platforms report `supported: false` via
/// `process_loopback_support` instead of interpreting this value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessLoopbackMode {
    /// Capture audio from the target process and every process it spawns.
    IncludeProcessTree,
    /// Capture audio from only the target process, not its children.
    ExcludeProcessTree,
}

/// Capability probe for per-process loopback capture, distinct from
/// `SystemAudioSupport` (which covers the existing all-system loopback path).
/// A platform can support all-system loopback while reporting `supported:
/// false` here (e.g. macOS/Linux today, or Windows older than the 2004
/// update, which lacks `AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK`).
#[derive(Debug, Clone, Serialize)]
pub struct ProcessLoopbackSupport {
    pub supported: bool,
    pub detail: String,
    pub platform: String,
}

/// A running process a caller can offer as a per-process loopback capture
/// target. `name` is the best-effort executable/display name; enumeration
/// must stay panic-free and skip processes it cannot read rather than fail
/// the whole listing.
#[derive(Debug, Clone, Serialize)]
pub struct CandidateProcess {
    pub pid: u32,
    pub name: String,
}

pub fn support_status() -> SystemAudioSupport {
    platform::support_status()
}

pub fn run_loopback_capture(
    stop: Arc<AtomicBool>,
    on_frame: impl FnMut(Vec<i16>) -> Result<(), String>,
    on_diagnostic: impl FnMut(&str) -> Result<(), String>,
) -> Result<String, CaptureBackendError> {
    platform::run_loopback_capture(stop, on_frame, on_diagnostic)
}

/// Capability probe for `run_process_loopback_capture`. Cheap enough to call
/// before showing per-process capture UI; platforms without an
/// implementation return `supported: false` rather than panicking.
pub fn process_loopback_support() -> ProcessLoopbackSupport {
    platform::process_loopback_support()
}

/// Lists candidate processes a caller may pick as a per-process loopback
/// target. Returns a typed `unsupported` `CaptureBackendError` on platforms
/// without an implementation.
pub fn list_candidate_processes() -> Result<Vec<CandidateProcess>, CaptureBackendError> {
    platform::list_candidate_processes()
}

/// Per-process loopback capture: only audio rendered by `process_id` (and,
/// depending on `mode`, its child processes) is captured, instead of the
/// whole system. Platforms without an implementation fail closed with a
/// typed `unsupported` `CaptureBackendError` rather than panicking or
/// silently falling back to all-system capture.
pub fn run_process_loopback_capture(
    process_id: u32,
    mode: ProcessLoopbackMode,
    stop: Arc<AtomicBool>,
    on_frame: impl FnMut(Vec<i16>) -> Result<(), String>,
    on_diagnostic: impl FnMut(&str) -> Result<(), String>,
) -> Result<String, CaptureBackendError> {
    platform::run_process_loopback_capture(process_id, mode, stop, on_frame, on_diagnostic)
}
