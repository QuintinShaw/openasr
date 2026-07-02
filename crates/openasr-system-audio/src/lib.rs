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
