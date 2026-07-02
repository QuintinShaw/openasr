use std::sync::{Arc, atomic::AtomicBool};

use crate::{CaptureBackendError, SystemAudioSupport};

pub fn support_status() -> SystemAudioSupport {
    SystemAudioSupport {
        supported: false,
        label: "System audio (M49C strategy)".to_string(),
        detail: unsupported_detail(),
        platform: std::env::consts::OS.to_string(),
    }
}

pub fn run_loopback_capture(
    _stop: Arc<AtomicBool>,
    _on_frame: impl FnMut(Vec<i16>) -> Result<(), String>,
    _on_diagnostic: impl FnMut(&str) -> Result<(), String>,
) -> Result<String, CaptureBackendError> {
    Err(CaptureBackendError {
        code: "unsupported",
        message: "System-audio smoke capture is not implemented on this platform.".to_string(),
        diagnostic: unsupported_detail(),
    })
}

fn unsupported_detail() -> String {
    // This stub only compiles for platforms without a real backend (macOS,
    // Linux, and Windows each route to their own module), so keep the message
    // platform-agnostic rather than describing macOS/Linux as unimplemented.
    "System-audio smoke capture is not implemented on this platform.".to_string()
}

#[cfg(test)]
mod tests {
    use super::support_status;

    #[test]
    fn stub_reports_not_supported() {
        let support = support_status();
        assert!(!support.supported);
        assert!(support.detail.contains("not implemented"));
    }
}
