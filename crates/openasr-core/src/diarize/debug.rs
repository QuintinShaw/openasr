//! Shared `OPENASR_DIARIZE_DEBUG` gate for diarization decision tracing.
//!
//! Both the realtime hook (`streaming.rs`) and the batch attribution path
//! consult this single switch so one env var lights up the whole pipeline.
//! The env read is cached, so a disabled gate costs one boolean load.

use std::sync::OnceLock;

const DIARIZE_DEBUG_ENV: &str = "OPENASR_DIARIZE_DEBUG";

/// Whether `OPENASR_DIARIZE_DEBUG` requests diarization decision traces on
/// stderr. Empty, `0`, and `false` (any case) mean off.
pub(crate) fn diarize_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(DIARIZE_DEBUG_ENV)
            .map(|value| {
                let value = value.trim();
                !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
            })
            .unwrap_or(false)
    })
}
