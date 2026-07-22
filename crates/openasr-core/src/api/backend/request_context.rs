//! Structured, privacy-safe `daemon.log` lines for one transcription
//! request's context (what kind of request, which model/quant/backend, how
//! much audio) and, on failure, why it failed. Companion to
//! [`crate::host::host_system_boot_summary`] and the existing
//! `stage=ggml_backend` boot line: together they let a bug report's
//! `daemon.log` (plus the desktop's `desktop.log`) stand on its own for
//! triage, without asking the reporter what model/OS/RAM they were on.
//!
//! **Privacy contract**: neither line here may ever include a file name, a
//! file path, or any audio/transcript content -- only request *shape*
//! (source, model, backend, audio duration/format) and host resource
//! counters. [`format_request_context_line`] is covered by a regression test
//! asserting this.

use crate::stage_timing;

/// Which call path originated a native transcription request. Named after
/// the concrete entry points that exist today (see `rg
/// "TranscriptionRequest::new"`), not an aspirational taxonomy: the realtime
/// websocket path serves both the desktop's dictation and live-captions
/// features through the same per-utterance native request, and the wire
/// protocol carries no field distinguishing them, so both log as
/// `server_realtime` rather than fabricating a distinction this crate cannot
/// actually observe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequestSource {
    /// `openasr transcribe` (CLI, one-shot local file).
    CliTranscribe,
    /// `openasr live` (CLI, mic capture loop), one request per utterance.
    CliLive,
    /// `POST /v1/audio/transcriptions` (server, uploaded file).
    ServerTranscribe,
    /// `POST /v1/audio/translations` (server, uploaded file, forced
    /// `task=translate`).
    ServerTranslate,
    /// `/v1/audio/realtime` websocket session (server), one request per
    /// finalized utterance. Covers both the desktop's dictation and
    /// live-captions features -- see the type-level doc above.
    ServerRealtime,
    /// The request was built without an explicit source (a test helper, or a
    /// caller that has not been updated yet). Never intentionally emitted by
    /// a real entry point; exists so adding this field did not require
    /// touching every one of this crate's existing `TranscriptionRequest`
    /// construction sites.
    #[default]
    Unspecified,
}

impl RequestSource {
    pub fn as_log_label(self) -> &'static str {
        match self {
            Self::CliTranscribe => "cli_transcribe",
            Self::CliLive => "cli_live",
            Self::ServerTranscribe => "server_transcribe",
            Self::ServerTranslate => "server_translate",
            Self::ServerRealtime => "server_realtime",
            Self::Unspecified => "unspecified",
        }
    }
}

/// Formats the per-request context line's payload (everything after the
/// `stage_timing` prefix and component name). Pure formatting, no I/O --
/// separated from [`log_request_context`] so the "no filename field" privacy
/// contract can be regression-tested without capturing stderr.
///
/// `model_id` must already be the resolved bare model id (e.g.
/// `"whisper-small"`), never a file path; `quant_tag` and `backend_label` are
/// short fixed-vocabulary tokens (`"q4_k"`, `"metal"`, ...); `container` is a
/// codec/extension tag (e.g. `"wav"`), not a file name.
///
/// `sample_rate_hz`/`channels` are the *source* audio's real format (before
/// this crate's normalization pipeline resamples/downmixes), not the
/// normalized-pipeline constants -- print `"unknown"` (matching
/// [`format_failure_context_line`]'s degrade style) rather than a fabricated
/// number when the caller could not determine them.
#[allow(clippy::too_many_arguments)]
pub fn format_request_context_line(
    source: RequestSource,
    model_id: &str,
    quant_tag: &str,
    backend_label: &str,
    audio_duration_seconds: f32,
    container: &str,
    sample_rate_hz: Option<u32>,
    channels: Option<u16>,
) -> String {
    let sample_rate_hz = sample_rate_hz.map_or_else(|| "unknown".to_string(), |hz| hz.to_string());
    let channels = channels.map_or_else(|| "unknown".to_string(), |count| count.to_string());
    format!(
        "stage=request_context source={} model={model_id} quant={quant_tag} backend={backend_label} audio_duration_s={audio_duration_seconds:.2} container={container} sample_rate_hz={sample_rate_hz} channels={channels}",
        source.as_log_label(),
    )
}

/// Logs [`format_request_context_line`]'s output as an unconditional (not
/// `OPENASR_TIMING`-gated) `daemon.log` line -- request-context is baseline
/// observability, not opt-in profiling, matching `stage=ggml_backend` and
/// `stage=host_system`'s always-on posture.
#[allow(clippy::too_many_arguments)]
pub fn log_request_context(
    source: RequestSource,
    model_id: &str,
    quant_tag: &str,
    backend_label: &str,
    audio_duration_seconds: f32,
    container: &str,
    sample_rate_hz: Option<u32>,
    channels: Option<u16>,
) {
    stage_timing::log_event(
        "native_transcribe",
        format_request_context_line(
            source,
            model_id,
            quant_tag,
            backend_label,
            audio_duration_seconds,
            container,
            sample_rate_hz,
            channels,
        ),
    );
}

/// Coarse failure-category buckets a `BackendError` collapses into for the
/// failure-context log line. Reuses `BackendError`'s existing variants (and,
/// for the variants that flatten internal detail into a `NativeFailClosed`
/// reason string, the same marker-text sniffing
/// `gpu_buffer_allocation_failure_backend` already relies on) rather than
/// introducing a second, parallel error taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureCategory {
    /// A compute-buffer/GPU allocation failed (ggml
    /// `BackendBufferAllocationFailed`, surfaced as `NativeFailClosed`).
    Alloc,
    /// The input audio file/container could not be read or normalized.
    AudioIo,
    /// The requested model/pack path/id could not be resolved or did not
    /// match the loaded runtime source.
    ModelResolve,
    /// The request asked for a capability (diarization, phrase bias,
    /// adapter, word-timestamp alignment, ...) the resolved backend/model
    /// does not support; rejected before any decode ran.
    UnsupportedCapability,
    /// The request was canceled by the caller at a slice boundary.
    Canceled,
    /// A transient, retryable condition (e.g. serve-batch decode busy).
    Transient,
    /// Any other fail-closed decode/dispatch error not covered above.
    Decode,
}

impl FailureCategory {
    pub fn as_log_label(self) -> &'static str {
        match self {
            Self::Alloc => "alloc",
            Self::AudioIo => "audio_io",
            Self::ModelResolve => "model_resolve",
            Self::UnsupportedCapability => "unsupported_capability",
            Self::Canceled => "canceled",
            Self::Transient => "transient",
            Self::Decode => "decode",
        }
    }
}

/// Formats the failure-context line's payload. `available_memory_mib` and
/// `gpu_memory_mib` (`(free, total)`) are `None` when the corresponding probe
/// is unavailable (unsupported platform, no GPU-class device) -- the field is
/// then omitted entirely rather than printed as a sentinel, matching this
/// crate's other best-effort diagnostic lines (e.g. `ggml_runtime_boot_summary`
/// omitting `gpu_selection=` when there is nothing to disambiguate).
pub fn format_failure_context_line(
    category: FailureCategory,
    available_memory_mib: Option<u64>,
    gpu_memory_mib: Option<(u64, u64)>,
) -> String {
    let mut line = format!(
        "stage=transcribe_failure error_category={}",
        category.as_log_label()
    );
    match available_memory_mib {
        Some(mib) => line.push_str(&format!(" mem_available_mib={mib}")),
        None => line.push_str(" mem_available_mib=unknown"),
    }
    if let Some((free_mib, total_mib)) = gpu_memory_mib {
        line.push_str(&format!(
            " gpu_mem_free_mib={free_mib} gpu_mem_total_mib={total_mib}"
        ));
    }
    line
}

/// Logs [`format_failure_context_line`]'s output at the moment a native
/// transcription request fails, using the process's current best-effort
/// available-memory reading (and, if a GPU-class device is present, its
/// free/total VRAM from the same device enumeration `stage=ggml_backend`
/// logs at boot) -- never re-probing anything expensive, and never a hard
/// dependency: a probe returning `None` just narrows the logged line.
pub fn log_failure_context(category: FailureCategory) {
    let available_memory_mib =
        crate::host_available_memory_bytes().map(|bytes| bytes / (1024 * 1024));
    let gpu_memory_mib = first_gpu_class_device_memory_mib();
    stage_timing::log_event(
        "native_transcribe",
        format_failure_context_line(category, available_memory_mib, gpu_memory_mib),
    );
}

/// Best-effort `(free_mib, total_mib)` for the first GPU-class device ggml's
/// device registry reports memory for. Reuses the same
/// `ggml_runtime_info()`/`GgmlBackendDevice` enumeration the boot-time
/// `stage=ggml_backend` line already walks, rather than a second device
/// probe -- this is a point-in-time snapshot at failure time, not the boot
/// snapshot, so it can catch VRAM pressure that developed since boot.
fn first_gpu_class_device_memory_mib() -> Option<(u64, u64)> {
    use crate::ggml_runtime::GgmlBackendKind;

    crate::ggml_runtime_info()
        .devices
        .iter()
        .find_map(|device| {
            let is_gpu_class = matches!(
                device.kind,
                GgmlBackendKind::Gpu
                    | GgmlBackendKind::IntegratedGpu
                    | GgmlBackendKind::Accelerator
            );
            let memory = device.memory?;
            is_gpu_class.then_some((
                (memory.free_bytes as u64) / (1024 * 1024),
                (memory.total_bytes as u64) / (1024 * 1024),
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_context_line_has_no_filename_or_path_field() {
        let line = format_request_context_line(
            RequestSource::ServerTranscribe,
            "whisper-small",
            "q4_k",
            "metal",
            12.34,
            "wav",
            Some(16_000),
            Some(1),
        );
        // Regression guard for the privacy contract: no field named
        // path/file/input in the formatted line, and no path separators that
        // would suggest one snuck in through another field.
        for forbidden in ["path=", "file=", "filename=", "input_path", "/", "\\"] {
            assert!(
                !line.contains(forbidden),
                "request context line unexpectedly contains {forbidden:?}: {line}"
            );
        }
        assert!(line.contains("source=server_transcribe"));
        assert!(line.contains("model=whisper-small"));
        assert!(line.contains("quant=q4_k"));
        assert!(line.contains("backend=metal"));
        assert!(line.contains("audio_duration_s=12.34"));
        assert!(line.contains("container=wav"));
        assert!(line.contains("sample_rate_hz=16000"));
        assert!(line.contains("channels=1"));
    }

    #[test]
    fn request_context_line_reports_the_true_source_format_not_a_normalized_constant() {
        // Regression guard for the field's honesty contract: a non-16 kHz,
        // non-mono source (e.g. a 44.1 kHz stereo m4a export) must show up as
        // its own real numbers, not silently collapse to the normalization
        // pipeline's 16000/1 constants.
        let line = format_request_context_line(
            RequestSource::ServerTranscribe,
            "whisper-small",
            "q4_k",
            "metal",
            12.34,
            "m4a",
            Some(44_100),
            Some(2),
        );
        assert!(line.contains("sample_rate_hz=44100"));
        assert!(line.contains("channels=2"));
        assert!(!line.contains("sample_rate_hz=16000"));
        assert!(!line.contains("channels=1"));
    }

    #[test]
    fn request_context_line_degrades_to_unknown_when_source_format_is_unavailable() {
        // Honesty contract, the other direction: a caller with no probed
        // source format passes `None`, which must render as `unknown`, never
        // a fabricated/default number.
        let line = format_request_context_line(
            RequestSource::CliTranscribe,
            "whisper-small",
            "q4_k",
            "cpu",
            5.0,
            "wav",
            None,
            None,
        );
        assert!(line.contains("sample_rate_hz=unknown"));
        assert!(line.contains("channels=unknown"));
    }

    #[test]
    fn request_source_labels_are_stable_and_distinct() {
        let all = [
            RequestSource::CliTranscribe,
            RequestSource::CliLive,
            RequestSource::ServerTranscribe,
            RequestSource::ServerTranslate,
            RequestSource::ServerRealtime,
            RequestSource::Unspecified,
        ];
        let mut labels: Vec<&'static str> =
            all.iter().map(|source| source.as_log_label()).collect();
        let original_len = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), original_len, "duplicate RequestSource labels");
    }

    #[test]
    fn failure_context_line_reports_unknown_when_probes_are_absent() {
        let line = format_failure_context_line(FailureCategory::Alloc, None, None);
        assert_eq!(
            line,
            "stage=transcribe_failure error_category=alloc mem_available_mib=unknown"
        );
    }

    #[test]
    fn failure_context_line_includes_gpu_memory_when_present() {
        let line =
            format_failure_context_line(FailureCategory::Alloc, Some(2048), Some((512, 8192)));
        assert!(line.contains("mem_available_mib=2048"));
        assert!(line.contains("gpu_mem_free_mib=512"));
        assert!(line.contains("gpu_mem_total_mib=8192"));
    }

    #[test]
    fn failure_category_labels_are_distinct() {
        let all = [
            FailureCategory::Alloc,
            FailureCategory::AudioIo,
            FailureCategory::ModelResolve,
            FailureCategory::UnsupportedCapability,
            FailureCategory::Canceled,
            FailureCategory::Transient,
            FailureCategory::Decode,
        ];
        let mut labels: Vec<&'static str> = all.iter().map(|c| c.as_log_label()).collect();
        let original_len = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(
            labels.len(),
            original_len,
            "duplicate FailureCategory labels"
        );
    }
}
