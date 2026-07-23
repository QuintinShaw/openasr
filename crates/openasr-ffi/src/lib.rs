//! C ABI for embedding the OpenASR engine in iOS/macOS apps (see
//! `scripts/build-xcframework.sh` and `docs/SDK_IOS_MACOS.md`).
//!
//! Two entry points, both fed the caller's own local `.oasr` pack:
//!
//! - **Batch**: load a pack, transcribe one whole in-memory 16 kHz mono PCM
//!   buffer (f32 or i16), get back text plus optional segment timestamps
//!   (`openasr_model_open` / `openasr_transcribe_pcm` / `openasr_result_*`).
//! - **Streaming**: open a live session, feed 16 kHz mono f32 PCM chunks and
//!   receive incremental partial/committed transcript events, then finish for
//!   the assembled final transcript (`openasr_streaming_*`). This is the
//!   in-process, transport-free path an iOS app uses for live captioning --
//!   iOS cannot spawn the desktop realtime server, so it links the same
//!   `openasr_core::StreamingSession` engine directly.
//!
//! No server, no model download -- SDK consumers manage their own model
//! distribution (the product's "no silent download" boundary is a CLI/server
//! concern, not an SDK one; see `AGENTS.md`).
//!
//! The API shape follows whisper.cpp's C API convention: opaque handles plus
//! index-based accessor functions for the result
//! (`openasr_result_segment_count` / `openasr_result_segment_text` / ...)
//! rather than a `#[repr(C)]` struct with owned Rust data (`String`, `Vec`)
//! inline -- that would not be a valid C type and cbindgen cannot represent it
//! either.
//!
//! # Safety / fail-closed posture
//!
//! Every exported function is wrapped in [`std::panic::catch_unwind`]: a
//! panic anywhere in the engine or this shim is caught and turned into
//! [`OpenAsrStatus::InternalPanic`] plus a last-error message, never
//! unwinds across the FFI boundary (that is undefined behavior in C). Null
//! handles, null/invalid pointers, malformed UTF-8 paths, non-16 kHz or
//! non-mono input, out-of-range segment indices, and unreadable/invalid
//! `.oasr` packs all fail closed with a typed status code (or a null/zero
//! sentinel for pure accessors) -- never a fabricated transcript, never
//! reading out of bounds.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::ptr;

use openasr_core::{
    NATIVE_RUNTIME_MODEL_ID_AUTO, NativeAsrHardwareTarget, NativeBackend, StreamingConfig,
    StreamingEvent, StreamingEventKind, StreamingSession, Transcription, TranscriptionBackend,
    TranscriptionRequest, validate_local_native_model_pack_path,
};

/// Model-market C ABI: fetch/verify the signed catalog, pull (download +
/// sha256-verify + install) a model pack under explicit consent, list installed
/// packs, and remove one. See the module for how it keeps the "no silent
/// download" and "verification stays in the open core" boundaries.
mod market;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

pub(crate) fn set_last_error(message: impl Into<String>) {
    let message = message.into();
    // A NUL byte inside the error text would make `CString::new` fail; that
    // can only happen for pathological upstream error strings, so fall back
    // to a fixed-ASCII placeholder rather than dropping the error silently.
    let cstring = CString::new(message).unwrap_or_else(|_| {
        CString::new("openasr: error message contained an embedded NUL byte").unwrap()
    });
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(cstring));
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Replaces embedded NUL bytes (which cannot appear in a C string) with a
/// space instead of dropping content the engine already produced.
pub(crate) fn cstring_lossy(text: String) -> CString {
    CString::new(text).unwrap_or_else(|error| {
        let sanitized: Vec<u8> = error
            .into_vec()
            .into_iter()
            .map(|byte| if byte == 0 { b' ' } else { byte })
            .collect();
        CString::new(sanitized).expect("NUL bytes were replaced above")
    })
}

/// Status codes returned by every `openasr_*` call that can fail. `Ok` is
/// always `0`; every other variant means the requested handle/buffer was not
/// produced (or not mutated) and the caller should consult
/// [`openasr_last_error_message`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAsrStatus {
    /// Call succeeded.
    Ok = 0,
    /// A required pointer was null, or an argument was out of the range this
    /// v1 API supports (e.g. a sample rate other than 16000, or a channel
    /// count other than mono).
    InvalidArgument = 1,
    /// The `.oasr` path was rejected: missing, not a recognized runtime
    /// format, or failed the local-path/format trust checks.
    ModelLoadFailed = 2,
    /// The engine returned an error while transcribing (bad/corrupt pack
    /// contents, unsupported model family, decode failure, ...).
    TranscribeFailed = 3,
    /// Writing the temporary PCM staging file failed (disk full, sandbox
    /// path unwritable, ...).
    IoError = 4,
    /// A panic was caught at the FFI boundary and converted to an error
    /// instead of unwinding into the caller. This should not happen in
    /// normal operation; it indicates an engine bug.
    InternalPanic = 5,
    /// The signed model catalog could not be fetched, verified, parsed, or
    /// (for a pull) the requested model/quant could not be resolved from it.
    /// The catalog is loaded through the same fail-closed signature pipeline
    /// the CLI uses; a bad signature, epoch rollback, or unknown reference all
    /// land here rather than producing an unverified result.
    CatalogFailed = 6,
    /// A model-pack pull or local-pack install failed: a network/transport
    /// error, a size/sha256 mismatch against the signed catalog, a failed
    /// GGUF/runtime preflight, a gated license not accepted, or an install I/O
    /// error. Never a partially-installed or unverified pack.
    PullFailed = 7,
    /// A pull was stopped because the caller's cancel callback returned true.
    /// Any partial download is cleaned up; nothing is installed.
    PullCanceled = 8,
}

/// PCM sample encoding accepted by [`openasr_transcribe_pcm`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAsrPcmFormat {
    /// 32-bit float samples in `[-1.0, 1.0]`.
    F32 = 0,
    /// 16-bit signed integer samples (standard PCM WAV encoding).
    S16 = 1,
}

/// Opaque handle to a validated local `.oasr` model pack path. Obtained from
/// [`openasr_model_open`], released with [`openasr_model_close`].
///
/// This does not keep decoded weights resident: the underlying engine loads
/// the pack fresh for each [`openasr_transcribe_pcm`] call (matching the CLI's
/// own per-request load path), so the handle's job is to validate the pack
/// once up front and fail closed before any transcription is attempted with a
/// bad path.
pub struct OpenAsrModel {
    pack_path: PathBuf,
}

struct OwnedSegment {
    start_seconds: f32,
    end_seconds: f32,
    text: CString,
}

/// Opaque transcription result. Free with [`openasr_result_free`]; read it
/// through `openasr_result_*` accessor functions.
pub struct OpenAsrResult {
    text: CString,
    language: Option<CString>,
    segments: Vec<OwnedSegment>,
}

/// Hardware target for a streaming session's decode, mirroring
/// [`openasr_core::NativeAsrHardwareTarget`]. `Auto` lets the runtime choose;
/// on iOS/macOS only `Auto`, `Cpu`, `Accelerated`, and `AppleSilicon` are
/// meaningful (the SDK is CPU-only today -- see `docs/SDK_IOS_MACOS.md`), but
/// the full set is mapped so the contract does not silently drop a value.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAsrHardwareTarget {
    Auto = 0,
    Cpu = 1,
    Accelerated = 2,
    AppleSilicon = 3,
    NvidiaCuda = 4,
    AmdGpu = 5,
    IntelCpu = 6,
    IntelGpu = 7,
}

impl From<OpenAsrHardwareTarget> for NativeAsrHardwareTarget {
    fn from(target: OpenAsrHardwareTarget) -> Self {
        match target {
            OpenAsrHardwareTarget::Auto => NativeAsrHardwareTarget::Auto,
            OpenAsrHardwareTarget::Cpu => NativeAsrHardwareTarget::Cpu,
            OpenAsrHardwareTarget::Accelerated => NativeAsrHardwareTarget::Accelerated,
            OpenAsrHardwareTarget::AppleSilicon => NativeAsrHardwareTarget::AppleSilicon,
            OpenAsrHardwareTarget::NvidiaCuda => NativeAsrHardwareTarget::NvidiaCuda,
            OpenAsrHardwareTarget::AmdGpu => NativeAsrHardwareTarget::AmdGpu,
            OpenAsrHardwareTarget::IntelCpu => NativeAsrHardwareTarget::IntelCpu,
            OpenAsrHardwareTarget::IntelGpu => NativeAsrHardwareTarget::IntelGpu,
        }
    }
}

/// The kind of an incremental streaming transcript event, mirroring
/// [`openasr_core::StreamingEventKind`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAsrStreamingEventKind {
    /// A mutable in-progress hypothesis for the active utterance, superseded by
    /// later events carrying the same segment id.
    Partial = 0,
    /// A settled segment, emitted at a VAD speech pause or at `finish`.
    Committed = 1,
    /// A post-final correction to an already-committed segment.
    Revision = 2,
}

/// Configuration for [`openasr_streaming_session_open`]. Pass a null pointer to
/// use the engine defaults (partial results on, word timestamps off, VAD on,
/// auto hardware, default poll cadence). All fields are plain C values so the
/// struct is a valid `#[repr(C)]` type; `language` is an optional borrowed C
/// string (null = auto-detect), read only during the open call.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OpenAsrStreamingConfig {
    /// Emit mutable `Partial` events as audio arrives.
    pub partial_results: bool,
    /// Attach per-word timings to events when the family supports them.
    pub word_timestamps: bool,
    /// Run an energy VAD that commits a segment at each speech pause. When
    /// false, the whole stream is one utterance finalized only at `finish`.
    pub enable_vad: bool,
    /// Optional decode language hint (e.g. "en"), or null to auto-detect. Only
    /// borrowed for the duration of the open call.
    pub language: *const c_char,
    /// Hardware target for the decode session.
    pub hardware_target: OpenAsrHardwareTarget,
    /// Inference thread cap; `0` uses the per-family default.
    pub inference_threads: u16,
    /// New audio (ms) to accumulate before polling the engine for a partial
    /// re-decode; `0` uses the engine default. Values below the 20 ms frame
    /// size are clamped up by the engine.
    pub partial_poll_interval_ms: u64,
}

/// Opaque handle to an in-process streaming transcription session over a local
/// `.oasr` pack. Created with [`openasr_streaming_session_open`], driven with
/// [`openasr_streaming_feed`], and consumed by [`openasr_streaming_finish`]
/// (which returns the final transcript and frees the session) or discarded with
/// [`openasr_streaming_free`].
pub struct OpenAsrStreamingSession {
    inner: StreamingSession,
}

/// One incremental transcript event, owning C-string copies of its text fields.
struct OwnedStreamingEvent {
    kind: OpenAsrStreamingEventKind,
    utterance_id: CString,
    segment_id: CString,
    revision: u64,
    text: CString,
    start_ms: u64,
    end_ms: u64,
    language: Option<CString>,
}

/// Opaque batch of streaming events produced by one [`openasr_streaming_feed`]
/// call. Read it through the `openasr_streaming_event_*` accessors, then free
/// it with [`openasr_streaming_events_free`]. Mirrors [`OpenAsrResult`]: the
/// events own Rust-side strings that are not a valid C value type, so they are
/// read by index rather than returned as a raw struct array.
pub struct OpenAsrStreamingEvents {
    events: Vec<OwnedStreamingEvent>,
}

/// Returns the engine version (the workspace product version), as a static,
/// NUL-terminated UTF-8 string. Never null, never freed by the caller.
#[unsafe(no_mangle)]
pub extern "C" fn openasr_version() -> *const c_char {
    // `CARGO_PKG_VERSION` has no embedded NUL, so this leaked, process-lifetime
    // allocation is safe and only happens once per process via the cache.
    static VERSION: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
    VERSION
        .get_or_init(|| CString::new(env!("CARGO_PKG_VERSION")).expect("version has no NUL"))
        .as_ptr()
}

/// Loads (validates) a local `.oasr` model pack and returns an opaque handle
/// through `out_model`. Fails closed -- with [`OpenAsrStatus::ModelLoadFailed`]
/// and no handle written -- if the path is missing, not UTF-8, a directory, or
/// not a recognized OpenASR runtime pack.
///
/// # Safety
/// `path` must be a valid, NUL-terminated UTF-8 C string. `out_model` must be
/// a valid, non-null pointer to a `*mut OpenAsrModel` the caller owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_model_open(
    path: *const c_char,
    out_model: *mut *mut OpenAsrModel,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::ModelLoadFailed, || {
        if out_model.is_null() {
            set_last_error("openasr_model_open: out_model must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        // SAFETY: caller contract requires `out_model` to be a valid pointer.
        unsafe {
            *out_model = ptr::null_mut();
        }
        let path = match unsafe { c_str_to_path(path) } {
            Ok(path) => path,
            Err(status) => return status,
        };
        match validate_local_native_model_pack_path(&path) {
            Ok(validated) => {
                let handle = Box::new(OpenAsrModel {
                    pack_path: validated,
                });
                // SAFETY: checked non-null above.
                unsafe {
                    *out_model = Box::into_raw(handle);
                }
                OpenAsrStatus::Ok
            }
            Err(error) => {
                set_last_error(format!("openasr_model_open: {error}"));
                OpenAsrStatus::ModelLoadFailed
            }
        }
    })
}

/// Frees a handle returned by [`openasr_model_open`]. Null is accepted and is
/// a no-op (matches `free`/whisper.cpp convention).
///
/// # Safety
/// `model`, if non-null, must be a handle previously returned by
/// [`openasr_model_open`] and not already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_model_close(model: *mut OpenAsrModel) {
    let _ = catch(OpenAsrStatus::Ok, || {
        if !model.is_null() {
            // SAFETY: caller contract requires this came from `Box::into_raw`
            // in `openasr_model_open` and is not double-freed.
            drop(unsafe { Box::from_raw(model) });
        }
        OpenAsrStatus::Ok
    });
}

/// Builds the [`TranscriptionRequest`] for one `openasr_transcribe_pcm` call.
/// Split out from that function so the `RequestSource` wiring is
/// unit-testable without a real model pack (this never touches the
/// filesystem or a backend).
fn ffi_transcription_request(staging_path: PathBuf, pack_path: PathBuf) -> TranscriptionRequest {
    TranscriptionRequest::new(staging_path, NATIVE_RUNTIME_MODEL_ID_AUTO)
        // The C ABI carries no field distinguishing which host feature called
        // in, so every embedder logs this one label -- see
        // `RequestSource::Ffi`'s doc comment.
        .with_source(openasr_core::RequestSource::Ffi)
        .with_model_pack_path(Some(pack_path))
        .with_word_timestamps(false)
}

/// Transcribes one whole in-memory 16 kHz mono PCM buffer and writes a result
/// through `out_result`. `pcm_len_samples` counts samples (not bytes/frames).
/// Only mono 16 kHz input is accepted in v1 -- resampling/downmixing is the
/// caller's responsibility; anything else fails closed with
/// [`OpenAsrStatus::InvalidArgument`] rather than silently reinterpreting the
/// buffer. Read the result with the `openasr_result_*` accessors, then free it
/// with [`openasr_result_free`].
///
/// # Safety
/// `model` must be a live handle from [`openasr_model_open`]. `pcm` must
/// point to at least `pcm_len_samples` samples of the given `format` (4 bytes
/// each for F32, 2 bytes each for S16). `out_result` must be a valid,
/// non-null pointer to a `*mut OpenAsrResult` the caller owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_transcribe_pcm(
    model: *mut OpenAsrModel,
    pcm: *const c_void,
    pcm_len_samples: usize,
    format: OpenAsrPcmFormat,
    sample_rate_hz: u32,
    with_segments: bool,
    out_result: *mut *mut OpenAsrResult,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::TranscribeFailed, || {
        if out_result.is_null() {
            set_last_error("openasr_transcribe_pcm: out_result must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        // SAFETY: caller contract requires `out_result` to be a valid pointer.
        unsafe {
            *out_result = ptr::null_mut();
        }
        if model.is_null() {
            set_last_error("openasr_transcribe_pcm: model handle must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        if pcm.is_null() && pcm_len_samples > 0 {
            set_last_error("openasr_transcribe_pcm: pcm must not be null when non-empty");
            return OpenAsrStatus::InvalidArgument;
        }
        if sample_rate_hz != 16_000 {
            set_last_error(format!(
                "openasr_transcribe_pcm: only 16000 Hz mono PCM is supported in v1, got {sample_rate_hz} Hz"
            ));
            return OpenAsrStatus::InvalidArgument;
        }
        if pcm_len_samples == 0 {
            set_last_error("openasr_transcribe_pcm: pcm_len_samples must be > 0");
            return OpenAsrStatus::InvalidArgument;
        }

        // SAFETY: caller contract guarantees `pcm` points at
        // `pcm_len_samples` samples of `format`; both arms read exactly that
        // many elements of the matching type and copy them out before this
        // function returns.
        let samples_f32: Vec<f32> = unsafe {
            match format {
                OpenAsrPcmFormat::F32 => {
                    std::slice::from_raw_parts(pcm as *const f32, pcm_len_samples).to_vec()
                }
                OpenAsrPcmFormat::S16 => {
                    std::slice::from_raw_parts(pcm as *const i16, pcm_len_samples)
                        .iter()
                        .map(|sample| f32::from(*sample) / f32::from(i16::MAX))
                        .collect()
                }
            }
        };

        // SAFETY: `model` was checked non-null above and, per the function's
        // safety contract, is a live handle from `openasr_model_open`.
        let model_ref = unsafe { &*model };

        let staging = match tempfile::Builder::new()
            .prefix("openasr-ffi-")
            .suffix(".wav")
            .tempfile()
        {
            Ok(file) => file,
            Err(error) => {
                set_last_error(format!(
                    "openasr_transcribe_pcm: could not create staging file: {error}"
                ));
                return OpenAsrStatus::IoError;
            }
        };
        let staging_path = staging.path().to_path_buf();
        if let Err(error) = write_wav_16khz_mono_f32(&staging_path, &samples_f32) {
            set_last_error(format!(
                "openasr_transcribe_pcm: could not stage PCM as WAV: {error}"
            ));
            return OpenAsrStatus::IoError;
        }

        let request = ffi_transcription_request(staging_path, model_ref.pack_path.clone());

        match NativeBackend.transcribe(request) {
            Ok(transcription) => {
                let result = Box::new(build_result(transcription, with_segments));
                // SAFETY: checked non-null above.
                unsafe {
                    *out_result = Box::into_raw(result);
                }
                OpenAsrStatus::Ok
            }
            Err(error) => {
                set_last_error(format!("openasr_transcribe_pcm: {error}"));
                OpenAsrStatus::TranscribeFailed
            }
        }
    })
}

/// Frees a result returned by [`openasr_transcribe_pcm`]. Null is accepted
/// and is a no-op.
///
/// # Safety
/// `result`, if non-null, must have been previously returned by
/// [`openasr_transcribe_pcm`] and not already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_result_free(result: *mut OpenAsrResult) {
    let _ = catch(OpenAsrStatus::Ok, || {
        if !result.is_null() {
            // SAFETY: caller contract requires this came from `Box::into_raw`
            // in `openasr_transcribe_pcm` and is not double-freed.
            drop(unsafe { Box::from_raw(result) });
        }
        OpenAsrStatus::Ok
    });
}

/// Returns the full transcript text, UTF-8 and NUL-terminated. Valid until
/// `result` is freed. Null only if `result` itself is null.
///
/// # Safety
/// `result`, if non-null, must be a live handle from [`openasr_transcribe_pcm`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_result_text(result: *const OpenAsrResult) -> *const c_char {
    if result.is_null() {
        return ptr::null();
    }
    // SAFETY: caller contract requires a live handle.
    unsafe { (*result).text.as_ptr() }
}

/// Returns the detected/requested language tag (e.g. "en"), UTF-8 and
/// NUL-terminated, or null if `result` is null or the model family does not
/// report a language. Valid until `result` is freed.
///
/// # Safety
/// `result`, if non-null, must be a live handle from [`openasr_transcribe_pcm`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_result_language(result: *const OpenAsrResult) -> *const c_char {
    if result.is_null() {
        return ptr::null();
    }
    // SAFETY: caller contract requires a live handle.
    unsafe { (*result).language.as_ref() }
        .map(|language| language.as_ptr())
        .unwrap_or(ptr::null())
}

/// Returns the number of segments in `result` (`0` if `result` is null or the
/// producing call passed `with_segments = false`).
///
/// # Safety
/// `result`, if non-null, must be a live handle from [`openasr_transcribe_pcm`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_result_segment_count(result: *const OpenAsrResult) -> usize {
    if result.is_null() {
        return 0;
    }
    // SAFETY: caller contract requires a live handle.
    unsafe { (*result).segments.len() }
}

/// Returns segment `index`'s start time in seconds, or `0.0` if `result` is
/// null or `index` is out of range.
///
/// # Safety
/// `result`, if non-null, must be a live handle from [`openasr_transcribe_pcm`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_result_segment_start(
    result: *const OpenAsrResult,
    index: usize,
) -> f32 {
    if result.is_null() {
        return 0.0;
    }
    // SAFETY: caller contract requires a live handle; `get` bounds-checks `index`.
    unsafe { (&*result).segments.get(index) }
        .map(|segment| segment.start_seconds)
        .unwrap_or(0.0)
}

/// Returns segment `index`'s end time in seconds, or `0.0` if `result` is
/// null or `index` is out of range.
///
/// # Safety
/// `result`, if non-null, must be a live handle from [`openasr_transcribe_pcm`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_result_segment_end(
    result: *const OpenAsrResult,
    index: usize,
) -> f32 {
    if result.is_null() {
        return 0.0;
    }
    // SAFETY: caller contract requires a live handle; `get` bounds-checks `index`.
    unsafe { (&*result).segments.get(index) }
        .map(|segment| segment.end_seconds)
        .unwrap_or(0.0)
}

/// Returns segment `index`'s UTF-8, NUL-terminated text, or null if `result`
/// is null or `index` is out of range. Valid until `result` is freed.
///
/// # Safety
/// `result`, if non-null, must be a live handle from [`openasr_transcribe_pcm`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_result_segment_text(
    result: *const OpenAsrResult,
    index: usize,
) -> *const c_char {
    if result.is_null() {
        return ptr::null();
    }
    // SAFETY: caller contract requires a live handle; `get` bounds-checks `index`.
    unsafe { (&*result).segments.get(index) }
        .map(|segment| segment.text.as_ptr())
        .unwrap_or(ptr::null())
}

/// Opens an in-process streaming transcription session over the local `.oasr`
/// pack at `path` and writes an opaque handle through `out_session`. Fails
/// closed -- with [`OpenAsrStatus::ModelLoadFailed`] and no handle written --
/// if the path is missing, not UTF-8, or not a pack a native model family
/// recognizes / can stream. Never touches the network.
///
/// Pass a null `config` to use the engine defaults; otherwise `config` is read
/// (and its `language`, if non-null, borrowed) only for the duration of this
/// call.
///
/// # Safety
/// `path` must be a valid, NUL-terminated UTF-8 C string. `config`, if
/// non-null, must point to a valid [`OpenAsrStreamingConfig`] whose `language`
/// is null or a valid NUL-terminated UTF-8 C string. `out_session` must be a
/// valid, non-null pointer to a `*mut OpenAsrStreamingSession` the caller owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_streaming_session_open(
    path: *const c_char,
    config: *const OpenAsrStreamingConfig,
    out_session: *mut *mut OpenAsrStreamingSession,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::ModelLoadFailed, || {
        if out_session.is_null() {
            set_last_error("openasr_streaming_session_open: out_session must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        // SAFETY: caller contract requires `out_session` to be a valid pointer.
        unsafe {
            *out_session = ptr::null_mut();
        }
        let path = match unsafe { c_str_to_path(path) } {
            Ok(path) => path,
            Err(status) => return status,
        };
        let cfg = match unsafe { streaming_config_from_c(config) } {
            Ok(cfg) => cfg,
            Err(status) => return status,
        };
        match StreamingSession::new(&path, cfg) {
            Ok(session) => {
                let handle = Box::new(OpenAsrStreamingSession { inner: session });
                // SAFETY: checked non-null above.
                unsafe {
                    *out_session = Box::into_raw(handle);
                }
                OpenAsrStatus::Ok
            }
            Err(error) => {
                set_last_error(format!("openasr_streaming_session_open: {error}"));
                OpenAsrStatus::ModelLoadFailed
            }
        }
    })
}

/// Feeds a chunk of 16 kHz mono `f32` PCM (any length, including zero) into an
/// open streaming session and writes the incremental events it produced through
/// `out_events`: `Partial`s for the active utterance and a `Committed` event
/// whenever a VAD speech pause closes one. An empty chunk is accepted and yields
/// an empty (non-null) event batch. Read the batch with the
/// `openasr_streaming_event_*` accessors, then free it with
/// [`openasr_streaming_events_free`].
///
/// # Safety
/// `session` must be a live handle from [`openasr_streaming_session_open`] (not
/// yet finished/freed). `pcm` must point to at least `pcm_len_samples` `f32`
/// samples (may be null only when `pcm_len_samples` is 0). `out_events` must be
/// a valid, non-null pointer to a `*mut OpenAsrStreamingEvents` the caller owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_streaming_feed(
    session: *mut OpenAsrStreamingSession,
    pcm: *const f32,
    pcm_len_samples: usize,
    out_events: *mut *mut OpenAsrStreamingEvents,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::TranscribeFailed, || {
        if out_events.is_null() {
            set_last_error("openasr_streaming_feed: out_events must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        // SAFETY: caller contract requires `out_events` to be a valid pointer.
        unsafe {
            *out_events = ptr::null_mut();
        }
        if session.is_null() {
            set_last_error("openasr_streaming_feed: session handle must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        if pcm.is_null() && pcm_len_samples > 0 {
            set_last_error("openasr_streaming_feed: pcm must not be null when non-empty");
            return OpenAsrStatus::InvalidArgument;
        }
        // SAFETY: `pcm` points at `pcm_len_samples` f32 samples per the safety
        // contract; the empty case uses a valid dangling-free slice so
        // `from_raw_parts` is never called with a null base pointer.
        let samples: &[f32] = if pcm_len_samples == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(pcm, pcm_len_samples) }
        };
        // SAFETY: `session` was checked non-null above and, per the safety
        // contract, is a live handle from `openasr_streaming_session_open`.
        let session_ref = unsafe { &mut *session };
        match session_ref.inner.feed(samples) {
            Ok(events) => {
                let batch = Box::new(OpenAsrStreamingEvents {
                    events: events.into_iter().map(owned_streaming_event).collect(),
                });
                // SAFETY: checked non-null above.
                unsafe {
                    *out_events = Box::into_raw(batch);
                }
                OpenAsrStatus::Ok
            }
            Err(error) => {
                set_last_error(format!("openasr_streaming_feed: {error}"));
                OpenAsrStatus::TranscribeFailed
            }
        }
    })
}

/// Finishes a streaming session: drains any buffered tail audio, finalizes the
/// active utterance, and writes the assembled final [`OpenAsrResult`] (with
/// per-segment timestamps) through `out_result`. This **consumes** `session`:
/// whether it succeeds or fails, the handle is freed and must not be reused or
/// passed to [`openasr_streaming_free`]. Read the result with the
/// `openasr_result_*` accessors and free it with [`openasr_result_free`].
///
/// # Safety
/// `session` must be a live handle from [`openasr_streaming_session_open`] that
/// has not already been finished or freed. `out_result` must be a valid,
/// non-null pointer to a `*mut OpenAsrResult` the caller owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_streaming_finish(
    session: *mut OpenAsrStreamingSession,
    out_result: *mut *mut OpenAsrResult,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::TranscribeFailed, || {
        if out_result.is_null() {
            set_last_error("openasr_streaming_finish: out_result must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        // SAFETY: caller contract requires `out_result` to be a valid pointer.
        unsafe {
            *out_result = ptr::null_mut();
        }
        if session.is_null() {
            set_last_error("openasr_streaming_finish: session handle must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        // SAFETY: caller contract requires this came from `Box::into_raw` in
        // `openasr_streaming_session_open` and has not been finished/freed.
        // `finish` consumes the session, so ownership is taken back here and the
        // box is dropped exactly once regardless of the decode outcome.
        let handle = unsafe { Box::from_raw(session) };
        match handle.inner.finish() {
            Ok(transcription) => {
                let result = Box::new(build_result(transcription, true));
                // SAFETY: checked non-null above.
                unsafe {
                    *out_result = Box::into_raw(result);
                }
                OpenAsrStatus::Ok
            }
            Err(error) => {
                set_last_error(format!("openasr_streaming_finish: {error}"));
                OpenAsrStatus::TranscribeFailed
            }
        }
    })
}

/// Frees a streaming session without finishing it (aborts the stream). Null is
/// accepted and is a no-op. Do not call this on a handle already consumed by
/// [`openasr_streaming_finish`].
///
/// # Safety
/// `session`, if non-null, must be a live handle from
/// [`openasr_streaming_session_open`] that has not been finished or freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_streaming_free(session: *mut OpenAsrStreamingSession) {
    let _ = catch(OpenAsrStatus::Ok, || {
        if !session.is_null() {
            // SAFETY: caller contract requires this came from `Box::into_raw`
            // in `openasr_streaming_session_open` and is not double-freed.
            drop(unsafe { Box::from_raw(session) });
        }
        OpenAsrStatus::Ok
    });
}

/// Returns the number of events in `events` (`0` if `events` is null).
///
/// # Safety
/// `events`, if non-null, must be a live handle from [`openasr_streaming_feed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_streaming_events_count(
    events: *const OpenAsrStreamingEvents,
) -> usize {
    if events.is_null() {
        return 0;
    }
    // SAFETY: caller contract requires a live handle.
    unsafe { (*events).events.len() }
}

/// Returns event `index`'s kind, or [`OpenAsrStreamingEventKind::Partial`] if
/// `events` is null or `index` is out of range (gate reads on
/// [`openasr_streaming_events_count`]).
///
/// # Safety
/// `events`, if non-null, must be a live handle from [`openasr_streaming_feed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_streaming_event_kind(
    events: *const OpenAsrStreamingEvents,
    index: usize,
) -> OpenAsrStreamingEventKind {
    if events.is_null() {
        return OpenAsrStreamingEventKind::Partial;
    }
    // SAFETY: caller contract requires a live handle; `get` bounds-checks `index`.
    unsafe { (&*events).events.get(index) }
        .map(|event| event.kind)
        .unwrap_or(OpenAsrStreamingEventKind::Partial)
}

/// Returns event `index`'s text (UTF-8, NUL-terminated), or null if `events` is
/// null or `index` is out of range. Valid until `events` is freed.
///
/// # Safety
/// `events`, if non-null, must be a live handle from [`openasr_streaming_feed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_streaming_event_text(
    events: *const OpenAsrStreamingEvents,
    index: usize,
) -> *const c_char {
    // SAFETY: caller contract requires a live handle from openasr_streaming_feed.
    unsafe { streaming_event_cstr(events, index, |event| Some(&event.text)) }
}

/// Returns event `index`'s utterance id (UTF-8, NUL-terminated), or null if
/// `events` is null or `index` is out of range. Valid until `events` is freed.
///
/// # Safety
/// `events`, if non-null, must be a live handle from [`openasr_streaming_feed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_streaming_event_utterance_id(
    events: *const OpenAsrStreamingEvents,
    index: usize,
) -> *const c_char {
    // SAFETY: caller contract requires a live handle from openasr_streaming_feed.
    unsafe { streaming_event_cstr(events, index, |event| Some(&event.utterance_id)) }
}

/// Returns event `index`'s segment id (UTF-8, NUL-terminated), or null if
/// `events` is null or `index` is out of range. Valid until `events` is freed.
///
/// # Safety
/// `events`, if non-null, must be a live handle from [`openasr_streaming_feed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_streaming_event_segment_id(
    events: *const OpenAsrStreamingEvents,
    index: usize,
) -> *const c_char {
    // SAFETY: caller contract requires a live handle from openasr_streaming_feed.
    unsafe { streaming_event_cstr(events, index, |event| Some(&event.segment_id)) }
}

/// Returns event `index`'s detected language tag (e.g. "en"), UTF-8 and
/// NUL-terminated, or null if `events` is null, `index` is out of range, or the
/// family reported no language. Valid until `events` is freed.
///
/// # Safety
/// `events`, if non-null, must be a live handle from [`openasr_streaming_feed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_streaming_event_language(
    events: *const OpenAsrStreamingEvents,
    index: usize,
) -> *const c_char {
    // SAFETY: caller contract requires a live handle from openasr_streaming_feed.
    unsafe { streaming_event_cstr(events, index, |event| event.language.as_ref()) }
}

/// Returns event `index`'s monotonic revision number (higher supersedes lower
/// for the same segment id), or `0` if `events` is null or `index` is out of
/// range.
///
/// # Safety
/// `events`, if non-null, must be a live handle from [`openasr_streaming_feed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_streaming_event_revision(
    events: *const OpenAsrStreamingEvents,
    index: usize,
) -> u64 {
    if events.is_null() {
        return 0;
    }
    // SAFETY: caller contract requires a live handle; `get` bounds-checks `index`.
    unsafe { (&*events).events.get(index) }
        .map(|event| event.revision)
        .unwrap_or(0)
}

/// Returns event `index`'s start time in milliseconds, or `0` if `events` is
/// null or `index` is out of range.
///
/// # Safety
/// `events`, if non-null, must be a live handle from [`openasr_streaming_feed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_streaming_event_start_ms(
    events: *const OpenAsrStreamingEvents,
    index: usize,
) -> u64 {
    if events.is_null() {
        return 0;
    }
    // SAFETY: caller contract requires a live handle; `get` bounds-checks `index`.
    unsafe { (&*events).events.get(index) }
        .map(|event| event.start_ms)
        .unwrap_or(0)
}

/// Returns event `index`'s end time in milliseconds, or `0` if `events` is null
/// or `index` is out of range.
///
/// # Safety
/// `events`, if non-null, must be a live handle from [`openasr_streaming_feed`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_streaming_event_end_ms(
    events: *const OpenAsrStreamingEvents,
    index: usize,
) -> u64 {
    if events.is_null() {
        return 0;
    }
    // SAFETY: caller contract requires a live handle; `get` bounds-checks `index`.
    unsafe { (&*events).events.get(index) }
        .map(|event| event.end_ms)
        .unwrap_or(0)
}

/// Frees an event batch returned by [`openasr_streaming_feed`]. Null is accepted
/// and is a no-op.
///
/// # Safety
/// `events`, if non-null, must have been previously returned by
/// [`openasr_streaming_feed`] and not already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_streaming_events_free(events: *mut OpenAsrStreamingEvents) {
    let _ = catch(OpenAsrStatus::Ok, || {
        if !events.is_null() {
            // SAFETY: caller contract requires this came from `Box::into_raw`
            // in `openasr_streaming_feed` and is not double-freed.
            drop(unsafe { Box::from_raw(events) });
        }
        OpenAsrStatus::Ok
    });
}

/// Returns the last error message set on the calling thread by a failing
/// `openasr_*` call, or null if none is set / the last call succeeded. Valid
/// until the next `openasr_*` call on the same thread; copy it out if you
/// need it longer. Never freed by the caller.
#[unsafe(no_mangle)]
pub extern "C" fn openasr_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|message| message.as_ptr())
            .unwrap_or(ptr::null())
    })
}

/// Runs `body`, catching any panic and converting it to `panic_status` plus a
/// last-error message instead of unwinding across the FFI boundary.
pub(crate) fn catch(
    panic_status: OpenAsrStatus,
    body: impl FnOnce() -> OpenAsrStatus,
) -> OpenAsrStatus {
    match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(status) => {
            if status == OpenAsrStatus::Ok {
                clear_last_error();
            }
            status
        }
        Err(payload) => {
            let message = panic_message(&payload);
            set_last_error(format!("openasr: internal panic: {message}"));
            panic_status
        }
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// # Safety
/// `path` must be a valid, NUL-terminated C string (or null).
pub(crate) unsafe fn c_str_to_path(path: *const c_char) -> Result<PathBuf, OpenAsrStatus> {
    if path.is_null() {
        set_last_error("path argument must not be null");
        return Err(OpenAsrStatus::InvalidArgument);
    }
    // SAFETY: caller contract requires a valid NUL-terminated C string.
    let c_str = unsafe { CStr::from_ptr(path) };
    match c_str.to_str() {
        Ok(text) if !text.is_empty() => Ok(PathBuf::from(text)),
        Ok(_) => {
            set_last_error("path argument must not be empty");
            Err(OpenAsrStatus::InvalidArgument)
        }
        Err(error) => {
            set_last_error(format!("path argument is not valid UTF-8: {error}"));
            Err(OpenAsrStatus::InvalidArgument)
        }
    }
}

fn write_wav_16khz_mono_f32(path: &std::path::Path, samples: &[f32]) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer =
        hound::WavWriter::create(path, spec).map_err(|error| format!("create wav: {error}"))?;
    for sample in samples {
        writer
            .write_sample(*sample)
            .map_err(|error| format!("write sample: {error}"))?;
    }
    writer
        .finalize()
        .map_err(|error| format!("finalize wav: {error}"))
}

fn build_result(transcription: Transcription, with_segments: bool) -> OpenAsrResult {
    let text = cstring_lossy(transcription.text);
    let language = transcription.language.map(cstring_lossy);
    let segments = if with_segments {
        transcription
            .segments
            .into_iter()
            .map(|segment| OwnedSegment {
                start_seconds: segment.start,
                end_seconds: segment.end,
                text: cstring_lossy(segment.text),
            })
            .collect()
    } else {
        Vec::new()
    };

    OpenAsrResult {
        text,
        language,
        segments,
    }
}

/// Builds a [`StreamingConfig`] from the optional C config. A null pointer
/// yields the engine defaults; a non-null pointer overrides only the fields it
/// carries (its `language`, if non-null, is borrowed for this call).
///
/// # Safety
/// `config`, if non-null, must point to a valid [`OpenAsrStreamingConfig`]
/// whose `language` is null or a valid NUL-terminated C string.
unsafe fn streaming_config_from_c(
    config: *const OpenAsrStreamingConfig,
) -> Result<StreamingConfig, OpenAsrStatus> {
    let mut cfg = StreamingConfig::default();
    if config.is_null() {
        return Ok(cfg);
    }
    // SAFETY: caller contract requires a valid pointer when non-null.
    let raw = unsafe { &*config };
    cfg.partial_results = raw.partial_results;
    cfg.word_timestamps = raw.word_timestamps;
    // `StreamingConfig::default()` already enables VAD; only clear it when the
    // caller opts out (whole stream treated as a single utterance).
    if !raw.enable_vad {
        cfg.vad = None;
    }
    cfg.language = if raw.language.is_null() {
        None
    } else {
        // SAFETY: caller contract requires a valid NUL-terminated C string when
        // `language` is non-null.
        let c_str = unsafe { CStr::from_ptr(raw.language) };
        match c_str.to_str() {
            Ok(text) if !text.is_empty() => Some(text.to_string()),
            Ok(_) => None,
            Err(error) => {
                set_last_error(format!(
                    "openasr_streaming_session_open: language is not valid UTF-8: {error}"
                ));
                return Err(OpenAsrStatus::InvalidArgument);
            }
        }
    };
    cfg.hardware_target = raw.hardware_target.into();
    cfg.inference_threads = if raw.inference_threads == 0 {
        None
    } else {
        Some(raw.inference_threads)
    };
    if raw.partial_poll_interval_ms != 0 {
        cfg.partial_poll_interval_ms = raw.partial_poll_interval_ms;
    }
    Ok(cfg)
}

fn owned_streaming_event(event: StreamingEvent) -> OwnedStreamingEvent {
    OwnedStreamingEvent {
        kind: match event.kind {
            StreamingEventKind::Partial => OpenAsrStreamingEventKind::Partial,
            StreamingEventKind::Committed => OpenAsrStreamingEventKind::Committed,
            StreamingEventKind::Revision => OpenAsrStreamingEventKind::Revision,
        },
        utterance_id: cstring_lossy(event.utterance_id),
        segment_id: cstring_lossy(event.segment_id),
        revision: event.revision,
        text: cstring_lossy(event.text),
        start_ms: event.start_ms,
        end_ms: event.end_ms,
        language: event.language.map(cstring_lossy),
    }
}

/// Shared body for the string-returning streaming-event accessors: fail closed
/// to null on a null handle or an out-of-range index, otherwise hand back a
/// borrowed pointer valid until the batch is freed.
///
/// # Safety
/// `events`, if non-null, must be a live handle from [`openasr_streaming_feed`].
unsafe fn streaming_event_cstr(
    events: *const OpenAsrStreamingEvents,
    index: usize,
    select: impl FnOnce(&OwnedStreamingEvent) -> Option<&CString>,
) -> *const c_char {
    if events.is_null() {
        return ptr::null();
    }
    // SAFETY: caller contract requires a live handle; `get` bounds-checks `index`.
    unsafe { (&*events).events.get(index) }
        .and_then(select)
        .map(|value| value.as_ptr())
        .unwrap_or(ptr::null())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString as StdCString;

    fn last_error() -> String {
        let ptr = openasr_last_error_message();
        if ptr.is_null() {
            return String::new();
        }
        // SAFETY: `openasr_last_error_message` returns either null or a valid
        // NUL-terminated string owned by this crate's thread-local state.
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    // Regression guard for the FFI entry point: the request built from a
    // `openasr_transcribe_pcm` call must log `RequestSource::Ffi`, not
    // `Unspecified` -- see that variant's doc comment for why the C ABI
    // still gets an intentional, non-default label despite carrying no
    // finer-grained caller context.
    #[test]
    fn ffi_transcription_request_labels_source_as_ffi() {
        let request = ffi_transcription_request(
            PathBuf::from("/tmp/openasr-ffi-staging.wav"),
            PathBuf::from("/nonexistent/model.oasr"),
        );
        assert_eq!(request.source, openasr_core::RequestSource::Ffi);
    }

    #[test]
    fn version_is_the_workspace_version_and_valid_utf8() {
        let version_ptr = openasr_version();
        assert!(!version_ptr.is_null());
        // SAFETY: `openasr_version` always returns a valid, static, NUL-terminated string.
        let version = unsafe { CStr::from_ptr(version_ptr) }.to_str().unwrap();
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn model_open_rejects_null_out_param() {
        let path = StdCString::new("/nonexistent/model.oasr").unwrap();
        // SAFETY: `path` is a valid C string; passing a null `out_model`
        // pointer is exactly the invalid-argument case under test.
        let status = unsafe { openasr_model_open(path.as_ptr(), ptr::null_mut()) };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
        assert!(last_error().contains("out_model"));
    }

    #[test]
    fn model_open_rejects_null_path() {
        let mut model: *mut OpenAsrModel = ptr::null_mut();
        // SAFETY: a null `path` with a valid `out_model` slot is the
        // invalid-argument case under test; `openasr_model_open` must not
        // dereference `path` before checking it.
        let status = unsafe { openasr_model_open(ptr::null(), &mut model) };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
        assert!(model.is_null());
    }

    #[test]
    fn model_open_fails_closed_on_missing_path() {
        let path = StdCString::new("/nonexistent/does-not-exist.oasr").unwrap();
        let mut model: *mut OpenAsrModel = ptr::null_mut();
        // SAFETY: valid C string, valid non-null out pointer.
        let status = unsafe { openasr_model_open(path.as_ptr(), &mut model) };
        assert_eq!(status, OpenAsrStatus::ModelLoadFailed);
        assert!(model.is_null());
        assert!(!last_error().is_empty());
    }

    #[test]
    fn model_close_accepts_null() {
        // SAFETY: null is an explicitly documented no-op.
        unsafe { openasr_model_close(ptr::null_mut()) };
    }

    #[test]
    fn result_free_accepts_null() {
        // SAFETY: null is an explicitly documented no-op.
        unsafe { openasr_result_free(ptr::null_mut()) };
    }

    #[test]
    fn transcribe_rejects_null_model() {
        let samples = [0i16; 16_000];
        let mut result: *mut OpenAsrResult = ptr::null_mut();
        // SAFETY: `samples` is a valid, fully-sized buffer for the sample
        // count passed; the null model handle is the case under test.
        let status = unsafe {
            openasr_transcribe_pcm(
                ptr::null_mut(),
                samples.as_ptr() as *const c_void,
                samples.len(),
                OpenAsrPcmFormat::S16,
                16_000,
                false,
                &mut result,
            )
        };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
        assert!(result.is_null());
    }

    #[test]
    fn transcribe_rejects_non_16khz_sample_rate() {
        // Needs a non-null (if never actually loaded) model handle: the
        // model-null check runs before the sample-rate check, and this test
        // wants to isolate the latter.
        let fake_model = Box::into_raw(Box::new(OpenAsrModel {
            pack_path: PathBuf::from("/nonexistent/model.oasr"),
        }));
        let samples = [0i16; 8_000];
        let mut result: *mut OpenAsrResult = ptr::null_mut();
        // SAFETY: `samples` is a valid, fully-sized buffer; `fake_model` is a
        // live handle (never dereferenced past the sample-rate check, which
        // fails first).
        let status = unsafe {
            openasr_transcribe_pcm(
                fake_model,
                samples.as_ptr() as *const c_void,
                samples.len(),
                OpenAsrPcmFormat::S16,
                8_000,
                false,
                &mut result,
            )
        };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
        assert!(last_error().contains("16000"));
        // SAFETY: `fake_model` came from `Box::into_raw` above and has not
        // been freed yet.
        unsafe { openasr_model_close(fake_model) };
    }

    #[test]
    fn transcribe_rejects_empty_buffer() {
        let mut result: *mut OpenAsrResult = ptr::null_mut();
        // SAFETY: null model + zero-length buffer is an invalid-argument
        // case caught before any dereference of `model` or `pcm`.
        let status = unsafe {
            openasr_transcribe_pcm(
                ptr::null_mut(),
                ptr::null(),
                0,
                OpenAsrPcmFormat::S16,
                16_000,
                false,
                &mut result,
            )
        };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
    }

    #[test]
    fn result_accessors_fail_closed_on_null_result() {
        // SAFETY: null `result` is the documented no-op / fail-closed case
        // for every accessor below.
        unsafe {
            assert!(openasr_result_text(ptr::null()).is_null());
            assert!(openasr_result_language(ptr::null()).is_null());
            assert_eq!(openasr_result_segment_count(ptr::null()), 0);
            assert_eq!(openasr_result_segment_start(ptr::null(), 0), 0.0);
            assert_eq!(openasr_result_segment_end(ptr::null(), 0), 0.0);
            assert!(openasr_result_segment_text(ptr::null(), 0).is_null());
        }
    }

    #[test]
    fn result_accessors_fail_closed_on_out_of_range_index() {
        let result = Box::new(build_result(
            Transcription {
                text: "hello".to_string(),
                segments: Vec::new(),
                longform: None,
                language: None,
            },
            true,
        ));
        let raw = Box::into_raw(result);
        // SAFETY: `raw` came from `Box::into_raw` above and is not freed
        // until `openasr_result_free` at the end of this test.
        unsafe {
            assert_eq!(openasr_result_segment_count(raw), 0);
            assert_eq!(openasr_result_segment_start(raw, 5), 0.0);
            assert_eq!(openasr_result_segment_end(raw, 5), 0.0);
            assert!(openasr_result_segment_text(raw, 5).is_null());
            openasr_result_free(raw);
        }
    }

    #[test]
    fn build_result_exposes_text_language_and_segments() {
        let transcription = Transcription {
            text: "hello world".to_string(),
            segments: vec![openasr_core::Segment {
                start: 0.0,
                end: 1.0,
                text: "hello world".to_string(),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            }],
            longform: None,
            language: Some("en".to_string()),
        };
        let result = Box::into_raw(Box::new(build_result(transcription, true)));

        // SAFETY: `result` is a live handle from the `Box::into_raw` above,
        // and is not freed until `openasr_result_free` at the end of the test.
        unsafe {
            let text = CStr::from_ptr(openasr_result_text(result));
            assert_eq!(text.to_str().unwrap(), "hello world");
            let language = CStr::from_ptr(openasr_result_language(result));
            assert_eq!(language.to_str().unwrap(), "en");
            assert_eq!(openasr_result_segment_count(result), 1);
            assert_eq!(openasr_result_segment_start(result, 0), 0.0);
            assert_eq!(openasr_result_segment_end(result, 0), 1.0);
            let segment_text = CStr::from_ptr(openasr_result_segment_text(result, 0));
            assert_eq!(segment_text.to_str().unwrap(), "hello world");
            openasr_result_free(result);
        }
    }

    fn sample_streaming_event(kind: StreamingEventKind, text: &str) -> StreamingEvent {
        StreamingEvent {
            kind,
            utterance_id: "utt_000001".to_string(),
            segment_id: "seg_000001".to_string(),
            revision: 3,
            text: text.to_string(),
            start_ms: 100,
            end_ms: 900,
            words: Vec::new(),
            language: Some("en".to_string()),
        }
    }

    #[test]
    fn streaming_session_open_rejects_null_out_param() {
        let path = StdCString::new("/nonexistent/model.oasr").unwrap();
        // SAFETY: valid C string; the null `out_session` is the case under test.
        let status =
            unsafe { openasr_streaming_session_open(path.as_ptr(), ptr::null(), ptr::null_mut()) };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
        assert!(last_error().contains("out_session"));
    }

    #[test]
    fn streaming_session_open_fails_closed_on_missing_pack() {
        let path = StdCString::new("/nonexistent/does-not-exist.oasr").unwrap();
        let mut session: *mut OpenAsrStreamingSession = ptr::null_mut();
        // SAFETY: valid C string, null config (defaults), valid out pointer.
        let status =
            unsafe { openasr_streaming_session_open(path.as_ptr(), ptr::null(), &mut session) };
        assert_eq!(status, OpenAsrStatus::ModelLoadFailed);
        assert!(session.is_null());
        assert!(!last_error().is_empty());
    }

    #[test]
    fn streaming_feed_rejects_null_session_and_out() {
        let samples = [0.0_f32; 320];
        let mut events: *mut OpenAsrStreamingEvents = ptr::null_mut();
        // SAFETY: null session with a valid out pointer is the case under test.
        let status = unsafe {
            openasr_streaming_feed(
                ptr::null_mut(),
                samples.as_ptr(),
                samples.len(),
                &mut events,
            )
        };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
        assert!(events.is_null());

        // SAFETY: null out pointer is separately rejected before any deref.
        let status = unsafe {
            openasr_streaming_feed(
                ptr::null_mut(),
                samples.as_ptr(),
                samples.len(),
                ptr::null_mut(),
            )
        };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
    }

    #[test]
    fn streaming_finish_rejects_null_session_and_out() {
        let mut result: *mut OpenAsrResult = ptr::null_mut();
        // SAFETY: null session with a valid out pointer is the case under test.
        let status = unsafe { openasr_streaming_finish(ptr::null_mut(), &mut result) };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
        assert!(result.is_null());

        // SAFETY: null out pointer is separately rejected before any deref.
        let status = unsafe { openasr_streaming_finish(ptr::null_mut(), ptr::null_mut()) };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
    }

    #[test]
    fn streaming_free_and_events_free_accept_null() {
        // SAFETY: null is an explicitly documented no-op for both.
        unsafe {
            openasr_streaming_free(ptr::null_mut());
            openasr_streaming_events_free(ptr::null_mut());
        }
    }

    #[test]
    fn streaming_event_accessors_fail_closed_on_null_batch() {
        // SAFETY: null `events` is the documented fail-closed case for every
        // accessor below.
        unsafe {
            assert_eq!(openasr_streaming_events_count(ptr::null()), 0);
            assert_eq!(
                openasr_streaming_event_kind(ptr::null(), 0),
                OpenAsrStreamingEventKind::Partial
            );
            assert!(openasr_streaming_event_text(ptr::null(), 0).is_null());
            assert!(openasr_streaming_event_utterance_id(ptr::null(), 0).is_null());
            assert!(openasr_streaming_event_segment_id(ptr::null(), 0).is_null());
            assert!(openasr_streaming_event_language(ptr::null(), 0).is_null());
            assert_eq!(openasr_streaming_event_revision(ptr::null(), 0), 0);
            assert_eq!(openasr_streaming_event_start_ms(ptr::null(), 0), 0);
            assert_eq!(openasr_streaming_event_end_ms(ptr::null(), 0), 0);
        }
    }

    #[test]
    fn streaming_events_roundtrip_through_accessors() {
        let batch = Box::into_raw(Box::new(OpenAsrStreamingEvents {
            events: vec![
                owned_streaming_event(sample_streaming_event(StreamingEventKind::Partial, "hello")),
                owned_streaming_event(sample_streaming_event(
                    StreamingEventKind::Committed,
                    "hello world",
                )),
            ],
        }));
        // SAFETY: `batch` came from `Box::into_raw` above and is freed at the
        // end of this test.
        unsafe {
            assert_eq!(openasr_streaming_events_count(batch), 2);

            assert_eq!(
                openasr_streaming_event_kind(batch, 0),
                OpenAsrStreamingEventKind::Partial
            );
            assert_eq!(
                CStr::from_ptr(openasr_streaming_event_text(batch, 0))
                    .to_str()
                    .unwrap(),
                "hello"
            );

            assert_eq!(
                openasr_streaming_event_kind(batch, 1),
                OpenAsrStreamingEventKind::Committed
            );
            assert_eq!(
                CStr::from_ptr(openasr_streaming_event_text(batch, 1))
                    .to_str()
                    .unwrap(),
                "hello world"
            );
            assert_eq!(
                CStr::from_ptr(openasr_streaming_event_utterance_id(batch, 1))
                    .to_str()
                    .unwrap(),
                "utt_000001"
            );
            assert_eq!(
                CStr::from_ptr(openasr_streaming_event_segment_id(batch, 1))
                    .to_str()
                    .unwrap(),
                "seg_000001"
            );
            assert_eq!(
                CStr::from_ptr(openasr_streaming_event_language(batch, 1))
                    .to_str()
                    .unwrap(),
                "en"
            );
            assert_eq!(openasr_streaming_event_revision(batch, 1), 3);
            assert_eq!(openasr_streaming_event_start_ms(batch, 1), 100);
            assert_eq!(openasr_streaming_event_end_ms(batch, 1), 900);

            // Out-of-range index fails closed rather than reading OOB.
            assert!(openasr_streaming_event_text(batch, 5).is_null());
            assert_eq!(openasr_streaming_event_revision(batch, 5), 0);

            openasr_streaming_events_free(batch);
        }
    }

    #[test]
    fn streaming_config_from_c_null_is_defaults() {
        // SAFETY: a null config pointer is the documented "use defaults" case.
        let cfg = unsafe { streaming_config_from_c(ptr::null()) }.unwrap();
        let default = StreamingConfig::default();
        assert_eq!(cfg.partial_results, default.partial_results);
        assert_eq!(cfg.word_timestamps, default.word_timestamps);
        assert!(cfg.vad.is_some());
        assert!(cfg.language.is_none());
    }

    #[test]
    fn streaming_config_from_c_applies_overrides() {
        let language = StdCString::new("es").unwrap();
        let raw = OpenAsrStreamingConfig {
            partial_results: false,
            word_timestamps: true,
            enable_vad: false,
            language: language.as_ptr(),
            hardware_target: OpenAsrHardwareTarget::Cpu,
            inference_threads: 4,
            partial_poll_interval_ms: 250,
        };
        // SAFETY: `raw` is a valid config with a valid, live `language` C string.
        let cfg = unsafe { streaming_config_from_c(&raw) }.unwrap();
        assert!(!cfg.partial_results);
        assert!(cfg.word_timestamps);
        assert!(cfg.vad.is_none());
        assert_eq!(cfg.language.as_deref(), Some("es"));
        assert_eq!(cfg.hardware_target, NativeAsrHardwareTarget::Cpu);
        assert_eq!(cfg.inference_threads, Some(4));
        assert_eq!(cfg.partial_poll_interval_ms, 250);
    }

    /// End-to-end C-ABI roundtrip against a real `.oasr` pack: open a session,
    /// feed synthetic PCM in chunks, drain events through the accessors, finish,
    /// read the transcript, and free everything -- asserting only that the C
    /// plumbing does not crash, leak, or fail (transcript quality is covered by
    /// `openasr-core`'s `streaming_matches_batch_transcribe`). Ignored by
    /// default because it needs model weights + a compiled ggml backend, which
    /// the weight-free default suite must not require. Run manually:
    ///
    /// ```text
    /// OPENASR_TEST_STREAMING_PACK=/path/to/moonshine-tiny-q8_0.oasr \
    ///   cargo test -p openasr-ffi streaming_c_abi_roundtrip -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires a real .oasr pack via OPENASR_TEST_STREAMING_PACK"]
    fn streaming_c_abi_roundtrip() {
        let pack = std::env::var("OPENASR_TEST_STREAMING_PACK")
            .expect("set OPENASR_TEST_STREAMING_PACK to a local .oasr pack");
        let pack_c = StdCString::new(pack).unwrap();

        let cfg = OpenAsrStreamingConfig {
            partial_results: true,
            word_timestamps: false,
            enable_vad: false,
            language: ptr::null(),
            hardware_target: OpenAsrHardwareTarget::Cpu,
            inference_threads: 0,
            partial_poll_interval_ms: 0,
        };

        let mut session: *mut OpenAsrStreamingSession = ptr::null_mut();
        // SAFETY: valid pack C string, valid config, valid out pointer.
        let status = unsafe { openasr_streaming_session_open(pack_c.as_ptr(), &cfg, &mut session) };
        assert_eq!(status, OpenAsrStatus::Ok, "open: {}", last_error());
        assert!(!session.is_null());

        // ~2s of a low-amplitude 440 Hz tone at 16 kHz, fed in 100 ms chunks.
        let total: Vec<f32> = (0..32_000)
            .map(|n| 0.1 * (2.0 * std::f32::consts::PI * 440.0 * n as f32 / 16_000.0).sin())
            .collect();
        for chunk in total.chunks(1_600) {
            let mut events: *mut OpenAsrStreamingEvents = ptr::null_mut();
            // SAFETY: live session, valid chunk pointer/len, valid out pointer.
            let status = unsafe {
                openasr_streaming_feed(session, chunk.as_ptr(), chunk.len(), &mut events)
            };
            assert_eq!(status, OpenAsrStatus::Ok, "feed: {}", last_error());
            assert!(!events.is_null());
            // SAFETY: `events` is a live batch; drain then free it.
            unsafe {
                for i in 0..openasr_streaming_events_count(events) {
                    let _ = openasr_streaming_event_kind(events, i);
                    let _ = openasr_streaming_event_text(events, i);
                }
                openasr_streaming_events_free(events);
            }
        }

        let mut result: *mut OpenAsrResult = ptr::null_mut();
        // SAFETY: live session (consumed here), valid out pointer.
        let status = unsafe { openasr_streaming_finish(session, &mut result) };
        assert_eq!(status, OpenAsrStatus::Ok, "finish: {}", last_error());
        assert!(!result.is_null());
        // SAFETY: `result` is a live handle from finish; read then free it.
        unsafe {
            let text_ptr = openasr_result_text(result);
            assert!(!text_ptr.is_null());
            let _ = CStr::from_ptr(text_ptr).to_str().unwrap();
            openasr_result_free(result);
        }
    }
}
