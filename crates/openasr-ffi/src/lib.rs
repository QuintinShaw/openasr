//! C ABI for embedding the OpenASR engine in iOS/macOS apps (see
//! `scripts/build-xcframework.sh` and `docs/SDK_IOS_MACOS.md`).
//!
//! Deliberately minimal for v1: load a local `.oasr` pack, transcribe one
//! whole in-memory 16 kHz mono PCM buffer (f32 or i16), get back text plus
//! optional segment timestamps. No streaming, no dictation, no server, no
//! model download -- SDK consumers manage their own model distribution (the
//! product's "no silent download" boundary is a CLI/server concern, not an
//! SDK one; see `AGENTS.md`).
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
    NATIVE_RUNTIME_MODEL_ID_AUTO, NativeBackend, Transcription, TranscriptionBackend,
    TranscriptionRequest, validate_local_native_model_pack_path,
};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(message: impl Into<String>) {
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
fn cstring_lossy(text: String) -> CString {
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

        let request = TranscriptionRequest::new(staging_path, NATIVE_RUNTIME_MODEL_ID_AUTO)
            .with_model_pack_path(Some(model_ref.pack_path.clone()))
            .with_word_timestamps(false);

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
fn catch(panic_status: OpenAsrStatus, body: impl FnOnce() -> OpenAsrStatus) -> OpenAsrStatus {
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
unsafe fn c_str_to_path(path: *const c_char) -> Result<PathBuf, OpenAsrStatus> {
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
}
