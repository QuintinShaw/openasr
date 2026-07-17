//! Model-market C ABI: the on-device model catalog + pull/install/remove path
//! for a native (SwiftUI) app that has no CLI or local server to lean on.
//!
//! # Where the trust boundary lives
//!
//! Every security-relevant decision stays in `openasr_core`, exactly as the
//! CLI does it -- this module only marshals C arguments in and typed
//! status/JSON out:
//!
//! - **Catalog fetch is fail-closed and signature-verified.**
//!   [`openasr_catalog_fetch`] calls [`openasr_core::load_model_catalog`], which
//!   requires a valid `catalog.signature.json` (ed25519, production trust root)
//!   and enforces the anti-rollback epoch floor before the catalog is ever
//!   parsed. A tampered or unsigned catalog yields [`OpenAsrStatus::CatalogFailed`]
//!   and no handle -- never an unverified model list.
//! - **No silent download.** This module never touches the network on its own.
//!   Fetching the catalog is an explicit call the app makes to render its market
//!   UI; pulling a model is a *separate* explicit call ([`openasr_pull_model`])
//!   the app makes only after showing the user the model, quant, size, host, and
//!   license (all present in the verified catalog JSON) and getting consent.
//!   There is no auto-install path and no transcription path that can trigger a
//!   download -- mirroring the CLI's "consent-pull is a command handler only"
//!   rule (see `AGENTS.md`).
//! - **Downloaded packs are verified in the core.** [`openasr_pull_model`] runs
//!   [`openasr_core::PullModelPackRequest`], which enforces https-only URLs,
//!   streams a sha256 checked against the catalog-pinned digest, runs the GGUF /
//!   runtime preflight, and installs atomically. [`openasr_install_local_pack`]
//!   verifies a user-provided `.oasr`'s sha256/size against the signed catalog
//!   before installing. The app never gets to hand over a URL or a hash -- only a
//!   catalog reference (`model:quant`), so it cannot redirect the download or
//!   bypass the digest check.
//!
//! The app supplies its own sandbox directory as the OpenASR home (the iOS
//! equivalent of `OPENASR_HOME`); packs install under `<home>/models/...`, and
//! the verified catalog / signature / epoch caches live directly under `<home>`.

use std::ffi::{CString, c_char, c_void};
use std::ptr;

use openasr_core::{
    CatalogPullRequest, DownloadSourcePref, InstalledPack, LicenseClass, ModelCatalog, PullError,
    PullModelPackRequest, PullProgress, install_catalog_model_pack_from_path, list_installed_packs,
    load_model_catalog, remove_model_pack, resolve_catalog_pull_with_profile, resolve_chain,
};

use crate::{OpenAsrStatus, c_str_to_path, catch, set_last_error};

/// Opaque handle to a fetched-and-verified model catalog. Obtained from
/// [`openasr_catalog_fetch`], read as JSON with [`openasr_catalog_json`], passed
/// to [`openasr_pull_model`] / [`openasr_install_local_pack`] as the verified
/// source of every model/quant/url/sha the pull may resolve, and released with
/// [`openasr_catalog_free`].
///
/// The catalog JSON is serialized once, at fetch time, from the verified +
/// forward-compatibility-filtered [`ModelCatalog`], so [`openasr_catalog_json`]
/// can hand back a borrowed pointer with no per-call allocation.
pub struct OpenAsrCatalog {
    inner: ModelCatalog,
    json: CString,
}

/// The stage a [`openasr_pull_model`] / [`openasr_install_local_pack`] progress
/// callback is reporting, mirroring [`openasr_core::PullProgress`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAsrPullPhase {
    /// The requested pack was already installed and verified on disk; no
    /// download happened. `bytes_done` / `bytes_total` are 0.
    UsingInstalled = 0,
    /// The download is starting (or resuming). `bytes_done` is the resume
    /// offset (0 for a fresh download); `bytes_total` is the full pack size.
    Started = 1,
    /// Download progress. `bytes_done` of `bytes_total` bytes fetched.
    Downloading = 2,
    /// The full pack is downloaded and its sha256 is being verified against the
    /// catalog-pinned digest. `bytes_done` is bytes hashed; `bytes_total` is 0.
    Verifying = 3,
    /// The verified pack has been installed. `bytes_done` / `bytes_total` are 0.
    Installed = 4,
}

/// Progress callback for a pull/install. Invoked synchronously on the calling
/// thread (the thread that called [`openasr_pull_model`] /
/// [`openasr_install_local_pack`]), so `user_data` need not be thread-safe. Pass
/// a null function pointer to receive no progress. The callback must not unwind
/// (a panic across it is undefined behavior); do the app-side marshalling to
/// another thread inside it, not around it.
pub type OpenAsrPullProgressCallback = Option<
    unsafe extern "C" fn(
        user_data: *mut c_void,
        phase: OpenAsrPullPhase,
        bytes_done: u64,
        bytes_total: u64,
    ),
>;

/// Cancellation callback for a pull. Polled synchronously on the calling thread
/// while the download runs; return `true` to cancel (the partial download is
/// cleaned up and [`OpenAsrStatus::PullCanceled`] is returned). Pass a null
/// function pointer to never cancel. Must not unwind.
pub type OpenAsrPullCancelCallback = Option<unsafe extern "C" fn(user_data: *mut c_void) -> bool>;

/// Fetches and verifies the signed model catalog, writing an opaque handle
/// through `out_catalog`. `catalog_url` is optional: null uses the built-in
/// production endpoint (the normal case); a non-null override is still held to
/// the same fail-closed signature/epoch pipeline. `home_dir` is the app's
/// OpenASR home (its sandbox directory), used for the verified catalog /
/// signature / epoch cache and as the offline fallback source.
///
/// Fails closed with [`OpenAsrStatus::CatalogFailed`] and no handle if the
/// catalog cannot be fetched, its signature does not verify, the epoch rolled
/// back, or it does not parse. Network access happens only inside this explicit
/// call.
///
/// # Safety
/// `catalog_url`, if non-null, must be a valid NUL-terminated UTF-8 C string.
/// `home_dir` must be a valid, non-empty, NUL-terminated UTF-8 C string.
/// `out_catalog` must be a valid, non-null pointer to a `*mut OpenAsrCatalog`
/// the caller owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_catalog_fetch(
    catalog_url: *const c_char,
    home_dir: *const c_char,
    out_catalog: *mut *mut OpenAsrCatalog,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::CatalogFailed, || {
        if out_catalog.is_null() {
            set_last_error("openasr_catalog_fetch: out_catalog must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        // SAFETY: caller contract requires `out_catalog` to be a valid pointer.
        unsafe {
            *out_catalog = ptr::null_mut();
        }
        let home = match unsafe { c_str_to_path(home_dir) } {
            Ok(home) => home,
            Err(status) => return status,
        };
        let url = match unsafe { opt_c_str(catalog_url, "catalog_url") } {
            Ok(url) => url,
            Err(status) => return status,
        };
        match load_model_catalog(url.as_deref(), &home) {
            Ok(catalog) => {
                let json = match serde_json::to_string(&catalog) {
                    Ok(json) => crate::cstring_lossy(json),
                    Err(error) => {
                        set_last_error(format!(
                            "openasr_catalog_fetch: could not serialize catalog: {error}"
                        ));
                        return OpenAsrStatus::CatalogFailed;
                    }
                };
                let handle = Box::new(OpenAsrCatalog {
                    inner: catalog,
                    json,
                });
                // SAFETY: checked non-null above.
                unsafe {
                    *out_catalog = Box::into_raw(handle);
                }
                OpenAsrStatus::Ok
            }
            Err(error) => {
                set_last_error(format!("openasr_catalog_fetch: {error}"));
                OpenAsrStatus::CatalogFailed
            }
        }
    })
}

/// Returns the verified catalog as a UTF-8, NUL-terminated JSON string (the
/// serialized [`ModelCatalog`]: models with their quants, sizes, sha256s,
/// languages, licenses, perf, etc.). Valid until `catalog` is freed; null only
/// if `catalog` is null. The app renders its market UI and its pull-consent
/// disclosure (model, quant, size, host, license) from this data.
///
/// # Safety
/// `catalog`, if non-null, must be a live handle from [`openasr_catalog_fetch`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_catalog_json(catalog: *const OpenAsrCatalog) -> *const c_char {
    if catalog.is_null() {
        return ptr::null();
    }
    // SAFETY: caller contract requires a live handle.
    unsafe { (*catalog).json.as_ptr() }
}

/// Frees a catalog handle returned by [`openasr_catalog_fetch`]. Null is
/// accepted and is a no-op.
///
/// # Safety
/// `catalog`, if non-null, must be a handle previously returned by
/// [`openasr_catalog_fetch`] and not already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_catalog_free(catalog: *mut OpenAsrCatalog) {
    let _ = catch(OpenAsrStatus::Ok, || {
        if !catalog.is_null() {
            // SAFETY: caller contract requires this came from `Box::into_raw`
            // in `openasr_catalog_fetch` and is not double-freed.
            drop(unsafe { Box::from_raw(catalog) });
        }
        OpenAsrStatus::Ok
    });
}

/// Downloads, verifies, and installs the model pack identified by `reference`
/// (a catalog id or `id:quant`) after the app has obtained the user's consent.
/// This is the app-side equivalent of `openasr pull`: it resolves `reference`
/// against the verified `catalog`, streams the download from the catalog-pinned
/// url over https, checks the sha256 against the catalog digest, runs the
/// GGUF/runtime preflight, and installs atomically under `<home_dir>/models`.
///
/// - `quant` is optional (null): with no quant pinned in `reference` or `quant`,
///   the largest quant that fits this device's memory budget is chosen (an
///   explicit `:quant` / `quant` always wins).
/// - `source` is optional (null): null uses the automatic download-source chain;
///   otherwise one of `"hf"`, `"hf-mirror"`, `"weights"`, or `"auto"`.
/// - `accept_license` must be true to pull a gated-license model; a gated model
///   pulled without it fails closed with [`OpenAsrStatus::PullFailed`], so
///   consent cannot silently become a license bypass (mirrors the CLI's
///   `--accept-license`).
/// - `progress_cb` / `cancel_cb` (both optional) are invoked synchronously on
///   this thread; return `true` from `cancel_cb` to abort.
/// - `out_installed_json`, if non-null, receives a freshly-allocated
///   UTF-8/NUL-terminated JSON object describing the installed pack (pull id,
///   path, quant, size, sha256, ...). Free it with [`openasr_string_free`].
///
/// Returns [`OpenAsrStatus::Ok`] on a verified install, [`OpenAsrStatus::PullCanceled`]
/// if `cancel_cb` aborted it, [`OpenAsrStatus::CatalogFailed`] if `reference`
/// could not be resolved, or [`OpenAsrStatus::PullFailed`] for a download /
/// verification / license / install failure. Never installs an unverified pack.
///
/// # Safety
/// `catalog` must be a live handle from [`openasr_catalog_fetch`]. `reference`
/// and `home_dir` must be valid, non-empty, NUL-terminated UTF-8 C strings.
/// `quant` / `source`, if non-null, must be valid NUL-terminated UTF-8 C
/// strings. The callbacks, if non-null, must be valid function pointers that do
/// not unwind. `out_installed_json`, if non-null, must point to a writable
/// `*mut c_char` the caller owns.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn openasr_pull_model(
    catalog: *const OpenAsrCatalog,
    reference: *const c_char,
    quant: *const c_char,
    source: *const c_char,
    accept_license: bool,
    home_dir: *const c_char,
    progress_cb: OpenAsrPullProgressCallback,
    progress_user_data: *mut c_void,
    cancel_cb: OpenAsrPullCancelCallback,
    cancel_user_data: *mut c_void,
    out_installed_json: *mut *mut c_char,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::PullFailed, || {
        clear_out_string(out_installed_json);
        if catalog.is_null() {
            set_last_error("openasr_pull_model: catalog handle must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        // SAFETY: caller contract requires a live catalog handle.
        let catalog = unsafe { &*catalog };
        let reference = match unsafe { required_c_str(reference, "reference") } {
            Ok(reference) => reference,
            Err(status) => return status,
        };
        let home = match unsafe { c_str_to_path(home_dir) } {
            Ok(home) => home,
            Err(status) => return status,
        };
        let quant = match unsafe { opt_c_str(quant, "quant") } {
            Ok(quant) => quant,
            Err(status) => return status,
        };
        let source = match unsafe { opt_c_str(source, "source") } {
            Ok(source) => source,
            Err(status) => return status,
        };

        // Resolve against the verified catalog: this (not the app) picks the
        // url + sha256 the download is pinned to, so the app can only choose a
        // reference, never redirect the fetch or weaken the digest check.
        let request = CatalogPullRequest {
            reference: reference.clone(),
            quant,
            size: None,
        };
        let resolved = match resolve_catalog_pull_with_profile(
            &catalog.inner,
            &request,
            Some(openasr_core::host_quant_recommendation_profile()),
        ) {
            Ok(resolved) => resolved,
            Err(error) => {
                set_last_error(format!("openasr_pull_model: {error}"));
                return OpenAsrStatus::CatalogFailed;
            }
        };

        if matches!(resolved.license_class, LicenseClass::Gated) && !accept_license {
            set_last_error(format!(
                "openasr_pull_model: model '{}' requires accepting a vendor license (review {}); call again with accept_license=true",
                resolved.model_id, resolved.license_url
            ));
            return OpenAsrStatus::PullFailed;
        }

        let source_chain = match source {
            Some(source) => match DownloadSourcePref::parse_env_value(&source) {
                Some(pref) => resolve_chain(&pref),
                None => {
                    set_last_error(format!(
                        "openasr_pull_model: unsupported download source '{source}'"
                    ));
                    return OpenAsrStatus::InvalidArgument;
                }
            },
            None => resolve_chain(&DownloadSourcePref::Auto),
        };

        let progress = |event: PullProgress| {
            report_progress(progress_cb, progress_user_data, &event);
        };
        let should_cancel = || match cancel_cb {
            // SAFETY: caller contract requires a valid, non-unwinding function
            // pointer when `cancel_cb` is non-null; it is only ever called on
            // this thread.
            Some(cancel) => unsafe { cancel(cancel_user_data) },
            None => false,
        };

        let result = PullModelPackRequest::new(&resolved, &home)
            .sources(&source_chain)
            .cancel(should_cancel)
            .execute(progress);

        finish_install(result, out_installed_json, "openasr_pull_model")
    })
}

/// Verifies and installs a `.oasr` pack the app already has on disk (e.g. one
/// side-loaded or copied into the app) without downloading it. The pack's
/// sha256/size must match an entry in the verified `catalog`, or the call fails
/// closed with [`OpenAsrStatus::PullFailed`] -- so a hand-supplied file cannot be
/// installed as a model the catalog never vouched for.
///
/// `oasr_path` is the local pack path; `home_dir` is the app's OpenASR home.
/// `progress_cb` (optional) reports the verify/install phases. `out_installed_json`,
/// if non-null, receives a JSON object for the installed pack (free with
/// [`openasr_string_free`]).
///
/// # Safety
/// `catalog` must be a live handle from [`openasr_catalog_fetch`]. `oasr_path`
/// and `home_dir` must be valid, non-empty, NUL-terminated UTF-8 C strings.
/// `progress_cb`, if non-null, must be a valid non-unwinding function pointer.
/// `out_installed_json`, if non-null, must point to a writable `*mut c_char`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_install_local_pack(
    catalog: *const OpenAsrCatalog,
    oasr_path: *const c_char,
    home_dir: *const c_char,
    progress_cb: OpenAsrPullProgressCallback,
    progress_user_data: *mut c_void,
    out_installed_json: *mut *mut c_char,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::PullFailed, || {
        clear_out_string(out_installed_json);
        if catalog.is_null() {
            set_last_error("openasr_install_local_pack: catalog handle must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        // SAFETY: caller contract requires a live catalog handle.
        let catalog = unsafe { &*catalog };
        let source_path = match unsafe { c_str_to_path(oasr_path) } {
            Ok(path) => path,
            Err(status) => return status,
        };
        let home = match unsafe { c_str_to_path(home_dir) } {
            Ok(home) => home,
            Err(status) => return status,
        };
        let progress = |event: PullProgress| {
            report_progress(progress_cb, progress_user_data, &event);
        };
        let result =
            install_catalog_model_pack_from_path(&catalog.inner, &source_path, &home, progress);
        finish_install(result, out_installed_json, "openasr_install_local_pack")
    })
}

/// Lists the installed model packs under `home_dir`, writing a
/// freshly-allocated UTF-8/NUL-terminated JSON array (one object per installed
/// pack: pull id, path, model id, quant, size, sha256, ...) through `out_json`.
/// An empty install set yields `"[]"`, not null. Free the string with
/// [`openasr_string_free`]. Never touches the network.
///
/// # Safety
/// `home_dir` must be a valid, non-empty, NUL-terminated UTF-8 C string.
/// `out_json` must be a valid, non-null pointer to a `*mut c_char` the caller
/// owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_list_installed_json(
    home_dir: *const c_char,
    out_json: *mut *mut c_char,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::IoError, || {
        if out_json.is_null() {
            set_last_error("openasr_list_installed_json: out_json must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        // SAFETY: checked non-null above.
        unsafe {
            *out_json = ptr::null_mut();
        }
        let home = match unsafe { c_str_to_path(home_dir) } {
            Ok(home) => home,
            Err(status) => return status,
        };
        match list_installed_packs(&home) {
            Ok(packs) => write_out_json(
                serde_json::to_string(&packs),
                out_json,
                "openasr_list_installed_json",
            ),
            Err(error) => {
                set_last_error(format!("openasr_list_installed_json: {error}"));
                OpenAsrStatus::IoError
            }
        }
    })
}

/// Removes an installed model pack (by `reference`, a pull id or `id:quant`)
/// from under `home_dir`. `out_removed`, if non-null, is set to true when a pack
/// was found and removed, false when nothing matched. Removing a non-existent
/// pack is not an error (`out_removed = false`, status [`OpenAsrStatus::Ok`]).
///
/// # Safety
/// `reference` and `home_dir` must be valid, non-empty, NUL-terminated UTF-8 C
/// strings. `out_removed`, if non-null, must point to a writable `bool` the
/// caller owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_remove_model(
    home_dir: *const c_char,
    reference: *const c_char,
    out_removed: *mut bool,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::IoError, || {
        if !out_removed.is_null() {
            // SAFETY: caller contract requires a writable pointer when non-null.
            unsafe {
                *out_removed = false;
            }
        }
        let home = match unsafe { c_str_to_path(home_dir) } {
            Ok(home) => home,
            Err(status) => return status,
        };
        let reference = match unsafe { required_c_str(reference, "reference") } {
            Ok(reference) => reference,
            Err(status) => return status,
        };
        match remove_model_pack(&home, &reference) {
            Ok(removed) => {
                if !out_removed.is_null() {
                    // SAFETY: checked non-null above.
                    unsafe {
                        *out_removed = removed.is_some();
                    }
                }
                OpenAsrStatus::Ok
            }
            Err(error) => {
                set_last_error(format!("openasr_remove_model: {error}"));
                OpenAsrStatus::IoError
            }
        }
    })
}

/// Frees a string returned through an `out_*_json` out-parameter by
/// [`openasr_pull_model`], [`openasr_install_local_pack`], or
/// [`openasr_list_installed_json`]. Null is accepted and is a no-op. Do not call
/// this on [`openasr_catalog_json`]'s return value (that is owned by the catalog
/// handle) or on [`openasr_last_error_message`]'s.
///
/// # Safety
/// `string`, if non-null, must be a pointer previously produced by one of the
/// `out_*_json` out-parameters above and not already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_string_free(string: *mut c_char) {
    let _ = catch(OpenAsrStatus::Ok, || {
        if !string.is_null() {
            // SAFETY: caller contract requires this came from `CString::into_raw`
            // in this module and is not double-freed.
            drop(unsafe { CString::from_raw(string) });
        }
        OpenAsrStatus::Ok
    });
}

/// Shared tail for [`openasr_pull_model`] / [`openasr_install_local_pack`]: turn
/// the core pull `Result` into a typed status, serializing the installed pack to
/// `out_installed_json` on success and mapping a cancellation to its own status.
fn finish_install(
    result: Result<InstalledPack, PullError>,
    out_installed_json: *mut *mut c_char,
    context: &str,
) -> OpenAsrStatus {
    match result {
        Ok(installed) => write_out_json(
            serde_json::to_string(&installed),
            out_installed_json,
            context,
        ),
        Err(PullError::Canceled { reference }) => {
            set_last_error(format!("{context}: pull canceled ({reference})"));
            OpenAsrStatus::PullCanceled
        }
        Err(error) => {
            set_last_error(format!("{context}: {error}"));
            OpenAsrStatus::PullFailed
        }
    }
}

/// Writes an already-serialized JSON string to a fresh owned C string through
/// `out` (skipped when `out` is null). A serialization failure is reported as an
/// internal error rather than a fabricated result.
fn write_out_json(
    json: serde_json::Result<String>,
    out: *mut *mut c_char,
    context: &str,
) -> OpenAsrStatus {
    if out.is_null() {
        return OpenAsrStatus::Ok;
    }
    match json {
        Ok(json) => {
            let owned = crate::cstring_lossy(json).into_raw();
            // SAFETY: caller contract requires `out` to be a writable pointer
            // when non-null; checked above.
            unsafe {
                *out = owned;
            }
            OpenAsrStatus::Ok
        }
        Err(error) => {
            set_last_error(format!("{context}: could not serialize result: {error}"));
            OpenAsrStatus::InternalPanic
        }
    }
}

/// Best-effort null-init of an optional out-string pointer at entry, so a caller
/// that ignores the return status never reads an uninitialized/stale pointer.
fn clear_out_string(out: *mut *mut c_char) {
    if !out.is_null() {
        // SAFETY: caller contract requires a writable pointer when non-null.
        unsafe {
            *out = ptr::null_mut();
        }
    }
}

fn report_progress(
    progress_cb: OpenAsrPullProgressCallback,
    user_data: *mut c_void,
    event: &PullProgress,
) {
    let Some(callback) = progress_cb else {
        return;
    };
    let (phase, bytes_done, bytes_total) = match event {
        PullProgress::UsingInstalled { .. } => (OpenAsrPullPhase::UsingInstalled, 0, 0),
        PullProgress::DownloadStarted {
            bytes_total,
            resume_from,
        } => (OpenAsrPullPhase::Started, *resume_from, *bytes_total),
        PullProgress::Downloading {
            bytes_done,
            bytes_total,
        } => (OpenAsrPullPhase::Downloading, *bytes_done, *bytes_total),
        PullProgress::Verifying { bytes_done } => (OpenAsrPullPhase::Verifying, *bytes_done, 0),
        PullProgress::Installed { .. } => (OpenAsrPullPhase::Installed, 0, 0),
    };
    // SAFETY: caller contract requires a valid, non-unwinding function pointer
    // when `progress_cb` is non-null; it is only ever called on this thread.
    unsafe {
        callback(user_data, phase, bytes_done, bytes_total);
    }
}

/// Reads a required C string argument (non-null, non-empty, UTF-8).
///
/// # Safety
/// `ptr` must be a valid NUL-terminated C string, or null.
unsafe fn required_c_str(ptr: *const c_char, name: &str) -> Result<String, OpenAsrStatus> {
    // Reuse the path validator's null/empty/UTF-8 checks, then hand back the
    // owned string rather than a PathBuf.
    match unsafe { c_str_to_path(ptr) } {
        Ok(path) => Ok(path.to_string_lossy().into_owned()),
        Err(status) => {
            // c_str_to_path already set a generic "path argument" message; make
            // it name the actual argument for a clearer error.
            set_last_error(format!("{name} argument must be a non-empty UTF-8 string"));
            Err(status)
        }
    }
}

/// Reads an optional C string argument: null yields `None`; a non-null pointer
/// must be a non-empty, UTF-8 string.
///
/// # Safety
/// `ptr` must be null or a valid NUL-terminated C string.
unsafe fn opt_c_str(ptr: *const c_char, name: &str) -> Result<Option<String>, OpenAsrStatus> {
    if ptr.is_null() {
        return Ok(None);
    }
    unsafe { required_c_str(ptr, name) }.map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;
    use std::ffi::CString as StdCString;
    use std::path::PathBuf;

    fn last_error() -> String {
        let ptr = crate::openasr_last_error_message();
        if ptr.is_null() {
            return String::new();
        }
        // SAFETY: `openasr_last_error_message` returns null or a valid
        // NUL-terminated string owned by this crate's thread-local state.
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn catalog_fetch_rejects_null_out_param() {
        let home = StdCString::new("/nonexistent/home").unwrap();
        // SAFETY: valid home C string; null `out_catalog` is the case under test.
        let status = unsafe { openasr_catalog_fetch(ptr::null(), home.as_ptr(), ptr::null_mut()) };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
        assert!(last_error().contains("out_catalog"));
    }

    #[test]
    fn catalog_fetch_rejects_null_home() {
        let mut catalog: *mut OpenAsrCatalog = ptr::null_mut();
        // SAFETY: null home with a valid out pointer is the invalid-argument
        // case; the url null means "default endpoint".
        let status = unsafe { openasr_catalog_fetch(ptr::null(), ptr::null(), &mut catalog) };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
        assert!(catalog.is_null());
    }

    #[test]
    fn catalog_json_and_free_accept_null() {
        // SAFETY: null is a documented fail-closed / no-op for both.
        unsafe {
            assert!(openasr_catalog_json(ptr::null()).is_null());
            openasr_catalog_free(ptr::null_mut());
        }
    }

    #[test]
    fn catalog_fetch_verifies_the_bundled_signed_catalog_from_a_local_home() {
        // The embedded, signed catalog is the offline fallback: pointing at an
        // empty home with no network still yields a verified catalog (or fails
        // closed), never an unverified one. Serialize round-trips to JSON.
        let temp = tempfile::tempdir().unwrap();
        let home = StdCString::new(temp.path().to_str().unwrap()).unwrap();
        // A bogus local file:// url forces the embedded-snapshot fallback path
        // deterministically without any network access.
        let bogus = StdCString::new("file:///nonexistent/catalog.json").unwrap();
        let mut catalog: *mut OpenAsrCatalog = ptr::null_mut();
        // SAFETY: valid C strings + out pointer.
        let status = unsafe { openasr_catalog_fetch(bogus.as_ptr(), home.as_ptr(), &mut catalog) };
        // Either the embedded catalog verifies (Ok) or verification/parse fails
        // closed (CatalogFailed) -- never a silent partial success with a handle
        // AND a failure status.
        if status == OpenAsrStatus::Ok {
            assert!(!catalog.is_null());
            // SAFETY: live handle from the Ok branch above.
            let json = unsafe { CStr::from_ptr(openasr_catalog_json(catalog)) }
                .to_str()
                .unwrap()
                .to_string();
            assert!(json.contains("\"models\""), "catalog json: {json}");
            // SAFETY: live handle; freed exactly once here.
            unsafe { openasr_catalog_free(catalog) };
        } else {
            assert_eq!(status, OpenAsrStatus::CatalogFailed);
            assert!(catalog.is_null());
        }
    }

    #[test]
    fn pull_rejects_null_catalog() {
        let reference = StdCString::new("moonshine-tiny").unwrap();
        let home = StdCString::new("/nonexistent/home").unwrap();
        // SAFETY: null catalog handle is the case under test; all other args are
        // valid and never reached.
        let status = unsafe {
            openasr_pull_model(
                ptr::null(),
                reference.as_ptr(),
                ptr::null(),
                ptr::null(),
                false,
                home.as_ptr(),
                None,
                ptr::null_mut(),
                None,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
        assert!(last_error().contains("catalog"));
    }

    #[test]
    fn install_local_pack_rejects_null_catalog() {
        let path = StdCString::new("/nonexistent/model.oasr").unwrap();
        let home = StdCString::new("/nonexistent/home").unwrap();
        // SAFETY: null catalog handle is the case under test.
        let status = unsafe {
            openasr_install_local_pack(
                ptr::null(),
                path.as_ptr(),
                home.as_ptr(),
                None,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
    }

    #[test]
    fn list_installed_is_empty_json_for_a_fresh_home() {
        let temp = tempfile::tempdir().unwrap();
        let home = StdCString::new(temp.path().to_str().unwrap()).unwrap();
        let mut json: *mut c_char = ptr::null_mut();
        // SAFETY: valid home + out pointer.
        let status = unsafe { openasr_list_installed_json(home.as_ptr(), &mut json) };
        assert_eq!(status, OpenAsrStatus::Ok, "{}", last_error());
        assert!(!json.is_null());
        // SAFETY: `json` is a live owned string from the Ok branch.
        let text = unsafe { CStr::from_ptr(json) }
            .to_str()
            .unwrap()
            .to_string();
        assert_eq!(text, "[]");
        // SAFETY: `json` came from `CString::into_raw`; freed once here.
        unsafe { openasr_string_free(json) };
    }

    #[test]
    fn list_installed_rejects_null_out() {
        let home = StdCString::new("/tmp").unwrap();
        // SAFETY: null out pointer is the case under test.
        let status = unsafe { openasr_list_installed_json(home.as_ptr(), ptr::null_mut()) };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
    }

    #[test]
    fn remove_missing_model_is_ok_and_not_removed() {
        let temp = tempfile::tempdir().unwrap();
        let home = StdCString::new(temp.path().to_str().unwrap()).unwrap();
        let reference = StdCString::new("moonshine-tiny:q8_0").unwrap();
        let mut removed = true;
        // SAFETY: valid home + reference + out pointer.
        let status =
            unsafe { openasr_remove_model(home.as_ptr(), reference.as_ptr(), &mut removed) };
        assert_eq!(status, OpenAsrStatus::Ok, "{}", last_error());
        assert!(!removed);
    }

    #[test]
    fn string_free_accepts_null() {
        // SAFETY: null is a documented no-op.
        unsafe { openasr_string_free(ptr::null_mut()) };
    }

    #[test]
    fn pull_phase_maps_every_progress_variant() {
        // The mapping is exhaustive over PullProgress; assert the boundary
        // cases carry their byte counts through.
        let mut seen: Vec<(OpenAsrPullPhase, u64, u64)> = Vec::new();
        extern "C" fn capture(
            user_data: *mut c_void,
            phase: OpenAsrPullPhase,
            done: u64,
            total: u64,
        ) {
            // SAFETY: `user_data` is the &mut Vec passed below, used only on
            // this (the test) thread.
            let seen = unsafe { &mut *(user_data as *mut Vec<(OpenAsrPullPhase, u64, u64)>) };
            seen.push((phase, done, total));
        }
        let cb: OpenAsrPullProgressCallback = Some(capture);
        let ud = &mut seen as *mut _ as *mut c_void;
        report_progress(
            cb,
            ud,
            &PullProgress::DownloadStarted {
                bytes_total: 100,
                resume_from: 10,
            },
        );
        report_progress(
            cb,
            ud,
            &PullProgress::Downloading {
                bytes_done: 50,
                bytes_total: 100,
            },
        );
        report_progress(cb, ud, &PullProgress::Verifying { bytes_done: 100 });
        report_progress(
            cb,
            ud,
            &PullProgress::Installed {
                path: PathBuf::from("/x"),
            },
        );
        report_progress(
            cb,
            ud,
            &PullProgress::UsingInstalled {
                path: PathBuf::from("/x"),
            },
        );
        assert_eq!(
            seen,
            vec![
                (OpenAsrPullPhase::Started, 10, 100),
                (OpenAsrPullPhase::Downloading, 50, 100),
                (OpenAsrPullPhase::Verifying, 100, 0),
                (OpenAsrPullPhase::Installed, 0, 0),
                (OpenAsrPullPhase::UsingInstalled, 0, 0),
            ]
        );
    }
}
