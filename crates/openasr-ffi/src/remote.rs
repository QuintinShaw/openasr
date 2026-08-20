//! Opt-in LAN remote-compute C ABI.
//!
//! Trust decisions (certificate fingerprint, pairing safety code, TOFU pin)
//! stay in `openasr-client`. This module only marshals C arguments and holds
//! session state. Bearer tokens live on the Rust handle / injected secret
//! store; they are never copied into a long-lived C string.

use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;
use std::sync::Arc;

use openasr_client::{
    ClientError, MemorySecretStore, PairingPoll, RealtimeWorker, RemoteClient, SecretStore,
};

use crate::{
    OpenAsrPcmFormat, OpenAsrResult, OpenAsrStatus, catch, result_from_text, set_last_error,
};

/// Opaque remote-compute client. Create with [`openasr_remote_client_create`]
/// or restore with [`openasr_remote_restore_connected`]; free with
/// [`openasr_remote_client_free`]. Holds the pairing session and, once
/// approved, the bearer token inside the injected secret store -- never as a
/// C string the caller is expected to keep.
///
/// **Not thread-safe.** Do not call any `openasr_remote_*` function on the same
/// client (or its realtime session) from multiple threads. Do not call any
/// `openasr_remote_*` function from an [`OpenAsrRemoteRealtimeEventCallback`].
pub struct OpenAsrRemoteClient {
    runtime: tokio::runtime::Runtime,
    inner: RemoteClient,
    device_name: String,
    last_safety_code: Option<CString>,
}

/// Opaque remote realtime session. Create with [`openasr_remote_realtime_start`],
/// stop with [`openasr_remote_realtime_stop`].
pub struct OpenAsrRemoteRealtime {
    worker: RealtimeWorker,
}

/// Outcome of [`openasr_remote_poll_pairing`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAsrRemotePairingStatus {
    Pending = 0,
    Approved = 1,
}

/// Optional injected secret store (keychain / callback). Pass null to
/// [`openasr_remote_client_create`] / [`openasr_remote_restore_connected`] to
/// use an in-memory store. A non-null vtable must provide `store`, `load`, and
/// `free_secret`; missing any of those fails closed instead of falling back to
/// an in-memory store. `delete_secret` is optional.
///
/// `store`: `secret` is valid only for the duration of this callback. The
/// implementor must copy the bytes if they need to retain them.
///
/// `load` must write a freshly allocated C string to `out_secret` on success
/// (0) or null if the account is missing. `free_secret` must free that string;
/// the FFI copies the secret into Rust and immediately calls `free_secret`. A
/// successful non-null `load` without `free_secret` is an error.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OpenAsrRemoteSecretStore {
    pub user_data: *mut c_void,
    pub store: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            account: *const c_char,
            secret: *const c_char,
        ) -> i32,
    >,
    pub load: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            account: *const c_char,
            out_secret: *mut *mut c_char,
        ) -> i32,
    >,
    pub free_secret: Option<unsafe extern "C" fn(secret: *mut c_char)>,
    pub delete_secret:
        Option<unsafe extern "C" fn(user_data: *mut c_void, account: *const c_char) -> i32>,
}

/// Incremental realtime event callback. `text` is valid only for the duration
/// of the call. The callback may run on a background thread. **Do not call any
/// `openasr_remote_*` function from this callback** -- the client and session
/// are not re-entrant.
pub type OpenAsrRemoteRealtimeEventCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, text: *const c_char)>;

struct FfiSecretStore {
    user_data: usize,
    store: OpenAsrRemoteSecretStore,
}

unsafe impl Send for FfiSecretStore {}
unsafe impl Sync for FfiSecretStore {}

impl SecretStore for FfiSecretStore {
    fn store_secret(&self, account: &str, secret: &str) -> Result<(), ClientError> {
        let Some(store) = self.store.store else {
            return Err(ClientError::new(
                "OpenASR remote secret store is missing a store callback.",
            ));
        };
        let account = CString::new(account)
            .map_err(|_| ClientError::new("Secret account contained an embedded NUL."))?;
        let secret = CString::new(secret)
            .map_err(|_| ClientError::new("Secret value contained an embedded NUL."))?;
        // SAFETY: caller-supplied keychain callback; account/secret are live C strings.
        let rc = unsafe {
            store(
                self.user_data as *mut c_void,
                account.as_ptr(),
                secret.as_ptr(),
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(ClientError::new(
                "OpenASR remote secret store rejected the token.",
            ))
        }
    }

    fn load_secret(&self, account: &str) -> Result<Option<String>, ClientError> {
        let Some(load) = self.store.load else {
            return Ok(None);
        };
        let account = CString::new(account)
            .map_err(|_| ClientError::new("Secret account contained an embedded NUL."))?;
        let mut out_secret: *mut c_char = ptr::null_mut();
        // SAFETY: caller-supplied keychain callback.
        let rc = unsafe {
            load(
                self.user_data as *mut c_void,
                account.as_ptr(),
                &mut out_secret,
            )
        };
        if rc != 0 {
            return Err(ClientError::new(
                "OpenASR remote secret store could not load the token.",
            ));
        }
        if out_secret.is_null() {
            return Ok(None);
        }
        let Some(free_secret) = self.store.free_secret else {
            return Err(ClientError::new(
                "OpenASR remote secret store loaded a token without a free_secret callback.",
            ));
        };
        // SAFETY: callback allocated a C string we copy then free.
        let copied = unsafe { CStr::from_ptr(out_secret) }
            .to_string_lossy()
            .into_owned();
        unsafe { free_secret(out_secret) };
        Ok(Some(copied))
    }

    fn delete_secret(&self, account: &str) -> Result<(), ClientError> {
        let Some(delete_secret) = self.store.delete_secret else {
            return Ok(());
        };
        let account = CString::new(account)
            .map_err(|_| ClientError::new("Secret account contained an embedded NUL."))?;
        let rc = unsafe { delete_secret(self.user_data as *mut c_void, account.as_ptr()) };
        if rc == 0 {
            Ok(())
        } else {
            Err(ClientError::new(
                "OpenASR remote secret store could not delete the token.",
            ))
        }
    }
}

fn map_remote_error(error: ClientError) -> OpenAsrStatus {
    set_last_error(error.message().to_string());
    OpenAsrStatus::RemoteFailed
}

unsafe fn required_c_str(ptr: *const c_char, what: &str) -> Result<String, OpenAsrStatus> {
    if ptr.is_null() {
        set_last_error(format!("{what} must not be null"));
        return Err(OpenAsrStatus::InvalidArgument);
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map(str::to_string)
        .map_err(|_| {
            set_last_error(format!("{what} must be valid UTF-8"));
            OpenAsrStatus::InvalidArgument
        })
}

fn secret_store_from_c(
    secret_store: *const OpenAsrRemoteSecretStore,
) -> Result<Arc<dyn SecretStore>, OpenAsrStatus> {
    if secret_store.is_null() {
        return Ok(Arc::new(MemorySecretStore::new()));
    }
    // SAFETY: the FFI caller passed a live `OpenAsrRemoteSecretStore` for the
    // duration of this call. Callbacks must remain valid for the client lifetime.
    let store = unsafe { *secret_store };
    if store.store.is_none() || store.load.is_none() || store.free_secret.is_none() {
        set_last_error(
            "OpenASR remote secret store is missing required store, load, or free_secret callbacks.",
        );
        return Err(OpenAsrStatus::InvalidArgument);
    }
    Ok(Arc::new(FfiSecretStore {
        user_data: store.user_data as usize,
        store,
    }))
}

fn build_remote_client(
    host: String,
    port: u16,
    device_name: String,
    secrets: Arc<dyn SecretStore>,
) -> Result<OpenAsrRemoteClient, OpenAsrStatus> {
    let inner = match RemoteClient::new(host, port, secrets) {
        Ok(inner) => inner,
        Err(error) => return Err(map_remote_error(error)),
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            set_last_error(format!("Could not start OpenASR remote runtime: {error}"));
            return Err(OpenAsrStatus::RemoteFailed);
        }
    };
    Ok(OpenAsrRemoteClient {
        runtime,
        inner,
        device_name,
        last_safety_code: None,
    })
}

/// Creates a remote-compute client for a LAN host/port. `device_name` is sent
/// on [`openasr_remote_begin_pairing`]. `secret_store` may be null (in-memory).
/// A non-null store must provide `store`, `load`, and `free_secret`.
///
/// **Not thread-safe.** See [`OpenAsrRemoteClient`].
///
/// # Safety
/// `host` and `device_name` must be valid NUL-terminated UTF-8. `out_client`
/// must point to writable storage for one `*mut OpenAsrRemoteClient`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_remote_client_create(
    host: *const c_char,
    port: u16,
    device_name: *const c_char,
    secret_store: *const OpenAsrRemoteSecretStore,
    out_client: *mut *mut OpenAsrRemoteClient,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::RemoteFailed, || {
        if out_client.is_null() {
            set_last_error("openasr_remote_client_create: out_client must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        unsafe { *out_client = ptr::null_mut() };
        let host = match unsafe { required_c_str(host, "host") } {
            Ok(host) => host,
            Err(status) => return status,
        };
        let device_name = match unsafe { required_c_str(device_name, "device_name") } {
            Ok(device_name) => device_name,
            Err(status) => return status,
        };
        let secrets = match secret_store_from_c(secret_store) {
            Ok(secrets) => secrets,
            Err(status) => return status,
        };
        match build_remote_client(host, port, device_name, secrets) {
            Ok(client) => {
                unsafe { *out_client = Box::into_raw(Box::new(client)) };
                OpenAsrStatus::Ok
            }
            Err(status) => status,
        }
    })
}

/// Restore a previously paired client from persisted host/port/fingerprint/
/// device id. The bearer token is loaded from `secret_store`; C must not pass
/// a token. Missing token or a fingerprint that does not match the stored
/// account fails closed.
///
/// **Not thread-safe.** See [`OpenAsrRemoteClient`].
///
/// # Safety
/// `host`, `device_name`, `fingerprint`, and `device_id` must be valid
/// NUL-terminated UTF-8. `out_client` must point to writable storage for one
/// `*mut OpenAsrRemoteClient`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_remote_restore_connected(
    host: *const c_char,
    port: u16,
    device_name: *const c_char,
    fingerprint: *const c_char,
    device_id: *const c_char,
    secret_store: *const OpenAsrRemoteSecretStore,
    out_client: *mut *mut OpenAsrRemoteClient,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::RemoteFailed, || {
        if out_client.is_null() {
            set_last_error("openasr_remote_restore_connected: out_client must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        unsafe { *out_client = ptr::null_mut() };
        let host = match unsafe { required_c_str(host, "host") } {
            Ok(host) => host,
            Err(status) => return status,
        };
        let device_name = match unsafe { required_c_str(device_name, "device_name") } {
            Ok(device_name) => device_name,
            Err(status) => return status,
        };
        let fingerprint = match unsafe { required_c_str(fingerprint, "fingerprint") } {
            Ok(fingerprint) => fingerprint,
            Err(status) => return status,
        };
        let device_id = match unsafe { required_c_str(device_id, "device_id") } {
            Ok(device_id) => device_id,
            Err(status) => return status,
        };
        let secrets = match secret_store_from_c(secret_store) {
            Ok(secrets) => secrets,
            Err(status) => return status,
        };
        let mut client = match build_remote_client(host, port, device_name, secrets) {
            Ok(client) => client,
            Err(status) => return status,
        };
        if let Err(error) = client.inner.restore_connected(&fingerprint, &device_id) {
            return map_remote_error(error);
        }
        unsafe { *out_client = Box::into_raw(Box::new(client)) };
        OpenAsrStatus::Ok
    })
}

/// Frees a remote client. Null is a no-op. Stop any realtime session first.
///
/// # Safety
/// `client`, if non-null, must be a live handle from [`openasr_remote_client_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_remote_client_free(client: *mut OpenAsrRemoteClient) {
    let _ = catch(OpenAsrStatus::Ok, || {
        if !client.is_null() {
            drop(unsafe { Box::from_raw(client) });
        }
        OpenAsrStatus::Ok
    });
}

/// Begin pairing. On success, `*out_safety_code` is a borrowed C string owned
/// by `client` (valid until the next mutating call or free). The TLS-vs-response
/// safety-code match has already been checked in `openasr-client`.
///
/// # Safety
/// `client` must be live. `out_safety_code` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_remote_begin_pairing(
    client: *mut OpenAsrRemoteClient,
    out_safety_code: *mut *const c_char,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::RemoteFailed, || {
        if client.is_null() {
            set_last_error("openasr_remote_begin_pairing: client must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        if out_safety_code.is_null() {
            set_last_error("openasr_remote_begin_pairing: out_safety_code must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        unsafe { *out_safety_code = ptr::null() };
        let client = unsafe { &mut *client };
        let device_name = client.device_name.clone();
        let start = match client
            .runtime
            .block_on(client.inner.begin_pairing(&device_name))
        {
            Ok(start) => start,
            Err(error) => return map_remote_error(error),
        };
        client.last_safety_code = Some(crate::cstring_lossy(start.safety_code));
        unsafe {
            *out_safety_code = client
                .last_safety_code
                .as_ref()
                .map(|code| code.as_ptr())
                .unwrap_or(ptr::null());
        }
        OpenAsrStatus::Ok
    })
}

/// Poll an in-progress pairing request. Does not return the bearer token.
///
/// # Safety
/// `client` must be live. `out_status` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_remote_poll_pairing(
    client: *mut OpenAsrRemoteClient,
    out_status: *mut OpenAsrRemotePairingStatus,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::RemoteFailed, || {
        if client.is_null() {
            set_last_error("openasr_remote_poll_pairing: client must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        if out_status.is_null() {
            set_last_error("openasr_remote_poll_pairing: out_status must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        let client = unsafe { &mut *client };
        match client.runtime.block_on(client.inner.poll_pairing()) {
            Ok(PairingPoll::Pending) => {
                unsafe { *out_status = OpenAsrRemotePairingStatus::Pending };
                OpenAsrStatus::Ok
            }
            Ok(PairingPoll::Approved { .. }) => {
                client.last_safety_code = None;
                unsafe { *out_status = OpenAsrRemotePairingStatus::Approved };
                OpenAsrStatus::Ok
            }
            Err(error) => map_remote_error(error),
        }
    })
}

/// Cancel in-progress pairing or disconnect a connected session. Drops the
/// stored token from the secret store when a device id is present.
///
/// # Safety
/// `client` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_remote_cancel(client: *mut OpenAsrRemoteClient) -> OpenAsrStatus {
    catch(OpenAsrStatus::RemoteFailed, || {
        if client.is_null() {
            set_last_error("openasr_remote_cancel: client must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        unsafe { &mut *client }.inner.cancel_pairing();
        OpenAsrStatus::Ok
    })
}

/// Transcribe in-memory PCM through the paired remote server. 16 kHz mono.
///
/// # Safety
/// Same pointer rules as [`crate::openasr_transcribe_pcm`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_remote_transcribe_pcm(
    client: *mut OpenAsrRemoteClient,
    model: *const c_char,
    pcm: *const c_void,
    pcm_len_samples: usize,
    format: OpenAsrPcmFormat,
    sample_rate_hz: u32,
    out_result: *mut *mut OpenAsrResult,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::RemoteFailed, || {
        if out_result.is_null() {
            set_last_error("openasr_remote_transcribe_pcm: out_result must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        unsafe { *out_result = ptr::null_mut() };
        if client.is_null() {
            set_last_error("openasr_remote_transcribe_pcm: client must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        if sample_rate_hz != 16_000 {
            set_last_error(
                "openasr_remote_transcribe_pcm: only 16000 Hz mono PCM is supported in v1",
            );
            return OpenAsrStatus::InvalidArgument;
        }
        if pcm.is_null() && pcm_len_samples > 0 {
            set_last_error("openasr_remote_transcribe_pcm: pcm must not be null when non-empty");
            return OpenAsrStatus::InvalidArgument;
        }
        let model = match unsafe { required_c_str(model, "model") } {
            Ok(model) => model,
            Err(status) => return status,
        };
        let client = unsafe { &mut *client };
        let response = match format {
            OpenAsrPcmFormat::F32 => {
                let samples = if pcm_len_samples == 0 {
                    &[][..]
                } else {
                    unsafe { std::slice::from_raw_parts(pcm as *const f32, pcm_len_samples) }
                };
                client.runtime.block_on(client.inner.transcribe_pcm_f32(
                    &model,
                    samples,
                    sample_rate_hz,
                ))
            }
            OpenAsrPcmFormat::S16 => {
                let samples = if pcm_len_samples == 0 {
                    &[][..]
                } else {
                    unsafe { std::slice::from_raw_parts(pcm as *const i16, pcm_len_samples) }
                };
                client.runtime.block_on(client.inner.transcribe_pcm_s16(
                    &model,
                    samples,
                    sample_rate_hz,
                ))
            }
        };
        let response = match response {
            Ok(response) => response,
            Err(error) => return map_remote_error(error),
        };
        let text = match response.text() {
            Ok(text) => text,
            Err(error) => return map_remote_error(error),
        };
        unsafe {
            *out_result = Box::into_raw(Box::new(result_from_text(text)));
        }
        OpenAsrStatus::Ok
    })
}

/// Start a pinned WSS realtime session. Completes the server handshake
/// (`audio.input.configure` then `session.start` for 16 kHz mono pcm16le) with
/// `model` before returning, so [`openasr_remote_realtime_feed`] is legal.
/// Incoming text frames are delivered to `on_message` (may be called from a
/// background thread, and must not call any `openasr_remote_*` function). The
/// bearer token stays on the Rust handle.
///
/// **Not thread-safe.** See [`OpenAsrRemoteClient`].
///
/// # Safety
/// `client` must be live and paired. `model` must be a valid NUL-terminated
/// UTF-8 C string. `out_session` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_remote_realtime_start(
    client: *mut OpenAsrRemoteClient,
    model: *const c_char,
    on_message: OpenAsrRemoteRealtimeEventCallback,
    user_data: *mut c_void,
    out_session: *mut *mut OpenAsrRemoteRealtime,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::RemoteFailed, || {
        if out_session.is_null() {
            set_last_error("openasr_remote_realtime_start: out_session must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        unsafe { *out_session = ptr::null_mut() };
        if client.is_null() {
            set_last_error("openasr_remote_realtime_start: client must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        let model = match unsafe { required_c_str(model, "model") } {
            Ok(model) if !model.trim().is_empty() => model,
            Ok(_) => {
                set_last_error("openasr_remote_realtime_start: model must not be empty");
                return OpenAsrStatus::InvalidArgument;
            }
            Err(status) => return status,
        };
        let client = unsafe { &mut *client };
        let user_data = user_data as usize;
        let worker =
            match client
                .runtime
                .block_on(client.inner.realtime_start_worker(&model, move |text| {
                    if let Some(on_message) = on_message {
                        let cstring = crate::cstring_lossy(text);
                        unsafe { on_message(user_data as *mut c_void, cstring.as_ptr()) };
                    }
                })) {
                Ok(worker) => worker,
                Err(error) => return map_remote_error(error),
            };
        unsafe {
            *out_session = Box::into_raw(Box::new(OpenAsrRemoteRealtime { worker }));
        }
        OpenAsrStatus::Ok
    })
}

/// Feed PCM16LE samples into an open remote realtime session. The session must
/// have been returned by a successful [`openasr_remote_realtime_start`]
/// (handshake completed). Feeding before start fails closed.
///
/// # Safety
/// `session` must be live. `pcm16le` must point to `sample_count` i16 samples.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_remote_realtime_feed(
    session: *mut OpenAsrRemoteRealtime,
    pcm16le: *const i16,
    sample_count: usize,
) -> OpenAsrStatus {
    catch(OpenAsrStatus::RemoteFailed, || {
        if session.is_null() {
            set_last_error("openasr_remote_realtime_feed: session must not be null");
            return OpenAsrStatus::InvalidArgument;
        }
        if pcm16le.is_null() || sample_count == 0 {
            set_last_error("openasr_remote_realtime_feed: audio frame must not be empty");
            return OpenAsrStatus::InvalidArgument;
        }
        let samples = unsafe { std::slice::from_raw_parts(pcm16le, sample_count) };
        let mut bytes = Vec::with_capacity(sample_count * 2);
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        match unsafe { &*session }.worker.try_send_pcm16le(bytes) {
            Ok(()) => OpenAsrStatus::Ok,
            Err(error) => map_remote_error(error),
        }
    })
}

/// Stop a remote realtime session. Null is a no-op.
///
/// # Safety
/// `session`, if non-null, must be a live handle from
/// [`openasr_remote_realtime_start`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn openasr_remote_realtime_stop(session: *mut OpenAsrRemoteRealtime) {
    let _ = catch(OpenAsrStatus::Ok, || {
        if !session.is_null() {
            let session = unsafe { Box::from_raw(session) };
            session.worker.try_close();
        }
        OpenAsrStatus::Ok
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use openasr_client::{
        ClientStatus, MemorySecretStore, RemoteClient, SecretStore, credential_account,
    };

    struct TestStore {
        map: Mutex<HashMap<String, String>>,
    }

    impl TestStore {
        fn new() -> Self {
            Self {
                map: Mutex::new(HashMap::new()),
            }
        }

        fn vtable(&self) -> OpenAsrRemoteSecretStore {
            OpenAsrRemoteSecretStore {
                user_data: self as *const TestStore as *mut c_void,
                store: Some(test_store_store),
                load: Some(test_store_load),
                free_secret: Some(test_store_free),
                delete_secret: Some(test_store_delete),
            }
        }
    }

    unsafe extern "C" fn test_store_store(
        user_data: *mut c_void,
        account: *const c_char,
        secret: *const c_char,
    ) -> i32 {
        let store = unsafe { &*(user_data as *const TestStore) };
        let account = unsafe { CStr::from_ptr(account) }
            .to_string_lossy()
            .into_owned();
        let secret = unsafe { CStr::from_ptr(secret) }
            .to_string_lossy()
            .into_owned();
        store
            .map
            .lock()
            .expect("test store")
            .insert(account, secret);
        0
    }

    unsafe extern "C" fn test_store_load(
        user_data: *mut c_void,
        account: *const c_char,
        out_secret: *mut *mut c_char,
    ) -> i32 {
        let store = unsafe { &*(user_data as *const TestStore) };
        let account = unsafe { CStr::from_ptr(account) }
            .to_string_lossy()
            .into_owned();
        let value = store.map.lock().expect("test store").get(&account).cloned();
        unsafe {
            *out_secret = match value {
                Some(secret) => CString::new(secret).unwrap().into_raw(),
                None => ptr::null_mut(),
            };
        }
        0
    }

    unsafe extern "C" fn test_store_free(secret: *mut c_char) {
        if !secret.is_null() {
            drop(unsafe { CString::from_raw(secret) });
        }
    }

    unsafe extern "C" fn test_store_delete(user_data: *mut c_void, account: *const c_char) -> i32 {
        let store = unsafe { &*(user_data as *const TestStore) };
        let account = unsafe { CStr::from_ptr(account) }
            .to_string_lossy()
            .into_owned();
        store.map.lock().expect("test store").remove(&account);
        0
    }

    fn last_error() -> String {
        let ptr = crate::openasr_last_error_message();
        if ptr.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn create_rejects_null_out_pointer() {
        let host = CString::new("127.0.0.1").unwrap();
        let name = CString::new("test").unwrap();
        let status = unsafe {
            openasr_remote_client_create(
                host.as_ptr(),
                8080,
                name.as_ptr(),
                ptr::null(),
                ptr::null_mut(),
            )
        };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
    }

    #[test]
    fn incomplete_secret_store_vtable_fails_closed() {
        let host = CString::new("127.0.0.1").unwrap();
        let name = CString::new("test").unwrap();
        let mut client = ptr::null_mut();
        let empty = OpenAsrRemoteSecretStore {
            user_data: ptr::null_mut(),
            store: None,
            load: None,
            free_secret: None,
            delete_secret: None,
        };
        let status = unsafe {
            openasr_remote_client_create(host.as_ptr(), 8080, name.as_ptr(), &empty, &mut client)
        };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
        assert!(client.is_null());
        assert!(
            last_error().contains("store, load, or free_secret"),
            "{}",
            last_error()
        );

        let store_and_load = OpenAsrRemoteSecretStore {
            user_data: ptr::null_mut(),
            store: Some(test_store_store),
            load: Some(test_store_load),
            free_secret: None,
            delete_secret: None,
        };
        let status = unsafe {
            openasr_remote_client_create(
                host.as_ptr(),
                8080,
                name.as_ptr(),
                &store_and_load,
                &mut client,
            )
        };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
        assert!(client.is_null());
    }

    #[test]
    fn restore_connected_fails_closed_without_token() {
        let host = CString::new("127.0.0.1").unwrap();
        let name = CString::new("test").unwrap();
        let fingerprint = CString::new("aa".repeat(32)).unwrap();
        let device_id = CString::new("device-1").unwrap();
        let mut client = ptr::null_mut();
        let status = unsafe {
            openasr_remote_restore_connected(
                host.as_ptr(),
                8080,
                name.as_ptr(),
                fingerprint.as_ptr(),
                device_id.as_ptr(),
                ptr::null(),
                &mut client,
            )
        };
        assert_eq!(status, OpenAsrStatus::RemoteFailed);
        assert!(client.is_null());
        assert!(
            last_error().contains("token is missing"),
            "{}",
            last_error()
        );
    }

    #[test]
    fn restore_connected_fails_closed_on_fingerprint_mismatch() {
        let store = TestStore::new();
        store.map.lock().unwrap().insert(
            credential_account(&"aa".repeat(32), "device-1"),
            "oasr_stored".to_string(),
        );
        let vtable = store.vtable();
        let host = CString::new("127.0.0.1").unwrap();
        let name = CString::new("test").unwrap();
        let fingerprint = CString::new("bb".repeat(32)).unwrap();
        let device_id = CString::new("device-1").unwrap();
        let mut client = ptr::null_mut();
        let status = unsafe {
            openasr_remote_restore_connected(
                host.as_ptr(),
                8080,
                name.as_ptr(),
                fingerprint.as_ptr(),
                device_id.as_ptr(),
                &vtable,
                &mut client,
            )
        };
        assert_eq!(status, OpenAsrStatus::RemoteFailed);
        assert!(client.is_null());
        assert!(
            last_error().contains("token is missing"),
            "{}",
            last_error()
        );
    }

    #[test]
    fn restore_connected_loads_token_from_secret_store() {
        let store = TestStore::new();
        let fingerprint = "aa".repeat(32);
        store.map.lock().unwrap().insert(
            credential_account(&fingerprint, "device-1"),
            "oasr_stored".to_string(),
        );
        let vtable = store.vtable();
        let host = CString::new("127.0.0.1").unwrap();
        let name = CString::new("test").unwrap();
        let fingerprint_c = CString::new(fingerprint).unwrap();
        let device_id = CString::new("device-1").unwrap();
        let mut client = ptr::null_mut();
        let status = unsafe {
            openasr_remote_restore_connected(
                host.as_ptr(),
                8080,
                name.as_ptr(),
                fingerprint_c.as_ptr(),
                device_id.as_ptr(),
                &vtable,
                &mut client,
            )
        };
        assert_eq!(status, OpenAsrStatus::Ok);
        assert!(!client.is_null());
        assert_eq!(unsafe { &*client }.inner.status(), ClientStatus::Connected);
        unsafe { openasr_remote_client_free(client) };
    }

    #[test]
    fn realtime_start_rejects_null_model() {
        let host = CString::new("127.0.0.1").unwrap();
        let name = CString::new("test").unwrap();
        let mut client = ptr::null_mut();
        assert_eq!(
            unsafe {
                openasr_remote_client_create(
                    host.as_ptr(),
                    8080,
                    name.as_ptr(),
                    ptr::null(),
                    &mut client,
                )
            },
            OpenAsrStatus::Ok
        );
        let mut session = ptr::null_mut();
        let status = unsafe {
            openasr_remote_realtime_start(client, ptr::null(), None, ptr::null_mut(), &mut session)
        };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
        assert!(session.is_null());
        unsafe { openasr_remote_client_free(client) };
    }

    #[test]
    fn realtime_feed_before_start_fails_closed() {
        let status =
            unsafe { openasr_remote_realtime_feed(ptr::null_mut(), [0i16; 320].as_ptr(), 320) };
        assert_eq!(status, OpenAsrStatus::InvalidArgument);
    }

    #[test]
    fn realtime_start_handshakes_then_allows_binary() {
        let home = tempfile::tempdir().unwrap();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = runtime.block_on(openasr_server::testing::spawn_loopback_pairing_server(
            home.path(),
        ));
        let secrets = Arc::new(MemorySecretStore::new());
        let mut paired =
            RemoteClient::new("127.0.0.1", server.addr.port(), secrets.clone()).unwrap();
        let start = runtime
            .block_on(paired.begin_pairing("FFI Realtime"))
            .unwrap();
        runtime.block_on(openasr_server::testing::approve_pending_pairing_request(
            &server,
            &start.request_id,
        ));
        runtime.block_on(paired.poll_pairing()).unwrap();
        let fingerprint = paired.server_fingerprint().unwrap().to_string();
        let device_id = paired.device_id().unwrap().to_string();
        let token = secrets
            .load_secret(&credential_account(&fingerprint, &device_id))
            .unwrap()
            .expect("paired token");

        let store = TestStore::new();
        store
            .map
            .lock()
            .unwrap()
            .insert(credential_account(&fingerprint, &device_id), token);
        let vtable = store.vtable();
        let host = CString::new("127.0.0.1").unwrap();
        let name = CString::new("FFI Realtime").unwrap();
        let fingerprint_c = CString::new(fingerprint).unwrap();
        let device_id_c = CString::new(device_id).unwrap();
        let mut client = ptr::null_mut();
        assert_eq!(
            unsafe {
                openasr_remote_restore_connected(
                    host.as_ptr(),
                    server.addr.port(),
                    name.as_ptr(),
                    fingerprint_c.as_ptr(),
                    device_id_c.as_ptr(),
                    &vtable,
                    &mut client,
                )
            },
            OpenAsrStatus::Ok
        );

        let model = CString::new("whisper-tiny").unwrap();
        let mut session = ptr::null_mut();
        let status = unsafe {
            openasr_remote_realtime_start(
                client,
                model.as_ptr(),
                None,
                ptr::null_mut(),
                &mut session,
            )
        };
        assert_eq!(status, OpenAsrStatus::Ok, "{}", last_error());
        assert!(!session.is_null());

        let pcm = [0i16; 320];
        let feed = unsafe { openasr_remote_realtime_feed(session, pcm.as_ptr(), pcm.len()) };
        assert_eq!(feed, OpenAsrStatus::Ok, "{}", last_error());

        unsafe {
            openasr_remote_realtime_stop(session);
            openasr_remote_client_free(client);
        }
    }
}
