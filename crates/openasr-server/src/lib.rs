mod realtime;
mod routes;

pub(crate) use routes::config::*;
pub(crate) use routes::history::*;
pub(crate) use routes::models_api::*;
pub(crate) use routes::pairing::*;
pub(crate) use routes::pull_jobs::*;
pub(crate) use routes::speakers::*;
pub(crate) use routes::transcription::*;
pub(crate) use routes::translation::*;

use std::{
    collections::{HashMap, HashSet},
    convert::Infallible,
    env,
    ffi::OsStr,
    fs,
    io::Write,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    Extension, Json, Router,
    extract::Request,
    extract::{
        DefaultBodyLimit, Multipart, Path as AxumPath, Query, State,
        multipart::{Field, MultipartRejection},
    },
    http::{HeaderMap, StatusCode, header},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{any, delete, get, patch, post},
    serve::Listener,
};
use futures_util::stream;
use openasr_core::api::backend::TranscriptionBackendCapabilities;
use openasr_core::config::{load_config_document, save_config_document};
pub use openasr_core::pairing_safety_code_for_certificate_fingerprint;
use openasr_core::realtime::history::{DaemonHistoryEntry, DaemonHistoryStoreError};
use openasr_core::{
    AudioPreparationError, BackendKind, CatalogError, CatalogMirror, CatalogPullRequest,
    InstalledPack, LaunchPackRequest, LicenseClass, OpenAsrHomeError, PullError,
    PullModelPackRequest, PullProgress, QuantPreference, RealtimeBackendCapabilities,
    ResolvedCatalogPull, certificate_fingerprint_sha256, default_pack_pointer_path,
    default_registry_dir, host_quant_recommendation_profile, install_catalog_model_pack_from_path,
    install_model_pack_from_path, list_installed_packs, load_config, load_model_catalog,
    load_registry, native_runtime_realtime_capabilities_for_path,
    native_runtime_transcription_capabilities_for_path, openasr_home, persist_default_pack_pointer,
    read_default_pack_pointer, remove_model_pack, resolve_catalog_pull,
    resolve_installed_pack_reference, resolve_installed_pack_reference_with_catalog,
    resolve_launch_pack, save_default_model_selection,
};
use rcgen::generate_simple_self_signed;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{Semaphore, watch},
    task,
};
use tokio_rustls::{TlsAcceptor, server::TlsStream};

// The current upload path buffers multipart fields in memory via `field.bytes()`,
// so keep the request ceiling conservative until uploads stream directly to disk.
const MAX_TRANSCRIPTION_UPLOAD_BYTES: usize = 64 * 1024 * 1024;
const SERVER_INSTANCE_TOKEN_ENV: &str = "OPENASR_SERVER_INSTANCE_TOKEN";
const MAX_CONCURRENT_PULL_JOBS_PER_HOME: usize = 1;
const PULL_JOB_PROGRESS_PERSIST_INTERVAL_BYTES: u64 = 8 * 1024 * 1024;
const PULL_JOB_PROGRESS_PERSIST_INTERVAL: Duration = Duration::from_secs(1);
pub(crate) const REMOTE_COMPUTE_HEADER: &str = "x-openasr-remote-compute";
pub(crate) const REMOTE_COMPUTE_CLIENT_VALUE: &str = "client";

static ATOMIC_FILE_COUNTER: AtomicU64 = AtomicU64::new(1);

pub fn app() -> Router {
    app_with_runtime(ServerRuntime::default())
}

pub fn app_with_runtime(runtime: ServerRuntime) -> Router {
    app_with_runtime_and_distribution(runtime, DistributionRuntime::default())
}

pub fn app_with_runtime_and_distribution(
    runtime: ServerRuntime,
    distribution_runtime: DistributionRuntime,
) -> Router {
    app_with_runtime_and_distribution_and_launch_options(
        runtime,
        distribution_runtime,
        ServerLaunchOptions::default(),
    )
}

pub fn app_with_runtime_and_distribution_and_launch_options(
    runtime: ServerRuntime,
    distribution_runtime: DistributionRuntime,
    launch_options: ServerLaunchOptions,
) -> Router {
    let distribution = DistributionContext::new(distribution_runtime);
    distribution.ensure_restart_resumes_started();
    let auth = launch_options.auth.clone();
    let health_identity = ServerHealthIdentity::from_launch_options(launch_options);
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/catalog", get(catalog))
        .route("/v1/config", get(get_config).put(put_config))
        .route("/v1/capabilities", get(capabilities))
        .route("/v1/history", get(history_list))
        .route("/v1/history/{id}", get(history_get).delete(history_delete))
        .route("/v1/speakers", get(list_speakers).post(create_speaker))
        .route(
            "/v1/speakers/{id}",
            patch(rename_speaker).delete(delete_speaker),
        )
        .route("/v1/speakers/{id}/reenroll", post(reenroll_speaker))
        .route("/v1/models/local", get(local_models))
        .route("/v1/models/local/import", post(import_local_model))
        .route(
            "/v1/models/default",
            get(default_model)
                .post(set_default_model)
                .put(set_default_model),
        )
        .route("/v1/models/{id}", delete(delete_model))
        .route("/v1/models/{id}/pull", post(start_pull_job))
        .route("/v1/models/pull/{job_id}", get(pull_job))
        .route("/v1/models/pull/{job_id}/events", get(pull_job_events))
        .route("/v1/models/pull/{job_id}/cancel", post(cancel_pull_job))
        .route("/v1/models/pull/{job_id}/pause", post(pause_pull_job))
        .route("/v1/models/pull/{job_id}/resume", post(resume_pull_job))
        .route(
            "/v1/pairing/requests",
            post(create_pairing_request).get(list_pairing_requests),
        )
        .route(
            "/v1/pairing/requests/{request_id}/approve",
            post(approve_pairing_request),
        )
        .route(
            "/v1/pairing/requests/{request_id}",
            delete(reject_pairing_request),
        )
        .route(
            "/v1/pairing/requests/{request_id}/credential",
            get(get_pairing_credential),
        )
        .route("/v1/pairing/credentials", get(list_pairing_credentials))
        .route(
            "/v1/pairing/credentials/{device_id}",
            delete(revoke_pairing_credential),
        )
        .route("/v1/audio/transcriptions", post(transcriptions))
        .route(
            "/v1/audio/transcriptions/progress",
            get(transcription_progress),
        )
        .route("/v1/audio/translations", post(translations))
        .route("/v1/audio/realtime", any(realtime::websocket))
        .layer(middleware::from_fn_with_state(
            auth.clone(),
            require_server_auth,
        ))
        .layer(Extension(auth))
        .layer(Extension(health_identity))
        .layer(Extension(distribution))
        .layer(DefaultBodyLimit::max(MAX_TRANSCRIPTION_UPLOAD_BYTES))
        .with_state(runtime)
}

pub async fn serve(addr: SocketAddr, runtime: ServerRuntime) -> anyhow::Result<()> {
    serve_with_launch_options(addr, runtime, ServerLaunchOptions::default()).await
}

pub async fn serve_with_launch_options(
    addr: SocketAddr,
    runtime: ServerRuntime,
    launch_options: ServerLaunchOptions,
) -> anyhow::Result<()> {
    validate_listen_security(addr, &launch_options)?;
    runtime.validate()?;
    let listener = TcpListener::bind(addr).await?;
    match &launch_options.tls.clone() {
        ServerTlsConfig::Disabled => {
            let app = app_with_runtime_and_distribution_and_launch_options(
                runtime,
                DistributionRuntime::default(),
                launch_options,
            );
            println!("OpenASR server listening on http://{addr}");
            axum::serve(listener, app).await?;
        }
        ServerTlsConfig::SelfSigned { subject_alt_names } => {
            let identity = self_signed_tls_identity(subject_alt_names)?;
            let mut launch_options = launch_options;
            launch_options.auth = launch_options.auth.with_pairing_safety_code(Some(
                pairing_safety_code_for_certificate_fingerprint(&identity.certificate_sha256),
            ));
            let app = app_with_runtime_and_distribution_and_launch_options(
                runtime,
                DistributionRuntime::default(),
                launch_options,
            );
            println!(
                "OpenASR server listening on https://{addr} (certificate sha256:{}, pairing code {})",
                identity.certificate_sha256, identity.pairing_safety_code
            );
            axum::serve(TlsListener::new(listener, identity.acceptor), app).await?;
        }
    }
    Ok(())
}

#[derive(Clone)]
struct TlsIdentity {
    acceptor: TlsAcceptor,
    certificate_sha256: String,
    pairing_safety_code: String,
    #[cfg(test)]
    certificate_der: CertificateDer<'static>,
}

fn self_signed_tls_identity(subject_alt_names: &[String]) -> anyhow::Result<TlsIdentity> {
    let certified = generate_simple_self_signed(subject_alt_names.to_vec())?;
    let certificate_der = CertificateDer::from(certified.serialize_der()?);
    let private_key_der = PrivateKeyDer::from(PrivatePkcs8KeyDer::from(
        certified.serialize_private_key_der(),
    ));
    let certificate_sha256 = certificate_fingerprint_sha256(certificate_der.as_ref());
    let config = rustls::ServerConfig::builder_with_provider(
        rustls::crypto::ring::default_provider().into(),
    )
    .with_safe_default_protocol_versions()?
    .with_no_client_auth()
    .with_single_cert(vec![certificate_der.clone()], private_key_der)?;
    Ok(TlsIdentity {
        acceptor: TlsAcceptor::from(Arc::new(config)),
        pairing_safety_code: pairing_safety_code_for_certificate_fingerprint(&certificate_sha256),
        certificate_sha256,
        #[cfg(test)]
        certificate_der,
    })
}

struct TlsListener {
    listener: TcpListener,
    acceptor: TlsAcceptor,
}

impl TlsListener {
    fn new(listener: TcpListener, acceptor: TlsAcceptor) -> Self {
        Self { listener, acceptor }
    }
}

impl Listener for TlsListener {
    type Io = TlsStream<TcpStream>;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            match self.listener.accept().await {
                Ok((stream, addr)) => match self.acceptor.accept(stream).await {
                    Ok(tls_stream) => return (tls_stream, addr),
                    Err(error) => {
                        eprintln!("OpenASR TLS handshake failed from {addr}: {error}");
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                },
                Err(error) => {
                    eprintln!("OpenASR TLS accept failed: {error}");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        self.listener.local_addr()
    }
}

pub fn server_certificate_fingerprint_for_subject_alt_names(
    subject_alt_names: impl IntoIterator<Item = impl Into<String>>,
) -> anyhow::Result<String> {
    let tls = ServerTlsConfig::self_signed(subject_alt_names);
    let ServerTlsConfig::SelfSigned { subject_alt_names } = tls else {
        unreachable!("self_signed always returns self-signed config")
    };
    Ok(self_signed_tls_identity(&subject_alt_names)?.certificate_sha256)
}

fn validate_listen_security(
    addr: SocketAddr,
    launch_options: &ServerLaunchOptions,
) -> anyhow::Result<()> {
    if addr.ip().is_loopback() {
        return Ok(());
    }
    if !launch_options.tls.is_enabled() {
        // Opt-in escape hatch for container / trusted-network deployments where an
        // OUTER boundary controls exposure (a Docker port-publish, a reverse proxy
        // terminating TLS, etc.). The default stays fail-closed; the desktop
        // pairing flow never sets this and always serves TLS.
        if insecure_non_loopback_bind_allowed() {
            eprintln!(
                "openasr-server: WARNING — binding non-loopback {addr} WITHOUT TLS because OPENASR_ALLOW_INSECURE_NON_LOOPBACK is set. Traffic is UNENCRYPTED; only do this behind a trusted boundary (container / reverse proxy)."
            );
            return Ok(());
        }
        anyhow::bail!(
            "OpenASR HTTP serve is local-only until TLS/WSS remote serving is enabled; bind a loopback address such as 127.0.0.1 instead of {addr} (or, only for a trusted/container deployment, set OPENASR_ALLOW_INSECURE_NON_LOOPBACK=1)"
        );
    }
    if !launch_options.auth.is_enabled() {
        anyhow::bail!(
            "OpenASR remote serve requires device authentication before binding a non-loopback address such as {addr}"
        );
    }
    Ok(())
}

fn insecure_non_loopback_bind_allowed() -> bool {
    std::env::var("OPENASR_ALLOW_INSECURE_NON_LOOPBACK")
        .map(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

#[derive(Debug, Clone, Default)]
pub struct ServerLaunchOptions {
    pub instance_token: Option<String>,
    pub auth: ServerAuth,
    pub tls: ServerTlsConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ServerTlsConfig {
    #[default]
    Disabled,
    SelfSigned {
        subject_alt_names: Vec<String>,
    },
}

impl ServerTlsConfig {
    pub fn self_signed(subject_alt_names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let subject_alt_names = subject_alt_names
            .into_iter()
            .map(Into::into)
            .map(|name| name.trim().to_string())
            .filter(|name| !name.is_empty())
            .collect::<Vec<_>>();
        Self::SelfSigned {
            subject_alt_names: if subject_alt_names.is_empty() {
                vec!["localhost".to_string()]
            } else {
                subject_alt_names
            },
        }
    }

    fn is_enabled(&self) -> bool {
        matches!(self, Self::SelfSigned { .. })
    }
}

#[derive(Debug, Clone)]
pub struct ServerAuth {
    mode: ServerAuthMode,
    pairing: Arc<Mutex<PairingRegistry>>,
    pairing_safety_code: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ServerAuthMode {
    Disabled,
    StaticBearer { token_hashes: HashSet<String> },
    Pairing { admin_token_hash: String },
}

impl Default for ServerAuth {
    fn default() -> Self {
        Self::disabled()
    }
}

impl ServerAuth {
    pub fn disabled() -> Self {
        Self {
            mode: ServerAuthMode::Disabled,
            pairing: Arc::new(Mutex::new(PairingRegistry::default())),
            pairing_safety_code: None,
        }
    }

    pub fn bearer(token: impl Into<String>) -> Self {
        let token = token.into().trim().to_string();
        if token.is_empty() {
            return Self::disabled();
        }
        Self::from_token_hashes([bearer_token_hash(&token)])
    }

    /// Enforces one of a set of pre-hashed bearer tokens (SHA-256 hex, matching
    /// `bearer_token_hash`). Used to wire the CLI's persisted API-key store
    /// (`openasr apikey create/list/revoke`, see `openasr_core::apikeys`) --
    /// only hashes ever cross the store/serve boundary, never plaintext keys.
    /// An empty set of hashes disables auth (loopback stays key-free by
    /// default until at least one key exists).
    pub fn from_token_hashes(token_hashes: impl IntoIterator<Item = String>) -> Self {
        let token_hashes: HashSet<String> = token_hashes
            .into_iter()
            .map(|hash| hash.trim().to_ascii_lowercase())
            .filter(|hash| !hash.is_empty())
            .collect();
        if token_hashes.is_empty() {
            return Self::disabled();
        }
        Self {
            mode: ServerAuthMode::StaticBearer { token_hashes },
            pairing: Arc::new(Mutex::new(PairingRegistry::default())),
            pairing_safety_code: None,
        }
    }

    pub fn pairing(admin_token: impl Into<String>) -> Self {
        Self::pairing_with_safety_code(admin_token, None::<String>)
    }

    pub fn pairing_with_safety_code(
        admin_token: impl Into<String>,
        safety_code: Option<impl Into<String>>,
    ) -> Self {
        let admin_token = admin_token.into().trim().to_string();
        if admin_token.is_empty() {
            return Self::disabled();
        }
        Self {
            mode: ServerAuthMode::Pairing {
                admin_token_hash: bearer_token_hash(&admin_token),
            },
            pairing: Arc::new(Mutex::new(PairingRegistry::default())),
            pairing_safety_code: safety_code
                .map(Into::into)
                .map(|code| code.trim().to_string())
                .filter(|code| !code.is_empty()),
        }
    }

    fn with_pairing_safety_code(mut self, safety_code: Option<String>) -> Self {
        if self.is_pairing_enabled() {
            self.pairing_safety_code = safety_code
                .map(|code| code.trim().to_string())
                .filter(|code| !code.is_empty());
        }
        self
    }

    /// Bind the pairing registry to a persistent JSON store at `path`. Existing
    /// device credentials + revocation state are loaded immediately, and later
    /// approve/revoke mutations are persisted, so paired devices and revocations
    /// survive the remote-server restarts the desktop performs on every daemon
    /// start. Only the credentials map (token *hashes*, never plaintext) is
    /// persisted; pending requests and one-time claims stay in memory. No-op
    /// unless pairing is enabled.
    pub fn with_pairing_store(self, path: PathBuf) -> Self {
        if self.is_pairing_enabled() {
            match load_pairing_credentials(&path) {
                Ok(credentials) => {
                    let mut pairing = self.lock_pairing();
                    for credential in credentials {
                        pairing
                            .credentials
                            .insert(credential.device_id.clone(), credential);
                    }
                    // Enable persistence only after a CLEAN load, so a later
                    // approve/revoke can never overwrite an unreadable file.
                    pairing.store_path = Some(path);
                }
                Err(error) => {
                    // Do NOT silently wipe a corrupt/tampered registry: keep the
                    // file for recovery and refuse to persist over it (store_path
                    // stays None) — degrade to "no paired devices this run" rather
                    // than permanent data loss.
                    eprintln!(
                        "openasr-server: refusing to load/persist pairing registry at {} (continuing with no paired devices): {error}",
                        path.display()
                    );
                }
            }
        }
        self
    }

    /// Lock the pairing registry, recovering from a poisoned mutex instead of
    /// crashing. A poisoned lock means some prior holder panicked while holding
    /// it; the registry is a plain device/request map whose mutations
    /// (insert/remove/get_mut/retain) cannot leave it memory-unsafe, so a single
    /// panicked operation must not permanently take down (DoS) every future
    /// pairing request — the server-wide blast radius `expect` previously had.
    fn lock_pairing(&self) -> std::sync::MutexGuard<'_, PairingRegistry> {
        self.pairing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn is_enabled(&self) -> bool {
        !matches!(self.mode, ServerAuthMode::Disabled)
    }

    fn is_pairing_enabled(&self) -> bool {
        matches!(self.mode, ServerAuthMode::Pairing { .. })
    }

    fn allows_unauthenticated_pair_request(&self, method: &axum::http::Method, path: &str) -> bool {
        if !self.is_pairing_enabled() {
            return false;
        }
        (*method == axum::http::Method::POST && path == "/v1/pairing/requests")
            || (*method == axum::http::Method::GET
                && path.starts_with("/v1/pairing/requests/")
                && path.ends_with("/credential"))
    }

    fn authorizes(&self, headers: &axum::http::HeaderMap) -> bool {
        match &self.mode {
            ServerAuthMode::Disabled => true,
            ServerAuthMode::StaticBearer { token_hashes } => header_bearer_token(headers)
                .is_some_and(|token| token_hashes.contains(&bearer_token_hash(token))),
            ServerAuthMode::Pairing { admin_token_hash } => header_bearer_token(headers)
                .is_some_and(|token| {
                    let token_hash = bearer_token_hash(token);
                    token_hash == *admin_token_hash
                        || self.pairing_authorizes_token_hash(&token_hash)
                }),
        }
    }

    fn authorizes_pairing_admin(&self, headers: &axum::http::HeaderMap) -> bool {
        let ServerAuthMode::Pairing { admin_token_hash } = &self.mode else {
            return false;
        };
        header_bearer_token(headers)
            .is_some_and(|token| bearer_token_hash(token) == *admin_token_hash)
    }

    fn authorizes_remote_compute_client(&self, headers: &axum::http::HeaderMap) -> bool {
        if !self.is_pairing_enabled() {
            return false;
        }
        header_bearer_token(headers)
            .is_some_and(|token| self.pairing_authorizes_token_hash(&bearer_token_hash(token)))
    }

    fn create_pairing_request(
        &self,
        device_name: impl Into<String>,
    ) -> Result<PairingRequestView, PairingError> {
        if !self.is_pairing_enabled() {
            return Err(PairingError::Disabled);
        }
        let device_name = normalized_device_name(device_name.into())?;
        let record = PairingRequestRecord {
            request_id: random_hex(16).map_err(|_| PairingError::Random)?,
            device_name,
            created_at_unix_secs: unix_now_secs(),
            safety_code: self.pairing_safety_code.clone(),
        };
        let view = PairingRequestView::from(&record);
        self.lock_pairing()
            .pending
            .insert(record.request_id.clone(), record);
        Ok(view)
    }

    fn pending_pairing_requests(&self) -> Result<Vec<PairingRequestView>, PairingError> {
        if !self.is_pairing_enabled() {
            return Err(PairingError::Disabled);
        }
        let mut requests: Vec<_> = self
            .lock_pairing()
            .pending
            .values()
            .map(PairingRequestView::from)
            .collect();
        requests.sort_by_key(|request| request.created_at_unix_secs);
        Ok(requests)
    }

    fn approve_pairing_request(
        &self,
        request_id: &str,
    ) -> Result<PairingApprovalView, PairingError> {
        if !self.is_pairing_enabled() {
            return Err(PairingError::Disabled);
        }
        let request_id = normalize_pairing_request_id(request_id)?;
        let mut pairing = self.lock_pairing();
        let request = pairing
            .pending
            .remove(&request_id)
            .ok_or(PairingError::NotFound)?;
        let bearer_token = format!("oasr_{}", random_hex(32).map_err(|_| PairingError::Random)?);
        let credential = DeviceCredentialRecord {
            device_id: random_hex(12).map_err(|_| PairingError::Random)?,
            device_name: request.device_name,
            token_hash: bearer_token_hash(&bearer_token),
            issued_at_unix_secs: unix_now_secs(),
            last_seen_unix_secs: None,
            revoked: false,
        };
        let claim = PairingCredentialView::from_record(&credential, bearer_token);
        let view = PairingApprovalView::from_record(&credential);
        pairing.credential_claims.insert(request_id, claim);
        pairing
            .credentials
            .insert(credential.device_id.clone(), credential);
        persist_pairing_credentials_locked(&pairing);
        Ok(view)
    }

    fn reject_pairing_request(&self, request_id: &str) -> Result<bool, PairingError> {
        if !self.is_pairing_enabled() {
            return Err(PairingError::Disabled);
        }
        let request_id = normalize_pairing_request_id(request_id)?;
        let mut pairing = self.lock_pairing();
        Ok(pairing.pending.remove(&request_id).is_some())
    }

    fn paired_devices(&self) -> Result<Vec<PairingDeviceView>, PairingError> {
        if !self.is_pairing_enabled() {
            return Err(PairingError::Disabled);
        }
        let mut devices: Vec<_> = self
            .lock_pairing()
            .credentials
            .values()
            .filter(|credential| !credential.revoked)
            .map(PairingDeviceView::from)
            .collect();
        devices.sort_by_key(|device| device.issued_at_unix_secs);
        Ok(devices)
    }

    fn pairing_credential(&self, request_id: &str) -> Result<PairingCredentialState, PairingError> {
        if !self.is_pairing_enabled() {
            return Err(PairingError::Disabled);
        }
        let request_id = normalize_pairing_request_id(request_id)?;
        let mut pairing = self.lock_pairing();
        if pairing.pending.contains_key(&request_id) {
            return Ok(PairingCredentialState::Pending);
        }
        // One-time: consume the plaintext claim on first successful fetch so the
        // bearer token cannot be re-fetched/replayed and does not linger in
        // server memory after delivery.
        if let Some(claim) = pairing.credential_claims.remove(&request_id) {
            return Ok(PairingCredentialState::Ready(claim));
        }
        Err(PairingError::NotFound)
    }

    fn revoke_pairing_credential(&self, device_id: &str) -> Result<bool, PairingError> {
        if !self.is_pairing_enabled() {
            return Err(PairingError::Disabled);
        }
        let device_id = normalize_pairing_device_id(device_id)?;
        let mut pairing = self.lock_pairing();
        let Some(credential) = pairing.credentials.get_mut(&device_id) else {
            return Ok(false);
        };
        credential.revoked = true;
        pairing
            .credential_claims
            .retain(|_, claim| claim.device_id != device_id);
        persist_pairing_credentials_locked(&pairing);
        Ok(true)
    }

    fn pairing_authorizes_token_hash(&self, token_hash: &str) -> bool {
        let mut pairing = self.lock_pairing();
        let Some(credential) = pairing
            .credentials
            .values_mut()
            .find(|credential| !credential.revoked && credential.token_hash == token_hash)
        else {
            return false;
        };
        credential.last_seen_unix_secs = Some(unix_now_secs());
        true
    }
}

fn header_bearer_token(headers: &axum::http::HeaderMap) -> Option<&str> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
}

fn bearer_token_hash(token: &str) -> String {
    let digest = Sha256::digest(token.as_bytes());
    hex_encode(&digest)
}

fn random_hex(byte_count: usize) -> Result<String, getrandom::Error> {
    let mut bytes = vec![0; byte_count];
    getrandom::fill(&mut bytes)?;
    Ok(hex_encode(&bytes))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ServerHealthIdentity {
    server_version: &'static str,
    pid: u32,
    instance_token: Option<String>,
}

impl ServerHealthIdentity {
    fn from_launch_options(launch_options: ServerLaunchOptions) -> Self {
        Self {
            server_version: env!("CARGO_PKG_VERSION"),
            pid: std::process::id(),
            instance_token: resolve_instance_token(launch_options.instance_token),
        }
    }
}

fn resolve_instance_token(launch_option: Option<String>) -> Option<String> {
    env::var(SERVER_INSTANCE_TOKEN_ENV)
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| launch_option.filter(|value| !value.is_empty()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerRuntime {
    pub backend: BackendKind,
    pub ffmpeg_bin: Option<std::path::PathBuf>,
    /// Whether `ffmpeg_bin` came from an explicit operator choice (CLI flag,
    /// env var, or config) rather than PATH auto-discovery -- see
    /// `AudioPreparationOptions::with_ffmpeg_bin_explicit`. Only an explicit
    /// choice skips the in-process symphonia decode path.
    pub ffmpeg_bin_explicit: bool,
    pub model_pack_path: Option<std::path::PathBuf>,
}

impl Default for ServerRuntime {
    fn default() -> Self {
        Self {
            backend: BackendKind::Mock,
            ffmpeg_bin: None,
            ffmpeg_bin_explicit: false,
            model_pack_path: None,
        }
    }
}

impl ServerRuntime {
    /// Validates the runtime is *safe to serve with*, not that a model is
    /// installed: no model bound at all (`model_pack_path: None`) is a normal
    /// state for a fresh install with zero pulled models, and the daemon must
    /// still start and answer `/health` in that state. Only a `model_pack_path`
    /// that is actually set but fails to validate (bad path, corrupt/foreign
    /// pack, ...) is a startup error -- that is a misconfiguration, not "no
    /// model yet". Requests that need a model fail closed at request time
    /// instead (see `validate_native_request_model`).
    pub fn validate(&self) -> Result<(), openasr_core::BackendError> {
        match self.backend {
            BackendKind::Mock => Ok(()),
            BackendKind::Native => {
                let Some(model_pack_path) = self.model_pack_path.as_deref() else {
                    return Ok(());
                };
                let pack_root =
                    openasr_core::validate_local_native_model_pack_path(model_pack_path)?;
                let _ = validate_native_runtime_pack(&pack_root)?;
                Ok(())
            }
        }
    }

    /// Whether a native model pack is currently bound (surfaced by `/health` so
    /// clients can distinguish "daemon not reachable" from "daemon ready, no
    /// model installed" without racing a separate models-list call).
    fn has_model_bound(&self) -> bool {
        match self.backend {
            BackendKind::Mock => true,
            BackendKind::Native => self.model_pack_path.is_some(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributionRuntime {
    pub openasr_home: Option<PathBuf>,
    pub catalog_url: Option<String>,
}

impl Default for DistributionRuntime {
    fn default() -> Self {
        Self {
            openasr_home: None,
            catalog_url: env::var("OPENASR_CATALOG_URL")
                .ok()
                .filter(|value| !value.trim().is_empty()),
        }
    }
}

#[derive(Clone)]
pub(crate) struct DistributionContext {
    runtime: DistributionRuntime,
    jobs: Arc<DistributionJobs>,
}

impl DistributionContext {
    fn new(runtime: DistributionRuntime) -> Self {
        Self {
            jobs: Arc::new(DistributionJobs::new(load_persisted_pull_jobs(&runtime))),
            runtime,
        }
    }

    fn openasr_home(&self) -> Result<PathBuf, ApiError> {
        self.runtime
            .openasr_home
            .clone()
            .map(Ok)
            .unwrap_or_else(openasr_home)
            .map_err(ApiError::Home)
    }

    fn catalog_url(&self) -> Option<&str> {
        self.runtime.catalog_url.as_deref()
    }

    fn next_job_id(&self) -> String {
        let seq = self.jobs.next.fetch_add(1, Ordering::Relaxed);
        let now_ms = unix_millis_now();
        format!("pull-{now_ms}-{seq}")
    }

    fn pulls_dir(&self) -> Result<PathBuf, ApiError> {
        Ok(self.openasr_home()?.join("pulls"))
    }

    fn insert_job(&self, snapshot: PullJobSnapshot) -> Result<(), ApiError> {
        self.persist_snapshot(&snapshot)?;
        let (sender, _receiver) = watch::channel(snapshot.clone());
        self.jobs
            .snapshots
            .lock()
            .expect("pull job snapshots mutex poisoned")
            .insert(snapshot.job_id.clone(), snapshot.clone());
        self.jobs
            .watchers
            .lock()
            .expect("pull job watchers mutex poisoned")
            .insert(snapshot.job_id.clone(), sender);
        Ok(())
    }

    fn update_job(
        &self,
        job_id: &str,
        update: impl FnOnce(&mut PullJobSnapshot),
    ) -> Result<bool, ApiError> {
        let mut snapshots = self
            .jobs
            .snapshots
            .lock()
            .expect("pull job snapshots mutex poisoned");
        let Some(current) = snapshots.get(job_id) else {
            return Ok(false);
        };
        let mut next = current.clone();
        update(&mut next);
        self.persist_snapshot(&next)?;
        snapshots.insert(job_id.to_string(), next.clone());
        drop(snapshots);
        self.notify_job_snapshot(&next);
        Ok(true)
    }

    fn update_job_best_effort(&self, job_id: &str, update: impl FnOnce(&mut PullJobSnapshot)) {
        if let Err(error) = self.update_job(job_id, update) {
            eprintln!("OpenASR warning: could not persist pull job '{job_id}': {error}");
        }
    }

    fn update_job_in_memory(
        &self,
        job_id: &str,
        update: impl FnOnce(&mut PullJobSnapshot),
    ) -> bool {
        let mut snapshots = self
            .jobs
            .snapshots
            .lock()
            .expect("pull job snapshots mutex poisoned");
        let Some(current) = snapshots.get(job_id) else {
            return false;
        };
        let mut next = current.clone();
        update(&mut next);
        snapshots.insert(job_id.to_string(), next.clone());
        drop(snapshots);
        self.notify_job_snapshot(&next);
        true
    }

    fn notify_job_snapshot(&self, snapshot: &PullJobSnapshot) {
        if let Some(sender) = self
            .jobs
            .watchers
            .lock()
            .expect("pull job watchers mutex poisoned")
            .get(&snapshot.job_id)
        {
            sender.send_replace(snapshot.clone());
        }
    }

    fn snapshot(&self, job_id: &str) -> Option<PullJobSnapshot> {
        self.jobs
            .snapshots
            .lock()
            .expect("pull job snapshots mutex poisoned")
            .get(job_id)
            .cloned()
    }

    fn subscribe_job(&self, job_id: &str) -> Option<watch::Receiver<PullJobSnapshot>> {
        self.jobs
            .watchers
            .lock()
            .expect("pull job watchers mutex poisoned")
            .get(job_id)
            .map(watch::Sender::subscribe)
    }

    fn nonterminal_snapshot_for_pull(
        &self,
        resolved: &ResolvedCatalogPull,
    ) -> Option<PullJobSnapshot> {
        self.jobs
            .snapshots
            .lock()
            .expect("pull job snapshots mutex poisoned")
            .values()
            .filter(|snapshot| {
                !snapshot.state.is_terminal()
                    && snapshot.model_id == resolved.model_id
                    && snapshot.quant == resolved.quant
                    && snapshot.pull == resolved.pull
                    && snapshot
                        .resolved
                        .as_ref()
                        .is_none_or(|spec| spec.filename == resolved.filename)
            })
            .cloned()
            .max_by(|left, right| left.job_id.cmp(&right.job_id))
    }

    fn restart_resumable_snapshots(&self) -> Vec<PullJobSnapshot> {
        self.jobs
            .snapshots
            .lock()
            .expect("pull job snapshots mutex poisoned")
            .values()
            .filter(|snapshot| snapshot.state.is_restart_resumable())
            .cloned()
            .collect()
    }

    fn ensure_restart_resumes_started(&self) {
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        if self
            .jobs
            .restart_resumes_started
            .swap(true, Ordering::AcqRel)
        {
            return;
        }
        for snapshot in self.restart_resumable_snapshots() {
            self.spawn_restart_resume_job(snapshot);
        }
    }

    fn spawn_restart_resume_job(&self, snapshot: PullJobSnapshot) {
        let job_id = snapshot.job_id.clone();
        let source_path = snapshot.source_path.clone();
        let distribution = self.clone();
        match resolved_pull_from_snapshot(&snapshot) {
            Ok(resolved) => match distribution.openasr_home() {
                Ok(home) => {
                    let cancel_flag = Arc::new(AtomicBool::new(false));
                    let pause_flag = Arc::new(AtomicBool::new(false));
                    spawn_pull_job(
                        distribution,
                        job_id,
                        home,
                        resolved,
                        source_path,
                        cancel_flag,
                        pause_flag,
                    );
                }
                Err(error) => fail_restart_resume(&distribution, &job_id, error),
            },
            Err(error) => fail_restart_resume(&distribution, &job_id, error),
        }
    }

    fn cancel_job(&self, job_id: &str) -> bool {
        self.jobs
            .active
            .lock()
            .expect("active pull job registry mutex poisoned")
            .get(job_id)
            .map(|job| {
                job.cancel_flag.store(true, Ordering::SeqCst);
            })
            .is_some()
    }

    fn pause_job(&self, job_id: &str) -> bool {
        self.jobs
            .active
            .lock()
            .expect("active pull job registry mutex poisoned")
            .get(job_id)
            .map(|job| {
                job.pause_flag.store(true, Ordering::SeqCst);
            })
            .is_some()
    }

    fn register_active_job(
        &self,
        job_id: &str,
        cancel_flag: Arc<AtomicBool>,
        pause_flag: Arc<AtomicBool>,
    ) {
        self.jobs
            .active
            .lock()
            .expect("active pull job registry mutex poisoned")
            .insert(
                job_id.to_string(),
                ActivePullJob {
                    cancel_flag,
                    pause_flag,
                },
            );
    }

    fn clear_active_job(&self, job_id: &str) {
        self.jobs
            .active
            .lock()
            .expect("active pull job registry mutex poisoned")
            .remove(job_id);
    }

    fn persist_snapshot(&self, snapshot: &PullJobSnapshot) -> Result<(), ApiError> {
        let pulls_dir = self.pulls_dir()?;
        fs::create_dir_all(&pulls_dir).map_err(|error| {
            ApiError::JobStore(format!(
                "Could not create pull job directory '{}': {error}",
                pulls_dir.display()
            ))
        })?;
        let path = pulls_dir.join(format!("{}.json", snapshot.job_id));
        let contents = serde_json::to_vec_pretty(snapshot).map_err(|error| {
            ApiError::JobStore(format!(
                "Could not serialize pull job '{}': {error}",
                snapshot.job_id
            ))
        })?;
        write_bytes_atomically(&path, &contents).map_err(|error| {
            ApiError::JobStore(format!(
                "Could not persist pull job '{}': {error}",
                path.display()
            ))
        })?;
        Ok(())
    }
}

fn write_bytes_atomically(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let temp_path = server_atomic_temp_path(path);
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)?;
        file.write_all(contents)?;
        file.flush()?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp_path, path)?;
        best_effort_sync_parent_dir(path);
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

fn server_atomic_temp_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("openasr.tmp");
    let sequence = ATOMIC_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    path.with_file_name(format!(
        ".{file_name}.{}.{}.{}.tmp",
        std::process::id(),
        now,
        sequence
    ))
}

fn best_effort_sync_parent_dir(path: &Path) {
    let Some(parent) = path.parent() else {
        return;
    };
    let _ = fs::File::open(parent).and_then(|file| file.sync_all());
}

struct DistributionJobs {
    next: AtomicU64,
    restart_resumes_started: AtomicBool,
    snapshots: Mutex<HashMap<String, PullJobSnapshot>>,
    watchers: Mutex<HashMap<String, watch::Sender<PullJobSnapshot>>>,
    active: Mutex<HashMap<String, ActivePullJob>>,
}

struct ActivePullJob {
    cancel_flag: Arc<AtomicBool>,
    pause_flag: Arc<AtomicBool>,
}

impl DistributionJobs {
    fn new(snapshots: HashMap<String, PullJobSnapshot>) -> Self {
        let watchers = snapshots
            .iter()
            .map(|(job_id, snapshot)| {
                let (sender, _receiver) = watch::channel(snapshot.clone());
                (job_id.clone(), sender)
            })
            .collect();
        Self {
            next: AtomicU64::new(0),
            restart_resumes_started: AtomicBool::new(false),
            snapshots: Mutex::new(snapshots),
            watchers: Mutex::new(watchers),
            active: Mutex::new(HashMap::new()),
        }
    }
}

fn load_persisted_pull_jobs(runtime: &DistributionRuntime) -> HashMap<String, PullJobSnapshot> {
    let home = runtime
        .openasr_home
        .clone()
        .map(Ok)
        .unwrap_or_else(openasr_home);
    let Ok(home) = home else {
        return HashMap::new();
    };
    let pulls_dir = home.join("pulls");
    let Ok(entries) = fs::read_dir(&pulls_dir) else {
        return HashMap::new();
    };
    let mut jobs = HashMap::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(contents) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(mut snapshot) = serde_json::from_str::<PullJobSnapshot>(&contents) else {
            continue;
        };
        if snapshot.state.is_restart_resumable() {
            if snapshot.resolved.is_some() {
                snapshot.state = PullJobState::Queued;
                snapshot.speed_bps = None;
                snapshot.eta_s = None;
                snapshot.error = Some(
                    "OpenASR daemon restarted before this pull completed; resume queued."
                        .to_string(),
                );
            } else {
                snapshot.state = PullJobState::Failed;
                snapshot.speed_bps = None;
                snapshot.eta_s = None;
                snapshot.error = Some(
                    "OpenASR daemon restarted before this pull completed, but the persisted job does not contain an immutable resolved model pack spec. Refusing to re-resolve the mutable catalog."
                        .to_string(),
                );
            }
        }
        jobs.insert(snapshot.job_id.clone(), snapshot);
    }
    jobs
}

async fn health(
    State(runtime): State<ServerRuntime>,
    Extension(identity): Extension<ServerHealthIdentity>,
) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        server_version: identity.server_version,
        pid: identity.pid,
        instance_token: identity.instance_token.clone(),
        model_installed: runtime.has_model_bound(),
    })
}

async fn models(State(runtime): State<ServerRuntime>) -> Result<Json<ModelsResponse>, ApiError> {
    let ids: Vec<String> = match runtime.backend {
        BackendKind::Mock => load_registry(default_registry_dir())
            .map_err(ApiError::Registry)?
            .into_iter()
            .map(|card| card.id)
            .collect(),
        BackendKind::Native => {
            // No model bound is a normal fresh-install state, not an error:
            // report an empty model list rather than fail-closed here (the
            // transcription path is the fail-closed boundary for "no model").
            match runtime.model_pack_path.as_deref() {
                None => Vec::new(),
                Some(model_pack_path) => {
                    let pack_root =
                        openasr_core::validate_local_native_model_pack_path(model_pack_path)
                            .map_err(ApiError::Backend)?;
                    let identity =
                        validate_native_runtime_pack(&pack_root).map_err(ApiError::Backend)?;
                    vec![identity.model_id]
                }
            }
        }
    };
    Ok(Json(ModelsResponse {
        object: "list",
        data: ids
            .into_iter()
            .map(|id| ModelResponse {
                id,
                object: "model",
                owned_by: "openasr",
            })
            .collect(),
    }))
}

async fn catalog(
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<openasr_core::ModelCatalog>, ApiError> {
    distribution.ensure_restart_resumes_started();
    let home = distribution.openasr_home()?;
    let catalog =
        load_model_catalog(distribution.catalog_url(), &home).map_err(ApiError::Catalog)?;
    Ok(Json(catalog))
}

async fn capabilities(
    State(runtime): State<ServerRuntime>,
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<CapabilitiesResponse>, ApiError> {
    let transcription = if runtime.backend == BackendKind::Native {
        runtime
            .model_pack_path
            .as_deref()
            .map(native_runtime_transcription_capabilities_for_path)
            .unwrap_or_else(|| TranscriptionBackendCapabilities::for_backend_kind(runtime.backend))
    } else {
        TranscriptionBackendCapabilities::for_backend_kind(runtime.backend)
    };
    Ok(Json(CapabilitiesResponse {
        object: "capabilities",
        transcription,
        realtime: realtime_capabilities_for_runtime_and_distribution(&runtime, &distribution),
    }))
}

pub(crate) fn realtime_capabilities_for_runtime(
    runtime: &ServerRuntime,
) -> RealtimeBackendCapabilities {
    let mut capabilities = if runtime.backend == BackendKind::Native {
        runtime
            .model_pack_path
            .as_deref()
            .map(cached_native_realtime_capabilities_for_path)
            .unwrap_or_else(|| RealtimeBackendCapabilities::for_backend_kind(runtime.backend))
    } else {
        RealtimeBackendCapabilities::for_backend_kind(runtime.backend)
    };
    // Model-pack capabilities are immutable for the daemon's lifetime (and so
    // cacheable), but diarization also depends on whether the active embedder
    // pack is installed — re-derive it fresh on every ask.
    capabilities.diarization =
        openasr_core::realtime::realtime_diarization_capability(capabilities.mode);
    capabilities
}

pub(crate) fn realtime_capabilities_for_runtime_and_distribution(
    runtime: &ServerRuntime,
    distribution: &DistributionContext,
) -> RealtimeBackendCapabilities {
    let mut capabilities = realtime_capabilities_for_runtime(runtime);
    capabilities.translation = translation_capability_for_distribution(distribution);
    capabilities
}

/// Deriving realtime capabilities reads GGUF metadata from disk; every session
/// asks 2-3 times, so memoize per pack path. Installed packs are
/// content-immutable (a model switch relaunches the daemon with a new
/// `--model-pack`), so path-keyed caching is safe for the daemon's lifetime.
fn cached_native_realtime_capabilities_for_path(path: &Path) -> RealtimeBackendCapabilities {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<PathBuf, RealtimeBackendCapabilities>>,
    > = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Ok(cache) = cache.lock()
        && let Some(capabilities) = cache.get(path)
    {
        return *capabilities;
    }
    let capabilities = native_runtime_realtime_capabilities_for_path(path);
    if let Ok(mut cache) = cache.lock() {
        cache.insert(path.to_path_buf(), capabilities);
    }
    capabilities
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    server_version: &'static str,
    pid: u32,
    instance_token: Option<String>,
    /// Whether a model is currently bound and ready to serve transcription
    /// requests. `false` on a fresh install with zero pulled models -- the
    /// daemon is still up and healthy, it just has nothing to transcribe with
    /// yet. Clients should treat this as "go install a model", not "daemon
    /// unreachable".
    model_installed: bool,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelResponse>,
}

#[derive(Serialize)]
struct ModelResponse {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct CapabilitiesResponse {
    object: &'static str,
    transcription: TranscriptionBackendCapabilities,
    realtime: RealtimeBackendCapabilities,
}

#[derive(Serialize)]
pub(crate) struct HistoryListResponse {
    pub(crate) object: &'static str,
    pub(crate) data: Vec<DaemonHistoryEntry>,
    /// Total rows matching the filter, ignoring `limit`/`offset` -- lets
    /// clients render pagination without a second request. Additive: absent
    /// in the pre-SQLite contract, so existing consumers that only read
    /// `data` are unaffected.
    pub(crate) total: usize,
    pub(crate) limit: usize,
    pub(crate) offset: usize,
}

#[derive(Serialize)]
pub(crate) struct DeleteHistoryResponse {
    pub(crate) deleted: bool,
    pub(crate) id: String,
}

#[derive(Serialize)]
struct LocalModelsResponse {
    object: &'static str,
    data: Vec<LocalModelResponse>,
}

#[derive(Serialize)]
struct LocalModelResponse {
    #[serde(flatten)]
    pack: InstalledPack,
    is_default: bool,
}

#[derive(Serialize)]
struct DeleteModelResponse {
    deleted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pack: Option<InstalledPack>,
}

#[derive(Debug, Deserialize)]
struct ImportLocalModelRequest {
    path: PathBuf,
}

#[derive(Serialize)]
struct ImportLocalModelResponse {
    object: &'static str,
    installed: InstalledPack,
}

#[derive(Debug, Deserialize)]
struct StartPullRequest {
    #[serde(default)]
    quant: Option<String>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    from: Option<PathBuf>,
    #[serde(default)]
    accept_license: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct SetDefaultRequest {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    quant: Option<String>,
    #[serde(default)]
    pull: Option<String>,
}

impl SetDefaultRequest {
    fn reference(&self) -> Result<String, ApiError> {
        if let Some(pull) = self
            .pull
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Ok(pull.to_string());
        }
        let id = self
            .id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                ApiError::BadRequest(
                    "Set-default request requires either 'pull' or 'id'.".to_string(),
                )
            })?;
        Ok(self
            .quant
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map_or_else(|| id.to_string(), |quant| format!("{id}:{quant}")))
    }

    fn is_auto_request(&self) -> bool {
        self.pull
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .is_none()
            && self
                .quant
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .is_none()
            && self
                .id
                .as_deref()
                .map(str::trim)
                .is_some_and(|value| !value.is_empty() && !value.contains(':'))
    }

    fn quant_preference_for_pack(&self, pack: &InstalledPack) -> QuantPreference {
        if self.is_auto_request() {
            QuantPreference::Auto
        } else {
            QuantPreference::pinned(&pack.quant)
        }
    }
}

#[derive(Serialize)]
struct DefaultModelResponse {
    object: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    default_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    default_pull: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pack: Option<InstalledPack>,
}

#[derive(Debug)]
pub(crate) enum ApiError {
    BadRequest(String),
    NotFound(String),
    Catalog(CatalogError),
    Config(openasr_core::ConfigError),
    Format(String),
    Home(OpenAsrHomeError),
    History(DaemonHistoryStoreError),
    JobStore(String),
    MultipartRejection(MultipartRejection),
    Multipart(axum::extract::multipart::MultipartError),
    AudioPreparation(AudioPreparationError),
    Backend(openasr_core::BackendError),
    BackendJoin(tokio::task::JoinError),
    Pull(PullError),
    Registry(openasr_core::RegistryError),
    Serialize(serde_json::Error),
    TempFile(std::io::Error),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadRequest(message) | Self::Format(message) => f.write_str(message),
            Self::NotFound(message) => f.write_str(message),
            Self::Catalog(error) => write!(f, "Could not load model catalog: {error}"),
            Self::Config(error) => write!(f, "Could not read or update OpenASR config: {error}"),
            Self::Home(error) => write!(f, "Could not resolve OpenASR home: {error}"),
            Self::History(error) => write!(f, "Could not update transcription history: {error}"),
            Self::JobStore(message) => f.write_str(message),
            Self::MultipartRejection(error) => write!(f, "Could not read multipart form: {error}"),
            Self::Multipart(error) => write!(f, "{}", multipart_error_message(error)),
            Self::AudioPreparation(error) => {
                write!(
                    f,
                    "Could not prepare uploaded audio for transcription: {error}"
                )
            }
            Self::Backend(error) => write!(f, "Could not transcribe audio: {error}"),
            Self::BackendJoin(error) => {
                write!(
                    f,
                    "Could not transcribe audio: backend task failed: {error}"
                )
            }
            Self::Pull(error) => write!(f, "Could not pull model pack: {error}"),
            Self::Registry(error) => write!(f, "Could not load model registry: {error}"),
            Self::Serialize(error) => write!(f, "Could not render transcription response: {error}"),
            Self::TempFile(error) => {
                write!(
                    f,
                    "Could not prepare uploaded audio for transcription: {error}"
                )
            }
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::BadRequest(message) | Self::Format(message) => (StatusCode::BAD_REQUEST, message),
            Self::NotFound(message) => (StatusCode::NOT_FOUND, message),
            Self::Catalog(error) => {
                let status = if matches!(
                    &error,
                    CatalogError::InvalidPullReference(_)
                        | CatalogError::UnknownModel { .. }
                        | CatalogError::AmbiguousModelRef { .. }
                        | CatalogError::UnknownQuant { .. }
                        | CatalogError::ConflictingQuant { .. }
                ) {
                    StatusCode::BAD_REQUEST
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                };
                (status, format!("Could not load model catalog: {error}"))
            }
            Self::Config(error) => (
                config_error_status(&error),
                format!("Could not read or update OpenASR config: {error}"),
            ),
            Self::Home(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Could not resolve OpenASR home: {error}"),
            ),
            Self::History(error) => {
                let status = if matches!(
                    error,
                    DaemonHistoryStoreError::InvalidId { .. }
                        | DaemonHistoryStoreError::InvalidRecord { .. }
                ) {
                    StatusCode::BAD_REQUEST
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                };
                (
                    status,
                    format!("Could not update transcription history: {error}"),
                )
            }
            Self::JobStore(message) => (StatusCode::INTERNAL_SERVER_ERROR, message),
            Self::MultipartRejection(error) => (
                StatusCode::BAD_REQUEST,
                format!("Could not read multipart form: {error}"),
            ),
            Self::Multipart(error) => {
                let status = error.status();
                let message = multipart_error_message(&error);
                (status, message)
            }
            Self::AudioPreparation(error) => (
                StatusCode::BAD_REQUEST,
                format!("Could not prepare uploaded audio for transcription: {error}"),
            ),
            Self::Backend(error) => {
                let status = match &error {
                    openasr_core::BackendError::DiarizationNotSupported { .. }
                    | openasr_core::BackendError::DiarizeSpeakersRequiresDiarization
                    | openasr_core::BackendError::PhraseBiasNotSupported { .. }
                    | openasr_core::BackendError::AdapterNotSupported { .. }
                    | openasr_core::BackendError::PhraseBiasUnsupportedByModel { .. }
                    | openasr_core::BackendError::RequestOptionUnsupportedByModel { .. }
                    | openasr_core::BackendError::NativeUnsupportedInputFormat { .. }
                    | openasr_core::BackendError::NativeModelSelectionMismatch { .. }
                    | openasr_core::BackendError::NativeModelPackPathRequired
                    | openasr_core::BackendError::NativeModelPackPathRejected { .. }
                    | openasr_core::BackendError::NativeFailClosed { .. } => {
                        // Native backend "fail-closed" is a deliberate, client-facing
                        // refusal (unexecutable/unsupported request or unusable pack),
                        // not a server fault — genuine internal panics surface via
                        // BackendJoin. Classify it as 400, matching the fail-closed
                        // contract the api.rs native tests assert.
                        StatusCode::BAD_REQUEST
                    }
                    openasr_core::BackendError::ServeBatchUnavailable { retryable, .. } => {
                        if *retryable {
                            StatusCode::TOO_MANY_REQUESTS
                        } else {
                            StatusCode::SERVICE_UNAVAILABLE
                        }
                    }
                };
                (status, format!("Could not transcribe audio: {error}"))
            }
            Self::BackendJoin(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Could not transcribe audio: backend task failed: {error}"),
            ),
            Self::Pull(error) => {
                let status = match &error {
                    PullError::InvalidTarget { .. }
                    | PullError::NonHttpsUrl { .. }
                    | PullError::NotInstalled { .. } => StatusCode::BAD_REQUEST,
                    PullError::LockHeld { .. } => StatusCode::CONFLICT,
                    PullError::InsufficientSpace { .. } => StatusCode::INSUFFICIENT_STORAGE,
                    _ => StatusCode::INTERNAL_SERVER_ERROR,
                };
                (status, format!("Could not pull model pack: {error}"))
            }
            Self::Registry(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Could not load model registry: {error}"),
            ),
            Self::Serialize(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Could not render transcription response: {error}"),
            ),
            Self::TempFile(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Could not prepare uploaded audio for transcription: {error}"),
            ),
        };

        // Log every failed request to stderr (captured in daemon.log by the
        // desktop sidecar) with the status and message that also went out in
        // the HTTP response body. Before this, a failing request left no
        // trace in the daemon log at all -- only the one-line startup banner
        // -- making field reports impossible to diagnose from logs alone.
        eprintln!("openasr-server: request failed status={status} message={message}");

        (
            status,
            Json(ErrorResponse {
                error: ErrorBody {
                    message,
                    r#type: match status {
                        StatusCode::BAD_REQUEST => "invalid_request_error",
                        StatusCode::CONFLICT => "conflict_error",
                        StatusCode::NOT_FOUND => "not_found_error",
                        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
                        StatusCode::SERVICE_UNAVAILABLE => "service_unavailable_error",
                        _ => "openasr_error",
                    },
                },
            }),
        )
            .into_response()
    }
}

fn config_error_status(error: &openasr_core::ConfigError) -> StatusCode {
    match error {
        openasr_core::ConfigError::UnknownKey(_)
        | openasr_core::ConfigError::UnsupportedBackend(_)
        | openasr_core::ConfigError::UnsupportedDownloadSource(_)
        | openasr_core::ConfigError::UnsupportedDefaultBackend(_)
        | openasr_core::ConfigError::UnsupportedPreferencesVersion { .. }
        | openasr_core::ConfigError::InvalidPreference { .. }
        | openasr_core::ConfigError::UnknownModel(_)
        | openasr_core::ConfigError::ModelResolution(_)
        | openasr_core::ConfigError::RuntimeModelResolution(_) => StatusCode::BAD_REQUEST,
        openasr_core::ConfigError::ReadConfig { .. }
        | openasr_core::ConfigError::ParseConfig { .. }
        | openasr_core::ConfigError::CreateHome { .. }
        | openasr_core::ConfigError::SerializeConfig(_)
        | openasr_core::ConfigError::WriteConfig { .. } => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// `axum::extract::multipart::MultipartError`'s `Display` always renders the
/// generic "Error parsing `multipart/form-data` request", even when the
/// underlying cause is the upload exceeding [`MAX_TRANSCRIPTION_UPLOAD_BYTES`]
/// (`DefaultBodyLimit`). That generic text reads like a malformed-body/encoding
/// bug when the real, actionable cause is "file too large". `MultipartError`'s
/// `status`/`body_text` already classify the underlying `multer` error
/// correctly, so use those instead of the raw `Display` for the client-facing
/// message.
fn multipart_error_message(error: &axum::extract::multipart::MultipartError) -> String {
    if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
        format!(
            "Uploaded file is too large. The daemon accepts uploads up to {} MB; \
             split the recording or use a smaller/compressed file.",
            MAX_TRANSCRIPTION_UPLOAD_BYTES / (1024 * 1024)
        )
    } else {
        format!("Could not read multipart form: {error}")
    }
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
    r#type: &'static str,
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
