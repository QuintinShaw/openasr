mod idle_activity;
mod realtime;
mod routes;

pub(crate) use idle_activity::{NativeActivityGuard, spawn_idle_unload_reaper};
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
pub use openasr_core::pairing_safety_code_for_certificate_fingerprint;
use openasr_core::realtime::history::{DaemonHistoryEntry, DaemonHistoryStoreError};
use openasr_core::{
    AudioPreparationError, BackendKind, CatalogError, CatalogMirror, CatalogPullRequest,
    InstalledPack, LaunchPackRequest, LicenseClass, ModelCatalog, OpenAsrHomeError, PullError,
    PullModelPackRequest, PullProgress, QuantPreference, RealtimeBackendCapabilities,
    ResolvedCatalogPull, certificate_fingerprint_sha256, host_quant_recommendation_profile,
    install_catalog_model_pack_from_path, install_model_pack_from_path, list_installed_packs,
    load_local_catalog_file_with_identity, load_model_catalog,
    native_runtime_realtime_capabilities_for_path,
    native_runtime_transcription_capabilities_for_path, openasr_home, remove_model_pack,
    resolve_catalog_pull, resolve_installed_pack_reference,
    resolve_installed_pack_reference_with_catalog, resolve_launch_pack, resolve_runtime_catalog,
    runtime_registry,
};
use rcgen::{Certificate, CertificateParams};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use time::OffsetDateTime;
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{Semaphore, watch},
    task,
};
use tokio_rustls::{TlsAcceptor, server::TlsStream};

// The `file` field of `/v1/audio/transcriptions` (and `/v1/audio/translations`)
// streams straight to a temp file in fixed-size chunks (see
// `write_upload_temp_file_streaming` in routes/transcription.rs), so memory use
// is O(chunk), not O(file); this ceiling is now just a finite abuse cap for a
// single request, not a memory guard. 2 GiB comfortably covers multi-hour
// uncompressed meeting recordings.
const MAX_TRANSCRIPTION_UPLOAD_BYTES: usize = 2 * 1024 * 1024 * 1024;
// Minimum free space the upload's temp volume must retain while an upload is
// streaming to disk. Checked periodically (see `DISK_SPACE_CHECK_INTERVAL_BYTES`)
// so a huge upload fails closed with a clear 507 instead of filling the disk;
// mirrors the headroom check `pull.rs` already does before downloading a model
// pack, via the shared `available_disk_space_bytes` probe.
const MIN_FREE_DISK_HEADROOM_BYTES: u64 = 256 * 1024 * 1024;
// How often (in bytes written) to re-check free disk space while streaming an
// upload to its temp file: frequent enough to catch a disk filling up
// mid-upload, infrequent enough that the statvfs probe doesn't dominate I/O.
const DISK_SPACE_CHECK_INTERVAL_BYTES: u64 = 8 * 1024 * 1024;
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
        .route("/v1/devices", get(devices))
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
        .route("/v1/models/pulls", get(list_pull_jobs))
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
        .route(
            "/v1/audio/transcriptions/{id}/cancel",
            post(cancel_transcription_job),
        )
        .route(
            "/v1/audio/transcriptions/{id}/pause",
            post(pause_transcription_job),
        )
        .route(
            "/v1/audio/transcriptions/{id}/resume",
            post(resume_transcription_job),
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
    // Boot stage timing: each phase logged as its own timestamped line so a
    // slow daemon start (validate/bind/router-build/model-bind) can be
    // attributed to a specific phase from `daemon.log` alone, instead of
    // guessing from wall-clock reads of surrounding, untimed events. Additive
    // only -- the existing "OpenASR server listening on ..." banners below are
    // left byte-for-byte unchanged, since `crates/openasr-cli/tests/cli.rs`
    // waits on that exact prefix and the desktop sidecar's readiness probe
    // must keep working unmodified.
    let boot_started = Instant::now();
    let mut stage_started = boot_started;
    validate_listen_security(addr, &launch_options)?;
    openasr_core::stage_timing::log_stage(
        "server_boot",
        "validate_listen_security",
        stage_started.elapsed(),
    );
    stage_started = Instant::now();
    runtime.validate()?;
    openasr_core::stage_timing::log_stage(
        "server_boot",
        "runtime_validate",
        stage_started.elapsed(),
    );
    stage_started = Instant::now();
    let listener = TcpListener::bind(addr).await?;
    // Shadow the requested `addr` (which may be an OS-assigned wildcard like
    // `:0`) with the listener's actual bound address so everything below --
    // the "listening on" banners and the stage-timing "ready" event -- reports
    // the real, connectable port instead of echoing the CLI's `--addr` input
    // back verbatim.
    let addr = listener.local_addr()?;
    openasr_core::stage_timing::log_stage(
        "server_boot",
        "tcp_listener_bind",
        stage_started.elapsed(),
    );
    // Warm the daemon's default bound native model pack in the background,
    // right after bind succeeds -- deliberately after this line and before
    // anything below that could block, so it never gates bind/serve/health.
    // See `spawn_boot_native_warmup`'s doc comment for the dedup story with a
    // concurrent real WS attach.
    realtime::spawn_boot_native_warmup(runtime.clone());
    // idle_unload: only spawn the reaper when the resolved policy is not
    // `never` (`idle_unload_after` is `None` for `never` and for every
    // existing caller/test that does not set it, so this is a no-op there).
    if let Some(idle_unload_after) = launch_options.idle_unload_after {
        spawn_idle_unload_reaper(idle_unload_after);
    }
    match &launch_options.tls.clone() {
        ServerTlsConfig::Disabled => {
            stage_started = Instant::now();
            let app = app_with_runtime_and_distribution_and_launch_options(
                runtime,
                DistributionRuntime::default(),
                launch_options,
            );
            openasr_core::stage_timing::log_stage(
                "server_boot",
                "router_build",
                stage_started.elapsed(),
            );
            openasr_core::stage_timing::log_event(
                "server_boot",
                format_args!(
                    "stage=ready total_ms={:.3} addr=http://{addr}",
                    boot_started.elapsed().as_secs_f64() * 1000.0
                ),
            );
            println!("OpenASR server listening on http://{addr}");
            axum::serve(listener, app).await?;
        }
        ServerTlsConfig::SelfSigned { subject_alt_names } => {
            stage_started = Instant::now();
            let identity = load_or_generate_self_signed_tls_identity(
                subject_alt_names,
                launch_options.tls_identity_store.as_deref(),
            )?;
            openasr_core::stage_timing::log_stage(
                "server_boot",
                "tls_self_signed_identity",
                stage_started.elapsed(),
            );
            let mut launch_options = launch_options;
            launch_options.auth = launch_options.auth.with_pairing_safety_code(Some(
                pairing_safety_code_for_certificate_fingerprint(&identity.certificate_sha256),
            ));
            stage_started = Instant::now();
            let app = app_with_runtime_and_distribution_and_launch_options(
                runtime,
                DistributionRuntime::default(),
                launch_options,
            );
            openasr_core::stage_timing::log_stage(
                "server_boot",
                "router_build",
                stage_started.elapsed(),
            );
            openasr_core::stage_timing::log_event(
                "server_boot",
                format_args!(
                    "stage=ready total_ms={:.3} addr=https://{addr}",
                    boot_started.elapsed().as_secs_f64() * 1000.0
                ),
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

/// Validity window for a freshly generated self-signed TLS identity: long
/// enough that "expired, regenerate" is a rare, load-bearing rotation rather
/// than something routine daemon restarts would ever hit, short enough to
/// still be a believable server certificate lifetime (this mirrors the
/// ~397-day cap modern browsers/CAs enforce for publicly-trusted TLS server
/// certs -- nothing here is publicly trusted, but there is no reason to mint
/// something longer-lived).
const TLS_IDENTITY_VALIDITY_DAYS: i64 = 397;
/// Backdate `not_before` by this much so a client whose clock runs slightly
/// behind the server's still sees an already-valid certificate.
const TLS_IDENTITY_CLOCK_SKEW_BACKDATE_HOURS: i64 = 24;

#[derive(Clone)]
struct TlsIdentity {
    acceptor: TlsAcceptor,
    certificate_sha256: String,
    pairing_safety_code: String,
    #[cfg(test)]
    certificate_der: CertificateDer<'static>,
}

/// On-disk shape of a persisted self-signed TLS identity: the private key +
/// certificate DER, plus the metadata needed to decide whether it is still
/// usable without re-parsing X.509 fields. Stored as plain JSON like
/// `pairing-registry.json`; the private key field is the sensitive one,
/// which is why `persist_tls_identity` writes the file with owner-only
/// (0600) permissions from the moment it is created -- on unix, via
/// `openasr_core::write_owner_only_file_atomically`, there is no
/// group/other-readable window at any point, matching the pairing store and
/// API key store's convention for secret-bearing files under OPENASR_HOME.
/// Windows has no equivalent owner-only chmod here; the file relies on the
/// user profile directory's default ACL (see `write_owner_only_file_atomically`'s
/// unix-only permission step) -- tracked as a gap, not fixed by this PR.
#[derive(Debug, Serialize, Deserialize)]
struct PersistedTlsIdentity {
    subject_alt_names: Vec<String>,
    certificate_der: Vec<u8>,
    private_key_der: Vec<u8>,
    not_before_unix_secs: u64,
    not_after_unix_secs: u64,
}

/// Generates a brand new self-signed identity for `subject_alt_names` with no
/// persistence -- every call mints a fresh keypair+certificate. Used directly
/// by tests that want an ephemeral identity, and as the fallback inside
/// `load_or_generate_self_signed_tls_identity` when no store path is given.
fn self_signed_tls_identity(subject_alt_names: &[String]) -> anyhow::Result<TlsIdentity> {
    let (certificate_der, private_key_der, ..) =
        generate_self_signed_tls_material(subject_alt_names)?;
    tls_identity_from_der(certificate_der, private_key_der)
}

/// Generates a new self-signed keypair + certificate for `subject_alt_names`.
/// Returns the raw DER encodings (so callers can persist them) alongside the
/// validity window's Unix-epoch bounds (so callers can record expiry without
/// re-parsing the certificate's X.509 fields later).
fn generate_self_signed_tls_material(
    subject_alt_names: &[String],
) -> anyhow::Result<(Vec<u8>, Vec<u8>, u64, u64)> {
    let mut params = CertificateParams::new(subject_alt_names.to_vec());
    let now = OffsetDateTime::now_utc();
    let not_before = now - time::Duration::hours(TLS_IDENTITY_CLOCK_SKEW_BACKDATE_HOURS);
    let not_after = now + time::Duration::days(TLS_IDENTITY_VALIDITY_DAYS);
    params.not_before = not_before;
    params.not_after = not_after;
    let certified = Certificate::from_params(params)?;
    let certificate_der = certified.serialize_der()?;
    let private_key_der = certified.serialize_private_key_der();
    Ok((
        certificate_der,
        private_key_der,
        not_before.unix_timestamp() as u64,
        not_after.unix_timestamp() as u64,
    ))
}

/// Builds a `TlsIdentity` (fingerprint, pairing safety code, rustls acceptor)
/// from raw DER bytes, whether they were just generated or loaded back from
/// `persist_tls_identity`'s store file. The single place that turns key
/// material into a live rustls `ServerConfig`, so a persisted identity and a
/// freshly generated one are wired up identically.
fn tls_identity_from_der(
    certificate_der: Vec<u8>,
    private_key_der: Vec<u8>,
) -> anyhow::Result<TlsIdentity> {
    let certificate_der = CertificateDer::from(certificate_der);
    let private_key_der = PrivateKeyDer::from(PrivatePkcs8KeyDer::from(private_key_der));
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

/// Loads a persisted self-signed TLS identity from `store_path` (if given)
/// and reuses it when it is readable, was issued for the same
/// `subject_alt_names`, and has not expired -- otherwise generates a fresh
/// identity and persists it to `store_path` for next time. `store_path: None`
/// keeps the previous always-generate-in-memory behavior (used by tests and
/// any caller with no durable home to persist under).
///
/// This is what keeps a paired remote client's TOFU-pinned certificate
/// fingerprint (and the human-readable pairing safety code derived from it,
/// see `pairing_safety_code_for_certificate_fingerprint`) stable across the
/// daemon restarts the desktop app performs on every model switch: before
/// this, `serve_with_launch_options` minted a brand new keypair+certificate
/// on *every* start, so every restart rotated the identity out from under
/// already-paired clients and forced them to re-pair.
///
/// Fail-closed on a damaged store: a present-but-corrupt or
/// permission-denied file is never silently treated as "no identity yet" --
/// only a genuinely missing file is. Any other read/parse error, *and* a
/// present-and-well-formed-JSON-but-unusable DER payload (truncated/bit-flipped
/// key or certificate bytes that parse as JSON but fail to build into a
/// rustls `ServerConfig` -- see the `tls_identity_from_der` call below), falls
/// through to generating (and re-persisting) a fresh identity, logging why,
/// rather than risking a hard startup failure over a damaged on-disk keypair.
/// A missing file, or an expired/SAN-mismatched identity, are ordinary
/// rotation and are not logged as errors.
fn load_or_generate_self_signed_tls_identity(
    subject_alt_names: &[String],
    store_path: Option<&Path>,
) -> anyhow::Result<TlsIdentity> {
    let Some(store_path) = store_path else {
        return self_signed_tls_identity(subject_alt_names);
    };
    harden_tls_identity_store_dir_permissions(store_path);
    match load_persisted_tls_identity(store_path, subject_alt_names) {
        Ok(Some((certificate_der, private_key_der))) => {
            match tls_identity_from_der(certificate_der, private_key_der) {
                Ok(identity) => return Ok(identity),
                Err(error) => {
                    // Legitimate JSON envelope, but the DER payload inside it
                    // does not parse/build into a usable rustls keypair
                    // (truncated write, bit flip, hand-edited file). Treat
                    // exactly like corrupt JSON: log and regenerate below,
                    // rather than let the `?`-propagated error here fail
                    // serve's startup outright and require manual deletion of
                    // the store file to recover.
                    eprintln!(
                        "openasr-server: persisted TLS identity at {} did not load as a valid keypair/certificate ({error}); treating as corrupt and regenerating (paired remote clients will need to re-confirm the new fingerprint)",
                        store_path.display()
                    );
                }
            }
        }
        Ok(None) => {
            // Missing, expired, or issued for different subject alt names:
            // ordinary rotation, already logged (if interesting) by
            // load_persisted_tls_identity. Fall through to regenerate.
        }
        Err(error) => {
            eprintln!(
                "openasr-server: could not load persisted TLS identity from {} ({error}); regenerating a new identity (paired remote clients will need to re-confirm the new fingerprint)",
                store_path.display()
            );
        }
    }
    let (certificate_der, private_key_der, not_before_unix_secs, not_after_unix_secs) =
        generate_self_signed_tls_material(subject_alt_names)?;
    persist_tls_identity(
        store_path,
        subject_alt_names,
        &certificate_der,
        &private_key_der,
        not_before_unix_secs,
        not_after_unix_secs,
    );
    tls_identity_from_der(certificate_der, private_key_der)
}

/// Best-effort owner-only (0700) hardening of the directory that will hold
/// `store_path` (in practice `OPENASR_HOME`), applied unconditionally on
/// every load/generate call -- not just the first time the directory is
/// created -- so a directory that predates this PR (or was widened by some
/// other tool) gets tightened back up on the next daemon start rather than
/// staying at whatever mode `create_dir_all` + the process umask produced.
/// Creates the directory if it does not exist yet (the TLS identity store,
/// unlike the pairing/API-key stores, has no other writer upstream in the
/// boot sequence that is guaranteed to have created `OPENASR_HOME` first).
/// Never fails serve over this: a create or chmod failure is logged and
/// swallowed, matching every other persistence step in this module.
// On non-unix targets the `#[cfg(unix)]` chmod block below does not exist, so
// this early `return` becomes the function's last statement and clippy (on
// those targets only) reads it as needless; it stays required on unix to skip
// that block after a failed `create_dir_all`.
#[cfg_attr(not(unix), allow(clippy::needless_return))]
fn harden_tls_identity_store_dir_permissions(store_path: &Path) {
    let Some(parent) = store_path.parent() else {
        return;
    };
    if let Err(error) = fs::create_dir_all(parent) {
        eprintln!(
            "openasr-server: could not create {} to hold the TLS identity store ({error}); continuing (persistence may fail)",
            parent.display()
        );
        return;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Err(error) = fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700)) {
            eprintln!(
                "openasr-server: could not tighten {} to owner-only (0700) permissions ({error}); continuing",
                parent.display()
            );
        }
    }
}

/// Returns `Ok(Some(..))` when `store_path` holds a still-usable identity for
/// `subject_alt_names`, `Ok(None)` when it is absent/expired/for different
/// names (normal, not-logged-as-error rotation), and `Err` when the file is
/// present but unreadable as a well-formed identity (corrupt JSON, empty key
/// material) -- the caller treats `Err` as a louder "this should not happen"
/// event distinct from routine rotation.
fn load_persisted_tls_identity(
    store_path: &Path,
    subject_alt_names: &[String],
) -> anyhow::Result<Option<(Vec<u8>, Vec<u8>)>> {
    let bytes = match fs::read(store_path) {
        Ok(bytes) => bytes,
        // Absent file = legitimately no persisted identity yet (fresh
        // install, or first ever --tls-self-signed run).
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        // Anything else (permission denied, I/O error) is NOT "no identity":
        // returning Err keeps the caller from treating a locked-out file as
        // license to silently mint (and overwrite) a new one without saying
        // why.
        Err(error) => return Err(error.into()),
    };
    let persisted: PersistedTlsIdentity = serde_json::from_slice(&bytes)?;
    if persisted.certificate_der.is_empty() || persisted.private_key_der.is_empty() {
        anyhow::bail!("persisted TLS identity has empty certificate or private key material");
    }
    if persisted.subject_alt_names != subject_alt_names {
        eprintln!(
            "openasr-server: persisted TLS identity at {} was issued for different subject alt names ({:?}, now requesting {:?}); rotating",
            store_path.display(),
            persisted.subject_alt_names,
            subject_alt_names
        );
        return Ok(None);
    }
    if unix_now_secs() >= persisted.not_after_unix_secs {
        eprintln!(
            "openasr-server: persisted TLS identity at {} expired (not_after unix {}); rotating",
            store_path.display(),
            persisted.not_after_unix_secs
        );
        return Ok(None);
    }
    Ok(Some((persisted.certificate_der, persisted.private_key_der)))
}

/// Persists `certificate_der`/`private_key_der` (plus the metadata needed to
/// validate reuse later) to `store_path`, atomically and with owner-only
/// (0600) permissions applied from the moment the temporary file is
/// created (on unix; see `openasr_core::write_owner_only_file_atomically`) --
/// this is the only place a self-signed TLS private key touches disk. Unlike
/// a plain atomic write followed by a post-rename `chmod`, there is no
/// window where the renamed file (or its temp-file predecessor) is
/// group/other readable: the raw PKCS8 key never sits on disk at a
/// permissive mode, even transiently. Best-effort: a write failure is logged
/// and otherwise swallowed (the freshly generated in-memory identity still
/// serves this boot; it simply will not survive the next restart), matching
/// `persist_pairing_credentials_locked`'s "never fail serve over a
/// persistence hiccup" posture.
fn persist_tls_identity(
    store_path: &Path,
    subject_alt_names: &[String],
    certificate_der: &[u8],
    private_key_der: &[u8],
    not_before_unix_secs: u64,
    not_after_unix_secs: u64,
) {
    let persisted = PersistedTlsIdentity {
        subject_alt_names: subject_alt_names.to_vec(),
        certificate_der: certificate_der.to_vec(),
        private_key_der: private_key_der.to_vec(),
        not_before_unix_secs,
        not_after_unix_secs,
    };
    match serde_json::to_vec_pretty(&persisted) {
        Ok(bytes) => {
            if let Err(error) = openasr_core::write_owner_only_file_atomically(store_path, &bytes) {
                eprintln!(
                    "openasr-server: could not persist TLS identity to {} (continuing without persistence; this identity will not survive a restart): {error}",
                    store_path.display()
                );
            }
        }
        Err(error) => {
            eprintln!(
                "openasr-server: could not serialize TLS identity (continuing without persistence): {error}"
            );
        }
    }
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
    /// Resolved `idle_unload` threshold (see
    /// `openasr_core::config::IdleUnloadPolicy::idle_threshold`); `None`
    /// (the default, and what `never` resolves to) never spawns the reaper,
    /// matching every existing caller/test that does not set this.
    pub idle_unload_after: Option<Duration>,
    /// Where to persist (and load back) the self-signed TLS private key +
    /// certificate across restarts -- see
    /// `load_or_generate_self_signed_tls_identity`. `None` keeps the
    /// pre-persistence behavior of generating a brand new identity on every
    /// `serve_with_launch_options` call; only a caller that sets this (the
    /// CLI, to `OPENASR_HOME/tls-identity.json`, alongside
    /// `pairing-registry.json`) gets an identity that survives a restart.
    /// No-op when `tls` is `ServerTlsConfig::Disabled`.
    pub tls_identity_store: Option<PathBuf>,
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

    /// Whether the bound model's runtime is resident right now (surfaced by
    /// `/health` as `model_resident`, gated on `has_model_bound` -- see the
    /// call site) so clients can tell "loaded, ready to transcribe
    /// instantly" apart from "bound but idle-unloaded, the next request pays
    /// a cold rebuild". `mock` has no runtime to unload -- it is resident
    /// whenever it is bound. `native`'s residency is the real, process-wide
    /// idle-unload signal in `idle_activity`, not a guess: it reads `true`
    /// only after a successful load/decode, and flips back to `false` the
    /// moment the `idle_unload` reaper evicts the cached runtime, without
    /// this method reaching into any per-thread cache itself.
    fn model_is_resident(&self) -> bool {
        match self.backend {
            BackendKind::Mock => true,
            BackendKind::Native => idle_activity::native_model_is_resident(),
        }
    }
}

/// Where to load the runtime model catalog from: either a URL/path string
/// used as both fetch source and verification identity (the pre-existing
/// `OPENASR_CATALOG_URL` / `--catalog-url` behavior), or a local file whose
/// bytes and verification identity are supplied separately (the
/// `OPENASR_CATALOG_FILE` + `OPENASR_CATALOG_IDENTITY` pair, resolved by the
/// shared [`openasr_core::resolve_local_catalog_env_override`] and also used
/// by `openasr-cli`'s startup catalog resolution). See
/// [`openasr_core::LocalCatalogEnvOverride`]'s doc comment for why the split
/// exists.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CatalogSource<'a> {
    Url(&'a str),
    LocalFile { path: &'a Path, identity: &'a str },
}

/// Loads the catalog for an explicit [`CatalogSource`], routing a `LocalFile`
/// source through [`load_local_catalog_file_with_identity`] (bytes from
/// `path`, signature checked against `identity`) and a `Url` source through
/// the existing [`load_model_catalog`] (fetch + verify against the same
/// string), unchanged from today's behavior.
pub(crate) fn load_catalog_for_source(
    source: CatalogSource<'_>,
    home: &Path,
) -> Result<ModelCatalog, CatalogError> {
    match source {
        CatalogSource::Url(url) => load_model_catalog(Some(url), home),
        CatalogSource::LocalFile { path, identity } => {
            load_local_catalog_file_with_identity(path, identity, home)
        }
    }
}

/// Same as [`load_catalog_for_source`], but `None` falls back to
/// [`load_model_catalog`]'s own default-catalog resolution (network/cache/
/// embedded) instead of skipping the load -- for call sites that always need
/// *some* catalog, matching the pre-existing `load_model_catalog(catalog_url,
/// home)` behavior when `catalog_url` was `None`.
pub(crate) fn load_catalog_for_optional_source(
    source: Option<CatalogSource<'_>>,
    home: &Path,
) -> Result<ModelCatalog, CatalogError> {
    match source {
        Some(source) => load_catalog_for_source(source, home),
        None => load_model_catalog(None, home),
    }
}

/// Same split as [`load_catalog_for_source`], but for the
/// [`resolve_runtime_catalog`] (network/cache tier + embedded-epoch-max
/// floor) resolution path rather than the plain network/cache
/// [`load_model_catalog`] one. A `LocalFile` override is authoritative on its
/// own -- like an explicit non-default URL override already is in
/// `resolve_runtime_catalog` -- so it is loaded directly without the embedded
/// epoch-max comparison.
pub(crate) fn resolve_runtime_catalog_for_source(
    source: CatalogSource<'_>,
    home: &Path,
) -> Result<ModelCatalog, CatalogError> {
    match source {
        CatalogSource::Url(url) => resolve_runtime_catalog(Some(url), home),
        CatalogSource::LocalFile { path, identity } => {
            load_local_catalog_file_with_identity(path, identity, home)
        }
    }
}

/// Reads the `OPENASR_CATALOG_FILE` + `OPENASR_CATALOG_IDENTITY` env var pair
/// via the shared [`openasr_core::resolve_local_catalog_env_override`],
/// surfacing a half-configured pair as a stderr warning (rather than silently
/// dropping half the config) so the misconfiguration is visible instead of
/// quietly changing trust behavior.
fn catalog_local_override_from_env() -> Option<openasr_core::LocalCatalogEnvOverride> {
    let (override_, warning) = openasr_core::resolve_local_catalog_env_override();
    if let Some(warning) = warning {
        eprintln!("warning: {warning} Falling back to OPENASR_CATALOG_URL / the default catalog.");
    }
    override_
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributionRuntime {
    pub openasr_home: Option<PathBuf>,
    pub catalog_url: Option<String>,
    /// Explicit local-file + declared-identity catalog override; see
    /// [`openasr_core::LocalCatalogEnvOverride`]. Takes precedence over
    /// `catalog_url` when set (see `DistributionContext::catalog_source`);
    /// `catalog_url`'s behavior is otherwise completely unchanged.
    pub catalog_local_override: Option<openasr_core::LocalCatalogEnvOverride>,
}

impl Default for DistributionRuntime {
    fn default() -> Self {
        Self {
            openasr_home: None,
            catalog_url: env::var("OPENASR_CATALOG_URL")
                .ok()
                .filter(|value| !value.trim().is_empty()),
            catalog_local_override: catalog_local_override_from_env(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct DistributionContext {
    runtime: DistributionRuntime,
    jobs: Arc<DistributionJobs>,
    // In-flight file transcriptions that opted into control, keyed by the
    // client-supplied transcription id. The pause/resume/cancel HTTP handlers
    // flip these flags while the blocking decode runs on a `spawn_blocking`
    // worker. In-session only: an entry lives just for one transcription's
    // lifetime and is cleared when the request returns.
    transcriptions: Arc<Mutex<HashMap<String, Arc<openasr_core::TranscriptionControl>>>>,
}

impl DistributionContext {
    fn new(runtime: DistributionRuntime) -> Self {
        Self {
            jobs: Arc::new(DistributionJobs::new(load_persisted_pull_jobs(&runtime))),
            transcriptions: Arc::new(Mutex::new(HashMap::new())),
            runtime,
        }
    }

    fn register_transcription(
        &self,
        transcription_id: &str,
        control: Arc<openasr_core::TranscriptionControl>,
    ) {
        self.transcriptions
            .lock()
            .expect("active transcription registry mutex poisoned")
            .insert(transcription_id.to_string(), control);
    }

    fn clear_transcription(&self, transcription_id: &str) {
        self.transcriptions
            .lock()
            .expect("active transcription registry mutex poisoned")
            .remove(transcription_id);
    }

    fn transcription_control(
        &self,
        transcription_id: &str,
    ) -> Option<Arc<openasr_core::TranscriptionControl>> {
        self.transcriptions
            .lock()
            .expect("active transcription registry mutex poisoned")
            .get(transcription_id)
            .cloned()
    }

    fn openasr_home(&self) -> Result<PathBuf, ApiError> {
        self.runtime
            .openasr_home
            .clone()
            .map(Ok)
            .unwrap_or_else(openasr_home)
            .map_err(ApiError::Home)
    }

    pub(crate) fn catalog_source(&self) -> Option<CatalogSource<'_>> {
        if let Some(local) = &self.runtime.catalog_local_override {
            return Some(CatalogSource::LocalFile {
                path: &local.path,
                identity: &local.identity,
            });
        }
        self.runtime.catalog_url.as_deref().map(CatalogSource::Url)
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
                    // A job with a pending cancel is on its way to a terminal
                    // `Canceled` state (the worker unwinds asynchronously). It
                    // must not be handed back as the live job for a fresh pull
                    // of the same pack: doing so silently coalesces the user's
                    // "download again" into the dying job, so no new download
                    // ever starts until the cancel fully settles. Excluding it
                    // here lets `start_pull_job` mint a new job instead (which
                    // then queues on the same pull lock the canceling job still
                    // holds, and proceeds once it releases).
                    && snapshot.control_requested != Some(PullControlRequest::Cancel)
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

    /// All jobs not yet in a terminal state, sorted by `job_id` for a stable
    /// listing order. Read-only: does not touch the restart-resume worker or
    /// persisted files, so calling it has no side effects on job state.
    fn nonterminal_snapshots(&self) -> Vec<PullJobSnapshot> {
        let mut snapshots: Vec<PullJobSnapshot> = self
            .jobs
            .snapshots
            .lock()
            .expect("pull job snapshots mutex poisoned")
            .values()
            .filter(|snapshot| !snapshot.state.is_terminal())
            .cloned()
            .collect();
        snapshots.sort_by(|left, right| left.job_id.cmp(&right.job_id));
        snapshots
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
    Extension(distribution): Extension<DistributionContext>,
) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        server_version: identity.server_version,
        pid: identity.pid,
        instance_token: identity.instance_token.clone(),
        model_installed: runtime.has_model_bound(),
        model_resident: runtime.has_model_bound() && runtime.model_is_resident(),
        native_active_count: idle_activity::native_activity_active_count() as u64,
        idle_seconds: idle_activity::native_activity_idle_seconds(Instant::now()),
        abandoned_worker_count: realtime::abandoned_stuck_worker_count() as u64,
        catalog_degraded: catalog_degraded_reason(&distribution),
    })
}

/// Best-effort read of the current "catalog degraded" status for `/health`.
/// Never fails the response -- an unresolvable `$OPENASR_HOME` or a
/// stale/missing status file both read as "not degraded" (`None`) rather than
/// an error, since this is a diagnostic surface, not a trust decision (see
/// `openasr_core::read_catalog_degraded_status`).
fn catalog_degraded_reason(distribution: &DistributionContext) -> Option<String> {
    let home = distribution.openasr_home().ok()?;
    openasr_core::read_catalog_degraded_status(home).map(|status| status.reason)
}

async fn models(State(runtime): State<ServerRuntime>) -> Result<Json<ModelsResponse>, ApiError> {
    let ids: Vec<String> = match runtime.backend {
        BackendKind::Mock => runtime_registry(None)
            .map_err(ApiError::from)?
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
    let catalog = load_catalog_for_optional_source(distribution.catalog_source(), &home)
        .map_err(ApiError::Catalog)?;
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

/// Read-only compute-device enumeration for the local UI's execution-target
/// picker. Derives the device list from *this* (the daemon's) ggml runtime, so
/// a shell built in a different backend shape than the sidecar (e.g. a CPU-only
/// desktop supervising a Vulkan sidecar on Windows) sees the backends inference
/// actually runs on instead of its own. Pure hardware facts, no secrets; sits
/// behind the same local auth layer as `/v1/capabilities`.
async fn devices() -> Json<DevicesResponse> {
    let runtime = openasr_core::GgmlRuntimeInfo::detect();
    let devices = openasr_core::compute_devices_from_runtime(&runtime);
    let default_execution_target = openasr_core::default_execution_target(&devices);
    Json(DevicesResponse {
        object: "devices",
        default_execution_target,
        devices,
    })
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

// TS export for the HTTP daemon response wire contract (`/health`,
// `/v1/models`, `/v1/capabilities`, `/v1/devices`): gated to `cfg(test)` so
// ts-rs is a dev-only dependency, never part of the shipped rlib. See
// `src/http_wire_bindings_test.rs` for the golden "regenerate == committed"
// guard, and its module doc for the directory-layout rationale (a few
// openasr-core capability leaf types are legitimately re-exported here,
// alongside their already-committed copy under
// `crates/openasr-core/generated/realtime-wire/`).
#[derive(Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export_to = "generated/http-wire/"))]
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
    /// Whether the bound model's runtime is currently resident in memory,
    /// i.e. ready to transcribe instantly with no cold-load latency. Added in
    /// 0.1.13 alongside the `idle_unload` reaper actually releasing the
    /// runtime after an idle period: `model_installed: true,
    /// model_resident: false` means a model is bound but its runtime has been
    /// unloaded (idle past the configured `idle_unload` threshold, or never
    /// loaded yet this boot) -- the next transcription request pays a cold
    /// rebuild before it can run, but is not itself an error. Always `false`
    /// when `model_installed` is `false` (nothing to be resident). Additive:
    /// absent in the pre-0.1.13 contract, so an older client that only reads
    /// `model_installed` keeps working unchanged.
    model_resident: bool,
    /// Debug-observability field: the process-wide count of currently active
    /// native requests/sessions (see the `idle_activity` module doc) --
    /// in-flight offline transcriptions/translations and attached realtime
    /// native-streaming sessions both count. Not gated on `model_installed`;
    /// reads `0` on the mock backend and on a fresh install with no model
    /// bound, since nothing ever enters the tracker there. Not a stable
    /// health signal on its own -- a transient nonzero count during a single
    /// request is normal, not a problem -- but useful when diagnosing why
    /// `idle_unload` has not fired ("is a session still counted active?").
    /// Additive: absent in the pre-0.1.14 contract.
    native_active_count: u64,
    /// Debug-observability field: seconds elapsed since the process-wide
    /// native activity count last returned to zero, as of this response
    /// (`0` while `native_active_count` is nonzero -- there is no
    /// meaningful idle duration mid-request). Pairs with
    /// `native_active_count` to diagnose `idle_unload` timing: compare
    /// against the configured `idle_unload` threshold to see how close the
    /// next sweep is. Additive: absent in the pre-0.1.14 contract.
    idle_seconds: u64,
    /// Debug-observability field: the process-wide count of native streaming
    /// decode workers the decode watchdog has abandoned because a decode never
    /// returned within its budget (each presumed a permanently wedged OS
    /// thread pinning a resident model runtime). Normally `0`; a nonzero value
    /// is a strong signal of a GPU/driver-level decode hang. The daemon fails
    /// loud and exits once this reaches its internal threshold (so a
    /// supervisor can restart it with a clean slate), making this the field to
    /// watch for that condition. Reads `0` on the mock backend. Additive:
    /// absent in the pre-0.1.15 contract.
    abandoned_worker_count: u64,
    /// `Some(reason)` when the model catalog most recently loaded on this
    /// machine came from a degraded tier (the on-disk signed cache, or the
    /// embedded offline snapshot) rather than a freshly verified primary
    /// source -- e.g. the primary catalog fetch/bundled resource failed, or a
    /// boot-local candidate's epoch sat below this machine's recorded
    /// anti-rollback floor. `reason` is the human-readable cause, for
    /// operator/shell diagnostics. `None` means either the last load used the
    /// primary source, or no catalog load has recorded a status yet (e.g. a
    /// backend that never touches the catalog). Best-effort: a stale/missing
    /// status file reads as `None`, never as an error -- see
    /// `openasr_core::read_catalog_degraded_status` and
    /// `docs/CATALOG_COMPATIBILITY.md`. Additive: absent in the pre-0.1.16
    /// contract.
    catalog_degraded: Option<String>,
}

#[derive(Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export_to = "generated/http-wire/"))]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelResponse>,
}

#[derive(Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export_to = "generated/http-wire/"))]
struct ModelResponse {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export_to = "generated/http-wire/"))]
struct CapabilitiesResponse {
    object: &'static str,
    transcription: TranscriptionBackendCapabilities,
    realtime: RealtimeBackendCapabilities,
}

#[derive(Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export_to = "generated/http-wire/"))]
struct DevicesResponse {
    object: &'static str,
    /// What the `auto` target resolves to on this daemon (`cpu` or
    /// `accelerated`), so a client can render the default without re-deriving
    /// it from the list.
    default_execution_target: String,
    devices: Vec<openasr_core::ComputeDevice>,
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
    /// Tri-state read straight off `openasr_core::default_selection::
    /// DefaultModelResolution`: "installed" (`pack`/`default_pull` populated),
    /// "not_installed" (a default is configured but has no matching installed
    /// pack -- `default_model` still names it), or "unset" (nothing configured
    /// at all). Always present, unlike the other fields, so a client can
    /// distinguish "reinstall your default" from "choose a model" without
    /// falling back to null-checking `pack`.
    default_model_status: &'static str,
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
    /// The upload's temp volume ran (or was about to run) low on free space
    /// mid-stream. The message is pre-built at the call site since it needs
    /// the probed byte counts and temp-dir path.
    InsufficientDiskSpace(String),
}

impl From<openasr_core::RuntimeRegistryError> for ApiError {
    fn from(error: openasr_core::RuntimeRegistryError) -> Self {
        match error {
            openasr_core::RuntimeRegistryError::Registry(error) => Self::Registry(error),
            openasr_core::RuntimeRegistryError::Catalog(error) => Self::Catalog(error),
        }
    }
}

impl From<openasr_core::default_selection::DefaultSelectionError> for ApiError {
    fn from(error: openasr_core::default_selection::DefaultSelectionError) -> Self {
        match error {
            openasr_core::default_selection::DefaultSelectionError::Config(error) => {
                Self::Config(error)
            }
            openasr_core::default_selection::DefaultSelectionError::Pull(error) => {
                Self::Pull(error)
            }
            openasr_core::default_selection::DefaultSelectionError::Catalog(error) => {
                Self::Catalog(error)
            }
        }
    }
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
            Self::InsufficientDiskSpace(message) => f.write_str(message),
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
                    | openasr_core::BackendError::WordTimestampAlignmentRequiresWordTimestamps
                    | openasr_core::BackendError::WordTimestampAlignmentPackMissing { .. }
                    | openasr_core::BackendError::WordTimestampAlignmentFailed { .. }
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
                    // A caller-requested cancel is not a fault: the client asked
                    // to stop this in-flight transcription. 409 distinguishes it
                    // from the 400 fail-closed refusals and the 5xx faults.
                    openasr_core::BackendError::TranscriptionCanceled => StatusCode::CONFLICT,
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
            Self::InsufficientDiskSpace(message) => (StatusCode::INSUFFICIENT_STORAGE, message),
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
                        StatusCode::INSUFFICIENT_STORAGE => "insufficient_storage_error",
                        _ => "openasr_error",
                    },
                    param: None,
                    code: None,
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
            "Uploaded file is too large. The daemon accepts uploads up to {} GB; \
             split the recording or use a smaller/compressed file.",
            MAX_TRANSCRIPTION_UPLOAD_BYTES / (1024 * 1024 * 1024)
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
    /// Always `null` today. Present (not skipped) so clients written against
    /// the OpenAI error envelope find every key they expect: `{"error":
    /// {"message", "type", "param", "code"}}`.
    param: Option<String>,
    code: Option<String>,
}

#[cfg(test)]
#[path = "http_wire_bindings_test.rs"]
mod http_wire_bindings_test;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
