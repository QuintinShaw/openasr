//! Integration-style unit tests for the server crate. Pure code-motion from `lib.rs`.

use futures_util::{SinkExt, StreamExt};
use openasr_core::RealtimeBackendMode;
use openasr_core::config::{HistoryRetentionPolicy, MAX_INFERENCE_THREADS, Preferences};
use openasr_core::realtime::history::{
    DaemonHistoryKind, DaemonHistoryProvenance, DaemonHistoryRecord, DaemonHistoryStore,
};
use openasr_core::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
use openasr_core::{
    ExecutionTarget, LongFormMode, NativeAsrHardwareTarget, ResponseFormat, Transcription,
    TranscriptionRequest,
};
use rustls::{ClientConfig, pki_types::ServerName};
use tokio_rustls::TlsConnector;
use tokio_tungstenite::tungstenite::{client::IntoClientRequest, protocol::Message as WsMessage};

use super::*;
use crate::testing::{
    LoopbackTlsServer, TestTofuVerifier, approve_loopback_pairing, bearer_auth_header,
    https_request, revoke_loopback_pairing, spawn_loopback_pairing_server,
};

#[test]
fn serve_batch_unavailable_retryable_maps_to_429() {
    let response = ApiError::Backend(openasr_core::BackendError::ServeBatchUnavailable {
        reason: "queue full".to_string(),
        retryable: true,
    })
    .into_response();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[test]
fn serve_batch_unavailable_non_retryable_maps_to_503() {
    let response = ApiError::Backend(openasr_core::BackendError::ServeBatchUnavailable {
        reason: "owner disconnected".to_string(),
        retryable: false,
    })
    .into_response();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[test]
fn non_loopback_tls_escape_still_requires_authentication() {
    let err = validate_listen_security_with_escape(
        "0.0.0.0:0".parse().unwrap(),
        &ServerLaunchOptions::default(),
        true,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("requires device authentication"),
        "unexpected error: {err:?}"
    );
}

fn header_map_with_bearer(token: &str) -> axum::http::HeaderMap {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        header::AUTHORIZATION,
        format!("Bearer {token}").parse().unwrap(),
    );
    headers
}

#[test]
fn from_token_hashes_authorizes_only_the_matching_token() {
    let auth = ServerAuth::from_token_hashes([bearer_token_hash("agent-secret")]);
    assert!(auth.authorizes(&header_map_with_bearer("agent-secret")));
    assert!(!auth.authorizes(&header_map_with_bearer("wrong-token")));
    assert!(!auth.authorizes(&axum::http::HeaderMap::new()));
}

#[test]
fn from_token_hashes_with_no_hashes_disables_auth() {
    let auth = ServerAuth::from_token_hashes(Vec::<String>::new());
    assert!(!auth.is_enabled());
    // Disabled auth authorizes everyone -- this is the loopback-default-free
    // state before any `openasr apikey create`.
    assert!(auth.authorizes(&axum::http::HeaderMap::new()));
}

#[test]
fn from_token_hashes_supports_multiple_concurrently_valid_keys() {
    let auth =
        ServerAuth::from_token_hashes([bearer_token_hash("key-a"), bearer_token_hash("key-b")]);
    assert!(auth.authorizes(&header_map_with_bearer("key-a")));
    assert!(auth.authorizes(&header_map_with_bearer("key-b")));
    assert!(!auth.authorizes(&header_map_with_bearer("key-c")));
}

#[test]
fn core_api_key_hash_matches_server_bearer_hash() {
    // `openasr-cli` persists `openasr_core::apikeys::ApiKeyStore` hashes and
    // hands them to `ServerAuth::from_token_hashes`; the two hash functions
    // must stay identical (SHA-256 hex) or every configured key would
    // silently stop authorizing at the API boundary.
    let token = "oasr_sk_test-drift-check-token";
    let core_hash = openasr_core::apikeys::hash_api_key_token(token);
    let auth = ServerAuth::from_token_hashes([core_hash]);
    assert!(auth.authorizes(&header_map_with_bearer(token)));
}

fn resolved_pull_fixture() -> ResolvedCatalogPull {
    ResolvedCatalogPull {
        requested: "moonshine-tiny:q8".to_string(),
        model_id: "moonshine-tiny".to_string(),
        catalog_family_id: "moonshine".to_string(),
        display_name: "Moonshine Tiny".to_string(),
        quant: "q8_0".to_string(),
        suffix: "q8".to_string(),
        pull: "moonshine-tiny:q8".to_string(),
        filename: "moonshine-tiny-q8_0.oasr".to_string(),
        url: "https://huggingface.co/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-q8_0.oasr".to_string(),
        mirrors: Vec::new(),
        hf_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
        sha256: "a".repeat(64),
        size_bytes: 3,
        license: "MIT".to_string(),
        license_url: "https://huggingface.co/UsefulSensors/moonshine-tiny".to_string(),
        license_class: LicenseClass::Permissive,
    }
}

fn distribution_context_for_test(home: &std::path::Path) -> DistributionContext {
    DistributionContext::new(DistributionRuntime {
        openasr_home: Some(home.to_path_buf()),
        catalog_url: None,
        catalog_local_override: None,
    })
}

fn distribution_context_with_pull_license_for_test(
    root: &std::path::Path,
    license_class: LicenseClass,
) -> (DistributionContext, PathBuf) {
    let source_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model-registry/catalog.json");
    let contents = fs::read_to_string(source_path).expect("read bundled catalog fixture");
    let mut catalog: serde_json::Value =
        serde_json::from_str(&contents).expect("parse bundled catalog fixture");
    let model = catalog["models"]
        .as_array_mut()
        .expect("catalog models array")
        .iter_mut()
        .find(|model| model["id"] == "moonshine-tiny")
        .expect("moonshine-tiny catalog fixture");
    model["license"] = serde_json::Value::String(
        match &license_class {
            LicenseClass::Permissive => "MIT",
            LicenseClass::Noncommercial => "CC-BY-NC-4.0",
            LicenseClass::Gated => "Vendor gated license",
            LicenseClass::Unknown => "Unknown",
        }
        .to_string(),
    );
    model["license_url"] =
        serde_json::Value::String("https://example.invalid/model-license".to_string());
    model["license_class"] =
        serde_json::to_value(&license_class).expect("serialize license class fixture");

    let catalog_path = root.join("catalog.json");
    let contents = serde_json::to_string(&catalog).expect("serialize catalog fixture");
    openasr_core::testing::write_local_dev_signed_catalog(&catalog_path, &contents, 1);
    let home = root.join("home");
    let distribution = DistributionContext::new(DistributionRuntime {
        openasr_home: Some(home.clone()),
        catalog_url: Some(format!("file://{}", catalog_path.display())),
        catalog_local_override: None,
    });
    (distribution, home)
}

fn local_import_fixture_with_license(
    root: &std::path::Path,
    license_class: LicenseClass,
) -> (DistributionContext, PathBuf, PathBuf) {
    let pack_path = root.join("moonshine-tiny-q8_0.oasr");
    let spec = TinyGgufFixtureSpec::moonshine_oasr_v1_runtime_ready("moonshine-tiny");
    write_tiny_gguf_runtime_source(&pack_path, &spec)
        .expect("write moonshine local import fixture");
    let pack_bytes = fs::read(&pack_path).expect("read local import fixture");
    let pack_sha256 = format!("{:x}", Sha256::digest(&pack_bytes));

    let source_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model-registry/catalog.json");
    let contents = fs::read_to_string(source_path).expect("read bundled catalog fixture");
    let mut catalog: serde_json::Value =
        serde_json::from_str(&contents).expect("parse bundled catalog fixture");
    let model = catalog["models"]
        .as_array_mut()
        .expect("catalog models array")
        .iter_mut()
        .find(|model| model["id"] == "moonshine-tiny")
        .expect("moonshine-tiny catalog fixture");
    model["license"] = serde_json::Value::String(
        match &license_class {
            LicenseClass::Permissive => "MIT",
            LicenseClass::Noncommercial => "CC-BY-NC-4.0",
            LicenseClass::Gated => "Vendor gated license",
            LicenseClass::Unknown => "Unknown",
        }
        .to_string(),
    );
    model["license_url"] =
        serde_json::Value::String("https://example.invalid/model-license".to_string());
    model["license_class"] =
        serde_json::to_value(&license_class).expect("serialize license class fixture");
    let quant = model["quants"]
        .as_array_mut()
        .expect("model quants array")
        .iter_mut()
        .find(|quant| quant["quant"] == "q8_0")
        .expect("moonshine q8 catalog fixture");
    quant["sha256"] = serde_json::Value::String(pack_sha256);
    quant["size_bytes"] = serde_json::Value::Number(pack_bytes.len().into());

    let catalog_path = root.join("catalog.json");
    let contents = serde_json::to_string(&catalog).expect("serialize catalog fixture");
    openasr_core::testing::write_local_dev_signed_catalog(&catalog_path, &contents, 1);
    let home = root.join("home");
    let distribution = DistributionContext::new(DistributionRuntime {
        openasr_home: Some(home.clone()),
        catalog_url: Some(format!("file://{}", catalog_path.display())),
        catalog_local_override: None,
    });
    (distribution, home, pack_path)
}

/// Copies the real, committed `model-registry/catalog.json` into `dir` and
/// re-signs the copy with the public local-dev key for the exact `file://`
/// path the test will pass as `catalog_url`. The committed catalog's own
/// signature is bound to the production HTTPS identity
/// (`https://catalog.openasr.org/v1/catalog.json`), not to an arbitrary local
/// path, so a test that wants to load the real bundled catalog contents
/// through a local `--catalog-url` override must sign a fresh, path-bound
/// copy rather than pointing straight at the committed file + its committed
/// signature.
fn bundled_catalog_url_for_test(dir: &std::path::Path) -> String {
    let source_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model-registry/catalog.json");
    let contents = fs::read_to_string(&source_path).expect("read bundled catalog fixture");
    let copy_path = dir.join("bundled-catalog-for-test.json");
    openasr_core::testing::write_local_dev_signed_catalog(&copy_path, &contents, 1);
    format!("file://{}", copy_path.display())
}

/// Copies the real, committed, PRODUCTION-signed `model-registry/catalog.json`
/// and its `catalog.signature.json` pair -- byte-for-byte what desktop
/// bundles into `Contents/Resources` -- into `dir`, returning the copied
/// catalog path.
fn copy_bundled_production_catalog_to(dir: &std::path::Path) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model-registry");
    let catalog_path = dir.join("catalog.json");
    fs::copy(root.join("catalog.json"), &catalog_path).expect("copy bundled catalog.json");
    fs::copy(
        root.join(openasr_core::CATALOG_SIGNATURE_FILE_NAME),
        catalog_path.with_file_name(openasr_core::CATALOG_SIGNATURE_FILE_NAME),
    )
    .expect("copy bundled catalog.signature.json");
    catalog_path
}

/// RAII guard restoring `OPENASR_CATALOG_URL` / `OPENASR_CATALOG_FILE` /
/// `OPENASR_CATALOG_IDENTITY` to their prior values. These three env vars are
/// process-global, so tests that mutate them must run under `cargo nextest`
/// (one process per test, per this repo's AGENTS.md) -- a plain multi-threaded
/// `cargo test` run within one binary could race a concurrently-running test
/// that calls `DistributionRuntime::default()`.
struct CatalogEnvGuard {
    url: Option<String>,
    file: Option<String>,
    identity: Option<String>,
}

impl CatalogEnvGuard {
    fn capture() -> Self {
        Self {
            url: env::var("OPENASR_CATALOG_URL").ok(),
            file: env::var("OPENASR_CATALOG_FILE").ok(),
            identity: env::var("OPENASR_CATALOG_IDENTITY").ok(),
        }
    }

    /// Sets up the OLD desktop wiring this PR replaces: a bare
    /// `OPENASR_CATALOG_URL=file://<path>` override, using the install path
    /// as both fetch source and verification identity.
    fn set_url_override(url: &str) -> Self {
        let guard = Self::capture();
        unsafe {
            env::set_var("OPENASR_CATALOG_URL", url);
            env::remove_var("OPENASR_CATALOG_FILE");
            env::remove_var("OPENASR_CATALOG_IDENTITY");
        }
        guard
    }

    /// Sets up the NEW mechanism: bytes read from `path`, verified against
    /// the separately-declared `identity`.
    fn set_local_file_override(path: &std::path::Path, identity: &str) -> Self {
        let guard = Self::capture();
        unsafe {
            env::set_var("OPENASR_CATALOG_FILE", path);
            env::set_var("OPENASR_CATALOG_IDENTITY", identity);
            env::remove_var("OPENASR_CATALOG_URL");
        }
        guard
    }
}

impl Drop for CatalogEnvGuard {
    fn drop(&mut self) {
        unsafe {
            match self.url.take() {
                Some(value) => env::set_var("OPENASR_CATALOG_URL", value),
                None => env::remove_var("OPENASR_CATALOG_URL"),
            }
            match self.file.take() {
                Some(value) => env::set_var("OPENASR_CATALOG_FILE", value),
                None => env::remove_var("OPENASR_CATALOG_FILE"),
            }
            match self.identity.take() {
                Some(value) => env::set_var("OPENASR_CATALOG_IDENTITY", value),
                None => env::remove_var("OPENASR_CATALOG_IDENTITY"),
            }
        }
    }
}

/// `OPENASR_CATALOG_URL`/`OPENASR_CATALOG_FILE`/`OPENASR_CATALOG_IDENTITY` are
/// process-global env vars, and `cargo test` (unlike `cargo nextest`, this
/// repo's canonical runner, which isolates each test in its own process) runs
/// tests within one binary on multiple threads by default -- so the 3 tests
/// mutating these env vars below must serialize against each other or they
/// race and read each other's overrides. Same pattern as
/// `realtime::tests::speaker_embedder_env_lock`.
fn catalog_env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn desktop_bundled_catalog_via_bare_file_url_override_is_rejected() {
    let _lock = catalog_env_lock();
    // Reproduces the 0.1.13 desktop packaging regression at the ACTUAL server
    // wiring surface (not just the underlying core primitive): the old
    // `sidecar.rs::resolve_catalog_url` set
    // `OPENASR_CATALOG_URL=file:///Applications/.../catalog.json`, using the
    // install path as both fetch source and verification identity for the
    // exact production-signed catalog desktop bundles. `DistributionRuntime`
    // picks this up via `catalog_url`, and `catalog_source()` /
    // `load_catalog_for_source` must reject it -- this is the crash
    // ("Could not load model catalog...") desktop hit on every core >=
    // dcce58b build.
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    let catalog_path = copy_bundled_production_catalog_to(temp.path());
    let file_url = format!("file://{}", catalog_path.display());

    let _guard = CatalogEnvGuard::set_url_override(&file_url);
    let runtime = DistributionRuntime {
        openasr_home: Some(home.clone()),
        ..DistributionRuntime::default()
    };
    let distribution = DistributionContext::new(runtime);
    let source = distribution
        .catalog_source()
        .expect("OPENASR_CATALOG_URL override must be picked up");
    let error = load_catalog_for_source(source, &home)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("Catalog signature manifest URL mismatch")
            || error.contains("no usable signed cache"),
        "{error}"
    );
}

#[test]
fn desktop_bundled_catalog_via_file_and_identity_override_loads() {
    let _lock = catalog_env_lock();
    // The fix: the SAME bundled bytes, but the server picks up
    // `OPENASR_CATALOG_FILE` (bytes) + `OPENASR_CATALOG_IDENTITY` (the real
    // production identity the signature is bound to) instead of folding both
    // into a single `file://` URL. `catalog_source()` must prefer this local
    // override over `OPENASR_CATALOG_URL`, and the load must succeed.
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    let catalog_path = copy_bundled_production_catalog_to(temp.path());
    let identity = openasr_core::default_catalog_url();

    let _guard = CatalogEnvGuard::set_local_file_override(&catalog_path, identity);
    let runtime = DistributionRuntime {
        openasr_home: Some(home.clone()),
        ..DistributionRuntime::default()
    };
    let distribution = DistributionContext::new(runtime);
    let source = distribution
        .catalog_source()
        .expect("OPENASR_CATALOG_FILE/IDENTITY override must be picked up");
    let catalog = load_catalog_for_source(source, &home)
        .expect("bundled catalog + declared identity must verify");
    assert!(!catalog.models.is_empty());
}

#[test]
fn catalog_local_override_takes_precedence_over_catalog_url() {
    let _lock = catalog_env_lock();
    // If both `OPENASR_CATALOG_URL` and `OPENASR_CATALOG_FILE`/`_IDENTITY` are
    // set (e.g. a stale env left over from a different launch path), the
    // explicit local-file override wins -- it is the more specific,
    // more recently introduced configuration, and silently preferring the
    // legacy `catalog_url` here would resurrect the exact regression this PR
    // fixes for any caller that (redundantly) sets both.
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    let catalog_path = copy_bundled_production_catalog_to(temp.path());
    let identity = openasr_core::default_catalog_url();
    let bogus_file_url = format!("file://{}", catalog_path.display());

    let _url_guard = CatalogEnvGuard::set_url_override(&bogus_file_url);
    unsafe {
        env::set_var("OPENASR_CATALOG_FILE", &catalog_path);
        env::set_var("OPENASR_CATALOG_IDENTITY", identity);
    }
    let runtime = DistributionRuntime {
        openasr_home: Some(home.clone()),
        ..DistributionRuntime::default()
    };
    let distribution = DistributionContext::new(runtime);
    let source = distribution.catalog_source().expect("override present");
    assert!(matches!(source, CatalogSource::LocalFile { .. }));
    load_catalog_for_source(source, &home).expect("local override must win and verify");
}

fn write_valid_installed_pack_for_test(
    home: &Path,
    model_id: &str,
    quant: &str,
    suffix: &str,
) -> InstalledPack {
    let filename = format!("{model_id}-{quant}.oasr");
    let models = home.join("models");

    // Build the pack bytes somewhere disposable, then publish them the way the
    // store holds a model: an immutable object named by its digest, plus a ref.
    let scratch = models.join("fixture-source");
    fs::create_dir_all(&scratch).expect("create fixture dir");
    let staged = scratch.join(&filename);
    write_mock_gguf_runtime_source(&staged, Some(model_id));
    let bytes = fs::read(&staged).expect("read installed pack fixture");
    fs::remove_dir_all(&scratch).expect("drop fixture staging dir");

    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let path = models.join("objects/sha256").join(&sha256).join("content");
    fs::create_dir_all(path.parent().expect("object parent")).expect("create object dir");
    fs::write(&path, &bytes).expect("write object");

    let pack = InstalledPack {
        model_id: model_id.to_string(),
        display_name: model_id.to_string(),
        quant: quant.to_string(),
        suffix: suffix.to_string(),
        pull: format!("{model_id}:{suffix}"),
        filename,
        path,
        url: format!("https://example.test/{model_id}-{quant}.oasr"),
        hf_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
        sha256,
        size_bytes: bytes.len() as u64,
        installed_at_unix_seconds: 1,
        source: None,
    };
    let ref_path = models
        .join("refs")
        .join(model_id)
        .join(format!("{quant}.json"));
    fs::create_dir_all(ref_path.parent().expect("ref parent")).expect("create ref dir");
    fs::write(
        &ref_path,
        serde_json::to_string_pretty(&pack).expect("serialize installed pack"),
    )
    .expect("write model ref");
    pack
}

fn write_mock_gguf_runtime_source(path: &std::path::Path, metadata_model_id: Option<&str>) {
    // Use the graph-complete, tokenizer-complete whisper fixture (not the
    // bare `whisper_oasr_v1_non_streaming_cpu`, which deliberately omits the
    // whisper runtime scalar keys): `list_installed_packs` now re-validates
    // on-disk packs through the shared runtime verifier on every lookup, so
    // an "installed" test fixture must satisfy that contract or it silently
    // stops being recognized as installed.
    let spec = match metadata_model_id {
        None => TinyGgufFixtureSpec::new(Default::default()),
        Some(model_id) if model_id.starts_with("moonshine") => {
            TinyGgufFixtureSpec::moonshine_oasr_v1_runtime_ready(model_id)
        }
        Some(model_id) if model_id.starts_with("cohere") => {
            TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready(model_id)
        }
        Some(model_id) if model_id.starts_with("qwen") => {
            TinyGgufFixtureSpec::qwen3_asr_oasr_v1_runtime_ready(model_id)
        }
        Some(model_id) => {
            TinyGgufFixtureSpec::whisper_oasr_v1_graph_ready_for_runtime_fail_closed(model_id)
        }
    };
    write_tiny_gguf_runtime_source(path, &spec).expect("write mock gguf runtime source");
}

fn remote_transcription_multipart_body() -> (String, Vec<u8>) {
    let boundary = "openasr-loopback-boundary";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\nnot a real wav\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n--{boundary}--\r\n"
    )
    .into_bytes();
    (format!("multipart/form-data; boundary={boundary}"), body)
}

async fn connect_loopback_realtime_websocket(
    server: &LoopbackTlsServer,
    bearer_token: &str,
) -> tokio_tungstenite::WebSocketStream<tokio_rustls::client::TlsStream<TcpStream>> {
    try_connect_loopback_realtime_websocket(server, bearer_token)
        .await
        .unwrap()
}

async fn try_connect_loopback_realtime_websocket(
    server: &LoopbackTlsServer,
    bearer_token: &str,
) -> Result<
    tokio_tungstenite::WebSocketStream<tokio_rustls::client::TlsStream<TcpStream>>,
    tokio_tungstenite::tungstenite::Error,
> {
    let fingerprint = Arc::new(Mutex::new(None));
    let verifier = Arc::new(TestTofuVerifier {
        fingerprint: fingerprint.clone(),
    });
    let config =
        ClientConfig::builder_with_provider(rustls::crypto::ring::default_provider().into())
            .with_safe_default_protocol_versions()
            .unwrap()
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_no_client_auth();
    let stream = TcpStream::connect(server.addr).await.unwrap();
    let server_name = ServerName::try_from("localhost").unwrap().to_owned();
    let tls = TlsConnector::from(Arc::new(config))
        .connect(server_name, stream)
        .await
        .unwrap();
    assert_eq!(
        fingerprint
            .lock()
            .expect("fingerprint mutex poisoned")
            .clone()
            .expect("server certificate fingerprint"),
        server.certificate_fingerprint
    );

    let mut request = format!("wss://localhost:{}/v1/audio/realtime", server.addr.port())
        .into_client_request()
        .unwrap();
    request.headers_mut().insert(
        header::AUTHORIZATION,
        format!("Bearer {bearer_token}").parse().unwrap(),
    );
    request.headers_mut().insert(
        REMOTE_COMPUTE_HEADER,
        REMOTE_COMPUTE_CLIENT_VALUE.parse().unwrap(),
    );

    let (websocket, response) = tokio_tungstenite::client_async(request, tls).await?;
    assert_eq!(response.status(), StatusCode::SWITCHING_PROTOCOLS);
    Ok(websocket)
}

#[test]
fn parse_inference_threads_field_validates_bounds() {
    assert_eq!(parse_inference_threads_field("1").unwrap(), 1);
    assert_eq!(
        parse_inference_threads_field(&MAX_INFERENCE_THREADS.to_string()).unwrap(),
        MAX_INFERENCE_THREADS
    );

    for value in ["0", "257"] {
        let error = parse_inference_threads_field(value)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("inference_threads must be between 1 and 256"),
            "{error}"
        );
    }
}

#[test]
fn parse_execution_target_field_accepts_supported_targets() {
    assert_eq!(
        parse_execution_target_field("auto").unwrap(),
        ExecutionTarget::Auto
    );
    assert_eq!(
        parse_execution_target_field("cpu").unwrap(),
        ExecutionTarget::Cpu
    );
    assert_eq!(
        parse_execution_target_field("accelerated").unwrap(),
        ExecutionTarget::Accelerated
    );
    let error = parse_execution_target_field("gpu0")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("Unsupported execution_target 'gpu0'"),
        "{error}"
    );
}

#[test]
fn native_execution_target_mapping_preserves_server_request_semantics() {
    assert_eq!(
        native_hardware_target_from_execution_target(None),
        NativeAsrHardwareTarget::Auto
    );
    assert_eq!(
        native_hardware_target_from_execution_target(Some(ExecutionTarget::Auto)),
        NativeAsrHardwareTarget::Auto
    );
    assert_eq!(
        native_hardware_target_from_execution_target(Some(ExecutionTarget::Cpu)),
        NativeAsrHardwareTarget::Cpu
    );
    assert_eq!(
        native_hardware_target_from_execution_target(Some(ExecutionTarget::Accelerated)),
        NativeAsrHardwareTarget::Accelerated
    );
}

#[test]
fn default_pack_lookup_resolves_series_alias_through_catalog() {
    let temp = tempfile::tempdir().unwrap();
    let pack = write_valid_installed_pack_for_test(temp.path(), "qwen3-asr-0.6b", "q8_0", "q8");
    let catalog_url = bundled_catalog_url_for_test(temp.path());

    let resolved = find_installed_pack_reference(
        temp.path(),
        Some(CatalogSource::Url(&catalog_url)),
        "qwen:q8",
    )
    .unwrap()
    .unwrap();

    assert_eq!(resolved.pull, pack.pull);
}

#[test]
fn form_model_resolution_preserves_native_request_id() {
    let temp = tempfile::tempdir().unwrap();
    let catalog_url = bundled_catalog_url_for_test(temp.path());
    let catalog = load_model_catalog(Some(&catalog_url), temp.path()).unwrap();

    let native_model =
        resolve_and_validate_form_model_id("qwen:q8", BackendKind::Native, Some(&catalog)).unwrap();
    assert_eq!(native_model, "qwen:q8");

    let mock_model =
        resolve_and_validate_form_model_id("qwen:q8", BackendKind::Mock, Some(&catalog)).unwrap();
    assert_eq!(mock_model, "qwen3-asr-0.6b");
}

#[test]
fn self_signed_tls_defaults_to_localhost_and_reports_certificate_fingerprint() {
    assert_eq!(
        ServerTlsConfig::self_signed(["", "  "]),
        ServerTlsConfig::SelfSigned {
            subject_alt_names: vec!["localhost".to_string()]
        }
    );

    let identity = self_signed_tls_identity(&["localhost".to_string()]).unwrap();
    assert_eq!(identity.certificate_sha256.len(), 64);
    assert!(
        identity
            .certificate_sha256
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    );
    assert_eq!(
        identity.certificate_sha256,
        hex_encode(&Sha256::digest(identity.certificate_der.as_ref()))
    );
    assert_eq!(
        identity.pairing_safety_code,
        pairing_safety_code_for_certificate_fingerprint(&identity.certificate_sha256)
    );
    assert_eq!(identity.pairing_safety_code.len(), "ABCD-1234".len());
}

#[test]
fn load_or_generate_self_signed_tls_identity_loads_persisted_identity() {
    let temp = tempfile::tempdir().unwrap();
    let store_path = temp.path().join("tls-identity.json");
    let sans = vec!["127.0.0.1".to_string()];

    let first = load_or_generate_self_signed_tls_identity(&sans, Some(&store_path)).unwrap();
    // A second call against the same store must load the persisted keypair +
    // certificate back rather than minting a new one -- this is the crux of
    // "restart does not rotate the pairing fingerprint".
    let second = load_or_generate_self_signed_tls_identity(&sans, Some(&store_path)).unwrap();

    assert_eq!(first.certificate_sha256, second.certificate_sha256);
    assert_eq!(first.certificate_der, second.certificate_der);
    assert_eq!(first.pairing_safety_code, second.pairing_safety_code);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(&store_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "persisted TLS identity file must be owner-only"
        );
    }
}

#[test]
fn load_or_generate_self_signed_tls_identity_generates_and_persists_when_store_missing() {
    let temp = tempfile::tempdir().unwrap();
    // Deliberately does not exist yet -- a fresh install / first ever
    // --tls-self-signed run.
    let store_path = temp.path().join("tls-identity.json");
    let sans = vec!["localhost".to_string()];
    assert!(!store_path.exists());

    let identity = load_or_generate_self_signed_tls_identity(&sans, Some(&store_path)).unwrap();

    assert!(store_path.exists());
    let persisted: PersistedTlsIdentity =
        serde_json::from_slice(&fs::read(&store_path).unwrap()).unwrap();
    assert_eq!(persisted.subject_alt_names, sans);
    assert_eq!(
        certificate_fingerprint_sha256(&persisted.certificate_der),
        identity.certificate_sha256
    );
    assert!(persisted.not_after_unix_secs > unix_now_secs());
}

#[test]
fn load_or_generate_self_signed_tls_identity_regenerates_on_corrupt_store() {
    let temp = tempfile::tempdir().unwrap();
    let store_path = temp.path().join("tls-identity.json");
    // Present but not valid JSON at all -- simulates disk corruption /
    // truncation, distinct from "file does not exist".
    fs::write(&store_path, b"not valid json { at all").unwrap();
    let sans = vec!["localhost".to_string()];

    // Must fail closed by regenerating rather than propagating the parse
    // error or, worse, serving with unusable key material.
    let identity = load_or_generate_self_signed_tls_identity(&sans, Some(&store_path)).unwrap();

    assert_eq!(identity.certificate_sha256.len(), 64);
    // The corrupt file must have been overwritten with a freshly generated,
    // well-formed identity -- not left corrupt for the next boot to trip over
    // again.
    let persisted: PersistedTlsIdentity =
        serde_json::from_slice(&fs::read(&store_path).unwrap()).unwrap();
    assert_eq!(
        certificate_fingerprint_sha256(&persisted.certificate_der),
        identity.certificate_sha256
    );
}

#[test]
fn load_or_generate_self_signed_tls_identity_regenerates_on_expired_certificate() {
    let temp = tempfile::tempdir().unwrap();
    let store_path = temp.path().join("tls-identity.json");
    let sans = vec!["localhost".to_string()];

    let (certificate_der, private_key_der, ..) = generate_self_signed_tls_material(&sans).unwrap();
    let expired_fingerprint = certificate_fingerprint_sha256(&certificate_der);
    let expired = PersistedTlsIdentity {
        subject_alt_names: sans.clone(),
        certificate_der,
        private_key_der,
        // Both bounds safely in the past: an already-expired identity, not
        // merely "expiring soon".
        not_before_unix_secs: unix_now_secs().saturating_sub(3600),
        not_after_unix_secs: unix_now_secs().saturating_sub(60),
    };
    fs::write(&store_path, serde_json::to_vec_pretty(&expired).unwrap()).unwrap();

    let identity = load_or_generate_self_signed_tls_identity(&sans, Some(&store_path)).unwrap();

    // A genuinely new identity was minted, not the expired one reused.
    assert_ne!(identity.certificate_sha256, expired_fingerprint);
    let persisted: PersistedTlsIdentity =
        serde_json::from_slice(&fs::read(&store_path).unwrap()).unwrap();
    assert!(persisted.not_after_unix_secs > unix_now_secs());
}

#[cfg(unix)]
#[test]
fn load_or_generate_self_signed_tls_identity_hardens_openasr_home_to_0700() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("openasr-home");
    // Simulate an OPENASR_HOME that predates this PR (or was widened by some
    // other tool/umask): world-traversable 0755, the `create_dir_all`
    // default under a typical 022 umask.
    fs::create_dir_all(&home).unwrap();
    fs::set_permissions(&home, fs::Permissions::from_mode(0o755)).unwrap();
    let store_path = home.join("tls-identity.json");
    let sans = vec!["localhost".to_string()];

    load_or_generate_self_signed_tls_identity(&sans, Some(&store_path)).unwrap();

    let mode = fs::metadata(&home).unwrap().permissions().mode() & 0o777;
    assert_eq!(
        mode, 0o700,
        "OPENASR_HOME must be tightened to owner-only even when it already existed wider"
    );
}

#[cfg(unix)]
#[test]
fn load_or_generate_self_signed_tls_identity_creates_and_hardens_missing_openasr_home() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    // Deliberately does not exist yet -- the TLS identity store has no other
    // writer upstream (unlike apikeys.json/pairing-registry.json) guaranteed
    // to have created OPENASR_HOME first.
    let home = temp.path().join("openasr-home");
    let store_path = home.join("tls-identity.json");
    let sans = vec!["localhost".to_string()];
    assert!(!home.exists());

    load_or_generate_self_signed_tls_identity(&sans, Some(&store_path)).unwrap();

    let mode = fs::metadata(&home).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o700);
}

#[test]
fn load_or_generate_self_signed_tls_identity_regenerates_on_corrupt_der_inside_valid_json() {
    let temp = tempfile::tempdir().unwrap();
    let store_path = temp.path().join("tls-identity.json");
    let sans = vec!["localhost".to_string()];

    // The JSON envelope itself is well-formed (unlike
    // `..._regenerates_on_corrupt_store`, which corrupts the JSON layer) --
    // only the DER payloads inside it are garbage, simulating a truncated
    // write or a bit-flipped disk that still leaves valid-looking JSON
    // structure. `load_persisted_tls_identity` only checks the DER fields are
    // non-empty, so this reaches `tls_identity_from_der` and must be handled
    // there (this is the regression test for S1: before the fix, the `?`
    // inside `tls_identity_from_der`'s rustls `with_single_cert` call
    // propagated straight out of `load_or_generate_self_signed_tls_identity`,
    // failing serve's startup instead of rotating).
    let corrupt = PersistedTlsIdentity {
        subject_alt_names: sans.clone(),
        certificate_der: vec![1, 2, 3, 4, 5, 6, 7, 8],
        private_key_der: vec![8, 7, 6, 5, 4, 3, 2, 1],
        not_before_unix_secs: unix_now_secs().saturating_sub(3600),
        not_after_unix_secs: unix_now_secs() + 3600,
    };
    fs::write(&store_path, serde_json::to_vec_pretty(&corrupt).unwrap()).unwrap();

    let identity = load_or_generate_self_signed_tls_identity(&sans, Some(&store_path))
        .expect("a DER-corrupt-but-JSON-valid store must regenerate, not fail startup");

    assert_eq!(identity.certificate_sha256.len(), 64);
    // The corrupt DER must have been overwritten with a freshly generated,
    // internally-consistent identity, not left in place for the next boot to
    // trip over again.
    let persisted: PersistedTlsIdentity =
        serde_json::from_slice(&fs::read(&store_path).unwrap()).unwrap();
    assert_eq!(
        certificate_fingerprint_sha256(&persisted.certificate_der),
        identity.certificate_sha256
    );
    // The regenerated identity must actually build into a usable rustls
    // config, i.e. round-trips through `tls_identity_from_der` cleanly.
    tls_identity_from_der(persisted.certificate_der, persisted.private_key_der)
        .expect("regenerated identity must itself load back as a valid keypair/certificate");
}

#[test]
fn load_or_generate_self_signed_tls_identity_regenerates_on_key_cert_mismatch() {
    let temp = tempfile::tempdir().unwrap();
    let store_path = temp.path().join("tls-identity.json");
    let sans = vec!["localhost".to_string()];

    // Each half is individually well-formed DER, but the private key does
    // not correspond to the certificate's public key -- rustls's
    // `with_single_cert` documents that it fails in exactly this case ("if
    // the SubjectPublicKeyInfo from the private key does not match the
    // public key for the end-entity certificate"). Simulates one field of a
    // persisted identity being replaced/corrupted independently of the
    // other.
    let (certificate_der, _matching_key_der, ..) =
        generate_self_signed_tls_material(&sans).unwrap();
    let (_other_certificate_der, mismatched_key_der, ..) =
        generate_self_signed_tls_material(&sans).unwrap();
    let mismatched = PersistedTlsIdentity {
        subject_alt_names: sans.clone(),
        certificate_der,
        private_key_der: mismatched_key_der,
        not_before_unix_secs: unix_now_secs().saturating_sub(3600),
        not_after_unix_secs: unix_now_secs() + 3600,
    };
    fs::write(&store_path, serde_json::to_vec_pretty(&mismatched).unwrap()).unwrap();

    let identity = load_or_generate_self_signed_tls_identity(&sans, Some(&store_path))
        .expect("a key/cert mismatch must regenerate, not fail startup");

    let persisted: PersistedTlsIdentity =
        serde_json::from_slice(&fs::read(&store_path).unwrap()).unwrap();
    assert_eq!(
        certificate_fingerprint_sha256(&persisted.certificate_der),
        identity.certificate_sha256
    );
    tls_identity_from_der(persisted.certificate_der, persisted.private_key_der)
        .expect("regenerated identity must have a matching key and certificate");
}

/// `write_bytes_atomically`'s rename is atomic, but there is no cross-process
/// file lock around "read store, decide to (re)generate, write store" -- the
/// TLS identity store has the same gap the review flagged for
/// `persist_pairing_credentials_locked` (whose `_locked` suffix is an
/// in-process `Mutex`, not an `flock`). Two daemons racing their first
/// `--tls-self-signed` start against the same `OPENASR_HOME` can each
/// generate their own identity and each call `persist_tls_identity`; the
/// atomic rename means the loser's write is fully overwritten (never a
/// torn/partial file), but the two in-memory server processes end up serving
/// *different* certificates for one boot cycle, and only one of the two
/// generated identities survives on disk.
///
/// This is a known, documented gap (see `load_or_generate_self_signed_tls_identity`'s
/// module-level discussion and the PR description) rather than something this
/// test suite adds cross-process locking for. What *is* guaranteed, and what
/// this test pins down, is that the loser's overwrite never corrupts the
/// store into something unusable: whichever identity's `persist_tls_identity`
/// call won the race is a complete, well-formed, self-consistent identity,
/// and the next daemon start (no race this time) loads it back rather than
/// tripping the corrupt-store regeneration path.
#[test]
fn concurrent_first_boot_race_self_heals_to_the_last_writer_on_next_start() {
    let temp = tempfile::tempdir().unwrap();
    let store_path = temp.path().join("tls-identity.json");
    let sans = vec!["localhost".to_string()];

    // Two "processes" independently generate an identity before either has
    // written anything -- the miss-then-generate race window.
    let (first_cert, first_key, first_not_before, first_not_after) =
        generate_self_signed_tls_material(&sans).unwrap();
    let (second_cert, second_key, second_not_before, second_not_after) =
        generate_self_signed_tls_material(&sans).unwrap();
    let first_fingerprint = certificate_fingerprint_sha256(&first_cert);
    let second_fingerprint = certificate_fingerprint_sha256(&second_cert);
    assert_ne!(
        first_fingerprint, second_fingerprint,
        "two independent generations must not coincidentally collide"
    );

    // Both persist, "first" then "second" -- last-writer-wins via atomic
    // rename (see `write_bytes_atomically` / `openasr_core::write_owner_only_file_atomically`).
    persist_tls_identity(
        &store_path,
        &sans,
        &first_cert,
        &first_key,
        first_not_before,
        first_not_after,
    );
    persist_tls_identity(
        &store_path,
        &sans,
        &second_cert,
        &second_key,
        second_not_before,
        second_not_after,
    );

    // Next daemon start (no concurrent writer this time) must load the
    // survivor back cleanly -- not trip the corrupt-store/DER-mismatch
    // regeneration path added for S1, and not silently keep serving the
    // loser's in-memory identity forever.
    let loaded = load_or_generate_self_signed_tls_identity(&sans, Some(&store_path)).unwrap();
    assert_eq!(loaded.certificate_sha256, second_fingerprint);
    assert_ne!(loaded.certificate_sha256, first_fingerprint);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(&store_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
}

#[tokio::test]
async fn loopback_tls_pairing_device_transcription_skips_server_history() {
    let temp = tempfile::tempdir().unwrap();
    let server = spawn_loopback_pairing_server(temp.path()).await;
    let credential = approve_loopback_pairing(&server).await;
    let bearer_auth = bearer_auth_header(&credential.bearer_token);

    let (content_type, body) = remote_transcription_multipart_body();
    let transcription = https_request(
        server.addr,
        "POST",
        "/v1/audio/transcriptions",
        &[
            ("Authorization", bearer_auth.as_str()),
            ("X-OpenASR-Remote-Compute", "client"),
            ("Content-Type", &content_type),
        ],
        body,
    )
    .await;
    assert_eq!(transcription.status, 200);
    let transcription_text = String::from_utf8(transcription.body).unwrap();
    assert!(transcription_text.contains("OpenASR mock transcription"));

    // S2: a paired *device* token is limited to compute — it cannot read the
    // operator's local history.
    let device_history = https_request(
        server.addr,
        "GET",
        "/v1/history",
        &[("Authorization", bearer_auth.as_str())],
        Vec::new(),
    )
    .await;
    assert_eq!(device_history.status, 403);

    // The admin token can read history, confirming the device transcript was
    // NOT recorded (the history-skip held).
    let history = https_request(
        server.addr,
        "GET",
        "/v1/history",
        &[("Authorization", "Bearer admin-secret")],
        Vec::new(),
    )
    .await;
    assert_eq!(history.status, 200);
    let history_json: serde_json::Value = serde_json::from_slice(&history.body).unwrap();
    assert_eq!(history_json["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn loopback_tls_pairing_device_realtime_skips_server_history() {
    let temp = tempfile::tempdir().unwrap();
    let server = spawn_loopback_pairing_server(temp.path()).await;
    let credential = approve_loopback_pairing(&server).await;
    let bearer_auth = bearer_auth_header(&credential.bearer_token);
    let mut websocket =
        connect_loopback_realtime_websocket(&server, &credential.bearer_token).await;

    let first = websocket
        .next()
        .await
        .expect("server sends realtime capabilities")
        .expect("realtime capabilities frame");
    match first {
        WsMessage::Text(text) => {
            let event: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(event["type"], "session.capabilities");
            assert_eq!(event["capabilities"]["supports_realtime_sessions"], true);
        }
        other => panic!("expected text capabilities frame, got {other:?}"),
    }

    websocket
        .send(WsMessage::Close(None))
        .await
        .expect("close realtime websocket");

    // S2: a paired *device* token is limited to compute — it cannot read the
    // operator's local history.
    let device_history = https_request(
        server.addr,
        "GET",
        "/v1/history",
        &[("Authorization", bearer_auth.as_str())],
        Vec::new(),
    )
    .await;
    assert_eq!(device_history.status, 403);

    // The admin token can read history, confirming the device transcript was
    // NOT recorded (the history-skip held).
    let history = https_request(
        server.addr,
        "GET",
        "/v1/history",
        &[("Authorization", "Bearer admin-secret")],
        Vec::new(),
    )
    .await;
    assert_eq!(history.status, 200);
    let history_json: serde_json::Value = serde_json::from_slice(&history.body).unwrap();
    assert_eq!(history_json["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn loopback_tls_revoked_pairing_device_cannot_access_remote_compute() {
    let temp = tempfile::tempdir().unwrap();
    let server = spawn_loopback_pairing_server(temp.path()).await;
    let credential = approve_loopback_pairing(&server).await;
    let bearer_auth = bearer_auth_header(&credential.bearer_token);
    revoke_loopback_pairing(&server, &credential.device_id).await;

    let (content_type, body) = remote_transcription_multipart_body();
    let transcription = https_request(
        server.addr,
        "POST",
        "/v1/audio/transcriptions",
        &[
            ("Authorization", bearer_auth.as_str()),
            ("X-OpenASR-Remote-Compute", "client"),
            ("Content-Type", &content_type),
        ],
        body,
    )
    .await;
    assert_eq!(transcription.status, 401);

    let error =
        match try_connect_loopback_realtime_websocket(&server, &credential.bearer_token).await {
            Ok(_) => panic!("revoked remote-compute token must not upgrade realtime websocket"),
            Err(error) => error,
        };
    assert!(error.to_string().contains("401"));
}

#[test]
fn pairing_device_authorization_updates_last_seen() {
    let auth = ServerAuth::pairing("admin-secret");
    let request = auth.create_pairing_request("MacBook Air").unwrap();
    let approved = auth.approve_pairing_request(&request.request_id).unwrap();
    let PairingCredentialState::Ready(credential) =
        auth.pairing_credential(&request.request_id).unwrap()
    else {
        panic!("expected approved pairing credential");
    };

    let before = auth.paired_devices().unwrap();
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].device_id, approved.device_id);
    assert_eq!(before[0].last_seen_unix_secs, None);

    let mut headers = HeaderMap::new();
    headers.insert(
        header::AUTHORIZATION,
        axum::http::HeaderValue::from_str(&format!("Bearer {}", credential.bearer_token)).unwrap(),
    );
    headers.insert(
        REMOTE_COMPUTE_HEADER,
        axum::http::HeaderValue::from_static(REMOTE_COMPUTE_CLIENT_VALUE),
    );
    assert!(auth.authorizes(&headers));
    assert!(is_remote_compute_client_request(&headers, &auth));

    let after = auth.paired_devices().unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].device_id, approved.device_id);
    assert!(after[0].last_seen_unix_secs.is_some());

    let mut admin_headers = HeaderMap::new();
    admin_headers.insert(
        header::AUTHORIZATION,
        axum::http::HeaderValue::from_static("Bearer admin-secret"),
    );
    admin_headers.insert(
        REMOTE_COMPUTE_HEADER,
        axum::http::HeaderValue::from_static(REMOTE_COMPUTE_CLIENT_VALUE),
    );
    assert!(auth.authorizes(&admin_headers));
    assert!(!is_remote_compute_client_request(&admin_headers, &auth));
}

#[test]
fn pairing_ops_recover_from_a_poisoned_registry_mutex_instead_of_crashing() {
    let auth = ServerAuth::pairing("admin-secret");
    let first = auth.create_pairing_request("Device A").unwrap();
    auth.approve_pairing_request(&first.request_id).unwrap();

    // Poison the pairing mutex the way a panic mid-mutation would: a thread
    // panics while holding the lock. Previously every later pairing op did
    // `.lock().expect(...)`, so this would permanently crash the server on the
    // next pairing request (server-wide DoS).
    let registry = auth.pairing.clone();
    let panicked = std::thread::spawn(move || {
        let _guard = registry.lock().unwrap();
        panic!("intentional poison for the recovery test");
    })
    .join();
    assert!(
        panicked.is_err(),
        "helper thread must panic to poison the mutex"
    );
    assert!(
        auth.pairing.is_poisoned(),
        "the pairing mutex must be poisoned now"
    );

    // Every pairing entry point must now RECOVER (via lock_pairing) and keep
    // serving, with prior state intact, rather than panic.
    let devices = auth.paired_devices().expect("list devices after poison");
    assert_eq!(devices.len(), 1, "the pre-poison approved device survives");
    let second = auth
        .create_pairing_request("Device B")
        .expect("create request after poison");
    auth.approve_pairing_request(&second.request_id)
        .expect("approve after poison");
    // reject also goes through lock_pairing; the already-approved id is no
    // longer pending, so it recovers and returns Ok(false) rather than panic.
    assert!(
        !auth
            .reject_pairing_request(&first.request_id)
            .expect("reject after poison"),
        "already-approved id is no longer pending"
    );
    assert_eq!(
        auth.paired_devices()
            .expect("list after second approve")
            .len(),
        2
    );
}

#[test]
fn pairing_credentials_and_revocations_survive_restart_and_claims_are_one_time() {
    let temp = tempfile::tempdir().unwrap();
    let store = temp.path().join("pairing-registry.json");

    let auth = ServerAuth::pairing("admin-secret").with_pairing_store(store.clone());
    let request = auth.create_pairing_request("Persisted Device").unwrap();
    auth.approve_pairing_request(&request.request_id).unwrap();

    // One-time claim: the first fetch yields the plaintext token, the second
    // must be gone (no replay, no lingering plaintext).
    let PairingCredentialState::Ready(claim) =
        auth.pairing_credential(&request.request_id).unwrap()
    else {
        panic!("expected approved pairing credential");
    };
    let device_token = claim.bearer_token.clone();
    let device_id = claim.device_id.clone();
    assert!(matches!(
        auth.pairing_credential(&request.request_id),
        Err(PairingError::NotFound)
    ));
    let token_hash = bearer_token_hash(&device_token);
    assert!(auth.pairing_authorizes_token_hash(&token_hash));

    // A fresh server instance bound to the same store reloads the credential,
    // so a paired device survives the remote-server restart the desktop does.
    let reloaded = ServerAuth::pairing("admin-secret").with_pairing_store(store.clone());
    assert!(reloaded.pairing_authorizes_token_hash(&token_hash));

    // Revocation must also be durable across a restart.
    assert!(reloaded.revoke_pairing_credential(&device_id).unwrap());
    let after_revoke = ServerAuth::pairing("admin-secret").with_pairing_store(store);
    assert!(!after_revoke.pairing_authorizes_token_hash(&token_hash));
}

#[test]
fn operator_only_paths_cover_history_config_and_model_mutations() {
    use axum::http::Method;
    // Operator-only (paired device token gets 403 in pairing mode):
    assert!(is_operator_only_path(&Method::GET, "/v1/history"));
    assert!(is_operator_only_path(&Method::DELETE, "/v1/history/abc"));
    assert!(is_operator_only_path(&Method::PUT, "/v1/config"));
    assert!(is_operator_only_path(&Method::GET, "/v1/config"));
    assert!(is_operator_only_path(&Method::POST, "/v1/models/default"));
    assert!(is_operator_only_path(&Method::DELETE, "/v1/models/whisper"));
    assert!(is_operator_only_path(
        &Method::POST,
        "/v1/models/whisper/pull"
    ));
    assert!(is_operator_only_path(
        &Method::POST,
        "/v1/models/local/import"
    ));
    assert!(is_operator_only_path(
        &Method::POST,
        "/v1/models/pull/job1/cancel"
    ));
    // Listing every in-flight job is broader exposure than a single-job GET
    // (which requires already knowing the job id), so it stays operator-only
    // even though the underlying handler is a GET.
    assert!(is_operator_only_path(&Method::GET, "/v1/models/pulls"));
    assert!(is_operator_only_path(&Method::GET, "/v1/voice-id/persons"));
    assert!(is_operator_only_path(&Method::POST, "/v1/voice-id/persons"));
    assert!(is_operator_only_path(
        &Method::PATCH,
        "/v1/voice-id/persons/person_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    ));
    assert!(is_operator_only_path(
        &Method::DELETE,
        "/v1/voice-id/samples/sample_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    ));
    // Open to paired compute clients:
    assert!(!is_operator_only_path(&Method::GET, "/v1/models/default"));
    assert!(!is_operator_only_path(&Method::GET, "/v1/models"));
    assert!(!is_operator_only_path(&Method::GET, "/v1/models/local"));
    assert!(!is_operator_only_path(&Method::GET, "/v1/capabilities"));
    assert!(!is_operator_only_path(
        &Method::POST,
        "/v1/audio/transcriptions"
    ));
    // The OpenAI-compat translations alias is a compute route, not operator-only.
    assert!(!is_operator_only_path(
        &Method::POST,
        "/v1/audio/translations"
    ));
    assert!(!is_operator_only_path(&Method::GET, "/v1/models/pull/job1"));
}

#[tokio::test]
async fn delete_model_allows_current_default_and_clears_default_selection() {
    let temp = tempfile::tempdir().unwrap();
    let pack = write_valid_installed_pack_for_test(temp.path(), "moonshine-tiny", "q8_0", "q8");
    persist_default_pack(temp.path(), &pack, QuantPreference::pinned(&pack.quant)).unwrap();
    let distribution = distribution_context_for_test(temp.path());
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack.path.clone()).into(),
    };

    let response = delete_model(
        State(runtime.clone()),
        AxumPath(pack.pull.clone()),
        Extension(distribution.clone()),
    )
    .await
    .unwrap();
    let response = response.0;

    assert!(response.deleted);
    assert_eq!(
        response.pack.as_ref().map(|pack| pack.pull.as_str()),
        Some("moonshine-tiny:q8")
    );
    assert!(list_installed_packs(temp.path()).unwrap().is_empty());
    let active = runtime.model_pack_path.current();
    let default = default_model_response(
        temp.path(),
        distribution.catalog_source(),
        active.as_deref(),
    )
    .unwrap();
    assert!(default.default_model.is_none());
    assert!(default.default_pull.is_none());
    assert!(default.pack.is_none());
    assert_eq!(default.default_model_status, "unset");
    assert_eq!(default.activation, DefaultModelActivationState::Unavailable);
    assert!(runtime.model_pack_path.current().is_none());
    let cleared =
        openasr_core::default_selection::read_active_model_selection_v2(temp.path()).unwrap();
    assert!(
        cleared.as_ref().is_none_or(|record| {
            record.status == openasr_core::default_selection::ActiveModelSelectionStatus::Unset
                && record.pull.is_none()
        }),
        "deleting the current default must clear durable V2: {cleared:?}"
    );
}

#[tokio::test]
async fn default_model_response_reports_installed_not_installed_and_unset() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = distribution_context_for_test(temp.path());

    let unset = default_model_response(temp.path(), distribution.catalog_source(), None).unwrap();
    assert_eq!(unset.default_model_status, "unset");
    assert!(unset.pack.is_none());
    assert_eq!(unset.activation, DefaultModelActivationState::Unavailable);

    let mut document = openasr_core::load_config_document(temp.path()).unwrap();
    document.config.default_model = Some("whisper-small".to_string());
    openasr_core::save_config_document(temp.path(), &document).unwrap();
    let not_installed =
        default_model_response(temp.path(), distribution.catalog_source(), None).unwrap();
    assert_eq!(not_installed.default_model_status, "not_installed");
    assert_eq!(
        not_installed.default_model.as_deref(),
        Some("whisper-small")
    );
    assert!(not_installed.pack.is_none());
    assert_eq!(
        not_installed.activation,
        DefaultModelActivationState::Unavailable
    );

    let pack = write_valid_installed_pack_for_test(temp.path(), "whisper-small", "q8_0", "q8");
    persist_default_pack(temp.path(), &pack, QuantPreference::pinned(&pack.quant)).unwrap();
    let installed = default_model_response(
        temp.path(),
        distribution.catalog_source(),
        Some(pack.path.as_path()),
    )
    .unwrap();
    assert_eq!(installed.default_model_status, "installed");
    assert_eq!(
        installed.pack.as_ref().map(|pack| pack.pull.as_str()),
        Some("whisper-small:q8")
    );
    assert_eq!(installed.activation, DefaultModelActivationState::Committed);
}

#[test]
fn transcription_preferences_fill_missing_thread_request_only() {
    let preferences = Preferences {
        inference_threads: Some(6),
        voice_id_segmenter: openasr_core::config::VoiceIdSegmenterPreference::Segmentation3_0,
        ..Default::default()
    };
    let mut request = TranscriptionRequest::new("fixtures/jfk.wav", "whisper-large-v3-turbo");

    apply_transcription_preferences(&mut request, &preferences);
    assert_eq!(request.inference_threads, Some(6));
    assert_eq!(
        request.voice_id_segmenter,
        openasr_core::config::VoiceIdSegmenterPreference::Segmentation3_0
    );

    request.inference_threads = Some(2);
    apply_transcription_preferences(&mut request, &preferences);
    assert_eq!(request.inference_threads, Some(2));
}

#[test]
fn record_file_transcription_history_round_trips_structured_metadata() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = distribution_context_for_test(temp.path());
    // auto_save only controls transcript-file exports; history recording is
    // governed by history_retention alone, so auto_save=false must still record.
    std::fs::write(
        temp.path().join("config.json"),
        serde_json::json!({
            "preferences": { "auto_save": false, "history_retention": "last5" }
        })
        .to_string(),
    )
    .unwrap();
    let request = TranscriptionRequest::new(temp.path().join("sample.wav"), "qwen3-asr-0.6b:q8")
        .with_display_file_name(Some("sample.wav".to_string()))
        .with_voice_id(true);
    let transcription = Transcription {
        truncated_decodes: Vec::new(),
        unnamed_speakers: Vec::new(),
        text: "hello with speaker".to_string(),
        segments: vec![openasr_core::Segment {
            start: 0.0,
            end: 2.0,
            text: "hello with speaker".to_string(),
            speaker: Some("Alice".to_string()),
            speaker_label: Some("SPEAKER_00".to_string()),
            speaker_person_id: None,
            speaker_snapshot_label: None,
            words: Vec::new(),
        }],
        longform: None,
        language: None,
        ..Default::default()
    };

    record_file_transcription_history(&distribution, &request, &transcription, ResponseFormat::Vtt)
        .unwrap();

    let store = DaemonHistoryStore::open(temp.path());
    let entries = store.list().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].output_format, Some(ResponseFormat::Vtt));
    assert_eq!(entries[0].diarization_active, Some(true));
    assert_eq!(
        entries[0].provenance,
        Some(DaemonHistoryProvenance::Recorded)
    );

    let detail = store.get(&entries[0].id).unwrap().unwrap();
    assert_eq!(detail.text, "hello with speaker");
    assert_eq!(detail.entry.output_format, Some(ResponseFormat::Vtt));
    assert_eq!(detail.entry.diarization_active, Some(true));
    assert_eq!(
        detail.entry.provenance,
        Some(DaemonHistoryProvenance::Recorded)
    );
}

#[test]
fn record_file_transcription_history_skips_write_when_retention_off() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = distribution_context_for_test(temp.path());
    // Even with auto_save enabled, "off" retention must skip the write:
    // history_retention is the only history switch.
    std::fs::write(
        temp.path().join("config.json"),
        serde_json::json!({
            "preferences": { "auto_save": true, "history_retention": "off" }
        })
        .to_string(),
    )
    .unwrap();
    let request = TranscriptionRequest::new(temp.path().join("sample.wav"), "qwen3-asr-0.6b:q8");
    let transcription = Transcription {
        truncated_decodes: Vec::new(),
        unnamed_speakers: Vec::new(),
        text: "never stored".to_string(),
        segments: Vec::new(),
        longform: None,
        language: None,
        ..Default::default()
    };

    record_file_transcription_history(
        &distribution,
        &request,
        &transcription,
        ResponseFormat::Text,
    )
    .unwrap();

    let store = DaemonHistoryStore::open(temp.path());
    assert!(store.list().unwrap().is_empty());
}

#[test]
fn history_retention_last5_prunes_store() {
    let temp = tempfile::tempdir().unwrap();
    let store = DaemonHistoryStore::open(temp.path());
    for index in 0..6 {
        store
            .record(DaemonHistoryRecord {
                kind: DaemonHistoryKind::File,
                model: "whisper-large-v3-turbo".to_string(),
                source_name: Some(format!("sample-{index}.wav")),
                duration_seconds: None,
                output_format: Some(ResponseFormat::Text),
                diarization_active: Some(false),
                provenance: Some(DaemonHistoryProvenance::Recorded),
                segments: Vec::new(),
                subtitle_cues: Vec::new(),
                timeline_quality: None,
                text: format!("transcript {index}"),
            })
            .unwrap();
    }

    assert_eq!(
        prune_history_store(&store, HistoryRetentionPolicy::Last5).unwrap(),
        1
    );

    let remaining = store.list().unwrap();
    assert_eq!(remaining.len(), 5);
    // The oldest entry (index 0) was pruned; every surviving row still serves
    // its transcript text from the SQLite store.
    for entry in &remaining {
        assert!(store.get(&entry.id).unwrap().is_some());
    }
    assert!(
        !remaining
            .iter()
            .any(|entry| entry.source_name.as_deref() == Some("sample-0.wav"))
    );
}

#[test]
fn history_retention_off_prunes_store_empty() {
    let temp = tempfile::tempdir().unwrap();
    let store = DaemonHistoryStore::open(temp.path());
    for index in 0..3 {
        store
            .record(DaemonHistoryRecord {
                kind: DaemonHistoryKind::File,
                model: "whisper-large-v3-turbo".to_string(),
                source_name: Some(format!("sample-{index}.wav")),
                duration_seconds: None,
                output_format: Some(ResponseFormat::Text),
                diarization_active: Some(false),
                provenance: Some(DaemonHistoryProvenance::Recorded),
                segments: Vec::new(),
                subtitle_cues: Vec::new(),
                timeline_quality: None,
                text: format!("transcript {index}"),
            })
            .unwrap();
    }

    // Switching to "Off" clears everything already stored, even though new
    // writes are skipped upstream at the record sites.
    assert_eq!(
        prune_history_store(&store, HistoryRetentionPolicy::Off).unwrap(),
        3
    );
    assert!(store.list().unwrap().is_empty());

    // "Forever" is the keep-all policy: it never prunes.
    let entry = store
        .record(DaemonHistoryRecord {
            kind: DaemonHistoryKind::File,
            model: "whisper-large-v3-turbo".to_string(),
            source_name: Some("kept.wav".to_string()),
            duration_seconds: None,
            output_format: Some(ResponseFormat::Text),
            diarization_active: Some(false),
            provenance: Some(DaemonHistoryProvenance::Recorded),
            segments: Vec::new(),
            subtitle_cues: Vec::new(),
            timeline_quality: None,
            text: "keep me".to_string(),
        })
        .unwrap();
    assert_eq!(
        prune_history_store(&store, HistoryRetentionPolicy::Forever).unwrap(),
        0
    );
    assert!(store.get(&entry.id).unwrap().is_some());
}

#[test]
fn realtime_capabilities_for_native_runtime_come_from_model_pack() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("server-pack.oasr");
    write_mock_gguf_runtime_source(&pack_root, Some("whisper-large-v3-turbo"));
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: crate::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root).into(),
    };

    let capabilities = realtime_capabilities_for_runtime(&runtime);

    // Realtime capability is registry-driven: the whisper family registers a
    // streaming executor, so its pack advertises true streaming with partials.
    assert_eq!(capabilities.mode, RealtimeBackendMode::TrueStreaming);
    assert!(capabilities.phrase_bias.supported);
    assert!(capabilities.supports_partial_results);
    assert!(!capabilities.diarization.supported);
    assert_eq!(
        capabilities.diarization.reason,
        Some(openasr_core::realtime::REALTIME_VOICE_ID_UNSUPPORTED_REASON)
    );
}

#[test]
fn bound_model_pack_path_is_shared_across_runtime_clones() {
    let temp = tempfile::tempdir().unwrap();
    let pack_a = temp.path().join("pack-a.oasr");
    let pack_b = temp.path().join("pack-b.oasr");
    write_mock_gguf_runtime_source(&pack_a, Some("whisper-tiny"));
    write_mock_gguf_runtime_source(&pack_b, Some("whisper-base"));
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_a.clone()).into(),
    };
    let cloned = runtime.clone();

    runtime
        .rebind_native_model_pack(Some(pack_b.clone()))
        .expect("idle native runtime must rebind in-process");

    assert_eq!(
        cloned.model_pack_path.current().as_deref(),
        Some(pack_b.as_path())
    );
    assert_eq!(
        runtime.model_pack_path.current().as_deref(),
        Some(pack_b.as_path())
    );
}

#[test]
fn rebind_native_model_pack_returns_conflict_while_session_is_active() {
    let temp = tempfile::tempdir().unwrap();
    let pack_a = temp.path().join("pack-a.oasr");
    let pack_b = temp.path().join("pack-b.oasr");
    write_mock_gguf_runtime_source(&pack_a, Some("whisper-tiny"));
    write_mock_gguf_runtime_source(&pack_b, Some("whisper-base"));
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_a.clone()).into(),
    };
    let _permit = runtime
        .acquire_native_execution("native:whisper-tiny@busy-rebind", None)
        .unwrap();

    let error = runtime
        .rebind_native_model_pack(Some(pack_b))
        .expect_err("busy rebind must fail closed");
    assert!(
        matches!(error, ApiError::Conflict(_)),
        "expected 409 conflict, got {error}"
    );
    assert_eq!(
        runtime.model_pack_path.current().as_deref(),
        Some(pack_a.as_path())
    );
}

#[tokio::test]
async fn set_default_model_http_returns_conflict_when_native_session_is_busy() {
    use axum::body::{Body, to_bytes};
    use tower::ServiceExt;

    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let pack_a = write_installed_pack_ref(
        &home,
        "whisper-tiny",
        "whisper-tiny:q4",
        "q4_0",
        "q4",
        "whisper-tiny",
    );
    let _pack_b = write_installed_pack_ref(
        &home,
        "whisper-base",
        "whisper-base:q4",
        "q4_0",
        "q4",
        "whisper-base",
    );
    let pack_a_installed = installed_pack_by_pull(&home, "whisper-tiny:q4");
    persist_default_pack(
        &home,
        &pack_a_installed,
        QuantPreference::pinned(&pack_a_installed.quant),
    )
    .unwrap();
    let previous_v2 =
        openasr_core::default_selection::read_active_model_selection_v2(&home).unwrap();

    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_a.clone()).into(),
    };
    let _permit = runtime
        .acquire_native_execution("native:whisper-tiny@busy-rebind-http", None)
        .unwrap();
    let app = app_with_runtime_and_distribution(
        runtime.clone(),
        DistributionRuntime {
            openasr_home: Some(home.clone()),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/default")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "pull": "whisper-base:q4" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        parsed["error"]["message"]
            .as_str()
            .unwrap()
            .contains("native transcription or realtime session is running")
    );
    assert_eq!(
        runtime.model_pack_path.current().as_deref(),
        Some(pack_a.as_path())
    );
    assert_eq!(
        openasr_core::default_selection::read_active_model_selection_v2(&home).unwrap(),
        previous_v2
    );
}

fn activation_probe_ok() -> Result<(), String> {
    Ok(())
}

fn activation_probe_fail() -> Result<(), String> {
    Err("injected activation probe failure".to_string())
}

fn installed_pack_by_pull(home: &std::path::Path, pull: &str) -> InstalledPack {
    list_installed_packs(home)
        .unwrap()
        .into_iter()
        .find(|pack| pack.pull == pull)
        .unwrap_or_else(|| panic!("installed pack {pull} must exist"))
}

#[tokio::test]
async fn set_default_model_http_keeps_previous_selection_when_activation_probe_fails() {
    use axum::body::{Body, to_bytes};
    use tower::ServiceExt;

    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let pack_a = write_installed_pack_ref(
        &home,
        "whisper-tiny",
        "whisper-tiny:q4",
        "q4_0",
        "q4",
        "whisper-tiny",
    );
    let _pack_b = write_installed_pack_ref(
        &home,
        "whisper-base",
        "whisper-base:q4",
        "q4_0",
        "q4",
        "whisper-base",
    );
    let pack_a_installed = installed_pack_by_pull(&home, "whisper-tiny:q4");
    persist_default_pack(
        &home,
        &pack_a_installed,
        QuantPreference::pinned(&pack_a_installed.quant),
    )
    .unwrap();
    let previous_v2 =
        openasr_core::default_selection::read_active_model_selection_v2(&home).unwrap();

    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_a.clone()).into(),
    };
    runtime
        .model_pack_path
        .set_activation_probe_failpoint(Some(activation_probe_fail()));
    let app = app_with_runtime_and_distribution(
        runtime.clone(),
        DistributionRuntime {
            openasr_home: Some(home.clone()),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/default")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "pull": "whisper-base:q4" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        parsed["error"]["message"]
            .as_str()
            .unwrap()
            .contains("injected activation probe failure"),
        "{parsed}"
    );
    assert_eq!(
        runtime.model_pack_path.current().as_deref(),
        Some(pack_a.as_path())
    );
    assert_eq!(
        openasr_core::default_selection::read_active_model_selection_v2(&home).unwrap(),
        previous_v2
    );
}

#[tokio::test]
async fn activation_rechecks_capacity_after_successful_forecast_and_preserves_old_state() {
    use axum::body::{Body, to_bytes};
    use openasr_core::device::execution_memory::{
        DeviceMemorySnapshot, DomainFootprint, DomainReservationRequest, MemoryDomainKey,
        MemoryObservationConfidence,
    };
    use tower::ServiceExt;

    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let previous_path = write_installed_pack_ref(
        &home,
        "whisper-tiny",
        "whisper-tiny:q4",
        "q4_0",
        "q4",
        "whisper-tiny",
    );
    let next_path = write_installed_pack_ref(
        &home,
        "whisper-base",
        "whisper-base:q4",
        "q4_0",
        "q4",
        "whisper-base",
    );
    let previous = installed_pack_by_pull(&home, "whisper-tiny:q4");
    persist_default_pack(&home, &previous, QuantPreference::pinned(&previous.quant)).unwrap();
    let previous_v2 =
        openasr_core::default_selection::read_active_model_selection_v2(&home).unwrap();

    let native_execution = NativeExecutionSupervisor::default();
    let services = Arc::clone(native_execution.execution_services());
    let broker = Arc::clone(services.memory_broker());
    let next = installed_pack_by_pull(&home, "whisper-base:q4");
    let verified_next = openasr_core::PackVerifier
        .verify_candidate(openasr_core::PackCandidate::new(next_path))
        .expect("forecast candidate pack must verify");
    let before_forecast = broker.usage(&MemoryDomainKey::SystemMemory);
    let forecast = openasr_core::resolve_default_model_activation(
        services.as_ref(),
        &verified_next,
        openasr_core::device::execution_policy::ExecutionIntent::CpuOnly,
        next.pull.clone(),
        next.path.clone(),
    )
    .expect("advisory activation facts must resolve")
    .quote()
    .expect("advisory activation forecast must succeed before pressure changes");
    drop(forecast);
    assert_eq!(
        broker.usage(&MemoryDomainKey::SystemMemory),
        before_forecast,
        "an advisory quote must not reserve or commit physical capacity"
    );

    // Capacity changes after the successful advisory forecast. The real
    // activation below must obtain a fresh quote/reservation and reject this
    // pressure rather than treating forecast success as authorization.
    let pressure_bytes = 1_u64 << 60;
    let mut pressure = broker
        .try_reserve_batch(vec![DomainReservationRequest::from_footprint(
            DomainFootprint {
                domain: MemoryDomainKey::SystemMemory,
                peak_bytes: pressure_bytes,
                retained_bytes: pressure_bytes,
                requires_reconciliation: false,
                resource_ids: vec!["external-host-pressure".to_string()],
            },
            DeviceMemorySnapshot {
                free_bytes: u64::MAX,
                total_bytes: u64::MAX,
                confidence: MemoryObservationConfidence::DeviceSnapshot,
            },
        )])
        .unwrap();
    pressure.commit_quoted().unwrap();
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution,
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(previous_path.clone()).into(),
    };
    let app = app_with_runtime_and_distribution(
        runtime.clone(),
        DistributionRuntime {
            openasr_home: Some(home.clone()),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/default")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "pull": "whisper-base:q4" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        parsed["error"]["message"]
            .as_str()
            .unwrap()
            .contains("activation reserve failed"),
        "{parsed}"
    );
    assert_eq!(
        runtime.model_pack_path.current().as_deref(),
        Some(previous_path.as_path())
    );
    assert_eq!(
        openasr_core::default_selection::read_active_model_selection_v2(&home).unwrap(),
        previous_v2
    );
    assert_eq!(
        services.runtime_receipts().reconcile_live_leases(&broker),
        openasr_core::runtime_receipts::LeaseReceiptShadow::Matched
    );
    drop(pressure);
}

#[tokio::test]
async fn set_default_model_http_keeps_previous_selection_when_persist_fails() {
    use axum::body::{Body, to_bytes};
    use tower::ServiceExt;

    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let pack_a = write_installed_pack_ref(
        &home,
        "whisper-tiny",
        "whisper-tiny:q4",
        "q4_0",
        "q4",
        "whisper-tiny",
    );
    let _pack_b = write_installed_pack_ref(
        &home,
        "whisper-base",
        "whisper-base:q4",
        "q4_0",
        "q4",
        "whisper-base",
    );
    let pack_a_installed = installed_pack_by_pull(&home, "whisper-tiny:q4");
    persist_default_pack(
        &home,
        &pack_a_installed,
        QuantPreference::pinned(&pack_a_installed.quant),
    )
    .unwrap();
    let previous_v2 =
        openasr_core::default_selection::read_active_model_selection_v2(&home).unwrap();

    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_a.clone()).into(),
    };
    runtime
        .model_pack_path
        .set_activation_probe_failpoint(Some(activation_probe_ok()));
    let app = app_with_runtime_and_distribution(
        runtime.clone(),
        DistributionRuntime {
            openasr_home: Some(home.clone()),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    openasr_core::default_selection::set_persist_commit_failpoint_for_test(true);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/default")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "pull": "whisper-base:q4" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    openasr_core::default_selection::set_persist_commit_failpoint_for_test(false);
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        parsed["error"]["message"]
            .as_str()
            .unwrap()
            .contains("injected persist failure"),
        "{parsed}"
    );
    assert_eq!(
        runtime.model_pack_path.current().as_deref(),
        Some(pack_a.as_path())
    );
    assert_eq!(
        openasr_core::default_selection::read_active_model_selection_v2(&home).unwrap(),
        previous_v2
    );
}

#[tokio::test]
async fn set_default_model_failure_matrix_preserves_precommit_state() {
    use axum::body::Body;
    use tower::ServiceExt;

    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let previous_path = write_installed_pack_ref(
        &home,
        "whisper-failure-a",
        "whisper-failure-a:q4",
        "q4_0",
        "q4",
        "whisper-failure-a",
    );
    let _next_path = write_installed_pack_ref(
        &home,
        "whisper-failure-b",
        "whisper-failure-b:q4",
        "q4_0",
        "q4",
        "whisper-failure-b",
    );
    let previous = installed_pack_by_pull(&home, "whisper-failure-a:q4");
    persist_default_pack(&home, &previous, QuantPreference::pinned(&previous.quant)).unwrap();
    let durable_before =
        openasr_core::default_selection::read_active_model_selection_v2(&home).unwrap();

    for failpoint in [
        ModelActivationFailpoint::PackVerification,
        ModelActivationFailpoint::CandidateResolution,
        ModelActivationFailpoint::QuoteObservation,
        ModelActivationFailpoint::BrokerReservation,
        ModelActivationFailpoint::NativeMaterialization,
        ModelActivationFailpoint::FirstComputeAttestation,
        ModelActivationFailpoint::Reconciliation,
        ModelActivationFailpoint::V2StagingWrite,
        ModelActivationFailpoint::V2StagingSync,
        ModelActivationFailpoint::AtomicBeforeReplace,
    ] {
        let runtime = ServerRuntime {
            backend: BackendKind::Native,
            native_execution: NativeExecutionSupervisor::default(),
            ffmpeg_bin: None,
            ffmpeg_bin_explicit: false,
            model_pack_path: Some(previous_path.clone()).into(),
        };
        runtime
            .model_pack_path
            .set_activation_probe_failpoint(Some(activation_probe_ok()));
        runtime
            .model_pack_path
            .set_activation_failpoint_for_test(Some(failpoint));
        let app = app_with_runtime_and_distribution(
            runtime.clone(),
            DistributionRuntime {
                openasr_home: Some(home.clone()),
                catalog_url: None,
                catalog_local_override: None,
            },
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/models/default")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::json!({ "pull": "whisper-failure-b:q4" }).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{failpoint:?}");
        assert_eq!(
            runtime.model_pack_path.current().as_deref(),
            Some(previous_path.as_path()),
            "live state changed at {failpoint:?}"
        );
        assert_eq!(
            openasr_core::default_selection::read_active_model_selection_v2(&home).unwrap(),
            durable_before,
            "durable state changed at {failpoint:?}"
        );
        let services = runtime.native_execution.execution_services();
        assert_eq!(
            services
                .runtime_receipts()
                .reconcile_live_leases_quiescent(services.memory_broker()),
            openasr_core::runtime_receipts::LeaseReceiptShadow::Matched,
            "owner ledger leaked at {failpoint:?}"
        );
        assert!(
            std::fs::read_dir(&home)
                .unwrap()
                .filter_map(Result::ok)
                .all(|entry| {
                    let name = entry.file_name();
                    let name = name.to_string_lossy();
                    !(name.starts_with(".openasr-") && name.ends_with(".tmp"))
                }),
            "orphan staging file after {failpoint:?}"
        );
    }
}

#[tokio::test]
async fn atomic_after_replace_commits_and_publishes_instead_of_rolling_back() {
    use axum::body::Body;
    use tower::ServiceExt;

    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let previous_path = write_installed_pack_ref(
        &home,
        "whisper-after-replace-a",
        "whisper-after-replace-a:q4",
        "q4_0",
        "q4",
        "whisper-after-replace-a",
    );
    let next_path = write_installed_pack_ref(
        &home,
        "whisper-after-replace-b",
        "whisper-after-replace-b:q4",
        "q4_0",
        "q4",
        "whisper-after-replace-b",
    );
    let previous = installed_pack_by_pull(&home, "whisper-after-replace-a:q4");
    persist_default_pack(&home, &previous, QuantPreference::pinned(&previous.quant)).unwrap();
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(previous_path).into(),
    };
    runtime
        .model_pack_path
        .set_activation_probe_failpoint(Some(activation_probe_ok()));
    runtime
        .model_pack_path
        .set_activation_failpoint_for_test(Some(ModelActivationFailpoint::AtomicAfterReplace));
    let app = app_with_runtime_and_distribution(
        runtime.clone(),
        DistributionRuntime {
            openasr_home: Some(home.clone()),
            catalog_url: None,
            catalog_local_override: None,
        },
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/default")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "pull": "whisper-after-replace-b:q4" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        runtime.model_pack_path.current().as_deref(),
        Some(next_path.as_path())
    );
    let durable = openasr_core::default_selection::read_active_model_selection_v2(&home)
        .unwrap()
        .unwrap();
    assert_eq!(durable.pull.as_deref(), Some("whisper-after-replace-b:q4"));
}

#[tokio::test]
async fn restart_reactivates_commit_that_preceded_live_pointer_exchange() {
    use axum::body::Body;
    use tower::ServiceExt;

    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let previous_path = write_installed_pack_ref(
        &home,
        "whisper-restart-a",
        "whisper-restart-a:q4",
        "q4_0",
        "q4",
        "whisper-restart-a",
    );
    let next_path = write_installed_pack_ref(
        &home,
        "whisper-restart-b",
        "whisper-restart-b:q4",
        "q4_0",
        "q4",
        "whisper-restart-b",
    );
    let previous = installed_pack_by_pull(&home, "whisper-restart-a:q4");
    persist_default_pack(&home, &previous, QuantPreference::pinned(&previous.quant)).unwrap();
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(previous_path.clone()).into(),
    };
    runtime
        .model_pack_path
        .set_activation_probe_failpoint(Some(activation_probe_ok()));
    runtime
        .model_pack_path
        .set_activation_failpoint_for_test(Some(
            ModelActivationFailpoint::DurableCommitBeforeLivePublish,
        ));
    let app = app_with_runtime_and_distribution(
        runtime.clone(),
        DistributionRuntime {
            openasr_home: Some(home.clone()),
            catalog_url: None,
            catalog_local_override: None,
        },
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/default")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "pull": "whisper-restart-b:q4" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        runtime.model_pack_path.current().as_deref(),
        Some(previous_path.as_path()),
        "the killed process never exchanged its live pointer"
    );
    let durable = openasr_core::default_selection::read_active_model_selection_v2(&home)
        .unwrap()
        .unwrap();
    assert_eq!(durable.pull.as_deref(), Some("whisper-restart-b:q4"));

    let restarted = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: ActiveRuntimeSlot::requested(Some(next_path.clone())),
    };
    restarted
        .model_pack_path
        .set_activation_probe_failpoint(Some(activation_probe_ok()));
    let next = installed_pack_by_pull(&home, "whisper-restart-b:q4");
    let intent =
        openasr_core::default_selection::execution_intent_from_v2_wire(&durable.execution_intent)
            .unwrap();
    activate_default_model_blocking(
        &restarted,
        &home,
        &next,
        durable.quant_preference.clone(),
        intent,
        DefaultModelActivationMode::ReactivateDurableSelection,
    )
    .unwrap();
    assert_eq!(
        restarted.model_pack_path.current().as_deref(),
        Some(next_path.as_path())
    );
    assert_eq!(
        openasr_core::default_selection::read_active_model_selection_v2(&home)
            .unwrap()
            .unwrap(),
        durable,
        "restart reactivation must not mint another durable generation"
    );
}

#[tokio::test]
async fn set_default_model_http_persists_only_after_activation_probe_succeeds() {
    use axum::body::{Body, to_bytes};
    use tower::ServiceExt;

    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let pack_a = write_installed_pack_ref(
        &home,
        "whisper-tiny",
        "whisper-tiny:q4",
        "q4_0",
        "q4",
        "whisper-tiny",
    );
    let pack_b = write_installed_pack_ref(
        &home,
        "whisper-base",
        "whisper-base:q4",
        "q4_0",
        "q4",
        "whisper-base",
    );
    let pack_a_installed = installed_pack_by_pull(&home, "whisper-tiny:q4");
    persist_default_pack(
        &home,
        &pack_a_installed,
        QuantPreference::pinned(&pack_a_installed.quant),
    )
    .unwrap();

    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_a).into(),
    };
    runtime
        .model_pack_path
        .set_activation_probe_failpoint(Some(activation_probe_ok()));
    let app = app_with_runtime_and_distribution(
        runtime.clone(),
        DistributionRuntime {
            openasr_home: Some(home.clone()),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/default")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "pull": "whisper-base:q4" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["default_model"], "whisper-base");
    assert_eq!(
        runtime.model_pack_path.current().as_deref(),
        Some(pack_b.as_path())
    );
    let persisted = openasr_core::default_selection::read_active_model_selection_v2(&home)
        .unwrap()
        .expect("successful activation must persist V2");
    assert_eq!(persisted.pull.as_deref(), Some("whisper-base:q4"));
    assert_eq!(
        persisted.status,
        openasr_core::default_selection::ActiveModelSelectionStatus::Installed
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_default_model_http_real_probe_attests_plan_lane_and_live_backend() {
    use axum::body::{Body, to_bytes};
    use tower::ServiceExt;

    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let mut config = openasr_core::load_config_document(&home).unwrap();
    config.preferences.execution_target = openasr_core::ExecutionTarget::Cpu;
    openasr_core::save_config_document(&home, &config).unwrap();
    let previous_path = write_installed_pack_ref(
        &home,
        "cohere-transcribe-a",
        "cohere-transcribe-a:q4",
        "q4_0",
        "q4",
        "cohere-transcribe-a",
    );
    let next_path = write_installed_pack_ref(
        &home,
        "cohere-transcribe-b",
        "cohere-transcribe-b:q4",
        "q4_0",
        "q4",
        "cohere-transcribe-b",
    );
    let previous = installed_pack_by_pull(&home, "cohere-transcribe-a:q4");
    persist_default_pack(&home, &previous, QuantPreference::pinned(&previous.quant)).unwrap();
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(previous_path).into(),
    };
    let app = app_with_runtime_and_distribution(
        runtime.clone(),
        DistributionRuntime {
            openasr_home: Some(home.clone()),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/default")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "pull": "cohere-transcribe-b:q4" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["activation"], "committed");
    assert_eq!(parsed["default_pull"], "cohere-transcribe-b:q4");
    assert_eq!(
        runtime.model_pack_path.current().as_deref(),
        Some(next_path.as_path())
    );
    let durable = openasr_core::default_selection::read_active_model_selection_v2(&home)
        .unwrap()
        .unwrap();
    assert_eq!(durable.execution_intent, "cpu_only");
    assert!(durable.architecture_id.is_some());
}

#[test]
fn active_runtime_barrier_closes_session_admission_race() {
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: None.into(),
    };

    let activation = runtime.begin_native_activation().unwrap();
    assert!(matches!(
        runtime.acquire_native_execution("activation-barrier", None),
        Err(ApiError::Conflict(_))
    ));
    drop(activation);

    let permit = runtime
        .acquire_native_execution("activation-barrier", None)
        .unwrap();
    let activation = runtime.begin_native_activation().unwrap();
    assert!(runtime.native_rebind_blocked());
    drop(activation);
    drop(permit);
}

#[test]
fn stale_active_runtime_snapshot_cannot_start_after_republication() {
    let pack = PathBuf::from("active-runtime-snapshot.oasr");
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack.clone()).into(),
    };
    let snapshot = runtime
        .model_pack_path
        .current_snapshot()
        .expect("initial active runtime snapshot");

    // Re-publishing even the same path is a new generation: the underlying
    // bytes may have been reinstalled in place, so path equality cannot be an
    // admission authority and must not create an ABA hole.
    runtime
        .model_pack_path
        .set_legacy_binding(Some(pack.clone()));

    assert!(matches!(
        runtime.acquire_native_execution_for_snapshot(&snapshot, "stale-snapshot", None),
        Err(ApiError::Conflict(_))
    ));
    assert!(!runtime.native_execution.has_active_sessions());

    let fresh = runtime
        .model_pack_path
        .current_snapshot()
        .expect("republished active runtime snapshot");
    let permit = runtime
        .acquire_native_execution_for_snapshot(&fresh, "fresh-snapshot", None)
        .expect("the current publication may be admitted");
    drop(permit);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn boot_reactivation_attests_v2_before_publishing_active_runtime() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let mut config = openasr_core::load_config_document(&home).unwrap();
    config.preferences.execution_target = openasr_core::ExecutionTarget::Cpu;
    openasr_core::save_config_document(&home, &config).unwrap();
    let pack_path = write_installed_pack_ref(
        &home,
        "cohere-transcribe-restart",
        "cohere-transcribe-restart:q4",
        "q4_0",
        "q4",
        "cohere-transcribe-restart",
    );
    let pack = installed_pack_by_pull(&home, "cohere-transcribe-restart:q4");
    let verified = openasr_core::PackVerifier
        .verify_candidate(openasr_core::PackCandidate::new(pack.path.clone()))
        .unwrap();
    openasr_core::default_selection::persist_activation_detailed(
        &home,
        &pack,
        QuantPreference::pinned(&pack.quant),
        verified.model_architecture(),
        &openasr_core::device::execution_policy::ExecutionIntent::CpuOnly,
    )
    .unwrap();
    let durable_before =
        openasr_core::default_selection::read_active_model_selection_v2(&home).unwrap();

    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: ActiveRuntimeSlot::requested(Some(pack_path.clone())),
    };
    assert!(runtime.model_pack_path.current().is_none());
    assert_eq!(
        runtime.model_pack_path.requested_path().as_deref(),
        Some(pack_path.as_path())
    );

    let reactivation = realtime::spawn_boot_native_warmup(runtime.clone(), home.clone());
    tokio::time::timeout(std::time::Duration::from_secs(30), reactivation)
        .await
        .expect("boot reactivation must finish within the integration-test deadline")
        .expect("boot reactivation worker must not panic");

    assert_eq!(
        runtime.model_pack_path.current().as_deref(),
        Some(pack_path.as_path())
    );
    assert!(
        runtime.model_is_resident(),
        "successful reactivation must publish the exact candidate resident marker without staling its warmup generation"
    );
    assert_eq!(
        openasr_core::default_selection::read_active_model_selection_v2(&home).unwrap(),
        durable_before,
        "startup reactivation must validate, not rewrite, durable V2"
    );
    let services = runtime.native_execution.execution_services();
    assert_eq!(
        services
            .runtime_receipts()
            .reconcile_live_leases_quiescent(services.memory_broker()),
        openasr_core::runtime_receipts::LeaseReceiptShadow::Matched
    );
}

#[test]
fn set_default_model_http_stage_does_not_publish_live() {
    let source = include_str!("routes/models_api.rs");
    assert!(
        !source.contains("rebind_native_model_pack"),
        "production model routes must not restore the legacy path-only rebind"
    );
    let owner_impl = source
        .split("impl NativeActivationStagedOwner")
        .nth(1)
        .expect("NativeActivationStagedOwner impl");
    let stage = owner_impl
        .split("fn discard_candidate")
        .next()
        .expect("stage precedes discard");
    assert!(
        stage.contains("fn stage("),
        "expected NativeActivationStagedOwner::stage in source audit window"
    );
    assert!(
        !stage.contains("rebind_native_model_pack"),
        "materialize/stage must not publish live: {stage}"
    );
    let set_default = source
        .split("pub(crate) fn activate_default_model_blocking")
        .nth(1)
        .expect("activate_default_model_blocking")
        .split("struct NativeActivationStagedOwner")
        .next()
        .expect("activation body");
    assert!(
        !set_default.contains("NoopActivationReservation"),
        "set_default_model must not reserve with NoopActivationReservation"
    );
    assert!(
        set_default.contains(".quote()") && set_default.contains(".reserve(services.as_ref())"),
        "set_default_model must separate quote observation from broker reservation"
    );
    assert!(
        !set_default.contains("ResolvedExecutionRoute::cpu()"),
        "set_default_model must not quote a dummy CPU candidate: {set_default}"
    );
    assert!(
        set_default.contains("PackVerifier")
            && set_default.contains("resolve_default_model_activation"),
        "set_default_model must quote the pack being activated on its real lane: {set_default}"
    );
    let persist_idx = set_default
        .find("commit_activation")
        .expect("persist/commit must exist");
    let publish_idx = set_default
        .find("publish_attested_native_model")
        .expect("live publication must exist after persist");
    assert!(
        persist_idx < publish_idx,
        "live publication must follow V2 commit, got persist@{persist_idx} publish@{publish_idx}"
    );
}

#[test]
fn spawn_boot_native_warmup_uses_set_default_transaction_entry() {
    let source = include_str!("realtime/native_worker.rs");
    let spawn = source
        .split("pub(crate) fn spawn_boot_native_warmup")
        .nth(1)
        .expect("spawn_boot_native_warmup")
        .split("/// Attest a candidate pack")
        .next()
        .expect("spawn_boot_native_warmup body");
    assert!(
        spawn.contains("activate_default_model_blocking(")
            && spawn.contains("ReactivateDurableSelection"),
        "boot warmup must use the same complete transaction entry as set-default: {spawn}"
    );
    assert!(
        !spawn.contains("warm_up_default_native_streaming_worker"),
        "boot warmup must not bypass probe_native_activation: {spawn}"
    );
    assert!(
        !spawn.contains("rebind_native_model_pack")
            && !spawn.contains("persist_detailed")
            && !spawn.contains("PersistSelection"),
        "boot reactivation must not bypass the read-only durable journal: {spawn}"
    );
}

fn write_installed_pack_ref(
    home: &std::path::Path,
    model_id: &str,
    pull: &str,
    quant: &str,
    suffix: &str,
    metadata_model_id: &str,
) -> std::path::PathBuf {
    use sha2::{Digest, Sha256};

    std::fs::create_dir_all(home).unwrap();
    let staging = home.join(format!("{model_id}-staging.oasr"));
    write_mock_gguf_runtime_source(&staging, Some(metadata_model_id));
    let bytes = std::fs::read(&staging).unwrap();
    std::fs::remove_file(&staging).unwrap();
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let object = home
        .join("models")
        .join("objects")
        .join("sha256")
        .join(&sha256)
        .join("content");
    std::fs::create_dir_all(object.parent().unwrap()).unwrap();
    std::fs::write(&object, &bytes).unwrap();
    let reference = home.join(format!("models/refs/{model_id}/{quant}.json"));
    std::fs::create_dir_all(reference.parent().unwrap()).unwrap();
    let pack = InstalledPack {
        model_id: model_id.to_string(),
        display_name: model_id.to_string(),
        quant: quant.to_string(),
        suffix: suffix.to_string(),
        pull: pull.to_string(),
        filename: format!("{model_id}-{quant}.oasr"),
        path: object.clone(),
        url: format!("https://example.invalid/{model_id}.oasr"),
        hf_revision: "test".to_string(),
        sha256,
        size_bytes: bytes.len() as u64,
        installed_at_unix_seconds: 1,
        source: None,
    };
    std::fs::write(reference, serde_json::to_vec_pretty(&pack).unwrap()).unwrap();
    object
}

#[tokio::test]
async fn devices_endpoint_enumerates_this_daemons_runtime() {
    // The endpoint reflects the daemon process's own ggml runtime -- the whole
    // point of moving enumeration server-side (a CPU-only desktop shell can no
    // longer under-report a GPU sidecar). It always offers at least Auto + CPU,
    // and the reported default matches Auto's effective target.
    let response = devices().await.0;
    assert_eq!(response.object, "devices");
    let ids: Vec<_> = response.devices.iter().map(|d| d.id.as_str()).collect();
    assert!(ids.contains(&"auto"), "auto target missing: {ids:?}");
    assert!(ids.contains(&"cpu"), "cpu target missing: {ids:?}");
    assert!(
        response.default_execution_target == "cpu"
            || response.default_execution_target == "accelerated",
        "unexpected default target: {}",
        response.default_execution_target
    );
    let auto = response.devices.iter().find(|d| d.id == "auto").unwrap();
    assert_eq!(auto.effective_target, response.default_execution_target);
}

#[test]
fn devices_endpoint_is_not_operator_gated() {
    // Local UI read: reachable like `/v1/capabilities`, not behind the
    // operator-only pull/write gate.
    assert!(!is_operator_only_path(
        &axum::http::Method::GET,
        "/v1/devices"
    ));
}

#[test]
fn native_server_runtime_rejects_directory_runtime_source() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("server-pack.openasr");
    std::fs::create_dir_all(&pack_root).unwrap();
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: crate::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root).into(),
    };
    let error = runtime.validate().unwrap_err().to_string();
    assert!(
        error.contains("could not verify and select a native model adapter"),
        "{error}"
    );
}

#[test]
fn eta_seconds_rounds_up_remaining_download_time() {
    assert_eq!(eta_seconds(90, 100, 20), Some(1));
    assert_eq!(eta_seconds(50, 101, 20), Some(3));
    assert_eq!(eta_seconds(100, 100, 20), Some(0));
    assert_eq!(eta_seconds(50, 100, 0), None);
}

#[test]
fn pull_progress_speed_is_smoothed_across_jittery_samples() {
    // Persistence fires on "累计 >=8MB OR distance >=1s from last point, first
    // to trigger", so consecutive sample windows vary widely in duration and
    // byte count even for a steady underlying transfer rate. That variance
    // alone makes the raw delta-bytes/delta-time speed swing wildly between
    // ticks. This drives `apply_progress` through such a jittery sequence and
    // asserts the EMA-smoothed `speed_bps` the snapshot exposes has much
    // lower variance than the instantaneous speed each tick implies.
    let resolved = resolved_pull_fixture();
    let bytes_total = 200 * 1024 * 1024;
    let mut snapshot =
        PullJobSnapshot::queued("pull-smooth-test".to_string(), &resolved, None, false);
    snapshot.apply_progress(PullProgress::DownloadStarted {
        bytes_total,
        resume_from: 0,
    });

    // (elapsed_ms, delta_bytes) ticks: a download effectively steady around
    // ~10 MB/s, but sampled on an irregular cadence so short/low-delta and
    // long/high-delta windows alternate -- exactly what production jitter
    // looks like.
    let ticks: &[(u128, u64)] = &[
        (1000, 10 * 1024 * 1024),
        (100, 500 * 1024),
        (2000, 30 * 1024 * 1024),
        (200, 4 * 1024 * 1024),
        (1500, 12 * 1024 * 1024),
        (50, 100 * 1024),
        (1000, 11 * 1024 * 1024),
        (2500, 20 * 1024 * 1024),
    ];

    let mut bytes_done = 0_u64;
    let mut instant_speeds = Vec::new();
    let mut smoothed_speeds = Vec::new();

    for &(elapsed_ms, delta_bytes) in ticks {
        bytes_done += delta_bytes;
        // Back-date the last-progress timestamp so `apply_progress` (which
        // stamps "now" internally) observes the simulated elapsed time,
        // without needing a real sleep.
        let now = unix_millis_now();
        snapshot.last_progress_at_unix_millis = Some(now.saturating_sub(elapsed_ms));
        instant_speeds.push(((delta_bytes as u128) * 1000 / elapsed_ms) as u64);
        snapshot.apply_progress(PullProgress::Downloading {
            bytes_done,
            bytes_total,
        });
        smoothed_speeds.push(
            snapshot
                .speed_bps
                .expect("speed_bps should be set after a progress tick"),
        );
    }

    fn variance(values: &[u64]) -> f64 {
        let mean = values.iter().sum::<u64>() as f64 / values.len() as f64;
        values
            .iter()
            .map(|&value| {
                let diff = value as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64
    }

    let instant_variance = variance(&instant_speeds);
    let smoothed_variance = variance(&smoothed_speeds);
    assert!(
        smoothed_variance < instant_variance * 0.5,
        "expected EMA-smoothed speed variance ({smoothed_variance}) to be well below \
         instantaneous speed variance ({instant_variance}); instant={instant_speeds:?} \
         smoothed={smoothed_speeds:?}"
    );
}

#[test]
fn ema_blend_seeds_from_instant_and_decays_toward_new_samples() {
    // alpha weight applied to the new sample: blend should sit strictly
    // between prev and instant for a mid-range alpha, and hit the endpoints
    // for alpha = 0.0 / 1.0.
    assert_eq!(ema_blend(100, 200, 0.0), 200); // alpha=0: ignore new sample
    assert_eq!(ema_blend(100, 200, 1.0), 100); // alpha=1: ignore history
    let blended = ema_blend(100, 200, 0.25);
    assert_eq!(blended, 175); // 0.25*100 + 0.75*200 = 175
    assert!(blended > 100 && blended < 200);
}

#[test]
fn pull_progress_persistence_is_throttled_between_boundaries() {
    let mut last_bytes = 0;
    let mut last_at = Instant::now();
    assert!(should_persist_pull_progress(
        &PullProgress::DownloadStarted {
            bytes_total: 32 * 1024 * 1024,
            resume_from: 0,
        },
        &mut last_bytes,
        &mut last_at,
    ));
    assert!(!should_persist_pull_progress(
        &PullProgress::Downloading {
            bytes_done: 64 * 1024,
            bytes_total: 32 * 1024 * 1024,
        },
        &mut last_bytes,
        &mut last_at,
    ));
    assert!(should_persist_pull_progress(
        &PullProgress::Downloading {
            bytes_done: PULL_JOB_PROGRESS_PERSIST_INTERVAL_BYTES,
            bytes_total: 32 * 1024 * 1024,
        },
        &mut last_bytes,
        &mut last_at,
    ));
}

#[test]
fn explicit_pull_license_acceptance_covers_every_license_class() {
    let mut resolved = resolved_pull_fixture();

    resolved.license_class = LicenseClass::Permissive;
    ensure_explicit_model_license_acceptance(&resolved, false)
        .expect("permissive models do not require acceptance");

    for license_class in [LicenseClass::Noncommercial, LicenseClass::Gated] {
        resolved.license_class = license_class;
        let error = ensure_explicit_model_license_acceptance(&resolved, false).unwrap_err();
        assert!(error.to_string().contains("accept_license=true"), "{error}");
        ensure_explicit_model_license_acceptance(&resolved, true)
            .expect("explicit acceptance permits a restricted model pull");
    }

    resolved.license_class = LicenseClass::Unknown;
    let error = ensure_explicit_model_license_acceptance(&resolved, true).unwrap_err();
    assert!(
        error.to_string().contains("unsupported license class"),
        "{error}"
    );
}

async fn assert_restricted_pull_is_rejected_before_side_effects(
    license_class: LicenseClass,
    from: Option<PathBuf>,
) -> String {
    let temp = tempfile::tempdir().unwrap();
    let (distribution, home) =
        distribution_context_with_pull_license_for_test(temp.path(), license_class);
    let source_path = from.map(|path| temp.path().join(path));

    let error = start_pull_job(
        AxumPath("moonshine-tiny".to_string()),
        Extension(distribution.clone()),
        Json(StartPullRequest {
            quant: Some("q8".to_string()),
            size: None,
            from: source_path.clone(),
            accept_license: None,
        }),
    )
    .await
    .unwrap_err();

    assert_eq!(
        distribution.jobs.next.load(Ordering::Relaxed),
        0,
        "license refusal must happen before allocating a job id"
    );
    assert!(
        distribution.jobs.snapshots.lock().unwrap().is_empty(),
        "license refusal must not publish a pull job"
    );
    assert!(
        distribution.jobs.active.lock().unwrap().is_empty(),
        "license refusal must not spawn a pull worker"
    );
    assert!(
        !home.join("pulls").exists(),
        "license refusal must happen before persisting a pull job"
    );
    if let Some(source_path) = source_path {
        assert!(
            !source_path.exists(),
            "the rejected local source must not be opened or created"
        );
    }

    match error {
        ApiError::BadRequest(message) => message,
        other => panic!("expected a bad-request license refusal, got {other}"),
    }
}

#[tokio::test]
async fn restricted_remote_pulls_require_acceptance_before_job_creation() {
    let noncommercial =
        assert_restricted_pull_is_rejected_before_side_effects(LicenseClass::Noncommercial, None)
            .await;
    assert!(noncommercial.contains("non-commercial use only"));

    let gated =
        assert_restricted_pull_is_rejected_before_side_effects(LicenseClass::Gated, None).await;
    assert!(gated.contains("vendor license"));
}

#[tokio::test]
async fn restricted_local_pulls_require_acceptance_before_source_access() {
    let local_source = || Some(PathBuf::from("source-must-not-be-read.oasr"));
    let noncommercial = assert_restricted_pull_is_rejected_before_side_effects(
        LicenseClass::Noncommercial,
        local_source(),
    )
    .await;
    assert!(noncommercial.contains("non-commercial use only"));

    let gated =
        assert_restricted_pull_is_rejected_before_side_effects(LicenseClass::Gated, local_source())
            .await;
    assert!(gated.contains("vendor license"));
}

async fn assert_local_import_license_policy(
    license_class: LicenseClass,
    accepted: Option<bool>,
) -> Result<ImportLocalModelResponse, ApiError> {
    let temp = tempfile::tempdir().unwrap();
    let (distribution, home, pack_path) =
        local_import_fixture_with_license(temp.path(), license_class);
    let result = import_local_model(
        Extension(distribution),
        Json(ImportLocalModelRequest {
            path: pack_path,
            accept_license: accepted,
        }),
    )
    .await;
    if result.is_err() {
        assert!(
            list_installed_packs(&home).unwrap().is_empty(),
            "license refusal must happen before installing local pack content"
        );
    }
    result.map(|response| response.0)
}

#[tokio::test]
async fn local_import_uses_the_shared_install_license_policy() {
    assert_local_import_license_policy(LicenseClass::Permissive, None)
        .await
        .expect("permissive local import needs no acceptance");

    for license_class in [LicenseClass::Noncommercial, LicenseClass::Gated] {
        let error = assert_local_import_license_policy(license_class.clone(), None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("accept_license=true"), "{error}");
        assert_local_import_license_policy(license_class, Some(true))
            .await
            .expect("explicit acceptance permits restricted local import");
    }

    let error = assert_local_import_license_policy(LicenseClass::Unknown, Some(true))
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("not present in the signed model catalog"),
        "{error}"
    );
}

#[test]
fn persisted_pull_license_proof_migrates_fail_closed() {
    let mut resolved = resolved_pull_fixture();

    // Simulate a pre-field snapshot by removing `license_accepted` from its
    // serialized form. Serde's default keeps old permissive jobs resumable.
    let permissive = PullJobSnapshot::queued("old-permissive".into(), &resolved, None, false);
    let mut old_json = serde_json::to_value(permissive).unwrap();
    old_json.as_object_mut().unwrap().remove("license_accepted");
    let migrated: PullJobSnapshot = serde_json::from_value(old_json).unwrap();
    assert!(!migrated.license_accepted);
    resolved_pull_from_snapshot(&migrated).expect("old permissive job remains resumable");

    for license_class in [LicenseClass::Noncommercial, LicenseClass::Gated] {
        resolved.license_class = license_class;
        let old_restricted =
            PullJobSnapshot::queued("old-restricted".into(), &resolved, None, false);
        let mut old_json = serde_json::to_value(old_restricted).unwrap();
        old_json.as_object_mut().unwrap().remove("license_accepted");
        let migrated: PullJobSnapshot = serde_json::from_value(old_json).unwrap();
        let error = resolved_pull_from_snapshot(&migrated).unwrap_err();
        assert!(error.to_string().contains("accept_license=true"), "{error}");

        let accepted = PullJobSnapshot::queued("accepted-restricted".into(), &resolved, None, true);
        assert!(
            serde_json::to_value(&accepted).unwrap()["license_accepted"] == true,
            "new restricted snapshots must persist acceptance proof"
        );
        resolved_pull_from_snapshot(&accepted)
            .expect("persisted explicit acceptance permits resume");
    }

    resolved.license_class = LicenseClass::Unknown;
    let unknown = PullJobSnapshot::queued("unknown".into(), &resolved, None, true);
    let error = resolved_pull_from_snapshot(&unknown).unwrap_err();
    assert!(
        error.to_string().contains("unsupported license class"),
        "{error}"
    );
}

#[tokio::test]
async fn pull_job_events_notify_paused_snapshot_and_reconnect_terminal_state() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = distribution_context_for_test(temp.path());
    let resolved = resolved_pull_fixture();
    let snapshot = PullJobSnapshot::queued("pull-test".to_string(), &resolved, None, false);
    distribution.insert_job(snapshot).unwrap();

    let mut receiver = distribution.subscribe_job("pull-test").unwrap();
    distribution
        .update_job("pull-test", |snapshot| {
            snapshot.state = PullJobState::Paused;
            snapshot.control_requested = None;
            snapshot.error = Some("Pull job was paused.".to_string());
        })
        .unwrap();
    receiver.changed().await.unwrap();
    let observed = receiver.borrow().clone();
    assert_eq!(observed.state, PullJobState::Paused);
    assert!(observed.state.is_event_terminal());

    let reconnected = distribution.subscribe_job("pull-test").unwrap();
    assert_eq!(reconnected.borrow().state, PullJobState::Paused);
}

#[tokio::test]
async fn pull_job_control_ack_sets_flag_without_terminal_state_flip() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = distribution_context_for_test(temp.path());
    let resolved = resolved_pull_fixture();
    let snapshot = PullJobSnapshot::queued("pull-control".to_string(), &resolved, None, false);
    distribution.insert_job(snapshot).unwrap();
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let pause_flag = Arc::new(AtomicBool::new(false));
    distribution.register_active_job("pull-control", cancel_flag.clone(), pause_flag.clone());

    assert!(distribution.pause_job("pull-control"));
    distribution
        .update_job("pull-control", |snapshot| {
            snapshot.control_requested = Some(PullControlRequest::Pause);
        })
        .unwrap();
    let snapshot = distribution.snapshot("pull-control").unwrap();
    assert_eq!(snapshot.state, PullJobState::Queued);
    assert_eq!(snapshot.control_requested, Some(PullControlRequest::Pause));
    assert!(pause_flag.load(Ordering::SeqCst));
    assert!(!cancel_flag.load(Ordering::SeqCst));

    assert!(distribution.cancel_job("pull-control"));
    assert!(cancel_flag.load(Ordering::SeqCst));
    distribution.clear_active_job("pull-control");
}

#[tokio::test]
async fn transcription_control_endpoints_flip_pause_resume_cancel_flags() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = distribution_context_for_test(temp.path());
    let control = Arc::new(openasr_core::TranscriptionControl::new());
    assert!(distribution.try_register_transcription("txn-1", Arc::clone(&control)));

    pause_transcription_job(
        AxumPath("txn-1".to_string()),
        Extension(distribution.clone()),
    )
    .await
    .unwrap();
    assert!(control.is_paused());

    resume_transcription_job(
        AxumPath("txn-1".to_string()),
        Extension(distribution.clone()),
    )
    .await
    .unwrap();
    assert!(!control.is_paused());

    cancel_transcription_job(
        AxumPath("txn-1".to_string()),
        Extension(distribution.clone()),
    )
    .await
    .unwrap();
    assert!(control.is_canceled());

    // Cleared entry (finished run) and unknown ids both fail closed with 404.
    assert!(distribution.clear_transcription_if_current("txn-1", &control));
    assert!(distribution.transcription_control("txn-1").is_none());
    let error = cancel_transcription_job(
        AxumPath("txn-1".to_string()),
        Extension(distribution.clone()),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, ApiError::NotFound(_)));
}

#[test]
fn transcription_canceled_backend_error_maps_to_409() {
    let response =
        ApiError::Backend(openasr_core::BackendError::TranscriptionCanceled).into_response();
    assert_eq!(response.status(), StatusCode::CONFLICT);
}

#[test]
fn external_diarization_fail_closed_errors_map_to_400() {
    let missing = ApiError::Backend(openasr_core::BackendError::DiarizationSegmenterUnavailable)
        .into_response();
    assert_eq!(missing.status(), StatusCode::BAD_REQUEST);

    let failed = ApiError::Backend(openasr_core::BackendError::ExternalDiarizationFailed {
        reason: "segmenter inference failed".to_string(),
    })
    .into_response();
    assert_eq!(failed.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn pull_job_reuses_existing_nonterminal_snapshot_for_same_pull() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = distribution_context_for_test(temp.path());
    let resolved = resolved_pull_fixture();
    distribution
        .insert_job(PullJobSnapshot::queued(
            "pull-existing".to_string(),
            &resolved,
            None,
            false,
        ))
        .unwrap();
    let mut completed =
        PullJobSnapshot::queued("pull-completed".to_string(), &resolved, None, false);
    completed.state = PullJobState::Completed;
    distribution.insert_job(completed).unwrap();

    let reused = distribution
        .nonterminal_snapshot_for_pull(&resolved)
        .unwrap();

    assert_eq!(reused.job_id, "pull-existing");
}

#[test]
fn nonterminal_snapshot_for_pull_skips_a_job_being_canceled() {
    // A job with a pending cancel is not yet terminal (its worker unwinds
    // asynchronously), but re-pulling the same pack while it drains must NOT
    // coalesce into that dying job -- otherwise the user's "download again"
    // silently attaches to a job on its way to `Canceled` and no new download
    // starts until the cancel fully settles. The re-pull must see "no live job
    // for this pack" so `start_pull_job` mints a fresh one instead.
    let temp = tempfile::tempdir().unwrap();
    let distribution = distribution_context_for_test(temp.path());
    let resolved = resolved_pull_fixture();

    let mut canceling =
        PullJobSnapshot::queued("pull-canceling".to_string(), &resolved, None, false);
    canceling.state = PullJobState::Downloading;
    canceling.control_requested = Some(PullControlRequest::Cancel);
    distribution.insert_job(canceling).unwrap();

    assert!(
        distribution
            .nonterminal_snapshot_for_pull(&resolved)
            .is_none(),
        "a cancel-in-flight job must not be reused for a fresh pull"
    );

    // A still-live (non-canceling) download for the same pack is still reused.
    let mut downloading = PullJobSnapshot::queued("pull-live".to_string(), &resolved, None, false);
    downloading.state = PullJobState::Downloading;
    distribution.insert_job(downloading).unwrap();

    assert_eq!(
        distribution
            .nonterminal_snapshot_for_pull(&resolved)
            .unwrap()
            .job_id,
        "pull-live"
    );
}

#[tokio::test]
async fn list_pull_jobs_returns_empty_list_when_no_jobs_exist() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = distribution_context_for_test(temp.path());

    let response = list_pull_jobs(Extension(distribution)).await;

    assert!(response.0.jobs.is_empty());
}

#[tokio::test]
async fn list_pull_jobs_surfaces_persisted_nonterminal_jobs_after_restart_without_side_effects() {
    let temp = tempfile::tempdir().unwrap();
    let resolved = resolved_pull_fixture();
    let pulls_dir = temp.path().join("pulls");
    std::fs::create_dir_all(&pulls_dir).unwrap();

    // A prior daemon process was killed mid-download and had persisted this
    // job to disk before exiting (mirrors what `persist_snapshot` writes).
    let mut downloading =
        PullJobSnapshot::queued("pull-inflight".to_string(), &resolved, None, false);
    downloading.state = PullJobState::Downloading;
    downloading.bytes_done = 1;
    downloading.bytes_total = resolved.size_bytes;
    std::fs::write(
        pulls_dir.join("pull-inflight.json"),
        serde_json::to_vec_pretty(&downloading).unwrap(),
    )
    .unwrap();

    // A job that already finished before the restart must not resurface.
    let mut completed = PullJobSnapshot::queued("pull-done".to_string(), &resolved, None, false);
    completed.state = PullJobState::Completed;
    std::fs::write(
        pulls_dir.join("pull-done.json"),
        serde_json::to_vec_pretty(&completed).unwrap(),
    )
    .unwrap();

    // Fresh `DistributionContext::new` is what happens when the daemon
    // restarts: it reads `~/.openasr/pulls/*.json` synchronously at
    // construction time, before any request has been served.
    let distribution = distribution_context_for_test(temp.path());

    let response = list_pull_jobs(Extension(distribution.clone())).await;
    let jobs = response.0.jobs;

    assert_eq!(jobs.len(), 1, "only the non-terminal job should be listed");
    assert_eq!(jobs[0].job_id, "pull-inflight");
    // `load_persisted_pull_jobs` normalizes a restart-resumable job that still
    // has a resolved spec back to `Queued` (in-process progress state like
    // the EMA speed and last-progress timestamp cannot have survived the
    // restart), so that is the state the listing endpoint must surface.
    assert_eq!(jobs[0].state, PullJobState::Queued);

    // Calling the endpoint must not itself start or resume anything: the
    // daemon never auto-resumes restart-interrupted jobs anymore (the
    // client decides via the resume/cancel routes), and no job was
    // registered active. Querying is not pulling.
    assert!(
        distribution.jobs.active.lock().unwrap().is_empty(),
        "listing jobs must not register/spawn any active pull job"
    );
}

/// A cancel (or pause) requested before the daemon died must survive the
/// restart as a settled state, not be resurrected into `Queued` and
/// re-downloaded against the user's last explicit instruction.
#[tokio::test]
async fn load_persisted_pull_jobs_finalizes_pending_control_requests_across_restart() {
    let temp = tempfile::tempdir().unwrap();
    let resolved = resolved_pull_fixture();
    let pulls_dir = temp.path().join("pulls");
    std::fs::create_dir_all(&pulls_dir).unwrap();

    let mut canceling =
        PullJobSnapshot::queued("pull-canceling".to_string(), &resolved, None, false);
    canceling.state = PullJobState::Downloading;
    canceling.bytes_done = 1;
    canceling.bytes_total = resolved.size_bytes;
    canceling.control_requested = Some(PullControlRequest::Cancel);
    std::fs::write(
        pulls_dir.join("pull-canceling.json"),
        serde_json::to_vec_pretty(&canceling).unwrap(),
    )
    .unwrap();

    let mut pausing = PullJobSnapshot::queued("pull-pausing".to_string(), &resolved, None, false);
    pausing.state = PullJobState::Verifying;
    pausing.bytes_total = resolved.size_bytes;
    pausing.control_requested = Some(PullControlRequest::Pause);
    std::fs::write(
        pulls_dir.join("pull-pausing.json"),
        serde_json::to_vec_pretty(&pausing).unwrap(),
    )
    .unwrap();

    let distribution = distribution_context_for_test(temp.path());

    let canceled = distribution.snapshot("pull-canceling").unwrap();
    assert_eq!(canceled.state, PullJobState::Canceled);
    assert!(canceled.state.is_terminal());
    assert_eq!(canceled.control_requested, None);
    assert!(
        canceled
            .error
            .as_deref()
            .unwrap()
            .contains("canceled before the OpenASR daemon restarted"),
        "cancel reason must survive the restart: {:?}",
        canceled.error
    );

    let paused = distribution.snapshot("pull-pausing").unwrap();
    assert_eq!(paused.state, PullJobState::Paused);
    assert_eq!(paused.control_requested, None);

    // The terminal canceled job must no longer count as active work.
    let jobs = list_pull_jobs(Extension(distribution)).await;
    assert_eq!(jobs.0.jobs.len(), 1);
    assert_eq!(jobs.0.jobs[0].job_id, "pull-pausing");
}

#[test]
fn pull_job_failure_log_line_carries_job_identity_and_message() {
    // Background pull failures must land in daemon.log with the same
    // identifying fields a failed HTTP request logs (see ApiError's
    // IntoResponse eprintln), so a failed download is diagnosable without
    // a client-side report.
    let line = pull_job_failure_log_line(
        "pull-7",
        "moonshine-tiny:q8",
        "Downloaded pack sha256 mismatch for '/x': expected a, got b",
    );
    assert_eq!(
        line,
        "openasr-server: pull job failed job_id=pull-7 pull=moonshine-tiny:q8 message=Downloaded pack sha256 mismatch for '/x': expected a, got b"
    );
}

/// The resume route is the client's explicit restart-resume decision: an
/// interrupted job (Queued, no worker) must actually start downloading
/// again. Uses a local source-path install so the worker settles without
/// any network access.
#[tokio::test]
async fn resume_pull_job_restarts_interrupted_job_to_completion() {
    let temp = tempfile::tempdir().unwrap();
    let pack_path = temp.path().join("source-moonshine-tiny-q8_0.oasr");
    let spec = TinyGgufFixtureSpec::moonshine_oasr_v1_runtime_ready("moonshine-tiny");
    write_tiny_gguf_runtime_source(&pack_path, &spec).unwrap();
    let bytes = std::fs::read(&pack_path).unwrap();

    let mut resolved = resolved_pull_fixture();
    resolved.sha256 = format!("{:x}", Sha256::digest(&bytes));
    resolved.size_bytes = bytes.len() as u64;

    let pulls_dir = temp.path().join("pulls");
    std::fs::create_dir_all(&pulls_dir).unwrap();
    let mut interrupted = PullJobSnapshot::queued(
        "pull-interrupted".to_string(),
        &resolved,
        Some(pack_path),
        false,
    );
    interrupted.state = PullJobState::Downloading;
    interrupted.bytes_done = 1;
    interrupted.bytes_total = resolved.size_bytes;
    std::fs::write(
        pulls_dir.join("pull-interrupted.json"),
        serde_json::to_vec_pretty(&interrupted).unwrap(),
    )
    .unwrap();

    let distribution = distribution_context_for_test(temp.path());
    assert_eq!(
        distribution.snapshot("pull-interrupted").unwrap().state,
        PullJobState::Queued
    );
    assert!(
        !distribution.is_job_active("pull-interrupted"),
        "load must not spawn any worker for the interrupted job"
    );

    let response = resume_pull_job(
        AxumPath("pull-interrupted".to_string()),
        Extension(distribution.clone()),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let final_snapshot = loop {
        let snapshot = distribution.snapshot("pull-interrupted").unwrap();
        if snapshot.state.is_terminal() {
            break snapshot;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "resumed job did not settle in time"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    };
    assert_eq!(
        final_snapshot.state,
        PullJobState::Completed,
        "resumed job must complete from its local source path: {:?}",
        final_snapshot.error
    );
    // The worker clears its active registration right after settling the
    // terminal state; poll for it (the snapshot flips a hair earlier).
    while distribution.is_job_active("pull-interrupted") {
        assert!(
            std::time::Instant::now() < deadline,
            "resumed job never released its active registration"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn resume_pull_job_never_double_spawns_an_active_job() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = distribution_context_for_test(temp.path());
    let resolved = resolved_pull_fixture();
    let snapshot = PullJobSnapshot::queued("pull-active".to_string(), &resolved, None, false);
    distribution.insert_job(snapshot).unwrap();
    distribution.register_active_job(
        "pull-active",
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );

    let response = resume_pull_job(
        AxumPath("pull-active".to_string()),
        Extension(distribution.clone()),
    )
    .await
    .unwrap();

    // A job with a live worker is already downloading: attach-only (200),
    // never a second competing worker.
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(distribution.jobs.active.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn resume_pull_job_still_fails_closed_on_unaccepted_restricted_license() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = distribution_context_for_test(temp.path());
    let mut resolved = resolved_pull_fixture();
    resolved.license_class = LicenseClass::Noncommercial;
    let snapshot = PullJobSnapshot::queued("pull-license-gate".to_string(), &resolved, None, false);
    distribution.insert_job(snapshot).unwrap();

    let error = resume_pull_job(
        AxumPath("pull-license-gate".to_string()),
        Extension(distribution.clone()),
    )
    .await
    .unwrap_err();

    let response = error.into_response();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        !distribution.is_job_active("pull-license-gate"),
        "a license-refused resume must not spawn a worker"
    );
}

#[tokio::test]
async fn cancel_pull_job_finalizes_workerless_interrupted_job() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = distribution_context_for_test(temp.path());
    let resolved = resolved_pull_fixture();
    let snapshot = PullJobSnapshot::queued("pull-idle".to_string(), &resolved, None, false);
    distribution.insert_job(snapshot).unwrap();
    assert!(!distribution.is_job_active("pull-idle"));

    let response = cancel_pull_job(
        AxumPath("pull-idle".to_string()),
        Extension(distribution.clone()),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let stored = distribution.snapshot("pull-idle").unwrap();
    assert_eq!(stored.state, PullJobState::Canceled);
    assert!(stored.state.is_terminal());
}

#[test]
fn pull_job_insert_failure_does_not_publish_in_memory_snapshot() {
    let temp = tempfile::tempdir().unwrap();
    let home_file = temp.path().join("openasr-home-file");
    std::fs::write(&home_file, b"not a directory").unwrap();
    let distribution = distribution_context_for_test(&home_file);
    let resolved = resolved_pull_fixture();
    let snapshot =
        PullJobSnapshot::queued("pull-persist-fails".to_string(), &resolved, None, false);

    let error = distribution.insert_job(snapshot).unwrap_err().to_string();

    assert!(
        error.contains("Could not create pull job directory"),
        "{error}"
    );
    assert!(distribution.snapshot("pull-persist-fails").is_none());
    assert!(distribution.subscribe_job("pull-persist-fails").is_none());
}

#[test]
fn pull_job_update_failure_does_not_publish_in_memory_snapshot() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = distribution_context_for_test(temp.path());
    let pulls_dir = temp.path().join("pulls");
    let resolved = resolved_pull_fixture();
    let snapshot = PullJobSnapshot::queued(
        "pull-update-persist-fails".to_string(),
        &resolved,
        None,
        false,
    );
    distribution.insert_job(snapshot).unwrap();
    std::fs::remove_dir_all(&pulls_dir).unwrap();
    std::fs::write(&pulls_dir, b"not a directory").unwrap();

    let error = distribution
        .update_job("pull-update-persist-fails", |snapshot| {
            snapshot.state = PullJobState::Completed;
        })
        .unwrap_err()
        .to_string();

    assert!(
        error.contains("Could not create pull job directory"),
        "{error}"
    );
    let stored = distribution.snapshot("pull-update-persist-fails").unwrap();
    assert_eq!(stored.state, PullJobState::Queued);
}

#[tokio::test]
async fn pull_job_limiter_is_per_home_and_single_concurrency() {
    let temp = tempfile::tempdir().unwrap();
    let limiter = pull_limiter_for_home(temp.path());
    let first = limiter.clone().acquire_owned().await.unwrap();

    assert!(limiter.clone().try_acquire_owned().is_err());

    drop(first);
    assert!(limiter.try_acquire_owned().is_ok());
}

#[test]
fn native_server_runtime_rejects_non_gguf_runtime_source_file() {
    let temp = tempfile::tempdir().unwrap();
    let pack_path = temp.path().join("server-pack.openasr");
    std::fs::write(&pack_path, b"not a directory").unwrap();
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: crate::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_path).into(),
    };
    let error = runtime.validate().unwrap_err().to_string();
    assert!(
        error.contains("could not verify and select a native model adapter"),
        "{error}"
    );
}

#[test]
fn native_server_runtime_rejects_directory_runtime_source_without_file_fallback() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("server-pack");
    std::fs::create_dir_all(&pack_root).unwrap();
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: crate::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root).into(),
    };
    let error = runtime.validate().unwrap_err().to_string();
    assert!(
        error.contains("could not verify and select a native model adapter"),
        "{error}"
    );
}

#[tokio::test]
async fn native_transcribe_stays_fail_closed_with_local_pack_only_validation() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("native-pack.oasr");
    write_mock_gguf_runtime_source(&pack_root, Some("whisper-large-v3-turbo"));
    let sample_wav =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: crate::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root).into(),
    };
    let request = TranscriptionRequest::new(sample_wav, "whisper-large-v3-turbo");
    let error = transcribe_with_runtime(
        runtime,
        request,
        std::sync::Arc::new(openasr_core::RequestExecutionContext::uncancellable(
            "test fixture",
        )),
    )
    .await
    .unwrap_err();
    let rendered = error.to_string();
    assert!(rendered.contains("Could not transcribe audio"));
}

/// End-to-end regression for the reported bug: a bare-ADTS `.aac` upload (the
/// exact shape WeChat and other recorders emit, not an m4a/mp4 container)
/// used to fail with HTTP 400 "Unsupported audio input ... the file has no
/// extension" -- a lie, since the client's upload plainly had a `.aac` name;
/// the extension was silently stripped from the upload's own temp file before
/// it ever reached the audio probe. This drives the exact same two steps a
/// real multipart POST to `/v1/audio/transcriptions` goes through --
/// `axum::extract::Multipart` extraction, then `parse_transcription_multipart`
/// -- with a real bare-ADTS `.aac` fixture, then feeds the resulting request
/// into the real native transcription path. The shared runtime-ready fixture
/// must complete transcription, proving both that the upload's extension
/// survived and that the decoded audio reached the unified native runtime.
#[tokio::test]
async fn native_transcribe_accepts_a_real_uploaded_aac_file_past_audio_preparation() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("native-pack.oasr");
    write_mock_gguf_runtime_source(&pack_root, Some("whisper-large-v3-turbo"));

    let fixture_bytes = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../openasr-core/tests/fixtures/tone_mono.aac"),
    )
    .unwrap();
    let boundary = "openasr-native-aac-e2e-boundary";
    let mut body = Vec::new();
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"voice-message.aac\"\r\nContent-Type: audio/aac\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(&fixture_bytes);
    body.extend_from_slice(
        format!(
            "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n--{boundary}--\r\n"
        )
        .as_bytes(),
    );
    let http_request = axum::http::Request::builder()
        .method("POST")
        .header(
            axum::http::header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(axum::body::Body::from(body))
        .unwrap();
    let multipart = <Multipart as axum::extract::FromRequest<()>>::from_request(http_request, &())
        .await
        .expect("a well-formed multipart body must extract");
    let mut parsed = parse_transcription_multipart(Ok(multipart), BackendKind::Native, None)
        .await
        .expect("a multipart body with a real .aac file + valid model field must parse");
    assert_eq!(
        parsed
            .request
            .input_path
            .extension()
            .and_then(|extension| extension.to_str()),
        Some("aac"),
        "the server's own upload temp file must keep the real .aac extension"
    );
    // This fixture intentionally carries the tiny CPU-only Whisper tensor
    // geometry. A neutral Windows host now exposes its bundled Vulkan module
    // to Auto, so leaving the request unconstrained would turn this audio
    // upload regression into an accidental GPU graph test. Keep the test on
    // its documented fixture backend; real Vulkan placement/correctness has a
    // separate exact-provider gate with production packs.
    parsed.request.execution_target = Some(ExecutionTarget::Cpu);

    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: crate::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root).into(),
    };
    let transcription = transcribe_with_runtime(
        runtime,
        parsed.request,
        std::sync::Arc::new(openasr_core::RequestExecutionContext::uncancellable(
            "test fixture",
        )),
    )
    .await
    .expect("a real .aac upload must complete through the unified native runtime");
    assert_eq!(transcription.text, "fixture0");
    assert_eq!(transcription.segments.len(), 1);
    assert!(transcription.segments[0].end > transcription.segments[0].start);
}

#[cfg(unix)]
#[tokio::test]
async fn native_audio_preparation_does_not_consume_model_capacity() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let pack_path = temp.path().join("native-preparation-capacity.oasr");
    write_mock_gguf_runtime_source(&pack_path, Some("whisper-large-v3-turbo"));
    let input_path = temp.path().join("blocked-preparation.mp3");
    std::fs::write(&input_path, b"not an mp3 stream").unwrap();
    let started_path = temp.path().join("preparation-started");
    let release_path = temp.path().join("release-preparation");
    let converter = temp.path().join("blocking-ffmpeg");
    std::fs::write(
        &converter,
        format!(
            "#!/bin/sh\ntouch '{}'\nwhile [ ! -f '{}' ]; do sleep 0.01; done\nexit 1\n",
            started_path.display(),
            release_path.display(),
        ),
    )
    .unwrap();
    std::fs::set_permissions(&converter, std::fs::Permissions::from_mode(0o755)).unwrap();

    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        native_execution: NativeExecutionSupervisor::new(std::num::NonZeroUsize::new(1).unwrap()),
        ffmpeg_bin: Some(converter),
        ffmpeg_bin_explicit: true,
        model_pack_path: Some(pack_path).into(),
    };
    let request_runtime = runtime.clone();
    let request = TranscriptionRequest::new(input_path, "whisper-large-v3-turbo");
    // Drive the native path as an async task. Nesting `spawn_blocking` +
    // `block_on` here starves the blocking pool under a full nextest run
    // (the native path already uses `spawn_blocking` for preparation), so the
    // converter never starts within the wait window.
    let request_task = tokio::spawn(async move {
        transcribe_with_runtime(
            request_runtime,
            request,
            std::sync::Arc::new(openasr_core::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        )
        .await
    });

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    while !started_path.exists() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "native audio preparation never reached the blocking converter"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    let permit = runtime
        .acquire_native_execution("test-content", None)
        .expect("audio-only preparation must not consume native model capacity");
    drop(permit);
    std::fs::write(&release_path, b"release").unwrap();
    let error = request_task
        .await
        .expect("preparation task must not panic")
        .expect_err("blocking converter exits unsuccessfully");
    assert!(matches!(error, ApiError::AudioPreparation(_)));
}

#[test]
fn parse_segment_mode_accepts_energy_and_rejects_unknown() {
    assert_eq!(parse_segment_mode("energy").unwrap(), LongFormMode::Energy);
    let error = parse_segment_mode("unknown").unwrap_err().to_string();
    assert!(error.contains("Unsupported segment_mode 'unknown'"));
}

#[test]
fn build_native_longform_options_validates_overlap() {
    let error = build_native_longform_options(
        Some("fixed"),
        Some(2.0),
        Some(2.0),
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("Invalid longform segmentation configuration"));
}

#[test]
fn build_native_longform_options_override_omits_default_server_values() {
    assert_eq!(
        build_native_longform_options_override(None, None, None, None, None, None, None, None)
            .unwrap(),
        None
    );
}

#[test]
fn build_native_longform_options_override_keeps_explicit_fields() {
    let options = build_native_longform_options_override(
        Some("energy"),
        None,
        Some(0.5),
        Some(-42.0),
        None,
        None,
        Some(1.0),
        Some(true),
    )
    .unwrap()
    .expect("explicit fields should preserve override");
    assert_eq!(options.mode, LongFormMode::Energy);
    assert_eq!(options.overlap_seconds, 0.5);
    assert_eq!(options.energy_silence_threshold_db, -42.0);
    assert_eq!(options.min_chunk_seconds, 1.0);
    assert!(options.suppress_silent_slices);
}
