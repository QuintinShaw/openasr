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
use rustls::{
    ClientConfig, DigitallySignedStruct, Error as TlsError, SignatureScheme,
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    pki_types::{ServerName, UnixTime},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_rustls::TlsConnector;
use tokio_tungstenite::tungstenite::{client::IntoClientRequest, protocol::Message as WsMessage};

use super::*;

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
    })
}

fn bundled_catalog_url_for_test() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model-registry/catalog.json");
    format!("file://{}", path.display())
}

fn write_valid_installed_pack_for_test(
    home: &Path,
    model_id: &str,
    quant: &str,
    suffix: &str,
) -> InstalledPack {
    let filename = format!("{model_id}-{quant}.oasr");
    let path = home
        .join("models")
        .join(model_id)
        .join(quant)
        .join(&filename);
    let parent = path.parent().expect("installed pack parent").to_path_buf();
    fs::create_dir_all(&parent).expect("create installed pack parent");
    write_mock_gguf_runtime_source(&path, Some(model_id));
    let bytes = fs::read(&path).expect("read installed pack fixture");
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
        sha256: format!("{:x}", Sha256::digest(&bytes)),
        size_bytes: bytes.len() as u64,
        installed_at_unix_seconds: 1,
        source: None,
    };
    fs::write(
        parent.join("installed.json"),
        serde_json::to_string_pretty(&pack).expect("serialize installed pack"),
    )
    .expect("write installed pack metadata");
    pack
}

fn write_mock_gguf_runtime_source(path: &std::path::Path, metadata_model_id: Option<&str>) {
    // Use the graph-complete whisper fixture (not the bare
    // `whisper_oasr_v1_non_streaming_cpu`, which deliberately omits the
    // whisper runtime scalar keys): `list_installed_packs` now re-validates
    // on-disk packs through `validate_native_runtime_model_pack_contract` on
    // every lookup, so an "installed" test fixture must satisfy that
    // contract or it silently stops being recognized as installed.
    let spec = metadata_model_id.map_or_else(
        || TinyGgufFixtureSpec::new(Default::default()),
        TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer,
    );
    write_tiny_gguf_runtime_source(path, &spec).expect("write mock gguf runtime source");
}

struct LoopbackTlsServer {
    addr: SocketAddr,
    certificate_fingerprint: String,
    _task: task::JoinHandle<()>,
}

impl Drop for LoopbackTlsServer {
    fn drop(&mut self) {
        self._task.abort();
    }
}

#[derive(Debug)]
struct TestTofuVerifier {
    fingerprint: Arc<Mutex<Option<String>>>,
}

impl ServerCertVerifier for TestTofuVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        *self.fingerprint.lock().expect("fingerprint mutex poisoned") =
            Some(certificate_fingerprint_sha256(end_entity.as_ref()));
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

struct TestHttpResponse {
    status: u16,
    body: Vec<u8>,
    certificate_fingerprint: String,
}

async fn spawn_loopback_pairing_server(home: &Path) -> LoopbackTlsServer {
    let identity = self_signed_tls_identity(&["127.0.0.1".to_string()]).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let certificate_fingerprint = identity.certificate_sha256.clone();
    let safety_code = pairing_safety_code_for_certificate_fingerprint(&certificate_fingerprint);
    let app = app_with_runtime_and_distribution_and_launch_options(
        ServerRuntime::default(),
        DistributionRuntime {
            openasr_home: Some(home.to_path_buf()),
            catalog_url: None,
        },
        ServerLaunchOptions {
            auth: ServerAuth::pairing_with_safety_code("admin-secret", Some(safety_code)),
            ..Default::default()
        },
    );
    let task = task::spawn(async move {
        let _ = axum::serve(TlsListener::new(listener, identity.acceptor), app).await;
    });
    LoopbackTlsServer {
        addr,
        certificate_fingerprint,
        _task: task,
    }
}

async fn https_request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Vec<u8>,
) -> TestHttpResponse {
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
    let stream = TcpStream::connect(addr).await.unwrap();
    let server_name = ServerName::try_from("localhost").unwrap().to_owned();
    let mut tls = TlsConnector::from(Arc::new(config))
        .connect(server_name, stream)
        .await
        .unwrap();
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\nContent-Length: {}\r\n",
        addr.port(),
        body.len()
    );
    for (name, value) in headers {
        request.push_str(name);
        request.push_str(": ");
        request.push_str(value);
        request.push_str("\r\n");
    }
    request.push_str("\r\n");
    tls.write_all(request.as_bytes()).await.unwrap();
    if !body.is_empty() {
        tls.write_all(&body).await.unwrap();
    }
    let mut response = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), tls.read_to_end(&mut response))
        .await
        .unwrap()
        .unwrap();
    let certificate_fingerprint = fingerprint
        .lock()
        .expect("fingerprint mutex poisoned")
        .clone()
        .expect("server certificate fingerprint");
    parse_test_http_response(&response, certificate_fingerprint)
}

fn parse_test_http_response(response: &[u8], certificate_fingerprint: String) -> TestHttpResponse {
    let header_end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("http response header terminator");
    let header_text = std::str::from_utf8(&response[..header_end]).unwrap();
    let mut lines = header_text.lines();
    let status = lines
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|status| status.parse::<u16>().ok())
        .expect("http status");
    let headers = lines
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((name.trim().to_ascii_lowercase(), value.trim().to_string()))
        })
        .collect::<Vec<_>>();
    let body = response[header_end + 4..].to_vec();
    let body = if headers
        .iter()
        .any(|(name, value)| name == "transfer-encoding" && value.eq_ignore_ascii_case("chunked"))
    {
        decode_chunked_body(&body)
    } else {
        body
    };
    TestHttpResponse {
        status,
        body,
        certificate_fingerprint,
    }
}

fn decode_chunked_body(body: &[u8]) -> Vec<u8> {
    let mut decoded = Vec::new();
    let mut cursor = 0;
    while let Some(line_end) = body[cursor..]
        .windows(2)
        .position(|window| window == b"\r\n")
        .map(|offset| cursor + offset)
    {
        let size_text = std::str::from_utf8(&body[cursor..line_end]).unwrap();
        let size = usize::from_str_radix(size_text.trim(), 16).unwrap();
        cursor = line_end + 2;
        if size == 0 {
            break;
        }
        decoded.extend_from_slice(&body[cursor..cursor + size]);
        cursor += size + 2;
    }
    decoded
}

fn remote_transcription_multipart_body() -> (String, Vec<u8>) {
    let boundary = "openasr-loopback-boundary";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\nnot a real wav\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n--{boundary}--\r\n"
    )
    .into_bytes();
    (format!("multipart/form-data; boundary={boundary}"), body)
}

fn bearer_auth_header(token: &str) -> String {
    format!("Bearer {token}")
}

struct LoopbackPairingCredential {
    device_id: String,
    bearer_token: String,
}

async fn approve_loopback_pairing(server: &LoopbackTlsServer) -> LoopbackPairingCredential {
    let create = https_request(
        server.addr,
        "POST",
        "/v1/pairing/requests",
        &[("Content-Type", "application/json")],
        br#"{"device_name":"Loopback Mac"}"#.to_vec(),
    )
    .await;
    assert_eq!(create.status, 202);
    assert_eq!(
        create.certificate_fingerprint,
        server.certificate_fingerprint
    );
    let create_json: serde_json::Value = serde_json::from_slice(&create.body).unwrap();
    let request_id = create_json["request_id"].as_str().unwrap();
    assert_eq!(
        create_json["safety_code"],
        pairing_safety_code_for_certificate_fingerprint(&server.certificate_fingerprint)
    );

    let approve = https_request(
        server.addr,
        "POST",
        &format!("/v1/pairing/requests/{request_id}/approve"),
        &[("Authorization", "Bearer admin-secret")],
        Vec::new(),
    )
    .await;
    assert_eq!(approve.status, 200);

    let credential = https_request(
        server.addr,
        "GET",
        &format!("/v1/pairing/requests/{request_id}/credential"),
        &[],
        Vec::new(),
    )
    .await;
    assert_eq!(credential.status, 200);
    let credential_json: serde_json::Value = serde_json::from_slice(&credential.body).unwrap();
    LoopbackPairingCredential {
        device_id: credential_json["device_id"]
            .as_str()
            .expect("approved credential device id")
            .to_string(),
        bearer_token: credential_json["bearer_token"]
            .as_str()
            .expect("approved credential token")
            .to_string(),
    }
}

async fn revoke_loopback_pairing(server: &LoopbackTlsServer, device_id: &str) {
    let revoke = https_request(
        server.addr,
        "DELETE",
        &format!("/v1/pairing/credentials/{device_id}"),
        &[("Authorization", "Bearer admin-secret")],
        Vec::new(),
    )
    .await;
    assert_eq!(revoke.status, 204);
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
    let catalog_url = bundled_catalog_url_for_test();

    let resolved = find_installed_pack_reference(temp.path(), Some(&catalog_url), "qwen:q8")
        .unwrap()
        .unwrap();

    assert_eq!(resolved.pull, pack.pull);
}

#[test]
fn form_model_resolution_preserves_native_request_id() {
    let temp = tempfile::tempdir().unwrap();
    let catalog_url = bundled_catalog_url_for_test();
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

    let response = delete_model(AxumPath(pack.pull.clone()), Extension(distribution.clone()))
        .await
        .unwrap();
    let response = response.0;

    assert!(response.deleted);
    assert_eq!(
        response.pack.as_ref().map(|pack| pack.pull.as_str()),
        Some("moonshine-tiny:q8")
    );
    assert!(list_installed_packs(temp.path()).unwrap().is_empty());
    let default = default_model_response(temp.path(), distribution.catalog_url()).unwrap();
    assert!(default.default_model.is_none());
    assert!(default.default_pull.is_none());
    assert!(default.pack.is_none());
}

#[test]
fn transcription_preferences_fill_missing_thread_request_only() {
    let preferences = Preferences {
        inference_threads: Some(6),
        ..Default::default()
    };
    let mut request = TranscriptionRequest::new("fixtures/jfk.wav", "whisper-large-v3-turbo");

    apply_transcription_preferences(&mut request, &preferences);
    assert_eq!(request.inference_threads, Some(6));

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
        .with_diarization(true);
    let transcription = Transcription {
        text: "hello with speaker".to_string(),
        segments: vec![openasr_core::Segment {
            start: 0.0,
            end: 2.0,
            text: "hello with speaker".to_string(),
            speaker: Some("Alice".to_string()),
            speaker_label: Some("SPEAKER_00".to_string()),
            speaker_profile_id: Some("vp_aaaaaaaaaaaaaaaa".to_string()),
            words: Vec::new(),
        }],
        longform: None,
        language: None,
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
        text: "never stored".to_string(),
        segments: Vec::new(),
        longform: None,
        language: None,
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
                formats: vec!["text".to_string()],
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
                formats: vec!["text".to_string()],
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
            formats: vec!["text".to_string()],
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
        ffmpeg_bin: None,
        model_pack_path: Some(pack_root),
    };

    let capabilities = realtime_capabilities_for_runtime(&runtime);

    // Realtime capability is registry-driven: the whisper family registers a
    // streaming executor, so its pack advertises true streaming with partials.
    assert_eq!(capabilities.mode, RealtimeBackendMode::TrueStreaming);
    assert!(capabilities.phrase_bias.supported);
    assert!(capabilities.supports_partial_results);
}

#[test]
fn native_server_runtime_rejects_directory_runtime_source() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("server-pack.openasr");
    std::fs::create_dir_all(&pack_root).unwrap();
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        ffmpeg_bin: None,
        model_pack_path: Some(pack_root),
    };
    let error = runtime.validate().unwrap_err().to_string();
    assert!(error.contains("must be a regular file"), "{error}");
}

#[test]
fn eta_seconds_rounds_up_remaining_download_time() {
    assert_eq!(eta_seconds(90, 100, 20), Some(1));
    assert_eq!(eta_seconds(50, 101, 20), Some(3));
    assert_eq!(eta_seconds(100, 100, 20), Some(0));
    assert_eq!(eta_seconds(50, 100, 0), None);
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

#[tokio::test]
async fn pull_job_events_notify_paused_snapshot_and_reconnect_terminal_state() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = distribution_context_for_test(temp.path());
    let resolved = resolved_pull_fixture();
    let snapshot = PullJobSnapshot::queued("pull-test".to_string(), &resolved, None);
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
    let snapshot = PullJobSnapshot::queued("pull-control".to_string(), &resolved, None);
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
        ))
        .unwrap();
    let mut completed = PullJobSnapshot::queued("pull-completed".to_string(), &resolved, None);
    completed.state = PullJobState::Completed;
    distribution.insert_job(completed).unwrap();

    let reused = distribution
        .nonterminal_snapshot_for_pull(&resolved)
        .unwrap();

    assert_eq!(reused.job_id, "pull-existing");
}

#[test]
fn pull_job_insert_failure_does_not_publish_in_memory_snapshot() {
    let temp = tempfile::tempdir().unwrap();
    let home_file = temp.path().join("openasr-home-file");
    std::fs::write(&home_file, b"not a directory").unwrap();
    let distribution = distribution_context_for_test(&home_file);
    let resolved = resolved_pull_fixture();
    let snapshot = PullJobSnapshot::queued("pull-persist-fails".to_string(), &resolved, None);

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
    let snapshot =
        PullJobSnapshot::queued("pull-update-persist-fails".to_string(), &resolved, None);
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
        ffmpeg_bin: None,
        model_pack_path: Some(pack_path),
    };
    let error = runtime.validate().unwrap_err().to_string();
    assert!(error.contains("has unknown magic bytes"), "{error}");
}

#[test]
fn native_server_runtime_rejects_directory_runtime_source_without_file_fallback() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("server-pack");
    std::fs::create_dir_all(&pack_root).unwrap();
    let runtime = ServerRuntime {
        backend: BackendKind::Native,
        ffmpeg_bin: None,
        model_pack_path: Some(pack_root),
    };
    let error = runtime.validate().unwrap_err().to_string();
    assert!(error.contains("must be a regular file"), "{error}");
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
        ffmpeg_bin: None,
        model_pack_path: Some(pack_root),
    };
    let request = TranscriptionRequest::new(sample_wav, "whisper-large-v3-turbo");
    let error = transcribe_with_runtime(runtime, request).await.unwrap_err();
    let rendered = error.to_string();
    assert!(rendered.contains("Could not transcribe audio"));
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
