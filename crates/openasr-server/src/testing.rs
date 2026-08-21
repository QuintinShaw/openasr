//! Loopback TLS + pairing fixture shared with `openasr-client` integration tests.
//!
//! This is not a product API. It exists so client-protocol tests can drive a
//! real `openasr-server` over TOFU TLS without copying the fixture.

use std::{
    net::SocketAddr,
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::http::StatusCode;
use rustls::{
    ClientConfig, DigitallySignedStruct, Error as TlsError, SignatureScheme,
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    pki_types::{CertificateDer, ServerName, UnixTime},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    task,
};
use tokio_rustls::TlsConnector;

use crate::{
    DistributionRuntime, ServerAuth, ServerLaunchOptions, ServerRuntime, TlsListener,
    app_with_runtime_and_distribution_and_launch_options, self_signed_tls_identity,
};
use openasr_core::{
    certificate_fingerprint_sha256, pairing_safety_code_for_certificate_fingerprint,
};

/// Administrator bearer token used by [`spawn_loopback_pairing_server`].
pub const PAIRING_ADMIN_TOKEN: &str = "admin-secret";

/// A loopback HTTPS OpenASR server with pairing enabled and a known certificate
/// fingerprint.
pub struct LoopbackTlsServer {
    pub addr: SocketAddr,
    pub certificate_fingerprint: String,
    _task: task::JoinHandle<()>,
}

impl Drop for LoopbackTlsServer {
    fn drop(&mut self) {
        self._task.abort();
    }
}

/// Trust-on-first-use verifier used only by this fixture's admin/helper HTTP
/// client. Production clients must use `openasr_client::TofuServerVerifier`.
#[derive(Debug)]
pub struct TestTofuVerifier {
    pub fingerprint: Arc<Mutex<Option<String>>>,
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

/// Parsed HTTPS response captured by [`https_request`].
pub struct TestHttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
    pub certificate_fingerprint: String,
}

/// Credential returned after a full create + approve + fetch cycle.
pub struct LoopbackPairingCredential {
    pub device_id: String,
    pub bearer_token: String,
}

/// Bind `127.0.0.1:0` with a self-signed cert for `127.0.0.1` and pairing auth.
pub async fn spawn_loopback_pairing_server(home: &Path) -> LoopbackTlsServer {
    spawn_loopback_pairing_server_with_sans(home, &["127.0.0.1".to_string()]).await
}

/// Same as [`spawn_loopback_pairing_server`], but with an explicit certificate
/// SAN list. Used to prove TOFU pinning does not treat hostname mismatch as a
/// hard failure (LAN clients connect by IP to a cert that may only name
/// localhost).
pub async fn spawn_loopback_pairing_server_with_sans(
    home: &Path,
    subject_alt_names: &[String],
) -> LoopbackTlsServer {
    let identity = self_signed_tls_identity(subject_alt_names).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let certificate_fingerprint = identity.certificate_sha256.clone();
    let safety_code = pairing_safety_code_for_certificate_fingerprint(&certificate_fingerprint);
    let app = app_with_runtime_and_distribution_and_launch_options(
        ServerRuntime::default(),
        DistributionRuntime {
            openasr_home: Some(home.to_path_buf()),
            catalog_url: None,
            catalog_local_override: None,
        },
        ServerLaunchOptions {
            auth: ServerAuth::pairing_with_safety_code(PAIRING_ADMIN_TOKEN, Some(safety_code)),
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

/// One-shot HTTPS request against a loopback pairing server using a TOFU
/// verifier that records the observed fingerprint (no pin).
pub async fn https_request(
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

/// Format a `Bearer` header value.
pub fn bearer_auth_header(token: &str) -> String {
    format!("Bearer {token}")
}

/// Create, admin-approve, and fetch a pairing credential against `server`.
pub async fn approve_loopback_pairing(server: &LoopbackTlsServer) -> LoopbackPairingCredential {
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

    approve_pending_pairing_request(server, request_id).await;

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

/// Admin-approve an already-created pairing request (the client owns create/poll).
pub async fn approve_pending_pairing_request(server: &LoopbackTlsServer, request_id: &str) {
    let authorize = format!("Bearer {PAIRING_ADMIN_TOKEN}");
    let approve = https_request(
        server.addr,
        "POST",
        &format!("/v1/pairing/requests/{request_id}/approve"),
        &[("Authorization", authorize.as_str())],
        Vec::new(),
    )
    .await;
    assert_eq!(approve.status, u16::from(StatusCode::OK));
}

/// Admin-revoke a paired device credential.
pub async fn revoke_loopback_pairing(server: &LoopbackTlsServer, device_id: &str) {
    let authorize = format!("Bearer {PAIRING_ADMIN_TOKEN}");
    let revoke = https_request(
        server.addr,
        "DELETE",
        &format!("/v1/pairing/credentials/{device_id}"),
        &[("Authorization", authorize.as_str())],
        Vec::new(),
    )
    .await;
    assert_eq!(revoke.status, 204);
}
