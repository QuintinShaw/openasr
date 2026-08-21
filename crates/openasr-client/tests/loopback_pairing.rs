//! Loopback TLS + pairing tests against a real `openasr-server`.
//!
//! The server fixture lives in `openasr_server::testing` so this crate does
//! not copy the loopback stack. Tokens stay in `MemorySecretStore`; public
//! client state never contains an `oasr_` bearer.

use std::sync::Arc;
use std::time::Duration;

use openasr_client::{
    ClientStatus, MemorySecretStore, PairingPoll, RemoteClient, SecretStore, credential_account,
};
use openasr_core::{
    certificate_fingerprint_sha256, pairing_safety_code_for_certificate_fingerprint,
};
use openasr_server::testing::{
    approve_pending_pairing_request, revoke_loopback_pairing, spawn_loopback_pairing_server,
    spawn_loopback_pairing_server_with_sans,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

struct TestTlsJsonServer {
    port: u16,
    _certificate_fingerprint: String,
    _safety_code: String,
    request: tokio::task::JoinHandle<String>,
}

async fn spawn_test_tls_json_server(
    status: &'static str,
    body: impl FnOnce(&str, &str) -> serde_json::Value,
    wait_for_body: bool,
) -> TestTlsJsonServer {
    let certified = rcgen::generate_simple_self_signed(vec!["127.0.0.1".to_string()]).unwrap();
    let certificate_der_vec = certified.serialize_der().unwrap();
    let certificate_fingerprint = certificate_fingerprint_sha256(&certificate_der_vec);
    let safety_code = pairing_safety_code_for_certificate_fingerprint(&certificate_fingerprint);
    let certificate_der = CertificateDer::from(certificate_der_vec);
    let private_key_der = PrivateKeyDer::from(PrivatePkcs8KeyDer::from(
        certified.serialize_private_key_der(),
    ));
    let tls_config = rustls::ServerConfig::builder_with_provider(
        rustls::crypto::ring::default_provider().into(),
    )
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(vec![certificate_der], private_key_der)
    .unwrap();
    let acceptor = TlsAcceptor::from(Arc::new(tls_config));
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let body = body(&certificate_fingerprint, &safety_code).to_string();
    let request = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let Ok(mut tls) = acceptor.accept(stream).await else {
            return String::new();
        };
        let mut request = Vec::new();
        let mut buffer = [0; 4096];
        loop {
            let read = tls.read(&mut buffer).await.unwrap();
            assert!(read > 0, "client closed before sending request");
            request.extend_from_slice(&buffer[..read]);
            if request.windows(4).any(|window| window == b"\r\n\r\n")
                && (!wait_for_body || request.ends_with(b"}"))
            {
                break;
            }
        }
        let request_text = String::from_utf8(request).unwrap();
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        tls.write_all(response.as_bytes()).await.unwrap();
        tls.shutdown().await.unwrap();
        request_text
    });
    TestTlsJsonServer {
        port,
        _certificate_fingerprint: certificate_fingerprint,
        _safety_code: safety_code,
        request,
    }
}

#[tokio::test]
async fn pairing_succeeds_over_loopback_tls_and_keeps_token_out_of_state() {
    let home = tempfile::tempdir().unwrap();
    let server = spawn_loopback_pairing_server(home.path()).await;
    let secrets = Arc::new(MemorySecretStore::new());
    let mut client = RemoteClient::new("127.0.0.1", server.addr.port(), secrets.clone()).unwrap();

    let start = client.begin_pairing("Loopback Client").await.unwrap();
    assert_eq!(start.server_fingerprint, server.certificate_fingerprint);
    assert_eq!(
        start.safety_code,
        pairing_safety_code_for_certificate_fingerprint(&server.certificate_fingerprint)
    );
    assert_eq!(client.status(), ClientStatus::Pairing);

    let pending = client.poll_pairing().await.unwrap();
    assert!(matches!(pending, PairingPoll::Pending));

    approve_pending_pairing_request(&server, &start.request_id).await;
    let approved = client.poll_pairing().await.unwrap();
    let PairingPoll::Approved { device_id, .. } = approved else {
        panic!("expected approved pairing");
    };
    assert_eq!(client.status(), ClientStatus::Connected);
    assert_eq!(client.device_id().unwrap(), device_id);
    assert!(client.safety_code().is_none());
    assert!(client.request_id().is_none());

    let debug = format!("{client:?}");
    assert!(!debug.contains("oasr_"));
    let token = secrets
        .load_secret(&credential_account(
            &server.certificate_fingerprint,
            &device_id,
        ))
        .unwrap()
        .expect("token stored");
    assert!(token.starts_with("oasr_"));
}

#[tokio::test]
async fn poll_rejects_invalid_request_id_before_the_network() {
    let mut client = RemoteClient::with_memory_store("127.0.0.1", 8080).unwrap();
    let error = client.poll_pairing().await.unwrap_err();
    assert!(error.message().contains("not in progress"));
}

#[tokio::test]
async fn pairing_response_with_invalid_request_id_is_rejected() {
    let server = spawn_test_tls_json_server(
        "202 Accepted",
        |_, safety_code| {
            serde_json::json!({
                "request_id": "req-test",
                "device_name": "Test",
                "created_at_unix_secs": 1,
                "safety_code": safety_code,
                "status": "pending"
            })
        },
        true,
    )
    .await;
    let mut client = RemoteClient::with_memory_store("127.0.0.1", server.port).unwrap();
    let error = client.begin_pairing("Test").await.unwrap_err();
    assert!(
        error
            .message()
            .contains("Remote pairing request id is invalid")
    );
    let _ = server.request.await.unwrap();
}

#[tokio::test]
async fn pairing_response_with_mismatched_safety_code_is_rejected() {
    let server = spawn_test_tls_json_server(
        "202 Accepted",
        |_, _| {
            serde_json::json!({
                "request_id": "0123456789abcdef0123456789abcdef",
                "device_name": "Test",
                "created_at_unix_secs": 1,
                "safety_code": "0000-0000",
                "status": "pending"
            })
        },
        true,
    )
    .await;
    let mut client = RemoteClient::with_memory_store("127.0.0.1", server.port).unwrap();
    let error = client.begin_pairing("Test").await.unwrap_err();
    assert!(error.message().contains("safety code did not match"));
    let _ = server.request.await.unwrap();
}

#[tokio::test]
async fn credential_poll_and_transcription_fail_closed_on_fingerprint_change() {
    let home = tempfile::tempdir().unwrap();
    let server_a = spawn_loopback_pairing_server(home.path()).await;
    let server_b = spawn_loopback_pairing_server(home.path()).await;
    assert_ne!(
        server_a.certificate_fingerprint,
        server_b.certificate_fingerprint
    );

    let secrets = Arc::new(MemorySecretStore::new());
    let mut client_a =
        RemoteClient::new("127.0.0.1", server_a.addr.port(), secrets.clone()).unwrap();
    let start = client_a.begin_pairing("Fingerprint Client").await.unwrap();
    approve_pending_pairing_request(&server_a, &start.request_id).await;
    let PairingPoll::Approved { device_id, .. } = client_a.poll_pairing().await.unwrap() else {
        panic!("expected approved pairing");
    };

    let mut poll_b = RemoteClient::with_memory_store("127.0.0.1", server_b.addr.port()).unwrap();
    poll_b
        .resume_pairing(&start.request_id, &server_a.certificate_fingerprint)
        .unwrap();
    let poll_error = poll_b.poll_pairing().await.unwrap_err();
    assert!(
        poll_error.message().contains("fingerprint changed"),
        "{}",
        poll_error.message()
    );

    let mut transcribe_b = RemoteClient::new("127.0.0.1", server_b.addr.port(), secrets).unwrap();
    transcribe_b
        .restore_connected(&server_a.certificate_fingerprint, &device_id)
        .unwrap();
    let transcribe_error = transcribe_b
        .transcribe_pcm_s16("whisper-tiny", &[0i16; 16], 16_000)
        .await
        .unwrap_err();
    assert!(
        transcribe_error.message().contains("fingerprint changed"),
        "{}",
        transcribe_error.message()
    );

    let ws_error = transcribe_b.realtime_connect().await.unwrap_err();
    assert!(
        ws_error.message().contains("fingerprint changed"),
        "{}",
        ws_error.message()
    );
}

#[tokio::test]
async fn transcribe_pcm_over_paired_loopback_tls() {
    let home = tempfile::tempdir().unwrap();
    let server = spawn_loopback_pairing_server(home.path()).await;
    let mut client = RemoteClient::with_memory_store("127.0.0.1", server.addr.port()).unwrap();
    let start = client.begin_pairing("Transcribe Client").await.unwrap();
    approve_pending_pairing_request(&server, &start.request_id).await;
    client.poll_pairing().await.unwrap();

    let samples = vec![0i16; 1600];
    let response = client
        .transcribe_pcm_s16("whisper-tiny", &samples, 16_000)
        .await
        .unwrap();
    assert_eq!(response.status, 200);
    let text = response.text().unwrap();
    assert!(text.contains("OpenASR mock transcription"), "{text}");
}

#[tokio::test]
async fn tofu_connects_by_ip_when_cert_san_is_only_localhost() {
    let home = tempfile::tempdir().unwrap();
    let server =
        spawn_loopback_pairing_server_with_sans(home.path(), &["localhost".to_string()]).await;
    let mut client = RemoteClient::with_memory_store("127.0.0.1", server.addr.port()).unwrap();
    let start = client.begin_pairing("LAN SAN Client").await.unwrap();
    assert_eq!(start.server_fingerprint, server.certificate_fingerprint);
}

#[tokio::test]
async fn realtime_connects_then_revoke_cannot_upgrade_ws() {
    let home = tempfile::tempdir().unwrap();
    let server = spawn_loopback_pairing_server(home.path()).await;
    let mut client = RemoteClient::with_memory_store("127.0.0.1", server.addr.port()).unwrap();
    let start = client.begin_pairing("Realtime Client").await.unwrap();
    approve_pending_pairing_request(&server, &start.request_id).await;
    let PairingPoll::Approved { device_id, .. } = client.poll_pairing().await.unwrap() else {
        panic!("expected approved pairing");
    };

    let mut session = client.realtime_connect().await.unwrap();
    let first = tokio::time::timeout(Duration::from_secs(5), session.next_text())
        .await
        .unwrap()
        .unwrap()
        .expect("capabilities");
    assert!(first.contains("session.capabilities"), "{first}");
    session.close().await.unwrap();

    revoke_loopback_pairing(&server, &device_id).await;
    let error = client.realtime_connect().await.unwrap_err();
    assert!(
        error.message().contains("401") || error.message().contains("WebSocket"),
        "{}",
        error.message()
    );
}

fn realtime_event_type(text: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(text)
        .ok()?
        .get("type")?
        .as_str()
        .map(str::to_string)
}

async fn pair_loopback_client(
    device_name: &str,
) -> (
    tempfile::TempDir,
    openasr_server::testing::LoopbackTlsServer,
    RemoteClient,
) {
    let home = tempfile::tempdir().unwrap();
    let server = spawn_loopback_pairing_server(home.path()).await;
    let mut client = RemoteClient::with_memory_store("127.0.0.1", server.addr.port()).unwrap();
    let start = client.begin_pairing(device_name).await.unwrap();
    approve_pending_pairing_request(&server, &start.request_id).await;
    client.poll_pairing().await.unwrap();
    (home, server, client)
}

async fn next_matching_event(
    session: &mut openasr_client::RealtimeSession,
    wanted: &str,
) -> String {
    loop {
        let text = tokio::time::timeout(Duration::from_secs(5), session.next_text())
            .await
            .unwrap_or_else(|_| panic!("timed out waiting for {wanted}"))
            .unwrap()
            .unwrap_or_else(|| panic!("realtime socket closed while waiting for {wanted}"));
        if realtime_event_type(&text).as_deref() == Some(wanted) {
            return text;
        }
        if realtime_event_type(&text).as_deref() == Some("error") {
            panic!("unexpected realtime error while waiting for {wanted}: {text}");
        }
    }
}

#[tokio::test]
async fn realtime_binary_before_session_start_is_rejected() {
    let (_home, _server, client) = pair_loopback_client("Realtime Gate Client").await;
    let mut session = client.realtime_connect().await.unwrap();
    let capabilities = tokio::time::timeout(Duration::from_secs(5), session.next_text())
        .await
        .unwrap()
        .unwrap()
        .expect("capabilities");
    assert!(
        capabilities.contains("session.capabilities"),
        "{capabilities}"
    );

    session.send_pcm16le(&[0u8; 640]).await.unwrap();
    let error = next_matching_event(&mut session, "error").await;
    assert!(error.contains("session.start first"), "{error}");
}

#[tokio::test]
async fn realtime_handshake_then_binary_is_accepted() {
    let (_home, _server, client) = pair_loopback_client("Realtime Handshake Client").await;
    let mut session = client.realtime_start("whisper-tiny").await.unwrap();
    session.send_pcm16le(&[0u8; 640]).await.unwrap();
    match tokio::time::timeout(Duration::from_millis(750), session.next_text()).await {
        Ok(Ok(Some(text))) => {
            assert_ne!(
                realtime_event_type(&text).as_deref(),
                Some("error"),
                "{text}"
            );
            assert!(!text.contains("session.start first"), "{text}");
        }
        Ok(Ok(None)) => panic!("realtime socket closed after a post-handshake binary frame"),
        Ok(Err(error)) => panic!("{}", error.message()),
        Err(_) => {}
    }
    session.close().await.unwrap();
}

#[tokio::test]
async fn realtime_worker_handshakes_before_feed() {
    let (_home, _server, client) = pair_loopback_client("Realtime Worker Client").await;
    let received = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let worker = client
        .realtime_start_worker("whisper-tiny", {
            let received = received.clone();
            move |text| received.lock().expect("callback mutex").push(text)
        })
        .await
        .unwrap();
    worker.try_send_pcm16le(vec![0u8; 640]).unwrap();
    tokio::time::sleep(Duration::from_millis(400)).await;
    worker.try_close();
    let frames = received.lock().expect("callback mutex").clone();
    assert!(
        frames.iter().all(
            |frame| realtime_event_type(frame).as_deref() != Some("error")
                || !frame.contains("session.start first")
        ),
        "{frames:?}"
    );
}

#[tokio::test]
async fn restore_connected_then_transcribe_over_loopback() {
    let home = tempfile::tempdir().unwrap();
    let server = spawn_loopback_pairing_server(home.path()).await;
    let secrets = Arc::new(MemorySecretStore::new());
    let mut paired = RemoteClient::new("127.0.0.1", server.addr.port(), secrets.clone()).unwrap();
    let start = paired.begin_pairing("Restore Client").await.unwrap();
    approve_pending_pairing_request(&server, &start.request_id).await;
    let PairingPoll::Approved { device_id, .. } = paired.poll_pairing().await.unwrap() else {
        panic!("expected approved pairing");
    };

    let mut restored = RemoteClient::new("127.0.0.1", server.addr.port(), secrets).unwrap();
    restored
        .restore_connected(&server.certificate_fingerprint, &device_id)
        .unwrap();
    let response = restored
        .transcribe_pcm_s16("whisper-tiny", &[0i16; 1600], 16_000)
        .await
        .unwrap();
    assert_eq!(response.status, 200);
    assert!(
        response
            .text()
            .unwrap()
            .contains("OpenASR mock transcription")
    );
}
