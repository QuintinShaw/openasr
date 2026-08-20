use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures_util::{SinkExt, StreamExt};
use openasr_core::pairing_safety_code_for_certificate_fingerprint;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::{
    Message as WsMessage,
    client::IntoClientRequest,
    http::{HeaderValue, header::AUTHORIZATION},
};

use crate::{
    ClientError, MemorySecretStore, SecretStore, credential_account,
    tls::{
        REMOTE_COMPUTE_CLIENT_VALUE, REMOTE_COMPUTE_HEADER, content_type_of, host_header,
        open_remote_tls_connection, send_remote_http_request,
    },
};

const PAIRING_REQUEST_ID_HEX_LEN: usize = 32;

/// Wire values for `audio.input.configure` / `session.start`. Must match
/// `openasr-server`'s `ClientMessage` and the Desktop realtime client: 16 kHz
/// mono `pcm_s16le`, 20 ms frames. Do not invent fields.
const REALTIME_AUDIO_ENCODING: &str = "pcm_s16le";
const REALTIME_SAMPLE_RATE_HZ: u32 = 16_000;
const REALTIME_CHANNELS: u16 = 1;
const REALTIME_FRAME_DURATION_MS: u32 = 20;
const REALTIME_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// High-level remote compute client: pairing, pinned HTTPS transcription, pinned WSS.
///
/// Trust decisions (certificate fingerprint, pairing safety code, TOFU pin
/// comparison) stay in this crate. Host apps display the safety code; they
/// never receive observed vs expected fingerprints to compare themselves.
pub struct RemoteClient {
    host: String,
    port: u16,
    secrets: Arc<dyn SecretStore>,
    status: ClientStatus,
    request_id: Option<String>,
    safety_code: Option<String>,
    server_fingerprint: Option<String>,
    device_id: Option<String>,
}

impl std::fmt::Debug for RemoteClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RemoteClient")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("status", &self.status)
            .field("request_id", &self.request_id)
            .field("safety_code", &self.safety_code)
            .field("server_fingerprint", &self.server_fingerprint)
            .field("device_id", &self.device_id)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientStatus {
    Idle,
    Pairing,
    Connected,
}

/// Values a host UI may display after [`RemoteClient::begin_pairing`].
///
/// `safety_code` is the human-readable pairing code derived from the pinned
/// certificate fingerprint. The TLS-vs-response match has already been
/// checked; do not re-compare fingerprints in the host app.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairingStart {
    pub request_id: String,
    pub safety_code: String,
    pub server_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairingPoll {
    Pending,
    Approved {
        device_id: String,
        issued_at_unix_secs: u64,
    },
}

#[derive(Debug, Clone)]
pub struct TranscriptionResponse {
    pub status: u16,
    pub content_type: String,
    pub body: String,
}

impl TranscriptionResponse {
    pub fn text(&self) -> Result<String, ClientError> {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&self.body)
            && let Some(text) = value.get("text").and_then(|value| value.as_str())
        {
            return Ok(text.to_string());
        }
        if self.status == 200 {
            return Ok(self.body.trim().to_string());
        }
        Err(ClientError::new(format!(
            "OpenASR remote transcription failed with HTTP {}: {}",
            self.status,
            self.body.trim()
        )))
    }
}

#[derive(Debug, Deserialize)]
struct PairingRequestResponse {
    request_id: String,
    safety_code: Option<String>,
    status: String,
}

#[derive(Debug, Deserialize)]
struct PairingCredentialResponse {
    device_id: String,
    issued_at_unix_secs: u64,
    bearer_token: String,
}

impl RemoteClient {
    pub fn new(
        host: impl Into<String>,
        port: u16,
        secrets: Arc<dyn SecretStore>,
    ) -> Result<Self, ClientError> {
        let host = normalize_host(&host.into())?;
        let port = normalize_port(port)?;
        Ok(Self {
            host,
            port,
            secrets,
            status: ClientStatus::Idle,
            request_id: None,
            safety_code: None,
            server_fingerprint: None,
            device_id: None,
        })
    }

    pub fn with_memory_store(host: impl Into<String>, port: u16) -> Result<Self, ClientError> {
        Self::new(host, port, Arc::new(MemorySecretStore::new()))
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn status(&self) -> ClientStatus {
        self.status
    }

    pub fn safety_code(&self) -> Option<&str> {
        self.safety_code.as_deref()
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub fn server_fingerprint(&self) -> Option<&str> {
        self.server_fingerprint.as_deref()
    }

    pub fn device_id(&self) -> Option<&str> {
        self.device_id.as_deref()
    }

    /// Start pairing: TOFU handshake, then fail closed unless the server's
    /// safety code matches the code derived from the observed certificate
    /// fingerprint.
    pub async fn begin_pairing(&mut self, device_name: &str) -> Result<PairingStart, ClientError> {
        let device_name = device_name.trim();
        if device_name.is_empty() {
            return Err(ClientError::new("Remote pairing device name is required."));
        }
        let body = serde_json::json!({ "device_name": device_name }).to_string();
        let response = send_remote_http_request(
            &self.host,
            self.port,
            "POST",
            "/v1/pairing/requests",
            body.into_bytes(),
            Some("application/json"),
            "application/json",
            None,
            None,
            &[],
        )
        .await?;
        let server_fingerprint = response.server_fingerprint;
        let expected_safety_code =
            pairing_safety_code_for_certificate_fingerprint(&server_fingerprint);
        if response.status != 202 {
            return Err(ClientError::new(format!(
                "OpenASR remote pairing request failed with HTTP {}: {}",
                response.status,
                String::from_utf8_lossy(&response.body).trim()
            )));
        }
        let parsed: PairingRequestResponse =
            serde_json::from_slice(&response.body).map_err(|error| {
                ClientError::new(format!(
                    "Could not parse OpenASR remote pairing response: {error}"
                ))
            })?;
        if parsed.status != "pending" {
            return Err(ClientError::new(format!(
                "OpenASR remote pairing returned unexpected status '{}'.",
                parsed.status
            )));
        }
        let request_id = normalize_pairing_request_id(&parsed.request_id)?;
        let safety_code = parsed.safety_code.ok_or_else(|| {
            ClientError::new(
                "OpenASR remote pairing response did not include a TLS-derived safety code.",
            )
        })?;
        if safety_code != expected_safety_code {
            return Err(ClientError::new(
                "OpenASR remote pairing safety code did not match the TLS certificate fingerprint.",
            ));
        }
        self.status = ClientStatus::Pairing;
        self.request_id = Some(request_id.clone());
        self.safety_code = Some(safety_code.clone());
        self.server_fingerprint = Some(server_fingerprint.clone());
        self.device_id = None;
        Ok(PairingStart {
            request_id,
            safety_code,
            server_fingerprint,
        })
    }

    pub async fn poll_pairing(&mut self) -> Result<PairingPoll, ClientError> {
        if self.status != ClientStatus::Pairing {
            return Err(ClientError::new("Remote pairing is not in progress."));
        }
        let request_id = self
            .request_id
            .clone()
            .ok_or_else(|| ClientError::new("Remote pairing is missing a request id."))?;
        let server_fingerprint = self.server_fingerprint.clone().ok_or_else(|| {
            ClientError::new("Remote pairing is missing a pinned server fingerprint.")
        })?;
        let request_id = normalize_pairing_request_id(&request_id)?;
        let response = send_remote_http_request(
            &self.host,
            self.port,
            "GET",
            &format!("/v1/pairing/requests/{request_id}/credential"),
            Vec::new(),
            None,
            "application/json",
            Some(&server_fingerprint),
            None,
            &[],
        )
        .await?;
        match response.status {
            202 => Ok(PairingPoll::Pending),
            200 => {
                let credential: PairingCredentialResponse = serde_json::from_slice(&response.body)
                    .map_err(|error| {
                        ClientError::new(format!(
                            "Could not parse OpenASR remote credential response: {error}"
                        ))
                    })?;
                if credential.device_id.trim().is_empty() {
                    return Err(ClientError::new(
                        "OpenASR remote credential response did not include a device id.",
                    ));
                }
                if !credential.bearer_token.starts_with("oasr_") {
                    return Err(ClientError::new(
                        "OpenASR remote credential response had an invalid token.",
                    ));
                }
                let account = credential_account(&server_fingerprint, &credential.device_id);
                self.secrets
                    .store_secret(&account, &credential.bearer_token)?;
                self.status = ClientStatus::Connected;
                self.device_id = Some(credential.device_id.clone());
                self.request_id = None;
                self.safety_code = None;
                Ok(PairingPoll::Approved {
                    device_id: credential.device_id,
                    issued_at_unix_secs: credential.issued_at_unix_secs,
                })
            }
            status => Err(ClientError::new(format!(
                "OpenASR remote credential request failed with HTTP {status}: {}",
                String::from_utf8_lossy(&response.body).trim()
            ))),
        }
    }

    pub fn cancel_pairing(&mut self) {
        if let (Some(fingerprint), Some(device_id)) = (&self.server_fingerprint, &self.device_id) {
            let account = credential_account(fingerprint, device_id);
            let _ = self.secrets.delete_secret(&account);
        }
        self.status = ClientStatus::Idle;
        self.request_id = None;
        self.safety_code = None;
        self.server_fingerprint = None;
        self.device_id = None;
    }

    /// Resume an in-progress pairing after a process restart. `request_id` and
    /// `server_fingerprint` must be the values previously returned by
    /// [`Self::begin_pairing`]; the host app must not derive or compare them.
    pub fn resume_pairing(
        &mut self,
        request_id: &str,
        server_fingerprint: &str,
    ) -> Result<(), ClientError> {
        let request_id = normalize_pairing_request_id(request_id)?;
        let server_fingerprint = crate::normalize_fingerprint(server_fingerprint);
        if server_fingerprint.len() != 64 {
            return Err(ClientError::new(
                "Remote pairing is missing a pinned server fingerprint.",
            ));
        }
        self.status = ClientStatus::Pairing;
        self.request_id = Some(request_id);
        self.server_fingerprint = Some(server_fingerprint);
        self.safety_code = None;
        self.device_id = None;
        Ok(())
    }

    /// Restore a connected session from persisted fingerprint + device id.
    /// The bearer token is loaded from the injected secret store, never from
    /// the caller.
    pub fn restore_connected(
        &mut self,
        server_fingerprint: &str,
        device_id: &str,
    ) -> Result<(), ClientError> {
        let server_fingerprint = crate::normalize_fingerprint(server_fingerprint);
        let device_id = device_id.trim();
        if server_fingerprint.len() != 64 || device_id.is_empty() {
            return Err(ClientError::new(
                "Remote compute pairing credentials are incomplete.",
            ));
        }
        let account = credential_account(&server_fingerprint, device_id);
        let token = self
            .secrets
            .load_secret(&account)?
            .ok_or_else(|| ClientError::new("Remote compute pairing token is missing."))?;
        if !token.starts_with("oasr_") {
            return Err(ClientError::new(
                "OpenASR remote credential store had an invalid token.",
            ));
        }
        self.status = ClientStatus::Connected;
        self.server_fingerprint = Some(server_fingerprint);
        self.device_id = Some(device_id.to_string());
        self.request_id = None;
        self.safety_code = None;
        Ok(())
    }

    fn bearer_token(&self) -> Result<String, ClientError> {
        if self.status != ClientStatus::Connected {
            return Err(ClientError::new("Remote compute client is not connected."));
        }
        let fingerprint = self
            .server_fingerprint
            .as_deref()
            .ok_or_else(|| ClientError::new("Remote compute server fingerprint is missing."))?;
        let device_id = self
            .device_id
            .as_deref()
            .ok_or_else(|| ClientError::new("Remote compute device id is missing."))?;
        self.secrets
            .load_secret(&credential_account(fingerprint, device_id))?
            .ok_or_else(|| ClientError::new("Remote compute pairing token is missing."))
    }

    pub async fn transcribe_wav_bytes(
        &self,
        model: &str,
        wav: &[u8],
        file_name: &str,
    ) -> Result<TranscriptionResponse, ClientError> {
        let fingerprint = self
            .server_fingerprint
            .clone()
            .ok_or_else(|| ClientError::new("Remote compute server fingerprint is missing."))?;
        let token = self.bearer_token()?;
        let upload = build_transcription_multipart(model, wav, file_name)?;
        let response = send_remote_http_request(
            &self.host,
            self.port,
            "POST",
            "/v1/audio/transcriptions",
            upload.body,
            Some(&upload.content_type),
            "application/json, text/plain",
            Some(&fingerprint),
            Some(&token),
            &[(REMOTE_COMPUTE_HEADER, REMOTE_COMPUTE_CLIENT_VALUE)],
        )
        .await?;
        let body = String::from_utf8(response.body).map_err(|error| {
            ClientError::new(format!(
                "OpenASR remote transcription response was not UTF-8: {error}"
            ))
        })?;
        Ok(TranscriptionResponse {
            status: response.status,
            content_type: content_type_of(&response.headers),
            body,
        })
    }

    pub async fn transcribe_pcm_f32(
        &self,
        model: &str,
        samples: &[f32],
        sample_rate_hz: u32,
    ) -> Result<TranscriptionResponse, ClientError> {
        let pcm: Vec<i16> = samples
            .iter()
            .map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)
            .collect();
        self.transcribe_pcm_s16(model, &pcm, sample_rate_hz).await
    }

    pub async fn transcribe_pcm_s16(
        &self,
        model: &str,
        samples: &[i16],
        sample_rate_hz: u32,
    ) -> Result<TranscriptionResponse, ClientError> {
        if sample_rate_hz == 0 {
            return Err(ClientError::new("PCM sample rate must be greater than 0."));
        }
        let wav = encode_wav_s16le_mono(samples, sample_rate_hz);
        self.transcribe_wav_bytes(model, &wav, "audio.wav").await
    }

    pub async fn realtime_connect(&self) -> Result<RealtimeSession, ClientError> {
        let fingerprint = self
            .server_fingerprint
            .as_deref()
            .ok_or_else(|| ClientError::new("Remote compute server fingerprint is missing."))?;
        let token = self.bearer_token()?;
        connect_remote_realtime_socket(&self.host, self.port, fingerprint, &token).await
    }

    /// Connect and complete the server handshake (`audio.input.configure` then
    /// `session.start`) so binary PCM frames are legal.
    pub async fn realtime_start(&self, model: &str) -> Result<RealtimeSession, ClientError> {
        let mut session = self.realtime_connect().await?;
        session.start_session(model).await?;
        Ok(session)
    }

    pub async fn realtime_start_worker(
        &self,
        model: &str,
        on_message: impl FnMut(String) + Send + 'static,
    ) -> Result<RealtimeWorker, ClientError> {
        let fingerprint = self
            .server_fingerprint
            .clone()
            .ok_or_else(|| ClientError::new("Remote compute server fingerprint is missing."))?;
        let token = self.bearer_token()?;
        spawn_realtime_worker(
            self.host.clone(),
            self.port,
            fingerprint,
            token,
            model.to_string(),
            on_message,
        )
        .await
    }
}

pub struct RealtimeSession {
    socket:
        tokio_tungstenite::WebSocketStream<tokio_rustls::client::TlsStream<tokio::net::TcpStream>>,
}

impl std::fmt::Debug for RealtimeSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("RealtimeSession")
    }
}

impl RealtimeSession {
    /// Send `audio.input.configure` then `session.start` and wait until the
    /// server admits audio (`audio.input.started`). Binary frames sent before
    /// this returns are rejected by the server.
    pub async fn start_session(&mut self, model: &str) -> Result<(), ClientError> {
        self.send_text(&realtime_audio_input_configure_json())
            .await?;
        self.send_text(&realtime_session_start_json(model)?).await?;
        tokio::time::timeout(REALTIME_HANDSHAKE_TIMEOUT, async {
            loop {
                match self.next_text().await? {
                    Some(text) => match realtime_event_type(&text).as_deref() {
                        Some("audio.input.started") => return Ok(()),
                        Some("error") => {
                            return Err(ClientError::new(format!(
                                "OpenASR remote realtime handshake failed: {}",
                                realtime_error_message(&text).as_deref().unwrap_or(&text)
                            )));
                        }
                        Some("session.closed") => {
                            return Err(ClientError::new(
                                "OpenASR remote realtime handshake closed before audio.input.started.",
                            ));
                        }
                        _ => {}
                    },
                    None => {
                        return Err(ClientError::new(
                            "OpenASR remote realtime handshake closed before audio.input.started.",
                        ));
                    }
                }
            }
        })
        .await
        .map_err(|_| {
            ClientError::new(
                "Timed out waiting for OpenASR remote realtime audio.input.started.",
            )
        })?
    }

    pub async fn send_pcm16le(&mut self, bytes: &[u8]) -> Result<(), ClientError> {
        if bytes.is_empty() {
            return Err(ClientError::new(
                "OpenASR remote realtime audio frame is empty.",
            ));
        }
        if !bytes.len().is_multiple_of(2) {
            return Err(ClientError::new(
                "OpenASR remote realtime audio frame must be PCM16LE bytes.",
            ));
        }
        self.socket
            .send(WsMessage::Binary(bytes.to_vec().into()))
            .await
            .map_err(|error| {
                ClientError::new(format!(
                    "Could not send OpenASR remote realtime audio frame: {error}"
                ))
            })
    }

    pub async fn send_text(&mut self, text: &str) -> Result<(), ClientError> {
        self.socket
            .send(WsMessage::Text(text.to_string().into()))
            .await
            .map_err(|error| {
                ClientError::new(format!(
                    "Could not send OpenASR remote realtime control message: {error}"
                ))
            })
    }

    pub async fn next_text(&mut self) -> Result<Option<String>, ClientError> {
        loop {
            match self.socket.next().await {
                Some(Ok(WsMessage::Text(text))) => return Ok(Some(text.to_string())),
                Some(Ok(WsMessage::Binary(_))) => {}
                Some(Ok(WsMessage::Ping(payload))) => {
                    let _ = self.socket.send(WsMessage::Pong(payload)).await;
                }
                Some(Ok(WsMessage::Pong(_))) => {}
                Some(Ok(WsMessage::Frame(_))) => {}
                Some(Ok(WsMessage::Close(_))) | None => return Ok(None),
                Some(Err(error)) => {
                    return Err(ClientError::new(format!(
                        "OpenASR remote realtime WebSocket failed: {error}"
                    )));
                }
            }
        }
    }

    pub async fn close(mut self) -> Result<(), ClientError> {
        let _ = self.socket.send(WsMessage::Close(None)).await;
        Ok(())
    }
}

/// Outbound frames for a background realtime worker used by the C ABI.
pub enum RealtimeOutbound {
    Text(String),
    Binary(Vec<u8>),
    Close,
}

pub struct RealtimeWorker {
    sender: mpsc::Sender<RealtimeOutbound>,
}

impl RealtimeWorker {
    pub fn try_send_pcm16le(&self, bytes: Vec<u8>) -> Result<(), ClientError> {
        if bytes.is_empty() {
            return Err(ClientError::new(
                "OpenASR remote realtime audio frame is empty.",
            ));
        }
        if !bytes.len().is_multiple_of(2) {
            return Err(ClientError::new(
                "OpenASR remote realtime audio frame must be PCM16LE bytes.",
            ));
        }
        self.sender
            .try_send(RealtimeOutbound::Binary(bytes))
            .map_err(|error| {
                ClientError::new(format!(
                    "Could not queue OpenASR remote realtime audio frame: {error}"
                ))
            })
    }

    pub fn try_send_text(&self, text: String) -> Result<(), ClientError> {
        self.sender
            .try_send(RealtimeOutbound::Text(text))
            .map_err(|error| {
                ClientError::new(format!(
                    "Could not queue OpenASR remote realtime control message: {error}"
                ))
            })
    }

    pub fn try_close(&self) {
        let _ = self.sender.try_send(RealtimeOutbound::Close);
    }
}

pub async fn spawn_realtime_worker(
    host: String,
    port: u16,
    expected_fingerprint: String,
    bearer_token: String,
    model: String,
    mut on_message: impl FnMut(String) + Send + 'static,
) -> Result<RealtimeWorker, ClientError> {
    let (sender, mut receiver) = mpsc::channel(128);
    let mut session =
        connect_remote_realtime_socket(&host, port, &expected_fingerprint, &bearer_token).await?;
    session.start_session(&model).await?;
    let RealtimeSession { mut socket } = session;
    tokio::spawn(async move {
        loop {
            tokio::select! {
                outbound = receiver.recv() => {
                    match outbound {
                        Some(RealtimeOutbound::Text(text)) => {
                            if socket.send(WsMessage::Text(text.into())).await.is_err() {
                                break;
                            }
                        }
                        Some(RealtimeOutbound::Binary(bytes)) => {
                            if socket.send(WsMessage::Binary(bytes.into())).await.is_err() {
                                break;
                            }
                        }
                        Some(RealtimeOutbound::Close) | None => {
                            let _ = socket.send(WsMessage::Close(None)).await;
                            break;
                        }
                    }
                }
                incoming = socket.next() => {
                    match incoming {
                        Some(Ok(WsMessage::Text(text))) => on_message(text.to_string()),
                        Some(Ok(WsMessage::Ping(payload))) => {
                            let _ = socket.send(WsMessage::Pong(payload)).await;
                        }
                        Some(Ok(WsMessage::Close(_))) | None => break,
                        Some(Err(_)) => break,
                        _ => {}
                    }
                }
            }
        }
    });
    Ok(RealtimeWorker { sender })
}

pub(crate) fn realtime_audio_input_configure_json() -> String {
    serde_json::json!({
        "type": "audio.input.configure",
        "format": {
            "encoding": REALTIME_AUDIO_ENCODING,
            "sample_rate_hz": REALTIME_SAMPLE_RATE_HZ,
            "channels": REALTIME_CHANNELS
        },
        "frame_duration_ms": REALTIME_FRAME_DURATION_MS
    })
    .to_string()
}

pub(crate) fn realtime_session_start_json(model: &str) -> Result<String, ClientError> {
    let model = model.trim();
    if model.is_empty() {
        return Err(ClientError::new(
            "Remote realtime session.model is required.",
        ));
    }
    Ok(serde_json::json!({
        "type": "session.start",
        "session": { "model": model }
    })
    .to_string())
}

fn realtime_event_type(text: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(text)
        .ok()?
        .get("type")?
        .as_str()
        .map(str::to_string)
}

fn realtime_error_message(text: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(text).ok()?;
    if value.get("type")?.as_str()? != "error" {
        return None;
    }
    value
        .get("message")
        .and_then(|message| message.as_str())
        .map(str::to_string)
}

async fn connect_remote_realtime_socket(
    host: &str,
    port: u16,
    expected_fingerprint: &str,
    bearer_token: &str,
) -> Result<RealtimeSession, ClientError> {
    let (tls, _) = open_remote_tls_connection(host, port, Some(expected_fingerprint)).await?;
    let url = format!("wss://{}/v1/audio/realtime", host_header(host, port));
    let mut request = url.into_client_request().map_err(|error| {
        ClientError::new(format!(
            "Could not build OpenASR remote realtime request: {error}"
        ))
    })?;
    let authorization =
        HeaderValue::from_str(&format!("Bearer {bearer_token}")).map_err(|error| {
            ClientError::new(format!(
                "OpenASR remote realtime token was not header-safe: {error}"
            ))
        })?;
    request.headers_mut().insert(AUTHORIZATION, authorization);
    request.headers_mut().insert(
        tokio_tungstenite::tungstenite::http::HeaderName::from_static("x-openasr-remote-compute"),
        HeaderValue::from_static(REMOTE_COMPUTE_CLIENT_VALUE),
    );
    let (socket, response) = tokio_tungstenite::client_async(request, tls)
        .await
        .map_err(|error| {
            ClientError::new(format!(
                "Could not open OpenASR remote realtime WebSocket: {error}"
            ))
        })?;
    if !response.status().is_informational() && response.status().as_u16() != 101 {
        return Err(ClientError::new(format!(
            "OpenASR remote realtime WebSocket failed with HTTP {}",
            response.status()
        )));
    }
    Ok(RealtimeSession { socket })
}

struct TranscriptionMultipart {
    content_type: String,
    body: Vec<u8>,
}

fn build_transcription_multipart(
    model: &str,
    wav: &[u8],
    file_name: &str,
) -> Result<TranscriptionMultipart, ClientError> {
    if model.trim().is_empty() {
        return Err(ClientError::new("Remote transcription model is required."));
    }
    let boundary = format!(
        "openasr-remote-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let mut body = Vec::with_capacity(wav.len() + 256);
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\nContent-Type: application/octet-stream\r\n\r\n",
            escape_multipart_token(file_name)
        )
        .as_bytes(),
    );
    body.extend_from_slice(wav);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{}\r\n--{boundary}--\r\n",
            model.trim()
        )
        .as_bytes(),
    );
    Ok(TranscriptionMultipart {
        content_type: format!("multipart/form-data; boundary={boundary}"),
        body,
    })
}

fn escape_multipart_token(value: &str) -> String {
    value
        .chars()
        .filter(|character| *character != '\r' && *character != '\n')
        .flat_map(|character| match character {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            character => vec![character],
        })
        .collect()
}

fn encode_wav_s16le_mono(samples: &[i16], sample_rate_hz: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate_hz.to_le_bytes());
    let byte_rate = sample_rate_hz.saturating_mul(2);
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

fn normalize_host(value: &str) -> Result<String, ClientError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ClientError::new("Remote server address is required."));
    }
    if trimmed.contains('/') || trimmed.contains('\\') {
        return Err(ClientError::new(
            "Host must be an address or hostname, not a URL.",
        ));
    }
    Ok(trimmed.to_string())
}

fn normalize_port(port: u16) -> Result<u16, ClientError> {
    if port == 0 {
        return Err(ClientError::new("Port must be between 1 and 65535."));
    }
    Ok(port)
}

pub(crate) fn normalize_pairing_request_id(request_id: &str) -> Result<String, ClientError> {
    let request_id = request_id.trim();
    if request_id.len() != PAIRING_REQUEST_ID_HEX_LEN
        || !request_id.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(ClientError::new("Remote pairing request id is invalid."));
    }
    Ok(request_id.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ClientStatus, MemorySecretStore, credential_account};
    use std::sync::Arc;

    #[test]
    fn host_rejects_urls() {
        assert!(normalize_host("https://192.168.1.2").is_err());
        assert_eq!(normalize_host(" 192.168.1.2 ").unwrap(), "192.168.1.2");
    }

    #[test]
    fn pairing_request_id_must_be_32_hex() {
        assert!(normalize_pairing_request_id("req-test").is_err());
        assert_eq!(
            normalize_pairing_request_id("0123456789abcdef0123456789ABCDEF").unwrap(),
            "0123456789abcdef0123456789abcdef"
        );
    }

    #[test]
    fn wav_header_is_44_bytes_plus_pcm() {
        let wav = encode_wav_s16le_mono(&[0, 1, -1], 16_000);
        assert_eq!(wav.len(), 44 + 6);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
    }

    #[test]
    fn audio_input_configure_json_matches_server_client_message() {
        let value: serde_json::Value =
            serde_json::from_str(&realtime_audio_input_configure_json()).unwrap();
        assert_eq!(value["type"], "audio.input.configure");
        assert_eq!(value["format"]["encoding"], "pcm_s16le");
        assert_eq!(value["format"]["sample_rate_hz"], 16_000);
        assert_eq!(value["format"]["channels"], 1);
        assert_eq!(value["frame_duration_ms"], 20);
        let format = value["format"].as_object().unwrap();
        assert_eq!(format.len(), 3);
        assert_eq!(value.as_object().unwrap().len(), 3);
    }

    #[test]
    fn session_start_json_matches_server_client_message_and_does_not_invent_fields() {
        let value: serde_json::Value =
            serde_json::from_str(&realtime_session_start_json("whisper-tiny").unwrap()).unwrap();
        assert_eq!(value["type"], "session.start");
        assert_eq!(value["session"]["model"], "whisper-tiny");
        assert_eq!(value["session"].as_object().unwrap().len(), 1);
        assert_eq!(value.as_object().unwrap().len(), 2);
        assert!(realtime_session_start_json("  ").is_err());
    }

    #[test]
    fn restore_connected_fails_closed_without_token() {
        let mut client = RemoteClient::with_memory_store("127.0.0.1", 8080).unwrap();
        let error = client
            .restore_connected(&"ab".repeat(32), "device-1")
            .unwrap_err();
        assert!(
            error.message().contains("token is missing"),
            "{}",
            error.message()
        );
    }

    #[test]
    fn restore_connected_fails_closed_when_fingerprint_does_not_match_stored_account() {
        let secrets = Arc::new(MemorySecretStore::new());
        secrets
            .store_secret(
                &credential_account(&"aa".repeat(32), "device-1"),
                "oasr_stored",
            )
            .unwrap();
        let mut client = RemoteClient::new("127.0.0.1", 8080, secrets).unwrap();
        let error = client
            .restore_connected(&"bb".repeat(32), "device-1")
            .unwrap_err();
        assert!(
            error.message().contains("token is missing"),
            "{}",
            error.message()
        );
    }

    #[test]
    fn restore_connected_loads_token_from_secret_store() {
        let fingerprint = "aa".repeat(32);
        let secrets = Arc::new(MemorySecretStore::new());
        secrets
            .store_secret(&credential_account(&fingerprint, "device-1"), "oasr_stored")
            .unwrap();
        let mut client = RemoteClient::new("127.0.0.1", 8080, secrets).unwrap();
        client.restore_connected(&fingerprint, "device-1").unwrap();
        assert_eq!(client.status(), ClientStatus::Connected);
        assert_eq!(client.device_id(), Some("device-1"));
        assert_eq!(
            client.server_fingerprint().as_deref(),
            Some(fingerprint.as_str())
        );
    }

    #[test]
    fn restore_connected_rejects_invalid_stored_token() {
        let fingerprint = "aa".repeat(32);
        let secrets = Arc::new(MemorySecretStore::new());
        secrets
            .store_secret(&credential_account(&fingerprint, "device-1"), "not-a-token")
            .unwrap();
        let mut client = RemoteClient::new("127.0.0.1", 8080, secrets).unwrap();
        let error = client
            .restore_connected(&fingerprint, "device-1")
            .unwrap_err();
        assert!(
            error.message().contains("invalid token"),
            "{}",
            error.message()
        );
    }
}
