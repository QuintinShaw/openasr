//! Pairing + remote-device auth: request/credential records, views, endpoint
//! handlers, and the server-auth middleware. Pure code-motion from `lib.rs`.

use crate::*;

pub(crate) const PAIRING_REQUEST_ID_HEX_LEN: usize = 32;
pub(crate) const PAIRING_DEVICE_ID_HEX_LEN: usize = 24;

pub(crate) fn normalize_pairing_request_id(request_id: &str) -> Result<String, PairingError> {
    normalize_hex_route_id(request_id, PAIRING_REQUEST_ID_HEX_LEN)
        .ok_or(PairingError::InvalidRequestId)
}

pub(crate) fn normalize_pairing_device_id(device_id: &str) -> Result<String, PairingError> {
    normalize_hex_route_id(device_id, PAIRING_DEVICE_ID_HEX_LEN)
        .ok_or(PairingError::InvalidDeviceId)
}

pub(crate) fn normalize_hex_route_id(value: &str, expected_len: usize) -> Option<String> {
    let value = value.trim();
    if value.len() != expected_len || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    Some(value.to_ascii_lowercase())
}

pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

pub(crate) fn normalized_device_name(device_name: String) -> Result<String, PairingError> {
    let device_name = device_name.trim();
    if device_name.is_empty() {
        return Err(PairingError::InvalidDeviceName);
    }
    Ok(device_name.chars().take(128).collect())
}

pub(crate) fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Debug, Default)]
pub(crate) struct PairingRegistry {
    pub(crate) pending: HashMap<String, PairingRequestRecord>,
    pub(crate) credentials: HashMap<String, DeviceCredentialRecord>,
    pub(crate) credential_claims: HashMap<String, PairingCredentialView>,
    pub(crate) store_path: Option<PathBuf>,
}

pub(crate) fn load_pairing_credentials(
    path: &Path,
) -> std::io::Result<Vec<DeviceCredentialRecord>> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        // Absent file = legitimately no paired devices yet.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    // A present-but-unparseable file is an error, NOT "zero credentials": returning
    // Err keeps the caller from wiping/overwriting it (see with_pairing_store).
    let records: Vec<DeviceCredentialRecord> = serde_json::from_slice(&bytes)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    // Treat the file as untrusted input: drop records whose ids/hashes are malformed.
    Ok(records
        .into_iter()
        .filter(|record| {
            record.device_id.len() == PAIRING_DEVICE_ID_HEX_LEN
                && record
                    .device_id
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
                && record.token_hash.len() == 64
                && record
                    .token_hash
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit())
        })
        .collect())
}

pub(crate) fn persist_pairing_credentials_locked(registry: &PairingRegistry) {
    let Some(path) = registry.store_path.as_ref() else {
        return;
    };
    // Persist only the credentials map (device id + token *hash* + revoked flag),
    // never the plaintext claims, so paired devices and revocations survive the
    // remote-server restarts the desktop performs on every daemon start.
    let snapshot: Vec<&DeviceCredentialRecord> = registry.credentials.values().collect();
    match serde_json::to_vec_pretty(&snapshot) {
        Ok(bytes) => {
            if let Err(error) = write_bytes_atomically(path, &bytes) {
                eprintln!(
                    "openasr-server: could not persist pairing registry to {} (continuing): {error}",
                    path.display()
                );
            } else {
                // Holds token *hashes* + device metadata; keep it owner-only like
                // the desktop client's credential state file.
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let _ = fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
                }
            }
        }
        Err(error) => {
            eprintln!("openasr-server: could not serialize pairing registry (continuing): {error}");
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PairingRequestRecord {
    pub(crate) request_id: String,
    pub(crate) device_name: String,
    pub(crate) created_at_unix_secs: u64,
    pub(crate) safety_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DeviceCredentialRecord {
    pub(crate) device_id: String,
    pub(crate) device_name: String,
    pub(crate) token_hash: String,
    pub(crate) issued_at_unix_secs: u64,
    pub(crate) last_seen_unix_secs: Option<u64>,
    pub(crate) revoked: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct PairingRequestView {
    pub(crate) request_id: String,
    pub(crate) device_name: String,
    pub(crate) created_at_unix_secs: u64,
    pub(crate) safety_code: Option<String>,
    pub(crate) status: &'static str,
}

impl From<&PairingRequestRecord> for PairingRequestView {
    fn from(record: &PairingRequestRecord) -> Self {
        Self {
            request_id: record.request_id.clone(),
            device_name: record.device_name.clone(),
            created_at_unix_secs: record.created_at_unix_secs,
            safety_code: record.safety_code.clone(),
            status: "pending",
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct PairingApprovalView {
    pub(crate) device_id: String,
    pub(crate) device_name: String,
    pub(crate) issued_at_unix_secs: u64,
    pub(crate) status: &'static str,
}

impl PairingApprovalView {
    pub(crate) fn from_record(record: &DeviceCredentialRecord) -> Self {
        Self {
            device_id: record.device_id.clone(),
            device_name: record.device_name.clone(),
            issued_at_unix_secs: record.issued_at_unix_secs,
            status: "approved",
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct PairingDeviceView {
    pub(crate) device_id: String,
    pub(crate) device_name: String,
    pub(crate) issued_at_unix_secs: u64,
    pub(crate) last_seen_unix_secs: Option<u64>,
}

impl From<&DeviceCredentialRecord> for PairingDeviceView {
    fn from(record: &DeviceCredentialRecord) -> Self {
        Self {
            device_id: record.device_id.clone(),
            device_name: record.device_name.clone(),
            issued_at_unix_secs: record.issued_at_unix_secs,
            last_seen_unix_secs: record.last_seen_unix_secs,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PairingCredentialView {
    pub(crate) device_id: String,
    pub(crate) device_name: String,
    pub(crate) issued_at_unix_secs: u64,
    pub(crate) bearer_token: String,
}

pub(crate) enum PairingCredentialState {
    Pending,
    Ready(PairingCredentialView),
}

#[derive(Debug, Serialize)]
pub(crate) struct PairingPendingView {
    pub(crate) status: &'static str,
}

impl PairingCredentialView {
    pub(crate) fn from_record(record: &DeviceCredentialRecord, bearer_token: String) -> Self {
        Self {
            device_id: record.device_id.clone(),
            device_name: record.device_name.clone(),
            issued_at_unix_secs: record.issued_at_unix_secs,
            bearer_token,
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct CreatePairingRequest {
    pub(crate) device_name: String,
}

#[derive(Debug)]
pub(crate) enum PairingError {
    Disabled,
    InvalidDeviceName,
    InvalidRequestId,
    InvalidDeviceId,
    NotFound,
    Random,
    AdminRequired,
}

impl IntoResponse for PairingError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::Disabled => (StatusCode::NOT_FOUND, "Pairing is not enabled."),
            Self::InvalidDeviceName => (StatusCode::BAD_REQUEST, "Device name is required."),
            Self::InvalidRequestId => (
                StatusCode::BAD_REQUEST,
                "Pairing request id must be 32 hex characters.",
            ),
            Self::InvalidDeviceId => (
                StatusCode::BAD_REQUEST,
                "Pairing device id must be 24 hex characters.",
            ),
            Self::NotFound => (StatusCode::NOT_FOUND, "Pairing request was not found."),
            Self::Random => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Could not generate OpenASR pairing credentials.",
            ),
            Self::AdminRequired => (
                StatusCode::UNAUTHORIZED,
                "Missing or invalid OpenASR pairing administrator credentials.",
            ),
        };
        (
            status,
            Json(serde_json::json!({
                "error": {
                    "message": message,
                    "type": "authentication_error"
                }
            })),
        )
            .into_response()
    }
}

pub(crate) async fn require_server_auth(
    State(auth): State<ServerAuth>,
    request: Request,
    next: Next,
) -> Response {
    if !auth.is_enabled()
        || request.uri().path() == "/health"
        || auth.allows_unauthenticated_pair_request(request.method(), request.uri().path())
    {
        return next.run(request).await;
    }
    if auth.authorizes(request.headers()) {
        // Operator-only routes (history / config / model management) require the
        // admin token in pairing mode; a paired remote-compute *device* token is
        // limited to /v1/audio/* (compute) + the informational reads, so it can
        // never read the operator's local transcripts or mutate its config/models.
        if auth.is_pairing_enabled()
            && is_operator_only_path(request.method(), request.uri().path())
            && !auth.authorizes_pairing_admin(request.headers())
        {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({
                    "error": {
                        "message": "Operator credentials are required for this endpoint.",
                        "type": "authorization_error"
                    }
                })),
            )
                .into_response();
        }
        return next.run(request).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        Json(serde_json::json!({
            "error": {
                "message": "Missing or invalid OpenASR server credentials.",
                "type": "authentication_error"
            }
        })),
    )
        .into_response()
}

pub(crate) fn is_operator_only_path(method: &axum::http::Method, path: &str) -> bool {
    use axum::http::Method;
    // The server operator's local data + model mutations. Reads of model/catalog
    // lists and capabilities stay open to paired compute clients.
    if path == "/v1/history" || path.starts_with("/v1/history/") {
        return true; // list / get / delete operator transcripts
    }
    if path == "/v1/config" {
        return true; // get / put operator config
    }
    if path == "/v1/speakers" || path.starts_with("/v1/speakers/") {
        return true; // operator-local voice-match profiles
    }
    if path == "/v1/models/default" {
        return method != Method::GET; // set-default (POST/PUT); GET is informational
    }
    if path.starts_with("/v1/models/pull/") {
        return method == Method::POST; // cancel / pause / resume; GET status stays open
    }
    if path == "/v1/models/local/import" {
        return method == Method::POST; // install a local .oasr into operator storage
    }
    if path.starts_with("/v1/models/") && path.ends_with("/pull") {
        return true; // start a pull (POST /v1/models/{id}/pull)
    }
    if path.starts_with("/v1/models/") && method == Method::DELETE {
        return true; // delete a model pack
    }
    false
}

pub(crate) async fn create_pairing_request(
    Extension(auth): Extension<ServerAuth>,
    Json(request): Json<CreatePairingRequest>,
) -> Result<(StatusCode, Json<PairingRequestView>), PairingError> {
    let request = auth.create_pairing_request(request.device_name)?;
    Ok((StatusCode::ACCEPTED, Json(request)))
}

pub(crate) async fn list_pairing_requests(
    Extension(auth): Extension<ServerAuth>,
    headers: axum::http::HeaderMap,
) -> Result<Json<Vec<PairingRequestView>>, PairingError> {
    require_pairing_admin(&auth, &headers)?;
    Ok(Json(auth.pending_pairing_requests()?))
}

pub(crate) async fn approve_pairing_request(
    Extension(auth): Extension<ServerAuth>,
    headers: axum::http::HeaderMap,
    AxumPath(request_id): AxumPath<String>,
) -> Result<Json<PairingApprovalView>, PairingError> {
    require_pairing_admin(&auth, &headers)?;
    Ok(Json(auth.approve_pairing_request(&request_id)?))
}

pub(crate) async fn reject_pairing_request(
    Extension(auth): Extension<ServerAuth>,
    headers: axum::http::HeaderMap,
    AxumPath(request_id): AxumPath<String>,
) -> Result<StatusCode, PairingError> {
    require_pairing_admin(&auth, &headers)?;
    if auth.reject_pairing_request(&request_id)? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(PairingError::NotFound)
    }
}

pub(crate) async fn list_pairing_credentials(
    Extension(auth): Extension<ServerAuth>,
    headers: axum::http::HeaderMap,
) -> Result<Json<Vec<PairingDeviceView>>, PairingError> {
    require_pairing_admin(&auth, &headers)?;
    Ok(Json(auth.paired_devices()?))
}

pub(crate) async fn get_pairing_credential(
    Extension(auth): Extension<ServerAuth>,
    AxumPath(request_id): AxumPath<String>,
) -> Result<Response, PairingError> {
    match auth.pairing_credential(&request_id)? {
        PairingCredentialState::Pending => Ok((
            StatusCode::ACCEPTED,
            Json(PairingPendingView { status: "pending" }),
        )
            .into_response()),
        PairingCredentialState::Ready(credential) => Ok(Json(credential).into_response()),
    }
}

pub(crate) async fn revoke_pairing_credential(
    Extension(auth): Extension<ServerAuth>,
    headers: axum::http::HeaderMap,
    AxumPath(device_id): AxumPath<String>,
) -> Result<StatusCode, PairingError> {
    require_pairing_admin(&auth, &headers)?;
    if auth.revoke_pairing_credential(&device_id)? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(PairingError::NotFound)
    }
}

pub(crate) fn require_pairing_admin(
    auth: &ServerAuth,
    headers: &axum::http::HeaderMap,
) -> Result<(), PairingError> {
    if auth.authorizes_pairing_admin(headers) {
        Ok(())
    } else {
        Err(PairingError::AdminRequired)
    }
}
