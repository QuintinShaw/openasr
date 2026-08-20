use std::fmt::{Display, Formatter};

/// Fail-closed error from the remote OpenASR client.
///
/// Trust failures (fingerprint change, safety-code mismatch) are ordinary
/// variants of this type. Callers must not recover by skipping TLS.
#[derive(Debug)]
pub struct ClientError {
    message: String,
}

impl ClientError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for ClientError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ClientError {}

impl From<String> for ClientError {
    fn from(message: String) -> Self {
        Self { message }
    }
}
