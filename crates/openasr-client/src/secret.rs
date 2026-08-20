use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::ClientError;

/// Injected store for pairing bearer tokens.
///
/// Tokens must never be written to a plaintext state file. Desktop/iOS supply
/// a keychain-backed implementation; tests use [`MemorySecretStore`].
pub trait SecretStore: Send + Sync {
    fn store_secret(&self, account: &str, secret: &str) -> Result<(), ClientError>;
    fn load_secret(&self, account: &str) -> Result<Option<String>, ClientError>;
    fn delete_secret(&self, account: &str) -> Result<(), ClientError>;
}

/// Process-memory secret store. Suitable for tests and short-lived sessions.
#[derive(Default)]
pub struct MemorySecretStore {
    inner: Mutex<HashMap<String, String>>,
}

impl MemorySecretStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for MemorySecretStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("MemorySecretStore")
    }
}

impl SecretStore for MemorySecretStore {
    fn store_secret(&self, account: &str, secret: &str) -> Result<(), ClientError> {
        self.inner
            .lock()
            .map_err(|_| ClientError::new("OpenASR secret store mutex poisoned."))?
            .insert(account.to_string(), secret.to_string());
        Ok(())
    }

    fn load_secret(&self, account: &str) -> Result<Option<String>, ClientError> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| ClientError::new("OpenASR secret store mutex poisoned."))?
            .get(account)
            .cloned())
    }

    fn delete_secret(&self, account: &str) -> Result<(), ClientError> {
        self.inner
            .lock()
            .map_err(|_| ClientError::new("OpenASR secret store mutex poisoned."))?
            .remove(account);
        Ok(())
    }
}

impl SecretStore for Arc<dyn SecretStore> {
    fn store_secret(&self, account: &str, secret: &str) -> Result<(), ClientError> {
        (**self).store_secret(account, secret)
    }

    fn load_secret(&self, account: &str) -> Result<Option<String>, ClientError> {
        (**self).load_secret(account)
    }

    fn delete_secret(&self, account: &str) -> Result<(), ClientError> {
        (**self).delete_secret(account)
    }
}

/// Account key for a paired device token: `{fingerprint}:{device_id}`.
pub fn credential_account(server_fingerprint: &str, device_id: &str) -> String {
    format!(
        "{}:{}",
        crate::normalize_fingerprint(server_fingerprint),
        device_id.trim()
    )
}
