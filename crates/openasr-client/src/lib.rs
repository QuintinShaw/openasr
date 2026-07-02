//! Client-side trust primitives for connecting to a remote OpenASR compute server.
//!
//! This crate is intentionally open source. It owns the code that decides
//! **which** TLS server certificate a client will trust — trust-on-first-use
//! (TOFU) fingerprint pinning. Keeping this auditable, rather than buried inside
//! an unauditable client, is the whole point of OpenASR's client trust model
//! (the same reason Signal's client is open): a privacy product whose
//! "where does my audio go?" decision is unauditable is worthless.
//!
//! The desktop (or any other client) links this crate and never reimplements the
//! trust decision itself.

use std::sync::{Arc, Mutex};

use openasr_core::certificate_fingerprint_sha256;
use rustls::{
    ClientConfig, DigitallySignedStruct, Error as TlsError, SignatureScheme,
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    pki_types::{CertificateDer, ServerName, UnixTime},
};

/// Normalize a certificate fingerprint to lowercase hex with all non-hex
/// characters stripped, so user-pasted fingerprints (with colons, spaces, or
/// mixed case) compare equal to the canonical form.
pub fn normalize_fingerprint(fingerprint: &str) -> String {
    fingerprint
        .chars()
        .filter(|character| character.is_ascii_hexdigit())
        .flat_map(|character| character.to_lowercase())
        .collect()
}

/// A rustls [`ServerCertVerifier`] that authenticates a server purely by its
/// SHA-256 certificate fingerprint: pinned to an expected value if one is given,
/// otherwise trust-on-first-use (the first observed fingerprint is recorded).
#[derive(Debug, Default)]
pub struct TofuServerVerifier {
    fingerprint: Mutex<Option<String>>,
    expected_fingerprint: Option<String>,
}

impl TofuServerVerifier {
    /// Create a verifier. When `expected_fingerprint` is `Some`, the handshake is
    /// rejected unless the server's certificate fingerprint matches (pinned);
    /// when `None`, the first observed fingerprint is recorded (trust-on-first-use).
    pub fn new(expected_fingerprint: Option<String>) -> Self {
        Self {
            fingerprint: Mutex::new(None),
            expected_fingerprint: expected_fingerprint
                .map(|fingerprint| normalize_fingerprint(&fingerprint))
                .filter(|fingerprint| !fingerprint.is_empty()),
        }
    }

    /// The fingerprint observed during the TLS handshake, if a connection was made.
    pub fn fingerprint(&self) -> Option<String> {
        self.fingerprint
            .lock()
            .expect("TOFU verifier fingerprint mutex poisoned")
            .clone()
    }
}

impl ServerCertVerifier for TofuServerVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        let fingerprint = certificate_fingerprint_sha256(end_entity.as_ref());
        if let Some(expected) = &self.expected_fingerprint
            && &fingerprint != expected
        {
            return Err(TlsError::General(
                "OpenASR remote server TLS fingerprint changed.".to_string(),
            ));
        }
        *self
            .fingerprint
            .lock()
            .expect("TOFU verifier fingerprint mutex poisoned") = Some(fingerprint);
        Ok(ServerCertVerified::assertion())
    }

    // The fingerprint pin (verify_server_cert) selects WHICH cert we trust; these
    // must still prove the peer HOLDS that cert's private key, otherwise an
    // on-path attacker who captured the (cleartext) cert DER could replay it and
    // pass the pin without the key. Delegate to rustls' real signature verifiers.
    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// Build a rustls [`ClientConfig`] that authenticates the server with the given
/// TOFU verifier instead of the platform root store. The server is trusted only
/// by its pinned fingerprint, plus a real signature check proving key possession.
pub fn tls_client_config(verifier: Arc<TofuServerVerifier>) -> Result<Arc<ClientConfig>, String> {
    Ok(Arc::new(
        ClientConfig::builder_with_provider(rustls::crypto::ring::default_provider().into())
            .with_safe_default_protocol_versions()
            .map_err(|error| format!("Could not configure OpenASR TLS client: {error}"))?
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_no_client_auth(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_fingerprint_lowercases_and_strips_non_hex() {
        assert_eq!(normalize_fingerprint("AB:CD ef:01"), "abcdef01");
        assert_eq!(normalize_fingerprint("  "), "");
        assert_eq!(normalize_fingerprint("ZzGg"), ""); // no ASCII hex digits
    }

    #[test]
    fn new_normalizes_expected_and_drops_empty() {
        assert!(
            TofuServerVerifier::new(Some("  ".into()))
                .expected_fingerprint
                .is_none()
        );
        assert!(TofuServerVerifier::new(None).expected_fingerprint.is_none());
        assert_eq!(
            TofuServerVerifier::new(Some("Ab:Cd".into()))
                .expected_fingerprint
                .as_deref(),
            Some("abcd"),
        );
    }

    #[test]
    fn tls_client_config_builds_with_pinned_verifier() {
        let verifier = Arc::new(TofuServerVerifier::new(Some("abcd".into())));
        assert!(tls_client_config(verifier).is_ok());
    }
}
