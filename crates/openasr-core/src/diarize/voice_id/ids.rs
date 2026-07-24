//! Stable Voice ID identifiers.
//!
//! Person and sample IDs are opaque, randomly generated, and never derived from
//! display names. Generation uses a UUID version 7 layout (unix-ms timestamp +
//! CSPRNG entropy) so IDs sort roughly by creation time without exposing a
//! counter or process id.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;

pub const PERSON_ID_PREFIX: &str = "person_";
pub const SAMPLE_ID_PREFIX: &str = "sample_";
pub const PROTOTYPE_ID_PREFIX: &str = "proto_";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PersonId(String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SampleId(String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PrototypeId(String);

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum IdError {
    #[error("invalid person id: {0}")]
    InvalidPersonId(String),
    #[error("invalid sample id: {0}")]
    InvalidSampleId(String),
    #[error("invalid prototype id: {0}")]
    InvalidPrototypeId(String),
}

impl PersonId {
    pub fn generate() -> Self {
        Self(format!("{PERSON_ID_PREFIX}{}", uuid_v7_hex()))
    }

    pub fn parse(raw: impl AsRef<str>) -> Result<Self, IdError> {
        let raw = raw.as_ref().trim();
        if is_prefixed_hex_id(raw, PERSON_ID_PREFIX) {
            Ok(Self(raw.to_string()))
        } else {
            Err(IdError::InvalidPersonId(raw.to_string()))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl SampleId {
    pub fn generate() -> Self {
        Self(format!("{SAMPLE_ID_PREFIX}{}", uuid_v7_hex()))
    }

    pub fn parse(raw: impl AsRef<str>) -> Result<Self, IdError> {
        let raw = raw.as_ref().trim();
        if is_prefixed_hex_id(raw, SAMPLE_ID_PREFIX) {
            Ok(Self(raw.to_string()))
        } else {
            Err(IdError::InvalidSampleId(raw.to_string()))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl PrototypeId {
    pub fn generate() -> Self {
        Self(format!("{PROTOTYPE_ID_PREFIX}{}", uuid_v7_hex()))
    }

    pub fn parse(raw: impl AsRef<str>) -> Result<Self, IdError> {
        let raw = raw.as_ref().trim();
        if is_prefixed_hex_id(raw, PROTOTYPE_ID_PREFIX) {
            Ok(Self(raw.to_string()))
        } else {
            Err(IdError::InvalidPrototypeId(raw.to_string()))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PersonId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Display for SampleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Display for PrototypeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl AsRef<str> for PersonId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for SampleId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for PrototypeId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

fn is_prefixed_hex_id(raw: &str, prefix: &str) -> bool {
    raw.starts_with(prefix)
        && raw.len() == prefix.len() + 32
        && raw[prefix.len()..].bytes().all(|b| b.is_ascii_hexdigit())
}

/// UUID v7 without dashes, lowercase hex (32 chars).
fn uuid_v7_hex() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let mut rand_bytes = [0u8; 10];
    // getrandom is the workspace CSPRNG; fall back to a process-unique mix only
    // if the OS entropy source is unavailable (should not happen on supported
    // hosts).
    if getrandom::fill(&mut rand_bytes).is_err() {
        let pid = std::process::id().to_le_bytes();
        let counter = std::time::Instant::now().elapsed().as_nanos().to_le_bytes();
        for (i, slot) in rand_bytes.iter_mut().enumerate() {
            *slot = pid[i % pid.len()] ^ counter[i % counter.len()] ^ (i as u8).wrapping_mul(17);
        }
    }

    let mut bytes = [0u8; 16];
    // 48-bit big-endian unix timestamp in milliseconds.
    bytes[0] = ((millis >> 40) & 0xff) as u8;
    bytes[1] = ((millis >> 32) & 0xff) as u8;
    bytes[2] = ((millis >> 24) & 0xff) as u8;
    bytes[3] = ((millis >> 16) & 0xff) as u8;
    bytes[4] = ((millis >> 8) & 0xff) as u8;
    bytes[5] = (millis & 0xff) as u8;
    // version 7 in the high nibble of byte 6.
    bytes[6] = (0x70) | (rand_bytes[0] & 0x0f);
    bytes[7] = rand_bytes[1];
    // RFC 4122 variant in the high two bits of byte 8.
    bytes[8] = (0x80) | (rand_bytes[2] & 0x3f);
    bytes[9..16].copy_from_slice(&rand_bytes[3..10]);

    let mut out = String::with_capacity(32);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn generated_ids_parse_and_are_unique() {
        let mut seen = HashSet::new();
        for _ in 0..64 {
            let person = PersonId::generate();
            let sample = SampleId::generate();
            let proto = PrototypeId::generate();
            assert!(PersonId::parse(person.as_str()).is_ok());
            assert!(SampleId::parse(sample.as_str()).is_ok());
            assert!(PrototypeId::parse(proto.as_str()).is_ok());
            assert!(seen.insert(person.as_str().to_string()));
            assert!(seen.insert(sample.as_str().to_string()));
            assert!(seen.insert(proto.as_str().to_string()));
        }
    }

    #[test]
    fn rejects_display_names_and_legacy_vp_ids() {
        assert!(PersonId::parse("Alice").is_err());
        assert!(PersonId::parse("vp_aaaaaaaaaaaaaaaa").is_err());
        assert!(SampleId::parse("person_0123456789abcdef0123456789abcdef").is_err());
    }
}
