use sha2::{Digest, Sha256};

pub fn certificate_fingerprint_sha256(certificate_der: &[u8]) -> String {
    hex_encode(&Sha256::digest(certificate_der))
}

pub fn pairing_safety_code_for_certificate_fingerprint(fingerprint: &str) -> String {
    let normalized = fingerprint
        .chars()
        .filter(|character| character.is_ascii_hexdigit())
        .flat_map(|character| character.to_lowercase())
        .collect::<String>();
    let digest = Sha256::digest(normalized.as_bytes());
    let code = hex_encode(&digest[..4]).to_ascii_uppercase();
    format!("{}-{}", &code[..4], &code[4..])
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certificate_fingerprint_is_lowercase_sha256_hex() {
        assert_eq!(
            certificate_fingerprint_sha256(b"openasr"),
            "5130f93ca3acb986ca033c244da44cd4dc2a36e2b2d58af1a23b8a778d41304d"
        );
    }

    #[test]
    fn pairing_safety_code_normalizes_fingerprint_text() {
        assert_eq!(
            pairing_safety_code_for_certificate_fingerprint("AB:cd ef"),
            pairing_safety_code_for_certificate_fingerprint("abcdef")
        );
        assert_eq!(
            pairing_safety_code_for_certificate_fingerprint("abcdef").len(),
            "ABCD-1234".len()
        );
    }
}
