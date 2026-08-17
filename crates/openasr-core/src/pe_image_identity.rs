//! Stable executable-image identities for Windows PE backend modules.
//!
//! Authenticode appends a certificate table and updates the PE checksum and
//! security-directory entry. Hashing the complete file therefore produces a
//! different value before and after release signing even though the executable
//! image is unchanged. The identity below follows the Authenticode exclusion
//! rules for exactly those mutable regions. It is used only as a build/runtime
//! contract; the manifest still carries and verifies the SHA-256 of the final
//! file bytes as the trust-bound corruption check. This module does not perform
//! WinVerifyTrust or certificate-chain validation and must not be described as
//! an Authenticode policy.

use sha2::{Digest as _, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PeImageIdentity {
    pub(crate) sha256: String,
    pub(crate) size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackendBundleContractEntry {
    pub(crate) filename: String,
    pub(crate) provider: String,
    pub(crate) image_sha256: String,
    pub(crate) image_size_bytes: u64,
}

pub(crate) fn pe_image_identity(bytes: &[u8]) -> Result<PeImageIdentity, String> {
    if bytes.get(..2) != Some(b"MZ") {
        return Err("file is not a PE image".to_string());
    }
    let pe_offset = read_u32(bytes, 0x3c)? as usize;
    if bytes.get(pe_offset..pe_offset.checked_add(4).ok_or("PE offset overflow")?)
        != Some(b"PE\0\0")
    {
        return Err("file has no PE signature".to_string());
    }
    let coff = pe_offset
        .checked_add(4)
        .ok_or("COFF header offset overflow")?;
    let optional_size =
        read_u16(bytes, coff.checked_add(16).ok_or("COFF header overflow")?)? as usize;
    let optional = coff
        .checked_add(20)
        .ok_or("optional header offset overflow")?;
    let optional_end = optional
        .checked_add(optional_size)
        .ok_or("optional header size overflow")?;
    if optional_end > bytes.len() {
        return Err("PE optional header is truncated".to_string());
    }
    let magic = read_u16(bytes, optional)?;
    let (directory_count_offset, directory_offset) = match magic {
        0x10b => (92_usize, 96_usize),
        0x20b => (108_usize, 112_usize),
        _ => return Err("PE optional header has an unsupported magic".to_string()),
    };
    let checksum = optional
        .checked_add(64)
        .ok_or("PE checksum offset overflow")?;
    let directory_count = read_u32(
        bytes,
        optional
            .checked_add(directory_count_offset)
            .ok_or("PE directory count offset overflow")?,
    )?;
    if directory_count < 5 {
        return Err("PE image has no security directory entry".to_string());
    }
    let security = optional
        .checked_add(directory_offset)
        .and_then(|value| value.checked_add(4 * 8))
        .ok_or("PE security directory offset overflow")?;
    if checksum.checked_add(4).is_none_or(|end| end > optional_end)
        || security.checked_add(8).is_none_or(|end| end > optional_end)
        || checksum >= security
    {
        return Err("PE security metadata is outside the optional header".to_string());
    }

    let certificate_offset = read_u32(bytes, security)? as usize;
    let certificate_size = read_u32(bytes, security + 4)? as usize;
    let certificate_range = if certificate_offset == 0 && certificate_size == 0 {
        bytes.len()..bytes.len()
    } else {
        let end = certificate_offset
            .checked_add(certificate_size)
            .ok_or("PE certificate table size overflow")?;
        if certificate_offset < optional_end || end > bytes.len() {
            return Err("PE certificate table is outside the file".to_string());
        }
        certificate_offset..end
    };

    let mut hasher = Sha256::new();
    hasher.update(&bytes[..checksum]);
    hasher.update([0_u8; 4]);
    hasher.update(&bytes[checksum + 4..security]);
    hasher.update([0_u8; 8]);
    hasher.update(&bytes[security + 8..certificate_range.start]);
    hasher.update(&bytes[certificate_range.end..]);
    Ok(PeImageIdentity {
        sha256: format!("{:x}", hasher.finalize()),
        size_bytes: u64::try_from(bytes.len() - certificate_range.len())
            .map_err(|_| "PE image size does not fit u64".to_string())?,
    })
}

pub(crate) fn backend_bundle_contract_sha256(
    host_abi_fingerprint: &str,
    entries: &[BackendBundleContractEntry],
) -> String {
    let mut entries = entries.to_vec();
    entries.sort_by(|left, right| {
        left.filename
            .to_ascii_lowercase()
            .cmp(&right.filename.to_ascii_lowercase())
            .then_with(|| left.provider.cmp(&right.provider))
    });
    let mut hasher = Sha256::new();
    update_field(&mut hasher, b"openasr-bundle-contract-v1");
    update_field(&mut hasher, host_abi_fingerprint.as_bytes());
    for entry in entries {
        update_field(&mut hasher, entry.filename.to_ascii_lowercase().as_bytes());
        update_field(&mut hasher, entry.provider.to_ascii_lowercase().as_bytes());
        update_field(&mut hasher, entry.image_sha256.as_bytes());
        hasher.update(entry.image_size_bytes.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn update_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, String> {
    let raw: [u8; 2] = bytes
        .get(offset..offset.checked_add(2).ok_or("PE offset overflow")?)
        .ok_or_else(|| "PE structure is truncated".to_string())?
        .try_into()
        .expect("slice length checked");
    Ok(u16::from_le_bytes(raw))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let raw: [u8; 4] = bytes
        .get(offset..offset.checked_add(4).ok_or("PE offset overflow")?)
        .ok_or_else(|| "PE structure is truncated".to_string())?
        .try_into()
        .expect("slice length checked");
    Ok(u32::from_le_bytes(raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_pe(certificate: &[u8]) -> Vec<u8> {
        let optional = 0x98_usize;
        let security = optional + 112 + 4 * 8;
        let certificate_offset = 0x200_usize;
        let mut bytes = vec![0_u8; certificate_offset];
        bytes[..2].copy_from_slice(b"MZ");
        bytes[0x3c..0x40].copy_from_slice(&(0x80_u32).to_le_bytes());
        bytes[0x80..0x84].copy_from_slice(b"PE\0\0");
        bytes[0x94..0x96].copy_from_slice(&(0xf0_u16).to_le_bytes());
        bytes[optional..optional + 2].copy_from_slice(&(0x20b_u16).to_le_bytes());
        bytes[optional + 108..optional + 112].copy_from_slice(&(16_u32).to_le_bytes());
        if !certificate.is_empty() {
            bytes[security..security + 4]
                .copy_from_slice(&(certificate_offset as u32).to_le_bytes());
            bytes[security + 4..security + 8]
                .copy_from_slice(&(certificate.len() as u32).to_le_bytes());
            bytes.extend_from_slice(certificate);
        }
        bytes
    }

    #[test]
    fn identity_ignores_authenticode_mutations_only() {
        let unsigned = minimal_pe(&[]);
        let mut signed = minimal_pe(b"certificate");
        signed[0x98 + 64..0x98 + 68].copy_from_slice(&123_u32.to_le_bytes());
        assert_eq!(
            pe_image_identity(&unsigned).unwrap(),
            pe_image_identity(&signed).unwrap()
        );

        let mut changed = unsigned.clone();
        changed[0x1f0] ^= 1;
        assert_ne!(
            pe_image_identity(&unsigned).unwrap(),
            pe_image_identity(&changed).unwrap()
        );
    }

    #[test]
    fn bundle_contract_is_order_independent_and_payload_bound() {
        let first = BackendBundleContractEntry {
            filename: "ggml.dll".to_string(),
            provider: "host".to_string(),
            image_sha256: "a".repeat(64),
            image_size_bytes: 10,
        };
        let second = BackendBundleContractEntry {
            filename: "ggml-vulkan.dll".to_string(),
            provider: "vulkan".to_string(),
            image_sha256: "b".repeat(64),
            image_size_bytes: 20,
        };
        let forward = backend_bundle_contract_sha256("c", &[first.clone(), second.clone()]);
        let reverse = backend_bundle_contract_sha256("c", &[second.clone(), first.clone()]);
        assert_eq!(forward, reverse);
        let mut changed = second;
        changed.image_size_bytes += 1;
        assert_ne!(
            forward,
            backend_bundle_contract_sha256("c", &[first, changed])
        );
    }
}
