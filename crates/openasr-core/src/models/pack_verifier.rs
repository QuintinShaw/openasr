//! Unforgeable `.oasr` package verification.
//!
//! A [`VerifiedPack`] proves that one exact open GGUF generation passed the
//! Rust-only package-envelope scan, the sandboxed metadata/tensor-index parse,
//! and its route-specific runtime contract. Downstream install/runtime code
//! receives this proof instead of a path that it could accidentally reopen.

use std::collections::BTreeSet;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlRuntimeSource, GgmlRuntimeSourcePathError, GgufRuntimeSourcePreflight,
    MAX_RUNTIME_GGUF_ARRAY_ELEMENTS, MAX_RUNTIME_GGUF_METADATA_ENTRIES,
    MAX_RUNTIME_GGUF_STRING_BYTES, MAX_RUNTIME_GGUF_TENSORS,
    RuntimeSourceMetadataAndTensorIndexPreflightError,
    load_runtime_source_metadata_and_tensor_index_from_source, validate_ggml_runtime_source_path,
};

use super::aux_pack_registry::AuxPackKind;
use super::oasr_metadata::OASR_PACKAGE_VERSION_V1;
use super::runtime_selection_metadata::selection_metadata_from_gguf;
use crate::arch::OpenAsrArchitectureRegistry;

const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
const MAX_GGUF_DIMS: u32 = 8;

/// Untrusted filesystem entry at the verification ingress.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackCandidate {
    path: PathBuf,
}

impl PackCandidate {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// The real execution route proven by one pack. Auxiliary packs intentionally
/// do not masquerade as ASR family descriptors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PackRoute {
    Asr {
        model_family: &'static str,
        model_architecture: &'static str,
    },
    Aux {
        kind: AuxPackKind,
        model_architecture: String,
    },
}

/// Proof that one exact source generation passed every package/runtime gate.
/// Its fields are private; only [`PackVerifier`] can construct it.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedPack {
    preflight: GgufRuntimeSourcePreflight,
    route: PackRoute,
}

impl VerifiedPack {
    pub fn preflight(&self) -> &GgufRuntimeSourcePreflight {
        &self.preflight
    }

    pub(crate) fn route(&self) -> &PackRoute {
        &self.route
    }

    pub fn content_id(&self) -> &str {
        self.preflight.runtime_source().content_id()
    }

    pub fn model_architecture(&self) -> &str {
        match &self.route {
            PackRoute::Asr {
                model_architecture, ..
            } => model_architecture,
            PackRoute::Aux {
                model_architecture, ..
            } => model_architecture,
        }
    }

    /// Diagnostic path for the exact open generation represented by this
    /// proof. The path is not an execution capability; consumers retain this
    /// `VerifiedPack` and pass the proof across the next seam.
    pub fn path(&self) -> &Path {
        self.preflight.runtime_source().path()
    }

    pub(crate) fn proves_asr_family(&self, model_family: &str, model_architecture: &str) -> bool {
        matches!(
            &self.route,
            PackRoute::Asr {
                model_family: proven_family,
                model_architecture: proven_architecture,
            } if *proven_family == model_family && *proven_architecture == model_architecture
        )
    }

    #[cfg(test)]
    pub(crate) fn from_unverified_preflight_for_test(
        preflight: GgufRuntimeSourcePreflight,
        model_architecture: &'static str,
    ) -> Self {
        let descriptor = OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(model_architecture)
            .expect("test runtime architecture must be registered");
        Self {
            preflight,
            route: PackRoute::Asr {
                model_family: descriptor.identity.model_family,
                model_architecture: descriptor.identity.model_architecture,
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn from_unverified_preflight_and_route_for_test(
        preflight: GgufRuntimeSourcePreflight,
        model_family: &'static str,
        model_architecture: &'static str,
    ) -> Self {
        Self {
            preflight,
            route: PackRoute::Asr {
                model_family,
                model_architecture,
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn from_unverified_aux_preflight_for_test(
        preflight: GgufRuntimeSourcePreflight,
        kind: AuxPackKind,
        model_architecture: impl Into<String>,
    ) -> Self {
        Self {
            preflight,
            route: PackRoute::Aux {
                kind,
                model_architecture: model_architecture.into(),
            },
        }
    }

    /// Canonical catalog family projected from the route proven from these
    /// exact bytes. This is the value download/install targets must bind to;
    /// a model id or filename is not a family proof.
    pub(crate) fn catalog_family_id(&self) -> Option<&'static str> {
        match &self.route {
            PackRoute::Asr {
                model_architecture, ..
            } => OpenAsrArchitectureRegistry::with_builtins()
                .find_by_model_architecture(model_architecture)
                .map(|descriptor| descriptor.identity.catalog_family_id),
            PackRoute::Aux {
                model_architecture, ..
            } => super::aux_pack_registry::auxiliary_catalog_family_id(model_architecture),
        }
    }

    /// Rebinds diagnostics after the verified inode is atomically exposed.
    /// The held descriptor/mmap and parsed views remain the same proof.
    pub(crate) fn with_display_path(mut self, path: PathBuf) -> Self {
        self.preflight = self.preflight.with_display_path(path);
        self
    }
}

/// Content-addressed package plus the exact verification proof produced while
/// its admission lease was pinned. Runtime/install code receives this type,
/// never a bare object path.
#[derive(Debug)]
pub(crate) struct AdmittedPack {
    content: crate::content_store::AdmittedContent<VerifiedPack>,
}

impl AdmittedPack {
    pub(crate) fn from_content(
        content: crate::content_store::AdmittedContent<VerifiedPack>,
    ) -> Result<Self, String> {
        let object_path = content.object_path.clone();
        let content = content.map_proof(|proof| proof.with_display_path(object_path));
        let expected = format!("sha256:{}", content.digest);
        let actual = content.proof().content_id();
        if actual != expected {
            return Err(format!(
                "admitted pack proof identity mismatch: expected {expected}, got {actual}"
            ));
        }
        let proven_size = content.proof().preflight().runtime_source().byte_len();
        if proven_size != content.size_bytes {
            return Err(format!(
                "admitted pack proof size mismatch: expected {}, got {proven_size}",
                content.size_bytes
            ));
        }
        match content.proof().route() {
            PackRoute::Asr {
                model_architecture: "",
                ..
            } => {
                return Err("admitted ASR pack route has an empty architecture".to_string());
            }
            PackRoute::Aux {
                model_architecture, ..
            } if model_architecture.is_empty() => {
                return Err("admitted auxiliary pack route has an empty architecture".to_string());
            }
            PackRoute::Asr { .. } | PackRoute::Aux { .. } => {}
        }
        Ok(Self { content })
    }

    pub(crate) fn digest(&self) -> &str {
        &self.content.digest
    }

    pub(crate) fn size_bytes(&self) -> u64 {
        self.content.size_bytes
    }

    pub(crate) fn object_path(&self) -> &Path {
        &self.content.object_path
    }

    pub(crate) fn catalog_family_id(&self) -> Option<&'static str> {
        self.content.proof().catalog_family_id()
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        crate::content_store::ContentLease,
        VerifiedPack,
        String,
        u64,
        PathBuf,
    ) {
        let digest = self.content.digest.clone();
        let size_bytes = self.content.size_bytes;
        let object_path = self.content.object_path.clone();
        let (lease, verified) = self.content.into_parts();
        (lease, verified, digest, size_bytes, object_path)
    }
}

#[derive(Debug, Error)]
pub enum PackVerificationError {
    #[error("runtime source path '{path}' is invalid: {source}")]
    RuntimeSource {
        path: PathBuf,
        #[source]
        source: GgmlRuntimeSourcePathError,
    },
    #[error("Rust-only package contract failed for '{path}': {reason}")]
    PackageContract { path: PathBuf, reason: String },
    #[error("sandboxed runtime preflight failed for '{path}': {source}")]
    RuntimePreflight {
        path: PathBuf,
        #[source]
        source: RuntimeSourceMetadataAndTensorIndexPreflightError,
    },
    #[error("runtime pack contract failed for '{path}': {reason}")]
    RuntimeContract { path: PathBuf, reason: String },
}

/// The sole constructor for verified package capability values.
#[derive(Debug, Default, Clone, Copy)]
pub struct PackVerifier;

impl PackVerifier {
    pub fn verify_candidate(
        self,
        candidate: PackCandidate,
    ) -> Result<VerifiedPack, PackVerificationError> {
        let path = candidate.path().to_path_buf();
        let runtime_source = validate_ggml_runtime_source_path(candidate.path())
            .map_err(|source| PackVerificationError::RuntimeSource { path, source })?;
        self.verify_runtime_source(runtime_source)
    }

    pub(crate) fn verify_admission_lease(
        self,
        lease: &crate::content_store::ContentLease,
    ) -> Result<VerifiedPack, PackVerificationError> {
        let path = lease.path().to_path_buf();
        let runtime_source = GgmlRuntimeSource::from_admission_lease(lease)
            .map_err(|source| PackVerificationError::RuntimeSource { path, source })?;
        self.verify_runtime_source(runtime_source)
    }

    pub(crate) fn verify_runtime_source(
        self,
        runtime_source: GgmlRuntimeSource,
    ) -> Result<VerifiedPack, PackVerificationError> {
        verify_package_contract_bytes(runtime_source.path(), runtime_source.backing_bytes())?;
        let path = runtime_source.path().to_path_buf();
        let preflight = load_runtime_source_metadata_and_tensor_index_from_source(&runtime_source)
            .map_err(|source| PackVerificationError::RuntimePreflight {
                path: path.clone(),
                source,
            })?;
        let route = self.verify_runtime_contract_preflight(&preflight)?;
        Ok(VerifiedPack { preflight, route })
    }

    pub(crate) fn verify_runtime_contract_preflight(
        self,
        preflight: &GgufRuntimeSourcePreflight,
    ) -> Result<PackRoute, PackVerificationError> {
        validate_route_contract(preflight).map_err(|reason| {
            PackVerificationError::RuntimeContract {
                path: preflight.runtime_source().path().to_path_buf(),
                reason,
            }
        })
    }
}

fn validate_route_contract(preflight: &GgufRuntimeSourcePreflight) -> Result<PackRoute, String> {
    let metadata = preflight.metadata();
    let tensor_index = preflight.tensor_index();
    if let Some((kind, result)) = super::aux_pack_registry::validate_aux_runtime_pack_contract(
        preflight.runtime_source().path(),
        metadata,
        tensor_index,
    ) {
        result.map_err(|error| format!("{}: {error}", kind.validation_failure_label()))?;
        let model_architecture = metadata
            .get_string(crate::arch::GENERAL_ARCHITECTURE_KEY)
            .ok_or_else(|| "auxiliary pack is missing general.architecture".to_string())?
            .to_string();
        return Ok(PackRoute::Aux {
            kind,
            model_architecture,
        });
    }

    let selection_metadata = selection_metadata_from_gguf(metadata);
    let descriptor = OpenAsrArchitectureRegistry::with_builtins()
        .select_ggml_adapter_from_gguf_metadata_v1(&selection_metadata)
        .map_err(|error| format!("runtime adapter selection failed: {error:?}"))?;
    let architecture = crate::arch::OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(descriptor.identity.model_architecture)
        .ok_or_else(|| {
            format!(
                "selected ASR architecture '{}' is absent from the canonical inventory",
                descriptor.identity.model_architecture
            )
        })?;
    (architecture.pack_contract.runtime_validator)(preflight)?;
    Ok(PackRoute::Asr {
        model_family: descriptor.identity.model_family,
        model_architecture: descriptor.identity.model_architecture,
    })
}

fn verify_package_contract_bytes(path: &Path, bytes: &[u8]) -> Result<(), PackVerificationError> {
    let file_len =
        u64::try_from(bytes.len()).map_err(|_| PackVerificationError::PackageContract {
            path: path.to_path_buf(),
            reason: "package length does not fit u64".to_string(),
        })?;
    GgufPackageContractReader::new(Cursor::new(bytes), file_len, path)
        .scan()
        .map_err(|reason| PackVerificationError::PackageContract {
            path: path.to_path_buf(),
            reason,
        })
}

struct GgufPackageContractReader<'a, R> {
    reader: R,
    file_len: u64,
    cursor: u64,
    path: &'a Path,
    alignment: u64,
    package_version: Option<String>,
}

impl<'a, R: Read + Seek> GgufPackageContractReader<'a, R> {
    fn new(reader: R, file_len: u64, path: &'a Path) -> Self {
        Self {
            reader,
            file_len,
            cursor: 0,
            path,
            alignment: GGUF_DEFAULT_ALIGNMENT,
            package_version: None,
        }
    }

    fn scan(&mut self) -> Result<(), String> {
        let mut magic = [0_u8; 4];
        self.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(format!("expected GGUF magic in '{}'", self.path.display()));
        }
        let version = self.read_u32()?;
        if version != 3 {
            return Err(format!("unsupported GGUF version {version}; expected 3"));
        }
        let tensor_count = self.read_u64()?;
        let kv_count = self.read_u64()?;
        if tensor_count == 0 || tensor_count > MAX_RUNTIME_GGUF_TENSORS {
            return Err(format!(
                "tensor count {tensor_count} is outside supported bounds"
            ));
        }
        if kv_count > MAX_RUNTIME_GGUF_METADATA_ENTRIES {
            return Err(format!(
                "metadata entry count {kv_count} is outside supported bounds"
            ));
        }
        let mut metadata_keys = BTreeSet::new();
        for _ in 0..kv_count {
            let key = self.read_metadata_entry()?;
            if !metadata_keys.insert(key.clone()) {
                return Err(format!("duplicate GGUF metadata key '{key}'"));
            }
        }
        let mut tensor_spans = Vec::with_capacity(usize::try_from(tensor_count).unwrap_or(0));
        let mut tensor_names = BTreeSet::new();
        for _ in 0..tensor_count {
            let name = self.read_string()?;
            if !tensor_names.insert(name.clone()) {
                return Err(format!("duplicate GGUF tensor name '{name}'"));
            }
            let n_dims = self.read_u32()?;
            if n_dims == 0 || n_dims > MAX_GGUF_DIMS {
                return Err(format!(
                    "tensor dim count {n_dims} is outside supported bounds"
                ));
            }
            let mut elements = 1_u64;
            for _ in 0..n_dims {
                let dim = self.read_u64()?;
                if dim == 0 {
                    return Err("tensor dimensions must be greater than zero".to_string());
                }
                elements = elements
                    .checked_mul(dim)
                    .ok_or_else(|| "tensor element count overflowed u64".to_string())?;
            }
            let ggml_type = self.read_u32()?;
            let offset = self.read_u64()?;
            let size = ggml_tensor_payload_size(ggml_type, elements)?;
            tensor_spans.push((offset, size));
        }
        let data_start = align_up_u64(self.cursor, self.alignment)?;
        if data_start > self.file_len {
            return Err("GGUF data section starts past end of file".to_string());
        }
        for (offset, size) in tensor_spans {
            let start = data_start
                .checked_add(offset)
                .ok_or_else(|| "tensor absolute offset overflowed u64".to_string())?;
            let end = start
                .checked_add(size)
                .ok_or_else(|| "tensor end offset overflowed u64".to_string())?;
            if end > self.file_len {
                return Err(format!(
                    "tensor payload range [{start}, {end}) exceeds file size {}",
                    self.file_len
                ));
            }
        }
        match self.package_version.as_deref() {
            Some(OASR_PACKAGE_VERSION_V1) => Ok(()),
            Some(value) => Err(format!(
                "unsupported OpenASR package version '{value}'; expected {OASR_PACKAGE_VERSION_V1}"
            )),
            None => Err(format!(
                "missing required metadata '{}'",
                super::oasr_metadata::OASR_METADATA_KEY_PACKAGE_VERSION
            )),
        }
    }

    fn read_metadata_entry(&mut self) -> Result<String, String> {
        let key = self.read_string()?;
        let value_type = self.read_u32()?;
        match value_type {
            0 | 1 | 7 => self.skip(1)?,
            2 | 3 => self.skip(2)?,
            4..=6 => {
                if key == "general.alignment" {
                    let value = self.read_u32()?;
                    self.set_alignment(u64::from(value))?;
                } else {
                    self.skip(4)?;
                }
            }
            8 => {
                let value = self.read_string()?;
                if key == super::oasr_metadata::OASR_METADATA_KEY_PACKAGE_VERSION {
                    self.package_version = Some(value);
                }
            }
            9 => self.skip_array_value()?,
            10..=12 => {
                if key == "general.alignment" && value_type == 10 {
                    let value = self.read_u64()?;
                    self.set_alignment(value)?;
                } else {
                    self.skip(8)?;
                }
            }
            other => return Err(format!("unsupported GGUF metadata value type {other}")),
        }
        Ok(key)
    }

    fn skip_array_value(&mut self) -> Result<(), String> {
        let item_type = self.read_u32()?;
        let item_count = self.read_u64()?;
        if item_count > MAX_RUNTIME_GGUF_ARRAY_ELEMENTS {
            return Err(format!(
                "GGUF array length {item_count} exceeds supported bounds"
            ));
        }
        match item_type {
            0 | 1 | 7 => self.skip(item_count)?,
            2 | 3 => self.skip(item_count.saturating_mul(2))?,
            4..=6 => self.skip(item_count.saturating_mul(4))?,
            8 => {
                for _ in 0..item_count {
                    let _ = self.read_string()?;
                }
            }
            10..=12 => self.skip(item_count.saturating_mul(8))?,
            other => return Err(format!("unsupported GGUF array item type {other}")),
        }
        Ok(())
    }

    fn set_alignment(&mut self, value: u64) -> Result<(), String> {
        if value == 0 || !value.is_power_of_two() || value > 4096 {
            return Err(format!("unsupported GGUF alignment {value}"));
        }
        self.alignment = value;
        Ok(())
    }

    fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_u64()?;
        if len > MAX_RUNTIME_GGUF_STRING_BYTES {
            return Err(format!("GGUF string length {len} exceeds supported bounds"));
        }
        let len_usize = usize::try_from(len).map_err(|_| "string length overflow".to_string())?;
        let mut bytes = vec![0_u8; len_usize];
        self.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(|source| format!("GGUF string is not UTF-8: {source}"))
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let mut bytes = [0_u8; 4];
        self.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        let mut bytes = [0_u8; 8];
        self.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_exact(&mut self, bytes: &mut [u8]) -> Result<(), String> {
        self.reader
            .read_exact(bytes)
            .map_err(|source| format!("unexpected EOF while scanning GGUF: {source}"))?;
        self.cursor = self.cursor.saturating_add(bytes.len() as u64);
        Ok(())
    }

    fn skip(&mut self, bytes: u64) -> Result<(), String> {
        let next = self
            .cursor
            .checked_add(bytes)
            .ok_or_else(|| "GGUF cursor overflowed while skipping".to_string())?;
        if next > self.file_len {
            return Err("GGUF metadata extends past end of file".to_string());
        }
        self.reader
            .seek(SeekFrom::Start(next))
            .map_err(|source| format!("could not seek while scanning GGUF: {source}"))?;
        self.cursor = next;
        Ok(())
    }
}

fn ggml_tensor_payload_size(ggml_type: u32, elements: u64) -> Result<u64, String> {
    match ggml_type {
        0 => elements
            .checked_mul(4)
            .ok_or_else(|| "f32 tensor size overflow".to_string()),
        1 | 30 => elements
            .checked_mul(2)
            .ok_or_else(|| "f16/bf16 tensor size overflow".to_string()),
        2 => block_payload(elements, 32, 18),
        8 => block_payload(elements, 32, 34),
        11 => block_payload(elements, 256, 110),
        12 => block_payload(elements, 256, 144),
        13 => block_payload(elements, 256, 176),
        14 => block_payload(elements, 256, 210),
        24 => Ok(elements),
        25 => elements
            .checked_mul(2)
            .ok_or_else(|| "i16 tensor size overflow".to_string()),
        26 => elements
            .checked_mul(4)
            .ok_or_else(|| "i32 tensor size overflow".to_string()),
        27 | 28 => elements
            .checked_mul(8)
            .ok_or_else(|| "i64/f64 tensor size overflow".to_string()),
        other => Err(format!("unsupported GGML tensor type {other}")),
    }
}

fn block_payload(elements: u64, block_elements: u64, block_bytes: u64) -> Result<u64, String> {
    let blocks = elements
        .checked_add(block_elements - 1)
        .ok_or_else(|| "quantized tensor block count overflow".to_string())?
        / block_elements;
    blocks
        .checked_mul(block_bytes)
        .ok_or_else(|| "quantized tensor size overflow".to_string())
}

fn align_up_u64(value: u64, alignment: u64) -> Result<u64, String> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| "alignment overflowed u64".to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::ggml_runtime::{GgufWriteTensor, GgufWriteTensorType, GgufWriteValue};

    fn scalar_tensor(name: &str) -> GgufWriteTensor {
        GgufWriteTensor {
            name: name.to_string(),
            dims: vec![1],
            tensor_type: GgufWriteTensorType::F32,
            data: 0.0_f32.to_le_bytes().to_vec(),
        }
    }

    fn common_metadata() -> BTreeMap<String, GgufWriteValue> {
        BTreeMap::from([(
            super::super::oasr_metadata::OASR_METADATA_KEY_PACKAGE_VERSION.to_string(),
            GgufWriteValue::String(OASR_PACKAGE_VERSION_V1.to_string()),
        )])
    }

    #[test]
    fn package_gate_rejects_missing_common_version_before_runtime_contract() {
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("missing-version.oasr");
        crate::ggml_runtime::write_gguf_file_v0(
            &path,
            &BTreeMap::new(),
            &[scalar_tensor("fixture.weight")],
        )
        .expect("write fixture");

        let error = PackVerifier
            .verify_candidate(PackCandidate::new(&path))
            .expect_err("missing package version must fail closed");
        assert!(matches!(
            error,
            PackVerificationError::PackageContract { reason, .. }
                if reason.contains("openasr.package.version")
        ));
    }

    #[test]
    fn package_gate_rejects_duplicate_metadata_keys_before_map_projection() {
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("duplicate-metadata.oasr");
        let mut metadata = common_metadata();
        metadata.insert(
            "openasr.test.a".to_string(),
            GgufWriteValue::String("first".to_string()),
        );
        metadata.insert(
            "openasr.test.b".to_string(),
            GgufWriteValue::String("second".to_string()),
        );
        crate::ggml_runtime::write_gguf_file_v0(
            &path,
            &metadata,
            &[scalar_tensor("fixture.weight")],
        )
        .expect("write fixture");

        let mut bytes = std::fs::read(&path).expect("read fixture");
        let needle = b"openasr.test.b";
        let replacement = b"openasr.test.a";
        let matches = bytes
            .windows(needle.len())
            .enumerate()
            .filter_map(|(offset, window)| (window == needle).then_some(offset))
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1, "fixture key must occur exactly once");
        let offset = matches[0];
        bytes[offset..offset + replacement.len()].copy_from_slice(replacement);
        std::fs::write(&path, bytes).expect("rewrite fixture");

        let error = PackVerifier
            .verify_candidate(PackCandidate::new(&path))
            .expect_err("duplicate metadata key must fail closed");
        assert!(matches!(
            error,
            PackVerificationError::PackageContract { reason, .. }
                if reason.contains("duplicate GGUF metadata key 'openasr.test.a'")
        ));
    }

    #[test]
    fn package_gate_rejects_duplicate_tensor_names_before_index_projection() {
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("duplicate-tensor.oasr");
        crate::ggml_runtime::write_gguf_file_v0(
            &path,
            &common_metadata(),
            &[
                scalar_tensor("fixture.alpha"),
                scalar_tensor("fixture.bravo"),
            ],
        )
        .expect("write fixture");

        // The production writer correctly rejects duplicate names, so corrupt
        // a valid serialized pack to exercise the independent reader gate.
        let mut bytes = std::fs::read(&path).expect("read fixture");
        let needle = b"fixture.bravo";
        let replacement = b"fixture.alpha";
        let matches = bytes
            .windows(needle.len())
            .enumerate()
            .filter_map(|(offset, window)| (window == needle).then_some(offset))
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1, "fixture name must occur exactly once");
        let offset = matches[0];
        bytes[offset..offset + replacement.len()].copy_from_slice(replacement);
        std::fs::write(&path, bytes).expect("rewrite fixture");

        let error = PackVerifier
            .verify_candidate(PackCandidate::new(&path))
            .expect_err("duplicate tensor name must fail closed");
        assert!(matches!(
            error,
            PackVerificationError::PackageContract { reason, .. }
                if reason.contains("duplicate GGUF tensor name 'fixture.alpha'")
        ));
    }
}
