//! `.oadp` adapter packs (OADP Phase 0): LoRA adapters for installed base packs.
//!
//! Format: a standard GGUF-v0 payload (same probe/magic rules as `.oasr`; the
//! reserved `OASR` magic stays rejected) with adapter manifest metadata and
//! per-target A/B tensors. No new container magic is introduced — `.oadp` is a
//! *naming + metadata* contract on top of GGUF, exactly like `.oasr`.
//!
//! Manifest keys (all required):
//!
//! ```text
//! openasr.package.kind            = "adapter-pack"
//! openasr.adapter.version         = 1 (u32)
//! openasr.adapter.id              = free-form adapter identity string
//! openasr.adapter.method          = "lora"
//! openasr.adapter.base.model_id   = exact `openasr.model.id` of the bound base pack
//! openasr.adapter.base.pack_sha256= sha256 (lowercase hex) of the bound base pack FILE
//! openasr.adapter.target_tensors  = string array of base 2-D linear tensor names
//! openasr.adapter.rank            = u32 LoRA rank (shared by all targets)
//! openasr.adapter.alpha           = u32 LoRA alpha (shared by all targets)
//! openasr.adapter.dtype           = "f16" | "f32" (storage dtype of A/B tensors)
//! openasr.adapter.min_openasr_version = "x.y.z"
//! ```
//!
//! Tensor layout per target `<t>` (matching `.oasr` projection orientation,
//! where a rank-2 `.weight` is ggml `[in, out]` so `mul_mat(W, x)` contracts
//! over `ne0 = in`):
//!
//! ```text
//! <t>.lora_a : [input_dim, rank]    => mul_mat(A, x)  -> [rank, tokens]
//! <t>.lora_b : [rank, output_dim]   => mul_mat(B, Ax) -> [out,  tokens]
//! y = W@x + (alpha/rank) * B@(A@x)
//! ```
//!
//! Trust model (Phase 0): adapters are LOCAL and UNSIGNED, but base-bound
//! fail-closed — the manifest's base pack sha256 must match the installed base
//! pack file exactly, and every mismatch class gets its own error.
//!
//! Activation surface (Phase 0): per-request, the execution options carry an
//! optional adapter path (`openasr transcribe --adapter` plumbs it through the
//! native transcription request). The `OPENASR_ADAPTER` environment variable
//! remains the SERVER-side process-level surface and acts as the fallback when
//! the request does not name an adapter. Only the moonshine family supports
//! dynamic adapters; every other family fails closed when an adapter is
//! active.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::ggml_runtime::{
    GGML_TYPE_F16, GGML_TYPE_F32, GgufTensorDataReader, GgufWriteTensor, GgufWriteTensorType,
    GgufWriteValue, read_gguf_metadata_from_runtime_source,
    read_gguf_tensor_index_from_runtime_source, write_gguf_file_v0,
};
use crate::nn::half::f32_to_f16_bits;
use crate::{GgufMetadata, validate_ggml_runtime_source_path};

/// SERVER-side process-level adapter activation for Phase 0. Holds one `.oadp`
/// path. The CLI does NOT set this; `openasr transcribe --adapter` plumbs the
/// path through the request options instead (see [`active_adapter_path`]).
pub const OPENASR_ADAPTER_ENV: &str = "OPENASR_ADAPTER";

/// Resolve the active adapter pack path for an execution: a request-level
/// adapter path wins; otherwise the server-side `OPENASR_ADAPTER` process
/// environment variable is consulted. Returns `None` when no adapter is
/// active.
pub fn active_adapter_path(request_adapter_path: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = request_adapter_path {
        return Some(path.to_path_buf());
    }
    std::env::var_os(OPENASR_ADAPTER_ENV).map(PathBuf::from)
}

/// Conventional extension for adapter packs (advisory; the reader is
/// magic-driven like `.oasr` and does not gate on the extension).
pub const OPENASR_ADAPTER_PACK_EXTENSION: &str = "oadp";

pub const OADP_KEY_PACKAGE_KIND: &str = "openasr.package.kind";
pub const OADP_PACKAGE_KIND_ADAPTER_PACK: &str = "adapter-pack";
pub const OADP_KEY_ADAPTER_VERSION: &str = "openasr.adapter.version";
pub const OADP_ADAPTER_VERSION_V1: u32 = 1;
pub const OADP_KEY_ADAPTER_ID: &str = "openasr.adapter.id";
pub const OADP_KEY_ADAPTER_METHOD: &str = "openasr.adapter.method";
pub const OADP_ADAPTER_METHOD_LORA: &str = "lora";
pub const OADP_KEY_BASE_MODEL_ID: &str = "openasr.adapter.base.model_id";
pub const OADP_KEY_BASE_PACK_SHA256: &str = "openasr.adapter.base.pack_sha256";
pub const OADP_KEY_TARGET_TENSORS: &str = "openasr.adapter.target_tensors";
pub const OADP_KEY_RANK: &str = "openasr.adapter.rank";
pub const OADP_KEY_ALPHA: &str = "openasr.adapter.alpha";
pub const OADP_KEY_DTYPE: &str = "openasr.adapter.dtype";
pub const OADP_KEY_MIN_OPENASR_VERSION: &str = "openasr.adapter.min_openasr_version";

const OADP_TENSOR_SUFFIX_A: &str = ".lora_a";
const OADP_TENSOR_SUFFIX_B: &str = ".lora_b";

/// `openasr.model.id` — the identity key burned into every base `.oasr` pack.
const BASE_PACK_MODEL_ID_KEY: &str = "openasr.model.id";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraAdapterDtype {
    F16,
    F32,
}

impl LoraAdapterDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "f16" => Some(Self::F16),
            "f32" => Some(Self::F32),
            _ => None,
        }
    }

    fn ggml_type(self) -> i32 {
        match self {
            Self::F16 => GGML_TYPE_F16,
            Self::F32 => GGML_TYPE_F32,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoraAdapterManifest {
    pub id: String,
    pub version: u32,
    pub base_model_id: String,
    pub base_pack_sha256: String,
    pub target_tensors: Vec<String>,
    pub rank: u32,
    pub alpha: u32,
    pub dtype: LoraAdapterDtype,
    pub min_openasr_version: String,
}

/// One LoRA target with its A/B factors dequantized to f32 host values
/// (ne0-major, matching GGUF row layout).
#[derive(Debug, Clone, PartialEq)]
pub struct LoraAdapterTarget {
    pub base_tensor: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub rank: usize,
    /// `[input_dim, rank]`, ne0-major.
    pub a_values: Vec<f32>,
    /// `[rank, output_dim]`, ne0-major. NOT pre-scaled; `alpha/rank` scaling is
    /// applied by the consuming runtime.
    pub b_values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoraAdapterPack {
    pub source_path: PathBuf,
    /// sha256 (lowercase hex) of the `.oadp` file itself: part of the runtime
    /// cache identity so prepared graphs are never reused across adapters.
    pub file_sha256: String,
    pub manifest: LoraAdapterManifest,
    pub targets: Vec<LoraAdapterTarget>,
}

impl LoraAdapterPack {
    /// Stable identity string for runtime/cgraph cache keys: id, content
    /// hash, rank, and target set. Prepared-graph reuse keyed only on the
    /// base pack would be a correctness bug; cache keys must include this.
    pub fn fingerprint(&self) -> String {
        format!(
            "lora:{}:{}:r{}:a{}:{}",
            self.manifest.id,
            self.file_sha256,
            self.manifest.rank,
            self.manifest.alpha,
            self.manifest.target_tensors.join(",")
        )
    }

    pub fn target(&self, base_tensor: &str) -> Option<&LoraAdapterTarget> {
        self.targets
            .iter()
            .find(|target| target.base_tensor == base_tensor)
    }
}

#[derive(Debug, Error)]
pub enum AdapterPackError {
    #[error("adapter pack '{path}' could not be read: {reason}")]
    Unreadable { path: PathBuf, reason: String },
    #[error(
        "'{path}' is not an adapter pack: metadata key '{key}' must be \
         '{expected}', got {found:?}"
    )]
    NotAnAdapterPack {
        path: PathBuf,
        key: &'static str,
        expected: &'static str,
        found: Option<String>,
    },
    #[error("adapter pack '{path}' has unsupported adapter version {found} (supported: 1)")]
    UnsupportedVersion { path: PathBuf, found: u32 },
    #[error("adapter pack '{path}' metadata key '{key}' is missing or has the wrong type")]
    MissingMetadata { path: PathBuf, key: &'static str },
    #[error("adapter pack '{path}' metadata key '{key}' is invalid: {reason}")]
    InvalidMetadata {
        path: PathBuf,
        key: &'static str,
        reason: String,
    },
    #[error("adapter pack '{path}' has unsupported dtype '{found}' (supported: f16, f32)")]
    UnsupportedDtype { path: PathBuf, found: String },
    #[error("adapter pack '{path}' declares no target tensors")]
    NoTargets { path: PathBuf },
    #[error("adapter pack '{path}' declares duplicate target tensor '{name}'")]
    DuplicateTarget { path: PathBuf, name: String },
    #[error("adapter pack '{path}' is missing adapter tensor '{name}'")]
    AdapterTensorMissing { path: PathBuf, name: String },
    #[error("adapter tensor '{name}' in '{path}' has invalid shape {dims:?}: {reason}")]
    AdapterTensorShapeInvalid {
        path: PathBuf,
        name: String,
        dims: Vec<u64>,
        reason: String,
    },
    #[error(
        "adapter tensor '{name}' in '{path}' is stored as ggml type {found_type_name} but the \
         manifest declares dtype '{declared}'"
    )]
    AdapterTensorDtypeMismatch {
        path: PathBuf,
        name: String,
        declared: &'static str,
        found_type_name: String,
    },
    #[error("base pack '{base_pack}' for adapter validation could not be read: {reason}")]
    BasePackUnreadable { base_pack: PathBuf, reason: String },
    #[error(
        "adapter base model id mismatch (fail-closed): adapter is bound to base model id \
         '{adapter_base_model_id}' but installed base pack '{base_pack}' has model id \
         {installed_model_id:?}"
    )]
    BaseModelIdMismatch {
        adapter_base_model_id: String,
        installed_model_id: Option<String>,
        base_pack: PathBuf,
    },
    #[error(
        "adapter base pack sha256 mismatch (fail-closed): adapter is bound to base pack sha256 \
         '{adapter_base_sha256}' but installed base pack '{base_pack}' has sha256 \
         '{installed_sha256}'"
    )]
    BasePackSha256Mismatch {
        adapter_base_sha256: String,
        installed_sha256: String,
        base_pack: PathBuf,
    },
    #[error(
        "adapter requires OpenASR version >= {required} but this build is {current} (fail-closed)"
    )]
    MinVersionUnsatisfied { required: String, current: String },
    #[error("adapter pack write failed for '{path}': {reason}")]
    WriteFailed { path: PathBuf, reason: String },
}

/// Read and structurally validate a `.oadp` adapter pack. This validates the
/// pack in isolation; base binding is checked separately by
/// [`validate_lora_adapter_base_binding`].
pub fn read_lora_adapter_pack(path: impl AsRef<Path>) -> Result<LoraAdapterPack, AdapterPackError> {
    let path = path.as_ref();
    let runtime_source =
        validate_ggml_runtime_source_path(path).map_err(|error| AdapterPackError::Unreadable {
            path: path.to_path_buf(),
            reason: error.to_string(),
        })?;
    let metadata = read_gguf_metadata_from_runtime_source(&runtime_source).map_err(|error| {
        AdapterPackError::Unreadable {
            path: path.to_path_buf(),
            reason: error.to_string(),
        }
    })?;

    let manifest = parse_manifest(path, &metadata)?;

    let tensor_index =
        read_gguf_tensor_index_from_runtime_source(&runtime_source).map_err(|error| {
            AdapterPackError::Unreadable {
                path: path.to_path_buf(),
                reason: error.to_string(),
            }
        })?;
    let reader = GgufTensorDataReader::from_tensor_index(tensor_index).map_err(|error| {
        AdapterPackError::Unreadable {
            path: path.to_path_buf(),
            reason: error.to_string(),
        }
    })?;

    let rank = manifest.rank as usize;
    let mut targets = Vec::with_capacity(manifest.target_tensors.len());
    for base_tensor in &manifest.target_tensors {
        let a_name = format!("{base_tensor}{OADP_TENSOR_SUFFIX_A}");
        let b_name = format!("{base_tensor}{OADP_TENSOR_SUFFIX_B}");
        let a = read_adapter_matrix(path, &reader, &a_name, manifest.dtype)?;
        let b = read_adapter_matrix(path, &reader, &b_name, manifest.dtype)?;
        if a.dims[1] != rank {
            return Err(AdapterPackError::AdapterTensorShapeInvalid {
                path: path.to_path_buf(),
                name: a_name,
                dims: a.dims.iter().map(|&dim| dim as u64).collect(),
                reason: format!("ne1 must equal manifest rank {rank} (A is [input_dim, rank])"),
            });
        }
        if b.dims[0] != rank {
            return Err(AdapterPackError::AdapterTensorShapeInvalid {
                path: path.to_path_buf(),
                name: b_name,
                dims: b.dims.iter().map(|&dim| dim as u64).collect(),
                reason: format!("ne0 must equal manifest rank {rank} (B is [rank, output_dim])"),
            });
        }
        targets.push(LoraAdapterTarget {
            base_tensor: base_tensor.clone(),
            input_dim: a.dims[0],
            output_dim: b.dims[1],
            rank,
            a_values: a.values,
            b_values: b.values,
        });
    }

    let file_sha256 = file_sha256_hex(path).map_err(|reason| AdapterPackError::Unreadable {
        path: path.to_path_buf(),
        reason,
    })?;

    Ok(LoraAdapterPack {
        source_path: path.to_path_buf(),
        file_sha256,
        manifest,
        targets,
    })
}

/// Fail-closed base binding: the installed base pack must match the adapter's
/// declared base EXACTLY (model id + file sha256) and this build must satisfy
/// the adapter's minimum OpenASR version. Each mismatch class is a distinct
/// error.
pub fn validate_lora_adapter_base_binding(
    pack: &LoraAdapterPack,
    base_pack_path: impl AsRef<Path>,
) -> Result<(), AdapterPackError> {
    let base_pack_path = base_pack_path.as_ref();

    check_min_openasr_version(&pack.manifest.min_openasr_version)?;

    let runtime_source = validate_ggml_runtime_source_path(base_pack_path).map_err(|error| {
        AdapterPackError::BasePackUnreadable {
            base_pack: base_pack_path.to_path_buf(),
            reason: error.to_string(),
        }
    })?;
    let base_metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).map_err(|error| {
            AdapterPackError::BasePackUnreadable {
                base_pack: base_pack_path.to_path_buf(),
                reason: error.to_string(),
            }
        })?;
    let installed_model_id = base_metadata.get_string(BASE_PACK_MODEL_ID_KEY);
    if installed_model_id != Some(pack.manifest.base_model_id.as_str()) {
        return Err(AdapterPackError::BaseModelIdMismatch {
            adapter_base_model_id: pack.manifest.base_model_id.clone(),
            installed_model_id: installed_model_id.map(str::to_string),
            base_pack: base_pack_path.to_path_buf(),
        });
    }

    let installed_sha256 =
        file_sha256_hex(base_pack_path).map_err(|reason| AdapterPackError::BasePackUnreadable {
            base_pack: base_pack_path.to_path_buf(),
            reason,
        })?;
    if installed_sha256 != pack.manifest.base_pack_sha256 {
        return Err(AdapterPackError::BasePackSha256Mismatch {
            adapter_base_sha256: pack.manifest.base_pack_sha256.clone(),
            installed_sha256,
            base_pack: base_pack_path.to_path_buf(),
        });
    }
    Ok(())
}

/// One write-side LoRA target. Dims are explicit so hand-made/test adapters
/// can also describe targets that do NOT exist in any base pack (for
/// fail-closed coverage).
#[derive(Debug, Clone, PartialEq)]
pub struct LoraAdapterWriteTarget {
    pub base_tensor: String,
    pub input_dim: usize,
    pub output_dim: usize,
    /// `[input_dim, rank]`, ne0-major.
    pub a_values: Vec<f32>,
    /// `[rank, output_dim]`, ne0-major.
    pub b_values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoraAdapterWriteRequest {
    pub output_path: PathBuf,
    pub id: String,
    pub base_model_id: String,
    pub base_pack_sha256: String,
    pub rank: u32,
    pub alpha: u32,
    pub dtype: LoraAdapterDtype,
    pub min_openasr_version: String,
    pub targets: Vec<LoraAdapterWriteTarget>,
}

/// Write a `.oadp` adapter pack (GGUF-v0 payload). Used by the hand-made
/// adapter tooling and the real-pack test suite.
pub fn write_lora_adapter_pack(request: &LoraAdapterWriteRequest) -> Result<(), AdapterPackError> {
    let path = request.output_path.as_path();
    let write_failed = |reason: String| AdapterPackError::WriteFailed {
        path: path.to_path_buf(),
        reason,
    };
    if request.targets.is_empty() {
        return Err(write_failed("at least one target is required".to_string()));
    }
    let rank = request.rank as usize;
    if rank == 0 {
        return Err(write_failed("rank must be > 0".to_string()));
    }

    let mut metadata = BTreeMap::new();
    metadata.insert(
        OADP_KEY_PACKAGE_KIND.to_string(),
        GgufWriteValue::String(OADP_PACKAGE_KIND_ADAPTER_PACK.to_string()),
    );
    metadata.insert(
        OADP_KEY_ADAPTER_VERSION.to_string(),
        GgufWriteValue::U32(OADP_ADAPTER_VERSION_V1),
    );
    metadata.insert(
        OADP_KEY_ADAPTER_ID.to_string(),
        GgufWriteValue::String(request.id.clone()),
    );
    metadata.insert(
        OADP_KEY_ADAPTER_METHOD.to_string(),
        GgufWriteValue::String(OADP_ADAPTER_METHOD_LORA.to_string()),
    );
    metadata.insert(
        OADP_KEY_BASE_MODEL_ID.to_string(),
        GgufWriteValue::String(request.base_model_id.clone()),
    );
    metadata.insert(
        OADP_KEY_BASE_PACK_SHA256.to_string(),
        GgufWriteValue::String(request.base_pack_sha256.clone()),
    );
    metadata.insert(
        OADP_KEY_TARGET_TENSORS.to_string(),
        GgufWriteValue::StringArray(
            request
                .targets
                .iter()
                .map(|target| target.base_tensor.clone())
                .collect(),
        ),
    );
    metadata.insert(OADP_KEY_RANK.to_string(), GgufWriteValue::U32(request.rank));
    metadata.insert(
        OADP_KEY_ALPHA.to_string(),
        GgufWriteValue::U32(request.alpha),
    );
    metadata.insert(
        OADP_KEY_DTYPE.to_string(),
        GgufWriteValue::String(request.dtype.as_str().to_string()),
    );
    metadata.insert(
        OADP_KEY_MIN_OPENASR_VERSION.to_string(),
        GgufWriteValue::String(request.min_openasr_version.clone()),
    );

    let mut tensors = Vec::with_capacity(request.targets.len() * 2);
    for target in &request.targets {
        let a_elements = target.input_dim.checked_mul(rank);
        if a_elements != Some(target.a_values.len()) {
            return Err(write_failed(format!(
                "target '{}' A has {} values but dims [{}, {rank}] require {:?}",
                target.base_tensor,
                target.a_values.len(),
                target.input_dim,
                a_elements
            )));
        }
        let b_elements = rank.checked_mul(target.output_dim);
        if b_elements != Some(target.b_values.len()) {
            return Err(write_failed(format!(
                "target '{}' B has {} values but dims [{rank}, {}] require {:?}",
                target.base_tensor,
                target.b_values.len(),
                target.output_dim,
                b_elements
            )));
        }
        tensors.push(adapter_write_tensor(
            format!("{}{OADP_TENSOR_SUFFIX_A}", target.base_tensor),
            &[target.input_dim as u64, rank as u64],
            &target.a_values,
            request.dtype,
        ));
        tensors.push(adapter_write_tensor(
            format!("{}{OADP_TENSOR_SUFFIX_B}", target.base_tensor),
            &[rank as u64, target.output_dim as u64],
            &target.b_values,
            request.dtype,
        ));
    }

    write_gguf_file_v0(path, &metadata, &tensors).map_err(|error| write_failed(error.to_string()))
}

/// The moonshine LoRA target contract: the 2-D linears that the dynamic
/// side-path can serve. `{enc,dec}.blk.<n>.{attn_q,attn_k,attn_v,attn_o,
/// ffn_up,ffn_down}.weight` plus the decoder's `cross_{q,k,v,o}.weight`.
/// Embedding and tied logits are NOT adapter targets in Phase 0.
pub fn is_moonshine_lora_target_tensor_name(name: &str) -> bool {
    let parts: Vec<&str> = name.split('.').collect();
    let [side, blk, layer_index, slot, weight] = parts.as_slice() else {
        return false;
    };
    if *blk != "blk" || *weight != "weight" || layer_index.parse::<usize>().is_err() {
        return false;
    }
    match *side {
        "enc" => matches!(
            *slot,
            "attn_q" | "attn_k" | "attn_v" | "attn_o" | "ffn_up" | "ffn_down"
        ),
        "dec" => matches!(
            *slot,
            "attn_q"
                | "attn_k"
                | "attn_v"
                | "attn_o"
                | "cross_q"
                | "cross_k"
                | "cross_v"
                | "cross_o"
                | "ffn_up"
                | "ffn_down"
        ),
        _ => false,
    }
}

/// The Qwen3-ASR LLM LoRA target contract: the 2-D linears in the LLM decoder
/// layers — `blk.<n>.{attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down}
/// .weight`. The 1-D norms (`attn_norm`, `attn_q_norm`, `attn_k_norm`,
/// `ffn_norm`), the token embedding, the tied logits head (`output.weight`), and
/// the audio encoder (`audio.*`) are NOT adapter targets.
pub fn is_qwen3_asr_lora_target_tensor_name(name: &str) -> bool {
    let parts: Vec<&str> = name.split('.').collect();
    // LLM decoder layers are `blk.<n>.<slot>.weight` (the audio encoder uses the
    // distinct `audio.blk.<n>.…` prefix, which has 5 parts and is rejected here).
    let [blk, layer_index, slot, weight] = parts.as_slice() else {
        return false;
    };
    if *blk != "blk" || *weight != "weight" || layer_index.parse::<usize>().is_err() {
        return false;
    }
    matches!(
        *slot,
        "attn_q" | "attn_k" | "attn_v" | "attn_output" | "ffn_gate" | "ffn_up" | "ffn_down"
    )
}

/// All adapter-targetable 2-D linears of a base pack matching `is_target`, with
/// their `[input_dim, output_dim]`. Shared by the per-family helpers below.
fn lora_targetable_tensors_with(
    base_pack_path: &Path,
    is_target: impl Fn(&str) -> bool,
) -> Result<Vec<(String, [usize; 2])>, AdapterPackError> {
    let unreadable = |reason: String| AdapterPackError::BasePackUnreadable {
        base_pack: base_pack_path.to_path_buf(),
        reason,
    };
    let runtime_source = validate_ggml_runtime_source_path(base_pack_path)
        .map_err(|error| unreadable(error.to_string()))?;
    let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
        .map_err(|error| unreadable(error.to_string()))?;
    let mut targets = Vec::new();
    for tensor in tensor_index.tensors() {
        if !is_target(&tensor.name) {
            continue;
        }
        let [ne0, ne1] = tensor.dims.as_slice() else {
            continue;
        };
        targets.push((tensor.name.clone(), [*ne0 as usize, *ne1 as usize]));
    }
    targets.sort();
    Ok(targets)
}

/// All adapter-targetable tensor names of a moonshine base pack, with their
/// `[input_dim, output_dim]`. Used by the hand-made adapter tooling/tests.
pub fn moonshine_lora_targetable_tensors(
    base_pack_path: impl AsRef<Path>,
) -> Result<Vec<(String, [usize; 2])>, AdapterPackError> {
    lora_targetable_tensors_with(
        base_pack_path.as_ref(),
        is_moonshine_lora_target_tensor_name,
    )
}

/// All Qwen3-ASR LLM-decoder LoRA-targetable tensors of a base pack, with dims.
pub fn qwen3_asr_lora_targetable_tensors(
    base_pack_path: impl AsRef<Path>,
) -> Result<Vec<(String, [usize; 2])>, AdapterPackError> {
    lora_targetable_tensors_with(
        base_pack_path.as_ref(),
        is_qwen3_asr_lora_target_tensor_name,
    )
}

/// Read `openasr.model.id` from a base pack (tooling helper).
pub fn base_pack_model_id(base_pack_path: impl AsRef<Path>) -> Result<String, AdapterPackError> {
    let base_pack_path = base_pack_path.as_ref();
    let unreadable = |reason: String| AdapterPackError::BasePackUnreadable {
        base_pack: base_pack_path.to_path_buf(),
        reason,
    };
    let runtime_source = validate_ggml_runtime_source_path(base_pack_path)
        .map_err(|error| unreadable(error.to_string()))?;
    let metadata = read_gguf_metadata_from_runtime_source(&runtime_source)
        .map_err(|error| unreadable(error.to_string()))?;
    metadata
        .get_string(BASE_PACK_MODEL_ID_KEY)
        .map(str::to_string)
        .ok_or_else(|| {
            unreadable(format!(
                "base pack has no '{BASE_PACK_MODEL_ID_KEY}' metadata"
            ))
        })
}

/// sha256 of a file as lowercase hex (streaming; used for base binding and
/// adapter cache identity).
pub fn file_sha256_hex(path: impl AsRef<Path>) -> Result<String, String> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn parse_manifest(
    path: &Path,
    metadata: &GgufMetadata,
) -> Result<LoraAdapterManifest, AdapterPackError> {
    let kind = metadata.get_string(OADP_KEY_PACKAGE_KIND);
    if kind != Some(OADP_PACKAGE_KIND_ADAPTER_PACK) {
        return Err(AdapterPackError::NotAnAdapterPack {
            path: path.to_path_buf(),
            key: OADP_KEY_PACKAGE_KIND,
            expected: OADP_PACKAGE_KIND_ADAPTER_PACK,
            found: kind.map(str::to_string),
        });
    }
    let version = require_u32(path, metadata, OADP_KEY_ADAPTER_VERSION)?;
    if version != OADP_ADAPTER_VERSION_V1 {
        return Err(AdapterPackError::UnsupportedVersion {
            path: path.to_path_buf(),
            found: version,
        });
    }
    let method = require_string(path, metadata, OADP_KEY_ADAPTER_METHOD)?;
    if method != OADP_ADAPTER_METHOD_LORA {
        return Err(AdapterPackError::InvalidMetadata {
            path: path.to_path_buf(),
            key: OADP_KEY_ADAPTER_METHOD,
            reason: format!("unsupported adapter method '{method}' (supported: lora)"),
        });
    }
    let id = require_string(path, metadata, OADP_KEY_ADAPTER_ID)?;
    if id.trim().is_empty() {
        return Err(AdapterPackError::InvalidMetadata {
            path: path.to_path_buf(),
            key: OADP_KEY_ADAPTER_ID,
            reason: "adapter id must be non-empty".to_string(),
        });
    }
    let base_model_id = require_string(path, metadata, OADP_KEY_BASE_MODEL_ID)?;
    if base_model_id.trim().is_empty() {
        return Err(AdapterPackError::InvalidMetadata {
            path: path.to_path_buf(),
            key: OADP_KEY_BASE_MODEL_ID,
            reason: "base model id must be non-empty".to_string(),
        });
    }
    let base_pack_sha256 = require_string(path, metadata, OADP_KEY_BASE_PACK_SHA256)?;
    crate::safety::validate_sha256(OADP_KEY_BASE_PACK_SHA256, base_pack_sha256).map_err(
        |reason| AdapterPackError::InvalidMetadata {
            path: path.to_path_buf(),
            key: OADP_KEY_BASE_PACK_SHA256,
            reason,
        },
    )?;
    let rank = require_u32(path, metadata, OADP_KEY_RANK)?;
    if rank == 0 {
        return Err(AdapterPackError::InvalidMetadata {
            path: path.to_path_buf(),
            key: OADP_KEY_RANK,
            reason: "rank must be > 0".to_string(),
        });
    }
    let alpha = require_u32(path, metadata, OADP_KEY_ALPHA)?;
    if alpha == 0 {
        return Err(AdapterPackError::InvalidMetadata {
            path: path.to_path_buf(),
            key: OADP_KEY_ALPHA,
            reason: "alpha must be > 0".to_string(),
        });
    }
    let dtype_raw = require_string(path, metadata, OADP_KEY_DTYPE)?;
    let dtype =
        LoraAdapterDtype::parse(dtype_raw).ok_or_else(|| AdapterPackError::UnsupportedDtype {
            path: path.to_path_buf(),
            found: dtype_raw.to_string(),
        })?;
    let min_openasr_version = require_string(path, metadata, OADP_KEY_MIN_OPENASR_VERSION)?;
    parse_semver_triple(min_openasr_version).map_err(|reason| {
        AdapterPackError::InvalidMetadata {
            path: path.to_path_buf(),
            key: OADP_KEY_MIN_OPENASR_VERSION,
            reason,
        }
    })?;

    let target_tensors = metadata
        .get_string_array(OADP_KEY_TARGET_TENSORS)
        .ok_or_else(|| AdapterPackError::MissingMetadata {
            path: path.to_path_buf(),
            key: OADP_KEY_TARGET_TENSORS,
        })?
        .to_vec();
    if target_tensors.is_empty() {
        return Err(AdapterPackError::NoTargets {
            path: path.to_path_buf(),
        });
    }
    let mut seen = std::collections::BTreeSet::new();
    for name in &target_tensors {
        if !seen.insert(name.as_str()) {
            return Err(AdapterPackError::DuplicateTarget {
                path: path.to_path_buf(),
                name: name.clone(),
            });
        }
    }

    Ok(LoraAdapterManifest {
        id: id.to_string(),
        version,
        base_model_id: base_model_id.to_string(),
        base_pack_sha256: base_pack_sha256.to_string(),
        target_tensors,
        rank,
        alpha,
        dtype,
        min_openasr_version: min_openasr_version.to_string(),
    })
}

struct AdapterMatrix {
    dims: [usize; 2],
    values: Vec<f32>,
}

fn read_adapter_matrix(
    path: &Path,
    reader: &GgufTensorDataReader,
    name: &str,
    dtype: LoraAdapterDtype,
) -> Result<AdapterMatrix, AdapterPackError> {
    let tensor =
        reader
            .tensor_index()
            .get(name)
            .ok_or_else(|| AdapterPackError::AdapterTensorMissing {
                path: path.to_path_buf(),
                name: name.to_string(),
            })?;
    let [ne0, ne1] = tensor.dims.as_slice() else {
        return Err(AdapterPackError::AdapterTensorShapeInvalid {
            path: path.to_path_buf(),
            name: name.to_string(),
            dims: tensor.dims.clone(),
            reason: "adapter tensors must be rank-2".to_string(),
        });
    };
    if *ne0 == 0 || *ne1 == 0 {
        return Err(AdapterPackError::AdapterTensorShapeInvalid {
            path: path.to_path_buf(),
            name: name.to_string(),
            dims: tensor.dims.clone(),
            reason: "adapter tensor dims must be > 0".to_string(),
        });
    }
    if tensor.ggml_type != dtype.ggml_type() {
        return Err(AdapterPackError::AdapterTensorDtypeMismatch {
            path: path.to_path_buf(),
            name: name.to_string(),
            declared: dtype.as_str(),
            found_type_name: tensor.type_name.clone(),
        });
    }
    let dims_u64 = [*ne0, *ne1];
    let values = reader
        .host_tensor_f32_copy_dequantized_by_name(name, &dims_u64)
        .map_err(|error| AdapterPackError::Unreadable {
            path: path.to_path_buf(),
            reason: error.to_string(),
        })?;
    Ok(AdapterMatrix {
        dims: [*ne0 as usize, *ne1 as usize],
        values,
    })
}

fn require_string<'a>(
    path: &Path,
    metadata: &'a GgufMetadata,
    key: &'static str,
) -> Result<&'a str, AdapterPackError> {
    metadata
        .get_string(key)
        .ok_or_else(|| AdapterPackError::MissingMetadata {
            path: path.to_path_buf(),
            key,
        })
}

fn require_u32(
    path: &Path,
    metadata: &GgufMetadata,
    key: &'static str,
) -> Result<u32, AdapterPackError> {
    metadata
        .get_u32(key)
        .ok_or_else(|| AdapterPackError::MissingMetadata {
            path: path.to_path_buf(),
            key,
        })
}

fn check_min_openasr_version(required: &str) -> Result<(), AdapterPackError> {
    let current = env!("CARGO_PKG_VERSION");
    let required_triple =
        parse_semver_triple(required).map_err(|_| AdapterPackError::MinVersionUnsatisfied {
            required: required.to_string(),
            current: current.to_string(),
        })?;
    let current_triple =
        parse_semver_triple(current).map_err(|_| AdapterPackError::MinVersionUnsatisfied {
            required: required.to_string(),
            current: current.to_string(),
        })?;
    if current_triple < required_triple {
        return Err(AdapterPackError::MinVersionUnsatisfied {
            required: required.to_string(),
            current: current.to_string(),
        });
    }
    Ok(())
}

fn parse_semver_triple(value: &str) -> Result<(u64, u64, u64), String> {
    let parts: Vec<&str> = value.split('.').collect();
    let [major, minor, patch] = parts.as_slice() else {
        return Err(format!("'{value}' is not a MAJOR.MINOR.PATCH version"));
    };
    let parse = |part: &str| {
        part.parse::<u64>()
            .map_err(|_| format!("'{value}' has a non-numeric version component '{part}'"))
    };
    Ok((parse(major)?, parse(minor)?, parse(patch)?))
}

fn adapter_write_tensor(
    name: String,
    dims: &[u64],
    values: &[f32],
    dtype: LoraAdapterDtype,
) -> GgufWriteTensor {
    let (tensor_type, data) = match dtype {
        LoraAdapterDtype::F32 => (
            GgufWriteTensorType::F32,
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<u8>>(),
        ),
        LoraAdapterDtype::F16 => (
            GgufWriteTensorType::F16,
            values
                .iter()
                .flat_map(|value| f32_to_f16_bits(*value).to_le_bytes())
                .collect::<Vec<u8>>(),
        ),
    };
    GgufWriteTensor {
        name,
        dims: dims.to_vec(),
        tensor_type,
        data,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use tempfile::TempDir;

    use super::*;
    use crate::ggml_runtime::{GgufWriteTensor, GgufWriteTensorType, GgufWriteValue};

    #[test]
    fn qwen3_asr_lora_target_name_contract() {
        // LLM-decoder 2-D linears are targets.
        for name in [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.3.attn_v.weight",
            "blk.11.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.5.ffn_down.weight",
        ] {
            assert!(
                is_qwen3_asr_lora_target_tensor_name(name),
                "{name} should be a qwen LoRA target"
            );
        }
        // Norms (1-D), bias, embeddings, tied head, and the audio encoder are NOT.
        for name in [
            "blk.0.attn_norm.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.attn_q.bias",
            "output.weight",
            "output_norm.weight",
            "token_embd.weight",
            "audio.blk.0.attn_q.weight",
            "blk.x.attn_q.weight",
        ] {
            assert!(
                !is_qwen3_asr_lora_target_tensor_name(name),
                "{name} should NOT be a qwen LoRA target"
            );
        }
    }

    fn base_request(dir: &Path) -> LoraAdapterWriteRequest {
        LoraAdapterWriteRequest {
            output_path: dir.join("adapter.oadp"),
            id: "test-adapter".to_string(),
            base_model_id: "moonshine-tiny".to_string(),
            base_pack_sha256: "0".repeat(64),
            rank: 2,
            alpha: 4,
            dtype: LoraAdapterDtype::F32,
            min_openasr_version: "0.1.0".to_string(),
            targets: vec![LoraAdapterWriteTarget {
                base_tensor: "dec.blk.0.attn_q.weight".to_string(),
                input_dim: 3,
                output_dim: 5,
                a_values: vec![0.5; 3 * 2],
                b_values: vec![0.25; 2 * 5],
            }],
        }
    }

    #[test]
    fn adapter_pack_roundtrip_f32() {
        let dir = TempDir::new().expect("tempdir");
        let request = base_request(dir.path());
        write_lora_adapter_pack(&request).expect("write adapter pack");

        let pack = read_lora_adapter_pack(&request.output_path).expect("read adapter pack");
        assert_eq!(pack.manifest.id, "test-adapter");
        assert_eq!(pack.manifest.version, 1);
        assert_eq!(pack.manifest.base_model_id, "moonshine-tiny");
        assert_eq!(pack.manifest.rank, 2);
        assert_eq!(pack.manifest.alpha, 4);
        assert_eq!(pack.manifest.dtype, LoraAdapterDtype::F32);
        assert_eq!(pack.targets.len(), 1);
        let target = &pack.targets[0];
        assert_eq!(target.base_tensor, "dec.blk.0.attn_q.weight");
        assert_eq!(target.input_dim, 3);
        assert_eq!(target.output_dim, 5);
        assert_eq!(target.rank, 2);
        assert_eq!(target.a_values, vec![0.5; 6]);
        assert_eq!(target.b_values, vec![0.25; 10]);
        assert!(pack.fingerprint().contains("test-adapter"));
        assert!(pack.fingerprint().contains(&pack.file_sha256));
    }

    #[test]
    fn adapter_pack_roundtrip_f16_preserves_exact_halves() {
        let dir = TempDir::new().expect("tempdir");
        let mut request = base_request(dir.path());
        request.dtype = LoraAdapterDtype::F16;
        // All values chosen f16-exact so the roundtrip must be lossless.
        request.targets[0].a_values = vec![0.5, -1.25, 2.0, 0.0, -0.75, 4.0];
        request.targets[0].b_values = vec![1.5; 10];
        write_lora_adapter_pack(&request).expect("write adapter pack");

        let pack = read_lora_adapter_pack(&request.output_path).expect("read adapter pack");
        assert_eq!(pack.manifest.dtype, LoraAdapterDtype::F16);
        assert_eq!(pack.targets[0].a_values, request.targets[0].a_values);
        assert_eq!(pack.targets[0].b_values, vec![1.5; 10]);
    }

    #[test]
    fn non_adapter_pack_fails_closed_on_kind() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("not-adapter.gguf");
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "openasr.model.id".to_string(),
            GgufWriteValue::String("moonshine-tiny".to_string()),
        );
        let tensors = [GgufWriteTensor {
            name: "fixture.tensor".to_string(),
            dims: vec![1],
            tensor_type: GgufWriteTensorType::F32,
            data: 0.0_f32.to_le_bytes().to_vec(),
        }];
        crate::ggml_runtime::write_gguf_file_v0(&path, &metadata, &tensors).expect("write fixture");

        let error = read_lora_adapter_pack(&path).expect_err("must fail closed");
        assert!(matches!(error, AdapterPackError::NotAnAdapterPack { .. }));
    }

    #[test]
    fn rank_mismatch_between_manifest_and_tensors_fails_closed() {
        let dir = TempDir::new().expect("tempdir");
        let mut request = base_request(dir.path());
        // Manifest says rank 2 (values sized accordingly), then we lie about
        // the rank by changing it post-hoc: sizes no longer match.
        request.rank = 3;
        let error = write_lora_adapter_pack(&request).expect_err("writer must reject");
        assert!(matches!(error, AdapterPackError::WriteFailed { .. }));
    }

    #[test]
    fn missing_adapter_tensor_fails_closed() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("missing-tensor.oadp");
        // Hand-build a manifest that names a target with no tensors present.
        let mut metadata = BTreeMap::new();
        for (key, value) in [
            (OADP_KEY_PACKAGE_KIND, OADP_PACKAGE_KIND_ADAPTER_PACK),
            (OADP_KEY_ADAPTER_ID, "broken"),
            (OADP_KEY_ADAPTER_METHOD, OADP_ADAPTER_METHOD_LORA),
            (OADP_KEY_BASE_MODEL_ID, "moonshine-tiny"),
            (OADP_KEY_DTYPE, "f32"),
            (OADP_KEY_MIN_OPENASR_VERSION, "0.1.0"),
        ] {
            metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
        }
        metadata.insert(
            OADP_KEY_BASE_PACK_SHA256.to_string(),
            GgufWriteValue::String("0".repeat(64)),
        );
        metadata.insert(OADP_KEY_ADAPTER_VERSION.to_string(), GgufWriteValue::U32(1));
        metadata.insert(OADP_KEY_RANK.to_string(), GgufWriteValue::U32(2));
        metadata.insert(OADP_KEY_ALPHA.to_string(), GgufWriteValue::U32(2));
        metadata.insert(
            OADP_KEY_TARGET_TENSORS.to_string(),
            GgufWriteValue::StringArray(vec!["dec.blk.0.attn_q.weight".to_string()]),
        );
        let tensors = [GgufWriteTensor {
            name: "unrelated.tensor".to_string(),
            dims: vec![1],
            tensor_type: GgufWriteTensorType::F32,
            data: 0.0_f32.to_le_bytes().to_vec(),
        }];
        crate::ggml_runtime::write_gguf_file_v0(&path, &metadata, &tensors).expect("write fixture");

        let error = read_lora_adapter_pack(&path).expect_err("must fail closed");
        assert!(matches!(
            error,
            AdapterPackError::AdapterTensorMissing { ref name, .. }
                if name == "dec.blk.0.attn_q.weight.lora_a"
        ));
    }

    #[test]
    fn base_binding_fails_closed_on_model_id_and_sha() {
        let dir = TempDir::new().expect("tempdir");
        // Tiny fake base pack with a model id.
        let base_path = dir.path().join("base.oasr");
        let mut base_metadata = BTreeMap::new();
        base_metadata.insert(
            "openasr.model.id".to_string(),
            GgufWriteValue::String("moonshine-tiny".to_string()),
        );
        let base_tensors = [GgufWriteTensor {
            name: "fixture.tensor".to_string(),
            dims: vec![1],
            tensor_type: GgufWriteTensorType::F32,
            data: 0.0_f32.to_le_bytes().to_vec(),
        }];
        crate::ggml_runtime::write_gguf_file_v0(&base_path, &base_metadata, &base_tensors)
            .expect("write base fixture");
        let base_sha = file_sha256_hex(&base_path).expect("hash base");

        // Correct binding passes.
        let mut request = base_request(dir.path());
        request.base_pack_sha256 = base_sha.clone();
        write_lora_adapter_pack(&request).expect("write adapter");
        let pack = read_lora_adapter_pack(&request.output_path).expect("read adapter");
        validate_lora_adapter_base_binding(&pack, &base_path).expect("binding must pass");

        // Wrong sha fails closed with the sha-specific error.
        let mut sha_mismatch = base_request(dir.path());
        sha_mismatch.output_path = dir.path().join("sha-mismatch.oadp");
        sha_mismatch.base_pack_sha256 = "f".repeat(64);
        write_lora_adapter_pack(&sha_mismatch).expect("write adapter");
        let pack = read_lora_adapter_pack(&sha_mismatch.output_path).expect("read adapter");
        let error = validate_lora_adapter_base_binding(&pack, &base_path)
            .expect_err("sha mismatch must fail closed");
        assert!(matches!(
            error,
            AdapterPackError::BasePackSha256Mismatch { .. }
        ));

        // Wrong base model id fails closed with the model-id-specific error.
        let mut id_mismatch = base_request(dir.path());
        id_mismatch.output_path = dir.path().join("id-mismatch.oadp");
        id_mismatch.base_model_id = "qwen3-asr-0.6b".to_string();
        id_mismatch.base_pack_sha256 = base_sha;
        write_lora_adapter_pack(&id_mismatch).expect("write adapter");
        let pack = read_lora_adapter_pack(&id_mismatch.output_path).expect("read adapter");
        let error = validate_lora_adapter_base_binding(&pack, &base_path)
            .expect_err("model id mismatch must fail closed");
        assert!(matches!(
            error,
            AdapterPackError::BaseModelIdMismatch { .. }
        ));
    }

    #[test]
    fn min_openasr_version_gate_fails_closed() {
        let dir = TempDir::new().expect("tempdir");
        let base_path = dir.path().join("base.oasr");
        let mut base_metadata = BTreeMap::new();
        base_metadata.insert(
            "openasr.model.id".to_string(),
            GgufWriteValue::String("moonshine-tiny".to_string()),
        );
        let base_tensors = [GgufWriteTensor {
            name: "fixture.tensor".to_string(),
            dims: vec![1],
            tensor_type: GgufWriteTensorType::F32,
            data: 0.0_f32.to_le_bytes().to_vec(),
        }];
        crate::ggml_runtime::write_gguf_file_v0(&base_path, &base_metadata, &base_tensors)
            .expect("write base fixture");

        let mut request = base_request(dir.path());
        request.min_openasr_version = "999.0.0".to_string();
        request.base_pack_sha256 = file_sha256_hex(&base_path).expect("hash base");
        write_lora_adapter_pack(&request).expect("write adapter");
        let pack = read_lora_adapter_pack(&request.output_path).expect("read adapter");
        let error = validate_lora_adapter_base_binding(&pack, &base_path)
            .expect_err("future min version must fail closed");
        assert!(matches!(
            error,
            AdapterPackError::MinVersionUnsatisfied { .. }
        ));
    }

    #[test]
    fn moonshine_target_name_contract() {
        for allowed in [
            "enc.blk.0.attn_q.weight",
            "enc.blk.7.ffn_down.weight",
            "dec.blk.3.cross_k.weight",
            "dec.blk.3.cross_v.weight",
            "dec.blk.0.ffn_up.weight",
        ] {
            assert!(is_moonshine_lora_target_tensor_name(allowed), "{allowed}");
        }
        for rejected in [
            "enc.blk.0.cross_k.weight", // encoder has no cross-attention
            "dec.emb.weight",           // embedding/tied logits not targetable
            "dec.blk.x.attn_q.weight",  // non-numeric layer index
            "dec.blk.0.attn_q.bias",
            "dec.blk.0.attn_norm.weight",
            "enc.conv1.weight",
        ] {
            assert!(
                !is_moonshine_lora_target_tensor_name(rejected),
                "{rejected}"
            );
        }
    }
}
