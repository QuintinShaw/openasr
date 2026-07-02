//! Shared inference-side dynamic-LoRA resolution (OADP Phase 0).
//!
//! Both the Moonshine and Qwen3-ASR side-paths resolve the active `.oadp` path
//! (request-level adapter option, falling back to the server-side
//! `OPENASR_ADAPTER` env var) against the base pack about to execute, fail-closed
//! on every mismatch class, and convert the adapter into runtime-ready host
//! tensors. The model-specific parts — which base tensors are valid LoRA
//! targets, and how the per-layer slots wire into each graph — stay in the model
//! modules; this module owns the format-agnostic resolution, fail-closed
//! validation, and the per-(adapter, base) resolution cache.
//!
//! - `A` stays `[input_dim, rank]` f32 (ne0-major), so `mul_mat(A, x)` contracts
//!   over the input dim;
//! - `B` is pre-scaled by `alpha/rank` at load time into `b_scaled_values`
//!   (`[rank, output_dim]`), so the in-graph side branch is exactly
//!   `y = W@x + B_scaled@(A@x)` with two `mul_mat` + one `add` — no extra `scale`
//!   node. Pre-scaling is mathematically identical to scaling the delta and keeps
//!   the zero-adapter case exact (0 * s == 0).
//!
//! Resolution results are cached per (adapter path, base pack path) with the
//! adapter file re-hashed (sha256) on every hit — adapters are small, so the hash
//! is cheap next to a decode and leaves no mtime-granularity TOCTOU window;
//! installed base packs are immutable by contract. The adapter fingerprint
//! participates in every runtime/cgraph cache key, so prepared graphs are never
//! reused across different adapters (or between adapter and no-adapter runs).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use thiserror::Error;

use crate::adapter_pack::{
    AdapterPackError, LoraAdapterPack, OPENASR_ADAPTER_ENV, active_adapter_path, file_sha256_hex,
    read_lora_adapter_pack, validate_lora_adapter_base_binding,
};
use crate::ggml_runtime::{GgmlCpuGraphError, GgmlStaticTensor, GgmlStaticTensorArena};
use crate::models::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight;

/// In-arena A / pre-scaled-B factors for one LoRA-decorated linear:
/// `y = W@x + B_scaled@(A@x)` (`alpha/rank` is folded into B at load time).
#[derive(Clone, Copy)]
pub(crate) struct LoraSlot {
    /// `[input_dim, rank]` f32.
    pub a: GgmlStaticTensor,
    /// `[rank, output_dim]` f32, pre-multiplied by `alpha/rank`.
    pub b_scaled: GgmlStaticTensor,
}

/// Allocate (but do not upload) the arena tensors for one LoRA target. The
/// caller queues the f32 payload uploads until all arena tensors exist (the
/// arena cannot extend once its backend buffer is allocated).
pub(crate) fn new_lora_slot_tensors(
    arena: &GgmlStaticTensorArena,
    target: &LoraTarget,
    a_name: &'static str,
    b_name: &'static str,
) -> Result<LoraSlot, GgmlCpuGraphError> {
    let a = arena.new_tensor_2d_f32(target.input_dim, target.rank, a_name)?;
    let b_scaled = arena.new_tensor_2d_f32(target.rank, target.output_dim, b_name)?;
    Ok(LoraSlot { a, b_scaled })
}

/// One LoRA-decorated 2-D linear, with values ready for arena upload.
#[derive(Debug, Clone)]
pub(crate) struct LoraTarget {
    pub rank: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    /// `[input_dim, rank]` f32, ne0-major.
    pub a_values: Vec<f32>,
    /// `[rank, output_dim]` f32, ne0-major, pre-multiplied by `alpha/rank`.
    pub b_scaled_values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedLoraAdapter {
    /// Cache-key identity: adapter id + .oadp sha256 + rank + alpha + targets.
    pub fingerprint: String,
    targets_by_base_tensor: HashMap<String, LoraTarget>,
}

impl ResolvedLoraAdapter {
    pub(crate) fn target(&self, base_tensor_name: &str) -> Option<&LoraTarget> {
        self.targets_by_base_tensor.get(base_tensor_name)
    }
}

/// Cache-key component for the runtime caches: empty when no adapter is active.
/// Keying prepared graphs only on the base pack would be a correctness bug
/// (stale adapter graphs would serve other requests).
pub(crate) fn adapter_cache_fingerprint(adapter: Option<&ResolvedLoraAdapter>) -> String {
    adapter
        .map(|adapter| adapter.fingerprint.clone())
        .unwrap_or_default()
}

#[derive(Debug, Error)]
pub(crate) enum LoraResolveError {
    #[error("adapter pack path is set but empty (--adapter / {OPENASR_ADAPTER_ENV})")]
    EmptyAdapterPath,
    #[error(transparent)]
    AdapterPack(#[from] AdapterPackError),
    #[error(
        "adapter target tensor '{name}' is not a {model_label} LoRA target \
         (allowed: {allowed}); fail-closed"
    )]
    TargetNotAllowed {
        name: String,
        model_label: &'static str,
        allowed: &'static str,
    },
    #[error("adapter target tensor '{name}' is missing from base pack '{base_pack}'; fail-closed")]
    TargetMissingFromBase { name: String, base_pack: PathBuf },
    #[error(
        "adapter target '{name}' dims mismatch base tensor (fail-closed): base is \
         [{base_in}, {base_out}], adapter A is [{adapter_in}, rank={rank}], adapter B is \
         [rank={rank}, {adapter_out}]"
    )]
    TargetDimsMismatch {
        name: String,
        base_in: usize,
        base_out: usize,
        adapter_in: usize,
        adapter_out: usize,
        rank: usize,
    },
    #[error("adapter '{path}' changed on disk since it was loaded; fail-closed")]
    AdapterFileChanged { path: PathBuf },
    #[error("adapter resolution cache is poisoned")]
    CachePoisoned,
}

type AdapterCacheKey = (PathBuf, PathBuf);

struct CachedAdapter {
    /// sha256 (lowercase hex) of the `.oadp` file at load time; every cache hit
    /// re-hashes the file and must match this exactly.
    file_sha256: String,
    adapter: Arc<ResolvedLoraAdapter>,
}

static RESOLVED_ADAPTERS: Mutex<Option<HashMap<AdapterCacheKey, CachedAdapter>>> = Mutex::new(None);

/// Resolve the active adapter (request-level `--adapter` path, falling back to
/// the server-side `OPENASR_ADAPTER` env var) for an execution. Returns
/// `Ok(None)` when no adapter is configured; otherwise the adapter must load AND
/// bind to this exact base pack or the whole transcription fails (fail-closed;
/// never silently ignored). `is_target` decides which base tensor names are valid
/// LoRA targets for this model family; `model_label` / `allowed` shape the
/// fail-closed [`LoraResolveError::TargetNotAllowed`] message.
pub(crate) fn resolve_lora_adapter(
    request_adapter_path: Option<&Path>,
    preflight: &GgmlAsrRuntimeSourcePreflight,
    is_target: fn(&str) -> bool,
    model_label: &'static str,
    allowed: &'static str,
) -> Result<Option<Arc<ResolvedLoraAdapter>>, LoraResolveError> {
    let Some(adapter_path) = active_adapter_path(request_adapter_path) else {
        return Ok(None);
    };
    if adapter_path.as_os_str().is_empty() {
        return Err(LoraResolveError::EmptyAdapterPath);
    }
    let base_pack_path = preflight.runtime_source.path().to_path_buf();
    let key = (adapter_path.clone(), base_pack_path.clone());

    // Content identity: hash the file up front. Adapters are a few MB at most, so
    // the sha256 is cheap next to a decode, and (unlike len+mtime) it leaves no
    // mtime-granularity TOCTOU window.
    let file_sha256 = adapter_file_sha256(&adapter_path)?;

    {
        let cache = RESOLVED_ADAPTERS
            .lock()
            .map_err(|_| LoraResolveError::CachePoisoned)?;
        if let Some(cached) = cache.as_ref().and_then(|map| map.get(&key))
            && cached.file_sha256 == file_sha256
        {
            return Ok(Some(Arc::clone(&cached.adapter)));
        }
    }

    let pack = read_lora_adapter_pack(&adapter_path)?;
    // Fail closed if the file mutated while we were reading it: the reader hashes
    // the file AFTER reading metadata/tensors, so a mismatch with the pre-read
    // hash means the loaded tensors cannot be trusted.
    if pack.file_sha256 != file_sha256 {
        return Err(LoraResolveError::AdapterFileChanged { path: adapter_path });
    }
    validate_lora_adapter_base_binding(&pack, &base_pack_path)?;
    let adapter = Arc::new(convert_validated_pack(
        &pack,
        preflight,
        is_target,
        model_label,
        allowed,
    )?);

    let mut cache = RESOLVED_ADAPTERS
        .lock()
        .map_err(|_| LoraResolveError::CachePoisoned)?;
    cache.get_or_insert_with(HashMap::new).insert(
        key,
        CachedAdapter {
            file_sha256,
            adapter: Arc::clone(&adapter),
        },
    );
    Ok(Some(adapter))
}

fn adapter_file_sha256(path: &Path) -> Result<String, LoraResolveError> {
    file_sha256_hex(path)
        .map_err(|reason| AdapterPackError::Unreadable {
            path: path.to_path_buf(),
            reason,
        })
        .map_err(LoraResolveError::from)
}

fn convert_validated_pack(
    pack: &LoraAdapterPack,
    preflight: &GgmlAsrRuntimeSourcePreflight,
    is_target: fn(&str) -> bool,
    model_label: &'static str,
    allowed: &'static str,
) -> Result<ResolvedLoraAdapter, LoraResolveError> {
    let base_pack_path = preflight.runtime_source.path();
    let alpha = pack.manifest.alpha as f32;
    let rank = pack.manifest.rank as f32;
    let scale = alpha / rank;

    let mut targets_by_base_tensor = HashMap::with_capacity(pack.targets.len());
    for target in &pack.targets {
        if !is_target(&target.base_tensor) {
            return Err(LoraResolveError::TargetNotAllowed {
                name: target.base_tensor.clone(),
                model_label,
                allowed,
            });
        }
        let base_tensor = preflight
            .tensor_index
            .get(&target.base_tensor)
            .ok_or_else(|| LoraResolveError::TargetMissingFromBase {
                name: target.base_tensor.clone(),
                base_pack: base_pack_path.to_path_buf(),
            })?;
        let base_dims: Vec<usize> = base_tensor.dims.iter().map(|&dim| dim as usize).collect();
        if base_dims.as_slice() != [target.input_dim, target.output_dim] {
            let (base_in, base_out) = match base_dims.as_slice() {
                [ne0, ne1] => (*ne0, *ne1),
                _ => (0, 0),
            };
            return Err(LoraResolveError::TargetDimsMismatch {
                name: target.base_tensor.clone(),
                base_in,
                base_out,
                adapter_in: target.input_dim,
                adapter_out: target.output_dim,
                rank: target.rank,
            });
        }
        let b_scaled_values: Vec<f32> =
            target.b_values.iter().map(|&value| value * scale).collect();
        targets_by_base_tensor.insert(
            target.base_tensor.clone(),
            LoraTarget {
                rank: target.rank,
                input_dim: target.input_dim,
                output_dim: target.output_dim,
                a_values: target.a_values.clone(),
                b_scaled_values,
            },
        );
    }

    Ok(ResolvedLoraAdapter {
        fingerprint: pack.fingerprint(),
        targets_by_base_tensor,
    })
}

#[cfg(test)]
pub(crate) fn lora_adapter_for_test(
    fingerprint: String,
    targets: Vec<(String, LoraTarget)>,
) -> ResolvedLoraAdapter {
    ResolvedLoraAdapter {
        fingerprint,
        targets_by_base_tensor: targets.into_iter().collect(),
    }
}
