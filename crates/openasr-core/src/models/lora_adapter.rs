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
//! Resolution results are cached by `(family contract, adapter content, base
//! content)` in a service-root-owned, byte-bounded single-flight cache. Cold
//! loads take an immutable snapshot of the already-open adapter generation;
//! metadata, tensors and sha256 identity therefore cannot cross file versions.
//! The admitted host owner and its lease share a lifetime, and the adapter
//! fingerprint participates in every runtime/cgraph cache key, so prepared
//! graphs are never reused across adapters (or adapter/no-adapter runs).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use thiserror::Error;

use crate::adapter_pack::{
    AdapterPackError, LoraAdapterPack, OPENASR_ADAPTER_ENV, active_adapter_path,
    plan_lora_adapter_resources, read_lora_adapter_pack_from_runtime_source,
    validate_lora_adapter_base_binding_from_runtime_source,
};
use crate::ggml_runtime::{
    GgmlCpuGraphError, GgmlStaticTensor, GgmlStaticTensorArena, validate_ggml_runtime_source_path,
};
use crate::models::admitted_host_object_cache::{
    AdmittedHostObjectCache, AdmittedHostObjectCacheLimits,
};
use crate::models::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight;
use crate::models::system_memory_owner::{
    AdmittedHostObject, SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
    SystemMemoryAllocationTransactionError, SystemMemoryCapacity, SystemMemoryOwner,
    SystemMemoryOwnerError,
};

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
    targets_by_base_tensor: Vec<(String, LoraTarget)>,
}

impl ResolvedLoraAdapter {
    pub(crate) fn target(&self, base_tensor_name: &str) -> Option<&LoraTarget> {
        self.targets_by_base_tensor
            .binary_search_by(|(name, _)| name.as_str().cmp(base_tensor_name))
            .ok()
            .map(|index| &self.targets_by_base_tensor[index].1)
    }

    fn retained_system_memory_bytes(&self) -> Result<u64, LoraResolveError> {
        let mut capacity = SystemMemoryCapacity::default();
        capacity
            .add_string(&self.fingerprint, "lora fingerprint")
            .map_err(LoraResolveError::MemoryAccounting)?;
        capacity
            .add_vec(&self.targets_by_base_tensor, "lora target table")
            .map_err(LoraResolveError::MemoryAccounting)?;
        for (name, target) in &self.targets_by_base_tensor {
            capacity
                .add_string(name, "lora target name")
                .map_err(LoraResolveError::MemoryAccounting)?;
            capacity
                .add_vec(&target.a_values, "lora A values")
                .map_err(LoraResolveError::MemoryAccounting)?;
            capacity
                .add_vec(&target.b_scaled_values, "lora scaled B values")
                .map_err(LoraResolveError::MemoryAccounting)?;
        }
        Ok(capacity.finish())
    }
}

pub(crate) type ResolvedLoraAdapterHandle = AdmittedHostObject<ResolvedLoraAdapter>;

pub(crate) fn resolved_lora_adapter(handle: &ResolvedLoraAdapterHandle) -> &ResolvedLoraAdapter {
    handle.as_ref()
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
    #[error("adapter resolution cache is poisoned")]
    CachePoisoned,
    #[error("adapter system-memory accounting failed: {0}")]
    MemoryAccounting(String),
    #[error("adapter system-memory admission failed: {0}")]
    MemoryAdmission(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AdapterCacheKey {
    contract_id: &'static str,
    adapter_content_id: String,
    base_content_id: String,
}

/// Service-root-owned, single-flight and byte-bounded cache. Dropping the
/// executor drops every idle adapter owner; an in-flight handle retains its
/// own SystemMemory admission lease until graph construction finishes.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedLoraAdapterCache {
    owners: AdmittedHostObjectCache<AdapterCacheKey, ResolvedLoraAdapter>,
}

impl Default for ResolvedLoraAdapterCache {
    fn default() -> Self {
        Self {
            owners: AdmittedHostObjectCache::new(AdmittedHostObjectCacheLimits::new(
                4,
                crate::host::host_available_memory_bytes().unwrap_or(u64::MAX),
            )),
        }
    }
}

impl ResolvedLoraAdapterCache {
    pub(crate) fn clear(&self) {
        self.owners.clear();
    }

    pub(crate) fn evict_base_content_id(&self, base_content_id: &str) {
        self.owners
            .evict_where(|key| key.base_content_id == base_content_id);
    }

    #[cfg(test)]
    fn usage_for_test(&self) -> (usize, u64) {
        self.owners.usage_for_test()
    }
}

/// Resolve the active adapter (request-level `--adapter` path, falling back to
/// the server-side `OPENASR_ADAPTER` env var) for an execution. Returns
/// `Ok(None)` when no adapter is configured; otherwise the adapter must load AND
/// bind to this exact base pack or the whole transcription fails (fail-closed;
/// never silently ignored). `is_target` decides which base tensor names are valid
/// LoRA targets for this model family; `model_label` / `allowed` shape the
/// fail-closed [`LoraResolveError::TargetNotAllowed`] message.
pub(crate) fn resolve_lora_adapter(
    cache: &ResolvedLoraAdapterCache,
    request_adapter_path: Option<&Path>,
    preflight: &GgmlAsrRuntimeSourcePreflight,
    contract_id: &'static str,
    is_target: fn(&str) -> bool,
    model_label: &'static str,
    allowed: &'static str,
) -> Result<Option<ResolvedLoraAdapterHandle>, LoraResolveError> {
    let Some(adapter_path) = active_adapter_path(request_adapter_path) else {
        return Ok(None);
    };
    if adapter_path.as_os_str().is_empty() {
        return Err(LoraResolveError::EmptyAdapterPath);
    }
    let adapter_source = validate_ggml_runtime_source_path(&adapter_path).map_err(|error| {
        AdapterPackError::Unreadable {
            path: adapter_path.clone(),
            reason: error.to_string(),
        }
    })?;
    let adapter_content_id = adapter_source.freshly_hashed_content_id();
    let base_content_id = preflight.runtime_source.content_id().to_owned();
    let key = AdapterCacheKey {
        contract_id,
        adapter_content_id: adapter_content_id.clone(),
        base_content_id: base_content_id.clone(),
    };

    cache
        .owners
        .get_or_try_insert_with(
            key,
            || {
                let resource_plan = plan_lora_adapter_resources(&adapter_source)?;
                let (materialization_peak, retained_bound) =
                    lora_materialization_memory_bounds(resource_plan)?;
                let peak_bound = adapter_source
                    .immutable_snapshot_construction_peak_bytes(materialization_peak)
                    .map_err(|error| capacity_quote_failure(error.to_string()))?;
                let quote = SystemMemoryAllocationQuote::new(
                    format!("lora:{contract_id}:{adapter_content_id}:{base_content_id}"),
                    peak_bound,
                    retained_bound,
                )
                .map_err(|error| LoraResolveError::MemoryAdmission(error.to_string()))?;
                Ok((retained_bound, (adapter_source, adapter_content_id, quote)))
            },
            |(adapter_source, expected_content_id, quote)| {
                let transaction =
                    SystemMemoryOwner::try_allocate_transaction(quote.clone(), || {
                        let snapshot = adapter_source
                            .immutable_snapshot_matching_content_id(&expected_content_id)
                            .map_err(|error| AdapterPackError::Unreadable {
                                path: adapter_path.clone(),
                                reason: error.to_string(),
                            })?;
                        drop(adapter_source);
                        let pack = read_lora_adapter_pack_from_runtime_source(&snapshot)?;
                        validate_lora_adapter_base_binding_from_runtime_source(
                            &pack,
                            &preflight.runtime_source,
                            &preflight.metadata,
                        )?;
                        let adapter = convert_validated_pack(
                            pack,
                            preflight,
                            is_target,
                            model_label,
                            allowed,
                        )?;
                        let retained = adapter.retained_system_memory_bytes()?;
                        Ok(SystemMemoryAllocationOutcome::new(
                            adapter,
                            quote.peak_bytes,
                            retained,
                        ))
                    });
                match transaction {
                    Ok(owner) => Ok(Arc::new(owner)),
                    Err(SystemMemoryAllocationTransactionError::Allocation(error)) => Err(error),
                    Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
                        Err(LoraResolveError::MemoryAdmission(error.to_string()))
                    }
                }
            },
            || LoraResolveError::CachePoisoned,
        )
        .map(Some)
}

fn lora_materialization_memory_bounds(
    plan: crate::adapter_pack::LoraAdapterResourcePlan,
) -> Result<(u64, u64), LoraResolveError> {
    // A valid target owns two tensors. Using ceil here also bounds malformed
    // odd descriptor counts until structural validation rejects them.
    let possible_targets = plan.declared_tensors.div_ceil(2);
    let target_table_bytes = possible_targets
        .checked_mul(std::mem::size_of::<(String, LoraTarget)>() as u64)
        .ok_or_else(|| capacity_quote_failure("LoRA target table byte count overflowed"))?;

    // Retained form: each serialized f16 payload expands by at most 2x to f32;
    // f32 is unchanged. Target names are bytes already present in the GGUF
    // header, so `2 * source` bounds payload + names. Inline Vec/String fields
    // and the fixed-size content fingerprint are added explicitly.
    let retained_bound = plan
        .source_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(target_table_bytes))
        .and_then(|bytes| bytes.checked_add(128))
        .ok_or_else(|| capacity_quote_failure("LoRA retained byte bound overflowed"))?;

    // Bounded GGUF parsing admits descriptor vectors exactly. Variable parser
    // payload is wire-derived: every string object corresponds to an eight-byte
    // serialized length slot. The C ABI reports its own temporary + retained
    // std::string ratio; Rust adds one ratio each for the simultaneously-live
    // metadata and tensor-index views. This is architecture/STL aware instead
    // of assuming a 24-byte string. The later f32 expansion is no larger than
    // this parser phase. Snapshot copying is a separate phase composed by the
    // caller with max(), not addition.
    let rust_string_wire_multiplier =
        (std::mem::size_of::<String>() as u64).div_ceil(std::mem::size_of::<u64>() as u64) + 1;
    let wire_multiplier = plan
        .bounded_parser_payload_wire_multiplier
        .checked_add(
            rust_string_wire_multiplier
                .checked_mul(2)
                .ok_or_else(|| capacity_quote_failure("Rust string byte ratio overflowed"))?,
        )
        .ok_or_else(|| capacity_quote_failure("GGUF wire byte ratio overflowed"))?;
    // A page per logical descriptor is the same allocator-independent upper
    // commitment used by the pure-Rust weight owners. It covers BTree/vector
    // node layout, allocator headers and size-class rounding without relying
    // on private std/STL implementation details.
    const HOST_DESCRIPTOR_COMMITMENT_BYTES: u64 = 4096;
    let rust_descriptor_bytes = plan
        .declared_tensors
        .checked_mul(HOST_DESCRIPTOR_COMMITMENT_BYTES)
        .and_then(|bytes| {
            plan.declared_kv
                .checked_mul(HOST_DESCRIPTOR_COMMITMENT_BYTES)
                .and_then(|kv| bytes.checked_add(kv))
        })
        .and_then(|bytes| bytes.checked_add(target_table_bytes))
        .ok_or_else(|| capacity_quote_failure("LoRA descriptor byte bound overflowed"))?;
    let materialization_peak = plan
        .source_bytes
        .checked_mul(wire_multiplier)
        .and_then(|bytes| bytes.checked_add(plan.bounded_parser_structural_bytes))
        .and_then(|bytes| bytes.checked_add(rust_descriptor_bytes))
        .ok_or_else(|| capacity_quote_failure("LoRA materialization peak overflowed"))?;
    Ok((materialization_peak.max(retained_bound), retained_bound))
}

fn capacity_quote_failure(reason: impl Into<String>) -> LoraResolveError {
    LoraResolveError::MemoryAdmission(
        SystemMemoryOwnerError::capacity_failure("lora_memory_quote", reason).to_string(),
    )
}

fn convert_validated_pack(
    pack: LoraAdapterPack,
    preflight: &GgmlAsrRuntimeSourcePreflight,
    is_target: fn(&str) -> bool,
    model_label: &'static str,
    allowed: &'static str,
) -> Result<ResolvedLoraAdapter, LoraResolveError> {
    let base_pack_path = preflight.runtime_source.path();
    let alpha = pack.manifest.alpha as f32;
    let rank = pack.manifest.rank as f32;
    let scale = alpha / rank;
    let fingerprint = pack.fingerprint();

    let mut targets_by_base_tensor = Vec::with_capacity(pack.targets.len());
    for mut target in pack.targets {
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
        for value in &mut target.b_values {
            *value *= scale;
        }
        targets_by_base_tensor.push((
            target.base_tensor,
            LoraTarget {
                rank: target.rank,
                input_dim: target.input_dim,
                output_dim: target.output_dim,
                a_values: target.a_values,
                b_scaled_values: target.b_values,
            },
        ));
    }
    targets_by_base_tensor.sort_unstable_by(|left, right| left.0.cmp(&right.0));

    Ok(ResolvedLoraAdapter {
        fingerprint,
        targets_by_base_tensor,
    })
}

#[cfg(test)]
pub(crate) fn lora_adapter_for_test(
    fingerprint: String,
    targets: Vec<(String, LoraTarget)>,
) -> ResolvedLoraAdapter {
    let mut targets = targets;
    targets.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    ResolvedLoraAdapter {
        fingerprint,
        targets_by_base_tensor: targets,
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    fn adapter(fingerprint: &str) -> ResolvedLoraAdapter {
        ResolvedLoraAdapter {
            fingerprint: fingerprint.to_string(),
            targets_by_base_tensor: Vec::new(),
        }
    }

    #[test]
    fn lora_memory_bounds_are_monotonic_and_phase_safe() {
        let plan = crate::adapter_pack::LoraAdapterResourcePlan {
            source_bytes: 1_024,
            declared_tensors: 14,
            declared_kv: 11,
            bounded_parser_structural_bytes: 4_096,
            bounded_parser_payload_wire_multiplier: 6,
        };
        let (peak, retained) = lora_materialization_memory_bounds(plan).expect("memory bounds");
        assert!(peak >= retained);

        let larger = crate::adapter_pack::LoraAdapterResourcePlan {
            source_bytes: 2_048,
            ..plan
        };
        let (larger_peak, larger_retained) =
            lora_materialization_memory_bounds(larger).expect("larger memory bounds");
        assert!(larger_peak > peak);
        assert!(larger_retained > retained);
    }

    #[test]
    fn lora_memory_bound_overflow_is_a_typed_capacity_failure() {
        let error =
            lora_materialization_memory_bounds(crate::adapter_pack::LoraAdapterResourcePlan {
                source_bytes: u64::MAX,
                declared_tensors: 0,
                declared_kv: 0,
                bounded_parser_structural_bytes: 0,
                bounded_parser_payload_wire_multiplier: 6,
            })
            .expect_err("overflow must fail");
        assert!(matches!(error, LoraResolveError::MemoryAdmission(_)));
    }

    fn insert(
        cache: &ResolvedLoraAdapterCache,
        contract_id: &'static str,
        fingerprint: &str,
    ) -> ResolvedLoraAdapterHandle {
        let key = AdapterCacheKey {
            contract_id,
            adapter_content_id: format!("sha256:{fingerprint}"),
            base_content_id: "sha256:base".to_string(),
        };
        cache
            .owners
            .get_or_try_insert_with(
                key,
                || Ok::<_, ()>((1, ())),
                |()| {
                    Ok::<_, ()>(Arc::new(
                        SystemMemoryOwner::with_committed_requested_bytes_for_test(
                            adapter(fingerprint),
                            1,
                        ),
                    ))
                },
                || (),
            )
            .expect("cache insert")
    }

    #[test]
    fn cache_drop_releases_idle_adapter_owner() {
        let cache = ResolvedLoraAdapterCache::default();
        let handle = insert(&cache, "qwen", "adapter-a");
        let weak = Arc::downgrade(&handle);
        drop(handle);
        assert_eq!(cache.usage_for_test(), (1, 1));

        drop(cache);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn clear_drops_idle_entry_but_preserves_in_flight_handle() {
        let cache = ResolvedLoraAdapterCache::default();
        let handle = insert(&cache, "qwen", "adapter-a");
        cache.clear();

        assert_eq!(cache.usage_for_test(), (0, 0));
        assert_eq!(handle.fingerprint, "adapter-a");
    }

    #[test]
    fn target_contract_is_part_of_cache_identity() {
        let cache = ResolvedLoraAdapterCache::default();
        drop(insert(&cache, "qwen", "same-content"));
        drop(insert(&cache, "moonshine", "same-content"));

        assert_eq!(cache.usage_for_test(), (2, 2));
    }
}
