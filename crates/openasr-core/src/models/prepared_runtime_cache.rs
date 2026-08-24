use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::time::Instant;

use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::models::admitted_host_object_cache::{
    AdmittedHostObjectCache, AdmittedHostObjectCacheLimits,
    DEFAULT_ADMITTED_HOST_OBJECT_CACHE_MAX_ENTRIES,
};
use crate::models::runtime_cache_coordinator::is_cacheable_pack_content_id;
use crate::models::system_memory_owner::{
    AdmittedHostObject, SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote,
    SystemMemoryAllocationTransactionError, SystemMemoryOwner, SystemMemoryOwnerError,
};
use crate::stage_timing;
use crate::{GgmlRuntimeSource, GgufMetadata, GgufTensorIndex};

// One cache entry per pack content id. Path alone is never a key -- same-path byte
// replacement resolves a different `pack_content_id` and must miss, which the
// map key already guarantees on its own; there is no generation/epoch here
// (removed -- see `runtime_cache_coordinator`'s module doc comment for why a
// shared counter in this kind of key was an audited bug: it invalidated every
// resident content identity on any unrelated cache's idle-unload / owner
// shutdown / pack replace). Idle unload evicts via [`PreparedRuntimeCache::clear`]
// (whole-cache) or [`PreparedRuntimeCache::evict_content_id`] (one entry);
// see each family's `unload_idle_state`.
//
// Failed and panicking builds remain retryable. `get_or_try_insert_with` runs
// `build()` behind `catch_unwind` so a panic never unwinds through the generic
// cache's per-key single-flight lock.
//
// Unreadable packs skip the map entirely (one-shot uncached build) rather than
// inserting an `unreadable:*` token that would poison or falsely collide later.
/// Compile-time declaration that a prepared value contains no backend handle,
/// scheduler, graph runner, device buffer, uploaded tensor arena, or cache key
/// derived from an execution candidate. Only such values may use the
/// content-only shared prepared cache; device-owning values belong in a
/// lane-keyed resident cache.
pub(crate) trait SystemMemoryMaterialization: Send + Sync + 'static {
    fn retained_system_memory_bytes(&self) -> Result<u64, String>;
}

/// Candidate facts available before host materialization starts. A family
/// oracle must quote the storage selected for this exact backend rather than
/// infer heap growth from the pack's file size.
#[derive(Clone, Copy)]
pub(crate) struct PreparedRuntimeQuoteContext<'a> {
    /// Engine architecture identity selected by the architecture registry.
    /// This is intentionally distinct from GGUF `general.architecture`, whose
    /// runtime-format alias is family-specific.
    pub(crate) model_architecture: &'a str,
    pub(crate) metadata: &'a GgufMetadata,
    pub(crate) tensor_index: &'a GgufTensorIndex,
    pub(crate) backend: GgmlCpuGraphBackend,
}

pub(crate) trait HostNeutralPreparedRuntime: SystemMemoryMaterialization {
    fn system_memory_quote(
        context: PreparedRuntimeQuoteContext<'_>,
        pack_content_id: &str,
    ) -> Result<SystemMemoryAllocationQuote, SystemMemoryOwnerError>;
}

/// A cache/in-flight handle whose admitted SystemMemory lease has exactly the
/// same lifetime as the materialized runtime. Removing the cache's clone does
/// not refund bytes while an execution still owns another clone.
pub(crate) type PreparedRuntimeHandle<T> = AdmittedHostObject<T>;

/// Host-neutral prepared runtimes report every retained Rust `Vec` request from
/// container capacity, recursively through their materialized bundles. This is
/// engine-requested heap capacity, not allocator usable-size or physical RSS.
enum PreparedRuntimeAllocationError<E> {
    Build(E),
    Measure(String),
}

/// Checked provisional quote assembled from one family's materialization
/// topology. Values are engine-requested heap capacity. The largest owned
/// allocation is added once more to the construction peak because loaders can
/// briefly hold a source/transposition buffer beside the retained destination.
/// Allocator overhead is intentionally not invented here; live host snapshots
/// and the broker's headroom policy cover it.
pub(crate) struct PreparedRuntimeQuoteBuilder {
    resource_id: String,
    retained_bytes: u64,
    largest_allocation_bytes: u64,
}

impl PreparedRuntimeQuoteBuilder {
    pub(crate) fn new<T: 'static>(pack_content_id: &str) -> Self {
        Self {
            resource_id: format!(
                "prepared-runtime:{}:{pack_content_id}",
                std::any::type_name::<T>()
            ),
            retained_bytes: 0,
            largest_allocation_bytes: 0,
        }
    }

    pub(crate) fn add_owned_bytes(
        &mut self,
        logical_bytes: u64,
        label: &str,
    ) -> Result<(), SystemMemoryOwnerError> {
        self.retained_bytes = self
            .retained_bytes
            .checked_add(logical_bytes)
            .ok_or_else(|| {
                SystemMemoryOwnerError::capacity_failure(
                    "prepared_runtime_quote",
                    format!("{label} retained-byte sum overflowed"),
                )
            })?;
        self.largest_allocation_bytes = self.largest_allocation_bytes.max(logical_bytes);
        Ok(())
    }

    pub(crate) fn add_owned_elements<T>(
        &mut self,
        elements: u64,
        label: &str,
    ) -> Result<(), SystemMemoryOwnerError> {
        let width = u64::try_from(std::mem::size_of::<T>()).map_err(|_| {
            SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                format!("{label} element width does not fit u64"),
            )
        })?;
        let logical_bytes = elements.checked_mul(width).ok_or_else(|| {
            SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                format!("{label} logical byte count overflowed"),
            )
        })?;
        self.add_owned_bytes(logical_bytes, label)
    }

    pub(crate) fn add_tensor_f32(
        &mut self,
        tensor_index: &GgufTensorIndex,
        name: &str,
    ) -> Result<(), SystemMemoryOwnerError> {
        let tensor = required_quote_tensor(tensor_index, name)?;
        let elements = tensor.num_elements().ok_or_else(|| {
            SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                format!("tensor '{name}' element count overflowed"),
            )
        })?;
        self.add_owned_elements::<f32>(elements, name)
    }

    pub(crate) fn add_tensor_f16(
        &mut self,
        tensor_index: &GgufTensorIndex,
        name: &str,
    ) -> Result<(), SystemMemoryOwnerError> {
        let tensor = required_quote_tensor(tensor_index, name)?;
        let elements = tensor.num_elements().ok_or_else(|| {
            SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                format!("tensor '{name}' element count overflowed"),
            )
        })?;
        self.add_owned_elements::<u16>(elements, name)
    }

    pub(crate) fn add_tensor_raw(
        &mut self,
        tensor_index: &GgufTensorIndex,
        name: &str,
    ) -> Result<(), SystemMemoryOwnerError> {
        self.add_owned_bytes(required_quote_tensor(tensor_index, name)?.size_bytes, name)
    }

    pub(crate) fn add_tensor_f32_or_raw_upper_bound(
        &mut self,
        tensor_index: &GgufTensorIndex,
        name: &str,
    ) -> Result<(), SystemMemoryOwnerError> {
        let tensor = required_quote_tensor(tensor_index, name)?;
        let elements = tensor.num_elements().ok_or_else(|| {
            SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                format!("tensor '{name}' element count overflowed"),
            )
        })?;
        let f32_bytes = elements.checked_mul(4).ok_or_else(|| {
            SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                format!("tensor '{name}' f32 byte count overflowed"),
            )
        })?;
        self.add_owned_bytes(tensor.size_bytes.max(f32_bytes), name)
    }

    pub(crate) fn add_tensor_metadata(
        &mut self,
        tensor_index: &GgufTensorIndex,
        name: &str,
    ) -> Result<(), SystemMemoryOwnerError> {
        let tensor = required_quote_tensor(tensor_index, name)?;
        let name_bytes = u64::try_from(tensor.name.len()).map_err(|_| {
            SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                format!("tensor '{name}' name length does not fit u64"),
            )
        })?;
        let dims = u64::try_from(tensor.dims.len()).map_err(|_| {
            SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                format!("tensor '{name}' rank does not fit u64"),
            )
        })?;
        self.add_owned_bytes(name_bytes, "tensor metadata name")?;
        self.add_owned_elements::<u64>(dims, "tensor metadata dims")
    }

    /// Exact Rust-owned metadata retained by one
    /// [`crate::ggml_runtime::GgufOwnedWeightTensorPayload`]. The mapped tensor
    /// bytes are intentionally absent: payload handles share the source mmap.
    pub(crate) fn add_owned_tensor_payload_metadata(
        &mut self,
        tensor_index: &GgufTensorIndex,
        name: &str,
    ) -> Result<(), SystemMemoryOwnerError> {
        let tensor = required_quote_tensor(tensor_index, name)?;
        self.add_owned_bytes(
            u64::try_from(tensor.name.len()).map_err(|_| {
                SystemMemoryOwnerError::capacity_failure(
                    "prepared_runtime_quote",
                    format!("tensor '{name}' name length does not fit u64"),
                )
            })?,
            "owned tensor metadata name",
        )?;
        self.add_owned_bytes(
            u64::try_from(tensor.type_name.len()).map_err(|_| {
                SystemMemoryOwnerError::capacity_failure(
                    "prepared_runtime_quote",
                    format!("tensor '{name}' type-name length does not fit u64"),
                )
            })?,
            "owned tensor metadata type name",
        )?;
        let rank = u64::try_from(tensor.dims.len()).map_err(|_| {
            SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                format!("tensor '{name}' rank does not fit u64"),
            )
        })?;
        // `GgufOwnedWeightTensorPayload` owns both metadata.dims (`u64`) and
        // its platform `usize` projection.
        self.add_owned_elements::<u64>(rank, "owned tensor metadata dims")?;
        self.add_owned_elements::<usize>(rank, "owned tensor platform dims")
    }

    pub(crate) fn add_structural_bytes(
        &mut self,
        logical_bytes: u64,
        label: &str,
    ) -> Result<(), SystemMemoryOwnerError> {
        self.add_owned_bytes(logical_bytes, label)
    }

    /// Adds retained capacity whose construction has no same-sized source
    /// buffer (for example zero-initialized stable KV). It must not become the
    /// builder's duplicate-allocation transient.
    #[allow(dead_code)] // Used by aggregate candidate quotes outside prepared-runtime caches.
    pub(crate) fn add_stable_owned_bytes(
        &mut self,
        logical_bytes: u64,
        label: &str,
    ) -> Result<(), SystemMemoryOwnerError> {
        self.retained_bytes = self
            .retained_bytes
            .checked_add(logical_bytes)
            .ok_or_else(|| {
                SystemMemoryOwnerError::capacity_failure(
                    "prepared_runtime_quote",
                    format!("{label} retained-byte sum overflowed"),
                )
            })?;
        Ok(())
    }

    /// Records a construction-only allocation without adding it to retained
    /// capacity. Materializers that stream one tensor at a time use this for
    /// their largest possible fallback/transposition buffer.
    #[allow(dead_code)] // Used by aggregate candidate quotes outside prepared-runtime caches.
    pub(crate) fn observe_transient_bytes(&mut self, logical_bytes: u64, _label: &str) {
        self.largest_allocation_bytes = self.largest_allocation_bytes.max(logical_bytes);
    }

    pub(crate) fn add_tokenizer_metadata(
        &mut self,
        metadata: &GgufMetadata,
        include_merges: bool,
    ) -> Result<(), SystemMemoryOwnerError> {
        let tokens = metadata
            .get_string_array("tokenizer.ggml.tokens")
            .unwrap_or_default();
        let merges = if include_merges {
            metadata
                .get_string_array("tokenizer.ggml.merges")
                .unwrap_or_default()
        } else {
            &[]
        };
        let token_text_bytes = tokens.iter().try_fold(0_u64, |total, value| {
            u64::try_from(value.len())
                .ok()
                .and_then(|len| total.checked_add(len))
        });
        let merge_text_bytes = merges.iter().try_fold(0_u64, |total, value| {
            u64::try_from(value.len())
                .ok()
                .and_then(|len| total.checked_add(len))
        });
        let (Some(token_text_bytes), Some(merge_text_bytes)) = (token_text_bytes, merge_text_bytes)
        else {
            return Err(SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                "tokenizer metadata byte count overflowed",
            ));
        };
        // The id table and reverse map each own token text; Qwen/Whisper also
        // own merge-map keys. Container payload uses checked type widths only;
        // BTree node and allocator overhead is deliberately left to broker
        // headroom rather than hidden behind an empirical per-entry constant.
        self.add_owned_bytes(
            token_text_bytes
                .checked_mul(2)
                .and_then(|value| value.checked_add(merge_text_bytes))
                .ok_or_else(|| {
                    SystemMemoryOwnerError::capacity_failure(
                        "prepared_runtime_quote",
                        "tokenizer owned text quote overflowed",
                    )
                })?,
            "tokenizer owned text",
        )?;
        let token_count = u64::try_from(tokens.len()).map_err(|_| {
            SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                "tokenizer token count does not fit u64",
            )
        })?;
        let merge_count = u64::try_from(merges.len()).map_err(|_| {
            SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                "tokenizer merge count does not fit u64",
            )
        })?;
        self.add_owned_elements::<Option<String>>(token_count, "tokenizer id table")?;
        self.add_owned_elements::<(String, u32)>(token_count, "tokenizer reverse-map payload")?;
        self.add_owned_elements::<(String, usize)>(merge_count, "tokenizer merge-map payload")
    }

    pub(crate) fn finish(self) -> Result<SystemMemoryAllocationQuote, SystemMemoryOwnerError> {
        let peak_bytes = self
            .retained_bytes
            .checked_add(self.largest_allocation_bytes)
            .ok_or_else(|| {
                SystemMemoryOwnerError::capacity_failure(
                    "prepared_runtime_quote",
                    "prepared-runtime construction peak overflowed",
                )
            })?;
        SystemMemoryAllocationQuote::new(self.resource_id, peak_bytes, self.retained_bytes)
    }
}

fn required_quote_tensor<'a>(
    tensor_index: &'a GgufTensorIndex,
    name: &str,
) -> Result<&'a crate::GgufTensorMetadata, SystemMemoryOwnerError> {
    tensor_index.get(name).ok_or_else(|| {
        SystemMemoryOwnerError::capacity_failure(
            "prepared_runtime_quote",
            format!("required prepared-runtime tensor '{name}' is missing from GGUF index"),
        )
    })
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedRuntimeCache<T: HostNeutralPreparedRuntime> {
    admitted_by_content_id: AdmittedHostObjectCache<String, T>,
}

impl<T: HostNeutralPreparedRuntime> Default for PreparedRuntimeCache<T> {
    fn default() -> Self {
        // The process-wide broker remains the aggregate SystemMemory authority.
        // A live available-memory observation gives this one cache a finite
        // retention ceiling without treating total RAM as currently usable;
        // unknown availability degrades to the entry cap rather than inventing
        // a byte limit. The broker still rechecks every cold materialization
        // against a fresh observation and its global headroom policy.
        let max_committed_requested_bytes =
            crate::host::host_available_memory_bytes().unwrap_or(u64::MAX);
        Self {
            admitted_by_content_id: AdmittedHostObjectCache::new(
                AdmittedHostObjectCacheLimits::new(
                    DEFAULT_ADMITTED_HOST_OBJECT_CACHE_MAX_ENTRIES,
                    max_committed_requested_bytes,
                ),
            ),
        }
    }
}

impl<T: HostNeutralPreparedRuntime> PreparedRuntimeCache<T> {
    /// Probe an already-admitted runtime by the validated content identity.
    /// A cold/building/staged entry is a miss: capacity planning must never
    /// trigger model materialization or wait behind it.
    pub(crate) fn ready(
        &self,
        runtime_source: &GgmlRuntimeSource,
    ) -> Option<PreparedRuntimeHandle<T>> {
        let pack_content_id = runtime_source.content_id();
        is_cacheable_pack_content_id(pack_content_id)
            .then(|| {
                self.admitted_by_content_id
                    .ready(&pack_content_id.to_string())
            })
            .flatten()
    }

    /// `runtime_source` must be the already-open, already-validated source
    /// for the pack being built -- never re-derive a fresh source from a
    /// path just to call this. Its `content_id()` (fd-derived, memoized) is
    /// the cache key; `build` still receives whatever the caller closed over
    /// (typically the same source) to actually materialize the runtime, so
    /// identity and bytes are provably read from one open handle.
    pub(crate) fn get_or_try_insert_with<E, F, M, C>(
        &self,
        runtime_source: &GgmlRuntimeSource,
        quote_context: PreparedRuntimeQuoteContext<'_>,
        build: F,
        map_poisoned_lock: M,
        map_capacity_error: C,
    ) -> Result<PreparedRuntimeHandle<T>, E>
    where
        F: FnOnce() -> Result<T, E>,
        M: Fn() -> E,
        C: Fn(SystemMemoryOwnerError) -> E,
    {
        let pack_content_id = runtime_source.content_id();
        if !is_cacheable_pack_content_id(pack_content_id) {
            // Fail closed on insert: unreadable / non-cacheable content ids never
            // enter the shared map. Still honor the caller's request with a
            // one-shot uncached build so a transient unreadable path does not
            // wedge the request path behind a permanent "unreadable" slot.
            return Self::build_once_uncached(
                quote_context,
                pack_content_id,
                build,
                map_poisoned_lock,
                map_capacity_error,
            );
        }
        let pack_content_id = pack_content_id.to_string();

        // Model pack loading (mmap + tensor materialization + context/graph
        // construction, up to inference-ready) happens exactly here, exactly
        // once per distinct content identity (subsequent calls hit the cache
        // check above). This one call site covers every builtin model family
        // that goes through this cache, so it is the single place to time
        // "how long did loading this pack take" without instrumenting each
        // family's build function separately.
        //
        // `build()` runs behind `catch_unwind` rather than being called
        // directly: this slot's `MutexGuard` (`slot_guard`) is held across the
        // call, and a `Mutex` is poisoned when a guard is dropped *while the
        // thread is unwinding from a panic*. Left uncaught, a single panicking
        // build would permanently wedge this one runtime identity -- every
        // future caller would get a poisoned-lock error instead of a clean
        // retry. `catch_unwind` fully absorbs the panic before this function
        // returns, so by the time `slot_guard` actually drops the thread is no
        // longer unwinding and the `Mutex` stays unpoisoned. `AssertUnwindSafe`
        // is sound here because `build()` is a pure host materialization
        // closure that never touches this cache's own state.
        self.admitted_by_content_id.get_or_try_insert_with(
            pack_content_id.clone(),
            || {
                let quote = T::system_memory_quote(quote_context, &pack_content_id)
                    .map_err(&map_capacity_error)?;
                Ok((quote.retained_bytes, quote))
            },
            |quote| {
                Self::allocate_once(
                    &pack_content_id,
                    quote,
                    build,
                    &map_poisoned_lock,
                    &map_capacity_error,
                    "",
                )
            },
            &map_poisoned_lock,
        )
    }

    fn build_once_uncached<E, F, M, C>(
        quote_context: PreparedRuntimeQuoteContext<'_>,
        pack_content_id: &str,
        build: F,
        map_poisoned_lock: M,
        map_capacity_error: C,
    ) -> Result<PreparedRuntimeHandle<T>, E>
    where
        F: FnOnce() -> Result<T, E>,
        M: Fn() -> E,
        C: Fn(SystemMemoryOwnerError) -> E,
    {
        let quote =
            T::system_memory_quote(quote_context, pack_content_id).map_err(&map_capacity_error)?;
        Self::allocate_once(
            pack_content_id,
            quote,
            build,
            &map_poisoned_lock,
            &map_capacity_error,
            " cache=skip_uncacheable_content_id",
        )
    }

    fn allocate_once<E, F, M, C>(
        pack_content_id: &str,
        quote: SystemMemoryAllocationQuote,
        build: F,
        map_poisoned_lock: &M,
        map_capacity_error: &C,
        log_suffix: &str,
    ) -> Result<PreparedRuntimeHandle<T>, E>
    where
        F: FnOnce() -> Result<T, E>,
        M: Fn() -> E,
        C: Fn(SystemMemoryOwnerError) -> E,
    {
        let load_started = Instant::now();
        match panic::catch_unwind(AssertUnwindSafe(|| {
            SystemMemoryOwner::try_allocate_transaction(quote, || {
                let prepared = build().map_err(PreparedRuntimeAllocationError::Build)?;
                let retained_bytes = prepared
                    .retained_system_memory_bytes()
                    .map_err(PreparedRuntimeAllocationError::Measure)?;
                Ok(SystemMemoryAllocationOutcome::new(
                    prepared,
                    retained_bytes,
                    retained_bytes,
                ))
            })
        })) {
            Ok(result) => {
                let owner = match result {
                    Ok(owner) => owner,
                    Err(SystemMemoryAllocationTransactionError::Allocation(
                        PreparedRuntimeAllocationError::Build(error),
                    )) => return Err(error),
                    Err(SystemMemoryAllocationTransactionError::Allocation(
                        PreparedRuntimeAllocationError::Measure(reason),
                    )) => {
                        return Err(map_capacity_error(
                            SystemMemoryOwnerError::capacity_failure(
                                "prepared_runtime_measure",
                                reason,
                            ),
                        ));
                    }
                    Err(SystemMemoryAllocationTransactionError::Capacity(error)) => {
                        return Err(map_capacity_error(error));
                    }
                };
                let prepared = Arc::new(owner);
                stage_timing::log_event(
                    "model_pack_load",
                    format_args!(
                        "pack_content_id={} duration_ms={:.3} admitted_requested_bytes={}{}",
                        pack_content_id,
                        load_started.elapsed().as_secs_f64() * 1000.0,
                        prepared.committed_requested_bytes(),
                        log_suffix,
                    ),
                );
                Ok(prepared)
            }
            Err(panic_payload) => {
                let payload_kind = if panic_payload.is::<&str>() || panic_payload.is::<String>() {
                    "string"
                } else {
                    "non-string"
                };
                stage_timing::log_event(
                    "model_pack_load_panicked",
                    format_args!(
                        "pack_content_id={} duration_ms={:.3} panic_payload={}{}",
                        pack_content_id,
                        load_started.elapsed().as_secs_f64() * 1000.0,
                        payload_kind,
                        log_suffix,
                    ),
                );
                Err(map_poisoned_lock())
            }
        }
    }

    /// Drops every cached prepared runtime, releasing the `Arc<T>` this cache
    /// holds. If nothing else is currently borrowing an entry (no in-flight
    /// request holding its own clone), this frees whatever native resources
    /// `T` owns -- mmap, materialized tensors, Metal/CPU graph context -- right
    /// away; otherwise the last outstanding clone's drop frees it once that
    /// request finishes. Used by the idle-unload reaper: a poisoned lock is
    /// swallowed (best-effort eviction, not a request-path operation) rather
    /// than propagated, since a subsequent `get_or_try_insert_with` will just
    /// rebuild on the next real request either way.
    ///
    /// This drops the per-content slots wholesale rather than resetting each
    /// slot's inner value to `None`: any build that is still in flight for a
    /// slot at the moment `clear()` runs holds its own `Arc` clone of that
    /// slot (taken before `clear()` could remove it from the map), so it still
    /// completes and populates the slot it is holding -- that slot is just no
    /// longer reachable from the map, so the next `get_or_try_insert_with` call
    /// for that content id creates a fresh slot and rebuilds, which is the same
    /// "pay the cold cost again" contract `clear()` has always documented.
    pub(crate) fn clear(&self) {
        self.admitted_by_content_id.clear();
    }

    /// Evicts exactly the slot for `pack_content_id`, leaving every other
    /// content identity's cached entry untouched. This is the "no global
    /// invalidation" eviction primitive: a pack install/replace only ever
    /// needs to drop the *old* content id's now-orphaned entry, never every
    /// resident entry in the cache (that used to be the audited bug -- see
    /// `runtime_cache_coordinator`'s module doc comment).
    pub(crate) fn evict_content_id(&self, pack_content_id: &str) {
        self.admitted_by_content_id
            .evict(&pack_content_id.to_string());
    }

    #[cfg(test)]
    fn len_for_test(&self) -> usize {
        self.admitted_by_content_id.usage_for_test().0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::path::PathBuf;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct StubRuntime {
        value: usize,
    }

    impl SystemMemoryMaterialization for StubRuntime {
        fn retained_system_memory_bytes(&self) -> Result<u64, String> {
            Ok(0)
        }
    }

    impl HostNeutralPreparedRuntime for StubRuntime {
        fn system_memory_quote(
            _context: PreparedRuntimeQuoteContext<'_>,
            pack_content_id: &str,
        ) -> Result<SystemMemoryAllocationQuote, SystemMemoryOwnerError> {
            SystemMemoryAllocationQuote::new(
                format!("prepared-runtime:test:{pack_content_id}"),
                64,
                64,
            )
        }
    }

    fn test_quote_context() -> PreparedRuntimeQuoteContext<'static> {
        static METADATA: std::sync::LazyLock<GgufMetadata> =
            std::sync::LazyLock::new(GgufMetadata::default);
        static TENSOR_INDEX: std::sync::LazyLock<GgufTensorIndex> =
            std::sync::LazyLock::new(|| {
                GgufTensorIndex::empty_for_test(PathBuf::from("stub.gguf"))
            });
        PreparedRuntimeQuoteContext {
            model_architecture: "test-stub",
            metadata: &METADATA,
            tensor_index: &TENSOR_INDEX,
            backend: GgmlCpuGraphBackend::Cpu,
        }
    }

    fn map_capacity_error(_error: SystemMemoryOwnerError) -> &'static str {
        "capacity"
    }

    /// Writes a minimal valid GGUF-magic fixture (`get_or_try_insert_with`
    /// now takes a `GgmlRuntimeSource`, which only ever admits GGUF-magic
    /// files) and returns its path.
    fn write_pack(dir: &tempfile::TempDir, name: &str, payload: &[u8]) -> PathBuf {
        let path = dir.path().join(name);
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(payload);
        std::fs::write(&path, bytes).expect("write pack");
        path
    }

    /// Every real caller of `get_or_try_insert_with` already holds a
    /// `GgmlRuntimeSource` (from a preflight); tests simulate that by
    /// validating fresh, exactly like a new request would.
    fn source_for(path: &std::path::Path) -> GgmlRuntimeSource {
        crate::validate_ggml_runtime_source_path(path).expect("validate runtime source")
    }

    #[test]
    fn reuses_cached_runtime_for_same_content() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_pack(&dir, "runtime.oasr", b"same-content");
        let cache = PreparedRuntimeCache::<StubRuntime>::default();

        let runtime_a = cache
            .get_or_try_insert_with(
                &source_for(&path),
                test_quote_context(),
                || Ok::<_, &'static str>(StubRuntime { value: 7 }),
                || "poisoned",
                map_capacity_error,
            )
            .expect("runtime a");
        let runtime_b = cache
            .get_or_try_insert_with(
                &source_for(&path),
                test_quote_context(),
                || Ok::<_, &'static str>(StubRuntime { value: 9 }),
                || "poisoned",
                map_capacity_error,
            )
            .expect("runtime b");

        assert!(Arc::ptr_eq(&runtime_a, &runtime_b));
        assert_eq!(runtime_b.value, 7);
        assert_eq!(cache.len_for_test(), 1);
    }

    #[test]
    fn reuses_cached_runtime_for_canonical_equivalent_paths() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = write_pack(&temp, "runtime.gguf", b"canonical-bytes");
        let dotted_path = temp.path().join(".").join("runtime.gguf");
        let cache = PreparedRuntimeCache::<StubRuntime>::default();
        let build_count = Cell::new(0usize);

        let runtime_a = cache
            .get_or_try_insert_with(
                &source_for(&dotted_path),
                test_quote_context(),
                || {
                    build_count.set(build_count.get() + 1);
                    Ok::<_, &'static str>(StubRuntime { value: 7 })
                },
                || "poisoned",
                map_capacity_error,
            )
            .expect("runtime a");
        let runtime_b = cache
            .get_or_try_insert_with(
                &source_for(&runtime_path),
                test_quote_context(),
                || {
                    build_count.set(build_count.get() + 1);
                    Ok::<_, &'static str>(StubRuntime { value: 9 })
                },
                || "poisoned",
                map_capacity_error,
            )
            .expect("runtime b");

        assert_eq!(build_count.get(), 1);
        assert!(Arc::ptr_eq(&runtime_a, &runtime_b));
        assert_eq!(runtime_b.value, 7);
    }

    #[test]
    fn same_path_byte_replacement_misses_cached_runtime() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_pack(&dir, "same-path.oasr", b"content-a-bytes");
        let cache = PreparedRuntimeCache::<StubRuntime>::default();
        let build_count = Cell::new(0usize);

        let runtime_a = cache
            .get_or_try_insert_with(
                &source_for(&path),
                test_quote_context(),
                || {
                    build_count.set(build_count.get() + 1);
                    Ok::<_, &'static str>(StubRuntime { value: 1 })
                },
                || "poisoned",
                map_capacity_error,
            )
            .expect("runtime a");
        assert_eq!(build_count.get(), 1);
        assert_eq!(cache.len_for_test(), 1);

        write_pack(&dir, "same-path.oasr", b"content-b-bytes-different");
        let runtime_b = cache
            .get_or_try_insert_with(
                &source_for(&path),
                test_quote_context(),
                || {
                    build_count.set(build_count.get() + 1);
                    Ok::<_, &'static str>(StubRuntime { value: 2 })
                },
                || "poisoned",
                map_capacity_error,
            )
            .expect("runtime b");

        assert_eq!(
            build_count.get(),
            2,
            "same path with different pack bytes must rebuild"
        );
        assert!(!Arc::ptr_eq(&runtime_a, &runtime_b));
        assert_eq!(runtime_b.value, 2);
        // Both content identities may remain until clear; that is intentional --
        // content A is still valid if referenced elsewhere.
        assert!(cache.len_for_test() >= 1);
    }

    /// Same bytes, two lookups (each with its own freshly-validated source,
    /// exactly like two separate requests): exactly one build (the
    /// warm-path hit), no generation/epoch anywhere in this cache to force a
    /// spurious rebuild.
    #[test]
    fn same_content_id_hits_across_repeated_lookups() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_pack(&dir, "stable.oasr", b"stable-bytes");
        let cache = PreparedRuntimeCache::<StubRuntime>::default();
        let build_count = Cell::new(0usize);

        let runtime_a = cache
            .get_or_try_insert_with(
                &source_for(&path),
                test_quote_context(),
                || {
                    build_count.set(build_count.get() + 1);
                    Ok::<_, &'static str>(StubRuntime { value: 1 })
                },
                || "poisoned",
                map_capacity_error,
            )
            .expect("runtime a");
        let runtime_b = cache
            .get_or_try_insert_with(
                &source_for(&path),
                test_quote_context(),
                || {
                    build_count.set(build_count.get() + 1);
                    Ok::<_, &'static str>(StubRuntime { value: 2 })
                },
                || "poisoned",
                map_capacity_error,
            )
            .expect("runtime b");

        assert_eq!(
            build_count.get(),
            1,
            "unchanged bytes must hit, not rebuild"
        );
        assert!(Arc::ptr_eq(&runtime_a, &runtime_b));
        assert_eq!(runtime_b.value, 1);
    }

    /// No global invalidation: evicting one pack's content id must not
    /// disturb a resident entry for a *different* pack in the same cache.
    /// Direct regression test for the audited bug -- a shared epoch baked
    /// into the cache used to invalidate every resident content identity at
    /// once (see `runtime_cache_coordinator`'s module doc comment).
    #[test]
    fn evict_content_id_leaves_a_different_pack_resident() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path_a = write_pack(&dir, "pack-a.oasr", b"pack-a-bytes");
        let path_b = write_pack(&dir, "pack-b.oasr", b"pack-b-different-bytes");
        let cache = PreparedRuntimeCache::<StubRuntime>::default();
        let build_count = Cell::new(0usize);

        let build = |value: usize| {
            build_count.set(build_count.get() + 1);
            Ok::<_, &'static str>(StubRuntime { value })
        };

        let runtime_a = cache
            .get_or_try_insert_with(
                &source_for(&path_a),
                test_quote_context(),
                || build(1),
                || "poisoned",
                map_capacity_error,
            )
            .expect("runtime a");
        let runtime_b = cache
            .get_or_try_insert_with(
                &source_for(&path_b),
                test_quote_context(),
                || build(2),
                || "poisoned",
                map_capacity_error,
            )
            .expect("runtime b");
        assert_eq!(build_count.get(), 2);
        assert_eq!(cache.len_for_test(), 2);

        let content_id_a = source_for(&path_a).content_id().to_string();
        cache.evict_content_id(&content_id_a);
        assert_eq!(cache.len_for_test(), 1, "only pack a's slot must be gone");

        // Pack a rebuilds (its slot was evicted); pack b is untouched --
        // still the exact same Arc, zero extra builds.
        let runtime_a_rebuilt = cache
            .get_or_try_insert_with(
                &source_for(&path_a),
                test_quote_context(),
                || build(3),
                || "poisoned",
                map_capacity_error,
            )
            .expect("runtime a rebuilt");
        let runtime_b_again = cache
            .get_or_try_insert_with(
                &source_for(&path_b),
                test_quote_context(),
                || build(4),
                || "poisoned",
                map_capacity_error,
            )
            .expect("runtime b again");

        assert_eq!(build_count.get(), 3, "only the evicted pack rebuilds");
        assert!(!Arc::ptr_eq(&runtime_a, &runtime_a_rebuilt));
        assert!(
            Arc::ptr_eq(&runtime_b, &runtime_b_again),
            "the untouched pack must still be the same cached Arc"
        );
    }

    #[test]
    fn clear_evicts_cached_entry_so_the_next_call_rebuilds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_pack(&dir, "clear.oasr", b"clear-bytes");
        let cache = PreparedRuntimeCache::<StubRuntime>::default();
        let build_count = Cell::new(0usize);

        let build = |value: usize| {
            build_count.set(build_count.get() + 1);
            Ok::<_, &'static str>(StubRuntime { value })
        };

        let runtime_a = cache
            .get_or_try_insert_with(
                &source_for(&path),
                test_quote_context(),
                || build(7),
                || "poisoned",
                map_capacity_error,
            )
            .expect("runtime a");
        assert_eq!(build_count.get(), 1);

        cache.clear();

        let runtime_b = cache
            .get_or_try_insert_with(
                &source_for(&path),
                test_quote_context(),
                || build(9),
                || "poisoned",
                map_capacity_error,
            )
            .expect("runtime b");

        assert_eq!(build_count.get(), 2, "clear must force a rebuild");
        assert!(!Arc::ptr_eq(&runtime_a, &runtime_b));
        assert_eq!(runtime_b.value, 9);
    }

    #[test]
    fn eviction_keeps_the_lease_until_the_last_in_flight_handle_drops() {
        use crate::device::execution_memory::MemoryDomainKey;
        use crate::models::native_execution_services::{
            NativeExecutionServices, install_native_execution_services,
        };

        let services = NativeExecutionServices::for_local_process().expect("native services");
        let broker = Arc::clone(services.memory_broker());
        let _scope = install_native_execution_services(&services);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_pack(&dir, "leased.oasr", b"leased-content");
        let source = source_for(&path);
        let content_id = source.content_id().to_string();
        let cache = PreparedRuntimeCache::<StubRuntime>::default();

        let in_flight = cache
            .get_or_try_insert_with(
                &source,
                test_quote_context(),
                || Ok::<_, &'static str>(StubRuntime { value: 7 }),
                || "poisoned",
                map_capacity_error,
            )
            .expect("leased runtime");
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            64
        );

        cache.evict_content_id(&content_id);
        assert_eq!(cache.len_for_test(), 0);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            64,
            "eviction must not refund an in-flight owner's lease"
        );

        drop(in_flight);
        assert_eq!(
            broker.usage(&MemoryDomainKey::SystemMemory).committed_bytes,
            0
        );
    }

    /// Proves the single-flight fix (see `get_or_try_insert_with`): two
    /// threads racing a cold miss on the *same* content identity must not
    /// both run `build()`. This used to need a retry loop to absorb a
    /// parallel test bumping the process-global epoch mid-race; with no
    /// epoch left in this cache at all, the race is now deterministic.
    #[test]
    fn concurrent_cold_miss_on_the_same_content_builds_exactly_once() {
        use std::sync::Barrier;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_pack(&dir, "concurrent.oasr", b"concurrent-bytes");

        let cache = Arc::new(PreparedRuntimeCache::<StubRuntime>::default());
        let build_count = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(2));

        let spawn_racer = |value: usize| {
            let cache = Arc::clone(&cache);
            let build_count = Arc::clone(&build_count);
            let barrier = Arc::clone(&barrier);
            let path = path.clone();
            thread::spawn(move || {
                let source = source_for(&path);
                barrier.wait();
                cache
                    .get_or_try_insert_with(
                        &source,
                        test_quote_context(),
                        || {
                            build_count.fetch_add(1, Ordering::SeqCst);
                            thread::sleep(std::time::Duration::from_millis(30));
                            Ok::<_, &'static str>(StubRuntime { value })
                        },
                        || "poisoned",
                        map_capacity_error,
                    )
                    .expect("runtime")
            })
        };

        let racer_a = spawn_racer(1);
        let racer_b = spawn_racer(2);
        let runtime_a = racer_a.join().expect("racer a joined");
        let runtime_b = racer_b.join().expect("racer b joined");

        assert_eq!(
            build_count.load(Ordering::SeqCst),
            1,
            "single build must be shared"
        );
        assert!(Arc::ptr_eq(&runtime_a, &runtime_b));
    }

    /// Proves a `build()` panic does not poison the slot `Mutex` for the next
    /// caller on the same content id.
    #[test]
    fn build_panic_does_not_poison_the_slot_for_the_next_caller() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_pack(&dir, "panic.oasr", b"panic-bytes");
        let cache = PreparedRuntimeCache::<StubRuntime>::default();

        let previous_hook = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let first_result = cache.get_or_try_insert_with(
            &source_for(&path),
            test_quote_context(),
            || -> Result<StubRuntime, &'static str> { panic!("simulated build panic") },
            || "poisoned",
            map_capacity_error,
        );
        panic::set_hook(previous_hook);

        assert!(
            matches!(first_result, Err("poisoned")),
            "a build() panic must be caught and mapped through map_poisoned_lock, not left \
             to unwind out of get_or_try_insert_with"
        );

        let second_result = cache
            .get_or_try_insert_with(
                &source_for(&path),
                test_quote_context(),
                || Ok::<_, &'static str>(StubRuntime { value: 42 }),
                || "poisoned",
                map_capacity_error,
            )
            .expect("build must succeed cleanly on retry after a prior build panic");
        assert_eq!(second_result.value, 42);
    }

    /// Proves `clear()` cannot be "undone" by a build that was already in
    /// flight when it ran: the in-flight winner still completes normally, but
    /// its result is orphaned once `clear()` removes the slot from the map.
    #[test]
    fn clear_during_in_flight_build_does_not_resurrect_the_evicted_slot() {
        use std::sync::Barrier;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_pack(&dir, "clear-in-flight.oasr", b"clear-in-flight-bytes");
        let cache = Arc::new(PreparedRuntimeCache::<StubRuntime>::default());
        let build_count = Arc::new(AtomicUsize::new(0));
        let builder_in_build = Arc::new(Barrier::new(2));

        let builder =
            {
                let cache = Arc::clone(&cache);
                let build_count = Arc::clone(&build_count);
                let barrier = Arc::clone(&builder_in_build);
                let path = path.clone();
                thread::spawn(move || {
                    let source = source_for(&path);
                    cache
                    .get_or_try_insert_with(
                        &source, test_quote_context(),
                        || {
                            build_count.fetch_add(1, Ordering::SeqCst);
                            barrier.wait();
                            thread::sleep(std::time::Duration::from_millis(50));
                            Ok::<_, &'static str>(StubRuntime { value: 1 })
                        },
                        || "poisoned", map_capacity_error,
                    )
                    .expect(
                        "in-flight build must still complete normally despite a concurrent clear()",
                    )
                })
            };

        builder_in_build.wait();
        cache.clear();

        let winner_runtime = builder.join().expect("builder thread joined");
        assert_eq!(build_count.load(Ordering::SeqCst), 1);

        let rebuilt_runtime = cache
            .get_or_try_insert_with(
                &source_for(&path),
                test_quote_context(),
                || {
                    build_count.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, &'static str>(StubRuntime { value: 2 })
                },
                || "poisoned",
                map_capacity_error,
            )
            .expect("rebuild after clear must succeed");

        assert_eq!(
            build_count.load(Ordering::SeqCst),
            2,
            "clear() during an in-flight build must force the next caller to rebuild, not \
             reuse the orphaned slot"
        );
        assert!(
            !Arc::ptr_eq(&winner_runtime, &rebuilt_runtime),
            "the post-clear rebuild must be a distinct Arc from the orphaned in-flight build's result"
        );
        assert_eq!(rebuilt_runtime.value, 2);
    }
}
