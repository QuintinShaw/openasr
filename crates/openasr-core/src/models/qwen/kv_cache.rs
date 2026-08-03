use std::ops::{Deref, DerefMut};

use crate::ggml_runtime::{
    GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuTensor, GgmlKvElementType,
};
use crate::models::system_memory_owner::{
    SystemMemoryAllocationOutcome, SystemMemoryAllocationQuote, SystemMemoryOwner,
};

fn host_kv_allocation_failure(detail: String) -> String {
    crate::models::native_execution_services::record_current_execution_candidate_failure(
        crate::device::execution_policy::ExecutionCandidateFailure::capacity(
            "decoder_host_state_allocate",
            detail.clone(),
        ),
    );
    detail
}

/// Validated logical and resident position spans for one Qwen-shaped causal
/// decoder invocation.
///
/// The two numbers are intentionally carried as one typed value so a caller
/// cannot accidentally size host history to the session reserve, or size a
/// reusable device graph to the current chunk. `logical_positions` is the
/// exact physical greedy-write span for this invocation: `P + G - 1` for a
/// non-empty generation budget, because the final sampled token is returned
/// rather than fed back into the decoder. The semantic `P + G` context limit is
/// validated separately. `resident_positions` is the stable physical-write
/// envelope shared by every legal chunk in the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Qwen3AsrKvCacheCapacity {
    logical_positions: usize,
    resident_positions: usize,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub(crate) enum Qwen3AsrKvCacheCapacityError {
    #[error("causal decoder request carries no planned persistent state")]
    DecoderStateNotPlanned,
    #[error("causal decoder state axis '{state}' is invalid: {source}")]
    InvalidStateAxis {
        state: &'static str,
        #[source]
        source: crate::capacity::topology::TopologyError,
    },
    #[error("causal decoder logical KV span must be positive")]
    ZeroLogicalPositions,
    #[error(
        "causal decoder resident KV span {resident_positions} does not cover logical span {logical_positions}"
    )]
    ResidentDoesNotCoverLogical {
        logical_positions: usize,
        resident_positions: usize,
    },
    #[error(
        "causal decoder runtime measured {measured_positions} KV positions, but the planner proved {planned_positions}"
    )]
    LogicalPositionMismatch {
        planned_positions: usize,
        measured_positions: usize,
    },
}

impl Qwen3AsrKvCacheCapacity {
    pub(crate) fn from_decoder_state(
        state: &crate::models::ggml_asr_executor::GgmlAsrDecoderState,
        state_id: &'static str,
    ) -> Result<Self, Qwen3AsrKvCacheCapacityError> {
        let crate::models::ggml_asr_executor::GgmlAsrDecoderState::Planned(plan) = state else {
            return Err(Qwen3AsrKvCacheCapacityError::DecoderStateNotPlanned);
        };
        let axis = plan
            .position_axis(
                state_id,
                crate::capacity::topology::StateKind::SelfAttentionKv,
            )
            .map_err(|source| Qwen3AsrKvCacheCapacityError::InvalidStateAxis {
                state: state_id,
                source,
            })?;
        Self::new(axis.logical_positions, axis.resident_positions)
    }

    pub(crate) fn new(
        logical_positions: usize,
        resident_positions: usize,
    ) -> Result<Self, Qwen3AsrKvCacheCapacityError> {
        if logical_positions == 0 {
            return Err(Qwen3AsrKvCacheCapacityError::ZeroLogicalPositions);
        }
        if resident_positions < logical_positions {
            return Err(Qwen3AsrKvCacheCapacityError::ResidentDoesNotCoverLogical {
                logical_positions,
                resident_positions,
            });
        }
        Ok(Self {
            logical_positions,
            resident_positions,
        })
    }

    pub(crate) const fn logical_positions(self) -> usize {
        self.logical_positions
    }

    pub(crate) const fn resident_positions(self) -> usize {
        self.resident_positions
    }

    /// Cross-check the planner against the real prompt and generation budget
    /// materialized by the executor. The planner is never allowed to silently
    /// replace runtime semantics, and the runtime is never allowed to fall
    /// back to a historical constant when the two drift.
    pub(crate) fn validate_measured_logical_positions(
        self,
        measured_positions: usize,
    ) -> Result<Self, Qwen3AsrKvCacheCapacityError> {
        if measured_positions != self.logical_positions {
            return Err(Qwen3AsrKvCacheCapacityError::LogicalPositionMismatch {
                planned_positions: self.logical_positions,
                measured_positions,
            });
        }
        Ok(self)
    }
}

/// Per-layer host KV cache shared by every Qwen-shaped decoder family.
///
/// Default storage is f32 (byte-identical to the historical path). Opt-in
/// `q8_0` stores native ggml q8_0 rows so host and resident paths share the
/// same packed layout without a full f32 staging buffer.
#[derive(Debug)]
pub(crate) struct Qwen3AsrLayerKvCacheState {
    max_positions: usize,
    kv_heads: usize,
    head_dim: usize,
    element_type: GgmlKvElementType,
    keys: HostKvStorage,
    values: HostKvStorage,
    written_positions: usize,
}

#[derive(Debug)]
enum HostKvStorage {
    F32(Vec<f32>),
    Q8(Vec<u8>),
}

impl HostKvStorage {
    fn empty(element_type: GgmlKvElementType) -> Self {
        match element_type {
            GgmlKvElementType::F32 => Self::F32(Vec::new()),
            GgmlKvElementType::Q8_0 => Self::Q8(Vec::new()),
            GgmlKvElementType::F16 => {
                // Host path never uses f16; resident path owns f16.
                Self::F32(Vec::new())
            }
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::F32(v) => v.is_empty(),
            Self::Q8(v) => v.is_empty(),
        }
    }

    fn f32_slice(&self) -> Result<&[f32], String> {
        match self {
            Self::F32(v) => Ok(v.as_slice()),
            Self::Q8(_) => Err("host kv storage is q8_0, not f32".to_string()),
        }
    }

    fn q8_slice(&self) -> Result<&[u8], String> {
        match self {
            Self::Q8(v) => Ok(v.as_slice()),
            Self::F32(_) => Err("host kv storage is f32, not q8_0".to_string()),
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen3AsrLayerKvCacheSnapshot {
    pub written_positions: usize,
    pub key_width: usize,
    pub value_width: usize,
    pub element_type: GgmlKvElementType,
}

pub(crate) struct Qwen3AsrLayerKvCacheHistory<'a> {
    pub max_positions: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub written_positions: usize,
    #[allow(dead_code)]
    pub element_type: GgmlKvElementType,
    pub keys_f32: Option<&'a [f32]>,
    pub values_f32: Option<&'a [f32]>,
    pub keys_q8: Option<&'a [u8]>,
    pub values_q8: Option<&'a [u8]>,
}

/// Whether one Qwen-shaped execution route owns a Rust host KV payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen3AsrHostKvMode {
    /// CPU/growing-graph and serve-batch refill paths read and update host KV.
    Materialized,
    /// Single-sequence persistent graph reuse keeps K/V entirely resident.
    ResidentOnly,
}

/// One transactionally-admitted host-KV owner for every layer of a
/// Qwen-shaped decoder invocation.
///
/// A single owner/lease covers the complete batch of K/V Vec allocations; no
/// layer can commit independently and leave a partially-accounted decoder.
#[derive(Debug)]
pub(crate) struct Qwen3AsrHostKvCacheOwner(SystemMemoryOwner<Vec<Qwen3AsrLayerKvCacheState>>);

impl Qwen3AsrHostKvCacheOwner {
    pub(crate) const fn empty() -> Self {
        Self(SystemMemoryOwner::without_allocation(Vec::new()))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new(
        resource_id: &'static str,
        layer_count: usize,
        capacity: Qwen3AsrKvCacheCapacity,
        kv_heads: usize,
        head_dim: usize,
        element_type: GgmlKvElementType,
        mode: Qwen3AsrHostKvMode,
    ) -> Result<Self, String> {
        Self::try_new_with_ownership(
            resource_id,
            layer_count,
            capacity,
            kv_heads,
            head_dim,
            element_type,
            mode,
            HostKvLeaseOwnership::Standalone,
        )
    }

    /// Builds materialized host KV inside an already-provisional parent
    /// [`SystemMemoryOwner`] transaction. The returned value owns the buffers
    /// but deliberately has no nested lease; the parent outcome must measure
    /// [`Self::retained_system_memory_bytes`] and bind the one candidate lease
    /// to the aggregate runtime + every session arena.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new_inside_parent_transaction(
        resource_id: &'static str,
        layer_count: usize,
        capacity: Qwen3AsrKvCacheCapacity,
        kv_heads: usize,
        head_dim: usize,
        element_type: GgmlKvElementType,
        mode: Qwen3AsrHostKvMode,
    ) -> Result<Self, String> {
        Self::try_new_with_ownership(
            resource_id,
            layer_count,
            capacity,
            kv_heads,
            head_dim,
            element_type,
            mode,
            HostKvLeaseOwnership::ParentTransaction,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn try_new_with_ownership(
        resource_id: &'static str,
        layer_count: usize,
        capacity: Qwen3AsrKvCacheCapacity,
        kv_heads: usize,
        head_dim: usize,
        element_type: GgmlKvElementType,
        mode: Qwen3AsrHostKvMode,
        ownership: HostKvLeaseOwnership,
    ) -> Result<Self, String> {
        let logical_positions = capacity.logical_positions();
        match mode {
            Qwen3AsrHostKvMode::ResidentOnly => {
                let layers = build_empty_layers(
                    layer_count,
                    logical_positions,
                    kv_heads,
                    head_dim,
                    element_type,
                )?;
                Ok(Self(SystemMemoryOwner::without_allocation(layers)))
            }
            Qwen3AsrHostKvMode::Materialized => {
                let quoted_bytes = qwen_host_kv_quoted_bytes(
                    layer_count,
                    logical_positions,
                    kv_heads,
                    head_dim,
                    element_type,
                )?;
                let allocate_layers =
                    || -> Result<(Vec<Qwen3AsrLayerKvCacheState>, u64), String> {
                        let mut layers = build_empty_layers(
                            layer_count,
                            logical_positions,
                            kv_heads,
                            head_dim,
                            element_type,
                        )?;
                        for layer in &mut layers {
                            layer.materialize_storage_for_owner()?;
                        }
                        let actual_bytes =
                            qwen_host_kv_actual_capacity_bytes(&layers, layers.capacity())?;
                        Ok((layers, actual_bytes))
                    };
                match ownership {
                    HostKvLeaseOwnership::Standalone => {
                        let quote = SystemMemoryAllocationQuote::new(
                            resource_id,
                            quoted_bytes,
                            quoted_bytes,
                        )
                        .map_err(|error| error.to_string())?;
                        let owner = SystemMemoryOwner::try_allocate(quote, || {
                            let (layers, actual_bytes) = allocate_layers()?;
                            Ok(SystemMemoryAllocationOutcome::new(
                                layers,
                                actual_bytes,
                                actual_bytes,
                            ))
                        })
                        .map_err(|error| error.to_string())?;
                        Ok(Self(owner))
                    }
                    HostKvLeaseOwnership::ParentTransaction => {
                        let (layers, _actual_bytes) = allocate_layers()?;
                        Ok(Self(SystemMemoryOwner::without_allocation(layers)))
                    }
                }
            }
        }
    }

    #[allow(dead_code)] // Reconciled by aggregate owners such as Hy-MT2.
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        qwen_host_kv_actual_capacity_bytes(&self.0, self.0.capacity())
    }

    /// Allocate a second admitted owner and copy one written prefix from an
    /// existing owner. The source remains charged while the destination is
    /// pending/committed, so the broker accounts for the real fork peak.
    #[cfg(test)]
    pub(crate) fn try_fork_prefix(
        resource_id: &'static str,
        source: &Self,
        written_positions: usize,
        max_positions: usize,
    ) -> Result<Self, String> {
        let first = source
            .first()
            .ok_or_else(|| "qwen-shaped host KV fork requires at least one layer".to_string())?;
        if source.iter().any(|layer| {
            layer.kv_heads != first.kv_heads
                || layer.head_dim != first.head_dim
                || layer.element_type != first.element_type
                || layer.written_positions < written_positions
        }) {
            return Err("qwen-shaped host KV fork source geometry/prefix mismatch".to_string());
        }
        if max_positions == 0 {
            return Err("qwen-shaped host KV fork max_positions must be positive".to_string());
        }
        let quoted_bytes = qwen_host_kv_quoted_bytes(
            source.len(),
            max_positions,
            first.kv_heads,
            first.head_dim,
            first.element_type,
        )?;
        let quote = SystemMemoryAllocationQuote::new(resource_id, quoted_bytes, quoted_bytes)
            .map_err(|error| error.to_string())?;
        let owner = SystemMemoryOwner::try_allocate(quote, || {
            let mut layers = Vec::new();
            layers.try_reserve_exact(source.len()).map_err(|error| {
                host_kv_allocation_failure(format!(
                    "qwen-shaped host KV fork layer table allocation failed: {error}"
                ))
            })?;
            for layer in source.iter() {
                layers.push(layer.fork_prefix_for_owner(written_positions, max_positions)?);
            }
            let actual_bytes = qwen_host_kv_actual_capacity_bytes(&layers, layers.capacity())?;
            Ok(SystemMemoryAllocationOutcome::new(
                layers,
                actual_bytes,
                actual_bytes,
            ))
        })
        .map_err(|error| error.to_string())?;
        Ok(Self(owner))
    }

    /// Copy a written prefix between two already-admitted, materialized owners
    /// without allocating. Long-lived session scratch can therefore reuse its
    /// stable envelope while a separate prefix-cache owner remains intact.
    pub(crate) fn replace_prefix_from(
        &mut self,
        source: &Self,
        written_positions: usize,
    ) -> Result<(), String> {
        if self.len() != source.len() || self.is_empty() {
            return Err(format!(
                "qwen-shaped host KV prefix copy layer mismatch: destination={} source={}",
                self.len(),
                source.len()
            ));
        }
        for (destination, source) in self.iter().zip(source.iter()) {
            destination.validate_prefix_copy_from(source, written_positions)?;
        }
        for (destination, source) in self.iter_mut().zip(source.iter()) {
            destination.copy_prefix_from_for_owner(source, written_positions)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn has_materialized_payload(&self) -> bool {
        self.iter()
            .all(Qwen3AsrLayerKvCacheState::has_materialized_storage)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostKvLeaseOwnership {
    Standalone,
    ParentTransaction,
}

impl Deref for Qwen3AsrHostKvCacheOwner {
    type Target = Vec<Qwen3AsrLayerKvCacheState>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Qwen3AsrHostKvCacheOwner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

fn build_empty_layers(
    layer_count: usize,
    max_positions: usize,
    kv_heads: usize,
    head_dim: usize,
    element_type: GgmlKvElementType,
) -> Result<Vec<Qwen3AsrLayerKvCacheState>, String> {
    let mut layers = Vec::new();
    layers.try_reserve_exact(layer_count).map_err(|error| {
        host_kv_allocation_failure(format!(
            "qwen-shaped host KV layer table allocation failed for {layer_count} layers: {error}"
        ))
    })?;
    for _ in 0..layer_count {
        layers.push(Qwen3AsrLayerKvCacheState::new_with_element_type(
            max_positions,
            kv_heads,
            head_dim,
            element_type,
        )?);
    }
    Ok(layers)
}

pub(crate) fn qwen_host_kv_quoted_bytes(
    layer_count: usize,
    max_positions: usize,
    kv_heads: usize,
    head_dim: usize,
    element_type: GgmlKvElementType,
) -> Result<u64, String> {
    let row_bytes = element_type.row_nbytes(head_dim)?;
    let payload = layer_count
        .checked_mul(2)
        .and_then(|value| value.checked_mul(max_positions))
        .and_then(|value| value.checked_mul(kv_heads))
        .and_then(|value| value.checked_mul(row_bytes))
        .ok_or_else(|| "qwen-shaped host KV quoted payload bytes overflowed".to_string())?;
    let table = layer_count
        .checked_mul(std::mem::size_of::<Qwen3AsrLayerKvCacheState>())
        .ok_or_else(|| "qwen-shaped host KV layer table bytes overflowed".to_string())?;
    u64::try_from(
        payload
            .checked_add(table)
            .ok_or_else(|| "qwen-shaped host KV quoted total bytes overflowed".to_string())?,
    )
    .map_err(|_| "qwen-shaped host KV quoted bytes exceed u64".to_string())
}

fn qwen_host_kv_actual_capacity_bytes(
    layers: &[Qwen3AsrLayerKvCacheState],
    layer_table_capacity: usize,
) -> Result<u64, String> {
    let table = layer_table_capacity
        .checked_mul(std::mem::size_of::<Qwen3AsrLayerKvCacheState>())
        .ok_or_else(|| "qwen-shaped host KV actual layer table bytes overflowed".to_string())?;
    let mut bytes = u64::try_from(table)
        .map_err(|_| "qwen-shaped host KV actual table bytes exceed u64".to_string())?;
    for layer in layers {
        bytes = bytes
            .checked_add(layer.storage_capacity_bytes()?)
            .ok_or_else(|| "qwen-shaped host KV actual capacity bytes overflowed".to_string())?;
    }
    Ok(bytes)
}

impl Qwen3AsrLayerKvCacheState {
    /// Host-F32 convenience constructor for tests and F32-pinned harnesses.
    /// Production whole-decoder paths use [`Self::new_with_element_type`] with
    /// the resolved `kv_cache_spec().host` so Q8 and F32 stay aligned.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn new(max_positions: usize, kv_heads: usize, head_dim: usize) -> Self {
        Self::new_with_element_type(max_positions, kv_heads, head_dim, GgmlKvElementType::F32)
            .expect("f32 host KV geometry is always valid")
    }

    pub(crate) fn new_with_element_type(
        max_positions: usize,
        kv_heads: usize,
        head_dim: usize,
        element_type: GgmlKvElementType,
    ) -> Result<Self, String> {
        match element_type {
            GgmlKvElementType::F32 | GgmlKvElementType::Q8_0 => {}
            GgmlKvElementType::F16 => {
                return Err("host KV cache does not use f16 storage; use resident f16".to_string());
            }
        }
        element_type.validate_head_dim(head_dim)?;
        Ok(Self {
            max_positions,
            kv_heads,
            head_dim,
            element_type,
            keys: HostKvStorage::empty(element_type),
            values: HostKvStorage::empty(element_type),
            written_positions: 0,
        })
    }

    pub(crate) fn element_type(&self) -> GgmlKvElementType {
        self.element_type
    }

    fn materialize_storage_for_owner(&mut self) -> Result<(), String> {
        if self.keys.is_empty() {
            self.keys = self.allocate_storage()?;
        }
        if self.values.is_empty() {
            self.values = self.allocate_storage()?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn has_materialized_storage(&self) -> bool {
        !self.keys.is_empty() && !self.values.is_empty()
    }

    fn storage_capacity_bytes(&self) -> Result<u64, String> {
        fn bytes(storage: &HostKvStorage) -> Result<u64, String> {
            let bytes = match storage {
                HostKvStorage::F32(values) => {
                    values.capacity().checked_mul(std::mem::size_of::<f32>())
                }
                HostKvStorage::Q8(values) => Some(values.capacity()),
            }
            .ok_or_else(|| "qwen-shaped host KV Vec capacity bytes overflowed".to_string())?;
            u64::try_from(bytes)
                .map_err(|_| "qwen-shaped host KV Vec capacity exceeds u64".to_string())
        }
        bytes(&self.keys)?
            .checked_add(bytes(&self.values)?)
            .ok_or_else(|| "qwen-shaped host KV K/V capacity sum overflowed".to_string())
    }

    pub(crate) fn write(
        &mut self,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), String> {
        if key.is_empty() || value.is_empty() {
            return Err("qwen3-asr kv-cache write rejected empty key/value row".to_string());
        }
        if key.iter().any(|v| !v.is_finite()) || value.iter().any(|v| !v.is_finite()) {
            return Err("qwen3-asr kv-cache write rejected non-finite key/value row".to_string());
        }
        if position >= self.max_positions {
            return Err(format!(
                "qwen3-asr kv-cache write position {position} exceeds max_positions={}",
                self.max_positions
            ));
        }

        let key_width = self.key_width();
        let value_width = self.value_width();
        self.ensure_shape_initialized(key_width, value_width)?;
        if key.len() != key_width || value.len() != value_width {
            return Err(format!(
                "qwen3-asr kv-cache row width mismatch: key={} (expected {}), value={} (expected {})",
                key.len(),
                key_width,
                value.len(),
                value_width
            ));
        }

        match self.element_type {
            GgmlKvElementType::F32 => {
                let keys = match &mut self.keys {
                    HostKvStorage::F32(v) => v,
                    HostKvStorage::Q8(_) => unreachable!("f32 element type with q8 storage"),
                };
                let values = match &mut self.values {
                    HostKvStorage::F32(v) => v,
                    HostKvStorage::Q8(_) => unreachable!("f32 element type with q8 storage"),
                };
                Self::write_history_row_f32(
                    keys,
                    self.max_positions,
                    self.kv_heads,
                    self.head_dim,
                    position,
                    key,
                )?;
                Self::write_history_row_f32(
                    values,
                    self.max_positions,
                    self.kv_heads,
                    self.head_dim,
                    position,
                    value,
                )?;
            }
            GgmlKvElementType::Q8_0 => {
                let row_nbytes = self.element_type.row_nbytes(self.head_dim)?;
                let keys = match &mut self.keys {
                    HostKvStorage::Q8(v) => v,
                    HostKvStorage::F32(_) => unreachable!("q8 element type with f32 storage"),
                };
                let values = match &mut self.values {
                    HostKvStorage::Q8(v) => v,
                    HostKvStorage::F32(_) => unreachable!("q8 element type with f32 storage"),
                };
                Self::write_history_row_q8(
                    keys,
                    self.max_positions,
                    self.kv_heads,
                    self.head_dim,
                    row_nbytes,
                    position,
                    key,
                )?;
                Self::write_history_row_q8(
                    values,
                    self.max_positions,
                    self.kv_heads,
                    self.head_dim,
                    row_nbytes,
                    position,
                    value,
                )?;
            }
            GgmlKvElementType::F16 => unreachable!("host KV rejects f16"),
        }
        self.written_positions = self.written_positions.max(position.saturating_add(1));
        Ok(())
    }

    pub(crate) fn upload_history_prefix_to_graph<'a>(
        &self,
        graph: &mut GgmlCpuGraphBuilder<'a>,
        key_history: GgmlCpuTensor<'a>,
        value_history: GgmlCpuTensor<'a>,
        token_count: usize,
        key_tensor_name: &'static str,
        value_tensor_name: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        let kv_span = token_count
            .checked_add(1)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "qwen kv-cache upload total_tokens overflow",
            })?;
        self.upload_history_prefix_to_fixed_span_graph(
            graph,
            key_history,
            value_history,
            token_count,
            kv_span,
            key_tensor_name,
            value_tensor_name,
        )
    }

    pub(crate) fn upload_history_prefix_to_fixed_span_graph<'a>(
        &self,
        graph: &mut GgmlCpuGraphBuilder<'a>,
        key_history: GgmlCpuTensor<'a>,
        value_history: GgmlCpuTensor<'a>,
        token_count: usize,
        kv_span: usize,
        key_tensor_name: &'static str,
        value_tensor_name: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        if token_count == 0 {
            return Ok(());
        }
        if kv_span < token_count {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "qwen kv-cache upload fixed span smaller than prefix",
            });
        }
        if token_count > self.written_positions {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "qwen kv-cache upload requested unwritten prefix",
            });
        }

        match self.element_type {
            GgmlKvElementType::F32 => {
                let per_head_len = token_count.checked_mul(self.head_dim).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "qwen kv-cache upload prefix overflow",
                    },
                )?;
                let per_head_stride = kv_span.checked_mul(self.head_dim).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "qwen kv-cache upload stride overflow",
                    },
                )?;
                let keys =
                    self.keys
                        .f32_slice()
                        .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                            reason: "qwen kv-cache upload f32 storage missing",
                        })?;
                let values =
                    self.values
                        .f32_slice()
                        .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                            reason: "qwen kv-cache upload f32 storage missing",
                        })?;
                Self::upload_storage_prefix_f32(
                    keys,
                    graph,
                    key_history,
                    self.max_positions,
                    self.kv_heads,
                    self.head_dim,
                    per_head_len,
                    per_head_stride,
                    key_tensor_name,
                )?;
                Self::upload_storage_prefix_f32(
                    values,
                    graph,
                    value_history,
                    self.max_positions,
                    self.kv_heads,
                    self.head_dim,
                    per_head_len,
                    per_head_stride,
                    value_tensor_name,
                )
            }
            GgmlKvElementType::Q8_0 => {
                let row_nbytes = self.element_type.row_nbytes(self.head_dim).map_err(|_| {
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "qwen kv-cache upload q8 row size invalid",
                    }
                })?;
                let per_head_len = token_count.checked_mul(row_nbytes).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "qwen kv-cache upload q8 prefix overflow",
                    },
                )?;
                let per_head_stride = kv_span.checked_mul(row_nbytes).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "qwen kv-cache upload q8 stride overflow",
                    },
                )?;
                let keys =
                    self.keys
                        .q8_slice()
                        .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                            reason: "qwen kv-cache upload q8 storage missing",
                        })?;
                let values =
                    self.values
                        .q8_slice()
                        .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                            reason: "qwen kv-cache upload q8 storage missing",
                        })?;
                Self::upload_storage_prefix_bytes(
                    keys,
                    graph,
                    key_history,
                    self.max_positions,
                    self.kv_heads,
                    row_nbytes,
                    per_head_len,
                    per_head_stride,
                    key_tensor_name,
                )?;
                Self::upload_storage_prefix_bytes(
                    values,
                    graph,
                    value_history,
                    self.max_positions,
                    self.kv_heads,
                    row_nbytes,
                    per_head_len,
                    per_head_stride,
                    value_tensor_name,
                )
            }
            GgmlKvElementType::F16 => Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "host kv-cache upload does not support f16 storage",
            }),
        }
    }

    pub(crate) fn max_positions(&self) -> usize {
        self.max_positions
    }

    pub(crate) fn clear_written_positions(&mut self) {
        self.written_positions = 0;
    }

    #[cfg(test)]
    pub(crate) fn written_positions(&self) -> usize {
        self.written_positions
    }

    pub(crate) fn truncate_written_positions(
        &mut self,
        written_positions: usize,
    ) -> Result<(), String> {
        if written_positions > self.written_positions {
            return Err(format!(
                "qwen3-asr kv-cache truncate target {written_positions} exceeds written_positions={}",
                self.written_positions
            ));
        }
        self.written_positions = written_positions;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn fork_prefix(
        &self,
        written_positions: usize,
        max_positions: usize,
    ) -> Result<Self, String> {
        if crate::models::native_execution_services::current_native_execution_scope_id().is_some() {
            return Err(host_kv_allocation_failure(
                "qwen-shaped host KV fork attempted without an admitted owner transaction"
                    .to_string(),
            ));
        }
        self.fork_prefix_for_owner(written_positions, max_positions)
    }

    #[cfg(test)]
    fn fork_prefix_for_owner(
        &self,
        written_positions: usize,
        max_positions: usize,
    ) -> Result<Self, String> {
        if written_positions > self.written_positions {
            return Err(format!(
                "qwen3-asr kv-cache fork target {written_positions} exceeds written_positions={}",
                self.written_positions
            ));
        }
        if max_positions < written_positions {
            return Err(format!(
                "qwen3-asr kv-cache fork max_positions={max_positions} is smaller than written_positions={written_positions}"
            ));
        }

        let mut forked = Self::new_with_element_type(
            max_positions,
            self.kv_heads,
            self.head_dim,
            self.element_type,
        )?;
        // An owner fork is a materialized host cache even when the copied
        // prefix is empty. Otherwise its first production write would need an
        // unadmitted lazy allocation after this transaction committed.
        forked.materialize_storage_for_owner()?;
        if written_positions == 0 {
            return Ok(forked);
        }
        match self.element_type {
            GgmlKvElementType::F32 => {
                let width = self.key_width();
                let old_expected_len = self
                    .max_positions
                    .checked_mul(width)
                    .ok_or_else(|| "qwen3-asr kv-cache fork old length overflowed".to_string())?;
                let keys = self.keys.f32_slice()?;
                let values = self.values.f32_slice()?;
                if keys.len() != old_expected_len || values.len() != old_expected_len {
                    return Err(format!(
                        "qwen3-asr kv-cache fork storage length mismatch: keys={} values={} expected={old_expected_len}",
                        keys.len(),
                        values.len()
                    ));
                }
                let new_keys = match &mut forked.keys {
                    HostKvStorage::F32(values) => values,
                    HostKvStorage::Q8(_) => unreachable!("f32 fork allocated q8 storage"),
                };
                Self::copy_history_prefix_to_span_f32(
                    keys,
                    new_keys,
                    self.max_positions,
                    max_positions,
                    self.kv_heads,
                    self.head_dim,
                    written_positions,
                )?;
                let new_values = match &mut forked.values {
                    HostKvStorage::F32(values) => values,
                    HostKvStorage::Q8(_) => unreachable!("f32 fork allocated q8 storage"),
                };
                Self::copy_history_prefix_to_span_f32(
                    values,
                    new_values,
                    self.max_positions,
                    max_positions,
                    self.kv_heads,
                    self.head_dim,
                    written_positions,
                )?;
            }
            GgmlKvElementType::Q8_0 => {
                let row_nbytes = self.element_type.row_nbytes(self.head_dim)?;
                let old_expected_len = self
                    .max_positions
                    .checked_mul(self.kv_heads)
                    .and_then(|n| n.checked_mul(row_nbytes))
                    .ok_or_else(|| {
                        "qwen3-asr kv-cache fork old q8 length overflowed".to_string()
                    })?;
                let keys = self.keys.q8_slice()?;
                let values = self.values.q8_slice()?;
                if keys.len() != old_expected_len || values.len() != old_expected_len {
                    return Err(format!(
                        "qwen3-asr kv-cache fork q8 storage length mismatch: keys={} values={} expected={old_expected_len}",
                        keys.len(),
                        values.len()
                    ));
                }
                let new_keys = match &mut forked.keys {
                    HostKvStorage::Q8(values) => values,
                    HostKvStorage::F32(_) => unreachable!("q8 fork allocated f32 storage"),
                };
                Self::copy_history_prefix_to_span_bytes(
                    keys,
                    new_keys,
                    self.max_positions,
                    max_positions,
                    self.kv_heads,
                    row_nbytes,
                    written_positions,
                )?;
                let new_values = match &mut forked.values {
                    HostKvStorage::Q8(values) => values,
                    HostKvStorage::F32(_) => unreachable!("q8 fork allocated f32 storage"),
                };
                Self::copy_history_prefix_to_span_bytes(
                    values,
                    new_values,
                    self.max_positions,
                    max_positions,
                    self.kv_heads,
                    row_nbytes,
                    written_positions,
                )?;
            }
            GgmlKvElementType::F16 => unreachable!("host KV rejects f16"),
        }
        forked.written_positions = written_positions;
        Ok(forked)
    }

    fn validate_prefix_copy_from(
        &self,
        source: &Self,
        written_positions: usize,
    ) -> Result<(), String> {
        if self.kv_heads != source.kv_heads
            || self.head_dim != source.head_dim
            || self.element_type != source.element_type
        {
            return Err("qwen-shaped host KV prefix copy geometry mismatch".to_string());
        }
        if written_positions > source.written_positions || written_positions > self.max_positions {
            return Err(format!(
                "qwen-shaped host KV prefix copy span {written_positions} exceeds source written={} or destination max={}",
                source.written_positions, self.max_positions
            ));
        }
        let source_expected = source.storage_len()?;
        let destination_expected = self.storage_len()?;
        match (&self.keys, &self.values, &source.keys, &source.values) {
            (
                HostKvStorage::F32(destination_keys),
                HostKvStorage::F32(destination_values),
                HostKvStorage::F32(source_keys),
                HostKvStorage::F32(source_values),
            ) if destination_keys.len() == destination_expected
                && destination_values.len() == destination_expected
                && source_keys.len() == source_expected
                && source_values.len() == source_expected => {}
            (
                HostKvStorage::Q8(destination_keys),
                HostKvStorage::Q8(destination_values),
                HostKvStorage::Q8(source_keys),
                HostKvStorage::Q8(source_values),
            ) if destination_keys.len() == destination_expected
                && destination_values.len() == destination_expected
                && source_keys.len() == source_expected
                && source_values.len() == source_expected => {}
            _ => {
                return Err(
                    "qwen-shaped host KV prefix copy requires fully materialized matching storage"
                        .to_string(),
                );
            }
        }
        Ok(())
    }

    fn copy_prefix_from_for_owner(
        &mut self,
        source: &Self,
        written_positions: usize,
    ) -> Result<(), String> {
        self.validate_prefix_copy_from(source, written_positions)?;
        match self.element_type {
            GgmlKvElementType::F32 => {
                let source_keys = source.keys.f32_slice()?;
                let source_values = source.values.f32_slice()?;
                let destination_keys = match &mut self.keys {
                    HostKvStorage::F32(values) => values,
                    HostKvStorage::Q8(_) => unreachable!("validated f32 destination became q8"),
                };
                Self::copy_history_prefix_to_span_f32(
                    source_keys,
                    destination_keys,
                    source.max_positions,
                    self.max_positions,
                    self.kv_heads,
                    self.head_dim,
                    written_positions,
                )?;
                let destination_values = match &mut self.values {
                    HostKvStorage::F32(values) => values,
                    HostKvStorage::Q8(_) => unreachable!("validated f32 destination became q8"),
                };
                Self::copy_history_prefix_to_span_f32(
                    source_values,
                    destination_values,
                    source.max_positions,
                    self.max_positions,
                    self.kv_heads,
                    self.head_dim,
                    written_positions,
                )?;
            }
            GgmlKvElementType::Q8_0 => {
                let row_nbytes = self.element_type.row_nbytes(self.head_dim)?;
                let source_keys = source.keys.q8_slice()?;
                let source_values = source.values.q8_slice()?;
                let destination_keys = match &mut self.keys {
                    HostKvStorage::Q8(values) => values,
                    HostKvStorage::F32(_) => unreachable!("validated q8 destination became f32"),
                };
                Self::copy_history_prefix_to_span_bytes(
                    source_keys,
                    destination_keys,
                    source.max_positions,
                    self.max_positions,
                    self.kv_heads,
                    row_nbytes,
                    written_positions,
                )?;
                let destination_values = match &mut self.values {
                    HostKvStorage::Q8(values) => values,
                    HostKvStorage::F32(_) => unreachable!("validated q8 destination became f32"),
                };
                Self::copy_history_prefix_to_span_bytes(
                    source_values,
                    destination_values,
                    source.max_positions,
                    self.max_positions,
                    self.kv_heads,
                    row_nbytes,
                    written_positions,
                )?;
            }
            GgmlKvElementType::F16 => unreachable!("host KV rejects f16"),
        }
        self.written_positions = written_positions;
        Ok(())
    }

    fn storage_len(&self) -> Result<usize, String> {
        match self.element_type {
            GgmlKvElementType::F32 => self
                .max_positions
                .checked_mul(self.key_width())
                .ok_or_else(|| "qwen-shaped host KV f32 storage length overflowed".to_string()),
            GgmlKvElementType::Q8_0 => {
                let row_bytes = self.element_type.row_nbytes(self.head_dim)?;
                self.max_positions
                    .checked_mul(self.kv_heads)
                    .and_then(|positions| positions.checked_mul(row_bytes))
                    .ok_or_else(|| "qwen-shaped host KV q8 storage length overflowed".to_string())
            }
            GgmlKvElementType::F16 => {
                Err("host KV storage length does not support f16".to_string())
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn resize_max_positions(&mut self, new_max_positions: usize) -> Result<(), String> {
        if new_max_positions == self.max_positions {
            return Ok(());
        }
        if new_max_positions < self.written_positions {
            return Err(format!(
                "qwen3-asr kv-cache resize target {new_max_positions} is smaller than written_positions={}",
                self.written_positions
            ));
        }
        if self.keys.is_empty() && self.values.is_empty() {
            self.max_positions = new_max_positions;
            return Ok(());
        }
        if crate::models::native_execution_services::current_native_execution_scope_id().is_some() {
            return Err(host_kv_allocation_failure(
                "qwen-shaped host KV resize attempted without an admitted owner replacement"
                    .to_string(),
            ));
        }

        match self.element_type {
            GgmlKvElementType::F32 => {
                let width = self.key_width();
                let old_expected_len = self
                    .max_positions
                    .checked_mul(width)
                    .ok_or_else(|| "qwen3-asr kv-cache resize old length overflowed".to_string())?;
                let keys = self.keys.f32_slice()?.to_vec();
                let values = self.values.f32_slice()?.to_vec();
                if keys.len() != old_expected_len || values.len() != old_expected_len {
                    return Err(format!(
                        "qwen3-asr kv-cache resize storage length mismatch: keys={} values={} expected={old_expected_len}",
                        keys.len(),
                        values.len()
                    ));
                }
                let new_len = new_max_positions
                    .checked_mul(width)
                    .ok_or_else(|| "qwen3-asr kv-cache resize new length overflowed".to_string())?;
                let mut new_keys = vec![0.0; new_len];
                let mut new_values = vec![0.0; new_len];
                Self::copy_history_prefix_to_span_f32(
                    &keys,
                    &mut new_keys,
                    self.max_positions,
                    new_max_positions,
                    self.kv_heads,
                    self.head_dim,
                    self.written_positions,
                )?;
                Self::copy_history_prefix_to_span_f32(
                    &values,
                    &mut new_values,
                    self.max_positions,
                    new_max_positions,
                    self.kv_heads,
                    self.head_dim,
                    self.written_positions,
                )?;
                self.max_positions = new_max_positions;
                self.keys = HostKvStorage::F32(new_keys);
                self.values = HostKvStorage::F32(new_values);
            }
            GgmlKvElementType::Q8_0 => {
                let row_nbytes = self.element_type.row_nbytes(self.head_dim)?;
                let old_expected_len = self
                    .max_positions
                    .checked_mul(self.kv_heads)
                    .and_then(|n| n.checked_mul(row_nbytes))
                    .ok_or_else(|| {
                        "qwen3-asr kv-cache resize old q8 length overflowed".to_string()
                    })?;
                let keys = self.keys.q8_slice()?.to_vec();
                let values = self.values.q8_slice()?.to_vec();
                if keys.len() != old_expected_len || values.len() != old_expected_len {
                    return Err(format!(
                        "qwen3-asr kv-cache resize q8 storage length mismatch: keys={} values={} expected={old_expected_len}",
                        keys.len(),
                        values.len()
                    ));
                }
                let new_len = new_max_positions
                    .checked_mul(self.kv_heads)
                    .and_then(|n| n.checked_mul(row_nbytes))
                    .ok_or_else(|| {
                        "qwen3-asr kv-cache resize new q8 length overflowed".to_string()
                    })?;
                let mut new_keys = vec![0_u8; new_len];
                let mut new_values = vec![0_u8; new_len];
                Self::copy_history_prefix_to_span_bytes(
                    &keys,
                    &mut new_keys,
                    self.max_positions,
                    new_max_positions,
                    self.kv_heads,
                    row_nbytes,
                    self.written_positions,
                )?;
                Self::copy_history_prefix_to_span_bytes(
                    &values,
                    &mut new_values,
                    self.max_positions,
                    new_max_positions,
                    self.kv_heads,
                    row_nbytes,
                    self.written_positions,
                )?;
                self.max_positions = new_max_positions;
                self.keys = HostKvStorage::Q8(new_keys);
                self.values = HostKvStorage::Q8(new_values);
            }
            GgmlKvElementType::F16 => unreachable!("host KV rejects f16"),
        }
        Ok(())
    }

    pub(crate) fn full_history_storage(&self) -> Result<Qwen3AsrLayerKvCacheHistory<'_>, String> {
        match self.element_type {
            GgmlKvElementType::F32 => {
                let expected_len = self
                    .max_positions
                    .checked_mul(self.key_width())
                    .ok_or_else(|| "qwen3-asr kv-cache storage length overflowed".to_string())?;
                let keys = self.keys.f32_slice()?;
                let values = self.values.f32_slice()?;
                if keys.len() != expected_len || values.len() != expected_len {
                    return Err(format!(
                        "qwen3-asr kv-cache storage length mismatch: keys={} values={} expected={}",
                        keys.len(),
                        values.len(),
                        expected_len
                    ));
                }
                Ok(Qwen3AsrLayerKvCacheHistory {
                    max_positions: self.max_positions,
                    kv_heads: self.kv_heads,
                    head_dim: self.head_dim,
                    written_positions: self.written_positions,
                    element_type: self.element_type,
                    keys_f32: Some(keys),
                    values_f32: Some(values),
                    keys_q8: None,
                    values_q8: None,
                })
            }
            GgmlKvElementType::Q8_0 => {
                let row_nbytes = self.element_type.row_nbytes(self.head_dim)?;
                let expected_len = self
                    .max_positions
                    .checked_mul(self.kv_heads)
                    .and_then(|n| n.checked_mul(row_nbytes))
                    .ok_or_else(|| "qwen3-asr kv-cache q8 storage length overflowed".to_string())?;
                let keys = self.keys.q8_slice()?;
                let values = self.values.q8_slice()?;
                if keys.len() != expected_len || values.len() != expected_len {
                    return Err(format!(
                        "qwen3-asr kv-cache q8 storage length mismatch: keys={} values={} expected={}",
                        keys.len(),
                        values.len(),
                        expected_len
                    ));
                }
                Ok(Qwen3AsrLayerKvCacheHistory {
                    max_positions: self.max_positions,
                    kv_heads: self.kv_heads,
                    head_dim: self.head_dim,
                    written_positions: self.written_positions,
                    element_type: self.element_type,
                    keys_f32: None,
                    values_f32: None,
                    keys_q8: Some(keys),
                    values_q8: Some(values),
                })
            }
            GgmlKvElementType::F16 => Err("host kv-cache history does not support f16".to_string()),
        }
    }

    #[cfg(test)]
    pub(crate) fn snapshot_written(&self) -> Result<Qwen3AsrLayerKvCacheSnapshot, String> {
        Ok(Qwen3AsrLayerKvCacheSnapshot {
            written_positions: self.written_positions,
            key_width: self.key_width(),
            value_width: self.value_width(),
            element_type: self.element_type,
        })
    }

    fn ensure_shape_initialized(
        &mut self,
        key_width: usize,
        value_width: usize,
    ) -> Result<(), String> {
        if key_width != self.key_width() || value_width != self.value_width() {
            return Err(format!(
                "qwen3-asr kv-cache shape mismatch: key_width={} (expected {}), value_width={} (expected {})",
                key_width,
                self.key_width(),
                value_width,
                self.value_width()
            ));
        }
        if (self.keys.is_empty() || self.values.is_empty())
            && crate::models::native_execution_services::current_native_execution_scope_id()
                .is_some()
        {
            return Err(host_kv_allocation_failure(
                "qwen-shaped host KV attempted lazy allocation inside a native execution scope; construct an admitted Qwen3AsrHostKvCacheOwner first"
                    .to_string(),
            ));
        }
        if self.keys.is_empty() {
            self.keys = self.allocate_storage()?;
        }
        if self.values.is_empty() {
            self.values = self.allocate_storage()?;
        }
        Ok(())
    }

    fn allocate_storage(&self) -> Result<HostKvStorage, String> {
        match self.element_type {
            GgmlKvElementType::F32 => {
                let len = self
                    .max_positions
                    .checked_mul(self.key_width())
                    .ok_or_else(|| "qwen3-asr kv-cache f32 allocation overflowed".to_string())?;
                let mut values = Vec::new();
                values.try_reserve_exact(len).map_err(|error| {
                    host_kv_allocation_failure(format!(
                        "qwen3-asr host KV f32 allocation failed for {len} elements: {error}"
                    ))
                })?;
                values.resize(len, 0.0);
                Ok(HostKvStorage::F32(values))
            }
            GgmlKvElementType::Q8_0 => {
                let row_nbytes = self.element_type.row_nbytes(self.head_dim)?;
                let len = self
                    .max_positions
                    .checked_mul(self.kv_heads)
                    .and_then(|n| n.checked_mul(row_nbytes))
                    .ok_or_else(|| "qwen3-asr kv-cache q8 allocation overflowed".to_string())?;
                let mut values = Vec::new();
                values.try_reserve_exact(len).map_err(|error| {
                    host_kv_allocation_failure(format!(
                        "qwen3-asr host KV q8 allocation failed for {len} bytes: {error}"
                    ))
                })?;
                values.resize(len, 0_u8);
                Ok(HostKvStorage::Q8(values))
            }
            GgmlKvElementType::F16 => Err("host KV rejects f16 storage".to_string()),
        }
    }

    fn key_width(&self) -> usize {
        self.kv_heads.saturating_mul(self.head_dim)
    }

    fn value_width(&self) -> usize {
        self.key_width()
    }

    fn write_history_row_f32(
        storage: &mut [f32],
        max_positions: usize,
        kv_heads: usize,
        head_dim: usize,
        position: usize,
        row: &[f32],
    ) -> Result<(), String> {
        for kv_head in 0..kv_heads {
            let row_start = kv_head
                .checked_mul(head_dim)
                .ok_or_else(|| "qwen3-asr kv-cache row indexing overflowed".to_string())?;
            let row_end = row_start
                .checked_add(head_dim)
                .ok_or_else(|| "qwen3-asr kv-cache row indexing overflowed".to_string())?;
            let storage_start = kv_head
                .checked_mul(max_positions)
                .and_then(|base| base.checked_add(position))
                .and_then(|slot| slot.checked_mul(head_dim))
                .ok_or_else(|| "qwen3-asr kv-cache storage indexing overflowed".to_string())?;
            let storage_end = storage_start
                .checked_add(head_dim)
                .ok_or_else(|| "qwen3-asr kv-cache storage indexing overflowed".to_string())?;
            storage[storage_start..storage_end].copy_from_slice(&row[row_start..row_end]);
        }
        Ok(())
    }

    fn write_history_row_q8(
        storage: &mut [u8],
        max_positions: usize,
        kv_heads: usize,
        head_dim: usize,
        row_nbytes: usize,
        position: usize,
        row: &[f32],
    ) -> Result<(), String> {
        for kv_head in 0..kv_heads {
            let row_start = kv_head
                .checked_mul(head_dim)
                .ok_or_else(|| "qwen3-asr kv-cache q8 row indexing overflowed".to_string())?;
            let row_end = row_start
                .checked_add(head_dim)
                .ok_or_else(|| "qwen3-asr kv-cache q8 row indexing overflowed".to_string())?;
            let packed = GgmlKvElementType::Q8_0.quantize_rows_from_f32(
                &row[row_start..row_end],
                head_dim,
                1,
            )?;
            if packed.len() != row_nbytes {
                return Err(format!(
                    "qwen3-asr kv-cache q8 row size mismatch: got {} expected {row_nbytes}",
                    packed.len()
                ));
            }
            let storage_start = kv_head
                .checked_mul(max_positions)
                .and_then(|base| base.checked_add(position))
                .and_then(|slot| slot.checked_mul(row_nbytes))
                .ok_or_else(|| "qwen3-asr kv-cache q8 storage indexing overflowed".to_string())?;
            let storage_end = storage_start
                .checked_add(row_nbytes)
                .ok_or_else(|| "qwen3-asr kv-cache q8 storage indexing overflowed".to_string())?;
            storage[storage_start..storage_end].copy_from_slice(&packed);
        }
        Ok(())
    }

    fn copy_history_prefix_to_span_f32(
        source: &[f32],
        target: &mut [f32],
        old_max_positions: usize,
        new_max_positions: usize,
        kv_heads: usize,
        head_dim: usize,
        written_positions: usize,
    ) -> Result<(), String> {
        let prefix_len = written_positions
            .checked_mul(head_dim)
            .ok_or_else(|| "qwen3-asr kv-cache resize prefix length overflowed".to_string())?;
        for kv_head in 0..kv_heads {
            let source_start = kv_head
                .checked_mul(old_max_positions)
                .and_then(|base| base.checked_mul(head_dim))
                .ok_or_else(|| {
                    "qwen3-asr kv-cache resize source indexing overflowed".to_string()
                })?;
            let source_end = source_start.checked_add(prefix_len).ok_or_else(|| {
                "qwen3-asr kv-cache resize source indexing overflowed".to_string()
            })?;
            let target_start = kv_head
                .checked_mul(new_max_positions)
                .and_then(|base| base.checked_mul(head_dim))
                .ok_or_else(|| {
                    "qwen3-asr kv-cache resize target indexing overflowed".to_string()
                })?;
            let target_end = target_start.checked_add(prefix_len).ok_or_else(|| {
                "qwen3-asr kv-cache resize target indexing overflowed".to_string()
            })?;
            target[target_start..target_end].copy_from_slice(&source[source_start..source_end]);
        }
        Ok(())
    }

    fn copy_history_prefix_to_span_bytes(
        source: &[u8],
        target: &mut [u8],
        old_max_positions: usize,
        new_max_positions: usize,
        kv_heads: usize,
        row_nbytes: usize,
        written_positions: usize,
    ) -> Result<(), String> {
        let prefix_len = written_positions
            .checked_mul(row_nbytes)
            .ok_or_else(|| "qwen3-asr kv-cache q8 resize prefix length overflowed".to_string())?;
        for kv_head in 0..kv_heads {
            let source_start = kv_head
                .checked_mul(old_max_positions)
                .and_then(|base| base.checked_mul(row_nbytes))
                .ok_or_else(|| {
                    "qwen3-asr kv-cache q8 resize source indexing overflowed".to_string()
                })?;
            let source_end = source_start.checked_add(prefix_len).ok_or_else(|| {
                "qwen3-asr kv-cache q8 resize source indexing overflowed".to_string()
            })?;
            let target_start = kv_head
                .checked_mul(new_max_positions)
                .and_then(|base| base.checked_mul(row_nbytes))
                .ok_or_else(|| {
                    "qwen3-asr kv-cache q8 resize target indexing overflowed".to_string()
                })?;
            let target_end = target_start.checked_add(prefix_len).ok_or_else(|| {
                "qwen3-asr kv-cache q8 resize target indexing overflowed".to_string()
            })?;
            target[target_start..target_end].copy_from_slice(&source[source_start..source_end]);
        }
        Ok(())
    }

    fn upload_storage_prefix_f32<'a>(
        storage: &[f32],
        graph: &mut GgmlCpuGraphBuilder<'a>,
        tensor: GgmlCpuTensor<'a>,
        max_positions: usize,
        kv_heads: usize,
        head_dim: usize,
        per_head_len: usize,
        per_head_stride: usize,
        tensor_name: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        for kv_head in 0..kv_heads {
            let storage_start = kv_head
                .checked_mul(max_positions)
                .and_then(|base| base.checked_mul(head_dim))
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "qwen kv-cache upload storage indexing overflow",
                })?;
            let storage_end = storage_start.checked_add(per_head_len).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "qwen kv-cache upload storage indexing overflow",
                },
            )?;
            let output_offset = kv_head.checked_mul(per_head_stride).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "qwen kv-cache upload output indexing overflow",
                },
            )?;
            graph.set_f32_slice_with_offset(
                tensor,
                output_offset,
                &storage[storage_start..storage_end],
                tensor_name,
            )?;
        }
        Ok(())
    }

    fn upload_storage_prefix_bytes<'a>(
        storage: &[u8],
        graph: &mut GgmlCpuGraphBuilder<'a>,
        tensor: GgmlCpuTensor<'a>,
        max_positions: usize,
        kv_heads: usize,
        row_nbytes: usize,
        per_head_len: usize,
        per_head_stride: usize,
        tensor_name: &'static str,
    ) -> Result<(), GgmlCpuGraphError> {
        for kv_head in 0..kv_heads {
            let storage_start = kv_head
                .checked_mul(max_positions)
                .and_then(|base| base.checked_mul(row_nbytes))
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "qwen kv-cache upload q8 storage indexing overflow",
                })?;
            let storage_end = storage_start.checked_add(per_head_len).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "qwen kv-cache upload q8 storage indexing overflow",
                },
            )?;
            let output_offset = kv_head.checked_mul(per_head_stride).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "qwen kv-cache upload q8 output indexing overflow",
                },
            )?;
            graph.set_bytes_slice_with_offset(
                tensor,
                output_offset,
                &storage[storage_start..storage_end],
                tensor_name,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::dequantize_q8_0_rows;

    #[test]
    fn resident_only_owner_has_no_host_payload_or_system_memory_charge() {
        let services = crate::models::native_execution_services::test_native_execution_services();
        let _scope = crate::models::native_execution_services::install_native_execution_services(
            services.as_ref(),
        );
        let capacity = Qwen3AsrKvCacheCapacity::new(4, 8).unwrap();
        let owner = Qwen3AsrHostKvCacheOwner::try_new(
            "test.qwen.resident-only",
            2,
            capacity,
            2,
            32,
            GgmlKvElementType::F32,
            Qwen3AsrHostKvMode::ResidentOnly,
        )
        .unwrap();
        assert!(!owner.has_materialized_payload());
        assert_eq!(
            services
                .memory_broker()
                .usage(&crate::device::execution_memory::MemoryDomainKey::SystemMemory)
                .committed_bytes,
            0
        );
    }

    #[test]
    fn materialized_owner_commits_actual_vec_capacities_until_drop() {
        let services = crate::models::native_execution_services::test_native_execution_services();
        let _scope = crate::models::native_execution_services::install_native_execution_services(
            services.as_ref(),
        );
        let capacity = Qwen3AsrKvCacheCapacity::new(4, 8).unwrap();
        let owner = Qwen3AsrHostKvCacheOwner::try_new(
            "test.qwen.materialized",
            2,
            capacity,
            2,
            32,
            GgmlKvElementType::F32,
            Qwen3AsrHostKvMode::Materialized,
        )
        .unwrap();
        assert!(owner.has_materialized_payload());
        let actual = qwen_host_kv_actual_capacity_bytes(&owner, owner.capacity()).unwrap();
        assert_eq!(
            services
                .memory_broker()
                .usage(&crate::device::execution_memory::MemoryDomainKey::SystemMemory)
                .committed_bytes,
            actual
        );
        drop(owner);
        assert_eq!(
            services
                .memory_broker()
                .usage(&crate::device::execution_memory::MemoryDomainKey::SystemMemory)
                .committed_bytes,
            0
        );
    }

    #[test]
    fn admitted_owner_fork_accounts_for_overlap_and_preserves_prefix() {
        let services = crate::models::native_execution_services::test_native_execution_services();
        let _scope = crate::models::native_execution_services::install_native_execution_services(
            services.as_ref(),
        );
        let capacity = Qwen3AsrKvCacheCapacity::new(4, 4).unwrap();
        let mut source = Qwen3AsrHostKvCacheOwner::try_new(
            "test.qwen.fork-source",
            1,
            capacity,
            2,
            2,
            GgmlKvElementType::F32,
            Qwen3AsrHostKvMode::Materialized,
        )
        .unwrap();
        source[0]
            .write(0, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .unwrap();
        source[0]
            .write(1, &[5.0, 6.0, 7.0, 8.0], &[50.0, 60.0, 70.0, 80.0])
            .unwrap();
        let source_bytes = qwen_host_kv_actual_capacity_bytes(&source, source.capacity()).unwrap();

        let fork =
            Qwen3AsrHostKvCacheOwner::try_fork_prefix("test.qwen.fork-destination", &source, 2, 6)
                .unwrap();
        let fork_bytes = qwen_host_kv_actual_capacity_bytes(&fork, fork.capacity()).unwrap();
        assert_eq!(
            services
                .memory_broker()
                .usage(&crate::device::execution_memory::MemoryDomainKey::SystemMemory)
                .committed_bytes,
            source_bytes + fork_bytes
        );
        let history = fork[0].full_history_storage().unwrap();
        assert_eq!(history.written_positions, 2);
        assert_eq!(
            history.keys_f32.unwrap(),
            &[
                1.0, 2.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 7.0, 8.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );

        drop(fork);
        assert_eq!(
            services
                .memory_broker()
                .usage(&crate::device::execution_memory::MemoryDomainKey::SystemMemory)
                .committed_bytes,
            source_bytes
        );
    }

    #[test]
    fn admitted_owner_reuses_destination_capacity_for_prefix_copy() {
        let services = crate::models::native_execution_services::test_native_execution_services();
        let _scope = crate::models::native_execution_services::install_native_execution_services(
            services.as_ref(),
        );
        let mut source = Qwen3AsrHostKvCacheOwner::try_new(
            "test.qwen.copy-source",
            1,
            Qwen3AsrKvCacheCapacity::new(4, 4).unwrap(),
            1,
            2,
            GgmlKvElementType::F32,
            Qwen3AsrHostKvMode::Materialized,
        )
        .unwrap();
        source[0].write(0, &[1.0, 2.0], &[10.0, 20.0]).unwrap();
        source[0].write(1, &[3.0, 4.0], &[30.0, 40.0]).unwrap();
        let mut destination = Qwen3AsrHostKvCacheOwner::try_new(
            "test.qwen.copy-destination",
            1,
            Qwen3AsrKvCacheCapacity::new(6, 6).unwrap(),
            1,
            2,
            GgmlKvElementType::F32,
            Qwen3AsrHostKvMode::Materialized,
        )
        .unwrap();
        let committed_before = services
            .memory_broker()
            .usage(&crate::device::execution_memory::MemoryDomainKey::SystemMemory)
            .committed_bytes;

        destination.replace_prefix_from(&source, 2).unwrap();

        assert_eq!(
            services
                .memory_broker()
                .usage(&crate::device::execution_memory::MemoryDomainKey::SystemMemory)
                .committed_bytes,
            committed_before
        );
        let history = destination[0].full_history_storage().unwrap();
        assert_eq!(history.written_positions, 2);
        assert_eq!(
            history.keys_f32.unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn production_scope_rejects_unadmitted_lazy_host_kv_allocation() {
        let services = crate::models::native_execution_services::test_native_execution_services();
        let _scope = crate::models::native_execution_services::install_native_execution_services(
            services.as_ref(),
        );
        let mut cache = Qwen3AsrLayerKvCacheState::new(2, 1, 32);
        let row = vec![0.0_f32; 32];
        let error = cache.write(0, &row, &row).unwrap_err();
        assert!(error.contains("admitted Qwen3AsrHostKvCacheOwner"));
    }

    #[test]
    fn planned_capacity_keeps_logical_and_resident_spans_distinct() {
        let capacity = Qwen3AsrKvCacheCapacity::new(1_290, 2_367).expect("capacity");
        assert_eq!(capacity.logical_positions(), 1_290);
        assert_eq!(capacity.resident_positions(), 2_367);
        assert_eq!(
            capacity
                .validate_measured_logical_positions(1_289)
                .expect_err("runtime/planner drift must fail closed"),
            Qwen3AsrKvCacheCapacityError::LogicalPositionMismatch {
                planned_positions: 1_290,
                measured_positions: 1_289,
            }
        );
    }

    #[test]
    fn planned_capacity_rejects_resident_span_below_logical_span() {
        assert_eq!(
            Qwen3AsrKvCacheCapacity::new(2_367, 1_290)
                .expect_err("reserve below logical must fail"),
            Qwen3AsrKvCacheCapacityError::ResidentDoesNotCoverLogical {
                logical_positions: 2_367,
                resident_positions: 1_290,
            }
        );
    }

    #[test]
    fn planned_capacity_requires_request_plan() {
        assert_eq!(
            Qwen3AsrKvCacheCapacity::from_decoder_state(
                &crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
                crate::models::qwen::capacity::QWEN3_SELF_KV_STATE_ID,
            )
            .expect_err("causal runtime must not invent a fallback capacity"),
            Qwen3AsrKvCacheCapacityError::DecoderStateNotPlanned,
        );
    }

    #[test]
    fn host_kv_cache_tracks_written_prefix() {
        let mut cache = Qwen3AsrLayerKvCacheState::new(8, 1, 2);
        cache
            .write(0, &[0.1, 0.2], &[0.3, 0.4])
            .expect("write row 0");
        cache
            .write(1, &[0.5, 0.6], &[0.7, 0.8])
            .expect("write row 1");

        let snapshot = cache.snapshot_written().expect("snapshot");
        assert_eq!(
            snapshot,
            Qwen3AsrLayerKvCacheSnapshot {
                written_positions: 2,
                key_width: 2,
                value_width: 2,
                element_type: GgmlKvElementType::F32,
            }
        );
    }

    #[test]
    fn host_kv_cache_resize_preserves_written_head_major_prefix() {
        let mut cache = Qwen3AsrLayerKvCacheState::new(3, 2, 2);
        cache
            .write(0, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .expect("row 0");
        cache
            .write(1, &[5.0, 6.0, 7.0, 8.0], &[50.0, 60.0, 70.0, 80.0])
            .expect("row 1");

        cache.resize_max_positions(5).expect("resize");

        let history = cache.full_history_storage().expect("history");
        assert_eq!(history.max_positions, 5);
        assert_eq!(history.written_positions, 2);
        assert_eq!(
            history.keys_f32.expect("f32 keys"),
            &[
                1.0, 2.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                3.0, 4.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
        );
        assert_eq!(
            history.values_f32.expect("f32 values"),
            &[
                10.0, 20.0, 50.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                30.0, 40.0, 70.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
        );
    }

    #[test]
    fn host_kv_cache_forks_and_truncates_written_prefix() {
        let mut cache = Qwen3AsrLayerKvCacheState::new(4, 2, 2);
        cache
            .write(0, &[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
            .expect("row 0");
        cache
            .write(1, &[5.0, 6.0, 7.0, 8.0], &[50.0, 60.0, 70.0, 80.0])
            .expect("row 1");
        cache
            .write(2, &[9.0, 10.0, 11.0, 12.0], &[90.0, 100.0, 110.0, 120.0])
            .expect("row 2");

        let fork = cache.fork_prefix(2, 5).expect("fork prefix");
        let history = fork.full_history_storage().expect("history");
        assert_eq!(history.max_positions, 5);
        assert_eq!(history.written_positions, 2);
        assert_eq!(
            history.keys_f32.expect("f32 keys"),
            &[
                1.0, 2.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                3.0, 4.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
        );

        cache.truncate_written_positions(1).expect("truncate");
        assert_eq!(cache.written_positions(), 1);
        assert!(cache.truncate_written_positions(2).is_err());
    }

    #[test]
    fn host_q8_kv_cache_write_resize_fork_preserve_logical_prefix() {
        let head_dim = 32;
        let mut cache = Qwen3AsrLayerKvCacheState::new_with_element_type(
            3,
            2,
            head_dim,
            GgmlKvElementType::Q8_0,
        )
        .expect("q8 cache");
        let mut row0 = vec![0.0_f32; head_dim * 2];
        let mut row1 = vec![0.0_f32; head_dim * 2];
        for i in 0..head_dim {
            row0[i] = (i as f32) * 0.01;
            row0[head_dim + i] = (i as f32) * -0.01;
            row1[i] = 0.5 + (i as f32) * 0.01;
            row1[head_dim + i] = -0.5 - (i as f32) * 0.01;
        }
        cache.write(0, &row0, &row0).expect("row0");
        cache.write(1, &row1, &row1).expect("row1");
        cache.resize_max_positions(5).expect("resize");
        let fork = cache.fork_prefix(2, 6).expect("fork");
        assert_eq!(fork.written_positions(), 2);
        assert_eq!(fork.max_positions(), 6);
        assert_eq!(fork.element_type(), GgmlKvElementType::Q8_0);

        let history = fork.full_history_storage().expect("history");
        let keys = history.keys_q8.expect("q8 keys");
        let row_nbytes = GgmlKvElementType::Q8_0.row_nbytes(head_dim).expect("row");
        // Dequantize head0 position0 and compare to source within q8 tolerance.
        let packed0 = &keys[0..row_nbytes];
        let restored = dequantize_q8_0_rows(packed0, head_dim, 1).expect("dequant");
        let mut max_abs = 0.0_f32;
        for (a, b) in row0[..head_dim].iter().zip(restored.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(max_abs < 0.05, "q8 host roundtrip max abs {max_abs}");
    }

    #[test]
    fn host_q8_kv_cache_rejects_misaligned_head_dim() {
        let err =
            Qwen3AsrLayerKvCacheState::new_with_element_type(4, 1, 30, GgmlKvElementType::Q8_0)
                .expect_err("head_dim 30 must fail");
        assert!(err.contains("block_size"), "{err}");
    }
}
