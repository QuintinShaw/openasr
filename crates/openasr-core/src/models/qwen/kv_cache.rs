use crate::ggml_runtime::{
    GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuTensor, GgmlKvElementType,
};

/// Per-layer host KV cache shared by every Qwen-shaped decoder family.
///
/// Default storage is f32 (byte-identical to the historical path). Opt-in
/// `q8_0` stores native ggml q8_0 rows so host and resident paths share the
/// same packed layout without a full f32 staging buffer.
#[derive(Debug, Clone)]
pub(crate) struct Qwen3AsrLayerKvCacheState {
    max_positions: usize,
    kv_heads: usize,
    head_dim: usize,
    element_type: GgmlKvElementType,
    keys: HostKvStorage,
    values: HostKvStorage,
    written_positions: usize,
}

#[derive(Debug, Clone)]
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

    pub(crate) fn fork_prefix(
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
                let new_len = max_positions
                    .checked_mul(width)
                    .ok_or_else(|| "qwen3-asr kv-cache fork new length overflowed".to_string())?;
                let mut new_keys = vec![0.0; new_len];
                let mut new_values = vec![0.0; new_len];
                Self::copy_history_prefix_to_span_f32(
                    keys,
                    &mut new_keys,
                    self.max_positions,
                    max_positions,
                    self.kv_heads,
                    self.head_dim,
                    written_positions,
                )?;
                Self::copy_history_prefix_to_span_f32(
                    values,
                    &mut new_values,
                    self.max_positions,
                    max_positions,
                    self.kv_heads,
                    self.head_dim,
                    written_positions,
                )?;
                forked.keys = HostKvStorage::F32(new_keys);
                forked.values = HostKvStorage::F32(new_values);
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
                let new_len = max_positions
                    .checked_mul(self.kv_heads)
                    .and_then(|n| n.checked_mul(row_nbytes))
                    .ok_or_else(|| {
                        "qwen3-asr kv-cache fork new q8 length overflowed".to_string()
                    })?;
                let mut new_keys = vec![0_u8; new_len];
                let mut new_values = vec![0_u8; new_len];
                Self::copy_history_prefix_to_span_bytes(
                    keys,
                    &mut new_keys,
                    self.max_positions,
                    max_positions,
                    self.kv_heads,
                    row_nbytes,
                    written_positions,
                )?;
                Self::copy_history_prefix_to_span_bytes(
                    values,
                    &mut new_values,
                    self.max_positions,
                    max_positions,
                    self.kv_heads,
                    row_nbytes,
                    written_positions,
                )?;
                forked.keys = HostKvStorage::Q8(new_keys);
                forked.values = HostKvStorage::Q8(new_values);
            }
            GgmlKvElementType::F16 => unreachable!("host KV rejects f16"),
        }
        forked.written_positions = written_positions;
        Ok(forked)
    }

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
                Ok(HostKvStorage::F32(vec![0.0; len]))
            }
            GgmlKvElementType::Q8_0 => {
                let row_nbytes = self.element_type.row_nbytes(self.head_dim)?;
                let len = self
                    .max_positions
                    .checked_mul(self.kv_heads)
                    .and_then(|n| n.checked_mul(row_nbytes))
                    .ok_or_else(|| "qwen3-asr kv-cache q8 allocation overflowed".to_string())?;
                Ok(HostKvStorage::Q8(vec![0_u8; len]))
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
