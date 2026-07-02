use crate::ggml_runtime::{
    GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor, GgmlStaticTensor,
    GgmlStaticTensorArena,
};

use super::graph_config::qwen_runtime_graph_config;
use super::runtime_contract::Qwen3AsrExecutionMetadata;

#[allow(dead_code)]
const QWEN3_LLM_KV_CACHE_ARENA_CONTEXT_BYTES: usize = 16 * 1024 * 1024;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen3AsrKvCacheLayout {
    pub head_dim: usize,
    pub kv_heads: usize,
    pub max_positions: usize,
    pub layers: usize,
    pub key_width: usize,
    pub value_width: usize,
}

#[allow(dead_code)]
impl Qwen3AsrKvCacheLayout {
    pub(crate) fn from_metadata(
        metadata: Qwen3AsrExecutionMetadata,
    ) -> Result<Self, GgmlCpuGraphError> {
        let key_width = metadata
            .llm_kv_heads
            .checked_mul(metadata.llm_head_dim)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "qwen kv key width overflow",
            })?;
        let value_width = key_width;
        Ok(Self {
            head_dim: metadata.llm_head_dim,
            kv_heads: metadata.llm_kv_heads,
            max_positions: metadata.llm_max_positions,
            layers: metadata.llm_layers,
            key_width,
            value_width,
        })
    }
}

#[allow(dead_code)]
pub(crate) struct Qwen3AsrPersistentKvCache {
    layout: Qwen3AsrKvCacheLayout,
    arena: GgmlStaticTensorArena,
    layers: Vec<Qwen3AsrPersistentKvLayer>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Qwen3AsrPersistentKvLayer {
    #[allow(dead_code)]
    pub key: GgmlStaticTensor,
    #[allow(dead_code)]
    pub value: GgmlStaticTensor,
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3AsrLayerKvCacheState {
    max_positions: usize,
    kv_heads: usize,
    head_dim: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
    written_positions: usize,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen3AsrLayerKvCacheSnapshot {
    pub written_positions: usize,
    pub key_width: usize,
    pub value_width: usize,
}

pub(crate) struct Qwen3AsrLayerKvCacheHistory<'a> {
    pub max_positions: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub written_positions: usize,
    pub keys: &'a [f32],
    pub values: &'a [f32],
}

#[allow(dead_code)]
impl Qwen3AsrPersistentKvCache {
    pub(crate) fn allocate(metadata: Qwen3AsrExecutionMetadata) -> Result<Self, GgmlCpuGraphError> {
        let layout = Qwen3AsrKvCacheLayout::from_metadata(metadata)?;
        let mut config = qwen_runtime_graph_config();
        config.context_bytes = QWEN3_LLM_KV_CACHE_ARENA_CONTEXT_BYTES;
        let runner = GgmlCpuGraphRunner::new(config)?;
        let mut arena = runner.start_static_tensor_arena(config.context_bytes)?;
        let mut layers = Vec::with_capacity(layout.layers);
        for layer_idx in 0..layout.layers {
            let key_name =
                Box::leak(format!("qwen_llm_kv_cache_key_layer_{layer_idx}").into_boxed_str());
            let value_name =
                Box::leak(format!("qwen_llm_kv_cache_value_layer_{layer_idx}").into_boxed_str());
            let key = arena.new_tensor_3d_f32(
                layout.head_dim,
                layout.max_positions,
                layout.kv_heads,
                key_name,
            )?;
            let value = arena.new_tensor_3d_f32(
                layout.head_dim,
                layout.max_positions,
                layout.kv_heads,
                value_name,
            )?;
            layers.push(Qwen3AsrPersistentKvLayer { key, value });
        }
        arena.allocate_backend_buffer()?;
        Ok(Self {
            layout,
            arena,
            layers,
        })
    }

    pub(crate) fn layout(&self) -> Qwen3AsrKvCacheLayout {
        self.layout
    }

    pub(crate) fn layer(&self, layer_idx: usize) -> Option<Qwen3AsrPersistentKvLayer> {
        self.layers.get(layer_idx).copied()
    }

    pub(crate) fn arena(&self) -> &GgmlStaticTensorArena {
        &self.arena
    }
}

impl Qwen3AsrLayerKvCacheState {
    pub(crate) fn new(max_positions: usize, kv_heads: usize, head_dim: usize) -> Self {
        Self {
            max_positions,
            kv_heads,
            head_dim,
            keys: Vec::new(),
            values: Vec::new(),
            written_positions: 0,
        }
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

        Self::write_history_row(
            &mut self.keys,
            self.max_positions,
            self.kv_heads,
            self.head_dim,
            position,
            key,
        )?;
        Self::write_history_row(
            &mut self.values,
            self.max_positions,
            self.kv_heads,
            self.head_dim,
            position,
            value,
        )?;
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

        let per_head_len =
            token_count
                .checked_mul(self.head_dim)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "qwen kv-cache upload prefix overflow",
                })?;
        let per_head_stride =
            kv_span
                .checked_mul(self.head_dim)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "qwen kv-cache upload stride overflow",
                })?;

        Self::upload_storage_prefix(
            &self.keys,
            graph,
            key_history,
            self.max_positions,
            self.kv_heads,
            self.head_dim,
            per_head_len,
            per_head_stride,
            key_tensor_name,
        )?;
        Self::upload_storage_prefix(
            &self.values,
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

        let mut forked = Self::new(max_positions, self.kv_heads, self.head_dim);
        if written_positions == 0 {
            return Ok(forked);
        }
        let width = self.key_width();
        let old_expected_len = self
            .max_positions
            .checked_mul(width)
            .ok_or_else(|| "qwen3-asr kv-cache fork old length overflowed".to_string())?;
        if self.keys.len() != old_expected_len || self.values.len() != old_expected_len {
            return Err(format!(
                "qwen3-asr kv-cache fork storage length mismatch: keys={} values={} expected={old_expected_len}",
                self.keys.len(),
                self.values.len()
            ));
        }
        let new_len = max_positions
            .checked_mul(width)
            .ok_or_else(|| "qwen3-asr kv-cache fork new length overflowed".to_string())?;
        forked.keys = vec![0.0; new_len];
        forked.values = vec![0.0; new_len];
        Self::copy_history_prefix_to_span(
            &self.keys,
            &mut forked.keys,
            self.max_positions,
            max_positions,
            self.kv_heads,
            self.head_dim,
            written_positions,
        )?;
        Self::copy_history_prefix_to_span(
            &self.values,
            &mut forked.values,
            self.max_positions,
            max_positions,
            self.kv_heads,
            self.head_dim,
            written_positions,
        )?;
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

        let width = self.key_width();
        let old_expected_len = self
            .max_positions
            .checked_mul(width)
            .ok_or_else(|| "qwen3-asr kv-cache resize old length overflowed".to_string())?;
        if self.keys.len() != old_expected_len || self.values.len() != old_expected_len {
            return Err(format!(
                "qwen3-asr kv-cache resize storage length mismatch: keys={} values={} expected={old_expected_len}",
                self.keys.len(),
                self.values.len()
            ));
        }
        let new_len = new_max_positions
            .checked_mul(width)
            .ok_or_else(|| "qwen3-asr kv-cache resize new length overflowed".to_string())?;
        let mut keys = vec![0.0; new_len];
        let mut values = vec![0.0; new_len];
        Self::copy_history_prefix_to_span(
            &self.keys,
            &mut keys,
            self.max_positions,
            new_max_positions,
            self.kv_heads,
            self.head_dim,
            self.written_positions,
        )?;
        Self::copy_history_prefix_to_span(
            &self.values,
            &mut values,
            self.max_positions,
            new_max_positions,
            self.kv_heads,
            self.head_dim,
            self.written_positions,
        )?;
        self.max_positions = new_max_positions;
        self.keys = keys;
        self.values = values;
        Ok(())
    }

    pub(crate) fn full_history_storage(&self) -> Result<Qwen3AsrLayerKvCacheHistory<'_>, String> {
        let expected_len = self
            .max_positions
            .checked_mul(self.key_width())
            .ok_or_else(|| "qwen3-asr kv-cache storage length overflowed".to_string())?;
        if self.keys.len() != expected_len || self.values.len() != expected_len {
            return Err(format!(
                "qwen3-asr kv-cache storage length mismatch: keys={} values={} expected={}",
                self.keys.len(),
                self.values.len(),
                expected_len
            ));
        }
        Ok(Qwen3AsrLayerKvCacheHistory {
            max_positions: self.max_positions,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            written_positions: self.written_positions,
            keys: &self.keys,
            values: &self.values,
        })
    }

    #[cfg(test)]
    pub(crate) fn snapshot_written(&self) -> Result<Qwen3AsrLayerKvCacheSnapshot, String> {
        Ok(Qwen3AsrLayerKvCacheSnapshot {
            written_positions: self.written_positions,
            key_width: self.key_width(),
            value_width: self.value_width(),
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
            let key_len = self
                .max_positions
                .checked_mul(key_width)
                .ok_or_else(|| "qwen3-asr kv-cache key allocation overflowed".to_string())?;
            self.keys = vec![0.0; key_len];
        }
        if self.values.is_empty() {
            let value_len = self
                .max_positions
                .checked_mul(value_width)
                .ok_or_else(|| "qwen3-asr kv-cache value allocation overflowed".to_string())?;
            self.values = vec![0.0; value_len];
        }
        Ok(())
    }

    fn key_width(&self) -> usize {
        self.kv_heads.saturating_mul(self.head_dim)
    }

    fn value_width(&self) -> usize {
        self.key_width()
    }

    fn write_history_row(
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

    fn copy_history_prefix_to_span(
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

    fn upload_storage_prefix<'a>(
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_metadata() -> Qwen3AsrExecutionMetadata {
        Qwen3AsrExecutionMetadata {
            sample_rate_hz: 16_000,
            n_mels: 8,
            n_fft: 400,
            win_length: 400,
            hop_length: 160,
            audio_layers: 2,
            audio_d_model: 16,
            audio_heads: 2,
            llm_layers: 3,
            llm_d_model: 16,
            llm_heads: 2,
            llm_kv_heads: 2,
            llm_head_dim: 8,
            vocab_size: 32,
            llm_max_positions: 64,
            audio_start_token_id: 2,
            audio_end_token_id: 3,
            audio_pad_token_id: 4,
            eos_token_id: 0,
            pad_token_id: 6,
        }
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
            history.keys,
            &[
                1.0, 2.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                3.0, 4.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
        );
        assert_eq!(
            history.values,
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
            history.keys,
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
    fn persistent_kv_cache_allocates_all_layers() {
        let cache =
            Qwen3AsrPersistentKvCache::allocate(tiny_metadata()).expect("allocate kv cache");
        let layout = cache.layout();

        assert_eq!(layout.layers, 3);
        assert_eq!(layout.head_dim, 8);
        assert_eq!(layout.kv_heads, 2);
        assert_eq!(layout.max_positions, 64);
        assert!(cache.layer(0).is_some());
        assert!(cache.layer(2).is_some());
        assert!(cache.layer(3).is_none());

        let _ = cache.arena();
    }
}
