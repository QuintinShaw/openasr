//! Hands-off: single-responsibility ggml graph transcription, guarded by
//! golden/parity tests. Do not split this module for "tidiness" -- the tensor
//! wiring is validated as a whole and refactoring here risks silent numeric
//! drift.

use std::fmt;

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor, GgmlRopeExtParams,
    GgmlStaticTensor, GgmlStaticTensorArena, GgufTensorDataReadError, GgufTensorDataReader,
    env_toggle_with_raw,
};

use super::graph_config::qwen_runtime_graph_config;
use super::kv_cache::Qwen3AsrLayerKvCacheState;
use super::logits_head::{
    Qwen3AsrLlmFusedLogitsHeadSpec, first_max_argmax_reverse_indices,
    first_max_token_id_from_reversed_argmax,
};
use super::lora::{QwenLayerLoraSlots, QwenLoraAdapter, new_qwen_lora_slot};
use super::runtime_contract::Qwen3AsrExecutionMetadata;
use super::tensor_names::llm_layer_tensor_names;
use crate::nn::decoder::{
    LlmDecoderStackConfig, LlmDecoderStackInputs, LlmLayerWeights, LlmResidentKvArena,
    LlmReusableDecodeGraph, allocate_zeroed_llm_resident_kv_arena,
    build_fixed_kv_attention_mask_bits, build_fixed_kv_attention_mask_bits_for_query_rows,
    build_fixed_kv_attention_mask_bits_for_sequences, compose_llm_decoder_layer_stack,
    reusable_decode_graph_supported_for_runner,
};
use crate::nn::half::f32_slice_to_f16_bits;

const DEFAULT_RMS_NORM_EPSILON: f32 = 1e-6;
// The whole-step decoder builds all layers into one graph per token, so its
// graph context must hold ~layer_count x the per-layer node/intermediate budget.
const QWEN3_LLM_WHOLE_DECODE_GRAPH_CONTEXT_BYTES: usize = 768 * 1024 * 1024;
// Correctness escape hatch for backend kernels with divergent native GQA
// behavior; keep it unless every backend's GQA path is verified.
const QWEN3_LLM_NATIVE_GQA_ENV: &str = "OPENASR_QWEN_LLM_NATIVE_GQA";
const QWEN3_LLM_CPU_SAFE_PREFILL_QUERY_TOKENS: usize = 8;
const QWEN3_LLM_GPU_SAFE_PREFILL_QUERY_TOKENS: usize = 1;
const QWEN3_LLM_HIP_SAFE_PREFILL_QUERY_TOKENS: usize = 2;
const QWEN3_LLM_HIP_SHORT_PREFILL_QUERY_TOKENS: usize = 8;
const QWEN3_LLM_HIP_SHORT_PREFILL_MAX_TOKENS: usize = 32;
/// Prefill chunk for HIP-like backends past the 32-token flash window. Chunks
/// in this range bypass the buggy flash MMA/TILE kernel (the graph swaps to the
/// unfused `llm_naive_masked_attention` path when `n_query > 2` and
/// `n_kv > 32`), so correctness no longer bounds the width — but performance
/// does: ggml's HIP `mul_mat` only takes the fast mmvq vector kernels for
/// `n_query <= 8` and beyond that switches to MMQ, which is pathologically slow
/// on RDNA4 Windows (measured on gfx1200: 8-token chunks decode at ~3 ms/token
/// while 16/32/64-token chunks blow up to seconds per chunk). Keep the chunk at
/// the mmvq ceiling; do not widen without re-measuring an offset>0 chunk on a
/// HIP host.
const QWEN3_LLM_HIP_NONFLASH_PREFILL_QUERY_TOKENS: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen3AsrLlmAttentionCoreOutput {
    pub attn_hidden: Vec<f32>,
    pub projected_k: Vec<f32>,
    pub projected_v: Vec<f32>,
    pub qk_width: usize,
    pub q_width: usize,
    pub k_width: usize,
    pub v_width: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Qwen3AsrLlmDecodeAttentionHistory<'a> {
    pub key_rows: &'a [f32],
    pub value_rows: &'a [f32],
    pub token_count: usize,
    pub position: usize,
    pub rope_theta: f32,
}

#[derive(Debug, Clone)]
struct DenseProjectionWeight {
    input_width: usize,
    output_width: usize,
    values: Vec<f32>,
    layout: DenseProjectionLayout,
    raw_ggml: Option<OwnedGgmlProjectionWeight>,
}

#[derive(Debug, Clone)]
struct OwnedGgmlProjectionWeight {
    ggml_type: i32,
    dims: Vec<usize>,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
struct FusedQkvProjectionWeight {
    input_width: usize,
    output_width: usize,
    raw_ggml: Option<OwnedGgmlProjectionWeight>,
    values: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenseProjectionLayout {
    InputByOutput,
    OutputByInput,
}

#[derive(Debug, Error)]
pub(crate) enum Qwen3AsrLlmTransformerError {
    #[error("qwen3-asr llm transformer tensor read failed: {reason}")]
    TensorReadFailed { reason: String },
    #[error("qwen3-asr llm transformer tensor '{tensor_name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        tensor_name: String,
        shape: String,
        reason: String,
    },
    #[error(
        "qwen3-asr llm transformer hidden state has invalid shape: got {got}, expected hidden_size={expected}"
    )]
    InvalidHiddenStateShape { got: usize, expected: usize },
    #[error("qwen3-asr llm transformer tensor '{tensor_name}' contains non-finite values")]
    NonFiniteTensorValues { tensor_name: String },
    #[error("qwen3-asr llm transformer projection values contain non-finite numbers")]
    NonFiniteProjectionValues,
    #[error(
        "qwen3-asr llm transformer projection values are unavailable for tensor '{tensor_name}'"
    )]
    ProjectionValuesUnavailable { tensor_name: String },
    #[error("qwen3-asr llm transformer tensor '{tensor_name}' projection overflowed allocation")]
    AllocationOverflow { tensor_name: String },
    #[error(
        "qwen3-asr llm transformer q/k norm width mismatch: vector_width={vector_width}, norm_width={norm_width}"
    )]
    QkNormWidthMismatch {
        vector_width: usize,
        norm_width: usize,
    },
    #[error("qwen3-asr llm transformer attention core has incompatible q/k widths")]
    IncompatibleQkWidths,
    #[error("qwen3-asr llm transformer attention core produced non-finite score")]
    NonFiniteAttentionScore,
    #[error(
        "qwen3-asr llm transformer decode history shape is invalid: key_len={key_len} (expected {expected_key_len}), value_len={value_len} (expected {expected_value_len}), token_count={token_count}"
    )]
    InvalidDecodeHistoryShape {
        key_len: usize,
        expected_key_len: usize,
        value_len: usize,
        expected_value_len: usize,
        token_count: usize,
    },
    #[error(
        "qwen3-asr llm transformer ffn projection width mismatch: gate_width={gate_width}, up_width={up_width}"
    )]
    FfnProjectionWidthMismatch { gate_width: usize, up_width: usize },
}

#[derive(Debug, Clone)]
pub(crate) enum Qwen3AsrLlmLayerAttentionProjection {
    Generic(Qwen3AsrLlmLayerAttentionProjectionGeneric),
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3AsrLlmLayerAttentionProjectionGeneric {
    d_model: usize,
    /// Explicit head width, since it can no longer always be inferred from
    /// `q_norm_weight.len()` (Qwen2-shaped projections have none).
    head_dim: usize,
    attn_norm_name: String,
    attn_q_name: String,
    attn_k_name: String,
    attn_v_name: String,
    attn_output_name: String,
    /// Native (zero-copy-bindable) pack names for gate/up/down, needed at
    /// `Qwen3AsrLlmWholeDecoderGraphExecutor` construction time to re-bind
    /// these tensors zero-copy from a freshly-reopened `GgmlLoadedWeightContext`
    /// (see `bind_or_arena_llm`). Their host payload is dropped after load
    /// (`dropped_projection_payload`), so the pack name is the ONLY way to
    /// find them again -- callers must not fall back to a family-fixed
    /// naming scheme (e.g. qwen3-asr's own `blk.N.*`) here, or a differently-
    /// named family's pack (e.g. firered-llm's `llm.blk.N.*`) fails to bind.
    ffn_gate_name: String,
    ffn_up_name: String,
    ffn_down_name: String,
    attn_norm_weight: Vec<f32>,
    q_weight: DenseProjectionWeight,
    k_weight: DenseProjectionWeight,
    v_weight: DenseProjectionWeight,
    attn_output_weight: DenseProjectionWeight,
    ffn_norm_weight: Vec<f32>,
    ffn_gate_weight: DenseProjectionWeight,
    ffn_up_weight: DenseProjectionWeight,
    ffn_down_weight: DenseProjectionWeight,
    /// Empty ⇒ no QK-norm (Qwen2's shape); non-empty (== `head_dim`) ⇒
    /// QK-norm applied (Qwen3's shape). Both must agree (both empty or both
    /// `head_dim`-wide) -- validated at load time.
    q_norm_weight: Vec<f32>,
    k_norm_weight: Vec<f32>,
    /// Empty ⇒ no attention bias (Qwen3's shape); non-empty ⇒ bias applied
    /// (Qwen2's shape). Independent of the QK-norm flag above -- the two
    /// axes happen to be inverted between Qwen2 and Qwen3 but are not
    /// coupled in the representation.
    q_bias: Vec<f32>,
    k_bias: Vec<f32>,
    v_bias: Vec<f32>,
}

#[allow(dead_code)]
impl Qwen3AsrLlmLayerAttentionProjection {
    pub(crate) fn run_attention_core_for_decode_boundary(
        &self,
        hidden: &[f32],
    ) -> Result<Qwen3AsrLlmAttentionCoreOutput, Qwen3AsrLlmTransformerError> {
        match self {
            Self::Generic(inner) => inner.run_attention_core_for_decode_boundary(hidden),
        }
    }
}

impl Qwen3AsrLlmLayerAttentionProjectionGeneric {
    pub(crate) fn run_attention_core_for_decode_boundary(
        &self,
        hidden: &[f32],
    ) -> Result<Qwen3AsrLlmAttentionCoreOutput, Qwen3AsrLlmTransformerError> {
        run_attention_core(
            self.d_model,
            hidden,
            &self.attn_norm_weight,
            &self.q_weight,
            &self.k_weight,
            &self.v_weight,
            &self.attn_output_weight,
            &self.q_norm_weight,
            &self.k_norm_weight,
            &self.attn_norm_name,
            &self.attn_q_name,
            &self.attn_k_name,
            &self.attn_v_name,
            &self.attn_output_name,
        )
    }
}

fn run_attention_core(
    d_model: usize,
    hidden: &[f32],
    attn_norm_weight: &[f32],
    q_weight: &DenseProjectionWeight,
    k_weight: &DenseProjectionWeight,
    v_weight: &DenseProjectionWeight,
    attn_output_weight: &DenseProjectionWeight,
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    attn_norm_name: &str,
    attn_q_name: &str,
    attn_k_name: &str,
    attn_v_name: &str,
    attn_output_name: &str,
) -> Result<Qwen3AsrLlmAttentionCoreOutput, Qwen3AsrLlmTransformerError> {
    run_attention_core_with_history(
        d_model,
        hidden,
        attn_norm_weight,
        q_weight,
        k_weight,
        v_weight,
        attn_output_weight,
        q_norm_weight,
        k_norm_weight,
        attn_norm_name,
        attn_q_name,
        attn_k_name,
        attn_v_name,
        attn_output_name,
        Qwen3AsrLlmDecodeAttentionHistory {
            key_rows: &[],
            value_rows: &[],
            token_count: 0,
            position: 0,
            rope_theta: 1_000_000.0,
        },
    )
}

fn run_attention_core_with_history(
    d_model: usize,
    hidden: &[f32],
    attn_norm_weight: &[f32],
    q_weight: &DenseProjectionWeight,
    k_weight: &DenseProjectionWeight,
    v_weight: &DenseProjectionWeight,
    attn_output_weight: &DenseProjectionWeight,
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    attn_norm_name: &str,
    attn_q_name: &str,
    attn_k_name: &str,
    attn_v_name: &str,
    attn_output_name: &str,
    history: Qwen3AsrLlmDecodeAttentionHistory<'_>,
) -> Result<Qwen3AsrLlmAttentionCoreOutput, Qwen3AsrLlmTransformerError> {
    if hidden.len() != d_model {
        return Err(Qwen3AsrLlmTransformerError::InvalidHiddenStateShape {
            got: hidden.len(),
            expected: d_model,
        });
    }
    if hidden.iter().any(|value| !value.is_finite()) {
        return Err(Qwen3AsrLlmTransformerError::NonFiniteProjectionValues);
    }
    let normed = rms_norm_with_weight(
        hidden,
        attn_norm_weight,
        DEFAULT_RMS_NORM_EPSILON,
        attn_norm_name,
    )?;
    let mut q = q_weight.project_row(&normed, attn_q_name)?;
    let mut k = k_weight.project_row(&normed, attn_k_name)?;
    let v = v_weight.project_row(&normed, attn_v_name)?;
    apply_segmented_rms_norm_with_weight(&mut q, q_norm_weight, DEFAULT_RMS_NORM_EPSILON)?;
    apply_segmented_rms_norm_with_weight(&mut k, k_norm_weight, DEFAULT_RMS_NORM_EPSILON)?;
    apply_rope_neox_in_place(
        &mut q,
        head_dim_from_norm(q_norm_weight)?,
        history.position,
        history.rope_theta,
    )?;
    apply_rope_neox_in_place(
        &mut k,
        head_dim_from_norm(k_norm_weight)?,
        history.position,
        history.rope_theta,
    )?;
    if q.iter().any(|value| !value.is_finite())
        || k.iter().any(|value| !value.is_finite())
        || v.iter().any(|value| !value.is_finite())
    {
        return Err(Qwen3AsrLlmTransformerError::NonFiniteProjectionValues);
    }

    let qk_width = q.len().min(k.len());
    if qk_width == 0 || q_norm_weight.is_empty() {
        return Err(Qwen3AsrLlmTransformerError::IncompatibleQkWidths);
    }

    let q_width = q.len();
    let k_width = k.len();
    let v_width = v.len();
    let head_dim = q_norm_weight.len();
    if q_width % head_dim != 0 || k_width % head_dim != 0 || v_width % head_dim != 0 {
        return Err(Qwen3AsrLlmTransformerError::IncompatibleQkWidths);
    }
    let q_heads = q_width / head_dim;
    let kv_heads = k_width / head_dim;
    let value_heads = v_width / head_dim;
    if q_heads == 0 || kv_heads == 0 || value_heads == 0 || kv_heads != value_heads {
        return Err(Qwen3AsrLlmTransformerError::IncompatibleQkWidths);
    }
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(Qwen3AsrLlmTransformerError::IncompatibleQkWidths);
    }
    let expected_key_len = history.token_count.checked_mul(k_width).ok_or(
        Qwen3AsrLlmTransformerError::InvalidDecodeHistoryShape {
            key_len: history.key_rows.len(),
            expected_key_len: usize::MAX,
            value_len: history.value_rows.len(),
            expected_value_len: usize::MAX,
            token_count: history.token_count,
        },
    )?;
    let expected_value_len = history.token_count.checked_mul(v_width).ok_or(
        Qwen3AsrLlmTransformerError::InvalidDecodeHistoryShape {
            key_len: history.key_rows.len(),
            expected_key_len,
            value_len: history.value_rows.len(),
            expected_value_len: usize::MAX,
            token_count: history.token_count,
        },
    )?;
    if history.key_rows.len() != expected_key_len || history.value_rows.len() != expected_value_len
    {
        return Err(Qwen3AsrLlmTransformerError::InvalidDecodeHistoryShape {
            key_len: history.key_rows.len(),
            expected_key_len,
            value_len: history.value_rows.len(),
            expected_value_len,
            token_count: history.token_count,
        });
    }
    debug_assert!(history.key_rows.iter().all(|v| v.is_finite()));
    debug_assert!(history.value_rows.iter().all(|v| v.is_finite()));

    let q_per_kv_group = q_heads / kv_heads;
    let total_tokens = history.token_count.saturating_add(1);
    let scale = (head_dim as f32).sqrt().recip();
    let mut context = vec![0.0_f32; q_width];
    let mut scores = Vec::with_capacity(total_tokens);
    let mut weights = Vec::with_capacity(total_tokens);

    for q_head in 0..q_heads {
        let kv_head = q_head / q_per_kv_group;
        let q_base = q_head * head_dim;
        let q_slice = &q[q_base..q_base + head_dim];
        let kv_base = kv_head * head_dim;
        let history_head_base = kv_head * history.token_count * head_dim;

        scores.clear();
        for token_idx in 0..history.token_count {
            let key_row_base = history_head_base + token_idx * head_dim;
            let Some(key_row) = history.key_rows.get(key_row_base..key_row_base + head_dim) else {
                return Err(Qwen3AsrLlmTransformerError::InvalidDecodeHistoryShape {
                    key_len: history.key_rows.len(),
                    expected_key_len,
                    value_len: history.value_rows.len(),
                    expected_value_len,
                    token_count: history.token_count,
                });
            };
            let mut dot = 0.0_f32;
            for idx in 0..head_dim {
                dot += q_slice[idx] * key_row[idx];
            }
            let scaled = dot * scale;
            if !scaled.is_finite() {
                return Err(Qwen3AsrLlmTransformerError::NonFiniteAttentionScore);
            }
            scores.push(scaled);
        }
        let current_k = &k[kv_base..kv_base + head_dim];
        let mut current_score = 0.0_f32;
        for idx in 0..head_dim {
            current_score += q_slice[idx] * current_k[idx];
        }
        let current_scaled = current_score * scale;
        if !current_scaled.is_finite() {
            return Err(Qwen3AsrLlmTransformerError::NonFiniteAttentionScore);
        }
        scores.push(current_scaled);

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max_score.is_finite() {
            return Err(Qwen3AsrLlmTransformerError::NonFiniteAttentionScore);
        }

        weights.clear();
        let mut denom = 0.0_f32;
        for score in scores.iter().copied() {
            let weight = (score - max_score).exp();
            if !weight.is_finite() {
                return Err(Qwen3AsrLlmTransformerError::NonFiniteAttentionScore);
            }
            denom += weight;
            weights.push(weight);
        }
        if !denom.is_finite() || denom <= 0.0 {
            return Err(Qwen3AsrLlmTransformerError::NonFiniteAttentionScore);
        }

        let out_slice = &mut context[q_base..q_base + head_dim];
        for (token_idx, weight) in weights.iter().copied().enumerate() {
            let norm_weight = weight / denom;
            let value_slice = if token_idx < history.token_count {
                let value_row_base = history_head_base + token_idx * head_dim;
                let Some(value_slice) = history
                    .value_rows
                    .get(value_row_base..value_row_base + head_dim)
                else {
                    return Err(Qwen3AsrLlmTransformerError::InvalidDecodeHistoryShape {
                        key_len: history.key_rows.len(),
                        expected_key_len,
                        value_len: history.value_rows.len(),
                        expected_value_len,
                        token_count: history.token_count,
                    });
                };
                value_slice
            } else {
                &v[kv_base..kv_base + head_dim]
            };
            for idx in 0..head_dim {
                out_slice[idx] += norm_weight * value_slice[idx];
            }
        }
    }

    let attn_hidden = attn_output_weight.project_row(&context, attn_output_name)?;
    if attn_hidden.iter().any(|value| !value.is_finite()) {
        return Err(Qwen3AsrLlmTransformerError::NonFiniteProjectionValues);
    }
    Ok(Qwen3AsrLlmAttentionCoreOutput {
        attn_hidden,
        projected_k: k,
        projected_v: v,
        qk_width,
        q_width,
        k_width,
        v_width,
    })
}

fn head_dim_from_norm(norm_weight: &[f32]) -> Result<usize, Qwen3AsrLlmTransformerError> {
    if norm_weight.is_empty() || !norm_weight.len().is_multiple_of(2) {
        return Err(Qwen3AsrLlmTransformerError::IncompatibleQkWidths);
    }
    Ok(norm_weight.len())
}

fn apply_rope_neox_in_place(
    values: &mut [f32],
    head_dim: usize,
    position: usize,
    rope_theta: f32,
) -> Result<(), Qwen3AsrLlmTransformerError> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) || !values.len().is_multiple_of(head_dim) {
        return Err(Qwen3AsrLlmTransformerError::IncompatibleQkWidths);
    }
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err(Qwen3AsrLlmTransformerError::NonFiniteProjectionValues);
    }

    let half = head_dim / 2;
    let position = position as f32;
    for head in values.chunks_exact_mut(head_dim) {
        for pair_idx in 0..half {
            let exponent = (2.0_f32 * pair_idx as f32) / head_dim as f32;
            let angle = position * rope_theta.powf(-exponent);
            let (sin_theta, cos_theta) = angle.sin_cos();
            let x0 = head[pair_idx];
            let x1 = head[pair_idx + half];
            head[pair_idx] = x0 * cos_theta - x1 * sin_theta;
            head[pair_idx + half] = x0 * sin_theta + x1 * cos_theta;
        }
    }

    Ok(())
}

impl DenseProjectionWeight {
    #[cfg(test)]
    fn from_tensor(
        tensor_name: &str,
        dims: &[u64],
        values: Vec<f32>,
        expected_input_width: usize,
    ) -> Result<Self, Qwen3AsrLlmTransformerError> {
        if dims.len() != 2 {
            return Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
                tensor_name: tensor_name.to_string(),
                shape: render_shape(dims),
                reason: "expected rank-2 matrix".to_string(),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Qwen3AsrLlmTransformerError::NonFiniteTensorValues {
                tensor_name: tensor_name.to_string(),
            });
        }

        let dim0 = dims[0] as usize;
        let dim1 = dims[1] as usize;
        if dim0 == expected_input_width {
            return Self::new(
                tensor_name,
                expected_input_width,
                dim1,
                values,
                DenseProjectionLayout::OutputByInput,
                None,
            );
        }
        if dim1 == expected_input_width {
            return Self::new(
                tensor_name,
                expected_input_width,
                dim0,
                values,
                DenseProjectionLayout::InputByOutput,
                None,
            );
        }
        Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: render_shape(dims),
            reason: format!("expected one dimension to equal hidden_size={expected_input_width}"),
        })
    }

    fn new(
        tensor_name: &str,
        input_width: usize,
        output_width: usize,
        values: Vec<f32>,
        layout: DenseProjectionLayout,
        raw_ggml: Option<OwnedGgmlProjectionWeight>,
    ) -> Result<Self, Qwen3AsrLlmTransformerError> {
        if !values.is_empty() && values.iter().any(|value| !value.is_finite()) {
            return Err(Qwen3AsrLlmTransformerError::NonFiniteTensorValues {
                tensor_name: tensor_name.to_string(),
            });
        }
        Ok(Self {
            input_width,
            output_width,
            values,
            layout,
            raw_ggml,
        })
    }

    fn project_row(
        &self,
        input: &[f32],
        tensor_name: &str,
    ) -> Result<Vec<f32>, Qwen3AsrLlmTransformerError> {
        if input.len() != self.input_width {
            return Err(Qwen3AsrLlmTransformerError::InvalidHiddenStateShape {
                got: input.len(),
                expected: self.input_width,
            });
        }
        self.project_row_rust(input, tensor_name)
    }

    fn project_row_rust(
        &self,
        input: &[f32],
        tensor_name: &str,
    ) -> Result<Vec<f32>, Qwen3AsrLlmTransformerError> {
        let expected_values_len = self.input_width.checked_mul(self.output_width).ok_or(
            Qwen3AsrLlmTransformerError::AllocationOverflow {
                tensor_name: tensor_name.to_string(),
            },
        )?;
        if self.values.len() != expected_values_len {
            return Err(Qwen3AsrLlmTransformerError::ProjectionValuesUnavailable {
                tensor_name: tensor_name.to_string(),
            });
        }
        let mut out = vec![0.0_f32; self.output_width];
        match self.layout {
            DenseProjectionLayout::InputByOutput => {
                for (input_idx, input_value) in input.iter().copied().enumerate() {
                    let row_start = input_idx.checked_mul(self.output_width).ok_or(
                        Qwen3AsrLlmTransformerError::AllocationOverflow {
                            tensor_name: tensor_name.to_string(),
                        },
                    )?;
                    let row = &self.values[row_start..row_start + self.output_width];
                    for (out_idx, weight) in row.iter().copied().enumerate() {
                        out[out_idx] += input_value * weight;
                    }
                }
            }
            DenseProjectionLayout::OutputByInput => {
                for (out_idx, out_value) in out.iter_mut().enumerate() {
                    let row_start = out_idx.checked_mul(self.input_width).ok_or(
                        Qwen3AsrLlmTransformerError::AllocationOverflow {
                            tensor_name: tensor_name.to_string(),
                        },
                    )?;
                    let row = &self.values[row_start..row_start + self.input_width];
                    let mut acc = 0.0_f32;
                    for (input_idx, weight) in row.iter().copied().enumerate() {
                        acc += input[input_idx] * weight;
                    }
                    *out_value = acc;
                }
            }
        }
        Ok(out)
    }
}

impl FusedQkvProjectionWeight {
    fn new(
        q_weight: &DenseProjectionWeight,
        k_weight: &DenseProjectionWeight,
        v_weight: &DenseProjectionWeight,
    ) -> Result<Option<Self>, GgmlCpuGraphError> {
        if q_weight.input_width != k_weight.input_width
            || q_weight.input_width != v_weight.input_width
        {
            return Ok(None);
        }

        let input_width = q_weight.input_width;
        let output_width = q_weight
            .output_width
            .checked_add(k_weight.output_width)
            .and_then(|value| value.checked_add(v_weight.output_width))
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "fused qkv projection width overflow",
            })?;

        if let Some(raw_ggml) = fuse_raw_qkv_projection_weights(q_weight, k_weight, v_weight)? {
            return Ok(Some(Self {
                input_width,
                output_width,
                raw_ggml: Some(raw_ggml),
                values: None,
            }));
        }

        // Dense f32 fallback: every contributing projection must carry
        // materialized values. Fail closed rather than concatenating a
        // short buffer if any weight is raw-only (e.g. a mixed raw/dense state).
        if q_weight.values.is_empty() || k_weight.values.is_empty() || v_weight.values.is_empty() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "fused qkv dense fallback requires materialized q/k/v values",
            });
        }

        let q_values = projection_values_for_ggml(
            q_weight.input_width,
            q_weight.output_width,
            &q_weight.values,
            q_weight.layout,
        )?;
        let k_values = projection_values_for_ggml(
            k_weight.input_width,
            k_weight.output_width,
            &k_weight.values,
            k_weight.layout,
        )?;
        let v_values = projection_values_for_ggml(
            v_weight.input_width,
            v_weight.output_width,
            &v_weight.values,
            v_weight.layout,
        )?;
        let mut values = Vec::with_capacity(output_width * input_width);
        values.extend_from_slice(&q_values);
        values.extend_from_slice(&k_values);
        values.extend_from_slice(&v_values);
        Ok(Some(Self {
            input_width,
            output_width,
            raw_ggml: None,
            values: Some(values),
        }))
    }
}

/// Default policy for the ggml-native GQA attention path (the KV-head broadcast
/// is done inside `flash_attn_ext`/`mul_mat` instead of by host-side head
/// expansion).
///
/// Correct and faster on CPU and Metal. On the discrete-GPU lane it is NOT
/// universally correct: the ROCm/HIP kernels mis-compute the GQA broadcast on
/// RDNA4 (measured on gfx1200 — qwen output degenerates into repeated garbage
/// tokens), and CUDA/Vulkan have never been validated for it. So it defaults OFF
/// on the discrete-GPU lane (attention falls back to the unfused head-expansion
/// path, the CPU/Metal-reference-correct attention) and ON for CPU and Metal.
///
/// A discrete GPU is re-enabled only after `tooling/qwen-gpu-parity` proves its
/// GPU transcript matches the CPU reference (a synthetic runtime self-check was
/// tried and rejected: it can false-pass when its probe shape does not hit the
/// exact op the real decoder mis-computes). `OPENASR_QWEN_LLM_NATIVE_GQA`
/// overrides the default either way.
fn qwen_llm_native_gqa_default_for_backend(backend: GgmlCpuGraphBackend) -> bool {
    !matches!(backend, GgmlCpuGraphBackend::Gpu)
}

fn qwen_llm_native_gqa_enabled(raw: Option<&str>, default_enabled: bool) -> bool {
    env_toggle_with_raw(None, raw, default_enabled)
}

/// Resolve whether to use native GQA on `backend`: `OPENASR_QWEN_LLM_NATIVE_GQA`
/// wins in either direction; otherwise the per-backend default applies.
fn qwen_llm_resolve_use_native_gqa(backend: GgmlCpuGraphBackend) -> bool {
    qwen_llm_native_gqa_enabled(
        std::env::var(QWEN3_LLM_NATIVE_GQA_ENV).ok().as_deref(),
        qwen_llm_native_gqa_default_for_backend(backend),
    )
}

/// A decode-layer 2D projection weight: either an arena tensor (f32-uploaded) or
/// a zero-copy leaf bound to the mmap'd pack (native q8/f16, no host copy). The
/// goals 7+8 LLM lever binds `output`/`gate`/`up`/`down` as `Loaded` to drop
/// their resident host bytes + per-encode arena copy; QKV/q/k/v stay `Arena`
/// (the fused-QKV synthetic tensor has no on-disk counterpart).
#[derive(Clone, Copy)]
enum LlmWeightHandle {
    Arena(GgmlStaticTensor),
    Loaded(crate::ggml_runtime::GgmlLoadedTensor),
}

impl LlmWeightHandle {
    fn as_graph_tensor<'a>(self, arena: &GgmlStaticTensorArena) -> GgmlCpuTensor<'a> {
        match self {
            Self::Arena(handle) => arena.graph_tensor(handle),
            Self::Loaded(tensor) => tensor.as_graph_tensor(),
        }
    }
    fn arena_handle(self) -> Option<GgmlStaticTensor> {
        match self {
            Self::Arena(handle) => Some(handle),
            Self::Loaded(_) => None,
        }
    }
}

/// Resident weight handles for one decode layer, allocated into a shared arena.
struct Qwen3AsrLlmLayerWeightHandles {
    attn_norm_weight: GgmlStaticTensor,
    qkv_weight: Option<GgmlStaticTensor>,
    q_weight: GgmlStaticTensor,
    k_weight: GgmlStaticTensor,
    v_weight: GgmlStaticTensor,
    /// `Some` only for a Qwen2-shaped projection (attention bias); Qwen3-ASR
    /// leaves these `None`.
    q_bias: Option<GgmlStaticTensor>,
    k_bias: Option<GgmlStaticTensor>,
    v_bias: Option<GgmlStaticTensor>,
    output_weight: LlmWeightHandle,
    /// `None` only for a Qwen2-shaped projection (no QK-norm); Qwen3-ASR
    /// always populates these.
    q_norm_weight: Option<GgmlStaticTensor>,
    k_norm_weight: Option<GgmlStaticTensor>,
    ffn_norm_weight: GgmlStaticTensor,
    gate_weight: LlmWeightHandle,
    up_weight: LlmWeightHandle,
    down_weight: LlmWeightHandle,
    /// Optional LoRA side-path slots (all `None` when no adapter is active).
    lora: QwenLayerLoraSlots,
}

struct Qwen3AsrLlmFusedLogitsHeadHandles {
    vocab_size: usize,
    rms_norm_epsilon: f32,
    output_norm_weight: GgmlStaticTensor,
    output_weight: LlmWeightHandle,
    argmax_reverse_indices: GgmlStaticTensor,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Qwen3AsrLlmDecodeDims {
    d_model: usize,
    q_width: usize,
    k_width: usize,
    v_width: usize,
    head_dim: usize,
    q_heads: usize,
    kv_heads: usize,
}

fn qwen_llm_stack_config(
    dims: Qwen3AsrLlmDecodeDims,
    rope: GgmlRopeExtParams,
    use_native_gqa: bool,
    rms_norm_epsilon: f32,
    token_count: usize,
    n_seq: usize,
    use_flash_attention: bool,
) -> LlmDecoderStackConfig {
    LlmDecoderStackConfig {
        d_model: dims.d_model,
        head_dim: dims.head_dim,
        q_heads: dims.q_heads,
        kv_heads: dims.kv_heads,
        q_width: dims.q_width,
        k_width: dims.k_width,
        v_width: dims.v_width,
        token_count,
        n_seq,
        rms_norm_epsilon,
        rope,
        // Batched (n_seq > 1) decode requires the native-GQA layout. NOTE: on the
        // discrete-GPU lane native GQA mis-computes on RDNA4 (see
        // `qwen_llm_native_gqa_default_for_backend`), so multi-sequence serve
        // batching on such GPUs can still garble; single-stream (n_seq == 1)
        // honours the backend-aware default and is correct. Fixing batched GPU
        // serve needs n_seq > 1 support in the unfused head-expansion path.
        use_native_gqa: use_native_gqa || n_seq > 1,
        use_flash_attention,
    }
}

fn qwen_llm_layer_weights<'a>(
    layer: &Qwen3AsrLlmLayerWeightHandles,
    arena: &GgmlStaticTensorArena,
) -> LlmLayerWeights<'a> {
    qwen_llm_layer_weights_with_lora(layer, arena)
}

fn qwen_llm_layer_weights_with_lora<'a>(
    layer: &Qwen3AsrLlmLayerWeightHandles,
    arena: &GgmlStaticTensorArena,
) -> LlmLayerWeights<'a> {
    use crate::nn::decoder::LlmLoraSlot;
    // Helper: convert an arena-resident QwenLoraSlot to a graph-level LlmLoraSlot.
    let to_graph = |s: crate::models::qwen::lora::QwenLoraSlot| -> LlmLoraSlot<'a> {
        LlmLoraSlot {
            a: arena.graph_tensor(s.a),
            b_scaled: arena.graph_tensor(s.b_scaled),
        }
    };
    LlmLayerWeights {
        attn_norm_weight: arena.graph_tensor(layer.attn_norm_weight),
        qkv_weight: layer.qkv_weight.map(|weight| weight.as_graph_tensor()),
        q_weight: layer.q_weight.as_graph_tensor(),
        k_weight: layer.k_weight.as_graph_tensor(),
        v_weight: layer.v_weight.as_graph_tensor(),
        q_bias: layer.q_bias.map(|t| arena.graph_tensor(t)),
        k_bias: layer.k_bias.map(|t| arena.graph_tensor(t)),
        v_bias: layer.v_bias.map(|t| arena.graph_tensor(t)),
        q_norm_weight: layer.q_norm_weight.map(|t| arena.graph_tensor(t)),
        k_norm_weight: layer.k_norm_weight.map(|t| arena.graph_tensor(t)),
        output_weight: layer.output_weight.as_graph_tensor(arena),
        ffn_norm_weight: arena.graph_tensor(layer.ffn_norm_weight),
        ffn_gate_weight: layer.gate_weight.as_graph_tensor(arena),
        ffn_up_weight: layer.up_weight.as_graph_tensor(arena),
        ffn_down_weight: layer.down_weight.as_graph_tensor(arena),
        q_lora: layer.lora.attn_q.map(to_graph),
        k_lora: layer.lora.attn_k.map(to_graph),
        v_lora: layer.lora.attn_v.map(to_graph),
        output_lora: layer.lora.attn_output.map(to_graph),
        ffn_gate_lora: layer.lora.ffn_gate.map(to_graph),
        ffn_up_lora: layer.lora.ffn_up.map(to_graph),
        ffn_down_lora: layer.lora.ffn_down.map(to_graph),
    }
}

pub(crate) struct Qwen3AsrLlmWholeStepOutput {
    pub hidden: Vec<f32>,
    pub layer_kv: Vec<(Vec<f32>, Vec<f32>)>,
    /// Microseconds spent building the graph (start_graph + appending all layer
    /// ops + KV uploads) vs the single compute/dispatch — for decode profiling.
    pub build_micros: u128,
    pub compute_micros: u128,
}

pub(crate) struct Qwen3AsrLlmWholeStepTop1Output {
    pub token_id: u32,
    pub layer_kv: Vec<(Vec<f32>, Vec<f32>)>,
    pub build_micros: u128,
    pub compute_micros: u128,
}

/// Validate and ALLOCATE (but do not upload) one decode layer's weight tensors
/// into `arena`. All layers must be allocated before ANY upload, because the
/// first upload freezes the arena's backend buffer (no further new_tensor). The
/// returned FusedQkvProjectionWeight is carried to the upload phase.
#[allow(clippy::too_many_arguments)]
fn allocate_decode_layer_tensors(
    arena: &mut GgmlStaticTensorArena,
    loaded: Option<&crate::ggml_runtime::GgmlLoadedWeightContext>,
    attn_norm_weight: &[f32],
    q_weight: &DenseProjectionWeight,
    k_weight: &DenseProjectionWeight,
    v_weight: &DenseProjectionWeight,
    // Empty ⇒ no bias (Qwen3's shape); non-empty ⇒ bias applied (Qwen2's shape).
    q_bias: &[f32],
    k_bias: &[f32],
    v_bias: &[f32],
    output_weight: &DenseProjectionWeight,
    // Empty ⇒ no QK-norm (Qwen2's shape); non-empty ⇒ QK-norm applied
    // (Qwen3's shape). `head_dim` is always required explicitly since it can
    // no longer be inferred from `q_norm_weight.len()` when norm is disabled.
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    head_dim: usize,
    ffn_norm_weight: &[f32],
    ffn_gate_weight: &DenseProjectionWeight,
    ffn_up_weight: &DenseProjectionWeight,
    ffn_down_weight: &DenseProjectionWeight,
    // Native (zero-copy-bindable) tensor names for output/gate/up/down --
    // callers own their family's tensor-naming scheme (qwen's `blk.N.*` vs
    // firered-llm's `llm.blk.N.*`), this function stays name-agnostic.
    output_weight_tensor_name: &str,
    ffn_gate_tensor_name: &str,
    ffn_up_tensor_name: &str,
    ffn_down_tensor_name: &str,
) -> Result<
    (
        Qwen3AsrLlmLayerWeightHandles,
        Qwen3AsrLlmDecodeDims,
        Option<FusedQkvProjectionWeight>,
    ),
    GgmlCpuGraphError,
> {
    let d_model = attn_norm_weight.len();
    if d_model == 0 || ffn_norm_weight.len() != d_model {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "decode layer norm weight width mismatch",
        });
    }
    let has_qk_norm = !q_norm_weight.is_empty() || !k_norm_weight.is_empty();
    if has_qk_norm && (q_norm_weight.len() != head_dim || k_norm_weight.len() != head_dim) {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "decode layer q/k norm width mismatch",
        });
    }
    if head_dim == 0 {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "decode layer head_dim must be positive",
        });
    }
    if q_weight.input_width != d_model
        || k_weight.input_width != d_model
        || v_weight.input_width != d_model
        || ffn_gate_weight.input_width != d_model
        || ffn_up_weight.input_width != d_model
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "decode layer input width mismatch",
        });
    }
    if !q_weight.output_width.is_multiple_of(head_dim)
        || !k_weight.output_width.is_multiple_of(head_dim)
        || !v_weight.output_width.is_multiple_of(head_dim)
        || k_weight.output_width != v_weight.output_width
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "decode layer q/k/v head shape mismatch",
        });
    }
    let has_qkv_bias = !q_bias.is_empty() || !k_bias.is_empty() || !v_bias.is_empty();
    if has_qkv_bias
        && (q_bias.len() != q_weight.output_width
            || k_bias.len() != k_weight.output_width
            || v_bias.len() != v_weight.output_width)
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "decode layer q/k/v bias width mismatch",
        });
    }
    if output_weight.input_width != q_weight.output_width
        || output_weight.output_width != d_model
        || ffn_gate_weight.output_width != ffn_up_weight.output_width
        || ffn_down_weight.input_width != ffn_gate_weight.output_width
        || ffn_down_weight.output_width != d_model
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "decode layer output projection shape mismatch",
        });
    }
    let q_heads = q_weight.output_width / head_dim;
    let kv_heads = k_weight.output_width / head_dim;
    if q_heads == 0 || kv_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "decode layer q/kv head ratio mismatch",
        });
    }
    // Bias forces the split (non-fused) QKV path (see `nn::decoder::LlmLayerWeights`'
    // doc comment) -- never build a fused-QKV synthetic tensor when bias is present.
    let allow_fused_qkv = !has_qkv_bias;

    let attn_norm = arena.new_tensor_2d_f32(d_model, 1, "qwen_llm_decode_attn_norm_weight")?;
    let q_norm = has_qk_norm
        .then(|| arena.new_tensor_2d_f32(head_dim, 1, "qwen_llm_decode_q_norm_weight"))
        .transpose()?;
    let k_norm = has_qk_norm
        .then(|| arena.new_tensor_2d_f32(head_dim, 1, "qwen_llm_decode_k_norm_weight"))
        .transpose()?;
    let q_bias_tensor = has_qkv_bias
        .then(|| arena.new_tensor_2d_f32(q_weight.output_width, 1, "qwen_llm_decode_q_bias"))
        .transpose()?;
    let k_bias_tensor = has_qkv_bias
        .then(|| arena.new_tensor_2d_f32(k_weight.output_width, 1, "qwen_llm_decode_k_bias"))
        .transpose()?;
    let v_bias_tensor = has_qkv_bias
        .then(|| arena.new_tensor_2d_f32(v_weight.output_width, 1, "qwen_llm_decode_v_bias"))
        .transpose()?;
    let ffn_norm = arena.new_tensor_2d_f32(d_model, 1, "qwen_llm_decode_ffn_norm_weight")?;
    let fused_qkv_weight = if allow_fused_qkv {
        FusedQkvProjectionWeight::new(q_weight, k_weight, v_weight)?
    } else {
        None
    };
    let qkv_weight_tensor = fused_qkv_weight
        .as_ref()
        .map(|weight| new_fused_qkv_tensor_in_arena(arena, weight, "qwen_llm_decode_qkv_weight"))
        .transpose()?;
    let q_weight_tensor =
        new_projection_tensor_in_arena(arena, q_weight, "qwen_llm_decode_q_weight")?;
    let k_weight_tensor =
        new_projection_tensor_in_arena(arena, k_weight, "qwen_llm_decode_k_weight")?;
    let v_weight_tensor =
        new_projection_tensor_in_arena(arena, v_weight, "qwen_llm_decode_v_weight")?;
    // Bind output/gate/up/down zero-copy from the mmap'd pack when present
    // (native q8/f16, no arena copy); else allocate an arena tensor. These four
    // are unentangled with the fused-QKV path. q/k/v stay arena (they feed the
    // fused-QKV synthetic tensor, which has no on-disk counterpart).
    let output_weight_tensor = bind_or_arena_llm(
        arena,
        loaded,
        output_weight,
        output_weight_tensor_name,
        "qwen_llm_decode_output_weight",
    )?;
    let gate_weight_tensor = bind_or_arena_llm(
        arena,
        loaded,
        ffn_gate_weight,
        ffn_gate_tensor_name,
        "qwen_llm_decode_gate_weight",
    )?;
    let up_weight_tensor = bind_or_arena_llm(
        arena,
        loaded,
        ffn_up_weight,
        ffn_up_tensor_name,
        "qwen_llm_decode_up_weight",
    )?;
    let down_weight_tensor = bind_or_arena_llm(
        arena,
        loaded,
        ffn_down_weight,
        ffn_down_tensor_name,
        "qwen_llm_decode_down_weight",
    )?;

    Ok((
        Qwen3AsrLlmLayerWeightHandles {
            attn_norm_weight: attn_norm,
            qkv_weight: qkv_weight_tensor,
            q_weight: q_weight_tensor,
            k_weight: k_weight_tensor,
            v_weight: v_weight_tensor,
            q_bias: q_bias_tensor,
            k_bias: k_bias_tensor,
            v_bias: v_bias_tensor,
            output_weight: output_weight_tensor,
            q_norm_weight: q_norm,
            k_norm_weight: k_norm,
            ffn_norm_weight: ffn_norm,
            gate_weight: gate_weight_tensor,
            up_weight: up_weight_tensor,
            down_weight: down_weight_tensor,
            // LoRA slots are populated by the caller after this returns.
            lora: QwenLayerLoraSlots::default(),
        },
        Qwen3AsrLlmDecodeDims {
            d_model,
            q_width: q_weight.output_width,
            k_width: k_weight.output_width,
            v_width: v_weight.output_width,
            head_dim,
            q_heads,
            kv_heads,
        },
        fused_qkv_weight,
    ))
}

/// Bind a decode 2D projection zero-copy from `loaded` (mmap'd pack, native
/// type) when present; else allocate an arena tensor. A `Loaded` handle carries
/// its mmap'd data already (no upload); an `Arena` handle is uploaded later.
fn bind_or_arena_llm(
    arena: &GgmlStaticTensorArena,
    loaded: Option<&crate::ggml_runtime::GgmlLoadedWeightContext>,
    weight: &DenseProjectionWeight,
    tensor_pack_name: &str,
    tensor_name: &'static str,
) -> Result<LlmWeightHandle, GgmlCpuGraphError> {
    // Only weights stored as native [in,out] (raw_ggml present — the loader
    // validated this orientation) are safe to bind zero-copy: the mmap'd dims
    // match what `mul_mat` expects. f32-fallback weights may sit [out,in] on
    // disk and depend on the arena path's transpose, so are NEVER bound.
    if weight.raw_ggml.is_some() {
        return match loaded.and_then(|context| context.tensor(tensor_pack_name)) {
            Some(tensor) => Ok(LlmWeightHandle::Loaded(tensor)),
            None => Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "decode native 2D weight could not be bound zero-copy (host payload was dropped)",
            }),
        };
    }
    Ok(LlmWeightHandle::Arena(new_projection_tensor_in_arena(
        arena,
        weight,
        tensor_name,
    )?))
}

/// Allocate LoRA A/B slots for one layer into the arena, collecting upload
/// payloads in `pending_uploads`.  Returns a `QwenLayerLoraSlots` with all
/// slots populated for matching targets, or `None` slots for non-targeted ones.
///
/// This must run during Pass 1 (before any upload), because allocating tensors
/// after the first upload freezes the backend buffer.
///
/// Target names come from the caller (the loaded projection's own recorded
/// pack names), not a family-fixed scheme -- the same "callers own their
/// family's tensor-naming scheme" rule `allocate_decode_layer_tensors` follows
/// for the zero-copy re-bind names above. `llm_layer_tensor_names(layer_index)`
/// only matches qwen3-asr's own `blk.N.*` on-disk names; a differently-prefixed
/// family's pack (e.g. firered-llm's `llm.blk.N.*`) would silently look up the
/// wrong LoRA target and drop the adapter for that tensor.
#[allow(clippy::too_many_arguments)]
fn allocate_layer_lora_slots(
    arena: &GgmlStaticTensorArena,
    adapter: Option<&QwenLoraAdapter>,
    attn_q_name: &str,
    attn_k_name: &str,
    attn_v_name: &str,
    attn_output_name: &str,
    ffn_gate_name: &str,
    ffn_up_name: &str,
    ffn_down_name: &str,
    pending_uploads: &mut Vec<(GgmlStaticTensor, Vec<f32>, &'static str)>,
) -> Result<QwenLayerLoraSlots, GgmlCpuGraphError> {
    let Some(adapter) = adapter else {
        return Ok(QwenLayerLoraSlots::default());
    };
    let mut slots = QwenLayerLoraSlots::default();
    // Allocate one LoRA slot for `target_name`, pushing the upload payload.
    let mut maybe_slot =
        |target_name: &str| -> Result<Option<super::lora::QwenLoraSlot>, GgmlCpuGraphError> {
            let Some(target) = adapter.target(target_name) else {
                return Ok(None);
            };
            let slot = new_qwen_lora_slot(arena, target, "qwen_lora_a", "qwen_lora_b")?;
            pending_uploads.push((slot.a, target.a_values.clone(), "qwen_lora_a"));
            pending_uploads.push((slot.b_scaled, target.b_scaled_values.clone(), "qwen_lora_b"));
            Ok(Some(slot))
        };
    slots.attn_q = maybe_slot(attn_q_name)?;
    slots.attn_k = maybe_slot(attn_k_name)?;
    slots.attn_v = maybe_slot(attn_v_name)?;
    slots.attn_output = maybe_slot(attn_output_name)?;
    slots.ffn_gate = maybe_slot(ffn_gate_name)?;
    slots.ffn_up = maybe_slot(ffn_up_name)?;
    slots.ffn_down = maybe_slot(ffn_down_name)?;
    Ok(slots)
}

fn allocate_fused_logits_head_tensors(
    arena: &mut GgmlStaticTensorArena,
    loaded: Option<&crate::ggml_runtime::GgmlLoadedWeightContext>,
    dims: Qwen3AsrLlmDecodeDims,
    spec: &Qwen3AsrLlmFusedLogitsHeadSpec<'_>,
) -> Result<Qwen3AsrLlmFusedLogitsHeadHandles, GgmlCpuGraphError> {
    if spec.d_model != dims.d_model {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "fused logits head hidden width mismatch",
        });
    }
    if spec.output_norm_weight.len() != dims.d_model {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "fused logits head norm width mismatch",
        });
    }
    if spec.output_weight_dims != [dims.d_model, spec.vocab_size] {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "fused logits head requires direct [hidden, vocab] output weight",
        });
    }
    if !spec.rms_norm_epsilon.is_finite() || spec.rms_norm_epsilon <= 0.0 {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "fused logits head rms norm epsilon must be finite and positive",
        });
    }

    let output_norm_weight =
        arena.new_tensor_1d_f32(dims.d_model, "qwen_llm_fused_output_norm_weight")?;
    let argmax_reverse_indices =
        arena.new_tensor_1d_i32(spec.vocab_size, "qwen_llm_fused_argmax_reverse_indices")?;
    let output_weight =
        match loaded.and_then(|context| context.tensor(spec.output_weight_tensor_name)) {
            Some(tensor) => LlmWeightHandle::Loaded(tensor),
            None => LlmWeightHandle::Arena(arena.new_matmul_weight_2d_typed(
                dims.d_model,
                spec.vocab_size,
                spec.output_weight_ggml_type,
                "qwen_llm_fused_output_weight",
            )?),
        };

    Ok(Qwen3AsrLlmFusedLogitsHeadHandles {
        vocab_size: spec.vocab_size,
        rms_norm_epsilon: spec.rms_norm_epsilon,
        output_norm_weight,
        output_weight,
        argmax_reverse_indices,
    })
}

fn upload_fused_logits_head_weights(
    arena: &mut GgmlStaticTensorArena,
    handles: &Qwen3AsrLlmFusedLogitsHeadHandles,
    spec: &Qwen3AsrLlmFusedLogitsHeadSpec<'_>,
) -> Result<(), GgmlCpuGraphError> {
    arena.set_f32_slice(
        handles.output_norm_weight,
        spec.output_norm_weight,
        "qwen_llm_fused_output_norm_weight",
    )?;
    arena.set_i32_slice(
        handles.argmax_reverse_indices,
        &first_max_argmax_reverse_indices(spec.vocab_size)?,
        "qwen_llm_fused_argmax_reverse_indices",
    )?;
    if let Some(output_weight) = handles.output_weight.arena_handle() {
        arena.set_bytes_slice(
            output_weight,
            spec.output_weight_bytes,
            "qwen_llm_fused_output_weight",
        )?;
    }
    Ok(())
}

fn build_fused_logits_top1<'a>(
    arena: &GgmlStaticTensorArena,
    logits_head: &Qwen3AsrLlmFusedLogitsHeadHandles,
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    state: GgmlCpuTensor<'a>,
    n_seq: usize,
) -> Result<GgmlCpuTensor<'a>, GgmlCpuGraphError> {
    if n_seq != 1 {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "fused whole-decoder top1 currently requires n_seq=1",
        });
    }
    let normed = graph.rms_norm(state, logits_head.rms_norm_epsilon)?;
    let normed = graph.mul(normed, arena.graph_tensor(logits_head.output_norm_weight))?;
    let logits = graph.mul_mat(logits_head.output_weight.as_graph_tensor(arena), normed)?;
    graph.top1_argmax_first_max_reversed(
        logits,
        arena.graph_tensor(logits_head.argmax_reverse_indices),
    )
}

fn validate_fused_top1_token_id(
    reversed_token_id: i32,
    vocab_size: usize,
) -> Result<u32, GgmlCpuGraphError> {
    let token_id =
        first_max_token_id_from_reversed_argmax(reversed_token_id, vocab_size).map_err(|_| {
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder fused top1 token id is outside vocab size",
            }
        })?;
    u32::try_from(token_id).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
        reason: "whole-decoder fused top1 token id is outside vocab size",
    })
}

/// Upload one decode layer's weight data into the previously-allocated arena
/// handles. Must run AFTER all layers' tensors are allocated (the first upload
/// freezes the arena's backend buffer).
#[allow(clippy::too_many_arguments)]
fn upload_decode_layer_weights(
    arena: &mut GgmlStaticTensorArena,
    handles: &Qwen3AsrLlmLayerWeightHandles,
    fused_qkv_weight: Option<&FusedQkvProjectionWeight>,
    attn_norm_weight: &[f32],
    q_weight: &DenseProjectionWeight,
    k_weight: &DenseProjectionWeight,
    v_weight: &DenseProjectionWeight,
    q_bias: &[f32],
    k_bias: &[f32],
    v_bias: &[f32],
    output_weight: &DenseProjectionWeight,
    q_norm_weight: &[f32],
    k_norm_weight: &[f32],
    ffn_norm_weight: &[f32],
    ffn_gate_weight: &DenseProjectionWeight,
    ffn_up_weight: &DenseProjectionWeight,
    ffn_down_weight: &DenseProjectionWeight,
) -> Result<(), GgmlCpuGraphError> {
    arena.set_f32_slice(
        handles.attn_norm_weight,
        attn_norm_weight,
        "qwen_llm_decode_attn_norm_weight",
    )?;
    if let Some(tensor) = handles.q_norm_weight {
        arena.set_f32_slice(tensor, q_norm_weight, "qwen_llm_decode_q_norm_weight")?;
    }
    if let Some(tensor) = handles.k_norm_weight {
        arena.set_f32_slice(tensor, k_norm_weight, "qwen_llm_decode_k_norm_weight")?;
    }
    if let Some(tensor) = handles.q_bias {
        arena.set_f32_slice(tensor, q_bias, "qwen_llm_decode_q_bias")?;
    }
    if let Some(tensor) = handles.k_bias {
        arena.set_f32_slice(tensor, k_bias, "qwen_llm_decode_k_bias")?;
    }
    if let Some(tensor) = handles.v_bias {
        arena.set_f32_slice(tensor, v_bias, "qwen_llm_decode_v_bias")?;
    }
    arena.set_f32_slice(
        handles.ffn_norm_weight,
        ffn_norm_weight,
        "qwen_llm_decode_ffn_norm_weight",
    )?;
    if let (Some(tensor), Some(weight)) = (handles.qkv_weight, fused_qkv_weight) {
        upload_fused_qkv_weight_to_arena(arena, tensor, weight, "qwen_llm_decode_qkv_weight")?;
    }
    upload_projection_weight_to_arena(
        arena,
        handles.q_weight,
        q_weight,
        "qwen_llm_decode_q_weight",
    )?;
    upload_projection_weight_to_arena(
        arena,
        handles.k_weight,
        k_weight,
        "qwen_llm_decode_k_weight",
    )?;
    upload_projection_weight_to_arena(
        arena,
        handles.v_weight,
        v_weight,
        "qwen_llm_decode_v_weight",
    )?;
    // output/gate/up/down: only `Arena` handles need an upload; `Loaded` ones
    // already carry their mmap'd data (zero-copy).
    if let Some(handle) = handles.output_weight.arena_handle() {
        upload_projection_weight_to_arena(
            arena,
            handle,
            output_weight,
            "qwen_llm_decode_output_weight",
        )?;
    }
    if let Some(handle) = handles.gate_weight.arena_handle() {
        upload_projection_weight_to_arena(
            arena,
            handle,
            ffn_gate_weight,
            "qwen_llm_decode_gate_weight",
        )?;
    }
    if let Some(handle) = handles.up_weight.arena_handle() {
        upload_projection_weight_to_arena(
            arena,
            handle,
            ffn_up_weight,
            "qwen_llm_decode_up_weight",
        )?;
    }
    if let Some(handle) = handles.down_weight.arena_handle() {
        upload_projection_weight_to_arena(
            arena,
            handle,
            ffn_down_weight,
            "qwen_llm_decode_down_weight",
        )?;
    }
    Ok(())
}

/// Builds the entire decode step (all layers) into ONE ggml graph per token,
/// mirroring whisper's whole-decoder graph, to collapse N graph builds + N
/// dispatches per token to 1+1. One runner, one arena holding all layers'
/// resident weights, one compute requesting the final hidden plus every layer's
/// projected K/V.
pub(crate) struct Qwen3AsrLlmWholeDecoderGraphExecutor {
    // `reuse` holds raw pointers into `runner` (backend/scheduler), `arena`
    // (resident weights), and `loaded` (zero-copy mmap'd weights), so it MUST
    // drop first — keep it the first field. `loaded` must outlive the graph but
    // its backend buffer is tied to `runner`, so it sits between them.
    reuse: Option<LlmReusableDecodeGraph>,
    // Never READ, but load-bearing: owns the mmap backing the zero-copy bound LLM
    // weights, so it MUST stay alive (and drop after `reuse`). Removing it would
    // dangle the bound tensor pointers (UB) — hence allow(dead_code), not deletion.
    #[allow(dead_code)]
    loaded: Option<crate::ggml_runtime::GgmlLoadedWeightContext>,
    runner: GgmlCpuGraphRunner,
    arena: GgmlStaticTensorArena,
    layers: Vec<Qwen3AsrLlmLayerWeightHandles>,
    fused_logits_head: Option<Qwen3AsrLlmFusedLogitsHeadHandles>,
    dims: Qwen3AsrLlmDecodeDims,
    use_native_gqa: bool,
    rms_norm_epsilon: f32,
}

impl fmt::Debug for Qwen3AsrLlmWholeDecoderGraphExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Qwen3AsrLlmWholeDecoderGraphExecutor")
            .field("layers", &self.layers.len())
            .field("d_model", &self.dims.d_model)
            .field("q_heads", &self.dims.q_heads)
            .field("kv_heads", &self.dims.kv_heads)
            .finish_non_exhaustive()
    }
}

impl Qwen3AsrLlmWholeDecoderGraphExecutor {
    pub(crate) fn new(
        projections: &[Qwen3AsrLlmLayerAttentionProjection],
        runtime_path: Option<&std::path::Path>,
    ) -> Result<Self, GgmlCpuGraphError> {
        Self::new_with_rms_norm_epsilon_and_fused_logits_head(
            projections,
            runtime_path,
            DEFAULT_RMS_NORM_EPSILON,
            None,
        )
    }

    /// Like [`new`] but with an optional LoRA adapter injected into the decoder
    /// graph.  Uses [`DEFAULT_RMS_NORM_EPSILON`] and no fused logits head.
    pub(crate) fn new_with_lora(
        projections: &[Qwen3AsrLlmLayerAttentionProjection],
        runtime_path: Option<&std::path::Path>,
        adapter: Option<&QwenLoraAdapter>,
    ) -> Result<Self, GgmlCpuGraphError> {
        Self::new_with_adapter(
            projections,
            runtime_path,
            DEFAULT_RMS_NORM_EPSILON,
            None,
            adapter,
        )
    }

    pub(crate) fn new_with_rms_norm_epsilon_and_fused_logits_head(
        projections: &[Qwen3AsrLlmLayerAttentionProjection],
        runtime_path: Option<&std::path::Path>,
        rms_norm_epsilon: f32,
        fused_logits_head: Option<Qwen3AsrLlmFusedLogitsHeadSpec<'_>>,
    ) -> Result<Self, GgmlCpuGraphError> {
        Self::new_with_adapter(
            projections,
            runtime_path,
            rms_norm_epsilon,
            fused_logits_head,
            None,
        )
    }

    /// Construct with an optional LoRA adapter.  The adapter's arena tensors
    /// are allocated in the SAME arena as the layer weights (so the entire
    /// graph lives in one backend buffer) and uploaded in the same pass.
    pub(crate) fn new_with_adapter(
        projections: &[Qwen3AsrLlmLayerAttentionProjection],
        runtime_path: Option<&std::path::Path>,
        rms_norm_epsilon: f32,
        fused_logits_head: Option<Qwen3AsrLlmFusedLogitsHeadSpec<'_>>,
        adapter: Option<&QwenLoraAdapter>,
    ) -> Result<Self, GgmlCpuGraphError> {
        if projections.is_empty() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder executor requires at least one layer",
            });
        }
        if !rms_norm_epsilon.is_finite() || rms_norm_epsilon <= 0.0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder rms norm epsilon must be finite and positive",
            });
        }
        let mut config = qwen_runtime_graph_config();
        config.context_bytes = QWEN3_LLM_WHOLE_DECODE_GRAPH_CONTEXT_BYTES;
        let use_native_gqa = qwen_llm_resolve_use_native_gqa(config.backend);
        let runner = GgmlCpuGraphRunner::new(config)?;
        // goals 7+8: bind output/gate/up/down zero-copy from the mmap'd pack
        // (native q8/f16) instead of copying them into the arena. The context is
        // owned by this executor (drops after `reuse`, before `runner`).
        let loaded = runtime_path.and_then(|path| runner.load_gguf_weight_context(path).ok());
        let mut arena = runner.start_static_tensor_arena(config.context_bytes)?;
        let mut layers = Vec::with_capacity(projections.len());
        let mut fused_qkvs = Vec::with_capacity(projections.len());
        // Pending LoRA uploads: (tensor, values, name). Collected in Pass 1,
        // then uploaded in Pass 2 alongside the base weights.
        let mut pending_lora_uploads: Vec<(GgmlStaticTensor, Vec<f32>, &'static str)> = Vec::new();
        let mut dims: Option<Qwen3AsrLlmDecodeDims> = None;
        // Pass 1: allocate ALL layers' tensors first — the first upload freezes
        // the arena's backend buffer, after which no new tensors may be created.
        for projection in projections.iter() {
            let Qwen3AsrLlmLayerAttentionProjection::Generic(inner) = projection;
            // Zero-copy re-bind names MUST come from the loaded projection's own
            // recorded pack names (`inner.attn_output_name`/`ffn_*_name`), not a
            // family-fixed scheme like `llm_layer_tensor_names` -- the latter only
            // happens to match qwen3-asr's own `blk.N.*` on-disk names and silently
            // fails to bind a differently-prefixed family's pack (e.g. firered-llm's
            // `llm.blk.N.*`) with "host payload was dropped", since these tensors'
            // host bytes are dropped after load and only re-derivable by name.
            let (mut handles, layer_dims, fused_qkv) = allocate_decode_layer_tensors(
                &mut arena,
                loaded.as_ref(),
                &inner.attn_norm_weight,
                &inner.q_weight,
                &inner.k_weight,
                &inner.v_weight,
                &inner.q_bias,
                &inner.k_bias,
                &inner.v_bias,
                &inner.attn_output_weight,
                &inner.q_norm_weight,
                &inner.k_norm_weight,
                inner.head_dim,
                &inner.ffn_norm_weight,
                &inner.ffn_gate_weight,
                &inner.ffn_up_weight,
                &inner.ffn_down_weight,
                &inner.attn_output_name,
                &inner.ffn_gate_name,
                &inner.ffn_up_name,
                &inner.ffn_down_name,
            )?;
            // Allocate LoRA slots for this layer (if an adapter is active).
            // Sourced from `inner`'s own recorded pack names -- see
            // `allocate_layer_lora_slots`'s doc comment for why a family-fixed
            // naming scheme must not be substituted here.
            handles.lora = allocate_layer_lora_slots(
                &arena,
                adapter,
                &inner.attn_q_name,
                &inner.attn_k_name,
                &inner.attn_v_name,
                &inner.attn_output_name,
                &inner.ffn_gate_name,
                &inner.ffn_up_name,
                &inner.ffn_down_name,
                &mut pending_lora_uploads,
            )?;
            match dims {
                None => {
                    dims = Some(layer_dims);
                }
                Some(existing) if existing != layer_dims => {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder layers have inconsistent dimensions",
                    });
                }
                Some(_) => {}
            }
            layers.push(handles);
            fused_qkvs.push(fused_qkv);
        }
        let dims = dims.expect("non-empty projections set dims");
        let fused_logits_head = match fused_logits_head {
            Some(spec) => {
                let handles =
                    allocate_fused_logits_head_tensors(&mut arena, loaded.as_ref(), dims, &spec)?;
                upload_fused_logits_head_weights(&mut arena, &handles, &spec)?;
                Some(handles)
            }
            None => None,
        };
        // Pass 2: upload all layers' weight data into the allocated handles.
        for (layer_index, projection) in projections.iter().enumerate() {
            let Qwen3AsrLlmLayerAttentionProjection::Generic(inner) = projection;
            upload_decode_layer_weights(
                &mut arena,
                &layers[layer_index],
                fused_qkvs[layer_index].as_ref(),
                &inner.attn_norm_weight,
                &inner.q_weight,
                &inner.k_weight,
                &inner.v_weight,
                &inner.q_bias,
                &inner.k_bias,
                &inner.v_bias,
                &inner.attn_output_weight,
                &inner.q_norm_weight,
                &inner.k_norm_weight,
                &inner.ffn_norm_weight,
                &inner.ffn_gate_weight,
                &inner.ffn_up_weight,
                &inner.ffn_down_weight,
            )?;
        }
        // Upload LoRA tensors (collected during Pass 1).
        for (tensor, values, name) in pending_lora_uploads {
            arena.set_f32_slice(tensor, &values, name)?;
        }
        Ok(Self {
            reuse: None,
            loaded,
            runner,
            arena,
            layers,
            fused_logits_head,
            dims,
            use_native_gqa,
            rms_norm_epsilon,
        })
    }

    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// `"<kind>:<ggml backend name>"`, for perf diagnostics (e.g. the
    /// `OPENASR_HYMT2_PROFILE` runtime-backend log line).
    pub(crate) fn backend_label(&self) -> String {
        format!(
            "{:?}:{}",
            self.runner.backend_kind(),
            self.runner.backend_name()
        )
    }

    /// Graph reuse is only correct on the single-backend GPU path (Metal or
    /// the generic discrete-GPU lane — HIP/CUDA/Vulkan); the non-scheduler CPU
    /// compute mis-recomputes a reused graph that writes its KV in place, while
    /// the multi-backend scheduler drops refreshed per-token inputs.
    pub(crate) fn supports_graph_reuse(&self) -> bool {
        reusable_decode_graph_supported_for_runner(&self.runner)
    }

    pub(crate) fn supports_fused_top1(&self) -> bool {
        self.fused_logits_head.is_some()
    }

    #[cfg(test)]
    pub(crate) fn reused_batch_width_for_test(&self) -> Option<usize> {
        self.reuse.as_ref().map(|reuse| reuse.n_seq)
    }

    pub(crate) fn reused_graph_matches(&self, n_seq: usize, max_positions: usize) -> bool {
        self.reuse
            .as_ref()
            .map(|reuse| reuse.n_seq == n_seq && reuse.max_positions == max_positions)
            .unwrap_or(false)
    }

    pub(crate) fn backend_is_metal(&self) -> bool {
        matches!(self.runner.backend_kind(), GgmlCpuGraphBackend::Metal)
    }

    /// Native-GQA multi-query prefill chunk width. Root cause of the historical
    /// GPU multi-query prefill divergence: the graph is correct on CPU at every
    /// span (byte-perfect to 256x8), but the ggml CUDA/HIP flash-attn MMA/TILE
    /// kernel mis-handles the per-query causal mask + GQA when `n_kv > 32` AND
    /// `n_query > 2`. `n_query <= 2` routes to the correct VEC kernel
    /// (`fattn.cu` `Q->ne[1] <= 2`), and `n_kv <= 32` fits a single K-tile and is
    /// correct at any chunk. On HIP-like backends, chunks that would trip the
    /// kernel bug now run through the unfused `llm_naive_masked_attention` graph
    /// instead of the fused kernel (`llm_prefill_uses_flash_attention`), so the
    /// chunk can be wide: HIP = 8 for prompts <= 32 tokens (flash, single
    /// K-tile), 64 beyond (non-flash). CPU = 8 (flash, correct everywhere);
    /// generic GPU (Vulkan/CUDA/unknown) = 1 (conservative, not yet
    /// per-backend gated or non-flash validated).
    pub(crate) fn safe_multi_query_prefill_chunk_size_for(
        &self,
        token_count: usize,
    ) -> Option<usize> {
        if !self.use_native_gqa {
            return None;
        }
        if self.runner.backend_kind().is_gpu_class() {
            return Some(qwen_llm_safe_gpu_prefill_query_tokens_for_backend_name(
                self.runner.backend_name(),
                token_count,
            ));
        }
        Some(QWEN3_LLM_CPU_SAFE_PREFILL_QUERY_TOKENS)
    }

    /// Chunk width for prefill that reads/writes the host KV cache mid-prompt
    /// (`prefill_tokens_at_offset_*`). Historically `None` on every GPU-class
    /// backend (forcing the ~50 ms/token serial host-step path); HIP-like
    /// backends now chunk like the full-prompt path because the non-flash
    /// attention graph is width-safe there. Generic GPU stays serial until the
    /// non-flash path is validated per backend.
    pub(crate) fn safe_host_cache_prefill_chunk_size_for(
        &self,
        token_count: usize,
    ) -> Option<usize> {
        if self.runner.backend_kind().is_gpu_class()
            && !qwen_llm_backend_is_hip_like(self.runner.backend_name())
        {
            return None;
        }
        self.safe_multi_query_prefill_chunk_size_for(token_count)
    }

    /// Decide fused-flash vs unfused attention for a prefill graph step.
    /// Flash everywhere it is numerically trusted: single/double-query steps
    /// (VEC kernel), short KV spans (single K-tile), CPU/Metal, and non-HIP
    /// GPUs (their chunk policy never produces wide queries). Only the exact
    /// combination the HIP MMA/TILE kernel mis-handles — `n_query > 2` with
    /// `n_kv > 32` on a HIP-like backend — swaps to
    /// `llm_naive_masked_attention`.
    fn llm_prefill_uses_flash_attention(&self, token_count: usize, kv_span: usize) -> bool {
        if token_count <= QWEN3_LLM_HIP_SAFE_PREFILL_QUERY_TOKENS
            || kv_span <= QWEN3_LLM_HIP_SHORT_PREFILL_MAX_TOKENS
        {
            return true;
        }
        if !self.runner.backend_kind().is_gpu_class() {
            return true;
        }
        !qwen_llm_backend_is_hip_like(self.runner.backend_name())
    }

    /// True when prefill chunk widths must stay even on this backend.
    /// Measured on gfx1200 (Windows ROCm 7.1): odd query widths of 3/5/7 in
    /// the prefill path stall for seconds per chunk (8.2 s at width 5) while
    /// widths 1/2/4/6/8 run in ~25 ms. Callers splitting a prompt into chunks
    /// must trim an odd width > 1 down by one token
    /// (`even_prefill_chunk_len`); the final single token then rides the
    /// fast width-1 step.
    pub(crate) fn prefill_chunks_require_even_width(&self) -> bool {
        self.runner.backend_kind().is_gpu_class()
            && qwen_llm_backend_is_hip_like(self.runner.backend_name())
    }

    /// Run one decode token through ALL layers in a single graph. Returns the
    /// final hidden state plus each layer's projected (K, V) for the caller to
    /// write back into the host KV caches. `layer_caches[i]` supplies layer i's
    /// history prefix (cache_position tokens) for in-graph attention.
    pub(crate) fn run_step(
        &mut self,
        hidden: &[f32],
        cache_position: usize,
        layer_caches: &[Qwen3AsrLayerKvCacheState],
        rope_theta: f32,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        let dims = self.dims;
        if hidden.len() != dims.d_model {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder hidden width mismatch",
            });
        }
        if layer_caches.len() != self.layers.len() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder layer/cache count mismatch",
            });
        }
        let total_tokens =
            cache_position
                .checked_add(1)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder token count overflow",
                })?;
        let row_index =
            i32::try_from(cache_position).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder cache index exceeds ggml int boundary",
            })?;
        let rope_position =
            i32::try_from(cache_position).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder rope position exceeds ggml int boundary",
            })?;
        let rope = GgmlRopeExtParams::qwen_neox(
            dims.head_dim,
            cache_position.saturating_add(1).max(1),
            rope_theta,
        )?;

        let build_started_at = std::time::Instant::now();
        let mut graph = self.runner.start_graph();
        let hidden_tensor = graph.new_tensor_2d_f32(dims.d_model, 1, "qwen_llm_whole_hidden")?;
        let row_indices = graph.new_tensor_1d_i32(1, "qwen_llm_whole_row_index")?;
        let positions = graph.new_tensor_1d_i32(1, "qwen_llm_whole_position")?;
        graph.set_input(hidden_tensor)?;
        graph.set_input(row_indices)?;
        graph.set_input(positions)?;

        let stack = compose_llm_decoder_layer_stack(
            &mut graph,
            self.layers.len(),
            qwen_llm_stack_config(
                dims,
                rope,
                self.use_native_gqa,
                self.rms_norm_epsilon,
                1,
                1,
                true,
            ),
            LlmDecoderStackInputs {
                state: hidden_tensor,
                row_indices,
                positions,
                attention_mask: None,
                kv_span: total_tokens,
                key_history_name: "qwen_llm_whole_key_history",
                value_history_name: "qwen_llm_whole_value_history",
            },
            None,
            |layer_index| qwen_llm_layer_weights(&self.layers[layer_index], &self.arena),
            |_step, source| source,
        )?;
        let state = stack.state;
        let kv_inputs = stack.kv_inputs;
        let kv_outputs = stack.kv_outputs;
        graph.set_output(state)?;

        graph.set_f32_slice(hidden_tensor, hidden, "qwen_llm_whole_hidden")?;
        for (layer_index, (key_history, value_history)) in kv_inputs.iter().enumerate() {
            layer_caches[layer_index].upload_history_prefix_to_graph(
                &mut graph,
                *key_history,
                *value_history,
                cache_position,
                "qwen_llm_whole_key_history",
                "qwen_llm_whole_value_history",
            )?;
        }
        graph.set_i32_slice(row_indices, &[row_index], "qwen_llm_whole_row_index")?;
        graph.set_i32_slice(positions, &[rope_position], "qwen_llm_whole_position")?;

        let mut requested: Vec<(GgmlCpuTensor, usize)> =
            Vec::with_capacity(1 + 2 * self.layers.len());
        requested.push((state, dims.d_model));
        for (k, v) in &kv_outputs {
            requested.push((*k, dims.k_width));
            requested.push((*v, dims.v_width));
        }
        let build_micros = build_started_at.elapsed().as_micros();
        let compute_started_at = std::time::Instant::now();
        let mut outputs = graph.compute_outputs_f32(&requested)?;
        let compute_micros = compute_started_at.elapsed().as_micros();
        let hidden_out = outputs.remove(0);
        let mut layer_kv = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            let k = outputs.remove(0);
            let v = outputs.remove(0);
            layer_kv.push((k, v));
        }
        Ok(Qwen3AsrLlmWholeStepOutput {
            hidden: hidden_out,
            layer_kv,
            build_micros,
            compute_micros,
        })
    }

    /// Single-token decode step that transparently prefers the persistent
    /// reuse graph (`run_step_reused`) whenever the backend supports it
    /// (`supports_graph_reuse`, GPU-only single-backend lane), falling back to
    /// the plain per-token graph build (`run_step`) everywhere else --
    /// byte-identical output either way. This is the one family-agnostic
    /// entry point every LLM-decoder-stage family driving this executor
    /// should call instead of `run_step` directly, so a new family gets the
    /// Metal/GPU graph-reuse speedup for free without re-deriving the
    /// reuse-eligibility branch itself.
    pub(crate) fn run_step_auto(
        &mut self,
        hidden: &[f32],
        cache_position: usize,
        layer_caches: &[Qwen3AsrLayerKvCacheState],
        rope_theta: f32,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        let reuse_max_positions = layer_caches
            .first()
            .map(Qwen3AsrLayerKvCacheState::max_positions)
            .filter(|_| self.supports_graph_reuse());
        if let Some(max_positions) = reuse_max_positions {
            self.run_step_reused(
                hidden,
                cache_position,
                layer_caches,
                rope_theta,
                max_positions,
            )
        } else {
            self.run_step(hidden, cache_position, layer_caches, rope_theta)
        }
    }

    /// Prompt prefill for families that keep the plain "whole prompt as one
    /// `run_prefill` call" CPU path (no HIP/GPU chunk tuning) but still want
    /// `run_step_auto`'s Metal/GPU decode-graph reuse: `run_step_auto` and
    /// bulk `run_prefill` cannot be mixed for one utterance, because the
    /// persistent resident-KV graph `run_step_auto` builds on its first call
    /// is zero-initialized and only ever gets a prompt token's real K/V by
    /// that token flowing through `run_step_auto` itself (`set_rows` writes
    /// accumulate per call) -- a prompt prefilled instead through the bulk
    /// host-cache `run_prefill` never touches that graph, so decode would
    /// resume attending over a KV history that was never populated for the
    /// prompt span. So on the graph-reuse-capable path this runs the prompt
    /// serially, ONE TOKEN AT A TIME through `run_step_auto`, exactly
    /// mirroring `qwen::ggml_executor`'s own
    /// `prefill_prompt_serial_and_compute_last_logits` gate
    /// (`safe_host_cache_prefill_chunk_size_for` returning `None` on
    /// GPU-class backends routes qwen through the identical serial path) --
    /// the one-time graph build happens on the first prompt token and every
    /// remaining prompt token plus every decode token afterwards reuses it.
    /// Returns `None` when the backend does not support graph reuse, so the
    /// caller falls back to its own bulk `run_prefill` + host-cache KV write.
    pub(crate) fn run_prefill_auto_last_hidden(
        &mut self,
        token_major_values: &[f32],
        token_count: usize,
        layer_caches: &[Qwen3AsrLayerKvCacheState],
        rope_theta: f32,
    ) -> Result<Option<Vec<f32>>, GgmlCpuGraphError> {
        if !self.supports_graph_reuse() {
            return Ok(None);
        }
        let d_model = self.dims.d_model;
        let mut last_hidden = None;
        for position in 0..token_count {
            let start = position * d_model;
            let end = start + d_model;
            let hidden_in =
                token_major_values
                    .get(start..end)
                    .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder prefill token slice out of bounds",
                    })?;
            let step = self.run_step_auto(hidden_in, position, layer_caches, rope_theta)?;
            last_hidden = Some(step.hidden);
        }
        Ok(last_hidden)
    }

    pub(crate) fn run_step_top1(
        &mut self,
        hidden: &[f32],
        cache_position: usize,
        layer_caches: &[Qwen3AsrLayerKvCacheState],
        rope_theta: f32,
    ) -> Result<Qwen3AsrLlmWholeStepTop1Output, GgmlCpuGraphError> {
        let dims = self.dims;
        let fused_logits_head =
            self.fused_logits_head
                .as_ref()
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder fused logits head is not configured",
                })?;
        let vocab_size = fused_logits_head.vocab_size;
        if hidden.len() != dims.d_model {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder hidden width mismatch",
            });
        }
        if layer_caches.len() != self.layers.len() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder layer/cache count mismatch",
            });
        }
        let total_tokens =
            cache_position
                .checked_add(1)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder token count overflow",
                })?;
        let row_index =
            i32::try_from(cache_position).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder cache index exceeds ggml int boundary",
            })?;
        let rope_position =
            i32::try_from(cache_position).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder rope position exceeds ggml int boundary",
            })?;
        let rope = GgmlRopeExtParams::qwen_neox(
            dims.head_dim,
            cache_position.saturating_add(1).max(1),
            rope_theta,
        )?;

        let build_started_at = std::time::Instant::now();
        let mut graph = self.runner.start_graph();
        let hidden_tensor = graph.new_tensor_2d_f32(dims.d_model, 1, "qwen_llm_whole_hidden")?;
        let row_indices = graph.new_tensor_1d_i32(1, "qwen_llm_whole_row_index")?;
        let positions = graph.new_tensor_1d_i32(1, "qwen_llm_whole_position")?;
        graph.set_input(hidden_tensor)?;
        graph.set_input(row_indices)?;
        graph.set_input(positions)?;

        let stack = compose_llm_decoder_layer_stack(
            &mut graph,
            self.layers.len(),
            qwen_llm_stack_config(
                dims,
                rope,
                self.use_native_gqa,
                self.rms_norm_epsilon,
                1,
                1,
                true,
            ),
            LlmDecoderStackInputs {
                state: hidden_tensor,
                row_indices,
                positions,
                attention_mask: None,
                kv_span: total_tokens,
                key_history_name: "qwen_llm_whole_key_history",
                value_history_name: "qwen_llm_whole_value_history",
            },
            None,
            |layer_index| qwen_llm_layer_weights(&self.layers[layer_index], &self.arena),
            |_step, source| source,
        )?;
        let state = stack.state;
        let kv_inputs = stack.kv_inputs;
        let kv_outputs = stack.kv_outputs;
        let top1 = build_fused_logits_top1(&self.arena, fused_logits_head, &mut graph, state, 1)?;
        graph.set_output(top1)?;

        graph.set_f32_slice(hidden_tensor, hidden, "qwen_llm_whole_hidden")?;
        for (layer_index, (key_history, value_history)) in kv_inputs.iter().enumerate() {
            layer_caches[layer_index].upload_history_prefix_to_graph(
                &mut graph,
                *key_history,
                *value_history,
                cache_position,
                "qwen_llm_whole_key_history",
                "qwen_llm_whole_value_history",
            )?;
        }
        graph.set_i32_slice(row_indices, &[row_index], "qwen_llm_whole_row_index")?;
        graph.set_i32_slice(positions, &[rope_position], "qwen_llm_whole_position")?;

        let mut requested_f32: Vec<(GgmlCpuTensor, usize)> =
            Vec::with_capacity(2 * self.layers.len());
        for (k, v) in &kv_outputs {
            requested_f32.push((*k, dims.k_width));
            requested_f32.push((*v, dims.v_width));
        }
        let build_micros = build_started_at.elapsed().as_micros();
        let compute_started_at = std::time::Instant::now();
        let (mut outputs, token_outputs) =
            graph.compute_outputs_f32_i32(&requested_f32, &[(top1, 1)])?;
        let compute_micros = compute_started_at.elapsed().as_micros();
        let token_id = token_outputs
            .first()
            .and_then(|values| values.first())
            .copied()
            .ok_or(GgmlCpuGraphError::OutputByteSizeMismatch {
                expected: std::mem::size_of::<i32>(),
                actual: 0,
            })
            .and_then(|token_id| validate_fused_top1_token_id(token_id, vocab_size))?;
        let mut layer_kv = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            let k = outputs.remove(0);
            let v = outputs.remove(0);
            layer_kv.push((k, v));
        }
        Ok(Qwen3AsrLlmWholeStepTop1Output {
            token_id,
            layer_kv,
            build_micros,
            compute_micros,
        })
    }

    /// Run an entire prompt prefix as one causal multi-query LLM graph. This is
    /// the prefill counterpart to `run_step`: K/V for all prompt rows are written
    /// by one `set_rows` call per layer, guarded by a `[kv, query, 1, 1]` causal
    /// mask, then returned to the caller for the host cache that seeds the
    /// resident batched decode graph.
    pub(crate) fn run_prefill(
        &mut self,
        token_major_hidden: &[f32],
        token_count: usize,
        rope_theta: f32,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        self.run_prefill_with_history(
            token_major_hidden,
            token_count,
            0,
            token_count,
            &[],
            rope_theta,
        )
    }

    pub(crate) fn run_prefill_chunk(
        &mut self,
        token_major_hidden: &[f32],
        token_count: usize,
        position_offset: usize,
        total_token_count: usize,
        layer_caches: &[Qwen3AsrLayerKvCacheState],
        rope_theta: f32,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        self.run_prefill_with_history(
            token_major_hidden,
            token_count,
            position_offset,
            total_token_count,
            layer_caches,
            rope_theta,
        )
    }

    pub(crate) fn run_prefill_batched_chunk(
        &mut self,
        sequence_major_hidden: &[f32],
        token_count: usize,
        n_seq: usize,
        position_offset: usize,
        total_token_count: usize,
        layer_caches_by_sequence: &[&[Qwen3AsrLayerKvCacheState]],
        rope_theta: f32,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        self.run_prefill_with_batched_history(
            sequence_major_hidden,
            token_count,
            n_seq,
            position_offset,
            total_token_count,
            layer_caches_by_sequence,
            rope_theta,
        )
    }

    pub(crate) fn run_prefill_into_reused_batched(
        &mut self,
        sequence_major_hidden: &[f32],
        token_count: usize,
        n_seq: usize,
        max_positions: usize,
        rope_theta: f32,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        let dims = self.dims;
        if token_count == 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder resident prefill token count must be positive",
            });
        }
        if n_seq == 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder resident prefill n_seq must be positive",
            });
        }
        if max_positions < token_count {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder resident prefill max-position span is too small",
            });
        }
        let output_tokens =
            token_count
                .checked_mul(n_seq)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder resident prefill token/sequence count overflow",
                })?;
        let expected_hidden = dims.d_model.checked_mul(output_tokens).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder resident prefill hidden width overflow",
            },
        )?;
        if sequence_major_hidden.len() != expected_hidden {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder resident prefill hidden width mismatch",
            });
        }
        if !self.reused_graph_matches(n_seq, max_positions) {
            self.rebuild_reused_batched_graph(n_seq, max_positions, rope_theta, None)?;
        }
        let use_flash_attention = self.llm_prefill_uses_flash_attention(token_count, max_positions);

        let mut row_indices = Vec::with_capacity(output_tokens);
        let mut row_indices_usize = Vec::with_capacity(output_tokens);
        let mut positions = Vec::with_capacity(output_tokens);
        for _sequence_index in 0..n_seq {
            for token_position in 0..token_count {
                row_indices_usize.push(token_position);
                row_indices.push(i32::try_from(token_position).map_err(|_| {
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder resident prefill cache index exceeds ggml int boundary",
                    }
                })?);
                positions.push(i32::try_from(token_position).map_err(|_| {
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder resident prefill rope position exceeds ggml int boundary",
                    }
                })?);
            }
        }

        let mut reuse = self
            .reuse
            .take()
            .expect("reuse graph was built before resident prefill");
        let result = (|| {
            let rope = GgmlRopeExtParams::qwen_neox(dims.head_dim, max_positions, rope_theta)?;
            let build_started_at = std::time::Instant::now();
            let resident_kv = reuse.resident_kv_arena_mut().graph_tensors();
            let mut graph = self.runner.start_graph();
            let hidden_tensor = graph.new_tensor_2d_f32(
                dims.d_model,
                output_tokens,
                "qwen_llm_prefill_resident_hidden",
            )?;
            let row_indices_tensor = graph.new_tensor_4d_i32(
                token_count,
                1,
                n_seq,
                1,
                "qwen_llm_prefill_resident_row_index",
            )?;
            let positions_tensor =
                graph.new_tensor_1d_i32(output_tokens, "qwen_llm_prefill_resident_position")?;
            let attention_mask = graph.new_tensor_4d_f16(
                max_positions,
                token_count,
                1,
                n_seq,
                "qwen_llm_prefill_resident_self_mask",
            )?;
            graph.set_input(hidden_tensor)?;
            graph.set_input(row_indices_tensor)?;
            graph.set_input(positions_tensor)?;
            graph.set_input(attention_mask)?;

            let stack = compose_llm_decoder_layer_stack(
                &mut graph,
                self.layers.len(),
                qwen_llm_stack_config(
                    dims,
                    rope,
                    self.use_native_gqa,
                    self.rms_norm_epsilon,
                    token_count,
                    n_seq,
                    use_flash_attention,
                ),
                LlmDecoderStackInputs {
                    state: hidden_tensor,
                    row_indices: row_indices_tensor,
                    positions: positions_tensor,
                    attention_mask: Some(attention_mask),
                    kv_span: max_positions,
                    key_history_name: "qwen_llm_prefill_resident_key_history",
                    value_history_name: "qwen_llm_prefill_resident_value_history",
                },
                Some(&resident_kv),
                |layer_index| qwen_llm_layer_weights(&self.layers[layer_index], &self.arena),
                |_step, source| source,
            )?;
            let state = stack.state;
            let (output_state, output_len) = if n_seq == 1 {
                let final_hidden_offset = token_count
                    .checked_sub(1)
                    .and_then(|position| position.checked_mul(dims.d_model))
                    .and_then(|offset| offset.checked_mul(std::mem::size_of::<f32>()))
                    .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder resident prefill final hidden offset overflow",
                    })?;
                let final_hidden = graph.view_2d(
                    state,
                    dims.d_model,
                    1,
                    dims.d_model * std::mem::size_of::<f32>(),
                    final_hidden_offset,
                )?;
                (final_hidden, dims.d_model)
            } else {
                (state, expected_hidden)
            };
            graph.set_output(output_state)?;
            graph.set_f32_slice(
                hidden_tensor,
                sequence_major_hidden,
                "qwen_llm_prefill_resident_hidden",
            )?;
            graph.set_i32_slice(
                row_indices_tensor,
                &row_indices,
                "qwen_llm_prefill_resident_row_index",
            )?;
            graph.set_i32_slice(
                positions_tensor,
                &positions,
                "qwen_llm_prefill_resident_position",
            )?;
            let mask_bits = build_fixed_kv_attention_mask_bits_for_query_rows(
                max_positions,
                token_count,
                n_seq,
                &row_indices_usize,
            )?;
            graph.set_f16_bits_slice(
                attention_mask,
                &mask_bits,
                "qwen_llm_prefill_resident_self_mask",
            )?;
            let build_micros = build_started_at.elapsed().as_micros();
            let compute_started_at = std::time::Instant::now();
            let hidden_out = graph.compute_output_f32(output_state, output_len)?;
            let compute_micros = compute_started_at.elapsed().as_micros();
            Ok(Qwen3AsrLlmWholeStepOutput {
                hidden: hidden_out,
                layer_kv: Vec::new(),
                build_micros,
                compute_micros,
            })
        })();
        self.reuse = Some(reuse);
        result
    }

    fn run_prefill_with_history(
        &mut self,
        token_major_hidden: &[f32],
        token_count: usize,
        position_offset: usize,
        total_token_count: usize,
        layer_caches: &[Qwen3AsrLayerKvCacheState],
        rope_theta: f32,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        let layer_caches_by_sequence = [layer_caches];
        self.run_prefill_with_batched_history(
            token_major_hidden,
            token_count,
            1,
            position_offset,
            total_token_count,
            &layer_caches_by_sequence,
            rope_theta,
        )
    }

    fn run_prefill_with_batched_history(
        &mut self,
        sequence_major_hidden: &[f32],
        token_count: usize,
        n_seq: usize,
        position_offset: usize,
        total_token_count: usize,
        layer_caches_by_sequence: &[&[Qwen3AsrLayerKvCacheState]],
        rope_theta: f32,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        let dims = self.dims;
        if token_count == 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill token count must be positive",
            });
        }
        if n_seq == 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill n_seq must be positive",
            });
        }
        let chunk_end = position_offset.checked_add(token_count).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill token span overflow",
            },
        )?;
        if total_token_count < chunk_end {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill total span smaller than query span",
            });
        }
        if position_offset > 0 && layer_caches_by_sequence.len() != n_seq {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill history sequence count mismatch",
            });
        }
        if position_offset > 0
            && layer_caches_by_sequence
                .iter()
                .any(|layer_caches| layer_caches.len() != self.layers.len())
        {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill history layer/cache count mismatch",
            });
        }
        let output_tokens =
            token_count
                .checked_mul(n_seq)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder prefill token/sequence count overflow",
                })?;
        let expected_hidden = dims.d_model.checked_mul(output_tokens).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill hidden width overflow",
            },
        )?;
        if sequence_major_hidden.len() != expected_hidden {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill hidden width mismatch",
            });
        }

        let mut row_indices = Vec::with_capacity(output_tokens);
        let mut row_indices_usize = Vec::with_capacity(output_tokens);
        let mut positions = Vec::with_capacity(output_tokens);
        for _sequence_index in 0..n_seq {
            for token_position in 0..token_count {
                let absolute_position = position_offset.checked_add(token_position).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder prefill position overflow",
                    },
                )?;
                row_indices_usize.push(absolute_position);
                row_indices.push(i32::try_from(absolute_position).map_err(|_| {
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder prefill cache index exceeds ggml int boundary",
                    }
                })?);
                positions.push(i32::try_from(absolute_position).map_err(|_| {
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder prefill rope position exceeds ggml int boundary",
                    }
                })?);
            }
        }

        let rope = GgmlRopeExtParams::qwen_neox(dims.head_dim, total_token_count, rope_theta)?;
        let use_flash_attention =
            self.llm_prefill_uses_flash_attention(token_count, total_token_count);
        let build_started_at = std::time::Instant::now();
        let mut graph = self.runner.start_graph();
        let hidden_tensor =
            graph.new_tensor_2d_f32(dims.d_model, output_tokens, "qwen_llm_prefill_hidden")?;
        let row_indices_tensor =
            graph.new_tensor_4d_i32(token_count, 1, n_seq, 1, "qwen_llm_prefill_row_index")?;
        let positions_tensor =
            graph.new_tensor_1d_i32(output_tokens, "qwen_llm_prefill_position")?;
        let attention_mask = graph.new_tensor_4d_f16(
            total_token_count,
            token_count,
            1,
            n_seq,
            "qwen_llm_prefill_self_mask",
        )?;
        graph.set_input(hidden_tensor)?;
        graph.set_input(row_indices_tensor)?;
        graph.set_input(positions_tensor)?;
        graph.set_input(attention_mask)?;

        let stack = compose_llm_decoder_layer_stack(
            &mut graph,
            self.layers.len(),
            qwen_llm_stack_config(
                dims,
                rope,
                self.use_native_gqa,
                self.rms_norm_epsilon,
                token_count,
                n_seq,
                use_flash_attention,
            ),
            LlmDecoderStackInputs {
                state: hidden_tensor,
                row_indices: row_indices_tensor,
                positions: positions_tensor,
                attention_mask: Some(attention_mask),
                kv_span: total_token_count,
                key_history_name: "qwen_llm_prefill_key_history",
                value_history_name: "qwen_llm_prefill_value_history",
            },
            None,
            |layer_index| qwen_llm_layer_weights(&self.layers[layer_index], &self.arena),
            |_step, source| source,
        )?;
        let state = stack.state;
        let kv_inputs = stack.kv_inputs;
        let kv_outputs = stack.kv_outputs;
        graph.set_output(state)?;

        graph.set_f32_slice(
            hidden_tensor,
            sequence_major_hidden,
            "qwen_llm_prefill_hidden",
        )?;
        graph.set_i32_slice(
            row_indices_tensor,
            &row_indices,
            "qwen_llm_prefill_row_index",
        )?;
        graph.set_i32_slice(positions_tensor, &positions, "qwen_llm_prefill_position")?;
        let mask_bits = build_fixed_kv_attention_mask_bits_for_query_rows(
            total_token_count,
            token_count,
            n_seq,
            &row_indices_usize,
        )?;
        graph.set_f16_bits_slice(attention_mask, &mask_bits, "qwen_llm_prefill_self_mask")?;
        for (layer_index, (key_history, value_history)) in kv_inputs.into_iter().enumerate() {
            let (key_values, value_values) = qwen_prefill_history_inputs_for_layer(
                dims,
                total_token_count,
                n_seq,
                layer_index,
                position_offset,
                layer_caches_by_sequence,
            )?;
            graph.set_f32_slice(key_history, &key_values, "qwen_llm_prefill_key_history")?;
            graph.set_f32_slice(
                value_history,
                &value_values,
                "qwen_llm_prefill_value_history",
            )?;
        }

        let mut requested: Vec<(GgmlCpuTensor, usize)> =
            Vec::with_capacity(1 + 2 * self.layers.len());
        requested.push((state, expected_hidden));
        let layer_kv_width = dims.k_width.checked_mul(output_tokens).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill KV output width overflow",
            },
        )?;
        for (k, v) in &kv_outputs {
            requested.push((*k, layer_kv_width));
            requested.push((*v, layer_kv_width));
        }
        let build_micros = build_started_at.elapsed().as_micros();
        let compute_started_at = std::time::Instant::now();
        let mut outputs = graph.compute_outputs_f32(&requested)?;
        let compute_micros = compute_started_at.elapsed().as_micros();
        let hidden_out = outputs.remove(0);
        let mut layer_kv = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            let k = outputs.remove(0);
            let v = outputs.remove(0);
            layer_kv.push((k, v));
        }
        Ok(Qwen3AsrLlmWholeStepOutput {
            hidden: hidden_out,
            layer_kv,
            build_micros,
            compute_micros,
        })
    }

    /// Fixed-max decode step that builds the graph ONCE into a persistent session
    /// and re-runs it every token, refreshing only the inputs (P9 graph reuse).
    /// The KV history is the full max_positions span with an additive f16 mask
    /// (0 for valid rows, -inf above) so the graph shape is constant and the
    /// build and Metal command-buffer encode are amortized across all decode tokens.
    /// Byte-identical to the growing-KV `run_step`; used on the Metal/scheduler
    /// path only (see `supports_graph_reuse`).
    pub(crate) fn run_step_reused(
        &mut self,
        hidden: &[f32],
        cache_position: usize,
        layer_caches: &[Qwen3AsrLayerKvCacheState],
        rope_theta: f32,
        max_positions: usize,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        let dims = self.dims;
        let n_seq = 1;
        let layer_count = self.layers.len();
        if hidden.len() != dims.d_model {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder hidden width mismatch",
            });
        }
        if layer_caches.len() != layer_count {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder layer/cache count mismatch",
            });
        }
        let total_tokens =
            cache_position
                .checked_add(1)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder token count overflow",
                })?;
        if max_positions < total_tokens {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder fixed KV span smaller than current token count",
            });
        }
        let row_index =
            i32::try_from(cache_position).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder cache index exceeds ggml int boundary",
            })?;
        let rope_position =
            i32::try_from(cache_position).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder rope position exceeds ggml int boundary",
            })?;

        let needs_build = self
            .reuse
            .as_ref()
            .map(|reuse| reuse.max_positions != max_positions || reuse.n_seq != n_seq)
            .unwrap_or(true);
        if needs_build {
            // n_ctx_orig is ignored (ext_factor=0); the rope position is supplied
            // by the `positions` input, so a constant here keeps the graph reusable.
            let rope = GgmlRopeExtParams::qwen_neox(dims.head_dim, max_positions, rope_theta)?;
            // S5: allocate device-resident per-layer KV in a persistent arena,
            // sized to the full max_positions span and zero-initialized so masked
            // (unwritten) positions never feed NaN/inf into flash-attn. The graph's
            // `set_rows` writes accumulate into this buffer across decode steps, so
            // there is no per-step host upload of the growing KV prefix.
            let resident_kv_arena = allocate_zeroed_llm_resident_kv_arena(
                &self.runner,
                QWEN3_LLM_WHOLE_DECODE_GRAPH_CONTEXT_BYTES,
                layer_count,
                dims.head_dim,
                max_positions,
                dims.kv_heads,
                n_seq,
                "qwen_llm_resident_kv",
            )?;

            let mut session = self
                .runner
                .start_persistent_graph_session(QWEN3_LLM_WHOLE_DECODE_GRAPH_CONTEXT_BYTES)?;
            let graph = session.builder();
            let hidden_tensor =
                graph.new_tensor_2d_f32(dims.d_model, n_seq, "qwen_llm_reuse_hidden")?;
            let row_indices =
                graph.new_tensor_4d_i32(1, 1, n_seq, 1, "qwen_llm_reuse_row_index")?;
            let positions = graph.new_tensor_1d_i32(n_seq, "qwen_llm_reuse_position")?;
            let attention_mask =
                graph.new_tensor_4d_f16(max_positions, 1, 1, n_seq, "qwen_llm_reuse_self_mask")?;
            graph.set_input(hidden_tensor)?;
            graph.set_input(row_indices)?;
            graph.set_input(positions)?;
            graph.set_input(attention_mask)?;
            let resident_kv = resident_kv_arena.graph_tensors();
            let stack = compose_llm_decoder_layer_stack(
                graph,
                self.layers.len(),
                qwen_llm_stack_config(
                    dims,
                    rope,
                    self.use_native_gqa,
                    self.rms_norm_epsilon,
                    1,
                    n_seq,
                    true,
                ),
                LlmDecoderStackInputs {
                    state: hidden_tensor,
                    row_indices,
                    positions,
                    attention_mask: Some(attention_mask),
                    kv_span: max_positions,
                    key_history_name: "qwen_llm_reuse_key_history",
                    value_history_name: "qwen_llm_reuse_value_history",
                },
                Some(&resident_kv),
                |layer_index| qwen_llm_layer_weights(&self.layers[layer_index], &self.arena),
                |_step, source| source,
            )?;
            let state = stack.state;
            let top1 = self
                .fused_logits_head
                .as_ref()
                .map(|logits_head| {
                    build_fused_logits_top1(&self.arena, logits_head, graph, state, n_seq)
                })
                .transpose()?;
            graph.set_output(state)?;
            if let Some(top1) = top1 {
                graph.set_output(top1)?;
            }
            // Load-bearing: reusable fused decode must prepare both roots. The
            // state root keeps host-hidden callers valid, and the top1 root keeps
            // later fused-token calls from reusing a graph allocated for state
            // only (round-4 regression).
            let mut prepared_outputs = vec![state];
            if let Some(top1) = top1 {
                prepared_outputs.push(top1);
            }
            graph.prepare_outputs_for_upload(&prepared_outputs)?;
            self.reuse = Some(LlmReusableDecodeGraph::new(
                session,
                resident_kv_arena,
                max_positions,
                n_seq,
                hidden_tensor,
                row_indices,
                positions,
                attention_mask,
                state,
                top1,
            ));
        }

        let reuse = self.reuse.as_mut().expect("reuse graph built above");
        let hidden_tensor = reuse.hidden_tensor;
        let row_indices = reuse.row_indices;
        let positions = reuse.positions;
        let attention_mask = reuse.attention_mask;
        let state = reuse.state;
        let graph = reuse.builder();

        graph.set_f32_slice(hidden_tensor, hidden, "qwen_llm_reuse_hidden")?;
        // S5: KV history is device-resident and accumulated in-graph by `set_rows`
        // across decode steps — no per-step host upload of the growing prefix
        // and no per-step K/V host readback. `layer_caches` is used only for the
        // layer-count check above on the resident path.
        let mask_bits = build_fixed_kv_attention_mask_bits(max_positions, total_tokens)?;
        graph.set_f16_bits_slice(attention_mask, &mask_bits, "qwen_llm_reuse_self_mask")?;
        graph.set_i32_slice(row_indices, &[row_index], "qwen_llm_reuse_row_index")?;
        graph.set_i32_slice(positions, &[rope_position], "qwen_llm_reuse_position")?;

        let compute_started_at = std::time::Instant::now();
        let hidden_out = graph.compute_output_f32(state, dims.d_model)?;
        let compute_micros = compute_started_at.elapsed().as_micros();
        Ok(Qwen3AsrLlmWholeStepOutput {
            hidden: hidden_out,
            layer_kv: Vec::new(),
            build_micros: 0,
            compute_micros,
        })
    }

    /// Fixed-max reusable decode for a static micro-batch. `hidden` is packed as
    /// `[d_model, n_seq]`; `cache_positions[i]` is the row/RoPE position for slot
    /// `i`. This is the graph-level entry point the serve-mode owner thread uses
    /// after it has packed active slots.
    #[allow(dead_code)]
    pub(crate) fn run_step_reused_batched(
        &mut self,
        hidden: &[f32],
        cache_positions: &[usize],
        rope_theta: f32,
        max_positions: usize,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        self.run_step_reused_batched_inner(hidden, cache_positions, rope_theta, max_positions, None)
    }

    /// Same graph as `run_step_reused_batched`, but seeds the resident KV arena
    /// with the serial prefill host caches before the first batched generated
    /// token. Passing a seed forces a graph/arena rebuild so stale slot KV from a
    /// previous static batch cannot leak into the new batch.
    #[allow(dead_code)]
    pub(crate) fn run_step_reused_batched_seeded(
        &mut self,
        hidden: &[f32],
        cache_positions: &[usize],
        seeded_layer_kv_by_sequence: &[&[Qwen3AsrLayerKvCacheState]],
        rope_theta: f32,
        max_positions: usize,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        self.run_step_reused_batched_inner(
            hidden,
            cache_positions,
            rope_theta,
            max_positions,
            Some(seeded_layer_kv_by_sequence),
        )
    }

    pub(crate) fn run_step_reused_batched_top1(
        &mut self,
        hidden: &[f32],
        cache_positions: &[usize],
        rope_theta: f32,
        max_positions: usize,
    ) -> Result<Qwen3AsrLlmWholeStepTop1Output, GgmlCpuGraphError> {
        self.run_step_reused_batched_top1_inner(
            hidden,
            cache_positions,
            rope_theta,
            max_positions,
            None,
        )
    }

    pub(crate) fn run_step_reused_batched_seeded_top1(
        &mut self,
        hidden: &[f32],
        cache_positions: &[usize],
        seeded_layer_kv_by_sequence: &[&[Qwen3AsrLayerKvCacheState]],
        rope_theta: f32,
        max_positions: usize,
    ) -> Result<Qwen3AsrLlmWholeStepTop1Output, GgmlCpuGraphError> {
        self.run_step_reused_batched_top1_inner(
            hidden,
            cache_positions,
            rope_theta,
            max_positions,
            Some(seeded_layer_kv_by_sequence),
        )
    }

    /// Rebuild the reusable batched graph and seed resident KV without executing
    /// a token step. This lets owner threads migrate a live batch to a different
    /// `n_seq` while preserving the boundary invariant: resident KV contains the
    /// prompt plus every generated token except the current last token.
    #[allow(dead_code)]
    pub(crate) fn reset_reused_batched_seeded(
        &mut self,
        seeded_layer_kv_by_sequence: &[&[Qwen3AsrLayerKvCacheState]],
        rope_theta: f32,
        max_positions: usize,
    ) -> Result<(), GgmlCpuGraphError> {
        let n_seq = seeded_layer_kv_by_sequence.len();
        if n_seq == 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder n_seq must be positive",
            });
        }
        let prefix_lengths = qwen_batched_seed_written_prefix_lengths(seeded_layer_kv_by_sequence)?;
        self.rebuild_reused_batched_graph(
            n_seq,
            max_positions,
            rope_theta,
            Some((&prefix_lengths, seeded_layer_kv_by_sequence)),
        )
    }

    #[allow(dead_code)]
    pub(crate) fn seed_reused_batched_slot(
        &mut self,
        slot_index: usize,
        cache_position: usize,
        layer_kv: &[Qwen3AsrLayerKvCacheState],
        max_positions: usize,
    ) -> Result<(), GgmlCpuGraphError> {
        let dims = self.dims;
        let reuse = self
            .reuse
            .as_mut()
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder batched reuse graph is not initialized",
            })?;
        if reuse.max_positions != max_positions {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder batched reuse max-position mismatch",
            });
        }
        if slot_index >= reuse.n_seq {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder batched reuse slot index out of range",
            });
        }
        seed_qwen_batched_resident_kv_slot(
            reuse.resident_kv_arena_mut(),
            dims.head_dim,
            max_positions,
            dims.kv_heads,
            slot_index,
            cache_position,
            layer_kv,
        )
    }

    #[allow(dead_code)]
    pub(crate) fn zero_reused_batched_slot(
        &mut self,
        slot_index: usize,
        max_positions: usize,
    ) -> Result<(), GgmlCpuGraphError> {
        let dims = self.dims;
        let reuse = self
            .reuse
            .as_mut()
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder batched reuse graph is not initialized",
            })?;
        if reuse.max_positions != max_positions {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder batched reuse max-position mismatch",
            });
        }
        if slot_index >= reuse.n_seq {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder batched reuse slot index out of range",
            });
        }
        zero_qwen_batched_resident_kv_slot(
            reuse.resident_kv_arena_mut(),
            dims.head_dim,
            max_positions,
            dims.kv_heads,
            slot_index,
        )
    }

    fn rebuild_reused_batched_graph(
        &mut self,
        n_seq: usize,
        max_positions: usize,
        rope_theta: f32,
        seed: Option<(&[usize], &[&[Qwen3AsrLayerKvCacheState]])>,
    ) -> Result<(), GgmlCpuGraphError> {
        if n_seq == 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder n_seq must be positive",
            });
        }
        let dims = self.dims;
        let rope = GgmlRopeExtParams::qwen_neox(dims.head_dim, max_positions, rope_theta)?;
        let mut resident_kv_arena = allocate_zeroed_llm_resident_kv_arena(
            &self.runner,
            QWEN3_LLM_WHOLE_DECODE_GRAPH_CONTEXT_BYTES,
            self.layers.len(),
            dims.head_dim,
            max_positions,
            dims.kv_heads,
            n_seq,
            "qwen_llm_resident_kv",
        )?;
        if let Some((prefix_lengths, seed_layers)) = seed {
            seed_qwen_batched_resident_kv_arena(
                &mut resident_kv_arena,
                dims.head_dim,
                max_positions,
                dims.kv_heads,
                prefix_lengths,
                seed_layers,
            )?;
        }

        let mut session = self
            .runner
            .start_persistent_graph_session(QWEN3_LLM_WHOLE_DECODE_GRAPH_CONTEXT_BYTES)?;
        let graph = session.builder();
        let hidden_tensor =
            graph.new_tensor_2d_f32(dims.d_model, n_seq, "qwen_llm_reuse_hidden")?;
        let row_indices_tensor =
            graph.new_tensor_4d_i32(1, 1, n_seq, 1, "qwen_llm_reuse_row_index")?;
        let positions = graph.new_tensor_1d_i32(n_seq, "qwen_llm_reuse_position")?;
        let attention_mask =
            graph.new_tensor_4d_f16(max_positions, 1, 1, n_seq, "qwen_llm_reuse_self_mask")?;
        graph.set_input(hidden_tensor)?;
        graph.set_input(row_indices_tensor)?;
        graph.set_input(positions)?;
        graph.set_input(attention_mask)?;
        let resident_kv = resident_kv_arena.graph_tensors();
        let stack = compose_llm_decoder_layer_stack(
            graph,
            self.layers.len(),
            qwen_llm_stack_config(
                dims,
                rope,
                self.use_native_gqa,
                self.rms_norm_epsilon,
                1,
                n_seq,
                true,
            ),
            LlmDecoderStackInputs {
                state: hidden_tensor,
                row_indices: row_indices_tensor,
                positions,
                attention_mask: Some(attention_mask),
                kv_span: max_positions,
                key_history_name: "qwen_llm_reuse_key_history",
                value_history_name: "qwen_llm_reuse_value_history",
            },
            Some(&resident_kv),
            |layer_index| qwen_llm_layer_weights(&self.layers[layer_index], &self.arena),
            |_step, source| source,
        )?;
        let state = stack.state;
        let top1 = self
            .fused_logits_head
            .as_ref()
            .map(|logits_head| {
                build_fused_logits_top1(&self.arena, logits_head, graph, state, n_seq)
            })
            .transpose()?;
        graph.set_output(state)?;
        if let Some(top1) = top1 {
            graph.set_output(top1)?;
        }
        // Load-bearing: reusable fused decode must prepare both roots. The state
        // root keeps host-hidden callers valid, and the top1 root keeps later
        // fused-token calls from reusing a graph allocated for state only
        // (round-4 regression).
        let mut prepared_outputs = vec![state];
        if let Some(top1) = top1 {
            prepared_outputs.push(top1);
        }
        graph.prepare_outputs_for_upload(&prepared_outputs)?;
        self.reuse = Some(LlmReusableDecodeGraph::new(
            session,
            resident_kv_arena,
            max_positions,
            n_seq,
            hidden_tensor,
            row_indices_tensor,
            positions,
            attention_mask,
            state,
            top1,
        ));
        Ok(())
    }

    fn run_step_reused_batched_inner(
        &mut self,
        hidden: &[f32],
        cache_positions: &[usize],
        rope_theta: f32,
        max_positions: usize,
        seeded_layer_kv_by_sequence: Option<&[&[Qwen3AsrLayerKvCacheState]]>,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        let dims = self.dims;
        let n_seq = cache_positions.len();
        if n_seq == 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder n_seq must be positive",
            });
        }
        let expected_hidden =
            dims.d_model
                .checked_mul(n_seq)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder hidden width overflow",
                })?;
        if hidden.len() != expected_hidden {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder hidden width mismatch",
            });
        }

        let mut total_tokens_by_sequence = Vec::with_capacity(n_seq);
        let mut row_indices = Vec::with_capacity(n_seq);
        let mut rope_positions = Vec::with_capacity(n_seq);
        for &cache_position in cache_positions {
            let total_tokens =
                cache_position
                    .checked_add(1)
                    .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder token count overflow",
                    })?;
            if max_positions < total_tokens {
                return Err(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder fixed KV span smaller than current token count",
                });
            }
            row_indices.push(i32::try_from(cache_position).map_err(|_| {
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder cache index exceeds ggml int boundary",
                }
            })?);
            rope_positions.push(i32::try_from(cache_position).map_err(|_| {
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder rope position exceeds ggml int boundary",
                }
            })?);
            total_tokens_by_sequence.push(total_tokens);
        }
        if let Some(seed) = seeded_layer_kv_by_sequence
            && seed.len() != n_seq
        {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder batched KV seed sequence count mismatch",
            });
        }

        let needs_build = self
            .reuse
            .as_ref()
            .map(|reuse| reuse.max_positions != max_positions || reuse.n_seq != n_seq)
            .unwrap_or(true)
            || seeded_layer_kv_by_sequence.is_some();
        if needs_build {
            let seed =
                seeded_layer_kv_by_sequence.map(|seed_layers| (cache_positions, seed_layers));
            self.rebuild_reused_batched_graph(n_seq, max_positions, rope_theta, seed)?;
        }

        let reuse = self.reuse.as_mut().expect("reuse graph built above");
        let hidden_tensor = reuse.hidden_tensor;
        let row_indices_tensor = reuse.row_indices;
        let positions = reuse.positions;
        let attention_mask = reuse.attention_mask;
        let state = reuse.state;
        let graph = reuse.builder();

        graph.set_f32_slice(hidden_tensor, hidden, "qwen_llm_reuse_hidden")?;
        let mask_bits = build_fixed_kv_attention_mask_bits_for_sequences(
            max_positions,
            &total_tokens_by_sequence,
        )?;
        graph.set_f16_bits_slice(attention_mask, &mask_bits, "qwen_llm_reuse_self_mask")?;
        graph.set_i32_slice(row_indices_tensor, &row_indices, "qwen_llm_reuse_row_index")?;
        graph.set_i32_slice(positions, &rope_positions, "qwen_llm_reuse_position")?;

        let compute_started_at = std::time::Instant::now();
        let hidden_out = graph.compute_output_f32(state, expected_hidden)?;
        let compute_micros = compute_started_at.elapsed().as_micros();
        Ok(Qwen3AsrLlmWholeStepOutput {
            hidden: hidden_out,
            layer_kv: Vec::new(),
            build_micros: 0,
            compute_micros,
        })
    }

    fn run_step_reused_batched_top1_inner(
        &mut self,
        hidden: &[f32],
        cache_positions: &[usize],
        rope_theta: f32,
        max_positions: usize,
        seeded_layer_kv_by_sequence: Option<&[&[Qwen3AsrLayerKvCacheState]]>,
    ) -> Result<Qwen3AsrLlmWholeStepTop1Output, GgmlCpuGraphError> {
        if self.fused_logits_head.is_none() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder fused logits head is not configured",
            });
        }
        let vocab_size = self
            .fused_logits_head
            .as_ref()
            .expect("checked above")
            .vocab_size;
        let dims = self.dims;
        let n_seq = cache_positions.len();
        if n_seq != 1 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder fused top1 currently requires n_seq=1",
            });
        }
        let expected_hidden =
            dims.d_model
                .checked_mul(n_seq)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder hidden width overflow",
                })?;
        if hidden.len() != expected_hidden {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder hidden width mismatch",
            });
        }

        let mut total_tokens_by_sequence = Vec::with_capacity(n_seq);
        let mut row_indices = Vec::with_capacity(n_seq);
        let mut rope_positions = Vec::with_capacity(n_seq);
        for &cache_position in cache_positions {
            let total_tokens =
                cache_position
                    .checked_add(1)
                    .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder token count overflow",
                    })?;
            if max_positions < total_tokens {
                return Err(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder fixed KV span smaller than current token count",
                });
            }
            row_indices.push(i32::try_from(cache_position).map_err(|_| {
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder cache index exceeds ggml int boundary",
                }
            })?);
            rope_positions.push(i32::try_from(cache_position).map_err(|_| {
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder rope position exceeds ggml int boundary",
                }
            })?);
            total_tokens_by_sequence.push(total_tokens);
        }
        if let Some(seed) = seeded_layer_kv_by_sequence
            && seed.len() != n_seq
        {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder batched KV seed sequence count mismatch",
            });
        }

        let needs_build = self
            .reuse
            .as_ref()
            .map(|reuse| {
                reuse.max_positions != max_positions || reuse.n_seq != n_seq || reuse.top1.is_none()
            })
            .unwrap_or(true)
            || seeded_layer_kv_by_sequence.is_some();
        if needs_build {
            let seed =
                seeded_layer_kv_by_sequence.map(|seed_layers| (cache_positions, seed_layers));
            self.rebuild_reused_batched_graph(n_seq, max_positions, rope_theta, seed)?;
        }

        let reuse = self.reuse.as_mut().expect("reuse graph built above");
        let hidden_tensor = reuse.hidden_tensor;
        let row_indices_tensor = reuse.row_indices;
        let positions = reuse.positions;
        let attention_mask = reuse.attention_mask;
        let top1 = reuse.top1.ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "whole-decoder fused top1 output was not prepared",
        })?;
        let graph = reuse.builder();

        graph.set_f32_slice(hidden_tensor, hidden, "qwen_llm_reuse_hidden")?;
        let mask_bits = build_fixed_kv_attention_mask_bits_for_sequences(
            max_positions,
            &total_tokens_by_sequence,
        )?;
        graph.set_f16_bits_slice(attention_mask, &mask_bits, "qwen_llm_reuse_self_mask")?;
        graph.set_i32_slice(row_indices_tensor, &row_indices, "qwen_llm_reuse_row_index")?;
        graph.set_i32_slice(positions, &rope_positions, "qwen_llm_reuse_position")?;

        let compute_started_at = std::time::Instant::now();
        let token_ids = graph.compute_output_i32(top1, 1)?;
        let compute_micros = compute_started_at.elapsed().as_micros();
        let token_id = token_ids
            .first()
            .copied()
            .ok_or(GgmlCpuGraphError::OutputByteSizeMismatch {
                expected: std::mem::size_of::<i32>(),
                actual: 0,
            })
            .and_then(|token_id| validate_fused_top1_token_id(token_id, vocab_size))?;
        Ok(Qwen3AsrLlmWholeStepTop1Output {
            token_id,
            layer_kv: Vec::new(),
            build_micros: 0,
            compute_micros,
        })
    }
}

fn qwen_prefill_history_inputs_for_layer(
    dims: Qwen3AsrLlmDecodeDims,
    kv_span: usize,
    n_seq: usize,
    layer_index: usize,
    prefix_tokens: usize,
    layer_caches_by_sequence: &[&[Qwen3AsrLayerKvCacheState]],
) -> Result<(Vec<f32>, Vec<f32>), GgmlCpuGraphError> {
    let plane_elems =
        dims.k_width
            .checked_mul(kv_span)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill history plane size overflow",
            })?;
    let total_elems =
        plane_elems
            .checked_mul(n_seq)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill history tensor size overflow",
            })?;
    let mut key_values = vec![0.0_f32; total_elems];
    let mut value_values = vec![0.0_f32; total_elems];
    if prefix_tokens == 0 {
        return Ok((key_values, value_values));
    }
    if layer_caches_by_sequence.len() != n_seq {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "whole-decoder prefill history sequence count mismatch",
        });
    }
    let prefix_per_head =
        prefix_tokens
            .checked_mul(dims.head_dim)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill history prefix size overflow",
            })?;
    let target_head_stride =
        kv_span
            .checked_mul(dims.head_dim)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill history stride overflow",
            })?;
    for (sequence_index, sequence_layers) in layer_caches_by_sequence.iter().enumerate() {
        let cache =
            sequence_layers
                .get(layer_index)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder prefill history layer/cache count mismatch",
                })?;
        let history =
            cache
                .full_history_storage()
                .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder prefill history host cache storage invalid",
                })?;
        if history.head_dim != dims.head_dim || history.kv_heads != dims.kv_heads {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill history host cache shape mismatch",
            });
        }
        if history.written_positions < prefix_tokens {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill history requested unwritten prefix",
            });
        }
        let source_head_stride = history.max_positions.checked_mul(dims.head_dim).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill history source stride overflow",
            },
        )?;
        let sequence_plane = sequence_index.checked_mul(plane_elems).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill history sequence offset overflow",
            },
        )?;
        for kv_head in 0..dims.kv_heads {
            let source_start = kv_head.checked_mul(source_head_stride).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder prefill history source offset overflow",
                },
            )?;
            let source_end = source_start.checked_add(prefix_per_head).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder prefill history source end overflow",
                },
            )?;
            let target_start = sequence_plane
                .checked_add(kv_head.checked_mul(target_head_stride).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder prefill history target offset overflow",
                    },
                )?)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder prefill history target offset overflow",
                })?;
            let target_end = target_start.checked_add(prefix_per_head).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder prefill history target end overflow",
                },
            )?;
            key_values[target_start..target_end]
                .copy_from_slice(&history.keys[source_start..source_end]);
            value_values[target_start..target_end]
                .copy_from_slice(&history.values[source_start..source_end]);
        }
    }
    Ok((key_values, value_values))
}

fn seed_qwen_batched_resident_kv_arena(
    resident_kv_arena: &mut LlmResidentKvArena,
    head_dim: usize,
    max_positions: usize,
    kv_heads: usize,
    prefix_lengths: &[usize],
    layer_kv_by_sequence: &[&[Qwen3AsrLayerKvCacheState]],
) -> Result<(), GgmlCpuGraphError> {
    if prefix_lengths.is_empty() {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "batched resident KV seed sequence count must be positive",
        });
    }
    if layer_kv_by_sequence.len() != prefix_lengths.len() {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "batched resident KV seed sequence count mismatch",
        });
    }
    let layer_count = resident_kv_arena.layers.len();
    if layer_kv_by_sequence
        .iter()
        .any(|sequence_layers| sequence_layers.len() != layer_count)
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "batched resident KV seed layer count mismatch",
        });
    }
    let plane_elems = qwen_resident_kv_plane_elems(head_dim, max_positions, kv_heads)?;
    let tensor_elems = plane_elems.checked_mul(prefix_lengths.len()).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "batched resident KV seed tensor size overflow",
        },
    )?;

    for layer_index in 0..layer_count {
        let mut key_planes = vec![0.0_f32; tensor_elems];
        let mut value_planes = vec![0.0_f32; tensor_elems];
        for (sequence_index, sequence_layers) in layer_kv_by_sequence.iter().enumerate() {
            let history = sequence_layers[layer_index]
                .full_history_storage()
                .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                    reason: "batched resident KV seed host cache storage invalid",
                })?;
            if history.head_dim != head_dim
                || history.max_positions != max_positions
                || history.kv_heads != kv_heads
            {
                return Err(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "batched resident KV seed host cache shape mismatch",
                });
            }
            if history.written_positions != prefix_lengths[sequence_index] {
                return Err(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "batched resident KV seed written prefix mismatch",
                });
            }
            if history.keys.len() != plane_elems || history.values.len() != plane_elems {
                return Err(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "batched resident KV seed host cache plane length mismatch",
                });
            }
            let plane_start = sequence_index.checked_mul(plane_elems).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "batched resident KV seed plane offset overflow",
                },
            )?;
            let plane_end = plane_start.checked_add(plane_elems).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "batched resident KV seed plane offset overflow",
                },
            )?;
            key_planes[plane_start..plane_end].copy_from_slice(history.keys);
            value_planes[plane_start..plane_end].copy_from_slice(history.values);
        }
        let layer = resident_kv_arena.layers[layer_index];
        // The resident arena is f16: convert the f32 host planes once at seed
        // time (set_rows performs the same cast for rows written in-graph).
        resident_kv_arena.arena.set_f16_bits_slice(
            layer.key,
            &f32_slice_to_f16_bits(&key_planes),
            "qwen_llm_resident_kv_seed_key",
        )?;
        resident_kv_arena.arena.set_f16_bits_slice(
            layer.value,
            &f32_slice_to_f16_bits(&value_planes),
            "qwen_llm_resident_kv_seed_value",
        )?;
    }
    Ok(())
}

fn qwen_batched_seed_written_prefix_lengths(
    layer_kv_by_sequence: &[&[Qwen3AsrLayerKvCacheState]],
) -> Result<Vec<usize>, GgmlCpuGraphError> {
    if layer_kv_by_sequence.is_empty() {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "batched resident KV seed sequence count must be positive",
        });
    }
    let mut prefix_lengths = Vec::with_capacity(layer_kv_by_sequence.len());
    for sequence_layers in layer_kv_by_sequence {
        let first_layer = sequence_layers
            .first()
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "batched resident KV seed layer count mismatch",
            })?;
        let first_history = first_layer.full_history_storage().map_err(|_| {
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "batched resident KV seed host cache storage invalid",
            }
        })?;
        let prefix_length = first_history.written_positions;
        for layer in *sequence_layers {
            let history =
                layer
                    .full_history_storage()
                    .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV seed host cache storage invalid",
                    })?;
            if history.written_positions != prefix_length {
                return Err(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "batched resident KV seed layer prefix mismatch",
                });
            }
        }
        prefix_lengths.push(prefix_length);
    }
    Ok(prefix_lengths)
}

fn qwen_resident_kv_plane_elems(
    head_dim: usize,
    max_positions: usize,
    kv_heads: usize,
) -> Result<usize, GgmlCpuGraphError> {
    head_dim
        .checked_mul(max_positions)
        .and_then(|n| n.checked_mul(kv_heads))
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "batched resident KV slot plane size overflow",
        })
}

#[allow(dead_code)]
fn seed_qwen_batched_resident_kv_slot(
    resident_kv_arena: &mut LlmResidentKvArena,
    head_dim: usize,
    max_positions: usize,
    kv_heads: usize,
    slot_index: usize,
    cache_position: usize,
    layer_kv: &[Qwen3AsrLayerKvCacheState],
) -> Result<(), GgmlCpuGraphError> {
    if layer_kv.len() != resident_kv_arena.layers.len() {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "batched resident KV slot seed layer count mismatch",
        });
    }
    let plane_elems = qwen_resident_kv_plane_elems(head_dim, max_positions, kv_heads)?;
    let plane_offset =
        slot_index
            .checked_mul(plane_elems)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "batched resident KV slot seed offset overflow",
            })?;
    for (layer_index, cache) in layer_kv.iter().enumerate() {
        let history =
            cache
                .full_history_storage()
                .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                    reason: "batched resident KV slot seed host cache storage invalid",
                })?;
        if history.head_dim != head_dim
            || history.max_positions != max_positions
            || history.kv_heads != kv_heads
        {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "batched resident KV slot seed host cache shape mismatch",
            });
        }
        if history.written_positions != cache_position {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "batched resident KV slot seed written prefix mismatch",
            });
        }
        if history.keys.len() != plane_elems || history.values.len() != plane_elems {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "batched resident KV slot seed host cache plane length mismatch",
            });
        }
        let layer = resident_kv_arena.layers[layer_index];
        resident_kv_arena.arena.set_f16_bits_slice_with_offset(
            layer.key,
            plane_offset,
            &f32_slice_to_f16_bits(history.keys),
            "qwen_llm_resident_kv_slot_seed_key",
        )?;
        resident_kv_arena.arena.set_f16_bits_slice_with_offset(
            layer.value,
            plane_offset,
            &f32_slice_to_f16_bits(history.values),
            "qwen_llm_resident_kv_slot_seed_value",
        )?;
    }
    Ok(())
}

#[allow(dead_code)]
fn zero_qwen_batched_resident_kv_slot(
    resident_kv_arena: &mut LlmResidentKvArena,
    head_dim: usize,
    max_positions: usize,
    kv_heads: usize,
    slot_index: usize,
) -> Result<(), GgmlCpuGraphError> {
    let plane_elems = qwen_resident_kv_plane_elems(head_dim, max_positions, kv_heads)?;
    let plane_offset =
        slot_index
            .checked_mul(plane_elems)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "batched resident KV slot zero offset overflow",
            })?;
    let zeros = vec![0_u16; plane_elems];
    for layer in &resident_kv_arena.layers {
        resident_kv_arena.arena.set_f16_bits_slice_with_offset(
            layer.key,
            plane_offset,
            &zeros,
            "qwen_llm_resident_kv_slot_zero_key",
        )?;
        resident_kv_arena.arena.set_f16_bits_slice_with_offset(
            layer.value,
            plane_offset,
            &zeros,
            "qwen_llm_resident_kv_slot_zero_value",
        )?;
    }
    Ok(())
}

fn new_projection_tensor_in_arena(
    arena: &GgmlStaticTensorArena,
    weight: &DenseProjectionWeight,
    tensor_name: &'static str,
) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
    if let Some(raw) = &weight.raw_ggml {
        return arena.new_matmul_weight_2d_typed(
            raw.dims[0],
            raw.dims[1],
            raw.ggml_type,
            tensor_name,
        );
    }
    arena.new_tensor_2d_f32(weight.input_width, weight.output_width, tensor_name)
}

fn new_fused_qkv_tensor_in_arena(
    arena: &GgmlStaticTensorArena,
    weight: &FusedQkvProjectionWeight,
    tensor_name: &'static str,
) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
    if let Some(raw) = &weight.raw_ggml {
        return arena.new_matmul_weight_2d_typed(
            raw.dims[0],
            raw.dims[1],
            raw.ggml_type,
            tensor_name,
        );
    }
    arena.new_tensor_2d_f32(weight.input_width, weight.output_width, tensor_name)
}

fn upload_projection_weight_to_arena(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &DenseProjectionWeight,
    tensor_name: &'static str,
) -> Result<(), GgmlCpuGraphError> {
    if let Some(raw) = &weight.raw_ggml {
        return arena.set_bytes_slice(tensor, &raw.bytes, tensor_name);
    }
    if weight.values.is_empty() {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "projection weight is missing materialized f32 values",
        });
    }
    let values = projection_values_for_ggml(
        weight.input_width,
        weight.output_width,
        &weight.values,
        weight.layout,
    )?;
    arena.set_f32_slice(tensor, &values, tensor_name)?;
    Ok(())
}

fn upload_fused_qkv_weight_to_arena(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &FusedQkvProjectionWeight,
    tensor_name: &'static str,
) -> Result<(), GgmlCpuGraphError> {
    if let Some(raw) = &weight.raw_ggml {
        return arena.set_bytes_slice(tensor, &raw.bytes, tensor_name);
    }
    let values = weight
        .values
        .as_ref()
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "fused qkv weight is missing upload payload",
        })?;
    arena.set_f32_slice(tensor, values, tensor_name)?;
    Ok(())
}

fn fuse_raw_qkv_projection_weights(
    q_weight: &DenseProjectionWeight,
    k_weight: &DenseProjectionWeight,
    v_weight: &DenseProjectionWeight,
) -> Result<Option<OwnedGgmlProjectionWeight>, GgmlCpuGraphError> {
    let (Some(q_raw), Some(k_raw), Some(v_raw)) = (
        q_weight.raw_ggml.as_ref(),
        k_weight.raw_ggml.as_ref(),
        v_weight.raw_ggml.as_ref(),
    ) else {
        return Ok(None);
    };

    if q_raw.ggml_type != k_raw.ggml_type
        || q_raw.ggml_type != v_raw.ggml_type
        || q_raw.dims.len() != 2
        || k_raw.dims.len() != 2
        || v_raw.dims.len() != 2
        || q_raw.dims[0] != k_raw.dims[0]
        || q_raw.dims[0] != v_raw.dims[0]
    {
        return Ok(None);
    }

    let output_width = q_raw.dims[1]
        .checked_add(k_raw.dims[1])
        .and_then(|value| value.checked_add(v_raw.dims[1]))
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "fused raw qkv projection width overflow",
        })?;
    let mut bytes = Vec::with_capacity(
        q_raw
            .bytes
            .len()
            .checked_add(k_raw.bytes.len())
            .and_then(|value| value.checked_add(v_raw.bytes.len()))
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "fused raw qkv byte width overflow",
            })?,
    );
    bytes.extend_from_slice(&q_raw.bytes);
    bytes.extend_from_slice(&k_raw.bytes);
    bytes.extend_from_slice(&v_raw.bytes);
    Ok(Some(OwnedGgmlProjectionWeight {
        ggml_type: q_raw.ggml_type,
        dims: vec![q_raw.dims[0], output_width],
        bytes,
    }))
}

fn projection_values_for_ggml(
    input_width: usize,
    output_width: usize,
    values: &[f32],
    layout: DenseProjectionLayout,
) -> Result<Vec<f32>, GgmlCpuGraphError> {
    match layout {
        DenseProjectionLayout::OutputByInput => Ok(values.to_vec()),
        DenseProjectionLayout::InputByOutput => {
            let mut transposed = vec![0.0_f32; values.len()];
            for input_idx in 0..input_width {
                let src_start = input_idx.checked_mul(output_width).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "dense projection transpose overflow",
                    },
                )?;
                for output_idx in 0..output_width {
                    let dst_idx = output_idx
                        .checked_mul(input_width)
                        .and_then(|base| base.checked_add(input_idx))
                        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                            reason: "dense projection transpose overflow",
                        })?;
                    transposed[dst_idx] = values[src_start + output_idx];
                }
            }
            Ok(transposed)
        }
    }
}

pub(crate) fn load_qwen3_llm_layer_attention_projection(
    reader: &GgufTensorDataReader,
    metadata: Qwen3AsrExecutionMetadata,
    layer_index: usize,
) -> Result<Qwen3AsrLlmLayerAttentionProjection, Qwen3AsrLlmTransformerError> {
    Ok(Qwen3AsrLlmLayerAttentionProjection::Generic(
        load_qwen3_llm_layer_attention_projection_generic(reader, metadata, layer_index, false)?,
    ))
}

pub(crate) fn load_qwen3_llm_attention_projections_from_reader(
    reader: &GgufTensorDataReader,
    metadata: Qwen3AsrExecutionMetadata,
) -> Result<Vec<Qwen3AsrLlmLayerAttentionProjection>, Qwen3AsrLlmTransformerError> {
    let mut projections = Vec::with_capacity(metadata.llm_layers);
    for layer_index in 0..metadata.llm_layers {
        let projection = load_qwen3_llm_layer_attention_projection(reader, metadata, layer_index)?;
        projections.push(projection);
    }
    Ok(projections)
}

pub(crate) fn load_qwen3_llm_attention_projections_from_reader_with_materialized_qkv(
    reader: &GgufTensorDataReader,
    metadata: Qwen3AsrExecutionMetadata,
) -> Result<Vec<Qwen3AsrLlmLayerAttentionProjection>, Qwen3AsrLlmTransformerError> {
    let mut projections = Vec::with_capacity(metadata.llm_layers);
    for layer_index in 0..metadata.llm_layers {
        projections.push(Qwen3AsrLlmLayerAttentionProjection::Generic(
            load_qwen3_llm_layer_attention_projection_generic(reader, metadata, layer_index, true)?,
        ));
    }
    Ok(projections)
}

fn load_qwen3_llm_layer_attention_projection_generic(
    reader: &GgufTensorDataReader,
    metadata: Qwen3AsrExecutionMetadata,
    layer_index: usize,
    materialize_qkv: bool,
) -> Result<Qwen3AsrLlmLayerAttentionProjectionGeneric, Qwen3AsrLlmTransformerError> {
    let names = llm_layer_tensor_names(layer_index);
    load_qwen_family_llm_layer_attention_projection_generic(
        reader,
        QwenFamilyLlmLayerTensorNames {
            attn_norm_name: names.attn_norm_weight,
            attn_q_name: names.attn_q_weight,
            attn_k_name: names.attn_k_weight,
            attn_v_name: names.attn_v_weight,
            attn_output_name: names.attn_output_weight,
            // Qwen3-ASR always has QK-norm and never has attention bias.
            q_norm_name: Some(names.attn_q_norm_weight),
            k_norm_name: Some(names.attn_k_norm_weight),
            q_bias_name: None,
            k_bias_name: None,
            v_bias_name: None,
            ffn_norm_name: names.ffn_norm_weight,
            ffn_gate_name: names.ffn_gate_weight,
            ffn_up_name: names.ffn_up_weight,
            ffn_down_name: names.ffn_down_weight,
        },
        metadata.llm_d_model,
        metadata.llm_heads,
        metadata.llm_kv_heads,
        metadata.llm_head_dim,
        materialize_qkv,
    )
}

/// Tensor names for one decoder layer, resolved by the caller's family-specific
/// naming scheme (qwen3-asr's `blk.N.*` vs firered-llm's `llm.blk.N.*`) --
/// this loader stays name-agnostic. `q_norm_name`/`k_norm_name` are `Some`
/// IFF the family applies QK-norm (Qwen3's shape); `*_bias_name` are `Some`
/// IFF the family has attention bias (Qwen2's shape, the inverse of Qwen3).
pub(crate) struct QwenFamilyLlmLayerTensorNames {
    pub attn_norm_name: String,
    pub attn_q_name: String,
    pub attn_k_name: String,
    pub attn_v_name: String,
    pub attn_output_name: String,
    pub q_norm_name: Option<String>,
    pub k_norm_name: Option<String>,
    pub q_bias_name: Option<String>,
    pub k_bias_name: Option<String>,
    pub v_bias_name: Option<String>,
    pub ffn_norm_name: String,
    pub ffn_gate_name: String,
    pub ffn_up_name: String,
    pub ffn_down_name: String,
}

/// Load one decoder-only LLM layer's projections from `reader`, parameterized
/// over the two axes that differ between Qwen2 and Qwen3 (QK-norm,
/// attention bias) via `names`' `Option` fields, rather than hard-coding
/// either family's shape. Shared by qwen3-asr
/// (`load_qwen3_llm_layer_attention_projection_generic`, always QK-norm, never
/// bias) and firered-llm (always bias, never QK-norm -- see
/// `models::firered_llm::llm_transformer`).
pub(crate) fn load_qwen_family_llm_layer_attention_projection_generic(
    reader: &GgufTensorDataReader,
    names: QwenFamilyLlmLayerTensorNames,
    d_model: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    materialize_qkv: bool,
) -> Result<Qwen3AsrLlmLayerAttentionProjectionGeneric, Qwen3AsrLlmTransformerError> {
    let attn_norm_weight = load_vector_weight(reader, &names.attn_norm_name, d_model)?;
    let q_norm_weight = match &names.q_norm_name {
        Some(name) => load_non_empty_vector_weight(reader, name)?,
        None => Vec::new(),
    };
    let k_norm_weight = match &names.k_norm_name {
        Some(name) => load_non_empty_vector_weight(reader, name)?,
        None => Vec::new(),
    };
    let q_output_width = projection_output_width(n_heads, head_dim)?;
    let kv_output_width = projection_output_width(n_kv_heads, head_dim)?;
    // q is non-square under GQA, so its storage orientation is unambiguous;
    // load it with explicit geometry and reuse its resolved layout for the
    // (possibly square) k/v projections. This guarantees q/k/v share one
    // orientation, so the fused-QKV builder never sees a mixed raw/dense state.
    let q_weight = load_projection_weight_with_input_output(
        reader,
        &names.attn_q_name,
        d_model,
        q_output_width,
        materialize_qkv,
    )?;
    // Fail closed on stale packs that stored projections in PyTorch [out,in]
    // order. Correct qwen-family packs follow the ggml [in,out] convention,
    // under which the non-square GQA q-projection resolves to
    // OutputByInput. A q resolving to InputByOutput means the dims were
    // written reversed by an older importer, which would otherwise silently
    // produce garbage tokens rather than fail.
    if q_weight.layout != DenseProjectionLayout::OutputByInput {
        return Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: names.attn_q_name.clone(),
            shape: format!("[output={q_output_width}, input={d_model}]"),
            reason: "qwen-family projection weights must use the ggml [in, out] dim order; \
                     this pack stores them as [out, in], which indicates it was built by \
                     an older importer — re-pack from source with the current build"
                .to_string(),
        });
    }
    let k_weight = load_projection_weight_with_layout(
        reader,
        &names.attn_k_name,
        d_model,
        kv_output_width,
        q_weight.layout,
        materialize_qkv,
    )?;
    let v_weight = load_projection_weight_with_layout(
        reader,
        &names.attn_v_name,
        d_model,
        kv_output_width,
        q_weight.layout,
        materialize_qkv,
    )?;
    let q_bias = match &names.q_bias_name {
        Some(name) => load_vector_weight(reader, name, q_weight.output_width)?,
        None => Vec::new(),
    };
    let k_bias = match &names.k_bias_name {
        Some(name) => load_vector_weight(reader, name, k_weight.output_width)?,
        None => Vec::new(),
    };
    let v_bias = match &names.v_bias_name {
        Some(name) => load_vector_weight(reader, name, v_weight.output_width)?,
        None => Vec::new(),
    };
    let attn_output_weight = load_projection_weight_with_input_output(
        reader,
        &names.attn_output_name,
        q_weight.output_width,
        d_model,
        false,
    )?;
    let ffn_norm_weight = load_vector_weight(reader, &names.ffn_norm_name, d_model)?;
    let ffn_gate_weight = load_projection_weight(reader, &names.ffn_gate_name, d_model)?;
    let ffn_up_weight = load_projection_weight(reader, &names.ffn_up_name, d_model)?;
    if ffn_gate_weight.output_width != ffn_up_weight.output_width {
        return Err(Qwen3AsrLlmTransformerError::FfnProjectionWidthMismatch {
            gate_width: ffn_gate_weight.output_width,
            up_width: ffn_up_weight.output_width,
        });
    }
    let ffn_down_weight = load_projection_weight_with_input_output(
        reader,
        &names.ffn_down_name,
        ffn_gate_weight.output_width,
        d_model,
        false,
    )?;

    Ok(Qwen3AsrLlmLayerAttentionProjectionGeneric {
        d_model,
        head_dim,
        attn_norm_name: names.attn_norm_name,
        attn_q_name: names.attn_q_name,
        attn_k_name: names.attn_k_name,
        attn_v_name: names.attn_v_name,
        attn_output_name: names.attn_output_name,
        ffn_gate_name: names.ffn_gate_name,
        ffn_up_name: names.ffn_up_name,
        ffn_down_name: names.ffn_down_name,
        attn_norm_weight,
        q_weight,
        k_weight,
        v_weight,
        // output/gate/up/down are bound zero-copy from the mmap'd pack at decode
        // (goals 7+8), so drop their resident host payload here — the ~hundreds
        // of MB this cached struct otherwise holds. `bind_or_arena_llm` fails
        // closed if the zero-copy binding is somehow unavailable.
        attn_output_weight: dropped_projection_payload(attn_output_weight),
        ffn_norm_weight,
        ffn_gate_weight: dropped_projection_payload(ffn_gate_weight),
        ffn_up_weight: dropped_projection_payload(ffn_up_weight),
        ffn_down_weight: dropped_projection_payload(ffn_down_weight),
        q_norm_weight,
        k_norm_weight,
        q_bias,
        k_bias,
        v_bias,
    })
}

/// Drop a projection's resident host payload (f32 values + raw native bytes),
/// keeping its shape metadata (input/output width, layout, dims/type). Used for
/// weights bound zero-copy at decode — the host copy is dead weight in the
/// cached prepared-runtime projections.
fn dropped_projection_payload(mut weight: DenseProjectionWeight) -> DenseProjectionWeight {
    // Only native [in,out] weights (raw_ggml present) are bound zero-copy — drop
    // their host bytes. f32-fallback weights KEEP their `values`: the arena path
    // is their only route (the loaded path can't fix their on-disk orientation).
    if let Some(raw) = weight.raw_ggml.as_mut() {
        raw.bytes = Vec::new();
    }
    weight
}

fn load_projection_weight(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    d_model: usize,
) -> Result<DenseProjectionWeight, Qwen3AsrLlmTransformerError> {
    let metadata = reader.tensor_index().get(tensor_name).ok_or_else(|| {
        Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: "[]".to_string(),
            reason: "tensor is missing from GGUF tensor index".to_string(),
        }
    })?;
    let dims = metadata.dims.clone();
    let (input_width, output_width, layout) =
        parse_projection_shape_for_input(tensor_name, &dims, d_model)?;
    let raw_ggml = load_direct_projection_weight_payload(
        reader,
        tensor_name,
        input_width,
        output_width,
        layout,
    )?;
    let values = if raw_ggml.is_none() {
        reader
            .host_tensor_f32_copy_dequantized_by_name(tensor_name, &dims)
            .map_err(map_tensor_read_error)?
    } else {
        Vec::new()
    };
    DenseProjectionWeight::new(
        tensor_name,
        input_width,
        output_width,
        values,
        layout,
        raw_ggml,
    )
}

fn load_projection_weight_with_input_output(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    input_width: usize,
    output_width: usize,
    materialize_if_raw: bool,
) -> Result<DenseProjectionWeight, Qwen3AsrLlmTransformerError> {
    let metadata = reader.tensor_index().get(tensor_name).ok_or_else(|| {
        Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: "[]".to_string(),
            reason: "tensor is missing from GGUF tensor index".to_string(),
        }
    })?;
    let dims = metadata.dims.clone();
    if dims.len() != 2 {
        return Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: render_shape(&dims),
            reason: "expected rank-2 matrix".to_string(),
        });
    }
    let dim0 = dims[0] as usize;
    let dim1 = dims[1] as usize;
    if dim0 == input_width && dim1 == output_width {
        let raw_ggml = load_direct_projection_weight_payload(
            reader,
            tensor_name,
            input_width,
            output_width,
            DenseProjectionLayout::OutputByInput,
        )?;
        let values = if raw_ggml.is_none() || materialize_if_raw {
            reader
                .host_tensor_f32_copy_dequantized_by_name(tensor_name, &dims)
                .map_err(map_tensor_read_error)?
        } else {
            Vec::new()
        };
        return DenseProjectionWeight::new(
            tensor_name,
            input_width,
            output_width,
            values,
            DenseProjectionLayout::OutputByInput,
            raw_ggml,
        );
    }
    if dim0 == output_width && dim1 == input_width {
        let values = reader
            .host_tensor_f32_copy_dequantized_by_name(tensor_name, &dims)
            .map_err(map_tensor_read_error)?;
        return DenseProjectionWeight::new(
            tensor_name,
            input_width,
            output_width,
            values,
            DenseProjectionLayout::InputByOutput,
            None,
        );
    }
    Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
        tensor_name: tensor_name.to_string(),
        shape: render_shape(&dims),
        reason: format!(
            "expected [{} x {}] or [{} x {}]",
            input_width, output_width, output_width, input_width
        ),
    })
}

fn projection_output_width(
    heads: usize,
    head_dim: usize,
) -> Result<usize, Qwen3AsrLlmTransformerError> {
    heads
        .checked_mul(head_dim)
        .ok_or_else(|| Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: "<qkv projection>".to_string(),
            shape: format!("heads={heads} head_dim={head_dim}"),
            reason: "qkv projection output width overflow".to_string(),
        })
}

/// Loads a projection weight with an explicit `(input, output)` geometry under a
/// caller-supplied storage `layout`, never guessing orientation.
///
/// q/k/v in one attention layer are written with a single orientation; the
/// square k/v matrices (when `kv_heads * head_dim == d_model`) are ambiguous on
/// their own, so the caller resolves the layout from the non-square q
/// projection and forces it here. This keeps all three projections on one
/// orientation, so the fused-QKV path cannot land on a mixed raw/dense state.
fn load_projection_weight_with_layout(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    input_width: usize,
    output_width: usize,
    layout: DenseProjectionLayout,
    materialize_if_raw: bool,
) -> Result<DenseProjectionWeight, Qwen3AsrLlmTransformerError> {
    let metadata = reader.tensor_index().get(tensor_name).ok_or_else(|| {
        Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: "[]".to_string(),
            reason: "tensor is missing from GGUF tensor index".to_string(),
        }
    })?;
    let dims = metadata.dims.clone();
    if dims.len() != 2 {
        return Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: render_shape(&dims),
            reason: "expected rank-2 matrix".to_string(),
        });
    }
    let expected = match layout {
        DenseProjectionLayout::OutputByInput => [input_width as u64, output_width as u64],
        DenseProjectionLayout::InputByOutput => [output_width as u64, input_width as u64],
    };
    if dims.as_slice() != expected {
        return Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: render_shape(&dims),
            reason: format!("expected {expected:?} for the layer's resolved projection layout"),
        });
    }
    let raw_ggml = load_direct_projection_weight_payload(
        reader,
        tensor_name,
        input_width,
        output_width,
        layout,
    )?;
    let values = if raw_ggml.is_none() || materialize_if_raw {
        reader
            .host_tensor_f32_copy_dequantized_by_name(tensor_name, &dims)
            .map_err(map_tensor_read_error)?
    } else {
        Vec::new()
    };
    DenseProjectionWeight::new(
        tensor_name,
        input_width,
        output_width,
        values,
        layout,
        raw_ggml,
    )
}

fn parse_projection_shape_for_input(
    tensor_name: &str,
    dims: &[u64],
    expected_input_width: usize,
) -> Result<(usize, usize, DenseProjectionLayout), Qwen3AsrLlmTransformerError> {
    if dims.len() != 2 {
        return Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: render_shape(dims),
            reason: "expected rank-2 matrix".to_string(),
        });
    }
    let dim0 = dims[0] as usize;
    let dim1 = dims[1] as usize;
    if dim0 == expected_input_width {
        return Ok((
            expected_input_width,
            dim1,
            DenseProjectionLayout::OutputByInput,
        ));
    }
    if dim1 == expected_input_width {
        return Ok((
            expected_input_width,
            dim0,
            DenseProjectionLayout::InputByOutput,
        ));
    }
    Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
        tensor_name: tensor_name.to_string(),
        shape: render_shape(dims),
        reason: format!("expected one dimension to equal hidden_size={expected_input_width}"),
    })
}

fn load_direct_projection_weight_payload(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    input_width: usize,
    output_width: usize,
    layout: DenseProjectionLayout,
) -> Result<Option<OwnedGgmlProjectionWeight>, Qwen3AsrLlmTransformerError> {
    if layout != DenseProjectionLayout::OutputByInput {
        return Ok(None);
    }
    let payload = reader
        .weight_tensor_payload_by_name(tensor_name)
        .map_err(map_tensor_read_error)?;
    if payload.dims.as_slice() != [input_width, output_width] {
        return Ok(None);
    }
    Ok(Some(OwnedGgmlProjectionWeight {
        ggml_type: payload.element_type.ggml_type(),
        dims: payload.dims,
        bytes: payload.bytes.to_vec(),
    }))
}

fn load_vector_weight(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
    expected_len: usize,
) -> Result<Vec<f32>, Qwen3AsrLlmTransformerError> {
    let dims = vec![expected_len as u64];
    let values = reader
        .host_tensor_f32_copy_dequantized_by_name(tensor_name, &dims)
        .map_err(map_tensor_read_error)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Qwen3AsrLlmTransformerError::NonFiniteTensorValues {
            tensor_name: tensor_name.to_string(),
        });
    }
    Ok(values)
}

fn load_non_empty_vector_weight(
    reader: &GgufTensorDataReader,
    tensor_name: &str,
) -> Result<Vec<f32>, Qwen3AsrLlmTransformerError> {
    let metadata = reader.tensor_index().get(tensor_name).ok_or_else(|| {
        Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: "[]".to_string(),
            reason: "tensor is missing from GGUF tensor index".to_string(),
        }
    })?;
    if metadata.dims.len() != 1 || metadata.dims[0] == 0 {
        return Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: render_shape(&metadata.dims),
            reason: "expected non-empty rank-1 vector".to_string(),
        });
    }
    let values = reader
        .host_tensor_f32_copy_dequantized_by_name(tensor_name, &metadata.dims)
        .map_err(map_tensor_read_error)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Qwen3AsrLlmTransformerError::NonFiniteTensorValues {
            tensor_name: tensor_name.to_string(),
        });
    }
    Ok(values)
}

/// Wraps `nn::norm::apply_rms_norm` for code paths that propagate `GgmlCpuGraphError` directly.
#[inline(always)]
fn rms_norm_with_weight(
    hidden: &[f32],
    weight: &[f32],
    epsilon: f32,
    tensor_name: &str,
) -> Result<Vec<f32>, Qwen3AsrLlmTransformerError> {
    if hidden.len() != weight.len() {
        return Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: format!("[{}]", weight.len()),
            reason: format!(
                "must match hidden_size={}, got {}",
                hidden.len(),
                weight.len()
            ),
        });
    }
    let mut ss = 0.0_f32;
    for value in hidden {
        ss += value * value;
    }
    let inv_rms = (ss / hidden.len() as f32 + epsilon).sqrt().recip();
    let mut out = vec![0.0_f32; hidden.len()];
    for idx in 0..hidden.len() {
        out[idx] = hidden[idx] * inv_rms * weight[idx];
    }
    Ok(out)
}

fn apply_segmented_rms_norm_with_weight(
    values: &mut [f32],
    weight: &[f32],
    epsilon: f32,
) -> Result<(), Qwen3AsrLlmTransformerError> {
    let norm_width = weight.len();
    if norm_width == 0 || !values.len().is_multiple_of(norm_width) {
        return Err(Qwen3AsrLlmTransformerError::QkNormWidthMismatch {
            vector_width: values.len(),
            norm_width,
        });
    }
    for chunk in values.chunks_exact_mut(norm_width) {
        let mut ss = 0.0_f32;
        for value in chunk.iter().copied() {
            ss += value * value;
        }
        let inv_rms = (ss / norm_width as f32 + epsilon).sqrt().recip();
        for idx in 0..norm_width {
            chunk[idx] = chunk[idx] * inv_rms * weight[idx];
        }
    }
    Ok(())
}

fn map_tensor_read_error(error: GgufTensorDataReadError) -> Qwen3AsrLlmTransformerError {
    Qwen3AsrLlmTransformerError::TensorReadFailed {
        reason: error.to_string(),
    }
}

fn render_shape(shape: &[u64]) -> String {
    let parts = shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{parts}]")
}

fn qwen_llm_safe_gpu_prefill_query_tokens_for_backend_name(
    backend_name: &str,
    token_count: usize,
) -> usize {
    if qwen_llm_backend_is_hip_like(backend_name) {
        // HIP-like backends are no longer bound by the flash MMA/TILE kernel
        // limits for wide prefill: any chunk that would trip the kernel bug
        // (n_query > 2 with n_kv > 32) is routed to the unfused
        // `llm_naive_masked_attention` path instead (see
        // `llm_prefill_uses_flash_attention`), which is correct at any width.
        if token_count <= QWEN3_LLM_HIP_SHORT_PREFILL_MAX_TOKENS {
            return QWEN3_LLM_HIP_SHORT_PREFILL_QUERY_TOKENS;
        }
        return QWEN3_LLM_HIP_NONFLASH_PREFILL_QUERY_TOKENS;
    }
    QWEN3_LLM_GPU_SAFE_PREFILL_QUERY_TOKENS
}

fn qwen_llm_backend_is_hip_like(backend_name: &str) -> bool {
    let backend_name = backend_name.to_ascii_lowercase();
    backend_name.contains("hip") || backend_name.contains("rocm")
}

/// Next chunk width for a prefill loop whose backend reports
/// `prefill_chunks_require_even_width`: an odd width > 1 is trimmed down by
/// one token so every multi-token chunk stays on the fast even-width HIP
/// kernels; the loop then finishes with a fast width-1 step.
pub(crate) fn even_prefill_chunk_len(remaining: usize, chunk_size: usize) -> usize {
    let width = remaining.min(chunk_size);
    if width > 1 && width % 2 == 1 {
        width - 1
    } else {
        width
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::ggml_runtime::{GGML_TYPE_F32, GgmlCpuGraphConfig, GgmlCpuGraphRunner};
    use crate::models::qwen::runtime_contract::parse_qwen3_execution_metadata;
    use crate::testing::with_forced_cpu_backend_for_test;
    use crate::{read_gguf_metadata_from_runtime_source, validate_ggml_runtime_source_path};

    const QWEN_PREFILL_REAL_PACK_ENV: &str = "OPENASR_QWEN_PREFILL_REAL_PACK";
    const QWEN_PREFILL_TOKENS_ENV: &str = "OPENASR_QWEN_PREFILL_TOKENS";
    const QWEN_PREFILL_CHUNK_TOKENS_ENV: &str = "OPENASR_QWEN_PREFILL_CHUNK_TOKENS";

    #[test]
    fn qwen_llm_native_gqa_default_is_on_for_cpu_metal_off_for_gpu() {
        assert!(qwen_llm_native_gqa_default_for_backend(
            GgmlCpuGraphBackend::Cpu
        ));
        assert!(qwen_llm_native_gqa_default_for_backend(
            GgmlCpuGraphBackend::Metal
        ));
        // The discrete-GPU lane (HIP/CUDA/Vulkan) mis-computes native GQA on
        // RDNA4 (gfx1200), so it must default off.
        assert!(!qwen_llm_native_gqa_default_for_backend(
            GgmlCpuGraphBackend::Gpu
        ));
    }

    #[test]
    fn qwen_llm_native_gqa_uses_backend_default_when_env_unset() {
        // No / unrecognized env value falls back to the supplied per-backend
        // default (true for CPU/Metal, false for the GPU lane).
        assert!(qwen_llm_native_gqa_enabled(None, true));
        assert!(qwen_llm_native_gqa_enabled(Some(""), true));
        assert!(qwen_llm_native_gqa_enabled(Some("native"), true));
        assert!(!qwen_llm_native_gqa_enabled(None, false));
        assert!(!qwen_llm_native_gqa_enabled(Some("maybe"), false));
    }

    #[test]
    fn qwen_llm_gpu_prefill_chunk_policy_widens_only_hip_like_backends() {
        assert_eq!(
            qwen_llm_safe_gpu_prefill_query_tokens_for_backend_name("HIP0", 8),
            QWEN3_LLM_HIP_SHORT_PREFILL_QUERY_TOKENS
        );
        assert_eq!(
            qwen_llm_safe_gpu_prefill_query_tokens_for_backend_name("HIP0", 32),
            QWEN3_LLM_HIP_SHORT_PREFILL_QUERY_TOKENS
        );
        assert_eq!(
            qwen_llm_safe_gpu_prefill_query_tokens_for_backend_name("HIP0", 33),
            QWEN3_LLM_HIP_NONFLASH_PREFILL_QUERY_TOKENS
        );
        assert_eq!(
            qwen_llm_safe_gpu_prefill_query_tokens_for_backend_name("ROCm0", 128),
            QWEN3_LLM_HIP_NONFLASH_PREFILL_QUERY_TOKENS
        );
        for backend_name in ["Vulkan0", "CUDA0", "Metal", "GPU", ""] {
            for token_count in [8, 32, 128] {
                assert_eq!(
                    qwen_llm_safe_gpu_prefill_query_tokens_for_backend_name(
                        backend_name,
                        token_count
                    ),
                    QWEN3_LLM_GPU_SAFE_PREFILL_QUERY_TOKENS,
                    "backend_name={backend_name} token_count={token_count}"
                );
            }
        }
    }

    #[test]
    fn even_prefill_chunk_len_trims_odd_multi_token_widths() {
        // Even widths and width 1 pass through untouched.
        assert_eq!(even_prefill_chunk_len(64, 8), 8);
        assert_eq!(even_prefill_chunk_len(6, 8), 6);
        assert_eq!(even_prefill_chunk_len(2, 8), 2);
        assert_eq!(even_prefill_chunk_len(1, 8), 1);
        // Odd widths > 1 are trimmed by one token so the chunk stays on the
        // fast even-width HIP kernels; the leftover token runs as width 1.
        assert_eq!(even_prefill_chunk_len(7, 8), 6);
        assert_eq!(even_prefill_chunk_len(5, 8), 4);
        assert_eq!(even_prefill_chunk_len(3, 8), 2);
        // The cap applies before the evenness trim.
        assert_eq!(even_prefill_chunk_len(65, 8), 8);
        assert_eq!(even_prefill_chunk_len(9, 7), 6);
    }

    #[test]
    fn qwen_llm_native_gqa_env_can_disable_or_enable() {
        // An explicit env value overrides the per-backend default both ways.
        assert!(!qwen_llm_native_gqa_enabled(Some("0"), true));
        assert!(!qwen_llm_native_gqa_enabled(Some("false"), true));
        assert!(qwen_llm_native_gqa_enabled(Some("1"), false));
        assert!(qwen_llm_native_gqa_enabled(Some("true"), false));
    }

    #[test]
    fn fused_logits_top1_selects_first_token_on_equal_logit_tie() {
        let config = GgmlCpuGraphConfig::default();
        let mut runner =
            GgmlCpuGraphRunner::new(config).expect("cpu graph runner should initialize");
        let mut arena = runner
            .start_static_tensor_arena(config.context_bytes)
            .expect("static arena should allocate");
        let dims = Qwen3AsrLlmDecodeDims {
            d_model: 2,
            q_width: 2,
            k_width: 2,
            v_width: 2,
            head_dim: 2,
            q_heads: 1,
            kv_heads: 1,
        };
        let output_weight_values = [
            0.1_f32, 0.0, //
            0.3, 0.0, //
            0.3, 0.0,
        ];
        let output_weight_bytes = output_weight_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let spec = Qwen3AsrLlmFusedLogitsHeadSpec {
            d_model: 2,
            vocab_size: 3,
            rms_norm_epsilon: DEFAULT_RMS_NORM_EPSILON,
            output_norm_weight: &[1.0, 1.0],
            output_weight_tensor_name: "synthetic.output.weight",
            output_weight_ggml_type: GGML_TYPE_F32,
            output_weight_dims: &[2, 3],
            output_weight_bytes: &output_weight_bytes,
        };
        let handles = allocate_fused_logits_head_tensors(&mut arena, None, dims, &spec)
            .expect("fused logits handles should allocate");
        upload_fused_logits_head_weights(&mut arena, &handles, &spec)
            .expect("fused logits weights should upload");

        let mut graph = runner.start_graph();
        let state = graph
            .new_tensor_2d_f32(2, 1, "synthetic_state")
            .expect("state tensor should allocate");
        graph.set_input(state).expect("state should be input");
        let top1 = build_fused_logits_top1(&arena, &handles, &mut graph, state, 1)
            .expect("fused top1 should build");
        graph.set_output(top1).expect("top1 should be output");
        graph
            .set_f32_slice(state, &[1.0, 0.0], "synthetic_state")
            .expect("state should upload");

        let reversed_top1 = graph
            .compute_output_i32(top1, 1)
            .expect("fused top1 should compute");
        let token_id = validate_fused_top1_token_id(reversed_top1[0], spec.vocab_size)
            .expect("top1 should map to a valid token");
        assert_eq!(token_id, 1);
    }

    #[test]
    fn dense_projection_accepts_both_matrix_layouts() {
        let input_by_output = DenseProjectionWeight::from_tensor(
            "blk.0.attn_q.weight",
            &[2, 3],
            vec![
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0,
            ],
            2,
        )
        .expect("input-by-output");
        let output_by_input = DenseProjectionWeight::from_tensor(
            "blk.0.attn_q.weight",
            &[3, 2],
            vec![
                1.0, 3.0, 5.0, //
                2.0, 4.0, 6.0,
            ],
            2,
        )
        .expect("output-by-input");

        let input = vec![2.0, 3.0];
        let lhs = input_by_output
            .project_row(&input, "blk.0.attn_q.weight")
            .expect("lhs");
        let rhs = output_by_input
            .project_row(&input, "blk.0.attn_q.weight")
            .expect("rhs");
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn fused_qkv_projection_weight_concatenates_f32_payloads() {
        let q_weight = DenseProjectionWeight {
            input_width: 2,
            output_width: 2,
            values: vec![1.0, 2.0, 3.0, 4.0],
            layout: DenseProjectionLayout::OutputByInput,
            raw_ggml: None,
        };
        let k_weight = DenseProjectionWeight {
            input_width: 2,
            output_width: 1,
            values: vec![5.0, 6.0],
            layout: DenseProjectionLayout::OutputByInput,
            raw_ggml: None,
        };
        let v_weight = DenseProjectionWeight {
            input_width: 2,
            output_width: 1,
            values: vec![7.0, 8.0],
            layout: DenseProjectionLayout::OutputByInput,
            raw_ggml: None,
        };

        let fused = FusedQkvProjectionWeight::new(&q_weight, &k_weight, &v_weight)
            .expect("fused qkv")
            .expect("available");
        assert_eq!(fused.input_width, 2);
        assert_eq!(fused.output_width, 4);
        assert!(fused.raw_ggml.is_none());
        assert_eq!(
            fused.values.expect("f32 fused payload"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn fused_qkv_projection_weight_concatenates_raw_ggml_payloads() {
        let q_weight = DenseProjectionWeight {
            input_width: 2,
            output_width: 2,
            values: vec![0.0; 4],
            layout: DenseProjectionLayout::OutputByInput,
            raw_ggml: Some(OwnedGgmlProjectionWeight {
                ggml_type: GGML_TYPE_F32,
                dims: vec![2, 2],
                bytes: vec![1, 2, 3, 4, 5, 6, 7, 8],
            }),
        };
        let k_weight = DenseProjectionWeight {
            input_width: 2,
            output_width: 1,
            values: vec![0.0; 2],
            layout: DenseProjectionLayout::OutputByInput,
            raw_ggml: Some(OwnedGgmlProjectionWeight {
                ggml_type: GGML_TYPE_F32,
                dims: vec![2, 1],
                bytes: vec![9, 10, 11, 12],
            }),
        };
        let v_weight = DenseProjectionWeight {
            input_width: 2,
            output_width: 1,
            values: vec![0.0; 2],
            layout: DenseProjectionLayout::OutputByInput,
            raw_ggml: Some(OwnedGgmlProjectionWeight {
                ggml_type: GGML_TYPE_F32,
                dims: vec![2, 1],
                bytes: vec![13, 14, 15, 16],
            }),
        };

        let fused = FusedQkvProjectionWeight::new(&q_weight, &k_weight, &v_weight)
            .expect("fused qkv")
            .expect("available");
        let raw = fused.raw_ggml.expect("raw fused payload");
        assert_eq!(raw.ggml_type, GGML_TYPE_F32);
        assert_eq!(raw.dims, vec![2, 4]);
        assert_eq!(
            raw.bytes,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
        assert!(fused.values.is_none());
    }

    #[test]
    fn qwen_batched_resident_kv_seed_packs_sequence_planes() {
        let runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut resident = allocate_zeroed_llm_resident_kv_arena(
            &runner,
            GgmlCpuGraphConfig::default().context_bytes,
            1,
            2,
            3,
            1,
            2,
            "test_qwen_seed_kv",
        )
        .expect("resident kv arena should allocate");
        let mut seq0 = Qwen3AsrLayerKvCacheState::new(3, 1, 2);
        seq0.write(0, &[1.0, 2.0], &[10.0, 20.0])
            .expect("seq0 row0");
        seq0.write(1, &[3.0, 4.0], &[30.0, 40.0])
            .expect("seq0 row1");
        let mut seq1 = Qwen3AsrLayerKvCacheState::new(3, 1, 2);
        seq1.write(0, &[5.0, 6.0], &[50.0, 60.0])
            .expect("seq1 row0");
        let seq0_layers = vec![seq0];
        let seq1_layers = vec![seq1];
        let seeds: [&[Qwen3AsrLayerKvCacheState]; 2] = [&seq0_layers, &seq1_layers];

        seed_qwen_batched_resident_kv_arena(&mut resident, 2, 3, 1, &[2, 1], &seeds)
            .expect("seed should upload");

        let layer = resident.layers[0];
        let key_values = resident
            .arena
            .read_f16_bits_slice(layer.key, 12)
            .expect("seeded key tensor should read back");
        let value_values = resident
            .arena
            .read_f16_bits_slice(layer.value, 12)
            .expect("seeded value tensor should read back");
        // Every expected value is exactly representable in f16, so the seeded
        // bits must equal the converted expectation bit-for-bit.
        assert_eq!(
            key_values,
            f32_slice_to_f16_bits(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            value_values,
            f32_slice_to_f16_bits(&[
                10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 50.0, 60.0, 0.0, 0.0, 0.0, 0.0
            ])
        );
    }

    #[test]
    fn qwen_batched_resident_kv_slot_seed_and_zero_touch_one_plane() {
        let runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut resident = allocate_zeroed_llm_resident_kv_arena(
            &runner,
            GgmlCpuGraphConfig::default().context_bytes,
            1,
            2,
            3,
            1,
            2,
            "test_qwen_slot_seed_kv",
        )
        .expect("resident kv arena should allocate");
        let mut seq1 = Qwen3AsrLayerKvCacheState::new(3, 1, 2);
        seq1.write(0, &[1.0, 2.0], &[10.0, 20.0])
            .expect("seq1 row0");
        seq1.write(1, &[3.0, 4.0], &[30.0, 40.0])
            .expect("seq1 row1");
        let seq1_layers = vec![seq1];

        seed_qwen_batched_resident_kv_slot(&mut resident, 2, 3, 1, 1, 2, &seq1_layers)
            .expect("slot seed should upload");

        let layer = resident.layers[0];
        let key_values = resident
            .arena
            .read_f16_bits_slice(layer.key, 12)
            .expect("seeded key tensor should read back");
        let value_values = resident
            .arena
            .read_f16_bits_slice(layer.value, 12)
            .expect("seeded value tensor should read back");
        assert_eq!(
            key_values,
            f32_slice_to_f16_bits(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0])
        );
        assert_eq!(
            value_values,
            f32_slice_to_f16_bits(&[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 0.0, 0.0
            ])
        );

        zero_qwen_batched_resident_kv_slot(&mut resident, 2, 3, 1, 1)
            .expect("slot zero should upload");
        let key_values = resident
            .arena
            .read_f16_bits_slice(layer.key, 12)
            .expect("zeroed key tensor should read back");
        let value_values = resident
            .arena
            .read_f16_bits_slice(layer.value, 12)
            .expect("zeroed value tensor should read back");
        assert_eq!(key_values, vec![0_u16; 12]);
        assert_eq!(value_values, vec![0_u16; 12]);
    }

    #[test]
    fn qwen_batched_resident_kv_seed_rejects_prefix_mismatch() {
        let runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())
            .expect("cpu graph runner should initialize");
        let mut resident = allocate_zeroed_llm_resident_kv_arena(
            &runner,
            GgmlCpuGraphConfig::default().context_bytes,
            1,
            2,
            3,
            1,
            1,
            "test_qwen_seed_kv",
        )
        .expect("resident kv arena should allocate");
        let mut seq0 = Qwen3AsrLayerKvCacheState::new(3, 1, 2);
        seq0.write(0, &[1.0, 2.0], &[10.0, 20.0])
            .expect("seq0 row0");
        let seq0_layers = vec![seq0];
        let seeds: [&[Qwen3AsrLayerKvCacheState]; 1] = [&seq0_layers];

        let error = seed_qwen_batched_resident_kv_arena(&mut resident, 2, 3, 1, &[2], &seeds)
            .expect_err("prefix mismatch must fail closed");
        assert!(matches!(
            error,
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "batched resident KV seed written prefix mismatch"
            }
        ));
    }

    #[test]
    fn qwen_batched_seed_written_prefix_lengths_reads_matching_layers() {
        let mut seq0_layer0 = Qwen3AsrLayerKvCacheState::new(4, 1, 2);
        seq0_layer0
            .write(0, &[1.0, 2.0], &[10.0, 20.0])
            .expect("seq0 layer0 row0");
        seq0_layer0
            .write(1, &[3.0, 4.0], &[30.0, 40.0])
            .expect("seq0 layer0 row1");
        let mut seq0_layer1 = Qwen3AsrLayerKvCacheState::new(4, 1, 2);
        seq0_layer1
            .write(0, &[5.0, 6.0], &[50.0, 60.0])
            .expect("seq0 layer1 row0");
        seq0_layer1
            .write(1, &[7.0, 8.0], &[70.0, 80.0])
            .expect("seq0 layer1 row1");
        let mut seq1_layer0 = Qwen3AsrLayerKvCacheState::new(4, 1, 2);
        seq1_layer0
            .write(0, &[9.0, 10.0], &[90.0, 100.0])
            .expect("seq1 layer0 row0");
        let mut seq1_layer1 = Qwen3AsrLayerKvCacheState::new(4, 1, 2);
        seq1_layer1
            .write(0, &[11.0, 12.0], &[110.0, 120.0])
            .expect("seq1 layer1 row0");
        let seq0_layers = vec![seq0_layer0, seq0_layer1];
        let seq1_layers = vec![seq1_layer0, seq1_layer1];
        let seeds: [&[Qwen3AsrLayerKvCacheState]; 2] = [&seq0_layers, &seq1_layers];

        let prefix_lengths =
            qwen_batched_seed_written_prefix_lengths(&seeds).expect("prefix lengths");
        assert_eq!(prefix_lengths, vec![2, 1]);
    }

    #[test]
    fn qwen_batched_seed_written_prefix_lengths_rejects_layer_mismatch() {
        let mut seq0_layer0 = Qwen3AsrLayerKvCacheState::new(4, 1, 2);
        seq0_layer0
            .write(0, &[1.0, 2.0], &[10.0, 20.0])
            .expect("seq0 layer0 row0");
        let mut seq0_layer1 = Qwen3AsrLayerKvCacheState::new(4, 1, 2);
        seq0_layer1
            .write(0, &[3.0, 4.0], &[30.0, 40.0])
            .expect("seq0 layer1 row0");
        seq0_layer1
            .write(1, &[5.0, 6.0], &[50.0, 60.0])
            .expect("seq0 layer1 row1");
        let seq0_layers = vec![seq0_layer0, seq0_layer1];
        let seeds: [&[Qwen3AsrLayerKvCacheState]; 1] = [&seq0_layers];

        let error = qwen_batched_seed_written_prefix_lengths(&seeds)
            .expect_err("layer prefix mismatch must fail closed");
        assert!(matches!(
            error,
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "batched resident KV seed layer prefix mismatch"
            }
        ));
    }

    #[test]
    fn segmented_rms_norm_rejects_mismatched_width() {
        let mut values = vec![1.0, 2.0, 3.0];
        let error = apply_segmented_rms_norm_with_weight(&mut values, &[1.0, 2.0], 1e-6)
            .expect_err("mismatch");
        assert!(matches!(
            error,
            Qwen3AsrLlmTransformerError::QkNormWidthMismatch { .. }
        ));
    }

    #[test]
    fn dense_projection_weight_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DenseProjectionWeight>();
        assert_send_sync::<Qwen3AsrLlmLayerAttentionProjection>();
    }

    #[test]
    #[ignore = "manual real-pack harness: set OPENASR_QWEN_PREFILL_REAL_PACK to a qwen .oasr model pack"]
    fn qwen_llm_prefill_real_pack_cpu_matches_serial() {
        with_forced_cpu_backend_for_test(|| {
            let report = run_qwen_real_pack_prefill_parity(Qwen3AsrPrefillParityMode::Whole);
            report.assert_close();
        });
    }

    #[test]
    #[ignore = "manual real-pack diagnostic: set OPENASR_QWEN_PREFILL_REAL_PACK and OPENASR_GGML_BACKEND=hip/vulkan/cuda/metal"]
    fn qwen_llm_prefill_real_pack_selected_backend_diagnostics() {
        let report = run_qwen_real_pack_prefill_parity(Qwen3AsrPrefillParityMode::Whole);
        report.assert_finite();
    }

    #[test]
    #[ignore = "manual real-pack GPU harness: set OPENASR_QWEN_PREFILL_REAL_PACK and OPENASR_GGML_BACKEND=hip/vulkan/cuda/metal"]
    fn qwen_llm_chunked_prefill_real_pack_selected_backend_matches_serial() {
        let report = run_qwen_real_pack_prefill_parity(Qwen3AsrPrefillParityMode::Chunked {
            chunk_size: qwen_prefill_chunk_size(),
        });
        report.assert_close();
    }

    #[test]
    #[ignore = "manual real-pack GPU harness: set OPENASR_QWEN_PREFILL_REAL_PACK and OPENASR_GGML_BACKEND=hip/vulkan/cuda/metal"]
    fn qwen_llm_policy_prefill_real_pack_selected_backend_matches_serial() {
        let report = run_qwen_real_pack_prefill_parity(Qwen3AsrPrefillParityMode::Policy);
        report.assert_close();
    }

    #[test]
    #[ignore = "manual real-pack harness: set OPENASR_QWEN_PREFILL_REAL_PACK to a qwen .oasr model pack"]
    fn qwen_llm_seed_only_reset_real_pack_rebuilds_reuse_graph() {
        let runtime_path = qwen_prefill_real_pack_path();
        let runtime_source =
            validate_ggml_runtime_source_path(&runtime_path).expect("valid qwen runtime source");
        let metadata = read_gguf_metadata_from_runtime_source(&runtime_source)
            .expect("read qwen runtime metadata");
        let metadata = parse_qwen3_execution_metadata(&metadata).expect("parse qwen metadata");
        let token_count = qwen_prefill_token_count(metadata).min(8);
        let hidden = deterministic_prefill_hidden(metadata.llm_d_model, token_count);
        let reader =
            GgufTensorDataReader::from_path(runtime_source.path()).expect("qwen tensor reader");
        let projections = load_qwen3_llm_attention_projections_from_reader(&reader, metadata)
            .expect("llm layers");
        let serial = run_qwen_serial_prefill(&projections, &runtime_path, metadata, &hidden);
        let seeds_two: [&[Qwen3AsrLayerKvCacheState]; 2] =
            [&serial.layer_kv_caches, &serial.layer_kv_caches];
        let seeds_one: [&[Qwen3AsrLayerKvCacheState]; 1] = [&serial.layer_kv_caches];

        let mut decoder =
            Qwen3AsrLlmWholeDecoderGraphExecutor::new(&projections, Some(&runtime_path))
                .expect("qwen decoder");
        decoder
            .reset_reused_batched_seeded(&seeds_two, 1_000_000.0, token_count)
            .expect("seed-only reset n_seq=2");
        let reuse = decoder.reuse.as_ref().expect("reuse graph after n_seq=2");
        assert_eq!(reuse.n_seq, 2);
        assert_eq!(reuse.max_positions, token_count);

        decoder
            .reset_reused_batched_seeded(&seeds_one, 1_000_000.0, token_count)
            .expect("seed-only reset n_seq=1");
        let reuse = decoder.reuse.as_ref().expect("reuse graph after n_seq=1");
        assert_eq!(reuse.n_seq, 1);
        assert_eq!(reuse.max_positions, token_count);
    }

    enum Qwen3AsrPrefillParityMode {
        Whole,
        Chunked { chunk_size: usize },
        Policy,
    }

    struct Qwen3AsrPrefillParityReport {
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
        token_count: usize,
        chunk_size: Option<usize>,
        hidden: VectorDiffStats,
        kv: VectorDiffStats,
    }

    impl Qwen3AsrPrefillParityReport {
        fn assert_finite(&self) {
            eprintln!(
                "qwen real-pack prefill parity backend={:?} token_count={} chunk_size={:?} hidden_max_abs={:.6} hidden_cosine={:.9} kv_max_abs={:.6} kv_cosine={:.9}",
                self.backend,
                self.token_count,
                self.chunk_size,
                self.hidden.max_abs,
                self.hidden.cosine(),
                self.kv.max_abs,
                self.kv.cosine()
            );
            assert!(
                self.hidden.is_finite() && self.kv.is_finite(),
                "qwen prefill parity produced non-finite stats"
            );
        }

        fn assert_close(&self) {
            self.assert_finite();
            assert!(
                self.hidden.max_abs <= 1.0e-3 && self.hidden.cosine() > 0.999,
                "qwen prefill hidden drift too far: max_abs={:.6} cosine={:.9}",
                self.hidden.max_abs,
                self.hidden.cosine()
            );
            assert!(
                self.kv.max_abs <= 1.0e-3 && self.kv.cosine() > 0.999,
                "qwen prefill KV drift too far: max_abs={:.6} cosine={:.9}",
                self.kv.max_abs,
                self.kv.cosine()
            );
        }
    }

    #[derive(Debug, Default, Clone, Copy)]
    struct VectorDiffStats {
        count: usize,
        max_abs: f32,
        dot: f64,
        lhs_norm: f64,
        rhs_norm: f64,
    }

    impl VectorDiffStats {
        fn push_pair(&mut self, lhs: f32, rhs: f32) {
            assert!(lhs.is_finite(), "lhs diff value is non-finite");
            assert!(rhs.is_finite(), "rhs diff value is non-finite");
            self.count += 1;
            self.max_abs = self.max_abs.max((lhs - rhs).abs());
            self.dot += lhs as f64 * rhs as f64;
            self.lhs_norm += lhs as f64 * lhs as f64;
            self.rhs_norm += rhs as f64 * rhs as f64;
        }

        fn extend_pairs(&mut self, lhs: &[f32], rhs: &[f32]) {
            assert_eq!(lhs.len(), rhs.len(), "diff vector length mismatch");
            for (&lhs, &rhs) in lhs.iter().zip(rhs) {
                self.push_pair(lhs, rhs);
            }
        }

        fn cosine(&self) -> f64 {
            if self.lhs_norm == 0.0 && self.rhs_norm == 0.0 {
                return 1.0;
            }
            self.dot / (self.lhs_norm.sqrt() * self.rhs_norm.sqrt())
        }

        fn is_finite(&self) -> bool {
            self.count > 0
                && self.max_abs.is_finite()
                && self.dot.is_finite()
                && self.lhs_norm.is_finite()
                && self.rhs_norm.is_finite()
                && self.cosine().is_finite()
        }
    }

    fn run_qwen_real_pack_prefill_parity(
        mode: Qwen3AsrPrefillParityMode,
    ) -> Qwen3AsrPrefillParityReport {
        let runtime_path = qwen_prefill_real_pack_path();
        let runtime_source =
            validate_ggml_runtime_source_path(&runtime_path).expect("valid qwen runtime source");
        let metadata = read_gguf_metadata_from_runtime_source(&runtime_source)
            .expect("read qwen runtime metadata");
        let metadata = parse_qwen3_execution_metadata(&metadata).expect("parse qwen metadata");
        let token_count = qwen_prefill_token_count(metadata);
        let hidden = deterministic_prefill_hidden(metadata.llm_d_model, token_count);
        let reader =
            GgufTensorDataReader::from_path(runtime_source.path()).expect("qwen tensor reader");
        let projections = load_qwen3_llm_attention_projections_from_reader(&reader, metadata)
            .expect("llm layers");

        let serial = run_qwen_serial_prefill(&projections, &runtime_path, metadata, &hidden);
        let mut selected_chunk_size = None;
        let prefill = match mode {
            Qwen3AsrPrefillParityMode::Whole => {
                run_qwen_whole_prefill(&projections, &runtime_path, token_count, &hidden)
            }
            Qwen3AsrPrefillParityMode::Chunked { chunk_size } => {
                selected_chunk_size = Some(chunk_size);
                run_qwen_chunked_prefill(
                    &projections,
                    &runtime_path,
                    metadata,
                    token_count,
                    chunk_size,
                    &hidden,
                )
            }
            Qwen3AsrPrefillParityMode::Policy => {
                let chunk_size =
                    qwen_policy_prefill_chunk_size(&projections, &runtime_path, token_count)
                        .expect("qwen policy should return a chunk size for native GQA");
                selected_chunk_size = Some(chunk_size);
                run_qwen_chunked_prefill(
                    &projections,
                    &runtime_path,
                    metadata,
                    token_count,
                    chunk_size,
                    &hidden,
                )
            }
        };

        let hidden_size = metadata.llm_d_model;
        let final_hidden_start = token_count
            .checked_sub(1)
            .and_then(|idx| idx.checked_mul(hidden_size))
            .expect("final hidden offset");
        let final_hidden_end = final_hidden_start
            .checked_add(hidden_size)
            .expect("final hidden end");
        let mut hidden_stats = VectorDiffStats::default();
        hidden_stats.extend_pairs(
            &prefill.hidden[final_hidden_start..final_hidden_end],
            &serial.final_hidden,
        );

        let kv_width = metadata
            .llm_kv_heads
            .checked_mul(metadata.llm_head_dim)
            .expect("kv width");
        let mut kv_stats = VectorDiffStats::default();
        for layer_index in 0..metadata.llm_layers {
            let (prefill_k, prefill_v) = &prefill.layer_kv[layer_index];
            for position in 0..token_count {
                let prefill_row_start = position.checked_mul(kv_width).expect("prefill row start");
                let prefill_row_end = prefill_row_start
                    .checked_add(kv_width)
                    .expect("prefill row end");
                let serial_key = serial_layer_kv_row(
                    &serial.layer_kv_caches[layer_index],
                    position,
                    kv_width,
                    KvRowKind::Key,
                );
                let serial_value = serial_layer_kv_row(
                    &serial.layer_kv_caches[layer_index],
                    position,
                    kv_width,
                    KvRowKind::Value,
                );
                kv_stats.extend_pairs(&prefill_k[prefill_row_start..prefill_row_end], &serial_key);
                kv_stats.extend_pairs(
                    &prefill_v[prefill_row_start..prefill_row_end],
                    &serial_value,
                );
            }
        }

        Qwen3AsrPrefillParityReport {
            backend: GgmlCpuGraphConfig::resolve_runtime_backend(),
            token_count,
            chunk_size: selected_chunk_size,
            hidden: hidden_stats,
            kv: kv_stats,
        }
    }

    struct Qwen3AsrSerialPrefillOutput {
        final_hidden: Vec<f32>,
        layer_kv_caches: Vec<Qwen3AsrLayerKvCacheState>,
    }

    fn run_qwen_serial_prefill(
        projections: &[Qwen3AsrLlmLayerAttentionProjection],
        runtime_path: &std::path::Path,
        metadata: Qwen3AsrExecutionMetadata,
        hidden: &[f32],
    ) -> Qwen3AsrSerialPrefillOutput {
        let token_count = hidden
            .len()
            .checked_div(metadata.llm_d_model)
            .expect("hidden token count");
        let mut decoder =
            Qwen3AsrLlmWholeDecoderGraphExecutor::new(projections, Some(runtime_path))
                .expect("serial qwen decoder");
        let mut layer_kv_caches = (0..metadata.llm_layers)
            .map(|_| {
                Qwen3AsrLayerKvCacheState::new(
                    token_count,
                    metadata.llm_kv_heads,
                    metadata.llm_head_dim,
                )
            })
            .collect::<Vec<_>>();
        let mut final_hidden = Vec::new();
        let kv_width = metadata
            .llm_kv_heads
            .checked_mul(metadata.llm_head_dim)
            .expect("kv width");
        for position in 0..token_count {
            let hidden_start = position
                .checked_mul(metadata.llm_d_model)
                .expect("hidden start");
            let hidden_end = hidden_start
                .checked_add(metadata.llm_d_model)
                .expect("hidden end");
            let step = decoder
                .run_step(
                    &hidden[hidden_start..hidden_end],
                    position,
                    &layer_kv_caches,
                    1_000_000.0,
                )
                .expect("serial qwen prefill step");
            for (layer_index, (key, value)) in step.layer_kv.iter().enumerate() {
                assert_eq!(key.len(), kv_width, "serial key width mismatch");
                assert_eq!(value.len(), kv_width, "serial value width mismatch");
                layer_kv_caches[layer_index]
                    .write(position, key, value)
                    .expect("serial KV write");
            }
            final_hidden = step.hidden;
        }
        Qwen3AsrSerialPrefillOutput {
            final_hidden,
            layer_kv_caches,
        }
    }

    fn run_qwen_whole_prefill(
        projections: &[Qwen3AsrLlmLayerAttentionProjection],
        runtime_path: &std::path::Path,
        token_count: usize,
        hidden: &[f32],
    ) -> Qwen3AsrLlmWholeStepOutput {
        let mut decoder =
            Qwen3AsrLlmWholeDecoderGraphExecutor::new(projections, Some(runtime_path))
                .expect("prefill qwen decoder");
        decoder
            .run_prefill(hidden, token_count, 1_000_000.0)
            .expect("qwen whole-prompt prefill")
    }

    fn run_qwen_chunked_prefill(
        projections: &[Qwen3AsrLlmLayerAttentionProjection],
        runtime_path: &std::path::Path,
        metadata: Qwen3AsrExecutionMetadata,
        token_count: usize,
        chunk_size: usize,
        hidden: &[f32],
    ) -> Qwen3AsrLlmWholeStepOutput {
        assert!(chunk_size > 0, "chunk size must be positive");
        let mut decoder =
            Qwen3AsrLlmWholeDecoderGraphExecutor::new(projections, Some(runtime_path))
                .expect("chunked prefill qwen decoder");
        let mut layer_kv_caches = (0..metadata.llm_layers)
            .map(|_| {
                Qwen3AsrLayerKvCacheState::new(
                    token_count,
                    metadata.llm_kv_heads,
                    metadata.llm_head_dim,
                )
            })
            .collect::<Vec<_>>();
        let kv_width = metadata
            .llm_kv_heads
            .checked_mul(metadata.llm_head_dim)
            .expect("kv width");
        let mut full_hidden = Vec::with_capacity(metadata.llm_d_model * token_count);
        let mut full_layer_kv = (0..metadata.llm_layers)
            .map(|_| {
                (
                    Vec::with_capacity(kv_width * token_count),
                    Vec::with_capacity(kv_width * token_count),
                )
            })
            .collect::<Vec<_>>();
        let mut position_offset = 0usize;
        while position_offset < token_count {
            let chunk_len = (token_count - position_offset).min(chunk_size);
            let hidden_start = position_offset
                .checked_mul(metadata.llm_d_model)
                .expect("chunk hidden start");
            let hidden_end = hidden_start
                .checked_add(chunk_len * metadata.llm_d_model)
                .expect("chunk hidden end");
            let total_token_count = position_offset
                .checked_add(chunk_len)
                .expect("chunk token span");
            let step = decoder
                .run_prefill_chunk(
                    &hidden[hidden_start..hidden_end],
                    chunk_len,
                    position_offset,
                    total_token_count,
                    &layer_kv_caches,
                    1_000_000.0,
                )
                .expect("chunked qwen prefill");
            full_hidden.extend_from_slice(&step.hidden);
            for (layer_index, (projected_k, projected_v)) in step.layer_kv.iter().enumerate() {
                full_layer_kv[layer_index].0.extend_from_slice(projected_k);
                full_layer_kv[layer_index].1.extend_from_slice(projected_v);
                for chunk_position in 0..chunk_len {
                    let row_start = chunk_position
                        .checked_mul(kv_width)
                        .expect("chunk row start");
                    let row_end = row_start.checked_add(kv_width).expect("chunk row end");
                    layer_kv_caches[layer_index]
                        .write(
                            position_offset + chunk_position,
                            &projected_k[row_start..row_end],
                            &projected_v[row_start..row_end],
                        )
                        .expect("chunked KV write");
                }
            }
            position_offset = total_token_count;
        }
        Qwen3AsrLlmWholeStepOutput {
            hidden: full_hidden,
            layer_kv: full_layer_kv,
            build_micros: 0,
            compute_micros: 0,
        }
    }

    fn qwen_policy_prefill_chunk_size(
        projections: &[Qwen3AsrLlmLayerAttentionProjection],
        runtime_path: &std::path::Path,
        token_count: usize,
    ) -> Option<usize> {
        let decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new(projections, Some(runtime_path))
            .expect("policy qwen decoder");
        decoder.safe_multi_query_prefill_chunk_size_for(token_count)
    }

    #[derive(Clone, Copy)]
    enum KvRowKind {
        Key,
        Value,
    }

    fn serial_layer_kv_row(
        cache: &Qwen3AsrLayerKvCacheState,
        position: usize,
        kv_width: usize,
        kind: KvRowKind,
    ) -> Vec<f32> {
        let history = cache.full_history_storage().expect("serial history");
        assert!(position < history.written_positions, "unwritten serial row");
        assert_eq!(history.kv_heads * history.head_dim, kv_width);
        let storage = match kind {
            KvRowKind::Key => history.keys,
            KvRowKind::Value => history.values,
        };
        let mut row = Vec::with_capacity(kv_width);
        for kv_head in 0..history.kv_heads {
            let row_start = kv_head
                .checked_mul(history.max_positions)
                .and_then(|base| base.checked_add(position))
                .and_then(|slot| slot.checked_mul(history.head_dim))
                .expect("serial row start");
            let row_end = row_start
                .checked_add(history.head_dim)
                .expect("serial row end");
            row.extend_from_slice(&storage[row_start..row_end]);
        }
        row
    }

    fn qwen_prefill_real_pack_path() -> PathBuf {
        std::env::var_os(QWEN_PREFILL_REAL_PACK_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                panic!("{QWEN_PREFILL_REAL_PACK_ENV} must point to a qwen .oasr model pack")
            })
    }

    fn qwen_prefill_token_count(metadata: Qwen3AsrExecutionMetadata) -> usize {
        let requested = std::env::var(QWEN_PREFILL_TOKENS_ENV)
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|&value| value > 1)
            .unwrap_or(8);
        requested.min(metadata.llm_max_positions).max(2)
    }

    fn qwen_prefill_chunk_size() -> usize {
        std::env::var(QWEN_PREFILL_CHUNK_TOKENS_ENV)
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(QWEN3_LLM_GPU_SAFE_PREFILL_QUERY_TOKENS)
    }

    fn deterministic_prefill_hidden(d_model: usize, token_count: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(d_model * token_count);
        for token in 0..token_count {
            for dim in 0..d_model {
                let mixed = (token * 17 + dim * 31 + token * dim * 3) % 97;
                values.push((mixed as f32 - 48.0) / 97.0);
            }
        }
        values
    }
}
