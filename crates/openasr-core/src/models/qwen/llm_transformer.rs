#![cfg_attr(test, allow(dead_code))]
// Production dead-code remains linted. Test builds retain the old all-layer
// host materializer only as a numerical parity oracle, so some oracle helpers
// are intentionally exercised by a subset of feature/test combinations.

//! Hands-off: single-responsibility ggml graph transcription, guarded by
//! golden/parity tests. Do not split this module for "tidiness" -- the tensor
//! wiring is validated as a whole and refactoring here risks silent numeric
//! drift.

use std::fmt;

use thiserror::Error;

#[cfg(test)]
use crate::GgmlRuntimeSource;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
    GgmlFlashAttentionPrecision, GgmlKvElementType, GgmlLoadedTensor, GgmlNativeGqaCapability,
    GgmlRopeExtParams, GgmlStaticTensor, GgmlStaticTensorArena, GgufTensorDataReadError,
    GgufTensorDataReader, env_toggle_with_raw, env_var_truthy,
};

use super::decoder_contract::{QwenDecoderContract, QwenDecoderContractGeometry};
use super::graph_config::qwen_decoder_graph_config;
use super::kv_cache::{Qwen3AsrKvCacheCapacity, Qwen3AsrLayerKvCacheState};
use crate::models::device_greedy_token::{
    first_max_argmax_reverse_indices, first_max_token_id_from_reversed_argmax,
};

use super::logits_head::{Qwen3AsrLlmFusedLogitsHeadSpec, Qwen3AsrLlmLogitsHead};
use super::lora::{QwenLayerLoraSlots, QwenLoraAdapter, new_qwen_lora_slot};
use super::runtime_contract::{Qwen3AsrExecutionMetadata, qwen3_asr_decoder_contract};
#[cfg(test)]
use super::tensor_names::llm_layer_tensor_names;
use crate::models::mapped_token_embedding::{
    MappedTokenEmbeddingDeviceSpec, MappedTokenEmbeddingTable,
};
use crate::models::prepared_runtime_cache::{
    PreparedRuntimeQuoteBuilder, PreparedRuntimeQuoteContext,
};
use crate::models::system_memory_owner::SystemMemoryOwnerError;
use crate::models::tensor_binding::{TensorBindingDescriptor, TensorBindingDescriptorRequirement};
use crate::nn::decoder::{
    LlmDecoderStackConfig, LlmDecoderStackInputs, LlmKvCachePolicy, LlmKvCacheSpec,
    LlmLayerWeights, LlmQkvWeights, LlmResidentKvArena, LlmReusableDecodeGraph,
    allocate_zeroed_llm_resident_kv_arena, build_fixed_kv_attention_mask_bits,
    build_fixed_kv_attention_mask_bits_for_query_rows,
    build_fixed_kv_attention_mask_bits_for_sequences, compose_llm_decoder_layer_stack,
    resolve_production_llm_kv_cache_policy_from_env, reusable_decode_graph_supported_for_runner,
};
use crate::nn::half::f32_slice_to_f16_bits;

const DEFAULT_RMS_NORM_EPSILON: f32 = 1e-6;
// The whole-step decoder already enforces this cgraph node limit. Size the
// no-alloc metadata contexts from that same topology contract instead of the
// former 768 MiB byte guess: ggml reserves the full context address range even
// though tensor payloads live in separate backend buffers.
const QWEN3_LLM_WHOLE_DECODE_GRAPH_SIZE: usize = 1usize << 12;
const QWEN3_LLM_RESIDENT_PREFILL_MAX_QUERY_TOKENS: usize = 256;
// Worst-case arena handles per layer: norms/biases (5), split QKV (3), four
// non-bindable projections, and seven two-tensor LoRA slots. Current Qwen3 ASR
// packs use materially fewer; this bound also covers Qwen2-shaped adapters.
const QWEN3_LLM_STATIC_TENSORS_PER_LAYER_MAX: usize = 26;
const QWEN3_LLM_STATIC_TENSOR_FIXED_MARGIN: usize = 16;

fn qwen_llm_graph_context_bytes() -> usize {
    GgmlCpuGraphConfig::metadata_context_bytes(QWEN3_LLM_WHOLE_DECODE_GRAPH_SIZE)
}

fn qwen_llm_weight_arena_context_bytes(layer_count: usize) -> Result<usize, GgmlCpuGraphError> {
    let tensor_count = layer_count
        .checked_mul(QWEN3_LLM_STATIC_TENSORS_PER_LAYER_MAX)
        .and_then(|count| count.checked_add(QWEN3_LLM_STATIC_TENSOR_FIXED_MARGIN))
        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "qwen decoder static tensor metadata count overflow",
        })?;
    let capacity =
        tensor_count
            .checked_next_power_of_two()
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "qwen decoder static tensor metadata capacity overflow",
            })?;
    Ok(GgmlCpuGraphConfig::metadata_context_bytes(capacity))
}

// Correctness escape hatch for backend kernels with divergent native GQA
// behavior; keep it unless every backend's GQA path is verified.
const QWEN3_LLM_NATIVE_GQA_ENV: &str = "OPENASR_QWEN_LLM_NATIVE_GQA";
const QWEN3_LLM_PREFILL_ALLOCATION_PROFILE_ENV: &str = "OPENASR_QWEN_PREFILL_ALLOCATION_PROFILE";
const QWEN3_LLM_CPU_SAFE_PREFILL_QUERY_TOKENS: usize = 8;
/// Conservative single-query width kept for backends that have not been
/// validated for multi-query host-cache prefill (legacy default). Discrete
/// GPU (CUDA/HIP/Vulkan) no longer uses this once the non-flash wide path is
/// selected — see `qwen_llm_safe_gpu_prefill_query_tokens_for_backend`.
const QWEN3_LLM_GPU_SAFE_PREFILL_QUERY_TOKENS: usize = 1;
/// Flash VEC kernel is trusted at `n_query <= 2` on every backend (see
/// `fattn.cu` `Q->ne[1] <= 2`). Above this, discrete-GPU flash MMA/TILE must
/// not be used for long KV spans.
const QWEN3_LLM_FLASH_SAFE_PREFILL_QUERY_TOKENS: usize = 2;
const QWEN3_LLM_FLASH_SAFE_PREFILL_MAX_KV_TOKENS: usize = 32;
const QWEN3_LLM_DISCRETE_GPU_SHORT_PREFILL_QUERY_TOKENS: usize = 8;
/// Prefill chunk for discrete-GPU backends (CUDA/HIP/Vulkan) past the 32-token
/// flash window. Chunks in this range bypass the buggy flash MMA/TILE kernel
/// (the graph swaps to the unfused `llm_naive_masked_attention` path when
/// `n_query > 2` and `n_kv > 32`), so correctness no longer bounds the width —
/// but performance still does on HIP: ggml's HIP `mul_mat` only takes the fast
/// mmvq vector kernels for `n_query <= 8` and beyond that switches to MMQ,
/// which is pathologically slow on RDNA4 Windows (measured on gfx1200: 8-token
/// chunks decode at ~3 ms/token while 16/32/64-token chunks blow up to seconds
/// per chunk). Keep the shared discrete-GPU host-cache chunk at the mmvq
/// ceiling so HIP stays fast; CUDA bulk resident prefill uses a separate
/// wider path (`run_prefill_into_reused_batched`) and is not limited by this.
const QWEN3_LLM_DISCRETE_GPU_NONFLASH_PREFILL_QUERY_TOKENS: usize = 8;

#[cfg(test)]
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

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Qwen3AsrLlmDecodeAttentionHistory<'a> {
    pub key_rows: &'a [f32],
    pub value_rows: &'a [f32],
    pub token_count: usize,
    pub position: usize,
    pub rope_theta: f32,
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct DenseProjectionWeight {
    input_width: usize,
    output_width: usize,
    values: Vec<f32>,
    layout: DenseProjectionLayout,
    raw_ggml: Option<OwnedGgmlProjectionWeight>,
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct OwnedGgmlProjectionWeight {
    ggml_type: i32,
    dims: Vec<usize>,
    bytes: Vec<u8>,
}

#[cfg(test)]
impl OwnedGgmlProjectionWeight {
    fn add_retained_system_memory(
        &self,
        bytes: &mut crate::models::system_memory_owner::SystemMemoryCapacity,
        label: &str,
    ) -> Result<(), String> {
        bytes.add_vec(&self.dims, label)?;
        bytes.add_vec(&self.bytes, label)
    }
}

#[cfg(test)]
impl DenseProjectionWeight {
    fn add_retained_system_memory(
        &self,
        bytes: &mut crate::models::system_memory_owner::SystemMemoryCapacity,
        label: &str,
    ) -> Result<(), String> {
        bytes.add_vec(&self.values, label)?;
        if let Some(raw) = &self.raw_ggml {
            raw.add_retained_system_memory(bytes, label)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct FusedQkvProjectionWeight {
    input_width: usize,
    output_width: usize,
    raw_ggml: Option<OwnedGgmlProjectionWeight>,
    values: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenseProjectionLayout {
    #[allow(dead_code)]
    InputByOutput,
    OutputByInput,
}

#[derive(Debug, Error)]
pub(crate) enum Qwen3AsrLlmTransformerError {
    #[cfg(test)]
    #[error("qwen3-asr llm transformer tensor read failed: {reason}")]
    TensorReadFailed { reason: String },
    #[error("qwen3-asr llm transformer tensor '{tensor_name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        tensor_name: String,
        shape: String,
        reason: String,
    },
    #[cfg(test)]
    #[error(
        "qwen3-asr llm transformer hidden state has invalid shape: got {got}, expected hidden_size={expected}"
    )]
    InvalidHiddenStateShape { got: usize, expected: usize },
    #[cfg(test)]
    #[error("qwen3-asr llm transformer tensor '{tensor_name}' contains non-finite values")]
    NonFiniteTensorValues { tensor_name: String },
    #[cfg(test)]
    #[error("qwen3-asr llm transformer projection values contain non-finite numbers")]
    NonFiniteProjectionValues,
    #[cfg(test)]
    #[error(
        "qwen3-asr llm transformer projection values are unavailable for tensor '{tensor_name}'"
    )]
    ProjectionValuesUnavailable { tensor_name: String },
    #[cfg(test)]
    #[error("qwen3-asr llm transformer tensor '{tensor_name}' projection overflowed allocation")]
    AllocationOverflow { tensor_name: String },
    #[error(
        "qwen3-asr llm transformer q/k norm width mismatch: vector_width={vector_width}, norm_width={norm_width}"
    )]
    #[cfg(test)]
    QkNormWidthMismatch {
        vector_width: usize,
        norm_width: usize,
    },
    #[cfg(test)]
    #[error("qwen3-asr llm transformer attention core has incompatible q/k widths")]
    IncompatibleQkWidths,
    #[cfg(test)]
    #[error("qwen3-asr llm transformer attention core produced non-finite score")]
    NonFiniteAttentionScore,
    #[cfg(test)]
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
    #[cfg(test)]
    FfnProjectionWidthMismatch { gate_width: usize, up_width: usize },
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) enum Qwen3AsrLlmLayerAttentionProjection {
    Generic(Qwen3AsrLlmLayerAttentionProjectionGeneric),
}

#[cfg(test)]
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

/// Metadata-only declaration of a Qwen-shaped whole decoder.
///
/// The plan deliberately owns no tensor payload. It is safe to retain in a
/// host prepared-runtime cache: resident construction reuses the exact
/// [`GgufTensorDataReader`] that validated the plan and streams one tensor (or
/// the three mmap-backed Q/K/V byte ranges) directly into the already-declared
/// arena before moving to the next layer.
#[derive(Debug, Clone)]
pub(crate) struct QwenWholeDecoderPlan {
    layers: Vec<QwenWholeDecoderLayerPlan>,
}

#[derive(Debug, Clone)]
struct QwenWholeDecoderLayerPlan {
    d_model: usize,
    head_dim: usize,
    attn_norm: VectorWeightPlan,
    q: ProjectionWeightPlan,
    k: ProjectionWeightPlan,
    v: ProjectionWeightPlan,
    q_bias: Option<VectorWeightPlan>,
    k_bias: Option<VectorWeightPlan>,
    v_bias: Option<VectorWeightPlan>,
    output: ProjectionWeightPlan,
    q_norm: Option<VectorWeightPlan>,
    k_norm: Option<VectorWeightPlan>,
    ffn_norm: VectorWeightPlan,
    gate: ProjectionWeightPlan,
    up: ProjectionWeightPlan,
    down: ProjectionWeightPlan,
}

#[derive(Debug, Clone)]
struct VectorWeightPlan {
    tensor_name: String,
    len: usize,
    ggml_type: i32,
    size_bytes: u64,
    offset_bytes: u64,
}

#[derive(Debug, Clone)]
struct ProjectionWeightPlan {
    tensor_name: String,
    input_width: usize,
    output_width: usize,
    storage_dims: [usize; 2],
    ggml_type: i32,
    size_bytes: u64,
    offset_bytes: u64,
    layout: DenseProjectionLayout,
}

/// The resident arena contains exactly one QKV representation per layer.
/// Per-projection LoRA requires split leaves; otherwise compatible native
/// rows are concatenated directly into one fused tensor during upload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QkvStorageMode {
    Fused { ggml_type: i32 },
    Split,
}

#[cfg(test)]
impl Qwen3AsrLlmLayerAttentionProjection {
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        let Self::Generic(inner) = self;
        for name in [
            &inner.attn_norm_name,
            &inner.attn_q_name,
            &inner.attn_k_name,
            &inner.attn_v_name,
            &inner.attn_output_name,
            &inner.ffn_gate_name,
            &inner.ffn_up_name,
            &inner.ffn_down_name,
        ] {
            bytes.add_string(name, "qwen llm tensor name")?;
        }
        for values in [
            &inner.attn_norm_weight,
            &inner.ffn_norm_weight,
            &inner.q_norm_weight,
            &inner.k_norm_weight,
            &inner.q_bias,
            &inner.k_bias,
            &inner.v_bias,
        ] {
            bytes.add_vec(values, "qwen llm vector weight")?;
        }
        for weight in [
            &inner.q_weight,
            &inner.k_weight,
            &inner.v_weight,
            &inner.attn_output_weight,
            &inner.ffn_gate_weight,
            &inner.ffn_up_weight,
            &inner.ffn_down_weight,
        ] {
            weight.add_retained_system_memory(&mut bytes, "qwen llm projection")?;
        }
        Ok(bytes.finish())
    }

    pub(crate) fn run_attention_core_for_decode_boundary(
        &self,
        hidden: &[f32],
    ) -> Result<Qwen3AsrLlmAttentionCoreOutput, Qwen3AsrLlmTransformerError> {
        match self {
            Self::Generic(inner) => inner.run_attention_core_for_decode_boundary(hidden),
        }
    }
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
fn head_dim_from_norm(norm_weight: &[f32]) -> Result<usize, Qwen3AsrLlmTransformerError> {
    if norm_weight.is_empty() || !norm_weight.len().is_multiple_of(2) {
        return Err(Qwen3AsrLlmTransformerError::IncompatibleQkWidths);
    }
    Ok(norm_weight.len())
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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
/// A discrete GPU is re-enabled only by a typed, validated Exact CUDA/Vulkan
/// route. A synthetic runtime self-check was rejected because a probe shape can
/// miss the real decoder op that diverges. `OPENASR_QWEN_LLM_NATIVE_GQA` is an
/// emergency opt-out only; it cannot promote an unsupported provider.
fn qwen_llm_native_gqa_default_for_backend(backend: GgmlCpuGraphBackend) -> bool {
    !matches!(backend, GgmlCpuGraphBackend::Gpu)
}

fn qwen_llm_native_gqa_enabled(raw: Option<&str>, capability: GgmlNativeGqaCapability) -> bool {
    capability.is_validated() && env_toggle_with_raw(None, raw, true)
}

fn qwen_llm_resolve_use_native_gqa_for_capability(capability: GgmlNativeGqaCapability) -> bool {
    qwen_llm_native_gqa_enabled(
        std::env::var(QWEN3_LLM_NATIVE_GQA_ENV).ok().as_deref(),
        capability,
    )
}

pub(crate) fn qwen_llm_effective_native_gqa_capability(
    capability: GgmlNativeGqaCapability,
) -> GgmlNativeGqaCapability {
    if qwen_llm_resolve_use_native_gqa_for_capability(capability) {
        GgmlNativeGqaCapability::Validated
    } else {
        GgmlNativeGqaCapability::Unsupported
    }
}

/// Conservative resolver for shared Qwen-shaped constructors. Only Qwen3-ASR
/// production requests pass the request's typed capability explicitly; other
/// families and the forced aligner retain the established CPU/Metal-on,
/// discrete-GPU-off behavior.
fn qwen_llm_resolve_use_native_gqa(backend: GgmlCpuGraphBackend) -> bool {
    let capability = if qwen_llm_native_gqa_default_for_backend(backend) {
        GgmlNativeGqaCapability::Validated
    } else {
        GgmlNativeGqaCapability::Unsupported
    };
    qwen_llm_resolve_use_native_gqa_for_capability(capability)
}

/// Production KV-cache policy for every Qwen-shaped whole-decoder constructor
/// (qwen / mimo / firered2 / moss / serve-batch).
///
/// Applies the shared phase-1 Q8 rules: CPU/Metal + native-GQA + flash geometry
/// selects `Q8_0`; discrete GPU and incomplete geometry stay on Default.
/// `OPENASR_QWEN_KV_CACHE_F32` opts back to host-F32 / resident-F16 for golden
/// pins. Decode always uses flash; CPU/Metal prefill also always uses flash, so
/// the construction-time flash flag is `true` for policy selection (discrete
/// GPU is already rejected by backend support).
pub(crate) fn resolve_qwen_family_production_kv_cache_policy(
    backend: GgmlCpuGraphBackend,
    head_dim: usize,
) -> LlmKvCachePolicy {
    let use_native_gqa = qwen_llm_resolve_use_native_gqa(backend);
    resolve_production_llm_kv_cache_policy_from_env(backend, head_dim, use_native_gqa, true)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum QwenQkvExecutionMode {
    FusedArena,
    SplitLoaded,
}

/// Resident weight handles for one decode layer, allocated into a shared arena.
#[derive(Clone, Copy)]
enum QwenQkvWeightHandles {
    Fused(GgmlStaticTensor),
    FusedQvSplitK {
        qv: GgmlStaticTensor,
        k: LlmWeightHandle,
    },
    Split {
        q: LlmWeightHandle,
        k: LlmWeightHandle,
        v: LlmWeightHandle,
    },
}

struct Qwen3AsrLlmLayerWeightHandles {
    attn_norm_weight: GgmlStaticTensor,
    qkv: QwenQkvWeightHandles,
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
    flash_attention_precision: GgmlFlashAttentionPrecision,
    kv_cache_spec: LlmKvCacheSpec,
    materialize_kv_outputs: bool,
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
        // Multi-sequence callers are validated before graph composition. Never
        // turn native GQA on here: doing so bypasses the typed provider gate and
        // corrupts output on HIP/RDNA4.
        use_native_gqa,
        use_flash_attention,
        flash_attention_precision,
        kv_cache_spec,
        materialize_kv_outputs,
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
        qkv: match layer.qkv {
            QwenQkvWeightHandles::Fused(weight) => LlmQkvWeights::Fused(weight.as_graph_tensor()),
            QwenQkvWeightHandles::FusedQvSplitK { qv, k } => LlmQkvWeights::FusedQvSplitK {
                qv: qv.as_graph_tensor(),
                k: k.as_graph_tensor(arena),
            },
            QwenQkvWeightHandles::Split { q, k, v } => LlmQkvWeights::Split {
                q: q.as_graph_tensor(arena),
                k: k.as_graph_tensor(arena),
                v: v.as_graph_tensor(arena),
            },
        },
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
    #[cfg(test)]
    pub layer_kv: Vec<(Vec<f32>, Vec<f32>)>,
    #[cfg(test)]
    pub build_micros: u128,
    #[cfg(test)]
    pub compute_micros: u128,
}

impl QwenWholeDecoderLayerPlan {
    fn dims(&self) -> Result<Qwen3AsrLlmDecodeDims, GgmlCpuGraphError> {
        if self.q.input_width != self.d_model
            || self.k.input_width != self.d_model
            || self.v.input_width != self.d_model
            || self.gate.input_width != self.d_model
            || self.up.input_width != self.d_model
            || self.k.output_width != self.v.output_width
            || !self.q.output_width.is_multiple_of(self.head_dim)
            || !self.k.output_width.is_multiple_of(self.head_dim)
        {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "qwen whole-decoder plan has inconsistent layer geometry",
            });
        }
        let q_heads = self.q.output_width / self.head_dim;
        let kv_heads = self.k.output_width / self.head_dim;
        if q_heads == 0 || kv_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "qwen whole-decoder plan has invalid q/kv head ratio",
            });
        }
        Ok(Qwen3AsrLlmDecodeDims {
            d_model: self.d_model,
            q_width: self.q.output_width,
            k_width: self.k.output_width,
            v_width: self.v.output_width,
            head_dim: self.head_dim,
            q_heads,
            kv_heads,
        })
    }

    fn qkv_storage_mode(&self, force_split: bool) -> QkvStorageMode {
        if !force_split
            && self.q.layout == DenseProjectionLayout::OutputByInput
            && self.k.layout == DenseProjectionLayout::OutputByInput
            && self.v.layout == DenseProjectionLayout::OutputByInput
            && self.q.ggml_type == self.k.ggml_type
            && self.q.ggml_type == self.v.ggml_type
            && self.q.storage_dims[0] == self.k.storage_dims[0]
            && self.q.storage_dims[0] == self.v.storage_dims[0]
        {
            QkvStorageMode::Fused {
                ggml_type: self.q.ggml_type,
            }
        } else {
            QkvStorageMode::Split
        }
    }
}

fn new_projection_tensor_from_plan(
    arena: &GgmlStaticTensorArena,
    plan: &ProjectionWeightPlan,
    tensor_name: &'static str,
) -> Result<GgmlStaticTensor, GgmlCpuGraphError> {
    match plan.layout {
        DenseProjectionLayout::OutputByInput => arena.new_matmul_weight_2d_typed(
            plan.storage_dims[0],
            plan.storage_dims[1],
            plan.ggml_type,
            tensor_name,
        ),
        DenseProjectionLayout::InputByOutput => {
            arena.new_tensor_2d_f32(plan.input_width, plan.output_width, tensor_name)
        }
    }
}

fn bind_or_arena_llm_plan(
    arena: &GgmlStaticTensorArena,
    loaded: Option<&crate::ggml_runtime::GgmlLoadedWeightContext>,
    plan: &ProjectionWeightPlan,
    tensor_name: &'static str,
) -> Result<LlmWeightHandle, GgmlCpuGraphError> {
    if plan.layout == DenseProjectionLayout::OutputByInput {
        return loaded
            .and_then(|context| context.tensor(&plan.tensor_name))
            .map(LlmWeightHandle::Loaded)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "decode native 2D weight could not be bound zero-copy from the planned runtime source",
            });
    }
    Ok(LlmWeightHandle::Arena(new_projection_tensor_from_plan(
        arena,
        plan,
        tensor_name,
    )?))
}

fn allocate_decode_layer_from_plan(
    arena: &mut GgmlStaticTensorArena,
    loaded: Option<&crate::ggml_runtime::GgmlLoadedWeightContext>,
    plan: &QwenWholeDecoderLayerPlan,
    force_split_qkv: bool,
    qkv_execution_mode: QwenQkvExecutionMode,
) -> Result<(Qwen3AsrLlmLayerWeightHandles, Qwen3AsrLlmDecodeDims), GgmlCpuGraphError> {
    let dims = plan.dims()?;
    let has_qk_norm = plan.q_norm.is_some();
    if has_qk_norm != plan.k_norm.is_some() {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "qwen whole-decoder plan has asymmetric q/k norm",
        });
    }
    let has_qkv_bias = plan.q_bias.is_some();
    if has_qkv_bias != plan.k_bias.is_some() || has_qkv_bias != plan.v_bias.is_some() {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "qwen whole-decoder plan has asymmetric q/k/v bias",
        });
    }
    let attn_norm = arena.new_tensor_2d_f32(plan.d_model, 1, "qwen_llm_decode_attn_norm_weight")?;
    let q_norm = has_qk_norm
        .then(|| arena.new_tensor_2d_f32(plan.head_dim, 1, "qwen_llm_decode_q_norm_weight"))
        .transpose()?;
    let k_norm = has_qk_norm
        .then(|| arena.new_tensor_2d_f32(plan.head_dim, 1, "qwen_llm_decode_k_norm_weight"))
        .transpose()?;
    let q_bias = has_qkv_bias
        .then(|| arena.new_tensor_2d_f32(plan.q.output_width, 1, "qwen_llm_decode_q_bias"))
        .transpose()?;
    let k_bias = has_qkv_bias
        .then(|| arena.new_tensor_2d_f32(plan.k.output_width, 1, "qwen_llm_decode_k_bias"))
        .transpose()?;
    let v_bias = has_qkv_bias
        .then(|| arena.new_tensor_2d_f32(plan.v.output_width, 1, "qwen_llm_decode_v_bias"))
        .transpose()?;
    let ffn_norm = arena.new_tensor_2d_f32(plan.d_model, 1, "qwen_llm_decode_ffn_norm_weight")?;
    let qkv_storage_mode = match qkv_execution_mode {
        QwenQkvExecutionMode::FusedArena => plan.qkv_storage_mode(force_split_qkv),
        QwenQkvExecutionMode::SplitLoaded => QkvStorageMode::Split,
    };
    let qkv = match qkv_storage_mode {
        QkvStorageMode::Fused { ggml_type } => {
            let output_width = plan
                .q
                .output_width
                .checked_add(plan.k.output_width)
                .and_then(|width| width.checked_add(plan.v.output_width))
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "fused qkv projection width overflow",
                })?;
            QwenQkvWeightHandles::Fused(arena.new_matmul_weight_2d_typed(
                plan.d_model,
                output_width,
                ggml_type,
                "qwen_llm_decode_qkv_weight",
            )?)
        }
        QkvStorageMode::Split => {
            let bind = |arena: &GgmlStaticTensorArena,
                        plan: &ProjectionWeightPlan,
                        tensor_name: &'static str|
             -> Result<LlmWeightHandle, GgmlCpuGraphError> {
                match qkv_execution_mode {
                    QwenQkvExecutionMode::FusedArena => Ok(LlmWeightHandle::Arena(
                        new_projection_tensor_from_plan(arena, plan, tensor_name)?,
                    )),
                    QwenQkvExecutionMode::SplitLoaded => {
                        bind_or_arena_llm_plan(arena, loaded, plan, tensor_name)
                    }
                }
            };
            let q = bind(arena, &plan.q, "qwen_llm_decode_q_weight")?;
            let k = bind(arena, &plan.k, "qwen_llm_decode_k_weight")?;
            let v = bind(arena, &plan.v, "qwen_llm_decode_v_weight")?;
            if qkv_execution_mode == QwenQkvExecutionMode::SplitLoaded
                && let (LlmWeightHandle::Loaded(q), LlmWeightHandle::Loaded(v)) = (q, v)
                && let Some(qv) =
                    arena.try_fuse_adjacent_loaded_tensors_2d(q, v, "qwen_llm_decode_qv_weight")?
            {
                QwenQkvWeightHandles::FusedQvSplitK { qv, k }
            } else {
                QwenQkvWeightHandles::Split { q, k, v }
            }
        }
    };
    Ok((
        Qwen3AsrLlmLayerWeightHandles {
            attn_norm_weight: attn_norm,
            qkv,
            q_bias,
            k_bias,
            v_bias,
            output_weight: bind_or_arena_llm_plan(
                arena,
                loaded,
                &plan.output,
                "qwen_llm_decode_output_weight",
            )?,
            q_norm_weight: q_norm,
            k_norm_weight: k_norm,
            ffn_norm_weight: ffn_norm,
            gate_weight: bind_or_arena_llm_plan(
                arena,
                loaded,
                &plan.gate,
                "qwen_llm_decode_gate_weight",
            )?,
            up_weight: bind_or_arena_llm_plan(
                arena,
                loaded,
                &plan.up,
                "qwen_llm_decode_up_weight",
            )?,
            down_weight: bind_or_arena_llm_plan(
                arena,
                loaded,
                &plan.down,
                "qwen_llm_decode_down_weight",
            )?,
            lora: QwenLayerLoraSlots::default(),
        },
        dims,
    ))
}

fn upload_vector_from_plan(
    reader: &GgufTensorDataReader,
    arena: &mut GgmlStaticTensorArena,
    handle: GgmlStaticTensor,
    plan: &VectorWeightPlan,
    upload_name: &'static str,
) -> Result<usize, GgmlCpuGraphError> {
    let expected_shape = [plan.len as u64];
    let values = reader
        .host_tensor_f32_copy_dequantized_by_name(&plan.tensor_name, &expected_shape)
        .map_err(map_tensor_read_error_to_graph)?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "qwen vector weight contains non-finite values",
        });
    }
    // The plan records the pack type so a same-name/same-shape source drift
    // cannot silently swap representations between validation and upload.
    let actual_type = required_tensor_metadata(reader, &plan.tensor_name)
        .map_err(map_transformer_error_to_graph)?
        .ggml_type;
    if actual_type != plan.ggml_type {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "qwen vector weight type changed after planning",
        });
    }
    let staging_bytes = values.capacity().saturating_mul(std::mem::size_of::<f32>());
    arena.set_f32_slice(handle, &values, upload_name)?;
    Ok(staging_bytes)
}

fn checked_projection_payload<'a>(
    reader: &'a GgufTensorDataReader,
    plan: &ProjectionWeightPlan,
) -> Result<crate::ggml_runtime::GgufWeightTensorPayload<'a>, GgmlCpuGraphError> {
    let payload = reader
        .weight_tensor_payload_by_name(&plan.tensor_name)
        .map_err(map_tensor_read_error_to_graph)?;
    if payload.dims.as_slice() != plan.storage_dims
        || payload.element_type.ggml_type() != plan.ggml_type
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "qwen projection payload disagrees with metadata plan",
        });
    }
    Ok(payload)
}

fn upload_projection_from_plan(
    reader: &GgufTensorDataReader,
    arena: &mut GgmlStaticTensorArena,
    handle: GgmlStaticTensor,
    plan: &ProjectionWeightPlan,
    upload_name: &'static str,
) -> Result<usize, GgmlCpuGraphError> {
    match plan.layout {
        DenseProjectionLayout::OutputByInput => {
            let payload = checked_projection_payload(reader, plan)?;
            arena.set_bytes_slice(handle, payload.bytes, upload_name)?;
            Ok(0)
        }
        DenseProjectionLayout::InputByOutput => {
            let expected_shape = [plan.storage_dims[0] as u64, plan.storage_dims[1] as u64];
            let values = reader
                .host_tensor_f32_copy_dequantized_by_name(&plan.tensor_name, &expected_shape)
                .map_err(map_tensor_read_error_to_graph)?;
            let transposed = projection_values_for_ggml(
                plan.input_width,
                plan.output_width,
                &values,
                plan.layout,
            )?;
            let staging_bytes = values
                .capacity()
                .saturating_add(transposed.capacity())
                .saturating_mul(std::mem::size_of::<f32>());
            arena.set_f32_slice(handle, &transposed, upload_name)?;
            Ok(staging_bytes)
        }
    }
}

fn upload_fused_qkv_from_plan(
    reader: &GgufTensorDataReader,
    arena: &mut GgmlStaticTensorArena,
    handle: GgmlStaticTensor,
    plan: &QwenWholeDecoderLayerPlan,
) -> Result<(), GgmlCpuGraphError> {
    let mut offset = 0usize;
    for projection in [&plan.q, &plan.k, &plan.v] {
        let payload = checked_projection_payload(reader, projection)?;
        arena.set_bytes_slice_with_offset(
            handle,
            offset,
            payload.bytes,
            "qwen_llm_decode_qkv_weight",
        )?;
        offset = offset.checked_add(payload.bytes.len()).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "fused qkv upload byte offset overflow",
            },
        )?;
    }
    Ok(())
}

fn upload_decode_layer_from_plan(
    reader: &GgufTensorDataReader,
    arena: &mut GgmlStaticTensorArena,
    handles: &Qwen3AsrLlmLayerWeightHandles,
    plan: &QwenWholeDecoderLayerPlan,
) -> Result<usize, GgmlCpuGraphError> {
    let mut peak_staging_bytes = 0usize;
    peak_staging_bytes = peak_staging_bytes.max(upload_vector_from_plan(
        reader,
        arena,
        handles.attn_norm_weight,
        &plan.attn_norm,
        "qwen_llm_decode_attn_norm_weight",
    )?);
    for (handle, vector, name) in [
        (
            handles.q_norm_weight,
            plan.q_norm.as_ref(),
            "qwen_llm_decode_q_norm_weight",
        ),
        (
            handles.k_norm_weight,
            plan.k_norm.as_ref(),
            "qwen_llm_decode_k_norm_weight",
        ),
        (
            handles.q_bias,
            plan.q_bias.as_ref(),
            "qwen_llm_decode_q_bias",
        ),
        (
            handles.k_bias,
            plan.k_bias.as_ref(),
            "qwen_llm_decode_k_bias",
        ),
        (
            handles.v_bias,
            plan.v_bias.as_ref(),
            "qwen_llm_decode_v_bias",
        ),
    ] {
        match (handle, vector) {
            (Some(handle), Some(vector)) => {
                peak_staging_bytes = peak_staging_bytes.max(upload_vector_from_plan(
                    reader, arena, handle, vector, name,
                )?);
            }
            (None, None) => {}
            _ => {
                return Err(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "qwen vector resident handle disagrees with metadata plan",
                });
            }
        }
    }
    peak_staging_bytes = peak_staging_bytes.max(upload_vector_from_plan(
        reader,
        arena,
        handles.ffn_norm_weight,
        &plan.ffn_norm,
        "qwen_llm_decode_ffn_norm_weight",
    )?);
    match handles.qkv {
        QwenQkvWeightHandles::Fused(handle) => {
            upload_fused_qkv_from_plan(reader, arena, handle, plan)?;
        }
        QwenQkvWeightHandles::FusedQvSplitK { k, .. } => {
            if let Some(handle) = k.arena_handle() {
                peak_staging_bytes = peak_staging_bytes.max(upload_projection_from_plan(
                    reader,
                    arena,
                    handle,
                    &plan.k,
                    "qwen_llm_decode_k_weight",
                )?);
            }
        }
        QwenQkvWeightHandles::Split { q, k, v } => {
            for (handle, projection, name) in [
                (q.arena_handle(), &plan.q, "qwen_llm_decode_q_weight"),
                (k.arena_handle(), &plan.k, "qwen_llm_decode_k_weight"),
                (v.arena_handle(), &plan.v, "qwen_llm_decode_v_weight"),
            ] {
                if let Some(handle) = handle {
                    peak_staging_bytes = peak_staging_bytes.max(upload_projection_from_plan(
                        reader, arena, handle, projection, name,
                    )?);
                }
            }
        }
    }
    for (handle, projection, name) in [
        (
            handles.output_weight.arena_handle(),
            &plan.output,
            "qwen_llm_decode_output_weight",
        ),
        (
            handles.gate_weight.arena_handle(),
            &plan.gate,
            "qwen_llm_decode_gate_weight",
        ),
        (
            handles.up_weight.arena_handle(),
            &plan.up,
            "qwen_llm_decode_up_weight",
        ),
        (
            handles.down_weight.arena_handle(),
            &plan.down,
            "qwen_llm_decode_down_weight",
        ),
    ] {
        if let Some(handle) = handle {
            peak_staging_bytes = peak_staging_bytes.max(upload_projection_from_plan(
                reader, arena, handle, projection, name,
            )?);
        }
    }
    Ok(peak_staging_bytes)
}

fn map_tensor_read_error_to_graph(error: GgufTensorDataReadError) -> GgmlCpuGraphError {
    let _ = error;
    GgmlCpuGraphError::UnsupportedInputs {
        reason: "qwen whole-decoder planned tensor materialization failed",
    }
}

fn map_transformer_error_to_graph(_error: Qwen3AsrLlmTransformerError) -> GgmlCpuGraphError {
    GgmlCpuGraphError::UnsupportedInputs {
        reason: "qwen whole-decoder tensor metadata changed after planning",
    }
}

/// Validate and ALLOCATE (but do not upload) one decode layer's weight tensors
/// into `arena`. All layers must be allocated before ANY upload, because the
/// first upload freezes the arena's backend buffer (no further new_tensor). The
/// returned FusedQkvProjectionWeight is carried to the upload phase.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
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
    force_split_qkv: bool,
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
    // QKV bias no longer forces the split path: the fused `[q|k|v]` matmul
    // produces the same per-projection columns, and the bias is added on the
    // (ggml-contiguous) per-projection view/copy downstream in `nn::decoder`
    // -- identical arithmetic to the split path, so Qwen2-shaped packs (bias,
    // e.g. funasr/mimo) get the single-matmul decode speedup too.
    let allow_fused_qkv = true;

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
    let fused_qkv_weight = if allow_fused_qkv && !force_split_qkv {
        FusedQkvProjectionWeight::new(q_weight, k_weight, v_weight)?
    } else {
        None
    };
    let qkv = match fused_qkv_weight.as_ref() {
        Some(weight) => QwenQkvWeightHandles::Fused(new_fused_qkv_tensor_in_arena(
            arena,
            weight,
            "qwen_llm_decode_qkv_weight",
        )?),
        None => QwenQkvWeightHandles::Split {
            q: LlmWeightHandle::Arena(new_projection_tensor_in_arena(
                arena,
                q_weight,
                "qwen_llm_decode_q_weight",
            )?),
            k: LlmWeightHandle::Arena(new_projection_tensor_in_arena(
                arena,
                k_weight,
                "qwen_llm_decode_k_weight",
            )?),
            v: LlmWeightHandle::Arena(new_projection_tensor_in_arena(
                arena,
                v_weight,
                "qwen_llm_decode_v_weight",
            )?),
        },
    };
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
            qkv,
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
#[cfg(test)]
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

/// Allocate LoRA A/B slots for one layer into the arena. Payloads remain owned
/// by the adapter and are uploaded by reference only after every arena tensor
/// has been declared; construction never clones all layer payloads into a
/// second pending-upload collection.
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
) -> Result<QwenLayerLoraSlots, GgmlCpuGraphError> {
    let Some(adapter) = adapter else {
        return Ok(QwenLayerLoraSlots::default());
    };
    let mut slots = QwenLayerLoraSlots::default();
    // Allocate one LoRA slot for `target_name`, pushing the upload payload.
    let maybe_slot =
        |target_name: &str| -> Result<Option<super::lora::QwenLoraSlot>, GgmlCpuGraphError> {
            let Some(target) = adapter.target(target_name) else {
                return Ok(None);
            };
            let slot = new_qwen_lora_slot(arena, target, "qwen_lora_a", "qwen_lora_b")?;
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

#[allow(clippy::too_many_arguments)]
fn upload_layer_lora_slots(
    arena: &mut GgmlStaticTensorArena,
    adapter: Option<&QwenLoraAdapter>,
    slots: &QwenLayerLoraSlots,
    attn_q_name: &str,
    attn_k_name: &str,
    attn_v_name: &str,
    attn_output_name: &str,
    ffn_gate_name: &str,
    ffn_up_name: &str,
    ffn_down_name: &str,
) -> Result<(), GgmlCpuGraphError> {
    let Some(adapter) = adapter else {
        return Ok(());
    };
    for (slot, target_name) in [
        (slots.attn_q, attn_q_name),
        (slots.attn_k, attn_k_name),
        (slots.attn_v, attn_v_name),
        (slots.attn_output, attn_output_name),
        (slots.ffn_gate, ffn_gate_name),
        (slots.ffn_up, ffn_up_name),
        (slots.ffn_down, ffn_down_name),
    ] {
        let Some(slot) = slot else { continue };
        let target = adapter
            .target(target_name)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "allocated qwen LoRA slot lost its adapter target",
            })?;
        arena.set_f32_slice(slot.a, &target.a_values, "qwen_lora_a")?;
        arena.set_f32_slice(slot.b_scaled, &target.b_scaled_values, "qwen_lora_b")?;
    }
    Ok(())
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
            None => LlmWeightHandle::Arena(arena.new_tensor_2d_typed(
                spec.output_weight_dims[0],
                spec.output_weight_dims[1],
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
#[cfg(test)]
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
    match (&handles.qkv, fused_qkv_weight) {
        (QwenQkvWeightHandles::Fused(tensor), Some(weight)) => {
            upload_fused_qkv_weight_to_arena(arena, *tensor, weight, "qwen_llm_decode_qkv_weight")?;
        }
        (QwenQkvWeightHandles::Split { q, k, v }, None) => {
            for (handle, weight, name) in [
                (q.arena_handle(), q_weight, "qwen_llm_decode_q_weight"),
                (k.arena_handle(), k_weight, "qwen_llm_decode_k_weight"),
                (v.arena_handle(), v_weight, "qwen_llm_decode_v_weight"),
            ] {
                let handle = handle.ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "fixture split QKV handle must be arena-backed",
                })?;
                upload_projection_weight_to_arena(arena, handle, weight, name)?;
            }
        }
        _ => {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "QKV resident handle/payload modes disagree",
            });
        }
    }
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

/// Prepare-time compile input for every Qwen-shaped whole-decoder adapter.
///
/// The owned typed plan is built once at prepare (host-neutral metadata only).
/// This request is the sole backend materialize seam: FunASR-Nano and
/// MOSS-Transcribe-Diarize (and every other Qwen-shaped consumer) pass the same
/// prepared plan through here rather than re-deriving layer geometry at first
/// decode checkout.
pub(crate) struct QwenPreparedDecoderGraphCompileRequest<'a> {
    pub plan: &'a QwenWholeDecoderPlan,
    pub preflight: &'a crate::ggml_runtime::GgufRuntimeSourcePreflight,
    pub rms_norm_epsilon: f32,
    pub fused_logits_head: Option<Qwen3AsrLlmFusedLogitsHeadSpec<'a>>,
    pub token_embedding: Option<MappedTokenEmbeddingDeviceSpec<'a>>,
    pub backend: GgmlCpuGraphBackend,
}

/// Compile a prepared [`QwenWholeDecoderPlan`] into a monomorphic whole-decoder
/// graph executor.
///
/// **Structural adoption (not performance-promoted):** every Qwen-shaped
/// production caller builds through this seam so assembly cannot fork. That is
/// a code-structure choice. Cold/warm/RSS/VRAM non-regression vs the pre-seam
/// path has **not** been proven on this branch; do not treat this as a
/// Completed Prepared Graph Plan promotion under ARCHITECTURE-DEEPENING-PLAN.
/// Callers must supply a plan that already passed prepare-time validation;
/// this function never walks family metadata or tensor-name tables.
pub(crate) fn compile_qwen_whole_decoder_graph_from_prepared_plan(
    request: QwenPreparedDecoderGraphCompileRequest<'_>,
) -> Result<Qwen3AsrLlmWholeDecoderGraphExecutor, GgmlCpuGraphError> {
    let graph_config = qwen_decoder_graph_config(request.backend);
    compile_qwen_whole_decoder_graph_from_prepared_plan_with_config(request, graph_config)
}

/// Typed Exact-provider variant for Qwen-shaped families that have completed
/// their own CUDA/Vulkan native-GQA parity gate. The capability is resolved on
/// the submitting thread and carried into the owner; worker threads never
/// infer a provider from a coarse `Gpu` backend or a backend name.
pub(crate) fn compile_qwen_whole_decoder_graph_from_prepared_plan_with_native_gqa(
    request: QwenPreparedDecoderGraphCompileRequest<'_>,
    native_gqa: GgmlNativeGqaCapability,
) -> Result<Qwen3AsrLlmWholeDecoderGraphExecutor, GgmlCpuGraphError> {
    let graph_config = qwen_decoder_graph_config(request.backend);
    compile_qwen_whole_decoder_graph_from_prepared_plan_with_config_and_native_gqa(
        request,
        graph_config,
        native_gqa,
    )
}

/// Compile a prepared Qwen-shaped decoder with a family-owned, already
/// resolved graph configuration. This is intentionally narrower than a
/// second materializer: tensor assembly remains centralized here, while a
/// Hybrid family such as MOSS may freeze its per-stage scheduler decision
/// without the generic Qwen policy re-applying placement defaults.
pub(crate) fn compile_qwen_whole_decoder_graph_from_prepared_plan_with_config(
    request: QwenPreparedDecoderGraphCompileRequest<'_>,
    graph_config: GgmlCpuGraphConfig,
) -> Result<Qwen3AsrLlmWholeDecoderGraphExecutor, GgmlCpuGraphError> {
    compile_qwen_whole_decoder_graph_from_prepared_plan_with_config_and_native_gqa_impl(
        request,
        graph_config,
        None,
    )
}

/// Family-owned graph-config variant for a typed native-GQA capability that
/// was frozen on the submitting thread. This keeps placement/thread policy in
/// the family while centralizing Qwen tensor assembly and provider gating.
pub(crate) fn compile_qwen_whole_decoder_graph_from_prepared_plan_with_config_and_native_gqa(
    request: QwenPreparedDecoderGraphCompileRequest<'_>,
    graph_config: GgmlCpuGraphConfig,
    native_gqa: GgmlNativeGqaCapability,
) -> Result<Qwen3AsrLlmWholeDecoderGraphExecutor, GgmlCpuGraphError> {
    compile_qwen_whole_decoder_graph_from_prepared_plan_with_config_and_native_gqa_impl(
        request,
        graph_config,
        Some(native_gqa),
    )
}

fn compile_qwen_whole_decoder_graph_from_prepared_plan_with_config_and_native_gqa_impl(
    request: QwenPreparedDecoderGraphCompileRequest<'_>,
    graph_config: GgmlCpuGraphConfig,
    native_gqa: Option<GgmlNativeGqaCapability>,
) -> Result<Qwen3AsrLlmWholeDecoderGraphExecutor, GgmlCpuGraphError> {
    if graph_config.backend != request.backend {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "prepared decoder graph config backend does not match compile request",
        });
    }
    Qwen3AsrLlmWholeDecoderGraphExecutor::new_from_plan_with_adapter_and_native_gqa(
        request.plan,
        request.rms_norm_epsilon,
        request.fused_logits_head,
        request.token_embedding,
        None,
        request.preflight,
        graph_config,
        native_gqa,
        QwenQkvExecutionMode::FusedArena,
    )
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
    device_token_embedding: Option<QwenDeviceTokenEmbedding>,
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
    kv_cache_spec: LlmKvCacheSpec,
    flash_attention_precision: GgmlFlashAttentionPrecision,
    #[cfg(test)]
    materialization_peak_staging_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct QwenDeviceTokenEmbedding {
    tensor: GgmlLoadedTensor,
    vocab_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QwenReusableDecodeInputKind {
    Hidden,
    TokenIds,
}

#[derive(Clone, Copy)]
enum QwenReusableDecodeInput<'a> {
    Hidden(&'a [f32]),
    TokenIds(&'a [u32]),
}

enum QwenResidentPrefillInput<'a> {
    Hidden(&'a [f32]),
    TokenIds {
        token_ids: &'a [u32],
        audio_rows: &'a [f32],
        audio_positions_in_chunk: &'a [usize],
    },
}

impl QwenReusableDecodeInput<'_> {
    fn kind(&self) -> QwenReusableDecodeInputKind {
        match self {
            Self::Hidden(_) => QwenReusableDecodeInputKind::Hidden,
            Self::TokenIds(_) => QwenReusableDecodeInputKind::TokenIds,
        }
    }
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
    #[allow(dead_code)] // Used by aggregate candidate memory quotes.
    pub(crate) fn quoted_retained_system_memory_bytes(layer_count: usize) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_usize(
            layer_count
                .checked_mul(std::mem::size_of::<Qwen3AsrLlmLayerWeightHandles>())
                .ok_or_else(|| "qwen decoder layer-handle quote overflowed".to_string())?,
            "qwen resident decoder layer handles",
        )?;
        Ok(bytes.finish())
    }

    #[allow(dead_code)] // Reconciled by aggregate candidate owners.
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_vec(&self.layers, "qwen resident decoder layer handles")?;
        Ok(bytes.finish())
    }

    pub(crate) fn graph_lane(&self) -> (GgmlCpuGraphBackend, bool) {
        (self.runner.backend_kind(), self.runner.uses_scheduler())
    }

    fn require_native_gqa_for_multi_sequence(&self, n_seq: usize) -> Result<(), GgmlCpuGraphError> {
        if n_seq > 1 && !self.use_native_gqa {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder multi-sequence execution requires a validated native GQA lane",
            });
        }
        Ok(())
    }

    pub(crate) fn loaded_weight_binding_identity(
        &self,
    ) -> Option<crate::ggml_runtime::GgmlLoadedWeightBindingIdentity> {
        self.loaded
            .as_ref()
            .map(|loaded| self.runner.loaded_weight_binding_identity(loaded))
    }

    pub(crate) fn uses_native_gqa(&self) -> bool {
        self.use_native_gqa
    }

    #[allow(dead_code)] // Conservative shared-family entry point; Qwen uses the typed variant.
    pub(crate) fn new_from_plan_with_preflight_and_lora(
        plan: &QwenWholeDecoderPlan,
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        fused_logits_head: Option<Qwen3AsrLlmFusedLogitsHeadSpec<'_>>,
        token_embedding: Option<MappedTokenEmbeddingDeviceSpec<'_>>,
        adapter: Option<&QwenLoraAdapter>,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, GgmlCpuGraphError> {
        Self::new_from_plan_with_adapter(
            plan,
            DEFAULT_RMS_NORM_EPSILON,
            fused_logits_head,
            token_embedding,
            adapter,
            preflight,
            qwen_decoder_graph_config(backend),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_from_plan_with_preflight_and_lora_for_qwen(
        plan: &QwenWholeDecoderPlan,
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        fused_logits_head: Option<Qwen3AsrLlmFusedLogitsHeadSpec<'_>>,
        token_embedding: Option<MappedTokenEmbeddingDeviceSpec<'_>>,
        adapter: Option<&QwenLoraAdapter>,
        backend: GgmlCpuGraphBackend,
        native_gqa: GgmlNativeGqaCapability,
        qkv_execution_mode: QwenQkvExecutionMode,
    ) -> Result<Self, GgmlCpuGraphError> {
        Self::new_from_plan_with_adapter_and_native_gqa(
            plan,
            DEFAULT_RMS_NORM_EPSILON,
            fused_logits_head,
            token_embedding,
            adapter,
            preflight,
            qwen_decoder_graph_config(backend),
            Some(native_gqa),
            qkv_execution_mode,
        )
    }

    pub(crate) fn new_from_plan_with_preflight(
        plan: &QwenWholeDecoderPlan,
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, GgmlCpuGraphError> {
        compile_qwen_whole_decoder_graph_from_prepared_plan(
            QwenPreparedDecoderGraphCompileRequest {
                plan,
                preflight,
                rms_norm_epsilon: DEFAULT_RMS_NORM_EPSILON,
                fused_logits_head: None,
                token_embedding: None,
                backend,
            },
        )
    }

    pub(crate) fn new_from_plan_with_preflight_and_token_embedding(
        plan: &QwenWholeDecoderPlan,
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        token_embedding: MappedTokenEmbeddingDeviceSpec<'_>,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, GgmlCpuGraphError> {
        compile_qwen_whole_decoder_graph_from_prepared_plan(
            QwenPreparedDecoderGraphCompileRequest {
                plan,
                preflight,
                rms_norm_epsilon: DEFAULT_RMS_NORM_EPSILON,
                fused_logits_head: None,
                token_embedding: Some(token_embedding),
                backend,
            },
        )
    }

    /// Private materializer used by the prepare-time compile seam and the
    /// LoRA-bearing production path. External adapters must not call this;
    /// they go through [`compile_qwen_whole_decoder_graph_from_prepared_plan`].
    #[allow(clippy::too_many_arguments)]
    fn new_from_plan_with_adapter(
        plan: &QwenWholeDecoderPlan,
        rms_norm_epsilon: f32,
        fused_logits_head: Option<Qwen3AsrLlmFusedLogitsHeadSpec<'_>>,
        token_embedding: Option<MappedTokenEmbeddingDeviceSpec<'_>>,
        adapter: Option<&QwenLoraAdapter>,
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        graph_config: GgmlCpuGraphConfig,
    ) -> Result<Self, GgmlCpuGraphError> {
        Self::new_from_plan_with_adapter_and_native_gqa(
            plan,
            rms_norm_epsilon,
            fused_logits_head,
            token_embedding,
            adapter,
            preflight,
            graph_config,
            None,
            QwenQkvExecutionMode::FusedArena,
        )
    }

    pub(crate) fn new_from_plan_with_preflight_and_token_embedding_for_qwen(
        plan: &QwenWholeDecoderPlan,
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        token_embedding: MappedTokenEmbeddingDeviceSpec<'_>,
        backend: GgmlCpuGraphBackend,
        native_gqa: GgmlNativeGqaCapability,
        qkv_execution_mode: QwenQkvExecutionMode,
    ) -> Result<Self, GgmlCpuGraphError> {
        Self::new_from_plan_with_adapter_and_native_gqa(
            plan,
            DEFAULT_RMS_NORM_EPSILON,
            None,
            Some(token_embedding),
            None,
            preflight,
            qwen_decoder_graph_config(backend),
            Some(native_gqa),
            qkv_execution_mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_from_plan_with_adapter_and_native_gqa(
        plan: &QwenWholeDecoderPlan,
        rms_norm_epsilon: f32,
        fused_logits_head: Option<Qwen3AsrLlmFusedLogitsHeadSpec<'_>>,
        token_embedding: Option<MappedTokenEmbeddingDeviceSpec<'_>>,
        adapter: Option<&QwenLoraAdapter>,
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        graph_config: GgmlCpuGraphConfig,
        native_gqa: Option<GgmlNativeGqaCapability>,
        qkv_execution_mode: QwenQkvExecutionMode,
    ) -> Result<Self, GgmlCpuGraphError> {
        if plan.layers.is_empty() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder plan requires at least one layer",
            });
        }
        if !rms_norm_epsilon.is_finite() || rms_norm_epsilon <= 0.0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder rms norm epsilon must be finite and positive",
            });
        }
        let reader = crate::ggml_runtime::build_runtime_tensor_reader_from_preflight(preflight)
            .map_err(|error| GgmlCpuGraphError::LoadedWeightContextFailed {
                reason: error.to_string(),
            })?;
        plan.validate_materialization_reader(&reader)?;
        let mut config = graph_config;
        config.graph_size = QWEN3_LLM_WHOLE_DECODE_GRAPH_SIZE;
        config.context_bytes = qwen_llm_graph_context_bytes();
        let use_native_gqa = native_gqa.map_or_else(
            || qwen_llm_resolve_use_native_gqa(config.backend),
            qwen_llm_resolve_use_native_gqa_for_capability,
        );
        let runner = GgmlCpuGraphRunner::new(config)?;
        let loaded = Some(runner.load_gguf_weight_context_from_preflight(preflight)?);
        let mut arena = runner
            .start_static_tensor_arena(qwen_llm_weight_arena_context_bytes(plan.layers.len())?)?;
        let mut layers = Vec::with_capacity(plan.layers.len());
        let mut dims = None;

        // Declaration pass. No payload is read and no backend buffer is
        // allocated until every base, LoRA and fused-head tensor exists.
        for layer_plan in &plan.layers {
            let layer_lora = allocate_layer_lora_slots(
                &arena,
                adapter,
                &layer_plan.q.tensor_name,
                &layer_plan.k.tensor_name,
                &layer_plan.v.tensor_name,
                &layer_plan.output.tensor_name,
                &layer_plan.gate.tensor_name,
                &layer_plan.up.tensor_name,
                &layer_plan.down.tensor_name,
            )?;
            let force_split_qkv = layer_lora.attn_q.is_some()
                || layer_lora.attn_k.is_some()
                || layer_lora.attn_v.is_some();
            let (mut handles, layer_dims) = allocate_decode_layer_from_plan(
                &mut arena,
                loaded.as_ref(),
                layer_plan,
                force_split_qkv,
                qkv_execution_mode,
            )?;
            handles.lora = layer_lora;
            match dims {
                None => dims = Some(layer_dims),
                Some(existing) if existing != layer_dims => {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder plan layers have inconsistent dimensions",
                    });
                }
                Some(_) => {}
            }
            layers.push(handles);
        }
        let dims = dims.expect("non-empty whole-decoder plan sets dimensions");
        let fused_logits_head = fused_logits_head
            .map(|spec| {
                let handles =
                    allocate_fused_logits_head_tensors(&mut arena, loaded.as_ref(), dims, &spec)?;
                upload_fused_logits_head_weights(&mut arena, &handles, &spec)?;
                Ok::<_, GgmlCpuGraphError>(handles)
            })
            .transpose()?;
        let device_token_embedding = token_embedding
            .map(|spec| {
                if spec.d_model != dims.d_model {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "device token embedding width does not match decoder d_model",
                    });
                }
                let metadata = reader.tensor_index().get(spec.tensor_name).ok_or_else(|| {
                    GgmlCpuGraphError::LoadedWeightContextFailed {
                        reason: format!(
                            "device token embedding tensor '{}' is missing",
                            spec.tensor_name
                        ),
                    }
                })?;
                if metadata.dims != [spec.d_model as u64, spec.vocab_size as u64] {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "device token embedding must use canonical [d_model, vocab] layout",
                    });
                }
                let tensor = loaded
                    .as_ref()
                    .and_then(|context| context.tensor(spec.tensor_name))
                    .ok_or_else(|| GgmlCpuGraphError::LoadedWeightContextFailed {
                        reason: format!(
                            "device token embedding tensor '{}' was not bound",
                            spec.tensor_name
                        ),
                    })?;
                Ok(QwenDeviceTokenEmbedding {
                    tensor,
                    vocab_size: spec.vocab_size,
                })
            })
            .transpose()?;

        // Materialization pass. Direct projections are borrowed from the mmap
        // and uploaded without a host copy. Fallback projections and vectors
        // are loaded one at a time and dropped before advancing to the next
        // tensor/layer. LoRA values remain owned by `adapter` and are likewise
        // uploaded by reference, never cloned into a pending list.
        let mut materialization_peak_staging_bytes = 0usize;
        for (layer_plan, handles) in plan.layers.iter().zip(&layers) {
            materialization_peak_staging_bytes = materialization_peak_staging_bytes.max(
                upload_decode_layer_from_plan(&reader, &mut arena, handles, layer_plan)?,
            );
            upload_layer_lora_slots(
                &mut arena,
                adapter,
                &handles.lora,
                &layer_plan.q.tensor_name,
                &layer_plan.k.tensor_name,
                &layer_plan.v.tensor_name,
                &layer_plan.output.tensor_name,
                &layer_plan.gate.tensor_name,
                &layer_plan.up.tensor_name,
                &layer_plan.down.tensor_name,
            )?;
        }
        let mut executor = Self {
            reuse: None,
            device_token_embedding,
            loaded,
            runner,
            arena,
            layers,
            fused_logits_head,
            dims,
            use_native_gqa,
            rms_norm_epsilon,
            kv_cache_spec: LlmKvCacheSpec::DEFAULT,
            flash_attention_precision: GgmlFlashAttentionPrecision::Default,
            #[cfg(test)]
            materialization_peak_staging_bytes,
        };
        let policy = resolve_qwen_family_production_kv_cache_policy(
            executor.runner.backend_kind(),
            executor.dims.head_dim,
        );
        executor.set_kv_cache_policy(policy)?;
        Ok(executor)
    }

    #[cfg(test)]
    pub(crate) fn new(
        projections: &[Qwen3AsrLlmLayerAttentionProjection],
        runtime_source: Option<&GgmlRuntimeSource>,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, GgmlCpuGraphError> {
        Self::new_with_rms_norm_epsilon_and_fused_logits_head(
            projections,
            runtime_source,
            DEFAULT_RMS_NORM_EPSILON,
            None,
            backend,
        )
    }

    /// Like [`new`] but with an optional LoRA adapter injected into the decoder
    /// graph.  Uses [`DEFAULT_RMS_NORM_EPSILON`].
    ///
    /// The fused logits head stays correct alongside an active adapter: LoRA
    /// slots only ever attach to the per-layer projections
    /// (`allocate_layer_lora_slots` -- q/k/v/o and the FFN gate/up/down), never
    /// to the output-norm/lm-head stage, and the host
    /// [`super::logits_head::Qwen3AsrLlmLogitsHead`] the spec is derived from
    /// is likewise loaded from the base pack only -- both selection paths see
    /// the identical head weights whether or not an adapter is active.
    #[cfg(test)]
    pub(crate) fn new_with_lora(
        projections: &[Qwen3AsrLlmLayerAttentionProjection],
        runtime_source: Option<&GgmlRuntimeSource>,
        fused_logits_head: Option<Qwen3AsrLlmFusedLogitsHeadSpec<'_>>,
        adapter: Option<&QwenLoraAdapter>,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, GgmlCpuGraphError> {
        Self::new_with_adapter(
            projections,
            runtime_source,
            DEFAULT_RMS_NORM_EPSILON,
            fused_logits_head,
            adapter,
            backend,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_with_rms_norm_epsilon_and_fused_logits_head(
        projections: &[Qwen3AsrLlmLayerAttentionProjection],
        runtime_source: Option<&GgmlRuntimeSource>,
        rms_norm_epsilon: f32,
        fused_logits_head: Option<Qwen3AsrLlmFusedLogitsHeadSpec<'_>>,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, GgmlCpuGraphError> {
        Self::new_with_adapter(
            projections,
            runtime_source,
            rms_norm_epsilon,
            fused_logits_head,
            None,
            backend,
        )
    }

    /// Construct with an optional LoRA adapter.  The adapter's arena tensors
    /// are allocated in the SAME arena as the layer weights (so the entire
    /// graph lives in one backend buffer) and uploaded in the same pass.
    ///
    /// `backend` is this family's already-resolved backend (see
    /// `GgmlAsrExecutionViewRequest::resolved_runtime`'s doc comment) -- the
    /// caller's explicit value, never re-derived here. `runtime_source` is
    /// the same already-open, already-validated source the caller's tensor
    /// reader was built from -- the zero-copy resident-weight bind below
    /// shares that one open mapping instead of a second `File::open` of the
    /// pack, so identity and weight bytes cannot come from different file
    /// generations.
    #[cfg(test)]
    pub(crate) fn new_with_adapter(
        projections: &[Qwen3AsrLlmLayerAttentionProjection],
        runtime_source: Option<&GgmlRuntimeSource>,
        rms_norm_epsilon: f32,
        fused_logits_head: Option<Qwen3AsrLlmFusedLogitsHeadSpec<'_>>,
        adapter: Option<&QwenLoraAdapter>,
        backend: GgmlCpuGraphBackend,
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
        // This single resident graph is reused across the whole pack lifetime
        // for both prompt-prefill chunks and single-token autoregressive
        // decode steps (the executor-owned resident decoder actor pool);
        // `n_threads` is fixed once here at construction, so one tier has to
        // be picked for the whole executor rather than per call. This
        // test-only numerical oracle uses the same `Decoder` tier that
        // production family runtimes inherit through the shared prepared-plan
        // compile seam. See `qwen_decoder_graph_config` for the rationale.
        let mut config = qwen_decoder_graph_config(backend);
        config.graph_size = QWEN3_LLM_WHOLE_DECODE_GRAPH_SIZE;
        config.context_bytes = qwen_llm_graph_context_bytes();
        let use_native_gqa = qwen_llm_resolve_use_native_gqa(config.backend);
        let runner = GgmlCpuGraphRunner::new(config)?;
        // goals 7+8: bind output/gate/up/down zero-copy from the mmap'd pack
        // (native q8/f16) instead of copying them into the arena. The context is
        // owned by this executor (drops after `reuse`, before `runner`).
        let loaded = runtime_source.and_then(|source| runner.load_gguf_weight_context(source).ok());
        let mut arena = runner
            .start_static_tensor_arena(qwen_llm_weight_arena_context_bytes(projections.len())?)?;
        let mut layers = Vec::with_capacity(projections.len());
        let mut fused_qkvs = Vec::with_capacity(projections.len());
        let mut dims: Option<Qwen3AsrLlmDecodeDims> = None;
        // Pass 1: allocate ALL layers' tensors first — the first upload freezes
        // the arena's backend buffer, after which no new tensors may be created.
        for projection in projections.iter() {
            let Qwen3AsrLlmLayerAttentionProjection::Generic(inner) = projection;
            // Allocate LoRA slots first so QKV storage can be selected once,
            // before any base tensor is declared. Per-projection Q/K/V LoRA
            // requires split weights; otherwise fused and split storage are
            // mutually exclusive.
            let layer_lora = allocate_layer_lora_slots(
                &arena,
                adapter,
                &inner.attn_q_name,
                &inner.attn_k_name,
                &inner.attn_v_name,
                &inner.attn_output_name,
                &inner.ffn_gate_name,
                &inner.ffn_up_name,
                &inner.ffn_down_name,
            )?;
            let force_split_qkv = layer_lora.attn_q.is_some()
                || layer_lora.attn_k.is_some()
                || layer_lora.attn_v.is_some();
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
                force_split_qkv,
            )?;
            handles.lora = layer_lora;
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
            upload_layer_lora_slots(
                &mut arena,
                adapter,
                &layers[layer_index].lora,
                &inner.attn_q_name,
                &inner.attn_k_name,
                &inner.attn_v_name,
                &inner.attn_output_name,
                &inner.ffn_gate_name,
                &inner.ffn_up_name,
                &inner.ffn_down_name,
            )?;
        }
        let mut executor = Self {
            reuse: None,
            device_token_embedding: None,
            loaded,
            runner,
            arena,
            layers,
            fused_logits_head,
            dims,
            use_native_gqa,
            rms_norm_epsilon,
            kv_cache_spec: LlmKvCacheSpec::DEFAULT,
            flash_attention_precision: GgmlFlashAttentionPrecision::Default,
            #[cfg(test)]
            materialization_peak_staging_bytes: 0,
        };
        // Shared production policy for every family that builds this executor
        // (qwen/mimo/firered2/moss/serve-batch). Discrete GPU and the
        // OPENASR_QWEN_KV_CACHE_F32 opt-out stay on Default.
        let policy = resolve_qwen_family_production_kv_cache_policy(
            executor.runner.backend_kind(),
            executor.dims.head_dim,
        );
        executor.set_kv_cache_policy(policy)?;
        Ok(executor)
    }

    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn kv_cache_spec(&self) -> LlmKvCacheSpec {
        self.kv_cache_spec
    }

    /// Set the host/resident KV element-type policy for this decoder.
    ///
    /// Production constructors already call this once via
    /// [`resolve_qwen_family_production_kv_cache_policy`]. Remaining call sites
    /// are explicit overrides (golden/parity harnesses that pin F32, or tests).
    /// Validates geometry against the decoder `head_dim`; backend/GQA/flash
    /// checks still happen when the graph is composed for a concrete path.
    pub(crate) fn set_kv_cache_policy(
        &mut self,
        policy: LlmKvCachePolicy,
    ) -> Result<(), GgmlCpuGraphError> {
        let spec = policy.to_spec();
        if let Err(_reason) = spec.validate_geometry(self.dims.head_dim) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "llm KV cache policy is incompatible with decoder head_dim",
            });
        }
        // Drop any resident reuse graph built under the previous element type.
        self.reuse = None;
        self.kv_cache_spec = spec;
        Ok(())
    }

    /// Select the precision contract used by fused self-attention graphs.
    ///
    /// This is an invocation-local graph policy, not a pack or backend global.
    /// Changing it discards a reusable graph so no session can retain kernels
    /// compiled under the previous precision contract.
    pub(crate) fn set_flash_attention_precision(&mut self, precision: GgmlFlashAttentionPrecision) {
        self.reuse = None;
        self.flash_attention_precision = precision;
    }

    /// Ends a decode session/slice: discards any reusable graph poisoned by an
    /// incomplete compute, then releases the CPU per-token grow-to-fit step
    /// buffer (see `GgmlCpuGraphRunner::release_cpu_step_buffer_pool`). Callers that cache
    /// this executor across sessions (qwen's `store_cached_whole_decoder`,
    /// mimo/firered2's decoder-runtime caches) MUST call this before storing
    /// it back so the buffer stays session-scoped instead of riding along
    /// with the cached decoder indefinitely. The buffer release is a no-op on
    /// Metal/GPU or when no CPU step ever ran.
    pub(crate) fn release_session_scoped_buffers(&mut self) {
        // A failed graph compute may have committed only a prefix of resident
        // KV writes. All qwen-family cache owners call this before putting the
        // whole decoder back (qwen, firered-llm, moss-td), so discard poisoned
        // graph + KV state here while retaining stateless loaded weights and the
        // cached backend/device handle.
        if self
            .reuse
            .as_ref()
            .is_some_and(LlmReusableDecodeGraph::is_poisoned)
        {
            self.reuse = None;
        }
        self.runner.release_cpu_step_buffer_pool();
    }

    /// Human-readable provider identity for diagnostics only. Executable
    /// policy in this family consumes the runner's typed backend kind or
    /// capabilities; callers must not parse this label.
    pub(crate) fn backend_label(&self) -> String {
        self.runner.backend_label()
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

    pub(crate) fn supports_device_token_embedding(&self) -> bool {
        self.device_token_embedding.is_some() && self.supports_graph_reuse()
    }

    #[cfg(test)]
    pub(crate) fn reused_batch_width_for_test(&self) -> Option<usize> {
        self.reuse.as_ref().map(|reuse| reuse.n_seq)
    }

    pub(crate) fn reused_graph_matches(&self, n_seq: usize, max_positions: usize) -> bool {
        self.reused_graph_matches_input(n_seq, max_positions, QwenReusableDecodeInputKind::Hidden)
    }

    fn reused_graph_matches_input(
        &self,
        n_seq: usize,
        max_positions: usize,
        input_kind: QwenReusableDecodeInputKind,
    ) -> bool {
        self.reuse
            .as_ref()
            .map(|reuse| {
                !reuse.is_poisoned()
                    && reuse.n_seq == n_seq
                    && reuse.max_positions == max_positions
                    && reuse.uses_token_ids()
                        == matches!(input_kind, QwenReusableDecodeInputKind::TokenIds)
            })
            .unwrap_or(false)
    }

    #[cfg(test)]
    pub(crate) fn backend_is_metal(&self) -> bool {
        matches!(self.runner.backend_kind(), GgmlCpuGraphBackend::Metal)
    }

    /// Native-GQA multi-query prefill chunk width. Root cause of the historical
    /// GPU multi-query prefill divergence: the graph is correct on CPU at every
    /// span (byte-perfect to 256x8), but the ggml CUDA/HIP flash-attn MMA/TILE
    /// kernel mis-handles the per-query causal mask + GQA when `n_kv > 32` AND
    /// `n_query > 2`. `n_query <= 2` routes to the correct VEC kernel
    /// (`fattn.cu` `Q->ne[1] <= 2`), and `n_kv <= 32` fits a single K-tile and is
    /// correct at any chunk. Discrete-GPU backends (CUDA/HIP/Vulkan) that would
    /// trip the kernel bug now run through the unfused
    /// `llm_naive_masked_attention` graph instead of the fused kernel
    /// (`llm_prefill_uses_flash_attention`), so the host-cache chunk can be
    /// wide (8). CPU/Metal keep flash at width 8 (trusted at every span).
    pub(crate) fn safe_multi_query_prefill_chunk_size_for(
        &self,
        token_count: usize,
    ) -> Option<usize> {
        if !self.use_native_gqa {
            return None;
        }
        if self.runner.backend_kind().is_gpu_class() {
            return Some(qwen_llm_safe_gpu_prefill_query_tokens_for_backend(
                self.runner.backend_capabilities(),
                token_count,
            ));
        }
        Some(QWEN3_LLM_CPU_SAFE_PREFILL_QUERY_TOKENS)
    }

    /// Chunk width for prefill that reads/writes the host KV cache mid-prompt
    /// (`prefill_tokens_at_offset_*`). Historically `None` on every GPU-class
    /// backend (forcing the ~50 ms/token serial host-step path). Discrete GPU
    /// backends now share the multi-query policy because wide steps are routed
    /// through the non-flash attention graph (`llm_prefill_uses_flash_attention`)
    /// and are width-safe. Callers that also decode via resident-graph reuse
    /// must still seed that arena from the same path (prefer
    /// `run_prefill_auto_last_hidden`); host-cache chunking alone does not
    /// populate the reuse arena.
    pub(crate) fn safe_host_cache_prefill_chunk_size_for(
        &self,
        token_count: usize,
    ) -> Option<usize> {
        self.safe_multi_query_prefill_chunk_size_for(token_count)
    }

    /// Decide fused-flash vs unfused attention for a prefill graph step.
    /// See `qwen_llm_prefill_uses_flash_attention_for_backend`.
    fn llm_prefill_uses_flash_attention(&self, token_count: usize, kv_span: usize) -> bool {
        qwen_llm_prefill_uses_flash_attention_for_backend(
            self.runner.backend_kind(),
            token_count,
            kv_span,
        )
    }

    /// True when prefill chunk widths must stay even on this backend.
    /// Measured on gfx1200 (Windows ROCm 7.1): odd query widths of 3/5/7 in
    /// the prefill path stall for seconds per chunk (8.2 s at width 5) while
    /// widths 1/2/4/6/8 run in ~25 ms. Callers splitting a prompt into chunks
    /// must trim an odd width > 1 down by one token
    /// (`even_prefill_chunk_len`); the final single token then rides the
    /// fast width-1 step.
    pub(crate) fn prefill_chunks_require_even_width(&self) -> bool {
        self.runner
            .backend_capabilities()
            .multi_query_prefill_width_multiple()
            > 1
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
                self.flash_attention_precision,
                self.kv_cache_spec,
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
        capacity: Qwen3AsrKvCacheCapacity,
        rope_theta: f32,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        self.validate_logical_cache_capacity(layer_caches, capacity)?;
        if self.supports_graph_reuse() {
            self.run_step_reused(
                hidden,
                cache_position,
                layer_caches,
                rope_theta,
                capacity.resident_positions(),
            )
        } else {
            self.run_step(hidden, cache_position, layer_caches, rope_theta)
        }
    }

    /// Device-token variant of [`Self::run_step_auto`]. Direct GPU lanes bind
    /// the canonical token-embedding tensor from the executor's existing
    /// pack-wide loaded context and run `get_rows` inside the persistent decode
    /// graph. CPU/scheduler paths return `None` so callers keep their existing
    /// mmap-backed host gather without changing numerical behavior.
    pub(crate) fn run_token_step_auto(
        &mut self,
        token_id: u32,
        cache_position: usize,
        layer_caches: &[Qwen3AsrLayerKvCacheState],
        capacity: Qwen3AsrKvCacheCapacity,
        rope_theta: f32,
    ) -> Result<Option<Qwen3AsrLlmWholeStepOutput>, GgmlCpuGraphError> {
        self.validate_logical_cache_capacity(layer_caches, capacity)?;
        if !self.supports_device_token_embedding() {
            return Ok(None);
        }
        self.run_token_step_reused_batched(
            &[token_id],
            &[cache_position],
            rope_theta,
            capacity.resident_positions(),
        )
        .map(Some)
    }

    /// Prompt prefill for families that keep the plain "whole prompt as one
    /// `run_prefill` call" CPU path (no HIP/GPU chunk tuning) but still want
    /// `run_step_auto`'s Metal/GPU decode-graph reuse: `run_step_auto` and
    /// bulk `run_prefill` cannot be mixed for one utterance, because the
    /// persistent resident-KV graph `run_step_auto` builds on its first call
    /// is zero-initialized and only ever gets a prompt token's real K/V by
    /// that token flowing through the SAME resident arena -- a prompt
    /// prefilled instead through the bulk host-cache `run_prefill` never
    /// touches that arena, so decode would resume attending over a KV
    /// history that was never populated for the prompt span.
    ///
    /// This seeds the resident arena via `run_prefill_into_reused_batched`
    /// (`n_seq=1`): ONE graph build plus ONE batched compute over the whole
    /// prompt, exactly like bulk `run_prefill`'s own single-shot efficiency,
    /// except its `set_rows` writes land in the persistent resident arena
    /// instead of a growing/host-cache graph. An earlier version of this
    /// method instead replayed the prompt token-by-token through
    /// `run_step_auto`: correct, but serializing what is otherwise one wide
    /// batched matmul into N narrow batch-1 matmuls (plus N Metal dispatches)
    /// measured slower than bulk prefill outright on an M1 (151 tokens: 8.4s
    /// bulk vs 14.6-19.0s serialized) -- a real loss, not merely "no gain",
    /// so it was replaced with this batched seeding instead.
    /// Returns `None` when the backend does not support graph reuse, so the
    /// caller falls back to its own bulk `run_prefill` + host-cache KV write.
    pub(crate) fn run_prefill_auto_last_hidden(
        &mut self,
        token_major_values: &[f32],
        token_count: usize,
        layer_caches: &[Qwen3AsrLayerKvCacheState],
        capacity: Qwen3AsrKvCacheCapacity,
        rope_theta: f32,
        control: &std::sync::Arc<crate::api::backend::TranscriptionControl>,
    ) -> Result<Option<Vec<f32>>, GgmlCpuGraphError> {
        self.validate_logical_cache_capacity(layer_caches, capacity)?;
        if !self.supports_graph_reuse() {
            return Ok(None);
        }
        let step = self.run_prefill_into_reused_batched(
            token_major_values,
            token_count,
            1,
            capacity.resident_positions(),
            rope_theta,
            control,
        )?;
        Ok(Some(step.hidden))
    }

    /// Device-resident counterpart to [`Self::run_prefill_auto_last_hidden`].
    /// Canonical prompt token embeddings are gathered from the pack-bound
    /// embedding tensor inside each resident prefill graph; encoder-produced
    /// audio rows replace only the declared contiguous placeholder span via
    /// `ggml_set_rows`. CPU/non-reuse backends return `None` so family wrappers
    /// retain their existing host gather + splice fallback.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_token_prefill_auto_last_hidden(
        &mut self,
        token_ids: &[u32],
        audio_rows: &[f32],
        audio_positions: &[usize],
        layer_caches: &[Qwen3AsrLayerKvCacheState],
        capacity: Qwen3AsrKvCacheCapacity,
        rope_theta: f32,
        control: &std::sync::Arc<crate::api::backend::TranscriptionControl>,
    ) -> Result<Option<Vec<f32>>, GgmlCpuGraphError> {
        self.validate_logical_cache_capacity(layer_caches, capacity)?;
        if !self.supports_device_token_embedding() {
            return Ok(None);
        }
        if token_ids.is_empty() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder resident token prefill requires at least one token",
            });
        }
        if !audio_rows.len().is_multiple_of(self.dims.d_model) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder resident token prefill audio row width mismatch",
            });
        }
        let audio_count = audio_rows.len() / self.dims.d_model;
        if audio_positions.len() != audio_count {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder resident token prefill audio position count mismatch",
            });
        }
        let mut previous = None;
        for &position in audio_positions {
            if position >= token_ids.len() || previous.is_some_and(|previous| position <= previous)
            {
                return Err(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder resident token prefill audio positions must be sorted, unique, and in range",
                });
            }
            previous = Some(position);
        }

        let mut final_step = None;
        for position_offset in
            (0..token_ids.len()).step_by(QWEN3_LLM_RESIDENT_PREFILL_MAX_QUERY_TOKENS)
        {
            if control.is_canceled() {
                return Err(GgmlCpuGraphError::Canceled);
            }
            let chunk_tokens = (token_ids.len() - position_offset)
                .min(QWEN3_LLM_RESIDENT_PREFILL_MAX_QUERY_TOKENS);
            let chunk_end = position_offset + chunk_tokens;
            let first_audio =
                audio_positions.partition_point(|&position| position < position_offset);
            let end_audio = audio_positions.partition_point(|&position| position < chunk_end);
            let chunk_audio_positions = audio_positions[first_audio..end_audio]
                .iter()
                .map(|position| position - position_offset)
                .collect::<Vec<_>>();
            let chunk_audio_rows = if first_audio < end_audio {
                let row_start = first_audio.checked_mul(self.dims.d_model).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder resident token prefill audio offset overflow",
                    },
                )?;
                let row_end = end_audio.checked_mul(self.dims.d_model).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder resident token prefill audio end overflow",
                    },
                )?;
                audio_rows
                    .get(row_start..row_end)
                    .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder resident token prefill audio slice is invalid",
                    })?
            } else {
                &[][..]
            };
            let is_final_chunk = chunk_end == token_ids.len();
            let step = self.run_prefill_chunk_into_reused_batched(
                QwenResidentPrefillInput::TokenIds {
                    token_ids: &token_ids[position_offset..chunk_end],
                    audio_rows: chunk_audio_rows,
                    audio_positions_in_chunk: &chunk_audio_positions,
                },
                chunk_tokens,
                1,
                position_offset,
                capacity.resident_positions(),
                rope_theta,
                is_final_chunk,
            )?;
            if is_final_chunk {
                final_step = Some(step);
            }
        }
        Ok(Some(
            final_step
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder resident token prefill produced no final chunk",
                })?
                .hidden,
        ))
    }

    /// Materialize canonical token rows and sparse audio replacements on the
    /// selected GPU without touching the host token table. This compatibility
    /// bridge is used by the multi-request serve-batch prefill and stateless
    /// consumers that still accept a hidden-row buffer; their transformer
    /// compute remains unchanged while the embedding lookup itself is no
    /// longer Rust/CPU model math.
    pub(crate) fn materialize_token_prompt_on_device(
        &mut self,
        token_ids: &[u32],
        audio_rows: &[f32],
        audio_positions: &[usize],
    ) -> Result<Option<Vec<f32>>, GgmlCpuGraphError> {
        let Some(embedding) = self.device_token_embedding else {
            return Ok(None);
        };
        if !self.runner.backend_kind().is_gpu_class() {
            return Ok(None);
        }
        if token_ids.is_empty() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder device prompt materialization requires at least one token",
            });
        }
        if token_ids
            .iter()
            .any(|&token_id| token_id as usize >= embedding.vocab_size)
        {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder device prompt token id exceeds vocabulary",
            });
        }
        if !audio_rows.len().is_multiple_of(self.dims.d_model) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder device prompt audio row width mismatch",
            });
        }
        let audio_count = audio_rows.len() / self.dims.d_model;
        if audio_positions.len() != audio_count
            || audio_positions
                .iter()
                .any(|&position| position >= token_ids.len())
            || audio_positions
                .windows(2)
                .any(|window| window[0] >= window[1])
        {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder device prompt audio positions are invalid",
            });
        }

        let token_values = token_ids
            .iter()
            .map(|&token_id| {
                i32::try_from(token_id).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder device prompt token id exceeds i32",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut graph = self.runner.start_graph();
        let token_tensor =
            graph.new_tensor_1d_i32(token_ids.len(), "qwen_llm_device_prompt_token_ids")?;
        graph.set_input(token_tensor)?;
        let token_rows = graph.get_rows(embedding.tensor.as_graph_tensor(), token_tensor)?;
        let mut audio_upload = None;
        let output = if audio_count == 0 {
            token_rows
        } else {
            let audio_tensor = graph.new_tensor_2d_f32(
                self.dims.d_model,
                audio_count,
                "qwen_llm_device_prompt_audio_rows",
            )?;
            let audio_indices_tensor =
                graph.new_tensor_1d_i32(audio_count, "qwen_llm_device_prompt_audio_indices")?;
            graph.set_input(audio_tensor)?;
            graph.set_input(audio_indices_tensor)?;
            let audio_indices = audio_positions
                .iter()
                .copied()
                .map(|position| {
                    i32::try_from(position).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder device prompt audio index exceeds i32",
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            audio_upload = Some((audio_tensor, audio_indices_tensor, audio_indices));
            graph.set_rows(token_rows, audio_tensor, audio_indices_tensor)?
        };
        graph.set_output(output)?;
        graph.prepare_outputs_for_upload(&[output])?;
        graph.set_i32_slice(
            token_tensor,
            &token_values,
            "qwen_llm_device_prompt_token_ids",
        )?;
        if let Some((audio_tensor, audio_indices_tensor, audio_indices)) = audio_upload {
            graph.set_f32_slice(
                audio_tensor,
                audio_rows,
                "qwen_llm_device_prompt_audio_rows",
            )?;
            graph.set_i32_slice(
                audio_indices_tensor,
                &audio_indices,
                "qwen_llm_device_prompt_audio_indices",
            )?;
        }
        let output_len = token_ids.len().checked_mul(self.dims.d_model).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder device prompt output width overflow",
            },
        )?;
        graph.compute_output_f32(output, output_len).map(Some)
    }

    fn validate_logical_cache_capacity(
        &self,
        layer_caches: &[Qwen3AsrLayerKvCacheState],
        capacity: Qwen3AsrKvCacheCapacity,
    ) -> Result<(), GgmlCpuGraphError> {
        if layer_caches.len() != self.layers.len() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder layer/cache count mismatch",
            });
        }
        if layer_caches
            .iter()
            .any(|cache| cache.max_positions() != capacity.logical_positions())
        {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder host KV span does not match planned logical capacity",
            });
        }
        Ok(())
    }

    #[cfg(test)]
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
                self.flash_attention_precision,
                self.kv_cache_spec,
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

    /// Compute a fused device-side top-1 token from an already-materialized
    /// decoder hidden row. This avoids allocating a separate full-vocabulary
    /// logits executor after a resident prefill graph has populated its KV.
    pub(crate) fn fused_logits_top1_from_hidden(
        &mut self,
        hidden: &[f32],
    ) -> Result<Option<u32>, GgmlCpuGraphError> {
        let Some(fused_logits_head) = self.fused_logits_head.as_ref() else {
            return Ok(None);
        };
        if hidden.len() != self.dims.d_model {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder fused top1 hidden width mismatch",
            });
        }
        let mut graph = self.runner.start_graph();
        let hidden_tensor =
            graph.new_tensor_2d_f32(self.dims.d_model, 1, "qwen_llm_fused_logits_hidden")?;
        graph.set_input(hidden_tensor)?;
        let top1 =
            build_fused_logits_top1(&self.arena, fused_logits_head, &mut graph, hidden_tensor, 1)?;
        graph.set_output(top1)?;
        graph.set_f32_slice(hidden_tensor, hidden, "qwen_llm_fused_logits_hidden")?;
        let token_id = graph
            .compute_output_i32(top1, 1)?
            .first()
            .copied()
            .ok_or(GgmlCpuGraphError::OutputByteSizeMismatch {
                expected: std::mem::size_of::<i32>(),
                actual: 0,
            })
            .and_then(|token_id| {
                validate_fused_top1_token_id(token_id, fused_logits_head.vocab_size)
            })?;
        Ok(Some(token_id))
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

    /// Run a complete prompt as one causal forward pass without materializing
    /// the per-layer K/V tensors for a later autoregressive decode.
    ///
    /// This is the exact execution contract needed by non-autoregressive
    /// consumers such as Qwen3-ForcedAligner: they consume the final hidden
    /// state once and never seed a decode cache. Keeping every layer's K/V as a
    /// graph output extends all of those tensors' lifetimes to the end of the
    /// graph and defeats the liveness allocator for no mathematical benefit.
    pub(crate) fn run_stateless_prefill(
        &mut self,
        token_major_hidden: &[f32],
        token_count: usize,
        rope_theta: f32,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        self.run_prefill_with_batched_history(
            token_major_hidden,
            token_count,
            1,
            0,
            token_count,
            &[&[]],
            rope_theta,
            false,
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
            true,
        )
    }

    fn sequence_major_prefill_chunk(
        sequence_major_hidden: &[f32],
        token_count: usize,
        n_seq: usize,
        d_model: usize,
        position_offset: usize,
        chunk_tokens: usize,
    ) -> Result<Vec<f32>, GgmlCpuGraphError> {
        let chunk_end = position_offset.checked_add(chunk_tokens).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder resident prefill chunk position overflow",
            },
        )?;
        if n_seq == 0 || chunk_tokens == 0 || chunk_end > token_count {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder resident prefill chunk span is invalid",
            });
        }
        let per_sequence =
            token_count
                .checked_mul(d_model)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder resident prefill sequence width overflow",
                })?;
        let expected =
            per_sequence
                .checked_mul(n_seq)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder resident prefill hidden width overflow",
                })?;
        if sequence_major_hidden.len() != expected {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder resident prefill hidden width mismatch",
            });
        }
        let chunk_width =
            chunk_tokens
                .checked_mul(d_model)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder resident prefill chunk width overflow",
                })?;
        let mut chunk = Vec::with_capacity(chunk_width.checked_mul(n_seq).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder resident prefill chunk allocation overflow",
            },
        )?);
        for sequence_index in 0..n_seq {
            let sequence_start = sequence_index.checked_mul(per_sequence).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder resident prefill sequence offset overflow",
                },
            )?;
            let token_start = position_offset.checked_mul(d_model).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder resident prefill token offset overflow",
                },
            )?;
            let start = sequence_start.checked_add(token_start).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder resident prefill chunk start overflow",
                },
            )?;
            let end =
                start
                    .checked_add(chunk_width)
                    .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder resident prefill chunk end overflow",
                    })?;
            chunk.extend_from_slice(sequence_major_hidden.get(start..end).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder resident prefill chunk slice is out of bounds",
                },
            )?);
        }
        Ok(chunk)
    }

    pub(crate) fn run_prefill_into_reused_batched(
        &mut self,
        sequence_major_hidden: &[f32],
        token_count: usize,
        n_seq: usize,
        max_positions: usize,
        rope_theta: f32,
        control: &std::sync::Arc<crate::api::backend::TranscriptionControl>,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        if token_count == 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder resident prefill token count must be positive",
            });
        }
        let mut final_step = None;
        for position_offset in (0..token_count).step_by(QWEN3_LLM_RESIDENT_PREFILL_MAX_QUERY_TOKENS)
        {
            // L1.2 cooperative cancel: poll between resident prefill chunks so
            // stop lands at a chunk boundary instead of waiting for the whole
            // prompt bulk (or the long-form slice boundary). Pause stays L0-only.
            if control.is_canceled() {
                return Err(GgmlCpuGraphError::Canceled);
            }
            let chunk_tokens =
                (token_count - position_offset).min(QWEN3_LLM_RESIDENT_PREFILL_MAX_QUERY_TOKENS);
            let chunk_hidden = Self::sequence_major_prefill_chunk(
                sequence_major_hidden,
                token_count,
                n_seq,
                self.dims.d_model,
                position_offset,
                chunk_tokens,
            )?;
            let is_final_chunk = position_offset.checked_add(chunk_tokens) == Some(token_count);
            let step = self.run_prefill_chunk_into_reused_batched(
                QwenResidentPrefillInput::Hidden(&chunk_hidden),
                chunk_tokens,
                n_seq,
                position_offset,
                max_positions,
                rope_theta,
                is_final_chunk,
            )?;
            if is_final_chunk {
                final_step = Some(step);
            }
        }
        final_step.ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "whole-decoder resident prefill produced no final chunk",
        })
    }

    fn run_prefill_chunk_into_reused_batched(
        &mut self,
        input: QwenResidentPrefillInput<'_>,
        token_count: usize,
        n_seq: usize,
        position_offset: usize,
        max_positions: usize,
        rope_theta: f32,
        materialize_last_hidden: bool,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        self.require_native_gqa_for_multi_sequence(n_seq)?;
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
        let chunk_end = position_offset.checked_add(token_count).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder resident prefill position span overflow",
            },
        )?;
        if max_positions < chunk_end {
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
        match &input {
            QwenResidentPrefillInput::Hidden(hidden) => {
                if hidden.len() != expected_hidden {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder resident prefill hidden width mismatch",
                    });
                }
            }
            QwenResidentPrefillInput::TokenIds {
                token_ids,
                audio_rows,
                audio_positions_in_chunk,
            } => {
                if n_seq != 1 || token_ids.len() != token_count {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder token prefill requires one token-id row per token",
                    });
                }
                if audio_rows.len() % dims.d_model != 0 {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder token prefill audio row width mismatch",
                    });
                }
                let audio_count = audio_rows.len() / dims.d_model;
                if audio_positions_in_chunk.len() != audio_count
                    || audio_positions_in_chunk
                        .iter()
                        .any(|&position| position >= token_count)
                    || audio_positions_in_chunk
                        .windows(2)
                        .any(|window| window[0] >= window[1])
                {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder token prefill audio positions are invalid",
                    });
                }
                let embedding =
                    self.device_token_embedding
                        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                            reason: "whole-decoder token prefill requires a device token embedding",
                        })?;
                if token_ids
                    .iter()
                    .any(|&token_id| token_id as usize >= embedding.vocab_size)
                {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder token prefill token id exceeds vocabulary",
                    });
                }
            }
        }
        if !self.reused_graph_matches(n_seq, max_positions) {
            self.rebuild_reused_batched_graph(
                n_seq,
                max_positions,
                rope_theta,
                None,
                QwenReusableDecodeInputKind::Hidden,
            )?;
        }
        let use_flash_attention = self.llm_prefill_uses_flash_attention(token_count, max_positions);

        let mut row_indices = Vec::with_capacity(output_tokens);
        let mut row_indices_usize = Vec::with_capacity(output_tokens);
        let mut positions = Vec::with_capacity(output_tokens);
        for _sequence_index in 0..n_seq {
            for token_position in 0..token_count {
                let absolute_position = position_offset.checked_add(token_position).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder resident prefill position overflow",
                    },
                )?;
                row_indices_usize.push(absolute_position);
                row_indices.push(i32::try_from(absolute_position).map_err(|_| {
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder resident prefill cache index exceeds ggml int boundary",
                    }
                })?);
                positions.push(i32::try_from(absolute_position).map_err(|_| {
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
            let mut hidden_upload = None;
            let mut token_upload = None;
            let mut audio_upload = None;
            let state = match input {
                QwenResidentPrefillInput::Hidden(hidden) => {
                    let hidden_tensor = graph.new_tensor_2d_f32(
                        dims.d_model,
                        output_tokens,
                        "qwen_llm_prefill_resident_hidden",
                    )?;
                    graph.set_input(hidden_tensor)?;
                    hidden_upload = Some((hidden_tensor, hidden));
                    hidden_tensor
                }
                QwenResidentPrefillInput::TokenIds {
                    token_ids,
                    audio_rows,
                    audio_positions_in_chunk,
                } => {
                    let embedding = self
                        .device_token_embedding
                        .expect("device token embedding validated before resident token prefill");
                    let token_values = token_ids
                        .iter()
                        .map(|&token_id| {
                            i32::try_from(token_id).map_err(|_| {
                                GgmlCpuGraphError::UnsupportedInputs {
                                    reason: "whole-decoder token prefill token id exceeds i32",
                                }
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let token_tensor = graph
                        .new_tensor_1d_i32(token_count, "qwen_llm_prefill_resident_token_ids")?;
                    graph.set_input(token_tensor)?;
                    let token_rows =
                        graph.get_rows(embedding.tensor.as_graph_tensor(), token_tensor)?;
                    token_upload = Some((token_tensor, token_values));
                    if audio_rows.is_empty() {
                        token_rows
                    } else {
                        let audio_count = audio_rows.len() / dims.d_model;
                        let audio_tensor = graph.new_tensor_2d_f32(
                            dims.d_model,
                            audio_count,
                            "qwen_llm_prefill_resident_audio_rows",
                        )?;
                        let audio_indices = audio_positions_in_chunk
                            .iter()
                            .copied()
                            .map(|index| {
                                i32::try_from(index).map_err(|_| {
                                    GgmlCpuGraphError::UnsupportedInputs {
                                        reason: "whole-decoder token prefill audio index exceeds i32",
                                    }
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let audio_indices_tensor = graph.new_tensor_1d_i32(
                            audio_count,
                            "qwen_llm_prefill_resident_audio_indices",
                        )?;
                        graph.set_input(audio_tensor)?;
                        graph.set_input(audio_indices_tensor)?;
                        let spliced =
                            graph.set_rows(token_rows, audio_tensor, audio_indices_tensor)?;
                        audio_upload = Some((
                            audio_tensor,
                            audio_rows,
                            audio_indices_tensor,
                            audio_indices,
                        ));
                        spliced
                    }
                }
            };
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
                    self.flash_attention_precision,
                    self.kv_cache_spec,
                    true,
                ),
                LlmDecoderStackInputs {
                    state,
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
            let (output_state, output_len) = if materialize_last_hidden && n_seq == 1 {
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
                let completion = graph.view_1d(state, 1, 0)?;
                (completion, 1)
            };
            graph.set_output(output_state)?;
            graph.prepare_outputs_for_upload(&[output_state])?;
            if let Some((tensor, values)) = hidden_upload {
                graph.set_f32_slice(tensor, values, "qwen_llm_prefill_resident_hidden")?;
            }
            if let Some((tensor, values)) = token_upload {
                graph.set_i32_slice(tensor, &values, "qwen_llm_prefill_resident_token_ids")?;
            }
            if let Some((audio, values, indices, index_values)) = audio_upload {
                graph.set_f32_slice(audio, values, "qwen_llm_prefill_resident_audio_rows")?;
                graph.set_i32_slice(
                    indices,
                    &index_values,
                    "qwen_llm_prefill_resident_audio_indices",
                )?;
            }
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
        if result.is_err() {
            // This temporary prefill graph writes into `reuse`'s resident KV
            // arena. Its own builder is ephemeral, so fail closed and propagate
            // any incomplete result to the persistent owner before cache reuse.
            reuse.mark_poisoned_after_failed_compute();
        }
        let restore_result = reuse.builder().restore_prepared_graph_allocation();
        self.reuse = Some(reuse);
        match result {
            Ok(output) => restore_result.map(|()| output),
            Err(error) => Err(error),
        }
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
            true,
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
        materialize_layer_kv: bool,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        self.require_native_gqa_for_multi_sequence(n_seq)?;
        let dims = self.dims;
        let kv_cache_spec = self.kv_cache_spec;
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
                self.flash_attention_precision,
                kv_cache_spec,
                materialize_layer_kv,
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

        let mut requested: Vec<(GgmlCpuTensor, usize)> =
            Vec::with_capacity(1 + usize::from(materialize_layer_kv) * 2 * self.layers.len());
        requested.push((state, expected_hidden));
        let layer_kv_width = dims.k_width.checked_mul(output_tokens).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill KV output width overflow",
            },
        )?;
        if materialize_layer_kv {
            for (k, v) in &kv_outputs {
                requested.push((*k, layer_kv_width));
                requested.push((*v, layer_kv_width));
            }
        }
        let requested_tensors = requested
            .iter()
            .map(|(tensor, _)| *tensor)
            .collect::<Vec<_>>();
        // Stateless consumers need the one-shot liveness plan before uploads;
        // otherwise the first upload allocates the declaration-sized CPU
        // arena. Autoregressive consumers deliberately keep the established
        // upload-first path: the fixed-span K/V inputs are persistent across
        // the prefill/decode boundary, while a scheduler plan created before
        // those uploads can recycle their backing storage and silently corrupt
        // the materialized cache.
        if !materialize_layer_kv {
            graph.prepare_one_shot_outputs_for_upload(&requested_tensors)?;
        }
        if env_var_truthy(QWEN3_LLM_PREFILL_ALLOCATION_PROFILE_ENV) {
            eprintln!(
                "openasr-qwen-prefill-allocation token_count={token_count} materialize_layer_kv={materialize_layer_kv} direct_graph_bytes={:?}",
                graph.prepared_direct_graph_allocation_bytes()
            );
        }

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
        // The first prefill has no host KV prefix. F32 history tensors get an
        // explicit zero-filled staging vector in `qwen_prefill_history_inputs_for_layer`.
        // Q8_0 must do the same rather than requiring a cache that only exists
        // after the prefill output has been written. A zero q8_0 block (scale
        // and quantized payload both zero) is the exact empty-history value.
        let q8_empty_history = if matches!(kv_cache_spec.host, GgmlKvElementType::Q8_0)
            && position_offset == 0
        {
            let row_nbytes = GgmlKvElementType::Q8_0
                .row_nbytes(dims.head_dim)
                .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                    reason: "q8_0 host prefill empty history row size invalid",
                })?;
            let history_nbytes = row_nbytes
                .checked_mul(dims.kv_heads)
                .and_then(|bytes| bytes.checked_mul(total_token_count))
                .and_then(|bytes| bytes.checked_mul(n_seq))
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "q8_0 host prefill empty history size overflow",
                })?;
            Some(vec![0_u8; history_nbytes])
        } else {
            None
        };
        for (layer_index, (key_history, value_history)) in kv_inputs.into_iter().enumerate() {
            match kv_cache_spec.host {
                GgmlKvElementType::F32 => {
                    let (key_values, value_values) = qwen_prefill_history_inputs_for_layer(
                        dims,
                        total_token_count,
                        n_seq,
                        layer_index,
                        position_offset,
                        layer_caches_by_sequence,
                    )?;
                    graph.set_f32_slice(
                        key_history,
                        &key_values,
                        "qwen_llm_prefill_key_history",
                    )?;
                    graph.set_f32_slice(
                        value_history,
                        &value_values,
                        "qwen_llm_prefill_value_history",
                    )?;
                }
                GgmlKvElementType::Q8_0 => {
                    // Host q8_0 stores packed rows; upload them directly into the
                    // matching q8_0 history tensors without a full f32 staging buffer.
                    if n_seq != 1 || layer_caches_by_sequence.len() != 1 {
                        return Err(GgmlCpuGraphError::UnsupportedInputs {
                            reason: "q8_0 host prefill history upload currently supports n_seq=1",
                        });
                    }
                    if position_offset == 0 {
                        let empty_history = q8_empty_history.as_deref().ok_or(
                            GgmlCpuGraphError::UnsupportedInputs {
                                reason: "q8_0 host prefill empty history was not initialized",
                            },
                        )?;
                        graph.set_bytes_slice(
                            key_history,
                            empty_history,
                            "qwen_llm_prefill_key_history",
                        )?;
                        graph.set_bytes_slice(
                            value_history,
                            empty_history,
                            "qwen_llm_prefill_value_history",
                        )?;
                    } else {
                        let cache = layer_caches_by_sequence[0].get(layer_index).ok_or(
                            GgmlCpuGraphError::UnsupportedInputs {
                                reason: "q8_0 host prefill history layer/cache count mismatch",
                            },
                        )?;
                        cache.upload_history_prefix_to_fixed_span_graph(
                            &mut graph,
                            key_history,
                            value_history,
                            position_offset,
                            total_token_count,
                            "qwen_llm_prefill_key_history",
                            "qwen_llm_prefill_value_history",
                        )?;
                    }
                }
                GgmlKvElementType::F16 => {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "host prefill history upload rejects f16 storage",
                    });
                }
            }
        }

        let build_micros = build_started_at.elapsed().as_micros();
        let compute_started_at = std::time::Instant::now();
        let mut outputs = graph.compute_outputs_f32(&requested)?;
        let compute_micros = compute_started_at.elapsed().as_micros();
        let hidden_out = outputs.remove(0);
        let mut layer_kv = Vec::with_capacity(if materialize_layer_kv {
            self.layers.len()
        } else {
            0
        });
        if materialize_layer_kv {
            for _ in 0..self.layers.len() {
                let k = outputs.remove(0);
                let v = outputs.remove(0);
                layer_kv.push((k, v));
            }
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
            .map(|reuse| {
                reuse.is_poisoned() || reuse.max_positions != max_positions || reuse.n_seq != n_seq
            })
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
                layer_count,
                dims.head_dim,
                max_positions,
                dims.kv_heads,
                n_seq,
                "qwen_llm_resident_kv",
                self.kv_cache_spec,
            )?;

            let mut session = self
                .runner
                .start_persistent_graph_session(qwen_llm_graph_context_bytes())?;
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
                    self.flash_attention_precision,
                    self.kv_cache_spec,
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
                Some(hidden_tensor),
                None,
                row_indices,
                positions,
                attention_mask,
                state,
                top1,
            ));
        }

        let reuse = self.reuse.as_mut().expect("reuse graph built above");
        let hidden_tensor = reuse
            .hidden_tensor
            .expect("serial hidden-input reuse graph built above");
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
        self.run_step_reused_batched_inner(
            QwenReusableDecodeInput::Hidden(hidden),
            cache_positions,
            rope_theta,
            max_positions,
            None,
        )
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
            QwenReusableDecodeInput::Hidden(hidden),
            cache_positions,
            rope_theta,
            max_positions,
            Some(seeded_layer_kv_by_sequence),
        )
    }

    #[cfg(test)]
    pub(crate) fn run_step_reused_batched_top1(
        &mut self,
        hidden: &[f32],
        cache_positions: &[usize],
        rope_theta: f32,
        max_positions: usize,
    ) -> Result<Qwen3AsrLlmWholeStepTop1Output, GgmlCpuGraphError> {
        self.run_step_reused_batched_top1_inner(
            QwenReusableDecodeInput::Hidden(hidden),
            cache_positions,
            rope_theta,
            max_positions,
            None,
        )
    }

    #[cfg(test)]
    pub(crate) fn run_step_reused_batched_seeded_top1(
        &mut self,
        hidden: &[f32],
        cache_positions: &[usize],
        seeded_layer_kv_by_sequence: &[&[Qwen3AsrLayerKvCacheState]],
        rope_theta: f32,
        max_positions: usize,
    ) -> Result<Qwen3AsrLlmWholeStepTop1Output, GgmlCpuGraphError> {
        self.run_step_reused_batched_top1_inner(
            QwenReusableDecodeInput::Hidden(hidden),
            cache_positions,
            rope_theta,
            max_positions,
            Some(seeded_layer_kv_by_sequence),
        )
    }

    pub(crate) fn run_token_step_reused_batched(
        &mut self,
        token_ids: &[u32],
        cache_positions: &[usize],
        rope_theta: f32,
        max_positions: usize,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        self.run_step_reused_batched_inner(
            QwenReusableDecodeInput::TokenIds(token_ids),
            cache_positions,
            rope_theta,
            max_positions,
            None,
        )
    }

    pub(crate) fn run_token_step_reused_batched_seeded(
        &mut self,
        token_ids: &[u32],
        cache_positions: &[usize],
        seeded_layer_kv_by_sequence: &[&[Qwen3AsrLayerKvCacheState]],
        rope_theta: f32,
        max_positions: usize,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        self.run_step_reused_batched_inner(
            QwenReusableDecodeInput::TokenIds(token_ids),
            cache_positions,
            rope_theta,
            max_positions,
            Some(seeded_layer_kv_by_sequence),
        )
    }

    pub(crate) fn run_token_step_reused_batched_top1(
        &mut self,
        token_ids: &[u32],
        cache_positions: &[usize],
        rope_theta: f32,
        max_positions: usize,
    ) -> Result<Qwen3AsrLlmWholeStepTop1Output, GgmlCpuGraphError> {
        self.run_step_reused_batched_top1_inner(
            QwenReusableDecodeInput::TokenIds(token_ids),
            cache_positions,
            rope_theta,
            max_positions,
            None,
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
            QwenReusableDecodeInputKind::Hidden,
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
        if reuse.is_poisoned() {
            return Err(GgmlCpuGraphError::GraphSessionPoisoned);
        }
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
            self.kv_cache_spec.resident,
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
        if reuse.is_poisoned() {
            return Err(GgmlCpuGraphError::GraphSessionPoisoned);
        }
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
            self.kv_cache_spec.resident,
        )
    }

    fn rebuild_reused_batched_graph(
        &mut self,
        n_seq: usize,
        max_positions: usize,
        rope_theta: f32,
        seed: Option<(&[usize], &[&[Qwen3AsrLayerKvCacheState]])>,
        input_kind: QwenReusableDecodeInputKind,
    ) -> Result<(), GgmlCpuGraphError> {
        self.require_native_gqa_for_multi_sequence(n_seq)?;
        if n_seq == 0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder n_seq must be positive",
            });
        }
        let dims = self.dims;
        let rope = GgmlRopeExtParams::qwen_neox(dims.head_dim, max_positions, rope_theta)?;
        // Switching the persistent graph from host-hidden input to token-id
        // input must preserve the prompt/generated KV already resident on the
        // device. Reuse the arena when only the graph input contract changes;
        // shape changes, poison, and explicit seeds allocate a fresh arena.
        let reusable_arena = if seed.is_none()
            && self.reuse.as_ref().is_some_and(|reuse| {
                !reuse.is_poisoned() && reuse.n_seq == n_seq && reuse.max_positions == max_positions
            }) {
            self.reuse
                .take()
                .map(LlmReusableDecodeGraph::into_resident_kv_arena)
        } else {
            self.reuse = None;
            None
        };
        let mut resident_kv_arena = match reusable_arena {
            Some(arena) => arena,
            None => allocate_zeroed_llm_resident_kv_arena(
                &self.runner,
                self.layers.len(),
                dims.head_dim,
                max_positions,
                dims.kv_heads,
                n_seq,
                "qwen_llm_resident_kv",
                self.kv_cache_spec,
            )?,
        };
        if let Some((prefix_lengths, seed_layers)) = seed {
            seed_qwen_batched_resident_kv_arena(
                &mut resident_kv_arena,
                dims.head_dim,
                max_positions,
                dims.kv_heads,
                prefix_lengths,
                seed_layers,
                self.kv_cache_spec.resident,
            )?;
        }

        let mut session = self
            .runner
            .start_persistent_graph_session(qwen_llm_graph_context_bytes())?;
        let graph = session.builder();
        let (hidden_tensor, token_ids, input_state) = match input_kind {
            QwenReusableDecodeInputKind::Hidden => {
                let hidden =
                    graph.new_tensor_2d_f32(dims.d_model, n_seq, "qwen_llm_reuse_hidden")?;
                graph.set_input(hidden)?;
                (Some(hidden), None, hidden)
            }
            QwenReusableDecodeInputKind::TokenIds => {
                let embedding =
                    self.device_token_embedding
                        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                            reason: "whole-decoder device token embedding is not configured",
                        })?;
                let token_ids = graph.new_tensor_1d_i32(n_seq, "qwen_llm_reuse_token_ids")?;
                graph.set_input(token_ids)?;
                let state = graph.get_rows(embedding.tensor.as_graph_tensor(), token_ids)?;
                (None, Some(token_ids), state)
            }
        };
        let row_indices_tensor =
            graph.new_tensor_4d_i32(1, 1, n_seq, 1, "qwen_llm_reuse_row_index")?;
        let positions = graph.new_tensor_1d_i32(n_seq, "qwen_llm_reuse_position")?;
        let attention_mask =
            graph.new_tensor_4d_f16(max_positions, 1, 1, n_seq, "qwen_llm_reuse_self_mask")?;
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
                self.flash_attention_precision,
                self.kv_cache_spec,
                true,
            ),
            LlmDecoderStackInputs {
                state: input_state,
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
            token_ids,
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
        input: QwenReusableDecodeInput<'_>,
        cache_positions: &[usize],
        rope_theta: f32,
        max_positions: usize,
        seeded_layer_kv_by_sequence: Option<&[&[Qwen3AsrLayerKvCacheState]]>,
    ) -> Result<Qwen3AsrLlmWholeStepOutput, GgmlCpuGraphError> {
        let dims = self.dims;
        let n_seq = cache_positions.len();
        self.require_native_gqa_for_multi_sequence(n_seq)?;
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
        let input_kind = input.kind();
        let token_ids_i32 = match input {
            QwenReusableDecodeInput::Hidden(hidden) => {
                if hidden.len() != expected_hidden {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder hidden width mismatch",
                    });
                }
                None
            }
            QwenReusableDecodeInput::TokenIds(token_ids) => {
                let embedding =
                    self.device_token_embedding
                        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                            reason: "whole-decoder device token embedding is not configured",
                        })?;
                if token_ids.len() != n_seq {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder token-id batch width mismatch",
                    });
                }
                Some(
                    token_ids
                        .iter()
                        .map(|&token_id| {
                            if usize::try_from(token_id).ok().is_none_or(|token_id| {
                                token_id >= embedding.vocab_size
                            }) {
                                return Err(GgmlCpuGraphError::UnsupportedInputs {
                                    reason: "whole-decoder token id is outside the embedding vocabulary",
                                });
                            }
                            i32::try_from(token_id).map_err(|_| {
                                GgmlCpuGraphError::UnsupportedInputs {
                                    reason: "whole-decoder token id exceeds ggml int boundary",
                                }
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
        };

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

        let needs_build = !self.reused_graph_matches_input(n_seq, max_positions, input_kind)
            || seeded_layer_kv_by_sequence.is_some();
        if needs_build {
            let seed =
                seeded_layer_kv_by_sequence.map(|seed_layers| (cache_positions, seed_layers));
            self.rebuild_reused_batched_graph(n_seq, max_positions, rope_theta, seed, input_kind)?;
        }

        let reuse = self.reuse.as_mut().expect("reuse graph built above");
        let hidden_tensor = reuse.hidden_tensor;
        let token_ids_tensor = reuse.token_ids;
        let row_indices_tensor = reuse.row_indices;
        let positions = reuse.positions;
        let attention_mask = reuse.attention_mask;
        let state = reuse.state;
        let graph = reuse.builder();

        match input {
            QwenReusableDecodeInput::Hidden(hidden) => {
                let hidden_tensor = hidden_tensor.ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder hidden-input graph is missing its input tensor",
                })?;
                graph.set_f32_slice(hidden_tensor, hidden, "qwen_llm_reuse_hidden")?;
            }
            QwenReusableDecodeInput::TokenIds(_) => {
                let token_ids = token_ids_tensor.ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder token-input graph is missing its input tensor",
                })?;
                graph.set_i32_slice(
                    token_ids,
                    token_ids_i32
                        .as_deref()
                        .expect("token input validated above"),
                    "qwen_llm_reuse_token_ids",
                )?;
            }
        }
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
        input: QwenReusableDecodeInput<'_>,
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
        let input_kind = input.kind();
        let token_ids_i32 = match input {
            QwenReusableDecodeInput::Hidden(hidden) => {
                if hidden.len() != expected_hidden {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder hidden width mismatch",
                    });
                }
                None
            }
            QwenReusableDecodeInput::TokenIds(token_ids) => {
                let embedding =
                    self.device_token_embedding
                        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                            reason: "whole-decoder device token embedding is not configured",
                        })?;
                if token_ids.len() != n_seq {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "whole-decoder token-id batch width mismatch",
                    });
                }
                Some(
                    token_ids
                        .iter()
                        .map(|&token_id| {
                            if usize::try_from(token_id).ok().is_none_or(|token_id| {
                                token_id >= embedding.vocab_size
                            }) {
                                return Err(GgmlCpuGraphError::UnsupportedInputs {
                                    reason: "whole-decoder token id is outside the embedding vocabulary",
                                });
                            }
                            i32::try_from(token_id).map_err(|_| {
                                GgmlCpuGraphError::UnsupportedInputs {
                                    reason: "whole-decoder token id exceeds ggml int boundary",
                                }
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
        };

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

        let needs_build = !self.reused_graph_matches_input(n_seq, max_positions, input_kind)
            || self.reuse.as_ref().is_none_or(|reuse| reuse.top1.is_none())
            || seeded_layer_kv_by_sequence.is_some();
        if needs_build {
            let seed =
                seeded_layer_kv_by_sequence.map(|seed_layers| (cache_positions, seed_layers));
            self.rebuild_reused_batched_graph(n_seq, max_positions, rope_theta, seed, input_kind)?;
        }

        let reuse = self.reuse.as_mut().expect("reuse graph built above");
        let hidden_tensor = reuse.hidden_tensor;
        let token_ids_tensor = reuse.token_ids;
        let row_indices_tensor = reuse.row_indices;
        let positions = reuse.positions;
        let attention_mask = reuse.attention_mask;
        let top1 = reuse.top1.ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "whole-decoder fused top1 output was not prepared",
        })?;
        let graph = reuse.builder();

        match input {
            QwenReusableDecodeInput::Hidden(hidden) => {
                let hidden_tensor = hidden_tensor.ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder hidden-input graph is missing its input tensor",
                })?;
                graph.set_f32_slice(hidden_tensor, hidden, "qwen_llm_reuse_hidden")?;
            }
            QwenReusableDecodeInput::TokenIds(_) => {
                let token_ids = token_ids_tensor.ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "whole-decoder token-input graph is missing its input tensor",
                })?;
                graph.set_i32_slice(
                    token_ids,
                    token_ids_i32
                        .as_deref()
                        .expect("token input validated above"),
                    "qwen_llm_reuse_token_ids",
                )?;
            }
        }
        let mask_bits = build_fixed_kv_attention_mask_bits_for_sequences(
            max_positions,
            &total_tokens_by_sequence,
        )?;
        graph.set_f16_bits_slice(attention_mask, &mask_bits, "qwen_llm_reuse_self_mask")?;
        graph.set_i32_slice(row_indices_tensor, &row_indices, "qwen_llm_reuse_row_index")?;
        graph.set_i32_slice(positions, &rope_positions, "qwen_llm_reuse_position")?;

        #[cfg(test)]
        let compute_started_at = std::time::Instant::now();
        let token_ids = graph.compute_output_i32(top1, 1)?;
        #[cfg(test)]
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
            #[cfg(test)]
            layer_kv: Vec::new(),
            #[cfg(test)]
            build_micros: 0,
            #[cfg(test)]
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
        if !matches!(cache.element_type(), GgmlKvElementType::F32) {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder f32 prefill history helper requires host f32 KV",
            });
        }
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
        let keys = history
            .keys_f32
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill history missing f32 keys",
            })?;
        let values = history
            .values_f32
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "whole-decoder prefill history missing f32 values",
            })?;
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
            key_values[target_start..target_end].copy_from_slice(&keys[source_start..source_end]);
            value_values[target_start..target_end]
                .copy_from_slice(&values[source_start..source_end]);
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
    resident_element_type: GgmlKvElementType,
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
    let host_type = layer_kv_by_sequence
        .first()
        .and_then(|layers| layers.first())
        .map(|cache| cache.element_type())
        .unwrap_or(GgmlKvElementType::F32);
    if layer_kv_by_sequence
        .iter()
        .any(|layers| layers.iter().any(|cache| cache.element_type() != host_type))
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "batched resident KV seed host element type mismatch",
        });
    }
    match (host_type, resident_element_type) {
        (GgmlKvElementType::F32, GgmlKvElementType::F16) => {}
        (GgmlKvElementType::Q8_0, GgmlKvElementType::Q8_0) => {}
        _ => {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "batched resident KV seed host/resident element type pair unsupported",
            });
        }
    }
    let plane_elems = qwen_resident_kv_plane_elems(head_dim, max_positions, kv_heads)?;
    let plane_nbytes = host_type
        .plane_nbytes(head_dim, max_positions, kv_heads)
        .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
            reason: "batched resident KV seed plane byte size overflow",
        })?;
    let tensor_elems = plane_elems.checked_mul(prefix_lengths.len()).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "batched resident KV seed tensor size overflow",
        },
    )?;
    let tensor_nbytes = plane_nbytes.checked_mul(prefix_lengths.len()).ok_or(
        GgmlCpuGraphError::UnsupportedInputs {
            reason: "batched resident KV seed tensor byte size overflow",
        },
    )?;

    for layer_index in 0..layer_count {
        match host_type {
            GgmlKvElementType::F32 => {
                let mut key_planes = vec![0.0_f32; tensor_elems];
                let mut value_planes = vec![0.0_f32; tensor_elems];
                for (sequence_index, sequence_layers) in layer_kv_by_sequence.iter().enumerate() {
                    let prefix_length = prefix_lengths[sequence_index];
                    let history = sequence_layers[layer_index]
                        .full_history_storage()
                        .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV seed host cache storage invalid",
                        })?;
                    if history.head_dim != head_dim
                        || history.kv_heads != kv_heads
                        || history.max_positions > max_positions
                        || prefix_length > history.max_positions
                    {
                        return Err(GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV seed host cache shape mismatch",
                        });
                    }
                    if history.written_positions != prefix_length {
                        return Err(GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV seed written prefix mismatch",
                        });
                    }
                    let keys = history
                        .keys_f32
                        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV seed missing f32 keys",
                        })?;
                    let values =
                        history
                            .values_f32
                            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                                reason: "batched resident KV seed missing f32 values",
                            })?;
                    let host_plane_elems =
                        qwen_resident_kv_plane_elems(head_dim, history.max_positions, kv_heads)?;
                    if keys.len() != host_plane_elems || values.len() != host_plane_elems {
                        return Err(GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV seed host cache plane length mismatch",
                        });
                    }
                    let plane_start = sequence_index.checked_mul(plane_elems).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV seed plane offset overflow",
                        },
                    )?;
                    let source_head_stride = history.max_positions.checked_mul(head_dim).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV seed source stride overflow",
                        },
                    )?;
                    let target_head_stride = max_positions.checked_mul(head_dim).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV seed target stride overflow",
                        },
                    )?;
                    let prefix_elems = prefix_length.checked_mul(head_dim).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV seed prefix size overflow",
                        },
                    )?;
                    for head in 0..kv_heads {
                        let source_start = head.checked_mul(source_head_stride).ok_or(
                            GgmlCpuGraphError::UnsupportedInputs {
                                reason: "batched resident KV seed source offset overflow",
                            },
                        )?;
                        let source_end = source_start.checked_add(prefix_elems).ok_or(
                            GgmlCpuGraphError::UnsupportedInputs {
                                reason: "batched resident KV seed source end overflow",
                            },
                        )?;
                        let target_start = plane_start
                            .checked_add(head.checked_mul(target_head_stride).ok_or(
                                GgmlCpuGraphError::UnsupportedInputs {
                                    reason: "batched resident KV seed target offset overflow",
                                },
                            )?)
                            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                                reason: "batched resident KV seed target offset overflow",
                            })?;
                        let target_end = target_start.checked_add(prefix_elems).ok_or(
                            GgmlCpuGraphError::UnsupportedInputs {
                                reason: "batched resident KV seed target end overflow",
                            },
                        )?;
                        key_planes[target_start..target_end]
                            .copy_from_slice(&keys[source_start..source_end]);
                        value_planes[target_start..target_end]
                            .copy_from_slice(&values[source_start..source_end]);
                    }
                }
                let layer = resident_kv_arena.layers[layer_index];
                // Default path: host f32 -> resident f16.
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
            GgmlKvElementType::Q8_0 => {
                let mut key_planes = vec![0_u8; tensor_nbytes];
                let mut value_planes = vec![0_u8; tensor_nbytes];
                for (sequence_index, sequence_layers) in layer_kv_by_sequence.iter().enumerate() {
                    let prefix_length = prefix_lengths[sequence_index];
                    let history = sequence_layers[layer_index]
                        .full_history_storage()
                        .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV q8 seed host cache storage invalid",
                        })?;
                    if history.head_dim != head_dim
                        || history.kv_heads != kv_heads
                        || history.max_positions > max_positions
                        || prefix_length > history.max_positions
                    {
                        return Err(GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV q8 seed host cache shape mismatch",
                        });
                    }
                    if history.written_positions != prefix_length {
                        return Err(GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV q8 seed written prefix mismatch",
                        });
                    }
                    let keys = history
                        .keys_q8
                        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV q8 seed missing q8 keys",
                        })?;
                    let values = history
                        .values_q8
                        .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV q8 seed missing q8 values",
                        })?;
                    let row_nbytes =
                        GgmlKvElementType::Q8_0.row_nbytes(head_dim).map_err(|_| {
                            GgmlCpuGraphError::UnsupportedInputs {
                                reason: "batched resident KV q8 seed row size overflow",
                            }
                        })?;
                    let host_plane_nbytes = GgmlKvElementType::Q8_0
                        .plane_nbytes(head_dim, history.max_positions, kv_heads)
                        .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV q8 seed host plane size overflow",
                        })?;
                    if keys.len() != host_plane_nbytes || values.len() != host_plane_nbytes {
                        return Err(GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV q8 seed host cache plane length mismatch",
                        });
                    }
                    let plane_start = sequence_index.checked_mul(plane_nbytes).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV q8 seed plane offset overflow",
                        },
                    )?;
                    let source_head_stride = history.max_positions.checked_mul(row_nbytes).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV q8 seed source stride overflow",
                        },
                    )?;
                    let target_head_stride = max_positions.checked_mul(row_nbytes).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV q8 seed target stride overflow",
                        },
                    )?;
                    let prefix_nbytes = prefix_length.checked_mul(row_nbytes).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV q8 seed prefix size overflow",
                        },
                    )?;
                    for head in 0..kv_heads {
                        let source_start = head.checked_mul(source_head_stride).ok_or(
                            GgmlCpuGraphError::UnsupportedInputs {
                                reason: "batched resident KV q8 seed source offset overflow",
                            },
                        )?;
                        let source_end = source_start.checked_add(prefix_nbytes).ok_or(
                            GgmlCpuGraphError::UnsupportedInputs {
                                reason: "batched resident KV q8 seed source end overflow",
                            },
                        )?;
                        let target_start = plane_start
                            .checked_add(head.checked_mul(target_head_stride).ok_or(
                                GgmlCpuGraphError::UnsupportedInputs {
                                    reason: "batched resident KV q8 seed target offset overflow",
                                },
                            )?)
                            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                                reason: "batched resident KV q8 seed target offset overflow",
                            })?;
                        let target_end = target_start.checked_add(prefix_nbytes).ok_or(
                            GgmlCpuGraphError::UnsupportedInputs {
                                reason: "batched resident KV q8 seed target end overflow",
                            },
                        )?;
                        key_planes[target_start..target_end]
                            .copy_from_slice(&keys[source_start..source_end]);
                        value_planes[target_start..target_end]
                            .copy_from_slice(&values[source_start..source_end]);
                    }
                }
                let layer = resident_kv_arena.layers[layer_index];
                // Direct packed q8_0 upload: no full f32 staging.
                resident_kv_arena.arena.set_bytes_slice(
                    layer.key,
                    &key_planes,
                    "qwen_llm_resident_kv_seed_key",
                )?;
                resident_kv_arena.arena.set_bytes_slice(
                    layer.value,
                    &value_planes,
                    "qwen_llm_resident_kv_seed_value",
                )?;
            }
            GgmlKvElementType::F16 => {
                return Err(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "batched resident KV seed rejects host f16 storage",
                });
            }
        }
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
    resident_element_type: GgmlKvElementType,
) -> Result<(), GgmlCpuGraphError> {
    if layer_kv.len() != resident_kv_arena.layers.len() {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "batched resident KV slot seed layer count mismatch",
        });
    }
    let host_type = layer_kv
        .first()
        .map(|cache| cache.element_type())
        .unwrap_or(GgmlKvElementType::F32);
    if layer_kv
        .iter()
        .any(|cache| cache.element_type() != host_type)
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "batched resident KV slot seed host element type mismatch",
        });
    }
    match (host_type, resident_element_type) {
        (GgmlKvElementType::F32, GgmlKvElementType::F16) => {
            let plane_elems = qwen_resident_kv_plane_elems(head_dim, max_positions, kv_heads)?;
            let plane_offset = slot_index.checked_mul(plane_elems).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "batched resident KV slot seed offset overflow",
                },
            )?;
            for (layer_index, cache) in layer_kv.iter().enumerate() {
                let mut key_plane = vec![0.0_f32; plane_elems];
                let mut value_plane = vec![0.0_f32; plane_elems];
                let history = cache.full_history_storage().map_err(|_| {
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV slot seed host cache storage invalid",
                    }
                })?;
                if history.head_dim != head_dim
                    || history.kv_heads != kv_heads
                    || history.max_positions > max_positions
                    || cache_position > history.max_positions
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
                let keys = history
                    .keys_f32
                    .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV slot seed missing f32 keys",
                    })?;
                let values = history
                    .values_f32
                    .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV slot seed missing f32 values",
                    })?;
                let host_plane_elems =
                    qwen_resident_kv_plane_elems(head_dim, history.max_positions, kv_heads)?;
                if keys.len() != host_plane_elems || values.len() != host_plane_elems {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV slot seed host cache plane length mismatch",
                    });
                }
                let source_head_stride = history.max_positions.checked_mul(head_dim).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV slot source stride overflow",
                    },
                )?;
                let target_head_stride = max_positions.checked_mul(head_dim).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV slot target stride overflow",
                    },
                )?;
                let prefix_elems = cache_position.checked_mul(head_dim).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV slot prefix size overflow",
                    },
                )?;
                for head in 0..kv_heads {
                    let source_start = head.checked_mul(source_head_stride).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV slot source offset overflow",
                        },
                    )?;
                    let source_end = source_start.checked_add(prefix_elems).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV slot source end overflow",
                        },
                    )?;
                    let target_start = head.checked_mul(target_head_stride).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV slot target offset overflow",
                        },
                    )?;
                    let target_end = target_start.checked_add(prefix_elems).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV slot target end overflow",
                        },
                    )?;
                    key_plane[target_start..target_end]
                        .copy_from_slice(&keys[source_start..source_end]);
                    value_plane[target_start..target_end]
                        .copy_from_slice(&values[source_start..source_end]);
                }
                let layer = resident_kv_arena.layers[layer_index];
                resident_kv_arena.arena.set_f16_bits_slice_with_offset(
                    layer.key,
                    plane_offset,
                    &f32_slice_to_f16_bits(&key_plane),
                    "qwen_llm_resident_kv_slot_seed_key",
                )?;
                resident_kv_arena.arena.set_f16_bits_slice_with_offset(
                    layer.value,
                    plane_offset,
                    &f32_slice_to_f16_bits(&value_plane),
                    "qwen_llm_resident_kv_slot_seed_value",
                )?;
            }
            Ok(())
        }
        (GgmlKvElementType::Q8_0, GgmlKvElementType::Q8_0) => {
            let plane_nbytes = GgmlKvElementType::Q8_0
                .plane_nbytes(head_dim, max_positions, kv_heads)
                .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                    reason: "batched resident KV q8 slot plane size overflow",
                })?;
            let plane_offset = slot_index.checked_mul(plane_nbytes).ok_or(
                GgmlCpuGraphError::UnsupportedInputs {
                    reason: "batched resident KV q8 slot seed offset overflow",
                },
            )?;
            for (layer_index, cache) in layer_kv.iter().enumerate() {
                let mut key_plane = vec![0_u8; plane_nbytes];
                let mut value_plane = vec![0_u8; plane_nbytes];
                let history = cache.full_history_storage().map_err(|_| {
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV q8 slot seed host cache storage invalid",
                    }
                })?;
                if history.head_dim != head_dim
                    || history.kv_heads != kv_heads
                    || history.max_positions > max_positions
                    || cache_position > history.max_positions
                {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV q8 slot seed host cache shape mismatch",
                    });
                }
                if history.written_positions != cache_position {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV q8 slot seed written prefix mismatch",
                    });
                }
                let keys = history
                    .keys_q8
                    .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV q8 slot seed missing q8 keys",
                    })?;
                let values = history
                    .values_q8
                    .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV q8 slot seed missing q8 values",
                    })?;
                let row_nbytes = GgmlKvElementType::Q8_0.row_nbytes(head_dim).map_err(|_| {
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV q8 slot row size overflow",
                    }
                })?;
                let host_plane_nbytes = GgmlKvElementType::Q8_0
                    .plane_nbytes(head_dim, history.max_positions, kv_heads)
                    .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV q8 slot host plane size overflow",
                    })?;
                if keys.len() != host_plane_nbytes || values.len() != host_plane_nbytes {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV q8 slot seed host cache plane length mismatch",
                    });
                }
                let source_head_stride = history.max_positions.checked_mul(row_nbytes).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV q8 slot source stride overflow",
                    },
                )?;
                let target_head_stride = max_positions.checked_mul(row_nbytes).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV q8 slot target stride overflow",
                    },
                )?;
                let prefix_nbytes = cache_position.checked_mul(row_nbytes).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "batched resident KV q8 slot prefix size overflow",
                    },
                )?;
                for head in 0..kv_heads {
                    let source_start = head.checked_mul(source_head_stride).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV q8 slot source offset overflow",
                        },
                    )?;
                    let source_end = source_start.checked_add(prefix_nbytes).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV q8 slot source end overflow",
                        },
                    )?;
                    let target_start = head.checked_mul(target_head_stride).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV q8 slot target offset overflow",
                        },
                    )?;
                    let target_end = target_start.checked_add(prefix_nbytes).ok_or(
                        GgmlCpuGraphError::UnsupportedInputs {
                            reason: "batched resident KV q8 slot target end overflow",
                        },
                    )?;
                    key_plane[target_start..target_end]
                        .copy_from_slice(&keys[source_start..source_end]);
                    value_plane[target_start..target_end]
                        .copy_from_slice(&values[source_start..source_end]);
                }
                let layer = resident_kv_arena.layers[layer_index];
                resident_kv_arena.arena.set_bytes_slice_with_offset(
                    layer.key,
                    plane_offset,
                    &key_plane,
                    "qwen_llm_resident_kv_slot_seed_key",
                )?;
                resident_kv_arena.arena.set_bytes_slice_with_offset(
                    layer.value,
                    plane_offset,
                    &value_plane,
                    "qwen_llm_resident_kv_slot_seed_value",
                )?;
            }
            Ok(())
        }
        _ => Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "batched resident KV slot seed host/resident element type pair unsupported",
        }),
    }
}

#[allow(dead_code)]
fn zero_qwen_batched_resident_kv_slot(
    resident_kv_arena: &mut LlmResidentKvArena,
    head_dim: usize,
    max_positions: usize,
    kv_heads: usize,
    slot_index: usize,
    resident_element_type: GgmlKvElementType,
) -> Result<(), GgmlCpuGraphError> {
    let plane_nbytes = resident_element_type
        .plane_nbytes(head_dim, max_positions, kv_heads)
        .map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
            reason: "batched resident KV slot zero plane size overflow",
        })?;
    let plane_offset =
        slot_index
            .checked_mul(plane_nbytes)
            .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "batched resident KV slot zero offset overflow",
            })?;
    let zeros = vec![0_u8; plane_nbytes];
    for layer in &resident_kv_arena.layers {
        resident_kv_arena.arena.set_bytes_slice_with_offset(
            layer.key,
            plane_offset,
            &zeros,
            "qwen_llm_resident_kv_slot_zero_key",
        )?;
        resident_kv_arena.arena.set_bytes_slice_with_offset(
            layer.value,
            plane_offset,
            &zeros,
            "qwen_llm_resident_kv_slot_zero_value",
        )?;
    }
    Ok(())
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
pub(crate) fn load_qwen3_llm_layer_attention_projection(
    reader: &GgufTensorDataReader,
    metadata: Qwen3AsrExecutionMetadata,
    layer_index: usize,
) -> Result<Qwen3AsrLlmLayerAttentionProjection, Qwen3AsrLlmTransformerError> {
    Ok(Qwen3AsrLlmLayerAttentionProjection::Generic(
        load_qwen3_llm_layer_attention_projection_generic(reader, metadata, layer_index, false)?,
    ))
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

fn quote_qwen_decoder_plan_names(
    layer_count: usize,
    mut names_for_layer: impl FnMut(usize) -> Result<QwenFamilyLlmLayerTensorNames, String>,
) -> Result<u64, String> {
    let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
    bytes.add_usize(
        layer_count
            .checked_mul(std::mem::size_of::<QwenWholeDecoderLayerPlan>())
            .ok_or_else(|| "qwen-family decoder-plan layer quote overflowed".to_string())?,
        "qwen-family whole-decoder layer plans",
    )?;
    for layer_index in 0..layer_count {
        let names = names_for_layer(layer_index)?;
        for name in [
            Some(names.attn_norm_name),
            Some(names.attn_q_name),
            Some(names.attn_k_name),
            Some(names.attn_v_name),
            Some(names.attn_output_name),
            names.q_norm_name,
            names.k_norm_name,
            names.q_bias_name,
            names.k_bias_name,
            names.v_bias_name,
            Some(names.ffn_norm_name),
            Some(names.ffn_gate_name),
            Some(names.ffn_up_name),
            Some(names.ffn_down_name),
        ]
        .into_iter()
        .flatten()
        {
            bytes.add_usize(name.len(), "qwen-family decoder-plan tensor name")?;
        }
    }
    Ok(bytes.finish())
}

/// Exact host-memory quote for the shared direct Qwen decoder constructor.
///
/// The construction topology is fixed by the shared Module: metadata plan,
/// logits head, token embedding, then the compiled graph. Family adapters pass
/// only the already-bound contract, so tensor names, dimensions, tied-head
/// policy, and layer count cannot drift from admission or materialization.
pub(crate) fn quoted_qwen_decoder_system_memory_bytes(
    reader: &GgufTensorDataReader,
    contract: &QwenDecoderContract,
    backend: GgmlCpuGraphBackend,
) -> Result<(u64, u64), String> {
    let geometry = contract.geometry();
    let tail = contract.tail();
    let graph_retained = Qwen3AsrLlmWholeDecoderGraphExecutor::quoted_retained_system_memory_bytes(
        geometry.n_layers,
    )?;
    let plan_transient = QwenWholeDecoderPlan::quoted_retained_system_memory_bytes(contract)?;
    let output_weight = tail.output_weight.unwrap_or(tail.token_embd);
    let (logits_peak, logits_retained) =
        Qwen3AsrLlmLogitsHead::quoted_system_memory_bytes_from_reader(
            reader,
            output_weight,
            geometry.d_model,
            geometry.vocab_size,
            backend,
        )?;
    let (embedding_peak, embedding_retained) =
        MappedTokenEmbeddingTable::quoted_system_memory_bytes_from_reader(
            reader,
            tail.token_embd,
            geometry.d_model,
            geometry.vocab_size,
        )?;

    let retained = graph_retained
        .checked_add(logits_retained)
        .and_then(|bytes| bytes.checked_add(embedding_retained))
        .ok_or_else(|| "qwen decoder retained quote overflowed".to_string())?;
    let logits_phase = plan_transient
        .checked_add(logits_peak)
        .ok_or_else(|| "qwen decoder logits construction quote overflowed".to_string())?;
    let embedding_phase = plan_transient
        .checked_add(logits_retained)
        .and_then(|bytes| bytes.checked_add(embedding_peak))
        .ok_or_else(|| "qwen decoder embedding construction quote overflowed".to_string())?;
    let graph_phase = plan_transient
        .checked_add(logits_retained)
        .and_then(|bytes| bytes.checked_add(embedding_retained))
        .and_then(|bytes| bytes.checked_add(graph_retained))
        .ok_or_else(|| "qwen decoder graph construction quote overflowed".to_string())?;
    Ok((logits_phase.max(embedding_phase).max(graph_phase), retained))
}

/// Add the retained host representation of a bound Qwen decoder to a prepared
/// runtime quote. This mirrors
/// [`super::decoder_tail::load_qwen_decoder_tail_from_contract`] without
/// materializing tensors and works for both tied and untied output heads.
pub(crate) fn add_qwen_decoder_prepared_runtime_quote(
    quote: &mut PreparedRuntimeQuoteBuilder,
    context: PreparedRuntimeQuoteContext<'_>,
    contract: &QwenDecoderContract,
) -> Result<(), SystemMemoryOwnerError> {
    let geometry = contract.geometry();
    let tail = contract.tail();
    let plan_bytes =
        QwenWholeDecoderPlan::quoted_retained_system_memory_bytes(contract).map_err(|reason| {
            SystemMemoryOwnerError::capacity_failure("prepared_runtime_quote", reason)
        })?;
    quote.add_structural_bytes(plan_bytes, "qwen decoder metadata plan")?;

    let embedding = context.tensor_index.get(tail.token_embd).ok_or_else(|| {
        SystemMemoryOwnerError::capacity_failure(
            "prepared_runtime_quote",
            format!("required tensor '{}' is missing", tail.token_embd),
        )
    })?;
    let canonical_dims = [geometry.d_model as u64, geometry.vocab_size as u64];
    if embedding.ggml_type == 0 || embedding.ggml_type == 1 || embedding.dims == canonical_dims {
        quote.add_owned_tensor_payload_metadata(context.tensor_index, tail.token_embd)?;
    } else {
        quote.add_tensor_f32(context.tensor_index, tail.token_embd)?;
    }

    quote.add_tensor_f32(context.tensor_index, tail.output_norm)?;
    let output_weight = tail.output_weight.unwrap_or(tail.token_embd);
    let output = context.tensor_index.get(output_weight).ok_or_else(|| {
        SystemMemoryOwnerError::capacity_failure(
            "prepared_runtime_quote",
            format!("required tensor '{output_weight}' is missing"),
        )
    })?;
    if super::logits_head::logits_head_ggml_enabled(context.backend)
        && output.dims == canonical_dims
    {
        quote.add_owned_tensor_payload_metadata(context.tensor_index, output_weight)?;
        quote.add_owned_elements::<usize>(
            u64::try_from(output.dims.len()).map_err(|_| {
                SystemMemoryOwnerError::capacity_failure(
                    "prepared_runtime_quote",
                    "qwen logits rank does not fit u64",
                )
            })?,
            "qwen logits raw dims",
        )?;
    } else {
        quote.add_tensor_f32(context.tensor_index, output_weight)?;
    }
    Ok(())
}

impl QwenWholeDecoderPlan {
    #[cfg(test)]
    pub(crate) fn quoted_retained_system_memory_bytes_for_qwen3_asr(
        layer_count: usize,
    ) -> Result<u64, String> {
        quote_qwen_decoder_plan_names(layer_count, |layer_index| {
            let names = llm_layer_tensor_names(layer_index);
            Ok(QwenFamilyLlmLayerTensorNames {
                attn_norm_name: names.attn_norm_weight,
                attn_q_name: names.attn_q_weight,
                attn_k_name: names.attn_k_weight,
                attn_v_name: names.attn_v_weight,
                attn_output_name: names.attn_output_weight,
                q_norm_name: Some(names.attn_q_norm_weight),
                k_norm_name: Some(names.attn_k_norm_weight),
                q_bias_name: None,
                k_bias_name: None,
                v_bias_name: None,
                ffn_norm_name: names.ffn_norm_weight,
                ffn_gate_name: names.ffn_gate_weight,
                ffn_up_name: names.ffn_up_weight,
                ffn_down_name: names.ffn_down_weight,
            })
        })
    }

    /// Count-only retained heap quote for any Qwen-shaped family decoder
    /// plan. The family supplies exactly the same tensor-name topology used by
    /// [`Self::for_qwen_family`], so planning and materialization cannot drift
    /// merely because two families prefix their GGUF tensor names differently.
    pub(crate) fn quoted_retained_system_memory_bytes(
        contract: &QwenDecoderContract,
    ) -> Result<u64, String> {
        let layer_count = contract.geometry().n_layers;
        quote_qwen_decoder_plan_names(layer_count, |layer_index| {
            contract
                .layer_projection(layer_index)
                .map(|(names, _)| names)
        })
    }

    pub(crate) fn for_qwen3_asr(
        reader: &GgufTensorDataReader,
        metadata: Qwen3AsrExecutionMetadata,
    ) -> Result<Self, Qwen3AsrLlmTransformerError> {
        let contract =
            qwen3_asr_decoder_contract(reader.tensor_index(), metadata).map_err(|error| {
                Qwen3AsrLlmTransformerError::InvalidTensorShape {
                    tensor_name: "<qwen3-asr decoder contract>".to_string(),
                    shape: "[]".to_string(),
                    reason: error.to_string(),
                }
            })?;
        Self::for_qwen_family(reader, &contract)
    }

    /// Build a whole-decoder plan from a **bound** [`QwenDecoderContract`].
    ///
    /// Production callers must bind geometry+profile once and pass that value
    /// here — not reassemble geometry/options/names at the call site. Shape
    /// expectations come only from the contract's descriptor expansion. The pack
    /// supplies tensor bytes/types/offsets; it cannot invent a second geometry.
    /// Transposed `[out, in]` projections fail closed.
    pub(crate) fn for_qwen_family(
        reader: &GgufTensorDataReader,
        contract: &QwenDecoderContract,
    ) -> Result<Self, Qwen3AsrLlmTransformerError> {
        let geometry = contract.geometry();
        let mut layers = Vec::with_capacity(geometry.n_layers);
        for layer_index in 0..geometry.n_layers {
            let (names, descriptors) =
                contract.layer_projection(layer_index).map_err(|reason| {
                    Qwen3AsrLlmTransformerError::InvalidTensorShape {
                        tensor_name: "<decoder layer contract>".to_string(),
                        shape: format!("{geometry:?}"),
                        reason,
                    }
                })?;
            layers.push(plan_qwen_family_layer(
                reader,
                names,
                geometry,
                &descriptors,
            )?);
        }
        Ok(Self { layers })
    }

    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        use crate::models::system_memory_owner::SystemMemoryCapacity;

        let mut bytes = SystemMemoryCapacity::default();
        bytes.add_vec(&self.layers, "qwen whole-decoder layer plans")?;
        for layer in &self.layers {
            for vector in [
                Some(&layer.attn_norm),
                layer.q_bias.as_ref(),
                layer.k_bias.as_ref(),
                layer.v_bias.as_ref(),
                layer.q_norm.as_ref(),
                layer.k_norm.as_ref(),
                Some(&layer.ffn_norm),
            ]
            .into_iter()
            .flatten()
            {
                bytes.add_string(&vector.tensor_name, "qwen vector plan tensor name")?;
            }
            for projection in [
                &layer.q,
                &layer.k,
                &layer.v,
                &layer.output,
                &layer.gate,
                &layer.up,
                &layer.down,
            ] {
                bytes.add_string(&projection.tensor_name, "qwen projection plan tensor name")?;
            }
        }
        Ok(bytes.finish())
    }

    fn validate_materialization_reader(
        &self,
        reader: &GgufTensorDataReader,
    ) -> Result<(), GgmlCpuGraphError> {
        for layer in &self.layers {
            for vector in [
                Some(&layer.attn_norm),
                layer.q_bias.as_ref(),
                layer.k_bias.as_ref(),
                layer.v_bias.as_ref(),
                layer.q_norm.as_ref(),
                layer.k_norm.as_ref(),
                Some(&layer.ffn_norm),
            ]
            .into_iter()
            .flatten()
            {
                let tensor = reader.tensor_index().get(&vector.tensor_name).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "planned qwen vector is missing from materialization source",
                    },
                )?;
                if tensor.dims.as_slice() != [vector.len as u64]
                    || tensor.ggml_type != vector.ggml_type
                    || tensor.size_bytes != vector.size_bytes
                    || tensor.offset_bytes != vector.offset_bytes
                {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "planned qwen vector metadata changed before materialization",
                    });
                }
            }
            for projection in [
                &layer.q,
                &layer.k,
                &layer.v,
                &layer.output,
                &layer.gate,
                &layer.up,
                &layer.down,
            ] {
                let tensor = reader.tensor_index().get(&projection.tensor_name).ok_or(
                    GgmlCpuGraphError::UnsupportedInputs {
                        reason: "planned qwen projection is missing from materialization source",
                    },
                )?;
                let planned_dims = projection
                    .storage_dims
                    .map(|dimension| u64::try_from(dimension).unwrap_or(u64::MAX));
                if tensor.dims.as_slice() != planned_dims
                    || tensor.ggml_type != projection.ggml_type
                    || tensor.size_bytes != projection.size_bytes
                    || tensor.offset_bytes != projection.offset_bytes
                {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "planned qwen projection metadata changed before materialization",
                    });
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn retained_weight_payload_bytes(&self) -> usize {
        // The representation intentionally has no byte/f32 payload field.
        0
    }
}

fn plan_qwen_family_layer(
    reader: &GgufTensorDataReader,
    names: QwenFamilyLlmLayerTensorNames,
    geometry: QwenDecoderContractGeometry,
    descriptors: &[TensorBindingDescriptor],
) -> Result<QwenWholeDecoderLayerPlan, Qwen3AsrLlmTransformerError> {
    let mut vectors = std::collections::HashMap::<String, VectorWeightPlan>::new();
    let mut projections = std::collections::HashMap::<String, ProjectionWeightPlan>::new();
    for descriptor in descriptors {
        match plan_weight_from_contract_descriptor(reader, descriptor)? {
            PlannedContractWeight::Vector(plan) => {
                vectors.insert(plan.tensor_name.clone(), plan);
            }
            PlannedContractWeight::Projection(plan) => {
                projections.insert(plan.tensor_name.clone(), plan);
            }
        }
    }

    let attn_norm = take_planned_vector(&mut vectors, &names.attn_norm_name)?;
    let q = take_planned_projection(&mut projections, &names.attn_q_name)?;
    let k = take_planned_projection(&mut projections, &names.attn_k_name)?;
    let v = take_planned_projection(&mut projections, &names.attn_v_name)?;
    let output = take_planned_projection(&mut projections, &names.attn_output_name)?;
    let ffn_norm = take_planned_vector(&mut vectors, &names.ffn_norm_name)?;
    let gate = take_planned_projection(&mut projections, &names.ffn_gate_name)?;
    let up = take_planned_projection(&mut projections, &names.ffn_up_name)?;
    let down = take_planned_projection(&mut projections, &names.ffn_down_name)?;

    let (q_bias, k_bias, v_bias) = if names.q_bias_name.is_some() {
        let q_bias_name = names.q_bias_name.as_deref().ok_or_else(|| {
            Qwen3AsrLlmTransformerError::InvalidTensorShape {
                tensor_name: "<q_bias>".to_string(),
                shape: "[]".to_string(),
                reason: "qwen decoder options.qkv_bias requires q_bias_name".to_string(),
            }
        })?;
        let k_bias_name = names.k_bias_name.as_deref().ok_or_else(|| {
            Qwen3AsrLlmTransformerError::InvalidTensorShape {
                tensor_name: "<k_bias>".to_string(),
                shape: "[]".to_string(),
                reason: "qwen decoder options.qkv_bias requires k_bias_name".to_string(),
            }
        })?;
        let v_bias_name = names.v_bias_name.as_deref().ok_or_else(|| {
            Qwen3AsrLlmTransformerError::InvalidTensorShape {
                tensor_name: "<v_bias>".to_string(),
                shape: "[]".to_string(),
                reason: "qwen decoder options.qkv_bias requires v_bias_name".to_string(),
            }
        })?;
        (
            Some(take_planned_vector(&mut vectors, q_bias_name)?),
            Some(take_planned_vector(&mut vectors, k_bias_name)?),
            Some(take_planned_vector(&mut vectors, v_bias_name)?),
        )
    } else {
        (None, None, None)
    };

    let (q_norm, k_norm) = if names.q_norm_name.is_some() {
        let q_norm_name = names.q_norm_name.as_deref().ok_or_else(|| {
            Qwen3AsrLlmTransformerError::InvalidTensorShape {
                tensor_name: "<q_norm>".to_string(),
                shape: "[]".to_string(),
                reason: "qwen decoder options.qk_norm requires q_norm_name".to_string(),
            }
        })?;
        let k_norm_name = names.k_norm_name.as_deref().ok_or_else(|| {
            Qwen3AsrLlmTransformerError::InvalidTensorShape {
                tensor_name: "<k_norm>".to_string(),
                shape: "[]".to_string(),
                reason: "qwen decoder options.qk_norm requires k_norm_name".to_string(),
            }
        })?;
        (
            Some(take_planned_vector(&mut vectors, q_norm_name)?),
            Some(take_planned_vector(&mut vectors, k_norm_name)?),
        )
    } else {
        (None, None)
    };

    if !vectors.is_empty() || !projections.is_empty() {
        let leftover: Vec<_> = vectors.keys().chain(projections.keys()).cloned().collect();
        return Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: leftover.join(","),
            shape: "[]".to_string(),
            reason: "contract descriptors projected weights that the layer plan could not bind"
                .to_string(),
        });
    }

    Ok(QwenWholeDecoderLayerPlan {
        d_model: geometry.d_model,
        head_dim: geometry.head_dim,
        attn_norm,
        q,
        k,
        v,
        q_bias,
        k_bias,
        v_bias,
        output,
        q_norm,
        k_norm,
        ffn_norm,
        gate,
        up,
        down,
    })
}

fn take_planned_vector(
    vectors: &mut std::collections::HashMap<String, VectorWeightPlan>,
    name: &str,
) -> Result<VectorWeightPlan, Qwen3AsrLlmTransformerError> {
    vectors
        .remove(name)
        .ok_or_else(|| Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: name.to_string(),
            shape: "[]".to_string(),
            reason: "contract descriptor did not project a vector weight for this name".to_string(),
        })
}

fn take_planned_projection(
    projections: &mut std::collections::HashMap<String, ProjectionWeightPlan>,
    name: &str,
) -> Result<ProjectionWeightPlan, Qwen3AsrLlmTransformerError> {
    projections
        .remove(name)
        .ok_or_else(|| Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: name.to_string(),
            shape: "[]".to_string(),
            reason: "contract descriptor did not project a projection weight for this name"
                .to_string(),
        })
}

enum PlannedContractWeight {
    Vector(VectorWeightPlan),
    Projection(ProjectionWeightPlan),
}

/// Project one admission descriptor into a loader weight plan. Shape authority
/// is the descriptor requirement (same expansion admission validates against).
fn plan_weight_from_contract_descriptor(
    reader: &GgufTensorDataReader,
    descriptor: &TensorBindingDescriptor,
) -> Result<PlannedContractWeight, Qwen3AsrLlmTransformerError> {
    let tensor = required_tensor_metadata(reader, &descriptor.tensor_name)?;
    match &descriptor.requirement {
        TensorBindingDescriptorRequirement::VectorLen(expected_len) => {
            if tensor.dims.as_slice() != [*expected_len as u64] {
                return Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
                    tensor_name: descriptor.tensor_name.clone(),
                    shape: render_shape(&tensor.dims),
                    reason: format!(
                        "contract requires vector len {expected_len} ({})",
                        descriptor.reason
                    ),
                });
            }
            Ok(PlannedContractWeight::Vector(VectorWeightPlan {
                tensor_name: descriptor.tensor_name.clone(),
                len: *expected_len,
                ggml_type: tensor.ggml_type,
                size_bytes: tensor.size_bytes,
                offset_bytes: tensor.offset_bytes,
            }))
        }
        TensorBindingDescriptorRequirement::ExactDims(expected) if expected.len() == 2 => {
            let input_width = expected[0];
            let output_width = expected[1];
            let canonical = [input_width as u64, output_width as u64];
            let transposed = [output_width as u64, input_width as u64];
            if tensor.dims.as_slice() == canonical {
                Ok(PlannedContractWeight::Projection(
                    projection_plan_from_metadata(
                        descriptor.tensor_name.clone(),
                        tensor,
                        input_width,
                        output_width,
                        DenseProjectionLayout::OutputByInput,
                    )?,
                ))
            } else if tensor.dims.as_slice() == transposed {
                Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
                    tensor_name: descriptor.tensor_name.clone(),
                    shape: render_shape(&tensor.dims),
                    reason: format!(
                        "qwen-family projection weights must use the ggml [in, out] dim order \
                         (contract ExactDims {:?}); this pack stores them as [out, in], which \
                         indicates it was built by an older importer - re-pack from source with \
                         the current build ({})",
                        expected, descriptor.reason
                    ),
                })
            } else {
                Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
                    tensor_name: descriptor.tensor_name.clone(),
                    shape: render_shape(&tensor.dims),
                    reason: format!(
                        "contract ExactDims {:?} not matched ({})",
                        expected, descriptor.reason
                    ),
                })
            }
        }
        other => Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: descriptor.tensor_name.clone(),
            shape: render_shape(&tensor.dims),
            reason: format!(
                "qwen decoder contract descriptor uses unsupported requirement {other:?}"
            ),
        }),
    }
}

fn required_tensor_metadata<'a>(
    reader: &'a GgufTensorDataReader,
    tensor_name: &str,
) -> Result<&'a crate::GgufTensorMetadata, Qwen3AsrLlmTransformerError> {
    reader.tensor_index().get(tensor_name).ok_or_else(|| {
        Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: tensor_name.to_string(),
            shape: "[]".to_string(),
            reason: "tensor is missing from GGUF tensor index".to_string(),
        }
    })
}

#[cfg(test)]
fn plan_projection_weight_for_input(
    reader: &GgufTensorDataReader,
    tensor_name: String,
    input_width: usize,
) -> Result<ProjectionWeightPlan, Qwen3AsrLlmTransformerError> {
    let tensor = required_tensor_metadata(reader, &tensor_name)?;
    let (input_width, output_width, layout) =
        parse_projection_shape_for_input(&tensor_name, &tensor.dims, input_width)?;
    projection_plan_from_metadata(tensor_name, tensor, input_width, output_width, layout)
}

#[cfg(test)]
fn plan_projection_weight_with_layout(
    reader: &GgufTensorDataReader,
    tensor_name: String,
    input_width: usize,
    output_width: usize,
    layout: DenseProjectionLayout,
) -> Result<ProjectionWeightPlan, Qwen3AsrLlmTransformerError> {
    let tensor = required_tensor_metadata(reader, &tensor_name)?;
    let expected = match layout {
        DenseProjectionLayout::OutputByInput => [input_width as u64, output_width as u64],
        DenseProjectionLayout::InputByOutput => [output_width as u64, input_width as u64],
    };
    if tensor.dims.as_slice() != expected {
        return Err(Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name,
            shape: render_shape(&tensor.dims),
            reason: format!("expected {expected:?} for the layer's resolved projection layout"),
        });
    }
    projection_plan_from_metadata(tensor_name, tensor, input_width, output_width, layout)
}

fn projection_plan_from_metadata(
    tensor_name: String,
    tensor: &crate::GgufTensorMetadata,
    input_width: usize,
    output_width: usize,
    layout: DenseProjectionLayout,
) -> Result<ProjectionWeightPlan, Qwen3AsrLlmTransformerError> {
    let dim0 = usize::try_from(tensor.dims[0]).map_err(|_| {
        Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: tensor_name.clone(),
            shape: render_shape(&tensor.dims),
            reason: "dimension 0 does not fit usize".to_string(),
        }
    })?;
    let dim1 = usize::try_from(tensor.dims[1]).map_err(|_| {
        Qwen3AsrLlmTransformerError::InvalidTensorShape {
            tensor_name: tensor_name.clone(),
            shape: render_shape(&tensor.dims),
            reason: "dimension 1 does not fit usize".to_string(),
        }
    })?;
    Ok(ProjectionWeightPlan {
        tensor_name,
        input_width,
        output_width,
        storage_dims: [dim0, dim1],
        ggml_type: tensor.ggml_type,
        size_bytes: tensor.size_bytes,
        offset_bytes: tensor.offset_bytes,
        layout,
    })
}

/// Load one decoder-only LLM layer's projections from `reader`, parameterized
/// over the two axes that differ between Qwen2 and Qwen3 (QK-norm,
/// attention bias) via `names`' `Option` fields, rather than hard-coding
/// either family's shape. Shared by qwen3-asr
/// (`load_qwen3_llm_layer_attention_projection_generic`, always QK-norm, never
/// bias) and firered-llm (always bias, never QK-norm -- see
/// `models::firered_llm::llm_transformer`).
#[cfg(test)]
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
#[cfg(test)]
fn dropped_projection_payload(mut weight: DenseProjectionWeight) -> DenseProjectionWeight {
    // Only native [in,out] weights (raw_ggml present) are bound zero-copy — drop
    // their host bytes. f32-fallback weights KEEP their `values`: the arena path
    // is their only route (the loaded path can't fix their on-disk orientation).
    if let Some(raw) = weight.raw_ggml.as_mut() {
        raw.bytes = Vec::new();
    }
    weight
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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
#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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
#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

/// Flash everywhere it is numerically trusted: single/double-query steps
/// (VEC kernel), short KV spans (single K-tile), and CPU/Metal at every
/// width. Discrete GPU (`GgmlCpuGraphBackend::Gpu`: CUDA/HIP/Vulkan) is
/// fail-closed: the flash MMA/TILE kernel mis-handles per-query causal
/// mask + GQA when `n_query > 2` and `n_kv > 32`, so that combination
/// always swaps to `llm_naive_masked_attention`. Do not re-enable wide
/// flash on discrete GPU without a per-backend golden that covers
/// `n_query > 2 && n_kv > 32` with native GQA.
fn qwen_llm_prefill_uses_flash_attention_for_backend(
    backend: GgmlCpuGraphBackend,
    token_count: usize,
    kv_span: usize,
) -> bool {
    if token_count <= QWEN3_LLM_FLASH_SAFE_PREFILL_QUERY_TOKENS
        || kv_span <= QWEN3_LLM_FLASH_SAFE_PREFILL_MAX_KV_TOKENS
    {
        return true;
    }
    // Metal + CPU: flash trusted at every validated width.
    // Discrete GPU lane: non-flash for the wide multi-query / long-KV case.
    !matches!(backend, GgmlCpuGraphBackend::Gpu)
}

fn qwen_llm_safe_gpu_prefill_query_tokens_for_backend(
    backend_capabilities: crate::ggml_runtime::GgmlBackendCapabilities,
    token_count: usize,
) -> usize {
    // Discrete GPU backends share the non-flash-backed wide host-cache chunk
    // policy. Flash MMA/TILE is fail-closed for `n_query > 2 && n_kv > 32` on
    // the whole `GgmlCpuGraphBackend::Gpu` lane (see
    // `llm_prefill_uses_flash_attention`), so CUDA/HIP/Vulkan can all use the
    // same width-safe chunk without re-entering the historical serial path.
    // Metal keeps flash at every width and is selected by backend kind, not
    // by this name helper's discrete-GPU branch.
    if backend_capabilities.is_known_discrete_gpu() {
        if token_count <= QWEN3_LLM_FLASH_SAFE_PREFILL_MAX_KV_TOKENS {
            return QWEN3_LLM_DISCRETE_GPU_SHORT_PREFILL_QUERY_TOKENS;
        }
        return QWEN3_LLM_DISCRETE_GPU_NONFLASH_PREFILL_QUERY_TOKENS;
    }
    QWEN3_LLM_GPU_SAFE_PREFILL_QUERY_TOKENS
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
    use std::{collections::BTreeMap, path::PathBuf};

    use crate::ggml_runtime::{GGML_TYPE_F32, GgmlCpuGraphConfig, GgmlCpuGraphRunner};
    use crate::models::qwen::runtime_contract::parse_qwen3_execution_metadata;
    use crate::testing::{
        TinyGgufFixtureSpec, with_forced_cpu_backend_for_test, write_tiny_gguf_runtime_source,
    };
    use crate::{read_gguf_metadata_from_runtime_source, validate_ggml_runtime_source_path};

    const QWEN_PREFILL_REAL_PACK_ENV: &str = "OPENASR_QWEN_PREFILL_REAL_PACK";
    const QWEN_PREFILL_TOKENS_ENV: &str = "OPENASR_QWEN_PREFILL_TOKENS";
    const QWEN_PREFILL_CHUNK_TOKENS_ENV: &str = "OPENASR_QWEN_PREFILL_CHUNK_TOKENS";

    fn quote_test_layer_names(layer: usize) -> QwenFamilyLlmLayerTensorNames {
        let prefix = format!("quote.blk.{layer}");
        QwenFamilyLlmLayerTensorNames {
            attn_norm_name: format!("{prefix}.attn_norm.weight"),
            attn_q_name: format!("{prefix}.attn_q.weight"),
            attn_k_name: format!("{prefix}.attn_k.weight"),
            attn_v_name: format!("{prefix}.attn_v.weight"),
            attn_output_name: format!("{prefix}.attn_output.weight"),
            q_norm_name: Some(format!("{prefix}.attn_q_norm.weight")),
            k_norm_name: Some(format!("{prefix}.attn_k_norm.weight")),
            q_bias_name: None,
            k_bias_name: None,
            v_bias_name: None,
            ffn_norm_name: format!("{prefix}.ffn_norm.weight"),
            ffn_gate_name: format!("{prefix}.ffn_gate.weight"),
            ffn_up_name: format!("{prefix}.ffn_up.weight"),
            ffn_down_name: format!("{prefix}.ffn_down.weight"),
        }
    }

    /// The shared host quote is not a paper estimate: it must equal the
    /// retained Rust containers produced by the same bound contract's real
    /// planner, tail loader, and graph compiler. Construction peak is an upper
    /// bound over their actual phase topology and therefore cannot be below
    /// the exact retained value.
    #[test]
    fn bound_decoder_quote_matches_real_materialization_retained_bytes() {
        use crate::models::qwen::{
            QwenDecoderContractGeometry, QwenDecoderTailTensorNames, QwenDecoderVariant,
            QwenFamilyDecoderProfile, load_qwen_decoder_tail_from_contract,
        };
        use crate::models::tensor_binding::project_fixture_tensors;

        let geometry = QwenDecoderContractGeometry {
            n_layers: 2,
            d_model: 16,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 32,
            vocab_size: 64,
        };
        let contract = QwenDecoderContract::bind(
            geometry,
            QwenFamilyDecoderProfile::new(
                QwenDecoderVariant::Qwen3,
                quote_test_layer_names,
                QwenDecoderTailTensorNames {
                    output_norm: "quote.output_norm.weight",
                    output_weight: Some("quote.output.weight"),
                    token_embd: "quote.token_embd.weight",
                },
            ),
        )
        .expect("bind quote-test contract");
        let descriptors = contract
            .runtime_tensor_descriptors()
            .expect("quote-test descriptors");
        let mut spec = TinyGgufFixtureSpec::new(BTreeMap::new());
        for (name, dims) in project_fixture_tensors(&descriptors) {
            spec = spec.with_tensor_shape(name, dims);
        }
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("bound-decoder-quote.oasr");
        write_tiny_gguf_runtime_source(&path, &spec).expect("write quote fixture");
        let preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index(&path)
                .expect("preflight quote fixture");
        let reader = GgufTensorDataReader::from_path(&path).expect("quote reader");

        let (quoted_peak, quoted_retained) =
            quoted_qwen_decoder_system_memory_bytes(&reader, &contract, GgmlCpuGraphBackend::Cpu)
                .expect("quote decoder");
        let plan = QwenWholeDecoderPlan::for_qwen_family(&reader, &contract).expect("plan decoder");
        let tail = load_qwen_decoder_tail_from_contract(
            &reader,
            &contract,
            DEFAULT_RMS_NORM_EPSILON,
            GgmlCpuGraphBackend::Cpu,
        )
        .expect("load tail");
        let graph = compile_qwen_whole_decoder_graph_from_prepared_plan(
            QwenPreparedDecoderGraphCompileRequest {
                plan: &plan,
                preflight: &preflight,
                rms_norm_epsilon: DEFAULT_RMS_NORM_EPSILON,
                fused_logits_head: tail.logits_head.fused_top1_spec(),
                token_embedding: None,
                backend: GgmlCpuGraphBackend::Cpu,
            },
        )
        .expect("compile graph");
        let actual_retained = graph
            .retained_system_memory_bytes()
            .and_then(|bytes| {
                tail.logits_head
                    .retained_system_memory_bytes()
                    .and_then(|logits| bytes.checked_add(logits).ok_or("retained overflow".into()))
            })
            .and_then(|bytes| {
                tail.token_embedding
                    .retained_system_memory_bytes()
                    .and_then(|embedding| {
                        bytes
                            .checked_add(embedding)
                            .ok_or_else(|| "retained overflow".to_string())
                    })
            })
            .expect("measure retained decoder");
        assert_eq!(quoted_retained, actual_retained);
        assert!(quoted_peak >= quoted_retained);
    }

    fn metadata_only_decoder_fixture(
        k_as_f16: bool,
    ) -> (
        tempfile::TempDir,
        crate::GgmlRuntimeSource,
        Qwen3AsrExecutionMetadata,
    ) {
        let metadata = Qwen3AsrExecutionMetadata {
            sample_rate_hz: 16_000,
            n_mels: 8,
            n_fft: 400,
            win_length: 400,
            hop_length: 160,
            audio_layers: 1,
            audio_d_model: 16,
            audio_heads: 2,
            llm_layers: 1,
            llm_d_model: 16,
            llm_heads: 2,
            llm_kv_heads: 2,
            llm_head_dim: 8,
            vocab_size: 32,
            llm_max_positions: 256,
            audio_start_token_id: 2,
            audio_end_token_id: 3,
            audio_pad_token_id: 4,
            eos_token_id: 0,
            pad_token_id: 6,
        };
        let names = llm_layer_tensor_names(0);
        let k_weight_name = names.attn_k_weight.clone();
        let mut spec = TinyGgufFixtureSpec::new(BTreeMap::new())
            .with_tensor_shape(names.attn_norm_weight, [16_u64])
            .with_tensor_shape(names.attn_q_weight, [16_u64, 16_u64])
            .with_tensor_shape(k_weight_name.clone(), [16_u64, 16_u64])
            .with_tensor_shape(names.attn_v_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_output_weight, [16_u64, 16_u64])
            .with_tensor_shape(names.attn_q_norm_weight, [8_u64])
            .with_tensor_shape(names.attn_k_norm_weight, [8_u64])
            .with_tensor_shape(names.ffn_norm_weight, [16_u64])
            // ggml [in, out]: gate/up = [d_model, ffn_dim], down = [ffn_dim, d_model]
            .with_tensor_shape(names.ffn_gate_weight, [16_u64, 32_u64])
            .with_tensor_shape(names.ffn_up_weight, [16_u64, 32_u64])
            .with_tensor_shape(names.ffn_down_weight, [32_u64, 16_u64]);
        if k_as_f16 {
            spec = spec.with_tensor_f16(k_weight_name);
        }
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("metadata-only-decoder.gguf");
        write_tiny_gguf_runtime_source(&path, &spec).expect("write decoder fixture");
        let source = validate_ggml_runtime_source_path(&path).expect("validate decoder fixture");
        (temp, source, metadata)
    }

    fn qwen2_scheduler_parity_layer_names(layer: usize) -> QwenFamilyLlmLayerTensorNames {
        let prefix = format!("qwen2-parity.blk.{layer}");
        QwenFamilyLlmLayerTensorNames {
            attn_norm_name: format!("{prefix}.attn_norm.weight"),
            attn_q_name: format!("{prefix}.attn_q.weight"),
            attn_k_name: format!("{prefix}.attn_k.weight"),
            attn_v_name: format!("{prefix}.attn_v.weight"),
            attn_output_name: format!("{prefix}.attn_output.weight"),
            q_norm_name: None,
            k_norm_name: None,
            q_bias_name: Some(format!("{prefix}.attn_q.bias")),
            k_bias_name: Some(format!("{prefix}.attn_k.bias")),
            v_bias_name: Some(format!("{prefix}.attn_v.bias")),
            ffn_norm_name: format!("{prefix}.ffn_norm.weight"),
            ffn_gate_name: format!("{prefix}.ffn_gate.weight"),
            ffn_up_name: format!("{prefix}.ffn_up.weight"),
            ffn_down_name: format!("{prefix}.ffn_down.weight"),
        }
    }

    fn run_qwen2_scheduler_parity_fixture(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        contract: &QwenDecoderContract,
        use_scheduler: bool,
        hidden: &[f32],
        token_count: usize,
    ) -> Qwen3AsrLlmWholeStepOutput {
        crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_GGML_BACKEND", Some("cpu".into())),
                (
                    "OPENASR_GGML_USE_SCHEDULER",
                    Some(if use_scheduler { "1" } else { "0" }.into()),
                ),
            ],
            || {
                let reader = GgufTensorDataReader::from_runtime_source(preflight.runtime_source())
                    .expect("qwen2 parity reader");
                let plan = QwenWholeDecoderPlan::for_qwen_family(&reader, contract)
                    .expect("qwen2 parity plan");
                let mut executor = compile_qwen_whole_decoder_graph_from_prepared_plan(
                    QwenPreparedDecoderGraphCompileRequest {
                        plan: &plan,
                        preflight,
                        rms_norm_epsilon: DEFAULT_RMS_NORM_EPSILON,
                        fused_logits_head: None,
                        token_embedding: None,
                        backend: GgmlCpuGraphBackend::Cpu,
                    },
                )
                .expect("qwen2 parity executor");
                executor
                    .run_prefill(hidden, token_count, 1_000_000.0)
                    .expect("qwen2 parity prefill")
            },
        )
    }

    fn assert_qwen2_scheduler_vectors_close(label: &str, scheduled: &[f32], direct: &[f32]) {
        assert_eq!(scheduled.len(), direct.len(), "{label} length mismatch");
        let mut max_abs = 0.0_f64;
        let mut squared_error = 0.0_f64;
        let mut squared_reference = 0.0_f64;
        for (&scheduled, &direct) in scheduled.iter().zip(direct) {
            assert!(
                scheduled.is_finite() && direct.is_finite(),
                "{label} non-finite"
            );
            let error = f64::from(scheduled) - f64::from(direct);
            max_abs = max_abs.max(error.abs());
            squared_error += error * error;
            squared_reference += f64::from(direct) * f64::from(direct);
        }
        let relative_l2 = if squared_reference == 0.0 {
            squared_error.sqrt()
        } else {
            (squared_error / squared_reference).sqrt()
        };
        eprintln!(
            "qwen2 scheduler parity {label}: max_abs={max_abs:.9} relative_l2={relative_l2:.12}"
        );
        assert!(
            relative_l2 <= 1.0e-5,
            "{label} scheduler drift exceeds numerical tolerance: max_abs={max_abs:.9} relative_l2={relative_l2:.12}"
        );
    }

    #[test]
    fn whole_decoder_plan_retains_no_weight_payload_and_materializes_one_layer_at_a_time() {
        let (_temp, source, metadata) = metadata_only_decoder_fixture(false);
        let reader = GgufTensorDataReader::from_runtime_source(&source).expect("reader");
        let plan = QwenWholeDecoderPlan::for_qwen3_asr(&reader, metadata).expect("decoder plan");
        assert_eq!(plan.retained_weight_payload_bytes(), 0);
        assert!(matches!(
            plan.layers[0].qkv_storage_mode(false),
            QkvStorageMode::Fused {
                ggml_type: GGML_TYPE_F32
            }
        ));
        assert!(matches!(
            plan.layers[0].qkv_storage_mode(true),
            QkvStorageMode::Split
        ));

        let preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index_from_source(
                &source,
            )
            .expect("decoder fixture preflight");
        let executor = Qwen3AsrLlmWholeDecoderGraphExecutor::new_from_plan_with_preflight(
            &plan,
            &preflight,
            GgmlCpuGraphBackend::Cpu,
        )
        .expect("materialize planned decoder");
        let one_layer_payload = [
            Some(&plan.layers[0].attn_norm),
            plan.layers[0].q_norm.as_ref(),
            plan.layers[0].k_norm.as_ref(),
            Some(&plan.layers[0].ffn_norm),
        ]
        .into_iter()
        .flatten()
        .map(|weight| usize::try_from(weight.size_bytes).expect("vector size"))
        .chain(
            [
                &plan.layers[0].q,
                &plan.layers[0].k,
                &plan.layers[0].v,
                &plan.layers[0].output,
                &plan.layers[0].gate,
                &plan.layers[0].up,
                &plan.layers[0].down,
            ]
            .into_iter()
            .map(|weight| usize::try_from(weight.size_bytes).expect("projection size")),
        )
        .sum::<usize>();
        assert!(
            executor.materialization_peak_staging_bytes <= one_layer_payload,
            "peak staging {} must fit within one layer payload {one_layer_payload}",
            executor.materialization_peak_staging_bytes
        );
    }

    #[test]
    fn split_loaded_qkv_binds_pack_roots_and_matches_fused_prefill() {
        let (_temp, source, metadata) = metadata_only_decoder_fixture(false);
        let reader = GgufTensorDataReader::from_runtime_source(&source).expect("reader");
        let plan = QwenWholeDecoderPlan::for_qwen3_asr(&reader, metadata).expect("decoder plan");
        let preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index_from_source(
                &source,
            )
            .expect("decoder fixture preflight");

        let mut fused =
            Qwen3AsrLlmWholeDecoderGraphExecutor::new_from_plan_with_preflight_and_lora_for_qwen(
                &plan,
                &preflight,
                None,
                None,
                None,
                GgmlCpuGraphBackend::Cpu,
                GgmlNativeGqaCapability::Validated,
                QwenQkvExecutionMode::FusedArena,
            )
            .expect("fused decoder");
        let mut split =
            Qwen3AsrLlmWholeDecoderGraphExecutor::new_from_plan_with_preflight_and_lora_for_qwen(
                &plan,
                &preflight,
                None,
                None,
                None,
                GgmlCpuGraphBackend::Cpu,
                GgmlNativeGqaCapability::Validated,
                QwenQkvExecutionMode::SplitLoaded,
            )
            .expect("split-loaded decoder");

        assert!(
            fused
                .layers
                .iter()
                .all(|layer| matches!(layer.qkv, QwenQkvWeightHandles::Fused(_)))
        );
        assert!(split.layers.iter().all(|layer| matches!(
            layer.qkv,
            QwenQkvWeightHandles::Split {
                q: LlmWeightHandle::Loaded(_),
                k: LlmWeightHandle::Loaded(_),
                v: LlmWeightHandle::Loaded(_),
            } | QwenQkvWeightHandles::FusedQvSplitK {
                k: LlmWeightHandle::Loaded(_),
                ..
            }
        )));

        let token_count = 7;
        let hidden = deterministic_prefill_hidden(metadata.llm_d_model, token_count);
        let fused = fused
            .run_prefill(&hidden, token_count, 1_000_000.0)
            .expect("fused prefill");
        let split = split
            .run_prefill(&hidden, token_count, 1_000_000.0)
            .expect("split-loaded prefill");
        assert_qwen2_scheduler_vectors_close("split-loaded hidden", &split.hidden, &fused.hidden);
        assert_eq!(split.layer_kv.len(), fused.layer_kv.len());
        for (layer_index, ((split_k, split_v), (fused_k, fused_v))) in
            split.layer_kv.iter().zip(&fused.layer_kv).enumerate()
        {
            assert_qwen2_scheduler_vectors_close(
                &format!("split-loaded layer {layer_index} key"),
                split_k,
                fused_k,
            );
            assert_qwen2_scheduler_vectors_close(
                &format!("split-loaded layer {layer_index} value"),
                split_v,
                fused_v,
            );
        }
    }

    #[test]
    fn stateless_prefill_matches_full_hidden_without_materializing_kv() {
        let (_temp, source, metadata) = metadata_only_decoder_fixture(false);
        let reader = GgufTensorDataReader::from_runtime_source(&source).expect("reader");
        let projections = load_qwen3_llm_attention_projections_from_reader(&reader, metadata)
            .expect("synthetic projections");
        let hidden = deterministic_prefill_hidden(metadata.llm_d_model, 7);

        let mut full = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
            &projections,
            Some(&source),
            GgmlCpuGraphBackend::Cpu,
        )
        .expect("full prefill executor");
        let full = full
            .run_prefill(&hidden, 7, 1_000_000.0)
            .expect("full prefill");

        let mut stateless = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
            &projections,
            Some(&source),
            GgmlCpuGraphBackend::Cpu,
        )
        .expect("stateless prefill executor");
        let stateless = stateless
            .run_stateless_prefill(&hidden, 7, 1_000_000.0)
            .expect("stateless prefill");

        assert_eq!(stateless.hidden, full.hidden);
        assert!(stateless.layer_kv.is_empty());
        assert_eq!(full.layer_kv.len(), metadata.llm_layers);
    }

    #[test]
    fn qwen2_materialized_prefill_matches_with_and_without_scheduler() {
        use crate::models::qwen::{
            QwenDecoderTailTensorNames, QwenDecoderVariant, QwenFamilyDecoderProfile,
        };
        use crate::models::tensor_binding::project_fixture_tensors;

        const LAYERS: usize = 4;
        const D_MODEL: usize = 64;
        const TOKEN_COUNT: usize = 48;
        let geometry = QwenDecoderContractGeometry {
            n_layers: LAYERS,
            d_model: D_MODEL,
            n_heads: 8,
            n_kv_heads: 4,
            head_dim: 8,
            ffn_dim: 128,
            vocab_size: 96,
        };
        let contract = QwenDecoderContract::bind(
            geometry,
            QwenFamilyDecoderProfile::new(
                QwenDecoderVariant::Qwen2,
                qwen2_scheduler_parity_layer_names,
                QwenDecoderTailTensorNames {
                    output_norm: "qwen2-parity.output_norm.weight",
                    output_weight: Some("qwen2-parity.output.weight"),
                    token_embd: "qwen2-parity.token_embd.weight",
                },
            ),
        )
        .expect("bind qwen2 scheduler parity contract");
        let descriptors = contract
            .runtime_tensor_descriptors()
            .expect("qwen2 scheduler parity descriptors");
        let mut spec = TinyGgufFixtureSpec::new(BTreeMap::new()).without_tensor("fixture.tensor");
        for (name, dims) in project_fixture_tensors(&descriptors) {
            spec = spec.with_tensor_shape(name, dims);
        }
        let temp = tempfile::tempdir().expect("qwen2 scheduler parity tempdir");
        let path = temp.path().join("qwen2-scheduler-parity.oasr");
        write_tiny_gguf_runtime_source(&path, &spec).expect("write qwen2 parity fixture");
        let preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index(&path)
                .expect("qwen2 parity preflight");
        let hidden = deterministic_prefill_hidden(D_MODEL, TOKEN_COUNT);

        let direct =
            run_qwen2_scheduler_parity_fixture(&preflight, &contract, false, &hidden, TOKEN_COUNT);
        let scheduled =
            run_qwen2_scheduler_parity_fixture(&preflight, &contract, true, &hidden, TOKEN_COUNT);

        assert_qwen2_scheduler_vectors_close("hidden", &scheduled.hidden, &direct.hidden);
        assert_eq!(scheduled.layer_kv.len(), direct.layer_kv.len());
        for (layer, ((scheduled_k, scheduled_v), (direct_k, direct_v))) in
            scheduled.layer_kv.iter().zip(&direct.layer_kv).enumerate()
        {
            assert_qwen2_scheduler_vectors_close(
                &format!("layer {layer} K"),
                scheduled_k,
                direct_k,
            );
            assert_qwen2_scheduler_vectors_close(
                &format!("layer {layer} V"),
                scheduled_v,
                direct_v,
            );
        }
    }

    #[test]
    fn prepared_plan_compile_seam_is_the_sole_backend_materialize_entry() {
        // Structural-adoption gate: the shared compile request is the only
        // way Qwen-shaped adapters turn a host-owned plan into a monomorphic
        // executor. Geometry is not re-derived from metadata here; performance
        // still requires an external receipt.
        let (_temp, source, metadata) = metadata_only_decoder_fixture(false);
        let reader = GgufTensorDataReader::from_runtime_source(&source).expect("reader");
        let plan = QwenWholeDecoderPlan::for_qwen3_asr(&reader, metadata).expect("decoder plan");
        let preflight =
            crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index_from_source(
                &source,
            )
            .expect("decoder fixture preflight");
        let executor = compile_qwen_whole_decoder_graph_from_prepared_plan(
            QwenPreparedDecoderGraphCompileRequest {
                plan: &plan,
                preflight: &preflight,
                rms_norm_epsilon: DEFAULT_RMS_NORM_EPSILON,
                fused_logits_head: None,
                token_embedding: None,
                backend: GgmlCpuGraphBackend::Cpu,
            },
        )
        .expect("prepare-time compile from owned plan");
        assert_eq!(executor.dims.d_model, metadata.llm_d_model);
        assert_eq!(executor.layers.len(), plan.layer_count());
    }

    #[test]
    fn whole_decoder_plan_uses_split_qkv_when_storage_types_differ() {
        let (_temp, source, metadata) = metadata_only_decoder_fixture(true);
        let reader = GgufTensorDataReader::from_runtime_source(&source).expect("reader");
        let plan = QwenWholeDecoderPlan::for_qwen3_asr(&reader, metadata).expect("decoder plan");
        assert!(matches!(
            plan.layers[0].qkv_storage_mode(false),
            QkvStorageMode::Split
        ));
    }

    #[test]
    fn resident_prefill_chunk_preserves_each_sequence_token_span() {
        const D_MODEL: usize = 2;
        for n_seq in [2, 3] {
            for token_count in [1, 255, 256, 257, 513] {
                let values = (0..n_seq * token_count * D_MODEL)
                    .map(|value| value as f32)
                    .collect::<Vec<_>>();
                for position_offset in (0..token_count).step_by(256) {
                    let chunk_tokens = (token_count - position_offset).min(256);
                    let actual =
                        Qwen3AsrLlmWholeDecoderGraphExecutor::sequence_major_prefill_chunk(
                            &values,
                            token_count,
                            n_seq,
                            D_MODEL,
                            position_offset,
                            chunk_tokens,
                        )
                        .expect("valid sequence-major chunk");
                    let expected = (0..n_seq)
                        .flat_map(|sequence_index| {
                            let start = (sequence_index * token_count + position_offset) * D_MODEL;
                            values[start..start + chunk_tokens * D_MODEL]
                                .iter()
                                .copied()
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        actual, expected,
                        "n_seq={n_seq}, token_count={token_count}, offset={position_offset}"
                    );
                }
            }
        }
    }

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
        assert!(qwen_llm_native_gqa_enabled(
            None,
            GgmlNativeGqaCapability::Validated
        ));
        assert!(qwen_llm_native_gqa_enabled(
            Some("native"),
            GgmlNativeGqaCapability::Validated
        ));
        assert!(!qwen_llm_native_gqa_enabled(
            None,
            GgmlNativeGqaCapability::Unsupported
        ));
        assert!(!qwen_llm_native_gqa_enabled(
            Some("maybe"),
            GgmlNativeGqaCapability::Unsupported
        ));
    }

    #[test]
    fn qwen_llm_gpu_prefill_chunk_policy_widens_discrete_gpu_backends() {
        for backend_name in ["HIP0", "ROCm0", "CUDA0", "cuda:0", "Vulkan0", "NVIDIA"] {
            let capabilities = crate::ggml_runtime::GgmlBackendCapabilities::from_backend_for_test(
                GgmlCpuGraphBackend::Gpu,
                backend_name,
            );
            assert_eq!(
                qwen_llm_safe_gpu_prefill_query_tokens_for_backend(capabilities, 8),
                QWEN3_LLM_DISCRETE_GPU_SHORT_PREFILL_QUERY_TOKENS,
                "backend_name={backend_name} short prompt"
            );
            assert_eq!(
                qwen_llm_safe_gpu_prefill_query_tokens_for_backend(capabilities, 32),
                QWEN3_LLM_DISCRETE_GPU_SHORT_PREFILL_QUERY_TOKENS,
                "backend_name={backend_name} flash-tile boundary"
            );
            assert_eq!(
                qwen_llm_safe_gpu_prefill_query_tokens_for_backend(capabilities, 33),
                QWEN3_LLM_DISCRETE_GPU_NONFLASH_PREFILL_QUERY_TOKENS,
                "backend_name={backend_name} non-flash window"
            );
            assert_eq!(
                qwen_llm_safe_gpu_prefill_query_tokens_for_backend(capabilities, 128),
                QWEN3_LLM_DISCRETE_GPU_NONFLASH_PREFILL_QUERY_TOKENS,
                "backend_name={backend_name} long prompt"
            );
        }
        // Metal / unknown names stay on the conservative single-query host
        // width; Metal bulk prefill goes through resident reuse instead.
        for backend_name in ["Metal", "GPU", ""] {
            let capabilities = crate::ggml_runtime::GgmlBackendCapabilities::from_backend_for_test(
                GgmlCpuGraphBackend::Gpu,
                backend_name,
            );
            for token_count in [8, 32, 128] {
                assert_eq!(
                    qwen_llm_safe_gpu_prefill_query_tokens_for_backend(capabilities, token_count),
                    QWEN3_LLM_GPU_SAFE_PREFILL_QUERY_TOKENS,
                    "backend_name={backend_name} token_count={token_count}"
                );
            }
        }
    }

    #[test]
    fn qwen_llm_prefill_flash_policy_is_fail_closed_on_discrete_gpu() {
        // Wide multi-query + long KV must never select flash on discrete GPU.
        assert!(qwen_llm_prefill_uses_flash_attention_for_backend(
            GgmlCpuGraphBackend::Gpu,
            /*token_count=*/ 1,
            /*kv_span=*/ 128
        ));
        assert!(qwen_llm_prefill_uses_flash_attention_for_backend(
            GgmlCpuGraphBackend::Gpu,
            /*token_count=*/ 2,
            /*kv_span=*/ 128
        ));
        assert!(qwen_llm_prefill_uses_flash_attention_for_backend(
            GgmlCpuGraphBackend::Gpu,
            /*token_count=*/ 8,
            /*kv_span=*/ 32
        ));
        assert!(!qwen_llm_prefill_uses_flash_attention_for_backend(
            GgmlCpuGraphBackend::Gpu,
            /*token_count=*/ 8,
            /*kv_span=*/ 33
        ));
        assert!(!qwen_llm_prefill_uses_flash_attention_for_backend(
            GgmlCpuGraphBackend::Gpu,
            /*token_count=*/ 256,
            /*kv_span=*/ 512
        ));
        // Metal/CPU remain flash-trusted past the discrete-GPU fail-closed window.
        assert!(qwen_llm_prefill_uses_flash_attention_for_backend(
            GgmlCpuGraphBackend::Metal,
            /*token_count=*/ 256,
            /*kv_span=*/ 512
        ));
        assert!(qwen_llm_prefill_uses_flash_attention_for_backend(
            GgmlCpuGraphBackend::Cpu,
            /*token_count=*/ 256,
            /*kv_span=*/ 512
        ));
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
    fn qwen_llm_native_gqa_env_can_disable_but_not_promote() {
        assert!(!qwen_llm_native_gqa_enabled(
            Some("0"),
            GgmlNativeGqaCapability::Validated
        ));
        assert!(!qwen_llm_native_gqa_enabled(
            Some("false"),
            GgmlNativeGqaCapability::Validated
        ));
        for raw in [Some("1"), Some("true"), None] {
            assert!(!qwen_llm_native_gqa_enabled(
                raw,
                GgmlNativeGqaCapability::Unsupported
            ));
        }
        crate::test_process_env::with_test_process_env(
            [(
                QWEN3_LLM_NATIVE_GQA_ENV,
                Some(std::ffi::OsString::from("0")),
            )],
            || {
                assert_eq!(
                    qwen_llm_effective_native_gqa_capability(GgmlNativeGqaCapability::Validated),
                    GgmlNativeGqaCapability::Unsupported
                );
            },
        );
    }

    #[test]
    fn qwen_llm_stack_config_never_promotes_multi_sequence_native_gqa() {
        let dims = Qwen3AsrLlmDecodeDims {
            d_model: 8,
            q_width: 8,
            k_width: 4,
            v_width: 4,
            head_dim: 4,
            q_heads: 2,
            kv_heads: 1,
        };
        let rope = GgmlRopeExtParams::qwen_neox(4, 8, 1_000_000.0).expect("rope");
        let config = qwen_llm_stack_config(
            dims,
            rope,
            false,
            DEFAULT_RMS_NORM_EPSILON,
            1,
            2,
            true,
            GgmlFlashAttentionPrecision::Default,
            LlmKvCacheSpec::DEFAULT,
            false,
        );
        assert!(!config.use_native_gqa);
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

    /// Pin the correctness contract every fused-top1 family (moss, and now
    /// qwen/mimo/firered-llm) rides: the device-graph argmax must select the
    /// exact token a host full-vocab logits row would (RMSNorm -> norm-weight
    /// mul -> [d_model x vocab] projection -> first-max argmax), across many
    /// deterministic pseudo-random hidden rows. The host reference below is
    /// computed independently in-test, mirroring
    /// `logits_head::Qwen3AsrLlmLogitsHead`'s `VocabHidden` fallback math.
    #[test]
    fn fused_logits_top1_matches_host_logits_argmax_over_many_hiddens() {
        const D_MODEL: usize = 8;
        const VOCAB: usize = 17;
        fn deterministic_f32_vec(seed: u64, len: usize) -> Vec<f32> {
            let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
            let mut out = Vec::with_capacity(len);
            for _ in 0..len {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = ((state >> 40) as u32 & 0x00FF_FFFF) as f32 / 16_777_216.0;
                out.push(unit * 2.0 - 1.0);
            }
            out
        }
        let output_weight_values = deterministic_f32_vec(0xF0_5ED0, D_MODEL * VOCAB);
        let output_norm_weight = deterministic_f32_vec(0x00BA_D001, D_MODEL);
        let output_weight_bytes = output_weight_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let config = GgmlCpuGraphConfig::default();
        let mut runner =
            GgmlCpuGraphRunner::new(config).expect("cpu graph runner should initialize");
        let mut arena = runner
            .start_static_tensor_arena(config.context_bytes)
            .expect("static arena should allocate");
        let dims = Qwen3AsrLlmDecodeDims {
            d_model: D_MODEL,
            q_width: D_MODEL,
            k_width: D_MODEL,
            v_width: D_MODEL,
            head_dim: D_MODEL,
            q_heads: 1,
            kv_heads: 1,
        };
        let spec = Qwen3AsrLlmFusedLogitsHeadSpec {
            d_model: D_MODEL,
            vocab_size: VOCAB,
            rms_norm_epsilon: DEFAULT_RMS_NORM_EPSILON,
            output_norm_weight: &output_norm_weight,
            output_weight_tensor_name: "synthetic.output.weight",
            output_weight_ggml_type: GGML_TYPE_F32,
            output_weight_dims: &[D_MODEL, VOCAB],
            output_weight_bytes: &output_weight_bytes,
        };
        let handles = allocate_fused_logits_head_tensors(&mut arena, None, dims, &spec)
            .expect("fused logits handles should allocate");
        upload_fused_logits_head_weights(&mut arena, &handles, &spec)
            .expect("fused logits weights should upload");

        for case in 0..32u64 {
            let hidden = deterministic_f32_vec(0x41D_0000 + case, D_MODEL);

            // Independent host reference: RMSNorm + norm-weight mul +
            // [d_model, vocab] matvec + first-max argmax.
            let mut sum_squares = 0.0_f32;
            for value in &hidden {
                sum_squares += value * value;
            }
            let inv_rms = (sum_squares / D_MODEL as f32 + DEFAULT_RMS_NORM_EPSILON)
                .sqrt()
                .recip();
            let mut host_best_token = 0usize;
            let mut host_best_logit = f32::NEG_INFINITY;
            for vocab_index in 0..VOCAB {
                let row = &output_weight_values[vocab_index * D_MODEL..(vocab_index + 1) * D_MODEL];
                let mut logit = 0.0_f32;
                for hidden_index in 0..D_MODEL {
                    logit += hidden[hidden_index]
                        * inv_rms
                        * output_norm_weight[hidden_index]
                        * row[hidden_index];
                }
                if logit > host_best_logit {
                    host_best_logit = logit;
                    host_best_token = vocab_index;
                }
            }

            let mut graph = runner.start_graph();
            let state = graph
                .new_tensor_2d_f32(D_MODEL, 1, "synthetic_state")
                .expect("state tensor should allocate");
            graph.set_input(state).expect("state should be input");
            let top1 = build_fused_logits_top1(&arena, &handles, &mut graph, state, 1)
                .expect("fused top1 should build");
            graph.set_output(top1).expect("top1 should be output");
            graph
                .set_f32_slice(state, &hidden, "synthetic_state")
                .expect("state should upload");
            let reversed_top1 = graph
                .compute_output_i32(top1, 1)
                .expect("fused top1 should compute");
            let fused_token = validate_fused_top1_token_id(reversed_top1[0], VOCAB)
                .expect("top1 should map to a valid token");
            assert_eq!(
                fused_token as usize, host_best_token,
                "fused device argmax diverged from host full-vocab argmax for case {case}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "vocab-hidden fused logits handles should allocate")]
    fn fused_logits_top1_rejects_vocab_hidden_output_layout() {
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
        // Physical [vocab, hidden] storage for the logical [hidden, vocab]
        // projection used by the ordinary logits path.
        let output_weight_bytes = [0.1_f32, 0.3, 0.3, 0.0, 0.0, 0.0]
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
            output_weight_dims: &[3, 2],
            output_weight_bytes: &output_weight_bytes,
        };
        let handles = allocate_fused_logits_head_tensors(&mut arena, None, dims, &spec)
            .expect("vocab-hidden fused logits handles should allocate");
        upload_fused_logits_head_weights(&mut arena, &handles, &spec)
            .expect("vocab-hidden fused logits weights should upload");
        let mut graph = runner.start_graph();
        let state = graph
            .new_tensor_2d_f32(2, 1, "synthetic_state")
            .expect("state");
        graph.set_input(state).expect("state input");
        let top1 = build_fused_logits_top1(&arena, &handles, &mut graph, state, 1)
            .expect("vocab-hidden fused top1 should build");
        graph.set_output(top1).expect("top1 output");
        graph
            .set_f32_slice(state, &[1.0, 0.0], "synthetic_state")
            .expect("state upload");
        let reversed = graph.compute_output_i32(top1, 1).expect("top1 compute");
        assert_eq!(
            validate_fused_top1_token_id(reversed[0], spec.vocab_size).expect("top1 token"),
            1
        );
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
            1,
            2,
            3,
            1,
            2,
            "test_qwen_seed_kv",
            LlmKvCacheSpec::DEFAULT,
        )
        .expect("resident kv arena should allocate");
        // Host planes follow each request's logical bound and may be narrower
        // than the shared resident span.
        let mut seq0 = Qwen3AsrLayerKvCacheState::new(2, 1, 2);
        seq0.write(0, &[1.0, 2.0], &[10.0, 20.0])
            .expect("seq0 row0");
        seq0.write(1, &[3.0, 4.0], &[30.0, 40.0])
            .expect("seq0 row1");
        let mut seq1 = Qwen3AsrLayerKvCacheState::new(1, 1, 2);
        seq1.write(0, &[5.0, 6.0], &[50.0, 60.0])
            .expect("seq1 row0");
        let seq0_layers = vec![seq0];
        let seq1_layers = vec![seq1];
        let seeds: [&[Qwen3AsrLayerKvCacheState]; 2] = [&seq0_layers, &seq1_layers];

        seed_qwen_batched_resident_kv_arena(
            &mut resident,
            2,
            3,
            1,
            &[2, 1],
            &seeds,
            LlmKvCacheSpec::DEFAULT.resident,
        )
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
            1,
            2,
            3,
            1,
            2,
            "test_qwen_slot_seed_kv",
            LlmKvCacheSpec::DEFAULT,
        )
        .expect("resident kv arena should allocate");
        // The active slot's logical host plane is narrower than the stable
        // three-position resident plane.
        let mut seq1 = Qwen3AsrLayerKvCacheState::new(2, 1, 2);
        seq1.write(0, &[1.0, 2.0], &[10.0, 20.0])
            .expect("seq1 row0");
        seq1.write(1, &[3.0, 4.0], &[30.0, 40.0])
            .expect("seq1 row1");
        let seq1_layers = vec![seq1];

        seed_qwen_batched_resident_kv_slot(
            &mut resident,
            2,
            3,
            1,
            1,
            2,
            &seq1_layers,
            LlmKvCacheSpec::DEFAULT.resident,
        )
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

        zero_qwen_batched_resident_kv_slot(
            &mut resident,
            2,
            3,
            1,
            1,
            LlmKvCacheSpec::DEFAULT.resident,
        )
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
            1,
            2,
            3,
            1,
            1,
            "test_qwen_seed_kv",
            LlmKvCacheSpec::DEFAULT,
        )
        .expect("resident kv arena should allocate");
        let mut seq0 = Qwen3AsrLayerKvCacheState::new(3, 1, 2);
        seq0.write(0, &[1.0, 2.0], &[10.0, 20.0])
            .expect("seq0 row0");
        let seq0_layers = vec![seq0];
        let seeds: [&[Qwen3AsrLayerKvCacheState]; 1] = [&seq0_layers];

        let error = seed_qwen_batched_resident_kv_arena(
            &mut resident,
            2,
            3,
            1,
            &[2],
            &seeds,
            LlmKvCacheSpec::DEFAULT.resident,
        )
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
    #[ignore = "manual real-pack cross-backend diagnostic: set OPENASR_QWEN_PREFILL_REAL_PACK and OPENASR_GGML_BACKEND=hip/vulkan/cuda/metal"]
    fn qwen_llm_prefill_real_pack_selected_backend_matches_cpu() {
        let runtime_path = qwen_prefill_real_pack_path();
        let runtime_source =
            validate_ggml_runtime_source_path(&runtime_path).expect("valid qwen runtime source");
        let metadata = read_gguf_metadata_from_runtime_source(&runtime_source)
            .expect("read qwen runtime metadata");
        let metadata = qwen_prefill_execution_metadata(&metadata);
        let token_count = qwen_prefill_token_count(metadata);
        let hidden = deterministic_prefill_hidden(metadata.llm_d_model, token_count);
        let reader =
            GgufTensorDataReader::from_path(runtime_source.path()).expect("qwen tensor reader");
        let projections = load_qwen3_llm_attention_projections_from_reader(&reader, metadata)
            .expect("llm layers");
        let selected_backend = GgmlCpuGraphConfig::runtime_default().backend;
        assert_ne!(
            selected_backend,
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
            "cross-backend diagnostic requires an accelerated backend"
        );
        let cpu = run_qwen_whole_prefill_on_backend(
            &projections,
            &runtime_source,
            token_count,
            &hidden,
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
        );
        let selected = run_qwen_whole_prefill_on_backend(
            &projections,
            &runtime_source,
            token_count,
            &hidden,
            selected_backend,
        );

        let mut hidden_stats = VectorDiffStats::default();
        hidden_stats.extend_pairs(&selected.hidden, &cpu.hidden);
        eprintln!(
            "qwen cross-backend prefill backend={selected_backend:?} token_count={token_count} hidden_max_abs={:.6} hidden_cosine={:.9}",
            hidden_stats.max_abs,
            hidden_stats.cosine(),
        );
        assert!(hidden_stats.is_finite(), "hidden diff stats must be finite");
        assert_eq!(selected.layer_kv.len(), cpu.layer_kv.len());
        for (layer, ((selected_k, selected_v), (cpu_k, cpu_v))) in
            selected.layer_kv.iter().zip(&cpu.layer_kv).enumerate()
        {
            let mut key_stats = VectorDiffStats::default();
            key_stats.extend_pairs(selected_k, cpu_k);
            let mut value_stats = VectorDiffStats::default();
            value_stats.extend_pairs(selected_v, cpu_v);
            eprintln!(
                "qwen cross-backend layer={layer} key_max_abs={:.6} key_cosine={:.9} value_max_abs={:.6} value_cosine={:.9}",
                key_stats.max_abs,
                key_stats.cosine(),
                value_stats.max_abs,
                value_stats.cosine(),
            );
            assert!(
                key_stats.is_finite() && value_stats.is_finite(),
                "layer {layer} KV diff stats must be finite"
            );
        }
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
        let metadata = qwen_prefill_execution_metadata(&metadata);
        let token_count = qwen_prefill_token_count(metadata).min(8);
        let hidden = deterministic_prefill_hidden(metadata.llm_d_model, token_count);
        let reader =
            GgufTensorDataReader::from_path(runtime_source.path()).expect("qwen tensor reader");
        let projections = load_qwen3_llm_attention_projections_from_reader(&reader, metadata)
            .expect("llm layers");
        let serial = run_qwen_serial_prefill(&projections, &runtime_source, metadata, &hidden);
        let seeds_two: [&[Qwen3AsrLayerKvCacheState]; 2] =
            [&serial.layer_kv_caches, &serial.layer_kv_caches];
        let seeds_one: [&[Qwen3AsrLayerKvCacheState]; 1] = [&serial.layer_kv_caches];

        let mut decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
            &projections,
            Some(&runtime_source),
            GgmlCpuGraphConfig::runtime_default().backend,
        )
        .expect("qwen decoder");
        decoder
            .set_kv_cache_policy(LlmKvCachePolicy::Default)
            .expect("pin f32 KV for seed-only reset harness");
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
        let metadata = qwen_prefill_execution_metadata(&metadata);
        let token_count = qwen_prefill_token_count(metadata);
        let hidden = deterministic_prefill_hidden(metadata.llm_d_model, token_count);
        let reader =
            GgufTensorDataReader::from_path(runtime_source.path()).expect("qwen tensor reader");
        let projections = load_qwen3_llm_attention_projections_from_reader(&reader, metadata)
            .expect("llm layers");

        let serial = run_qwen_serial_prefill(&projections, &runtime_source, metadata, &hidden);
        let mut selected_chunk_size = None;
        let prefill = match mode {
            Qwen3AsrPrefillParityMode::Whole => {
                run_qwen_whole_prefill(&projections, &runtime_source, token_count, &hidden)
            }
            Qwen3AsrPrefillParityMode::Chunked { chunk_size } => {
                selected_chunk_size = Some(chunk_size);
                run_qwen_chunked_prefill(
                    &projections,
                    &runtime_source,
                    metadata,
                    token_count,
                    chunk_size,
                    &hidden,
                )
            }
            Qwen3AsrPrefillParityMode::Policy => {
                let chunk_size =
                    qwen_policy_prefill_chunk_size(&projections, &runtime_source, token_count)
                        .expect("qwen policy should return a chunk size for native GQA");
                selected_chunk_size = Some(chunk_size);
                run_qwen_chunked_prefill(
                    &projections,
                    &runtime_source,
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
            backend: GgmlCpuGraphConfig::runtime_default().backend,
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
        runtime_source: &crate::GgmlRuntimeSource,
        metadata: Qwen3AsrExecutionMetadata,
        hidden: &[f32],
    ) -> Qwen3AsrSerialPrefillOutput {
        let token_count = hidden
            .len()
            .checked_div(metadata.llm_d_model)
            .expect("hidden token count");
        let mut decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
            projections,
            Some(runtime_source),
            GgmlCpuGraphConfig::runtime_default().backend,
        )
        .expect("serial qwen decoder");
        // Prefill parity compares against f32 host history rows. Pin Default so
        // production Q8 does not silently break the manual harness.
        decoder
            .set_kv_cache_policy(LlmKvCachePolicy::Default)
            .expect("pin f32 KV for prefill parity");
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
        runtime_source: &crate::GgmlRuntimeSource,
        token_count: usize,
        hidden: &[f32],
    ) -> Qwen3AsrLlmWholeStepOutput {
        run_qwen_whole_prefill_on_backend(
            projections,
            runtime_source,
            token_count,
            hidden,
            GgmlCpuGraphConfig::runtime_default().backend,
        )
    }

    fn run_qwen_whole_prefill_on_backend(
        projections: &[Qwen3AsrLlmLayerAttentionProjection],
        runtime_source: &crate::GgmlRuntimeSource,
        token_count: usize,
        hidden: &[f32],
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Qwen3AsrLlmWholeStepOutput {
        let mut decoder =
            Qwen3AsrLlmWholeDecoderGraphExecutor::new(projections, Some(runtime_source), backend)
                .expect("prefill qwen decoder");
        decoder
            .set_kv_cache_policy(LlmKvCachePolicy::Default)
            .expect("pin f32 KV for prefill parity");
        decoder
            .run_prefill(hidden, token_count, 1_000_000.0)
            .expect("qwen whole-prompt prefill")
    }

    fn qwen_prefill_execution_metadata(
        metadata: &crate::ggml_runtime::GgufMetadata,
    ) -> Qwen3AsrExecutionMetadata {
        match metadata
            .get_string(crate::arch::GENERAL_ARCHITECTURE_KEY)
            .expect("qwen architecture metadata")
            .trim()
        {
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID => {
                parse_qwen3_execution_metadata(metadata).expect("parse qwen metadata")
            }
            crate::models::qwen::QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID => {
                crate::models::qwen::forced_aligner_runtime::parse_forced_aligner_runtime_metadata(
                    metadata,
                )
                .expect("parse forced-aligner metadata")
                .as_embedding_execution_metadata()
            }
            architecture => panic!("unsupported qwen prefill architecture '{architecture}'"),
        }
    }

    fn run_qwen_chunked_prefill(
        projections: &[Qwen3AsrLlmLayerAttentionProjection],
        runtime_source: &crate::GgmlRuntimeSource,
        metadata: Qwen3AsrExecutionMetadata,
        token_count: usize,
        chunk_size: usize,
        hidden: &[f32],
    ) -> Qwen3AsrLlmWholeStepOutput {
        assert!(chunk_size > 0, "chunk size must be positive");
        let mut decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
            projections,
            Some(runtime_source),
            GgmlCpuGraphConfig::runtime_default().backend,
        )
        .expect("chunked prefill qwen decoder");
        decoder
            .set_kv_cache_policy(LlmKvCachePolicy::Default)
            .expect("pin f32 KV for prefill parity");
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
        runtime_source: &crate::GgmlRuntimeSource,
        token_count: usize,
    ) -> Option<usize> {
        let decoder = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
            projections,
            Some(runtime_source),
            GgmlCpuGraphConfig::runtime_default().backend,
        )
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
        assert!(
            matches!(history.element_type, GgmlKvElementType::F32),
            "serial_layer_kv_row helper is f32-only"
        );
        let storage = match kind {
            KvRowKind::Key => history.keys_f32.expect("f32 keys"),
            KvRowKind::Value => history.values_f32.expect("f32 values"),
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
