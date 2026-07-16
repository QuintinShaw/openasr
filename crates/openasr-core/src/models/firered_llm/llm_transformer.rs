//! The Qwen2-parameterized decoder-only LLM stage: loads the LoRA-merged
//! Qwen2-7B-Instruct decoder's projections (`llm.blk.N.*`, has attention
//! bias, no QK-norm -- see `tensor_names`' module doc) through
//! `qwen::load_qwen_family_llm_layer_attention_projection_generic` (T3's
//! shared, family-agnostic loader) and drives them through
//! `qwen::Qwen3AsrLlmWholeDecoderGraphExecutor` (T3's shared whole-decoder
//! ggml graph executor, also family-agnostic once QK-norm/bias are
//! parameterized) for prefill + single-token decode, seeding/growing the
//! host-side per-layer GQA KV cache (`qwen::Qwen3AsrLayerKvCacheState`,
//! dimension-driven, not Qwen2/3-specific) exactly the way
//! `qwen::ggml_executor`'s own prefill/decode loop does.
//!
//! Deliberately does NOT replicate qwen's HIP/discrete-GPU prefill-chunk
//! tuning or persistent-graph-session reuse (`qwen::llm_transformer`'s
//! `safe_*_prefill_chunk_size_for` / `Qwen3AsrLlmWholeDecoderGraphExecutor`'s
//! GPU-reuse path): those exist to squeeze ROCm/CUDA decode latency for a
//! shipped, GPU-tuned family. FireRedASR2-LLM's stage-4 goal is a correct,
//! single-shot CPU/Metal transcription path (the upstream 40s hard cap keeps
//! prompts short -- well under any chunking threshold), so this module always
//! runs the plain (build-graph-per-token) path. Re-add GPU chunk tuning here
//! if/when this family ships a GPU-accelerated build.

use thiserror::Error;

use crate::ggml_runtime::GgufTensorDataReadError;
use crate::models::qwen::Qwen3AsrTokenEmbeddingTable;
use crate::models::qwen::{
    Qwen3AsrLayerKvCacheState, Qwen3AsrLlmLayerAttentionProjection, Qwen3AsrLlmLogitsHead,
    Qwen3AsrLlmWholeDecoderGraphExecutor, Qwen3AsrPromptEmbeddings, QwenFamilyLlmLayerTensorNames,
    load_llm_logits_head_from_reader_with_tensor_names,
    load_qwen_family_llm_layer_attention_projection_generic,
    load_token_embedding_table_from_reader_with_tensor_name,
};

use super::runtime_contract::{
    FIRERED_LLM_RMS_NORM_EPSILON, FIRERED_LLM_ROPE_THETA, FireRedLlmDecoderMetadata,
};
use super::tensor_names::{
    LLM_OUTPUT_NORM_WEIGHT, LLM_OUTPUT_WEIGHT, LLM_TOKEN_EMBD_WEIGHT, qwen2_llm_layer_tensor_names,
};

#[derive(Debug, Error)]
pub(crate) enum FireRedLlmDecoderError {
    #[error("firered-llm decoder tensor read failed: {reason}")]
    TensorReadFailed { reason: String },
    #[error("firered-llm decoder graph failed: {reason}")]
    GraphFailed { reason: String },
    #[error("firered-llm decoder token-embedding gather failed: {reason}")]
    TokenEmbeddingFailed { reason: String },
    #[error("firered-llm decoder logits head failed: {reason}")]
    LogitsHeadFailed { reason: String },
    #[error("firered-llm decoder KV cache write failed: {reason}")]
    KvCacheFailed { reason: String },
    #[error("firered-llm decoder prefill produced no final hidden state")]
    EmptyPrefillOutput,
}

fn load_qwen2_layer_projections(
    reader: &crate::ggml_runtime::GgufTensorDataReader,
    metadata: &FireRedLlmDecoderMetadata,
) -> Result<Vec<Qwen3AsrLlmLayerAttentionProjection>, FireRedLlmDecoderError> {
    let mut projections = Vec::with_capacity(metadata.n_layers);
    for layer_index in 0..metadata.n_layers {
        let names = qwen2_llm_layer_tensor_names(layer_index);
        let generic = load_qwen_family_llm_layer_attention_projection_generic(
            reader,
            QwenFamilyLlmLayerTensorNames {
                attn_norm_name: names.attn_norm_weight,
                attn_q_name: names.attn_q_weight,
                attn_k_name: names.attn_k_weight,
                attn_v_name: names.attn_v_weight,
                attn_output_name: names.attn_out_weight,
                // Qwen2 has no QK-norm (unlike Qwen3-ASR).
                q_norm_name: None,
                k_norm_name: None,
                // Qwen2 has attention bias on q/k/v (unlike Qwen3-ASR); o_proj
                // never has bias (verified against the official Qwen2
                // architecture -- see `package_import`'s remap comment).
                q_bias_name: Some(names.attn_q_bias),
                k_bias_name: Some(names.attn_k_bias),
                v_bias_name: Some(names.attn_v_bias),
                ffn_norm_name: names.ffn_norm_weight,
                ffn_gate_name: names.ffn_gate_weight,
                ffn_up_name: names.ffn_up_weight,
                ffn_down_name: names.ffn_down_weight,
            },
            metadata.d_model,
            metadata.n_heads,
            metadata.n_kv_heads,
            metadata.head_dim,
            false,
        )
        .map_err(|error| FireRedLlmDecoderError::TensorReadFailed {
            reason: error.to_string(),
        })?;
        projections.push(Qwen3AsrLlmLayerAttentionProjection::Generic(generic));
    }
    Ok(projections)
}

/// The Qwen2 decoder-only stack for one loaded pack: layer weights + logits
/// head + token embedding table, ready to prefill/decode against a fresh set
/// of per-utterance KV caches (`new_kv_caches`).
pub(crate) struct FireRedLlmDecoderRuntime {
    whole_decoder: Qwen3AsrLlmWholeDecoderGraphExecutor,
    logits_head: Qwen3AsrLlmLogitsHead,
    token_embedding: Qwen3AsrTokenEmbeddingTable,
    metadata: FireRedLlmDecoderMetadata,
}

impl FireRedLlmDecoderRuntime {
    pub(crate) fn new(
        runtime_path: &std::path::Path,
        metadata: FireRedLlmDecoderMetadata,
    ) -> Result<Self, FireRedLlmDecoderError> {
        let reader = crate::ggml_runtime::GgufTensorDataReader::from_path(runtime_path)
            .map_err(map_tensor_read_error)?;
        let projections = load_qwen2_layer_projections(&reader, &metadata)?;
        let whole_decoder =
            Qwen3AsrLlmWholeDecoderGraphExecutor::new_with_rms_norm_epsilon_and_fused_logits_head(
                &projections,
                Some(runtime_path),
                FIRERED_LLM_RMS_NORM_EPSILON,
                None,
            )
            .map_err(|error| FireRedLlmDecoderError::GraphFailed {
                reason: error.to_string(),
            })?;
        let logits_head = load_llm_logits_head_from_reader_with_tensor_names(
            &reader,
            metadata.d_model,
            metadata.vocab_size,
            LLM_OUTPUT_NORM_WEIGHT,
            LLM_OUTPUT_WEIGHT,
            FIRERED_LLM_RMS_NORM_EPSILON,
        )
        .map_err(|error| FireRedLlmDecoderError::LogitsHeadFailed {
            reason: error.to_string(),
        })?;
        let token_embedding = load_token_embedding_table_from_reader_with_tensor_name(
            &reader,
            LLM_TOKEN_EMBD_WEIGHT,
            metadata.d_model,
            metadata.vocab_size,
        )
        .map_err(|error| FireRedLlmDecoderError::TokenEmbeddingFailed {
            reason: error.to_string(),
        })?;
        Ok(Self {
            whole_decoder,
            logits_head,
            token_embedding,
            metadata,
        })
    }

    pub(crate) fn new_kv_caches(&self) -> Vec<Qwen3AsrLayerKvCacheState> {
        (0..self.metadata.n_layers)
            .map(|_| {
                Qwen3AsrLayerKvCacheState::new(
                    self.metadata.max_positions,
                    self.metadata.n_kv_heads,
                    self.metadata.head_dim,
                )
            })
            .collect()
    }

    pub(crate) fn gather_token_embedding(
        &self,
        token_id: u32,
    ) -> Result<Vec<f32>, FireRedLlmDecoderError> {
        self.token_embedding
            .gather_rows(&[token_id])
            .map_err(|error| FireRedLlmDecoderError::TokenEmbeddingFailed {
                reason: error.to_string(),
            })
    }

    /// Run the entire ChatML+speech prompt as one causal prefill pass,
    /// seeding `layer_kv_caches` with every prompt token's K/V, and return the
    /// logits row for the token immediately following the prompt (i.e. the
    /// first generated token's distribution) -- mirrors
    /// `qwen::ggml_executor`'s `write_prefill_step_outputs_and_compute_last_logits`.
    pub(crate) fn prefill(
        &mut self,
        prompt_embeddings: &Qwen3AsrPromptEmbeddings,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
    ) -> Result<Vec<f32>, FireRedLlmDecoderError> {
        let token_count = prompt_embeddings.token_count;
        let step = self
            .whole_decoder
            .run_prefill(
                &prompt_embeddings.token_major_values,
                token_count,
                FIRERED_LLM_ROPE_THETA,
            )
            .map_err(|error| FireRedLlmDecoderError::GraphFailed {
                reason: error.to_string(),
            })?;
        let final_hidden = self.write_prefill_outputs(0, token_count, &step, layer_kv_caches)?;
        self.logits_head
            .compute_logits_for_last_hidden(&final_hidden)
            .map_err(|error| FireRedLlmDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            })
    }

    /// Run one incremental decode step for `token_id` at `cache_position`
    /// (the position this token's own K/V will occupy), updating
    /// `layer_kv_caches`, and return the logits row for the NEXT token.
    pub(crate) fn decode_step(
        &mut self,
        token_id: u32,
        cache_position: usize,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
    ) -> Result<Vec<f32>, FireRedLlmDecoderError> {
        let hidden = self.gather_token_embedding(token_id)?;
        let step = self
            .whole_decoder
            .run_step(
                &hidden,
                cache_position,
                layer_kv_caches,
                FIRERED_LLM_ROPE_THETA,
            )
            .map_err(|error| FireRedLlmDecoderError::GraphFailed {
                reason: error.to_string(),
            })?;
        write_layer_kv(
            cache_position,
            1,
            &step.layer_kv,
            self.metadata.n_kv_heads * self.metadata.head_dim,
            layer_kv_caches,
        )?;
        self.logits_head
            .compute_logits_for_last_hidden(&step.hidden)
            .map_err(|error| FireRedLlmDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            })
    }

    fn write_prefill_outputs(
        &self,
        position_offset: usize,
        token_count: usize,
        step: &crate::models::qwen::Qwen3AsrLlmWholeStepOutput,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
    ) -> Result<Vec<f32>, FireRedLlmDecoderError> {
        let kv_row_width = self.metadata.n_kv_heads * self.metadata.head_dim;
        write_layer_kv(
            position_offset,
            token_count,
            &step.layer_kv,
            kv_row_width,
            layer_kv_caches,
        )?;
        let hidden_size = self.metadata.d_model;
        let final_hidden_start = token_count
            .checked_sub(1)
            .and_then(|position| position.checked_mul(hidden_size))
            .ok_or(FireRedLlmDecoderError::EmptyPrefillOutput)?;
        let final_hidden_end = final_hidden_start
            .checked_add(hidden_size)
            .ok_or(FireRedLlmDecoderError::EmptyPrefillOutput)?;
        step.hidden
            .get(final_hidden_start..final_hidden_end)
            .map(<[f32]>::to_vec)
            .ok_or(FireRedLlmDecoderError::EmptyPrefillOutput)
    }
}

/// Write `token_count` rows (starting at `position_offset`) of every layer's
/// projected K/V into the corresponding host KV cache. Mirrors
/// `qwen::ggml_executor::write_prefill_chunk_outputs`'s per-token,
/// per-layer write loop (that function is private to `qwen::ggml_executor`,
/// so this is a small parallel copy rather than a cross-module reuse --
/// unlike the executor/loader machinery above, this is a ~15-line loop, not
/// worth threading a new pub(crate) export through for).
fn write_layer_kv(
    position_offset: usize,
    token_count: usize,
    layer_kv: &[(Vec<f32>, Vec<f32>)],
    kv_row_width: usize,
    layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
) -> Result<(), FireRedLlmDecoderError> {
    if layer_kv.len() != layer_kv_caches.len() {
        return Err(FireRedLlmDecoderError::KvCacheFailed {
            reason: "layer-KV count mismatch".to_string(),
        });
    }
    for token_position in 0..token_count {
        let absolute_position = position_offset + token_position;
        let row_start = token_position * kv_row_width;
        let row_end = row_start + kv_row_width;
        for (layer_index, (projected_k, projected_v)) in layer_kv.iter().enumerate() {
            let key_row = projected_k.get(row_start..row_end).ok_or_else(|| {
                FireRedLlmDecoderError::KvCacheFailed {
                    reason: "K row out of bounds".to_string(),
                }
            })?;
            let value_row = projected_v.get(row_start..row_end).ok_or_else(|| {
                FireRedLlmDecoderError::KvCacheFailed {
                    reason: "V row out of bounds".to_string(),
                }
            })?;
            layer_kv_caches[layer_index]
                .write(absolute_position, key_row, value_row)
                .map_err(|reason| FireRedLlmDecoderError::KvCacheFailed { reason })?;
        }
    }
    Ok(())
}

fn map_tensor_read_error(error: GgufTensorDataReadError) -> FireRedLlmDecoderError {
    FireRedLlmDecoderError::TensorReadFailed {
        reason: error.to_string(),
    }
}
