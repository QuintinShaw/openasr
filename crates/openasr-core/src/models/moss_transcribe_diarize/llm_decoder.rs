//! The Qwen3-0.6B decoder-only LLM stage, reusing `qwen`'s family-agnostic
//! decoder machinery byte-for-byte: `qwen::load_qwen_family_llm_layer_attention_projection_generic`
//! for layer projections (QK-norm present, no attention bias -- the inverse
//! of `firered_llm`'s Qwen2 parameterization, but the SAME shared loader,
//! just with the `Option` fields flipped), `qwen::Qwen3AsrLlmWholeDecoderGraphExecutor`
//! for the whole-decoder ggml graph, `qwen::Qwen3AsrLayerKvCacheState` for the
//! host-side per-layer GQA KV cache, and `qwen::Qwen3AsrLlmLogitsHead` /
//! `qwen::Qwen3AsrTokenEmbeddingTable` for the output/embedding stage.
//! Mirrors `firered_llm::llm_transformer`'s exact shape (see that module's
//! doc comment for why this crate does not replicate qwen's GPU-tuned
//! prefill-chunk/persistent-session machinery here: correctness-first single-
//! shot decode, GPU perf tuning is out of scope this stage).

use thiserror::Error;

use crate::ggml_runtime::GgufTensorDataReadError;
use crate::models::qwen::{
    Qwen3AsrLayerKvCacheState, Qwen3AsrLlmLayerAttentionProjection, Qwen3AsrLlmLogitsHead,
    Qwen3AsrLlmWholeDecoderGraphExecutor, Qwen3AsrPromptEmbeddings, Qwen3AsrTokenEmbeddingTable,
    QwenFamilyLlmLayerTensorNames, load_llm_logits_head_from_reader_with_tensor_names,
    load_qwen_family_llm_layer_attention_projection_generic,
    load_token_embedding_table_from_reader_with_tensor_name,
};

use super::runtime_contract::{
    MOSS_TD_RMS_NORM_EPSILON, MOSS_TD_ROPE_THETA, MossTdDecoderMetadata, moss_td_kv_cache_positions,
};
use super::tensor_names::{
    LLM_OUTPUT_NORM_WEIGHT, LLM_TOKEN_EMBD_WEIGHT, moss_llm_layer_tensor_names,
};

#[derive(Debug, Error)]
pub(crate) enum MossTdDecoderError {
    #[error("moss-transcribe-diarize decoder tensor read failed: {reason}")]
    TensorReadFailed { reason: String },
    #[error("moss-transcribe-diarize decoder graph failed: {reason}")]
    GraphFailed { reason: String },
    #[error("moss-transcribe-diarize decoder token-embedding gather failed: {reason}")]
    TokenEmbeddingFailed { reason: String },
    #[error("moss-transcribe-diarize decoder logits head failed: {reason}")]
    LogitsHeadFailed { reason: String },
    #[error("moss-transcribe-diarize decoder KV cache write failed: {reason}")]
    KvCacheFailed { reason: String },
    #[error("moss-transcribe-diarize decoder prefill produced no final hidden state")]
    EmptyPrefillOutput,
}

fn load_moss_layer_projections(
    reader: &crate::ggml_runtime::GgufTensorDataReader,
    metadata: &MossTdDecoderMetadata,
) -> Result<Vec<Qwen3AsrLlmLayerAttentionProjection>, MossTdDecoderError> {
    let mut projections = Vec::with_capacity(metadata.n_layers);
    for layer_index in 0..metadata.n_layers {
        let names = moss_llm_layer_tensor_names(layer_index);
        let generic = load_qwen_family_llm_layer_attention_projection_generic(
            reader,
            QwenFamilyLlmLayerTensorNames {
                attn_norm_name: names.attn_norm_weight,
                attn_q_name: names.attn_q_weight,
                attn_k_name: names.attn_k_weight,
                attn_v_name: names.attn_v_weight,
                attn_output_name: names.attn_output_weight,
                // Qwen3 has QK-norm (unlike Qwen2/firered-llm).
                q_norm_name: Some(names.attn_q_norm_weight),
                k_norm_name: Some(names.attn_k_norm_weight),
                // Qwen3 has no attention bias (unlike Qwen2/firered-llm).
                q_bias_name: None,
                k_bias_name: None,
                v_bias_name: None,
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
        .map_err(|error| MossTdDecoderError::TensorReadFailed {
            reason: error.to_string(),
        })?;
        projections.push(Qwen3AsrLlmLayerAttentionProjection::Generic(generic));
    }
    Ok(projections)
}

/// The Qwen3-0.6B decoder-only stack for one loaded pack: layer weights +
/// logits head + token embedding table (tied to the same tensor as the
/// logits head's output weight -- `config.tie_word_embeddings=true`, see
/// `package_import`'s module doc), ready to prefill/decode against a fresh
/// set of per-utterance KV caches (`new_kv_caches`).
pub(crate) struct MossTdDecoderRuntime {
    whole_decoder: Qwen3AsrLlmWholeDecoderGraphExecutor,
    logits_head: Qwen3AsrLlmLogitsHead,
    token_embedding: Qwen3AsrTokenEmbeddingTable,
    metadata: MossTdDecoderMetadata,
}

impl MossTdDecoderRuntime {
    pub(crate) fn new(
        runtime_path: &std::path::Path,
        metadata: MossTdDecoderMetadata,
    ) -> Result<Self, MossTdDecoderError> {
        let reader = crate::ggml_runtime::GgufTensorDataReader::from_path(runtime_path)
            .map_err(map_tensor_read_error)?;
        let projections = load_moss_layer_projections(&reader, &metadata)?;
        let whole_decoder =
            Qwen3AsrLlmWholeDecoderGraphExecutor::new_with_rms_norm_epsilon_and_fused_logits_head(
                &projections,
                Some(runtime_path),
                MOSS_TD_RMS_NORM_EPSILON,
                None,
            )
            .map_err(|error| MossTdDecoderError::GraphFailed {
                reason: error.to_string(),
            })?;
        let logits_head = load_llm_logits_head_from_reader_with_tensor_names(
            &reader,
            metadata.d_model,
            metadata.vocab_size,
            LLM_OUTPUT_NORM_WEIGHT,
            // Tied embeddings: the output projection reuses the token
            // embedding tensor -- no separate lm_head tensor exists in the
            // pack (see `package_import`).
            LLM_TOKEN_EMBD_WEIGHT,
            MOSS_TD_RMS_NORM_EPSILON,
        )
        .map_err(|error| MossTdDecoderError::LogitsHeadFailed {
            reason: error.to_string(),
        })?;
        let token_embedding = load_token_embedding_table_from_reader_with_tensor_name(
            &reader,
            LLM_TOKEN_EMBD_WEIGHT,
            metadata.d_model,
            metadata.vocab_size,
        )
        .map_err(|error| MossTdDecoderError::TokenEmbeddingFailed {
            reason: error.to_string(),
        })?;
        Ok(Self {
            whole_decoder,
            logits_head,
            token_embedding,
            metadata,
        })
    }

    pub(crate) fn backend_label(&self) -> String {
        self.whole_decoder.backend_label()
    }

    pub(crate) fn new_kv_caches(&self) -> Vec<Qwen3AsrLayerKvCacheState> {
        // Clamp the RoPE context limit (`max_positions`, up to 131072) down to
        // the KV-cache preallocation cap: `Qwen3AsrLayerKvCacheState::new`
        // eagerly allocates the full `max_positions` span on first write, so an
        // uncapped 131072 reserves ~30 GB across the 28 layers (see
        // `moss_td_kv_cache_positions`). This makes packs built before the cap
        // (131072 baked into the GGUF) allocate the same ~1.9 GB as fresh ones.
        let cache_positions = moss_td_kv_cache_positions(self.metadata.max_positions);
        (0..self.metadata.n_layers)
            .map(|_| {
                Qwen3AsrLayerKvCacheState::new(
                    cache_positions,
                    self.metadata.n_kv_heads,
                    self.metadata.head_dim,
                )
            })
            .collect()
    }

    pub(crate) fn gather_token_embedding(
        &self,
        token_id: u32,
    ) -> Result<Vec<f32>, MossTdDecoderError> {
        self.token_embedding
            .gather_rows(&[token_id])
            .map_err(|error| MossTdDecoderError::TokenEmbeddingFailed {
                reason: error.to_string(),
            })
    }

    /// Run the entire ChatML+audio prompt as one causal prefill pass, seeding
    /// `layer_kv_caches` with every prompt token's K/V, and return the logits
    /// row for the token immediately following the prompt.
    pub(crate) fn prefill(
        &mut self,
        prompt_embeddings: &Qwen3AsrPromptEmbeddings,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
    ) -> Result<Vec<f32>, MossTdDecoderError> {
        let token_count = prompt_embeddings.token_count;
        let step = self
            .whole_decoder
            .run_prefill(
                &prompt_embeddings.token_major_values,
                token_count,
                MOSS_TD_ROPE_THETA,
            )
            .map_err(|error| MossTdDecoderError::GraphFailed {
                reason: error.to_string(),
            })?;
        let final_hidden = self.write_prefill_outputs(0, token_count, &step, layer_kv_caches)?;
        self.logits_head
            .compute_logits_for_last_hidden(&final_hidden)
            .map_err(|error| MossTdDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            })
    }

    /// Run one incremental decode step for `token_id` at `cache_position`,
    /// updating `layer_kv_caches`, and return the logits row for the NEXT
    /// token.
    pub(crate) fn decode_step(
        &mut self,
        token_id: u32,
        cache_position: usize,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
    ) -> Result<Vec<f32>, MossTdDecoderError> {
        let hidden = self.gather_token_embedding(token_id)?;
        let step = self
            .whole_decoder
            .run_step(&hidden, cache_position, layer_kv_caches, MOSS_TD_ROPE_THETA)
            .map_err(|error| MossTdDecoderError::GraphFailed {
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
            .map_err(|error| MossTdDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            })
    }

    fn write_prefill_outputs(
        &self,
        position_offset: usize,
        token_count: usize,
        step: &crate::models::qwen::Qwen3AsrLlmWholeStepOutput,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
    ) -> Result<Vec<f32>, MossTdDecoderError> {
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
            .ok_or(MossTdDecoderError::EmptyPrefillOutput)?;
        let final_hidden_end = final_hidden_start
            .checked_add(hidden_size)
            .ok_or(MossTdDecoderError::EmptyPrefillOutput)?;
        step.hidden
            .get(final_hidden_start..final_hidden_end)
            .map(<[f32]>::to_vec)
            .ok_or(MossTdDecoderError::EmptyPrefillOutput)
    }
}

fn write_layer_kv(
    position_offset: usize,
    token_count: usize,
    layer_kv: &[(Vec<f32>, Vec<f32>)],
    kv_row_width: usize,
    layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
) -> Result<(), MossTdDecoderError> {
    if layer_kv.len() != layer_kv_caches.len() {
        return Err(MossTdDecoderError::KvCacheFailed {
            reason: "layer-KV count mismatch".to_string(),
        });
    }
    for token_position in 0..token_count {
        let absolute_position = position_offset + token_position;
        let row_start = token_position * kv_row_width;
        let row_end = row_start + kv_row_width;
        for (layer_index, (projected_k, projected_v)) in layer_kv.iter().enumerate() {
            let key_row = projected_k.get(row_start..row_end).ok_or_else(|| {
                MossTdDecoderError::KvCacheFailed {
                    reason: "K row out of bounds".to_string(),
                }
            })?;
            let value_row = projected_v.get(row_start..row_end).ok_or_else(|| {
                MossTdDecoderError::KvCacheFailed {
                    reason: "V row out of bounds".to_string(),
                }
            })?;
            layer_kv_caches[layer_index]
                .write(absolute_position, key_row, value_row)
                .map_err(|reason| MossTdDecoderError::KvCacheFailed { reason })?;
        }
    }
    Ok(())
}

fn map_tensor_read_error(error: GgufTensorDataReadError) -> MossTdDecoderError {
    MossTdDecoderError::TensorReadFailed {
        reason: error.to_string(),
    }
}
