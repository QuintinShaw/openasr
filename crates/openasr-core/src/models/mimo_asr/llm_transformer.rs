//! The 36L Qwen2 backbone stage: loads `blk.N.*` projections (qkv bias on, no
//! QK-norm -- the same shape `firered_llm::llm_transformer` already
//! parameterizes `qwen::load_qwen_family_llm_layer_attention_projection_generic`
//! for) and drives them through `qwen::Qwen3AsrLlmWholeDecoderGraphExecutor`
//! for prefill + single-token decode, exactly mirroring
//! `firered_llm::llm_transformer`'s shape (see that module's doc comment for
//! why GPU/HIP prefill-chunk tuning is deliberately not replicated here, and
//! for why decode DOES go through `run_step_auto`'s persistent-graph reuse on
//! Metal/single-GPU: this ~8B decoder has the same host-graph-construction-
//! bound decode profile as firered2-llm).

use thiserror::Error;

use crate::models::qwen::Qwen3AsrTokenEmbeddingTable;
use crate::models::qwen::{
    Qwen3AsrLayerKvCacheState, Qwen3AsrLlmLayerAttentionProjection, Qwen3AsrLlmLogitsHead,
    Qwen3AsrLlmWholeDecoderGraphExecutor, Qwen3AsrPromptEmbeddings, QwenFamilyLlmLayerTensorNames,
    load_llm_logits_head_from_reader_with_tensor_names,
    load_qwen_family_llm_layer_attention_projection_generic,
    load_token_embedding_table_from_reader_with_tensor_name,
};

use super::runtime_contract::MimoLlmMetadata;
use super::tensor_names::{
    OUTPUT_NORM_WEIGHT, OUTPUT_WEIGHT, TOKEN_EMBD_WEIGHT, mimo_llm_layer_tensor_names,
};

#[derive(Debug, Error)]
pub(crate) enum MimoLlmDecoderError {
    #[error("mimo-asr backbone tensor read failed: {reason}")]
    TensorReadFailed { reason: String },
    #[error("mimo-asr backbone graph failed: {reason}")]
    GraphFailed { reason: String },
    #[error("mimo-asr backbone token-embedding gather failed: {reason}")]
    TokenEmbeddingFailed { reason: String },
    #[error("mimo-asr backbone logits head failed: {reason}")]
    LogitsHeadFailed { reason: String },
    #[error("mimo-asr backbone KV cache write failed: {reason}")]
    KvCacheFailed { reason: String },
    #[error("mimo-asr backbone prefill produced no final hidden state")]
    EmptyPrefillOutput,
}

fn load_layer_projections(
    reader: &crate::ggml_runtime::GgufTensorDataReader,
    metadata: &MimoLlmMetadata,
) -> Result<Vec<Qwen3AsrLlmLayerAttentionProjection>, MimoLlmDecoderError> {
    let mut projections = Vec::with_capacity(metadata.n_layers);
    for layer_index in 0..metadata.n_layers {
        let names = mimo_llm_layer_tensor_names(layer_index);
        let generic = load_qwen_family_llm_layer_attention_projection_generic(
            reader,
            QwenFamilyLlmLayerTensorNames {
                attn_norm_name: names.attn_norm_weight,
                attn_q_name: names.attn_q_weight,
                attn_k_name: names.attn_k_weight,
                attn_v_name: names.attn_v_weight,
                attn_output_name: names.attn_output_weight,
                // MiMo's backbone is Qwen2 (not Qwen3): no QK-norm.
                q_norm_name: None,
                k_norm_name: None,
                // Qwen2 has attention bias on q/k/v; o_proj never has bias
                // (verified against config.json's attention_bias=true and
                // the safetensors index, P2.0 findings SS1.1).
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
        .map_err(|error| MimoLlmDecoderError::TensorReadFailed {
            reason: error.to_string(),
        })?;
        projections.push(Qwen3AsrLlmLayerAttentionProjection::Generic(generic));
    }
    Ok(projections)
}

pub(crate) struct MimoLlmDecoderRuntime {
    whole_decoder: Qwen3AsrLlmWholeDecoderGraphExecutor,
    logits_head: Qwen3AsrLlmLogitsHead,
    token_embedding: Qwen3AsrTokenEmbeddingTable,
    metadata: MimoLlmMetadata,
}

impl MimoLlmDecoderRuntime {
    pub(crate) fn new(
        runtime_path: &std::path::Path,
        metadata: MimoLlmMetadata,
    ) -> Result<Self, MimoLlmDecoderError> {
        let reader = crate::ggml_runtime::GgufTensorDataReader::from_path(runtime_path).map_err(
            |error| MimoLlmDecoderError::TensorReadFailed {
                reason: error.to_string(),
            },
        )?;
        let projections = load_layer_projections(&reader, &metadata)?;
        let whole_decoder =
            Qwen3AsrLlmWholeDecoderGraphExecutor::new_with_rms_norm_epsilon_and_fused_logits_head(
                &projections,
                Some(runtime_path),
                metadata.rms_norm_epsilon,
                None,
            )
            .map_err(|error| MimoLlmDecoderError::GraphFailed {
                reason: error.to_string(),
            })?;
        let logits_head = load_llm_logits_head_from_reader_with_tensor_names(
            &reader,
            metadata.d_model,
            metadata.vocab_size,
            OUTPUT_NORM_WEIGHT,
            OUTPUT_WEIGHT,
            metadata.rms_norm_epsilon,
        )
        .map_err(|error| MimoLlmDecoderError::LogitsHeadFailed {
            reason: error.to_string(),
        })?;
        let token_embedding = load_token_embedding_table_from_reader_with_tensor_name(
            &reader,
            TOKEN_EMBD_WEIGHT,
            metadata.d_model,
            metadata.vocab_size,
        )
        .map_err(|error| MimoLlmDecoderError::TokenEmbeddingFailed {
            reason: error.to_string(),
        })?;
        Ok(Self {
            whole_decoder,
            logits_head,
            token_embedding,
            metadata,
        })
    }

    /// `capacity` should be the request-sized bound (prompt tokens + the
    /// generation budget), NOT the model's native `max_positions`: see
    /// `firered_llm::llm_transformer::FireRedLlmDecoderRuntime::new_kv_caches`'s
    /// doc comment (both families drive the same shared executor, and the
    /// same measured regression applies here) -- `capacity` becomes the
    /// persistent reuse graph's fixed attention span on Metal/GPU, which is a
    /// per-token compute cost there, not just a host allocation ceiling.
    pub(crate) fn new_kv_caches(&self, capacity: usize) -> Vec<Qwen3AsrLayerKvCacheState> {
        (0..self.metadata.n_layers)
            .map(|_| {
                Qwen3AsrLayerKvCacheState::new(
                    capacity,
                    self.metadata.n_kv_heads,
                    self.metadata.head_dim,
                )
            })
            .collect()
    }

    pub(crate) fn gather_token_embedding(
        &self,
        token_id: u32,
    ) -> Result<Vec<f32>, MimoLlmDecoderError> {
        self.token_embedding
            .gather_rows(&[token_id])
            .map_err(|error| MimoLlmDecoderError::TokenEmbeddingFailed {
                reason: error.to_string(),
            })
    }

    /// On a backend that supports persistent decode-graph reuse (Metal/
    /// single-GPU), this runs the prompt through
    /// `run_prefill_auto_last_hidden` instead of the bulk `run_prefill`
    /// below: `decode_step` reuses that same persistent graph, and it can
    /// only see a prompt token's K/V if the prompt flowed through it too
    /// (see that method's doc comment) -- prefilling in bulk and decoding
    /// via reuse would silently attend over an empty KV history for the
    /// whole prompt span. See `firered_llm::llm_transformer::prefill`'s
    /// identical structure (both drive the same shared executor).
    pub(crate) fn prefill(
        &mut self,
        prompt_embeddings: &Qwen3AsrPromptEmbeddings,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
    ) -> Result<Vec<f32>, MimoLlmDecoderError> {
        let token_count = prompt_embeddings.token_count;
        if let Some(final_hidden) = self
            .whole_decoder
            .run_prefill_auto_last_hidden(
                &prompt_embeddings.token_major_values,
                token_count,
                layer_kv_caches,
                self.metadata.rope_theta,
            )
            .map_err(|error| MimoLlmDecoderError::GraphFailed {
                reason: error.to_string(),
            })?
        {
            return self
                .logits_head
                .compute_logits_for_last_hidden(&final_hidden)
                .map_err(|error| MimoLlmDecoderError::LogitsHeadFailed {
                    reason: error.to_string(),
                });
        }
        let step = self
            .whole_decoder
            .run_prefill(
                &prompt_embeddings.token_major_values,
                token_count,
                self.metadata.rope_theta,
            )
            .map_err(|error| MimoLlmDecoderError::GraphFailed {
                reason: error.to_string(),
            })?;
        let final_hidden = self.write_prefill_outputs(0, token_count, &step, layer_kv_caches)?;
        self.logits_head
            .compute_logits_for_last_hidden(&final_hidden)
            .map_err(|error| MimoLlmDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            })
    }

    pub(crate) fn decode_step(
        &mut self,
        token_id: u32,
        cache_position: usize,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
    ) -> Result<Vec<f32>, MimoLlmDecoderError> {
        let hidden = self.gather_token_embedding(token_id)?;
        // `run_step_auto` transparently reuses the persistent decode graph on
        // the Metal/single-GPU lane (see `Qwen3AsrLlmWholeDecoderGraphExecutor
        // ::run_step_auto`'s doc comment); CPU stays on the per-token rebuild
        // path exactly as before.
        let step = self
            .whole_decoder
            .run_step_auto(
                &hidden,
                cache_position,
                layer_kv_caches,
                self.metadata.rope_theta,
            )
            .map_err(|error| MimoLlmDecoderError::GraphFailed {
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
            .map_err(|error| MimoLlmDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            })
    }

    fn write_prefill_outputs(
        &self,
        position_offset: usize,
        token_count: usize,
        step: &crate::models::qwen::Qwen3AsrLlmWholeStepOutput,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
    ) -> Result<Vec<f32>, MimoLlmDecoderError> {
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
            .ok_or(MimoLlmDecoderError::EmptyPrefillOutput)?;
        let final_hidden_end = final_hidden_start
            .checked_add(hidden_size)
            .ok_or(MimoLlmDecoderError::EmptyPrefillOutput)?;
        step.hidden
            .get(final_hidden_start..final_hidden_end)
            .map(<[f32]>::to_vec)
            .ok_or(MimoLlmDecoderError::EmptyPrefillOutput)
    }
}

/// `layer_kv` is empty whenever the step came from the persistent reuse
/// graph (`run_step_auto`'s reused path): that graph's KV lives resident
/// device-side and is never read back to the host, so there is nothing to
/// write and this is a deliberate no-op -- not a mismatch. See
/// `firered_llm::llm_transformer::write_layer_kv`'s identical doc comment.
fn write_layer_kv(
    position_offset: usize,
    token_count: usize,
    layer_kv: &[(Vec<f32>, Vec<f32>)],
    kv_row_width: usize,
    layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
) -> Result<(), MimoLlmDecoderError> {
    if layer_kv.is_empty() {
        return Ok(());
    }
    if layer_kv.len() != layer_kv_caches.len() {
        return Err(MimoLlmDecoderError::KvCacheFailed {
            reason: "layer-KV count mismatch".to_string(),
        });
    }
    for token_position in 0..token_count {
        let absolute_position = position_offset + token_position;
        let row_start = token_position * kv_row_width;
        let row_end = row_start + kv_row_width;
        for (layer_index, (projected_k, projected_v)) in layer_kv.iter().enumerate() {
            let key_row = projected_k.get(row_start..row_end).ok_or_else(|| {
                MimoLlmDecoderError::KvCacheFailed {
                    reason: "K row out of bounds".to_string(),
                }
            })?;
            let value_row = projected_v.get(row_start..row_end).ok_or_else(|| {
                MimoLlmDecoderError::KvCacheFailed {
                    reason: "V row out of bounds".to_string(),
                }
            })?;
            layer_kv_caches[layer_index]
                .write(absolute_position, key_row, value_row)
                .map_err(|reason| MimoLlmDecoderError::KvCacheFailed { reason })?;
        }
    }
    Ok(())
}
