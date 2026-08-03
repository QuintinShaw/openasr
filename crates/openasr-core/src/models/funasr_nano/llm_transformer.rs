//! The Qwen3-0.6B decoder-only LLM stage, reusing `qwen`'s family-agnostic
//! decoder machinery byte-for-byte: `qwen::load_qwen_family_llm_layer_attention_projection_generic`
//! for layer projections (QK-norm present, no attention bias -- the Qwen3
//! parameterization, same `Option` flips `moss_transcribe_diarize::llm_decoder`
//! uses), `qwen::Qwen3AsrLlmWholeDecoderGraphExecutor` for the whole-decoder
//! ggml graph, `qwen::Qwen3AsrLayerKvCacheState` for the host-side per-layer
//! GQA KV cache, and `qwen::Qwen3AsrLlmLogitsHead` /
//! `qwen::Qwen3AsrTokenEmbeddingTable` for the output/embedding stage. Mirrors
//! `moss_transcribe_diarize::llm_decoder`'s exact shape (both drive the same
//! shared executor with a stock Qwen3-0.6B decoder).

use thiserror::Error;

use crate::ggml_runtime::{GgmlCpuGraphBackend, GgufTensorDataReadError, GgufTensorDataReader};
use crate::models::qwen::{
    Qwen3AsrHostKvCacheOwner, Qwen3AsrHostKvMode, Qwen3AsrKvCacheCapacity,
    Qwen3AsrLayerKvCacheState, Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadRuntime,
    Qwen3AsrLlmWholeDecoderGraphExecutor, Qwen3AsrPromptEmbeddings, Qwen3AsrTokenEmbeddingTable,
    QwenFamilyLlmLayerTensorNames, QwenWholeDecoderPlan,
    load_llm_logits_head_from_reader_with_tensor_names,
    load_token_embedding_table_from_reader_with_tensor_name,
};

use super::runtime_contract::{
    FUNASR_NANO_RMS_NORM_EPSILON, FUNASR_NANO_ROPE_THETA, FunasrNanoDecoderMetadata,
};
use super::tensor_names::{
    LLM_OUTPUT_NORM_WEIGHT, LLM_OUTPUT_WEIGHT, LLM_TOKEN_EMBD_WEIGHT,
    funasr_nano_llm_layer_tensor_names,
};

/// Exact Rust/system-memory quote for one resident FunASR-Nano decoder actor.
/// Native ggml arenas account their own backend-domain allocations; this quote
/// covers only graph-handle containers and any materialized host logits or
/// token-embedding matrices. Construction-phase liveness follows `new`: the
/// temporary decoder plan survives until the whole-decoder graph is built.
pub(crate) fn quoted_funasr_nano_decoder_system_memory_bytes(
    reader: &GgufTensorDataReader,
    metadata: &FunasrNanoDecoderMetadata,
    backend: GgmlCpuGraphBackend,
) -> Result<(u64, u64), String> {
    let graph_retained = Qwen3AsrLlmWholeDecoderGraphExecutor::quoted_retained_system_memory_bytes(
        metadata.n_layers,
    )?;
    let plan_transient = QwenWholeDecoderPlan::quoted_retained_system_memory_bytes_for_family(
        metadata.n_layers,
        |layer_index| {
            let names = funasr_nano_llm_layer_tensor_names(layer_index);
            QwenFamilyLlmLayerTensorNames {
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
            }
        },
    )?;
    let (logits_peak, logits_retained) =
        Qwen3AsrLlmLogitsHead::quoted_system_memory_bytes_from_reader(
            reader,
            LLM_OUTPUT_WEIGHT,
            metadata.d_model,
            metadata.vocab_size,
            backend,
        )?;
    let (embedding_peak, embedding_retained) =
        Qwen3AsrTokenEmbeddingTable::quoted_system_memory_bytes_from_reader(
            reader,
            LLM_TOKEN_EMBD_WEIGHT,
            metadata.d_model,
            metadata.vocab_size,
        )?;

    let retained_bytes = graph_retained
        .checked_add(logits_retained)
        .and_then(|bytes| bytes.checked_add(embedding_retained))
        .ok_or_else(|| "funasr-nano decoder retained quote overflowed".to_string())?;
    let logits_phase_peak = plan_transient
        .checked_add(logits_peak)
        .ok_or_else(|| "funasr-nano logits construction peak quote overflowed".to_string())?;
    let embedding_phase_peak = plan_transient
        .checked_add(logits_retained)
        .and_then(|bytes| bytes.checked_add(embedding_peak))
        .ok_or_else(|| "funasr-nano embedding construction peak quote overflowed".to_string())?;
    let graph_phase_peak = plan_transient
        .checked_add(logits_retained)
        .and_then(|bytes| bytes.checked_add(embedding_retained))
        .and_then(|bytes| bytes.checked_add(graph_retained))
        .ok_or_else(|| "funasr-nano graph construction peak quote overflowed".to_string())?;
    Ok((
        logits_phase_peak
            .max(embedding_phase_peak)
            .max(graph_phase_peak),
        retained_bytes,
    ))
}

#[derive(Debug, Error)]
pub(crate) enum FunasrNanoDecoderError {
    #[error("funasr-nano decoder tensor read failed: {reason}")]
    TensorReadFailed { reason: String },
    #[error("funasr-nano decoder graph failed: {reason}")]
    GraphFailed { reason: String },
    #[error("funasr-nano decoder token-embedding gather failed: {reason}")]
    TokenEmbeddingFailed { reason: String },
    #[error("funasr-nano decoder logits head failed: {reason}")]
    LogitsHeadFailed { reason: String },
    #[error("funasr-nano decoder KV cache write failed: {reason}")]
    KvCacheFailed { reason: String },
    #[error("funasr-nano decoder prefill produced no final hidden state")]
    EmptyPrefillOutput,
}

fn plan_whole_decoder(
    reader: &crate::ggml_runtime::GgufTensorDataReader,
    metadata: &FunasrNanoDecoderMetadata,
) -> Result<QwenWholeDecoderPlan, FunasrNanoDecoderError> {
    QwenWholeDecoderPlan::for_qwen_family(
        reader,
        metadata.n_layers,
        metadata.d_model,
        metadata.n_heads,
        metadata.n_kv_heads,
        metadata.head_dim,
        |layer_index| {
            let names = funasr_nano_llm_layer_tensor_names(layer_index);
            QwenFamilyLlmLayerTensorNames {
                attn_norm_name: names.attn_norm_weight,
                attn_q_name: names.attn_q_weight,
                attn_k_name: names.attn_k_weight,
                attn_v_name: names.attn_v_weight,
                attn_output_name: names.attn_output_weight,
                // Qwen3 has QK-norm (unlike Qwen2).
                q_norm_name: Some(names.attn_q_norm_weight),
                k_norm_name: Some(names.attn_k_norm_weight),
                // Qwen3 has no attention bias.
                q_bias_name: None,
                k_bias_name: None,
                v_bias_name: None,
                ffn_norm_name: names.ffn_norm_weight,
                ffn_gate_name: names.ffn_gate_weight,
                ffn_up_name: names.ffn_up_weight,
                ffn_down_name: names.ffn_down_weight,
            }
        },
    )
    .map_err(|error| FunasrNanoDecoderError::TensorReadFailed {
        reason: error.to_string(),
    })
}

pub(crate) struct FunasrNanoDecoderRuntime {
    whole_decoder: Qwen3AsrLlmWholeDecoderGraphExecutor,
    logits_head: Qwen3AsrLlmLogitsHead,
    logits_runtime: Qwen3AsrLlmLogitsHeadRuntime,
    token_embedding: Qwen3AsrTokenEmbeddingTable,
    metadata: FunasrNanoDecoderMetadata,
}

pub(crate) struct FunasrNanoPrefillOutput {
    pub(crate) logits: Vec<f32>,
    pub(crate) greedy_token_hint: Option<u32>,
}

impl FunasrNanoDecoderRuntime {
    pub(crate) fn new(
        runtime_source: &crate::GgmlRuntimeSource,
        metadata: FunasrNanoDecoderMetadata,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, FunasrNanoDecoderError> {
        let reader = crate::ggml_runtime::GgufTensorDataReader::from_runtime_source(runtime_source)
            .map_err(map_tensor_read_error)?;
        let decoder_plan = plan_whole_decoder(&reader, &metadata)?;
        let logits_head = load_llm_logits_head_from_reader_with_tensor_names(
            &reader,
            runtime_source,
            metadata.d_model,
            metadata.vocab_size,
            LLM_OUTPUT_NORM_WEIGHT,
            // The pack carries a materialized `output.weight` (identical to the
            // tied `token_embd.weight`; both present, see the importer).
            LLM_OUTPUT_WEIGHT,
            FUNASR_NANO_RMS_NORM_EPSILON,
            backend,
        )
        .map_err(|error| FunasrNanoDecoderError::LogitsHeadFailed {
            reason: error.to_string(),
        })?;
        let token_embedding = load_token_embedding_table_from_reader_with_tensor_name(
            &reader,
            LLM_TOKEN_EMBD_WEIGHT,
            metadata.d_model,
            metadata.vocab_size,
        )
        .map_err(|error| FunasrNanoDecoderError::TokenEmbeddingFailed {
            reason: error.to_string(),
        })?;
        let whole_decoder =
            Qwen3AsrLlmWholeDecoderGraphExecutor::new_from_plan_with_rms_norm_epsilon_and_fused_logits_head(
                &decoder_plan,
                runtime_source,
                FUNASR_NANO_RMS_NORM_EPSILON,
                logits_head.fused_top1_spec(),
                backend,
            )
            .map_err(|error| FunasrNanoDecoderError::GraphFailed {
                reason: error.to_string(),
            })?;
        let logits_runtime = logits_head.new_runtime(backend).map_err(|error| {
            FunasrNanoDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            }
        })?;
        Ok(Self {
            whole_decoder,
            logits_head,
            logits_runtime,
            token_embedding,
            metadata,
        })
    }

    /// Exact post-build Rust container capacity retained by this actor.
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add(
            self.whole_decoder.retained_system_memory_bytes()?,
            "funasr-nano decoder graph handles",
        )?;
        bytes.add(
            self.logits_head.retained_system_memory_bytes()?,
            "funasr-nano logits head",
        )?;
        bytes.add(
            self.token_embedding.retained_system_memory_bytes()?,
            "funasr-nano token embedding",
        )?;
        Ok(bytes.finish())
    }

    pub(crate) fn release_session_scoped_buffers(&mut self) {
        self.whole_decoder.release_session_scoped_buffers();
    }

    /// Allocate only the exact logical host history. The stable resident span
    /// is carried separately to the reusable GPU graph.
    pub(crate) fn new_kv_caches(
        &self,
        capacity: Qwen3AsrKvCacheCapacity,
    ) -> Result<Qwen3AsrHostKvCacheOwner, String> {
        let host = self.whole_decoder.kv_cache_spec().host;
        let mode = if self.whole_decoder.supports_graph_reuse() {
            Qwen3AsrHostKvMode::ResidentOnly
        } else {
            Qwen3AsrHostKvMode::Materialized
        };
        Qwen3AsrHostKvCacheOwner::try_new(
            "funasr-nano.decoder.self-kv.host",
            self.metadata.n_layers,
            capacity,
            self.metadata.n_kv_heads,
            self.metadata.head_dim,
            host,
            mode,
        )
    }

    pub(crate) fn gather_token_embedding(
        &self,
        token_id: u32,
    ) -> Result<Vec<f32>, FunasrNanoDecoderError> {
        self.token_embedding
            .gather_rows(&[token_id])
            .map_err(|error| FunasrNanoDecoderError::TokenEmbeddingFailed {
                reason: error.to_string(),
            })
    }

    pub(crate) fn prefill(
        &mut self,
        prompt_embeddings: &Qwen3AsrPromptEmbeddings,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
        capacity: Qwen3AsrKvCacheCapacity,
        control: &std::sync::Arc<crate::api::backend::TranscriptionControl>,
    ) -> Result<FunasrNanoPrefillOutput, FunasrNanoDecoderError> {
        let token_count = prompt_embeddings.token_count;
        if let Some(final_hidden) = self
            .whole_decoder
            .run_prefill_auto_last_hidden(
                &prompt_embeddings.token_major_values,
                token_count,
                layer_kv_caches,
                capacity,
                FUNASR_NANO_ROPE_THETA,
                control,
            )
            .map_err(|error| FunasrNanoDecoderError::GraphFailed {
                reason: error.to_string(),
            })?
        {
            if let Some(token_id) = self
                .whole_decoder
                .fused_logits_top1_from_hidden(&final_hidden)
                .map_err(|error| FunasrNanoDecoderError::GraphFailed {
                    reason: error.to_string(),
                })?
            {
                return Ok(FunasrNanoPrefillOutput {
                    logits: Vec::new(),
                    greedy_token_hint: Some(token_id),
                });
            }
            let logits = self
                .logits_runtime
                .compute_logits_for_last_hidden(&self.logits_head, &final_hidden)
                .map_err(|error| FunasrNanoDecoderError::LogitsHeadFailed {
                    reason: error.to_string(),
                })?;
            return Ok(FunasrNanoPrefillOutput {
                logits,
                greedy_token_hint: None,
            });
        }
        let step = self
            .whole_decoder
            .run_prefill(
                &prompt_embeddings.token_major_values,
                token_count,
                FUNASR_NANO_ROPE_THETA,
            )
            .map_err(|error| FunasrNanoDecoderError::GraphFailed {
                reason: error.to_string(),
            })?;
        let final_hidden = self.write_prefill_outputs(0, token_count, &step, layer_kv_caches)?;
        let logits = self
            .logits_runtime
            .compute_logits_for_last_hidden(&self.logits_head, &final_hidden)
            .map_err(|error| FunasrNanoDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            })?;
        Ok(FunasrNanoPrefillOutput {
            logits,
            greedy_token_hint: None,
        })
    }

    pub(crate) fn decode_step(
        &mut self,
        token_id: u32,
        cache_position: usize,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
        capacity: Qwen3AsrKvCacheCapacity,
    ) -> Result<Vec<f32>, FunasrNanoDecoderError> {
        let hidden = self.gather_token_embedding(token_id)?;
        let step = self
            .whole_decoder
            .run_step_auto(
                &hidden,
                cache_position,
                layer_kv_caches,
                capacity,
                FUNASR_NANO_ROPE_THETA,
            )
            .map_err(|error| FunasrNanoDecoderError::GraphFailed {
                reason: error.to_string(),
            })?;
        write_layer_kv(
            cache_position,
            1,
            &step.layer_kv,
            self.metadata.n_kv_heads * self.metadata.head_dim,
            layer_kv_caches,
        )?;
        self.logits_runtime
            .compute_logits_for_last_hidden(&self.logits_head, &step.hidden)
            .map_err(|error| FunasrNanoDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            })
    }

    /// On the resident Metal/GPU reuse graph, return the decoder's device-side
    /// argmax directly. This family's registered policy has no suppression or
    /// phrase bias, so the shared greedy driver can safely consume this as a
    /// validated `greedy_token_hint`; CPU falls back to the full host logits
    /// path.
    pub(crate) fn decode_step_reused_top1(
        &mut self,
        token_id: u32,
        cache_position: usize,
        layer_kv_caches: &[Qwen3AsrLayerKvCacheState],
        capacity: Qwen3AsrKvCacheCapacity,
    ) -> Result<Option<u32>, FunasrNanoDecoderError> {
        if !self.whole_decoder.supports_graph_reuse() || !self.whole_decoder.supports_fused_top1() {
            return Ok(None);
        }
        if layer_kv_caches.is_empty() {
            return Err(FunasrNanoDecoderError::KvCacheFailed {
                reason: "funasr-nano decoder has no layer KV caches".to_string(),
            });
        }
        let hidden = self.gather_token_embedding(token_id)?;
        let step = self
            .whole_decoder
            .run_step_reused_batched_top1(
                &hidden,
                &[cache_position],
                FUNASR_NANO_ROPE_THETA,
                capacity.resident_positions(),
            )
            .map_err(|error| FunasrNanoDecoderError::GraphFailed {
                reason: error.to_string(),
            })?;
        Ok(Some(step.token_id))
    }

    fn write_prefill_outputs(
        &self,
        position_offset: usize,
        token_count: usize,
        step: &crate::models::qwen::Qwen3AsrLlmWholeStepOutput,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
    ) -> Result<Vec<f32>, FunasrNanoDecoderError> {
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
            .ok_or(FunasrNanoDecoderError::EmptyPrefillOutput)?;
        let final_hidden_end = final_hidden_start
            .checked_add(hidden_size)
            .ok_or(FunasrNanoDecoderError::EmptyPrefillOutput)?;
        step.hidden
            .get(final_hidden_start..final_hidden_end)
            .map(<[f32]>::to_vec)
            .ok_or(FunasrNanoDecoderError::EmptyPrefillOutput)
    }
}

/// `layer_kv` is empty whenever the step came from the persistent reuse graph
/// (`run_step_auto`/`run_prefill_auto`'s reused path): that graph's KV lives
/// resident device-side and is never read back to the host, so there is nothing
/// to write and this is a deliberate no-op -- not a mismatch (mirrors
/// `moss_transcribe_diarize::llm_decoder::write_layer_kv`).
fn write_layer_kv(
    position_offset: usize,
    token_count: usize,
    layer_kv: &[(Vec<f32>, Vec<f32>)],
    kv_row_width: usize,
    layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
) -> Result<(), FunasrNanoDecoderError> {
    if layer_kv.is_empty() {
        return Ok(());
    }
    if layer_kv.len() != layer_kv_caches.len() {
        return Err(FunasrNanoDecoderError::KvCacheFailed {
            reason: "layer-KV count mismatch".to_string(),
        });
    }
    for token_position in 0..token_count {
        let absolute_position = position_offset + token_position;
        let row_start = token_position * kv_row_width;
        let row_end = row_start + kv_row_width;
        for (layer_index, (projected_k, projected_v)) in layer_kv.iter().enumerate() {
            let key_row = projected_k.get(row_start..row_end).ok_or_else(|| {
                FunasrNanoDecoderError::KvCacheFailed {
                    reason: "K row out of bounds".to_string(),
                }
            })?;
            let value_row = projected_v.get(row_start..row_end).ok_or_else(|| {
                FunasrNanoDecoderError::KvCacheFailed {
                    reason: "V row out of bounds".to_string(),
                }
            })?;
            layer_kv_caches[layer_index]
                .write(absolute_position, key_row, value_row)
                .map_err(|reason| FunasrNanoDecoderError::KvCacheFailed { reason })?;
        }
    }
    Ok(())
}

fn map_tensor_read_error(error: GgufTensorDataReadError) -> FunasrNanoDecoderError {
    FunasrNanoDecoderError::TensorReadFailed {
        reason: error.to_string(),
    }
}
