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

use crate::ggml_runtime::{GgmlCpuGraphBackend, GgufTensorDataReader, ResolvedFamilyRuntimeInput};

use crate::models::mapped_token_embedding::MappedTokenEmbeddingTable;
use crate::models::qwen::{
    Qwen3AsrHostKvCacheOwner, Qwen3AsrHostKvMode, Qwen3AsrKvCacheCapacity,
    Qwen3AsrLayerKvCacheState, Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadRuntime,
    Qwen3AsrLlmWholeDecoderGraphExecutor, Qwen3AsrPromptEmbeddings, QwenDecoderTail,
    QwenDecoderTailLoadError, QwenPreparedDecoderGraphCompileRequest, QwenWholeDecoderPlan,
    build_qwen3_prompt_embeddings_with_audio_positions,
    compile_qwen_whole_decoder_graph_from_prepared_plan_with_native_gqa,
    load_qwen_decoder_tail_from_contract, quoted_qwen_decoder_system_memory_bytes,
};

use super::runtime_contract::{MimoLlmMetadata, mimo_asr_qwen_decoder_contract};

pub(crate) fn quoted_mimo_llm_decoder_system_memory_bytes(
    reader: &GgufTensorDataReader,
    metadata: &MimoLlmMetadata,
    backend: GgmlCpuGraphBackend,
) -> Result<(u64, u64), String> {
    let contract = mimo_asr_qwen_decoder_contract(metadata).map_err(|error| error.to_string())?;
    quoted_qwen_decoder_system_memory_bytes(reader, &contract, backend)
}

#[derive(Debug, Error)]
pub(crate) enum MimoLlmDecoderError {
    #[error("mimo-asr backbone tensor read failed: {reason}")]
    TensorReadFailed { reason: String },
    #[error("mimo-asr backbone graph failed: {reason}")]
    GraphFailed { reason: String },
    #[error("mimo-asr backbone token-embedding gather failed: {reason}")]
    TokenEmbeddingFailed { reason: String },
    #[error("mimo-asr backbone prompt embedding failed: {reason}")]
    PromptEmbeddingFailed { reason: String },
    #[error("mimo-asr backbone logits head failed: {reason}")]
    LogitsHeadFailed { reason: String },
    #[error("mimo-asr backbone KV cache write failed: {reason}")]
    KvCacheFailed { reason: String },
    #[error("mimo-asr backbone prefill produced no final hidden state")]
    EmptyPrefillOutput,
}

fn map_tail_load_error(error: QwenDecoderTailLoadError) -> MimoLlmDecoderError {
    match error {
        QwenDecoderTailLoadError::TokenEmbedding(error) => {
            MimoLlmDecoderError::TokenEmbeddingFailed {
                reason: error.to_string(),
            }
        }
        other => MimoLlmDecoderError::LogitsHeadFailed {
            reason: other.to_string(),
        },
    }
}

pub(crate) struct MimoLlmDecoderRuntime {
    whole_decoder: Qwen3AsrLlmWholeDecoderGraphExecutor,
    logits_head: Qwen3AsrLlmLogitsHead,
    logits_runtime: Qwen3AsrLlmLogitsHeadRuntime,
    token_embedding: MappedTokenEmbeddingTable,
    metadata: MimoLlmMetadata,
}

/// Prefill output for the shared greedy driver's step 0: the host logits row
/// for the first generated token, or (on the fused Metal/GPU lane) a device
/// argmax hint with no host row. Mirrors
/// `moss_transcribe_diarize::llm_decoder::MossTdPrefillOutput`.
pub(crate) struct MimoLlmPrefillOutput {
    pub(crate) logits: Vec<f32>,
    pub(crate) greedy_token_hint: Option<u32>,
}

impl MimoLlmDecoderRuntime {
    pub(crate) fn take_compute_evidence(
        &mut self,
    ) -> Option<crate::ggml_runtime::GgmlSelectionEvidenceRef> {
        self.whole_decoder
            .take_fused_compute_evidence()
            .or_else(|| self.logits_runtime.take_compute_evidence())
    }

    pub(crate) fn new_from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        metadata: MimoLlmMetadata,
        resolved_runtime: ResolvedFamilyRuntimeInput,
    ) -> Result<Self, MimoLlmDecoderError> {
        let backend = resolved_runtime.backend();
        let reader =
            crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight(preflight)
                .map_err(|error| MimoLlmDecoderError::TensorReadFailed {
                    reason: error.to_string(),
                })?;
        // Bind the Qwen decoder contract exactly once for plan + tail + compile.
        let contract = mimo_asr_qwen_decoder_contract(&metadata).map_err(|error| {
            MimoLlmDecoderError::TensorReadFailed {
                reason: error.to_string(),
            }
        })?;
        let decoder_plan =
            QwenWholeDecoderPlan::for_qwen_family(&reader, &contract).map_err(|error| {
                MimoLlmDecoderError::TensorReadFailed {
                    reason: error.to_string(),
                }
            })?;
        let QwenDecoderTail {
            logits_head,
            token_embedding,
        } = load_qwen_decoder_tail_from_contract(
            &reader,
            &contract,
            metadata.rms_norm_epsilon,
            backend,
        )
        .map_err(map_tail_load_error)?;
        // Keep the output projection in the same static arena as the resident
        // decoder graph so Metal/GPU decode can return a device-side top-1
        // token per step instead of building a separate full-vocab logits
        // graph and reading the whole row back to the host -- mirrors
        // `moss_transcribe_diarize::llm_decoder`'s identical wiring (mimo's
        // registered policy has no suppression or phrase bias, so the shared
        // driver can always honor the hint).
        let whole_decoder = compile_qwen_whole_decoder_graph_from_prepared_plan_with_native_gqa(
            QwenPreparedDecoderGraphCompileRequest {
                plan: &decoder_plan,
                preflight,
                rms_norm_epsilon: metadata.rms_norm_epsilon,
                fused_logits_head: logits_head.fused_top1_spec(),
                token_embedding: token_embedding.device_graph_spec(),
                resolved_runtime,
            },
        )
        .map_err(|error| MimoLlmDecoderError::GraphFailed {
            reason: error.to_string(),
        })?;
        // The graph constructor has copied/bound every planned tensor handle;
        // release the heap-heavy transient plan before materializing the token
        // embedding so construction peak follows the quoted phase topology.
        // (Tail load already ran above so the peak topology is logits-then-graph;
        // dropping the plan still frees layer-plan heap before the runtime is
        // retained.)
        drop(decoder_plan);
        let logits_runtime = logits_head.new_runtime(backend).map_err(|error| {
            MimoLlmDecoderError::LogitsHeadFailed {
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

    /// Exact post-build Rust container capacity retained by the resident
    /// decoder actor. Native graph arenas and backend buffers are admitted by
    /// their constructors and intentionally excluded here.
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add(
            self.whole_decoder.retained_system_memory_bytes()?,
            "mimo-asr decoder graph handles",
        )?;
        bytes.add(
            self.logits_head.retained_system_memory_bytes()?,
            "mimo-asr logits head",
        )?;
        bytes.add(
            self.token_embedding.retained_system_memory_bytes()?,
            "mimo-asr token embedding",
        )?;
        Ok(bytes.finish())
    }

    /// Allocate only the invocation's exact logical host history; the stable
    /// session reserve is passed independently to the reusable GPU graph.
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
            "mimo-asr.decoder.self-kv.host",
            self.metadata.n_layers,
            capacity,
            self.metadata.n_kv_heads,
            self.metadata.head_dim,
            host,
            mode,
        )
    }

    /// Releases the CPU per-token grow-to-fit step buffer before this decoder
    /// goes back into the cross-request resident cache, so it stays scoped to
    /// one utterance instead of living on indefinitely with the cached
    /// decoder. A no-op on Metal/GPU or when no CPU step ever ran. Mirrors
    /// `firered_llm::llm_transformer::FireRedLlmDecoderRuntime::
    /// release_session_scoped_buffers` (both drive the same shared executor).
    pub(crate) fn release_session_scoped_buffers(&mut self) {
        self.whole_decoder.release_session_scoped_buffers();
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
        capacity: Qwen3AsrKvCacheCapacity,
        control: &std::sync::Arc<crate::api::backend::TranscriptionControl>,
    ) -> Result<MimoLlmPrefillOutput, MimoLlmDecoderError> {
        let token_count = prompt_embeddings.token_count;
        if let Some(final_hidden) = self
            .whole_decoder
            .run_prefill_auto_last_hidden(
                &prompt_embeddings.token_major_values,
                token_count,
                layer_kv_caches,
                capacity,
                self.metadata.rope_theta,
                control,
            )
            .map_err(|error| MimoLlmDecoderError::GraphFailed {
                reason: error.to_string(),
            })?
        {
            if let Some(token_id) = self
                .whole_decoder
                .fused_logits_top1_from_hidden(&final_hidden)
                .map_err(|error| MimoLlmDecoderError::GraphFailed {
                    reason: error.to_string(),
                })?
            {
                return Ok(MimoLlmPrefillOutput {
                    logits: Vec::new(),
                    greedy_token_hint: Some(token_id),
                });
            }
            let logits = self
                .logits_runtime
                .compute_logits_for_last_hidden(&self.logits_head, &final_hidden)
                .map_err(|error| MimoLlmDecoderError::LogitsHeadFailed {
                    reason: error.to_string(),
                })?;
            return Ok(MimoLlmPrefillOutput {
                logits,
                greedy_token_hint: None,
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
        let logits = self
            .logits_runtime
            .compute_logits_for_last_hidden(&self.logits_head, &final_hidden)
            .map_err(|error| MimoLlmDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            })?;
        Ok(MimoLlmPrefillOutput {
            logits,
            greedy_token_hint: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_token_ids_with_audio(
        &mut self,
        token_ids: &[u32],
        audio_rows: &[f32],
        audio_positions: &[usize],
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
        capacity: Qwen3AsrKvCacheCapacity,
        control: &std::sync::Arc<crate::api::backend::TranscriptionControl>,
    ) -> Result<MimoLlmPrefillOutput, MimoLlmDecoderError> {
        if let Some(final_hidden) = self
            .whole_decoder
            .run_token_prefill_auto_last_hidden(
                token_ids,
                audio_rows,
                audio_positions,
                layer_kv_caches,
                capacity,
                self.metadata.rope_theta,
                control,
            )
            .map_err(|error| MimoLlmDecoderError::GraphFailed {
                reason: error.to_string(),
            })?
        {
            if let Some(token_id) = self
                .whole_decoder
                .fused_logits_top1_from_hidden(&final_hidden)
                .map_err(|error| MimoLlmDecoderError::GraphFailed {
                    reason: error.to_string(),
                })?
            {
                return Ok(MimoLlmPrefillOutput {
                    logits: Vec::new(),
                    greedy_token_hint: Some(token_id),
                });
            }
            let logits = self
                .logits_runtime
                .compute_logits_for_last_hidden(&self.logits_head, &final_hidden)
                .map_err(|error| MimoLlmDecoderError::LogitsHeadFailed {
                    reason: error.to_string(),
                })?;
            return Ok(MimoLlmPrefillOutput {
                logits,
                greedy_token_hint: None,
            });
        }
        let token_rows = self
            .token_embedding
            .gather_rows(token_ids)
            .map_err(|error| MimoLlmDecoderError::TokenEmbeddingFailed {
                reason: error.to_string(),
            })?;
        let prompt = build_qwen3_prompt_embeddings_with_audio_positions(
            token_ids.len(),
            audio_positions,
            self.metadata.d_model,
            token_rows,
            audio_rows,
        )
        .map_err(|error| MimoLlmDecoderError::PromptEmbeddingFailed {
            reason: error.to_string(),
        })?;
        self.prefill(&prompt, layer_kv_caches, capacity, control)
    }

    pub(crate) fn decode_step(
        &mut self,
        token_id: u32,
        cache_position: usize,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
        capacity: Qwen3AsrKvCacheCapacity,
    ) -> Result<Vec<f32>, MimoLlmDecoderError> {
        let device_step = self
            .whole_decoder
            .run_token_step_auto(
                token_id,
                cache_position,
                layer_kv_caches,
                capacity,
                self.metadata.rope_theta,
            )
            .map_err(|error| MimoLlmDecoderError::GraphFailed {
                reason: error.to_string(),
            })?;
        let step = match device_step {
            Some(step) => step,
            None => {
                let hidden = self.gather_token_embedding(token_id)?;
                self.whole_decoder
                    .run_step_auto(
                        &hidden,
                        cache_position,
                        layer_kv_caches,
                        capacity,
                        self.metadata.rope_theta,
                    )
                    .map_err(|error| MimoLlmDecoderError::GraphFailed {
                        reason: error.to_string(),
                    })?
            }
        };
        write_layer_kv(
            cache_position,
            1,
            &step.layer_kv,
            self.metadata.n_kv_heads * self.metadata.head_dim,
            layer_kv_caches,
        )?;
        if let Some(logits) = step.fused_logits {
            return Ok(logits);
        }
        self.logits_runtime
            .compute_logits_for_last_hidden(&self.logits_head, &step.hidden)
            .map_err(|error| MimoLlmDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            })
    }

    /// On the resident Metal/GPU reuse graph, return the decoder's device-side
    /// argmax directly. mimo-asr's registered policy has no suppression or
    /// phrase bias, so the shared greedy driver can safely consume this as a
    /// validated `greedy_token_hint`; CPU and any non-reuse backend fall back
    /// to the full host logits path above. Mirrors
    /// `moss_transcribe_diarize::llm_decoder::decode_step_reused_top1`.
    pub(crate) fn decode_step_reused_top1(
        &mut self,
        token_id: u32,
        cache_position: usize,
        layer_kv_caches: &[Qwen3AsrLayerKvCacheState],
        capacity: Qwen3AsrKvCacheCapacity,
    ) -> Result<Option<u32>, MimoLlmDecoderError> {
        if !self.whole_decoder.supports_device_token_embedding()
            || !self.whole_decoder.supports_fused_top1()
        {
            return Ok(None);
        }
        if layer_kv_caches.is_empty() {
            return Err(MimoLlmDecoderError::KvCacheFailed {
                reason: "mimo-asr backbone has no layer KV caches".to_string(),
            });
        }
        let step = self
            .whole_decoder
            .run_token_step_reused_batched_top1(
                &[token_id],
                &[cache_position],
                self.metadata.rope_theta,
                capacity.resident_positions(),
            )
            .map_err(|error| MimoLlmDecoderError::GraphFailed {
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
