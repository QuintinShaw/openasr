//! The Qwen3-0.6B decoder-only LLM stage, reusing `qwen`'s family-agnostic
//! decoder machinery byte-for-byte: `qwen::load_qwen_family_llm_layer_attention_projection_generic`
//! for layer projections (QK-norm present, no attention bias -- the Qwen3
//! parameterization, same `Option` flips `moss_transcribe_diarize::llm_decoder`
//! uses), `qwen::Qwen3AsrLlmWholeDecoderGraphExecutor` for the whole-decoder
//! ggml graph, `qwen::Qwen3AsrLayerKvCacheState` for the host-side per-layer
//! GQA KV cache, and `qwen::Qwen3AsrLlmLogitsHead` /
//! `qwen::MappedTokenEmbeddingTable` for the output/embedding stage. Mirrors
//! `moss_transcribe_diarize::llm_decoder`'s exact shape (both drive the same
//! shared executor with a stock Qwen3-0.6B decoder).

use thiserror::Error;

use crate::ggml_runtime::{GgmlCpuGraphBackend, GgufTensorDataReader, ResolvedFamilyRuntimeInput};
use crate::models::mapped_token_embedding::MappedTokenEmbeddingTable;
use crate::models::qwen::{
    Qwen3AsrHostKvCacheOwner, Qwen3AsrHostKvMode, Qwen3AsrKvCacheCapacity,
    Qwen3AsrLayerKvCacheState, Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadRuntime,
    Qwen3AsrLlmWholeDecoderGraphExecutor, Qwen3AsrPromptEmbeddings, QwenDecoderTail,
    QwenDecoderTailLoadError, QwenPreparedDecoderGraphCompileRequest, QwenWholeDecoderPlan,
    build_qwen3_prompt_embeddings_with_audio_positions,
    compile_qwen_whole_decoder_graph_from_prepared_plan, load_qwen_decoder_tail_from_contract,
    quoted_qwen_decoder_system_memory_bytes,
};

use super::runtime_contract::{
    FUNASR_NANO_RMS_NORM_EPSILON, FUNASR_NANO_ROPE_THETA, FunasrNanoDecoderMetadata,
    funasr_nano_qwen_decoder_contract,
};
#[cfg(test)]
use super::tensor_names::LLM_TOKEN_EMBD_WEIGHT;

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
    let contract =
        funasr_nano_qwen_decoder_contract(metadata).map_err(|error| error.to_string())?;
    quoted_qwen_decoder_system_memory_bytes(reader, &contract, backend)
}

#[derive(Debug, Error)]
pub(crate) enum FunasrNanoDecoderError {
    #[error("funasr-nano decoder tensor read failed: {reason}")]
    TensorReadFailed { reason: String },
    #[error("funasr-nano decoder graph failed: {reason}")]
    GraphFailed { reason: String },
    #[error("funasr-nano decoder token-embedding gather failed: {reason}")]
    TokenEmbeddingFailed { reason: String },
    #[error("funasr-nano decoder prompt embedding failed: {reason}")]
    PromptEmbeddingFailed { reason: String },
    #[error("funasr-nano decoder logits head failed: {reason}")]
    LogitsHeadFailed { reason: String },
    #[error("funasr-nano decoder KV cache write failed: {reason}")]
    KvCacheFailed { reason: String },
    #[error("funasr-nano decoder prefill produced no final hidden state")]
    EmptyPrefillOutput,
}

fn map_tail_load_error(error: QwenDecoderTailLoadError) -> FunasrNanoDecoderError {
    match error {
        QwenDecoderTailLoadError::TokenEmbedding(error) => {
            FunasrNanoDecoderError::TokenEmbeddingFailed {
                reason: error.to_string(),
            }
        }
        other => FunasrNanoDecoderError::LogitsHeadFailed {
            reason: other.to_string(),
        },
    }
}

pub(crate) struct FunasrNanoDecoderRuntime {
    whole_decoder: Qwen3AsrLlmWholeDecoderGraphExecutor,
    logits_head: Qwen3AsrLlmLogitsHead,
    logits_runtime: Qwen3AsrLlmLogitsHeadRuntime,
    token_embedding: MappedTokenEmbeddingTable,
    metadata: FunasrNanoDecoderMetadata,
}

pub(crate) struct FunasrNanoPrefillOutput {
    pub(crate) logits: Vec<f32>,
    pub(crate) greedy_token_hint: Option<u32>,
}

impl FunasrNanoDecoderRuntime {
    pub(crate) fn graph_lane(&self) -> (crate::ggml_runtime::GgmlCpuGraphBackend, bool) {
        self.whole_decoder.graph_lane()
    }

    pub(crate) fn loaded_weight_binding_identity(
        &self,
    ) -> Option<crate::ggml_runtime::GgmlLoadedWeightBindingIdentity> {
        self.whole_decoder.loaded_weight_binding_identity()
    }

    pub(crate) fn new_from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        metadata: FunasrNanoDecoderMetadata,
        resolved_runtime: ResolvedFamilyRuntimeInput,
    ) -> Result<Self, FunasrNanoDecoderError> {
        let backend = resolved_runtime.backend();
        let reader =
            crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight(preflight)
                .map_err(|error| FunasrNanoDecoderError::TensorReadFailed {
                    reason: error.to_string(),
                })?;
        // Decoder fail-closed is admission-time
        // (`validate_funasr_nano_runtime_tensors_with_index`) plus known-name
        // shape checks inside the shared Qwen planner / contract-projected tail
        // loader. Do NOT install a shared-index allowlist here: the whole-
        // decoder graph materializer enumerates every pack tensor through
        // `load_gguf_weight_context_from_preflight`, and FunASR ships a
        // combined encoder+adapter+decoder pack -- a decoder-only allowlist
        // would hide the non-decoder weights and break production load.
        //
        // Bind the Qwen decoder contract exactly once: planner + tail + compile
        // all consume this value (no second geometry/options/names assembly).
        let contract = super::runtime_contract::funasr_nano_qwen_decoder_contract(&metadata)
            .map_err(|error| FunasrNanoDecoderError::TensorReadFailed {
                reason: error.to_string(),
            })?;
        let decoder_plan =
            QwenWholeDecoderPlan::for_qwen_family(&reader, &contract).map_err(|error| {
                FunasrNanoDecoderError::TensorReadFailed {
                    reason: error.to_string(),
                }
            })?;
        let QwenDecoderTail {
            logits_head,
            token_embedding,
        } = load_qwen_decoder_tail_from_contract(
            &reader,
            &contract,
            FUNASR_NANO_RMS_NORM_EPSILON,
            backend,
        )
        .map_err(map_tail_load_error)?;
        // Structural Prepared Graph Plan adoption: plan is host-owned metadata
        // built at prepare; the shared compile seam is the only backend
        // materialize path (same entry MOSS-TD uses). No performance claim is
        // implied, and no family-local graph assembly remains here.
        let whole_decoder = compile_qwen_whole_decoder_graph_from_prepared_plan(
            QwenPreparedDecoderGraphCompileRequest {
                plan: &decoder_plan,
                preflight,
                rms_norm_epsilon: FUNASR_NANO_RMS_NORM_EPSILON,
                fused_logits_head: logits_head.fused_top1_spec(),
                token_embedding: token_embedding.device_graph_spec(),
                resolved_runtime,
            },
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_token_ids_with_audio(
        &mut self,
        token_ids: &[u32],
        audio_rows: &[f32],
        audio_positions: &[usize],
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
        capacity: Qwen3AsrKvCacheCapacity,
        control: &std::sync::Arc<crate::api::backend::TranscriptionControl>,
    ) -> Result<FunasrNanoPrefillOutput, FunasrNanoDecoderError> {
        if let Some(final_hidden) = self
            .whole_decoder
            .run_token_prefill_auto_last_hidden(
                token_ids,
                audio_rows,
                audio_positions,
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
        let token_rows = self
            .token_embedding
            .gather_rows(token_ids)
            .map_err(|error| FunasrNanoDecoderError::TokenEmbeddingFailed {
                reason: error.to_string(),
            })?;
        let prompt = build_qwen3_prompt_embeddings_with_audio_positions(
            token_ids.len(),
            audio_positions,
            self.metadata.d_model,
            token_rows,
            audio_rows,
        )
        .map_err(|error| FunasrNanoDecoderError::PromptEmbeddingFailed {
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
    ) -> Result<Vec<f32>, FunasrNanoDecoderError> {
        let device_step = self
            .whole_decoder
            .run_token_step_auto(
                token_id,
                cache_position,
                layer_kv_caches,
                capacity,
                FUNASR_NANO_ROPE_THETA,
            )
            .map_err(|error| FunasrNanoDecoderError::GraphFailed {
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
                        FUNASR_NANO_ROPE_THETA,
                    )
                    .map_err(|error| FunasrNanoDecoderError::GraphFailed {
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
        if !self.whole_decoder.supports_device_token_embedding()
            || !self.whole_decoder.supports_fused_top1()
        {
            return Ok(None);
        }
        if layer_kv_caches.is_empty() {
            return Err(FunasrNanoDecoderError::KvCacheFailed {
                reason: "funasr-nano decoder has no layer KV caches".to_string(),
            });
        }
        let step = self
            .whole_decoder
            .run_token_step_reused_batched_top1(
                &[token_id],
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
                .map_err(|reason| FunasrNanoDecoderError::KvCacheFailed {
                    reason: format!(
                        "layer {layer_index} token {absolute_position} write failed: {reason}"
                    ),
                })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod trace_tests {
    use super::*;
    use crate::ggml_runtime::GgmlCpuGraphBackend;
    use crate::models::funasr_nano::runtime_contract::{
        FunasrNanoDecoderMetadata, funasr_nano_decoder_read_guard,
        funasr_nano_decoder_tensor_descriptors, funasr_nano_qwen_decoder_contract,
        parse_funasr_nano_decoder_metadata,
    };
    use crate::models::tensor_binding::{
        assert_trace_matches_descriptor_set, project_fixture_tensors,
    };
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};

    fn tiny_decoder_metadata() -> FunasrNanoDecoderMetadata {
        FunasrNanoDecoderMetadata {
            n_layers: 1,
            d_model: 16,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 32,
            vocab_size: 64,
            max_positions: 128,
            chatml_im_start_token_id: 1,
            chatml_im_end_token_id: 2,
            endoftext_token_id: 0,
        }
    }

    /// Logical-binding read-set evidence for the decoder half: run the real
    /// plan, logits, and embedding loaders over a synthetic pack projected
    /// from the decoder contract itself, with the index access trace enabled,
    /// and assert the traced read set equals the decoder descriptor set name
    /// for name and shape for shape.
    ///
    /// This covers the logical loaders only. It does NOT exercise the physical
    /// whole-decoder weight-context enumeration
    /// (`load_gguf_weight_context_from_preflight`), which walks every pack
    /// tensor -- that path is covered by
    /// `combo_pack_decoder_new_from_preflight_succeeds`. Encoder and adaptor
    /// halves have their own trace certificates
    /// (`encoder_graph::trace_tests`, `adapter_graph::trace_tests`); this is
    /// not a whole-family access-trace claim.
    #[test]
    fn decoder_logical_loader_read_trace_equals_the_contract_descriptors() {
        let metadata = tiny_decoder_metadata();
        let descriptors = funasr_nano_decoder_tensor_descriptors(&metadata)
            .expect("tiny decoder geometry must expand");
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("funasr-nano-decoder-trace.oasr");
        let mut spec = TinyGgufFixtureSpec::new(std::collections::BTreeMap::new());
        for (name, dims) in project_fixture_tensors(&descriptors) {
            spec = spec.with_tensor_shape(name, dims);
        }
        write_tiny_gguf_runtime_source(&path, &spec).expect("write trace pack");

        let reader = crate::ggml_runtime::GgufTensorDataReader::from_path(&path).expect("reader");
        reader.tensor_index().enable_access_trace();

        // Same single-bind production shape: one contract value drives plan + tail.
        let contract = funasr_nano_qwen_decoder_contract(&metadata).expect("bind decoder contract");
        QwenWholeDecoderPlan::for_qwen_family(&reader, &contract).expect("plan decoder");
        load_qwen_decoder_tail_from_contract(
            &reader,
            &contract,
            FUNASR_NANO_RMS_NORM_EPSILON,
            GgmlCpuGraphBackend::Cpu,
        )
        .expect("load decoder tail");

        assert_trace_matches_descriptor_set(&reader.tensor_index().access_trace(), &descriptors);
    }

    /// Local contract-name guard: the decoder descriptor set is the
    /// authoritative logical read list. Shared Qwen loaders do not take a
    /// guard parameter, so this stays a local `contains` check rather than a
    /// shared-index policy.
    #[test]
    fn decoder_read_guard_lists_only_contract_names() {
        let metadata = tiny_decoder_metadata();
        let guard = funasr_nano_decoder_read_guard(&metadata).expect("decoder guard");
        assert!(
            !guard.contains("off.contract.weight"),
            "off-contract names must not be in the decoder read set"
        );
        assert!(
            guard.contains(LLM_TOKEN_EMBD_WEIGHT),
            "required contract tensors must remain listed"
        );
        for descriptor in funasr_nano_decoder_tensor_descriptors(&metadata)
            .expect("tiny decoder geometry must expand")
        {
            assert!(
                guard.contains(&descriptor.tensor_name),
                "descriptor {} must be listed",
                descriptor.tensor_name
            );
        }
    }

    /// Production regression gate: FunASR ships a combined
    /// encoder+adapter+decoder pack. `new_from_preflight` must succeed on that
    /// combo pack because the whole-decoder graph materializer enumerates every
    /// pack tensor (not only decoder contract names).
    #[test]
    fn combo_pack_decoder_new_from_preflight_succeeds() {
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("funasr-nano-combo-decoder.oasr");
        let spec = TinyGgufFixtureSpec::funasr_nano_oasr_v1_runtime_ready("funasr-nano-combo");
        write_tiny_gguf_runtime_source(&path, &spec).expect("write combo pack");
        let preflight = crate::ggml_runtime::load_runtime_source_metadata_and_tensor_index(&path)
            .expect("combo pack preflight");
        let decoder = parse_funasr_nano_decoder_metadata(preflight.metadata())
            .expect("parse decoder metadata from combo pack");
        let resolved_runtime = ResolvedFamilyRuntimeInput::resolve(
            Some(crate::ggml_runtime::RequestBackendPreference::CpuOnly),
            crate::ggml_runtime::AutoGpuPolicy::AllBackends,
        );
        FunasrNanoDecoderRuntime::new_from_preflight(&preflight, decoder, resolved_runtime)
            .expect("combo pack must load through whole-decoder weight context");
    }

    /// Missing a decoder-contract tensor still fails closed through the real
    /// production constructor (planner / logits / embedding known-name checks).
    #[test]
    fn decoder_new_from_preflight_fails_closed_when_a_contract_tensor_is_absent() {
        let metadata = tiny_decoder_metadata();
        let mut descriptors = funasr_nano_decoder_tensor_descriptors(&metadata)
            .expect("tiny decoder geometry must expand");
        // Drop one required layer weight.
        descriptors.retain(|d| d.tensor_name != "blk.0.attn_q.weight");
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("funasr-nano-decoder-missing.oasr");
        let mut spec = TinyGgufFixtureSpec::new(std::collections::BTreeMap::new());
        for (name, dims) in project_fixture_tensors(&descriptors) {
            // Also stamp a few off-contract names so the pack is non-empty beyond
            // the decoder half -- the fail path must still be the missing contract
            // tensor, not an empty-pack edge case.
            spec = spec.with_tensor_shape(name, dims);
        }
        spec = spec.with_tensor_shape("enc.blk.0.attn_norm.weight".to_string(), vec![16]);
        write_tiny_gguf_runtime_source(&path, &spec).expect("write pack");

        let preflight = crate::ggml_runtime::load_runtime_source_metadata_and_tensor_index(&path)
            .expect("preflight");
        let resolved_runtime = ResolvedFamilyRuntimeInput::resolve(
            Some(crate::ggml_runtime::RequestBackendPreference::CpuOnly),
            crate::ggml_runtime::AutoGpuPolicy::AllBackends,
        );
        let result =
            FunasrNanoDecoderRuntime::new_from_preflight(&preflight, metadata, resolved_runtime);
        let error = match result {
            Ok(_) => panic!("missing attn_q must fail closed"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(
            message.contains("attn_q") || message.contains("missing"),
            "unexpected error: {message}"
        );
    }
}
