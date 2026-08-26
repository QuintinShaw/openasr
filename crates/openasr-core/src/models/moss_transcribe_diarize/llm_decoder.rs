//! The Qwen3-0.6B decoder-only LLM stage, reusing `qwen`'s family-agnostic
//! decoder machinery byte-for-byte: `qwen::load_qwen_family_llm_layer_attention_projection_generic`
//! for layer projections (QK-norm present, no attention bias -- the inverse
//! of `firered_llm`'s Qwen2 parameterization, but the SAME shared loader,
//! just with the `Option` fields flipped), `qwen::Qwen3AsrLlmWholeDecoderGraphExecutor`
//! for the whole-decoder ggml graph, `qwen::Qwen3AsrLayerKvCacheState` for the
//! host-side per-layer GQA KV cache, and `qwen::Qwen3AsrLlmLogitsHead` /
//! `qwen::MappedTokenEmbeddingTable` for the output/embedding stage.
//! Mirrors `firered_llm::llm_transformer`'s exact shape (see that module's
//! doc comment for why this crate does not replicate qwen's GPU-tuned
//! prefill-chunk/persistent-session machinery here: correctness-first single-
//! shot decode, GPU perf tuning is out of scope this stage).

use std::sync::Arc;

use thiserror::Error;

use crate::ggml_runtime::ResolvedFamilyRuntimeInput;
use crate::models::mapped_token_embedding::MappedTokenEmbeddingTable;
use crate::models::qwen::{
    Qwen3AsrHostKvCacheOwner, Qwen3AsrHostKvMode, Qwen3AsrKvCacheCapacity,
    Qwen3AsrLayerKvCacheState, Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadRuntime,
    Qwen3AsrLlmWholeDecoderGraphExecutor, Qwen3AsrPromptEmbeddings,
    QwenPreparedDecoderGraphCompileRequest, QwenWholeDecoderPlan,
    build_qwen3_prompt_embeddings_with_audio_positions,
    compile_qwen_whole_decoder_graph_from_prepared_plan_with_config_and_native_gqa,
};

use super::runtime_contract::{
    MOSS_TD_RMS_NORM_EPSILON, MOSS_TD_ROPE_THETA, MossTdDecoderMetadata,
};

/// Host-path prefill segment width (see [`MossTdDecoderRuntime::prefill`]). A
/// prompt longer than this is fed to the decoder in segments of this many
/// tokens so the attention working set stays `chunk x total` instead of
/// `total x total`. Chosen above every sub-40s clip's prompt length so short
/// audio (jfk/en_zh and the 40s regression clip) keeps the byte-for-byte
/// single-shot path, while a multi-minute prompt still splits into a handful of
/// segments that fit a modest memory budget.
const MOSS_TD_PREFILL_CHUNK_TOKENS: usize = 512;

#[derive(Debug, Error)]
pub(crate) enum MossTdDecoderError {
    #[error("moss-transcribe-diarize decoder graph failed: {reason}")]
    GraphFailed { reason: String },
    #[error("moss-transcribe-diarize decoder token-embedding gather failed: {reason}")]
    TokenEmbeddingFailed { reason: String },
    #[error("moss-transcribe-diarize decoder prompt embedding failed: {reason}")]
    PromptEmbeddingFailed { reason: String },
    #[error("moss-transcribe-diarize decoder logits head failed: {reason}")]
    LogitsHeadFailed { reason: String },
    #[error("moss-transcribe-diarize decoder KV cache write failed: {reason}")]
    KvCacheFailed { reason: String },
    #[error("moss-transcribe-diarize decoder prefill produced no final hidden state")]
    EmptyPrefillOutput,
}

/// The Qwen3-0.6B decoder-only stack for one loaded pack: layer weights +
/// logits head + token embedding table (tied to the same tensor as the
/// logits head's output weight -- `config.tie_word_embeddings=true`, see
/// `package_import`'s module doc), ready to prefill/decode against a fresh
/// set of per-utterance KV caches (`new_kv_caches`).
pub(crate) struct MossTdDecoderRuntime {
    whole_decoder: Qwen3AsrLlmWholeDecoderGraphExecutor,
    logits_head: Arc<Qwen3AsrLlmLogitsHead>,
    logits_runtime: Qwen3AsrLlmLogitsHeadRuntime,
    token_embedding: Arc<MappedTokenEmbeddingTable>,
    metadata: MossTdDecoderMetadata,
}

pub(crate) struct MossTdPrefillOutput {
    pub(crate) logits: Vec<f32>,
    pub(crate) greedy_token_hint: Option<u32>,
}

impl MossTdDecoderRuntime {
    pub(crate) fn take_compute_evidence(
        &mut self,
    ) -> Option<crate::ggml_runtime::GgmlSelectionEvidenceRef> {
        self.logits_runtime.take_compute_evidence()
    }

    pub(crate) fn graph_lanes(
        &self,
    ) -> (
        (crate::ggml_runtime::GgmlCpuGraphBackend, bool),
        Option<(crate::ggml_runtime::GgmlCpuGraphBackend, bool)>,
    ) {
        (
            self.whole_decoder.graph_lane(),
            self.logits_runtime.graph_lane(),
        )
    }

    pub(crate) fn loaded_weight_binding_identity(
        &self,
    ) -> Option<crate::ggml_runtime::GgmlLoadedWeightBindingIdentity> {
        self.whole_decoder.loaded_weight_binding_identity()
    }

    /// Quote the actor-local graph handles from the already-prepared plan.
    ///
    /// Accepting the plan instead of a raw layer count keeps runtime admission
    /// on the same source of truth as materialization.
    pub(crate) fn quoted_resident_system_memory_bytes(
        plan: &QwenWholeDecoderPlan,
    ) -> Result<u64, String> {
        Qwen3AsrLlmWholeDecoderGraphExecutor::quoted_retained_system_memory_bytes(
            plan.layer_count(),
        )
    }

    pub(crate) fn resident_system_memory_bytes(&self) -> Result<u64, String> {
        self.whole_decoder.retained_system_memory_bytes()
    }

    pub(crate) fn new_with_prepared_state_from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        metadata: MossTdDecoderMetadata,
        decoder_plan: Arc<QwenWholeDecoderPlan>,
        logits_head: Arc<Qwen3AsrLlmLogitsHead>,
        token_embedding: Arc<MappedTokenEmbeddingTable>,
        graph_config: crate::ggml_runtime::GgmlCpuGraphConfig,
        resolved_runtime: ResolvedFamilyRuntimeInput,
    ) -> Result<Self, MossTdDecoderError> {
        // Structural Prepared Graph Plan adoption: host prepare already owns
        // the typed plan and compiles through the shared seam (same entry
        // FunASR-Nano uses). Performance remains an evidence-only claim.
        let whole_decoder =
            compile_qwen_whole_decoder_graph_from_prepared_plan_with_config_and_native_gqa(
                QwenPreparedDecoderGraphCompileRequest {
                    plan: &decoder_plan,
                    preflight,
                    rms_norm_epsilon: MOSS_TD_RMS_NORM_EPSILON,
                    fused_logits_head: logits_head.fused_top1_spec(),
                    token_embedding: token_embedding.device_graph_spec(),
                    resolved_runtime,
                },
                graph_config,
            )
            .map_err(|error| MossTdDecoderError::GraphFailed {
                reason: error.to_string(),
            })?;
        let logits_runtime = logits_head
            .new_runtime_with_graph_config(graph_config)
            .map_err(|error| MossTdDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            })?;
        Ok(Self {
            whole_decoder,
            logits_head,
            logits_runtime,
            token_embedding,
            metadata,
        })
    }

    pub(crate) fn backend_label(&self) -> String {
        self.whole_decoder.backend_label()
    }

    pub(crate) fn uses_native_gqa(&self) -> bool {
        self.whole_decoder.uses_native_gqa()
    }

    pub(crate) fn supports_graph_reuse(&self) -> bool {
        self.whole_decoder.supports_graph_reuse()
    }

    /// Frees this decoder's per-token grow-to-fit host step buffer. Call
    /// after every decode, right before this runtime goes back into
    /// `executor.rs`'s owner-actor checkout pool -- without it, a session-
    /// scoped allocation sized for one utterance would otherwise ride along
    /// on the cached runtime into the next, unrelated request. Mirrors
    /// `qwen::ggml_executor`'s identical call around its own resident
    /// whole-decoder cache.
    pub(crate) fn release_session_scoped_buffers(&mut self) {
        self.whole_decoder.release_session_scoped_buffers();
    }

    /// Allocate only the exact current-invocation host history. The executor
    /// validates both logical and stable resident spans against the family cap
    /// before this call; no allocation path clamps or substitutes that cap.
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
            "moss-transcribe-diarize.decoder.self-kv.host",
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
    ) -> Result<Vec<f32>, MossTdDecoderError> {
        self.token_embedding
            .gather_rows(&[token_id])
            .map_err(|error| MossTdDecoderError::TokenEmbeddingFailed {
                reason: error.to_string(),
            })
    }

    /// Run the entire ChatML+audio prompt as one causal prefill pass, seeding
    /// the decoder's KV history with every prompt token's K/V, and return the
    /// logits row for the token immediately following the prompt.
    ///
    /// Three paths, all numerically equivalent (causal attention over the same
    /// KV order):
    /// - Metal/single-GPU reuse: `run_prefill_auto_last_hidden` seeds the
    ///   persistent decode graph's resident-KV arena in one batched compute so
    ///   `decode_step`'s reused graph can attend over the prompt (see the branch
    ///   below).
    /// - Host path, long prompt: split into `MOSS_TD_PREFILL_CHUNK_TOKENS`-token
    ///   segments fed in KV order, so the single-shot graph's
    ///   `token_count x token_count` attention working set stays `chunk x total`
    ///   -- a secondary memory lever for multi-minute clips (mirrors qwen3-asr's
    ///   own `ggml_executor` prefill-chunking).
    /// - Host path, short prompt (every sub-40s clip, incl. jfk/en_zh): a single
    ///   bulk `run_prefill`.
    pub(crate) fn prefill(
        &mut self,
        prompt_embeddings: &Qwen3AsrPromptEmbeddings,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
        capacity: Qwen3AsrKvCacheCapacity,
        control: &std::sync::Arc<crate::api::backend::TranscriptionControl>,
    ) -> Result<MossTdPrefillOutput, MossTdDecoderError> {
        let token_count = prompt_embeddings.token_count;
        // On a backend with persistent decode-graph reuse (Metal/single-GPU),
        // seed the resident-KV arena in one batched compute instead of the bulk
        // host-cache `run_prefill` below. `decode_step` reuses that same
        // persistent graph and can only see a prompt token's K/V if the prompt
        // flowed through it too (see `run_prefill_auto_last_hidden`'s doc), so
        // mixing bulk prefill with reuse decode would attend over an empty KV
        // history for the whole prompt span. Returns `None` on the host/CPU
        // path, which falls through to the chunked/single-shot prefill.
        if let Some(final_hidden) = self
            .whole_decoder
            .run_prefill_auto_last_hidden(
                &prompt_embeddings.token_major_values,
                token_count,
                layer_kv_caches,
                capacity,
                MOSS_TD_ROPE_THETA,
                control,
            )
            .map_err(|error| MossTdDecoderError::GraphFailed {
                reason: error.to_string(),
            })?
        {
            if let Some(token_id) = self
                .whole_decoder
                .fused_logits_top1_from_hidden(&final_hidden)
                .map_err(|error| MossTdDecoderError::GraphFailed {
                    reason: error.to_string(),
                })?
            {
                return Ok(MossTdPrefillOutput {
                    logits: Vec::new(),
                    greedy_token_hint: Some(token_id),
                });
            }
            let logits = self
                .logits_runtime
                .compute_logits_for_last_hidden(&self.logits_head, &final_hidden)
                .map_err(|error| MossTdDecoderError::LogitsHeadFailed {
                    reason: error.to_string(),
                })?;
            return Ok(MossTdPrefillOutput {
                logits,
                greedy_token_hint: None,
            });
        }
        let host_chunked = self
            .whole_decoder
            .safe_host_cache_prefill_chunk_size_for(token_count)
            .is_some();
        let final_hidden = if host_chunked && token_count > MOSS_TD_PREFILL_CHUNK_TOKENS {
            self.prefill_chunked(prompt_embeddings, layer_kv_caches)?
        } else {
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
            self.write_prefill_outputs(0, token_count, &step, layer_kv_caches)?
        };
        let logits = self
            .logits_runtime
            .compute_logits_for_last_hidden(&self.logits_head, &final_hidden)
            .map_err(|error| MossTdDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            })?;
        Ok(MossTdPrefillOutput {
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
    ) -> Result<MossTdPrefillOutput, MossTdDecoderError> {
        if let Some(final_hidden) = self
            .whole_decoder
            .run_token_prefill_auto_last_hidden(
                token_ids,
                audio_rows,
                audio_positions,
                layer_kv_caches,
                capacity,
                MOSS_TD_ROPE_THETA,
                control,
            )
            .map_err(|error| MossTdDecoderError::GraphFailed {
                reason: error.to_string(),
            })?
        {
            if let Some(token_id) = self
                .whole_decoder
                .fused_logits_top1_from_hidden(&final_hidden)
                .map_err(|error| MossTdDecoderError::GraphFailed {
                    reason: error.to_string(),
                })?
            {
                return Ok(MossTdPrefillOutput {
                    logits: Vec::new(),
                    greedy_token_hint: Some(token_id),
                });
            }
            let logits = self
                .logits_runtime
                .compute_logits_for_last_hidden(&self.logits_head, &final_hidden)
                .map_err(|error| MossTdDecoderError::LogitsHeadFailed {
                    reason: error.to_string(),
                })?;
            return Ok(MossTdPrefillOutput {
                logits,
                greedy_token_hint: None,
            });
        }
        let token_rows = self
            .token_embedding
            .gather_rows(token_ids)
            .map_err(|error| MossTdDecoderError::TokenEmbeddingFailed {
                reason: error.to_string(),
            })?;
        let prompt = build_qwen3_prompt_embeddings_with_audio_positions(
            token_ids.len(),
            audio_positions,
            self.metadata.d_model,
            token_rows,
            audio_rows,
        )
        .map_err(|error| MossTdDecoderError::PromptEmbeddingFailed {
            reason: error.to_string(),
        })?;
        self.prefill(&prompt, layer_kv_caches, capacity, control)
    }

    /// Segmented prefill (see [`Self::prefill`]): feed the prompt to the decoder
    /// in `MOSS_TD_PREFILL_CHUNK_TOKENS`-token segments, each attending to the
    /// KV history the prior segments wrote, and return the final segment's last
    /// hidden state. Every segment's K/V is written into `layer_kv_caches`
    /// before the next segment runs, exactly reproducing the single-shot pass's
    /// KV order.
    fn prefill_chunked(
        &mut self,
        prompt_embeddings: &Qwen3AsrPromptEmbeddings,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
    ) -> Result<Vec<f32>, MossTdDecoderError> {
        let token_count = prompt_embeddings.token_count;
        let hidden_size = self.metadata.d_model;
        let mut position_offset = 0usize;
        let mut final_hidden: Option<Vec<f32>> = None;
        while position_offset < token_count {
            let chunk_len = (token_count - position_offset).min(MOSS_TD_PREFILL_CHUNK_TOKENS);
            let hidden_start = position_offset
                .checked_mul(hidden_size)
                .ok_or(MossTdDecoderError::EmptyPrefillOutput)?;
            let hidden_end = position_offset
                .checked_add(chunk_len)
                .and_then(|end| end.checked_mul(hidden_size))
                .ok_or(MossTdDecoderError::EmptyPrefillOutput)?;
            let total_token_count = position_offset + chunk_len;
            let chunk_values = prompt_embeddings
                .token_major_values
                .get(hidden_start..hidden_end)
                .ok_or(MossTdDecoderError::EmptyPrefillOutput)?;
            let step = self
                .whole_decoder
                .run_prefill_chunk(
                    chunk_values,
                    chunk_len,
                    position_offset,
                    total_token_count,
                    layer_kv_caches,
                    MOSS_TD_ROPE_THETA,
                )
                .map_err(|error| MossTdDecoderError::GraphFailed {
                    reason: error.to_string(),
                })?;
            final_hidden = Some(self.write_prefill_outputs(
                position_offset,
                chunk_len,
                &step,
                layer_kv_caches,
            )?);
            position_offset = total_token_count;
        }
        final_hidden.ok_or(MossTdDecoderError::EmptyPrefillOutput)
    }

    /// Run one incremental decode step for `token_id` at `cache_position`,
    /// updating `layer_kv_caches`, and return the logits row for the NEXT
    /// token.
    pub(crate) fn decode_step(
        &mut self,
        token_id: u32,
        cache_position: usize,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
        capacity: Qwen3AsrKvCacheCapacity,
    ) -> Result<Vec<f32>, MossTdDecoderError> {
        // `run_step_auto` transparently reuses the persistent decode graph on
        // the Metal/single-GPU lane (avoiding the per-token graph rebuild whose
        // growing-KV re-upload and re-allocation is what exhausts Metal's
        // wired memory over a long decode); CPU stays on the per-token
        // `run_step` rebuild, byte-identical to before. The host KV write
        // below keeps the host cache in sync for the CPU (and any non-reuse)
        // path; on the reuse path `step.layer_kv` comes back empty by design
        // (see `write_layer_kv`'s doc) and the write below is a real no-op,
        // not merely a harmless-looking one -- `write_layer_kv` must treat an
        // empty slice as intentional rather than a count-mismatch error.
        let device_step = self
            .whole_decoder
            .run_token_step_auto(
                token_id,
                cache_position,
                layer_kv_caches,
                capacity,
                MOSS_TD_ROPE_THETA,
            )
            .map_err(|error| MossTdDecoderError::GraphFailed {
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
                        MOSS_TD_ROPE_THETA,
                    )
                    .map_err(|error| MossTdDecoderError::GraphFailed {
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
            .map_err(|error| MossTdDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            })
    }

    /// On the resident Metal/GPU reuse graph, return the decoder's device-side
    /// argmax directly. MOSS's registered policy has no suppression or phrase
    /// bias, so the shared greedy driver can safely consume this as a validated
    /// `greedy_token_hint`; CPU and any non-reuse backend fall back to the full
    /// host logits path above.
    pub(crate) fn decode_step_reused_top1(
        &mut self,
        token_id: u32,
        cache_position: usize,
        layer_kv_caches: &[Qwen3AsrLayerKvCacheState],
        capacity: Qwen3AsrKvCacheCapacity,
    ) -> Result<Option<u32>, MossTdDecoderError> {
        if !self.whole_decoder.supports_device_token_embedding()
            || !self.whole_decoder.supports_fused_top1()
        {
            return Ok(None);
        }
        if layer_kv_caches.is_empty() {
            return Err(MossTdDecoderError::KvCacheFailed {
                reason: "moss-transcribe-diarize decoder has no layer KV caches".to_string(),
            });
        }
        let step = self
            .whole_decoder
            .run_token_step_reused_batched_top1(
                &[token_id],
                &[cache_position],
                MOSS_TD_ROPE_THETA,
                capacity.resident_positions(),
            )
            .map_err(|error| MossTdDecoderError::GraphFailed {
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
    // `run_step_reused` (the Metal/single-GPU decode-graph-reuse path) returns
    // an intentionally EMPTY `layer_kv`: its K/V lives resident in the
    // persistent decode graph's device-side arena, never read back to the
    // host, so there is nothing to mirror into `layer_kv_caches` -- exactly
    // mirroring qwen3-asr's own decode step, whose `step.layer_kv.iter()`
    // loop (`qwen::ggml_executor`'s `decode_step`) simply iterates zero times
    // on that path instead of asserting a count. Treating empty as a no-op
    // here (rather than failing the strict count check below) is what makes
    // `decode_step`'s call into this function byte-for-byte the same
    // reuse-tolerant contract; every OTHER caller (prefill, and decode's own
    // non-reuse `run_step` rebuild) always produces one entry per layer, so a
    // non-empty-but-wrong count still fails closed instead of silently
    // dropping/misaligning KV writes.
    if layer_kv.is_empty() {
        return Ok(());
    }
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
