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
//! tuning (`qwen::llm_transformer`'s `safe_*_prefill_chunk_size_for`): that
//! exists to squeeze ROCm/CUDA prefill latency for a shipped, GPU-tuned
//! family, and FireRedASR2-LLM's stage-4 goal is a correct, single-shot
//! CPU/Metal transcription path (the upstream 40s hard cap keeps prompts
//! short -- well under any chunking threshold), so prefill always runs the
//! plain per-chunk path here.
//!
//! Single-token decode, however, DOES go through
//! `Qwen3AsrLlmWholeDecoderGraphExecutor::run_step_auto`, which transparently
//! reuses the persistent decode graph on the Metal/single-GPU lane: an 8B,
//! 28-layer decoder rebuilding its whole graph every token makes host graph
//! construction (CPU-bound) dominate over Metal compute, starving the GPU
//! (low utilization, one CPU core pegged). That is a generic property of any
//! large LLM-decoder-stage family driving this shared executor, not a
//! qwen-specific GPU tuning knob, so it is on by default here exactly as it
//! is for qwen (see `run_step_auto`'s doc comment for the CPU-vs-GPU
//! eligibility rule).

use thiserror::Error;

use crate::ggml_runtime::{GgmlCpuGraphBackend, GgufTensorDataReader};
use crate::models::mapped_token_embedding::MappedTokenEmbeddingTable;
use crate::models::qwen::{
    Qwen3AsrHostKvCacheOwner, Qwen3AsrHostKvMode, Qwen3AsrKvCacheCapacity,
    Qwen3AsrLayerKvCacheState, Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadRuntime,
    Qwen3AsrLlmWholeDecoderGraphExecutor, Qwen3AsrPromptEmbeddings, QwenDecoderTail,
    QwenDecoderTailLoadError, QwenPreparedDecoderGraphCompileRequest, QwenWholeDecoderPlan,
    compile_qwen_whole_decoder_graph_from_prepared_plan, load_qwen_decoder_tail_from_contract,
    quoted_qwen_decoder_system_memory_bytes,
};
#[cfg(test)]
use crate::models::qwen::{
    Qwen3AsrLlmLayerAttentionProjection, load_qwen_family_llm_layer_attention_projection_generic,
};

#[cfg(test)]
use super::runtime_contract::firered_llm_qwen_family_layer_names;
use super::runtime_contract::{
    FIRERED_LLM_RMS_NORM_EPSILON, FIRERED_LLM_ROPE_THETA, FireRedLlmDecoderMetadata,
    firered_llm_qwen_decoder_contract,
};

/// Quotes the host-memory shape retained by one FireRed-LLM decoder actor.
///
/// The actor owns only Rust-side graph handles and the small f32 fallback
/// matrices. GGUF tensor payloads are views into the request's already-open
/// mmap, so their file-backed bytes are deliberately absent from this quote;
/// only the payload metadata vectors/strings are counted. The native ggml
/// arenas and backend allocations are admitted by their graph constructors,
/// not duplicated here.
///
/// `peak_bytes` differs from `retained_bytes` only for a quantized
/// hidden-major token embedding. That compatibility path first materializes a
/// source f32 matrix and then transposes it into the retained token-major
/// matrix, so both matrices are live during construction. All arithmetic is
/// checked because malformed GGUF dimensions must fail closed rather than
/// wrap into an under-quote.
pub(crate) fn quoted_firered_llm_decoder_system_memory_bytes(
    reader: &GgufTensorDataReader,
    metadata: &FireRedLlmDecoderMetadata,
    backend: GgmlCpuGraphBackend,
) -> Result<(u64, u64), String> {
    let contract =
        firered_llm_qwen_decoder_contract(metadata).map_err(|error| error.to_string())?;
    quoted_qwen_decoder_system_memory_bytes(reader, &contract, backend)
}

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

fn map_tail_load_error(error: QwenDecoderTailLoadError) -> FireRedLlmDecoderError {
    match error {
        QwenDecoderTailLoadError::TokenEmbedding(error) => {
            FireRedLlmDecoderError::TokenEmbeddingFailed {
                reason: error.to_string(),
            }
        }
        other => FireRedLlmDecoderError::LogitsHeadFailed {
            reason: other.to_string(),
        },
    }
}

/// The Qwen2 decoder-only stack for one loaded pack: layer weights + logits
/// head + token embedding table, ready to prefill/decode against a fresh set
/// of per-utterance KV caches (`new_kv_caches`).
pub(crate) struct FireRedLlmDecoderRuntime {
    whole_decoder: Qwen3AsrLlmWholeDecoderGraphExecutor,
    logits_head: Qwen3AsrLlmLogitsHead,
    logits_runtime: Qwen3AsrLlmLogitsHeadRuntime,
    token_embedding: MappedTokenEmbeddingTable,
    metadata: FireRedLlmDecoderMetadata,
}

/// Prefill output for the shared greedy driver's step 0: the host logits row
/// for the first generated token, or (on the fused Metal/GPU lane) a device
/// argmax hint with no host row. Mirrors
/// `moss_transcribe_diarize::llm_decoder::MossTdPrefillOutput`.
pub(crate) struct FireRedLlmPrefillOutput {
    pub(crate) logits: Vec<f32>,
    pub(crate) greedy_token_hint: Option<u32>,
}

impl FireRedLlmDecoderRuntime {
    pub(crate) fn new_from_preflight(
        preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
        metadata: FireRedLlmDecoderMetadata,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, FireRedLlmDecoderError> {
        let reader =
            crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight(preflight)
                .map_err(|error| FireRedLlmDecoderError::TensorReadFailed {
                    reason: error.to_string(),
                })?;
        // Bind the Qwen decoder contract exactly once for plan + tail + compile.
        let contract = firered_llm_qwen_decoder_contract(&metadata).map_err(|error| {
            FireRedLlmDecoderError::TensorReadFailed {
                reason: error.to_string(),
            }
        })?;
        let decoder_plan =
            QwenWholeDecoderPlan::for_qwen_family(&reader, &contract).map_err(|error| {
                FireRedLlmDecoderError::TensorReadFailed {
                    reason: error.to_string(),
                }
            })?;
        let QwenDecoderTail {
            logits_head,
            token_embedding,
        } = load_qwen_decoder_tail_from_contract(
            &reader,
            &contract,
            FIRERED_LLM_RMS_NORM_EPSILON,
            backend,
        )
        .map_err(map_tail_load_error)?;
        let whole_decoder = compile_qwen_whole_decoder_graph_from_prepared_plan(
            QwenPreparedDecoderGraphCompileRequest {
                plan: &decoder_plan,
                preflight,
                rms_norm_epsilon: FIRERED_LLM_RMS_NORM_EPSILON,
                fused_logits_head: logits_head.fused_top1_spec(),
                backend,
            },
        )
        .map_err(|error| FireRedLlmDecoderError::GraphFailed {
            reason: error.to_string(),
        })?;
        let logits_runtime = logits_head.new_runtime(backend).map_err(|error| {
            FireRedLlmDecoderError::LogitsHeadFailed {
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

    /// `"<backend-kind>:<ggml backend name>"` for the Qwen2 decoder graph
    /// (e.g. `Metal:Metal` or `Cpu:CPU`), for perf diagnostics -- surfaced
    /// through the executor's `OPENASR_FIRERED_LLM_PROFILE` log line so a
    /// maintainer can confirm which backend the 7B decoder actually ran on.
    pub(crate) fn backend_label(&self) -> String {
        self.whole_decoder.backend_label()
    }

    /// Exact post-build Rust container capacity retained by the resident
    /// decoder actor. Native backend arenas are intentionally excluded: their
    /// graph constructors own admission for the concrete device/lane.
    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add(
            self.whole_decoder.retained_system_memory_bytes()?,
            "firered-llm decoder graph handles",
        )?;
        bytes.add(
            self.logits_head.retained_system_memory_bytes()?,
            "firered-llm logits head",
        )?;
        bytes.add(
            self.token_embedding.retained_system_memory_bytes()?,
            "firered-llm token embedding",
        )?;
        Ok(bytes.finish())
    }

    /// Releases the CPU per-token grow-to-fit step buffer this decoder's
    /// runner accumulated over the just-finished decode session/slice.
    /// `store_cached_decoder_runtime`'s caller MUST call this before storing
    /// the decoder back into the cross-chunk cache, so the buffer stays
    /// scoped to one session/slice instead of living on indefinitely with the
    /// cached decoder. A no-op on Metal/GPU or when no CPU step ever ran.
    pub(crate) fn release_session_scoped_buffers(&mut self) {
        self.whole_decoder.release_session_scoped_buffers();
    }

    /// Host history is sized to the current invocation's exact logical bound.
    /// The stable session reserve is carried separately into the reusable GPU
    /// graph, so changing chunk length within one envelope neither inflates
    /// host memory nor rebuilds the resident graph.
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
            "firered-llm.decoder.self-kv.host",
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
    ) -> Result<Vec<f32>, FireRedLlmDecoderError> {
        self.token_embedding
            .gather_rows(&[token_id])
            .map_err(|error| FireRedLlmDecoderError::TokenEmbeddingFailed {
                reason: error.to_string(),
            })
    }

    /// Run the entire ChatML+speech prompt as one causal prefill, seeding
    /// `layer_kv_caches` with every prompt token's K/V (unless the
    /// graph-reuse path handles it, see below), and return the logits row
    /// for the token immediately following the prompt (i.e. the first
    /// generated token's distribution) -- mirrors `qwen::ggml_executor`'s
    /// `write_prefill_step_outputs_and_compute_last_logits`.
    ///
    /// On a backend that supports persistent decode-graph reuse (Metal/
    /// single-GPU), this runs the prompt through
    /// `run_prefill_auto_last_hidden` instead of the bulk `run_prefill`
    /// below: `decode_step` reuses that same persistent graph, and it can
    /// only see a prompt token's K/V if the prompt flowed through it too
    /// (see that method's doc comment) -- prefilling in bulk and decoding
    /// via reuse would silently attend over an empty KV history for the
    /// whole prompt span.
    pub(crate) fn prefill(
        &mut self,
        prompt_embeddings: &Qwen3AsrPromptEmbeddings,
        layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
        capacity: Qwen3AsrKvCacheCapacity,
        control: &std::sync::Arc<crate::api::backend::TranscriptionControl>,
    ) -> Result<FireRedLlmPrefillOutput, FireRedLlmDecoderError> {
        let token_count = prompt_embeddings.token_count;
        if let Some(final_hidden) = self
            .whole_decoder
            .run_prefill_auto_last_hidden(
                &prompt_embeddings.token_major_values,
                token_count,
                layer_kv_caches,
                capacity,
                FIRERED_LLM_ROPE_THETA,
                control,
            )
            .map_err(|error| FireRedLlmDecoderError::GraphFailed {
                reason: error.to_string(),
            })?
        {
            if let Some(token_id) = self
                .whole_decoder
                .fused_logits_top1_from_hidden(&final_hidden)
                .map_err(|error| FireRedLlmDecoderError::GraphFailed {
                    reason: error.to_string(),
                })?
            {
                return Ok(FireRedLlmPrefillOutput {
                    logits: Vec::new(),
                    greedy_token_hint: Some(token_id),
                });
            }
            let logits = self
                .logits_runtime
                .compute_logits_for_last_hidden(&self.logits_head, &final_hidden)
                .map_err(|error| FireRedLlmDecoderError::LogitsHeadFailed {
                    reason: error.to_string(),
                })?;
            return Ok(FireRedLlmPrefillOutput {
                logits,
                greedy_token_hint: None,
            });
        }
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
        let logits = self
            .logits_runtime
            .compute_logits_for_last_hidden(&self.logits_head, &final_hidden)
            .map_err(|error| FireRedLlmDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            })?;
        Ok(FireRedLlmPrefillOutput {
            logits,
            greedy_token_hint: None,
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
        capacity: Qwen3AsrKvCacheCapacity,
    ) -> Result<Vec<f32>, FireRedLlmDecoderError> {
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
                capacity,
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
        self.logits_runtime
            .compute_logits_for_last_hidden(&self.logits_head, &step.hidden)
            .map_err(|error| FireRedLlmDecoderError::LogitsHeadFailed {
                reason: error.to_string(),
            })
    }

    /// On the resident Metal/GPU reuse graph, return the decoder's device-side
    /// argmax directly. firered-llm's registered policy has no suppression or
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
    ) -> Result<Option<u32>, FireRedLlmDecoderError> {
        if !self.whole_decoder.supports_graph_reuse() || !self.whole_decoder.supports_fused_top1() {
            return Ok(None);
        }
        if layer_kv_caches.is_empty() {
            return Err(FireRedLlmDecoderError::KvCacheFailed {
                reason: "firered-llm decoder has no layer KV caches".to_string(),
            });
        }
        let hidden = self.gather_token_embedding(token_id)?;
        let step = self
            .whole_decoder
            .run_step_reused_batched_top1(
                &hidden,
                &[cache_position],
                FIRERED_LLM_ROPE_THETA,
                capacity.resident_positions(),
            )
            .map_err(|error| FireRedLlmDecoderError::GraphFailed {
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
///
/// `layer_kv` is empty whenever the step came from the persistent reuse
/// graph (`run_step_auto`'s reused path): that graph's KV lives resident
/// device-side and is never read back to the host (see
/// `Qwen3AsrLlmWholeDecoderGraphExecutor::run_step_reused`'s doc comment), so
/// there is nothing to write and this is a deliberate no-op -- not a
/// mismatch -- exactly like `qwen::ggml_executor::run_llm_layers_with_kv`'s
/// own (unconditional, non-validating) write loop over the same empty case.
fn write_layer_kv(
    position_offset: usize,
    token_count: usize,
    layer_kv: &[(Vec<f32>, Vec<f32>)],
    kv_row_width: usize,
    layer_kv_caches: &mut [Qwen3AsrLayerKvCacheState],
) -> Result<(), FireRedLlmDecoderError> {
    if layer_kv.is_empty() {
        return Ok(());
    }
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

/// T5 (per-segment numeric parity against an independent PyTorch reference):
/// dumps embedding / single-decoder-block / final_norm+lm_head outputs on
/// fixed synthetic inputs to flat files that
/// `scratchpad/fr2-t5-parity/compare_parity.py` reads and diffs against a
/// from-scratch `transformers.Qwen2DecoderLayer` / manual embedding-gather /
/// RMSNorm+matmul reference built from the same merged safetensors the `.oasr`
/// pack was converted from. Deliberately tests the REAL production load path
/// (`Qwen3AsrLlmWholeDecoderGraphExecutor`, `Qwen3AsrLlmLogitsHead`,
/// `MappedTokenEmbeddingTable`) against the real q8_0 dev pack, not a
/// hand-rolled parallel implementation -- this is what caught the
/// zero-copy-bind tensor-naming bug this module's history fixed (see
/// `new_with_adapter`'s doc comment on `inner.attn_output_name`/`ffn_*_name`).
#[cfg(test)]
mod parity_tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::models::runtime_contract::ScalarMetadataView;

    fn dev_pack_path() -> Option<PathBuf> {
        crate::testing::external_test_fixture_path(
            "OPENASR_FIRERED_LLM_PACK",
            "FireRed2 LLM .oasr pack",
        )
        .inspect_err(|skip| eprintln!("skipping: {skip}"))
        .ok()
    }

    fn dump_dir() -> Option<PathBuf> {
        crate::testing::external_test_fixture_path(
            "OPENASR_FIRERED_LLM_PARITY_DUMP_DIR",
            "FireRed2 LLM parity output directory",
        )
        .inspect_err(|skip| eprintln!("skipping: {skip}"))
        .ok()
    }

    /// Deterministic pseudo-random f32 generator (xorshift64*, no external
    /// `rand` dependency needed for a test-only fixture) -- values scaled to a
    /// modest range so summed multi-layer activations stay well clear of f16/
    /// q8_0 dynamic-range edge cases that would make the parity check about
    /// quantization noise rather than wiring correctness.
    fn deterministic_f32_vec(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed ^ 0x9E3779B97F4A7C15;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // top 24 bits -> uniform in [-1, 1)
            let unit = ((state >> 40) as u32 & 0x00FF_FFFF) as f32 / 16_777_216.0;
            out.push(unit * 2.0 - 1.0);
        }
        out
    }

    fn write_f32_dump(dir: &Path, name: &str, values: &[f32]) {
        fs::create_dir_all(dir).expect("create dump dir");
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(dir.join(format!("{name}.f32le")), bytes)
            .unwrap_or_else(|error| panic!("write dump {name}: {error}"));
    }

    fn write_json_dump(dir: &Path, name: &str, json: &serde_json::Value) {
        fs::create_dir_all(dir).expect("create dump dir");
        fs::write(
            dir.join(format!("{name}.json")),
            serde_json::to_vec_pretty(json).expect("serialize json dump"),
        )
        .unwrap_or_else(|error| panic!("write json dump {name}: {error}"));
    }

    /// Load ONE decoder layer's projections directly by real layer index (not
    /// via `FireRedLlmDecoderRuntime::new`'s all-28-layers loop), so a
    /// single-layer `Qwen3AsrLlmWholeDecoderGraphExecutor` can be built and
    /// run in isolation -- this is what makes block-1 / block-14 / block-28
    /// (real indices 0/13/27) independently testable segments rather than
    /// only observable as one opaque 28-layer stack output.
    fn load_one_layer_projection(
        reader: &crate::ggml_runtime::GgufTensorDataReader,
        metadata: &FireRedLlmDecoderMetadata,
        layer_index: usize,
    ) -> Qwen3AsrLlmLayerAttentionProjection {
        let generic = load_qwen_family_llm_layer_attention_projection_generic(
            reader,
            firered_llm_qwen_family_layer_names(layer_index),
            metadata.d_model,
            metadata.n_heads,
            metadata.n_kv_heads,
            metadata.head_dim,
            false,
        )
        .unwrap_or_else(|error| panic!("load layer {layer_index} projection: {error}"));
        Qwen3AsrLlmLayerAttentionProjection::Generic(generic)
    }

    /// Dump one decoder block's isolated 3-position causal-prefill forward
    /// (real positions 0/1/2, real RoPE theta, real GQA/bias wiring) on a
    /// fixed synthetic hidden-state input -- independent of every other layer
    /// and of the token embedding table, so a mismatch localizes to this one
    /// block.
    fn dump_one_block_segment(
        reader: &crate::ggml_runtime::GgufTensorDataReader,
        runtime_source: &crate::GgmlRuntimeSource,
        metadata: &FireRedLlmDecoderMetadata,
        layer_index: usize,
        segment_name: &str,
        dir: &Path,
    ) {
        let token_count = 3usize;
        let input = deterministic_f32_vec(
            0xB10C_0000 + layer_index as u64,
            token_count * metadata.d_model,
        );
        let projection = load_one_layer_projection(reader, metadata, layer_index);
        let mut executor = Qwen3AsrLlmWholeDecoderGraphExecutor::new(
            &[projection],
            Some(runtime_source),
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
        )
        .unwrap_or_else(|error| panic!("{segment_name} single-layer executor: {error}"));
        let step = executor
            .run_prefill(&input, token_count, FIRERED_LLM_ROPE_THETA)
            .unwrap_or_else(|error| panic!("{segment_name} prefill: {error}"));

        write_f32_dump(dir, &format!("{segment_name}_input"), &input);
        write_f32_dump(dir, &format!("{segment_name}_output"), &step.hidden);
        write_json_dump(
            dir,
            &format!("{segment_name}_meta"),
            &serde_json::json!({
                "real_layer_index": layer_index,
                "token_count": token_count,
                "d_model": metadata.d_model,
                "n_heads": metadata.n_heads,
                "n_kv_heads": metadata.n_kv_heads,
                "head_dim": metadata.head_dim,
                "rope_theta": FIRERED_LLM_ROPE_THETA,
                "rms_norm_epsilon": FIRERED_LLM_RMS_NORM_EPSILON,
            }),
        );
    }

    #[test]
    #[ignore = "requires the private ~8.9GB dev-only firered2-llm-q8_0.oasr pack; dumps fixed-input \
                per-segment outputs to scratchpad/fr2-t5-parity for compare_parity.py to diff \
                against an independent PyTorch reference -- see this module's parity_tests doc"]
    fn dump_parity_segments_for_python_reference_comparison() {
        let Some(pack_path) = dev_pack_path() else {
            return;
        };
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return;
        }
        let Some(dir) = dump_dir() else {
            return;
        };

        let gguf_metadata =
            crate::ggml_runtime::read_gguf_metadata(&pack_path).expect("read gguf metadata");
        let decoder_metadata =
            super::super::runtime_contract::parse_firered_llm_decoder_metadata(&gguf_metadata)
                .expect("parse decoder metadata");
        eprintln!("decoder_metadata = {decoder_metadata:?}");
        write_json_dump(
            &dir,
            "manifest",
            &serde_json::json!({
                "n_layers": decoder_metadata.n_layers,
                "d_model": decoder_metadata.d_model,
                "n_heads": decoder_metadata.n_heads,
                "n_kv_heads": decoder_metadata.n_kv_heads,
                "head_dim": decoder_metadata.head_dim,
                "vocab_size": decoder_metadata.vocab_size,
                "block_segments": ["block0", "block13", "block27"],
                "block_real_layer_indices": [0, 13, decoder_metadata.n_layers - 1],
            }),
        );

        let runtime_source =
            crate::validate_ggml_runtime_source_path(&pack_path).expect("runtime source");
        let reader =
            crate::ggml_runtime::GgufTensorDataReader::from_runtime_source(&runtime_source)
                .expect("open gguf tensor reader");

        // --- Segment: embedding gather ---
        let contract =
            firered_llm_qwen_decoder_contract(&decoder_metadata).expect("bind decoder contract");
        let QwenDecoderTail {
            logits_head,
            token_embedding,
        } = load_qwen_decoder_tail_from_contract(
            &reader,
            &contract,
            FIRERED_LLM_RMS_NORM_EPSILON,
            crate::ggml_runtime::GgmlCpuGraphBackend::Cpu,
        )
        .expect("load decoder tail");
        let embedding_token_ids: Vec<u32> = vec![0, 1000, 50_000, 100_000, 151_643, 151_646];
        let embedding_rows = token_embedding
            .gather_rows(&embedding_token_ids)
            .expect("gather embedding rows");
        write_json_dump(
            &dir,
            "embedding_token_ids",
            &serde_json::json!({ "token_ids": embedding_token_ids }),
        );
        write_f32_dump(&dir, "embedding_output", &embedding_rows);

        // --- Segments: block 0 (first), block 13 (14th), block 27 (last) ---
        dump_one_block_segment(
            &reader,
            &runtime_source,
            &decoder_metadata,
            0,
            "block0",
            &dir,
        );
        dump_one_block_segment(
            &reader,
            &runtime_source,
            &decoder_metadata,
            13,
            "block13",
            &dir,
        );
        dump_one_block_segment(
            &reader,
            &runtime_source,
            &decoder_metadata,
            decoder_metadata.n_layers - 1,
            "block27",
            &dir,
        );

        // --- Segment: final_norm -> lm_head (fused; the only exposed API) ---
        let mut logits_runtime = logits_head
            .new_runtime(crate::ggml_runtime::GgmlCpuGraphBackend::Cpu)
            .expect("build logits runtime");
        let final_hidden = deterministic_f32_vec(0xF14A_1000, decoder_metadata.d_model);
        let logits = logits_runtime
            .compute_logits_for_last_hidden(&logits_head, &final_hidden)
            .expect("compute final logits");
        write_f32_dump(&dir, "final_norm_lm_head_input", &final_hidden);
        write_f32_dump(&dir, "final_norm_lm_head_output", &logits);

        eprintln!("dumped parity segments to {}", dir.display());
    }

    #[test]
    #[ignore = "requires the private ~8.9GB dev-only firered2-llm-q8_0.oasr pack; construction-only \
                smoke check for the zero-copy tensor-name wiring this module's history fixed"]
    fn probe_decoder_runtime_construction_against_real_pack() {
        let Some(pack_path) = dev_pack_path() else {
            return;
        };
        if !pack_path.exists() {
            eprintln!("skipping: {} not present", pack_path.display());
            return;
        }
        let metadata = crate::ggml_runtime::read_gguf_metadata(&pack_path).expect("read metadata");
        let _ = metadata.get_string_scalar("firered_llm.llm.n_layers");
        let decoder_metadata =
            super::super::runtime_contract::parse_firered_llm_decoder_metadata(&metadata)
                .expect("parse decoder metadata");
        let runtime_source =
            crate::validate_ggml_runtime_source_path(&pack_path).expect("runtime source");
        let preflight = crate::models::runtime_preflight::load_runtime_source_metadata_and_tensor_index_from_source(
            &runtime_source,
        )
        .expect("runtime preflight");
        FireRedLlmDecoderRuntime::new_from_preflight(
            &preflight,
            decoder_metadata,
            crate::ggml_runtime::GgmlCpuGraphConfig::runtime_default().backend,
        )
        .expect("decoder runtime constructs against the real pack");
    }
}
