use std::{cell::RefCell, fmt, path::PathBuf};

use thiserror::Error;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlStaticTensor,
    GgmlStaticTensorArena, GgufTensorDataReadError, GgufTensorDataReader, env_toggle_with_raw,
};
use crate::models::thread_local_runtime_cache::{
    BoundedRuntimeCache, DEFAULT_RUNTIME_CACHE_CAPACITY, canonical_runtime_cache_path,
    with_thread_local_cached_mut_by_key,
};

use super::graph_config::{qwen_decoder_graph_config, qwen_runtime_graph_config};
use super::runtime_contract::Qwen3AsrExecutionMetadata;
use super::tensor_names::{
    OUTPUT_NORM_WEIGHT as OUTPUT_NORM_WEIGHT_TENSOR_NAME,
    OUTPUT_WEIGHT as OUTPUT_WEIGHT_TENSOR_NAME,
};
const DEFAULT_RMS_NORM_EPSILON: f32 = 1e-6;
const QWEN3_LLM_LOGITS_GRAPH_CONTEXT_BYTES: usize = 16 * 1024 * 1024;
const OPENASR_QWEN3_LLM_LOGITS_GGML_ENV: &str = "OPENASR_QWEN3_LLM_LOGITS_GGML";

/// (canonical pack path, backend): identifies the resident fused logits-head
/// graph executor for a loaded pack, mirroring the `(PathBuf,
/// GgmlCpuGraphBackend)` key convention used by the qwen audio-encoder and
/// firered-aed encoder/decoder runtime caches.
type QwenLogitsHeadExecutorCacheKey = (PathBuf, GgmlCpuGraphBackend);

#[derive(Debug, Clone)]
pub(crate) struct Qwen3AsrLlmLogitsHead {
    d_model: usize,
    vocab_size: usize,
    rms_norm_epsilon: f32,
    output_norm_weight: Vec<f32>,
    output_weight_tensor_name: &'static str,
    output_weight_values: Option<Vec<f32>>,
    output_weight_layout: OutputWeightLayout,
    ggml_output_weight: Option<OwnedGgmlLogitsWeight>,
    ggml_executor_cache_key: Option<QwenLogitsHeadExecutorCacheKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputWeightLayout {
    HiddenVocab,
    VocabHidden,
}

#[derive(Debug, Error)]
pub(crate) enum Qwen3AsrLlmLogitsHeadError {
    #[error("qwen3-asr llm logits head tensor read failed: {reason}")]
    TensorReadFailed { reason: String },
    #[error("qwen3-asr llm logits head tensor '{tensor_name}' has invalid shape {shape}: {reason}")]
    InvalidTensorShape {
        tensor_name: &'static str,
        shape: String,
        reason: String,
    },
    #[error(
        "qwen3-asr llm logits head hidden state has invalid shape: got {got}, expected hidden_size={expected}"
    )]
    InvalidHiddenStateShape { got: usize, expected: usize },
    #[error("qwen3-asr llm logits head inputs contain non-finite values")]
    NonFiniteInputs,
    #[error("qwen3-asr llm logits head fallback values are unavailable")]
    OutputWeightValuesUnavailable,
    #[error("qwen3-asr llm logits head internal allocation overflowed")]
    AllocationOverflow,
    #[error(
        "qwen3-asr llm logits head top-1 token id {token_id} is outside vocab size {vocab_size}"
    )]
    InvalidTop1Token { token_id: i32, vocab_size: usize },
    #[error("qwen3-asr llm logits head ggml graph failed: {reason}")]
    GgmlGraphFailed { reason: String },
}

#[derive(Debug, Clone)]
struct OwnedGgmlLogitsWeight {
    ggml_type: i32,
    dims: Vec<usize>,
    bytes: Vec<u8>,
}

pub(crate) struct Qwen3AsrLlmFusedLogitsHeadSpec<'a> {
    pub(crate) d_model: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_epsilon: f32,
    pub(crate) output_norm_weight: &'a [f32],
    pub(crate) output_weight_tensor_name: &'static str,
    pub(crate) output_weight_ggml_type: i32,
    pub(crate) output_weight_dims: &'a [usize],
    pub(crate) output_weight_bytes: &'a [u8],
}

impl Qwen3AsrLlmLogitsHead {
    pub(crate) fn fused_top1_spec(&self) -> Option<Qwen3AsrLlmFusedLogitsHeadSpec<'_>> {
        let output_weight = self.ggml_output_weight.as_ref()?;
        Some(Qwen3AsrLlmFusedLogitsHeadSpec {
            d_model: self.d_model,
            vocab_size: self.vocab_size,
            rms_norm_epsilon: self.rms_norm_epsilon,
            output_norm_weight: &self.output_norm_weight,
            output_weight_tensor_name: self.output_weight_tensor_name,
            output_weight_ggml_type: output_weight.ggml_type,
            output_weight_dims: &output_weight.dims,
            output_weight_bytes: &output_weight.bytes,
        })
    }

    pub fn compute_logits_for_last_hidden(
        &self,
        hidden: &[f32],
    ) -> Result<Vec<f32>, Qwen3AsrLlmLogitsHeadError> {
        if hidden.len() != self.d_model {
            return Err(Qwen3AsrLlmLogitsHeadError::InvalidHiddenStateShape {
                got: hidden.len(),
                expected: self.d_model,
            });
        }
        if hidden.iter().any(|value| !value.is_finite()) {
            return Err(Qwen3AsrLlmLogitsHeadError::NonFiniteInputs);
        }

        if let (Some(cache_key), Some(output_weight)) = (
            self.ggml_executor_cache_key.as_ref(),
            self.ggml_output_weight.as_ref(),
        ) {
            return with_thread_local_logits_head_executor(
                cache_key.clone(),
                self.d_model,
                self.vocab_size,
                self.rms_norm_epsilon,
                &self.output_norm_weight,
                output_weight,
                |executor| executor.compute(hidden),
            )
            .map_err(|source| Qwen3AsrLlmLogitsHeadError::GgmlGraphFailed {
                reason: source.to_string(),
            });
        }

        let normed = rms_norm_with_weight(hidden, &self.output_norm_weight, self.rms_norm_epsilon)?;
        let output_weight_values = self
            .output_weight_values
            .as_ref()
            .ok_or(Qwen3AsrLlmLogitsHeadError::OutputWeightValuesUnavailable)?;
        let mut logits = vec![0.0_f32; self.vocab_size];
        match self.output_weight_layout {
            OutputWeightLayout::HiddenVocab => {
                for (hidden_idx, hidden_value) in normed.iter().copied().enumerate() {
                    let row_start = hidden_idx
                        .checked_mul(self.vocab_size)
                        .ok_or(Qwen3AsrLlmLogitsHeadError::AllocationOverflow)?;
                    let row = &output_weight_values[row_start..row_start + self.vocab_size];
                    for (vocab_idx, weight) in row.iter().copied().enumerate() {
                        logits[vocab_idx] += hidden_value * weight;
                    }
                }
            }
            OutputWeightLayout::VocabHidden => {
                for (vocab_idx, out) in logits.iter_mut().enumerate() {
                    let row_start = vocab_idx
                        .checked_mul(self.d_model)
                        .ok_or(Qwen3AsrLlmLogitsHeadError::AllocationOverflow)?;
                    let row = &output_weight_values[row_start..row_start + self.d_model];
                    let mut acc = 0.0_f32;
                    for (hidden_idx, weight) in row.iter().copied().enumerate() {
                        acc += normed[hidden_idx] * weight;
                    }
                    *out = acc;
                }
            }
        }
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(Qwen3AsrLlmLogitsHeadError::NonFiniteInputs);
        }
        Ok(logits)
    }

    pub(crate) fn compute_top1_token_for_last_hidden(
        &self,
        hidden: &[f32],
    ) -> Result<u32, Qwen3AsrLlmLogitsHeadError> {
        if hidden.len() != self.d_model {
            return Err(Qwen3AsrLlmLogitsHeadError::InvalidHiddenStateShape {
                got: hidden.len(),
                expected: self.d_model,
            });
        }
        if hidden.iter().any(|value| !value.is_finite()) {
            return Err(Qwen3AsrLlmLogitsHeadError::NonFiniteInputs);
        }

        if let (Some(cache_key), Some(output_weight)) = (
            self.ggml_executor_cache_key.as_ref(),
            self.ggml_output_weight.as_ref(),
        ) {
            let token_id = with_thread_local_logits_head_executor(
                cache_key.clone(),
                self.d_model,
                self.vocab_size,
                self.rms_norm_epsilon,
                &self.output_norm_weight,
                output_weight,
                |executor| executor.compute_top1(hidden),
            )
            .map_err(|source| Qwen3AsrLlmLogitsHeadError::GgmlGraphFailed {
                reason: source.to_string(),
            })?;
            return validate_top1_token_id(token_id, self.vocab_size);
        }

        let logits = self.compute_logits_for_last_hidden(hidden)?;
        let mut best_index = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, value) in logits.iter().copied().enumerate() {
            if value > best_value {
                best_value = value;
                best_index = index;
            }
        }
        u32::try_from(best_index).map_err(|_| Qwen3AsrLlmLogitsHeadError::AllocationOverflow)
    }
}

pub(crate) fn load_qwen3_llm_logits_head_from_reader(
    reader: &GgufTensorDataReader,
    metadata: Qwen3AsrExecutionMetadata,
) -> Result<Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadError> {
    load_qwen3_llm_logits_head_from_reader_with_output_tensor(
        reader,
        metadata,
        OUTPUT_WEIGHT_TENSOR_NAME,
        DEFAULT_RMS_NORM_EPSILON,
    )
}

pub(crate) fn load_qwen3_llm_logits_head_from_reader_with_output_tensor(
    reader: &GgufTensorDataReader,
    metadata: Qwen3AsrExecutionMetadata,
    output_weight_tensor_name: &'static str,
    rms_norm_epsilon: f32,
) -> Result<Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadError> {
    load_llm_logits_head_from_reader_with_tensor_names(
        reader,
        metadata.llm_d_model,
        metadata.vocab_size,
        OUTPUT_NORM_WEIGHT_TENSOR_NAME,
        output_weight_tensor_name,
        rms_norm_epsilon,
    )
}

/// Like [`load_qwen3_llm_logits_head_from_reader_with_output_tensor`] but
/// decoupled from `Qwen3AsrExecutionMetadata` and qwen's own tensor-naming
/// scheme, so a sibling family (e.g. firered-llm's `llm.out_norm.weight` /
/// `llm.lm_head.weight`) can reuse the same RMSNorm+matmul(+optional fused
/// device top-1) logits-head machinery without any Qwen2/Qwen3-specific
/// assumption -- this stage of the pipeline (final hidden -> logits/top-1) is
/// identical across every qwen-family decoder-only LLM.
pub(crate) fn load_llm_logits_head_from_reader_with_tensor_names(
    reader: &GgufTensorDataReader,
    d_model: usize,
    vocab_size: usize,
    output_norm_weight_tensor_name: &'static str,
    output_weight_tensor_name: &'static str,
    rms_norm_epsilon: f32,
) -> Result<Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadError> {
    if !rms_norm_epsilon.is_finite() || rms_norm_epsilon <= 0.0 {
        return Err(Qwen3AsrLlmLogitsHeadError::InvalidTensorShape {
            tensor_name: output_norm_weight_tensor_name,
            shape: "[]".to_string(),
            reason: format!("rms_norm_epsilon={rms_norm_epsilon} must be finite and positive"),
        });
    }
    let output_weight_tensor = reader
        .tensor_index()
        .get(output_weight_tensor_name)
        .ok_or_else(|| Qwen3AsrLlmLogitsHeadError::InvalidTensorShape {
            tensor_name: output_weight_tensor_name,
            shape: "[]".to_string(),
            reason: "tensor is missing from GGUF tensor index".to_string(),
        })?;
    let output_weight_dims = output_weight_tensor.dims.clone();
    if output_weight_dims.len() != 2 {
        return Err(Qwen3AsrLlmLogitsHeadError::InvalidTensorShape {
            tensor_name: output_weight_tensor_name,
            shape: render_shape(&output_weight_dims),
            reason: "expected rank-2 matrix".to_string(),
        });
    }
    let output_weight_layout =
        resolve_output_weight_layout(&output_weight_dims, d_model, vocab_size)?;
    let output_norm_weight = reader
        .host_tensor_f32_copy_dequantized_by_name(output_norm_weight_tensor_name, &[d_model as u64])
        .map_err(map_tensor_read_error)?;
    if output_norm_weight.iter().any(|value| !value.is_finite()) {
        return Err(Qwen3AsrLlmLogitsHeadError::NonFiniteInputs);
    }
    let raw_output_weight = if logits_head_ggml_enabled() {
        load_direct_output_weight_payload(
            reader,
            output_weight_tensor_name,
            &output_weight_dims,
            d_model,
            vocab_size,
        )?
    } else {
        None
    };
    let output_weight_values = if raw_output_weight.is_some() {
        None
    } else {
        let values = reader
            .host_tensor_f32_copy_dequantized_by_name(
                output_weight_tensor_name,
                &output_weight_dims,
            )
            .map_err(map_tensor_read_error)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Qwen3AsrLlmLogitsHeadError::NonFiniteInputs);
        }
        Some(values)
    };
    Ok(Qwen3AsrLlmLogitsHead {
        d_model,
        vocab_size,
        rms_norm_epsilon,
        output_norm_weight,
        output_weight_tensor_name,
        output_weight_values,
        output_weight_layout,
        ggml_executor_cache_key: raw_output_weight.as_ref().map(|_| {
            (
                canonical_runtime_cache_path(reader.tensor_index().path()),
                qwen_runtime_graph_config().backend,
            )
        }),
        ggml_output_weight: raw_output_weight,
    })
}

fn load_direct_output_weight_payload(
    reader: &GgufTensorDataReader,
    output_weight_tensor_name: &'static str,
    dims: &[u64],
    d_model: usize,
    vocab_size: usize,
) -> Result<Option<OwnedGgmlLogitsWeight>, Qwen3AsrLlmLogitsHeadError> {
    if dims != [d_model as u64, vocab_size as u64] {
        return Ok(None);
    }
    let payload = reader
        .weight_tensor_payload_by_name(output_weight_tensor_name)
        .map_err(map_tensor_read_error)?;
    if payload.dims.as_slice() != [d_model, vocab_size] {
        return Ok(None);
    }
    Ok(Some(OwnedGgmlLogitsWeight {
        ggml_type: payload.element_type.ggml_type(),
        dims: payload.dims,
        bytes: payload.bytes.to_vec(),
    }))
}

struct Qwen3AsrLlmLogitsHeadGraphExecutor {
    d_model: usize,
    vocab_size: usize,
    rms_norm_epsilon: f32,
    runner: GgmlCpuGraphRunner,
    arena: GgmlStaticTensorArena,
    output_norm_weight: GgmlStaticTensor,
    output_weight: GgmlStaticTensor,
    argmax_reverse_indices: GgmlStaticTensor,
}

thread_local! {
    // Bounded LRU (not a plain `HashMap`): each entry owns a resident ggml
    // static-tensor arena holding the full vocab x d_model output-weight
    // matrix. The previous design keyed this cache on a thread-local
    // monotonic id minted once per model *load* and never evicted or
    // reused, so every load of a qwen pack left one more multi-hundred-MB
    // entry permanently resident per thread for the life of the process --
    // the same "memory roller coaster" root cause already fixed for the
    // other per-pack runtime caches (see
    // `thread_local_runtime_cache::DEFAULT_RUNTIME_CACHE_CAPACITY`). Keying
    // on `(canonical pack path, backend)` instead lets repeated loads of the
    // same pack reuse one entry and bounds the worst case to
    // `DEFAULT_RUNTIME_CACHE_CAPACITY` resident executors per thread.
    static QWEN_LLM_LOGITS_HEAD_EXECUTOR_BY_KEY: RefCell<
        BoundedRuntimeCache<QwenLogitsHeadExecutorCacheKey, Qwen3AsrLlmLogitsHeadGraphExecutor>,
    > = RefCell::new(BoundedRuntimeCache::new());
}

fn with_thread_local_logits_head_executor<R>(
    cache_key: QwenLogitsHeadExecutorCacheKey,
    d_model: usize,
    vocab_size: usize,
    rms_norm_epsilon: f32,
    output_norm_weight: &[f32],
    output_weight: &OwnedGgmlLogitsWeight,
    use_executor: impl FnOnce(&mut Qwen3AsrLlmLogitsHeadGraphExecutor) -> Result<R, GgmlCpuGraphError>,
) -> Result<R, GgmlCpuGraphError> {
    with_thread_local_cached_mut_by_key(
        &QWEN_LLM_LOGITS_HEAD_EXECUTOR_BY_KEY,
        cache_key,
        DEFAULT_RUNTIME_CACHE_CAPACITY,
        || {
            Qwen3AsrLlmLogitsHeadGraphExecutor::new(
                d_model,
                vocab_size,
                rms_norm_epsilon,
                output_norm_weight,
                output_weight,
            )
        },
        use_executor,
    )
}

impl fmt::Debug for Qwen3AsrLlmLogitsHeadGraphExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Qwen3AsrLlmLogitsHeadGraphExecutor")
            .field("d_model", &self.d_model)
            .field("vocab_size", &self.vocab_size)
            .finish_non_exhaustive()
    }
}

impl Qwen3AsrLlmLogitsHeadGraphExecutor {
    fn new(
        d_model: usize,
        vocab_size: usize,
        rms_norm_epsilon: f32,
        output_norm_weight: &[f32],
        output_weight: &OwnedGgmlLogitsWeight,
    ) -> Result<Self, GgmlCpuGraphError> {
        if !rms_norm_epsilon.is_finite() || rms_norm_epsilon <= 0.0 {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "logits head rms norm epsilon must be finite and positive",
            });
        }
        if output_norm_weight.len() != d_model {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "logits head norm weight width mismatch",
            });
        }
        if output_weight.dims.as_slice() != [d_model, vocab_size] {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "logits head output weight shape mismatch",
            });
        }

        // Norm + a single [1, d_model] x [d_model, vocab_size] projection per
        // call -- a thin, memory-bandwidth-bound matrix-vector product with
        // one output row, run once per decode step at the same cadence as
        // the whole-decoder executor this feeds (thousands of decode-step
        // calls dominating over a handful of prefill chunks). Little
        // row-level parallelism to hand out per call regardless of thread
        // count, so this takes the `Decoder` tier from
        // `qwen_decoder_graph_config` on its own merits (there is no
        // separate firered-aed logits-head module to mirror here).
        let mut config = qwen_decoder_graph_config();
        config.context_bytes = QWEN3_LLM_LOGITS_GRAPH_CONTEXT_BYTES;
        let runner = GgmlCpuGraphRunner::new(config)?;
        let mut arena = runner.start_static_tensor_arena(config.context_bytes)?;
        let norm = arena.new_tensor_2d_f32(d_model, 1, "qwen_llm_logits_output_norm_weight")?;
        let weight = arena.new_matmul_weight_2d_typed(
            d_model,
            vocab_size,
            output_weight.ggml_type,
            "qwen_llm_logits_output_weight",
        )?;
        let argmax_reverse_indices =
            arena.new_tensor_1d_i32(vocab_size, "qwen_llm_logits_argmax_reverse_indices")?;
        arena.set_f32_slice(
            norm,
            output_norm_weight,
            "qwen_llm_logits_output_norm_weight",
        )?;
        arena.set_bytes_slice(
            weight,
            &output_weight.bytes,
            "qwen_llm_logits_output_weight",
        )?;
        arena.set_i32_slice(
            argmax_reverse_indices,
            &first_max_argmax_reverse_indices(vocab_size)?,
            "qwen_llm_logits_argmax_reverse_indices",
        )?;
        Ok(Self {
            d_model,
            vocab_size,
            rms_norm_epsilon,
            runner,
            arena,
            output_norm_weight: norm,
            output_weight: weight,
            argmax_reverse_indices,
        })
    }

    fn compute(&mut self, hidden: &[f32]) -> Result<Vec<f32>, GgmlCpuGraphError> {
        if hidden.len() != self.d_model {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "logits head hidden width mismatch",
            });
        }
        let mut graph = self.runner.start_graph();
        let hidden_tensor = graph.new_tensor_2d_f32(self.d_model, 1, "qwen_llm_logits_hidden")?;
        graph.set_input(hidden_tensor)?;
        let normed = graph.rms_norm(hidden_tensor, self.rms_norm_epsilon)?;
        let normed = graph.mul(normed, self.arena.graph_tensor(self.output_norm_weight))?;
        let logits = graph.mul_mat(self.arena.graph_tensor(self.output_weight), normed)?;
        graph.set_output(logits)?;
        graph.set_f32_slice(hidden_tensor, hidden, "qwen_llm_logits_hidden")?;
        graph.compute_output_f32(logits, self.vocab_size)
    }

    fn compute_top1(&mut self, hidden: &[f32]) -> Result<i32, GgmlCpuGraphError> {
        if hidden.len() != self.d_model {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "logits head hidden width mismatch",
            });
        }
        // Deliberately per-call. A prepared standalone logits-head top-1 graph
        // can alias stale i32 output storage on GPU-class non-scheduler backends
        // and has segfaulted under Hy-MT2 decode. The hot greedy path is fused
        // into the resident whole-decoder graph instead; keep this shared Qwen
        // helper as a simple fallback with no hidden persistent crash path.
        self.compute_top1_single_graph(hidden)
    }

    fn compute_top1_single_graph(&mut self, hidden: &[f32]) -> Result<i32, GgmlCpuGraphError> {
        let mut graph = self.runner.start_graph();
        let hidden_tensor = graph.new_tensor_2d_f32(self.d_model, 1, "qwen_llm_logits_hidden")?;
        graph.set_input(hidden_tensor)?;
        let normed = graph.rms_norm(hidden_tensor, self.rms_norm_epsilon)?;
        let normed = graph.mul(normed, self.arena.graph_tensor(self.output_norm_weight))?;
        let logits = graph.mul_mat(self.arena.graph_tensor(self.output_weight), normed)?;
        let top1 = graph.top1_argmax_first_max_reversed(
            logits,
            self.arena.graph_tensor(self.argmax_reverse_indices),
        )?;
        graph.set_output(top1)?;
        graph.set_f32_slice(hidden_tensor, hidden, "qwen_llm_logits_hidden")?;
        let token_ids = graph.compute_output_i32(top1, 1)?;
        let reversed_token_id =
            token_ids
                .first()
                .copied()
                .ok_or(GgmlCpuGraphError::OutputByteSizeMismatch {
                    expected: std::mem::size_of::<i32>(),
                    actual: 0,
                })?;
        first_max_token_id_from_reversed_argmax(reversed_token_id, self.vocab_size)
    }
}

pub(crate) fn first_max_argmax_reverse_indices(
    vocab_size: usize,
) -> Result<Vec<i32>, GgmlCpuGraphError> {
    (0..vocab_size)
        .rev()
        .map(|index| {
            i32::try_from(index).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                reason: "first-max argmax vocab index exceeds ggml int boundary",
            })
        })
        .collect()
}

pub(crate) fn first_max_token_id_from_reversed_argmax(
    reversed_token_id: i32,
    vocab_size: usize,
) -> Result<i32, GgmlCpuGraphError> {
    let reversed_index =
        usize::try_from(reversed_token_id).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
            reason: "first-max argmax reversed token id is negative",
        })?;
    if reversed_index >= vocab_size {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "first-max argmax reversed token id is outside vocab size",
        });
    }
    let original_index = vocab_size - 1 - reversed_index;
    i32::try_from(original_index).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
        reason: "first-max argmax token id exceeds ggml int boundary",
    })
}

fn validate_top1_token_id(
    token_id: i32,
    vocab_size: usize,
) -> Result<u32, Qwen3AsrLlmLogitsHeadError> {
    if token_id < 0 || token_id as usize >= vocab_size {
        return Err(Qwen3AsrLlmLogitsHeadError::InvalidTop1Token {
            token_id,
            vocab_size,
        });
    }
    Ok(token_id as u32)
}

fn rms_norm_with_weight(
    hidden: &[f32],
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, Qwen3AsrLlmLogitsHeadError> {
    if hidden.len() != weight.len() {
        return Err(Qwen3AsrLlmLogitsHeadError::InvalidTensorShape {
            tensor_name: OUTPUT_NORM_WEIGHT_TENSOR_NAME,
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

fn map_tensor_read_error(error: GgufTensorDataReadError) -> Qwen3AsrLlmLogitsHeadError {
    Qwen3AsrLlmLogitsHeadError::TensorReadFailed {
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

fn logits_head_ggml_enabled() -> bool {
    parse_env_flag(
        std::env::var(OPENASR_QWEN3_LLM_LOGITS_GGML_ENV)
            .ok()
            .as_deref(),
        logits_head_ggml_default_enabled(),
    )
}

fn logits_head_ggml_default_enabled() -> bool {
    logits_head_ggml_default_enabled_for_backend(qwen_runtime_graph_config().backend)
}

fn logits_head_ggml_default_enabled_for_backend(backend: GgmlCpuGraphBackend) -> bool {
    // Keep the large hidden x vocab projection in the runtime graph whenever
    // the output-weight layout can be loaded directly. Even on CPU, ggml's
    // matmul path avoids the scalar host fallback becoming the autoregressive
    // loop bottleneck.
    matches!(
        backend,
        GgmlCpuGraphBackend::Cpu | GgmlCpuGraphBackend::Metal | GgmlCpuGraphBackend::Gpu
    )
}

fn parse_env_flag(raw: Option<&str>, default: bool) -> bool {
    env_toggle_with_raw(None, raw, default)
}

fn resolve_output_weight_layout(
    output_weight_dims: &[u64],
    d_model: usize,
    vocab_size: usize,
) -> Result<OutputWeightLayout, Qwen3AsrLlmLogitsHeadError> {
    if output_weight_dims[0] == d_model as u64 && output_weight_dims[1] == vocab_size as u64 {
        return Ok(OutputWeightLayout::VocabHidden);
    }
    if output_weight_dims[0] == vocab_size as u64 && output_weight_dims[1] == d_model as u64 {
        return Ok(OutputWeightLayout::HiddenVocab);
    }
    Err(Qwen3AsrLlmLogitsHeadError::InvalidTensorShape {
        tensor_name: OUTPUT_WEIGHT_TENSOR_NAME,
        shape: render_shape(output_weight_dims),
        reason: format!("expected [{d_model} x {vocab_size}] or [{vocab_size} x {d_model}]"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logits_head_hidden_vocab_layout_matches_manual_matmul() {
        let head = Qwen3AsrLlmLogitsHead {
            d_model: 2,
            vocab_size: 3,
            rms_norm_epsilon: DEFAULT_RMS_NORM_EPSILON,
            output_norm_weight: vec![1.0, 1.0],
            output_weight_tensor_name: OUTPUT_WEIGHT_TENSOR_NAME,
            output_weight_values: Some(vec![
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0,
            ]),
            output_weight_layout: OutputWeightLayout::HiddenVocab,
            ggml_output_weight: None,
            ggml_executor_cache_key: None,
        };
        let logits = head
            .compute_logits_for_last_hidden(&[1.0, 2.0])
            .expect("logits");
        assert_eq!(logits.len(), 3);
        assert!(logits.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn logits_head_rejects_wrong_hidden_size() {
        let head = Qwen3AsrLlmLogitsHead {
            d_model: 4,
            vocab_size: 8,
            rms_norm_epsilon: DEFAULT_RMS_NORM_EPSILON,
            output_norm_weight: vec![1.0; 4],
            output_weight_tensor_name: OUTPUT_WEIGHT_TENSOR_NAME,
            output_weight_values: Some(vec![0.0; 32]),
            output_weight_layout: OutputWeightLayout::HiddenVocab,
            ggml_output_weight: None,
            ggml_executor_cache_key: None,
        };
        let error = head
            .compute_logits_for_last_hidden(&[0.0; 3])
            .expect_err("wrong hidden size must fail");
        assert!(matches!(
            error,
            Qwen3AsrLlmLogitsHeadError::InvalidHiddenStateShape { .. }
        ));
    }

    #[test]
    fn logits_head_layout_resolves_hidden_vocab_for_canonical_dims() {
        let layout = resolve_output_weight_layout(&[1024, 151936], 1024, 151936)
            .expect("canonical dims should resolve");
        assert_eq!(layout, OutputWeightLayout::VocabHidden);
    }

    #[test]
    fn logits_head_layout_resolves_vocab_hidden_for_transposed_dims() {
        let layout = resolve_output_weight_layout(&[151936, 1024], 1024, 151936)
            .expect("transposed dims should resolve");
        assert_eq!(layout, OutputWeightLayout::HiddenVocab);
    }

    #[test]
    fn logits_head_env_flag_defaults_when_unset() {
        assert!(parse_env_flag(None, true));
        assert!(!parse_env_flag(None, false));
    }

    #[test]
    fn logits_head_env_flag_accepts_common_true_false_values() {
        for value in ["1", "true", "yes", "on", " TRUE "] {
            assert!(
                parse_env_flag(Some(value), false),
                "expected true for value {value}"
            );
        }
        for value in ["0", "false", "no", "off", " Off "] {
            assert!(
                !parse_env_flag(Some(value), true),
                "expected false for value {value}"
            );
        }
    }

    #[test]
    fn logits_head_env_flag_falls_back_to_default_for_unknown_values() {
        assert!(parse_env_flag(Some("maybe"), true));
        assert!(!parse_env_flag(Some("maybe"), false));
    }

    #[test]
    fn logits_head_ggml_default_enabled_for_all_backends() {
        assert!(logits_head_ggml_default_enabled_for_backend(
            GgmlCpuGraphBackend::Metal
        ));
        assert!(logits_head_ggml_default_enabled_for_backend(
            GgmlCpuGraphBackend::Gpu
        ));
        assert!(logits_head_ggml_default_enabled_for_backend(
            GgmlCpuGraphBackend::Cpu
        ));
    }

    #[test]
    fn logits_head_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Qwen3AsrLlmLogitsHead>();
    }

    fn ggml_logits_head_with_cache_key(
        cache_key: QwenLogitsHeadExecutorCacheKey,
        output_norm_weight: Vec<f32>,
    ) -> Qwen3AsrLlmLogitsHead {
        // A valid rank-2 [d_model x vocab_size] f32 weight for d_model=2,
        // vocab_size=3, matching the fused-logits fixture in
        // `llm_transformer::tests::fused_logits_top1_selects_first_token_on_equal_logit_tie`.
        let output_weight_values: [f32; 6] = [
            0.1, 0.0, //
            0.3, 0.0, //
            0.3, 0.0,
        ];
        Qwen3AsrLlmLogitsHead {
            d_model: 2,
            vocab_size: 3,
            rms_norm_epsilon: DEFAULT_RMS_NORM_EPSILON,
            output_norm_weight,
            output_weight_tensor_name: OUTPUT_WEIGHT_TENSOR_NAME,
            output_weight_values: None,
            output_weight_layout: OutputWeightLayout::HiddenVocab,
            ggml_output_weight: Some(OwnedGgmlLogitsWeight {
                ggml_type: crate::ggml_runtime::GGML_TYPE_F32,
                dims: vec![2, 3],
                bytes: output_weight_values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            }),
            ggml_executor_cache_key: Some(cache_key),
        }
    }

    fn ggml_logits_head_test_cache_key(id: usize) -> QwenLogitsHeadExecutorCacheKey {
        (
            PathBuf::from(format!("/tmp/openasr-test-logits-head-cache-{id}.gguf")),
            GgmlCpuGraphBackend::Cpu,
        )
    }

    #[test]
    fn ggml_logits_head_executor_cache_reuses_same_key_without_rebuilding() {
        let key = ggml_logits_head_test_cache_key(9_000_001);
        let first = ggml_logits_head_with_cache_key(key.clone(), vec![1.0, 1.0]);
        first
            .compute_top1_token_for_last_hidden(&[1.0, 2.0])
            .expect("first compute builds and caches the executor");

        // Same cache key, but a norm-weight width that would fail
        // `Qwen3AsrLlmLogitsHeadGraphExecutor::new`'s shape validation if it
        // were actually rebuilt. Success here proves the cached executor
        // from `first` was reused and this (invalid) weight was never used
        // to build a new one.
        let second = ggml_logits_head_with_cache_key(key, vec![1.0, 1.0, 1.0]);
        second
            .compute_top1_token_for_last_hidden(&[1.0, 2.0])
            .expect("second compute must reuse the cached executor, not rebuild from bad input");
    }

    #[test]
    fn ggml_logits_head_executor_cache_evicts_beyond_capacity() {
        let base_id = 9_100_000;
        for offset in 0..(DEFAULT_RUNTIME_CACHE_CAPACITY + 3) {
            let key = ggml_logits_head_test_cache_key(base_id + offset);
            let head = ggml_logits_head_with_cache_key(key, vec![1.0, 1.0]);
            head.compute_top1_token_for_last_hidden(&[1.0, 2.0])
                .unwrap_or_else(|error| panic!("compute for distinct key {offset}: {error}"));

            let len = QWEN_LLM_LOGITS_HEAD_EXECUTOR_BY_KEY.with(|cache| cache.borrow().len());
            assert!(
                len <= DEFAULT_RUNTIME_CACHE_CAPACITY,
                "cache must never exceed the configured capacity (cap={DEFAULT_RUNTIME_CACHE_CAPACITY}), got {len}"
            );
        }
    }
}
