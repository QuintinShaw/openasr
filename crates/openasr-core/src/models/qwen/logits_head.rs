use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;

use crate::GgmlRuntimeSource;
use crate::ggml_runtime::{
    GgmlComputeOutput, GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlSelectionEvidenceRef, GgmlStaticTensor, GgmlStaticTensorArena,
    GgufOwnedWeightTensorPayload, GgufTensorDataReadError, GgufTensorDataReader,
    env_toggle_with_raw,
};
#[cfg(test)]
use crate::models::device_greedy_token::device_top1_token_id;

use super::graph_config::{qwen_decoder_graph_config, qwen_runtime_graph_config};
use super::runtime_contract::Qwen3AsrExecutionMetadata;
use super::tensor_names::{
    OUTPUT_NORM_WEIGHT as OUTPUT_NORM_WEIGHT_TENSOR_NAME,
    OUTPUT_WEIGHT as OUTPUT_WEIGHT_TENSOR_NAME,
};
pub(crate) const DEFAULT_RMS_NORM_EPSILON: f32 = 1e-6;
// The longest graph is input -> RMS norm -> affine -> projection -> first-max
// argmax. Keep a small structural margin for metadata-only views without
// coupling this helper to the whole decoder's much larger graph budget.
const QWEN3_LLM_LOGITS_GRAPH_NODE_CAPACITY: usize = 64;
const QWEN3_LLM_LOGITS_STATIC_TENSOR_COUNT: usize = 2;
const OPENASR_QWEN3_LLM_LOGITS_GGML_ENV: &str = "OPENASR_QWEN3_LLM_LOGITS_GGML";
static NEXT_LOGITS_HEAD_RUNTIME_IDENTITY: AtomicU64 = AtomicU64::new(1);

fn next_logits_head_runtime_identity() -> u64 {
    NEXT_LOGITS_HEAD_RUNTIME_IDENTITY
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .expect("qwen logits-head runtime identity space exhausted")
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3AsrLlmLogitsHead {
    runtime_identity: u64,
    d_model: usize,
    vocab_size: usize,
    rms_norm_epsilon: f32,
    output_norm_weight: Vec<f32>,
    #[cfg(test)]
    output_weight_tensor_name: &'static str,
    output_weight_values: Option<Vec<f32>>,
    output_weight_layout: OutputWeightLayout,
    ggml_output_weight: Option<OwnedGgmlLogitsWeight>,
}

/// Mutable native logits graph owned by the decoder runtime that consumes it.
/// Prepared weights remain host-neutral and shareable, while this graph runner
/// and its static arena are pinned to one concrete execution lane and follow
/// that decoder owner's eviction lifetime.
pub(crate) struct Qwen3AsrLlmLogitsHeadRuntime {
    head_runtime_identity: u64,
    executor: Option<Qwen3AsrLlmLogitsHeadGraphExecutor>,
    last_compute_evidence: Option<GgmlSelectionEvidenceRef>,
}

impl fmt::Debug for Qwen3AsrLlmLogitsHeadRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Qwen3AsrLlmLogitsHeadRuntime")
            .field("head_runtime_identity", &self.head_runtime_identity)
            .field("native_executor", &self.executor.is_some())
            .finish()
    }
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
    #[error(
        "qwen3-asr llm logits head hidden rows have invalid shape: got {got} values for {row_count} row(s) of hidden_size={hidden_size}"
    )]
    InvalidHiddenRowsShape {
        got: usize,
        row_count: usize,
        hidden_size: usize,
    },
    #[error(
        "qwen3-asr llm logits head output rows have invalid shape: got {got} values for {row_count} row(s) of vocab_size={vocab_size}"
    )]
    InvalidLogitsRowsShape {
        got: usize,
        row_count: usize,
        vocab_size: usize,
    },
    #[error("qwen3-asr llm logits head inputs contain non-finite values")]
    NonFiniteInputs,
    #[error("qwen3-asr llm logits head outputs contain non-finite values")]
    NonFiniteOutputs,
    #[error("qwen3-asr llm logits head fallback values are unavailable")]
    OutputWeightValuesUnavailable,
    #[error("qwen3-asr llm logits runtime was paired with a different prepared head")]
    RuntimeHeadMismatch,
    #[error("qwen3-asr llm logits head internal allocation overflowed")]
    AllocationOverflow,
    #[cfg(test)]
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
    payload: LogitsWeightPayload,
}

#[derive(Debug, Clone)]
enum LogitsWeightPayload {
    Mapped(GgufOwnedWeightTensorPayload),
    #[cfg(test)]
    TestBytes(Vec<u8>),
}

impl LogitsWeightPayload {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Mapped(payload) => payload.bytes(),
            #[cfg(test)]
            Self::TestBytes(bytes) => bytes,
        }
    }

    fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        match self {
            Self::Mapped(payload) => payload.retained_system_memory_bytes(),
            #[cfg(test)]
            Self::TestBytes(bytes) => {
                let mut capacity =
                    crate::models::system_memory_owner::SystemMemoryCapacity::default();
                capacity.add_vec(bytes, "qwen test logits payload")?;
                Ok(capacity.finish())
            }
        }
    }
}

#[cfg(test)]
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

#[cfg(not(test))]
pub(crate) struct Qwen3AsrLlmFusedLogitsHeadSpec<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
}

impl Qwen3AsrLlmLogitsHead {
    /// Quotes the exact host representation branch used by the logits loader.
    /// A directly executable canonical output matrix remains mmap-backed;
    /// otherwise the loader retains one dequantized f32 matrix. The native
    /// graph arena is quoted independently by the backend allocator.
    pub(crate) fn quoted_system_memory_bytes_from_reader(
        reader: &GgufTensorDataReader,
        output_weight_tensor_name: &'static str,
        d_model: usize,
        vocab_size: usize,
        backend: GgmlCpuGraphBackend,
    ) -> Result<(u64, u64), String> {
        let output = reader
            .tensor_index()
            .get(output_weight_tensor_name)
            .ok_or_else(|| format!("required tensor '{output_weight_tensor_name}' is missing"))?;
        resolve_output_weight_layout(&output.dims, d_model, vocab_size)
            .map_err(|error| error.to_string())?;

        let mut retained = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        let norm_bytes = d_model
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "logits output norm quote overflowed".to_string())?;
        retained.add_usize(norm_bytes, "logits output norm quote")?;

        let direct_mapped =
            logits_head_ggml_enabled(backend) && output.dims == [d_model as u64, vocab_size as u64];
        if direct_mapped {
            retained.add(
                GgufOwnedWeightTensorPayload::quoted_retained_system_memory_bytes(output)?,
                "mapped logits payload metadata quote",
            )?;
            retained.add_usize(
                output
                    .dims
                    .len()
                    .checked_mul(std::mem::size_of::<usize>())
                    .ok_or_else(|| "mapped logits raw dims quote overflowed".to_string())?,
                "mapped logits raw dims quote",
            )?;
        } else {
            let matrix_bytes = d_model
                .checked_mul(vocab_size)
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| "logits output matrix quote overflowed".to_string())?;
            retained.add_usize(matrix_bytes, "logits output matrix quote")?;
        }
        let retained = retained.finish();
        Ok((retained, retained))
    }

    pub(crate) fn new_runtime(
        &self,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Qwen3AsrLlmLogitsHeadRuntime, Qwen3AsrLlmLogitsHeadError> {
        self.new_runtime_with_graph_config(qwen_decoder_graph_config(backend))
    }

    pub(crate) fn supports_native_runtime(&self) -> bool {
        self.ggml_output_weight.is_some()
    }

    pub(crate) fn new_runtime_with_graph_config(
        &self,
        graph_config: GgmlCpuGraphConfig,
    ) -> Result<Qwen3AsrLlmLogitsHeadRuntime, Qwen3AsrLlmLogitsHeadError> {
        let executor = self
            .ggml_output_weight
            .as_ref()
            .map(|output_weight| {
                Qwen3AsrLlmLogitsHeadGraphExecutor::new(
                    self.d_model,
                    self.vocab_size,
                    self.rms_norm_epsilon,
                    &self.output_norm_weight,
                    output_weight,
                    graph_config,
                )
            })
            .transpose()
            .map_err(|source| Qwen3AsrLlmLogitsHeadError::GgmlGraphFailed {
                reason: source.to_string(),
            })?;
        Ok(Qwen3AsrLlmLogitsHeadRuntime {
            head_runtime_identity: self.runtime_identity,
            executor,
            last_compute_evidence: None,
        })
    }

    pub(crate) fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        let mut bytes = crate::models::system_memory_owner::SystemMemoryCapacity::default();
        bytes.add_vec(&self.output_norm_weight, "qwen logits output norm")?;
        if let Some(values) = &self.output_weight_values {
            bytes.add_vec(values, "qwen logits output weight f32")?;
        }
        if let Some(weight) = &self.ggml_output_weight {
            bytes.add_vec(&weight.dims, "qwen logits raw dims")?;
            bytes.add(
                weight.payload.retained_system_memory_bytes()?,
                "qwen logits mapped payload metadata",
            )?;
        }
        Ok(bytes.finish())
    }

    #[cfg(test)]
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
            output_weight_bytes: output_weight.payload.bytes(),
        })
    }

    #[cfg(not(test))]
    pub(crate) fn fused_top1_spec(&self) -> Option<Qwen3AsrLlmFusedLogitsHeadSpec<'_>> {
        // Native compact output is a test-only seam until the shared planner
        // receives device-specific evidence for this exact graph.
        None
    }

    #[cfg(test)]
    pub(crate) fn mapped_output_weight_payload(&self) -> Option<&GgufOwnedWeightTensorPayload> {
        match &self.ggml_output_weight.as_ref()?.payload {
            LogitsWeightPayload::Mapped(payload) => Some(payload),
            LogitsWeightPayload::TestBytes(_) => None,
        }
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
            return Err(Qwen3AsrLlmLogitsHeadError::NonFiniteOutputs);
        }
        Ok(logits)
    }

    fn compute_logits_for_hidden_rows(
        &self,
        hidden: &[f32],
        row_count: usize,
    ) -> Result<Vec<f32>, Qwen3AsrLlmLogitsHeadError> {
        let expected = self
            .d_model
            .checked_mul(row_count)
            .ok_or(Qwen3AsrLlmLogitsHeadError::AllocationOverflow)?;
        if row_count == 0 || hidden.len() != expected {
            return Err(Qwen3AsrLlmLogitsHeadError::InvalidHiddenRowsShape {
                got: hidden.len(),
                row_count,
                hidden_size: self.d_model,
            });
        }
        let output_len = self
            .vocab_size
            .checked_mul(row_count)
            .ok_or(Qwen3AsrLlmLogitsHeadError::AllocationOverflow)?;
        let mut logits = Vec::with_capacity(output_len);
        for row in hidden.chunks_exact(self.d_model) {
            logits.extend(self.compute_logits_for_last_hidden(row)?);
        }
        Ok(logits)
    }

    #[cfg(test)]
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

impl Qwen3AsrLlmLogitsHeadRuntime {
    pub(crate) fn graph_lane(&self) -> Option<(GgmlCpuGraphBackend, bool)> {
        self.executor.as_ref().map(|executor| {
            (
                executor.runner.backend_kind(),
                executor.runner.uses_scheduler(),
            )
        })
    }

    pub(crate) fn compute_logits_for_hidden_rows(
        &mut self,
        head: &Qwen3AsrLlmLogitsHead,
        hidden: &[f32],
        row_count: usize,
    ) -> Result<Vec<f32>, Qwen3AsrLlmLogitsHeadError> {
        self.last_compute_evidence = None;
        self.validate_head(head)?;
        let expected_hidden = head
            .d_model
            .checked_mul(row_count)
            .ok_or(Qwen3AsrLlmLogitsHeadError::AllocationOverflow)?;
        if row_count == 0 || hidden.len() != expected_hidden {
            return Err(Qwen3AsrLlmLogitsHeadError::InvalidHiddenRowsShape {
                got: hidden.len(),
                row_count,
                hidden_size: head.d_model,
            });
        }
        if hidden.iter().any(|value| !value.is_finite()) {
            return Err(Qwen3AsrLlmLogitsHeadError::NonFiniteInputs);
        }
        let logits = if let Some(executor) = self.executor.as_mut() {
            let output = executor.compute_rows(hidden, row_count).map_err(|source| {
                Qwen3AsrLlmLogitsHeadError::GgmlGraphFailed {
                    reason: source.to_string(),
                }
            })?;
            let (logits, evidence) = output.into_parts();
            self.last_compute_evidence = evidence;
            logits
        } else {
            head.compute_logits_for_hidden_rows(hidden, row_count)?
        };
        validate_logits_rows(logits, row_count, head.vocab_size)
    }

    pub(crate) fn compute_logits_for_last_hidden(
        &mut self,
        head: &Qwen3AsrLlmLogitsHead,
        hidden: &[f32],
    ) -> Result<Vec<f32>, Qwen3AsrLlmLogitsHeadError> {
        self.last_compute_evidence = None;
        self.validate_head(head)?;
        if let Some(executor) = self.executor.as_mut() {
            if hidden.len() != head.d_model {
                return Err(Qwen3AsrLlmLogitsHeadError::InvalidHiddenStateShape {
                    got: hidden.len(),
                    expected: head.d_model,
                });
            }
            if hidden.iter().any(|value| !value.is_finite()) {
                return Err(Qwen3AsrLlmLogitsHeadError::NonFiniteInputs);
            }
            let output = executor.compute(hidden).map_err(|source| {
                Qwen3AsrLlmLogitsHeadError::GgmlGraphFailed {
                    reason: source.to_string(),
                }
            })?;
            let (logits, evidence) = output.into_parts();
            self.last_compute_evidence = evidence;
            return validate_logits_rows(logits, 1, head.vocab_size);
        }
        head.compute_logits_for_last_hidden(hidden)
    }

    pub(crate) fn take_compute_evidence(&mut self) -> Option<GgmlSelectionEvidenceRef> {
        self.last_compute_evidence.take()
    }

    #[cfg(test)]
    pub(crate) fn compute_top1_token_for_last_hidden(
        &mut self,
        head: &Qwen3AsrLlmLogitsHead,
        hidden: &[f32],
    ) -> Result<u32, Qwen3AsrLlmLogitsHeadError> {
        self.validate_head(head)?;
        if let Some(executor) = self.executor.as_mut() {
            if hidden.len() != head.d_model {
                return Err(Qwen3AsrLlmLogitsHeadError::InvalidHiddenStateShape {
                    got: hidden.len(),
                    expected: head.d_model,
                });
            }
            if hidden.iter().any(|value| !value.is_finite()) {
                return Err(Qwen3AsrLlmLogitsHeadError::NonFiniteInputs);
            }
            let token_id = executor.compute_top1(hidden).map_err(|source| {
                Qwen3AsrLlmLogitsHeadError::GgmlGraphFailed {
                    reason: source.to_string(),
                }
            })?;
            return validate_top1_token_id(token_id, head.vocab_size);
        }
        head.compute_top1_token_for_last_hidden(hidden)
    }

    #[cfg(test)]
    pub(crate) fn compute_top1_tokens_for_hidden_rows(
        &mut self,
        head: &Qwen3AsrLlmLogitsHead,
        hidden: &[f32],
        row_count: usize,
    ) -> Result<Vec<u32>, Qwen3AsrLlmLogitsHeadError> {
        self.validate_head(head)?;
        let expected_hidden = head
            .d_model
            .checked_mul(row_count)
            .ok_or(Qwen3AsrLlmLogitsHeadError::AllocationOverflow)?;
        if row_count == 0 || hidden.len() != expected_hidden {
            return Err(Qwen3AsrLlmLogitsHeadError::InvalidHiddenRowsShape {
                got: hidden.len(),
                row_count,
                hidden_size: head.d_model,
            });
        }
        if hidden.iter().any(|value| !value.is_finite()) {
            return Err(Qwen3AsrLlmLogitsHeadError::NonFiniteInputs);
        }
        if let Some(executor) = self.executor.as_mut() {
            let token_ids = executor
                .compute_top1_rows(hidden, row_count)
                .map_err(|source| Qwen3AsrLlmLogitsHeadError::GgmlGraphFailed {
                    reason: source.to_string(),
                })?;
            return token_ids
                .into_iter()
                .map(|token_id| validate_top1_token_id(token_id, head.vocab_size))
                .collect();
        }
        let logits = head.compute_logits_for_hidden_rows(hidden, row_count)?;
        let expected = head
            .vocab_size
            .checked_mul(row_count)
            .ok_or(Qwen3AsrLlmLogitsHeadError::AllocationOverflow)?;
        if logits.len() != expected {
            return Err(Qwen3AsrLlmLogitsHeadError::InvalidLogitsRowsShape {
                got: logits.len(),
                row_count,
                vocab_size: head.vocab_size,
            });
        }
        logits
            .chunks_exact(head.vocab_size)
            .map(|row| {
                let mut best_index = 0usize;
                let mut best_value = f32::NEG_INFINITY;
                for (index, value) in row.iter().copied().enumerate() {
                    if value > best_value {
                        best_index = index;
                        best_value = value;
                    }
                }
                u32::try_from(best_index)
                    .map_err(|_| Qwen3AsrLlmLogitsHeadError::AllocationOverflow)
            })
            .collect()
    }

    fn validate_head(
        &self,
        head: &Qwen3AsrLlmLogitsHead,
    ) -> Result<(), Qwen3AsrLlmLogitsHeadError> {
        if self.head_runtime_identity == head.runtime_identity {
            Ok(())
        } else {
            Err(Qwen3AsrLlmLogitsHeadError::RuntimeHeadMismatch)
        }
    }
}

fn validate_logits_rows(
    logits: Vec<f32>,
    row_count: usize,
    vocab_size: usize,
) -> Result<Vec<f32>, Qwen3AsrLlmLogitsHeadError> {
    let expected = vocab_size
        .checked_mul(row_count)
        .ok_or(Qwen3AsrLlmLogitsHeadError::AllocationOverflow)?;
    if logits.len() != expected {
        return Err(Qwen3AsrLlmLogitsHeadError::InvalidLogitsRowsShape {
            got: logits.len(),
            row_count,
            vocab_size,
        });
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(Qwen3AsrLlmLogitsHeadError::NonFiniteOutputs);
    }
    Ok(logits)
}

#[cfg(test)]
pub(crate) fn load_qwen3_llm_logits_head_from_reader(
    reader: &GgufTensorDataReader,
    _runtime_source: &GgmlRuntimeSource,
    metadata: Qwen3AsrExecutionMetadata,
    backend: GgmlCpuGraphBackend,
) -> Result<Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadError> {
    load_qwen3_llm_logits_head_from_reader_with_output_tensor(
        reader,
        _runtime_source,
        metadata,
        OUTPUT_WEIGHT_TENSOR_NAME,
        DEFAULT_RMS_NORM_EPSILON,
        backend,
    )
}

pub(crate) fn load_qwen3_llm_logits_head_from_reader_with_output_tensor(
    reader: &GgufTensorDataReader,
    _runtime_source: &GgmlRuntimeSource,
    metadata: Qwen3AsrExecutionMetadata,
    output_weight_tensor_name: &'static str,
    rms_norm_epsilon: f32,
    backend: GgmlCpuGraphBackend,
) -> Result<Qwen3AsrLlmLogitsHead, Qwen3AsrLlmLogitsHeadError> {
    load_llm_logits_head_from_reader_with_tensor_names(
        reader,
        metadata.llm_d_model,
        metadata.vocab_size,
        OUTPUT_NORM_WEIGHT_TENSOR_NAME,
        output_weight_tensor_name,
        rms_norm_epsilon,
        backend,
    )
}

/// Like [`load_qwen3_llm_logits_head_from_reader_with_output_tensor`] but
/// decoupled from `Qwen3AsrExecutionMetadata` and qwen's own tensor-naming
/// scheme, so a sibling family (e.g. firered-llm's `llm.out_norm.weight` /
/// `llm.lm_head.weight`) can reuse the same RMSNorm+matmul(+optional fused
/// device top-1) logits-head machinery without any Qwen2/Qwen3-specific
/// assumption -- this stage of the pipeline (final hidden -> logits/top-1) is
/// identical across every qwen-family decoder-only LLM.
///
/// Prefer [`super::load_qwen_decoder_tail_from_contract`] at production family
/// call sites so final-norm / logits / embedding shapes stay projected from the
/// shared decoder-tail descriptors rather than a second hand-written geometry.
pub(crate) fn load_llm_logits_head_from_reader_with_tensor_names(
    reader: &GgufTensorDataReader,
    d_model: usize,
    vocab_size: usize,
    output_norm_weight_tensor_name: &'static str,
    output_weight_tensor_name: &'static str,
    rms_norm_epsilon: f32,
    backend: GgmlCpuGraphBackend,
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
    let raw_output_weight = if logits_head_ggml_enabled(backend) {
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
        runtime_identity: next_logits_head_runtime_identity(),
        d_model,
        vocab_size,
        rms_norm_epsilon,
        output_norm_weight,
        #[cfg(test)]
        output_weight_tensor_name,
        output_weight_values,
        output_weight_layout,
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
        .owned_weight_tensor_payload_by_name(output_weight_tensor_name)
        .map_err(map_tensor_read_error)?;
    if payload.dims.as_slice() != [d_model, vocab_size] {
        return Ok(None);
    }
    Ok(Some(OwnedGgmlLogitsWeight {
        ggml_type: payload.element_type.ggml_type(),
        dims: payload.dims.clone(),
        payload: LogitsWeightPayload::Mapped(payload),
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
        graph_config: GgmlCpuGraphConfig,
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

        // The regular autoregressive path projects one row per call, while
        // forced alignment projects every timestamp row in one matrix. Both
        // use the decoder execution tier because they share the same output
        // matrix and execution lane; the batched form amortizes graph setup
        // and GPU submission without introducing a second runtime owner.
        let mut config = graph_config;
        config.set_graph_node_capacity(QWEN3_LLM_LOGITS_GRAPH_NODE_CAPACITY);
        let runner = GgmlCpuGraphRunner::new(config)?;
        let mut arena = runner.start_static_tensor_arena(
            GgmlCpuGraphConfig::metadata_context_bytes(QWEN3_LLM_LOGITS_STATIC_TENSOR_COUNT),
        )?;
        let norm = arena.new_tensor_2d_f32(d_model, 1, "qwen_llm_logits_output_norm_weight")?;
        let weight = arena.new_matmul_weight_2d_typed(
            d_model,
            vocab_size,
            output_weight.ggml_type,
            "qwen_llm_logits_output_weight",
        )?;
        arena.set_f32_slice(
            norm,
            output_norm_weight,
            "qwen_llm_logits_output_norm_weight",
        )?;
        arena.set_bytes_slice(
            weight,
            output_weight.payload.bytes(),
            "qwen_llm_logits_output_weight",
        )?;
        Ok(Self {
            d_model,
            vocab_size,
            rms_norm_epsilon,
            runner,
            arena,
            output_norm_weight: norm,
            output_weight: weight,
        })
    }

    fn compute(
        &mut self,
        hidden: &[f32],
    ) -> Result<GgmlComputeOutput<Vec<f32>>, GgmlCpuGraphError> {
        self.compute_rows(hidden, 1)
    }

    fn compute_rows(
        &mut self,
        hidden: &[f32],
        row_count: usize,
    ) -> Result<GgmlComputeOutput<Vec<f32>>, GgmlCpuGraphError> {
        let expected_hidden =
            self.d_model
                .checked_mul(row_count)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "logits head hidden rows shape overflow",
                })?;
        if row_count == 0 || hidden.len() != expected_hidden {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "logits head hidden rows shape mismatch",
            });
        }
        let output_len =
            self.vocab_size
                .checked_mul(row_count)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "logits head output rows shape overflow",
                })?;
        let mut graph = self.runner.start_graph();
        let hidden_tensor =
            graph.new_tensor_2d_f32(self.d_model, row_count, "qwen_llm_logits_hidden_rows")?;
        graph.set_input(hidden_tensor)?;
        let normed = graph.rms_norm(hidden_tensor, self.rms_norm_epsilon)?;
        let normed = graph.mul(normed, self.arena.graph_tensor(self.output_norm_weight))?;
        let logits = graph.mul_mat(self.arena.graph_tensor(self.output_weight), normed)?;
        graph.set_output(logits)?;
        graph.prepare_outputs_for_upload(&[logits])?;
        graph.set_f32_slice(hidden_tensor, hidden, "qwen_llm_logits_hidden_rows")?;
        graph.compute_output_f32_with_evidence(logits, output_len)
    }

    #[cfg(test)]
    fn compute_top1(&mut self, hidden: &[f32]) -> Result<i32, GgmlCpuGraphError> {
        if hidden.len() != self.d_model {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "logits head hidden width mismatch",
            });
        }
        // Deliberately per-call. A prepared standalone logits-head top-1 graph
        // can alias stale i32 output storage on GPU-class non-scheduler backends
        // and has segfaulted under scheduler-backed decode. The hot greedy path is fused
        // into the resident whole-decoder graph instead; keep this shared Qwen
        // helper as a simple fallback with no hidden persistent crash path.
        self.compute_top1_rows(hidden, 1)?.into_iter().next().ok_or(
            GgmlCpuGraphError::OutputByteSizeMismatch {
                expected: std::mem::size_of::<i32>(),
                actual: 0,
            },
        )
    }

    #[cfg(test)]
    fn compute_top1_rows(
        &mut self,
        hidden: &[f32],
        row_count: usize,
    ) -> Result<Vec<i32>, GgmlCpuGraphError> {
        let expected_hidden =
            self.d_model
                .checked_mul(row_count)
                .ok_or(GgmlCpuGraphError::UnsupportedInputs {
                    reason: "logits head top-1 hidden rows shape overflow",
                })?;
        if row_count == 0 || hidden.len() != expected_hidden {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "logits head top-1 hidden rows shape mismatch",
            });
        }
        let mut graph = self.runner.start_graph();
        let hidden_tensor =
            graph.new_tensor_2d_f32(self.d_model, row_count, "qwen_llm_logits_top1_hidden_rows")?;
        graph.set_input(hidden_tensor)?;
        let normed = graph.rms_norm(hidden_tensor, self.rms_norm_epsilon)?;
        let normed = graph.mul(normed, self.arena.graph_tensor(self.output_norm_weight))?;
        let logits = graph.mul_mat(self.arena.graph_tensor(self.output_weight), normed)?;
        let top1 = graph.top1_argmax_first_max(logits)?;
        graph.set_output(top1)?;
        graph.prepare_outputs_for_upload(&[top1])?;
        graph.set_f32_slice(hidden_tensor, hidden, "qwen_llm_logits_top1_hidden_rows")?;
        graph
            .compute_output_i32(top1, row_count)?
            .into_iter()
            .map(|token_id| {
                device_top1_token_id(token_id, self.vocab_size).and_then(|id| {
                    i32::try_from(id).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
                        reason: "device top-1 token id does not fit i32",
                    })
                })
            })
            .collect()
    }
}

#[cfg(test)]
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

pub(crate) fn logits_head_ggml_enabled(backend: GgmlCpuGraphBackend) -> bool {
    parse_env_flag(
        std::env::var(OPENASR_QWEN3_LLM_LOGITS_GGML_ENV)
            .ok()
            .as_deref(),
        logits_head_ggml_default_enabled(backend),
    )
}

fn logits_head_ggml_default_enabled(backend: GgmlCpuGraphBackend) -> bool {
    logits_head_ggml_default_enabled_for_backend(qwen_runtime_graph_config(backend).backend)
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
    use std::collections::BTreeMap;
    use std::ffi::OsString;

    use crate::models::mapped_token_embedding::load_mapped_token_embedding_table_from_reader;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};

    use super::*;

    #[test]
    fn tied_embedding_and_logits_share_one_mmap_payload_range() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime_path = temp.path().join("qwen-tied-embedding.gguf");
        let tied_weight_name = "token_embd.weight";
        let output_norm_name = "output_norm.weight";
        let spec = TinyGgufFixtureSpec::new(BTreeMap::new())
            .with_tensor_shape(tied_weight_name, [2_u64, 3_u64])
            .with_tensor_shape(output_norm_name, [2_u64]);
        write_tiny_gguf_runtime_source(&runtime_path, &spec).expect("write tied fixture");
        let runtime_source =
            crate::validate_ggml_runtime_source_path(&runtime_path).expect("validate tied fixture");
        let reader = GgufTensorDataReader::from_runtime_source(&runtime_source).expect("reader");

        crate::test_process_env::with_test_process_env(
            [(OPENASR_QWEN3_LLM_LOGITS_GGML_ENV, Some(OsString::from("1")))],
            || {
                let embedding =
                    load_mapped_token_embedding_table_from_reader(&reader, tied_weight_name, 2, 3)
                        .expect("mapped embedding");
                let logits = load_llm_logits_head_from_reader_with_tensor_names(
                    &reader,
                    2,
                    3,
                    output_norm_name,
                    tied_weight_name,
                    DEFAULT_RMS_NORM_EPSILON,
                    GgmlCpuGraphBackend::Cpu,
                )
                .expect("mapped tied logits");
                let embedding_payload = embedding.mapped_payload().expect("mapped embedding view");
                let logits_payload = logits
                    .mapped_output_weight_payload()
                    .expect("mapped logits view");
                assert!(
                    embedding_payload.shares_backing_range(logits_payload),
                    "tied consumers must retain views into one physical mmap range"
                );
            },
        );
    }

    #[test]
    fn logits_head_hidden_vocab_layout_matches_manual_matmul() {
        let head = Qwen3AsrLlmLogitsHead {
            runtime_identity: next_logits_head_runtime_identity(),
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
            runtime_identity: next_logits_head_runtime_identity(),
            d_model: 4,
            vocab_size: 8,
            rms_norm_epsilon: DEFAULT_RMS_NORM_EPSILON,
            output_norm_weight: vec![1.0; 4],
            output_weight_tensor_name: OUTPUT_WEIGHT_TENSOR_NAME,
            output_weight_values: Some(vec![0.0; 32]),
            output_weight_layout: OutputWeightLayout::HiddenVocab,
            ggml_output_weight: None,
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
    fn logits_head_host_batch_matches_concatenated_single_rows() {
        let head = Qwen3AsrLlmLogitsHead {
            runtime_identity: next_logits_head_runtime_identity(),
            d_model: 2,
            vocab_size: 3,
            rms_norm_epsilon: DEFAULT_RMS_NORM_EPSILON,
            output_norm_weight: vec![0.75, 1.25],
            output_weight_tensor_name: OUTPUT_WEIGHT_TENSOR_NAME,
            output_weight_values: Some(vec![
                0.1, 0.2, //
                0.3, -0.4, //
                -0.5, 0.6,
            ]),
            output_weight_layout: OutputWeightLayout::VocabHidden,
            ggml_output_weight: None,
        };
        let hidden = [1.0, 2.0, 2.0, 1.0, -1.0, 0.5];
        let batch = head
            .compute_logits_for_hidden_rows(&hidden, 3)
            .expect("host batch logits");
        let expected = hidden
            .chunks_exact(2)
            .flat_map(|row| {
                head.compute_logits_for_last_hidden(row)
                    .expect("single-row logits")
            })
            .collect::<Vec<_>>();
        assert_eq!(batch, expected);
    }

    #[test]
    fn logits_head_batch_rejects_empty_mismatched_and_non_finite_rows() {
        let head = ggml_logits_head(vec![1.0, 1.0]);
        let mut runtime = head
            .new_runtime(GgmlCpuGraphBackend::Cpu)
            .expect("build explicit logits runtime");
        assert!(matches!(
            runtime.compute_top1_tokens_for_hidden_rows(&head, &[], 0),
            Err(Qwen3AsrLlmLogitsHeadError::InvalidHiddenRowsShape { .. })
        ));
        assert!(matches!(
            runtime.compute_logits_for_hidden_rows(&head, &[], 0),
            Err(Qwen3AsrLlmLogitsHeadError::InvalidHiddenRowsShape { .. })
        ));
        assert!(matches!(
            runtime.compute_top1_tokens_for_hidden_rows(&head, &[1.0, 2.0, 3.0], 2),
            Err(Qwen3AsrLlmLogitsHeadError::InvalidHiddenRowsShape { .. })
        ));
        assert!(matches!(
            runtime.compute_logits_for_hidden_rows(&head, &[1.0, 2.0, 3.0], 2),
            Err(Qwen3AsrLlmLogitsHeadError::InvalidHiddenRowsShape { .. })
        ));
        assert!(matches!(
            runtime.compute_top1_tokens_for_hidden_rows(&head, &[1.0, f32::NAN, 2.0, 3.0], 2,),
            Err(Qwen3AsrLlmLogitsHeadError::NonFiniteInputs)
        ));
        assert!(matches!(
            runtime.compute_logits_for_hidden_rows(&head, &[1.0, f32::NAN, 2.0, 3.0], 2,),
            Err(Qwen3AsrLlmLogitsHeadError::NonFiniteInputs)
        ));
    }

    #[test]
    fn logits_row_postcondition_rejects_wrong_shape_and_non_finite_outputs() {
        assert!(matches!(
            validate_logits_rows(vec![1.0, 2.0], 1, 3),
            Err(Qwen3AsrLlmLogitsHeadError::InvalidLogitsRowsShape { .. })
        ));
        assert!(matches!(
            validate_logits_rows(vec![1.0, f32::NAN, 2.0], 1, 3),
            Err(Qwen3AsrLlmLogitsHeadError::NonFiniteOutputs)
        ));
        assert!(matches!(
            validate_logits_rows(vec![1.0, f32::INFINITY, 2.0], 1, 3),
            Err(Qwen3AsrLlmLogitsHeadError::NonFiniteOutputs)
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

    fn ggml_logits_head(output_norm_weight: Vec<f32>) -> Qwen3AsrLlmLogitsHead {
        // A valid rank-2 [d_model x vocab_size] f32 weight for d_model=2,
        // vocab_size=3, matching the fused-logits fixture in
        // `llm_transformer::tests::fused_logits_top1_selects_first_token_on_equal_logit_tie`.
        let output_weight_values: [f32; 6] = [
            0.1, 0.0, //
            0.3, 0.0, //
            0.3, 0.0,
        ];
        Qwen3AsrLlmLogitsHead {
            runtime_identity: next_logits_head_runtime_identity(),
            d_model: 2,
            vocab_size: 3,
            rms_norm_epsilon: DEFAULT_RMS_NORM_EPSILON,
            output_norm_weight,
            output_weight_tensor_name: OUTPUT_WEIGHT_TENSOR_NAME,
            output_weight_values: Some(output_weight_values.to_vec()),
            output_weight_layout: OutputWeightLayout::VocabHidden,
            ggml_output_weight: Some(OwnedGgmlLogitsWeight {
                ggml_type: crate::ggml_runtime::GGML_TYPE_F32,
                dims: vec![2, 3],
                payload: LogitsWeightPayload::TestBytes(
                    output_weight_values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                ),
            }),
        }
    }

    #[test]
    fn explicit_runtime_reuses_one_native_graph_and_rejects_head_aliasing() {
        let first = ggml_logits_head(vec![1.0, 1.0]);
        let mut runtime = first
            .new_runtime(GgmlCpuGraphBackend::Cpu)
            .expect("build explicit logits runtime");
        let first_token = runtime
            .compute_top1_token_for_last_hidden(&first, &[1.0, 2.0])
            .expect("first call");
        let second_token = runtime
            .compute_top1_token_for_last_hidden(&first, &[1.0, 2.0])
            .expect("same owner reuses its graph");
        assert_eq!(first_token, second_token);

        let distinct_head = ggml_logits_head(vec![1.0, 1.0]);
        assert!(matches!(
            runtime.compute_top1_token_for_last_hidden(&distinct_head, &[1.0, 2.0]),
            Err(Qwen3AsrLlmLogitsHeadError::RuntimeHeadMismatch)
        ));
    }

    #[test]
    fn explicit_runtime_batch_top1_matches_repeated_single_row_graphs() {
        let head = ggml_logits_head(vec![1.0, 0.75]);
        let hidden = [1.0, 2.0, 2.0, 1.0, -1.0, 0.5];
        let mut runtime = head
            .new_runtime(GgmlCpuGraphBackend::Cpu)
            .expect("build explicit logits runtime");
        let batched_logits = runtime
            .compute_logits_for_hidden_rows(&head, &hidden, 3)
            .expect("batched logits");
        let repeated_logits = hidden
            .chunks_exact(2)
            .flat_map(|row| {
                runtime
                    .compute_logits_for_last_hidden(&head, row)
                    .expect("single-row logits")
            })
            .collect::<Vec<_>>();
        assert_eq!(batched_logits, repeated_logits);
        let batched = runtime
            .compute_top1_tokens_for_hidden_rows(&head, &hidden, 3)
            .expect("batched top-1");
        let repeated = hidden
            .chunks_exact(2)
            .map(|row| {
                runtime
                    .compute_top1_token_for_last_hidden(&head, row)
                    .expect("single-row top-1")
            })
            .collect::<Vec<_>>();
        assert_eq!(batched, repeated);
    }
}
