//! [`LongFormVadProvider`] backed by the causal Stream-VAD DFSMN model: the
//! sole long-form VAD engine, run over the whole long-form utterance.

use thiserror::Error;

use super::frontend::SAMPLE_RATE_HZ;
use super::ggml_runtime::FireRedStreamVadGgmlRuntime;
use super::model::{FRAME_SHIFT_MS, FireRedStreamVadModel};
use super::streaming::FireRedStreamingVad;
use super::weights::FireRedStreamVadWeightsError;
use crate::NativeExecutionServices;
use crate::device::{
    execution_policy::{ExecutionIntent, ExecutionPlacement},
    execution_route::enumerate_compute_devices_from_ggml,
};
use crate::ggml_runtime::GgmlCpuGraphBackend;
use crate::longform::{
    LongFormOptions, LongFormVadProvider, LongFormVadProviderError, LongFormVadProviderKind,
    LongFormVadSlice,
};

#[derive(Debug, Error)]
pub enum FireRedStreamVadError {
    #[error("firered Stream-VAD model is unavailable: {0}")]
    Unavailable(#[from] FireRedStreamVadWeightsError),
    #[error("firered Stream-VAD requires {expected} Hz mono audio, got {actual} Hz")]
    UnsupportedSampleRate { expected: u32, actual: u32 },
    #[error("firered Stream-VAD was canceled")]
    Canceled,
    #[error("firered Stream-VAD device graph failed: {reason}")]
    Graph { reason: String },
    #[error("firered Stream-VAD execution policy failed: {reason}")]
    ExecutionPolicy { reason: String },
    #[error("firered Stream-VAD realtime runtime failed: {reason}")]
    RealtimeRuntime { reason: String },
}

/// Neural VAD provider over the process-wide shared Stream-VAD model. Cheap
/// to construct (it only borrows the model), so build one per request.
pub struct FireRedStreamVadProvider {
    model: &'static FireRedStreamVadModel,
    backend: GgmlCpuGraphBackend,
    placement: ExecutionPlacement,
}

const CPU_OFFLINE_CHUNK_SECONDS: usize = 1;
const ACCELERATED_OFFLINE_CHUNK_SECONDS: usize = 32;

pub(super) fn offline_chunk_samples_for_backend(backend: GgmlCpuGraphBackend) -> usize {
    let seconds = if backend == GgmlCpuGraphBackend::Cpu {
        CPU_OFFLINE_CHUNK_SECONDS
    } else {
        ACCELERATED_OFFLINE_CHUNK_SECONDS
    };
    seconds * SAMPLE_RATE_HZ as usize
}

impl FireRedStreamVadProvider {
    fn offline_chunk_samples(&self) -> usize {
        offline_chunk_samples_for_backend(self.backend)
    }

    /// Borrow the shared Stream-VAD model. Returns `None` when the vendored
    /// weights could not be loaded.
    pub fn shared() -> Option<Self> {
        Self::shared_for_backend_and_placement(
            GgmlCpuGraphBackend::Cpu,
            ExecutionPlacement::CpuOnly,
        )
    }

    pub(crate) fn shared_for_backend_and_placement(
        backend: GgmlCpuGraphBackend,
        placement: ExecutionPlacement,
    ) -> Option<Self> {
        super::shared_model().map(|model| Self {
            model,
            backend,
            placement,
        })
    }

    /// Resolve the request's VAD placement through the same request-local
    /// execution policy used by long-form slicing. This keeps external
    /// diarization from silently pinning its VAD sub-stage to CPU when the
    /// caller explicitly selected Metal.
    pub(crate) fn shared_for_intent(
        execution_services: &NativeExecutionServices,
        intent: &ExecutionIntent,
    ) -> Result<Option<Self>, FireRedStreamVadError> {
        let inventory = enumerate_compute_devices_from_ggml(&crate::ggml_available_devices());
        let plan = execution_services
            .policy_resolver()
            .resolve(
                intent.clone(),
                super::AUTO_GPU_POLICY,
                super::execution_capabilities(),
                &inventory,
            )
            .map_err(|error| FireRedStreamVadError::ExecutionPolicy {
                reason: error.to_string(),
            })?;
        let candidate =
            plan.candidates()
                .first()
                .ok_or_else(|| FireRedStreamVadError::ExecutionPolicy {
                    reason: "execution policy returned an empty candidate plan".to_string(),
                })?;
        let backend =
            crate::models::policy_resolved_aux_runtime::resolved_runtime_for_auxiliary_candidate(
                candidate,
            )
            .backend();
        Ok(Self::shared_for_backend_and_placement(
            backend,
            candidate.placement,
        ))
    }

    /// Direct access to per-frame probabilities, for diagnostics/tests.
    pub fn probabilities(&self, samples: &[f32]) -> Vec<f32> {
        self.model.probabilities(samples)
    }

    /// Recording-length-independent host peak for one bounded offline step.
    /// CPU keeps one-second cancellation checkpoints. Accelerated execution
    /// batches 32 seconds of this causal DFSMN per graph: the 15-minute Pareto
    /// sweep found this faster and lower-memory than every larger candidate,
    /// while preserving the exact per-frame output hash. The raw buffer can
    /// contain one chunk plus the fbank overlap tail; geometric Vec growth is
    /// bounded by twice that payload.
    ///
    /// Native ggml contexts, uploaded weights, and graph workspaces are quoted
    /// and admitted by the shared backend-allocation layer when the accelerated
    /// runtime materializes. They must not be charged again here: this outer
    /// reservation owns only the family-local Rust/frontend payload that the
    /// backend cannot observe.
    pub(crate) fn invocation_scratch_peak_bytes(&self) -> u64 {
        let buffered_samples = self.offline_chunk_samples() + super::frontend::FRAME_LENGTH;
        let raw_buffer_bytes = (buffered_samples as u64)
            .saturating_mul(std::mem::size_of::<f32>() as u64)
            .saturating_mul(2);
        raw_buffer_bytes.saturating_add(
            self.model
                .quoted_streaming_chunk_peak_bytes(buffered_samples),
        )
    }

    /// Offline speech slicing with bounded cancellation latency. PCM is scored
    /// in backend-appropriate bounded chunks while the causal DFSMN cache and
    /// fbank overlap tail remain continuous, so chunk size changes scheduling
    /// only, never the output sequence. Realtime streaming remains on its
    /// caller-provided cadence and does not use this offline batching policy.
    pub(crate) fn compute_speech_slices_cancellable(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        options: &LongFormOptions,
        canceled: &dyn Fn() -> bool,
    ) -> Result<Vec<LongFormVadSlice>, FireRedStreamVadError> {
        if sample_rate_hz != SAMPLE_RATE_HZ {
            return Err(FireRedStreamVadError::UnsupportedSampleRate {
                expected: SAMPLE_RATE_HZ,
                actual: sample_rate_hz,
            });
        }
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let mut streaming = FireRedStreamingVad::from_model(self.model);
        let mut device_runtime = if self.backend == GgmlCpuGraphBackend::Cpu {
            None
        } else {
            Some(
                FireRedStreamVadGgmlRuntime::new(self.model, self.backend, self.placement)
                    .map_err(|error| FireRedStreamVadError::Graph {
                        reason: error.to_string(),
                    })?,
            )
        };
        let mut probabilities = Vec::with_capacity(samples.len().div_ceil(FRAME_SAMPLES));
        for chunk in samples.chunks(self.offline_chunk_samples()) {
            if canceled() {
                return Err(FireRedStreamVadError::Canceled);
            }
            let chunk_probabilities = if let Some(runtime) = device_runtime.as_mut() {
                streaming
                    .accept_f32_chunk_with(chunk, |features, frames, cache| {
                        runtime.forward_chunk(features, frames, cache)
                    })
                    .map_err(|error| FireRedStreamVadError::Graph {
                        reason: error.to_string(),
                    })?
            } else {
                streaming.accept_f32_chunk(chunk)
            };
            probabilities.extend(chunk_probabilities);
        }
        if canceled() {
            return Err(FireRedStreamVadError::Canceled);
        }
        Ok(spans_from_probs(&probabilities, samples.len(), options))
    }
}

impl LongFormVadProvider for FireRedStreamVadProvider {
    fn provider_kind(&self) -> LongFormVadProviderKind {
        LongFormVadProviderKind::Custom
    }

    fn compute_speech_slices(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        options: &LongFormOptions,
    ) -> Result<Vec<LongFormVadSlice>, String> {
        self.compute_speech_slices_cancellable(samples, sample_rate_hz, options, &|| false)
            .map_err(|error| error.to_string())
    }

    fn compute_speech_slices_cancellable(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        options: &LongFormOptions,
        canceled: &dyn Fn() -> bool,
    ) -> Result<Vec<LongFormVadSlice>, LongFormVadProviderError> {
        FireRedStreamVadProvider::compute_speech_slices_cancellable(
            self,
            samples,
            sample_rate_hz,
            options,
            canceled,
        )
        .map_err(|error| match error {
            FireRedStreamVadError::Canceled => LongFormVadProviderError::Canceled,
            other => LongFormVadProviderError::Failed {
                reason: other.to_string(),
            },
        })
    }
}

/// Samples consumed per probability frame (10 ms at 16 kHz).
const FRAME_SAMPLES: usize = (SAMPLE_RATE_HZ as u64 * FRAME_SHIFT_MS as u64 / 1000) as usize;

/// Convert per-frame speech probabilities into sample-space speech spans with
/// threshold gating plus min-speech / min-silence hysteresis.
pub(super) fn spans_from_probs(
    probs: &[f32],
    total_samples: usize,
    options: &LongFormOptions,
) -> Vec<LongFormVadSlice> {
    let threshold = options.vad.threshold.clamp(0.0, 1.0);
    let min_speech_frames = ms_to_frames(options.vad.min_speech_duration_ms);
    let min_silence_frames = ms_to_frames(options.vad.min_silence_duration_ms);

    let mut spans = Vec::new();
    let mut in_speech = false;
    let mut speech_start = 0usize;
    let mut trailing_silence = 0usize;

    for (idx, &prob) in probs.iter().enumerate() {
        if prob >= threshold {
            if !in_speech {
                in_speech = true;
                speech_start = idx;
            }
            trailing_silence = 0;
            continue;
        }
        if !in_speech {
            continue;
        }
        trailing_silence += 1;
        if trailing_silence < min_silence_frames {
            continue;
        }
        let speech_end = idx + 1 - trailing_silence;
        push_span(
            &mut spans,
            speech_start,
            speech_end,
            min_speech_frames,
            total_samples,
        );
        in_speech = false;
        trailing_silence = 0;
    }
    if in_speech {
        let speech_end = probs.len() - trailing_silence;
        push_span(
            &mut spans,
            speech_start,
            speech_end,
            min_speech_frames,
            total_samples,
        );
    }
    spans
}

fn push_span(
    spans: &mut Vec<LongFormVadSlice>,
    start_frame: usize,
    end_frame: usize,
    min_speech_frames: usize,
    total_samples: usize,
) {
    if end_frame <= start_frame || end_frame - start_frame < min_speech_frames {
        return;
    }
    let start_sample = (start_frame * FRAME_SAMPLES).min(total_samples);
    let end_sample = (end_frame * FRAME_SAMPLES).min(total_samples);
    if end_sample > start_sample {
        spans.push(LongFormVadSlice {
            start_sample,
            end_sample,
        });
    }
}

fn ms_to_frames(ms: u32) -> usize {
    (ms.div_ceil(FRAME_SHIFT_MS)).max(1) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_chunk_policy_keeps_cpu_responsive_and_batches_accelerators() {
        let model = super::super::shared_model().expect("vendored Stream-VAD model");
        let cpu = FireRedStreamVadProvider {
            model,
            backend: GgmlCpuGraphBackend::Cpu,
            placement: ExecutionPlacement::CpuOnly,
        };
        let metal = FireRedStreamVadProvider {
            model,
            backend: GgmlCpuGraphBackend::Metal,
            placement: ExecutionPlacement::FullDevice,
        };
        assert_eq!(
            cpu.offline_chunk_samples(),
            offline_chunk_samples_for_backend(GgmlCpuGraphBackend::Cpu)
        );
        assert_eq!(
            metal.offline_chunk_samples(),
            offline_chunk_samples_for_backend(GgmlCpuGraphBackend::Metal)
        );
    }
}
