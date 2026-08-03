//! MiMo-ASR decoder-state topology. Its exact position bound is assembled
//! from pack metadata, the tokenized prompt wrapper, and the same frontend and
//! generation shape facts used by execution.
//!
//! Reuses, never re-derives:
//! - KV geometry comes straight off the parsed pack (`mimo.llm.*` keys).
//! - The audio-token shape is the pack-carried stride product: mel
//!   `sample_rate/hop_length` through the
//!   tokenizer conv stem (`conv1_stride * conv2_stride * down_sample_stride`,
//!   25Hz RVQ frames for the shipped pack) down to one LLM position per
//!   `group_size` frames (the input-local group downcast).
//! - The generation budget reuses the executor's own constant, so planning
//!   and allocation share one `prompt + generation` contract.

use crate::capacity::KvGeometry;
use crate::capacity::topology::{
    DecoderStateDemandScope, DecoderStateTopology, InvocationEnvelope, InvocationShapeInput,
    PositionBoundProof, StateDemand, StateKind, TopologyError,
    causal_prefix_positions_with_context_cap,
};
use crate::models::audio_frontend::{PadMode, StftFramer};
use crate::models::ggml_asr_executor::{
    GgmlAsrDecoderStatePlanningError, GgmlAsrDecoderStatePlanningInput,
};
use crate::nn::decoder::LlmKvCacheSpec;

use super::executor::MIMO_ASR_MAX_GENERATED_TOKENS;
use super::mel_frontend::{MimoResampleShapeError, mimo_resampled_output_sample_count};
use super::runtime_contract::{
    MimoAudiotokMetadata, MimoInlocalMetadata, MimoLlmMetadata, MimoMelMetadata,
};

pub(crate) const MIMO_ASR_SELF_KV_STATE_ID: &str = "mimo-asr.decoder.self_kv";
pub(crate) const MIMO_ASR_DECODER_STATE_STREAMS:
    &[crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract] = &[
    crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract::new(
        MIMO_ASR_SELF_KV_STATE_ID,
        StateKind::SelfAttentionKv,
    ),
];

pub(crate) fn plan_mimo_asr_decoder_state(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    let family = "mimo-asr";
    let metadata = input.preflight.metadata.as_ref();
    let map_metadata = |error: super::runtime_contract::MimoMetadataError| {
        GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
            family,
            reason: error.to_string(),
        }
    };
    let llm = super::runtime_contract::parse_mimo_llm_metadata(metadata).map_err(map_metadata)?;
    let mel = super::runtime_contract::parse_mimo_mel_metadata(metadata).map_err(map_metadata)?;
    let audiotok =
        super::runtime_contract::parse_mimo_audiotok_metadata(metadata).map_err(map_metadata)?;
    let inlocal =
        super::runtime_contract::parse_mimo_inlocal_metadata(metadata).map_err(map_metadata)?;
    let special =
        super::runtime_contract::parse_mimo_special_tokens(metadata).map_err(map_metadata)?;
    let tokenizer = super::tokenizer::MimoAsrTokenizer::from_gguf_metadata(metadata, special)
        .map_err(
            |error| GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
                family,
                reason: error.to_string(),
            },
        )?;
    let prompt =
        super::decode_prompt::build_mimo_asr_decode_prompt(&tokenizer, 1).map_err(|error| {
            GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
                family,
                reason: error.to_string(),
            }
        })?;
    let fixed_prompt_tokens = prompt.token_ids.len().checked_sub(1).ok_or_else(|| {
        GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
            family,
            reason: "dummy audio span was absent from the tokenized prompt".to_string(),
        }
    })?;
    let geometry = mimo_asr_kv_geometry(&llm);
    let spec = crate::models::qwen::resolve_qwen_family_production_kv_cache_policy(
        input.backend,
        geometry.head_dim,
    )
    .to_spec();
    crate::capacity::topology::DecoderStatePlan::build(
        &MimoAsrDecoderStateTopology::new(llm, mel, audiotok, inlocal, fixed_prompt_tokens, spec),
        input.invocation,
        input.envelope,
    )
    .map_err(|source| GgmlAsrDecoderStatePlanningError::Topology { family, source })
}

/// The backbone KV geometry the loaded pack advertises.
pub(crate) fn mimo_asr_kv_geometry(llm: &MimoLlmMetadata) -> KvGeometry {
    KvGeometry {
        n_layers: llm.n_layers,
        kv_heads: llm.n_kv_heads,
        head_dim: llm.head_dim,
    }
}

const MIMO_MAX_INPUT_MILLISECONDS: u128 = 30_000;

fn mimo_conv_out_len(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<usize, TopologyError> {
    if stride == 0 {
        return Err(TopologyError::DivisionByZero);
    }
    let padded = input
        .checked_add(
            padding
                .checked_mul(2)
                .ok_or(TopologyError::ArithmeticOverflow {
                    operation: "mimo convolution padding",
                })?,
        )
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "mimo convolution padded length",
        })?;
    let Some(tail) = padded.checked_sub(kernel) else {
        return Ok(0);
    };
    (tail / stride)
        .checked_add(1)
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "mimo convolution output length",
        })
}

/// Exact number of MiMo LLM audio positions produced from prepared input
/// samples, following resample -> centered STFT -> conv1/conv2/downsample ->
/// input-local whole-group truncation.
pub(crate) fn mimo_asr_audio_group_count_for_samples(
    mel: &MimoMelMetadata,
    audiotok: &MimoAudiotokMetadata,
    inlocal: &MimoInlocalMetadata,
    input_samples: usize,
    input_sample_rate_hz: usize,
) -> Result<usize, TopologyError> {
    if mel.hop_length == 0 || inlocal.group_size == 0 {
        return Err(TopologyError::DivisionByZero);
    }
    let resampled =
        mimo_resampled_output_sample_count(input_samples, input_sample_rate_hz, mel.sample_rate_hz)
            .map_err(|error| match error {
                MimoResampleShapeError::ZeroSampleRate => TopologyError::DivisionByZero,
                MimoResampleShapeError::ArithmeticOverflow { operation } => {
                    TopologyError::ArithmeticOverflow { operation }
                }
            })?;
    let mel_frames = StftFramer::output_frame_count_for(
        mel.n_fft,
        mel.hop_length,
        PadMode::ReflectCenter,
        resampled,
    )
    .map_err(|error| TopologyError::Unavailable {
        reason: format!("mimo-asr STFT shape is invalid: {error}"),
    })?;
    let conv_padding = audiotok.conv_kernel_size / 2;
    let conv1 = mimo_conv_out_len(
        mel_frames,
        audiotok.conv_kernel_size,
        audiotok.conv1_stride,
        conv_padding,
    )?;
    let conv2 = mimo_conv_out_len(
        conv1,
        audiotok.conv_kernel_size,
        audiotok.conv2_stride,
        conv_padding,
    )?;
    let rvq_frames = mimo_conv_out_len(conv2, 2, audiotok.down_sample_stride, 0)?;
    let audio_groups = rvq_frames / inlocal.group_size;
    if audio_groups == 0 {
        return Err(TopologyError::Unavailable {
            reason: "mimo-asr audio is too short to produce one input-local group".to_string(),
        });
    }
    Ok(audio_groups)
}

#[derive(Debug, Clone)]
pub(crate) struct MimoAsrDecoderStateTopology {
    llm: MimoLlmMetadata,
    mel: MimoMelMetadata,
    audiotok: MimoAudiotokMetadata,
    inlocal: MimoInlocalMetadata,
    fixed_prompt_tokens: usize,
    kv_spec: LlmKvCacheSpec,
}

impl MimoAsrDecoderStateTopology {
    pub(crate) fn new(
        llm: MimoLlmMetadata,
        mel: MimoMelMetadata,
        audiotok: MimoAudiotokMetadata,
        inlocal: MimoInlocalMetadata,
        fixed_prompt_tokens: usize,
        kv_spec: LlmKvCacheSpec,
    ) -> Self {
        Self {
            llm,
            mel,
            audiotok,
            inlocal,
            fixed_prompt_tokens,
            kv_spec,
        }
    }
}

impl DecoderStateTopology for MimoAsrDecoderStateTopology {
    fn demands(
        &self,
        scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
    ) -> Result<Vec<StateDemand>, TopologyError> {
        let invocation = match scope {
            DecoderStateDemandScope::ExactInvocation(invocation) => invocation,
            DecoderStateDemandScope::StableEnvelope(envelope) => envelope.maximum_invocation(),
        };
        let duration_numerator = u128::try_from(invocation.samples())
            .ok()
            .and_then(|samples| samples.checked_mul(1_000))
            .ok_or(TopologyError::ArithmeticOverflow {
                operation: "mimo-asr input duration numerator",
            })?;
        let duration_limit = u128::from(invocation.sample_rate_hz().get())
            .checked_mul(MIMO_MAX_INPUT_MILLISECONDS)
            .ok_or(TopologyError::ArithmeticOverflow {
                operation: "mimo-asr input duration limit",
            })?;
        if duration_numerator > duration_limit {
            let max_samples = usize::try_from(duration_limit / 1_000).map_err(|_| {
                TopologyError::ArithmeticOverflow {
                    operation: "mimo-asr input duration limit conversion",
                }
            })?;
            return Err(TopologyError::InvocationSampleLimitExceeded {
                required_samples: invocation.samples(),
                max_samples,
            });
        }
        let audio_groups = mimo_asr_audio_group_count_for_samples(
            &self.mel,
            &self.audiotok,
            &self.inlocal,
            invocation.samples(),
            invocation.sample_rate_hz().get() as usize,
        )?;
        let prompt_positions = self.fixed_prompt_tokens.checked_add(audio_groups).ok_or(
            TopologyError::ArithmeticOverflow {
                operation: "mimo-asr prompt positions",
            },
        )?;
        let positions = causal_prefix_positions_with_context_cap(
            MIMO_ASR_SELF_KV_STATE_ID,
            prompt_positions,
            MIMO_ASR_MAX_GENERATED_TOKENS,
            self.llm.max_positions,
        )?;
        Ok(vec![StateDemand::from_llm_kv_geometry(
            MIMO_ASR_SELF_KV_STATE_ID,
            StateKind::SelfAttentionKv,
            positions,
            self.llm.max_positions,
            mimo_asr_kv_geometry(&self.llm),
            self.kv_spec,
            invocation.sequences().get() as usize,
            PositionBoundProof::Exact,
        )?])
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::capacity::kv_bytes_per_position;
    use crate::capacity::topology::{DecoderStatePlan, InvocationEnvelope, StateKind};
    use crate::nn::decoder::LlmKvCacheSpec;

    /// Real-checkpoint-shaped facts (the same values `runtime_contract`'s
    /// `full_metadata` fixture parses: 36L Qwen2 backbone with 8 KV heads at
    /// head_dim 128; 24kHz/240-hop mel; stride-1/2/2 conv stem; group 4).
    fn reference_llm() -> MimoLlmMetadata {
        MimoLlmMetadata {
            n_layers: 36,
            d_model: 4096,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            ffn_dim: 11008,
            vocab_size: 151_680,
            max_positions: 8192,
            rms_norm_epsilon: 1e-6,
            rope_theta: 640_000.0,
        }
    }

    fn reference_mel() -> MimoMelMetadata {
        MimoMelMetadata {
            sample_rate_hz: 24_000,
            n_fft: 960,
            hop_length: 240,
            win_length: 960,
            n_mels: 128,
            log_clip: 1e-7,
        }
    }

    fn reference_audiotok() -> MimoAudiotokMetadata {
        MimoAudiotokMetadata {
            n_layers: 32,
            d_model: 1280,
            n_heads: 20,
            head_dim: 64,
            ffn_dim: 5120,
            skip_layer_id: 3,
            conv_kernel_size: 3,
            conv1_stride: 1,
            conv2_stride: 2,
            down_sample_stride: 2,
            rope_theta: 10_000.0,
            rvq_packed: 8,
            codebook_sizes: vec![1024, 1024, 128, 128, 128, 128, 128, 128],
        }
    }

    fn reference_inlocal() -> MimoInlocalMetadata {
        MimoInlocalMetadata {
            n_layers: 6,
            d_model: 1024,
            n_heads: 64,
            head_dim: 16,
            ffn_dim: 4096,
            rope_theta: 640_000.0,
            group_size: 4,
            audio_channels: 8,
        }
    }

    #[test]
    fn kv_geometry_reads_the_llm_metadata() {
        let geometry = mimo_asr_kv_geometry(&reference_llm());
        assert_eq!(
            geometry,
            KvGeometry {
                n_layers: 36,
                kv_heads: 8,
                head_dim: 128,
            }
        );
        // Feeds the shared KV byte model without error (576 rows/position,
        // the same shape the runtime_contract capacity anchor pins).
        let default = kv_bytes_per_position(&geometry, LlmKvCacheSpec::DEFAULT).expect("default");
        assert_eq!(default.total(), 576 * 768);
    }

    #[test]
    fn exact_topology_includes_the_resampler_terminal_flush() {
        let envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(30_000).unwrap(),
        )
        .unwrap();
        assert_eq!(
            mimo_resampled_output_sample_count(480_000, 16_000, 24_000).unwrap(),
            724_992
        );
        assert_eq!(
            mimo_asr_audio_group_count_for_samples(
                &reference_mel(),
                &reference_audiotok(),
                &reference_inlocal(),
                480_000,
                16_000,
            )
            .unwrap(),
            188
        );
        let plan = DecoderStatePlan::for_envelope(
            &MimoAsrDecoderStateTopology::new(
                reference_llm(),
                reference_mel(),
                reference_audiotok(),
                reference_inlocal(),
                32,
                LlmKvCacheSpec::DEFAULT,
            ),
            envelope,
        )
        .unwrap();
        assert_eq!(
            plan.reserve_positions(StateKind::SelfAttentionKv),
            Some(731)
        );
    }

    #[test]
    fn duration_and_context_boundaries_follow_resample_group_and_greedy_schedule() {
        let rate = NonZeroU32::new(16_000).unwrap();
        let topology = MimoAsrDecoderStateTopology::new(
            reference_llm(),
            reference_mel(),
            reference_audiotok(),
            reference_inlocal(),
            32,
            LlmKvCacheSpec::DEFAULT,
        );
        for seconds in [1, 30] {
            let samples = seconds * 16_000;
            let groups = mimo_asr_audio_group_count_for_samples(
                &reference_mel(),
                &reference_audiotok(),
                &reference_inlocal(),
                samples,
                16_000,
            )
            .unwrap();
            let expected_positions = 32 + groups + MIMO_ASR_MAX_GENERATED_TOKENS - 1;
            let plan = DecoderStatePlan::for_envelope(
                &topology,
                InvocationEnvelope::new(rate, samples).unwrap(),
            )
            .unwrap();
            assert_eq!(
                plan.reserve_positions_by_id(MIMO_ASR_SELF_KV_STATE_ID),
                Some(expected_positions)
            );
        }
        for seconds in [60, 300] {
            assert!(matches!(
                DecoderStatePlan::for_envelope(
                    &topology,
                    InvocationEnvelope::new(rate, seconds * 16_000).unwrap(),
                ),
                Err(TopologyError::InvocationSampleLimitExceeded { .. })
            ));
        }

        let one_second_groups = mimo_asr_audio_group_count_for_samples(
            &reference_mel(),
            &reference_audiotok(),
            &reference_inlocal(),
            16_000,
            16_000,
        )
        .unwrap();
        let physical = 32 + one_second_groups + MIMO_ASR_MAX_GENERATED_TOKENS - 1;
        let mut small = reference_llm();
        small.max_positions = physical;
        assert!(matches!(
            DecoderStatePlan::for_envelope(
                &MimoAsrDecoderStateTopology::new(
                    small,
                    reference_mel(),
                    reference_audiotok(),
                    reference_inlocal(),
                    32,
                    LlmKvCacheSpec::DEFAULT,
                ),
                InvocationEnvelope::new(rate, 16_000).unwrap(),
            ),
            Err(TopologyError::SemanticContextCapExceeded { .. })
        ));
    }
}
