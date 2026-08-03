//! granite-speech capacity derivation: two related surfaces.
//!
//! # 1. Single-decode max audio length
//!
//! Computed from the decoder's trained context window, the fixed prompt text
//! overhead, the generation backstop, and the Q-Former audio-token rate --
//! never a margin-note magic number.
//!
//! ```text
//! audio_token_budget = decoder_max_positions
//!                    - fixed_prompt_tokens
//!                    - max_generated_tokens
//! block_budget       = floor(audio_token_budget / queries_per_block)
//! max_encoder_frames = block_budget * qformer_window_size
//! max_input_samples  = exact_inverse(frontend_shape, max_encoder_frames)
//! ```
//!
//! The nominal rate is 10 tokens/s, but it is not an admissible capacity
//! formula: Q-Former rounds the encoder frames up to complete 15-frame
//! windows. The inverse above includes that integer padding exactly. For the
//! shipped geometry, 381 whole seconds fit and 382 seconds require one token
//! too many.
//!
//! The executor rejects a single buffer longer than
//! [`GRANITE_SPEECH_MAX_INPUT_SECONDS`] with a typed `AudioTooLong` error
//! (fail-closed, no silent truncation). Longer recordings are the shared
//! longform `SharedWindow` slicer's job (`DEFAULT_ENCODER_CHUNK_SECONDS` =
//! 30s); that slice is well under the decoder-context-derived ceiling, so the
//! two bounds do not fight -- the executor protects the 4096-token context
//! against a direct over-limit buffer, the slicer keeps ordinary longform
//! work inside a comfortable window.
//!
//! # 2. Persistent decoder state
//!
//! [`GraniteSpeechDecoderStateTopology`] reuses the exact Q-Former integer
//! shape oracle and decode generation budget. It returns the current logical
//! self-KV span and the stable session-envelope span independently. Physical
//! admission is performed later from the native backend quote; this family
//! module never guesses host/device bytes from a synthetic two-copy model.
//!
//! Production reads the derived max-input-seconds constant below. The pure
//! derivation helpers stay unit-tested so a future pack-carried
//! `max_position_embeddings` key can replace the architecture constant
//! without silent drift.
use crate::capacity::topology::{
    DecoderStateDemandScope, DecoderStateTopology, InvocationEnvelope, InvocationShapeInput,
    PositionBoundProof, StateBytes, StateDemand, StateKind, TopologyError,
    causal_prefix_positions_with_context_cap,
};
use crate::models::ggml_asr_executor::{
    GgmlAsrDecoderStatePlanningError, GgmlAsrDecoderStatePlanningInput,
};

use super::decoder_graph::GraniteSpeechDecoderConfig;
use super::executor::GRANITE_SPEECH_MAX_GENERATED_TOKENS;
use super::frontend::{HOP_LENGTH, SAMPLE_RATE_HZ};
use super::qformer::GraniteSpeechProjectorConfig;

/// Decoder training context (`text_config.max_position_embeddings` on the
/// shipped `ibm-granite/granite-speech-4.1-2b` checkpoint = 4096). Pack
/// metadata does not yet carry a max-position key, so this architecture
/// constant is the source of truth (cited by `encoder_graph.rs`'s long-audio
/// note). When a pack revision adds the key, prefer the pack value and pin
/// it equal to this constant for the shipped 4.1-2b shape.
pub(crate) const GRANITE_SPEECH_DECODER_MAX_POSITIONS: usize = 4096;
pub(crate) const GRANITE_SPEECH_SELF_KV_STATE_ID: &str = "granite-speech.decoder.self_kv";
pub(crate) const GRANITE_SPEECH_DECODER_STATE_STREAMS:
    &[crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract] = &[
    crate::models::ggml_asr_executor::GgmlAsrDecoderStateStreamContract::new(
        GRANITE_SPEECH_SELF_KV_STATE_ID,
        StateKind::SelfAttentionKv,
    ),
];

/// Front-end frame-stack factor: after the 80-mel STFT the extractor drops an
/// odd trailing frame and concatenates pairs of 80-dim frames into 160-dim
/// (`frontend.rs`), halving the frame rate before the Conformer encoder.
pub(crate) const GRANITE_SPEECH_ENCODER_FRAME_STACK: usize = 2;
pub(crate) const GRANITE_SPEECH_SAMPLE_RATE_HZ: u32 = 16_000;

/// Fixed text-token overhead of the default transcription prompt
/// (`USER: <|audio|>can you transcribe the speech into a written format?\n ASSISTANT:`)
/// excluding the expanded audio-token span. Measured with the granite-4.0
/// GPT2-BPE tokenizer against the shipped checkpoint: 19 non-audio tokens
/// once the placeholder expands to at least one audio token (the leading
/// space after `USER:` becomes its own token in that case). A `Keywords:`
/// suffix grows this term; SharedWindow 30s slices leave thousands of tokens
/// of headroom even with long keyword lists, so the default-prompt figure is
/// the right basis for the audio-length ceiling (the binding constraint is
/// audio tokens, not KWB text).
#[cfg(test)]
pub(crate) const GRANITE_SPEECH_FIXED_PROMPT_TOKENS: usize = 19;

/// Audio tokens per second of input for the shipped geometry (10.0).
pub(crate) fn granite_speech_audio_tokens_per_second() -> f32 {
    let projector = GraniteSpeechProjectorConfig::granite_speech_4_1_2b();
    let samples_per_token = HOP_LENGTH
        .checked_mul(GRANITE_SPEECH_ENCODER_FRAME_STACK)
        .and_then(|value| value.checked_mul(projector.downsample_rate))
        .expect("shipped Granite frontend stride fits usize");
    SAMPLE_RATE_HZ / samples_per_token as f32
}

/// Exact inverse of [`granite_speech_audio_token_count_for_samples`] for the
/// largest whole-second input that fits the decoder after prompt and decode
/// positions are reserved. Every operation is checked integer arithmetic;
/// the nominal 10-token/s rate is diagnostics only and never used for
/// admission.
#[cfg(test)]
pub(crate) fn derive_max_input_whole_seconds(
    decoder_max_positions: usize,
    fixed_prompt_tokens: usize,
    max_generated_tokens: usize,
    projector: &GraniteSpeechProjectorConfig,
) -> Result<usize, TopologyError> {
    if GRANITE_SPEECH_SAMPLE_RATE_HZ == 0
        || HOP_LENGTH == 0
        || GRANITE_SPEECH_ENCODER_FRAME_STACK == 0
        || projector.window_size == 0
        || projector.downsample_rate == 0
    {
        return Err(TopologyError::DivisionByZero);
    }
    if !projector
        .window_size
        .is_multiple_of(projector.downsample_rate)
    {
        return Err(TopologyError::Unavailable {
            reason: format!(
                "granite projector window_size {} is not divisible by downsample_rate {}",
                projector.window_size, projector.downsample_rate
            ),
        });
    }
    // This inverse defines which audio requests are semantically legal, not
    // how many rows the current greedy schedule writes. The final sampled
    // token still counts against the model context even though it consumes no
    // self-KV row.
    let audio_token_budget = decoder_max_positions
        .checked_sub(fixed_prompt_tokens)
        .and_then(|value| value.checked_sub(max_generated_tokens))
        .unwrap_or(0);
    let queries_per_block = projector.window_size / projector.downsample_rate;
    let block_budget = audio_token_budget / queries_per_block;
    if block_budget == 0 {
        return Ok(0);
    }
    let max_encoder_frames = block_budget.checked_mul(projector.window_size).ok_or(
        TopologyError::ArithmeticOverflow {
            operation: "granite maximum encoder frames",
        },
    )?;
    // Runtime shape:
    //   mel_frames     = floor(samples / hop) + 1
    //   encoder_frames = floor(mel_frames / frame_stack)
    // Therefore encoder_frames <= E iff
    //   floor(samples / hop) <= (E + 1) * frame_stack - 2.
    let max_hop_quotient = max_encoder_frames
        .checked_add(1)
        .and_then(|value| value.checked_mul(GRANITE_SPEECH_ENCODER_FRAME_STACK))
        .and_then(|value| value.checked_sub(2))
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "granite inverse frontend frame count",
        })?;
    let max_samples = max_hop_quotient
        .checked_add(1)
        .and_then(|value| value.checked_mul(HOP_LENGTH))
        .and_then(|value| value.checked_sub(1))
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "granite maximum input samples",
        })?;
    Ok(max_samples / GRANITE_SPEECH_SAMPLE_RATE_HZ as usize)
}

/// Derived single-decode max input seconds for the shipped 4.1-2b geometry:
/// the nominal division suggests 382.1 seconds, but the exact Q-Former window
/// shape makes 382 seconds consume 3822 audio positions, so
/// `19 + 3822 + 256 = 4097`. Production and the executor use 381 seconds; the
/// test below pins it to [`derive_max_input_whole_seconds`].
pub(crate) const GRANITE_SPEECH_MAX_INPUT_SECONDS: u32 =
    crate::arch::GRANITE_SPEECH_MAX_INVOCATION_SECONDS;
pub(crate) const GRANITE_SPEECH_MAX_INPUT_SAMPLES: usize =
    GRANITE_SPEECH_MAX_INPUT_SECONDS as usize * GRANITE_SPEECH_SAMPLE_RATE_HZ as usize;

/// Exact Q-Former audio-token count for the runtime's centered-STFT,
/// pair-stacking and padded-window projector shape.
pub(crate) fn granite_speech_audio_token_count_for_samples(
    samples: usize,
    projector: &GraniteSpeechProjectorConfig,
) -> Result<usize, TopologyError> {
    if HOP_LENGTH == 0 || projector.window_size == 0 || projector.downsample_rate == 0 {
        return Err(TopologyError::DivisionByZero);
    }
    if !projector
        .window_size
        .is_multiple_of(projector.downsample_rate)
    {
        return Err(TopologyError::Unavailable {
            reason: format!(
                "granite projector window_size {} is not divisible by downsample_rate {}",
                projector.window_size, projector.downsample_rate
            ),
        });
    }
    // ReflectCenter with the shipped even n_fft yields floor(samples/hop)+1
    // frames; the frontend drops one odd tail then stacks pairs.
    let mel_frames =
        (samples / HOP_LENGTH)
            .checked_add(1)
            .ok_or(TopologyError::ArithmeticOverflow {
                operation: "granite mel frame count",
            })?;
    let encoder_frames = mel_frames / GRANITE_SPEECH_ENCODER_FRAME_STACK;
    if encoder_frames == 0 {
        return Err(TopologyError::Unavailable {
            reason: "granite-speech audio is too short to produce one stacked frame".to_string(),
        });
    }
    let blocks = encoder_frames.div_ceil(projector.window_size);
    let queries_per_block = projector.window_size / projector.downsample_rate;
    blocks
        .checked_mul(queries_per_block)
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "granite projector audio-token count",
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GraniteSpeechKvResidence {
    Host,
    Resident,
}

pub(crate) fn plan_granite_speech_decoder_state(
    input: &GgmlAsrDecoderStatePlanningInput<'_>,
) -> Result<crate::capacity::topology::DecoderStatePlan, GgmlAsrDecoderStatePlanningError> {
    let family = "granite-speech";
    let metadata = input.preflight.metadata.as_ref();
    let decoder = super::runtime_contract::parse_decoder_metadata(metadata).map_err(|error| {
        GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
            family,
            reason: error.to_string(),
        }
    })?;
    let projector =
        super::runtime_contract::parse_projector_metadata(metadata).map_err(|error| {
            GgmlAsrDecoderStatePlanningError::MetadataUnavailable {
                family,
                reason: error.to_string(),
            }
        })?;
    let tokenizer = super::tokenizer::GraniteSpeechTokenizer::from_gguf_metadata(metadata)
        .map_err(
            |error| GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
                family,
                reason: error.to_string(),
            },
        )?;
    let prompt_text =
        super::prompt::build_granite_speech_prompt_text(input.request_options.phrase_bias.as_ref());
    let one_audio_prompt = super::prompt::build_audio_prompt_token_ids(&tokenizer, &prompt_text, 1)
        .map_err(
            |error| GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
                family,
                reason: error.to_string(),
            },
        )?;
    let fixed_prompt_tokens = one_audio_prompt.len().checked_sub(1).ok_or_else(|| {
        GgmlAsrDecoderStatePlanningError::PromptTokenCountUnavailable {
            family,
            reason: "granite prompt did not contain its one dummy audio token".to_string(),
        }
    })?;
    let residence = match input.backend {
        crate::ggml_runtime::GgmlCpuGraphBackend::Cpu => GraniteSpeechKvResidence::Host,
        crate::ggml_runtime::GgmlCpuGraphBackend::Metal
        | crate::ggml_runtime::GgmlCpuGraphBackend::Gpu => GraniteSpeechKvResidence::Resident,
    };
    crate::capacity::topology::DecoderStatePlan::build(
        &GraniteSpeechDecoderStateTopology::new(decoder, projector, residence, fixed_prompt_tokens),
        input.invocation,
        input.envelope,
    )
    .map_err(|source| GgmlAsrDecoderStatePlanningError::Topology { family, source })
}

fn granite_speech_state_bytes(
    decoder: &GraniteSpeechDecoderConfig,
    positions: usize,
    sequences: usize,
    residence: GraniteSpeechKvResidence,
) -> Result<StateBytes, TopologyError> {
    let bytes = decoder
        .num_layers
        .checked_mul(2)
        .and_then(|value| value.checked_mul(decoder.num_kv_heads))
        .and_then(|value| value.checked_mul(decoder.head_dim))
        .and_then(|value| value.checked_mul(positions))
        .and_then(|value| value.checked_mul(sequences))
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(TopologyError::ArithmeticOverflow {
            operation: "granite decoder KV bytes",
        })?;
    Ok(match residence {
        GraniteSpeechKvResidence::Host => StateBytes {
            host: bytes,
            resident: 0,
        },
        GraniteSpeechKvResidence::Resident => StateBytes {
            host: 0,
            resident: bytes,
        },
    })
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GraniteSpeechDecoderStateTopology {
    decoder: GraniteSpeechDecoderConfig,
    projector: GraniteSpeechProjectorConfig,
    residence: GraniteSpeechKvResidence,
    fixed_prompt_tokens: usize,
}

impl GraniteSpeechDecoderStateTopology {
    pub(crate) const fn new(
        decoder: GraniteSpeechDecoderConfig,
        projector: GraniteSpeechProjectorConfig,
        residence: GraniteSpeechKvResidence,
        fixed_prompt_tokens: usize,
    ) -> Self {
        Self {
            decoder,
            projector,
            residence,
            fixed_prompt_tokens,
        }
    }
}

impl DecoderStateTopology for GraniteSpeechDecoderStateTopology {
    fn demands(
        &self,
        scope: DecoderStateDemandScope<InvocationShapeInput, InvocationEnvelope>,
    ) -> Result<Vec<StateDemand>, TopologyError> {
        let invocation = match scope {
            DecoderStateDemandScope::ExactInvocation(invocation) => invocation,
            DecoderStateDemandScope::StableEnvelope(envelope) => envelope.maximum_invocation(),
        };
        if invocation.sample_rate_hz().get() != GRANITE_SPEECH_SAMPLE_RATE_HZ {
            return Err(TopologyError::UnsupportedSampleRate {
                expected_hz: GRANITE_SPEECH_SAMPLE_RATE_HZ,
                actual_hz: invocation.sample_rate_hz().get(),
            });
        }
        if invocation.samples() > GRANITE_SPEECH_MAX_INPUT_SAMPLES {
            return Err(TopologyError::InvocationSampleLimitExceeded {
                required_samples: invocation.samples(),
                max_samples: GRANITE_SPEECH_MAX_INPUT_SAMPLES,
            });
        }
        let audio_tokens =
            granite_speech_audio_token_count_for_samples(invocation.samples(), &self.projector)?;
        let prompt_positions = self.fixed_prompt_tokens.checked_add(audio_tokens).ok_or(
            TopologyError::ArithmeticOverflow {
                operation: "granite prompt positions",
            },
        )?;
        let positions = causal_prefix_positions_with_context_cap(
            GRANITE_SPEECH_SELF_KV_STATE_ID,
            prompt_positions,
            GRANITE_SPEECH_MAX_GENERATED_TOKENS,
            GRANITE_SPEECH_DECODER_MAX_POSITIONS,
        )?;
        Ok(vec![StateDemand::new(
            GRANITE_SPEECH_SELF_KV_STATE_ID,
            StateKind::SelfAttentionKv,
            positions,
            GRANITE_SPEECH_DECODER_MAX_POSITIONS,
            granite_speech_state_bytes(
                &self.decoder,
                positions,
                invocation.sequences().get() as usize,
                self.residence,
            )?,
            PositionBoundProof::Exact,
        )?])
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::capacity::topology::{DecoderStatePlan, InvocationEnvelope, StateKind};

    #[test]
    fn shipped_geometry_is_ten_audio_tokens_per_second() {
        assert_eq!(granite_speech_audio_tokens_per_second(), 10.0);
    }

    #[test]
    fn derived_max_input_seconds_matches_published_constant() {
        let projector = GraniteSpeechProjectorConfig::granite_speech_4_1_2b();
        let derived = derive_max_input_whole_seconds(
            GRANITE_SPEECH_DECODER_MAX_POSITIONS,
            GRANITE_SPEECH_FIXED_PROMPT_TOKENS,
            GRANITE_SPEECH_MAX_GENERATED_TOKENS,
            &projector,
        )
        .unwrap();
        assert_eq!(
            derived, GRANITE_SPEECH_MAX_INPUT_SECONDS as usize,
            "derived limit drifted off the published constant"
        );
        assert_eq!(derived, 381, "Q-Former tail padding must be included");
    }

    #[test]
    fn derived_budget_leaves_room_for_prompt_and_generation() {
        let projector = GraniteSpeechProjectorConfig::granite_speech_4_1_2b();
        let audio_tokens_at_limit = granite_speech_audio_token_count_for_samples(
            GRANITE_SPEECH_MAX_INPUT_SAMPLES,
            &projector,
        )
        .unwrap();
        let total = GRANITE_SPEECH_FIXED_PROMPT_TOKENS
            + audio_tokens_at_limit
            + GRANITE_SPEECH_MAX_GENERATED_TOKENS;
        assert!(
            total <= GRANITE_SPEECH_DECODER_MAX_POSITIONS,
            "limit must keep prompt+audio+generation inside the 4096 context (got {total})"
        );
        // Next whole second would overflow the floored budget.
        let next_second_samples = (GRANITE_SPEECH_MAX_INPUT_SECONDS as usize + 1)
            .checked_mul(GRANITE_SPEECH_SAMPLE_RATE_HZ as usize)
            .unwrap();
        let over = GRANITE_SPEECH_FIXED_PROMPT_TOKENS
            + granite_speech_audio_token_count_for_samples(next_second_samples, &projector)
                .unwrap()
            + GRANITE_SPEECH_MAX_GENERATED_TOKENS;
        assert!(
            over > GRANITE_SPEECH_DECODER_MAX_POSITIONS,
            "limit + 1s must be the first whole second past the context"
        );
        assert_eq!(audio_tokens_at_limit, 3_810);
        assert_eq!(over, 4_097);
    }

    #[test]
    fn exhausted_context_has_no_legal_whole_second() {
        assert_eq!(
            derive_max_input_whole_seconds(
                256,
                19,
                256,
                &GraniteSpeechProjectorConfig::granite_speech_4_1_2b(),
            ),
            Ok(0)
        );
    }

    #[test]
    fn frontend_constants_match_geometry_inputs() {
        // Drift guard: geometry reads the live frontend / projector constants
        // so a hop or downsample change moves the capacity figure automatically.
        assert_eq!(SAMPLE_RATE_HZ as usize, 16_000);
        assert_eq!(HOP_LENGTH, 160);
        assert_eq!(
            GraniteSpeechProjectorConfig::granite_speech_4_1_2b().downsample_rate,
            5
        );
        assert_eq!(
            GraniteSpeechProjectorConfig::granite_speech_4_1_2b().window_size,
            15
        );
    }

    #[test]
    fn topology_counts_projector_padding_with_integer_shapes() {
        let envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(30_000).unwrap(),
        )
        .unwrap();
        let projector = GraniteSpeechProjectorConfig::granite_speech_4_1_2b();
        assert_eq!(
            granite_speech_audio_token_count_for_samples(480_000, &projector).unwrap(),
            300
        );
        let plan = DecoderStatePlan::for_envelope(
            &GraniteSpeechDecoderStateTopology::new(
                GraniteSpeechDecoderConfig::granite_speech_4_1_2b(),
                projector,
                GraniteSpeechKvResidence::Resident,
                GRANITE_SPEECH_FIXED_PROMPT_TOKENS,
            ),
            envelope,
        )
        .unwrap();
        assert_eq!(
            plan.reserve_positions(StateKind::SelfAttentionKv),
            Some(574)
        );
        assert_eq!(plan.reserve_bytes().host, 0);
        assert!(plan.reserve_bytes().resident > 0);
    }

    #[test]
    fn one_through_three_hundred_seconds_follow_qformer_windows_exactly() {
        let rate = NonZeroU32::new(GRANITE_SPEECH_SAMPLE_RATE_HZ).unwrap();
        let projector = GraniteSpeechProjectorConfig::granite_speech_4_1_2b();
        let topology = GraniteSpeechDecoderStateTopology::new(
            GraniteSpeechDecoderConfig::granite_speech_4_1_2b(),
            projector,
            GraniteSpeechKvResidence::Resident,
            GRANITE_SPEECH_FIXED_PROMPT_TOKENS,
        );
        for seconds in [1, 30, 60, 300, 381] {
            let samples = seconds * GRANITE_SPEECH_SAMPLE_RATE_HZ as usize;
            let audio_positions =
                granite_speech_audio_token_count_for_samples(samples, &projector).unwrap();
            let expected = GRANITE_SPEECH_FIXED_PROMPT_TOKENS
                + audio_positions
                + GRANITE_SPEECH_MAX_GENERATED_TOKENS
                - 1;
            let plan = DecoderStatePlan::for_envelope(
                &topology,
                InvocationEnvelope::new(rate, samples).unwrap(),
            )
            .unwrap();
            assert_eq!(
                plan.reserve_positions_by_id(GRANITE_SPEECH_SELF_KV_STATE_ID),
                Some(expected),
                "unexpected Granite capacity at {seconds}s"
            );
            assert!(expected < GRANITE_SPEECH_DECODER_MAX_POSITIONS);
        }

        let over = InvocationEnvelope::new(
            rate,
            (GRANITE_SPEECH_MAX_INPUT_SECONDS as usize + 1)
                * GRANITE_SPEECH_SAMPLE_RATE_HZ as usize,
        )
        .unwrap();
        assert!(matches!(
            DecoderStatePlan::for_envelope(&topology, over),
            Err(TopologyError::InvocationSampleLimitExceeded { .. })
        ));

        let one_sample_over =
            InvocationEnvelope::new(rate, GRANITE_SPEECH_MAX_INPUT_SAMPLES + 1).unwrap();
        assert!(matches!(
            DecoderStatePlan::for_envelope(&topology, one_sample_over),
            Err(TopologyError::InvocationSampleLimitExceeded {
                required_samples,
                max_samples: GRANITE_SPEECH_MAX_INPUT_SAMPLES,
            }) if required_samples == GRANITE_SPEECH_MAX_INPUT_SAMPLES + 1
        ));
    }

    #[test]
    fn semantic_context_rejects_physical_span_equal_to_cap() {
        let projector = GraniteSpeechProjectorConfig::granite_speech_4_1_2b();
        let audio_positions = granite_speech_audio_token_count_for_samples(
            GRANITE_SPEECH_SAMPLE_RATE_HZ as usize,
            &projector,
        )
        .unwrap();
        let fixed_prompt_tokens = GRANITE_SPEECH_DECODER_MAX_POSITIONS
            .checked_add(1)
            .and_then(|positions| positions.checked_sub(audio_positions))
            .and_then(|positions| positions.checked_sub(GRANITE_SPEECH_MAX_GENERATED_TOKENS))
            .unwrap();
        assert!(matches!(
            DecoderStatePlan::for_envelope(
                &GraniteSpeechDecoderStateTopology::new(
                    GraniteSpeechDecoderConfig::granite_speech_4_1_2b(),
                    projector,
                    GraniteSpeechKvResidence::Resident,
                    fixed_prompt_tokens,
                ),
                InvocationEnvelope::new(
                    NonZeroU32::new(GRANITE_SPEECH_SAMPLE_RATE_HZ).unwrap(),
                    GRANITE_SPEECH_SAMPLE_RATE_HZ as usize,
                )
                .unwrap(),
            ),
            Err(TopologyError::SemanticContextCapExceeded {
                required: 4_097,
                hard_cap: 4_096,
                ..
            })
        ));
    }

    #[test]
    fn topology_accounts_for_the_exact_request_prompt_size() {
        let envelope = InvocationEnvelope::from_milliseconds(
            NonZeroU32::new(16_000).unwrap(),
            NonZeroU32::new(30_000).unwrap(),
        )
        .unwrap();
        let projector = GraniteSpeechProjectorConfig::granite_speech_4_1_2b();
        let baseline = DecoderStatePlan::for_envelope(
            &GraniteSpeechDecoderStateTopology::new(
                GraniteSpeechDecoderConfig::granite_speech_4_1_2b(),
                projector,
                GraniteSpeechKvResidence::Resident,
                GRANITE_SPEECH_FIXED_PROMPT_TOKENS,
            ),
            envelope,
        )
        .unwrap();
        let keyword_prompt = DecoderStatePlan::for_envelope(
            &GraniteSpeechDecoderStateTopology::new(
                GraniteSpeechDecoderConfig::granite_speech_4_1_2b(),
                projector,
                GraniteSpeechKvResidence::Resident,
                GRANITE_SPEECH_FIXED_PROMPT_TOKENS + 7,
            ),
            envelope,
        )
        .unwrap();

        assert_eq!(
            keyword_prompt.reserve_positions(StateKind::SelfAttentionKv),
            baseline
                .reserve_positions(StateKind::SelfAttentionKv)
                .map(|positions| positions + 7)
        );
    }
}
