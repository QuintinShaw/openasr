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
//! max_input_seconds  = floor(audio_token_budget / audio_tokens_per_second)
//! ```
//!
//! where `audio_tokens_per_second` is the frontend geometry product
//! `sample_rate / (hop_length * frame_stack * projector_downsample)`
//! (16_000 / (160 * 2 * 5) = 10 tok/s -- one projector token per 100ms, the
//! same rate `encoder_graph.rs`'s long-audio note documents).
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
//! # 2. Host-memory admission (decoder KV)
//!
//! [`granite_speech_kv_geometry`] / [`granite_speech_admission_required_positions`]
//! / [`GRANITE_SPEECH_ADMISSION_KV_SPEC`] feed the shared host-memory admission
//! check ([`crate::capacity::evaluate_host_memory_admission`]) so a pack whose
//! weights + decode KV plainly cannot fit is refused BEFORE the ggml graph
//! build turns the shortfall into an opaque allocation failure.
//!
//! Reuses, never re-derives:
//! - KV geometry comes straight off the parsed pack
//!   (`granite_speech.decoder.{num_hidden_layers,num_key_value_heads,head_dim}`).
//! - Audio-token rate is [`granite_speech_audio_tokens_per_second`] (the same
//!   frontend geometry product the max-input-seconds derivation uses).
//! - Fixed prompt overhead is [`GRANITE_SPEECH_FIXED_PROMPT_TOKENS`]; generation
//!   headroom is [`super::executor::GRANITE_SPEECH_MAX_GENERATED_TOKENS`] (the
//!   same figure `decode_session`'s resident-KV arena headroom reserves).
//! - The single-decode admission window is the shared longform safety chunk
//!   ([`crate::arch::DEFAULT_ENCODER_SAFE_CHUNK_SECONDS`]): granite is
//!   `SharedWindow`, so no single decode's KV cache ever spans more audio than
//!   one slice. Clamping here keeps a multi-hour recording from being judged
//!   (and falsely rejected) as if it decoded whole. A user who forces a larger
//!   `longform.chunk_seconds` only makes admission MORE permissive (an under-
//!   estimate resolves to "allow", per `crate::capacity`'s fail-open
//!   invariant), never falsely rejecting.
//!
//! KV element-type truth source: the runtime keeps a SINGLE f32 copy of the
//! decode KV -- host growing history on CPU (`decode_session`'s `k_history` /
//! `v_history`), device-resident f32 arena on Metal
//! (`GRANITE_RESIDENT_KV_SPEC` in `decode_session.rs`). The shared position
//! model charges host + resident halves, so
//! [`GRANITE_SPEECH_ADMISSION_KV_SPEC`] models that single f32 copy as two f16
//! halves whose `.total()` equals one f32 (the same cohere / firered-aed
//! modeling stand-in). Hard-coding `LlmKvCacheSpec::DEFAULT` (f32 host + f16
//! resident) would overstate by 1.5x in the false-reject direction admission
//! forbids.
//!
//! Production reads the derived max-input-seconds constant below. The pure
//! derivation helpers stay unit-tested so a future pack-carried
//! `max_position_embeddings` key can replace the architecture constant
//! without silent drift.
#![cfg_attr(not(test), allow(dead_code))]

use crate::capacity::{FrontendGeometry, KvGeometry, audio_tokens_per_second};
use crate::ggml_runtime::GgmlKvElementType;
use crate::nn::decoder::LlmKvCacheSpec;

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

/// Front-end frame-stack factor: after the 80-mel STFT the extractor drops an
/// odd trailing frame and concatenates pairs of 80-dim frames into 160-dim
/// (`frontend.rs`), halving the frame rate before the Conformer encoder.
pub(crate) const GRANITE_SPEECH_ENCODER_FRAME_STACK: usize = 2;

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
pub(crate) const GRANITE_SPEECH_FIXED_PROMPT_TOKENS: usize = 19;

/// Shipped 4.1-2b frontend geometry: 16 kHz / 160-hop mel, 2x frame-stack,
/// Q-Former `downsample_rate=5` (window_size=15 / num_queries=3). Yields
/// 10 audio tokens per second of input.
pub(crate) fn granite_speech_frontend_geometry() -> FrontendGeometry {
    FrontendGeometry {
        sample_rate_hz: SAMPLE_RATE_HZ as usize,
        hop_length: HOP_LENGTH,
        encoder_conv_stride: GRANITE_SPEECH_ENCODER_FRAME_STACK,
        // Q-Former emits `window_size / downsample_rate` queries per window, so
        // the frames-per-audio-token factor equals `downsample_rate` (the
        // shipped default; pack metadata carries the same value under
        // `granite_speech.downsample_rate`).
        adaptor_merge_size: GraniteSpeechProjectorConfig::granite_speech_4_1_2b().downsample_rate,
    }
}

/// Audio tokens per second of input for the shipped geometry (10.0).
pub(crate) fn granite_speech_audio_tokens_per_second() -> f32 {
    audio_tokens_per_second(&granite_speech_frontend_geometry())
}

/// Pure derivation: largest whole-second audio span whose projected tokens
/// still fit `decoder_max_positions` after reserving the fixed prompt and the
/// generation backstop. Returns `0.0` on a non-positive rate (fail-closed)
/// rather than inventing a limit.
pub(crate) fn derive_max_input_seconds(
    decoder_max_positions: usize,
    fixed_prompt_tokens: usize,
    max_generated_tokens: usize,
    audio_tokens_per_second: f32,
) -> f32 {
    if !audio_tokens_per_second.is_finite() || audio_tokens_per_second <= 0.0 {
        return 0.0;
    }
    let budget = decoder_max_positions
        .saturating_sub(fixed_prompt_tokens)
        .saturating_sub(max_generated_tokens);
    // Floor so a request at the published limit cannot land past the token
    // budget even with Q-Former's partial-window pad (at most one window of
    // queries -- 3 tokens / 0.3s at the shipped geometry -- of slack below
    // the unfloored quotient).
    (budget as f32 / audio_tokens_per_second).floor()
}

/// Derived single-decode max input seconds for the shipped 4.1-2b geometry:
/// `(4096 - 19 - 256) / 10 = 382.1 -> 382.0`. Production and the executor
/// fail-closed check read this constant; the unit test below pins it equal
/// to [`derive_max_input_seconds`] so the arithmetic cannot drift silently.
pub(crate) const GRANITE_SPEECH_MAX_INPUT_SECONDS: f32 = 382.0;

/// The audio a single granite-speech decode is estimated to fold in, for
/// host-memory admission. granite-speech is
/// [`crate::arch::OpenAsrLongformSliceShape::SharedWindow`]: a recording
/// longer than the shared longform safety chunk is sliced, and each slice is
/// its own decode, so no single decode's KV cache ever spans more than one
/// window's audio. Clamping the admission estimate here (the analogue of
/// qwen3's `QWEN3_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS` clamp) is what
/// keeps a multi-hour recording from being judged -- and falsely rejected --
/// as if it decoded whole. The executor's own 382s decoder-context ceiling
/// is a separate fail-closed bound on a direct over-limit buffer; ordinary
/// longform work never approaches it.
pub(crate) const GRANITE_SPEECH_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS: f32 =
    crate::arch::DEFAULT_ENCODER_SAFE_CHUNK_SECONDS;

/// Two f16 copies = 4 B/value total: byte-for-byte the single f32 copy the
/// runtime actually keeps (host growing history on CPU; device-resident f32
/// arena on Metal -- see the module doc). Only
/// [`crate::capacity::KvBytesPerPosition::total`] is consumed by admission,
/// so the host/resident split is a modeling stand-in, not a claim that both
/// copies exist at once.
pub(crate) const GRANITE_SPEECH_ADMISSION_KV_SPEC: LlmKvCacheSpec = LlmKvCacheSpec {
    host: GgmlKvElementType::F16,
    resident: GgmlKvElementType::F16,
};

/// The decoder KV geometry the loaded pack advertises.
pub(crate) fn granite_speech_kv_geometry(decoder: &GraniteSpeechDecoderConfig) -> KvGeometry {
    KvGeometry {
        n_layers: decoder.num_layers,
        kv_heads: decoder.num_kv_heads,
        head_dim: decoder.head_dim,
    }
}

/// Decoder positions a single decode of `audio_duration_seconds` requires --
/// the admission figure `evaluate_host_memory_admission` charges KV bytes
/// against. Mirrors the runtime's own KV-capacity sizing
/// (`prompt_tokens + GRANITE_SPEECH_MAX_GENERATED_TOKENS` /
/// `GRANITE_RESIDENT_DECODE_HEADROOM` in `decode_session`): fixed prompt
/// overhead + projected audio tokens at the frontend rate + the full
/// generation backstop, with the audio clamped to the SharedWindow single-
/// decode window.
pub(crate) fn granite_speech_admission_required_positions(audio_duration_seconds: f32) -> usize {
    let admission_seconds =
        audio_duration_seconds.clamp(0.0, GRANITE_SPEECH_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS);
    let rate = granite_speech_audio_tokens_per_second();
    // Ceil so a fractional token still reserves a full position (matches the
    // Q-Former's partial-window pad behaviour at the frame boundary). A non-
    // positive rate fails closed to zero audio tokens -- an under-estimate
    // resolves to "allow" per `crate::capacity`'s fail-open invariant.
    let audio_tokens = if rate.is_finite() && rate > 0.0 {
        (admission_seconds * rate).ceil() as usize
    } else {
        0
    };
    GRANITE_SPEECH_FIXED_PROMPT_TOKENS
        .saturating_add(audio_tokens)
        .saturating_add(GRANITE_SPEECH_MAX_GENERATED_TOKENS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacity::{
        AudioFrontendCapacityBasis, KvGeometry, audio_tokens_per_second, frontend_capacity_basis,
        kv_bytes_per_position,
    };

    #[test]
    fn shipped_geometry_is_ten_audio_tokens_per_second() {
        let geometry = granite_speech_frontend_geometry();
        assert_eq!(geometry.sample_rate_hz, 16_000);
        assert_eq!(geometry.hop_length, 160);
        assert_eq!(geometry.encoder_conv_stride, 2);
        assert_eq!(geometry.adaptor_merge_size, 5);
        assert_eq!(audio_tokens_per_second(&geometry), 10.0);
        assert_eq!(granite_speech_audio_tokens_per_second(), 10.0);
    }

    #[test]
    fn derived_max_input_seconds_matches_published_constant() {
        let derived = derive_max_input_seconds(
            GRANITE_SPEECH_DECODER_MAX_POSITIONS,
            GRANITE_SPEECH_FIXED_PROMPT_TOKENS,
            GRANITE_SPEECH_MAX_GENERATED_TOKENS,
            granite_speech_audio_tokens_per_second(),
        );
        assert_eq!(
            derived, GRANITE_SPEECH_MAX_INPUT_SECONDS,
            "derived limit drifted off the published constant"
        );
        // Explicit arithmetic pin: (4096 - 19 - 256) / 10 = 382.1 -> floor 382.
        assert_eq!(
            derive_max_input_seconds(4096, 19, 256, 10.0),
            382.0,
            "fixed-input derivation must stay bit-stable"
        );
    }

    #[test]
    fn derived_budget_leaves_room_for_prompt_and_generation() {
        let tokens_per_sec = granite_speech_audio_tokens_per_second();
        let audio_tokens_at_limit =
            (GRANITE_SPEECH_MAX_INPUT_SECONDS * tokens_per_sec).round() as usize;
        let total = GRANITE_SPEECH_FIXED_PROMPT_TOKENS
            + audio_tokens_at_limit
            + GRANITE_SPEECH_MAX_GENERATED_TOKENS;
        assert!(
            total <= GRANITE_SPEECH_DECODER_MAX_POSITIONS,
            "limit must keep prompt+audio+generation inside the 4096 context (got {total})"
        );
        // Next whole second would overflow the floored budget.
        let over = GRANITE_SPEECH_FIXED_PROMPT_TOKENS
            + ((GRANITE_SPEECH_MAX_INPUT_SECONDS + 1.0) * tokens_per_sec).round() as usize
            + GRANITE_SPEECH_MAX_GENERATED_TOKENS;
        assert!(
            over > GRANITE_SPEECH_DECODER_MAX_POSITIONS,
            "limit + 1s must be the first whole second past the context"
        );
    }

    #[test]
    fn non_positive_rate_fails_closed_to_zero() {
        assert_eq!(derive_max_input_seconds(4096, 19, 256, 0.0), 0.0);
        assert_eq!(derive_max_input_seconds(4096, 19, 256, -1.0), 0.0);
        assert_eq!(derive_max_input_seconds(4096, 19, 256, f32::NAN), 0.0);
    }

    #[test]
    fn capacity_frontend_registry_matches_family_geometry() {
        let basis = frontend_capacity_basis(crate::arch::GRANITE_SPEECH_AUDIO_FRONTEND_ID)
            .expect("granite-speech frontend must be registered for Derived capacity");
        let AudioFrontendCapacityBasis::Constant(geometry) = *basis else {
            panic!("granite-speech frontend capacity basis must be Constant geometry");
        };
        assert_eq!(geometry, granite_speech_frontend_geometry());
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
    fn kv_geometry_reads_the_llm_decoder_metadata() {
        let geometry =
            granite_speech_kv_geometry(&GraniteSpeechDecoderConfig::granite_speech_4_1_2b());
        assert_eq!(
            geometry,
            KvGeometry {
                n_layers: 40,
                kv_heads: 4,
                head_dim: 128,
            }
        );
        // Two f16 halves = one f32 copy: 40 layers * 2 (K+V) * 4 kv-heads *
        // 128 * 4 B = 163_840 B/position.
        let per_pos =
            kv_bytes_per_position(&geometry, GRANITE_SPEECH_ADMISSION_KV_SPEC).expect("spec");
        assert_eq!(per_pos.total(), 40 * 2 * 4 * 128 * 4);
    }

    #[test]
    fn required_positions_is_clamped_to_the_shared_window() {
        // 30s and 3600s clamp to the same SharedWindow single-decode window.
        let at_window = granite_speech_admission_required_positions(
            GRANITE_SPEECH_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS,
        );
        let at_hour = granite_speech_admission_required_positions(3600.0);
        assert_eq!(at_window, at_hour);
        let short = granite_speech_admission_required_positions(5.0);
        assert!(short < at_window, "short={short} window={at_window}");
        // Explicit arithmetic pin at the SharedWindow: 19 fixed + ceil(30*10)
        // audio + 256 generation = 19 + 300 + 256 = 575.
        assert_eq!(at_window, 575);
        assert_eq!(granite_speech_admission_required_positions(30.0), 575);
    }
}
