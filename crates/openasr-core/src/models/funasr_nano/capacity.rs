//! funasr-nano capacity derivation: the family's Qwen3 decoder KV geometry
//! and the decoder positions a single decode of a given audio span needs,
//! assembled from PACK METADATA (`FunasrNanoDecoderMetadata`) plus the same
//! frontend / audio-token / generation constants the executor uses, so the
//! shared host-memory admission check
//! ([`crate::capacity::evaluate_host_memory_admission`]) can refuse a request
//! that plainly cannot fit BEFORE the ggml graph build turns the shortfall
//! into an opaque allocation failure. The funasr-nano analogue of
//! [`crate::models::firered_llm::capacity`] / [`crate::models::qwen::capacity`].
//!
//! Reuses, never re-derives:
//! - KV geometry comes straight off the parsed pack (`funasr.llm.*` are
//!   required metadata keys).
//! - The audio-token count walks the exact runtime pipeline arithmetic:
//!   snip-edges kaldi fbank framing (SenseVoice `WavFrontend` constants the
//!   executor reuses -- 16 kHz / 25 ms / 10 ms), FunASR LFR stacking
//!   (`LFR_N = 6`), and the official low-frame-rate truncation
//!   ([`super::decode_prompt::funasr_nano_audio_token_count`], the same
//!   function the real decode path calls).
//! - The generation budget and the single-decode window reuse the executor's
//!   own constants ([`super::executor::FUNASR_NANO_MAX_GENERATED_TOKENS`],
//!   [`super::executor::FUNASR_NANO_MAX_INPUT_SECONDS`]), so admission
//!   charges the same `prompt + generation` KV capacity
//!   `FunasrNanoDecoderRuntime::new_kv_caches` actually allocates.

use crate::capacity::KvGeometry;
use crate::models::sensevoice::frontend::{LFR_N, SAMPLE_RATE_HZ};

use super::decode_prompt::funasr_nano_audio_token_count;
use super::executor::{FUNASR_NANO_MAX_GENERATED_TOKENS, FUNASR_NANO_MAX_INPUT_SECONDS};
use super::runtime_contract::FunasrNanoDecoderMetadata;

/// Kaldi fbank frame length / shift the SenseVoice `WavFrontend` pins
/// (25 ms / 10 ms @ 16 kHz) -- the same constants
/// `crate::models::sensevoice::frontend`'s frontend config carries. Kept local
/// (rather than reaching into the private frontend config) so the admission
/// arithmetic stays a pure function of publicly named family facts.
const FBANK_FRAME_LENGTH_SAMPLES: usize = 400;
const FBANK_FRAME_SHIFT_SAMPLES: usize = 160;

/// The audio a single funasr-nano decode is estimated to fold in, for
/// admission. The executor fails closed above this upstream hard cap
/// (`FUNASR_NANO_MAX_INPUT_SECONDS`, ~40s OOD-repeat bound), and longer
/// recordings reach it only as longform slices each at or under this bound --
/// so no single decode's KV cache ever spans more audio. Clamping here is the
/// funasr-nano analogue of firered-llm's
/// `FIRERED_LLM_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS` clamp: a multi-hour
/// recording is judged by what one decode actually needs, never falsely
/// rejected as if it decoded whole.
pub(crate) const FUNASR_NANO_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS: f32 =
    FUNASR_NANO_MAX_INPUT_SECONDS;

/// Fixed ChatML wrapper tokens around the audio span
/// (`<|im_start|>system ... <|im_start|>user\n语音转写：` prefix +
/// `<|im_end|>\n<|im_start|>assistant\n` suffix, see
/// `super::decode_prompt::build_funasr_nano_decode_prompt`). A small
/// conservative figure, dwarfed by the audio-token term at any real duration;
/// its exact value cannot swing an admission decision.
const FUNASR_NANO_ADMISSION_FIXED_PROMPT_TOKENS: usize = 32;

/// The decoder KV geometry the loaded pack advertises.
pub(crate) fn funasr_nano_kv_geometry(decoder: &FunasrNanoDecoderMetadata) -> KvGeometry {
    KvGeometry {
        n_layers: decoder.n_layers,
        kv_heads: decoder.n_kv_heads,
        head_dim: decoder.head_dim,
    }
}

/// Decoder positions a single decode of `audio_duration_seconds` requires --
/// the admission figure `evaluate_host_memory_admission` charges KV bytes
/// against. Mirrors the runtime's own KV-capacity sizing
/// (`decode_prompt.token_ids.len() + FUNASR_NANO_MAX_GENERATED_TOKENS` in
/// `super::executor`): fixed wrapper + one audio token per kept adaptor
/// output frame + the full runaway-generation backstop, with the audio
/// clamped to the family's single-decode window.
pub(crate) fn funasr_nano_admission_required_positions(audio_duration_seconds: f32) -> usize {
    let admission_seconds =
        audio_duration_seconds.clamp(0.0, FUNASR_NANO_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS);
    let samples = (admission_seconds * SAMPLE_RATE_HZ as f32) as usize;
    // snip_edges fbank framing: 1 + (len - frame_length) / frame_shift.
    let mel_frames = samples
        .checked_sub(FBANK_FRAME_LENGTH_SAMPLES)
        .map(|tail| 1 + tail / FBANK_FRAME_SHIFT_SAMPLES)
        .unwrap_or(0);
    // FunASR LFR stacking: ceil(mel_frames / LFR_N), matching
    // `sensevoice::frontend::apply_lfr`'s output-frame count (computed from
    // the original, pre-padding frame count).
    let lfr_frames = if mel_frames == 0 {
        0
    } else {
        mel_frames.div_ceil(LFR_N)
    };
    // Official low-frame-rate truncation -- the same function the executor
    // calls after the adaptor. A degenerate frame count resolves to a tiny
    // positive audio-token count (the formula bottoms at 1); an under-estimate
    // resolves to "allow" per `crate::capacity`'s fail-open invariant.
    let audio_tokens = funasr_nano_audio_token_count(lfr_frames);
    FUNASR_NANO_ADMISSION_FIXED_PROMPT_TOKENS
        .saturating_add(audio_tokens)
        .saturating_add(FUNASR_NANO_MAX_GENERATED_TOKENS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacity::kv_bytes_per_position;
    use crate::nn::decoder::LlmKvCacheSpec;

    /// Real-checkpoint-shaped metadata (Qwen3-0.6B decoder: 28 layers, GQA
    /// with 8 KV heads, head_dim 128, 40960-position context -- the same
    /// values `runtime_contract`'s `full_metadata` fixture parses).
    fn reference_decoder() -> FunasrNanoDecoderMetadata {
        FunasrNanoDecoderMetadata {
            n_layers: 28,
            d_model: 1024,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            ffn_dim: 3072,
            vocab_size: 151_936,
            max_positions: 40_960,
            chatml_im_start_token_id: 151_644,
            chatml_im_end_token_id: 151_645,
            endoftext_token_id: 151_643,
        }
    }

    #[test]
    fn kv_geometry_reads_the_llm_decoder_metadata() {
        let geometry = funasr_nano_kv_geometry(&reference_decoder());
        assert_eq!(
            geometry,
            KvGeometry {
                n_layers: 28,
                kv_heads: 8,
                head_dim: 128,
            }
        );
        // Feeds the shared KV byte model without error (448 rows/position,
        // the same shape the qwen3 / moss capacity anchors pin).
        let default = kv_bytes_per_position(&geometry, LlmKvCacheSpec::DEFAULT).expect("default");
        assert_eq!(default.total(), 448 * 768);
    }

    #[test]
    fn required_positions_is_clamped_to_the_single_decode_window() {
        // 40s and 3600s clamp to the same single-decode window: the executor
        // hard-caps a single decode at 40s and longform slices the rest, so a
        // multi-hour recording is never judged as one giant decode.
        let at_window = funasr_nano_admission_required_positions(
            FUNASR_NANO_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS,
        );
        let at_hour = funasr_nano_admission_required_positions(3600.0);
        assert_eq!(at_window, at_hour);
        // A short clip needs strictly fewer positions than a full window.
        let short = funasr_nano_admission_required_positions(5.0);
        assert!(short < at_window, "short={short} window={at_window}");
        // Sanity: well under the pack's 40960 context ceiling at any window.
        assert!(at_window < reference_decoder().max_positions);
    }

    #[test]
    fn required_positions_walks_the_runtime_pipeline_arithmetic() {
        // 40s at 16kHz snip-edges fbank: 1 + (640000-400)/160 = 3998 mel
        // frames -> LFR ceil(3998/6) = 667 lfr frames ->
        // funasr_nano_audio_token_count(667):
        //   conv(t) = 1 + (t - 3 + 2) / 2
        //   conv(667) = 1 + 666/2 = 334
        //   conv(334) = 1 + 333/2 = 167
        //   n_aud = (167 - 1) / 2 + 1 = 84
        // required = 32 fixed + 84 audio + 512 generation backstop = 628.
        let positions = funasr_nano_admission_required_positions(40.0);
        assert_eq!(positions, 628);
        assert_eq!(funasr_nano_audio_token_count(667), 84);
    }
}
