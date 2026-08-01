//! firered-llm capacity derivation: the family's Qwen2 decoder KV geometry
//! and the decoder positions a single decode of a given audio span needs,
//! assembled from PACK METADATA (`FireRedLlmDecoderMetadata` +
//! `FireRedLlmAdapterMetadata`) plus family constants, so the shared
//! host-memory admission check
//! ([`crate::capacity::evaluate_host_memory_admission`]) can refuse a request
//! that plainly cannot fit BEFORE the ggml graph build turns the shortfall
//! into an opaque allocation failure. The firered-llm analogue of
//! [`crate::models::qwen::capacity`].
//!
//! Reuses, never re-derives:
//! - KV geometry comes straight off the parsed pack (`firered_llm.llm.*` are
//!   required metadata keys).
//! - The audio-token count walks the exact runtime pipeline arithmetic:
//!   snip-edges fbank framing (`firered_aed::frontend`'s constants), the
//!   `2x Conv2d(k3,s2)` stem
//!   ([`crate::models::firered_aed::encoder_graph::predicted_encoder_time_frames`],
//!   the same function the real encoder path calls), and the adapter's
//!   pack-carried `downsample_rate` frame stacking.
//! - The generation budget and the single-decode window reuse the executor's
//!   own constants ([`super::executor::FIRERED_LLM_MAX_GENERATED_TOKENS`],
//!   [`super::executor::FIRERED_LLM_MAX_INPUT_SECONDS`]), so admission
//!   charges the same `prompt + generation` KV capacity
//!   `FireRedLlmDecoderRuntime::new_kv_caches` actually allocates.

use crate::capacity::KvGeometry;
use crate::models::firered_aed::encoder_graph::predicted_encoder_time_frames;
use crate::models::firered_aed::frontend::{
    FRAME_LENGTH_SAMPLES, FRAME_SHIFT_SAMPLES, SAMPLE_RATE_HZ,
};

use super::executor::{FIRERED_LLM_MAX_GENERATED_TOKENS, FIRERED_LLM_MAX_INPUT_SECONDS};
use super::runtime_contract::{FireRedLlmAdapterMetadata, FireRedLlmDecoderMetadata};

/// The audio a single firered-llm decode is estimated to fold in, for
/// admission. The executor fails closed above this upstream hard cap
/// (`FIRERED_LLM_MAX_INPUT_SECONDS`, "single 40s max input"), and longer
/// recordings reach it only as longform slices each at or under this bound --
/// so no single decode's KV cache ever spans more audio. Clamping here is the
/// firered-llm analogue of qwen3's
/// `QWEN3_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS` clamp: a multi-hour
/// recording is judged by what one decode actually needs, never falsely
/// rejected as if it decoded whole.
pub(crate) const FIRERED_LLM_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS: f32 =
    FIRERED_LLM_MAX_INPUT_SECONDS;

/// Fixed ChatML wrapper tokens around the speech span
/// (`<|im_start|>user\n` prefix + instruction + `<|im_end|>\n<|im_start|>
/// assistant\n` suffix, see `super::decode_prompt::build_firered_llm_decode_prompt`).
/// A small conservative figure, dwarfed by the speech-token term at any real
/// duration; its exact value cannot swing an admission decision.
const FIRERED_LLM_ADMISSION_FIXED_PROMPT_TOKENS: usize = 32;

/// The decoder KV geometry the loaded pack advertises.
pub(crate) fn firered_llm_kv_geometry(decoder: &FireRedLlmDecoderMetadata) -> KvGeometry {
    KvGeometry {
        n_layers: decoder.n_layers,
        kv_heads: decoder.n_kv_heads,
        head_dim: decoder.head_dim,
    }
}

/// Decoder positions a single decode of `audio_duration_seconds` requires --
/// the admission figure `evaluate_host_memory_admission` charges KV bytes
/// against. Mirrors the runtime's own KV-capacity sizing
/// (`decode_prompt.token_ids.len() + FIRERED_LLM_MAX_GENERATED_TOKENS` in
/// `super::executor`): fixed wrapper + one speech token per adapter output
/// frame + the full runaway-generation backstop, with the audio clamped to
/// the family's single-decode window.
pub(crate) fn firered_llm_admission_required_positions(
    adapter: &FireRedLlmAdapterMetadata,
    audio_duration_seconds: f32,
) -> usize {
    let admission_seconds =
        audio_duration_seconds.clamp(0.0, FIRERED_LLM_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS);
    let samples = (admission_seconds * SAMPLE_RATE_HZ as f32) as usize;
    // snip_edges fbank framing: 1 + (len - frame_length) / frame_shift.
    let mel_frames = samples
        .checked_sub(FRAME_LENGTH_SAMPLES)
        .map(|tail| 1 + tail / FRAME_SHIFT_SAMPLES)
        .unwrap_or(0);
    // The 2x Conv2d(k3,s2) stem, via the same prediction the real encoder
    // path uses. A degenerate frame count (too-short audio) resolves to zero
    // speech tokens -- an under-estimate resolves to "allow" per
    // `crate::capacity`'s fail-open invariant.
    let encoder_frames = predicted_encoder_time_frames(mel_frames).unwrap_or(0);
    // Adapter frame stacking: `downsample_rate` adjacent frames concatenate
    // into one LLM speech token (trailing remainder dropped, exactly the
    // reshape `super::adapter_graph` performs).
    let speech_tokens = encoder_frames / adapter.downsample_rate.max(1);
    FIRERED_LLM_ADMISSION_FIXED_PROMPT_TOKENS
        .saturating_add(speech_tokens)
        .saturating_add(FIRERED_LLM_MAX_GENERATED_TOKENS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacity::kv_bytes_per_position;
    use crate::nn::decoder::LlmKvCacheSpec;

    /// Real-checkpoint-shaped metadata (the Qwen2-7B decoder: 28 layers, GQA
    /// with 4 KV heads, head_dim 128, 32768-position context -- the same
    /// values `runtime_contract`'s `full_metadata` fixture parses).
    fn reference_decoder() -> FireRedLlmDecoderMetadata {
        FireRedLlmDecoderMetadata {
            n_layers: 28,
            d_model: 3584,
            n_heads: 28,
            n_kv_heads: 4,
            head_dim: 128,
            ffn_dim: 18944,
            vocab_size: 152_064,
            max_positions: 32_768,
            chatml_im_start_token_id: 151_644,
            chatml_im_end_token_id: 151_645,
            endoftext_token_id: 151_643,
            speech_token_id: 151_646,
        }
    }

    fn reference_adapter() -> FireRedLlmAdapterMetadata {
        FireRedLlmAdapterMetadata {
            downsample_rate: 2,
            llm_dim: 3584,
        }
    }

    #[test]
    fn kv_geometry_reads_the_llm_decoder_metadata() {
        let geometry = firered_llm_kv_geometry(&reference_decoder());
        assert_eq!(
            geometry,
            KvGeometry {
                n_layers: 28,
                kv_heads: 4,
                head_dim: 128,
            }
        );
        // Feeds the shared KV byte model without error (224 rows/position,
        // the same shape the runtime_contract capacity anchor pins).
        let default = kv_bytes_per_position(&geometry, LlmKvCacheSpec::DEFAULT).expect("default");
        assert_eq!(default.total(), 224 * 768);
    }

    #[test]
    fn required_positions_is_clamped_to_the_single_decode_window() {
        let adapter = reference_adapter();
        // 40s and 3600s clamp to the same single-decode window: the executor
        // hard-caps a single decode at 40s and longform slices the rest, so a
        // multi-hour recording is never judged as one giant decode.
        let at_window = firered_llm_admission_required_positions(
            &adapter,
            FIRERED_LLM_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS,
        );
        let at_hour = firered_llm_admission_required_positions(&adapter, 3600.0);
        assert_eq!(at_window, at_hour);
        // A short clip needs strictly fewer positions than a full window.
        let short = firered_llm_admission_required_positions(&adapter, 5.0);
        assert!(short < at_window, "short={short} window={at_window}");
        // Sanity: well under the pack's 32768 context ceiling at any window.
        assert!(at_window < reference_decoder().max_positions);
    }

    #[test]
    fn required_positions_walks_the_runtime_pipeline_arithmetic() {
        // 40s at 16kHz snip-edges fbank: 1 + (640000-400)/160 = 3998 mel
        // frames -> conv stem ((3998+6-1)/2 -> 2001, (2001-1)/2 -> 1000)
        // -> /2 adapter stacking = 500 speech tokens.
        // required = 32 fixed + 500 speech + 512 generation backstop = 1044.
        let positions = firered_llm_admission_required_positions(&reference_adapter(), 40.0);
        assert_eq!(positions, 1044);
    }
}
