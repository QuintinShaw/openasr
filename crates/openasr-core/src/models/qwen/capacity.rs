//! qwen3-asr capacity derivation: the family's decoder KV geometry and the
//! decoder positions a single decode of a given audio span needs, assembled
//! from PACK METADATA (`Qwen3AsrExecutionMetadata`) plus family constants, so
//! the shared host-memory admission check
//! ([`crate::capacity::evaluate_host_memory_admission`]) can refuse a request
//! that plainly cannot fit BEFORE the ggml graph build turns the shortfall
//! into an opaque `ggml cpu graph backend buffer allocation failed` (issue
//! #159's root cause on CPU). It is the qwen3 analogue of
//! [`crate::models::moss_transcribe_diarize::capacity`].
//!
//! Reuses, never re-derives:
//! - KV geometry comes straight off the parsed pack (`llm_layers` /
//!   `llm_kv_heads` / `llm_head_dim` are required metadata keys).
//! - The prompt's audio-token count is
//!   [`super::audio_encoder::qwen3_audio_token_count_for_mel_frames`], the
//!   exact row count the decode path splices into the prompt.
//! - The generation budget reuses the same `QWEN3_DECODE_*` constants and the
//!   `llm_max_positions` context clamp the runtime's
//!   `qwen3_generated_token_budget` / `required_max_positions_for_job` apply,
//!   so admission argues from the same position ceiling the real decode does.

use crate::capacity::KvGeometry;
use crate::models::decode_token_history::context_window_budget;

use super::audio_encoder::qwen3_audio_token_count_for_mel_frames;
use super::ggml_executor::{
    QWEN3_DECODE_MIN_GENERATED_TOKENS, QWEN3_DECODE_TOKEN_BUDGET_MARGIN,
    QWEN3_DECODE_TOKENS_PER_AUDIO_SECOND,
};
use super::runtime_contract::Qwen3AsrExecutionMetadata;

/// The audio a single qwen3 decode is estimated to fold in, for admission.
/// qwen3-asr is [`crate::arch::OpenAsrLongformSliceShape::SharedWindow`]: a
/// recording longer than this is sliced by the shared slicer and each slice is
/// its own decode, so no single decode's KV cache ever spans more than one
/// window's audio. Clamping the admission estimate here (the analogue of
/// moss's `integral_seconds` clamp) is what keeps a multi-hour recording from
/// being judged -- and falsely rejected -- as if it decoded whole. The shared
/// slicer's generic safety chunk is the honest, backend-independent single-
/// slice bound; a user who forces a larger `longform.chunk_seconds` only makes
/// admission MORE permissive (an under-estimate resolves to "allow", per
/// `crate::capacity`'s fail-open invariant), never falsely rejecting.
pub(crate) const QWEN3_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS: f32 =
    crate::arch::DEFAULT_ENCODER_SAFE_CHUNK_SECONDS;

/// Fixed ChatML wrapper tokens around the audio span (`<|im_start|>system ...
/// <|audio_start|>` prefix + `<|audio_end|> ... <|im_start|>assistant` suffix,
/// see `super::decode_prompt::build_qwen3_decode_prompt`). A small conservative
/// figure: it is dwarfed by the audio-token term at any real duration, and the
/// whole required-positions estimate is bounded by `llm_max_positions`, so its
/// exact value cannot swing an admission decision.
const QWEN3_ADMISSION_FIXED_PROMPT_TOKENS: usize = 32;

/// The decoder KV geometry the loaded pack advertises.
pub(crate) fn qwen3_kv_geometry(metadata: &Qwen3AsrExecutionMetadata) -> KvGeometry {
    KvGeometry {
        n_layers: metadata.llm_layers,
        kv_heads: metadata.llm_kv_heads,
        head_dim: metadata.llm_head_dim,
    }
}

/// Desired generation budget (before the context clamp) for `audio_seconds` of
/// audio: the family's per-second token allowance plus a fixed margin, floored
/// at the short-audio minimum. The seconds-based twin of the sample-count
/// arithmetic in `super::ggml_executor::qwen3_generated_token_budget`; both
/// read the same `QWEN3_DECODE_*` constants and
/// `qwen3_admission_desired_generation_matches_runtime_budget` pins them equal
/// on whole-second inputs so the two never drift.
fn qwen3_desired_generated_tokens_for_seconds(audio_seconds: f32) -> usize {
    let audio_rate_budget =
        (audio_seconds.max(0.0) * QWEN3_DECODE_TOKENS_PER_AUDIO_SECOND as f32).ceil() as usize;
    audio_rate_budget
        .saturating_add(QWEN3_DECODE_TOKEN_BUDGET_MARGIN)
        .max(QWEN3_DECODE_MIN_GENERATED_TOKENS)
}

/// Decoder positions a single decode of `audio_duration_seconds` requires, the
/// admission figure `evaluate_host_memory_admission` charges KV bytes against.
/// The audio is clamped to [`QWEN3_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS`]
/// (one slice's worth), the prompt is the fixed wrapper plus the exact spliced
/// audio-token count, the generation budget is the family's measured demand
/// clamped to the remaining context, and the whole figure is capped at
/// `llm_max_positions` -- the same ceiling the runtime allocates against.
pub(crate) fn qwen3_admission_required_positions(
    metadata: &Qwen3AsrExecutionMetadata,
    audio_duration_seconds: f32,
) -> usize {
    let admission_seconds =
        audio_duration_seconds.clamp(0.0, QWEN3_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS);
    let sample_rate = metadata.sample_rate_hz.max(1) as f32;
    let hop = metadata.hop_length.max(1) as f32;
    let mel_frames = (admission_seconds * sample_rate / hop) as usize;
    let audio_tokens = qwen3_audio_token_count_for_mel_frames(mel_frames);
    let prompt_tokens = QWEN3_ADMISSION_FIXED_PROMPT_TOKENS.saturating_add(audio_tokens);
    let desired_generation = qwen3_desired_generated_tokens_for_seconds(admission_seconds);
    // Same context clamp the runtime applies: generation may only use what the
    // decoder positions past the prompt leave; a prompt that already fills the
    // context leaves nothing to generate (the runtime fails that request
    // closed separately -- admission just stops charging past the ceiling).
    let generation = context_window_budget(metadata.llm_max_positions, prompt_tokens)
        .map(|remaining| desired_generation.min(remaining))
        .unwrap_or(0);
    prompt_tokens
        .saturating_add(generation)
        .min(metadata.llm_max_positions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacity::{KvGeometry, kv_bytes_per_position};
    use crate::nn::decoder::LlmKvCacheSpec;

    /// Real-checkpoint-shaped metadata (the 1.7B config: 28-layer GQA decoder,
    /// 8 KV heads, head_dim 128, 40960-position context, 16kHz/160-hop mel).
    fn reference_metadata() -> Qwen3AsrExecutionMetadata {
        Qwen3AsrExecutionMetadata {
            sample_rate_hz: 16_000,
            n_mels: 128,
            n_fft: 400,
            win_length: 400,
            hop_length: 160,
            audio_layers: 18,
            audio_d_model: 896,
            audio_heads: 14,
            llm_layers: 28,
            llm_d_model: 1024,
            llm_heads: 16,
            llm_kv_heads: 8,
            llm_head_dim: 128,
            vocab_size: 151_936,
            llm_max_positions: 40_960,
            audio_start_token_id: 11,
            audio_end_token_id: 12,
            audio_pad_token_id: 13,
            eos_token_id: 14,
            pad_token_id: 15,
        }
    }

    #[test]
    fn kv_geometry_reads_the_llm_decoder_metadata() {
        let geometry = qwen3_kv_geometry(&reference_metadata());
        assert_eq!(
            geometry,
            KvGeometry {
                n_layers: 28,
                kv_heads: 8,
                head_dim: 128,
            }
        );
        // The geometry feeds the shared KV byte model without error (the same
        // 448-row/position shape the runtime_contract capacity anchor pins).
        let default = kv_bytes_per_position(&geometry, LlmKvCacheSpec::DEFAULT).expect("default");
        assert_eq!(default.total(), 448 * 768);
    }

    #[test]
    fn required_positions_is_clamped_to_the_single_decode_window() {
        let metadata = reference_metadata();
        // 30s and 3600s clamp to the same single-decode window, so a multi-hour
        // recording is never judged as one giant decode (the false-reject the
        // clamp exists to prevent).
        let at_window = qwen3_admission_required_positions(
            &metadata,
            QWEN3_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS,
        );
        let at_hour = qwen3_admission_required_positions(&metadata, 3600.0);
        assert_eq!(at_window, at_hour);
        // A short clip needs strictly fewer positions than a full window.
        let short = qwen3_admission_required_positions(&metadata, 5.0);
        assert!(short < at_window, "short={short} window={at_window}");
        // Every estimate stays within the pack's advertised context ceiling.
        assert!(at_window <= metadata.llm_max_positions);
    }

    #[test]
    fn required_positions_counts_audio_tokens_and_generation() {
        let metadata = reference_metadata();
        // 30s at 16kHz/160-hop = 3000 mel frames -> ceil(3000/100)=30 chunks *
        // conv_out_len^3(100)=13 = 390 audio tokens; prompt = 32 + 390 = 422.
        // Desired generation = ceil(30*12)+32 = 392 (> the 128 floor).
        // required = 422 + 392 = 814, well under the 40960 ceiling.
        let positions = qwen3_admission_required_positions(&metadata, 30.0);
        assert_eq!(positions, 814);
    }

    #[test]
    fn required_positions_is_capped_at_llm_max_positions() {
        // A pack with a tiny context ceiling: the prompt alone overruns it, so
        // admission charges exactly the ceiling and never more (saturating, not
        // wrapping into a small "fits" number).
        let mut metadata = reference_metadata();
        metadata.llm_max_positions = 64;
        let positions = qwen3_admission_required_positions(&metadata, 30.0);
        assert_eq!(positions, 64);
    }

    #[test]
    fn qwen3_admission_desired_generation_matches_runtime_budget() {
        // The seconds-based generation budget equals the runtime's sample-count
        // arithmetic (`prepared_audio.len() * 12 / sample_rate`, ceil) on a
        // whole-second input -- the shared-constant contract that keeps the
        // admission estimate from drifting off the real decode budget.
        for seconds in [1usize, 3, 10, 30] {
            let sample_rate = 16_000usize;
            let samples = seconds * sample_rate;
            let runtime_audio_rate = samples
                .checked_mul(QWEN3_DECODE_TOKENS_PER_AUDIO_SECOND)
                .and_then(|value| value.checked_add(sample_rate - 1))
                .and_then(|value| value.checked_div(sample_rate))
                .expect("no overflow");
            let runtime_desired = runtime_audio_rate
                .saturating_add(QWEN3_DECODE_TOKEN_BUDGET_MARGIN)
                .max(QWEN3_DECODE_MIN_GENERATED_TOKENS);
            let admission_desired = qwen3_desired_generated_tokens_for_seconds(seconds as f32);
            assert_eq!(
                admission_desired, runtime_desired,
                "generation budget drifted at {seconds}s"
            );
        }
    }
}
