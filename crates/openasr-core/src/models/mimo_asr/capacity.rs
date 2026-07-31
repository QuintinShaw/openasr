//! mimo-asr capacity derivation: the family's Qwen2 backbone KV geometry and
//! the decoder positions a single decode of a given audio span needs,
//! assembled from PACK METADATA (`MimoLlmMetadata` + `MimoMelMetadata` +
//! `MimoAudiotokMetadata` + `MimoInlocalMetadata`) plus family constants, for
//! the shared host-memory admission check
//! ([`crate::capacity::evaluate_host_memory_admission`]). The mimo-asr
//! analogue of [`crate::models::qwen::capacity`].
//!
//! Reuses, never re-derives:
//! - KV geometry comes straight off the parsed pack (`mimo.llm.*` keys).
//! - The audio-token rate is the pack-carried stride product the registry row
//!   in `crate::capacity` records: mel `sample_rate/hop_length` through the
//!   tokenizer conv stem (`conv1_stride * conv2_stride * down_sample_stride`,
//!   25Hz RVQ frames for the shipped pack) down to one LLM position per
//!   `group_size` frames (the input-local group downcast).
//! - The generation budget and the single-decode window reuse the executor's
//!   own constants ([`super::executor::MIMO_ASR_MAX_GENERATED_TOKENS`],
//!   [`super::executor::MIMO_ASR_MAX_INPUT_SECONDS`]), so admission charges
//!   the same `prompt + generation` KV capacity
//!   `MimoLlmDecoderRuntime::new_kv_caches` actually allocates.

use crate::capacity::KvGeometry;

use super::executor::{MIMO_ASR_MAX_GENERATED_TOKENS, MIMO_ASR_MAX_INPUT_SECONDS};
use super::runtime_contract::{
    MimoAudiotokMetadata, MimoInlocalMetadata, MimoLlmMetadata, MimoMelMetadata,
};

/// The audio a single mimo-asr decode is estimated to fold in, for admission.
/// The executor fails closed above this per-chunk cap (the reference
/// `preprocess_input`'s own 30s re-chunk bound), and longer recordings reach
/// it only as longform slices each at or under this bound -- so no single
/// decode's KV cache ever spans more audio. Same clamp rationale as qwen3's
/// `QWEN3_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS`.
pub(crate) const MIMO_ASR_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS: f32 = MIMO_ASR_MAX_INPUT_SECONDS;

/// Fixed ChatML/`<|sosp|>`/`<|eosp|>` wrapper tokens around the audio span
/// (see `super::decode_prompt::build_mimo_asr_decode_prompt`). A small
/// conservative figure, dwarfed by the audio-group term at any real duration;
/// its exact value cannot swing an admission decision.
const MIMO_ASR_ADMISSION_FIXED_PROMPT_TOKENS: usize = 32;

/// The backbone KV geometry the loaded pack advertises.
pub(crate) fn mimo_asr_kv_geometry(llm: &MimoLlmMetadata) -> KvGeometry {
    KvGeometry {
        n_layers: llm.n_layers,
        kv_heads: llm.n_kv_heads,
        head_dim: llm.head_dim,
    }
}

/// Decoder positions a single decode of `audio_duration_seconds` requires --
/// the admission figure `evaluate_host_memory_admission` charges KV bytes
/// against. Mirrors the runtime's own KV-capacity sizing
/// (`decode_prompt.token_ids.len() + MIMO_ASR_MAX_GENERATED_TOKENS` in
/// `super::executor`): fixed wrapper + one position per `group_size` RVQ
/// frames + the full runaway-generation backstop, with the audio clamped to
/// the family's single-decode window.
pub(crate) fn mimo_asr_admission_required_positions(
    mel: &MimoMelMetadata,
    audiotok: &MimoAudiotokMetadata,
    inlocal: &MimoInlocalMetadata,
    audio_duration_seconds: f32,
) -> usize {
    let admission_seconds =
        audio_duration_seconds.clamp(0.0, MIMO_ASR_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS);
    let mel_frames =
        (admission_seconds * mel.sample_rate_hz as f32 / mel.hop_length.max(1) as f32) as usize;
    // The tokenizer conv stem's time-axis stride product (conv1 is stride 1
    // for the shipped pack; all three are pack-carried facts).
    let stem_stride = audiotok
        .conv1_stride
        .max(1)
        .saturating_mul(audiotok.conv2_stride.max(1))
        .saturating_mul(audiotok.down_sample_stride.max(1));
    let rvq_frames = mel_frames / stem_stride;
    // The input-local group downcast folds `group_size` RVQ frames into one
    // spliced LLM position (trailing remainder truncated, exactly as the
    // executor truncates codes to a whole-group multiple).
    let audio_groups = rvq_frames / inlocal.group_size.max(1);
    MIMO_ASR_ADMISSION_FIXED_PROMPT_TOKENS
        .saturating_add(audio_groups)
        .saturating_add(MIMO_ASR_MAX_GENERATED_TOKENS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacity::kv_bytes_per_position;
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
    fn required_positions_is_clamped_to_the_single_decode_window() {
        let (mel, audiotok, inlocal) = (reference_mel(), reference_audiotok(), reference_inlocal());
        let at_window = mimo_asr_admission_required_positions(
            &mel,
            &audiotok,
            &inlocal,
            MIMO_ASR_ADMISSION_SINGLE_DECODE_WINDOW_SECONDS,
        );
        let at_hour = mimo_asr_admission_required_positions(&mel, &audiotok, &inlocal, 3600.0);
        assert_eq!(at_window, at_hour);
        let short = mimo_asr_admission_required_positions(&mel, &audiotok, &inlocal, 5.0);
        assert!(short < at_window, "short={short} window={at_window}");
        assert!(at_window < reference_llm().max_positions);
    }

    #[test]
    fn required_positions_walks_the_runtime_pipeline_arithmetic() {
        // 30s at 24kHz/240-hop = 3000 mel frames -> stem stride 1*2*2 = 750
        // RVQ frames (25Hz) -> group 4 = 187 audio groups (truncated).
        // required = 32 fixed + 187 groups + 512 generation backstop = 731.
        let positions = mimo_asr_admission_required_positions(
            &reference_mel(),
            &reference_audiotok(),
            &reference_inlocal(),
            30.0,
        );
        assert_eq!(positions, 731);
    }
}
