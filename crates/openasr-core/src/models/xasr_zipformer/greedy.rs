//! RNN-T greedy search for X-ASR.

use super::decoder::XasrDecoder;
use super::joiner::XasrJoiner;
use super::tokenizer::XasrZipformerTokenizer;

pub(crate) const DEFAULT_MAX_SYMBOLS_PER_FRAME: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrGreedyDecodeResult {
    pub token_ids: Vec<u32>,
    /// Absolute encoder frame each token was emitted on (parallel to
    /// `token_ids`).
    pub emit_frames: Vec<usize>,
    /// Joiner softmax probability of each emitted token (parallel to
    /// `token_ids`).
    pub emit_probabilities: Vec<f32>,
    /// Total encoder frames the emission frames index into.
    pub encoder_frames: usize,
    pub text: String,
}

pub(crate) fn greedy_decode_frames(
    encoder_frames: &[f32],
    frame_count: usize,
    encoder_dim: usize,
    decoder: &XasrDecoder,
    joiner: &XasrJoiner,
    tokenizer: &XasrZipformerTokenizer,
    blank_id: u32,
) -> Result<XasrGreedyDecodeResult, String> {
    greedy_decode_frames_with_limit(
        encoder_frames,
        frame_count,
        encoder_dim,
        decoder,
        joiner,
        tokenizer,
        blank_id,
        DEFAULT_MAX_SYMBOLS_PER_FRAME,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn greedy_decode_frames_with_limit(
    encoder_frames: &[f32],
    frame_count: usize,
    encoder_dim: usize,
    decoder: &XasrDecoder,
    joiner: &XasrJoiner,
    tokenizer: &XasrZipformerTokenizer,
    blank_id: u32,
    max_symbols_per_frame: usize,
) -> Result<XasrGreedyDecodeResult, String> {
    let mut context = decoder.initial_context();
    let mut emitted = Vec::new();
    let mut emit_frames = Vec::new();
    let mut emit_probabilities = Vec::new();
    greedy_decode_frames_incremental(
        encoder_frames,
        frame_count,
        encoder_dim,
        decoder,
        joiner,
        blank_id,
        max_symbols_per_frame,
        &mut context,
        &mut emitted,
        &mut emit_frames,
        &mut emit_probabilities,
        0,
    )?;
    let text = tokenizer.decode(&emitted)?;
    Ok(XasrGreedyDecodeResult {
        token_ids: emitted,
        emit_frames,
        emit_probabilities,
        encoder_frames: frame_count,
        text,
    })
}

/// Greedy RNN-T over `frame_count` encoder frames, continuing from the given
/// decoder `context` and appending to `emitted`. Each emission also records
/// its absolute encoder frame (`frame_offset` + local index) into
/// `emit_frames` and its joiner softmax probability into
/// `emit_probabilities`, both kept parallel to `emitted` — the alignment and
/// the per-token score transducers get for free.
///
/// Per-step cost discipline: the encoder projection is computed once per
/// frame, and the decoder state + its projection are recomputed only after a
/// non-blank emission changes the context — across the (overwhelmingly
/// common) blank-only frames, each step runs just the vocab output linear.
/// The probability is computed only on emission (non-blank), so blank-only
/// frames pay nothing extra.
#[allow(clippy::too_many_arguments)]
pub(crate) fn greedy_decode_frames_incremental(
    encoder_frames: &[f32],
    frame_count: usize,
    encoder_dim: usize,
    decoder: &XasrDecoder,
    joiner: &XasrJoiner,
    blank_id: u32,
    max_symbols_per_frame: usize,
    context: &mut Vec<u32>,
    emitted: &mut Vec<u32>,
    emit_frames: &mut Vec<usize>,
    emit_probabilities: &mut Vec<f32>,
    frame_offset: usize,
) -> Result<usize, String> {
    let expected = frame_count
        .checked_mul(encoder_dim)
        .ok_or_else(|| "xasr greedy encoder shape overflow".to_string())?;
    if encoder_frames.len() != expected {
        return Err(format!(
            "xasr greedy got {} encoder values, expected {expected}",
            encoder_frames.len()
        ));
    }
    let start_len = emitted.len();
    let mut scratch = joiner.scratch();
    let mut decoder_projection_valid = false;
    for frame_idx in 0..frame_count {
        let frame = &encoder_frames[frame_idx * encoder_dim..(frame_idx + 1) * encoder_dim];
        joiner.project_encoder_frame(frame, &mut scratch)?;
        for _ in 0..max_symbols_per_frame {
            if !decoder_projection_valid {
                let decoder_state = decoder.decode_context(context)?;
                joiner.project_decoder_state(&decoder_state, &mut scratch)?;
                decoder_projection_valid = true;
            }
            let logits = joiner.logits_from_projected(&mut scratch)?;
            let Some(token_id) = argmax(logits) else {
                return Err("xasr joiner produced no logits".to_string());
            };
            if token_id == blank_id {
                break;
            }
            emitted.push(token_id);
            emit_frames.push(frame_offset + frame_idx);
            emit_probabilities.push(
                crate::models::seq2seq_greedy_decode::token_softmax_probability(
                    logits,
                    token_id as usize,
                ),
            );
            context.remove(0);
            context.push(token_id);
            decoder_projection_valid = false;
        }
    }
    Ok(emitted.len() - start_len)
}

fn argmax(values: &[f32]) -> Option<u32> {
    values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::xasr_zipformer::decoder::XasrDecoder;
    use crate::models::xasr_zipformer::joiner::XasrJoiner;
    use crate::models::xasr_zipformer::weights::{
        NamedTensor, StoredLinear, XasrDecoderWeights, XasrJoinerWeights,
    };

    #[test]
    fn argmax_ignores_nan() {
        assert_eq!(argmax(&[0.0, f32::NAN, 2.0]), Some(2));
    }

    #[test]
    fn greedy_emits_until_blank_and_advances_context() {
        let tokenizer = XasrZipformerTokenizer::new(
            vec![
                "<blk>".to_string(),
                "\u{2581}A".to_string(),
                "\u{2581}B".to_string(),
            ],
            0,
        )
        .unwrap();
        let decoder = XasrDecoder::new(decoder_weights(), 2, 0);
        let joiner = XasrJoiner::new(joiner_weights());
        let result = greedy_decode_frames_with_limit(
            &[1.0, 0.0, 0.0, 1.0],
            2,
            2,
            &decoder,
            &joiner,
            &tokenizer,
            0,
            1,
        )
        .unwrap();
        assert_eq!(result.token_ids, vec![1, 2]);
        assert_eq!(result.text, "A B");
        assert_eq!(result.emit_frames, vec![0, 1]);
        assert_eq!(result.encoder_frames, 2);
        assert_eq!(result.emit_probabilities.len(), 2);
        // The fixture joiner separates the winner by 8 logits; its softmax
        // probability must reflect near-certainty.
        assert!(result.emit_probabilities.iter().all(|p| *p > 0.99));
    }

    #[test]
    fn incremental_emit_frames_are_offset_to_absolute_stream_frames() {
        let decoder = XasrDecoder::new(decoder_weights(), 2, 0);
        let joiner = XasrJoiner::new(joiner_weights());
        let mut context = decoder.initial_context();
        let mut emitted = Vec::new();
        let mut emit_frames = Vec::new();
        let mut emit_probabilities = Vec::new();
        greedy_decode_frames_incremental(
            &[1.0, 0.0, 0.0, 1.0],
            2,
            2,
            &decoder,
            &joiner,
            0,
            1,
            &mut context,
            &mut emitted,
            &mut emit_frames,
            &mut emit_probabilities,
            7,
        )
        .unwrap();
        assert_eq!(emitted.len(), emit_frames.len());
        assert_eq!(emitted.len(), emit_probabilities.len());
        assert_eq!(emit_frames, vec![7, 8]);
    }

    fn decoder_weights() -> XasrDecoderWeights {
        XasrDecoderWeights {
            embedding: StoredLinear {
                name: "emb".to_string(),
                input_dim: 2,
                output_dim: 3,
                values: vec![
                    0.0, 0.0, // blank
                    1.0, 0.0, // token 1
                    0.0, 1.0, // token 2
                ],
            },
            conv_weight: NamedTensor {
                name: "conv".to_string(),
                dims: vec![2, 2, 2],
                values: vec![
                    0.0, 0.0, 1.0, 0.0, // out0 reads second token channel 0
                    0.0, 0.0, 0.0, 1.0, // out1 reads second token channel 1
                ],
            },
            groups: 1,
        }
    }

    fn joiner_weights() -> XasrJoinerWeights {
        XasrJoinerWeights {
            encoder_proj_weight: identity("enc", 2),
            encoder_proj_bias: vec![0.0, 0.0],
            decoder_proj_weight: StoredLinear {
                name: "dec".to_string(),
                input_dim: 2,
                output_dim: 2,
                values: vec![-1.0, 0.0, 0.0, -1.0],
            },
            decoder_proj_bias: vec![0.0, 0.0],
            output_linear_weight: StoredLinear {
                name: "out".to_string(),
                input_dim: 2,
                output_dim: 3,
                values: vec![
                    -4.0, -4.0, // blank
                    4.0, -4.0, // token 1
                    -4.0, 4.0, // token 2
                ],
            },
            output_linear_bias: vec![0.0, 0.0, 0.0],
        }
    }

    fn identity(name: &str, dim: usize) -> StoredLinear {
        let mut values = vec![0.0_f32; dim * dim];
        for i in 0..dim {
            values[i * dim + i] = 1.0;
        }
        StoredLinear {
            name: name.to_string(),
            input_dim: dim,
            output_dim: dim,
            values,
        }
    }
}
