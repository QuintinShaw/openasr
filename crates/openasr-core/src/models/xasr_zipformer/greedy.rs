//! RNN-T greedy search for X-ASR.

use super::decoder::XasrDecoder;
use super::joiner::{XasrJoiner, XasrJoinerScratch};
use super::tokenizer::XasrZipformerTokenizer;

pub(crate) const DEFAULT_MAX_SYMBOLS_PER_FRAME: usize = 8;

pub(crate) trait XasrGreedyDecodeBackend {
    fn project_encoder_frame(&mut self, frame: &[f32]) -> Result<(), String>;
    fn project_decoder_context(&mut self, context: &[u32]) -> Result<(), String>;
    fn next_token(&mut self) -> Result<u32, String>;
    fn token_probability(&self, token: u32) -> Result<f32, String>;

    fn speculative_blank_prefix_len(
        &mut self,
        _context: Option<&[u32]>,
        _encoder_frames: &[f32],
        _frame_count: usize,
        _encoder_dim: usize,
    ) -> Result<Option<usize>, String> {
        Ok(None)
    }
}

struct HostXasrGreedyDecodeBackend<'a> {
    decoder: &'a XasrDecoder,
    joiner: &'a XasrJoiner,
    scratch: XasrJoinerScratch,
}

impl<'a> HostXasrGreedyDecodeBackend<'a> {
    fn new(decoder: &'a XasrDecoder, joiner: &'a XasrJoiner) -> Self {
        Self {
            decoder,
            joiner,
            scratch: joiner.scratch(),
        }
    }
}

impl XasrGreedyDecodeBackend for HostXasrGreedyDecodeBackend<'_> {
    fn project_encoder_frame(&mut self, frame: &[f32]) -> Result<(), String> {
        self.joiner.project_encoder_frame(frame, &mut self.scratch)
    }

    fn project_decoder_context(&mut self, context: &[u32]) -> Result<(), String> {
        let decoder_state = self.decoder.decode_context(context)?;
        self.joiner
            .project_decoder_state(&decoder_state, &mut self.scratch)
    }

    fn next_token(&mut self) -> Result<u32, String> {
        let logits = self.joiner.logits_from_projected(&mut self.scratch)?;
        argmax(logits).ok_or_else(|| "xasr joiner produced no logits".to_string())
    }

    fn token_probability(&self, token: u32) -> Result<f32, String> {
        self.joiner.token_probability(&self.scratch, token)
    }
}

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
        &|| false,
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
    is_canceled: &dyn Fn() -> bool,
) -> Result<usize, String> {
    let mut backend = HostXasrGreedyDecodeBackend::new(decoder, joiner);
    greedy_decode_frames_incremental_with_backend(
        encoder_frames,
        frame_count,
        encoder_dim,
        &mut backend,
        blank_id,
        max_symbols_per_frame,
        context,
        emitted,
        emit_frames,
        emit_probabilities,
        frame_offset,
        is_canceled,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn greedy_decode_frames_incremental_with_backend<B: XasrGreedyDecodeBackend>(
    encoder_frames: &[f32],
    frame_count: usize,
    encoder_dim: usize,
    backend: &mut B,
    blank_id: u32,
    max_symbols_per_frame: usize,
    context: &mut Vec<u32>,
    emitted: &mut Vec<u32>,
    emit_frames: &mut Vec<usize>,
    emit_probabilities: &mut Vec<f32>,
    frame_offset: usize,
    is_canceled: &dyn Fn() -> bool,
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
    let mut decoder_projection_valid = false;
    let mut frame_idx = 0usize;
    while frame_idx < frame_count {
        // The token-control loop remains host-side on every backend, so poll at
        // each encoder-frame boundary in addition to the shared graph-abort
        // callback used by device graph execution.
        if is_canceled() {
            return Err(format!(
                "xasr-zipformer decode canceled at encoder frame {frame_idx}"
            ));
        }
        let remaining_frames = frame_count - frame_idx;
        let remaining_values = &encoder_frames[frame_idx * encoder_dim..];
        let speculative_context = (!decoder_projection_valid).then_some(context.as_slice());
        if let Some(blank_prefix_len) = backend.speculative_blank_prefix_len(
            speculative_context,
            remaining_values,
            remaining_frames,
            encoder_dim,
        )? {
            if blank_prefix_len > remaining_frames {
                return Err("xasr speculative blank prefix exceeds remaining frames".to_string());
            }
            if speculative_context.is_some() {
                decoder_projection_valid = true;
            }
            frame_idx += blank_prefix_len;
            if frame_idx == frame_count {
                break;
            }
        }
        let frame = &encoder_frames[frame_idx * encoder_dim..(frame_idx + 1) * encoder_dim];
        backend.project_encoder_frame(frame)?;
        for _ in 0..max_symbols_per_frame {
            if !decoder_projection_valid {
                backend.project_decoder_context(context)?;
                decoder_projection_valid = true;
            }
            let token_id = backend.next_token()?;
            if token_id == blank_id {
                break;
            }
            emitted.push(token_id);
            emit_frames.push(frame_offset + frame_idx);
            emit_probabilities.push(backend.token_probability(token_id)?);
            context.remove(0);
            context.push(token_id);
            decoder_projection_valid = false;
        }
        frame_idx += 1;
    }
    Ok(emitted.len() - start_len)
}

pub(super) fn argmax(values: &[f32]) -> Option<u32> {
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
    fn argmax_uses_last_index_on_exact_ties() {
        assert_eq!(argmax(&[3.0, 7.0, 7.0, 2.0]), Some(2));
        assert_eq!(argmax(&[f32::NAN, 7.0, 7.0]), Some(2));
        assert_eq!(
            argmax(&[2.0, 1.0, 5.0, 5.0]),
            Some(3),
            "XASR host last-max must keep the last equal maximum"
        );
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
            &|| false,
        )
        .unwrap();
        assert_eq!(emitted.len(), emit_frames.len());
        assert_eq!(emitted.len(), emit_probabilities.len());
        assert_eq!(emit_frames, vec![7, 8]);
    }

    #[test]
    fn greedy_decode_polls_cancellation_at_frame_boundaries() {
        let decoder = XasrDecoder::new(decoder_weights(), 2, 0);
        let joiner = XasrJoiner::new(joiner_weights());
        let mut context = decoder.initial_context();
        let mut emitted = Vec::new();
        let mut emit_frames = Vec::new();
        let mut emit_probabilities = Vec::new();
        // Already canceled before the first frame: the dedicated transducer
        // loop must fail closed without emitting anything, mirroring the
        // shared cooperative cancellation contract (the parakeet-tdt precedent).
        let error = greedy_decode_frames_incremental(
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
            0,
            &|| true,
        )
        .expect_err("a canceled decode must fail closed");
        assert!(error.contains("canceled"), "{error}");
        assert!(
            emitted.is_empty(),
            "cancel polling must remain frame-local and emit nothing"
        );
    }

    struct SpeculativeTestBackend {
        blank_prefix: usize,
        projected_frames: Vec<Vec<f32>>,
        projected_contexts: usize,
        tokens: std::collections::VecDeque<u32>,
    }

    impl XasrGreedyDecodeBackend for SpeculativeTestBackend {
        fn project_encoder_frame(&mut self, frame: &[f32]) -> Result<(), String> {
            self.projected_frames.push(frame.to_vec());
            Ok(())
        }

        fn project_decoder_context(&mut self, _context: &[u32]) -> Result<(), String> {
            self.projected_contexts += 1;
            Ok(())
        }

        fn next_token(&mut self) -> Result<u32, String> {
            self.tokens
                .pop_front()
                .ok_or_else(|| "speculative test backend ran out of tokens".to_string())
        }

        fn token_probability(&self, _token: u32) -> Result<f32, String> {
            Ok(0.75)
        }

        fn speculative_blank_prefix_len(
            &mut self,
            context: Option<&[u32]>,
            _encoder_frames: &[f32],
            _frame_count: usize,
            _encoder_dim: usize,
        ) -> Result<Option<usize>, String> {
            if context.is_some() {
                self.projected_contexts += 1;
            }
            Ok(Some(self.blank_prefix))
        }
    }

    #[test]
    fn speculative_blank_prefix_skips_only_confirmed_frames_then_uses_scalar_path() {
        let mut backend = SpeculativeTestBackend {
            blank_prefix: 2,
            projected_frames: Vec::new(),
            projected_contexts: 0,
            tokens: [1, 0].into_iter().collect(),
        };
        let mut context = vec![0, 0];
        let mut emitted = Vec::new();
        let mut frames = Vec::new();
        let mut probabilities = Vec::new();

        let count = greedy_decode_frames_incremental_with_backend(
            &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
            3,
            2,
            &mut backend,
            0,
            2,
            &mut context,
            &mut emitted,
            &mut frames,
            &mut probabilities,
            7,
            &|| false,
        )
        .expect("speculative blank prefix decode");

        assert_eq!(count, 1);
        assert_eq!(emitted, vec![1]);
        assert_eq!(frames, vec![9]);
        assert_eq!(probabilities, vec![0.75]);
        assert_eq!(context, vec![0, 1]);
        assert_eq!(backend.projected_frames, vec![vec![30.0, 31.0]]);
        assert_eq!(backend.projected_contexts, 2);
    }

    #[test]
    fn speculative_blank_prefix_rejects_backend_overrun() {
        let mut backend = SpeculativeTestBackend {
            blank_prefix: 3,
            projected_frames: Vec::new(),
            projected_contexts: 0,
            tokens: std::collections::VecDeque::new(),
        };
        let mut context = vec![0, 0];
        let mut emitted = Vec::new();
        let mut frames = Vec::new();
        let mut probabilities = Vec::new();

        let error = greedy_decode_frames_incremental_with_backend(
            &[1.0, 2.0, 3.0, 4.0],
            2,
            2,
            &mut backend,
            0,
            1,
            &mut context,
            &mut emitted,
            &mut frames,
            &mut probabilities,
            0,
            &|| false,
        )
        .expect_err("speculative prefix beyond remaining frames must fail closed");

        assert!(error.contains("exceeds remaining frames"), "{error}");
        assert!(emitted.is_empty());
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
                native: None,
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
                native: None,
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
                native: None,
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
            native: None,
        }
    }
}
