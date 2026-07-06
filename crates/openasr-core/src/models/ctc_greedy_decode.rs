//! Non-autoregressive CTC greedy decode (goal-1 `Ctc` orchestration shape).
//!
//! CTC greedy: per-frame argmax over the `(vocab + blank)` logit row, collapse
//! consecutive duplicate ids (the "transition" rule: emit only when the argmax
//! changes), drop the blank id, then detokenize the survivors. There is NO
//! autoregressive loop, NO KV cache, NO step executor — the opposite of the
//! seq2seq decode path, which is exactly why the `Ctc` shape needs its own
//! decode entry point rather than reusing `run_seq2seq_greedy_decode_loop_*`.

use thiserror::Error;

use crate::models::ctc_prefix_beam::{CtcContextGraph, run_ctc_prefix_beam_decode};
use crate::models::phrase_bias_decode::TokenPhraseBias;
use crate::models::seq2seq_greedy_decode::token_softmax_probability;

/// Static CTC decode parameters carried from the pack metadata.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CtcGreedyDecodeConfig {
    /// The CTC blank token id (read from pack metadata; parakeet-ctc-0.6b = 1024).
    /// MUST be read from metadata, never assumed to equal `vocab_size`.
    pub blank_token_id: u32,
    /// Width of each frame's logit row = vocabulary size including the blank.
    pub vocab_size: usize,
    pub phrase_biases: Vec<TokenPhraseBias>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum CtcGreedyDecodeError {
    #[error("ctc greedy decode: no frame logits")]
    EmptyFrames,
    #[error("ctc greedy decode: blank_token_id {blank} out of range for vocab_size {vocab_size}")]
    BlankOutOfRange { blank: u32, vocab_size: usize },
    #[error("ctc greedy decode: frame {frame} logit row width {got} != vocab_size {expected}")]
    FrameWidthMismatch {
        frame: usize,
        got: usize,
        expected: usize,
    },
    #[error("ctc greedy decode: frame {frame} has non-finite logits")]
    NonFiniteLogits { frame: usize },
    #[error("ctc greedy decode: detokenize failed: {reason}")]
    DetokenizeFailed { reason: String },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CtcTokenFrameSpan {
    pub token_id: u32,
    pub start_frame: usize,
    /// Exclusive end frame for the CTC argmax run that emitted this token.
    pub end_frame: usize,
    /// Mean softmax probability of the argmax over the run's frames.
    pub probability: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CtcGreedyDecodeResult {
    pub token_ids: Vec<u32>,
    pub token_spans: Vec<CtcTokenFrameSpan>,
    pub frame_count: usize,
    pub text: String,
}

#[derive(Debug, Clone)]
pub(crate) struct IncrementalCtcGreedyDecoder {
    config: CtcGreedyDecodeConfig,
    token_ids: Vec<u32>,
    token_spans: Vec<CtcTokenFrameSpan>,
    prev_argmax: Option<u32>,
    run_start_frame: usize,
    run_probability_sum: f32,
    processed_frames: usize,
}

impl IncrementalCtcGreedyDecoder {
    pub(crate) fn new(config: CtcGreedyDecodeConfig) -> Result<Self, CtcGreedyDecodeError> {
        if config.blank_token_id as usize >= config.vocab_size {
            return Err(CtcGreedyDecodeError::BlankOutOfRange {
                blank: config.blank_token_id,
                vocab_size: config.vocab_size,
            });
        }
        Ok(Self {
            config,
            token_ids: Vec::new(),
            token_spans: Vec::new(),
            prev_argmax: None,
            run_start_frame: 0,
            run_probability_sum: 0.0,
            processed_frames: 0,
        })
    }

    pub(crate) fn append_frames(
        &mut self,
        frame_logits: &[&[f32]],
    ) -> Result<(), CtcGreedyDecodeError> {
        for row in frame_logits {
            self.append_frame(row)?;
        }
        Ok(())
    }

    pub(crate) fn append_frame(&mut self, row: &[f32]) -> Result<(), CtcGreedyDecodeError> {
        let frame = self.processed_frames;
        if row.len() != self.config.vocab_size {
            return Err(CtcGreedyDecodeError::FrameWidthMismatch {
                frame,
                got: row.len(),
                expected: self.config.vocab_size,
            });
        }
        // The greedy path is blank-collapse argmax with NO phrase biasing: hotword
        // biasing on CTC runs through the prefix-beam decoder instead (see
        // `run_ctc_greedy_decode`), so this decoder only ever sees an empty bias
        // set and its output stays byte-identical to the un-biased decode.
        let argmax = self.argmax_frame(row, frame)?;
        // Confidence of this frame's pick; a token's probability is the mean over
        // its argmax run.
        let probability = token_softmax_probability(row, argmax as usize);
        match self.prev_argmax {
            None => {
                self.prev_argmax = Some(argmax);
                self.run_start_frame = frame;
                self.run_probability_sum = probability;
            }
            Some(previous) if previous != argmax => {
                push_ctc_token_run(
                    previous,
                    self.run_start_frame,
                    frame,
                    self.run_probability_sum,
                    self.config.blank_token_id,
                    &mut self.token_ids,
                    &mut self.token_spans,
                );
                self.prev_argmax = Some(argmax);
                self.run_start_frame = frame;
                self.run_probability_sum = probability;
            }
            Some(_) => {
                self.run_probability_sum += probability;
            }
        }
        self.processed_frames = self.processed_frames.saturating_add(1);
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn committed_token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    #[allow(dead_code)]
    pub(crate) fn committed_token_spans(&self) -> &[CtcTokenFrameSpan] {
        &self.token_spans
    }

    #[allow(dead_code)]
    pub(crate) fn processed_frame_count(&self) -> usize {
        self.processed_frames
    }

    pub(crate) fn finish<E>(
        mut self,
        decode_text_token_ids: impl Fn(&[u32]) -> Result<String, E>,
        map_err: impl Fn(E) -> CtcGreedyDecodeError,
    ) -> Result<CtcGreedyDecodeResult, CtcGreedyDecodeError> {
        if self.processed_frames == 0 {
            return Err(CtcGreedyDecodeError::EmptyFrames);
        }
        if let Some(previous) = self.prev_argmax {
            push_ctc_token_run(
                previous,
                self.run_start_frame,
                self.processed_frames,
                self.run_probability_sum,
                self.config.blank_token_id,
                &mut self.token_ids,
                &mut self.token_spans,
            );
        }

        let text = decode_text_token_ids(&self.token_ids).map_err(map_err)?;
        Ok(CtcGreedyDecodeResult {
            token_ids: self.token_ids,
            token_spans: self.token_spans,
            frame_count: self.processed_frames,
            text,
        })
    }

    fn argmax_frame(&self, row: &[f32], frame: usize) -> Result<u32, CtcGreedyDecodeError> {
        // Argmax over raw logits: softmax is monotonic, so it is unnecessary.
        // Fail closed on NaN/Inf rather than letting a bogus argmax win.
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (index, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(CtcGreedyDecodeError::NonFiniteLogits { frame });
            }
            if value > best_val {
                best_val = value;
                best_idx = index;
            }
        }
        Ok(best_idx as u32)
    }
}

/// Collapse CTC frame logits to token ids and detokenize. `frame_logits[t]` is
/// the length-`vocab_size` logit row for frame `t`; `decode_text_token_ids` maps
/// the collapsed ids to text (e.g. a SentencePiece detokenizer). Fail-closed on
/// every malformed input.
///
/// Two decode paths, selected by whether the config carries hotwords:
///
/// - No phrase biases (the default): per-frame argmax -> merge consecutive
///   duplicates -> drop blank. Byte-identical to the historical greedy decode;
///   zero added work.
/// - With hotwords: a CTC prefix-beam search biased by an Aho-Corasick context
///   graph (`ctc_prefix_beam`). This is the ONLY place hotwords touch the CTC
///   decode -- the greedy argmax above is never phrase-biased, because per-frame
///   logit nudging cannot bias a maximum-alignment decoder without wrecking the
///   transcript. If the biases produce no positive context (e.g. only negative
///   anti-context, which the prefix beam does not represent), fall back to the
///   plain greedy path.
pub(crate) fn run_ctc_greedy_decode<E>(
    config: CtcGreedyDecodeConfig,
    frame_logits: &[&[f32]],
    decode_text_token_ids: impl Fn(&[u32]) -> Result<String, E>,
    map_err: impl Fn(E) -> CtcGreedyDecodeError,
) -> Result<CtcGreedyDecodeResult, CtcGreedyDecodeError> {
    if config.blank_token_id as usize >= config.vocab_size {
        return Err(CtcGreedyDecodeError::BlankOutOfRange {
            blank: config.blank_token_id,
            vocab_size: config.vocab_size,
        });
    }
    if !config.phrase_biases.is_empty()
        && let Some(graph) = CtcContextGraph::from_token_phrase_biases(&config.phrase_biases)
    {
        return run_ctc_prefix_beam_decode(
            config.blank_token_id,
            config.vocab_size,
            &graph,
            frame_logits,
            decode_text_token_ids,
            map_err,
        );
    }
    let mut decoder = IncrementalCtcGreedyDecoder::new(config)?;
    decoder.append_frames(frame_logits)?;
    decoder.finish(decode_text_token_ids, map_err)
}

fn push_ctc_token_run(
    token_id: u32,
    start_frame: usize,
    end_frame: usize,
    probability_sum: f32,
    blank_token_id: u32,
    token_ids: &mut Vec<u32>,
    token_spans: &mut Vec<CtcTokenFrameSpan>,
) {
    if token_id == blank_token_id {
        return;
    }
    let run_frames = end_frame.saturating_sub(start_frame).max(1);
    token_ids.push(token_id);
    token_spans.push(CtcTokenFrameSpan {
        token_id,
        start_frame,
        end_frame: end_frame.max(start_frame.saturating_add(1)),
        probability: (probability_sum / run_frames as f32).clamp(0.0, 1.0),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB: usize = 5; // ids 0..=3 real, 4 = blank
    const BLANK: u32 = 4;

    fn cfg() -> CtcGreedyDecodeConfig {
        CtcGreedyDecodeConfig {
            blank_token_id: BLANK,
            vocab_size: VOCAB,
            phrase_biases: Vec::new(),
        }
    }

    /// One-hot logit row peaking at `id`.
    fn frame(id: u32) -> Vec<f32> {
        let mut row = vec![0.0f32; VOCAB];
        row[id as usize] = 10.0;
        row
    }

    fn decode_ids(ids: &[u32]) -> Result<String, std::convert::Infallible> {
        Ok(ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(","))
    }

    fn run(rows: &[Vec<f32>]) -> Result<CtcGreedyDecodeResult, CtcGreedyDecodeError> {
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        run_ctc_greedy_decode(cfg(), &refs, decode_ids, |never| match never {})
    }

    /// Softmax probability of the one-hot peak in [`frame`] rows; every
    /// `frame(id)` row yields the same value by symmetry.
    fn one_hot_probability() -> f32 {
        token_softmax_probability(&frame(1), 1)
    }

    #[test]
    fn collapses_consecutive_duplicate_runs() {
        let r = run(&[frame(1), frame(1), frame(1)]).unwrap();
        assert_eq!(r.frame_count, 3);
        assert_eq!(r.token_ids, vec![1]);
        assert_eq!(
            r.token_spans,
            vec![CtcTokenFrameSpan {
                token_id: 1,
                start_frame: 0,
                end_frame: 3,
                probability: one_hot_probability(),
            }]
        );
    }

    #[test]
    fn drops_blanks() {
        let r = run(&[frame(1), frame(BLANK), frame(BLANK), frame(2)]).unwrap();
        assert_eq!(r.token_ids, vec![1, 2]);
        assert_eq!(
            r.token_spans,
            vec![
                CtcTokenFrameSpan {
                    token_id: 1,
                    start_frame: 0,
                    end_frame: 1,
                    probability: one_hot_probability(),
                },
                CtcTokenFrameSpan {
                    token_id: 2,
                    start_frame: 3,
                    end_frame: 4,
                    probability: one_hot_probability(),
                },
            ]
        );
        assert_eq!(r.text, "1,2");
    }

    #[test]
    fn token_probability_is_the_mean_over_its_argmax_run() {
        // Two frames pick id 1 with different margins; the emitted token's
        // probability is the mean of the two per-frame softmax values.
        let sharp = frame(1);
        let mut soft = vec![0.0f32; VOCAB];
        soft[1] = 1.0;
        let expected =
            (token_softmax_probability(&sharp, 1) + token_softmax_probability(&soft, 1)) / 2.0;
        let r = run(&[sharp, soft]).unwrap();
        assert_eq!(r.token_ids, vec![1]);
        assert!((r.token_spans[0].probability - expected).abs() < 1e-6);
    }

    #[test]
    fn blank_separated_duplicate_is_emitted_twice() {
        // A blank between two identical ids breaks the collapse run: 1 _ 1 -> 1 1.
        let r = run(&[frame(1), frame(BLANK), frame(1)]).unwrap();
        assert_eq!(r.token_ids, vec![1, 1]);
    }

    #[test]
    fn adjacent_duplicate_without_blank_collapses() {
        // 1 1 2 2 3 -> 1 2 3 (no blanks between, runs merge).
        let r = run(&[frame(1), frame(1), frame(2), frame(2), frame(3)]).unwrap();
        assert_eq!(r.token_ids, vec![1, 2, 3]);
    }

    #[test]
    fn rejects_empty_frames() {
        assert_eq!(run(&[]), Err(CtcGreedyDecodeError::EmptyFrames));
    }

    #[test]
    fn rejects_blank_out_of_range() {
        let bad = CtcGreedyDecodeConfig {
            blank_token_id: 9,
            vocab_size: VOCAB,
            phrase_biases: Vec::new(),
        };
        let row = frame(1);
        let refs: [&[f32]; 1] = [row.as_slice()];
        assert_eq!(
            run_ctc_greedy_decode(bad, &refs, decode_ids, |never| match never {}),
            Err(CtcGreedyDecodeError::BlankOutOfRange {
                blank: 9,
                vocab_size: VOCAB
            })
        );
    }

    #[test]
    fn rejects_frame_width_mismatch() {
        let good = frame(1);
        let short = vec![0.0f32; VOCAB - 1];
        let refs: [&[f32]; 2] = [good.as_slice(), short.as_slice()];
        assert_eq!(
            run_ctc_greedy_decode(cfg(), &refs, decode_ids, |never| match never {}),
            Err(CtcGreedyDecodeError::FrameWidthMismatch {
                frame: 1,
                got: VOCAB - 1,
                expected: VOCAB
            })
        );
    }

    #[test]
    fn rejects_non_finite_logits() {
        let mut bad = frame(0);
        bad[2] = f32::NAN;
        assert_eq!(
            run(&[frame(1), bad]),
            Err(CtcGreedyDecodeError::NonFiniteLogits { frame: 1 })
        );
    }

    #[test]
    fn detokenize_error_is_propagated_fail_closed() {
        let row = frame(1);
        let refs: [&[f32]; 1] = [row.as_slice()];
        let result = run_ctc_greedy_decode(
            cfg(),
            &refs,
            |_ids| Err("tokenizer exploded".to_string()),
            |reason| CtcGreedyDecodeError::DetokenizeFailed { reason },
        );
        assert_eq!(
            result,
            Err(CtcGreedyDecodeError::DetokenizeFailed {
                reason: "tokenizer exploded".to_string()
            })
        );
    }

    #[test]
    fn non_empty_phrase_bias_routes_to_the_prefix_beam_and_can_change_the_label() {
        // A non-empty positive hotword makes `run_ctc_greedy_decode` route to the
        // prefix-beam decoder, whose accumulated context score can flip the label
        // away from the greedy argmax (token 2) to the hotword (token 1). The
        // greedy path itself is never phrase-biased.
        let rows = [
            vec![0.0, 0.8, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 10.0],
        ];
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let result = run_ctc_greedy_decode(
            CtcGreedyDecodeConfig {
                blank_token_id: BLANK,
                vocab_size: VOCAB,
                phrase_biases: vec![TokenPhraseBias::new(vec![vec![1]], 5.0).unwrap()],
            },
            &refs,
            decode_ids,
            |never| match never {},
        )
        .unwrap();

        assert_eq!(result.token_ids, vec![1]);
    }

    #[test]
    fn incremental_decoder_matches_batch_across_chunks() {
        let rows = [
            frame(1),
            frame(1),
            frame(BLANK),
            frame(2),
            frame(2),
            frame(3),
            frame(BLANK),
        ];
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let batch = run_ctc_greedy_decode(cfg(), &refs, decode_ids, |never| match never {})
            .expect("batch ctc");

        let mut incremental = IncrementalCtcGreedyDecoder::new(cfg()).expect("decoder");
        incremental.append_frames(&refs[0..2]).expect("first chunk");
        incremental
            .append_frames(&refs[2..5])
            .expect("second chunk");
        incremental.append_frames(&refs[5..]).expect("third chunk");
        assert_eq!(incremental.processed_frame_count(), refs.len());
        let streamed = incremental
            .finish(decode_ids, |never| match never {})
            .expect("streaming ctc");

        assert_eq!(streamed, batch);
        assert_eq!(streamed.token_ids, vec![1, 2, 3]);
    }

    #[test]
    fn incremental_decoder_commits_only_closed_runs_before_finish() {
        let mut decoder = IncrementalCtcGreedyDecoder::new(cfg()).expect("decoder");

        decoder.append_frame(&frame(1)).expect("first frame");
        decoder.append_frame(&frame(1)).expect("duplicate frame");
        assert!(decoder.committed_token_ids().is_empty());
        assert!(decoder.committed_token_spans().is_empty());

        decoder.append_frame(&frame(BLANK)).expect("blank boundary");
        assert_eq!(decoder.committed_token_ids(), &[1]);
        assert_eq!(
            decoder.committed_token_spans(),
            &[CtcTokenFrameSpan {
                token_id: 1,
                start_frame: 0,
                end_frame: 2,
                probability: one_hot_probability(),
            }]
        );

        decoder.append_frame(&frame(2)).expect("new active run");
        assert_eq!(decoder.committed_token_ids(), &[1]);

        let finished = decoder
            .finish(decode_ids, |never| match never {})
            .expect("finish");
        assert_eq!(finished.token_ids, vec![1, 2]);
    }

    #[test]
    fn incremental_greedy_decoder_ignores_phrase_biases() {
        // The incremental greedy decoder is un-biased: hotwords are handled by the
        // prefix-beam batch path, never here. Even with a bias in the config, this
        // decoder emits the plain greedy argmax (token 2 wins frame 0).
        let rows = [
            vec![0.0, 0.8, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 10.0],
        ];
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let config = CtcGreedyDecodeConfig {
            blank_token_id: BLANK,
            vocab_size: VOCAB,
            phrase_biases: vec![TokenPhraseBias::new(vec![vec![1]], 5.0).unwrap()],
        };
        let mut incremental = IncrementalCtcGreedyDecoder::new(config).expect("decoder");
        incremental.append_frame(refs[0]).expect("first");
        incremental.append_frame(refs[1]).expect("second");
        let streamed = incremental
            .finish(decode_ids, |never| match never {})
            .expect("streaming ctc");

        assert_eq!(streamed.token_ids, vec![2]);
    }
}
