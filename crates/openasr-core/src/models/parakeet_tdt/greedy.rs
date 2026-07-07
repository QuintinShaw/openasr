//! TDT (Token-and-Duration Transducer) greedy decode.
//!
//! Per step the joint head produces one fused logit row over
//! `[vocab+blank | durations]`: the token half is argmaxed for the emission
//! and the duration half is argmaxed (separately — it is trained as an n-way
//! classifier) for the frame skip. Semantics follow NeMo's
//! `GreedyTDTInfer._greedy_decode` (cross-checked against the CrispASR ggml
//! reimplementation):
//!
//! - blank, duration > 0: no emission; advance `t` by the duration.
//! - blank, duration = 0: no emission and no advance; the retry counts
//!   toward the shared per-frame `max_symbols_per_step` budget.
//! - token, duration >= 0: emit; advance the predictor by the token; advance
//!   `t` by the duration (0 keeps decoding more symbols on this frame).
//!
//! LOOP INVARIANT (progress guarantee): every outer iteration either advances
//! `t` by at least 1 (any duration > 0, or the forced `t += 1` when the inner
//! budget is exhausted by duration-0 steps), so the decode terminates after
//! at most `frames * max_symbols_per_step` joint evaluations.

use crate::models::seq2seq_greedy_decode::token_softmax_probability;

use super::encoder_weights::ParakeetTdtJointWeights;
use super::predictor::{ParakeetTdtPredictor, dot_f32};
use super::runtime_contract::ParakeetTdtExecutionMetadata;

/// One emitted token with its TDT-native frame span: `start_frame` is the
/// encoder frame at emission, `end_frame` = start + predicted duration
/// (clamped to the frame count). The duration head is the model's own
/// alignment output, which is what makes parakeet-tdt word timestamps native
/// rather than approximate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParakeetTdtEmittedToken {
    pub token_id: u32,
    pub start_frame: usize,
    pub end_frame: usize,
    /// Softmax probability of the emitted token over the token half of the
    /// joint logits (duration half excluded — separate classifier).
    pub probability: f32,
}

/// Precomputed host-side joint: the predictor projection + fused output head.
/// The encoder projection is already applied in-graph (the encoder output IS
/// `enc_proj(enc)`), so per frame the joint costs only the ReLU add + the
/// output head matvec; the predictor projection reruns only after an
/// emission changes the predictor state.
#[derive(Debug, Clone)]
pub(crate) struct ParakeetTdtJoint {
    weights: ParakeetTdtJointWeights,
    joint_hidden: usize,
    out_rows: usize,
}

pub(crate) struct ParakeetTdtJointScratch {
    pred_proj: Vec<f32>,
    mid: Vec<f32>,
    logits: Vec<f32>,
    pred_out: Vec<f32>,
}

impl ParakeetTdtJoint {
    pub(crate) fn new(weights: ParakeetTdtJointWeights, joint_hidden: usize) -> Self {
        let out_rows = weights.out_bias.values.len();
        Self {
            weights,
            joint_hidden,
            out_rows,
        }
    }

    pub(crate) fn scratch(&self) -> ParakeetTdtJointScratch {
        ParakeetTdtJointScratch {
            pred_proj: vec![0.0; self.joint_hidden],
            mid: vec![0.0; self.joint_hidden],
            logits: vec![0.0; self.out_rows],
            pred_out: Vec::with_capacity(self.joint_hidden),
        }
    }

    /// `pred_proj = joint.pred @ pred_out + bias` — recomputed only when the
    /// predictor state changes (i.e. after an emission).
    fn project_predictor(&self, pred_out: &[f32], scratch: &mut ParakeetTdtJointScratch) {
        let in_dim = pred_out.len();
        for (row, out) in scratch.pred_proj.iter_mut().enumerate() {
            let w = &self.weights.pred_weight.values[row * in_dim..(row + 1) * in_dim];
            *out = dot_f32(w, pred_out) + self.weights.pred_bias.values[row];
        }
    }

    /// `logits = joint.out @ relu(enc_frame + pred_proj) + bias`.
    fn logits<'s>(&self, enc_frame: &[f32], scratch: &'s mut ParakeetTdtJointScratch) -> &'s [f32] {
        for ((mid, &enc), &pred) in scratch
            .mid
            .iter_mut()
            .zip(enc_frame)
            .zip(&scratch.pred_proj)
        {
            *mid = (enc + pred).max(0.0);
        }
        let joint = self.joint_hidden;
        for (row, out) in scratch.logits.iter_mut().enumerate() {
            let w = &self.weights.out_weight.values[row * joint..(row + 1) * joint];
            *out = dot_f32(w, &scratch.mid) + self.weights.out_bias.values[row];
        }
        &scratch.logits
    }
}

/// Greedy TDT over `frame_count` encoder frames of `enc_features`
/// (frame-major `[frame][joint_hidden]`, already encoder-projected).
pub(crate) fn tdt_greedy_decode(
    enc_features: &[f32],
    frame_count: usize,
    metadata: &ParakeetTdtExecutionMetadata,
    predictor: &ParakeetTdtPredictor,
    joint: &ParakeetTdtJoint,
) -> Result<Vec<ParakeetTdtEmittedToken>, String> {
    let joint_hidden = metadata.joint_hidden;
    let expected = frame_count
        .checked_mul(joint_hidden)
        .ok_or_else(|| "parakeet-tdt greedy encoder shape overflow".to_string())?;
    if enc_features.len() != expected {
        return Err(format!(
            "parakeet-tdt greedy got {} encoder values, expected {expected}",
            enc_features.len()
        ));
    }
    let vocab = metadata.vocab_size; // includes the blank (last id)
    let n_durations = metadata.n_durations;
    if joint.out_rows != vocab + n_durations {
        return Err(format!(
            "parakeet-tdt joint head has {} rows, expected vocab {vocab} + durations {n_durations}",
            joint.out_rows
        ));
    }
    let blank = metadata.blank_token_id;
    let max_symbols = metadata.max_symbols_per_step.max(1);

    let mut emitted = Vec::new();
    let mut state = predictor.initial_state();
    let mut scratch = joint.scratch();

    // SOS: feed the blank through the LSTM from the zero state (NeMo
    // convention; the blank embedding row is the trained padding_idx zeros).
    let mut pred_out = std::mem::take(&mut scratch.pred_out);
    predictor.step(blank, &mut state, &mut pred_out)?;
    joint.project_predictor(&pred_out, &mut scratch);

    let mut t = 0usize;
    while t < frame_count {
        let enc_frame = &enc_features[t * joint_hidden..(t + 1) * joint_hidden];
        let mut symbols_this_frame = 0usize;
        // Inner loop: symbols decoded without leaving frame `t`. Exits when a
        // duration > 0 advances `t`, or the budget forces `t += 1`.
        loop {
            let logits = joint.logits(enc_frame, &mut scratch);
            let token_id = argmax(&logits[..vocab])
                .ok_or_else(|| "parakeet-tdt joint produced no token logits".to_string())?;
            let duration = argmax(&logits[vocab..vocab + n_durations])
                .ok_or_else(|| "parakeet-tdt joint produced no duration logits".to_string())?
                as usize; // durations are the contiguous range 0..n (validated)

            if token_id != blank {
                let probability = token_softmax_probability(&logits[..vocab], token_id as usize);
                emitted.push(ParakeetTdtEmittedToken {
                    token_id,
                    start_frame: t,
                    end_frame: (t + duration).min(frame_count),
                    probability,
                });
                predictor.step(token_id, &mut state, &mut pred_out)?;
                joint.project_predictor(&pred_out, &mut scratch);
            }

            if duration > 0 {
                t += duration;
                break;
            }
            // duration == 0 (blank or token): stay on this frame, but count
            // against the budget so a degenerate joint cannot spin forever.
            symbols_this_frame += 1;
            if symbols_this_frame >= max_symbols {
                // Forced progress: NeMo's `if symbols_added == max_symbols:
                // time_idx += 1` guard.
                t += 1;
                break;
            }
        }
    }
    scratch.pred_out = pred_out;
    Ok(emitted)
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
    use crate::models::parakeet_tdt::encoder_weights::{
        NamedTensor, ParakeetTdtJointWeights, ParakeetTdtLstmLayerWeights,
        ParakeetTdtPredictorWeights,
    };

    fn named(name: &str, values: Vec<f32>) -> NamedTensor {
        NamedTensor {
            name: name.to_string(),
            dims: vec![values.len()],
            values,
        }
    }

    fn metadata(
        vocab: usize,
        n_durations: usize,
        max_symbols: usize,
    ) -> ParakeetTdtExecutionMetadata {
        ParakeetTdtExecutionMetadata {
            n_layers: 1,
            hidden_size: 2,
            n_heads: 1,
            head_dim: 2,
            ffn_dim: 2,
            conv_kernel: 9,
            n_mels: 128,
            subsampling_factor: 8,
            subsampling_channels: 2,
            scale_input: false,
            vocab_size: vocab,
            blank_token_id: (vocab - 1) as u32,
            pred_hidden: 2,
            pred_layers: 2,
            joint_hidden: 2,
            n_durations,
            max_symbols_per_step: max_symbols,
        }
    }

    /// Identity-ish fixture: predictor output is ~constant, joint passes the
    /// encoder frame through (pred projection zero), and the output head maps
    /// joint dims so the winning token/duration is controlled per frame by
    /// the encoder features directly.
    fn fixture() -> (ParakeetTdtPredictor, ParakeetTdtJoint) {
        let hidden = 2;
        let layer = || ParakeetTdtLstmLayerWeights {
            w_ih: named("w_ih", vec![0.0; 4 * hidden * hidden]),
            w_hh: named("w_hh", vec![0.0; 4 * hidden * hidden]),
            b_ih: named("b_ih", vec![0.0; 4 * hidden]),
            b_hh: named("b_hh", vec![0.0; 4 * hidden]),
        };
        let predictor = ParakeetTdtPredictor::new(
            ParakeetTdtPredictorWeights {
                embedding: named("embed", vec![0.0; 3 * hidden]),
                lstm_layers: vec![layer(), layer()],
            },
            hidden,
            3,
        );
        // vocab = 3 (ids 0, 1, blank=2), durations = 2 (0, 1) -> 5 rows.
        // Row layout over mid=[m0, m1]: tok0 = m0, tok1 = -m0, blank = m1,
        // dur0 = m1, dur1 = m0.
        let joint = ParakeetTdtJoint::new(
            ParakeetTdtJointWeights {
                pred_weight: named("pred_w", vec![0.0; 2 * hidden]),
                pred_bias: named("pred_b", vec![0.0; hidden]),
                out_weight: named(
                    "out_w",
                    vec![
                        1.0, 0.0, // tok0
                        -1.0, 0.0, // tok1
                        0.0, 1.0, // blank
                        0.0, 1.0, // dur 0
                        1.0, 0.0, // dur 1
                    ],
                ),
                out_bias: named("out_b", vec![0.0; 5]),
            },
            hidden,
        );
        (predictor, joint)
    }

    #[test]
    fn emits_token_with_duration_skip_and_blank_advances() {
        let (predictor, joint) = fixture();
        // Frame 0: m=[2, 0] (ReLU keeps it) -> tok0 wins, dur1 wins -> emit
        // tok0 with span [0, 1), advance to frame 1.
        // Frame 1: m=[0, 2] -> blank wins, dur0 wins -> budget spins to the
        // forced advance.
        // Frame 2: m=[2, 0] -> tok0 again.
        let enc = vec![2.0, 0.0, 0.0, 2.0, 2.0, 0.0];
        let emitted =
            tdt_greedy_decode(&enc, 3, &metadata(3, 2, 4), &predictor, &joint).expect("decode");
        assert_eq!(emitted.len(), 2);
        assert_eq!(emitted[0].token_id, 0);
        assert_eq!(emitted[0].start_frame, 0);
        assert_eq!(emitted[0].end_frame, 1);
        assert_eq!(emitted[1].start_frame, 2);
        assert!(emitted.iter().all(|e| e.probability > 0.0));
    }

    /// Adversarial: joint always answers (token, duration 0). The
    /// max_symbols_per_step budget must force one-frame progress and
    /// terminate with at most `frames * max_symbols` emissions.
    #[test]
    fn duration_zero_loop_is_bounded_by_max_symbols_budget() {
        let (predictor, _) = fixture();
        let hidden = 2;
        let joint = ParakeetTdtJoint::new(
            ParakeetTdtJointWeights {
                pred_weight: named("pred_w", vec![0.0; 2 * hidden]),
                pred_bias: named("pred_b", vec![0.0; hidden]),
                out_weight: named(
                    "out_w",
                    vec![
                        1.0, 1.0, // tok0 always wins
                        0.0, 0.0, // tok1
                        -1.0, -1.0, // blank never wins
                        1.0, 1.0, // dur 0 always wins
                        0.0, 0.0, // dur 1
                    ],
                ),
                out_bias: named("out_b", vec![0.0; 5]),
            },
            hidden,
        );
        let enc = vec![1.0, 1.0, 1.0, 1.0];
        let emitted =
            tdt_greedy_decode(&enc, 2, &metadata(3, 2, 3), &predictor, &joint).expect("decode");
        // 2 frames x budget 3 duration-0 symbols each.
        assert_eq!(emitted.len(), 6);
    }

    /// Blank with duration 0 must not emit and must still terminate (budget
    /// path), matching NeMo's blank-retry semantics.
    #[test]
    fn blank_duration_zero_terminates_without_emissions() {
        let (predictor, _) = fixture();
        let hidden = 2;
        let joint = ParakeetTdtJoint::new(
            ParakeetTdtJointWeights {
                pred_weight: named("pred_w", vec![0.0; 2 * hidden]),
                pred_bias: named("pred_b", vec![0.0; hidden]),
                out_weight: named(
                    "out_w",
                    vec![
                        0.0, 0.0, // tok0
                        0.0, 0.0, // tok1
                        1.0, 1.0, // blank always wins
                        1.0, 1.0, // dur 0 always wins
                        0.0, 0.0, // dur 1
                    ],
                ),
                out_bias: named("out_b", vec![0.0; 5]),
            },
            hidden,
        );
        let enc = vec![1.0, 1.0, 1.0, 1.0];
        let emitted =
            tdt_greedy_decode(&enc, 2, &metadata(3, 2, 10), &predictor, &joint).expect("decode");
        assert!(emitted.is_empty());
    }

    #[test]
    fn rejects_mismatched_joint_head_rows() {
        let (predictor, joint) = fixture();
        // metadata claims 3 durations but the fixture head has vocab 3 + 2.
        let err = tdt_greedy_decode(&[1.0, 1.0], 1, &metadata(3, 3, 4), &predictor, &joint)
            .expect_err("row mismatch must fail closed");
        assert!(err.contains("joint head"));
    }
}
