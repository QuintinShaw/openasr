//! parakeet-tdt prediction network: token embedding -> 2-layer LSTM.
//!
//! Host-side f32 (the xasr decoder/joiner precedent: per-symbol matvecs, not
//! ggml graph matmuls). PyTorch LSTM semantics, gates packed `[i|f|g|o]`:
//!
//! ```text
//!   gates = w_ih @ x + b_ih + w_hh @ h + b_hh          [4H]
//!   i = sigmoid(gates[0..H])    f = sigmoid(gates[H..2H])
//!   g = tanh(gates[2H..3H])     o = sigmoid(gates[3H..4H])
//!   c' = f*c + i*g              h' = o * tanh(c')
//! ```
//!
//! The decode start symbol is the BLANK token fed through the LSTM from a
//! zero state (NeMo convention; the blank embedding row is the trained
//! `padding_idx` zero vector, so the first step is an LSTM step on zeros).

use super::encoder_weights::ParakeetTdtPredictorWeights;

#[derive(Debug, Clone)]
pub(crate) struct ParakeetTdtPredictor {
    weights: ParakeetTdtPredictorWeights,
    hidden: usize,
    vocab_size: usize,
}

/// Per-utterance recurrent state: one (h, c) pair per LSTM layer.
#[derive(Debug, Clone)]
pub(crate) struct ParakeetTdtLstmState {
    h: Vec<Vec<f32>>,
    c: Vec<Vec<f32>>,
    /// Scratch gate buffer reused across steps.
    gates: Vec<f32>,
}

impl ParakeetTdtPredictor {
    pub(crate) fn new(
        weights: ParakeetTdtPredictorWeights,
        hidden: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            weights,
            hidden,
            vocab_size,
        }
    }

    pub(crate) fn initial_state(&self) -> ParakeetTdtLstmState {
        let layers = self.weights.lstm_layers.len();
        ParakeetTdtLstmState {
            h: vec![vec![0.0; self.hidden]; layers],
            c: vec![vec![0.0; self.hidden]; layers],
            gates: vec![0.0; 4 * self.hidden],
        }
    }

    /// Advance the prediction network by one token. Writes the top layer's
    /// new hidden state into `output` (`pred_hidden` wide).
    pub(crate) fn step(
        &self,
        token_id: u32,
        state: &mut ParakeetTdtLstmState,
        output: &mut Vec<f32>,
    ) -> Result<(), String> {
        let hidden = self.hidden;
        let token = token_id as usize;
        if token >= self.vocab_size {
            return Err(format!(
                "parakeet-tdt predictor token id {token_id} out of range for vocab {}",
                self.vocab_size
            ));
        }
        let embed_row = &self.weights.embedding.values[token * hidden..(token + 1) * hidden];

        // Layer 0 consumes the embedding; layer n consumes layer n-1's new h.
        // The per-layer input is copied so the borrow of `state.h` stays local.
        let mut input: Vec<f32> = embed_row.to_vec();
        for (layer_idx, layer) in self.weights.lstm_layers.iter().enumerate() {
            lstm_step(
                &input,
                &layer.w_ih.values,
                &layer.b_ih.values,
                &layer.w_hh.values,
                &layer.b_hh.values,
                &mut state.h[layer_idx],
                &mut state.c[layer_idx],
                &mut state.gates,
                hidden,
            )?;
            input.clear();
            input.extend_from_slice(&state.h[layer_idx]);
        }
        output.clear();
        output.extend_from_slice(&input);
        Ok(())
    }
}

/// One PyTorch LSTM cell step over packed `[i|f|g|o]` gates. `w_ih` is
/// row-major `[4H][input]`, `w_hh` row-major `[4H][H]` (the flat safetensors
/// layout; the pack's reversed ggml dims do not change the buffer).
#[allow(clippy::too_many_arguments)]
fn lstm_step(
    input: &[f32],
    w_ih: &[f32],
    b_ih: &[f32],
    w_hh: &[f32],
    b_hh: &[f32],
    h: &mut [f32],
    c: &mut [f32],
    gates: &mut [f32],
    hidden: usize,
) -> Result<(), String> {
    let rows = 4 * hidden;
    if w_ih.len() != rows * input.len()
        || w_hh.len() != rows * hidden
        || b_ih.len() != rows
        || b_hh.len() != rows
        || h.len() != hidden
        || c.len() != hidden
        || gates.len() != rows
    {
        return Err(format!(
            "parakeet-tdt lstm shape mismatch: input {} w_ih {} w_hh {} b_ih {} b_hh {} h {} c {}",
            input.len(),
            w_ih.len(),
            w_hh.len(),
            b_ih.len(),
            b_hh.len(),
            h.len(),
            c.len()
        ));
    }
    for (row, gate) in gates.iter_mut().enumerate() {
        let wi = &w_ih[row * input.len()..(row + 1) * input.len()];
        let wh = &w_hh[row * hidden..(row + 1) * hidden];
        *gate = b_ih[row] + b_hh[row] + dot_f32(wi, input) + dot_f32(wh, h);
    }
    for j in 0..hidden {
        let i_gate = sigmoid(gates[j]);
        let f_gate = sigmoid(gates[hidden + j]);
        let g_gate = gates[2 * hidden + j].tanh();
        let o_gate = sigmoid(gates[3 * hidden + j]);
        c[j] = f_gate * c[j] + i_gate * g_gate;
        h[j] = o_gate * c[j].tanh();
    }
    Ok(())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Four-accumulator dot product (breaks the FP-add latency chain; same kernel
/// shape as the xasr joiner matvec).
#[inline]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut s0, mut s1, mut s2, mut s3) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut i = 0usize;
    while i + 16 <= n {
        let ca = &a[i..i + 16];
        let cb = &b[i..i + 16];
        s0 += ca[0] * cb[0] + ca[1] * cb[1] + ca[2] * cb[2] + ca[3] * cb[3];
        s1 += ca[4] * cb[4] + ca[5] * cb[5] + ca[6] * cb[6] + ca[7] * cb[7];
        s2 += ca[8] * cb[8] + ca[9] * cb[9] + ca[10] * cb[10] + ca[11] * cb[11];
        s3 += ca[12] * cb[12] + ca[13] * cb[13] + ca[14] * cb[14] + ca[15] * cb[15];
        i += 16;
    }
    let mut tail = 0.0_f32;
    while i < n {
        tail += a[i] * b[i];
        i += 1;
    }
    s0 + s1 + s2 + s3 + tail
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::parakeet_tdt::encoder_weights::{
        NamedTensor, ParakeetTdtLstmLayerWeights, ParakeetTdtPredictorWeights,
    };

    fn named(name: &str, dims: Vec<usize>, values: Vec<f32>) -> NamedTensor {
        NamedTensor {
            name: name.to_string(),
            dims,
            values,
        }
    }

    /// Single-layer H=1 LSTM with hand-computable weights: w_ih = [1,1,1,1]^T
    /// on a scalar input, w_hh = 0, biases 0. For x = 0.5:
    /// i = f = o = sigmoid(0.5), g = tanh(0.5); c' = i*g; h' = o*tanh(c').
    #[test]
    fn lstm_step_matches_manual_reference() {
        let mut h = vec![0.0f32];
        let mut c = vec![0.0f32];
        let mut gates = vec![0.0f32; 4];
        lstm_step(
            &[0.5],
            &[1.0, 1.0, 1.0, 1.0],
            &[0.0; 4],
            &[0.0; 4],
            &[0.0; 4],
            &mut h,
            &mut c,
            &mut gates,
            1,
        )
        .expect("lstm step");
        let s = 1.0 / (1.0 + (-0.5f32).exp());
        let g = 0.5f32.tanh();
        let c_want = s * g;
        let h_want = s * c_want.tanh();
        assert!((c[0] - c_want).abs() < 1.0e-6);
        assert!((h[0] - h_want).abs() < 1.0e-6);
    }

    /// The forget gate must read the OLD cell state: run two steps and check
    /// the second step's c' = f*c1 + i*g (with w_hh=0 the gates depend only on
    /// the input, so the recurrence is exactly the cell-state chain).
    #[test]
    fn lstm_cell_state_chains_across_steps() {
        let mut h = vec![0.0f32];
        let mut c = vec![0.0f32];
        let mut gates = vec![0.0f32; 4];
        let step = |h: &mut Vec<f32>, c: &mut Vec<f32>, gates: &mut Vec<f32>| {
            lstm_step(
                &[1.0],
                &[1.0, 1.0, 1.0, 1.0],
                &[0.0; 4],
                &[0.0; 4],
                &[0.0; 4],
                h,
                c,
                gates,
                1,
            )
            .expect("lstm step");
        };
        step(&mut h, &mut c, &mut gates);
        let c1 = c[0];
        step(&mut h, &mut c, &mut gates);
        let s = 1.0 / (1.0 + (-1.0f32).exp());
        let g = 1.0f32.tanh();
        let c2_want = s * c1 + s * g;
        assert!((c[0] - c2_want).abs() < 1.0e-6);
    }

    #[test]
    fn predictor_steps_two_layers_and_rejects_out_of_range_tokens() {
        let hidden = 2;
        let layer = || ParakeetTdtLstmLayerWeights {
            w_ih: named(
                "w_ih",
                vec![hidden, 4 * hidden],
                vec![0.1; 4 * hidden * hidden],
            ),
            w_hh: named(
                "w_hh",
                vec![hidden, 4 * hidden],
                vec![0.1; 4 * hidden * hidden],
            ),
            b_ih: named("b_ih", vec![4 * hidden], vec![0.0; 4 * hidden]),
            b_hh: named("b_hh", vec![4 * hidden], vec![0.0; 4 * hidden]),
        };
        let predictor = ParakeetTdtPredictor::new(
            ParakeetTdtPredictorWeights {
                embedding: named(
                    "embed",
                    vec![hidden, 3],
                    vec![0.0, 0.0, 1.0, -1.0, 0.5, 0.5],
                ),
                lstm_layers: vec![layer(), layer()],
            },
            hidden,
            3,
        );
        let mut state = predictor.initial_state();
        let mut output = Vec::new();
        predictor
            .step(1, &mut state, &mut output)
            .expect("step token 1");
        assert_eq!(output.len(), hidden);
        assert!(output.iter().all(|v| v.is_finite()));
        assert!(predictor.step(3, &mut state, &mut output).is_err());
    }
}
