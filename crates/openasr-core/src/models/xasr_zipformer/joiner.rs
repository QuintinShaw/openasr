//! X-ASR transducer joiner: tanh(enc_proj + dec_proj) -> vocab logits.
//!
//! The greedy loop runs the joiner once per symbol step, but its two input
//! projections change at different rates: the encoder projection only changes
//! per frame, and the decoder projection only changes when a non-blank token
//! is emitted (most frames emit none). The split API below lets the caller
//! compute each projection exactly when its input changes and reuse scratch
//! buffers across the whole utterance.

use super::weights::XasrJoinerWeights;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct XasrJoiner {
    weights: XasrJoinerWeights,
}

/// Reusable buffers for the per-step joiner pipeline. `enc` and `dec` hold the
/// two projections at their own cadence; `joined` and `logits` are transient.
#[derive(Debug, Clone)]
pub(crate) struct XasrJoinerScratch {
    enc: Vec<f32>,
    dec: Vec<f32>,
    joined: Vec<f32>,
    logits: Vec<f32>,
}

impl XasrJoiner {
    pub(crate) fn new(weights: XasrJoinerWeights) -> Self {
        Self { weights }
    }

    pub(crate) fn scratch(&self) -> XasrJoinerScratch {
        XasrJoinerScratch {
            enc: vec![0.0; self.weights.encoder_proj_weight.output_dim],
            dec: vec![0.0; self.weights.decoder_proj_weight.output_dim],
            joined: vec![0.0; self.weights.encoder_proj_weight.output_dim],
            logits: vec![0.0; self.weights.output_linear_weight.output_dim],
        }
    }

    pub(crate) fn project_encoder_frame(
        &self,
        encoder_frame: &[f32],
        scratch: &mut XasrJoinerScratch,
    ) -> Result<(), String> {
        self.weights.encoder_proj_weight.apply_into(
            encoder_frame,
            Some(&self.weights.encoder_proj_bias),
            &mut scratch.enc,
        )
    }

    pub(crate) fn project_decoder_state(
        &self,
        decoder_state: &[f32],
        scratch: &mut XasrJoinerScratch,
    ) -> Result<(), String> {
        self.weights.decoder_proj_weight.apply_into(
            decoder_state,
            Some(&self.weights.decoder_proj_bias),
            &mut scratch.dec,
        )
    }

    /// tanh(enc + dec) -> vocab logits, from the projections already staged in
    /// `scratch`. Returns a borrow of the scratch logits buffer.
    pub(crate) fn logits_from_projected<'s>(
        &self,
        scratch: &'s mut XasrJoinerScratch,
    ) -> Result<&'s [f32], String> {
        if scratch.enc.len() != scratch.dec.len() {
            return Err(format!(
                "xasr joiner projected dims differ: encoder {} vs decoder {}",
                scratch.enc.len(),
                scratch.dec.len()
            ));
        }
        for ((joined, enc), dec) in scratch
            .joined
            .iter_mut()
            .zip(&scratch.enc)
            .zip(&scratch.dec)
        {
            *joined = (enc + dec).tanh();
        }
        self.weights.output_linear_weight.apply_into(
            &scratch.joined,
            Some(&self.weights.output_linear_bias),
            &mut scratch.logits,
        )?;
        Ok(&scratch.logits)
    }

    pub(crate) fn logits(
        &self,
        encoder_frame: &[f32],
        decoder_state: &[f32],
    ) -> Result<Vec<f32>, String> {
        let mut scratch = self.scratch();
        self.project_encoder_frame(encoder_frame, &mut scratch)?;
        self.project_decoder_state(decoder_state, &mut scratch)?;
        self.logits_from_projected(&mut scratch)?;
        Ok(scratch.logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::xasr_zipformer::weights::{StoredLinear, XasrJoinerWeights};

    #[test]
    fn applies_tanh_before_output_linear() {
        let joiner = XasrJoiner::new(XasrJoinerWeights {
            encoder_proj_weight: identity("enc", 2),
            encoder_proj_bias: vec![0.0, 0.0],
            decoder_proj_weight: identity("dec", 2),
            decoder_proj_bias: vec![0.0, 0.0],
            output_linear_weight: StoredLinear {
                name: "out".to_string(),
                input_dim: 2,
                output_dim: 2,
                values: vec![1.0, 0.0, 0.0, 1.0],
            },
            output_linear_bias: vec![0.0, 0.0],
        });
        let logits = joiner.logits(&[1.0, -1.0], &[1.0, 3.0]).unwrap();
        assert!((logits[0] - 2.0_f32.tanh()).abs() < 1.0e-6);
        assert!((logits[1] - 2.0_f32.tanh()).abs() < 1.0e-6);
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
